"""
Tests for Fused Layer Normalization

This test suite validates the fused layer normalization implementation against
PyTorch's standard LayerNorm for correctness, performance, and edge cases.
"""

import pytest
import torch
import torch.nn as nn

from rosellm.rosetrainer.fusions import FusedLayerNorm, LayerNormConfig


class TestFusedLayerNorm:
    """Test suite for fused layer normalization."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def config(self, device):
        """Create default config."""
        return LayerNormConfig(
            hidden_size=768,
            eps=1e-5,
            persist_layer_norm=False,  # Use standard for testing
            zero_centered_gamma=False,
            memory_efficient=False,
            device=device,
        )

    def test_forward_accuracy(self, config, device):
        """Test forward pass accuracy against PyTorch LayerNorm."""
        batch_size, seq_length = 2, 128
        hidden_size = config.hidden_size

        # Create input
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_length, hidden_size, device=device)

        # PyTorch LayerNorm
        pytorch_ln = nn.LayerNorm(hidden_size, eps=config.eps).to(device)

        # Fused LayerNorm
        fused_ln = FusedLayerNorm(config)

        # Copy weights
        with torch.no_grad():
            fused_ln.weight.copy_(pytorch_ln.weight)
            fused_ln.bias.copy_(pytorch_ln.bias)

        # Forward pass
        pytorch_out = pytorch_ln(x)
        fused_out = fused_ln(x)

        # Check accuracy
        assert torch.allclose(pytorch_out, fused_out, rtol=1e-5, atol=1e-6)

    def test_backward_accuracy(self, config, device):
        """Test backward pass accuracy against PyTorch LayerNorm."""
        # Skip if using CPU fallback (lower precision expected)
        try:
            from apex.normalization.fused_layer_norm import (  # pyright: ignore
                FusedLayerNormAffineFunction,
            )

            # Just check it's available
            assert FusedLayerNormAffineFunction is not None
        except ImportError:
            pytest.skip("Apex not available, CPU fallback has lower gradient precision")

        batch_size, seq_length = 2, 64
        hidden_size = config.hidden_size

        # Create inputs
        torch.manual_seed(42)
        x1 = torch.randn(
            batch_size, seq_length, hidden_size, device=device, requires_grad=True
        )
        x2 = x1.clone().detach().requires_grad_(True)
        grad_output = torch.randn_like(x1)

        # PyTorch LayerNorm
        pytorch_ln = nn.LayerNorm(hidden_size, eps=config.eps).to(device)

        # Fused LayerNorm
        fused_ln = FusedLayerNorm(config)

        # Copy weights
        with torch.no_grad():
            fused_ln.weight.copy_(pytorch_ln.weight)
            fused_ln.bias.copy_(pytorch_ln.bias)

        # Forward and backward
        pytorch_out = pytorch_ln(x1)
        pytorch_out.backward(grad_output)

        fused_out = fused_ln(x2)
        fused_out.backward(grad_output)

        # Check gradient accuracy (CPU fallback has slightly lower precision)
        assert x1.grad is not None and x2.grad is not None
        assert pytorch_ln.weight.grad is not None and fused_ln.weight.grad is not None
        assert pytorch_ln.bias.grad is not None and fused_ln.bias.grad is not None

        assert torch.allclose(x1.grad, x2.grad, rtol=1e-2, atol=1e-2)
        assert torch.allclose(
            pytorch_ln.weight.grad, fused_ln.weight.grad, rtol=1e-3, atol=1e-3
        )
        assert torch.allclose(
            pytorch_ln.bias.grad, fused_ln.bias.grad, rtol=1e-3, atol=1e-3
        )

    def test_zero_centered_gamma(self, device):
        """Test zero-centered gamma functionality."""
        config = LayerNormConfig(
            hidden_size=512, eps=1e-5, zero_centered_gamma=True, device=device
        )

        ln = FusedLayerNorm(config)

        # Check initialization
        assert torch.all(ln.weight == 0)
        assert torch.all(ln.bias == 0)

        # Test forward pass
        x = torch.randn(2, 32, 512, device=device)
        output = ln(x)
        assert output.shape == x.shape

    def test_different_shapes(self, device):
        """Test various input shapes."""
        test_cases = [
            (1, 512),  # 2D input
            (4, 128, 768),  # 3D input
            (2, 8, 64, 256),  # 4D input
        ]

        for shape in test_cases:
            hidden_size = shape[-1]
            config = LayerNormConfig(hidden_size=hidden_size, device=device)

            ln = FusedLayerNorm(config)
            x = torch.randn(*shape, device=device)

            output = ln(x)
            assert output.shape == x.shape

    def test_persistent_kernel_sizes(self, device):
        """Test persistent kernel for supported sizes."""
        # Only test if persistent kernel is available
        try:
            from apex.contrib.layer_norm.layer_norm import (  # pyright: ignore
                FastLayerNormFN,
            )

            # Just check it's available
            assert FastLayerNormFN is not None
        except ImportError:
            pytest.skip("Apex not available for persistent kernel test")

        # Test a supported size
        config = LayerNormConfig(
            hidden_size=4096, persist_layer_norm=True, device=device  # Supported size
        )

        ln = FusedLayerNorm(config)
        # Check if using persistent kernel
        from rosellm.rosetrainer.fusions import LayerNormKernelType

        assert ln.kernel.get_type() == LayerNormKernelType.PERSISTENT

        # Test unsupported size
        config = LayerNormConfig(
            hidden_size=777, persist_layer_norm=True, device=device  # Unsupported size
        )

        ln = FusedLayerNorm(config)
        # Check if not using persistent kernel
        from rosellm.rosetrainer.fusions import LayerNormKernelType

        assert ln.kernel.get_type() != LayerNormKernelType.PERSISTENT

    def test_memory_efficient_mode(self, device):
        """Test memory efficient backward pass."""
        config = LayerNormConfig(hidden_size=1024, memory_efficient=True, device=device)

        ln = FusedLayerNorm(config)
        x = torch.randn(2, 128, 1024, device=device, requires_grad=True)

        # Forward and backward should work
        output = ln(x)
        output.mean().backward()

        assert x.grad is not None

    def test_gradient_flow(self, config, device):
        """Test gradient flow through layer norm."""
        ln = FusedLayerNorm(config)
        x = torch.randn(2, 32, config.hidden_size, device=device, requires_grad=True)

        # Forward pass
        output = ln(x)
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert ln.weight.grad is not None
        assert ln.bias.grad is not None

        # Check gradients are non-zero
        assert torch.any(x.grad != 0)
        assert torch.any(ln.weight.grad != 0)
        assert torch.any(ln.bias.grad != 0)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_mixed_precision(self, device, dtype):
        """Test layer norm with different precisions."""
        if device.type == "cpu" and dtype == torch.float16:
            pytest.skip("CPU doesn't support float16 well")

        config = LayerNormConfig(hidden_size=512, device=device)

        ln = FusedLayerNorm(config).to(dtype)
        x = torch.randn(2, 64, 512, device=device, dtype=dtype)

        output = ln(x)
        assert output.dtype == dtype
        assert output.shape == x.shape

    def test_numerical_stability(self, config, device):
        """Test numerical stability with extreme values."""
        ln = FusedLayerNorm(config)

        # Test with very small values
        x_small = torch.full((2, 32, config.hidden_size), 1e-8, device=device)
        output_small = ln(x_small)
        assert torch.all(torch.isfinite(output_small))

        # Test with very large values
        x_large = torch.full((2, 32, config.hidden_size), 1e8, device=device)
        output_large = ln(x_large)
        assert torch.all(torch.isfinite(output_large))

        # Test with mixed scales
        x_mixed = torch.randn(2, 32, config.hidden_size, device=device)
        x_mixed[:, :16] *= 1e8
        x_mixed[:, 16:] *= 1e-8
        output_mixed = ln(x_mixed)
        assert torch.all(torch.isfinite(output_mixed))

    def test_sequence_parallel_flag(self, device):
        """Test sequence parallel flag setting."""
        config = LayerNormConfig(hidden_size=768, sequence_parallel=True, device=device)

        ln = FusedLayerNorm(config)

        # Check flags are set
        assert hasattr(ln.weight, "sequence_parallel")
        assert hasattr(ln.bias, "sequence_parallel")
        assert getattr(ln.weight, "sequence_parallel", False) is True
        assert getattr(ln.bias, "sequence_parallel", False) is True

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for performance test"
    )
    def test_performance_improvement(self):
        """Test that fused version is faster than standard."""
        device = torch.device("cuda")
        hidden_size = 4096
        batch_size, seq_length = 4, 512

        # Create configs
        config = LayerNormConfig(
            hidden_size=hidden_size, persist_layer_norm=True, device=device
        )

        # Create layers
        standard_ln = nn.LayerNorm(hidden_size).to(device)
        fused_ln = FusedLayerNorm(config)

        # Create input
        x = torch.randn(batch_size, seq_length, hidden_size, device=device)

        # Warmup
        for _ in range(10):
            _ = standard_ln(x)
            _ = fused_ln(x)

        # Time standard
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record(stream=torch.cuda.current_stream())
        for _ in range(100):
            _ = standard_ln(x)
        end.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        standard_time = start.elapsed_time(end)

        # Time fused
        start.record(stream=torch.cuda.current_stream())
        for _ in range(100):
            _ = fused_ln(x)
        end.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        fused_time = start.elapsed_time(end)

        # Fused should be faster (or at least not significantly slower)
        print(f"Standard: {standard_time:.2f}ms, Fused: {fused_time:.2f}ms")
        assert fused_time < standard_time * 1.1  # Allow 10% tolerance


class TestLayerNormConfig:
    """Test LayerNormConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LayerNormConfig(hidden_size=768)

        assert config.hidden_size == 768
        assert config.eps == 1e-5
        assert config.persist_layer_norm is True
        assert config.zero_centered_gamma is False
        assert config.sequence_parallel is False
        assert config.memory_efficient is False
        assert config.device is None

    def test_custom_config(self):
        """Test custom configuration values."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = LayerNormConfig(
            hidden_size=1024,
            eps=1e-6,
            persist_layer_norm=False,
            zero_centered_gamma=True,
            sequence_parallel=True,
            memory_efficient=True,
            device=device,
        )

        assert config.hidden_size == 1024
        assert config.eps == 1e-6
        assert config.persist_layer_norm is False
        assert config.zero_centered_gamma is True
        assert config.sequence_parallel is True
        assert config.memory_efficient is True
        assert config.device == device


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
