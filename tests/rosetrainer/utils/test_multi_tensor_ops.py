"""
Comprehensive unit tests for multi-tensor gradient operations.

This test suite validates the multi-tensor operations with:
- Automatic backend detection and fallback
- Correctness validation against PyTorch reference
- Performance benchmarking
- Edge case handling
- Memory efficiency testing
"""

from typing import List
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from rosellm.rosetrainer.utils.multi_tensor_ops import (
    Backend,
    BackendInfo,
    MultiTensorOperator,
    get_default_operator,
    multi_tensor_clip_grad_norm,
    multi_tensor_norm,
    multi_tensor_scale,
)


class TestBackendDetection:
    """Test backend detection and selection."""

    def test_backend_detection(self):
        """Test that backend detection works correctly."""
        operator = MultiTensorOperator()

        # Should always have PyTorch backend
        assert Backend.PYTORCH in operator.backends
        assert operator.backends[Backend.PYTORCH].available

        # Check backend info structure
        pytorch_info = operator.backends[Backend.PYTORCH]
        assert pytorch_info.name == Backend.PYTORCH
        assert pytorch_info.version == torch.__version__
        assert pytorch_info.device_support is not None
        assert "cpu" in pytorch_info.device_support

    def test_backend_selection_priority(self):
        """Test backend selection follows correct priority."""
        # Mock all backends as available
        with patch.object(MultiTensorOperator, "_detect_transformer_engine") as mock_te:
            with patch.object(MultiTensorOperator, "_detect_apex") as mock_apex:
                mock_te.return_value = BackendInfo(
                    name=Backend.TRANSFORMER_ENGINE,
                    available=True,
                    version="1.0.0",
                )
                mock_apex.return_value = BackendInfo(
                    name=Backend.APEX,
                    available=True,
                    version="1.0.0",
                )

                # Should select Transformer Engine when available
                operator = MultiTensorOperator()
                if torch.cuda.is_available():
                    assert operator.backend.name == Backend.TRANSFORMER_ENGINE

    def test_fallback_to_pytorch(self):
        """Test fallback to PyTorch when other backends unavailable."""
        with patch.object(MultiTensorOperator, "_detect_transformer_engine") as mock_te:
            with patch.object(MultiTensorOperator, "_detect_apex") as mock_apex:
                mock_te.return_value = BackendInfo(
                    name=Backend.TRANSFORMER_ENGINE,
                    available=False,
                )
                mock_apex.return_value = BackendInfo(
                    name=Backend.APEX,
                    available=False,
                )

                operator = MultiTensorOperator()
                assert operator.backend.name == Backend.PYTORCH

    def test_preferred_backend_selection(self):
        """Test that preferred backend is selected when available."""
        operator = MultiTensorOperator(preferred_backend=Backend.PYTORCH)
        assert operator.backend.name == Backend.PYTORCH


class TestMultiTensorNorm:
    """Test multi-tensor norm calculations."""

    def setup_method(self):
        """Setup test tensors."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)

    def create_test_tensors(
        self, num_tensors: int = 5, size: int = 1000
    ) -> List[torch.Tensor]:
        """Create test tensors with known properties."""
        tensors = []
        for i in range(num_tensors):
            # Create tensors with different scales to test numerical stability
            scale = 10 ** (i - 2)
            tensor = torch.randn(size, device=self.device) * scale
            tensors.append(tensor)
        return tensors

    def test_l2_norm_correctness(self):
        """Test L2 norm calculation correctness."""
        tensors = self.create_test_tensors()
        operator = MultiTensorOperator(device=self.device)

        # Calculate using multi-tensor ops
        mt_norm = operator.calculate_norm(tensors, norm_type=2.0)

        # Calculate reference norm
        ref_norm_squared = sum(t.pow(2).sum() for t in tensors)
        ref_norm = torch.sqrt(ref_norm_squared)  # type: ignore[arg-type]

        # Should match within tolerance
        torch.testing.assert_close(mt_norm, ref_norm, rtol=1e-5, atol=1e-6)

    def test_l1_norm_correctness(self):
        """Test L1 norm calculation correctness."""
        tensors = self.create_test_tensors()
        operator = MultiTensorOperator(device=self.device)

        # Calculate using multi-tensor ops
        mt_norm = operator.calculate_norm(tensors, norm_type=1.0)

        # Calculate reference norm
        ref_norm = sum(t.abs().sum() for t in tensors)

        # Should match within tolerance
        torch.testing.assert_close(mt_norm, ref_norm, rtol=1e-5, atol=1e-6)

    def test_inf_norm_correctness(self):
        """Test infinity norm calculation correctness."""
        tensors = self.create_test_tensors()
        operator = MultiTensorOperator(device=self.device)

        # Calculate using multi-tensor ops
        mt_norm = operator.calculate_norm(tensors, norm_type=float("inf"))

        # Calculate reference norm
        ref_norm = max(t.abs().max() for t in tensors)

        # Should match within tolerance
        torch.testing.assert_close(mt_norm, ref_norm, rtol=1e-5, atol=1e-6)

    def test_per_tensor_norms(self):
        """Test per-tensor norm calculation."""
        tensors = self.create_test_tensors()
        operator = MultiTensorOperator(device=self.device)

        # Calculate per-tensor norms
        per_tensor_norms = operator.calculate_norm(
            tensors, norm_type=2.0, per_tensor=True
        )

        # Verify each norm
        assert len(per_tensor_norms) == len(tensors)
        for tensor, norm in zip(tensors, per_tensor_norms):
            ref_norm = torch.norm(tensor, p=2.0)
            torch.testing.assert_close(norm, ref_norm, rtol=1e-5, atol=1e-6)

    def test_empty_tensor_list(self):
        """Test handling of empty tensor list."""
        operator = MultiTensorOperator(device=self.device)

        norm = operator.calculate_norm([], norm_type=2.0)
        assert isinstance(norm, torch.Tensor)
        assert norm.item() == 0.0

    def test_non_finite_handling(self):
        """Test handling of non-finite values."""
        operator = MultiTensorOperator(device=self.device)

        # Create tensors with NaN and Inf
        tensors = [
            torch.tensor([1.0, 2.0, float("nan")], device=self.device),
            torch.tensor([3.0, float("inf"), 4.0], device=self.device),
            torch.tensor([5.0, 6.0, 7.0], device=self.device),
        ]

        # Should handle non-finite values gracefully
        norm = operator.calculate_norm(tensors, norm_type=2.0)
        assert isinstance(norm, torch.Tensor)
        assert torch.isfinite(norm)

    def test_mixed_dtypes(self):
        """Test handling of mixed tensor dtypes."""
        operator = MultiTensorOperator(device=self.device)

        tensors = [
            torch.randn(100, device=self.device, dtype=torch.float32),
            torch.randn(100, device=self.device, dtype=torch.float16),
            torch.randn(100, device=self.device, dtype=torch.float64),
        ]

        # Should handle mixed dtypes
        norm = operator.calculate_norm(tensors, norm_type=2.0)
        assert isinstance(norm, torch.Tensor)
        assert torch.isfinite(norm)


class TestGradientClipping:
    """Test gradient clipping operations."""

    def setup_method(self):
        """Setup test model and parameters."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)

        # Create a simple model
        self.model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        ).to(self.device)

        # Create dummy gradients
        for param in self.model.parameters():
            param.grad = torch.randn_like(param) * 10.0

    def test_clip_grad_norm(self):
        """Test gradient norm clipping."""
        operator = MultiTensorOperator(device=self.device)

        # Get initial norm
        params = list(self.model.parameters())
        grads = [p.grad for p in params if p.grad is not None]
        initial_norm = operator.calculate_norm(grads, norm_type=2.0)

        # Clip gradients
        max_norm = 1.0
        clip_stats = operator.clip_grad_norm(params, max_norm, norm_type=2.0)

        # Check clipping was applied
        assert isinstance(initial_norm, torch.Tensor)
        if initial_norm.item() > max_norm:
            assert clip_stats["was_clipped"]
            assert clip_stats["clip_coeff"] < 1.0

            # Verify new norm
            new_norm = operator.calculate_norm(grads, norm_type=2.0)
            torch.testing.assert_close(
                new_norm, torch.tensor(max_norm), rtol=1e-4, atol=1e-5
            )

    def test_no_clipping_when_below_threshold(self):
        """Test that no clipping occurs when norm is below threshold."""
        operator = MultiTensorOperator(device=self.device)

        # Scale down gradients
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.mul_(0.01)

        params = list(self.model.parameters())

        # Clip with high threshold
        max_norm = 100.0
        clip_stats = operator.clip_grad_norm(params, max_norm, norm_type=2.0)

        # Should not clip
        assert not clip_stats["was_clipped"]
        assert clip_stats["clip_coeff"] == 1.0

    def test_clip_with_non_finite_error(self):
        """Test error handling for non-finite gradients."""
        operator = MultiTensorOperator(device=self.device)

        # Add non-finite gradient - make entire gradient non-finite
        params = list(self.model.parameters())
        if params[0].grad is not None:
            params[0].grad.fill_(float("inf"))

        # Should not raise when error_if_nonfinite=False
        clip_stats = operator.clip_grad_norm(params, 1.0, error_if_nonfinite=False)
        assert "total_norm" in clip_stats

        # Create a case where the norm itself is non-finite
        # This happens when all gradients are infinite
        for param in params:
            if param.grad is not None:
                param.grad.fill_(float("inf"))

        # In this case, the norm calculation will handle it gracefully
        clip_stats = operator.clip_grad_norm(params, 1.0, error_if_nonfinite=False)
        assert "total_norm" in clip_stats


class TestTensorScaling:
    """Test tensor scaling operations."""

    def setup_method(self):
        """Setup test tensors."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)

    def test_scale_tensors(self):
        """Test tensor scaling correctness."""
        operator = MultiTensorOperator(device=self.device)

        tensors = [torch.randn(100, device=self.device) for _ in range(5)]
        original = [t.clone() for t in tensors]

        scale = 0.5
        scaled = operator.scale_tensors(tensors, scale, in_place=True)

        # Check scaling was applied
        for orig, scaled_t in zip(original, scaled):
            torch.testing.assert_close(scaled_t, orig * scale, rtol=1e-5, atol=1e-6)

    def test_scale_with_tensor_factor(self):
        """Test scaling with tensor scale factor."""
        operator = MultiTensorOperator(device=self.device)

        tensors = [torch.ones(100, device=self.device) for _ in range(3)]
        scale = torch.tensor(2.0, device=self.device)

        scaled = operator.scale_tensors(tensors, scale, in_place=True)

        for t in scaled:
            assert torch.allclose(t, torch.ones_like(t) * 2.0)

    def test_non_inplace_scaling(self):
        """Test non-inplace scaling preserves original tensors."""
        operator = MultiTensorOperator(device=self.device)

        tensors = [torch.randn(100, device=self.device) for _ in range(3)]
        original = [t.clone() for t in tensors]

        scale = 0.5
        scaled = operator.scale_tensors(tensors, scale, in_place=False)

        # Original should be unchanged
        for orig, tensor in zip(original, tensors):
            torch.testing.assert_close(tensor, orig)

        # Scaled should be different
        for orig, scaled_t in zip(original, scaled):
            torch.testing.assert_close(scaled_t, orig * scale)


class TestValidation:
    """Test validation against reference implementations."""

    def setup_method(self):
        """Setup test environment."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)

    def test_validate_against_reference(self):
        """Test validation functionality."""
        operator = MultiTensorOperator(device=self.device)

        tensors = [torch.randn(1000, device=self.device) for _ in range(10)]

        # Validate norm calculation
        validation = operator.validate_against_reference(
            tensors, operation="norm", norm_type=2.0
        )

        assert "operation" in validation
        assert "backend" in validation
        assert "current_result" in validation
        assert "reference_result" in validation
        assert "matches" in validation

        # Should match within tolerance
        assert validation["matches"]
        assert validation["relative_difference"] < 1e-5


class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""

    def setup_method(self):
        """Setup benchmark environment."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)

    @pytest.mark.skip(reason="Performance test - run manually")
    def test_norm_performance(self):
        """Benchmark norm calculation performance."""
        operator = MultiTensorOperator(device=self.device, enable_benchmarking=True)

        # Create large tensor list
        tensors = [torch.randn(10000, device=self.device) for _ in range(100)]

        # Warm up
        for _ in range(5):
            operator.calculate_norm(tensors, norm_type=2.0)

        # Benchmark
        import time

        start = time.perf_counter()
        for _ in range(100):
            operator.calculate_norm(tensors, norm_type=2.0)
        elapsed = time.perf_counter() - start

        # Get performance stats
        stats = operator.get_performance_stats()

        # Should have recorded performance
        assert len(stats) > 0

        print(f"\nNorm calculation: {elapsed:.3f}s for 100 iterations")
        print(f"Performance stats: {stats}")

    @pytest.mark.skip(reason="Performance test - run manually")
    def test_scaling_performance(self):
        """Benchmark tensor scaling performance."""
        operator = MultiTensorOperator(device=self.device, enable_benchmarking=True)

        # Create large tensor list
        tensors = [torch.randn(10000, device=self.device) for _ in range(100)]

        # Warm up
        for _ in range(5):
            operator.scale_tensors(tensors, 0.5, in_place=True)

        # Benchmark
        import time

        start = time.perf_counter()
        for _ in range(100):
            operator.scale_tensors(tensors, 0.5, in_place=True)
        elapsed = time.perf_counter() - start

        print(f"\nTensor scaling: {elapsed:.3f}s for 100 iterations")


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def setup_method(self):
        """Setup test environment."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)

    def test_default_operator(self):
        """Test default operator creation and caching."""
        op1 = get_default_operator()
        op2 = get_default_operator()

        # Should return same instance
        assert op1 is op2

        # Reset should create new instance
        op3 = get_default_operator(reset=True)
        assert op3 is not op2

    def test_multi_tensor_norm_function(self):
        """Test multi_tensor_norm convenience function."""
        tensors = [torch.randn(100, device=self.device) for _ in range(5)]

        norm = multi_tensor_norm(tensors, norm_type=2.0)

        # Should return correct norm
        ref_norm_squared = sum(t.pow(2).sum() for t in tensors)
        ref_norm = torch.sqrt(ref_norm_squared)  # type: ignore[arg-type]
        torch.testing.assert_close(norm, ref_norm, rtol=1e-5, atol=1e-6)

    def test_multi_tensor_clip_function(self):
        """Test multi_tensor_clip_grad_norm convenience function."""
        model = nn.Linear(10, 10).to(self.device)

        # Create gradients
        for param in model.parameters():
            param.grad = torch.randn_like(param) * 10.0

        # Clip using convenience function
        clip_stats = multi_tensor_clip_grad_norm(model, max_norm=1.0)

        assert "total_norm" in clip_stats
        assert "was_clipped" in clip_stats

    def test_multi_tensor_scale_function(self):
        """Test multi_tensor_scale convenience function."""
        tensors = [torch.ones(100, device=self.device) for _ in range(5)]

        scaled = multi_tensor_scale(tensors, 2.0)

        for t in scaled:
            assert torch.allclose(t, torch.ones_like(t) * 2.0)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Setup test environment."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_single_tensor(self):
        """Test operations with single tensor."""
        operator = MultiTensorOperator(device=self.device)

        tensor = torch.randn(100, device=self.device)
        norm = operator.calculate_norm([tensor], norm_type=2.0)

        ref_norm = torch.norm(tensor, p=2.0)
        torch.testing.assert_close(norm, ref_norm)

    def test_very_large_tensors(self):
        """Test with very large tensors."""
        operator = MultiTensorOperator(device=self.device)

        # Create large tensor that exceeds chunk size
        large_tensor = torch.randn(10**7, device=self.device)

        norm = operator.calculate_norm([large_tensor], norm_type=2.0)
        assert isinstance(norm, torch.Tensor)
        assert torch.isfinite(norm)

    def test_zero_tensors(self):
        """Test with zero tensors."""
        operator = MultiTensorOperator(device=self.device)

        tensors = [torch.zeros(100, device=self.device) for _ in range(5)]

        norm = operator.calculate_norm(tensors, norm_type=2.0)
        assert isinstance(norm, torch.Tensor)
        assert norm.item() == 0.0

        # Clipping should not affect zero gradients
        clip_stats = operator.clip_grad_norm(tensors, max_norm=1.0)
        assert not clip_stats["was_clipped"]

    def test_mixed_devices_error(self):
        """Test that mixed devices are handled properly."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        operator = MultiTensorOperator()

        # Create tensors on different devices
        tensors = [
            torch.randn(100, device="cpu"),
            torch.randn(100, device="cuda"),
        ]

        # Should handle gracefully (group by device internally)
        norm = operator.calculate_norm(tensors, norm_type=2.0)
        assert isinstance(norm, torch.Tensor)
        assert torch.isfinite(norm)


class TestMemoryEfficiency:
    """Test memory efficiency of operations."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_memory_usage(self):
        """Test that operations are memory efficient."""
        device = torch.device("cuda")
        operator = MultiTensorOperator(device=device)

        # Create large tensors
        num_tensors = 100
        tensor_size = 10000
        tensors = [torch.randn(tensor_size, device=device) for _ in range(num_tensors)]

        # Measure memory before
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()

        # Perform operations
        for _ in range(10):
            _ = operator.calculate_norm(tensors, norm_type=2.0)
            operator.scale_tensors(tensors, 0.99, in_place=True)

        # Measure memory after
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated()

        # Memory increase should be minimal (just temporary buffers)
        mem_increase = mem_after - mem_before
        expected_increase = tensor_size * 4 * 2  # Roughly 2 temporary buffers

        assert (
            mem_increase < expected_increase
        ), f"Memory increase {mem_increase} exceeds expected {expected_increase}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
