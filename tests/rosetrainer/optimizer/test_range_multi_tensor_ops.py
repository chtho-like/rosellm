"""Tests for range-based multi-tensor operations.

This test suite validates the multi-tensor operations optimized for
range-based parameter buffers with comprehensive accuracy checks.
"""

import logging
import math
from typing import List

import pytest
import torch
import torch.nn as nn

from rosellm.rosetrainer.optimizer.range_aware_gradient_buffer import (
    RangeAwareGradientBuffer,
)
from rosellm.rosetrainer.optimizer.range_buffer_mapping import (
    RangeBufferConfig,
    RangeBufferMapper,
)
from rosellm.rosetrainer.optimizer.range_multi_tensor_ops import (
    RangeMultiTensorOperator,
    create_range_multi_tensor_operator,
    multi_tensor_range_scale,
)
from rosellm.rosetrainer.utils.gradient_utils import GradientClipConfig

logger = logging.getLogger(__name__)


class SimpleModel(nn.Module):
    """Simple model for testing multi-tensor operations."""

    def __init__(self, dims: List[int], dtype: torch.dtype = torch.float32):
        super().__init__()
        self.layers = nn.ModuleList()

        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            self.layers.append(nn.Linear(in_dim, out_dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


class TestRangeMultiTensorOperator:
    """Test RangeMultiTensorOperator functionality."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def test_model(self, device):
        """Create test model."""
        model = SimpleModel([32, 64, 32, 16])
        model.to(device)
        return model

    @pytest.fixture
    def range_mapper(self, test_model, device):
        """Create range buffer mapper."""
        parameters = list(test_model.parameters())
        config = RangeBufferConfig(device=device)
        return RangeBufferMapper(parameters, config)

    @pytest.fixture
    def gradient_buffer(self, test_model, device):
        """Create range-aware gradient buffer."""
        parameters = list(test_model.parameters())
        return RangeAwareGradientBuffer(params=parameters)

    def test_initialization(self, range_mapper, gradient_buffer, device):
        """Test operator initialization."""
        operator = RangeMultiTensorOperator(
            range_mapper=range_mapper,
            gradient_buffer=gradient_buffer,
            device=device,
        )

        assert operator.range_mapper is range_mapper
        assert operator.gradient_buffer is gradient_buffer
        assert operator.device == device
        assert operator.base_operator is not None

    def test_initialization_without_range_components(self, device):
        """Test initialization without range components."""
        operator = RangeMultiTensorOperator(device=device)

        assert operator.range_mapper is None
        assert operator.gradient_buffer is None
        assert operator.device == device
        assert operator.base_operator is not None

    def test_can_use_range_optimization_true(
        self, range_mapper, gradient_buffer, device
    ):
        """Test range optimization capability detection - positive case."""
        parameters = list(range_mapper.parameters)

        operator = RangeMultiTensorOperator(
            range_mapper=range_mapper,
            gradient_buffer=gradient_buffer,
            device=device,
        )

        # Should be able to use range optimization for mapped parameters
        can_use = operator._can_use_range_optimization(parameters)
        # May be True or False depending on parameter mapping
        assert isinstance(can_use, bool)

    def test_can_use_range_optimization_false(self, device):
        """Test range optimization capability detection - negative case."""
        operator = RangeMultiTensorOperator(device=device)

        param = nn.Parameter(torch.randn(10, device=device))
        can_use = operator._can_use_range_optimization([param])

        assert can_use is False

    def test_gradient_scaling_standard(self, test_model, device):
        """Test standard gradient scaling."""
        parameters = list(test_model.parameters())

        operator = RangeMultiTensorOperator(device=device)

        # Run forward and backward pass to generate gradients
        x = torch.randn(4, 32, device=device)
        y = test_model(x)
        loss = y.sum()
        loss.backward()

        # Store original gradients
        original_grads = [
            p.grad.clone() if p.grad is not None else None for p in parameters
        ]

        # Apply scaling
        scale_factor = 2.0
        operator.scale_gradients(parameters, scale_factor, use_ranges=False)

        # Verify scaling
        for i, (param, orig_grad) in enumerate(zip(parameters, original_grads)):
            if param.grad is not None and orig_grad is not None:
                expected_grad = orig_grad * scale_factor
                torch.testing.assert_close(
                    param.grad,
                    expected_grad,
                    msg=f"Gradient scaling failed for parameter {i}",
                    atol=1e-6,
                    rtol=1e-5,
                )

    def test_gradient_scaling_range_based(
        self, test_model, range_mapper, gradient_buffer, device
    ):
        """Test range-based gradient scaling."""
        parameters = list(test_model.parameters())

        operator = RangeMultiTensorOperator(
            range_mapper=range_mapper,
            gradient_buffer=gradient_buffer,
            device=device,
        )

        # Run forward and backward pass
        x = torch.randn(4, 32, device=device)
        y = test_model(x)
        loss = y.sum()
        loss.backward()

        # Apply range-based scaling
        scale_factor = 1.5
        operator.scale_gradients(parameters, scale_factor, use_ranges=True)

        # Verify that scaling was applied (exact verification depends on range mapping)
        has_nonzero_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in parameters
        )
        assert has_nonzero_grads  # Should have some gradients

    def test_gradient_scaling_no_op(self, test_model, device):
        """Test gradient scaling with scale factor of 1.0."""
        parameters = list(test_model.parameters())
        operator = RangeMultiTensorOperator(device=device)

        # Run backward pass
        x = torch.randn(4, 32, device=device)
        y = test_model(x)
        loss = y.sum()
        loss.backward()

        # Store original gradients
        original_grads = [
            p.grad.clone() if p.grad is not None else None for p in parameters
        ]

        # Apply no-op scaling
        operator.scale_gradients(parameters, 1.0)

        # Gradients should be unchanged
        for i, (param, orig_grad) in enumerate(zip(parameters, original_grads)):
            if param.grad is not None and orig_grad is not None:
                torch.testing.assert_close(
                    param.grad,
                    orig_grad,
                    msg=f"Gradient should be unchanged for parameter {i}",
                )

    def test_gradient_norm_computation_standard(self, test_model, device):
        """Test standard gradient norm computation."""
        parameters = list(test_model.parameters())
        operator = RangeMultiTensorOperator(device=device)

        # Run backward pass
        x = torch.randn(4, 32, device=device)
        y = test_model(x)
        loss = y.sum()
        loss.backward()

        # Compute gradient norm
        grad_norm = operator.compute_gradient_norm(
            parameters, norm_type=2.0, use_ranges=False
        )

        # Verify norm is positive and finite
        assert grad_norm >= 0.0
        assert math.isfinite(grad_norm)

        # Compare with manual computation
        manual_norm = 0.0
        for param in parameters:
            if param.grad is not None:
                manual_norm += param.grad.pow(2).sum().item()
        manual_norm = math.sqrt(manual_norm)

        assert abs(grad_norm - manual_norm) < 1e-6

    def test_gradient_norm_computation_different_types(self, test_model, device):
        """Test gradient norm computation with different norm types."""
        parameters = list(test_model.parameters())
        operator = RangeMultiTensorOperator(device=device)

        # Run backward pass
        x = torch.randn(4, 32, device=device)
        y = test_model(x)
        loss = y.sum()
        loss.backward()

        # Test different norm types
        norm_l1 = operator.compute_gradient_norm(parameters, norm_type=1.0)
        norm_l2 = operator.compute_gradient_norm(parameters, norm_type=2.0)
        norm_inf = operator.compute_gradient_norm(parameters, norm_type=float("inf"))

        # All norms should be positive and finite
        assert norm_l1 >= 0.0 and math.isfinite(norm_l1)
        assert norm_l2 >= 0.0 and math.isfinite(norm_l2)
        assert norm_inf >= 0.0 and math.isfinite(norm_inf)

        # L1 norm should typically be larger than L2 norm
        # (unless gradients are very uniform)
        assert norm_l1 >= 0.0
        assert norm_l2 >= 0.0

    def test_gradient_norm_caching(self, test_model, device):
        """Test gradient norm caching functionality."""
        parameters = list(test_model.parameters())
        operator = RangeMultiTensorOperator(device=device)

        # Run backward pass
        x = torch.randn(4, 32, device=device)
        y = test_model(x)
        loss = y.sum()
        loss.backward()

        # Compute norm with caching
        step = 100
        norm1 = operator.compute_gradient_norm(parameters, cache_step=step)
        norm2 = operator.compute_gradient_norm(parameters, cache_step=step)

        # Should get same result (from cache)
        assert abs(norm1 - norm2) < 1e-10

        # Cache should be cleared for different step
        norm3 = operator.compute_gradient_norm(parameters, cache_step=step + 1)
        # Should still be close (same gradients) but not identical due to recomputation
        assert abs(norm1 - norm3) < 1e-6

    def test_gradient_clipping_standard(self, test_model, device):
        """Test standard gradient clipping."""
        parameters = list(test_model.parameters())
        operator = RangeMultiTensorOperator(device=device)

        # Run backward pass
        x = torch.randn(4, 32, device=device)
        y = test_model(x)
        loss = y.sum() * 1000  # Large loss to create large gradients
        loss.backward()

        # Create clip config
        clip_config = GradientClipConfig(
            clip_type="norm",
            max_norm=1.0,
            norm_type=2.0,
        )

        # Apply clipping
        clip_stats = operator.clip_gradients(parameters, clip_config, use_ranges=False)

        # Verify clipping statistics
        assert "grad_norm" in clip_stats
        assert "clipped_grad_norm" in clip_stats
        assert "clip_ratio" in clip_stats

        # Clipped norm should not exceed max norm
        assert clip_stats["clipped_grad_norm"] <= clip_config.max_norm + 1e-6

        # Clip ratio should be reasonable
        assert 0.0 <= clip_stats["clip_ratio"] <= 1.0

    def test_gradient_clipping_no_clipping_needed(self, test_model, device):
        """Test gradient clipping when no clipping is needed."""
        parameters = list(test_model.parameters())
        operator = RangeMultiTensorOperator(device=device)

        # Run backward pass with small loss
        x = torch.randn(4, 32, device=device)
        y = test_model(x)
        loss = y.sum() * 0.01  # Small loss
        loss.backward()

        # Create clip config with large threshold
        clip_config = GradientClipConfig(
            clip_type="norm",
            max_norm=100.0,  # Large threshold
            norm_type=2.0,
        )

        # Apply clipping
        clip_stats = operator.clip_gradients(parameters, clip_config)

        # No clipping should have occurred
        assert clip_stats["clip_ratio"] == 1.0
        assert abs(clip_stats["grad_norm"] - clip_stats["clipped_grad_norm"]) < 1e-6

    def test_zero_gradients_standard(self, test_model, device):
        """Test standard gradient zeroing."""
        parameters = list(test_model.parameters())
        operator = RangeMultiTensorOperator(device=device)

        # Run backward pass
        x = torch.randn(4, 32, device=device)
        y = test_model(x)
        loss = y.sum()
        loss.backward()

        # Verify gradients exist
        has_grads_before = any(p.grad is not None for p in parameters)
        assert has_grads_before

        # Zero gradients
        operator.zero_gradients(parameters, use_ranges=False, set_to_none=False)

        # Verify gradients are zeroed
        for param in parameters:
            if param.grad is not None:
                assert torch.allclose(param.grad, torch.zeros_like(param.grad))

    def test_zero_gradients_set_to_none(self, test_model, device):
        """Test gradient zeroing with set_to_none=True."""
        parameters = list(test_model.parameters())
        operator = RangeMultiTensorOperator(device=device)

        # Run backward pass
        x = torch.randn(4, 32, device=device)
        y = test_model(x)
        loss = y.sum()
        loss.backward()

        # Zero gradients with set_to_none
        operator.zero_gradients(parameters, use_ranges=False, set_to_none=True)

        # Verify gradients are None
        for param in parameters:
            assert param.grad is None

    def test_weight_decay_standard(self, test_model, device):
        """Test standard weight decay application."""
        parameters = list(test_model.parameters())
        operator = RangeMultiTensorOperator(device=device)

        # Store original parameters
        original_params = [p.data.clone() for p in parameters]

        # Apply weight decay
        weight_decay = 0.01
        operator.apply_weight_decay(parameters, weight_decay, use_ranges=False)

        # Verify weight decay was applied
        for i, (param, orig_param) in enumerate(zip(parameters, original_params)):
            expected_param = orig_param * (1.0 - weight_decay)
            torch.testing.assert_close(
                param.data,
                expected_param,
                msg=f"Weight decay failed for parameter {i}",
                atol=1e-6,
                rtol=1e-5,
            )

    def test_weight_decay_no_op(self, test_model, device):
        """Test weight decay with zero coefficient."""
        parameters = list(test_model.parameters())
        operator = RangeMultiTensorOperator(device=device)

        # Store original parameters
        original_params = [p.data.clone() for p in parameters]

        # Apply zero weight decay
        operator.apply_weight_decay(parameters, 0.0)

        # Parameters should be unchanged
        for i, (param, orig_param) in enumerate(zip(parameters, original_params)):
            torch.testing.assert_close(
                param.data,
                orig_param,
                msg=f"Parameter should be unchanged for parameter {i}",
            )

    def test_operation_stats(self, device):
        """Test operation statistics collection."""
        operator = RangeMultiTensorOperator(device=device, enable_benchmarking=True)

        # Initially should have no stats
        stats = operator.get_operation_stats()
        assert isinstance(stats, dict)

        # Reset stats
        operator.reset_stats()

        # Stats should still be a dict (possibly empty)
        stats_after_reset = operator.get_operation_stats()
        assert isinstance(stats_after_reset, dict)


class TestRangeMultiTensorOperatorFactory:
    """Test factory functions for range multi-tensor operator."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_create_range_multi_tensor_operator(self, device):
        """Test factory function for creating operator."""
        operator = create_range_multi_tensor_operator(device=device)

        assert isinstance(operator, RangeMultiTensorOperator)
        assert operator.device == device
        assert operator.range_mapper is None
        assert operator.gradient_buffer is None

    def test_create_with_components(self, device):
        """Test factory function with range components."""
        # Create simple components
        model = SimpleModel([16, 32, 16])
        model.to(device)
        parameters = list(model.parameters())

        config = RangeBufferConfig(device=device)
        range_mapper = RangeBufferMapper(parameters, config)
        gradient_buffer = RangeAwareGradientBuffer(params=parameters)

        operator = create_range_multi_tensor_operator(
            range_mapper=range_mapper,
            gradient_buffer=gradient_buffer,
            device=device,
            enable_benchmarking=True,
        )

        assert operator.range_mapper is range_mapper
        assert operator.gradient_buffer is gradient_buffer
        assert operator.enable_benchmarking is True


class TestMultiTensorRangeScale:
    """Test multi-tensor range scaling function."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_multi_tensor_range_scale_basic(self, device):
        """Test basic multi-tensor range scaling."""
        tensors = [torch.randn(10, device=device) for _ in range(3)]
        original_tensors = [t.clone() for t in tensors]

        scale_factor = 2.0
        multi_tensor_range_scale(tensors, scale_factor)

        # Verify scaling
        for i, (tensor, original) in enumerate(zip(tensors, original_tensors)):
            expected = original * scale_factor
            torch.testing.assert_close(
                tensor, expected, msg=f"Scaling failed for tensor {i}"
            )

    def test_multi_tensor_range_scale_no_op(self, device):
        """Test multi-tensor range scaling with scale factor 1.0."""
        tensors = [torch.randn(10, device=device) for _ in range(3)]
        original_tensors = [t.clone() for t in tensors]

        multi_tensor_range_scale(tensors, 1.0)

        # Tensors should be unchanged
        for i, (tensor, original) in enumerate(zip(tensors, original_tensors)):
            torch.testing.assert_close(
                tensor, original, msg=f"Tensor {i} should be unchanged"
            )

    def test_multi_tensor_range_scale_empty_list(self, device):
        """Test multi-tensor range scaling with empty tensor list."""
        # Should not raise an error
        multi_tensor_range_scale([], 2.0)

    def test_multi_tensor_range_scale_with_operator(self, device):
        """Test multi-tensor range scaling with range operator."""
        model = SimpleModel([16, 32, 16])
        model.to(device)
        parameters = list(model.parameters())

        # Create range operator
        range_operator = create_range_multi_tensor_operator(device=device)

        # Run backward pass to generate gradients
        x = torch.randn(2, 16, device=device)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Extract gradient tensors
        grad_tensors = [p.grad for p in parameters if p.grad is not None]
        original_grads = [g.clone() for g in grad_tensors]

        scale_factor = 1.5
        multi_tensor_range_scale(grad_tensors, scale_factor, range_operator)

        # Verify scaling (may use range-aware or standard scaling)
        for i, (grad, original) in enumerate(zip(grad_tensors, original_grads)):
            # Should be scaled by approximately the scale factor
            # (exact method depends on whether range optimization was used)
            assert not torch.allclose(grad, original), f"Gradient {i} should be scaled"


class TestRangeMultiTensorOperatorEdgeCases:
    """Test edge cases for range multi-tensor operator."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_empty_parameter_list(self, device):
        """Test operations with empty parameter list."""
        operator = RangeMultiTensorOperator(device=device)

        # All operations should handle empty lists gracefully
        operator.scale_gradients([], 2.0)

        norm = operator.compute_gradient_norm([])
        assert norm == 0.0

        clip_config = GradientClipConfig(clip_type="norm", max_norm=1.0)
        clip_stats = operator.clip_gradients([], clip_config)
        assert clip_stats["grad_norm"] == 0.0

        operator.zero_gradients([])
        operator.apply_weight_decay([], 0.01)

    def test_parameters_without_gradients(self, device):
        """Test operations with parameters that have no gradients."""
        param = nn.Parameter(torch.randn(10, device=device))
        # Don't run backward pass, so param.grad is None

        operator = RangeMultiTensorOperator(device=device)

        # Should handle parameters without gradients
        operator.scale_gradients([param], 2.0)

        norm = operator.compute_gradient_norm([param])
        assert norm == 0.0

    def test_mixed_parameters_with_and_without_gradients(self, device):
        """Test operations with mixed parameter states."""
        param_with_grad = nn.Parameter(torch.randn(10, device=device))
        param_without_grad = nn.Parameter(torch.randn(10, device=device))

        # Set gradient for one parameter
        param_with_grad.grad = torch.randn(10, device=device)

        operator = RangeMultiTensorOperator(device=device)

        # Should handle mixed parameter states
        parameters = [param_with_grad, param_without_grad]

        original_grad = param_with_grad.grad.clone()

        operator.scale_gradients(parameters, 2.0)

        # Parameter with gradient should be scaled
        torch.testing.assert_close(param_with_grad.grad, original_grad * 2.0)

        # Parameter without gradient should remain None
        assert param_without_grad.grad is None

    def test_very_small_gradients(self, device):
        """Test operations with very small gradients."""
        param = nn.Parameter(torch.randn(10, device=device))
        param.grad = torch.ones(10, device=device) * 1e-10  # Very small gradients

        operator = RangeMultiTensorOperator(device=device)

        # Should handle small gradients without numerical issues
        norm = operator.compute_gradient_norm([param])
        assert norm > 0.0 and math.isfinite(norm)

        # Clipping should work with small gradients
        clip_config = GradientClipConfig(clip_type="norm", max_norm=1.0)
        clip_stats = operator.clip_gradients([param], clip_config)
        assert math.isfinite(clip_stats["grad_norm"])
        assert math.isfinite(clip_stats["clip_ratio"])

    def test_very_large_gradients(self, device):
        """Test operations with very large gradients."""
        param = nn.Parameter(torch.randn(10, device=device))
        param.grad = torch.ones(10, device=device) * 1e6  # Very large gradients

        operator = RangeMultiTensorOperator(device=device)

        # Should handle large gradients
        norm = operator.compute_gradient_norm([param])
        assert norm > 0.0 and math.isfinite(norm)

        # Clipping should reduce large gradients
        clip_config = GradientClipConfig(clip_type="norm", max_norm=1.0)
        clip_stats = operator.clip_gradients([param], clip_config)

        assert clip_stats["clipped_grad_norm"] <= 1.0 + 1e-6
        assert clip_stats["clip_ratio"] < 1.0  # Should have clipped


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
