"""
Comprehensive tests for gradient clipping utilities.

Tests cover:
- L2 norm clipping with single and multi-tensor operations
- Value clipping with parameter tracking
- Adaptive clipping based on statistics
- Distributed training with proper norm reduction
- Megatron-LM compatibility mode
- Edge cases and error handling
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from rosellm.rosetrainer.gradient.clip_grads import (
    ClipType,
    GradientClipper,
    clip_grad_norm,
    clip_grad_value,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_size: int = 10, hidden_size: int = 20):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


@pytest.fixture
def simple_model():
    """Create a simple model with gradients."""
    model = SimpleModel()

    # Create dummy input and compute gradients
    x = torch.randn(4, 10)
    output = model(x)
    loss = output.sum()
    loss.backward()

    return model


@pytest.fixture
def large_gradient_model():
    """Create a model with large gradients for clipping tests."""
    model = SimpleModel()

    # Create large gradients
    for param in model.parameters():
        param.grad = torch.randn_like(param) * 100

    return model


class TestGradientClipper:
    """Test GradientClipper class."""

    def test_init_default(self):
        """Test default initialization."""
        clipper = GradientClipper()

        assert clipper.max_norm is None
        assert clipper.max_value is None
        assert clipper.clip_type == ClipType.NORM
        assert clipper.use_multi_tensor is True
        assert clipper.megatron_compatible is False
        assert clipper.check_for_nan_in_grad is False
        assert clipper.log_stats is False

    def test_init_custom(self):
        """Test custom initialization."""
        clipper = GradientClipper(
            max_norm=1.0,
            max_value=0.5,
            clip_type=ClipType.VALUE,
            use_multi_tensor=False,
            megatron_compatible=True,
            check_for_nan_in_grad=True,
            log_stats=True,
        )

        assert clipper.max_norm == 1.0
        assert clipper.max_value == 0.5
        assert clipper.clip_type == ClipType.VALUE
        assert clipper.use_multi_tensor is False
        assert clipper.megatron_compatible is True
        assert clipper.check_for_nan_in_grad is True
        assert clipper.log_stats is True

    def test_norm_clipping_basic(self, simple_model):
        """Test basic L2 norm clipping."""
        clipper = GradientClipper(max_norm=1.0, clip_type=ClipType.NORM)

        # Compute original norm
        original_norm = 0.0
        for param in simple_model.parameters():
            if param.grad is not None:
                original_norm += param.grad.data.norm() ** 2
        original_norm = original_norm**0.5

        # Clip gradients
        stats = clipper.clip_gradients(simple_model)

        # Check statistics
        assert "total_norm" in stats
        assert "clip_coef" in stats
        assert "clipped" in stats
        assert "max_norm" in stats

        # Verify norm is clipped
        if original_norm > 1.0:
            assert stats["clipped"] is True

            # Compute new norm
            new_norm = 0.0
            for param in simple_model.parameters():
                if param.grad is not None:
                    new_norm += param.grad.data.norm() ** 2
            new_norm = new_norm**0.5

            # Should be approximately max_norm
            assert abs(new_norm - 1.0) < 1e-5

    def test_norm_clipping_large_gradients(self, large_gradient_model):
        """Test norm clipping with large gradients."""
        clipper = GradientClipper(max_norm=1.0, clip_type=ClipType.NORM)

        # Clip gradients
        stats = clipper.clip_gradients(large_gradient_model)

        # Should definitely be clipped
        assert stats["clipped"] is True
        assert stats["clip_coef"] < 1.0

        # Verify final norm
        final_norm = 0.0
        for param in large_gradient_model.parameters():
            if param.grad is not None:
                final_norm += param.grad.data.norm() ** 2
        final_norm = final_norm**0.5

        assert abs(final_norm - 1.0) < 1e-5

    def test_value_clipping_basic(self, simple_model):
        """Test basic value clipping."""
        clipper = GradientClipper(max_value=0.1, clip_type=ClipType.VALUE)

        # Clip gradients
        stats = clipper.clip_gradients(simple_model)

        # Check statistics
        assert "max_grad" in stats
        assert "num_clipped" in stats
        assert "clipped" in stats
        assert "max_value" in stats

        # Verify all gradients are within bounds
        for param in simple_model.parameters():
            if param.grad is not None:
                assert param.grad.abs().max() <= 0.1

    def test_value_clipping_large_gradients(self, large_gradient_model):
        """Test value clipping with large gradients."""
        clipper = GradientClipper(max_value=1.0, clip_type=ClipType.VALUE)

        # Clip gradients
        stats = clipper.clip_gradients(large_gradient_model)

        # Should have clipped parameters
        assert stats["clipped"] is True
        assert stats["num_clipped"] > 0

        # Verify all gradients are within bounds
        for param in large_gradient_model.parameters():
            if param.grad is not None:
                assert param.grad.abs().max() <= 1.0

    def test_adaptive_clipping(self, simple_model):
        """Test adaptive gradient clipping."""
        clipper = GradientClipper(
            max_norm=10.0,  # High threshold for testing
            clip_type=ClipType.ADAPTIVE,
        )

        # Clip gradients
        stats = clipper.clip_gradients(simple_model)

        # Check statistics
        assert "mean_norm" in stats
        assert "std_norm" in stats
        assert "adaptive_threshold" in stats
        assert "total_norm" in stats
        assert "clipped" in stats

        # Adaptive threshold should be computed
        expected_threshold = stats["mean_norm"] + 2 * stats["std_norm"]
        assert abs(stats["adaptive_threshold"] - expected_threshold) < 1e-5

    def test_parameter_list_input(self, simple_model):
        """Test clipping with parameter list input."""
        clipper = GradientClipper(max_norm=1.0, clip_type=ClipType.NORM)

        # Get parameter list
        params = list(simple_model.parameters())

        # Clip gradients
        stats = clipper.clip_gradients(params)

        assert "total_norm" in stats
        assert stats["max_norm"] == 1.0

    def test_no_gradients(self):
        """Test handling of parameters without gradients."""
        model = SimpleModel()  # No backward pass
        clipper = GradientClipper(max_norm=1.0, clip_type=ClipType.NORM)

        stats = clipper.clip_gradients(model)

        assert stats["total_norm"] == 0.0
        assert stats["clipped"] is False

    def test_nan_gradient_check(self, simple_model):
        """Test NaN gradient detection."""
        # Inject NaN gradient
        param = next(simple_model.parameters())
        param.grad[0, 0] = float("nan")

        clipper = GradientClipper(
            max_norm=1.0,
            clip_type=ClipType.NORM,
            check_for_nan_in_grad=True,
        )

        with pytest.raises(ValueError, match="NaN gradient found"):
            clipper.clip_gradients(simple_model)

    def test_invalid_max_norm(self, simple_model):
        """Test error on invalid max_norm."""
        clipper = GradientClipper(max_norm=-1.0, clip_type=ClipType.NORM)

        with pytest.raises(ValueError, match="max_norm must be positive"):
            clipper.clip_gradients(simple_model)

    def test_invalid_max_value(self, simple_model):
        """Test error on invalid max_value."""
        clipper = GradientClipper(max_value=-1.0, clip_type=ClipType.VALUE)

        with pytest.raises(ValueError, match="max_value must be positive"):
            clipper.clip_gradients(simple_model)

    def test_multi_tensor_fallback(self, simple_model):
        """Test fallback when multi-tensor ops fail."""
        with patch(
            "rosellm.rosetrainer.gradient.clip_grads.get_default_operator"
        ) as mock_get:
            # Simulate failure to get operator
            mock_get.side_effect = Exception("No multi-tensor support")

            clipper = GradientClipper(
                max_norm=1.0,
                clip_type=ClipType.NORM,
                use_multi_tensor=True,
            )

            # Should fall back to single tensor
            assert clipper.use_multi_tensor is False
            assert clipper.operator is None

            # Should still work
            stats = clipper.clip_gradients(simple_model)
            assert "total_norm" in stats

    def test_gradient_grouping(self, simple_model):
        """Test gradient grouping by dtype and device."""
        clipper = GradientClipper(max_norm=1.0, clip_type=ClipType.NORM)

        params = list(simple_model.parameters())
        grouped = clipper._group_gradients(params)

        # Should group by (dtype, device)
        for key, grads in grouped.items():
            dtype, device = key
            assert isinstance(dtype, torch.dtype)
            assert isinstance(device, torch.device)
            assert all(g.dtype == dtype for g in grads)
            assert all(g.device == device for g in grads)

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.all_reduce")
    def test_distributed_norm_aggregation(
        self, mock_all_reduce, mock_is_init, simple_model
    ):
        """Test distributed norm aggregation."""
        mock_is_init.return_value = True

        # Create mock process group
        mock_group = MagicMock()

        clipper = GradientClipper(
            max_norm=1.0,
            clip_type=ClipType.NORM,
            model_parallel_group=mock_group,
        )

        # Test norm aggregation
        norm = 2.0
        _ = clipper._aggregate_norm_distributed(norm)

        # Should call all_reduce
        mock_all_reduce.assert_called_once()
        call_args = mock_all_reduce.call_args
        assert call_args[1]["group"] == mock_group

    def test_statistics_logging(self, simple_model, capsys):
        """Test statistics logging."""
        clipper = GradientClipper(
            max_norm=1.0,
            clip_type=ClipType.NORM,
            log_stats=True,
        )

        stats = clipper.clip_gradients(simple_model)

        # Check console output
        captured = capsys.readouterr()
        assert "Gradient norm:" in captured.out
        assert "Clipped:" in captured.out
        assert "Scale:" in captured.out

        # Check stored stats
        assert clipper.stats["total_norm"] == stats["total_norm"]
        assert clipper.stats["clipped"] == stats["clipped"]


class TestClipGradNorm:
    """Test clip_grad_norm function."""

    def test_basic_usage(self, simple_model):
        """Test basic usage compatible with PyTorch."""
        original_norm = 0.0
        for param in simple_model.parameters():
            if param.grad is not None:
                original_norm += param.grad.data.norm() ** 2
        original_norm = original_norm**0.5

        # Clip gradients
        total_norm = clip_grad_norm(simple_model.parameters(), max_norm=1.0)

        # Should return original norm
        assert abs(total_norm - original_norm) < 1e-5

        # Check clipped norm
        clipped_norm = 0.0
        for param in simple_model.parameters():
            if param.grad is not None:
                clipped_norm += param.grad.data.norm() ** 2
        clipped_norm = clipped_norm**0.5

        if original_norm > 1.0:
            assert abs(clipped_norm - 1.0) < 1e-5

    def test_model_input(self, simple_model):
        """Test with model as input."""
        total_norm = clip_grad_norm(simple_model, max_norm=1.0)

        assert isinstance(total_norm, float)
        assert total_norm >= 0

    def test_error_on_nonfinite(self, simple_model):
        """Test error_if_nonfinite parameter."""
        # Inject inf gradient
        param = next(simple_model.parameters())
        param.grad[0, 0] = float("inf")

        with pytest.raises(RuntimeError, match="Gradient norm is"):
            clip_grad_norm(
                simple_model.parameters(),
                max_norm=1.0,
                error_if_nonfinite=True,
            )

    def test_non_l2_norm_warning(self, simple_model):
        """Test warning for non-L2 norms."""
        with pytest.warns(UserWarning, match="Only L2 norm"):
            # Should fall back to PyTorch implementation
            total_norm = clip_grad_norm(
                simple_model.parameters(),
                max_norm=1.0,
                norm_type=1.0,  # L1 norm
            )

        assert isinstance(total_norm, (float, torch.Tensor))

    @patch("torch.distributed.is_initialized")
    def test_with_model_parallel_group(self, mock_is_init, simple_model):
        """Test with model parallel group."""
        mock_is_init.return_value = False  # Avoid actual distributed ops

        mock_group = MagicMock()

        total_norm = clip_grad_norm(
            simple_model.parameters(),
            max_norm=1.0,
            model_parallel_group=mock_group,
        )

        assert isinstance(total_norm, float)


class TestClipGradValue:
    """Test clip_grad_value function."""

    def test_basic_usage(self, simple_model):
        """Test basic value clipping."""
        num_clipped = clip_grad_value(simple_model.parameters(), clip_value=0.1)

        # Verify all gradients are within bounds
        for param in simple_model.parameters():
            if param.grad is not None:
                assert param.grad.abs().max() <= 0.1

        assert isinstance(num_clipped, int)
        assert num_clipped >= 0

    def test_model_input(self, simple_model):
        """Test with model as input."""
        num_clipped = clip_grad_value(simple_model, clip_value=0.1)

        # Verify all gradients are within bounds
        for param in simple_model.parameters():
            if param.grad is not None:
                assert param.grad.abs().max() <= 0.1

        assert isinstance(num_clipped, int)

    def test_large_gradients(self, large_gradient_model):
        """Test with large gradients."""
        num_clipped = clip_grad_value(large_gradient_model, clip_value=1.0)

        # Should have clipped all parameters
        assert num_clipped > 0

        # Verify all gradients are within bounds
        for param in large_gradient_model.parameters():
            if param.grad is not None:
                assert param.grad.abs().max() <= 1.0


class TestIntegration:
    """Integration tests with real scenarios."""

    def test_training_loop_simulation(self):
        """Simulate a training loop with gradient clipping."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Training loop
        for _ in range(5):
            # Forward pass
            x = torch.randn(8, 10)
            output = model(x)
            loss = output.sum()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            total_norm = clip_grad_norm(list(model.parameters()), max_norm=1.0)

            # Optimizer step
            optimizer.step()

            # Verify norm was computed
            assert isinstance(total_norm, float)
            assert total_norm >= 0

    def test_mixed_precision_compatibility(self):
        """Test compatibility with mixed precision training."""
        model = SimpleModel()

        # Convert model to half precision
        model = model.half()

        # Create gradients in half precision
        x = torch.randn(4, 10, dtype=torch.float16)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Clip gradients
        clipper = GradientClipper(max_norm=1.0, clip_type=ClipType.NORM)
        stats = clipper.clip_gradients(model)

        assert "total_norm" in stats
        assert stats["max_norm"] == 1.0

        # Verify gradients are still half precision
        for param in model.parameters():
            if param.grad is not None:
                assert param.grad.dtype == torch.float16

    def test_multiple_devices_simulation(self):
        """Test gradient grouping with multiple device types."""
        model1 = SimpleModel()
        model2 = SimpleModel()

        # Create gradients
        for model in [model1, model2]:
            x = torch.randn(4, 10)
            output = model(x)
            loss = output.sum()
            loss.backward()

        # Combine parameters from both models
        all_params = list(model1.parameters()) + list(model2.parameters())

        # Clip combined gradients
        clipper = GradientClipper(max_norm=1.0, clip_type=ClipType.NORM)
        stats = clipper.clip_gradients(all_params)

        assert "total_norm" in stats
        assert "clipped" in stats

    def test_gradient_accumulation_scenario(self):
        """Test clipping with gradient accumulation."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        accumulation_steps = 4

        # Accumulate gradients
        for step in range(accumulation_steps):
            x = torch.randn(2, 10)
            output = model(x)
            loss = output.sum() / accumulation_steps
            loss.backward()

        # Clip accumulated gradients
        total_norm = clip_grad_norm(list(model.parameters()), max_norm=1.0)

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        assert isinstance(total_norm, float)
        assert total_norm >= 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_max_norm(self, simple_model):
        """Test with zero max_norm."""
        clipper = GradientClipper(max_norm=0, clip_type=ClipType.NORM)

        with pytest.raises(ValueError, match="max_norm must be positive"):
            clipper.clip_gradients(simple_model)

    def test_infinite_max_norm(self, simple_model):
        """Test with infinite max_norm."""
        clipper = GradientClipper(max_norm=float("inf"), clip_type=ClipType.NORM)

        # Should not clip anything
        stats = clipper.clip_gradients(simple_model)
        assert stats["clipped"] is False

    def test_single_parameter(self):
        """Test with single parameter."""
        param = nn.Parameter(torch.randn(10, 10))
        param.grad = torch.randn_like(param)

        clipper = GradientClipper(max_norm=1.0, clip_type=ClipType.NORM)
        stats = clipper.clip_gradients([param])

        assert "total_norm" in stats

    def test_empty_model(self):
        """Test with model without parameters."""
        model = nn.Module()  # Empty module

        clipper = GradientClipper(max_norm=1.0, clip_type=ClipType.NORM)
        stats = clipper.clip_gradients(model)

        assert stats["total_norm"] == 0.0
        assert stats["clipped"] is False

    def test_mixed_gradient_presence(self):
        """Test with some parameters having gradients and others not."""
        model = SimpleModel()

        # Only compute gradients for first layer
        x = torch.randn(4, 10)
        output = model.linear1(x)
        loss = output.sum()
        loss.backward()

        # linear2 parameters won't have gradients
        clipper = GradientClipper(max_norm=1.0, clip_type=ClipType.NORM)
        stats = clipper.clip_gradients(model)

        assert "total_norm" in stats
        assert stats["total_norm"] > 0  # Should have some norm from linear1
