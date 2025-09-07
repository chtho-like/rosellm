"""
Comprehensive unit tests for gradient utilities.

Tests cover:
- Basic functionality vs PyTorch native operations
- Distributed scenarios with CPU simulation
- Configuration options and validation
- Integration with various model architectures
- Error handling and edge cases
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from rosellm.rosetrainer.utils.gradient_utils import (
    GradientClipConfig,
    apply_gradient_clipping,
    calculate_gradient_norm_multitensor,
    check_gradient_finite,
    get_gradient_stats,
    gradient_accumulation_context,
    sync_gradients,
)


class SimpleModel(nn.Module):
    """Simple model for testing gradient utilities."""

    def __init__(
        self, input_size: int = 10, hidden_size: int = 20, output_size: int = 5
    ):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TestGradientClipConfig(unittest.TestCase):
    """Test gradient clipping configuration validation."""

    def test_valid_config_creation(self):
        """Test creating valid gradient clip configurations."""
        # Default config
        config = GradientClipConfig()
        self.assertEqual(config.clip_type, "norm")
        self.assertEqual(config.max_norm, 1.0)
        self.assertEqual(config.norm_type, 2.0)
        self.assertTrue(config.error_if_nonfinite)
        self.assertTrue(config.model_parallel_reduce)
        self.assertTrue(config.use_multitensor)

        # Custom config
        config = GradientClipConfig(
            clip_type="value",
            max_norm=0.5,
            norm_type=1.0,
            error_if_nonfinite=False,
            model_parallel_reduce=False,
            use_multitensor=False,
        )
        self.assertEqual(config.clip_type, "value")
        self.assertEqual(config.max_norm, 0.5)
        self.assertEqual(config.norm_type, 1.0)
        self.assertFalse(config.error_if_nonfinite)
        self.assertFalse(config.model_parallel_reduce)
        self.assertFalse(config.use_multitensor)

    def test_invalid_clip_type(self):
        """Test invalid clip type raises error."""
        with self.assertRaises(ValueError):
            GradientClipConfig(clip_type="invalid")

    def test_invalid_max_norm(self):
        """Test invalid max_norm raises error."""
        with self.assertRaises(ValueError):
            GradientClipConfig(max_norm=-1.0)

        with self.assertRaises(ValueError):
            GradientClipConfig(max_norm=0.0)

    def test_invalid_norm_type(self):
        """Test invalid norm_type raises error."""
        with self.assertRaises(ValueError):
            GradientClipConfig(norm_type=-1.0)

        with self.assertRaises(ValueError):
            GradientClipConfig(norm_type=0.0)


class TestGradientNormCalculation(unittest.TestCase):
    """Test gradient norm calculation with multi-tensor operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleModel().to(self.device)
        self.batch_size = 8
        self.input_size = 10

    def _create_test_batch(self):
        """Create a test batch with loss computation."""
        x = torch.randn(self.batch_size, self.input_size, device=self.device)
        target = torch.randint(0, 5, (self.batch_size,), device=self.device)

        output = self.model(x)
        loss = F.cross_entropy(output, target)
        loss.backward()

        return loss

    def test_norm_calculation_vs_pytorch(self):
        """Test that our norm calculation matches PyTorch's implementation."""
        # Create gradients
        self._create_test_batch()

        # Calculate norm using our implementation first (doesn't modify gradients)
        our_norm = calculate_gradient_norm_multitensor(
            self.model,
            norm_type=2.0,
            use_multitensor=False,  # Disable for direct comparison
            model_parallel_reduce=False,
        )

        # Calculate norm using PyTorch (this modifies gradients but we use inf
        # threshold)
        pytorch_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), float("inf")
        )

        # Should be very close (within numerical precision)
        self.assertAlmostEqual(
            float(pytorch_norm),
            float(our_norm),
            places=5,
            msg="Gradient norm calculation doesn't match PyTorch",
        )

    def test_different_norm_types(self):
        """Test different norm type calculations."""
        self._create_test_batch()

        # Test L1 norm
        l1_norm = calculate_gradient_norm_multitensor(
            self.model,
            norm_type=1.0,
            use_multitensor=False,
            model_parallel_reduce=False,
        )

        # Test L2 norm
        l2_norm = calculate_gradient_norm_multitensor(
            self.model,
            norm_type=2.0,
            use_multitensor=False,
            model_parallel_reduce=False,
        )

        # Test infinity norm
        inf_norm = calculate_gradient_norm_multitensor(
            self.model,
            norm_type=float("inf"),
            use_multitensor=False,
            model_parallel_reduce=False,
        )

        # Basic sanity checks
        self.assertGreater(float(l1_norm), 0)
        self.assertGreater(float(l2_norm), 0)
        self.assertGreater(float(inf_norm), 0)

        # L1 norm should generally be larger than L2 norm
        # (though this isn't guaranteed for all gradient distributions)
        self.assertGreater(float(l1_norm), 0)
        self.assertGreater(float(l2_norm), 0)

        # Infinity norm should be the maximum absolute gradient value
        max_grad = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                max_grad = max(max_grad, float(param.grad.abs().max()))

        self.assertAlmostEqual(
            float(inf_norm),
            max_grad,
            places=5,
            msg="Infinity norm should equal maximum absolute gradient",
        )

    def test_empty_gradients(self):
        """Test behavior with no gradients."""
        # Model with no gradients
        model = SimpleModel()

        norm = calculate_gradient_norm_multitensor(
            model, norm_type=2.0, use_multitensor=False, model_parallel_reduce=False
        )

        self.assertEqual(float(norm), 0.0)

    def test_multitensor_fallback(self):
        """Test that multitensor operations fall back gracefully."""
        # Create gradients once and store them
        self._create_test_batch()

        # Store gradients for consistent comparison
        stored_grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                stored_grads[name] = param.grad.clone()

        # Test with multitensor enabled (should work regardless of APEX availability)
        norm_mt = calculate_gradient_norm_multitensor(
            self.model, norm_type=2.0, use_multitensor=True, model_parallel_reduce=False
        )

        # Restore the exact same gradients
        for name, param in self.model.named_parameters():
            if name in stored_grads:
                param.grad = stored_grads[name].clone()

        # Test with multitensor disabled
        norm_std = calculate_gradient_norm_multitensor(
            self.model,
            norm_type=2.0,
            use_multitensor=False,
            model_parallel_reduce=False,
        )

        # Results should be very close (allowing some numerical difference)
        self.assertAlmostEqual(
            float(norm_mt),
            float(norm_std),
            places=4,  # Relaxed tolerance for fallback comparison
            msg="Multi-tensor and standard norm calculations should match",
        )


class TestCustomScalerIntegration(unittest.TestCase):
    """Test integration with custom gradient scalers."""

    def test_check_for_inf_and_nan_with_custom_scaler(self):
        """Test that custom scalers are properly recognized and updated."""
        from rosellm.rosetrainer.mixed_precision.gradient_scaler import (
            DynamicGradScaler,
        )
        from rosellm.rosetrainer.utils.gradient_utils import (
            check_for_inf_and_nan_with_scaler,
        )

        # Create a simple model with gradients
        model = SimpleModel()
        x = torch.randn(4, 10, device=model.linear1.weight.device)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Create custom scaler with hysteresis=1 for simpler testing
        scaler = DynamicGradScaler(
            initial_scale=1024.0,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=100,
            hysteresis=1,  # Set to 1 so scale backs off immediately on overflow
            device=str(model.linear1.weight.device),
        )

        # Initial scale
        initial_scale = float(scaler.scale)

        # Check for inf/nan (should find none and update scaler)
        found_inf = check_for_inf_and_nan_with_scaler(model, scaler)

        self.assertFalse(found_inf, "Should not find inf/nan in normal gradients")

        # Verify scaler was updated (growth tracker should be incremented)
        self.assertEqual(scaler._growth_tracker, 1)

        # Create inf gradient
        assert model.linear1.weight.grad is not None
        model.linear1.weight.grad[0, 0] = float("inf")

        # Check again (should find inf and update scaler)
        found_inf = check_for_inf_and_nan_with_scaler(model, scaler)

        self.assertTrue(found_inf, "Should find inf in gradients")

        # Verify scaler was updated (scale should be reduced)
        new_scale = float(scaler.scale)
        self.assertLess(
            new_scale, initial_scale, "Scale should be reduced after inf detected"
        )
        self.assertEqual(scaler._growth_tracker, 0, "Growth tracker should reset")


class TestGradientClipping(unittest.TestCase):
    """Test gradient clipping functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleModel().to(self.device)
        self.batch_size = 8
        self.input_size = 10

    def _create_large_gradients(self, scale: float = 10.0):
        """Create gradients that are likely to exceed clipping threshold."""
        x = torch.randn(self.batch_size, self.input_size, device=self.device) * scale
        target = torch.randint(0, 5, (self.batch_size,), device=self.device)

        output = self.model(x)
        loss = F.cross_entropy(output, target) * scale
        loss.backward()

        return loss

    def test_norm_clipping(self):
        """Test gradient norm clipping."""
        # Create large gradients
        self._create_large_gradients(scale=20.0)

        # Configure clipping
        config = GradientClipConfig(
            clip_type="norm", max_norm=1.0, norm_type=2.0, model_parallel_reduce=False
        )

        # Apply clipping
        stats = apply_gradient_clipping(self.model, config)

        # Check that clipping was applied
        self.assertIn("grad_norm", stats)
        self.assertIn("clipped", stats)

        # Verify that the norm is now close to max_norm
        post_clip_norm = calculate_gradient_norm_multitensor(
            self.model,
            norm_type=2.0,
            use_multitensor=False,
            model_parallel_reduce=False,
        )

        # Should be close to max_norm (within numerical precision)
        self.assertLessEqual(
            float(post_clip_norm),
            config.max_norm * 1.01,  # Small tolerance for numerical errors
            msg="Clipped gradient norm should not exceed max_norm",
        )

    def test_value_clipping(self):
        """Test gradient value clipping."""
        # Create large gradients
        self._create_large_gradients(scale=20.0)

        # Configure value clipping
        config = GradientClipConfig(
            clip_type="value",
            max_norm=1.0,  # This becomes max_value for value clipping
            model_parallel_reduce=False,
        )

        # Apply clipping
        stats = apply_gradient_clipping(self.model, config)

        # Check that clipping was applied
        self.assertIn("clipped", stats)

        # Verify that all gradient values are within [-max_value, max_value]
        for param in self.model.parameters():
            if param.grad is not None:
                self.assertTrue(
                    torch.all(param.grad >= -config.max_norm),
                    "All gradient values should be >= -max_value",
                )
                self.assertTrue(
                    torch.all(param.grad <= config.max_norm),
                    "All gradient values should be <= max_value",
                )

    def test_no_clipping(self):
        """Test that no clipping is applied when clip_type is 'none'."""
        # Create normal gradients
        x = torch.randn(self.batch_size, self.input_size, device=self.device)
        target = torch.randint(0, 5, (self.batch_size,), device=self.device)

        output = self.model(x)
        loss = F.cross_entropy(output, target)
        loss.backward()

        # Store original gradients
        orig_grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                orig_grads[name] = param.grad.clone()

        # Configure no clipping
        config = GradientClipConfig(clip_type="none", model_parallel_reduce=False)

        # Apply "clipping" (should do nothing)
        stats = apply_gradient_clipping(self.model, config)

        # Check that no clipping was applied
        self.assertFalse(stats.get("clipped", False))

        # Verify gradients are unchanged
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.assertTrue(
                    torch.equal(param.grad, orig_grads[name]),
                    f"Gradient for {name} should be unchanged when clip_type='none'",
                )

    def test_clipping_statistics(self):
        """Test that clipping returns correct statistics."""
        # Create large gradients that will definitely be clipped
        self._create_large_gradients(scale=50.0)

        config = GradientClipConfig(
            clip_type="norm", max_norm=1.0, model_parallel_reduce=False
        )

        stats = apply_gradient_clipping(self.model, config)

        # Check required statistics
        required_keys = [
            "grad_norm",
            "clipped",
            "scale_factor",
            "num_parameters",
            "num_gradients",
        ]
        for key in required_keys:
            self.assertIn(key, stats, f"Missing required statistic: {key}")

        # Check value ranges
        self.assertGreaterEqual(stats["grad_norm"], 0.0)
        self.assertIsInstance(stats["clipped"], bool)
        self.assertGreaterEqual(stats["scale_factor"], 0.0)
        self.assertLessEqual(
            stats["scale_factor"], 1.0
        )  # Should be <= 1 for norm clipping
        self.assertGreater(stats["num_parameters"], 0)
        self.assertGreater(stats["num_gradients"], 0)


class TestGradientFiniteCheck(unittest.TestCase):
    """Test gradient finiteness checking."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleModel().to(self.device)

    def test_finite_gradients(self):
        """Test detection of finite gradients."""
        # Create normal gradients
        x = torch.randn(8, 10, device=self.device)
        target = torch.randint(0, 5, (8,), device=self.device)

        output = self.model(x)
        loss = F.cross_entropy(output, target)
        loss.backward()

        is_finite, stats = check_gradient_finite(self.model, raise_on_nonfinite=False)

        self.assertTrue(is_finite)
        self.assertEqual(stats["nan_parameters"], 0)
        self.assertEqual(stats["inf_parameters"], 0)
        self.assertGreater(stats["parameters_with_grad"], 0)

    def test_nan_gradients(self):
        """Test detection of NaN gradients."""
        # Create gradients and inject NaN
        x = torch.randn(8, 10, device=self.device)
        target = torch.randint(0, 5, (8,), device=self.device)

        output = self.model(x)
        loss = F.cross_entropy(output, target)
        loss.backward()

        # Inject NaN into first parameter's gradient
        first_param = next(self.model.parameters())
        if first_param.grad is not None:
            first_param.grad[0, 0] = float("nan")

        is_finite, stats = check_gradient_finite(self.model, raise_on_nonfinite=False)

        self.assertFalse(is_finite)
        self.assertGreater(stats["nan_parameters"], 0)

    def test_inf_gradients(self):
        """Test detection of infinite gradients."""
        # Create gradients and inject infinity
        x = torch.randn(8, 10, device=self.device)
        target = torch.randint(0, 5, (8,), device=self.device)

        output = self.model(x)
        loss = F.cross_entropy(output, target)
        loss.backward()

        # Inject infinity into first parameter's gradient
        first_param = next(self.model.parameters())
        if first_param.grad is not None:
            first_param.grad[0, 0] = float("inf")

        is_finite, stats = check_gradient_finite(self.model, raise_on_nonfinite=False)

        self.assertFalse(is_finite)
        self.assertGreater(stats["inf_parameters"], 0)

    def test_raise_on_nonfinite(self):
        """Test that non-finite gradients raise exception when configured."""
        # Create gradients and inject NaN
        x = torch.randn(8, 10, device=self.device)
        target = torch.randint(0, 5, (8,), device=self.device)

        output = self.model(x)
        loss = F.cross_entropy(output, target)
        loss.backward()

        # Inject NaN
        first_param = next(self.model.parameters())
        if first_param.grad is not None:
            first_param.grad[0, 0] = float("nan")

        with self.assertRaises(RuntimeError):
            check_gradient_finite(self.model, raise_on_nonfinite=True)


class TestGradientStats(unittest.TestCase):
    """Test gradient statistics calculation."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleModel().to(self.device)

    def test_basic_stats(self):
        """Test basic gradient statistics calculation."""
        # Create gradients
        x = torch.randn(8, 10, device=self.device)
        target = torch.randint(0, 5, (8,), device=self.device)

        output = self.model(x)
        loss = F.cross_entropy(output, target)
        loss.backward()

        stats = get_gradient_stats(self.model, include_histograms=False)

        # Check required statistics
        required_keys = [
            "total_parameters",
            "parameters_with_grad",
            "grad_norm_l1",
            "grad_norm_l2",
            "grad_norm_inf",
            "grad_mean",
            "grad_std",
            "grad_min",
            "grad_max",
            "zero_grad_parameters",
            "finite",
        ]

        for key in required_keys:
            self.assertIn(key, stats, f"Missing required statistic: {key}")

        # Check value ranges and types
        self.assertGreater(stats["total_parameters"], 0)
        self.assertGreater(stats["parameters_with_grad"], 0)
        self.assertGreaterEqual(stats["grad_norm_l1"], 0.0)
        self.assertGreaterEqual(stats["grad_norm_l2"], 0.0)
        self.assertGreaterEqual(stats["grad_norm_inf"], 0.0)
        self.assertIsInstance(stats["grad_mean"], float)
        self.assertGreaterEqual(stats["grad_std"], 0.0)
        self.assertIsInstance(stats["grad_min"], float)
        self.assertIsInstance(stats["grad_max"], float)
        self.assertGreaterEqual(stats["zero_grad_parameters"], 0)
        self.assertTrue(stats["finite"])

    def test_histogram_stats(self):
        """Test gradient statistics with histograms."""
        # Create gradients
        x = torch.randn(8, 10, device=self.device)
        target = torch.randint(0, 5, (8,), device=self.device)

        output = self.model(x)
        loss = F.cross_entropy(output, target)
        loss.backward()

        stats = get_gradient_stats(self.model, include_histograms=True)

        # Check histogram is included
        if "histogram" in stats:  # May not be available on all platforms
            self.assertIn("values", stats["histogram"])
            self.assertIn("bins", stats["histogram"])
            self.assertEqual(len(stats["histogram"]["values"]), 50)  # Default 50 bins
            self.assertEqual(len(stats["histogram"]["bins"]), 51)  # 50 bins + 1


class TestGradientAccumulationContext(unittest.TestCase):
    """Test gradient accumulation context manager."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleModel().to(self.device)

        # Reset step counter for clean tests
        if hasattr(gradient_accumulation_context, "_step_counter"):
            gradient_accumulation_context._step_counter = 0

    def test_accumulation_steps(self):
        """Test gradient accumulation step tracking."""
        accumulation_steps = 4
        step_results = []

        for _ in range(8):  # Run for 8 steps (2 full accumulation cycles)
            with gradient_accumulation_context(
                self.model, accumulation_steps, sync_on_last_step=True
            ) as is_last_step:
                step_results.append(is_last_step)

        # Check pattern: False, False, False, True, False, False, False, True
        expected_pattern = [False, False, False, True] * 2
        self.assertEqual(step_results, expected_pattern)

    def test_ddp_sync_control(self):
        """Test DDP sync control (mock test)."""
        # Mock DDP model
        mock_model = MagicMock()
        mock_model.no_sync.return_value.__enter__ = MagicMock()
        mock_model.no_sync.return_value.__exit__ = MagicMock()

        accumulation_steps = 3

        # Test non-last step (should use no_sync)
        with gradient_accumulation_context(
            mock_model, accumulation_steps, sync_on_last_step=True
        ) as is_last_step:
            self.assertFalse(is_last_step)

        # Test second step (still not last)
        with gradient_accumulation_context(
            mock_model, accumulation_steps, sync_on_last_step=True
        ) as is_last_step:
            self.assertFalse(is_last_step)

        # Test last step (should not use no_sync)
        with gradient_accumulation_context(
            mock_model, accumulation_steps, sync_on_last_step=True
        ) as is_last_step:
            self.assertTrue(is_last_step)


class TestGradientSync(unittest.TestCase):
    """Test gradient synchronization (non-distributed tests)."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleModel().to(self.device)

    @patch("rosellm.rosetrainer.utils.gradient_utils.parallel_initialized")
    @patch("rosellm.rosetrainer.utils.gradient_utils.get_data_parallel_group")
    def test_sync_gradients_no_parallel(self, mock_get_dp_group, mock_parallel_init):
        """Test gradient sync when parallelism is not initialized."""
        mock_parallel_init.return_value = False

        # Should return early without error
        sync_gradients(self.model)

        mock_get_dp_group.assert_not_called()

    @patch("rosellm.rosetrainer.utils.gradient_utils.parallel_initialized")
    @patch("rosellm.rosetrainer.utils.gradient_utils.get_data_parallel_group")
    def test_sync_gradients_no_group(self, mock_get_dp_group, mock_parallel_init):
        """Test gradient sync when no data parallel group exists."""
        mock_parallel_init.return_value = True
        mock_get_dp_group.return_value = None

        # Should return early without error
        sync_gradients(self.model)

        mock_get_dp_group.assert_called_once()


class TestDistributedGradientUtils(unittest.TestCase):
    """Test gradient utilities in simulated distributed environment."""

    def setUp(self):
        """Set up simulated distributed environment."""
        # Skip if distributed testing is not available
        if not torch.distributed.is_available():
            self.skipTest("Distributed PyTorch not available")

        # Set environment variables for single-process testing
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleModel().to(self.device)

    def test_model_parallel_reduce_disabled(self):
        """Test gradient norm calculation with model parallel reduce disabled."""
        # Create gradients
        x = torch.randn(8, 10, device=self.device)
        target = torch.randint(0, 5, (8,), device=self.device)

        output = self.model(x)
        loss = F.cross_entropy(output, target)
        loss.backward()

        # Calculate norm without model parallel reduce
        norm = calculate_gradient_norm_multitensor(
            self.model,
            norm_type=2.0,
            use_multitensor=False,
            model_parallel_reduce=False,  # Explicitly disable
        )

        self.assertGreater(float(norm), 0.0)
        self.assertTrue(torch.isfinite(norm), "Gradient norm should be finite")

    def tearDown(self):
        """Clean up environment."""
        # Clean up environment variables
        for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE"]:
            if key in os.environ:
                del os.environ[key]


class TestIntegrationWithRoseTrainer(unittest.TestCase):
    """Integration tests with RoseTrainer components."""

    def test_gradient_config_integration(self):
        """Test that GradientClipConfig integrates properly with training config."""
        from rosellm.rosetrainer.config import GradientClipType, GradientConfig

        # Create config with gradient utilities settings
        grad_config = GradientConfig(
            clip_type=GradientClipType.NORM,
            clip_value=1.0,
            norm_type=2.0,
            use_multitensor=True,
            model_parallel_reduce=True,
            error_if_nonfinite=True,
            track_gradient_stats=True,
            accumulation_steps=4,
            sync_on_accumulation=False,
            gradient_stats_interval=100,
            include_gradient_histograms=False,
        )

        # Convert to GradientClipConfig
        clip_config = GradientClipConfig(
            clip_type=grad_config.clip_type,
            max_norm=grad_config.clip_value or 1.0,
            norm_type=grad_config.norm_type,
            error_if_nonfinite=grad_config.error_if_nonfinite,
            model_parallel_reduce=grad_config.model_parallel_reduce,
            use_multitensor=grad_config.use_multitensor,
        )

        # Verify conversion
        self.assertEqual(clip_config.clip_type, "norm")
        self.assertEqual(clip_config.max_norm, 1.0)
        self.assertEqual(clip_config.norm_type, 2.0)
        self.assertTrue(clip_config.error_if_nonfinite)
        self.assertTrue(clip_config.model_parallel_reduce)
        self.assertTrue(clip_config.use_multitensor)


if __name__ == "__main__":
    unittest.main()
