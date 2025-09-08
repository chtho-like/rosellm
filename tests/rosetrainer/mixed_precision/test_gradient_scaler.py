"""
Unit tests for custom gradient scalers.

This module tests the gradient scaler implementations, including:
- Constant scaler behavior
- Dynamic scaler with hysteresis
- State dict save/load
- Bit-to-bit validation with Megatron-LM behavior
"""

import unittest

import torch
import torch.nn as nn

from rosellm.rosetrainer.mixed_precision import ConstantGradScaler
from rosellm.rosetrainer.mixed_precision import (
    EnhancedDynamicGradScaler as DynamicGradScaler,
)
from rosellm.rosetrainer.mixed_precision import GradScalerConfig
from rosellm.rosetrainer.mixed_precision.gradient_scaler import (
    DynamicGradScaler as LegacyDynamicGradScaler,
)
from rosellm.rosetrainer.mixed_precision.gradient_scaler import check_for_inf_and_nan


class SimpleModel(nn.Module):
    """Simple model for testing gradient operations."""

    def __init__(self, input_size: int = 10, hidden_size: int = 20):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


class TestConstantGradScaler(unittest.TestCase):
    """Test cases for ConstantGradScaler."""

    def test_initialization(self):
        """Test scaler initialization with various scales."""
        scaler = ConstantGradScaler(1024.0)
        self.assertEqual(scaler.scale.item(), 1024.0)
        self.assertAlmostEqual(scaler.inv_scale.item(), 1.0 / 1024.0, places=7)

    def test_scale_remains_constant(self):
        """Test that constant scaler never changes scale."""
        scaler = ConstantGradScaler(1024.0)
        initial_scale = scaler.scale.item()

        # Update with overflow - scale should not change
        scaler.update(found_inf=True)
        self.assertEqual(scaler.scale.item(), initial_scale)

        # Update without overflow - scale should not change
        scaler.update(found_inf=False)
        self.assertEqual(scaler.scale.item(), initial_scale)

        # Multiple updates - scale should remain constant
        for _ in range(100):
            scaler.update(found_inf=False)
        self.assertEqual(scaler.scale.item(), initial_scale)

    def test_state_dict(self):
        """Test state dict save and load."""
        scaler1 = ConstantGradScaler(2048.0)
        state_dict = scaler1.state_dict()

        # Create new scaler and load state
        scaler2 = ConstantGradScaler(1.0)  # Different initial value
        scaler2.load_state_dict(state_dict)

        self.assertEqual(scaler2.scale.item(), 2048.0)

    def test_scale_loss(self):
        """Test loss scaling operation."""
        scaler = ConstantGradScaler(1024.0)
        loss = torch.tensor(0.5)
        scaled_loss = scaler.scale_loss(loss)
        self.assertEqual(scaled_loss.item(), 512.0)

    def test_unscale_gradients(self):
        """Test gradient unscaling with model."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        scaler = ConstantGradScaler(1024.0, device=device)
        model = SimpleModel().to(device)

        # Create dummy gradients
        for param in model.parameters():
            param.grad = torch.ones_like(param) * 1024.0

        # Unscale gradients
        scaler.unscale_gradients(model)

        # Check all gradients are unscaled
        for param in model.parameters():
            if param.grad is not None:
                self.assertTrue(torch.allclose(param.grad, torch.ones_like(param)))

    def test_device_placement(self):
        """Test scaler works on different devices."""
        if torch.cuda.is_available():
            scaler = ConstantGradScaler(1024.0, device="cuda")
            self.assertEqual(scaler.scale.device.type, "cuda")
            self.assertEqual(scaler.inv_scale.device.type, "cuda")

        # CPU should always work
        scaler = ConstantGradScaler(1024.0, device="cpu")
        self.assertEqual(scaler.scale.device.type, "cpu")
        self.assertEqual(scaler.inv_scale.device.type, "cpu")


class TestDynamicGradScaler(unittest.TestCase):
    """Test cases for DynamicGradScaler."""

    def test_initialization(self):
        """Test dynamic scaler initialization."""
        scaler = DynamicGradScaler(
            initial_scale=1024.0,
            min_scale=1.0,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=100,
            hysteresis=2,
        )
        self.assertEqual(scaler.scale.item(), 1024.0)
        self.assertEqual(scaler._growth_tracker, 0)
        self.assertEqual(scaler._hysteresis_tracker, 2)

    def test_backoff_with_hysteresis(self):
        """Test that scaler backs off after consecutive overflows."""
        scaler = DynamicGradScaler(
            initial_scale=1024.0, backoff_factor=0.5, hysteresis=2
        )

        # First overflow - no backoff yet
        scaler.update(found_inf=True)
        self.assertEqual(scaler.scale.item(), 1024.0)
        self.assertEqual(scaler._hysteresis_tracker, 1)

        # Second overflow - should trigger backoff
        scaler.update(found_inf=True)
        self.assertEqual(scaler.scale.item(), 512.0)
        self.assertEqual(scaler._hysteresis_tracker, 2)  # Reset

    def test_growth_after_interval(self):
        """Test that scaler grows after successful iterations."""
        scaler = DynamicGradScaler(
            initial_scale=1024.0, growth_factor=2.0, growth_interval=100
        )

        # Run successful iterations
        for i in range(99):
            scaler.update(found_inf=False)
            self.assertEqual(scaler.scale.item(), 1024.0)  # No growth yet

        # 100th iteration should trigger growth
        scaler.update(found_inf=False)
        self.assertEqual(scaler.scale.item(), 2048.0)
        self.assertEqual(scaler._growth_tracker, 0)  # Reset

    def test_min_scale_enforcement(self):
        """Test that scale doesn't go below min_scale."""
        scaler = DynamicGradScaler(
            initial_scale=4.0, min_scale=2.0, backoff_factor=0.5, hysteresis=1
        )

        # Trigger multiple backoffs
        scaler.update(found_inf=True)
        self.assertEqual(scaler.scale.item(), 2.0)  # 4 * 0.5 = 2

        scaler.update(found_inf=True)
        self.assertEqual(scaler.scale.item(), 2.0)  # Should stay at min_scale

    def test_growth_reset_on_overflow(self):
        """Test that growth tracker resets on overflow."""
        scaler = DynamicGradScaler(
            initial_scale=1024.0, growth_interval=100, hysteresis=3
        )

        # Accumulate growth
        for _ in range(50):
            scaler.update(found_inf=False)
        self.assertEqual(scaler._growth_tracker, 50)

        # Overflow should reset growth tracker
        scaler.update(found_inf=True)
        self.assertEqual(scaler._growth_tracker, 0)
        self.assertEqual(scaler._hysteresis_tracker, 2)

    def test_state_dict_save_load(self):
        """Test state dict with all internal states."""
        scaler1 = DynamicGradScaler(
            initial_scale=1024.0, growth_interval=100, hysteresis=2
        )

        # Modify internal state
        for _ in range(50):
            scaler1.update(found_inf=False)
        scaler1.update(found_inf=True)

        # Save state
        state_dict = scaler1.state_dict()

        # Create new scaler and load state
        scaler2 = DynamicGradScaler(
            initial_scale=1.0, growth_interval=100, hysteresis=2
        )
        scaler2.load_state_dict(state_dict)

        # Verify all states match
        self.assertEqual(scaler2.scale.item(), scaler1.scale.item())
        self.assertEqual(scaler2._growth_tracker, scaler1._growth_tracker)
        self.assertEqual(scaler2._hysteresis_tracker, scaler1._hysteresis_tracker)

    def test_parameter_validation(self):
        """Test that invalid parameters raise errors."""
        # Negative initial scale
        with self.assertRaises(ValueError):
            DynamicGradScaler(initial_scale=-1.0)

        # Min scale > initial scale
        with self.assertRaises(ValueError):
            DynamicGradScaler(initial_scale=10.0, min_scale=20.0)

        # Growth factor <= 1
        with self.assertRaises(ValueError):
            DynamicGradScaler(initial_scale=10.0, growth_factor=0.5)

        # Backoff factor out of range
        with self.assertRaises(ValueError):
            DynamicGradScaler(initial_scale=10.0, backoff_factor=1.5)

        # Negative growth interval
        with self.assertRaises(ValueError):
            DynamicGradScaler(initial_scale=10.0, growth_interval=-1)

        # Zero hysteresis
        with self.assertRaises(ValueError):
            DynamicGradScaler(initial_scale=10.0, hysteresis=0)


class TestGradScalerConfig(unittest.TestCase):
    """Test cases for GradScalerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GradScalerConfig()
        self.assertEqual(config.scaler_type, "dynamic")
        self.assertEqual(config.initial_scale, 2**16)
        self.assertEqual(config.min_scale, 1.0)
        self.assertEqual(config.growth_factor, 2.0)
        self.assertEqual(config.backoff_factor, 0.5)
        self.assertEqual(config.growth_interval, 2000)
        self.assertEqual(config.hysteresis, 2)

    def test_create_constant_scaler(self):
        """Test creating constant scaler from config."""
        config = GradScalerConfig(scaler_type="constant", initial_scale=1024.0)
        scaler = config.create_scaler()
        self.assertIsInstance(scaler, ConstantGradScaler)
        assert scaler is not None
        self.assertEqual(scaler.scale.item(), 1024.0)

    def test_create_dynamic_scaler(self):
        """Test creating dynamic scaler from config."""
        config = GradScalerConfig(
            scaler_type="dynamic", initial_scale=512.0, growth_interval=500
        )
        scaler = config.create_scaler()
        self.assertIsInstance(scaler, LegacyDynamicGradScaler)
        assert scaler is not None
        self.assertEqual(scaler.scale.item(), 512.0)
        if isinstance(scaler, LegacyDynamicGradScaler):
            self.assertEqual(scaler.growth_interval, 500)

    def test_create_none_scaler(self):
        """Test that 'none' type returns None."""
        config = GradScalerConfig(scaler_type="none")
        scaler = config.create_scaler()
        self.assertIsNone(scaler)

    def test_invalid_scaler_type(self):
        """Test that invalid scaler type raises error."""
        with self.assertRaises(ValueError):
            GradScalerConfig(scaler_type="invalid")

    def test_config_validation(self):
        """Test configuration parameter validation."""
        # Invalid initial scale
        with self.assertRaises(ValueError):
            GradScalerConfig(initial_scale=-1.0)

        # Invalid min scale
        with self.assertRaises(ValueError):
            GradScalerConfig(min_scale=0.0)

        # Invalid growth factor
        with self.assertRaises(ValueError):
            GradScalerConfig(growth_factor=0.5)

        # Invalid backoff factor
        with self.assertRaises(ValueError):
            GradScalerConfig(backoff_factor=2.0)

        # Invalid growth interval
        with self.assertRaises(ValueError):
            GradScalerConfig(growth_interval=0)

        # Invalid hysteresis
        with self.assertRaises(ValueError):
            GradScalerConfig(hysteresis=-1)


class TestMegatronParity(unittest.TestCase):
    """Test bit-to-bit parity with Megatron-LM behavior."""

    def test_dynamic_scaler_megatron_behavior(self):
        """Validate that our implementation matches Megatron-LM's behavior."""
        # Create scaler with Megatron-LM default parameters
        rosellm_scaler = DynamicGradScaler(
            initial_scale=65536.0,
            min_scale=1.0,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=1000,
            hysteresis=2,
        )

        # Simulate Megatron-LM state tracking
        megatron_scale = 65536.0
        megatron_growth_tracker = 0
        megatron_hysteresis_tracker = 2

        # Test sequence: growth, overflow, recovery
        test_sequence = (
            [False] * 500  # Partial growth
            + [True, True]  # Trigger backoff
            + [False] * 1000  # Full growth cycle
        )

        for found_inf in test_sequence:
            # Update RoseLLM scaler
            rosellm_scaler.update(found_inf)

            # Simulate Megatron-LM logic
            if found_inf:
                megatron_growth_tracker = 0
                megatron_hysteresis_tracker -= 1
                if megatron_hysteresis_tracker <= 0:
                    megatron_scale = max(megatron_scale * 0.5, 1.0)
                    megatron_hysteresis_tracker = 2
            else:
                megatron_growth_tracker += 1
                if megatron_growth_tracker == 1000:
                    megatron_scale = megatron_scale * 2.0
                    megatron_growth_tracker = 0
                    megatron_hysteresis_tracker = 2

            # Verify bit-to-bit accuracy
            self.assertAlmostEqual(
                rosellm_scaler.scale.item(),
                megatron_scale,
                places=6,
                msg=f"Scale mismatch: RoseLLM={rosellm_scaler.scale.item()}, "
                f"Megatron={megatron_scale}",
            )

    def test_edge_case_min_scale(self):
        """Test edge case where scale hits minimum."""
        rosellm_scaler = DynamicGradScaler(
            initial_scale=2.0, min_scale=1.0, backoff_factor=0.5, hysteresis=1
        )

        # Simulate hitting min scale
        megatron_scale = 2.0

        # Multiple overflows to hit min scale
        for _ in range(3):
            rosellm_scaler.update(found_inf=True)
            megatron_scale = max(megatron_scale * 0.5, 1.0)
            self.assertAlmostEqual(
                rosellm_scaler.scale.item(), megatron_scale, places=6
            )

        # Should stay at min scale
        self.assertEqual(rosellm_scaler.scale.item(), 1.0)


class TestCheckForInfAndNan(unittest.TestCase):
    """Test the check_for_inf_and_nan utility function."""

    def test_no_overflow(self):
        """Test detection with finite gradients."""
        model = SimpleModel()

        # Set finite gradients
        for param in model.parameters():
            param.grad = torch.ones_like(param)

        found_inf = check_for_inf_and_nan(model)
        self.assertFalse(found_inf)

    def test_detect_nan(self):
        """Test detection of NaN gradients."""
        model = SimpleModel()

        # Set one gradient to NaN
        for i, param in enumerate(model.parameters()):
            if i == 0:
                param.grad = torch.full_like(param, float("nan"))
            else:
                param.grad = torch.ones_like(param)

        found_inf = check_for_inf_and_nan(model)
        self.assertTrue(found_inf)

    def test_detect_inf(self):
        """Test detection of infinite gradients."""
        model = SimpleModel()

        # Set one gradient to inf
        for i, param in enumerate(model.parameters()):
            if i == 1:
                param.grad = torch.full_like(param, float("inf"))
            else:
                param.grad = torch.ones_like(param)

        found_inf = check_for_inf_and_nan(model)
        self.assertTrue(found_inf)

    def test_with_scaler_update(self):
        """Test that scaler is updated when provided."""
        model = SimpleModel()
        scaler = DynamicGradScaler(initial_scale=1024.0, hysteresis=1)

        # Set finite gradients
        for param in model.parameters():
            param.grad = torch.ones_like(param)

        # Check without overflow
        found_inf = check_for_inf_and_nan(model, scaler)
        self.assertFalse(found_inf)
        self.assertEqual(scaler._growth_tracker, 1)

        # Set infinite gradient
        next(model.parameters()).grad = torch.full_like(
            next(model.parameters()), float("inf")
        )

        # Check with overflow
        found_inf = check_for_inf_and_nan(model, scaler)
        self.assertTrue(found_inf)
        self.assertEqual(scaler._growth_tracker, 0)

    def test_parameter_list(self):
        """Test with list of parameters instead of model."""
        params = [nn.Parameter(torch.randn(10, 10)), nn.Parameter(torch.randn(5, 5))]

        # Set gradients
        params[0].grad = torch.ones_like(params[0])
        params[1].grad = torch.ones_like(params[1])

        found_inf = check_for_inf_and_nan(params)
        self.assertFalse(found_inf)

        # Add NaN
        params[1].grad = torch.full_like(params[1], float("nan"))
        found_inf = check_for_inf_and_nan(params)
        self.assertTrue(found_inf)

    def test_none_gradients(self):
        """Test handling of None gradients."""
        model = SimpleModel()

        # No gradients set (all None)
        found_inf = check_for_inf_and_nan(model)
        self.assertFalse(found_inf)

        # Mix of None and finite gradients
        for i, param in enumerate(model.parameters()):
            if i % 2 == 0:
                param.grad = torch.ones_like(param)
            # else: grad remains None

        found_inf = check_for_inf_and_nan(model)
        self.assertFalse(found_inf)


class TestIntegration(unittest.TestCase):
    """Integration tests for gradient scalers."""

    def test_full_training_step_simulation(self):
        """Simulate a full training step with gradient scaling."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Setup
        model = SimpleModel().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler = DynamicGradScaler(initial_scale=1024.0)

        # Simulate training step
        for step in range(10):
            optimizer.zero_grad()

            # Forward pass
            x = torch.randn(32, 10).cuda()
            output = model(x)
            loss = output.mean()

            # Scale loss and backward
            scaled_loss = scaler.scale_loss(loss)
            scaled_loss.backward()

            # Check for overflow
            found_inf = check_for_inf_and_nan(model, scaler)

            if not found_inf:
                # Unscale gradients
                scaler.unscale_gradients(model)

                # Clip gradients (optional)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Optimizer step
                optimizer.step()

        # Verify scaler is working
        self.assertIsNotNone(scaler.scale)
        self.assertGreater(scaler.scale.item(), 0)

    def test_state_persistence(self):
        """Test saving and loading full training state."""
        model = SimpleModel()
        scaler = DynamicGradScaler(initial_scale=512.0)

        # Simulate some training
        for _ in range(50):
            scaler.update(found_inf=False)

        # Save state
        checkpoint = {"model": model.state_dict(), "scaler": scaler.state_dict()}

        # Create new instances
        new_model = SimpleModel()
        new_scaler = DynamicGradScaler(initial_scale=1.0)

        # Load state
        new_model.load_state_dict(checkpoint["model"])
        new_scaler.load_state_dict(checkpoint["scaler"])

        # Verify state is restored
        self.assertEqual(new_scaler.scale.item(), scaler.scale.item())
        self.assertEqual(new_scaler._growth_tracker, scaler._growth_tracker)


if __name__ == "__main__":
    unittest.main()
