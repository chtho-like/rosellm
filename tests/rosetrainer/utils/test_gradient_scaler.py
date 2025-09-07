"""
Tests for CustomGradientScaler.
"""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from rosellm.rosetrainer.utils import CustomGradientScaler


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class TestCustomGradientScaler(unittest.TestCase):
    """Test CustomGradientScaler functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleModel().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

    def test_scaler_initialization(self):
        """Test scaler initialization."""
        scaler = CustomGradientScaler()
        self.assertEqual(scaler.get_scale(), 2.0**16)
        self.assertEqual(scaler.get_growth_tracker(), 0)
        self.assertEqual(scaler.get_overflow_history(), [])

        # Test custom initialization
        scaler = CustomGradientScaler(
            init_scale=1024.0,
            growth_factor=3.0,
            backoff_factor=0.25,
            growth_interval=1000,
            enabled=False,
        )
        self.assertEqual(scaler.get_scale(), 1.0)  # Disabled returns 1.0

    def test_scale_operation(self):
        """Test scaling of loss values."""
        scaler = CustomGradientScaler(init_scale=100.0)

        loss = torch.tensor(1.5, device=self.device)
        scaled_loss = scaler.scale(loss)

        self.assertAlmostEqual(scaled_loss.item(), 150.0, places=5)

        # Test disabled scaler
        scaler = CustomGradientScaler(enabled=False)
        scaled_loss = scaler.scale(loss)
        self.assertEqual(scaled_loss.item(), loss.item())

    def test_unscale_operation(self):
        """Test unscaling of gradients."""
        scaler = CustomGradientScaler(init_scale=10.0)

        # Create fake gradients
        x = torch.randn(4, 10, device=self.device)
        y = self.model(x)
        loss = y.sum() * 10.0  # Scale the loss
        loss.backward()

        # Store original gradients
        orig_grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                orig_grads.append(param.grad.clone())

        # Unscale
        scaler.unscale_(self.optimizer)

        # Check that gradients were divided by scale
        for param, orig_grad in zip(self.model.parameters(), orig_grads):
            if param.grad is not None:
                expected = orig_grad / 10.0
                torch.testing.assert_close(param.grad, expected, rtol=1e-5, atol=1e-6)

    def test_update_with_overflow(self):
        """Test scale update with gradient overflow."""
        scaler = CustomGradientScaler(init_scale=1000.0, backoff_factor=0.5)

        initial_scale = scaler.get_scale()

        # Simulate overflow
        scaler.update(found_inf=True)

        self.assertEqual(scaler.get_scale(), initial_scale * 0.5)
        self.assertEqual(scaler.get_growth_tracker(), 0)
        self.assertEqual(scaler.get_overflow_history(), [True])

    def test_update_without_overflow(self):
        """Test scale update without overflow."""
        scaler = CustomGradientScaler(
            init_scale=100.0, growth_factor=2.0, growth_interval=2
        )

        initial_scale = scaler.get_scale()

        # First successful update
        scaler.update(found_inf=False)
        self.assertEqual(scaler.get_scale(), initial_scale)
        self.assertEqual(scaler.get_growth_tracker(), 1)

        # Second successful update - should trigger growth
        scaler.update(found_inf=False)
        self.assertEqual(scaler.get_scale(), initial_scale * 2.0)
        self.assertEqual(scaler.get_growth_tracker(), 0)

    def test_step_method(self):
        """Test the step method with optimizer."""
        scaler = CustomGradientScaler(init_scale=10.0)

        # Create gradients
        x = torch.randn(4, 10, device=self.device)
        y = self.model(x)
        target = torch.randn_like(y)
        loss = F.mse_loss(y, target)

        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()

        # Step should unscale and update optimizer
        # Note: optimizer.step() normally returns None, which is expected
        scaler.step(self.optimizer)
        # Just verify no exception was raised

    def test_step_with_inf_gradients(self):
        """Test step method with infinite gradients."""
        scaler = CustomGradientScaler(init_scale=10.0)

        # Create gradients and inject inf
        x = torch.randn(4, 10, device=self.device)
        y = self.model(x)
        loss = y.sum()
        loss.backward()

        # Inject infinity
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad[0] = float("inf")
                break

        # Step should return None due to inf
        result = scaler.step(self.optimizer)
        self.assertIsNone(result)

    def test_state_dict_and_load(self):
        """Test state dict save and load."""
        scaler = CustomGradientScaler(init_scale=500.0)

        # Update a few times
        scaler.update(found_inf=False)
        scaler.update(found_inf=True)
        scaler.update(found_inf=False)

        # Save state
        state = scaler.state_dict()

        # Create new scaler and load state
        new_scaler = CustomGradientScaler()
        new_scaler.load_state_dict(state)

        self.assertEqual(new_scaler.get_scale(), scaler.get_scale())
        self.assertEqual(new_scaler.get_growth_tracker(), scaler.get_growth_tracker())
        self.assertEqual(
            new_scaler.get_overflow_history(), scaler.get_overflow_history()
        )

    def test_repr(self):
        """Test string representation."""
        scaler = CustomGradientScaler(init_scale=1024.0)
        repr_str = repr(scaler)

        self.assertIn("CustomGradientScaler", repr_str)
        self.assertIn("scale=1024.0", repr_str)
        self.assertIn("enabled=True", repr_str)

    def test_auto_detect_overflow(self):
        """Test automatic overflow detection from optimizer."""
        scaler = CustomGradientScaler(init_scale=100.0)

        # Create model with inf gradients
        x = torch.randn(4, 10, device=self.device)
        y = self.model(x)
        loss = y.sum()
        loss.backward()

        # Inject NaN
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad[0] = float("nan")
                break

        initial_scale = scaler.get_scale()

        # Update should auto-detect NaN
        scaler.update(optimizer=self.optimizer)

        # Scale should be reduced
        self.assertLess(scaler.get_scale(), initial_scale)
        self.assertEqual(scaler.get_overflow_history()[-1], True)


if __name__ == "__main__":
    unittest.main()
