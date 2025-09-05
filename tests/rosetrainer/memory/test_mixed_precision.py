import unittest

import torch
import torch.nn as nn

from rosellm.rosetrainer.memory.mixed_precision import (
    DynamicLossScaler,
    check_overflow,
    convert_model_to_fp16,
)


class SimpleModel(nn.Module):
    """Simple model for testing mixed precision training."""

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(10, 50)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(50, 1)
        self.layernorm = nn.LayerNorm(50)

    def forward(self, x):
        x = self.linear1(x)
        x = self.layernorm(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class TestMixedPrecision(unittest.TestCase):
    """Tests for the mixed precision training utilities."""

    def setUp(self):
        """Set up for each test."""
        self.model = SimpleModel()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Create a loss scaler with small values for testing
        self.loss_scaler = DynamicLossScaler(
            init_scale=8.0,
            scale_factor=2.0,
            scale_window=3,
            min_scale=1.0,
            max_scale=32.0,
        )

    def test_loss_scaling(self):
        """Test loss scaling functionality."""
        # Create a tensor to scale
        loss = torch.tensor(2.0, device=self.device, requires_grad=True)

        # Scale the loss
        scaled_loss = self.loss_scaler.scale(loss)

        # Check scaling
        self.assertEqual(scaled_loss.item(), loss.item() * self.loss_scaler.cur_scale)

        # Test unscaling
        grad = torch.tensor(3.0, device=self.device)
        self.loss_scaler.unscale(grad)
        self.assertEqual(grad.item(), 3.0 / self.loss_scaler.cur_scale)

    def test_dynamic_scaling_update(self):
        """Test dynamic update of loss scale."""
        # Initial scale should be 8.0
        self.assertEqual(self.loss_scaler.cur_scale, 8.0)

        # Update without overflow
        for _ in range(3):  # scale_window is 3
            self.loss_scaler.update_scale(overflow=False)

        # Scale should double after scale_window steps
        self.assertEqual(self.loss_scaler.cur_scale, 16.0)

        # Update with overflow
        self.loss_scaler.update_scale(overflow=True)

        # Scale should halve on overflow
        self.assertEqual(self.loss_scaler.cur_scale, 8.0)

    def test_check_overflow(self):
        """Test overflow detection."""
        # Create parameters with normal gradients
        param1 = nn.Parameter(torch.randn(10, 10, device=self.device))
        param1.grad = torch.randn(10, 10, device=self.device)

        # No overflow
        self.assertFalse(check_overflow([param1]))

        # Create a parameter with NaN gradient
        param2 = nn.Parameter(torch.randn(10, 10, device=self.device))
        param2.grad = torch.tensor(float("nan"), device=self.device).expand(10, 10)

        # Should detect overflow
        self.assertTrue(check_overflow([param1, param2]))

        # Create a parameter with inf gradient
        param3 = nn.Parameter(torch.randn(10, 10, device=self.device))
        param3.grad = torch.tensor(float("inf"), device=self.device).expand(10, 10)

        # Should detect overflow
        self.assertTrue(check_overflow([param1, param3]))

    def test_convert_model_to_fp16(self):
        """Test conversion of model to FP16."""
        # Model should start in FP32
        for name, param in self.model.named_parameters():
            self.assertEqual(param.dtype, torch.float32)

        # Convert model to FP16
        convert_model_to_fp16(self.model)

        # Check that parameters are converted (except LayerNorm)
        for name, param in self.model.named_parameters():
            if "layernorm" in name:
                # LayerNorm should stay in FP32
                self.assertEqual(param.dtype, torch.float32)
            else:
                # Other parameters should be in FP16
                self.assertEqual(param.dtype, torch.float16)


if __name__ == "__main__":
    unittest.main()
