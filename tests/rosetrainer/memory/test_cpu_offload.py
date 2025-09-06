import unittest

import torch
import torch.nn as nn
import torch.optim as optim

from rosellm.rosetrainer.memory.cpu_offload import (
    CPUOffloadOptimizer,
    ParameterOffloader,
)


class SimpleModel(nn.Module):
    """Simple model for testing CPU offloading."""

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 20))
        self.decoder = nn.Sequential(nn.Linear(20, 50), nn.ReLU(), nn.Linear(50, 10))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class TestCPUOffload(unittest.TestCase):
    """Tests for the CPU offloading utilities."""

    def setUp(self):
        """Set up for each test."""
        self.model = SimpleModel()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # If device is already CPU, we need to handle the tests differently
        self.using_cpu = self.device.type == "cpu"

        # Move model to device
        self.model.to(self.device)

        # Create an optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def test_offload_optimizer(self):
        """Test the CPU offload optimizer."""
        # Skip if already on CPU
        if self.using_cpu:
            self.skipTest("Test requires GPU to be meaningful")

        # Wrap the optimizer
        offload_optimizer = CPUOffloadOptimizer(
            optimizer=self.optimizer, pin_memory=False, offload_params=False
        )

        # Check that param groups are preserved
        self.assertEqual(
            len(offload_optimizer.param_groups), len(self.optimizer.param_groups)
        )

        # Create sample input and target
        input_data = torch.randn(4, 10, device=self.device)
        target = torch.randn(4, 10, device=self.device)

        # Run a training step
        def train_step():
            offload_optimizer.zero_grad()
            output = self.model(input_data)
            loss = nn.MSELoss()(output, target)
            loss.backward()
            offload_optimizer.step()
            return loss.item()

        # Run multiple steps to ensure it works consistently
        for _ in range(3):
            loss = train_step()
            self.assertGreater(loss, 0)  # Just check that loss is valid

    def test_parameter_offloader(self):
        """Test parameter offloading functionality."""
        # Skip if already on CPU
        if self.using_cpu:
            self.skipTest("Test requires GPU to be meaningful")

        # Create a parameter offloader
        param_offloader = ParameterOffloader(
            model=self.model, device=self.device, pin_memory=False
        )

        # Check that all parameters are on GPU
        for name, param in self.model.named_parameters():
            self.assertEqual(param.device.type, "cuda")

        # Offload all parameters
        param_offloader.offload_all_parameters()

        # Check that all parameters are on CPU
        for name, param in self.model.named_parameters():
            self.assertEqual(param.device.type, "cpu")

        # Load them back
        param_offloader.load_all_parameters()

        # Check that all parameters are back on GPU
        for name, param in self.model.named_parameters():
            self.assertEqual(param.device.type, "cuda")

    def test_partial_offloading(self):
        """Test offloading only specific modules."""
        # Skip if already on CPU
        if self.using_cpu:
            self.skipTest("Test requires GPU to be meaningful")

        # Create a parameter offloader
        param_offloader = ParameterOffloader(
            model=self.model, device=self.device, pin_memory=False
        )

        # Offload only encoder parameters
        param_offloader.offload_module_parameters(["encoder"])

        # Check that encoder parameters are on CPU and decoder parameters on GPU
        for name, param in self.model.named_parameters():
            if name.startswith("encoder"):
                self.assertEqual(param.device.type, "cpu")
            else:
                self.assertEqual(param.device.type, "cuda")

        # Load only encoder parameters back
        param_offloader.load_module_parameters(["encoder"])

        # Check that all parameters are back on GPU
        for name, param in self.model.named_parameters():
            self.assertEqual(param.device.type, "cuda")

    def test_optimizer_state_dict(self):
        """Test saving and loading state dict with offloaded optimizer."""
        # Skip if already on CPU
        if self.using_cpu:
            self.skipTest("Test requires GPU to be meaningful")

        # Wrap the optimizer
        offload_optimizer = CPUOffloadOptimizer(
            optimizer=self.optimizer, pin_memory=False, offload_params=True
        )

        # Save state dict
        state_dict = offload_optimizer.state_dict()

        # Check that state dict contains the extra keys
        self.assertIn("param_to_device", state_dict)
        self.assertIn("pin_memory", state_dict)
        self.assertIn("offload_params", state_dict)

        # Create a new optimizer and model
        new_model = SimpleModel().to(self.device)
        new_optimizer = optim.Adam(new_model.parameters(), lr=0.002)  # Different lr

        # Create new offloaded optimizer
        new_offload_optimizer = CPUOffloadOptimizer(
            optimizer=new_optimizer,
            pin_memory=True,  # Different from original
            offload_params=False,  # Different from original
        )

        # Load state dict
        new_offload_optimizer.load_state_dict(state_dict)

        # Check that settings were loaded correctly
        self.assertEqual(new_offload_optimizer.pin_memory, False)
        self.assertEqual(new_offload_optimizer.offload_params, True)

        # Check that optimizer settings were loaded
        for group in new_offload_optimizer.optimizer.param_groups:
            self.assertEqual(group["lr"], 0.001)  # Should now match original


if __name__ == "__main__":
    unittest.main()
