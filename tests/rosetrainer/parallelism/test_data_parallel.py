import unittest

import torch
import torch.nn as nn
import torch.optim as optim

from rosellm.rosetrainer.parallelism.data_parallel import DataParallelTrainer


class SimpleModel(nn.Module):
    """Simple model for testing DataParallelTrainer."""

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, input_ids=None, **kwargs):
        outputs = self.linear(input_ids)
        loss = outputs.sum()
        return type("ModelOutput", (), {"loss": loss})()


class TestDataParallelTrainer(unittest.TestCase):
    """Tests for the DataParallelTrainer class."""

    def setUp(self):
        """Set up for each test."""
        self.model = SimpleModel()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create a single-process trainer for testing
        self.trainer = DataParallelTrainer(
            model=self.model,
            device=self.device,
            local_rank=0,
            world_size=1,
            gradient_accumulation_steps=2,
            fp16=False,
            grad_clip=1.0,
        )

    def test_initialization(self):
        """Test trainer initialization."""
        self.assertIsNotNone(self.trainer)
        self.assertEqual(self.trainer.device, self.device)
        self.assertEqual(self.trainer.accumulation_step, 0)

    def test_forward_backward(self):
        """Test forward and backward passes with gradient accumulation."""
        # Create a simple batch
        batch_size = 4
        input_ids = torch.randn(batch_size, 10, device=self.device)
        batch = {"input_ids": input_ids}

        # Create optimizer
        optimizer = optim.Adam(self.trainer.model.parameters(), lr=0.001)

        # Zero gradients initially
        optimizer.zero_grad()

        # First accumulation step
        _ = self.trainer.forward_backward(batch, optimizer)  # loss1
        self.assertEqual(self.trainer.accumulation_step, 1)

        # Check that optimizer step was not called yet
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    # Gradient exists but optimizer step not called
                    has_grad = True
        self.assertTrue(has_grad)

        # Second accumulation step (should update parameters)
        _ = self.trainer.forward_backward(batch, optimizer)  # loss2
        self.assertEqual(self.trainer.accumulation_step, 0)  # Reset after accumulation

    def test_get_model(self):
        """Test getting the underlying model."""
        model = self.trainer.get_model()
        self.assertIsInstance(model, SimpleModel)

        # If using DDP, should unwrap the model
        if hasattr(self.trainer.model, "module"):
            self.assertEqual(model, self.trainer.model.module)
        else:
            self.assertEqual(model, self.trainer.model)


if __name__ == "__main__":
    unittest.main()
