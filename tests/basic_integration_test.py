"""Basic integration test for RoseLLM."""

import unittest

import torch
import torch.nn as nn
import torch.optim as optim

from rosellm.rosetrainer.engine import RoseTrainer


class SimpleModel(nn.Module):
    """Simple model for integration testing."""

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, input_ids=None, **kwargs):
        outputs = self.linear(input_ids)
        loss = outputs.sum()
        return type("ModelOutput", (), {"loss": loss})()


class BasicIntegrationTest(unittest.TestCase):
    """Basic integration test for RoseLLM."""

    def test_basic_training(self):
        """Test training flow and checkpoint functionality."""
        # Create a simple model
        model = SimpleModel()

        # Set up device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create an optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Create trainer
        trainer = RoseTrainer(
            model=model,
            optimizer=optimizer,
            config={"max_grad_norm": 1.0},
            local_rank=0,
            world_size=1,
        )

        # Create dummy batch
        batch_size = 4
        input_ids = torch.randn(batch_size, 10, device=device)
        batch = {"input_ids": input_ids}

        # Perform initial training
        for i in range(3):
            result = trainer.train_step(batch)
            print(f"Step {i+1}, Loss: {result['loss']}")

        # Test checkpoint saving/loading
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")

            # Save checkpoint
            trainer.save_checkpoint(checkpoint_path)

            # Check file exists
            self.assertTrue(os.path.exists(checkpoint_path))

            # Change model parameters
            for param in model.parameters():
                param.data.add_(0.1)

            # Load checkpoint
            trainer.load_checkpoint(checkpoint_path)

            # Verify training continues
            result = trainer.train_step(batch)
            self.assertIsNotNone(result["loss"])


if __name__ == "__main__":
    unittest.main()
