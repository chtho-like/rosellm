import os
import unittest
import torch
import torch.nn as nn
import torch.optim as optim

from rosellm.rosetrainer.engine import RoseTrainer


class SimpleModel(nn.Module):
    """Simple model for testing RoseTrainer."""
    
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 2)
        
    def forward(self, input_ids=None, **kwargs):
        outputs = self.linear(input_ids)
        loss = outputs.sum()
        return type('ModelOutput', (), {'loss': loss})()


class TestRoseTrainer(unittest.TestCase):
    """Tests for the RoseTrainer class."""
    
    def setUp(self):
        """Set up for each test."""
        self.model = SimpleModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.config = {
            "max_grad_norm": 1.0,
        }
        
        # Create a single-process trainer for testing
        self.trainer = RoseTrainer(
            model=self.model,
            optimizer=self.optimizer,
            config=self.config,
            local_rank=0,
            world_size=1
        )
        
    def test_initialization(self):
        """Test trainer initialization."""
        self.assertIsNotNone(self.trainer)
        self.assertEqual(self.trainer.device.type, 
                        "cuda" if torch.cuda.is_available() else "cpu")
        self.assertFalse(self.trainer.distributed)
        
    def test_train_step(self):
        """Test a single training step."""
        # Create a simple batch
        batch_size = 4
        input_ids = torch.randn(batch_size, 10, device=self.trainer.device)
        batch = {"input_ids": input_ids}
        
        # Perform a training step
        result = self.trainer.train_step(batch)
        
        # Check the results
        self.assertIn("loss", result)
        self.assertIsInstance(result["loss"], float)
        
    def test_save_and_load_checkpoint(self):
        """Test saving and loading checkpoints."""
        import tempfile
        import os
        
        # Create a temporary directory for checkpoints
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")
            
            # Save the checkpoint
            self.trainer.save_checkpoint(checkpoint_path)
            
            # Verify the file exists
            self.assertTrue(os.path.isfile(checkpoint_path))
            
            # Change model parameters
            old_params = next(self.model.parameters()).clone()
            for param in self.model.parameters():
                param.data.add_(torch.randn_like(param))
            
            # Check that parameters have changed
            new_params = next(self.model.parameters()).clone()
            self.assertFalse(torch.allclose(old_params, new_params))
            
            # Load the checkpoint
            self.trainer.load_checkpoint(checkpoint_path)
            
            # Verify parameters are restored
            loaded_params = next(self.model.parameters()).clone()
            self.assertTrue(torch.allclose(old_params, loaded_params))
            

if __name__ == "__main__":
    unittest.main() 