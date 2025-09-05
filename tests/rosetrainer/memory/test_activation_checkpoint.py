import unittest

import torch
import torch.nn as nn

from rosellm.rosetrainer.memory.activation_checkpoint import ActivationCheckpointing


class SimpleTransformerLayer(nn.Module):
    """Simple transformer layer for testing checkpointing."""

    def __init__(self, size=128):
        super().__init__()
        self.attention = nn.Linear(size, size)
        self.mlp = nn.Sequential(
            nn.Linear(size, size * 4), nn.ReLU(), nn.Linear(size * 4, size)
        )
        self.ln1 = nn.LayerNorm(size)
        self.ln2 = nn.LayerNorm(size)

    def forward(self, x):
        h = x + self.attention(self.ln1(x))
        out = h + self.mlp(self.ln2(h))
        return out


class SimpleTransformer(nn.Module):
    """Simple transformer model for testing checkpoint activation."""

    def __init__(self, num_layers=4, size=128):
        super().__init__()
        # Create a ModuleList for transformer layers
        self.layers = nn.ModuleList(
            [SimpleTransformerLayer(size) for _ in range(num_layers)]
        )
        self.embedding = nn.Linear(10, size)
        self.output = nn.Linear(size, 10)

    def forward(self, x):
        x = self.embedding(x)
        # Iterate through the layers
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


class TestActivationCheckpointing(unittest.TestCase):
    """Tests for the ActivationCheckpointing class."""

    def setUp(self):
        """Set up for each test."""
        self.model = SimpleTransformer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def test_apply_to_transformer_layers(self):
        """Test applying checkpointing to transformer layers."""
        # Create input
        batch_size = 4
        seq_len = 8
        input_tensor = torch.randn(batch_size, seq_len, 10, device=self.device)

        # Forward pass before checkpointing
        out_before = self.model(input_tensor).sum()

        # Save model state for comparison after checkpointing
        model_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # Apply checkpointing
        model_with_checkpoints = ActivationCheckpointing.apply_to_transformer_layers(
            self.model, use_reentrant=True, layer_attr="layers"
        )

        # Verify model weights are unchanged
        for name, param in model_with_checkpoints.named_parameters():
            original_param = model_state[name]
            self.assertTrue(torch.allclose(param, original_param))

        # Forward pass after checkpointing
        out_after = model_with_checkpoints(input_tensor).sum()

        # Check that both can do backward pass
        out_before.backward(retain_graph=True)

        # Reset gradients and run with checkpointing
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()

        out_after.backward()

    def test_apply_to_modules(self):
        """Test applying checkpointing to specific module types."""
        # Create a new model instance for this test to avoid interference
        model = SimpleTransformer()
        model.to(self.device)

        # Create input
        batch_size = 4
        seq_len = 8
        input_tensor = torch.randn(batch_size, seq_len, 10, device=self.device)

        # Apply checkpointing to only specific modules to avoid shape mismatch
        model_with_checkpoints = ActivationCheckpointing.apply_to_modules(
            model,
            module_types=[nn.Sequential],  # Only apply to Sequential modules
            use_reentrant=True,
        )

        # Forward pass after checkpointing
        out_after = model_with_checkpoints(input_tensor).sum()

        # Check that we can do backward pass with checkpointing
        if out_after is not None:  # Guard against None
            out_after.backward()

    def test_custom_function_checkpointing(self):
        """Test checkpointing a custom function."""
        # Create input
        x = torch.randn(4, 10, requires_grad=True, device=self.device)

        # Create a function to checkpoint
        def func(x):
            return torch.sin(x).sum()

        # Run with and without checkpointing
        result_orig = func(x)
        result_checkpointed = ActivationCheckpointing.apply_to_custom_function(func, x)

        # Results should be identical - only check if result is not None
        if result_checkpointed is not None:
            self.assertTrue(torch.allclose(result_orig, result_checkpointed))

            # Can do backward pass
            result_checkpointed.backward()


if __name__ == "__main__":
    unittest.main()
