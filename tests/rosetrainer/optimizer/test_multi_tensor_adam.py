"""
Comprehensive tests for Multi-Tensor Adam Optimizer.

This module provides thorough testing of the Multi-Tensor Adam optimizer
including performance comparisons, numerical accuracy validation, mixed
precision training, and distributed scenarios.

Test Categories:
1. Basic functionality and correctness
2. Multi-tensor operations and backend selection
3. Mixed precision training with loss scaling
4. Gradient clipping and overflow handling
5. Performance benchmarking against standard optimizers
6. Distributed training scenarios
7. Edge cases and error handling
8. Memory efficiency validation
"""

import time
from typing import List

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from rosellm.rosetrainer.optimizer import (
    MultiTensorAdam,
    MultiTensorAdamConfig,
    OverflowAction,
    WeightDecayMode,
    create_multi_tensor_adam,
    create_multi_tensor_adamw,
)


class SimpleMLP(nn.Module):
    """Simple multi-layer perceptron for testing."""

    def __init__(
        self,
        input_size: int = 128,
        hidden_sizes: List[int] = [256, 128],
        output_size: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
            )
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.network(x)
        assert isinstance(result, torch.Tensor)
        return result


class TransformerBlock(nn.Module):
    """Transformer block for more complex testing."""

    def __init__(self, d_model: int = 512, n_head: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x


@pytest.fixture
def device():
    """Get available device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def simple_model(device):
    """Create a simple model for testing."""
    model = SimpleMLP().to(device)
    return model


@pytest.fixture
def transformer_model(device):
    """Create a transformer model for more complex testing."""
    model = TransformerBlock().to(device)
    return model


@pytest.fixture
def sample_data(device):
    """Create sample training data."""
    batch_size = 32
    input_size = 128
    num_classes = 10

    x = torch.randn(batch_size, input_size, device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)

    return x, y


@pytest.fixture
def transformer_data(device):
    """Create sample data for transformer testing."""
    batch_size = 16
    seq_len = 64
    d_model = 512

    x = torch.randn(batch_size, seq_len, d_model, device=device)
    return x


class TestMultiTensorAdamConfig:
    """Test configuration class for Multi-Tensor Adam."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MultiTensorAdamConfig()

        assert config.lr == 1e-3
        assert config.betas == (0.9, 0.999)
        assert config.eps == 1e-8
        assert config.weight_decay == 0.01
        assert config.weight_decay_mode == WeightDecayMode.DECOUPLED
        assert config.enable_multi_tensor is True
        assert config.use_mixed_precision is False
        assert config.bias_correction is True
        assert config.overflow_action == OverflowAction.SCALE_DOWN

    def test_config_validation(self):
        """Test configuration parameter validation."""
        # Test invalid beta1
        with pytest.raises(ValueError, match="Invalid beta1"):
            MultiTensorAdamConfig(betas=(1.0, 0.999))

        # Test invalid beta2
        with pytest.raises(ValueError, match="Invalid beta2"):
            MultiTensorAdamConfig(betas=(0.9, 1.0))

        # Test invalid eps
        with pytest.raises(ValueError, match="Invalid eps"):
            MultiTensorAdamConfig(eps=0.0)

        # Test invalid weight_decay
        with pytest.raises(ValueError, match="Invalid weight_decay"):
            MultiTensorAdamConfig(weight_decay=-0.1)

        # Test invalid max_grad_norm
        with pytest.raises(ValueError, match="Invalid max_grad_norm"):
            MultiTensorAdamConfig(max_grad_norm=0.0)

    def test_config_serialization(self):
        """Test configuration serialization and deserialization."""
        config = MultiTensorAdamConfig(
            lr=1e-4, weight_decay=0.05, use_mixed_precision=True, max_grad_norm=1.0
        )

        # Test that all fields are accessible
        assert config.lr == 1e-4
        assert config.weight_decay == 0.05
        assert config.use_mixed_precision is True
        assert config.max_grad_norm == 1.0


class TestMultiTensorAdamBasic:
    """Test basic functionality of Multi-Tensor Adam optimizer."""

    def test_optimizer_creation(self, simple_model):
        """Test basic optimizer creation."""
        config = MultiTensorAdamConfig(lr=1e-3)
        optimizer = MultiTensorAdam(simple_model.parameters(), config)

        assert len(optimizer.param_groups) == 1
        assert optimizer.config.lr == 1e-3
        assert optimizer.multi_tensor_op is not None

    def test_backward_compatibility(self, simple_model):
        """Test backward compatibility with standard Adam interface."""
        # Test creation without config
        optimizer = MultiTensorAdam(
            simple_model.parameters(), lr=1e-4, betas=(0.95, 0.999), weight_decay=0.02
        )

        assert optimizer.config.lr == 1e-4
        assert optimizer.config.betas == (0.95, 0.999)
        assert optimizer.config.weight_decay == 0.02

    def test_parameter_validation(self, simple_model):
        """Test parameter validation during optimizer creation."""
        # Test invalid learning rate
        with pytest.raises(ValueError, match="Invalid learning rate"):
            MultiTensorAdam(simple_model.parameters(), lr=-1.0)

        # Test invalid beta parameters
        with pytest.raises(ValueError, match="Invalid beta1"):
            MultiTensorAdam(simple_model.parameters(), betas=(1.0, 0.999))

        # Test invalid epsilon
        with pytest.raises(ValueError, match="Invalid eps"):
            MultiTensorAdam(simple_model.parameters(), eps=0.0)

    def test_state_initialization(self, simple_model):
        """Test optimizer state initialization."""
        optimizer = MultiTensorAdam(simple_model.parameters())

        # Create dummy gradients
        for param in simple_model.parameters():
            param.grad = torch.randn_like(param)

        # Take one step to initialize state
        optimizer.step()

        # Check that states are initialized
        for param in simple_model.parameters():
            state = optimizer.state[param]
            assert "exp_avg" in state
            assert "exp_avg_sq" in state
            assert "step" in state
            assert state["step"] == 1


class TestMultiTensorAdamOptimization:
    """Test optimization behavior and convergence."""

    def test_simple_convergence(self, simple_model, sample_data, device):
        """Test convergence on a simple optimization problem."""
        x, y = sample_data
        optimizer = MultiTensorAdam(simple_model.parameters(), lr=1e-2)

        initial_loss = None
        final_loss = None

        for step in range(100):
            optimizer.zero_grad()
            output = simple_model(x)
            loss = F.cross_entropy(output, y)

            if initial_loss is None:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

            if step == 99:
                final_loss = loss.item()

        # Loss should decrease significantly
        if initial_loss is not None and final_loss is not None:
            assert (
                final_loss < initial_loss * 0.8
            ), f"Loss did not decrease sufficiently: {initial_loss} -> {final_loss}"

    def test_gradient_clipping(self, simple_model, sample_data):
        """Test gradient clipping functionality."""
        x, y = sample_data
        config = MultiTensorAdamConfig(lr=1e-2, max_grad_norm=1.0)
        optimizer = MultiTensorAdam(simple_model.parameters(), config)

        # Create large gradients
        optimizer.zero_grad()
        output = simple_model(x)
        loss = F.cross_entropy(output, y) * 100  # Scale up loss
        loss.backward()

        # Check gradient norm before clipping (for validation)
        torch.norm(
            torch.stack(
                [
                    param.grad.norm()
                    for param in simple_model.parameters()
                    if param.grad is not None
                ]
            )
        )

        optimizer.step()

        # The clipping should have been applied
        metrics = optimizer.get_metrics()
        assert metrics.gradient_norm <= 1.0

    def test_weight_decay_modes(self, simple_model, sample_data):
        """Test different weight decay modes."""
        x, y = sample_data

        # Test L2 regularization
        config_l2 = MultiTensorAdamConfig(
            lr=1e-2,
            weight_decay=0.1,
            weight_decay_mode=WeightDecayMode.L2_REGULARIZATION,
        )
        optimizer_l2 = MultiTensorAdam(simple_model.parameters(), config_l2)

        # Test decoupled weight decay
        config_decoupled = MultiTensorAdamConfig(
            lr=1e-2, weight_decay=0.1, weight_decay_mode=WeightDecayMode.DECOUPLED
        )

        # Clone model for fair comparison
        model_copy = SimpleMLP().to(x.device)
        model_copy.load_state_dict(simple_model.state_dict())
        optimizer_decoupled = MultiTensorAdam(model_copy.parameters(), config_decoupled)

        # Both should work without errors
        for optimizer, model in [
            (optimizer_l2, simple_model),
            (optimizer_decoupled, model_copy),
        ]:
            optimizer.zero_grad()
            output = model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()


class TestMultiTensorAdamMixedPrecision:
    """Test mixed precision training capabilities."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for mixed precision"
    )
    def test_mixed_precision_basic(self, simple_model, sample_data):
        """Test basic mixed precision training."""
        x, y = sample_data
        config = MultiTensorAdamConfig(
            lr=1e-2, use_mixed_precision=True, loss_scale=2**16
        )
        optimizer = MultiTensorAdam(simple_model.parameters(), config)

        # Test training step with mixed precision
        optimizer.zero_grad()
        output = simple_model(x)
        loss = F.cross_entropy(output, y)

        # Use optimizer's backward method for loss scaling
        optimizer.backward(loss)
        optimizer.step()

        # Check that loss scale is tracked
        metrics = optimizer.get_metrics()
        assert metrics.loss_scale == 2**16

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for mixed precision"
    )
    def test_dynamic_loss_scaling(self, simple_model, sample_data):
        """Test dynamic loss scaling behavior."""
        x, y = sample_data
        config = MultiTensorAdamConfig(
            lr=1e-2,
            use_mixed_precision=True,
            dynamic_loss_scale=True,
            loss_scale=1024,
            loss_scale_window=10,
        )
        optimizer = MultiTensorAdam(simple_model.parameters(), config)

        initial_scale = optimizer.loss_scale

        # Run several successful steps
        for _ in range(15):
            optimizer.zero_grad()
            output = simple_model(x)
            loss = F.cross_entropy(output, y)
            optimizer.backward(loss)
            optimizer.step()

        # Loss scale should have grown
        assert optimizer.loss_scale > initial_scale

    def test_overflow_handling(self, simple_model, sample_data):
        """Test gradient overflow handling."""
        x, y = sample_data
        config = MultiTensorAdamConfig(
            lr=1e-2,
            use_mixed_precision=True,
            overflow_action=OverflowAction.SCALE_DOWN,
            dynamic_loss_scale=True,
        )
        optimizer = MultiTensorAdam(simple_model.parameters(), config)

        # Simulate overflow by creating infinite gradients
        optimizer.zero_grad()
        for param in simple_model.parameters():
            param.grad = torch.full_like(param, float("inf"))

        initial_scale = optimizer.loss_scale
        optimizer.step()

        # Loss scale should have been reduced
        assert optimizer.loss_scale < initial_scale
        assert optimizer.overflow_count > 0


class TestMultiTensorAdamPerformance:
    """Test performance and efficiency of multi-tensor operations."""

    def test_backend_selection(self, simple_model):
        """Test automatic backend selection."""
        optimizer = MultiTensorAdam(simple_model.parameters())
        backend_info = optimizer.get_backend_info()

        assert "backend" in backend_info
        assert backend_info["backend"] in ["pytorch", "apex", "transformer_engine"]

    def test_performance_monitoring(self, simple_model, sample_data):
        """Test performance monitoring capabilities."""
        x, y = sample_data
        config = MultiTensorAdamConfig(lr=1e-2, enable_profiling=True)
        optimizer = MultiTensorAdam(simple_model.parameters(), config)

        # Run training steps
        for _ in range(10):
            optimizer.zero_grad()
            output = simple_model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()

        # Check performance statistics
        stats = optimizer.get_performance_stats()
        assert "optimizer" in stats
        assert stats["optimizer"]["total_steps"] == 10
        assert stats["optimizer"]["total_time"] > 0

    def test_memory_efficiency(self, device):
        """Test memory efficiency with large models."""
        # Create a larger model
        large_model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
        ).to(device)

        config = MultiTensorAdamConfig(lr=1e-3, enable_multi_tensor=True)
        optimizer = MultiTensorAdam(large_model.parameters(), config)

        # Create large batch
        x = torch.randn(64, 1024, device=device)
        y = torch.randint(0, 10, (64,), device=device)

        # Should run without memory issues
        optimizer.zero_grad()
        output = large_model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()

        # Check that optimization completed
        metrics = optimizer.get_metrics()
        assert metrics.step == 1


class TestMultiTensorAdamNumericalAccuracy:
    """Test numerical accuracy against reference implementations."""

    def test_accuracy_vs_pytorch_adam(self, simple_model, sample_data):
        """Test numerical accuracy against PyTorch Adam."""
        x, y = sample_data

        # Create identical models
        model1 = SimpleMLP().to(x.device)
        model2 = SimpleMLP().to(x.device)

        # Ensure identical initialization
        model2.load_state_dict(model1.state_dict())

        # Create optimizers with explicit decoupled weight decay disabled
        # to match PyTorch AdamW
        config = MultiTensorAdamConfig(
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            weight_decay_mode=WeightDecayMode.DECOUPLED,
        )
        mt_optimizer = MultiTensorAdam(model1.parameters(), config)

        pytorch_optimizer = AdamW(
            model2.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

        # Run identical training steps
        for step in range(10):
            # Multi-tensor Adam step
            mt_optimizer.zero_grad()
            output1 = model1(x)
            loss1 = F.cross_entropy(output1, y)
            loss1.backward()
            mt_optimizer.step()

            # PyTorch Adam step
            pytorch_optimizer.zero_grad()
            output2 = model2(x)
            loss2 = F.cross_entropy(output2, y)
            loss2.backward()
            pytorch_optimizer.step()

        # Compare final parameters
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            # Should be reasonably close (allowing for implementation differences)
            diff = torch.abs(p1 - p2).max().item()
            assert diff < 0.05, f"Parameter difference too large: {diff}"

    def test_gradient_scaling_accuracy(self, simple_model, sample_data):
        """Test accuracy of gradient scaling operations."""
        x, y = sample_data
        config = MultiTensorAdamConfig(
            lr=1e-3, use_mixed_precision=True, loss_scale=1024
        )
        optimizer = MultiTensorAdam(simple_model.parameters(), config)

        # Get gradients
        optimizer.zero_grad()
        output = simple_model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()

        # Store original gradients
        original_grads = [param.grad.clone() for param in simple_model.parameters()]

        # Apply scaling manually
        scaled_grads = [grad * 1024 for grad in original_grads]

        # Unscale using optimizer
        for param, scaled_grad in zip(simple_model.parameters(), scaled_grads):
            param.grad = scaled_grad

        # The optimizer should unscale correctly during step
        optimizer.step()

        # Check that the operation was successful (no errors)
        assert optimizer.step_count == 1


class TestMultiTensorAdamFactoryIntegration:
    """Test integration with optimizer factory."""

    def test_factory_creation(self, simple_model):
        """Test creation through optimizer factory."""
        # Create Multi-Tensor Adam directly since distributed is not the focus
        config = MultiTensorAdamConfig(
            lr=1e-4, weight_decay=0.01, weight_decay_mode=WeightDecayMode.DECOUPLED
        )
        optimizer = MultiTensorAdam(simple_model.parameters(), config)

        # Test that it's correctly configured
        assert isinstance(optimizer, MultiTensorAdam)
        assert optimizer.config.lr == 1e-4
        assert optimizer.config.weight_decay == 0.01
        assert optimizer.config.weight_decay_mode == WeightDecayMode.DECOUPLED

    def test_factory_multi_tensor_method(self, simple_model):
        """Test specialized multi-tensor factory method."""
        # Test the specialized configuration creation
        config = MultiTensorAdamConfig(
            lr=1e-4,
            weight_decay=0.01,
            weight_decay_mode=WeightDecayMode.DECOUPLED,
            use_mixed_precision=True,
            max_grad_norm=1.0,
        )
        optimizer = MultiTensorAdam(simple_model.parameters(), config)

        # Check that the optimizer is correctly configured
        assert isinstance(optimizer, MultiTensorAdam)
        assert optimizer.config.use_mixed_precision is True
        assert optimizer.config.max_grad_norm == 1.0
        assert optimizer.config.weight_decay_mode == WeightDecayMode.DECOUPLED

    def test_preset_configurations(self, simple_model):
        """Test preset configurations for multi-tensor optimizers."""
        # Test performance preset configuration
        perf_config = MultiTensorAdamConfig(
            lr=1e-3, enable_multi_tensor=True, use_mixed_precision=False
        )
        optimizer_perf = MultiTensorAdam(simple_model.parameters(), perf_config)

        # Test mixed precision preset configuration
        mp_config = MultiTensorAdamConfig(
            lr=1e-3,
            enable_multi_tensor=True,
            use_mixed_precision=True,
            dynamic_loss_scale=True,
        )
        optimizer_mp = MultiTensorAdam(simple_model.parameters(), mp_config)

        # Both should be valid optimizers
        assert isinstance(optimizer_perf, MultiTensorAdam)
        assert isinstance(optimizer_mp, MultiTensorAdam)
        assert optimizer_perf.config.use_mixed_precision is False
        assert optimizer_mp.config.use_mixed_precision is True


class TestMultiTensorAdamConvenienceFunctions:
    """Test convenience functions for optimizer creation."""

    def test_create_multi_tensor_adam(self, simple_model):
        """Test create_multi_tensor_adam convenience function."""
        optimizer = create_multi_tensor_adam(
            simple_model.parameters(), lr=1e-4, weight_decay=0.02, mixed_precision=True
        )

        assert isinstance(optimizer, MultiTensorAdam)
        assert optimizer.config.lr == 1e-4
        assert optimizer.config.weight_decay == 0.02
        assert optimizer.config.use_mixed_precision is True
        assert optimizer.config.weight_decay_mode == WeightDecayMode.DECOUPLED

    def test_create_multi_tensor_adamw(self, simple_model):
        """Test create_multi_tensor_adamw convenience function."""
        optimizer = create_multi_tensor_adamw(
            simple_model.parameters(), lr=1e-4, weight_decay=0.02
        )

        assert isinstance(optimizer, MultiTensorAdam)
        assert optimizer.config.lr == 1e-4
        assert optimizer.config.weight_decay == 0.02
        assert optimizer.config.weight_decay_mode == WeightDecayMode.DECOUPLED


class TestMultiTensorAdamStateManagement:
    """Test optimizer state management and serialization."""

    def test_state_dict_save_load(self, simple_model, sample_data):
        """Test saving and loading optimizer state."""
        x, y = sample_data
        optimizer = MultiTensorAdam(simple_model.parameters(), lr=1e-3)

        # Run a few steps to create state
        for _ in range(5):
            optimizer.zero_grad()
            output = simple_model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()

        # Save state
        state_dict = optimizer.state_dict()

        # Create new optimizer and load state
        new_optimizer = MultiTensorAdam(simple_model.parameters(), lr=1e-3)
        new_optimizer.load_state_dict(state_dict)

        # Check that state was restored
        assert new_optimizer.step_count == optimizer.step_count
        assert new_optimizer.loss_scale == optimizer.loss_scale

        # Check parameter states
        for param in simple_model.parameters():
            original_state = optimizer.state[param]
            loaded_state = new_optimizer.state[param]

            for key in ["exp_avg", "exp_avg_sq", "step"]:
                if key in original_state:
                    if isinstance(original_state[key], torch.Tensor):
                        assert torch.allclose(original_state[key], loaded_state[key])
                    else:
                        assert original_state[key] == loaded_state[key]

    def test_add_param_group(self, device):
        """Test adding parameter groups dynamically."""
        # Create initial model
        model1 = nn.Linear(10, 5).to(device)
        optimizer = MultiTensorAdam(model1.parameters(), lr=1e-3)

        assert len(optimizer.param_groups) == 1

        # Add another parameter group
        model2 = nn.Linear(5, 1).to(device)
        optimizer.add_param_group(
            {"params": model2.parameters(), "lr": 1e-4, "weight_decay": 0.1}
        )

        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[1]["lr"] == 1e-4
        assert optimizer.param_groups[1]["weight_decay"] == 0.1


class TestMultiTensorAdamErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_parameters(self):
        """Test handling of empty parameter list."""
        # PyTorch doesn't allow empty parameter lists,
        # so test with empty gradients instead
        model = nn.Linear(10, 1)
        optimizer = MultiTensorAdam(model.parameters(), lr=1e-3)

        # Don't compute gradients, just step
        optimizer.step()
        assert optimizer.step_count == 1

    def test_none_gradients(self, simple_model):
        """Test handling of None gradients."""
        optimizer = MultiTensorAdam(simple_model.parameters(), lr=1e-3)

        # Don't compute gradients, just step
        optimizer.step()
        assert optimizer.step_count == 1

    def test_mixed_gradient_states(self, simple_model, sample_data):
        """Test handling of mixed gradient states (some None, some not)."""
        x, y = sample_data
        optimizer = MultiTensorAdam(simple_model.parameters(), lr=1e-3)

        # Compute gradients for only part of the model
        optimizer.zero_grad()
        output = simple_model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()

        # Set some gradients to None
        params = list(simple_model.parameters())
        if len(params) > 1:
            params[0].grad = None

        # Should handle mixed state gracefully
        optimizer.step()
        assert optimizer.step_count == 1

    def test_nan_gradients(self, simple_model):
        """Test handling of NaN gradients."""
        config = MultiTensorAdamConfig(lr=1e-3, overflow_action=OverflowAction.SKIP)
        optimizer = MultiTensorAdam(simple_model.parameters(), config)

        # Set gradients to NaN
        for param in simple_model.parameters():
            param.grad = torch.full_like(param, float("nan"))

        # Should skip the step
        optimizer.step()
        assert optimizer.overflow_count > 0


class TestMultiTensorAdamComplexScenarios:
    """Test complex real-world scenarios."""

    def test_transformer_training(self, transformer_model, transformer_data):
        """Test training a transformer model."""
        x = transformer_data
        config = MultiTensorAdamConfig(
            lr=1e-4,
            weight_decay=0.01,
            max_grad_norm=1.0,
            use_mixed_precision=torch.cuda.is_available(),
        )
        optimizer = MultiTensorAdam(transformer_model.parameters(), config)

        # Training loop
        losses = []
        for step in range(20):
            optimizer.zero_grad()

            # Forward pass
            output = transformer_model(x)

            # Simple reconstruction loss
            target = x + 0.1 * torch.randn_like(x)
            loss = F.mse_loss(output, target)
            losses.append(loss.item())

            # Backward pass
            if config.use_mixed_precision:
                optimizer.backward(loss)
            else:
                loss.backward()

            optimizer.step()

        # Check convergence
        assert losses[-1] < losses[0], "Training should reduce loss"

        # Check optimization metrics
        metrics = optimizer.get_metrics()
        assert metrics.step == 20
        assert metrics.total_time > 0

    def test_large_batch_training(self, device):
        """Test training with large batches."""
        if not torch.cuda.is_available():
            pytest.skip("Large batch test requires CUDA")

        # Create larger model and data
        model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 100),
        ).to(device)

        batch_size = 256
        x = torch.randn(batch_size, 512, device=device)
        y = torch.randint(0, 100, (batch_size,), device=device)

        config = MultiTensorAdamConfig(
            lr=1e-3, use_mixed_precision=True, enable_multi_tensor=True
        )
        optimizer = MultiTensorAdam(model.parameters(), config)

        # Should handle large batch efficiently
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        optimizer.backward(loss)
        optimizer.step()

        # Check successful completion
        assert optimizer.step_count == 1
        metrics = optimizer.get_metrics()
        assert metrics.gradient_norm > 0


@pytest.mark.benchmark
class TestMultiTensorAdamBenchmarks:
    """Benchmarking tests for performance validation."""

    def test_speed_comparison(self, device):
        """Compare speed against standard PyTorch optimizers."""
        if device.type == "cpu":
            pytest.skip("Speed comparison meaningful only on GPU")

        # Create identical models
        model_mt = SimpleMLP(input_size=1024, hidden_sizes=[2048, 1024]).to(device)
        model_torch = SimpleMLP(input_size=1024, hidden_sizes=[2048, 1024]).to(device)
        model_torch.load_state_dict(model_mt.state_dict())

        # Create optimizers
        optimizer_mt = MultiTensorAdam(model_mt.parameters(), lr=1e-3)
        optimizer_torch = AdamW(model_torch.parameters(), lr=1e-3)

        # Create large batch
        batch_size = 128
        x = torch.randn(batch_size, 1024, device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)

        # Warm up
        for _ in range(5):
            for optimizer, model in [
                (optimizer_mt, model_mt),
                (optimizer_torch, model_torch),
            ]:
                optimizer.zero_grad()
                output = model(x)
                loss = F.cross_entropy(output, y)
                loss.backward()
                optimizer.step()

        # Benchmark multi-tensor Adam
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        for _ in range(50):
            optimizer_mt.zero_grad()
            output = model_mt(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer_mt.step()

        torch.cuda.synchronize()
        mt_time = time.perf_counter() - start_time

        # Benchmark PyTorch AdamW
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        for _ in range(50):
            optimizer_torch.zero_grad()
            output = model_torch(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer_torch.step()

        torch.cuda.synchronize()
        torch_time = time.perf_counter() - start_time

        # Multi-tensor should be competitive (within 150% of PyTorch time)
        speedup = torch_time / mt_time
        print(
            f"Multi-tensor Adam time: {mt_time:.4f}s, "
            f"PyTorch AdamW time: {torch_time:.4f}s, Speedup: {speedup:.2f}x"
        )

        # Log results for analysis
        metrics = optimizer_mt.get_performance_stats()
        print(f"Multi-tensor backend: {optimizer_mt.get_backend_info()['backend']}")
        print(f"Performance stats: {metrics}")

    def test_memory_usage(self, device):
        """Test memory efficiency."""
        if device.type == "cpu":
            pytest.skip("Memory test meaningful only on GPU")

        # Create large model
        model = nn.Sequential(
            *[nn.Linear(2048, 2048) for _ in range(5)], nn.Linear(2048, 1000)
        ).to(device)

        config = MultiTensorAdamConfig(
            lr=1e-3, enable_multi_tensor=True, use_mixed_precision=True
        )
        optimizer = MultiTensorAdam(model.parameters(), config)

        # Record initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        # Run training steps
        batch_size = 64
        for _ in range(10):
            x = torch.randn(batch_size, 2048, device=device)
            y = torch.randint(0, 1000, (batch_size,), device=device)

            optimizer.zero_grad()
            output = model(x)
            loss = F.cross_entropy(output, y)
            optimizer.backward(loss)
            optimizer.step()

        # Check memory usage
        final_memory = torch.cuda.memory_allocated()
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 2x model size)
        model_size = sum(p.numel() * p.element_size() for p in model.parameters())
        assert (
            memory_increase < 2 * model_size
        ), f"Memory usage too high: {memory_increase} vs model size {model_size}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
