"""Tests for decoupled gradient storage module."""

import gc
from typing import Tuple
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from rosellm.rosetrainer.gradient.decoupled_grad import (
    DecoupledGradientBuffer,
    DecoupledGradientConfig,
    DecoupledGradientManager,
)
from rosellm.rosetrainer.optimizer.config import DistributedOptimizerConfig
from rosellm.rosetrainer.optimizer.distributed_optimizer import DistributedOptimizer


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(
        self, input_size: int = 10, hidden_size: int = 20, output_size: int = 5
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TestDecoupledGradientConfig:
    """Tests for DecoupledGradientConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = DecoupledGradientConfig()
        assert config.enabled is True
        assert config.dtype == torch.float32
        assert config.device == "cuda"
        assert config.contiguous_buffer is True
        assert config.persistent_storage is True
        assert config.lazy_init is False
        assert config.buffer_growth_factor == 1.5
        assert config.min_buffer_size_mb == 1.0
        assert config.max_buffer_size_mb == 1024.0

    def test_config_validation(self) -> None:
        """Test configuration validation."""
        # Invalid buffer growth factor
        with pytest.raises(ValueError, match="buffer_growth_factor must be > 1.0"):
            DecoupledGradientConfig(buffer_growth_factor=0.5)

        # Invalid min buffer size
        with pytest.raises(ValueError, match="min_buffer_size_mb must be > 0"):
            DecoupledGradientConfig(min_buffer_size_mb=-1.0)

        # Invalid max < min buffer size
        with pytest.raises(ValueError, match="max_buffer_size_mb.*must be >="):
            DecoupledGradientConfig(min_buffer_size_mb=10.0, max_buffer_size_mb=5.0)

        # Invalid device
        with pytest.raises(ValueError, match="device must be 'cuda' or 'cpu'"):
            DecoupledGradientConfig(device="invalid")

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = DecoupledGradientConfig(
            enabled=False,
            dtype=torch.float16,
            device="cpu",
            contiguous_buffer=False,
            persistent_storage=False,
            lazy_init=True,
            buffer_growth_factor=2.0,
            min_buffer_size_mb=2.0,
            max_buffer_size_mb=512.0,
            enable_profiling=True,
            use_pinned_memory=True,
            async_gpu_transfer=False,
        )
        assert config.enabled is False
        assert config.dtype == torch.float16
        assert config.device == "cpu"
        assert config.contiguous_buffer is False
        assert config.buffer_growth_factor == 2.0


class TestDecoupledGradientBuffer:
    """Tests for DecoupledGradientBuffer."""

    @pytest.fixture
    def simple_params(self) -> Tuple[nn.Parameter, nn.Parameter]:
        """Create simple parameters for testing."""
        param1 = nn.Parameter(torch.randn(10, 5))
        param2 = nn.Parameter(torch.randn(5, 3))
        return param1, param2

    def test_buffer_initialization(
        self, simple_params: Tuple[nn.Parameter, nn.Parameter]
    ) -> None:
        """Test buffer initialization."""
        param1, param2 = simple_params
        config = DecoupledGradientConfig(device="cpu")

        buffer = DecoupledGradientBuffer([param1, param2], config)

        assert buffer.total_numel == param1.numel() + param2.numel()
        assert len(buffer.param_shapes) == 2
        assert buffer.param_shapes[0] == param1.shape
        assert buffer.param_shapes[1] == param2.shape
        assert buffer.gradient_buffer is not None
        assert buffer.gradient_buffer.shape == (buffer.total_numel,)

    def test_lazy_initialization(
        self, simple_params: Tuple[nn.Parameter, nn.Parameter]
    ) -> None:
        """Test lazy buffer initialization."""
        param1, param2 = simple_params
        config = DecoupledGradientConfig(device="cpu", lazy_init=True)

        buffer = DecoupledGradientBuffer([param1, param2], config)

        # Buffer should not be initialized yet
        assert buffer.gradient_buffer is None

        # Trigger initialization through gradient hook
        param1.grad = torch.ones_like(param1)
        buffer._gradient_hook(param1.grad, 0)

        # Now buffer should be initialized
        assert buffer.gradient_buffer is not None

    def test_gradient_storage(
        self, simple_params: Tuple[nn.Parameter, nn.Parameter]
    ) -> None:
        """Test gradient storage and retrieval."""
        param1, param2 = simple_params
        config = DecoupledGradientConfig(device="cpu")

        buffer = DecoupledGradientBuffer([param1, param2], config)

        # Set gradients
        grad1 = torch.ones_like(param1)
        grad2 = torch.ones_like(param2) * 2

        buffer.set_gradient(param1, grad1)
        buffer.set_gradient(param2, grad2)

        # Retrieve gradients
        retrieved_grad1 = buffer.get_gradient(param1)
        retrieved_grad2 = buffer.get_gradient(param2)

        assert retrieved_grad1 is not None
        assert retrieved_grad2 is not None
        assert torch.allclose(retrieved_grad1, grad1)
        assert torch.allclose(retrieved_grad2, grad2)

    def test_gradient_accumulation(
        self, simple_params: Tuple[nn.Parameter, nn.Parameter]
    ) -> None:
        """Test gradient accumulation."""
        param1, param2 = simple_params
        config = DecoupledGradientConfig(device="cpu", persistent_storage=True)

        buffer = DecoupledGradientBuffer([param1, param2], config)

        # Accumulate gradients
        grad1 = torch.ones_like(param1)
        buffer.accumulate_gradient(param1, grad1)
        buffer.accumulate_gradient(param1, grad1)

        # Check accumulated gradient
        accumulated = buffer.get_gradient(param1)
        assert accumulated is not None
        assert torch.allclose(accumulated, grad1 * 2)

    def test_zero_gradients(
        self, simple_params: Tuple[nn.Parameter, nn.Parameter]
    ) -> None:
        """Test zeroing gradients."""
        param1, param2 = simple_params
        config = DecoupledGradientConfig(device="cpu")

        buffer = DecoupledGradientBuffer([param1, param2], config)

        # Set gradients
        buffer.set_gradient(param1, torch.ones_like(param1))
        buffer.set_gradient(param2, torch.ones_like(param2))

        # Zero gradients
        buffer.zero_gradients()

        # Check gradients are zero
        grad1 = buffer.get_gradient(param1)
        grad2 = buffer.get_gradient(param2)

        assert grad1 is not None
        assert grad2 is not None
        assert torch.allclose(grad1, torch.zeros_like(param1))
        assert torch.allclose(grad2, torch.zeros_like(param2))

    def test_scale_gradients(
        self, simple_params: Tuple[nn.Parameter, nn.Parameter]
    ) -> None:
        """Test gradient scaling."""
        param1, param2 = simple_params
        config = DecoupledGradientConfig(device="cpu")

        buffer = DecoupledGradientBuffer([param1, param2], config)

        # Set gradients
        buffer.set_gradient(param1, torch.ones_like(param1))
        buffer.set_gradient(param2, torch.ones_like(param2) * 2)

        # Scale gradients
        scale_factor = 0.5
        buffer.scale_gradients(scale_factor)

        # Check scaled gradients
        grad1 = buffer.get_gradient(param1)
        grad2 = buffer.get_gradient(param2)

        assert grad1 is not None
        assert grad2 is not None
        assert torch.allclose(grad1, torch.ones_like(param1) * scale_factor)
        assert torch.allclose(grad2, torch.ones_like(param2) * 2 * scale_factor)

    def test_gradient_hook(
        self, simple_params: Tuple[nn.Parameter, nn.Parameter]
    ) -> None:
        """Test gradient hook functionality."""
        param1, param2 = simple_params
        config = DecoupledGradientConfig(device="cpu", enabled=True)

        buffer = DecoupledGradientBuffer([param1, param2], config)

        # Simulate gradient from backward pass
        grad = torch.ones_like(param1)
        result = buffer._gradient_hook(grad, 0)

        # Check that gradient was stored
        stored_grad = buffer.get_gradient(param1)
        assert stored_grad is not None
        assert torch.allclose(stored_grad, grad)

        # Check that zero tensor was returned
        assert torch.allclose(result, torch.zeros_like(grad))

    def test_device_transfer(self) -> None:
        """Test moving buffer between devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        param = nn.Parameter(torch.randn(10, 5))
        config = DecoupledGradientConfig(device="cpu")

        buffer = DecoupledGradientBuffer([param], config)

        # Initially on CPU
        assert buffer.gradient_buffer is not None
        assert buffer.gradient_buffer.device.type == "cpu"

        # Move to CUDA
        buffer.to("cuda:0")
        assert buffer.gradient_buffer.device.type == "cuda"
        assert buffer.gradient_buffer.device.index == 0

        # Move back to CPU
        buffer.to("cpu")
        assert buffer.gradient_buffer.device.type == "cpu"

    def test_memory_usage_reporting(
        self, simple_params: Tuple[nn.Parameter, nn.Parameter]
    ) -> None:
        """Test memory usage reporting."""
        param1, param2 = simple_params
        config = DecoupledGradientConfig(device="cpu", enable_profiling=True)

        buffer = DecoupledGradientBuffer([param1, param2], config)

        stats = buffer.get_memory_usage()

        assert "allocated_mb" in stats
        assert "peak_mb" in stats
        assert "num_parameters" in stats
        assert "total_elements" in stats
        assert "accumulation_count" in stats
        assert stats["num_parameters"] == 2
        assert stats["total_elements"] == param1.numel() + param2.numel()


class TestDecoupledGradientManager:
    """Tests for DecoupledGradientManager."""

    @pytest.fixture
    def simple_model(self) -> SimpleModel:
        """Create a simple model for testing."""
        return SimpleModel()

    def test_manager_initialization(self, simple_model: SimpleModel) -> None:
        """Test manager initialization."""
        config = DecoupledGradientConfig(device="cpu")

        manager = DecoupledGradientManager(simple_model, config)

        # Check that buffers were created
        assert len(manager.buffers) > 0
        assert len(manager.all_parameters) == sum(1 for _ in simple_model.parameters())

        # Check that all parameters with gradients have buffers
        for param in simple_model.parameters():
            if param.requires_grad:
                assert param in manager.param_to_buffer

    def test_gradient_operations(self, simple_model: SimpleModel) -> None:
        """Test gradient operations through manager."""
        config = DecoupledGradientConfig(device="cpu")

        manager = DecoupledGradientManager(simple_model, config)

        # Set gradients for all parameters
        for param in simple_model.parameters():
            if param.requires_grad:
                grad = torch.ones_like(param)
                manager.set_gradient(param, grad)

        # Scale gradients
        scale_factor = 0.5
        manager.scale_gradients(scale_factor)

        # Check scaled gradients
        for param in simple_model.parameters():
            if param.requires_grad:
                grad = manager.get_gradient(param)
                assert grad is not None
                expected = torch.ones_like(param) * scale_factor
                assert torch.allclose(grad, expected)

        # Zero gradients
        manager.zero_gradients()

        # Check zeroed gradients
        for param in simple_model.parameters():
            if param.requires_grad:
                grad = manager.get_gradient(param)
                assert grad is not None
                assert torch.allclose(grad, torch.zeros_like(param))

    def test_gradient_sync(self, simple_model: SimpleModel) -> None:
        """Test gradient synchronization between parameters and buffers."""
        config = DecoupledGradientConfig(device="cpu")

        manager = DecoupledGradientManager(simple_model, config)

        # Set parameter gradients
        for param in simple_model.parameters():
            if param.requires_grad:
                param.grad = torch.ones_like(param)

        # Sync from parameters to buffers
        manager.sync_gradients_from_params()

        # Check that gradients are in buffers
        for param in simple_model.parameters():
            if param.requires_grad:
                grad = manager.get_gradient(param)
                assert grad is not None
                assert torch.allclose(grad, torch.ones_like(param))
                # Parameter gradients should be cleared
                assert param.grad is None

        # Modify buffer gradients
        manager.scale_gradients(2.0)

        # Sync from buffers to parameters
        manager.sync_gradients_to_params()

        # Check that parameters have updated gradients
        for param in simple_model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.allclose(param.grad, torch.ones_like(param) * 2.0)

    def test_param_groups(self, simple_model: SimpleModel) -> None:
        """Test creating buffers for parameter groups."""
        config = DecoupledGradientConfig(device="cpu")

        manager = DecoupledGradientManager(simple_model, config)

        # Create parameter groups
        params = list(simple_model.parameters())
        param_groups = [
            {"params": params[:2], "lr": 0.01},
            {"params": params[2:], "lr": 0.001},
        ]

        buffers = manager.create_param_groups(param_groups)

        # Check that correct number of buffers were created
        assert len(buffers) == 2

        # Check that parameters are in correct buffers
        for i, group in enumerate(param_groups):
            # Handle both list and single param cases
            params_obj = group["params"]
            params_list = params_obj if isinstance(params_obj, list) else [params_obj]
            for param in params_list:
                if param.requires_grad:
                    assert manager.param_to_buffer[param] == buffers[i]

    def test_memory_usage(self, simple_model: SimpleModel) -> None:
        """Test memory usage reporting."""
        config = DecoupledGradientConfig(device="cpu", enable_profiling=True)

        manager = DecoupledGradientManager(simple_model, config)

        stats = manager.get_memory_usage()

        assert "total_allocated_mb" in stats
        assert "total_peak_mb" in stats
        assert "num_buffers" in stats
        assert "num_parameters" in stats
        assert "gradient_sync_count" in stats
        assert "buffer_stats" in stats

        assert stats["num_buffers"] > 0
        assert stats["num_parameters"] == sum(1 for _ in simple_model.parameters())


class TestDistributedOptimizerIntegration:
    """Test integration with DistributedOptimizer."""

    @pytest.fixture
    def simple_model(self) -> SimpleModel:
        """Create a simple model for testing."""
        return SimpleModel()

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    def test_optimizer_with_decoupled_gradients(
        self,
        mock_rank: MagicMock,
        mock_world_size: MagicMock,
        mock_is_initialized: MagicMock,
        simple_model: SimpleModel,
    ) -> None:
        """Test DistributedOptimizer with decoupled gradient storage."""
        # Mock distributed environment
        mock_is_initialized.return_value = True
        mock_world_size.return_value = 1
        mock_rank.return_value = 0

        # Create configurations
        optimizer_config = DistributedOptimizerConfig(
            partition_parameters=False,
            verbose=False,
        )

        grad_config = DecoupledGradientConfig(
            enabled=True,
            device="cpu",
            contiguous_buffer=True,
        )

        # Create optimizer with decoupled gradients
        optimizer = DistributedOptimizer(
            simple_model.parameters(),
            torch.optim.Adam,
            {"lr": 0.001},
            optimizer_config,
            decoupled_grad_config=grad_config,
            model=simple_model,
        )

        # Check that decoupled gradient manager was created
        assert optimizer.decoupled_grad_manager is not None

        # Perform a forward/backward pass
        x = torch.randn(32, 10)
        y = simple_model(x)
        loss = y.mean()
        loss.backward()

        # Step optimizer
        optimizer.step()

        # Check that gradients were processed
        assert optimizer.step_count == 1

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    def test_optimizer_gradient_operations(
        self,
        mock_rank: MagicMock,
        mock_world_size: MagicMock,
        mock_is_initialized: MagicMock,
        simple_model: SimpleModel,
    ) -> None:
        """Test gradient operations in DistributedOptimizer."""
        # Mock distributed environment
        mock_is_initialized.return_value = True
        mock_world_size.return_value = 1
        mock_rank.return_value = 0

        # Create configurations
        optimizer_config = DistributedOptimizerConfig(
            partition_parameters=False,
            verbose=False,
        )

        grad_config = DecoupledGradientConfig(
            enabled=True,
            device="cpu",
        )

        # Create optimizer
        optimizer = DistributedOptimizer(
            simple_model.parameters(),
            torch.optim.SGD,
            {"lr": 0.01},
            optimizer_config,
            decoupled_grad_config=grad_config,
            model=simple_model,
        )

        # Set gradients using decoupled storage
        for param in simple_model.parameters():
            if param.requires_grad:
                grad = torch.ones_like(param) * 0.1
                optimizer._set_gradient(param, grad)

        # Check gradients
        for param in simple_model.parameters():
            if param.requires_grad:
                grad = optimizer._get_gradient(param)
                assert grad is not None
                assert torch.allclose(grad, torch.ones_like(param) * 0.1)

        # Zero gradients
        optimizer.zero_grad()

        # Check that gradients are zeroed
        for param in simple_model.parameters():
            if param.requires_grad:
                assert param.grad is None
                # Decoupled gradients should also be zeroed
                if optimizer.decoupled_grad_manager:
                    grad = optimizer.decoupled_grad_manager.get_gradient(param)
                    assert grad is not None
                    assert torch.allclose(grad, torch.zeros_like(param))


class TestEndToEndTraining:
    """End-to-end training tests with decoupled gradients."""

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    def test_training_loop(
        self,
        mock_rank: MagicMock,
        mock_world_size: MagicMock,
        mock_is_initialized: MagicMock,
    ) -> None:
        """Test complete training loop with decoupled gradients."""
        # Mock distributed environment
        mock_is_initialized.return_value = True
        mock_world_size.return_value = 1
        mock_rank.return_value = 0

        # Create model and data
        model = SimpleModel()
        criterion = nn.MSELoss()

        # Create configurations
        optimizer_config = DistributedOptimizerConfig(
            partition_parameters=False,
            verbose=False,
            grad_clip_value=1.0,
        )

        grad_config = DecoupledGradientConfig(
            enabled=True,
            device="cpu",
            contiguous_buffer=True,
            persistent_storage=True,
        )

        # Create optimizer with decoupled gradients
        optimizer = DistributedOptimizer(
            model.parameters(),
            torch.optim.Adam,
            {"lr": 0.001},
            optimizer_config,
            decoupled_grad_config=grad_config,
            model=model,
        )

        # Training loop
        num_steps = 10
        batch_size = 16
        input_size = 10
        output_size = 5

        initial_loss = None
        final_loss = None

        for step in range(num_steps):
            # Generate dummy data
            inputs = torch.randn(batch_size, input_size)
            targets = torch.randn(batch_size, output_size)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if step == 0:
                initial_loss = loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Optimizer step
            optimizer.step()

            if step == num_steps - 1:
                final_loss = loss.item()

        # Check that training happened
        assert optimizer.step_count == num_steps
        assert initial_loss is not None
        assert final_loss is not None
        # Loss should generally decrease (though not guaranteed)
        print(f"Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")

    def test_memory_efficiency(self) -> None:
        """Test memory efficiency of decoupled gradient storage."""
        # Create a larger model
        large_model = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100),
        )

        # Test with decoupled gradients
        config = DecoupledGradientConfig(
            enabled=True,
            device="cpu",
            contiguous_buffer=True,
        )

        manager = DecoupledGradientManager(large_model, config)

        # Simulate gradient computation
        for param in large_model.parameters():
            if param.requires_grad:
                # Simulate gradient
                grad = torch.randn_like(param) * 0.01
                manager.set_gradient(param, grad)
                # Clear parameter gradient to save memory
                param.grad = None

        # Get memory usage
        stats = manager.get_memory_usage()
        total_mb = stats["total_allocated_mb"]

        # Calculate expected memory usage - only for parameters requiring gradients
        total_params = sum(
            p.numel() for p in large_model.parameters() if p.requires_grad
        )
        expected_mb = (total_params * 4) / (1024 * 1024)  # float32 = 4 bytes

        # Check that memory usage is reasonable (within 10% tolerance due to overhead)
        tolerance = max(1.0, expected_mb * 0.1)  # At least 1 MB or 10% tolerance
        assert abs(total_mb - expected_mb) < tolerance

        # Clean up
        manager.release()
        del manager
        gc.collect()

    def test_gradient_accumulation(self) -> None:
        """Test gradient accumulation with decoupled storage."""
        model = SimpleModel()
        config = DecoupledGradientConfig(
            enabled=True,
            device="cpu",
            persistent_storage=True,
        )

        manager = DecoupledGradientManager(model, config)

        # Accumulate gradients over multiple steps
        accumulation_steps = 4

        for step in range(accumulation_steps):
            # Simulate gradients
            for param in model.parameters():
                if param.requires_grad:
                    grad = torch.ones_like(param) * 0.1
                    if step == 0:
                        manager.set_gradient(param, grad)
                    else:
                        manager.accumulate_gradient(param, grad)

        # Check accumulated gradients
        for param in model.parameters():
            if param.requires_grad:
                grad = manager.get_gradient(param)
                assert grad is not None
                expected = torch.ones_like(param) * 0.1 * accumulation_steps
                assert torch.allclose(grad, expected, rtol=1e-5)

        # Scale by accumulation steps
        manager.scale_gradients(1.0 / accumulation_steps)

        # Check scaled gradients
        for param in model.parameters():
            if param.requires_grad:
                grad = manager.get_gradient(param)
                assert grad is not None
                expected = torch.ones_like(param) * 0.1
                assert torch.allclose(grad, expected, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
