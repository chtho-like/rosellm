"""Tests for distributed optimizer with parameter partitioning."""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD, Adam

from rosellm.rosetrainer.optimizer import (
    DistributedOptimizer,
    DistributedOptimizerConfig,
    ParameterPartitioner,
    ParameterRange,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(
        self, input_size: int = 10, hidden_size: int = 20, output_size: int = 5
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestDistributedOptimizerConfig:
    """Test configuration for distributed optimizer."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DistributedOptimizerConfig()

        assert config.partition_parameters is True
        assert config.partition_gradients is True
        assert config.partition_optimizer_states is True
        assert config.contiguous_gradients is True
        assert config.mixed_precision is False
        assert config.bucket_size_mb == 25

    def test_config_validation(self):
        """Test configuration validation."""
        # Test that partition_parameters forces partition_gradients
        config = DistributedOptimizerConfig(
            partition_parameters=True, partition_gradients=False
        )
        assert config.partition_gradients is True

        # Test invalid overlap_grad_reduce
        with pytest.raises(ValueError, match="overlap_grad_reduce requires"):
            config = DistributedOptimizerConfig(
                overlap_grad_reduce=True, contiguous_gradients=False
            )

        # Test invalid cpu_offload
        with pytest.raises(ValueError, match="cpu_offload requires"):
            config = DistributedOptimizerConfig(
                cpu_offload=True, partition_optimizer_states=False
            )

    def test_memory_estimation(self):
        """Test memory usage estimation."""
        config = DistributedOptimizerConfig()

        # 1M parameters, Adam optimizer (2 states)
        memory_gb = config.get_memory_usage_gb(1_000_000, optimizer_state_size=2)

        # Should be reasonable
        assert 0.001 < memory_gb < 0.1  # Between 1MB and 100MB

    def test_mixed_precision_dtype(self):
        """Test that mixed precision sets dtype correctly."""
        config = DistributedOptimizerConfig(mixed_precision=True)
        assert config.dtype == torch.float16


class TestParameterRange:
    """Test parameter range functionality."""

    def test_parameter_range_creation(self):
        """Test creating a parameter range."""
        param_range = ParameterRange(
            rank=0,
            start_idx=0,
            end_idx=2,
            param_start_offset=0,
            param_end_offset=100,
            total_elements=200,
            param_indices=[0, 1],
        )

        assert param_range.rank == 0
        assert param_range.total_elements == 200
        assert param_range.contains_param(0)
        assert param_range.contains_param(1)
        assert not param_range.contains_param(2)

    def test_get_param_slice(self):
        """Test getting parameter slices."""
        param_range = ParameterRange(
            rank=0,
            start_idx=0,
            end_idx=3,
            param_start_offset=50,
            param_end_offset=75,
            total_elements=200,
            param_indices=[0, 1, 2],
        )

        # First parameter (partial)
        slice_info = param_range.get_param_slice(0, 100)
        assert slice_info == (50, 100)

        # Middle parameter (full)
        slice_info = param_range.get_param_slice(1, 100)
        assert slice_info == (0, 100)

        # Last parameter (partial)
        slice_info = param_range.get_param_slice(2, 100)
        assert slice_info == (0, 75)

        # Parameter not in range
        slice_info = param_range.get_param_slice(3, 100)
        assert slice_info is None


class TestParameterPartitioner:
    """Test parameter partitioner functionality."""

    def test_partition_single_rank(self):
        """Test partitioning with single rank."""
        partitioner = ParameterPartitioner(world_size=1, rank=0)

        # Create test parameters
        params = [
            nn.Parameter(torch.randn(10, 10)),  # 100 elements
            nn.Parameter(torch.randn(20, 5)),  # 100 elements
        ]

        ranges = partitioner.compute_partition_ranges(params)

        assert len(ranges) == 1
        assert ranges[0].rank == 0
        assert ranges[0].total_elements == 200
        assert ranges[0].param_indices == [0, 1]

    def test_partition_multiple_ranks(self):
        """Test partitioning across multiple ranks."""
        world_size = 4

        # Create test parameters
        params = [
            nn.Parameter(torch.randn(100)),  # 100 elements
            nn.Parameter(torch.randn(100)),  # 100 elements
            nn.Parameter(torch.randn(100)),  # 100 elements
            nn.Parameter(torch.randn(100)),  # 100 elements
        ]

        for rank in range(world_size):
            partitioner = ParameterPartitioner(world_size=world_size, rank=rank)
            ranges = partitioner.compute_partition_ranges(params)

            assert len(ranges) == world_size

            # Each rank should get approximately equal elements (with alignment)
            for r in ranges:
                # With alignment, sizes may vary more
                assert 80 <= r.total_elements <= 120  # Allow for alignment variance

            # Check local range
            local_range = partitioner.get_local_param_range()
            assert local_range is not None
            assert local_range.rank == rank

    def test_uneven_partition(self):
        """Test partitioning with uneven parameter sizes."""
        partitioner = ParameterPartitioner(world_size=2, rank=0)

        # Create uneven parameters
        params = [
            nn.Parameter(torch.randn(150)),  # 150 elements
            nn.Parameter(torch.randn(50)),  # 50 elements
            nn.Parameter(torch.randn(25)),  # 25 elements
        ]

        ranges = partitioner.compute_partition_ranges(params)

        assert len(ranges) == 2

        # With alignment, exact distribution may vary
        # Total is 225 elements across 2 ranks
        total_assigned = sum(r.total_elements for r in ranges)
        assert total_assigned == 225

        # Each rank should get roughly half
        assert 100 <= ranges[0].total_elements <= 130
        assert 95 <= ranges[1].total_elements <= 125

    def test_create_partition_buffer(self):
        """Test creating contiguous buffer for partition."""
        partitioner = ParameterPartitioner(world_size=2, rank=0)

        # Create test parameters
        params = [
            nn.Parameter(torch.ones(100)),
            nn.Parameter(torch.ones(100) * 2),
        ]

        partitioner.compute_partition_ranges(params)
        buffer, offsets = partitioner.create_partition_buffer(params)

        # Check buffer size
        local_range = partitioner.get_local_param_range()
        assert local_range is not None
        assert buffer.numel() == local_range.total_elements

        # Check that data was copied correctly
        assert torch.allclose(buffer[:100], torch.ones(100))

    def test_scatter_parameters(self):
        """Test scattering buffer back to parameters."""
        partitioner = ParameterPartitioner(world_size=1, rank=0)

        # Create test parameters
        params = [
            nn.Parameter(torch.zeros(100)),
            nn.Parameter(torch.zeros(100)),
        ]

        partitioner.compute_partition_ranges(params)
        buffer, offsets = partitioner.create_partition_buffer(params)

        # Modify buffer
        buffer.fill_(3.14)

        # Scatter back
        partitioner.scatter_parameters(buffer, params, offsets)

        # Check parameters were updated
        assert torch.allclose(params[0], torch.ones(100) * 3.14)
        assert torch.allclose(params[1], torch.ones(100) * 3.14)


class TestDistributedOptimizer:
    """Test distributed optimizer functionality."""

    @pytest.fixture
    def mock_dist(self):
        """Mock distributed training environment."""
        with patch("torch.distributed.is_initialized", return_value=True), patch(
            "torch.distributed.get_world_size", return_value=2
        ), patch("torch.distributed.get_rank", return_value=0), patch(
            "torch.distributed.all_reduce"
        ), patch(
            "torch.distributed.all_gather_into_tensor"
        ):
            yield

    def test_optimizer_creation(self, mock_dist):
        """Test creating distributed optimizer."""
        model = SimpleModel()
        config = DistributedOptimizerConfig()

        optimizer = DistributedOptimizer(
            params=model.parameters(),
            optimizer_class=Adam,
            optimizer_kwargs={"lr": 0.001},
            config=config,
        )

        assert optimizer.world_size == 2
        assert optimizer.rank == 0
        assert optimizer.config == config
        assert len(optimizer.local_params) > 0

    def test_zero_grad(self, mock_dist):
        """Test zeroing gradients."""
        model = SimpleModel()
        # Don't partition for this test
        config = DistributedOptimizerConfig(partition_parameters=False)

        optimizer = DistributedOptimizer(
            params=model.parameters(),
            optimizer_class=SGD,
            optimizer_kwargs={"lr": 0.01},
            config=config,
        )

        # Set some gradients
        for param in model.parameters():
            param.grad = torch.randn_like(param)

        # Zero gradients
        optimizer.zero_grad()

        # Check gradients are None
        for param in model.parameters():
            assert param.grad is None

    def test_gradient_reduction(self, mock_dist):
        """Test gradient reduction across ranks."""
        model = SimpleModel()
        config = DistributedOptimizerConfig(
            contiguous_gradients=True,
            partition_parameters=False,  # Don't partition for this test
        )

        optimizer = DistributedOptimizer(
            params=model.parameters(),
            optimizer_class=Adam,
            optimizer_kwargs={"lr": 0.001},
            config=config,
        )

        # Set gradients
        for param in model.parameters():
            param.grad = torch.ones_like(param)

        # Mock all_reduce to verify it's called
        with patch("torch.distributed.all_reduce") as mock_all_reduce:
            optimizer._reduce_gradients()

            # Should have called all_reduce
            assert mock_all_reduce.called

        # Gradients should be scaled by world size (only check local params)
        for param in optimizer.local_params:
            if param.grad is not None:
                expected_value = 1.0 / optimizer.world_size
                expected_grad = torch.ones_like(param) * expected_value
                assert torch.allclose(param.grad, expected_grad)

    def test_gradient_clipping(self, mock_dist):
        """Test gradient clipping."""
        model = SimpleModel()
        config = DistributedOptimizerConfig(
            grad_clip_value=1.0,
            partition_parameters=False,  # Don't partition for this test
        )

        optimizer = DistributedOptimizer(
            params=model.parameters(),
            optimizer_class=Adam,
            optimizer_kwargs={"lr": 0.001},
            config=config,
        )

        # Set large gradients (only on local params)
        for param in optimizer.local_params:
            param.grad = torch.ones_like(param) * 100

        # Clip gradients
        grad_norm = optimizer._clip_gradients()

        # Check norm was computed
        assert grad_norm > 0

        # Check gradients were clipped (only check local params)
        total_norm = 0
        for param in optimizer.local_params:
            if param.grad is not None:
                total_norm += param.grad.norm().item() ** 2
        total_norm = total_norm**0.5

        # Should be close to clip value
        # (config.grad_clip_value is not None due to line 339)
        assert config.grad_clip_value is not None
        assert total_norm <= config.grad_clip_value * 1.1  # Allow small tolerance

    def test_mixed_precision(self, mock_dist):
        """Test mixed precision training."""
        model = SimpleModel().half()  # FP16 model
        config = DistributedOptimizerConfig(
            mixed_precision=True,
            partition_parameters=False,  # Don't partition for clearer testing
        )

        optimizer = DistributedOptimizer(
            params=model.parameters(),
            optimizer_class=Adam,
            optimizer_kwargs={"lr": 0.001},
            config=config,
        )

        # Check FP32 main parameters were created
        assert hasattr(optimizer, "fp32_params")
        # Should have FP32 params for all local params
        assert len(optimizer.fp32_params) == len(optimizer.local_params)

        # Check all FP32 params are float32
        for param, fp32_param in optimizer.fp32_params.items():
            assert fp32_param.dtype == torch.float32

    def test_step_with_valid_gradients(self, mock_dist):
        """Test optimizer step with valid gradients."""
        model = SimpleModel()
        config = DistributedOptimizerConfig(check_gradients=True)

        optimizer = DistributedOptimizer(
            params=model.parameters(),
            optimizer_class=SGD,
            optimizer_kwargs={"lr": 0.01},
            config=config,
        )

        # Set valid gradients
        for param in model.parameters():
            param.grad = torch.randn_like(param) * 0.01

        # Take step
        optimizer.step()

        assert optimizer.step_count == 1
        assert optimizer.overflow_count == 0

    def test_step_with_invalid_gradients(self, mock_dist):
        """Test optimizer step with NaN gradients."""
        model = SimpleModel()
        config = DistributedOptimizerConfig(check_gradients=True)

        optimizer = DistributedOptimizer(
            params=model.parameters(),
            optimizer_class=Adam,
            optimizer_kwargs={"lr": 0.001},
            config=config,
        )

        # Set NaN gradients
        for param in model.parameters():
            param.grad = torch.full_like(param, float("nan"))

        # Take step - should skip due to NaN
        optimizer.step()

        assert optimizer.step_count == 0
        assert optimizer.overflow_count == 1

    def test_state_dict(self, mock_dist):
        """Test saving and loading state dict."""
        model = SimpleModel()
        config = DistributedOptimizerConfig()

        optimizer = DistributedOptimizer(
            params=model.parameters(),
            optimizer_class=Adam,
            optimizer_kwargs={"lr": 0.001},
            config=config,
        )

        # Take a step to create state
        for param in model.parameters():
            param.grad = torch.randn_like(param) * 0.01
        optimizer.step()

        # Get state dict
        state_dict = optimizer.state_dict()

        assert "state" in state_dict
        assert "param_groups" in state_dict
        assert "step_count" in state_dict
        assert state_dict["step_count"] == 1

        # Create new optimizer and load state
        optimizer2 = DistributedOptimizer(
            params=model.parameters(),
            optimizer_class=Adam,
            optimizer_kwargs={"lr": 0.001},
            config=config,
        )

        optimizer2.load_state_dict(state_dict)
        assert optimizer2.step_count == 1

    def test_memory_usage_reporting(self, mock_dist):
        """Test memory usage statistics."""
        model = SimpleModel()
        config = DistributedOptimizerConfig()

        optimizer = DistributedOptimizer(
            params=model.parameters(),
            optimizer_class=Adam,
            optimizer_kwargs={"lr": 0.001},
            config=config,
        )

        # Set gradients to get gradient memory
        for param in model.parameters():
            param.grad = torch.randn_like(param)

        memory_stats = optimizer.get_memory_usage()

        assert "parameters_mb" in memory_stats
        assert "gradients_mb" in memory_stats
        assert "optimizer_states_mb" in memory_stats
        assert "total_mb" in memory_stats

        # Should have some memory usage
        assert memory_stats["total_mb"] > 0

    def test_parameter_groups(self, mock_dist):
        """Test optimizer with multiple parameter groups."""
        model = SimpleModel()

        # Create parameter groups with different learning rates
        param_groups = [
            {"params": model.fc1.parameters(), "lr": 0.001},
            {"params": model.fc2.parameters(), "lr": 0.01},
        ]

        config = DistributedOptimizerConfig()

        optimizer = DistributedOptimizer(
            params=param_groups,
            optimizer_class=SGD,
            optimizer_kwargs={},  # LR specified in groups
            config=config,
        )

        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["lr"] == 0.001
        assert optimizer.param_groups[1]["lr"] == 0.01


class TestIntegration:
    """Integration tests for distributed optimizer."""

    @pytest.fixture
    def mock_dist_env(self):
        """Mock full distributed environment."""
        with patch("torch.distributed.is_initialized", return_value=True), patch(
            "torch.distributed.get_world_size", return_value=4
        ), patch("torch.distributed.get_rank", return_value=0), patch(
            "torch.distributed.all_reduce"
        ) as mock_all_reduce, patch(
            "torch.distributed.all_gather_into_tensor"
        ) as mock_all_gather:
            # Make all_reduce modify tensor in place
            def all_reduce_side_effect(tensor, **kwargs):
                tensor.div_(4)  # Simulate averaging across 4 ranks
                return tensor

            mock_all_reduce.side_effect = all_reduce_side_effect

            yield {"all_reduce": mock_all_reduce, "all_gather": mock_all_gather}

    def test_end_to_end_training_step(self, mock_dist_env):
        """Test complete training step with distributed optimizer."""
        # Create model and data
        model = SimpleModel()
        input_data = torch.randn(32, 10)
        target = torch.randn(32, 5)

        # Create optimizer
        config = DistributedOptimizerConfig(
            partition_parameters=True,
            contiguous_gradients=True,
            grad_clip_value=1.0,
            check_gradients=True,
        )

        optimizer = DistributedOptimizer(
            params=model.parameters(),
            optimizer_class=Adam,
            optimizer_kwargs={"lr": 0.001},
            config=config,
        )

        # Forward pass
        output = model(input_data)
        loss = nn.MSELoss()(output, target)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Verify step was taken
        assert optimizer.step_count == 1
        assert optimizer.overflow_count == 0

        # Verify all_reduce was called for gradient reduction
        assert mock_dist_env["all_reduce"].called

        # Zero gradients for next iteration
        optimizer.zero_grad()

        # Verify gradients are cleared (only check local params when partitioning)
        for param in optimizer.local_params:
            assert param.grad is None

    def test_memory_efficiency(self, mock_dist_env):
        """Test that partitioning reduces memory usage."""

        # Create large model
        class LargeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(1000, 1000) for _ in range(10)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        model = LargeModel()

        # Create partitioned optimizer
        config = DistributedOptimizerConfig(
            partition_parameters=True, partition_optimizer_states=True
        )

        optimizer = DistributedOptimizer(
            params=model.parameters(),
            optimizer_class=Adam,
            optimizer_kwargs={"lr": 0.001},
            config=config,
        )

        # With 4 ranks, each should have ~1/4 of parameters
        total_params = sum(p.numel() for p in model.parameters())
        local_params = sum(p.numel() for p in optimizer.local_params)

        # Local params should be significantly less than total
        # (allowing for some imbalance due to partitioning)
        assert local_params < total_params * 0.4
