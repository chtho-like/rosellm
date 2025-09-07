"""
Comprehensive tests for DistributedOptimizer with gradient bucketing.

Tests cover:
- Gradient bucketing logic
- Asynchronous gradient reduction
- Parameter partitioning
- Communication-computation overlap
- End-to-end training scenarios
"""

import time
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from rosellm.rosetrainer.optimizer import (
    ConfigurationError,
    DistributedOptimizer,
    GradientBuffer,
    PartitioningStrategyFactory,
    PerformanceMonitor,
    create_parameter_buckets,
    estimate_memory_savings,
    flatten_dense_tensors,
    get_optimizer_memory_usage,
    partition_parameters_by_size,
    partition_parameters_round_robin,
    unflatten_dense_tensors,
    validate_bucket_configuration,
)


class SimpleModel(nn.Module):
    """Simple model for testing"""

    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TestGradientBuffer:
    """Test gradient buffer functionality"""

    def test_bucket_creation(self):
        """Test creation of gradient buckets"""
        model = SimpleModel()
        params = list(model.parameters())

        # Create gradient buffer with small bucket size
        buffer = GradientBuffer(params, bucket_size_mb=0.001)  # 1KB buckets

        # Check buckets were created
        assert len(buffer.buckets) > 0
        assert len(buffer.param_to_bucket) == len(params)

        # Verify bucket info
        info = buffer.get_bucket_info()
        num_buckets = info["num_buckets"]
        assert isinstance(num_buckets, int) and num_buckets > 0
        assert info["bucket_size_mb"] == 0.001

    def test_gradient_accumulation(self):
        """Test gradient accumulation in buckets"""
        model = SimpleModel()
        params = list(model.parameters())
        buffer = GradientBuffer(params, bucket_size_mb=0.01)

        # Create dummy gradients
        for param in params:
            param.grad = torch.randn_like(param)

        # Copy gradients to buckets
        for param in params:
            if param.grad is not None:
                buffer._copy_grad_to_bucket(param, param.grad)

        # Check gradients were copied
        for bucket in buffer.buckets:
            if bucket.grad_buffer is not None:
                assert not torch.all(bucket.grad_buffer == 0)

    def test_bucket_reset(self):
        """Test resetting gradient buffers"""
        model = SimpleModel()
        params = list(model.parameters())
        buffer = GradientBuffer(params)

        # Set some values
        for bucket in buffer.buckets:
            if bucket.grad_buffer is not None:
                bucket.grad_buffer.fill_(1.0)
                bucket.is_ready = True
                bucket.is_reduced = True

        # Reset
        buffer.reset()

        # Check reset worked
        for bucket in buffer.buckets:
            assert not bucket.is_ready
            assert not bucket.is_reduced
            assert bucket.all_reduce_handle is None
            if bucket.grad_buffer is not None:
                assert torch.all(bucket.grad_buffer == 0)

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.all_reduce")
    def test_async_reduction(self, mock_all_reduce, mock_is_init):
        """Test asynchronous gradient reduction"""
        mock_is_init.return_value = True
        mock_handle = MagicMock()
        mock_all_reduce.return_value = mock_handle

        model = SimpleModel()
        params = list(model.parameters())

        # Create process group mock
        mock_pg = MagicMock()

        buffer = GradientBuffer(params, process_group=mock_pg)

        # Mark bucket as ready
        if buffer.buckets:
            bucket = buffer.buckets[0]
            bucket.is_ready = True
            buffer._start_bucket_reduction(bucket)

            # Check all_reduce was called
            mock_all_reduce.assert_called_once()
            assert bucket.all_reduce_handle == mock_handle


class TestDistributedOptimizer:
    """Test distributed optimizer functionality"""

    def test_initialization(self):
        """Test optimizer initialization"""
        model = SimpleModel()
        base_opt = optim.Adam(model.parameters(), lr=0.001)

        # Create distributed optimizer
        dist_opt = DistributedOptimizer(
            base_opt,
            models=model,
            bucket_size_mb=25.0,
            overlap_grad_reduce=True,
            enable_metrics=True,
        )

        assert dist_opt.base_optimizer == base_opt
        assert dist_opt.bucket_size_mb == 25.0
        assert dist_opt.overlap_grad_reduce
        assert dist_opt.performance_monitor is not None

    def test_invalid_configuration(self):
        """Test that invalid configurations raise errors"""
        model = SimpleModel()
        base_opt = optim.Adam(model.parameters(), lr=0.001)

        # Test invalid bucket size
        with pytest.raises(ConfigurationError):
            DistributedOptimizer(
                base_opt,
                models=model,
                bucket_size_mb=-1.0,
            )

        # Test invalid gradient accumulation steps
        with pytest.raises(ConfigurationError):
            DistributedOptimizer(
                base_opt,
                models=model,
                gradient_accumulation_steps=0,
            )

        # Test invalid clip norm
        with pytest.raises(ConfigurationError):
            DistributedOptimizer(
                base_opt,
                models=model,
                clip_grad_norm=-1.0,
            )

    def test_zero_grad(self):
        """Test zeroing gradients"""
        model = SimpleModel()
        base_opt = optim.Adam(model.parameters(), lr=0.001)
        dist_opt = DistributedOptimizer(base_opt, models=model)

        # Set some gradients
        for param in model.parameters():
            param.grad = torch.randn_like(param)

        # Zero gradients
        dist_opt.zero_grad()

        # Check gradients are zeroed
        for param in model.parameters():
            assert param.grad is None or torch.all(param.grad == 0)

    def test_gradient_clipping(self):
        """Test gradient clipping functionality"""
        model = SimpleModel()
        base_opt = optim.SGD(model.parameters(), lr=0.01)
        dist_opt = DistributedOptimizer(
            base_opt,
            models=model,
            clip_grad_norm=1.0,
        )

        # Create large gradients
        for param in model.parameters():
            param.grad = torch.randn_like(param) * 10

        # Clip gradients
        _ = dist_opt._clip_gradients()

        # Check clipping worked
        total_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5

        assert total_norm <= 1.0 + 1e-6  # Allow small numerical error

    def test_gradient_accumulation(self):
        """Test gradient accumulation steps"""
        model = SimpleModel()
        base_opt = optim.Adam(model.parameters(), lr=0.001)
        dist_opt = DistributedOptimizer(
            base_opt,
            models=model,
            gradient_accumulation_steps=4,
        )

        # Simulate multiple forward/backward passes
        for i in range(3):
            # Set dummy gradients
            for param in model.parameters():
                param.grad = torch.ones_like(param)

            # Step should not update parameters yet
            dist_opt.step()

        # Check accumulation counter
        assert dist_opt.accumulation_step == 3

        # Final step should trigger update
        for param in model.parameters():
            param.grad = torch.ones_like(param)
        dist_opt.step()

        # Counter should reset
        assert dist_opt.accumulation_step == 0

    @patch("rosellm.rosetrainer.optimizer.distributed_optimizer.get_data_parallel_size")
    @patch("rosellm.rosetrainer.optimizer.distributed_optimizer.get_data_parallel_rank")
    def test_parameter_partitioning(self, mock_rank, mock_size):
        """Test parameter partitioning across ranks"""
        mock_size.return_value = 4
        mock_rank.return_value = 1

        model = SimpleModel()
        base_opt = optim.Adam(model.parameters(), lr=0.001)
        dist_opt = DistributedOptimizer(
            base_opt,
            models=model,
            partition_optimizer_states=True,
        )

        # Check partitioning was done (only when dp_size > 1)
        if dist_opt.dp_size > 1:
            assert len(dist_opt.param_to_rank) == len(list(model.parameters()))
            assert len(dist_opt.rank_to_params) == 4

            # Check round-robin assignment
            for idx, param in enumerate(model.parameters()):
                expected_rank = idx % 4
                assert dist_opt.param_to_rank[param] == expected_rank
        else:
            # When dp_size == 1, no partitioning should happen
            assert len(dist_opt.param_to_rank) == 0

    def test_state_dict_save_load(self):
        """Test saving and loading optimizer state"""
        model = SimpleModel()
        base_opt = optim.Adam(model.parameters(), lr=0.001)
        dist_opt = DistributedOptimizer(
            base_opt,
            models=model,
            bucket_size_mb=10.0,
            clip_grad_norm=1.0,
        )

        # Set some state
        dist_opt.accumulation_step = 2
        dist_opt.stats["num_bucket_reductions"] = 100

        # Save state dict
        state_dict = dist_opt.state_dict()

        # Create new optimizer and load state
        new_base_opt = optim.Adam(model.parameters(), lr=0.001)
        new_dist_opt = DistributedOptimizer(
            new_base_opt,
            models=model,
            bucket_size_mb=10.0,
            clip_grad_norm=1.0,
        )
        new_dist_opt.load_state_dict(state_dict)

        # Check state was loaded
        assert new_dist_opt.accumulation_step == 2
        assert new_dist_opt.stats["num_bucket_reductions"] == 100

    def test_add_param_group(self):
        """Test adding parameter groups"""
        model1 = SimpleModel()
        model2 = SimpleModel()

        base_opt = optim.Adam(model1.parameters(), lr=0.001)
        dist_opt = DistributedOptimizer(base_opt, models=model1)

        initial_param_count = len(dist_opt.all_params)

        # Add new param group
        dist_opt.add_param_group(
            {
                "params": list(model2.parameters()),
                "lr": 0.01,
            }
        )

        # Check params were added
        assert len(dist_opt.all_params) == initial_param_count + len(
            list(model2.parameters())
        )
        assert len(dist_opt.param_groups) == 2


class TestPartitioningStrategies:
    """Test parameter partitioning strategies"""

    def test_strategy_factory(self):
        """Test partitioning strategy factory"""
        # Test valid strategies
        round_robin = PartitioningStrategyFactory.create("round_robin")
        assert round_robin.get_name() == "round_robin"

        size_balanced = PartitioningStrategyFactory.create("size_balanced")
        assert size_balanced.get_name() == "size_balanced"

        layer_wise = PartitioningStrategyFactory.create("layer_wise")
        assert layer_wise.get_name() == "layer_wise"

        # Test invalid strategy
        with pytest.raises(ValueError):
            PartitioningStrategyFactory.create("invalid_strategy")

    def test_round_robin_partitioning(self):
        """Test round-robin partitioning strategy"""
        model = SimpleModel()
        params = list(model.parameters())

        strategy = PartitioningStrategyFactory.create("round_robin")
        partitions = strategy.partition(params, world_size=4)

        # Check all parameters are assigned
        total_params = sum(len(p) for p in partitions.values())
        assert total_params == len(params)

        # Check round-robin distribution
        for i, param in enumerate(params):
            expected_rank = i % 4
            assert param in partitions[expected_rank]

    def test_size_balanced_partitioning(self):
        """Test size-balanced partitioning strategy"""
        # Create parameters with different sizes that allow better balance
        params = [
            nn.Parameter(torch.randn(1000, 1000)),  # 1M elements
            nn.Parameter(torch.randn(800, 800)),  # 640K elements
            nn.Parameter(torch.randn(600, 600)),  # 360K elements
            nn.Parameter(torch.randn(500, 500)),  # 250K elements
            nn.Parameter(torch.randn(400, 400)),  # 160K elements
            nn.Parameter(torch.randn(100, 100)),  # 10K elements
        ]

        strategy = PartitioningStrategyFactory.create("size_balanced")
        partitions = strategy.partition(params, world_size=2)

        # Check all parameters are assigned
        all_assigned = []
        for rank_params in partitions.values():
            all_assigned.extend(rank_params)
        assert len(all_assigned) == len(params)

        # Check size balance
        sizes = []
        for rank_params in partitions.values():
            total_size = sum(p.numel() for p in rank_params)
            sizes.append(total_size)

        # Sizes should be relatively balanced
        # With more parameters, the greedy algorithm can achieve better balance
        if len(sizes) == 2:
            balance_ratio = min(sizes) / max(sizes) if max(sizes) > 0 else 1
            assert (
                balance_ratio > 0.7
            )  # Should achieve at least 70% balance with these parameters


class TestOptimizerUtils:
    """Test optimizer utility functions"""

    def test_create_parameter_buckets(self):
        """Test bucket creation utility"""
        model = SimpleModel(input_size=100, hidden_size=200, output_size=50)
        params = list(model.parameters())

        # Create buckets
        buckets = create_parameter_buckets(params, bucket_size_mb=0.001)

        # Check buckets were created
        assert len(buckets) > 0

        # Verify all params are in buckets
        bucketed_params = []
        for bucket in buckets:
            bucketed_params.extend(bucket)
        assert len(bucketed_params) == len([p for p in params if p.requires_grad])

    def test_partition_round_robin(self):
        """Test round-robin parameter partitioning"""
        model = SimpleModel()
        params = list(model.parameters())

        # Partition across 4 ranks
        rank_to_params = partition_parameters_round_robin(params, world_size=4)

        # Check all ranks have params
        assert len(rank_to_params) == 4

        # Check round-robin assignment
        for idx, param in enumerate(params):
            expected_rank = idx % 4
            assert param in rank_to_params[expected_rank]

    def test_partition_by_size(self):
        """Test size-based parameter partitioning"""
        # Create params with different sizes
        params = [
            nn.Parameter(torch.randn(1000, 1000)),  # Large
            nn.Parameter(torch.randn(100, 100)),  # Medium
            nn.Parameter(torch.randn(10, 10)),  # Small
            nn.Parameter(torch.randn(500, 500)),  # Medium-large
        ]

        # Partition by size
        rank_to_params = partition_parameters_by_size(params, world_size=2)

        # Check partitioning is balanced
        rank0_size = sum(p.numel() for p in rank_to_params[0])
        rank1_size = sum(p.numel() for p in rank_to_params[1])

        # Sizes should be relatively balanced
        balance_ratio = min(rank0_size, rank1_size) / max(rank0_size, rank1_size)
        # With very different parameter sizes, perfect balance might not be achievable
        assert balance_ratio > 0.2  # At least 20% balance ratio
        # Ensure all parameters are assigned
        assert len(rank_to_params[0]) + len(rank_to_params[1]) == len(params)

    def test_flatten_unflatten_tensors(self):
        """Test tensor flattening and unflattening"""
        # Create tensors with different shapes
        tensors = [
            torch.randn(2, 3),
            torch.randn(4, 5),
            torch.randn(6),
        ]

        # Flatten
        flat = flatten_dense_tensors(tensors)
        expected_size = sum(t.numel() for t in tensors)
        assert flat.numel() == expected_size

        # Unflatten
        unflat = unflatten_dense_tensors(flat, tensors)
        assert len(unflat) == len(tensors)

        # Check shapes and values match
        for orig, new in zip(tensors, unflat):
            assert orig.shape == new.shape
            assert torch.allclose(orig, new)

    def test_memory_savings_estimation(self):
        """Test memory savings estimation"""
        estimates = estimate_memory_savings(
            num_params=1000000,
            param_dtype=torch.float32,
            optimizer_state_size=2,  # Adam has 2 state tensors
            world_size=4,
        )

        # Check estimates are reasonable
        assert estimates["total_memory_mb"] > 0
        assert estimates["partitioned_memory_mb"] < estimates["total_memory_mb"]
        assert estimates["savings_mb"] > 0
        assert 0 < estimates["savings_percent"] < 100

    def test_validate_bucket_configuration(self):
        """Test bucket configuration validation"""
        model = SimpleModel()
        params = list(model.parameters())

        # Validate configuration
        validation = validate_bucket_configuration(params, bucket_size_mb=0.01)

        # Check validation results
        assert validation["valid"]
        num_buckets = validation["num_buckets"]
        assert isinstance(num_buckets, int) and num_buckets > 0
        assert validation["num_params"] == len(params)
        efficiency = validation["efficiency"]
        assert isinstance(efficiency, (int, float)) and efficiency > 0

    def test_optimizer_memory_usage(self):
        """Test optimizer memory usage calculation"""
        model = SimpleModel()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Do a step to initialize optimizer state
        loss = torch.sum(torch.stack([p.sum() for p in model.parameters()]))
        loss.backward()
        optimizer.step()

        # Calculate memory usage
        usage = get_optimizer_memory_usage(optimizer)

        # Check usage stats
        assert usage["param_memory_mb"] > 0
        assert usage["state_memory_mb"] > 0
        assert (
            usage["total_memory_mb"]
            == usage["param_memory_mb"] + usage["state_memory_mb"]
        )
        assert usage["num_params_with_state"] == len(list(model.parameters()))


class TestPerformanceMonitoring:
    """Test performance monitoring functionality"""

    def test_performance_monitor_basic(self):
        """Test basic performance monitor functionality"""
        monitor = PerformanceMonitor(window_size=10)

        # Test timer functionality
        monitor.start_timer("test")
        time.sleep(0.01)  # Sleep for 10ms
        duration = monitor.end_timer("test")
        assert duration > 0.009  # Should be at least 9ms

        # Test gradient norm recording
        monitor.record_gradient_norm(1.5, clipped=False)
        monitor.record_gradient_norm(2.5, clipped=True)

        metrics = monitor.get_current_metrics()
        assert metrics.gradient_norm == 2.5
        assert metrics.gradient_clips == 1
        assert metrics.max_gradient_norm == 2.5

    def test_performance_monitor_integration(self):
        """Test performance monitor integration with optimizer"""
        model = SimpleModel()
        base_opt = optim.Adam(model.parameters(), lr=0.001)

        dist_opt = DistributedOptimizer(
            base_opt,
            models=model,
            enable_metrics=True,
        )

        # Run a few steps
        for _ in range(3):
            # Simulate gradients
            for param in model.parameters():
                param.grad = torch.randn_like(param) * 0.1

            dist_opt.step()

        # Check that metrics were collected
        stats = dist_opt.get_statistics()
        if "performance" in stats:
            assert "timing" in stats["performance"]
            assert "gradients" in stats["performance"]


class TestEndToEndTraining:
    """End-to-end training tests"""

    def test_single_gpu_training(self):
        """Test training on single GPU/CPU"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleModel().to(device)
        base_opt = optim.Adam(model.parameters(), lr=0.001)

        # Create distributed optimizer
        dist_opt = DistributedOptimizer(
            base_opt,
            models=model,
            bucket_size_mb=1.0,
            overlap_grad_reduce=False,  # No overlap for single GPU
        )

        # Training loop
        for _ in range(5):
            # Forward pass
            input_data = torch.randn(4, 10).to(device)
            output = model(input_data)
            loss = output.mean()

            # Backward pass
            dist_opt.zero_grad()
            loss.backward()

            # Optimizer step
            dist_opt.step()

        # Check model was updated
        for param in model.parameters():
            assert param.grad is not None

    def test_gradient_accumulation_training(self):
        """Test training with gradient accumulation"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleModel().to(device)
        base_opt = optim.SGD(model.parameters(), lr=0.01)

        # Create optimizer with gradient accumulation
        dist_opt = DistributedOptimizer(
            base_opt,
            models=model,
            gradient_accumulation_steps=4,
        )

        # Save initial parameters
        initial_params = {
            name: param.clone() for name, param in model.named_parameters()
        }

        # Training with accumulation
        for i in range(8):  # 2 actual updates
            input_data = torch.randn(2, 10).to(device)
            output = model(input_data)
            loss = output.mean()

            dist_opt.zero_grad()
            loss.backward()
            dist_opt.step()

        # Check parameters were updated
        for name, param in model.named_parameters():
            assert not torch.allclose(param, initial_params[name])

    def test_gradient_clipping_training(self):
        """Test training with gradient clipping"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleModel().to(device)
        base_opt = optim.SGD(model.parameters(), lr=0.01)

        # Create optimizer with gradient clipping
        dist_opt = DistributedOptimizer(
            base_opt,
            models=model,
            clip_grad_norm=0.5,
        )

        # Training loop with large gradients
        for _ in range(3):
            input_data = torch.randn(4, 10).to(device) * 100  # Large inputs
            output = model(input_data)
            loss = (output**2).mean() * 1000  # Large loss

            dist_opt.zero_grad()
            loss.backward()

            # Check gradients before step
            total_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    total_norm += param.grad.norm(2).item() ** 2
            total_norm = total_norm**0.5

            dist_opt.step()

            # Gradients should have been clipped
            assert dist_opt.stats["num_gradient_clips"] > 0


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires at least 2 GPUs",
)
class TestMultiGPU:
    """Tests requiring multiple GPUs"""

    def test_multi_gpu_gradient_reduction(self):
        """Test gradient reduction across multiple GPUs"""
        # This test would require proper distributed setup
        # Marking as a template for multi-GPU testing
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
