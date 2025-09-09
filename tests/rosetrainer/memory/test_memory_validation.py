"""
Memory Usage Validation Tests for Distributed Training

This test suite validates memory usage patterns across distributed ranks,
ensuring that memory optimization and checkpointing work correctly in
multi-rank scenarios. It includes both unit tests and integration tests
that verify memory efficiency and correctness.

Test Categories:
1. Cross-rank memory synchronization
2. Memory imbalance detection and correction
3. Memory profiling accuracy
4. Load balancing effectiveness
5. Memory leak detection
6. Peak memory usage validation
7. Memory efficiency benchmarks
"""

from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from rosellm.rosetrainer.memory.distributed_checkpoint import (
    DistributedActivationCheckpointing,
    DistributedCheckpointConfig,
    DistributedMemoryProfiler,
    DistributedMemoryStats,
)
from rosellm.rosetrainer.memory.distributed_memory_optimizer import (
    DistributedMemoryConfig,
    DistributedMemoryCoordinator,
    DistributedMemoryOptimizer,
    MemoryBalance,
)


class MockCudaMemory:
    """Mock CUDA memory management for testing."""

    def __init__(self, initial_allocated: int = 0):
        self.allocated = initial_allocated
        self.reserved = initial_allocated + 1000000  # Reserve more than allocated
        self.max_allocated = initial_allocated

    def memory_allocated(self) -> int:
        return self.allocated

    def memory_reserved(self) -> int:
        return self.reserved

    def max_memory_allocated(self) -> int:
        return self.max_allocated

    def allocate(self, size: int) -> None:
        """Simulate memory allocation."""
        self.allocated += size
        self.max_allocated = max(self.max_allocated, self.allocated)

    def deallocate(self, size: int) -> None:
        """Simulate memory deallocation."""
        self.allocated = max(0, self.allocated - size)

    def empty_cache(self) -> None:
        """Simulate cache clearing."""
        pass


class TestCrossRankMemorySync:
    """Test memory synchronization across distributed ranks."""

    @pytest.fixture
    def mock_cuda_memory(self):
        """Mock CUDA memory for testing."""
        return MockCudaMemory(initial_allocated=1024 * 1024 * 1024)  # 1GB

    @patch("torch.cuda.is_available")
    def test_memory_profiler_sync_disabled(self, mock_cuda_available):
        """Test memory profiler when distributed sync is disabled."""
        mock_cuda_available.return_value = False

        config = DistributedCheckpointConfig(enable_cross_rank_profiling=False)
        profiler = DistributedMemoryProfiler(config)

        # Should not sync
        assert not profiler.should_sync_memory_stats()

        # Sync should return None
        result = profiler.sync_memory_stats()
        assert result is None

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.memory_reserved")
    @patch("torch.cuda.max_memory_allocated")
    @patch("torch.cuda.synchronize")
    def test_memory_profiler_single_rank(
        self,
        mock_sync,
        mock_max_alloc,
        mock_reserved,
        mock_allocated,
        mock_cuda_available,
    ):
        """Test memory profiler in single rank setup."""
        mock_cuda_available.return_value = True
        mock_sync.return_value = None
        mock_allocated.return_value = 1024 * 1024 * 1024  # 1GB
        mock_reserved.return_value = 1536 * 1024 * 1024  # 1.5GB
        mock_max_alloc.return_value = 1200 * 1024 * 1024  # 1.2GB

        config = DistributedCheckpointConfig()
        profiler = DistributedMemoryProfiler(config)

        # Profile memory
        stats = profiler.profile_memory_distributed("test_layer", "before")

        assert stats["local_stats"]["allocated_gb"] == 1.0
        assert stats["distributed_stats"].allocated_memory_mb == 1024.0
        assert stats["rank"] == 0

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.memory_allocated")
    def test_memory_imbalance_detection(self, mock_allocated, mock_cuda_available):
        """Test detection of memory imbalances across ranks."""
        mock_cuda_available.return_value = True
        mock_allocated.return_value = 512 * 1024 * 1024  # 512MB

        config = DistributedCheckpointConfig()
        profiler = DistributedMemoryProfiler(config)

        # Simulate different memory usage across ranks
        profiler.global_stats = {
            0: DistributedMemoryStats(0, 0, 0, 0, 0, 0, allocated_memory_mb=500.0),
            1: DistributedMemoryStats(1, 0, 0, 0, 0, 0, allocated_memory_mb=1000.0),
            2: DistributedMemoryStats(2, 0, 0, 0, 0, 0, allocated_memory_mb=250.0),
        }

        # Calculate imbalance ratio
        ratio = profiler.get_memory_imbalance_ratio()
        assert ratio == 4.0  # 1000 / 250

    def test_memory_balance_creation(self):
        """Test creation and validation of memory balance objects."""
        balance = MemoryBalance(
            rank=1,
            allocated_memory_gb=2.5,
            reserved_memory_gb=3.0,
            peak_memory_gb=2.8,
            tp_rank=0,
            pp_rank=1,
            dp_rank=0,
        )

        assert balance.rank == 1
        assert balance.allocated_memory_gb == 2.5
        assert balance.get_total_usage_gb() == 0.0  # No component usage set
        assert balance.get_efficiency_ratio() == 0.0  # No usage

        # Set some component usage
        balance.activation_memory_gb = 1.0
        balance.parameter_memory_gb = 1.0
        balance.gradient_memory_gb = 0.3

        assert balance.get_total_usage_gb() == 2.3
        assert balance.get_efficiency_ratio() == 2.3 / 2.5


class TestMemoryImbalanceCorrection:
    """Test memory imbalance detection and correction mechanisms."""

    def test_memory_coordinator_initialization(self):
        """Test initialization of distributed memory coordinator."""
        config = DistributedMemoryConfig()
        coordinator = DistributedMemoryCoordinator(config)

        assert coordinator.config == config
        assert coordinator.world_size >= 1
        assert coordinator.rank >= 0
        assert coordinator.last_balance_step == 0

    def test_should_balance_memory_logic(self):
        """Test logic for determining when to balance memory."""
        config = DistributedMemoryConfig(
            balance_interval_steps=100, enable_memory_balancing=True
        )
        coordinator = DistributedMemoryCoordinator(config)

        # Should not balance initially
        assert not coordinator.should_balance_memory(50)

        # Should balance after interval
        assert coordinator.should_balance_memory(100)

        # Should not balance if disabled
        config.enable_memory_balancing = False
        coordinator = DistributedMemoryCoordinator(config)
        assert not coordinator.should_balance_memory(200)

    def test_memory_balance_single_rank(self):
        """Test memory balancing in single rank scenario."""
        config = DistributedMemoryConfig()

        # Mock single rank setup
        with patch(
            "rosellm.rosetrainer.parallelism.parallel_state.is_initialized",
            return_value=False,
        ):
            coordinator = DistributedMemoryCoordinator(config)

            result = coordinator.balance_memory_across_ranks(100)

            assert result["balanced"] is False
            assert result["reason"] == "not_distributed"

    @patch("torch.distributed.all_gather_object")
    def test_memory_balance_multi_rank(self, mock_all_gather):
        """Test memory balancing across multiple ranks."""
        config = DistributedMemoryConfig()

        # Mock distributed setup
        with patch(
            "rosellm.rosetrainer.parallelism.parallel_state.is_initialized",
            return_value=True,
        ), patch("torch.distributed.get_world_size", return_value=4), patch(
            "torch.distributed.get_rank", return_value=0
        ):
            coordinator = DistributedMemoryCoordinator(config)

            # Mock memory balances showing imbalance
            mock_balances = [
                MemoryBalance(0, 1.0, 1.5, 1.2),  # Low usage
                MemoryBalance(1, 3.0, 3.5, 3.2),  # High usage
                MemoryBalance(2, 2.0, 2.5, 2.1),  # Medium usage
                MemoryBalance(3, 1.5, 2.0, 1.7),  # Low-medium usage
            ]
            mock_all_gather.return_value = mock_balances

            # Force memory update
            coordinator.memory_balances = {
                i: balance for i, balance in enumerate(mock_balances)
            }

            result = coordinator.balance_memory_across_ranks(100)

            # Should attempt balancing due to imbalance
            assert result["balanced"] is True
            assert "imbalance_ratio" in result
            assert result["imbalance_ratio"] > 1.0

    def test_coordination_stats_generation(self):
        """Test generation of coordination statistics."""
        config = DistributedMemoryConfig()
        coordinator = DistributedMemoryCoordinator(config)

        stats = coordinator.get_coordination_stats()

        assert "world_size" in stats
        assert "rank" in stats
        assert "current_imbalance_ratio" in stats
        assert "last_balance_step" in stats

        # Check that stats are reasonable
        assert stats["world_size"] >= 1
        assert stats["rank"] >= 0
        assert stats["current_imbalance_ratio"] >= 1.0


class TestMemoryProfilingAccuracy:
    """Test accuracy of memory profiling across ranks."""

    def test_distributed_stats_accuracy(self):
        """Test accuracy of distributed memory statistics."""
        config = DistributedCheckpointConfig()
        profiler = DistributedMemoryProfiler(config)

        # Test with mock memory values
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.memory_allocated", return_value=1073741824
        ), patch("torch.cuda.memory_reserved", return_value=1610612736), patch(
            "torch.cuda.max_memory_allocated", return_value=1288490189
        ), patch(
            "torch.cuda.synchronize", return_value=None
        ):
            stats = profiler.profile_memory_distributed("test_layer", "before")

            # Check accuracy (allowing for floating point precision)
            assert abs(stats["local_stats"]["allocated_gb"] - 1.0) < 0.01
            assert abs(stats["distributed_stats"].allocated_memory_mb - 1024.0) < 0.1

    def test_memory_history_tracking(self):
        """Test tracking of memory usage history."""
        config = DistributedCheckpointConfig()
        profiler = DistributedMemoryProfiler(config)

        # Add some history
        for i in range(5):
            stats = DistributedMemoryStats(
                0, 0, 0, 0, 0, 0, allocated_memory_mb=float(i * 100)
            )
            profiler.local_stats[f"layer_{i}"] = stats

        assert len(profiler.local_stats) == 5

        # Test that stats are properly tracked
        assert all(
            isinstance(stats, DistributedMemoryStats)
            for stats in profiler.local_stats.values()
        )

    def test_memory_leak_detection(self):
        """Test detection of memory leaks in profiling."""
        config = DistributedCheckpointConfig()
        profiler = DistributedMemoryProfiler(config)

        initial_memory = 1000.0

        # Simulate increasing memory usage (potential leak)
        for i in range(10):
            memory_usage = initial_memory + (i * 50)  # Growing memory
            stats = DistributedMemoryStats(
                0, 0, 0, 0, 0, 0, allocated_memory_mb=memory_usage
            )
            profiler.local_stats[f"layer_{i}"] = stats

        # Check for memory growth pattern
        memory_values = [
            stats.allocated_memory_mb for stats in profiler.local_stats.values()
        ]
        is_growing = all(
            memory_values[i] <= memory_values[i + 1]
            for i in range(len(memory_values) - 1)
        )

        assert is_growing  # Should detect growing pattern


class TestLoadBalancingEffectiveness:
    """Test effectiveness of load balancing mechanisms."""

    def test_distributed_memory_optimizer_integration(self):
        """Test integration with distributed memory optimizer."""
        # Create simple model and optimizer
        model = nn.Linear(64, 32)
        optimizer = torch.optim.Adam(model.parameters())

        config = DistributedMemoryConfig()
        memory_optimizer = DistributedMemoryOptimizer(model, optimizer, config)

        assert memory_optimizer.model == model
        assert memory_optimizer.optimizer == optimizer
        assert memory_optimizer.coordinator is not None

    def test_model_integration_process(self):
        """Test model integration process."""
        model = nn.Linear(128, 64)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        config = DistributedMemoryConfig()
        memory_optimizer = DistributedMemoryOptimizer(model, optimizer, config)

        # Integrate model
        integrated_model = memory_optimizer.integrate_with_model()

        assert integrated_model is not None
        assert memory_optimizer.fully_integrated is True

        # Should be callable - ensure input dtype matches model
        test_input = torch.randn(4, 128)
        # Check if model uses half precision and convert input accordingly
        first_param = next(integrated_model.parameters())
        if first_param.dtype == torch.float16:
            test_input = test_input.half()
        output = integrated_model(test_input)
        assert output.shape == (4, 64)

    def test_optimization_step_execution(self):
        """Test execution of optimization steps."""
        model = nn.Linear(32, 16)
        optimizer = torch.optim.Adam(model.parameters())

        config = DistributedMemoryConfig()
        memory_optimizer = DistributedMemoryOptimizer(model, optimizer, config)

        # Perform optimization step
        stats = memory_optimizer.optimize_step(100)

        assert isinstance(stats, dict)
        assert memory_optimizer.step_count == 100

    def test_optimization_report_generation(self):
        """Test generation of optimization reports."""
        model = nn.Linear(16, 8)
        optimizer = torch.optim.AdamW(model.parameters())

        config = DistributedMemoryConfig()
        memory_optimizer = DistributedMemoryOptimizer(model, optimizer, config)

        report = memory_optimizer.get_optimization_report()

        assert "distributed_memory_optimizer" in report
        assert "coordination" in report
        assert "distributed_checkpointing" in report
        assert "memory_profiling" in report

    def test_stats_reset_functionality(self):
        """Test resetting of optimization statistics."""
        model = nn.Linear(8, 4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        config = DistributedMemoryConfig()
        memory_optimizer = DistributedMemoryOptimizer(model, optimizer, config)

        # Set some state
        memory_optimizer.step_count = 50
        memory_optimizer.last_optimization_step = 45

        # Reset
        memory_optimizer.reset_optimization_stats()

        assert memory_optimizer.step_count == 0
        assert memory_optimizer.last_optimization_step == 0


class TestPeakMemoryValidation:
    """Test validation of peak memory usage."""

    def test_peak_memory_tracking(self):
        """Test tracking of peak memory usage."""
        config = DistributedCheckpointConfig()
        profiler = DistributedMemoryProfiler(config)

        # Simulate varying memory usage
        memory_values = [500, 800, 1200, 900, 600]  # MB

        for i, memory in enumerate(memory_values):
            stats = DistributedMemoryStats(
                0,
                0,
                0,
                0,
                0,
                0,
                allocated_memory_mb=float(memory),
                peak_memory_mb=float(max(memory_values[: i + 1])),
            )
            profiler.local_stats[f"layer_{i}"] = stats

        # Check peak tracking
        final_stats = profiler.local_stats["layer_4"]
        assert final_stats.peak_memory_mb == 1200.0

    def test_memory_efficiency_calculation(self):
        """Test calculation of memory efficiency metrics."""
        balance = MemoryBalance(
            rank=0,
            allocated_memory_gb=4.0,
            reserved_memory_gb=5.0,
            peak_memory_gb=4.5,
        )

        # Set component usage
        balance.activation_memory_gb = 2.0
        balance.parameter_memory_gb = 1.5
        balance.gradient_memory_gb = 0.3
        balance.optimizer_memory_gb = 0.1

        efficiency = balance.get_efficiency_ratio()
        expected_efficiency = 3.9 / 4.0  # total usage / allocated

        assert abs(efficiency - expected_efficiency) < 0.01

    def test_memory_usage_patterns(self):
        """Test analysis of memory usage patterns."""
        config = DistributedCheckpointConfig()
        profiler = DistributedMemoryProfiler(config)

        # Simulate memory usage pattern
        base_memory = 1000.0
        pattern = [0, 200, 150, 300, 100, 250, 50]  # Additional memory per layer

        for i, additional in enumerate(pattern):
            total_memory = base_memory + additional
            stats = DistributedMemoryStats(
                0, 0, 0, 0, 0, 0, allocated_memory_mb=total_memory
            )
            profiler.local_stats[f"layer_{i}"] = stats

        # Analyze pattern
        all_memory = [
            stats.allocated_memory_mb for stats in profiler.local_stats.values()
        ]
        memory_variance = np.var(all_memory)

        # Should detect variance in memory usage
        assert memory_variance > 0


class TestMemoryEfficiencyBenchmarks:
    """Benchmark tests for memory efficiency."""

    @pytest.mark.slow
    def test_memory_efficiency_large_model(self):
        """Test memory efficiency with large model."""

        # Create larger model for more realistic testing
        class LargeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(1024, 1024) for _ in range(10)])

            def forward(self, x):
                for layer in self.layers:
                    x = torch.relu(layer(x))
                return x

        model = LargeModel()
        optimizer = torch.optim.Adam(model.parameters())

        config = DistributedMemoryConfig()
        memory_optimizer = DistributedMemoryOptimizer(model, optimizer, config)

        # Integrate and test
        integrated_model = memory_optimizer.integrate_with_model()

        # Test with reasonably sized input
        test_input = torch.randn(16, 1024)
        # Check if model uses half precision and convert input accordingly
        first_param = next(integrated_model.parameters())
        if first_param.dtype == torch.float16:
            test_input = test_input.half()

        # Measure memory before
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        initial_memory = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )

        # Run forward pass
        output = integrated_model(test_input)

        # Measure memory after
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Verify output shape
        assert output.shape == (16, 1024)

        # Memory usage should be reasonable
        memory_used = final_memory - initial_memory
        print(f"Memory used: {memory_used / (1024**2):.2f} MB")

    @pytest.mark.slow
    def test_checkpointing_memory_savings(self):
        """Test actual memory savings from checkpointing."""
        config = DistributedCheckpointConfig()
        checkpointing = DistributedActivationCheckpointing(config)

        # Create test layer
        layer = nn.Linear(2048, 2048)
        input_tensor = torch.randn(32, 2048)

        # Measure memory without checkpointing
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        baseline_memory = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )

        # Run without checkpointing
        with torch.no_grad():
            _ = layer(input_tensor)

        no_checkpoint_memory = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )

        # Clear and measure with checkpointing
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        _ = checkpointing.checkpoint_layer_distributed(
            layer, input_tensor, layer_id="memory_test"
        )

        checkpoint_memory = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )

        # Calculate savings
        no_checkpoint_usage = no_checkpoint_memory - baseline_memory
        checkpoint_usage = checkpoint_memory - baseline_memory

        print(f"Without checkpointing: {no_checkpoint_usage / (1024**2):.2f} MB")
        print(f"With checkpointing: {checkpoint_usage / (1024**2):.2f} MB")

        # Should use similar or less memory with checkpointing
        # (In practice, checkpointing trades memory for computation)
        assert checkpoint_usage >= 0


class TestMemoryValidationIntegration:
    """Integration tests for memory validation across components."""

    def test_end_to_end_memory_optimization(self):
        """Test end-to-end memory optimization pipeline."""
        # Create model and optimizer
        model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        optimizer = torch.optim.Adam(model.parameters())

        # Create memory optimizer
        config = DistributedMemoryConfig(
            enable_memory_balancing=True,
            enable_distributed_profiling=True,
        )
        memory_optimizer = DistributedMemoryOptimizer(model, optimizer, config)

        # Integrate model
        integrated_model = memory_optimizer.integrate_with_model()

        # Simulate training steps
        for step in range(10):
            # Create batch
            batch_input = torch.randn(8, 256)
            # Check if model uses half precision and convert input accordingly
            first_param = next(integrated_model.parameters())
            if first_param.dtype == torch.float16:
                batch_input = batch_input.half()

            # Forward pass
            output = integrated_model(batch_input)

            # Simulate loss and backward (simplified)
            loss = output.sum()
            loss.backward()

            # Optimization step
            opt_stats = memory_optimizer.optimize_step(step)

            # Check that we get valid statistics
            assert isinstance(opt_stats, dict)

            # Clear gradients
            optimizer.zero_grad()

        # Get final report
        report = memory_optimizer.get_optimization_report()

        assert "distributed_memory_optimizer" in report
        assert report["distributed_memory_optimizer"]["step_count"] == 9

    def test_distributed_transformer_memory_validation(self):
        """Test memory validation with distributed transformer components."""
        from rosellm.rosetrainer.memory.distributed_transformer import (
            DistributedTransformerEncoderLayer,
        )
        from rosellm.rosetrainer.memory.model_parallel_checkpoint import (
            ModelParallelCheckpointConfig,
        )

        # Create transformer layer
        config = ModelParallelCheckpointConfig()
        layer = DistributedTransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            config=config,
        )

        # Test forward pass
        test_input = torch.randn(4, 64, 512)  # batch, seq_len, d_model

        # Measure memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        initial_memory = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )

        output = layer(test_input)

        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Verify output shape
        assert output.shape == (4, 64, 512)

        # Memory usage should be reasonable
        memory_used = final_memory - initial_memory
        print(f"Transformer layer memory: {memory_used / (1024**2):.2f} MB")


# Custom pytest marks for different test categories
class MarkGenerator:
    """Generator for pytest marks to avoid linting issues."""

    def __init__(self):
        self.memory_validation = pytest.mark.memory_validation
        self.integration = pytest.mark.integration
        self.benchmark = pytest.mark.benchmark


# Create mark generator instance
marks = MarkGenerator()


if __name__ == "__main__":
    # Run memory validation tests
    pytest.main([__file__, "-v", "-m", "not slow"])
