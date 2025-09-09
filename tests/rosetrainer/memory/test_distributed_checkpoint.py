"""
Comprehensive Tests for Distributed Activation Checkpointing

This test suite validates all aspects of the distributed activation checkpointing
system, including coordination across ranks, memory profiling, and integration
with various parallelism dimensions.

Test Categories:
1. Core distributed checkpointing functionality
2. Memory profiler cross-rank synchronization
3. Checkpoint coordinator decision making
4. Model parallel activation management
5. Distributed strategies validation
6. Integration with transformer layers
7. Memory optimization coordination
8. Performance and benchmark tests
"""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

# Import the modules we're testing
from rosellm.rosetrainer.memory.distributed_checkpoint import (
    DistributedActivationCheckpointing,
    DistributedCheckpointConfig,
    DistributedCheckpointCoordinator,
    DistributedCheckpointStrategy,
    DistributedMemoryProfiler,
    DistributedMemoryStats,
    create_distributed_checkpointing,
)
from rosellm.rosetrainer.memory.selective_recompute import (
    SelectionStrategy,
    SelectiveCheckpointConfig,
)


class TestDistributedCheckpointConfig:
    """Test distributed checkpoint configuration validation and functionality."""

    def test_config_creation(self):
        """Test basic configuration creation."""
        config = DistributedCheckpointConfig()

        assert config.strategy == DistributedCheckpointStrategy.COORDINATED
        assert config.coordinate_across_tp == True
        assert config.coordinate_across_pp == False
        assert config.enable_load_balancing == True

    def test_config_validation_success(self):
        """Test successful configuration validation."""
        config = DistributedCheckpointConfig(
            load_balance_threshold=0.5,
            rebalance_interval=50,
            memory_sync_interval=25,
            max_memory_imbalance_ratio=2.0,
        )

        # Should not raise any exception
        config.validate()

    def test_config_validation_failures(self):
        """Test configuration validation failures."""
        # Test invalid load balance threshold
        with pytest.raises(
            ValueError, match="load_balance_threshold must be between 0 and 1"
        ):
            config = DistributedCheckpointConfig(load_balance_threshold=1.5)
            config.validate()

        # Test invalid rebalance interval
        with pytest.raises(ValueError, match="rebalance_interval must be positive"):
            config = DistributedCheckpointConfig(rebalance_interval=0)
            config.validate()

        # Test invalid memory sync interval
        with pytest.raises(ValueError, match="memory_sync_interval must be positive"):
            config = DistributedCheckpointConfig(memory_sync_interval=-1)
            config.validate()

        # Test invalid memory imbalance ratio
        with pytest.raises(
            ValueError, match="max_memory_imbalance_ratio must be >= 1.0"
        ):
            config = DistributedCheckpointConfig(max_memory_imbalance_ratio=0.5)
            config.validate()


class TestDistributedMemoryStats:
    """Test distributed memory statistics functionality."""

    def test_memory_stats_creation(self):
        """Test creation of memory statistics."""
        stats = DistributedMemoryStats(
            rank=0,
            tensor_parallel_rank=0,
            pipeline_parallel_rank=0,
            data_parallel_rank=0,
            context_parallel_rank=0,
            expert_parallel_rank=0,
            allocated_memory_mb=1024.0,
            reserved_memory_mb=1536.0,
        )

        assert stats.rank == 0
        assert stats.allocated_memory_mb == 1024.0
        assert stats.reserved_memory_mb == 1536.0
        assert stats.active_checkpoints == 0

    def test_memory_stats_calculations(self):
        """Test memory statistics calculations."""
        stats = DistributedMemoryStats(
            rank=0,
            tensor_parallel_rank=0,
            pipeline_parallel_rank=0,
            data_parallel_rank=0,
            context_parallel_rank=0,
            expert_parallel_rank=0,
            allocated_memory_mb=1024.0,
            checkpoint_memory_saved_mb=256.0,
        )

        # Test that timestamp is set
        assert stats.timestamp > 0


class TestDistributedMemoryProfiler:
    """Test distributed memory profiler functionality."""

    def test_profiler_creation(self):
        """Test profiler creation."""
        config = DistributedCheckpointConfig()
        profiler = DistributedMemoryProfiler(config)

        assert profiler.config == config
        assert profiler.world_size >= 1
        assert profiler.rank >= 0

    def test_profile_memory_distributed(self):
        """Test distributed memory profiling."""
        config = DistributedCheckpointConfig()
        profiler = DistributedMemoryProfiler(config)

        # Profile a layer
        stats = profiler.profile_memory_distributed("test_layer", "before")

        assert "local_stats" in stats
        assert "distributed_stats" in stats
        assert "rank" in stats
        assert "parallel_ranks" in stats

    def test_should_sync_memory_stats(self):
        """Test memory synchronization decision logic."""
        config = DistributedCheckpointConfig(enable_cross_rank_profiling=True)
        profiler = DistributedMemoryProfiler(config)

        # Should sync when cross-rank profiling is enabled
        should_sync = profiler.should_sync_memory_stats()
        assert isinstance(should_sync, bool)

    def test_memory_imbalance_calculation(self):
        """Test memory imbalance ratio calculation."""
        config = DistributedCheckpointConfig()
        profiler = DistributedMemoryProfiler(config)

        # Add mock global stats
        profiler.global_stats = {
            0: DistributedMemoryStats(0, 0, 0, 0, 0, 0, allocated_memory_mb=1000.0),
            1: DistributedMemoryStats(1, 0, 0, 0, 0, 0, allocated_memory_mb=2000.0),
        }

        ratio = profiler.get_memory_imbalance_ratio()
        assert ratio == 2.0

    def test_distributed_memory_report(self):
        """Test generation of distributed memory report."""
        config = DistributedCheckpointConfig()
        profiler = DistributedMemoryProfiler(config)

        report = profiler.get_distributed_memory_report()

        assert "local_report" in report
        assert "distributed_enabled" in report
        assert "rank" in report

        if profiler.is_distributed:
            assert "world_size" in report
            assert "parallel_ranks" in report
            assert "global_memory_stats" in report

    def test_reset_distributed_stats(self):
        """Test resetting distributed statistics."""
        config = DistributedCheckpointConfig()
        profiler = DistributedMemoryProfiler(config)

        # Add some data
        profiler.local_stats["test"] = DistributedMemoryStats(0, 0, 0, 0, 0, 0)

        # Reset
        profiler.reset_distributed_stats()

        assert len(profiler.local_stats) == 0
        assert len(profiler.global_stats) == 0


class TestDistributedCheckpointCoordinator:
    """Test distributed checkpoint coordinator functionality."""

    def test_coordinator_creation(self):
        """Test coordinator creation."""
        config = DistributedCheckpointConfig()
        coordinator = DistributedCheckpointCoordinator(config)

        assert coordinator.config == config
        assert coordinator.world_size >= 1
        assert coordinator.rank >= 0

    def test_coordination_decision_single_rank(self):
        """Test coordination decisions in single-rank setup."""
        config = DistributedCheckpointConfig()

        with patch(
            "rosellm.rosetrainer.parallelism.parallel_state.is_initialized",
            return_value=False,
        ):
            coordinator = DistributedCheckpointCoordinator(config)

            # Should always return True for single rank
            decision = coordinator.coordinate_checkpoint_decision("test_layer")
            assert decision == True

    def test_coordination_stats(self):
        """Test coordination statistics generation."""
        config = DistributedCheckpointConfig()
        coordinator = DistributedCheckpointCoordinator(config)

        stats = coordinator.get_coordination_stats()

        assert "strategy" in stats
        assert "is_distributed" in stats
        assert "world_size" in stats
        assert "rank" in stats
        assert "parallel_config" in stats
        assert "coordination_settings" in stats

    def test_reset_coordination_state(self):
        """Test resetting coordination state."""
        config = DistributedCheckpointConfig()
        coordinator = DistributedCheckpointCoordinator(config)

        # Add some state
        coordinator.checkpoint_decisions["test"] = True

        # Reset
        coordinator.reset_coordination_state()

        assert len(coordinator.checkpoint_decisions) == 0


class TestDistributedActivationCheckpointing:
    """Test main distributed activation checkpointing class."""

    def test_checkpointing_creation(self):
        """Test creation of distributed checkpointing."""
        config = DistributedCheckpointConfig()
        checkpointing = DistributedActivationCheckpointing(config)

        assert checkpointing.config == config
        assert checkpointing.profiler is not None
        assert checkpointing.coordinator is not None
        assert checkpointing.step_count == 0

    def test_checkpoint_layer_distributed(self):
        """Test distributed layer checkpointing."""
        config = DistributedCheckpointConfig()
        checkpointing = DistributedActivationCheckpointing(config)

        # Create a simple test layer
        def test_layer(x):
            return x * 2

        input_tensor = torch.randn(4, 8)

        # Should not raise exception
        result = checkpointing.checkpoint_layer_distributed(
            test_layer, input_tensor, layer_id="test_layer"
        )

        # Should return expected result
        expected = test_layer(input_tensor)
        assert torch.allclose(result, expected)

    def test_checkpoint_layer_with_error_handling(self):
        """Test error handling in layer checkpointing."""
        config = DistributedCheckpointConfig()
        checkpointing = DistributedActivationCheckpointing(config)

        # Test with None layer
        with pytest.raises(ValueError, match="Layer cannot be None"):
            checkpointing.checkpoint_layer_distributed(None, torch.randn(4, 8))

    def test_profiling_report_generation(self):
        """Test generation of distributed profiling report."""
        config = DistributedCheckpointConfig()
        checkpointing = DistributedActivationCheckpointing(config)

        report = checkpointing.get_distributed_profiling_report()

        assert "distributed_checkpointing" in report
        assert "selective_checkpointing" in report
        assert "activation_checkpointing" in report

        # Check distributed checkpointing section
        dist_section = report["distributed_checkpointing"]
        assert "config" in dist_section
        assert "memory_profiling" in dist_section
        assert "coordination" in dist_section

    def test_reset_distributed_profiling(self):
        """Test resetting distributed profiling statistics."""
        config = DistributedCheckpointConfig()
        checkpointing = DistributedActivationCheckpointing(config)

        # Increment step count
        checkpointing.step_count = 10

        # Reset
        checkpointing.reset_distributed_profiling()

        assert checkpointing.step_count == 0


class TestDistributedCheckpointingIntegration:
    """Test integration with transformer layers and models."""

    def test_transformer_integration(self):
        """Test integration with transformer layers."""
        config = DistributedCheckpointConfig()
        checkpointing = DistributedActivationCheckpointing(config)

        # Create a simple model structure
        class SimpleTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = nn.ModuleDict(
                    {
                        "h": nn.ModuleList(
                            [
                                nn.Linear(64, 64),
                                nn.Linear(64, 64),
                            ]
                        )
                    }
                )

            def forward(self, x):
                for layer in self.transformer.h:
                    x = layer(x)
                return x

        model = SimpleTransformer()

        # Apply distributed checkpointing
        enhanced_model = checkpointing.apply_to_transformer_layers_distributed(
            model, layer_attr="transformer.h"
        )

        # Model should still be callable
        input_tensor = torch.randn(2, 64)
        output = enhanced_model(input_tensor)
        assert output.shape == (2, 64)

    def test_factory_function(self):
        """Test factory function for creating distributed checkpointing."""
        checkpointing = create_distributed_checkpointing(
            strategy=DistributedCheckpointStrategy.HIERARCHICAL,
            coordinate_tp=True,
            enable_load_balancing=True,
        )

        assert isinstance(checkpointing, DistributedActivationCheckpointing)
        assert (
            checkpointing.config.strategy == DistributedCheckpointStrategy.HIERARCHICAL
        )
        assert checkpointing.config.coordinate_across_tp == True
        assert checkpointing.config.enable_load_balancing == True


class TestDistributedCheckpointingPerformance:
    """Test performance characteristics of distributed checkpointing."""

    def test_memory_overhead(self):
        """Test memory overhead of distributed checkpointing."""
        config = DistributedCheckpointConfig()
        checkpointing = DistributedActivationCheckpointing(config)

        # Create test layer
        layer = nn.Linear(1024, 1024)
        input_tensor = torch.randn(16, 1024)

        # Measure baseline memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        baseline_memory = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )

        # Run with checkpointing
        result = checkpointing.checkpoint_layer_distributed(
            layer, input_tensor, layer_id="test_layer"
        )

        # Measure memory with checkpointing
        checkpoint_memory = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )

        # Verify result is correct
        expected = layer(input_tensor)
        assert torch.allclose(result, expected)

        # Memory overhead should be reasonable (this is a basic sanity check)
        memory_overhead = checkpoint_memory - baseline_memory
        assert memory_overhead >= 0  # Should not have negative overhead

    def test_computation_overhead(self):
        """Test computational overhead of distributed checkpointing."""
        import time

        config = DistributedCheckpointConfig()
        checkpointing = DistributedActivationCheckpointing(config)

        # Create test layer
        layer = nn.Linear(512, 512)
        input_tensor = torch.randn(32, 512)

        # Measure baseline time
        start_time = time.time()
        for _ in range(10):
            _ = layer(input_tensor)
        baseline_time = time.time() - start_time

        # Measure time with checkpointing
        start_time = time.time()
        for i in range(10):
            _ = checkpointing.checkpoint_layer_distributed(
                layer, input_tensor, layer_id=f"test_layer_{i}"
            )
        checkpoint_time = time.time() - start_time

        # Overhead should be reasonable (less than 5x)
        overhead_ratio = checkpoint_time / baseline_time
        assert overhead_ratio < 5.0, f"Overhead ratio too high: {overhead_ratio}"


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    def test_invalid_configuration(self):
        """Test handling of invalid configurations."""
        # Should raise error during validation
        with pytest.raises(ValueError):
            config = DistributedCheckpointConfig(load_balance_threshold=2.0)
            config.validate()

    def test_layer_execution_failure(self):
        """Test handling of layer execution failures."""
        config = DistributedCheckpointConfig()
        checkpointing = DistributedActivationCheckpointing(config)

        def failing_layer(x):
            raise RuntimeError("Simulated layer failure")

        input_tensor = torch.randn(4, 8)

        with pytest.raises(RuntimeError):
            checkpointing.checkpoint_layer_distributed(
                failing_layer, input_tensor, layer_id="failing_layer"
            )

    def test_empty_model_integration(self):
        """Test integration with empty or invalid models."""
        config = DistributedCheckpointConfig()
        checkpointing = DistributedActivationCheckpointing(config)

        # Model without transformer structure
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()

        # Should handle gracefully (issue warning but not fail)
        enhanced_model = checkpointing.apply_to_transformer_layers_distributed(model)

        # Model should still work
        input_tensor = torch.randn(2, 10)
        output = enhanced_model(input_tensor)
        assert output.shape == (2, 10)


class TestMultiRankSimulation:
    """Test multi-rank scenarios using mocks."""

    @patch("rosellm.rosetrainer.parallelism.parallel_state.is_initialized")
    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    def test_multi_rank_coordination(
        self, mock_rank, mock_world_size, mock_dist_init, mock_parallel_init
    ):
        """Test coordination across multiple ranks using mocks."""
        # Setup multi-rank scenario
        mock_parallel_init.return_value = True
        mock_dist_init.return_value = True
        mock_world_size.return_value = 4
        mock_rank.return_value = 0

        config = DistributedCheckpointConfig()
        coordinator = DistributedCheckpointCoordinator(config)

        assert coordinator.is_distributed == True
        assert coordinator.world_size == 4
        assert coordinator.rank == 0

        # Test coordination decision
        decision = coordinator.coordinate_checkpoint_decision("test_layer")
        assert isinstance(decision, bool)

    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    @patch("rosellm.rosetrainer.parallelism.parallel_state.is_initialized")
    @patch(
        "rosellm.rosetrainer.parallelism.parallel_state.get_tensor_model_parallel_size"
    )
    @patch(
        "rosellm.rosetrainer.parallelism.parallel_state.get_tensor_model_parallel_group"
    )
    @patch(
        "rosellm.rosetrainer.parallelism.parallel_state.get_pipeline_model_parallel_group"
    )
    @patch("rosellm.rosetrainer.parallelism.parallel_state.get_data_parallel_group")
    @patch("rosellm.rosetrainer.parallelism.parallel_state.get_context_parallel_group")
    @patch(
        "rosellm.rosetrainer.parallelism.parallel_state.get_expert_model_parallel_group"
    )
    @patch(
        "rosellm.rosetrainer.parallelism.parallel_state.get_pipeline_model_parallel_size"
    )
    @patch("rosellm.rosetrainer.parallelism.parallel_state.get_data_parallel_size")
    @patch("rosellm.rosetrainer.parallelism.parallel_state.get_context_parallel_size")
    @patch(
        "rosellm.rosetrainer.parallelism.parallel_state.get_expert_model_parallel_size"
    )
    @patch(
        "rosellm.rosetrainer.parallelism.parallel_state.get_tensor_model_parallel_rank"
    )
    @patch(
        "rosellm.rosetrainer.parallelism.parallel_state.get_pipeline_model_parallel_rank"
    )
    @patch("rosellm.rosetrainer.parallelism.parallel_state.get_data_parallel_rank")
    @patch("rosellm.rosetrainer.parallelism.parallel_state.get_context_parallel_rank")
    @patch(
        "rosellm.rosetrainer.parallelism.parallel_state.get_expert_model_parallel_rank"
    )
    def test_tensor_parallel_coordination(
        self,
        mock_ep_rank,
        mock_cp_rank,
        mock_dp_rank,
        mock_pp_rank,
        mock_tp_rank,
        mock_ep_size,
        mock_cp_size,
        mock_dp_size,
        mock_pp_size,
        mock_ep_group,
        mock_cp_group,
        mock_dp_group,
        mock_pp_group,
        mock_tp_group,
        mock_tp_size,
        mock_parallel_init,
        mock_rank,
        mock_world_size,
    ):
        """Test coordination with tensor parallelism."""
        mock_parallel_init.return_value = True
        mock_world_size.return_value = 2
        mock_rank.return_value = 0
        mock_tp_size.return_value = 2
        mock_pp_size.return_value = 1
        mock_dp_size.return_value = 1
        mock_cp_size.return_value = 1
        mock_ep_size.return_value = 1
        mock_tp_rank.return_value = 0
        mock_pp_rank.return_value = 0
        mock_dp_rank.return_value = 0
        mock_cp_rank.return_value = 0
        mock_ep_rank.return_value = 0

        # Mock the groups
        mock_tp_group.return_value = None
        mock_pp_group.return_value = None
        mock_dp_group.return_value = None
        mock_cp_group.return_value = None
        mock_ep_group.return_value = None

        config = DistributedCheckpointConfig(coordinate_across_tp=True)
        coordinator = DistributedCheckpointCoordinator(config)

        # Should coordinate across TP ranks
        assert coordinator.tp_size == 2


# Pytest fixtures for common test setup
@pytest.fixture
def basic_config():
    """Fixture providing basic distributed checkpoint configuration."""
    return DistributedCheckpointConfig()


@pytest.fixture
def distributed_profiler(basic_config):
    """Fixture providing distributed memory profiler."""
    return DistributedMemoryProfiler(basic_config)


@pytest.fixture
def checkpoint_coordinator(basic_config):
    """Fixture providing checkpoint coordinator."""
    return DistributedCheckpointCoordinator(basic_config)


@pytest.fixture
def distributed_checkpointing(basic_config):
    """Fixture providing distributed activation checkpointing."""
    return DistributedActivationCheckpointing(basic_config)


@pytest.fixture
def simple_model():
    """Fixture providing simple test model."""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(64, 128)
            self.linear2 = nn.Linear(128, 64)

        def forward(self, x):
            x = self.linear1(x)
            x = torch.relu(x)
            x = self.linear2(x)
            return x

    return SimpleModel()


# Benchmark tests (marked as slow)
@pytest.mark.slow
class TestDistributedCheckpointingBenchmarks:
    """Benchmark tests for distributed checkpointing performance."""

    def test_large_model_checkpointing_benchmark(self, distributed_checkpointing):
        """Benchmark checkpointing with large model."""
        # Create large layer
        large_layer = nn.Linear(4096, 4096)
        input_tensor = torch.randn(64, 4096)

        import time

        # Benchmark multiple runs
        times = []
        for i in range(5):
            start_time = time.time()
            _ = distributed_checkpointing.checkpoint_layer_distributed(
                large_layer, input_tensor, layer_id=f"benchmark_layer_{i}"
            )
            times.append(time.time() - start_time)

        avg_time = sum(times) / len(times)
        print(f"Average checkpointing time: {avg_time:.4f}s")

        # Should complete in reasonable time (less than 1 second)
        assert avg_time < 1.0

    def test_memory_scaling_benchmark(self, distributed_checkpointing):
        """Benchmark memory usage scaling."""
        layer_sizes = [256, 512, 1024, 2048]
        memory_usage = []

        for size in layer_sizes:
            layer = nn.Linear(size, size)
            input_tensor = torch.randn(32, size)

            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            initial_memory = (
                torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            )

            _ = distributed_checkpointing.checkpoint_layer_distributed(
                layer, input_tensor, layer_id=f"scale_test_{size}"
            )

            final_memory = (
                torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            )
            memory_usage.append(final_memory - initial_memory)

        print(f"Memory usage scaling: {memory_usage}")

        # Memory usage should scale reasonably with layer size
        # (This is a basic sanity check - actual validation would be more sophisticated)
        assert all(usage >= 0 for usage in memory_usage)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
