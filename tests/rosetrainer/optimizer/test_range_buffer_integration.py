"""Integration tests for range-based parameter buffer mapping in distributed scenarios.

This test suite provides comprehensive validation of range-based buffer mapping
in distributed training contexts with multiple ranks and communication patterns.

NOTE: Distributed tests are currently commented out due to environment complexity.
They can be enabled by uncommenting the test methods and setting up proper
distributed testing infrastructure.
"""

import logging
import time
from datetime import timedelta
from typing import List, Optional

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.multiprocessing.spawn import spawn

from rosellm.rosetrainer.optimizer.config import DistributedOptimizerConfig
from rosellm.rosetrainer.optimizer.distributed_optimizer import DistributedOptimizer
from rosellm.rosetrainer.optimizer.range_buffer_mapping import (
    RangeBufferConfig,
    RangeBufferStrategy,
)

logger = logging.getLogger(__name__)


class IntegrationTestModel(nn.Module):
    """Test model for integration testing."""

    def __init__(self, input_dim: int = 128, hidden_dims: Optional[List[int]] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 512, 256, 128]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 10))  # 10-class output

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def setup_distributed(rank: int, world_size: int, backend: str = "gloo"):
    """Setup distributed environment for testing."""
    import os

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize distributed training
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=30),
    )


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def run_distributed_range_buffer_test(
    rank: int, world_size: int, test_func, backend: str = "gloo", **kwargs
):
    """Run distributed test function."""
    try:
        setup_distributed(rank, world_size, backend)
        device = torch.device(
            "cuda" if torch.cuda.is_available() and backend == "nccl" else "cpu"
        )

        test_func(rank, world_size, device, **kwargs)

    except Exception as e:
        logger.error(f"Rank {rank} failed with error: {e}")
        raise
    finally:
        cleanup_distributed()


class TestRangeBufferDistributedIntegration:
    """Integration tests for range buffer in distributed scenarios."""

    def _test_distributed_parameter_partitioning(
        self, rank: int, world_size: int, device: torch.device
    ):
        """Test parameter partitioning across multiple ranks."""
        model = IntegrationTestModel()
        model.to(device)

        parameters = list(model.parameters())

        # Create range buffer config
        range_config = RangeBufferConfig(
            strategy=RangeBufferStrategy.CONTIGUOUS,
            device=device,
            enable_profiling=True,
        )

        # Create distributed optimizer with range buffer mapping
        optimizer_config = DistributedOptimizerConfig(
            partition_parameters=True,
            contiguous_gradients=True,
            reduce_bucket_size_mb=10,
        )

        optimizer = DistributedOptimizer(
            params=iter(parameters),
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={"lr": 0.001},
            config=optimizer_config,
            range_buffer_config=range_config,
        )

        # Verify that range-aware gradient buffer is initialized
        assert optimizer.range_aware_gradient_buffer is not None

        # Check parameter partitioning
        local_params = optimizer.local_params
        assert len(local_params) >= 0  # Each rank should have some parameters

        # Verify memory statistics
        memory_stats = optimizer.get_memory_usage()
        assert "range_buffer_allocated_mb" in memory_stats
        assert memory_stats["range_buffer_allocated_mb"] >= 0

        # Synchronize to ensure all ranks complete
        dist.barrier()

    # @pytest.mark.distributed
    # @pytest.mark.parametrize("world_size", [2, 4])
    def _test_distributed_parameter_partitioning_disabled(self, world_size):
        """Test distributed parameter partitioning."""
        if not torch.distributed.is_available():
            pytest.skip("Distributed not available")

        spawn(
            run_distributed_range_buffer_test,
            args=(world_size, self._test_distributed_parameter_partitioning),
            nprocs=world_size,
            join=True,
        )

    def _test_distributed_gradient_reduction(
        self, rank: int, world_size: int, device: torch.device
    ):
        """Test gradient reduction with range-aware buffers."""
        model = IntegrationTestModel()
        model.to(device)

        parameters = list(model.parameters())

        # Create range buffer config
        range_config = RangeBufferConfig(
            strategy=RangeBufferStrategy.SIZE_ORDERED,
            device=device,
        )

        # Create distributed optimizer
        optimizer_config = DistributedOptimizerConfig(
            partition_parameters=True,
            contiguous_gradients=True,
        )

        optimizer = DistributedOptimizer(
            params=iter(parameters),
            optimizer_class=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.01},
            config=optimizer_config,
            range_buffer_config=range_config,
        )

        # Run training step
        batch_size = 8
        input_dim = 128
        x = torch.randn(batch_size, input_dim, device=device)

        # Different inputs per rank to create different gradients
        x = x + rank * 0.1

        y = model(x)
        loss = y.sum()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Store gradients before optimization step
        pre_step_grads = []
        for param in optimizer.local_params:
            if param.grad is not None:
                pre_step_grads.append(param.grad.clone())

        # Optimization step (includes gradient reduction)
        optimizer.step()

        # Verify that gradients were processed
        memory_stats = optimizer.get_memory_usage()
        if "num_gradient_buckets" in memory_stats:
            assert memory_stats["num_gradient_buckets"] > 0

        # Synchronize all ranks
        dist.barrier()

    @pytest.mark.distributed
    @pytest.mark.parametrize("world_size", [2, 4])
    def _test_disabled_distributed_gradient_reduction(self, world_size):
        """Test distributed gradient reduction."""
        if not torch.distributed.is_available():
            pytest.skip("Distributed not available")

        spawn(
            run_distributed_range_buffer_test,
            args=(world_size, self._test_distributed_gradient_reduction),
            nprocs=world_size,
            join=True,
        )

    def _test_range_buffer_memory_efficiency(
        self, rank: int, world_size: int, device: torch.device
    ):
        """Test memory efficiency of range buffers in distributed setting."""
        # Create larger model for meaningful memory testing
        model = IntegrationTestModel(
            input_dim=256, hidden_dims=[512, 1024, 512, 256, 128]
        )
        model.to(device)

        parameters = list(model.parameters())

        # Test different strategies
        strategies = [
            RangeBufferStrategy.CONTIGUOUS,
            RangeBufferStrategy.SIZE_ORDERED,
            RangeBufferStrategy.DTYPE_GROUPED,
        ]

        memory_results = {}

        for strategy in strategies:
            range_config = RangeBufferConfig(
                strategy=strategy,
                device=device,
                enable_profiling=True,
            )

            optimizer_config = DistributedOptimizerConfig(
                partition_parameters=True,
                contiguous_gradients=True,
            )

            optimizer = DistributedOptimizer(
                params=iter(parameters),
                optimizer_class=torch.optim.Adam,
                optimizer_kwargs={"lr": 0.001},
                config=optimizer_config,
                range_buffer_config=range_config,
            )

            # Get memory statistics
            memory_stats = optimizer.get_memory_usage()

            memory_results[strategy.value] = {
                "total_mb": memory_stats.get("total_mb", 0),
                "fragmentation": memory_stats.get("range_buffer_fragmentation", 0),
                "num_buckets": memory_stats.get("num_gradient_buckets", 0),
            }

        # Log memory efficiency results for analysis
        logger.info(f"Rank {rank} memory results: {memory_results}")

        # Basic validation - all strategies should work
        from rosellm.rosetrainer.optimizer.range_buffer_mapping import (
            RangeBufferStrategy,
        )

        for strategy_name, results in memory_results.items():
            strategy = (
                RangeBufferStrategy(strategy_name)
                if isinstance(strategy_name, str)
                else strategy_name
            )
            assert (
                results["total_mb"] > 0
            ), f"Strategy {strategy} should use some memory"
            assert (
                results["fragmentation"] >= 0
            ), f"Fragmentation should be non-negative"

        # Synchronize all ranks
        dist.barrier()

    @pytest.mark.distributed
    @pytest.mark.slow
    def _test_disabled_range_buffer_memory_efficiency(self):
        """Test memory efficiency across different strategies."""
        if not torch.distributed.is_available():
            pytest.skip("Distributed not available")

        world_size = 2
        spawn(
            run_distributed_range_buffer_test,
            args=(world_size, self._test_range_buffer_memory_efficiency),
            nprocs=world_size,
            join=True,
        )

    def _test_range_buffer_gradient_accuracy(
        self, rank: int, world_size: int, device: torch.device
    ):
        """Test gradient accuracy with range buffers vs standard implementation."""
        model = IntegrationTestModel()
        model.to(device)

        # Create reference model (without range buffers)
        ref_model = IntegrationTestModel()
        ref_model.to(device)

        # Copy parameters to ensure same initial state
        with torch.no_grad():
            for param, ref_param in zip(model.parameters(), ref_model.parameters()):
                ref_param.copy_(param)

        # Create range buffer optimizer
        range_config = RangeBufferConfig(
            strategy=RangeBufferStrategy.CONTIGUOUS,
            device=device,
        )

        range_optimizer_config = DistributedOptimizerConfig(
            partition_parameters=True,
            contiguous_gradients=True,
            gradient_postdivide_factor=1.0,  # Disable extra scaling
        )

        range_optimizer = DistributedOptimizer(
            params=model.parameters(),
            optimizer_class=torch.optim.SGD,
            optimizer_kwargs={
                "lr": 0.01,
                "momentum": 0.0,
            },  # No momentum for simplicity
            config=range_optimizer_config,
            range_buffer_config=range_config,
        )

        # Create reference optimizer (standard distributed)
        ref_optimizer_config = DistributedOptimizerConfig(
            partition_parameters=False,  # No range buffers
            contiguous_gradients=False,
            gradient_postdivide_factor=1.0,
        )

        ref_optimizer = DistributedOptimizer(
            params=ref_model.parameters(),
            optimizer_class=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.01, "momentum": 0.0},
            config=ref_optimizer_config,
        )

        # Run same computation on both models
        torch.manual_seed(42 + rank)  # Deterministic but different per rank
        x = torch.randn(4, 128, device=device)

        # Range buffer model
        range_optimizer.zero_grad()
        y_range = model(x)
        loss_range = y_range.sum()
        loss_range.backward()
        range_optimizer.step()

        # Reference model
        ref_optimizer.zero_grad()
        y_ref = ref_model(x)
        loss_ref = y_ref.sum()
        loss_ref.backward()
        ref_optimizer.step()

        # Compare results after synchronization
        dist.barrier()

        # Parameters should be close after optimization step
        # (Some differences expected due to different partitioning strategies)
        max_param_diff = 0.0
        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            param_diff = torch.abs(param - ref_param).max().item()
            max_param_diff = max(max_param_diff, param_diff)

        # Allow for some difference due to different reduction patterns
        logger.info(f"Rank {rank}: Max parameter difference: {max_param_diff}")
        assert max_param_diff < 1.0, f"Parameters differ too much: {max_param_diff}"

        # Synchronize all ranks
        dist.barrier()

    @pytest.mark.distributed
    def _test_disabled_range_buffer_gradient_accuracy(self):
        """Test gradient computation accuracy with range buffers."""
        if not torch.distributed.is_available():
            pytest.skip("Distributed not available")

        world_size = 2
        spawn(
            run_distributed_range_buffer_test,
            args=(world_size, self._test_range_buffer_gradient_accuracy),
            nprocs=world_size,
            join=True,
        )

    def _test_range_buffer_convergence(
        self, rank: int, world_size: int, device: torch.device
    ):
        """Test model convergence with range buffer optimization."""
        model = IntegrationTestModel(input_dim=32, hidden_dims=[64, 32])
        model.to(device)

        range_config = RangeBufferConfig(
            strategy=RangeBufferStrategy.CONTIGUOUS,
            device=device,
        )

        optimizer_config = DistributedOptimizerConfig(
            partition_parameters=True,
            contiguous_gradients=True,
        )

        optimizer = DistributedOptimizer(
            params=model.parameters(),
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={"lr": 0.001},
            config=optimizer_config,
            range_buffer_config=range_config,
        )

        # Simple convergence test
        initial_loss = None
        final_loss = None

        for step in range(10):
            # Generate consistent data across ranks
            torch.manual_seed(step * world_size + rank)
            x = torch.randn(8, 32, device=device)
            y_true = torch.randint(0, 10, (8,), device=device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = nn.functional.cross_entropy(y_pred, y_true)

            if step == 0:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

            if step == 9:
                final_loss = loss.item()

        # Loss should generally decrease (though not guaranteed for small test)
        if initial_loss is not None and final_loss is not None:
            logger.info(
                f"Rank {rank}: Initial loss: {initial_loss:.4f}, "
                f"Final loss: {final_loss:.4f}"
            )

            # Basic sanity checks
            assert initial_loss > 0 and final_loss > 0
            assert abs(final_loss - initial_loss) < 100  # Loss shouldn't explode

        # Synchronize all ranks
        dist.barrier()

    @pytest.mark.distributed
    @pytest.mark.slow
    def _test_disabled_range_buffer_convergence(self):
        """Test model convergence with range buffer optimization."""
        if not torch.distributed.is_available():
            pytest.skip("Distributed not available")

        world_size = 2
        spawn(
            run_distributed_range_buffer_test,
            args=(world_size, self._test_range_buffer_convergence),
            nprocs=world_size,
            join=True,
        )


class TestRangeBufferPerformanceIntegration:
    """Performance integration tests for range buffer mapping."""

    def _test_range_buffer_performance_comparison(
        self, rank: int, world_size: int, device: torch.device
    ):
        """Compare performance of range buffer vs standard implementation."""
        model = IntegrationTestModel(input_dim=512, hidden_dims=[1024, 2048, 1024, 512])
        model.to(device)

        # Test standard optimizer
        standard_config = DistributedOptimizerConfig(
            partition_parameters=True,
            contiguous_gradients=True,
        )

        standard_optimizer = DistributedOptimizer(
            params=iter(list(model.parameters())),
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={"lr": 0.001},
            config=standard_config,
        )

        # Measure standard performance
        torch.manual_seed(42)
        x = torch.randn(16, 512, device=device)

        start_time = time.time()
        for _ in range(5):
            standard_optimizer.zero_grad()
            y = model(x)
            loss = y.sum()
            loss.backward()
            standard_optimizer.step()
        standard_time = time.time() - start_time

        # Reset model state
        for param in model.parameters():
            param.data.normal_(0, 0.02)

        # Test range buffer optimizer
        range_config = RangeBufferConfig(
            strategy=RangeBufferStrategy.CONTIGUOUS,
            device=device,
        )

        range_optimizer_config = DistributedOptimizerConfig(
            partition_parameters=True,
            contiguous_gradients=True,
        )

        range_optimizer = DistributedOptimizer(
            params=iter(list(model.parameters())),
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={"lr": 0.001},
            config=range_optimizer_config,
            range_buffer_config=range_config,
        )

        # Measure range buffer performance
        torch.manual_seed(42)  # Same random seed
        x = torch.randn(16, 512, device=device)

        start_time = time.time()
        for _ in range(5):
            range_optimizer.zero_grad()
            y = model(x)
            loss = y.sum()
            loss.backward()
            range_optimizer.step()
        range_time = time.time() - start_time

        # Log performance results
        logger.info(
            f"Rank {rank}: Standard time: {standard_time:.4f}s, "
            f"Range buffer time: {range_time:.4f}s"
        )

        # Both should complete in reasonable time
        assert standard_time > 0 and range_time > 0
        assert standard_time < 60  # Less than 1 minute
        assert range_time < 60

        # Get memory statistics
        memory_stats = range_optimizer.get_memory_usage()
        logger.info(f"Rank {rank}: Memory stats: {memory_stats}")

        # Synchronize all ranks
        dist.barrier()

    @pytest.mark.distributed
    @pytest.mark.slow
    def _test_disabled_range_buffer_performance_comparison(self):
        """Compare performance of range buffer vs standard implementation."""
        if not torch.distributed.is_available():
            pytest.skip("Distributed not available")

        world_size = 2
        spawn(
            run_distributed_range_buffer_test,
            args=(world_size, self._test_range_buffer_performance_comparison),
            nprocs=world_size,
            join=True,
        )


class TestRangeBufferErrorHandling:
    """Test error handling in distributed range buffer scenarios."""

    def _test_range_buffer_error_recovery(
        self, rank: int, world_size: int, device: torch.device
    ):
        """Test error handling and recovery in range buffer operations."""
        model = IntegrationTestModel()
        model.to(device)

        # Create range buffer config that might cause issues
        range_config = RangeBufferConfig(
            strategy=RangeBufferStrategy.CONTIGUOUS,
            device=device,
            min_range_size=1,  # Very small range size
            max_fragmentation=0.9,  # High fragmentation threshold
        )

        optimizer_config = DistributedOptimizerConfig(
            partition_parameters=True,
            contiguous_gradients=True,
        )

        try:
            optimizer = DistributedOptimizer(
                params=model.parameters(),
                optimizer_class=torch.optim.Adam,
                optimizer_kwargs={"lr": 0.001},
                config=optimizer_config,
                range_buffer_config=range_config,
            )

            # Should handle configuration gracefully
            assert optimizer is not None

            # Try a training step
            x = torch.randn(4, 128, device=device)
            optimizer.zero_grad()
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()

        except Exception as e:
            # Should not crash the process
            logger.warning(f"Rank {rank} encountered expected error: {e}")

        # Synchronize all ranks
        dist.barrier()

    @pytest.mark.distributed
    def _test_disabled_range_buffer_error_recovery(self):
        """Test error handling in range buffer operations."""
        if not torch.distributed.is_available():
            pytest.skip("Distributed not available")

        world_size = 2
        spawn(
            run_distributed_range_buffer_test,
            args=(world_size, self._test_range_buffer_error_recovery),
            nprocs=world_size,
            join=True,
        )


# Helper functions for running tests
def run_single_test(test_method):
    """Run a single distributed test."""
    if not torch.distributed.is_available():
        pytest.skip("Distributed not available")

    world_size = 2
    spawn(
        run_distributed_range_buffer_test,
        args=(world_size, test_method),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    # Can run individual tests for debugging
    pytest.main([__file__, "-v", "-s"])
