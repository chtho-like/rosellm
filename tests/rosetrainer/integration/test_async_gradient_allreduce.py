"""
Integration tests for async gradient allreduce functionality.

These tests validate the complete async gradient allreduce workflow including:
- Integration with parallel linear layers
- End-to-end gradient synchronization
- Performance with different strategies
- Compatibility with existing parallelism systems
"""

import time
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from rosellm.rosetrainer.parallelism import (
    AsyncAllreduceConfig,
    AsyncAllreduceStrategy,
    AsyncGradientAllreduce,
    ColumnParallelLinear,
    RowParallelLinear,
    async_allreduce_step,
    destroy_async_allreduce,
    initialize_async_allreduce,
    register_parameter_for_async_allreduce,
    sync_async_allreduce,
)


class SimpleModel(nn.Module):
    """Simple model for testing async gradient allreduce."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class ParallelModel(nn.Module):
    """Model using parallel linear layers for testing."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        tp_size: int = 2,
        enable_async: bool = False,
    ):
        super().__init__()

        # Create mock process group for testing
        self.tp_group = MagicMock(spec=dist.ProcessGroup)

        self.column_linear = ColumnParallelLinear(
            input_size,
            hidden_size,
            bias=True,
            tp_group=self.tp_group,
            tp_size=tp_size,
            tp_rank=0,
            enable_async_allreduce=enable_async,
            layer_name="column_layer",
        )

        self.row_linear = RowParallelLinear(
            hidden_size // tp_size,  # Input is partitioned
            output_size,
            bias=True,
            tp_group=self.tp_group,
            tp_size=tp_size,
            tp_rank=0,
            enable_async_allreduce=enable_async,
            layer_name="row_layer",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.column_linear(x))
        return self.row_linear(x)


class TestAsyncGradientAllreduceIntegration:
    """Integration tests for async gradient allreduce."""

    def setup_method(self):
        """Setup for each test method."""
        # Clean up any existing global state
        destroy_async_allreduce()

    def teardown_method(self):
        """Cleanup after each test method."""
        destroy_async_allreduce()

    def test_basic_async_allreduce_workflow(self):
        """Test basic async gradient allreduce workflow without distributed setup."""
        config = AsyncAllreduceConfig(
            enabled=True,
            strategy=AsyncAllreduceStrategy.IMMEDIATE,
            warmup_steps=0,
        )

        # Mock distributed components
        with patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_group"
        ) as mock_get_dp_group, patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_size"
        ) as mock_get_dp_size, patch(
            "torch.distributed.all_reduce"
        ) as mock_all_reduce:
            mock_process_group = MagicMock()
            mock_get_dp_group.return_value = mock_process_group
            mock_get_dp_size.return_value = 4

            mock_work = MagicMock()
            mock_work.is_completed.return_value = False
            mock_all_reduce.return_value = mock_work

            # Initialize async allreduce
            manager = initialize_async_allreduce(config)
            assert manager is not None

            # Create model and register parameters
            model = SimpleModel(10, 20, 5)

            for name, param in model.named_parameters():
                register_parameter_for_async_allreduce(param, name)

            # Simulate forward and backward pass
            x = torch.randn(2, 10)
            y_true = torch.randint(0, 5, (2,))

            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y_true)
            loss.backward()

            # Complete async allreduce step
            async_allreduce_step()

            # Verify all_reduce was called for each parameter
            assert mock_all_reduce.call_count == len(list(model.parameters()))

    def test_bucketed_strategy_integration(self):
        """Test bucketed async allreduce strategy."""
        config = AsyncAllreduceConfig(
            enabled=True,
            strategy=AsyncAllreduceStrategy.BUCKETED,
            bucket_size=1024,  # Small bucket for testing
            max_buckets=2,
            warmup_steps=0,
        )

        with patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_group"
        ) as mock_get_dp_group, patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_size"
        ) as mock_get_dp_size:
            mock_process_group = MagicMock()
            mock_get_dp_group.return_value = mock_process_group
            mock_get_dp_size.return_value = 4

            manager = initialize_async_allreduce(config)

            # Create model with larger parameters to test bucketing
            model = SimpleModel(100, 200, 50)

            for name, param in model.named_parameters():
                register_parameter_for_async_allreduce(param, name)

            # Check that buckets were created
            assert len(manager.buckets) == config.max_buckets

            # Simulate training step
            x = torch.randn(4, 100)
            y_true = torch.randint(0, 50, (4,))

            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y_true)
            loss.backward()

            # Complete async allreduce step
            async_allreduce_step()

    def test_parallel_linear_layers_with_async(self):
        """Test parallel linear layers with async gradient allreduce."""
        config = AsyncAllreduceConfig(
            enabled=True,
            strategy=AsyncAllreduceStrategy.PRIORITY_BASED,
            priority_layers=["column_layer"],
            warmup_steps=0,
        )

        with patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_group"
        ) as mock_get_dp_group, patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_size"
        ) as mock_get_dp_size, patch(
            "torch.distributed.all_reduce"
        ) as mock_all_reduce:
            mock_process_group = MagicMock()
            mock_get_dp_group.return_value = mock_process_group
            mock_get_dp_size.return_value = 4

            mock_work = MagicMock()
            mock_work.is_completed.return_value = False
            mock_all_reduce.return_value = mock_work

            initialize_async_allreduce(config)

            # Create parallel model with async enabled
            model = ParallelModel(20, 40, 10, tp_size=2, enable_async=True)

            # Simulate training
            x = torch.randn(3, 20)
            y_true = torch.randint(0, 10, (3,))

            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y_true)
            loss.backward()

            # Complete async allreduce step
            async_allreduce_step()

            # Verify async allreduce was triggered
            assert mock_all_reduce.call_count > 0

    def test_warmup_behavior(self):
        """Test warmup behavior of async gradient allreduce."""
        config = AsyncAllreduceConfig(
            enabled=True,
            strategy=AsyncAllreduceStrategy.IMMEDIATE,
            warmup_steps=3,
        )

        with patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_group"
        ) as mock_get_dp_group, patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_size"
        ) as mock_get_dp_size, patch(
            "torch.distributed.all_reduce"
        ) as mock_all_reduce:
            mock_process_group = MagicMock()
            mock_get_dp_group.return_value = mock_process_group
            mock_get_dp_size.return_value = 4

            mock_work = MagicMock()
            mock_work.is_completed.return_value = False
            mock_all_reduce.return_value = mock_work

            initialize_async_allreduce(config)

            model = SimpleModel(10, 20, 5)
            for name, param in model.named_parameters():
                register_parameter_for_async_allreduce(param, name)

            # Run warmup steps
            for step in range(5):
                x = torch.randn(2, 10)
                y_true = torch.randint(0, 5, (2,))

                output = model(x)
                loss = nn.CrossEntropyLoss()(output, y_true)
                loss.backward()

                async_allreduce_step()

                if step < config.warmup_steps:
                    # During warmup, no async allreduce should happen
                    assert mock_all_reduce.call_count == 0
                else:
                    # After warmup, async allreduce should happen
                    assert mock_all_reduce.call_count > 0

                # Reset call count for next iteration
                mock_all_reduce.reset_mock()

    def test_skip_layers_functionality(self):
        """Test skip layers functionality in async gradient allreduce."""
        config = AsyncAllreduceConfig(
            enabled=True,
            strategy=AsyncAllreduceStrategy.IMMEDIATE,
            skip_layers=["fc2.weight", "fc2.bias"],
            warmup_steps=0,
        )

        with patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_group"
        ) as mock_get_dp_group, patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_size"
        ) as mock_get_dp_size, patch(
            "torch.distributed.all_reduce"
        ) as mock_all_reduce:
            mock_process_group = MagicMock()
            mock_get_dp_group.return_value = mock_process_group
            mock_get_dp_size.return_value = 4

            mock_work = MagicMock()
            mock_work.is_completed.return_value = False
            mock_all_reduce.return_value = mock_work

            initialize_async_allreduce(config)

            model = SimpleModel(10, 20, 5)

            # Register all parameters (some will be skipped by the manager)
            for name, param in model.named_parameters():
                register_parameter_for_async_allreduce(param, name)

            # Simulate training
            x = torch.randn(2, 10)
            y_true = torch.randint(0, 5, (2,))

            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y_true)
            loss.backward()

            async_allreduce_step()

            # Should only call all_reduce for non-skipped layers (fc1.weight, fc1.bias)
            assert mock_all_reduce.call_count == 2

    def test_statistics_collection(self):
        """Test statistics collection in async gradient allreduce."""
        config = AsyncAllreduceConfig(
            enabled=True,
            strategy=AsyncAllreduceStrategy.IMMEDIATE,
            log_communication_stats=True,
            warmup_steps=0,
        )

        with patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_group"
        ) as mock_get_dp_group, patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_size"
        ) as mock_get_dp_size, patch(
            "torch.distributed.all_reduce"
        ) as mock_all_reduce:
            mock_process_group = MagicMock()
            mock_get_dp_group.return_value = mock_process_group
            mock_get_dp_size.return_value = 4

            # Simulate some delay in all_reduce
            def delayed_all_reduce(*args, **kwargs):
                time.sleep(0.001)  # 1ms delay
                mock_work = MagicMock()
                mock_work.is_completed.return_value = False
                return mock_work

            mock_all_reduce.side_effect = delayed_all_reduce

            manager = initialize_async_allreduce(config)

            model = SimpleModel(10, 20, 5)
            for name, param in model.named_parameters():
                register_parameter_for_async_allreduce(param, name)

            # Run a few training steps
            for _ in range(3):
                x = torch.randn(2, 10)
                y_true = torch.randint(0, 5, (2,))

                output = model(x)
                loss = nn.CrossEntropyLoss()(output, y_true)
                loss.backward()

                async_allreduce_step()

            # Check statistics
            stats = manager.get_statistics()

            assert stats["step_count"] == 3
            assert stats["world_size"] == 4
            assert "avg_comm_time" in stats
            assert "min_comm_time" in stats
            assert "max_comm_time" in stats
            assert stats["avg_comm_time"] > 0

    def test_synchronization_behavior(self):
        """Test synchronization behavior of async operations."""
        config = AsyncAllreduceConfig(
            enabled=True,
            strategy=AsyncAllreduceStrategy.BUCKETED,
            bucket_size=512,
            max_buckets=2,
            warmup_steps=0,
        )

        with patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_group"
        ) as mock_get_dp_group, patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_size"
        ) as mock_get_dp_size:
            mock_process_group = MagicMock()
            mock_get_dp_group.return_value = mock_process_group
            mock_get_dp_size.return_value = 4

            manager = initialize_async_allreduce(config)

            # Add mock handles to pending operations
            mock_handle1 = MagicMock()
            mock_handle1.is_completed.return_value = False
            mock_handle2 = MagicMock()
            mock_handle2.is_completed.return_value = True

            manager.pending_handles.add(mock_handle1)
            manager.pending_handles.add(mock_handle2)

            # Test explicit synchronization
            sync_async_allreduce()

            # Should wait for incomplete handle
            mock_handle1.wait.assert_called_once()
            mock_handle2.wait.assert_not_called()

            # Pending handles should be cleared
            assert len(manager.pending_handles) == 0

    def test_memory_cleanup(self):
        """Test memory cleanup in async gradient allreduce."""
        config = AsyncAllreduceConfig(
            enabled=True,
            strategy=AsyncAllreduceStrategy.BUCKETED,
            bucket_size=1024,
            max_buckets=2,
            warmup_steps=0,
        )

        with patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_group"
        ) as mock_get_dp_group, patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_size"
        ) as mock_get_dp_size:
            mock_process_group = MagicMock()
            mock_get_dp_group.return_value = mock_process_group
            mock_get_dp_size.return_value = 4

            manager = initialize_async_allreduce(config)

            # Simulate adding gradients to buckets
            grad1 = torch.randn(50)
            grad2 = torch.randn(100)

            bucket = manager.buckets[0]
            bucket.add_gradient(grad1)
            bucket.add_gradient(grad2)
            bucket.prepare_buffer()

            # Verify buffer is created
            assert bucket.buffer is not None

            # Reset bucket (should clean up memory)
            bucket.reset()

            # Verify cleanup
            assert bucket.buffer is None
            assert len(bucket.gradients) == 0
            assert len(bucket.gradient_views) == 0
            assert bucket.current_size == 0

    def test_error_handling_invalid_config(self):
        """Test error handling with invalid configuration."""
        # Test with world_size <= 1 and async enabled
        config = AsyncAllreduceConfig(enabled=True)

        with patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_group"
        ) as mock_get_dp_group, patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_size"
        ) as mock_get_dp_size:
            mock_process_group = MagicMock()
            mock_get_dp_group.return_value = mock_process_group
            mock_get_dp_size.return_value = 1  # Single process

            with pytest.raises(
                ValueError,
                match="Async allreduce cannot be enabled with world_size <= 1",
            ):
                AsyncGradientAllreduce(config, mock_process_group)

    def test_disabled_async_allreduce(self):
        """Test behavior when async allreduce is disabled."""
        config = AsyncAllreduceConfig(enabled=False)

        with patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_group"
        ) as mock_get_dp_group, patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_size"
        ) as mock_get_dp_size, patch(
            "torch.distributed.all_reduce"
        ) as mock_all_reduce:
            mock_process_group = MagicMock()
            mock_get_dp_group.return_value = mock_process_group
            mock_get_dp_size.return_value = 4

            initialize_async_allreduce(config)

            model = SimpleModel(10, 20, 5)
            for name, param in model.named_parameters():
                register_parameter_for_async_allreduce(param, name)

            # Simulate training
            x = torch.randn(2, 10)
            y_true = torch.randint(0, 5, (2,))

            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y_true)
            loss.backward()

            async_allreduce_step()

            # No async allreduce should happen when disabled
            assert mock_all_reduce.call_count == 0

    def test_layerwise_strategy_fallback(self):
        """Test layerwise strategy fallback behavior."""
        config = AsyncAllreduceConfig(
            enabled=True,
            strategy=AsyncAllreduceStrategy.LAYERWISE,
            warmup_steps=0,
        )

        with patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_group"
        ) as mock_get_dp_group, patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_size"
        ) as mock_get_dp_size, patch(
            "torch.distributed.all_reduce"
        ) as mock_all_reduce:
            mock_process_group = MagicMock()
            mock_get_dp_group.return_value = mock_process_group
            mock_get_dp_size.return_value = 4

            mock_work = MagicMock()
            mock_work.is_completed.return_value = False
            mock_all_reduce.return_value = mock_work

            initialize_async_allreduce(config)

            model = SimpleModel(10, 20, 5)
            for name, param in model.named_parameters():
                register_parameter_for_async_allreduce(param, name)

            # Simulate training
            x = torch.randn(2, 10)
            y_true = torch.randint(0, 5, (2,))

            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y_true)
            loss.backward()

            async_allreduce_step()

            # Layerwise should fallback to immediate strategy currently
            assert mock_all_reduce.call_count > 0


class TestAsyncGradientAllreducePerformance:
    """Performance-focused integration tests."""

    def setup_method(self):
        """Setup for each test method."""
        destroy_async_allreduce()

    def teardown_method(self):
        """Cleanup after each test method."""
        destroy_async_allreduce()

    def test_performance_comparison_strategies(self):
        """Compare performance of different async strategies."""
        strategies = [
            AsyncAllreduceStrategy.IMMEDIATE,
            AsyncAllreduceStrategy.BUCKETED,
            AsyncAllreduceStrategy.PRIORITY_BASED,
        ]

        results = {}

        for strategy in strategies:
            config = AsyncAllreduceConfig(
                enabled=True,
                strategy=strategy,
                bucket_size=2048,
                max_buckets=3,
                warmup_steps=0,
                log_communication_stats=True,
            )

            with patch(
                "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_group"
            ) as mock_get_dp_group, patch(
                "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_size"
            ) as mock_get_dp_size, patch(
                "torch.distributed.all_reduce"
            ) as mock_all_reduce:
                mock_process_group = MagicMock()
                mock_get_dp_group.return_value = mock_process_group
                mock_get_dp_size.return_value = 4

                # Simulate some delay
                def delayed_all_reduce(*args, **kwargs):
                    time.sleep(0.001)
                    mock_work = MagicMock()
                    mock_work.is_completed.return_value = False
                    return mock_work

                mock_all_reduce.side_effect = delayed_all_reduce

                manager = initialize_async_allreduce(config)

                model = SimpleModel(50, 100, 20)
                for name, param in model.named_parameters():
                    register_parameter_for_async_allreduce(param, name)

                # Time multiple training steps
                start_time = time.time()

                for _ in range(10):
                    x = torch.randn(4, 50)
                    y_true = torch.randint(0, 20, (4,))

                    output = model(x)
                    loss = nn.CrossEntropyLoss()(output, y_true)
                    loss.backward()

                    async_allreduce_step()

                elapsed_time = time.time() - start_time
                stats = manager.get_statistics()

                results[strategy.value] = {
                    "elapsed_time": elapsed_time,
                    "stats": stats,
                    "all_reduce_calls": mock_all_reduce.call_count,
                }

                destroy_async_allreduce()

        # Verify all strategies completed
        assert len(results) == len(strategies)

        # Log results for comparison
        for strategy, result in results.items():
            print(
                f"Strategy {strategy}: "
                f"time={result['elapsed_time']:.4f}s, "
                f"calls={result['all_reduce_calls']}"
            )

    def test_memory_usage_with_large_model(self):
        """Test memory usage with larger model."""
        config = AsyncAllreduceConfig(
            enabled=True,
            strategy=AsyncAllreduceStrategy.BUCKETED,
            bucket_size=10 * 1024,  # 10KB buckets
            max_buckets=5,
            warmup_steps=0,
        )

        with patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_group"
        ) as mock_get_dp_group, patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_size"
        ) as mock_get_dp_size:
            mock_process_group = MagicMock()
            mock_get_dp_group.return_value = mock_process_group
            mock_get_dp_size.return_value = 4

            initialize_async_allreduce(config)

            # Create larger model
            model = SimpleModel(500, 1000, 100)
            for name, param in model.named_parameters():
                register_parameter_for_async_allreduce(param, name)

            # Check memory usage is reasonable
            total_params = sum(p.numel() for p in model.parameters())
            total_param_bytes = total_params * 4  # float32

            # Buckets should not use excessive memory
            total_bucket_memory = config.max_buckets * config.bucket_size

            # Memory overhead should be reasonable (less than 10% of model size)
            assert total_bucket_memory < total_param_bytes * 0.1

            # Simulate training to trigger bucket usage
            x = torch.randn(8, 500)
            y_true = torch.randint(0, 100, (8,))

            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y_true)
            loss.backward()

            async_allreduce_step()


if __name__ == "__main__":
    pytest.main([__file__])
