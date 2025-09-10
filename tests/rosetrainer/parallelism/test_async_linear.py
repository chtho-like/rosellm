"""
Unit tests for async gradient allreduce functionality.

Tests cover:
- AsyncAllreduceConfig creation and validation
- GradientBucket operations and buffer management
- AsyncGradientAllreduce manager functionality
- Parameter registration and gradient hooks
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist

from rosellm.rosetrainer.parallelism.async_config import (
    AsyncAllreduceConfig,
    AsyncAllreduceStrategy,
    GradientBucketingStrategy,
)
from rosellm.rosetrainer.parallelism.async_linear import (
    AsyncGradientAllreduce,
    GradientBucket,
    async_allreduce_step,
    destroy_async_allreduce,
    get_async_allreduce_manager,
    initialize_async_allreduce,
    register_parameter_for_async_allreduce,
    sync_async_allreduce,
)


class TestAsyncAllreduceConfig:
    """Test cases for AsyncAllreduceConfig."""

    def test_default_config_creation(self):
        """Test creating config with default parameters."""
        config = AsyncAllreduceConfig()

        assert config.enabled is True
        assert config.strategy == AsyncAllreduceStrategy.BUCKETED
        assert config.bucket_size == 25 * 1024 * 1024
        assert config.max_buckets == 4
        assert config.bucketing_strategy == GradientBucketingStrategy.SIZE_BASED
        assert config.overlap_threshold == 0.1
        assert config.warmup_steps == 10
        assert config.buffer_growth_factor == 1.25
        assert config.max_buffer_size == 100 * 1024 * 1024
        assert config.enable_async_param_sync is True
        assert config.async_op_timeout == 30.0
        assert config.gradient_predivision is True

    def test_config_validation_valid_params(self):
        """Test config validation with valid parameters."""
        config = AsyncAllreduceConfig(
            bucket_size=10 * 1024 * 1024,
            max_buckets=2,
            overlap_threshold=0.05,
            warmup_steps=5,
            buffer_growth_factor=1.5,
            max_buffer_size=50 * 1024 * 1024,
            async_op_timeout=60.0,
        )
        # Should not raise any exceptions
        assert config.bucket_size == 10 * 1024 * 1024

    def test_config_validation_invalid_bucket_size(self):
        """Test config validation with invalid bucket size."""
        with pytest.raises(ValueError, match="bucket_size must be positive"):
            AsyncAllreduceConfig(bucket_size=0)

        with pytest.raises(ValueError, match="bucket_size must be positive"):
            AsyncAllreduceConfig(bucket_size=-1)

    def test_config_validation_invalid_max_buckets(self):
        """Test config validation with invalid max buckets."""
        with pytest.raises(ValueError, match="max_buckets must be positive"):
            AsyncAllreduceConfig(max_buckets=0)

    def test_config_validation_invalid_overlap_threshold(self):
        """Test config validation with invalid overlap threshold."""
        with pytest.raises(ValueError, match="overlap_threshold must be non-negative"):
            AsyncAllreduceConfig(overlap_threshold=-0.1)

    def test_config_validation_invalid_warmup_steps(self):
        """Test config validation with invalid warmup steps."""
        with pytest.raises(ValueError, match="warmup_steps must be non-negative"):
            AsyncAllreduceConfig(warmup_steps=-1)

    def test_config_validation_invalid_buffer_growth_factor(self):
        """Test config validation with invalid buffer growth factor."""
        with pytest.raises(
            ValueError, match="buffer_growth_factor must be greater than 1.0"
        ):
            AsyncAllreduceConfig(buffer_growth_factor=1.0)

        with pytest.raises(
            ValueError, match="buffer_growth_factor must be greater than 1.0"
        ):
            AsyncAllreduceConfig(buffer_growth_factor=0.5)

    def test_config_validation_invalid_max_buffer_size(self):
        """Test config validation with invalid max buffer size."""
        with pytest.raises(ValueError, match="max_buffer_size must be positive"):
            AsyncAllreduceConfig(max_buffer_size=0)

    def test_config_validation_invalid_timeout(self):
        """Test config validation with invalid timeout."""
        with pytest.raises(
            ValueError, match="async_op_timeout must be positive or None"
        ):
            AsyncAllreduceConfig(async_op_timeout=0)

        with pytest.raises(
            ValueError, match="async_op_timeout must be positive or None"
        ):
            AsyncAllreduceConfig(async_op_timeout=-10)

        # None should be valid
        config = AsyncAllreduceConfig(async_op_timeout=None)
        assert config.async_op_timeout is None

    def test_config_validation_bucket_size_exceeds_max_buffer(self):
        """Test config validation when bucket size exceeds max buffer size."""
        with pytest.raises(
            ValueError, match="bucket_size cannot exceed max_buffer_size"
        ):
            AsyncAllreduceConfig(
                bucket_size=50 * 1024 * 1024, max_buffer_size=25 * 1024 * 1024
            )

    def test_create_optimized_config(self):
        """Test creating optimized configuration."""
        config = AsyncAllreduceConfig.create_optimized_config(
            world_size=8, model_size_mb=1000, gpu_memory_gb=32
        )

        assert config.enabled is True
        assert config.strategy == AsyncAllreduceStrategy.BUCKETED
        assert config.bucket_size >= 1024 * 1024  # At least 1MB
        assert config.bucket_size <= 25 * 1024 * 1024  # At most 25MB
        assert config.max_buckets >= 2
        assert config.max_buckets <= 8
        assert config.warmup_steps >= 8  # At least world_size
        assert config.overlap_threshold == 0.05  # Lower threshold for large world size

    def test_create_optimized_config_small_world_size(self):
        """Test creating optimized config for small world size."""
        config = AsyncAllreduceConfig.create_optimized_config(
            world_size=2, model_size_mb=100, gpu_memory_gb=8
        )

        assert config.overlap_threshold == 0.1  # Higher threshold for small world size
        assert config.warmup_steps == 10  # Default minimum

    def test_validate_for_world_size_valid(self):
        """Test world size validation with valid configuration."""
        config = AsyncAllreduceConfig()

        # Should not raise for valid world sizes
        config.validate_for_world_size(4)
        config.validate_for_world_size(8)

    def test_validate_for_world_size_invalid(self):
        """Test world size validation with invalid configuration."""
        config = AsyncAllreduceConfig(enabled=True)

        with pytest.raises(
            ValueError, match="Async allreduce cannot be enabled with world_size <= 1"
        ):
            config.validate_for_world_size(1)

    def test_validate_for_world_size_warning(self):
        """Test world size validation with warning condition."""
        config = AsyncAllreduceConfig(max_buckets=8)

        with pytest.warns(
            UserWarning, match="max_buckets .* is larger than world_size"
        ):
            config.validate_for_world_size(2)


class TestGradientBucket:
    """Test cases for GradientBucket."""

    @pytest.fixture
    def mock_process_group(self):
        """Create a mock process group for testing."""
        return MagicMock(spec=dist.ProcessGroup)

    @pytest.fixture
    def gradient_bucket(self, mock_process_group):
        """Create a gradient bucket for testing."""
        device = torch.device("cpu")
        dtype = torch.float32
        return GradientBucket(
            bucket_id=0,
            max_size=1024,  # Small size for testing
            device=device,
            dtype=dtype,
            process_group=mock_process_group,
        )

    def test_gradient_bucket_creation(self, gradient_bucket):
        """Test gradient bucket creation."""
        assert gradient_bucket.bucket_id == 0
        assert gradient_bucket.max_size == 1024
        assert gradient_bucket.current_size == 0
        assert len(gradient_bucket.gradients) == 0
        assert len(gradient_bucket.gradient_views) == 0
        assert gradient_bucket.is_ready is False
        assert gradient_bucket.allreduce_handle is None
        assert gradient_bucket.buffer is None

    def test_add_gradient_success(self, gradient_bucket):
        """Test successfully adding gradients to bucket."""
        grad1 = torch.randn(10)  # 40 bytes (10 * 4)
        grad2 = torch.randn(20)  # 80 bytes (20 * 4)

        # Both should fit in 1024 byte bucket
        assert gradient_bucket.add_gradient(grad1) is True
        assert len(gradient_bucket.gradients) == 1
        assert gradient_bucket.current_size == 40

        assert gradient_bucket.add_gradient(grad2) is True
        assert len(gradient_bucket.gradients) == 2
        assert gradient_bucket.current_size == 120

    def test_add_gradient_bucket_full(self, gradient_bucket):
        """Test adding gradient when bucket is full."""
        # Add a large gradient that fills most of the bucket
        large_grad = torch.randn(200)  # 800 bytes
        assert gradient_bucket.add_gradient(large_grad) is True

        # Try to add another gradient that would exceed bucket size
        another_grad = torch.randn(100)  # 400 bytes, would make total 1200 > 1024
        assert gradient_bucket.add_gradient(another_grad) is False

        # Bucket should still contain only the first gradient
        assert len(gradient_bucket.gradients) == 1
        assert gradient_bucket.current_size == 800

    def test_add_gradient_empty_bucket_accepts_large_gradient(self, gradient_bucket):
        """Test that empty bucket accepts gradient larger than max_size."""
        # Empty bucket should accept even oversized gradients
        large_grad = torch.randn(500)  # 2000 bytes > 1024 max_size
        assert gradient_bucket.add_gradient(large_grad) is True
        assert len(gradient_bucket.gradients) == 1

    def test_prepare_buffer(self, gradient_bucket):
        """Test buffer preparation."""
        grad1 = torch.randn(10)
        grad2 = torch.randn(20)

        gradient_bucket.add_gradient(grad1)
        gradient_bucket.add_gradient(grad2)

        # Prepare buffer
        gradient_bucket.prepare_buffer()

        assert gradient_bucket.buffer is not None
        assert gradient_bucket.buffer.numel() == 30  # 10 + 20 elements
        assert gradient_bucket.buffer.dtype == torch.float32

    def test_prepare_buffer_empty_bucket(self, gradient_bucket):
        """Test buffer preparation with empty bucket."""
        gradient_bucket.prepare_buffer()
        # Should handle empty bucket gracefully
        assert gradient_bucket.buffer is None

    @patch("torch.distributed.all_reduce")
    def test_start_allreduce(self, mock_all_reduce, gradient_bucket):
        """Test starting allreduce operation."""
        grad = torch.randn(10)
        gradient_bucket.add_gradient(grad)

        # Mock the all_reduce function
        mock_work = MagicMock()
        mock_all_reduce.return_value = mock_work

        # Start allreduce
        handle = gradient_bucket.start_allreduce()

        assert handle == mock_work
        assert gradient_bucket.allreduce_handle == mock_work
        mock_all_reduce.assert_called_once()

    def test_start_allreduce_already_started(self, gradient_bucket):
        """Test starting allreduce when already started."""
        existing_handle = MagicMock()
        gradient_bucket.allreduce_handle = existing_handle

        # Should return existing handle
        handle = gradient_bucket.start_allreduce()
        assert handle == existing_handle

    def test_wait_and_copy_back(self, gradient_bucket):
        """Test waiting for allreduce and copying results back."""
        grad = torch.randn(10)
        gradient_bucket.add_gradient(grad)
        gradient_bucket.prepare_buffer()

        # Mock allreduce handle
        mock_handle = MagicMock()
        gradient_bucket.allreduce_handle = mock_handle

        # Modify buffer to simulate allreduce result
        gradient_bucket.buffer.fill_(1.0)

        # Wait and copy back
        gradient_bucket.wait_and_copy_back()

        mock_handle.wait.assert_called_once()
        # Gradient should be updated with buffer values
        assert torch.allclose(grad.view(-1), torch.ones(10))
        assert gradient_bucket.allreduce_handle is None

    def test_wait_and_copy_back_no_handle(self, gradient_bucket):
        """Test wait_and_copy_back when no allreduce handle exists."""
        # Should handle gracefully
        gradient_bucket.wait_and_copy_back()
        # No exception should be raised

    def test_reset(self, gradient_bucket):
        """Test bucket reset."""
        grad = torch.randn(10)
        gradient_bucket.add_gradient(grad)
        gradient_bucket.prepare_buffer()
        gradient_bucket.allreduce_handle = MagicMock()
        gradient_bucket.is_ready = True

        gradient_bucket.reset()

        assert len(gradient_bucket.gradients) == 0
        assert len(gradient_bucket.gradient_views) == 0
        assert gradient_bucket.current_size == 0
        assert gradient_bucket.buffer_offset == 0
        assert gradient_bucket.is_ready is False
        assert gradient_bucket.buffer is None
        assert gradient_bucket.allreduce_handle is None

    def test_thread_safety(self, gradient_bucket):
        """Test thread safety of bucket operations."""
        results = []
        exceptions = []

        def add_gradients():
            try:
                for i in range(10):
                    grad = torch.randn(5)
                    result = gradient_bucket.add_gradient(grad)
                    results.append(result)
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                exceptions.append(e)

        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=add_gradients)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should not have any exceptions
        assert len(exceptions) == 0
        # Some gradients should be accepted
        assert any(results)


class TestAsyncGradientAllreduce:
    """Test cases for AsyncGradientAllreduce manager."""

    @pytest.fixture
    def mock_process_group(self):
        """Create a mock process group for testing."""
        return MagicMock(spec=dist.ProcessGroup)

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AsyncAllreduceConfig(
            enabled=True,
            strategy=AsyncAllreduceStrategy.BUCKETED,
            bucket_size=1024,
            max_buckets=2,
            warmup_steps=2,
        )

    @patch("rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_group")
    @patch("rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_size")
    def test_manager_initialization(
        self, mock_get_dp_size, mock_get_dp_group, config, mock_process_group
    ):
        """Test async gradient allreduce manager initialization."""
        mock_get_dp_group.return_value = mock_process_group
        mock_get_dp_size.return_value = 4

        manager = AsyncGradientAllreduce(config, mock_process_group)

        assert manager.config == config
        assert manager.process_group == mock_process_group
        assert manager.world_size == 4
        assert manager.step_count == 0
        assert len(manager.buckets) == config.max_buckets
        assert manager.current_bucket_idx == 0

    @patch("rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_group")
    @patch("rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_size")
    def test_manager_with_disabled_config(
        self, mock_get_dp_size, mock_get_dp_group, mock_process_group
    ):
        """Test manager with disabled async allreduce."""
        mock_get_dp_group.return_value = mock_process_group
        mock_get_dp_size.return_value = 4

        config = AsyncAllreduceConfig(enabled=False)
        manager = AsyncGradientAllreduce(config, mock_process_group)

        assert len(manager.buckets) == 0

    @patch("rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_group")
    @patch("rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_size")
    def test_manager_single_process(
        self, mock_get_dp_size, mock_get_dp_group, config, mock_process_group
    ):
        """Test manager with single process (no communication needed)."""
        mock_get_dp_group.return_value = mock_process_group
        mock_get_dp_size.return_value = 1

        manager = AsyncGradientAllreduce(config, mock_process_group)

        # Should have no buckets for single process
        assert len(manager.buckets) == 0

    def test_register_gradient_hook_disabled(self, config, mock_process_group):
        """Test registering gradient hook when disabled."""
        config.enabled = False

        with patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_group"
        ) as mock_get_dp_group, patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_size"
        ) as mock_get_dp_size:
            mock_get_dp_group.return_value = mock_process_group
            mock_get_dp_size.return_value = 4

            manager = AsyncGradientAllreduce(config, mock_process_group)

            param = torch.nn.Parameter(torch.randn(10))
            original_hooks = (
                len(param._backward_hooks)
                if hasattr(param, "_backward_hooks")
                and param._backward_hooks is not None
                else 0
            )

            manager.register_gradient_hook(param, "test_layer")

            # No hook should be registered when disabled
            current_hooks = (
                len(param._backward_hooks)
                if hasattr(param, "_backward_hooks")
                and param._backward_hooks is not None
                else 0
            )
            assert current_hooks == original_hooks

    def test_register_gradient_hook_skip_layer(self, config, mock_process_group):
        """Test registering gradient hook for skipped layer."""
        config.skip_layers = ["test_layer"]

        with patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_group"
        ) as mock_get_dp_group, patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_size"
        ) as mock_get_dp_size:
            mock_get_dp_group.return_value = mock_process_group
            mock_get_dp_size.return_value = 4

            manager = AsyncGradientAllreduce(config, mock_process_group)

            param = torch.nn.Parameter(torch.randn(10))
            original_hooks = (
                len(param._backward_hooks)
                if hasattr(param, "_backward_hooks")
                and param._backward_hooks is not None
                else 0
            )

            manager.register_gradient_hook(param, "test_layer")

            # No hook should be registered for skipped layer
            current_hooks = (
                len(param._backward_hooks)
                if hasattr(param, "_backward_hooks")
                and param._backward_hooks is not None
                else 0
            )
            assert current_hooks == original_hooks

    @patch("torch.distributed.all_reduce")
    def test_handle_immediate_allreduce(
        self, mock_all_reduce, config, mock_process_group
    ):
        """Test immediate allreduce strategy."""
        config.strategy = AsyncAllreduceStrategy.IMMEDIATE
        config.gradient_predivision = True

        with patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_group"
        ) as mock_get_dp_group, patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_size"
        ) as mock_get_dp_size:
            mock_get_dp_group.return_value = mock_process_group
            mock_get_dp_size.return_value = 4

            manager = AsyncGradientAllreduce(config, mock_process_group)
            manager.step_count = 10  # Beyond warmup

            grad = torch.ones(10)
            original_grad = grad.clone()

            mock_work = MagicMock()
            mock_all_reduce.return_value = mock_work

            manager._handle_immediate_allreduce(grad)

            # Gradient should be divided by world size
            expected_grad = original_grad / 4
            assert torch.allclose(grad, expected_grad)

            # All-reduce should be called
            mock_all_reduce.assert_called_once_with(
                grad, group=mock_process_group, async_op=True
            )

    def test_synchronize(self, config, mock_process_group):
        """Test synchronizing all async operations."""
        with patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_group"
        ) as mock_get_dp_group, patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_size"
        ) as mock_get_dp_size:
            mock_get_dp_group.return_value = mock_process_group
            mock_get_dp_size.return_value = 4

            manager = AsyncGradientAllreduce(config, mock_process_group)

            # Add mock handles
            mock_handle1 = MagicMock()
            mock_handle1.is_completed.return_value = False
            mock_handle2 = MagicMock()
            mock_handle2.is_completed.return_value = True

            manager.pending_handles.add(mock_handle1)
            manager.pending_handles.add(mock_handle2)

            # Mock bucket with pending allreduce
            mock_bucket = MagicMock()
            mock_bucket.allreduce_handle = MagicMock()
            manager.buckets.append(mock_bucket)

            manager.synchronize()

            # Should wait for incomplete handle
            mock_handle1.wait.assert_called_once()
            mock_handle2.wait.assert_not_called()

            # Should synchronize bucket
            mock_bucket.wait_and_copy_back.assert_called_once()
            mock_bucket.reset.assert_called_once()

            # Pending handles should be cleared
            assert len(manager.pending_handles) == 0

    def test_step(self, config, mock_process_group):
        """Test completing a training step."""
        with patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_group"
        ) as mock_get_dp_group, patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_size"
        ) as mock_get_dp_size:
            mock_get_dp_group.return_value = mock_process_group
            mock_get_dp_size.return_value = 4

            manager = AsyncGradientAllreduce(config, mock_process_group)
            initial_step = manager.step_count

            with patch.object(manager, "synchronize") as mock_sync:
                manager.step()

                mock_sync.assert_called_once()
                assert manager.step_count == initial_step + 1

    def test_get_statistics(self, config, mock_process_group):
        """Test getting performance statistics."""
        with patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_group"
        ) as mock_get_dp_group, patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_size"
        ) as mock_get_dp_size:
            mock_get_dp_group.return_value = mock_process_group
            mock_get_dp_size.return_value = 4

            manager = AsyncGradientAllreduce(config, mock_process_group)
            manager.step_count = 5
            manager.communication_times.append(0.1)
            manager.communication_times.append(0.2)
            manager.overlap_ratios.append(0.8)
            manager.overlap_ratios.append(0.9)

            stats = manager.get_statistics()

            assert stats["step_count"] == 5
            assert stats["world_size"] == 4
            assert stats["num_buckets"] == len(manager.buckets)
            assert stats["config"] == config
            assert abs(stats["avg_comm_time"] - 0.15) < 1e-10
            assert stats["min_comm_time"] == 0.1
            assert stats["max_comm_time"] == 0.2
            assert abs(stats["avg_overlap_ratio"] - 0.85) < 1e-10

    def test_reset_statistics(self, config, mock_process_group):
        """Test resetting performance statistics."""
        with patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_group"
        ) as mock_get_dp_group, patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_size"
        ) as mock_get_dp_size:
            mock_get_dp_group.return_value = mock_process_group
            mock_get_dp_size.return_value = 4

            manager = AsyncGradientAllreduce(config, mock_process_group)
            manager.step_count = 10
            manager.communication_times.append(0.1)
            manager.overlap_ratios.append(0.8)

            manager.reset_statistics()

            assert manager.step_count == 0
            assert len(manager.communication_times) == 0
            assert len(manager.overlap_ratios) == 0


class TestGlobalFunctions:
    """Test cases for global async allreduce functions."""

    def setup_method(self):
        """Clean up global state before each test."""
        destroy_async_allreduce()

    def teardown_method(self):
        """Clean up global state after each test."""
        destroy_async_allreduce()

    def test_initialize_and_get_manager(self):
        """Test initializing and getting global async allreduce manager."""
        config = AsyncAllreduceConfig(
            enabled=False
        )  # Disabled to avoid distributed requirements

        with patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_group"
        ) as mock_get_dp_group, patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_size"
        ) as mock_get_dp_size:
            mock_get_dp_group.return_value = MagicMock()
            mock_get_dp_size.return_value = 1

            manager = initialize_async_allreduce(config)

            assert manager is not None
            assert get_async_allreduce_manager() == manager

    def test_initialize_replace_existing(self):
        """Test replacing existing global manager."""
        config1 = AsyncAllreduceConfig(enabled=False)
        config2 = AsyncAllreduceConfig(enabled=False, bucket_size=2048)

        with patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_group"
        ) as mock_get_dp_group, patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_size"
        ) as mock_get_dp_size:
            mock_get_dp_group.return_value = MagicMock()
            mock_get_dp_size.return_value = 1

            manager1 = initialize_async_allreduce(config1)

            with pytest.warns(UserWarning, match="already initialized"):
                manager2 = initialize_async_allreduce(config2)

            assert manager2 != manager1
            assert get_async_allreduce_manager() == manager2
            assert manager2.config.bucket_size == 2048

    def test_destroy_manager(self):
        """Test destroying global manager."""
        config = AsyncAllreduceConfig(enabled=False)

        with patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_group"
        ) as mock_get_dp_group, patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_size"
        ) as mock_get_dp_size:
            mock_get_dp_group.return_value = MagicMock()
            mock_get_dp_size.return_value = 1

            initialize_async_allreduce(config)
            assert get_async_allreduce_manager() is not None

            destroy_async_allreduce()
            assert get_async_allreduce_manager() is None

    def test_register_parameter_no_manager(self):
        """Test registering parameter when no global manager exists."""
        param = torch.nn.Parameter(torch.randn(10))

        # Should not raise an error
        register_parameter_for_async_allreduce(param, "test_layer")

    def test_register_parameter_with_manager(self):
        """Test registering parameter with global manager."""
        config = AsyncAllreduceConfig(enabled=False)
        param = torch.nn.Parameter(torch.randn(10))

        with patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_group"
        ) as mock_get_dp_group, patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_size"
        ) as mock_get_dp_size:
            mock_get_dp_group.return_value = MagicMock()
            mock_get_dp_size.return_value = 1

            manager = initialize_async_allreduce(config)

            with patch.object(manager, "register_gradient_hook") as mock_register:
                register_parameter_for_async_allreduce(param, "test_layer")
                mock_register.assert_called_once_with(param, "test_layer")

    def test_async_allreduce_step_no_manager(self):
        """Test async allreduce step when no manager exists."""
        # Should not raise an error
        async_allreduce_step()

    def test_async_allreduce_step_with_manager(self):
        """Test async allreduce step with global manager."""
        config = AsyncAllreduceConfig(enabled=False)

        with patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_group"
        ) as mock_get_dp_group, patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_size"
        ) as mock_get_dp_size:
            mock_get_dp_group.return_value = MagicMock()
            mock_get_dp_size.return_value = 1

            manager = initialize_async_allreduce(config)

            with patch.object(manager, "step") as mock_step:
                async_allreduce_step()
                mock_step.assert_called_once()

    def test_sync_async_allreduce_no_manager(self):
        """Test sync async allreduce when no manager exists."""
        # Should not raise an error
        sync_async_allreduce()

    def test_sync_async_allreduce_with_manager(self):
        """Test sync async allreduce with global manager."""
        config = AsyncAllreduceConfig(enabled=False)

        with patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_group"
        ) as mock_get_dp_group, patch(
            "rosellm.rosetrainer.parallelism.async_linear.get_data_parallel_size"
        ) as mock_get_dp_size:
            mock_get_dp_group.return_value = MagicMock()
            mock_get_dp_size.return_value = 1

            manager = initialize_async_allreduce(config)

            with patch.object(manager, "synchronize") as mock_sync:
                sync_async_allreduce()
                mock_sync.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
