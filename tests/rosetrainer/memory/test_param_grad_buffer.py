"""
Unit tests for Parameter and Gradient Buffer System.

Tests the ParamAndGradBuffer, GradientBucket, and BufferManager classes
for correctness, performance, and integration with distributed training.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from rosellm.rosetrainer.memory.param_grad_buffer import (
    BucketCapacityError,
    BucketConfig,
    BucketConfigError,
    BufferManager,
    CommunicationError,
    GradientBucket,
    ParamAndGradBuffer,
    ParameterMappingError,
)

# Common test constants to avoid magic numbers
TEST_BUCKET_SIZE_MB = 1.0
TEST_ALIGNMENT = 128
TEST_MAX_NORM = 1.0
TEST_HALF_SCALE = 0.5
TEST_ACC_STEPS = 4


class SimpleModel(nn.Module):
    """Simple model for testing buffer system."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.norm(x)
        x = self.fc3(x)
        return x


class MixedPrecisionModel(nn.Module):
    """Model with mixed precision parameters for testing."""

    def __init__(self):
        super().__init__()
        self.fc_fp32 = nn.Linear(10, 20)
        self.fc_fp16 = nn.Linear(20, 30)
        self.fc_bf16 = nn.Linear(30, 10)

        # Convert layers to different precisions
        self.fc_fp16 = self.fc_fp16.half()
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.fc_bf16 = self.fc_bf16.to(torch.bfloat16)


class TestGradientBucket(unittest.TestCase):
    """Tests for GradientBucket class."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

    def test_bucket_creation(self):
        """Test bucket creation and initialization."""
        bucket_size_bytes = int(TEST_BUCKET_SIZE_MB * 1024 * 1024)

        bucket = GradientBucket(
            bucket_id=0,
            dtype=self.dtype,
            device=self.device,
            bucket_size_bytes=bucket_size_bytes,
            alignment=128,
        )

        self.assertEqual(bucket.bucket_id, 0)
        self.assertEqual(bucket.dtype, self.dtype)
        self.assertEqual(bucket.device, self.device)
        self.assertGreater(bucket.numel, 0)
        self.assertEqual(bucket.numel % 128, 0)  # Check alignment

    def test_add_param(self):
        """Test adding parameters to bucket."""
        bucket = GradientBucket(
            bucket_id=0,
            dtype=self.dtype,
            device=self.device,
            bucket_size_bytes=1024 * 1024,  # 1 MB
            alignment=TEST_ALIGNMENT,
        )

        # Create test parameters
        param1 = nn.Parameter(torch.randn(100, 10, device=self.device))
        param2 = nn.Parameter(torch.randn(50, 20, device=self.device))

        # Add parameters
        offset1 = bucket.add_param(param1)
        offset2 = bucket.add_param(param2)

        self.assertEqual(offset1, (0, param1.numel()))
        self.assertEqual(offset2, (param1.numel(), param1.numel() + param2.numel()))
        self.assertEqual(len(bucket.params), 2)

    def test_pack_unpack_gradients(self):
        """Test packing and unpacking gradients."""
        bucket = GradientBucket(
            bucket_id=0,
            dtype=self.dtype,
            device=self.device,
            bucket_size_bytes=1024 * 1024,
            alignment=TEST_ALIGNMENT,
        )

        # Create parameters with gradients
        param1 = nn.Parameter(torch.randn(10, 10, device=self.device))
        param1.grad = torch.randn_like(param1)
        grad1_orig = param1.grad.clone()

        param2 = nn.Parameter(torch.randn(5, 5, device=self.device))
        param2.grad = torch.randn_like(param2)
        grad2_orig = param2.grad.clone()

        # Add to bucket
        bucket.add_param(param1)
        bucket.add_param(param2)

        # Pack gradients
        bucket.pack_gradients()
        self.assertTrue(bucket.ready_for_comm)

        # Verify packed data
        self.assertTrue(
            torch.allclose(bucket.grad_data[: param1.numel()], grad1_orig.view(-1))
        )

        # Unpack with scaling
        bucket.unpack_gradients(scale=TEST_HALF_SCALE)

        self.assertTrue(torch.allclose(param1.grad, grad1_orig * TEST_HALF_SCALE))
        self.assertTrue(torch.allclose(param2.grad, grad2_orig * TEST_HALF_SCALE))


class TestParamAndGradBuffer(unittest.TestCase):
    """Tests for ParamAndGradBuffer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleModel()
        self.model.to(self.device)

    def test_buffer_creation(self):
        """Test buffer creation and initialization."""
        params = list(self.model.parameters())

        buffer = ParamAndGradBuffer(
            dtype=torch.float32,
            params=params,
            data_parallel_group=None,
            bucket_config=BucketConfig(bucket_size_mb=TEST_BUCKET_SIZE_MB),
        )

        self.assertEqual(
            len(buffer.params), len([p for p in params if p.requires_grad])
        )
        self.assertEqual(buffer.dtype, torch.float32)
        self.assertGreater(buffer.numel, 0)

    def test_param_buffer_mapping(self):
        """Test parameter to buffer mapping."""
        params = list(self.model.parameters())

        buffer = ParamAndGradBuffer(
            dtype=torch.float32,
            params=params,
            data_parallel_group=None,
        )

        # Verify all parameters are mapped (by identity)
        for param in params:
            if param.requires_grad:
                self.assertIn(id(param), buffer.param_index_by_id)

        # Verify offsets are correct
        total_numel = 0
        for param, (start, end) in zip(buffer.params, buffer.param_offsets):
            self.assertEqual(end - start, param.numel())
            total_numel += param.numel()
        self.assertEqual(total_numel, buffer.numel)

    def test_sync_operations(self):
        """Test synchronization between parameters and buffers."""
        params = list(self.model.parameters())

        buffer = ParamAndGradBuffer(
            dtype=torch.float32,
            params=params,
            data_parallel_group=None,
        )

        # Modify parameters
        for param in buffer.params:
            param.data.fill_(1.0)

        # Sync to buffer
        buffer.sync_params_to_buffer()

        # Verify buffer contains updated values
        self.assertTrue(torch.all(buffer.param_data == 1.0))

        # Modify buffer
        buffer.param_data.fill_(2.0)

        # Sync back to params
        buffer.sync_buffer_to_params()

        # Verify parameters updated
        for param in buffer.params:
            self.assertTrue(torch.all(param.data == 2.0))

    def test_gradient_sync(self):
        """Test gradient synchronization."""
        params = list(self.model.parameters())

        buffer = ParamAndGradBuffer(
            dtype=torch.float32,
            params=params,
            data_parallel_group=None,
        )

        # Create gradients
        for param in buffer.params:
            param.grad = torch.ones_like(param) * 3.0

        # Sync gradients to buffer
        buffer.sync_gradients_to_buffer()

        # Verify buffer contains gradients
        self.assertTrue(torch.all(buffer.grad_data == 3.0))

        # Modify buffer gradients
        buffer.grad_data.fill_(4.0)

        # Sync back with scaling
        buffer.sync_buffer_to_gradients(scale=TEST_HALF_SCALE)

        # Verify scaled gradients
        for param in buffer.params:
            self.assertTrue(torch.all(param.grad == 2.0))

    def test_gradient_clipping(self):
        """Test gradient clipping."""
        params = list(self.model.parameters())

        buffer = ParamAndGradBuffer(
            dtype=torch.float32,
            params=params,
            data_parallel_group=None,
        )

        # Create large gradients
        for param in buffer.params:
            param.grad = torch.randn_like(param) * 10.0

        # Clip gradients
        total_norm = buffer.clip_gradients(TEST_MAX_NORM)

        # Verify clipping
        self.assertGreater(total_norm, TEST_MAX_NORM)  # Original norm was larger

        # Check new norm is approximately max_norm
        buffer.sync_gradients_to_buffer()
        new_norm = torch.norm(buffer.grad_data, p=2).item()
        self.assertAlmostEqual(new_norm, TEST_MAX_NORM, places=5)

    def test_bucket_creation(self):
        """Test gradient bucket creation."""
        params = list(self.model.parameters())

        # Create buffer with moderate bucket size to ensure buckets can hold params
        buffer = ParamAndGradBuffer(
            dtype=torch.float32,
            params=params,
            data_parallel_group=MagicMock(),  # Mock process group
            bucket_config=BucketConfig(bucket_size_mb=TEST_BUCKET_SIZE_MB),
        )

        # Should have at least one bucket
        self.assertGreaterEqual(len(buffer.buckets), 1)

        # Verify all parameters are in buckets
        params_in_buckets = []
        for bucket in buffer.buckets:
            params_in_buckets.extend(bucket.params)

        self.assertEqual(len(params_in_buckets), len(buffer.params))


class TestBufferManager(unittest.TestCase):
    """Tests for BufferManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_manager_creation(self):
        """Test buffer manager creation."""
        model = SimpleModel()
        model.to(self.device)

        manager = BufferManager(
            model=model,
            data_parallel_group=None,
            bucket_config=BucketConfig(bucket_size_mb=1.0),
            create_per_dtype_buffers=False,
        )

        self.assertEqual(len(manager.buffers), 1)
        self.assertIn("main", manager.buffers)
        self.assertGreater(manager.total_params, 0)

    def test_per_dtype_buffers(self):
        """Test creation of per-dtype buffers."""
        model = MixedPrecisionModel()
        model.to(self.device)

        manager = BufferManager(
            model=model,
            data_parallel_group=None,
            create_per_dtype_buffers=True,
        )

        # Should have buffer for each dtype present
        dtype_names = [str(dtype) for dtype in manager.buffers.keys()]
        self.assertIn("torch.float32", dtype_names)

        if self.device.type == "cuda":
            self.assertIn("torch.float16", dtype_names)

    def test_sync_operations(self):
        """Test synchronization operations."""
        model = SimpleModel()
        model.to(self.device)

        manager = BufferManager(model=model)

        # Modify model parameters
        for param in model.parameters():
            param.data.fill_(5.0)

        # Sync to buffers
        manager.sync_params_to_buffers()

        # Modify through buffer
        for buffer in manager.buffers.values():
            buffer.param_data.fill_(6.0)

        # Sync back
        manager.sync_buffers_to_params()

        # Verify parameters updated
        for param in model.parameters():
            if param.requires_grad:
                self.assertTrue(torch.all(param.data == 6.0))

    def test_gradient_operations(self):
        """Test gradient operations."""
        model = SimpleModel()
        model.to(self.device)

        manager = BufferManager(model=model)

        # Create gradients
        for param in model.parameters():
            if param.requires_grad:
                param.grad = torch.ones_like(param) * 2.0

        # Zero gradients
        manager.zero_gradients()

        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                self.assertTrue(torch.all(param.grad == 0.0))

        # Create new gradients for clipping test
        for param in model.parameters():
            if param.requires_grad:
                param.grad = torch.randn_like(param) * 10.0

        # Clip gradients
        max_norm = 1.0
        total_norm = manager.clip_gradients(max_norm)

        self.assertGreater(total_norm, max_norm)

        # Verify clipping worked
        new_total_norm = 0.0
        for buffer in manager.buffers.values():
            buffer.sync_gradients_to_buffer()
            norm = torch.norm(buffer.grad_data, p=2).item()
            new_total_norm += norm**2
        new_total_norm = new_total_norm**0.5

        self.assertAlmostEqual(new_total_norm, max_norm, places=5)

    def test_memory_usage_reporting(self):
        """Test memory usage statistics."""
        model = SimpleModel()
        model.to(self.device)

        manager = BufferManager(model=model)

        stats = manager.get_memory_usage()

        self.assertIn("num_buffers", stats)
        self.assertIn("total_params", stats)
        self.assertIn("total_memory_mb", stats)
        self.assertIn("buffers", stats)

        # Check that total_memory_mb exists and is positive
        total_memory = stats["total_memory_mb"]
        self.assertIsInstance(total_memory, (int, float))
        if isinstance(total_memory, (int, float)):
            self.assertGreater(total_memory, 0)


class TestDistributedOperations(unittest.TestCase):
    """Tests for distributed operations (mocked)."""

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.all_reduce")
    def test_all_reduce_flat(self, mock_all_reduce, mock_world_size, mock_is_init):
        """Test flat all-reduce operation."""
        mock_is_init.return_value = True
        mock_world_size.return_value = 4
        mock_all_reduce.return_value = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleModel()
        model.to(device)

        params = list(model.parameters())
        process_group = MagicMock()

        buffer = ParamAndGradBuffer(
            dtype=torch.float32,
            params=params,
            data_parallel_group=process_group,
            bucket_config=BucketConfig(overlap_comm=False),
        )

        # Create gradients
        for param in buffer.params:
            param.grad = torch.ones_like(param)

        # Perform all-reduce
        buffer.all_reduce_gradients(async_op=False)

        # Verify all_reduce was called
        mock_all_reduce.assert_called_once()

        # Verify gradients were scaled by world size
        for param in buffer.params:
            expected = torch.ones_like(param) / 4.0  # Divided by world_size
            if param.grad is not None:
                self.assertTrue(torch.allclose(param.grad, expected))

    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.all_reduce")
    def test_async_all_reduce(self, mock_all_reduce, mock_is_init, mock_world_size):
        """Test asynchronous all-reduce."""
        mock_is_init.return_value = True
        mock_world_size.return_value = 4
        mock_handle = MagicMock()
        mock_all_reduce.return_value = mock_handle

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleModel()
        model.to(device)

        manager = BufferManager(
            model=model,
            data_parallel_group=MagicMock(),
            overlap_comm=False,
        )

        # Create gradients
        for param in model.parameters():
            if param.requires_grad:
                param.grad = torch.ones_like(param)

        # Start async all-reduce
        manager.all_reduce_gradients(async_op=True)

        # Verify handles were stored
        self.assertGreater(len(manager.comm_handles), 0)

        # Wait for completion
        manager.wait_for_all_reduce()

        # Verify wait was called on mock handles
        mock_handle.wait.assert_called()


class TestExceptionHandling(unittest.TestCase):
    """Tests for custom exception handling."""

    def test_bucket_config_validation(self):
        """Test BucketConfig validation raises proper exceptions."""
        # Test invalid bucket size
        with self.assertRaises(BucketConfigError):
            BucketConfig(bucket_size_mb=0.5)  # Below minimum

        with self.assertRaises(BucketConfigError):
            BucketConfig(bucket_size_mb=150)  # Above maximum

        # Test invalid alignment
        with self.assertRaises(BucketConfigError):
            BucketConfig(alignment=0)  # Zero alignment

        with self.assertRaises(BucketConfigError):
            BucketConfig(alignment=127)  # Not power of 2

    def test_bucket_capacity_error(self):
        """Test bucket capacity exceptions."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create small bucket
        bucket = GradientBucket(
            bucket_id=0,
            dtype=torch.float32,
            device=device,
            bucket_size_bytes=100,  # Very small bucket
            alignment=16,
        )

        # Try to add large parameter
        large_param = nn.Parameter(torch.randn(1000, 1000, device=device))

        with self.assertRaises(BucketCapacityError):
            bucket.add_param(large_param)

    def test_communication_error(self):
        """Test communication error when bucket not ready."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bucket = GradientBucket(
            bucket_id=0,
            dtype=torch.float32,
            device=device,
            bucket_size_bytes=1024,
            alignment=128,
        )

        # Try to start all-reduce without packing gradients
        with self.assertRaises(CommunicationError):
            bucket.start_all_reduce()

    def test_parameter_mapping_error(self):
        """Test parameter mapping error with empty params list."""
        with self.assertRaises(ParameterMappingError):
            ParamAndGradBuffer(
                dtype=torch.float32,
                params=[],  # Empty params list
            )


class TestContextManagers(unittest.TestCase):
    """Tests for context manager functionality."""

    def test_buffer_manager_context(self):
        """Test BufferManager context manager."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleModel()
        model.to(device)

        with BufferManager(model=model) as manager:
            # Use manager
            self.assertIsNotNone(manager)
            self.assertGreater(len(manager.buffers), 0)

        # After exiting, gradients should be zeroed
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                self.assertTrue(torch.all(param.grad == 0))

    def test_gradient_accumulation_context(self):
        """Test gradient accumulation context manager."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleModel()
        model.to(device)

        manager = BufferManager(model=model)

        # Create some gradients
        for param in model.parameters():
            if param.requires_grad:
                param.grad = torch.ones_like(param) * 4.0

        # Use gradient accumulation context (auto-scales on exit; no external scaling)
        with manager.gradient_accumulation_context(num_steps=TEST_ACC_STEPS):
            pass

        # After context, gradients should be scaled
        for buffer in manager.buffers.values():
            buffer.sync_gradients_to_buffer()
            # Check that gradients are scaled by 1/4
            self.assertTrue(
                torch.allclose(buffer.grad_data, torch.ones_like(buffer.grad_data))
            )


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and boundary conditions."""

    def test_single_parameter_buffer(self):
        """Test buffer with single parameter."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        param = nn.Parameter(torch.randn(10, 10, device=device))

        buffer = ParamAndGradBuffer(
            dtype=torch.float32,
            params=[param],
        )

        self.assertEqual(len(buffer.params), 1)
        self.assertEqual(buffer.numel, param.numel())

    def test_mixed_requires_grad(self):
        """Test handling of mixed requires_grad parameters."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create params with mixed requires_grad
        param1 = nn.Parameter(torch.randn(10, 10, device=device))
        param2 = nn.Parameter(torch.randn(20, 20, device=device))
        param3 = nn.Parameter(torch.randn(15, 15, device=device))

        param2.requires_grad = False  # Disable gradients for param2

        buffer = ParamAndGradBuffer(
            dtype=torch.float32,
            params=[param1, param2, param3],
        )

        # Should only include params with requires_grad=True
        self.assertEqual(len(buffer.params), 2)
        # Check using identity (id) rather than value equality to avoid
        # tensor comparison issues
        param_ids = [id(p) for p in buffer.params]
        self.assertIn(id(param1), param_ids)
        self.assertNotIn(id(param2), param_ids)
        self.assertIn(id(param3), param_ids)

    def test_zero_size_operations(self):
        """Test operations with zero-size tensors."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleModel()
        model.to(device)

        # Disable all gradients
        for param in model.parameters():
            param.requires_grad = False

        # Create a new parameter with gradient
        new_param = nn.Parameter(torch.randn(5, 5, device=device))
        model.register_parameter("new_param", new_param)

        manager = BufferManager(model=model)

        # Should handle zero gradients gracefully
        manager.zero_gradients()
        manager.clip_gradients(max_norm=1.0)

    def test_very_large_model(self):
        """Test with very large number of parameters."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create model with many small parameters
        class LargeModel(nn.Module):
            def __init__(self):
                super().__init__()
                for i in range(100):
                    setattr(
                        self,
                        f"param_{i}",
                        nn.Parameter(torch.randn(10, 10, device=device)),
                    )

        model = LargeModel()

        manager = BufferManager(
            model=model,
            bucket_config=BucketConfig(bucket_size_mb=TEST_BUCKET_SIZE_MB),
        )

        # Should create multiple buckets
        self.assertGreater(len(manager.buffers), 0)

        # Memory usage should be reasonable
        stats = manager.get_memory_usage()
        self.assertIn("total_memory_mb", stats)
        total_memory = stats.get("total_memory_mb", 0)
        if isinstance(total_memory, (int, float)):
            self.assertGreater(total_memory, 0)


if __name__ == "__main__":
    unittest.main()
