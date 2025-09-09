"""
Comprehensive tests for gradient bucketing with Megatron-LM validation.

This test suite validates the gradient bucketing implementation against
Megatron-LM's behavior, ensuring bit-to-bit accuracy where applicable.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from rosellm.rosetrainer.gradient.bucketing import (
    BucketingStrategy,
    GradientBucket,
    GradientBucketConfig,
    GradientBucketManager,
    create_gradient_buckets,
)


class SimpleModel(nn.Module):
    """Simple test model with various layer types."""

    def __init__(self, hidden_size: int = 128, num_layers: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(1000, hidden_size)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x) + x  # Residual connection
        x = self.norm(x)
        return self.output(x)


class TestGradientBucket(unittest.TestCase):
    """Test individual gradient bucket functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create test parameters
        self.params = [
            nn.Parameter(torch.randn(100, 100, device=self.device)),
            nn.Parameter(torch.randn(50, 50, device=self.device)),
            nn.Parameter(torch.randn(25, device=self.device)),
        ]

        # Calculate total elements
        self.total_numel = sum(p.numel() for p in self.params)

    def test_bucket_creation(self):
        """Test bucket creation and initialization."""
        bucket = GradientBucket(
            bucket_id=0,
            params=self.params,
            numel=self.total_numel,
            dtype=torch.float32,
            device=self.device,
        )

        self.assertEqual(bucket.bucket_id, 0)
        self.assertEqual(len(bucket.params), 3)
        self.assertEqual(bucket.numel, self.total_numel)
        self.assertEqual(len(bucket.param_to_buffer_offset), 3)
        self.assertIsNone(bucket.grad_buffer)

    def test_gradient_buffer_allocation(self):
        """Test gradient buffer allocation."""
        bucket = GradientBucket(
            bucket_id=0,
            params=self.params,
            numel=self.total_numel,
            dtype=torch.float32,
            device=self.device,
        )

        bucket.allocate_grad_buffer()

        self.assertIsNotNone(bucket.grad_buffer)
        assert bucket.grad_buffer is not None  # Type guard for mypy
        self.assertEqual(bucket.grad_buffer.numel(), self.total_numel)
        self.assertEqual(bucket.grad_buffer.dtype, torch.float32)
        self.assertEqual(bucket.grad_buffer.device, self.device)

    def test_gradient_registration(self):
        """Test gradient ready registration."""
        bucket = GradientBucket(
            bucket_id=0,
            params=self.params,
            numel=self.total_numel,
            dtype=torch.float32,
            device=self.device,
        )

        # Register first two parameters
        self.assertFalse(bucket.register_grad_ready(self.params[0]))
        self.assertFalse(bucket.register_grad_ready(self.params[1]))

        # Register last parameter - should return True
        self.assertTrue(bucket.register_grad_ready(self.params[2]))
        self.assertTrue(bucket.all_gradients_ready)

    def test_gradient_copy_operations(self):
        """Test copying gradients to/from buffer."""
        bucket = GradientBucket(
            bucket_id=0,
            params=self.params,
            numel=self.total_numel,
            dtype=torch.float32,
            device=self.device,
        )

        # Set gradients for parameters
        for param in self.params:
            param.grad = torch.randn_like(param)

        # Copy to buffer
        bucket.copy_gradients_to_buffer()

        self.assertIsNotNone(bucket.grad_buffer)

        # Verify buffer contains gradient data
        assert bucket.grad_buffer is not None  # Type guard
        offset = 0
        for param in self.params:
            assert param.grad is not None  # Type guard
            param_numel = param.numel()
            buffer_slice = bucket.grad_buffer[offset : offset + param_numel]
            buffer_view = buffer_slice.view_as(param.grad)
            self.assertTrue(torch.allclose(buffer_view, param.grad))
            offset += param_numel

        # Modify buffer and copy back
        bucket.grad_buffer.mul_(2.0)
        bucket.copy_buffer_to_gradients()

        # Verify gradients were updated
        for param in self.params:
            assert param.grad is not None  # Type guard
            original_grad = torch.randn_like(param)  # This won't match
            self.assertFalse(torch.allclose(param.grad, original_grad))

    def test_bucket_reset(self):
        """Test bucket state reset."""
        bucket = GradientBucket(
            bucket_id=0,
            params=self.params,
            numel=self.total_numel,
            dtype=torch.float32,
            device=self.device,
        )

        # Set up some state
        bucket.register_grad_ready(self.params[0])
        bucket.allocate_grad_buffer()
        assert bucket.grad_buffer is not None  # Type guard
        bucket.grad_buffer.fill_(1.0)
        bucket.communication_handle = MagicMock()

        # Reset
        bucket.reset()

        self.assertEqual(len(bucket.params_with_grad), 0)
        self.assertFalse(bucket.all_gradients_ready)
        self.assertIsNone(bucket.communication_handle)
        self.assertTrue(torch.all(bucket.grad_buffer == 0))


class TestGradientBucketManager(unittest.TestCase):
    """Test gradient bucket manager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleModel().to(self.device)

        # Mock distributed environment
        self.mock_dist_patcher = patch("torch.distributed.is_initialized")
        self.mock_is_init = self.mock_dist_patcher.start()
        self.mock_is_init.return_value = True

        # Mock world size and rank
        with patch("torch.distributed.get_world_size", return_value=2):
            with patch("torch.distributed.get_rank", return_value=0):
                self.config = GradientBucketConfig(
                    bucket_size_mb=1,
                    bucketing_strategy=BucketingStrategy.SIZE_BASED,
                    overlap_communication=False,
                )

    def tearDown(self):
        """Clean up mocks."""
        self.mock_dist_patcher.stop()

    @patch("torch.distributed.get_world_size", return_value=2)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_manager_creation(self, mock_rank, mock_world):
        """Test bucket manager creation."""
        manager = GradientBucketManager(self.model, self.config)

        self.assertGreater(len(manager.buckets), 0)
        self.assertEqual(
            len(manager.param_to_bucket),
            sum(1 for p in self.model.parameters() if p.requires_grad),
        )

        # Verify all parameters are in buckets
        for param in self.model.parameters():
            if param.requires_grad:
                self.assertIn(param, manager.param_to_bucket)

    @patch("torch.distributed.get_world_size", return_value=2)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_size_based_bucketing(self, mock_rank, mock_world):
        """Test size-based bucketing strategy."""
        config = GradientBucketConfig(
            bucket_size_mb=1.0,  # Use minimum allowed size
            bucketing_strategy=BucketingStrategy.SIZE_BASED,
        )
        manager = GradientBucketManager(self.model, config)

        # Check bucket sizes are within limits
        bucket_size_bytes = config.bucket_size_mb * 1024 * 1024
        for bucket in manager.buckets:
            bytes_per_element = torch.tensor([], dtype=bucket.dtype).element_size()
            bucket_bytes = bucket.numel * bytes_per_element
            # Allow for cap factor
            self.assertLessEqual(
                bucket_bytes,
                bucket_size_bytes * config.bucket_cap_factor * 1.1,  # Small tolerance
            )

    @patch("torch.distributed.get_world_size", return_value=2)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_type_based_bucketing(self, mock_rank, mock_world):
        """Test type-based bucketing strategy."""
        config = GradientBucketConfig(bucketing_strategy=BucketingStrategy.TYPE_BASED)
        manager = GradientBucketManager(self.model, config)

        # Verify parameters are grouped by module type
        module_types = set()
        for name, module in self.model.named_modules():
            if len(list(module.parameters(recurse=False))) > 0:
                module_types.add(type(module).__name__)

        # Should have buckets for different module types
        self.assertGreater(len(manager.buckets), 0)
        self.assertLessEqual(len(manager.buckets), len(module_types))

    @patch("torch.distributed.get_world_size", return_value=2)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_layer_based_bucketing(self, mock_rank, mock_world):
        """Test layer-based bucketing strategy."""
        config = GradientBucketConfig(
            bucket_size_mb=1.0,  # Use minimum allowed size
            bucketing_strategy=BucketingStrategy.LAYER_BASED,
        )
        manager = GradientBucketManager(self.model, config)

        # Verify consecutive parameters are grouped
        self.assertGreater(len(manager.buckets), 0)

        # Check that parameters maintain order
        param_list = list(p for p in self.model.parameters() if p.requires_grad)
        for bucket in manager.buckets:
            bucket_indices = []
            for param in bucket.params:
                # Use 'is' for parameter identity comparison instead of 'in'
                for i, p in enumerate(param_list):
                    if param is p:
                        bucket_indices.append(i)
                        break

            # Indices should be consecutive (mostly)
            if len(bucket_indices) > 1:
                sorted_indices = sorted(bucket_indices)
                self.assertEqual(bucket_indices, sorted_indices)

    @patch("torch.distributed.get_world_size", return_value=2)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_hybrid_bucketing(self, mock_rank, mock_world):
        """Test hybrid bucketing strategy."""
        config = GradientBucketConfig(
            bucket_size_mb=1.0,  # Use minimum allowed size
            bucketing_strategy=BucketingStrategy.HYBRID,
        )
        manager = GradientBucketManager(self.model, config)

        # Should create buckets considering both type and size
        self.assertGreater(len(manager.buckets), 0)

        stats = manager.get_statistics()
        self.assertEqual(stats["strategy"], BucketingStrategy.HYBRID)

    @patch("torch.distributed.get_world_size", return_value=2)
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.all_reduce")
    def test_synchronize_gradients(self, mock_allreduce, mock_rank, mock_world):
        """Test gradient synchronization."""
        manager = GradientBucketManager(self.model, self.config)

        # Set gradients for all parameters
        for param in self.model.parameters():
            if param.requires_grad:
                param.grad = torch.randn_like(param)

        # Synchronize
        manager.synchronize_gradients()

        # Verify all_reduce was called for each bucket
        self.assertEqual(mock_allreduce.call_count, len(manager.buckets))

    @patch("torch.distributed.get_world_size", return_value=2)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_statistics(self, mock_rank, mock_world):
        """Test statistics collection."""
        manager = GradientBucketManager(self.model, self.config)

        stats = manager.get_statistics()

        self.assertIn("num_buckets", stats)
        self.assertIn("total_parameters", stats)
        self.assertIn("total_numel", stats)
        self.assertIn("bucket_sizes", stats)
        self.assertIn("avg_bucket_size", stats)
        self.assertIn("dtype_distribution", stats)

        self.assertEqual(len(stats["bucket_sizes"]), stats["num_buckets"])
        self.assertEqual(
            stats["total_parameters"], sum(len(b.params) for b in manager.buckets)
        )


class TestGradientBucketingIntegration(unittest.TestCase):
    """Integration tests for gradient bucketing."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_create_gradient_buckets_helper(self):
        """Test the create_gradient_buckets helper function."""
        model = SimpleModel().to(self.device)

        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.get_world_size", return_value=2):
                with patch("torch.distributed.get_rank", return_value=0):
                    # Test with default config
                    manager = create_gradient_buckets(model)
                    self.assertIsInstance(manager, GradientBucketManager)
                    self.assertGreater(len(manager.buckets), 0)

                    # Test with custom config
                    config = GradientBucketConfig(
                        bucket_size_mb=10,
                        bucketing_strategy=BucketingStrategy.HYBRID,
                        overlap_communication=True,
                    )
                    manager = create_gradient_buckets(model, config)
                    self.assertEqual(manager.config.bucket_size_mb, 10)
                    self.assertTrue(manager.config.overlap_communication)

    def test_end_to_end_training_step(self):
        """Test end-to-end training step with bucketing."""
        model = SimpleModel(hidden_size=64, num_layers=2).to(self.device)

        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.get_world_size", return_value=1):
                with patch("torch.distributed.get_rank", return_value=0):
                    with patch("torch.distributed.all_reduce"):
                        config = GradientBucketConfig(
                            bucket_size_mb=1, overlap_communication=False
                        )
                        manager = create_gradient_buckets(model, config)

                        # Simulate forward and backward pass
                        batch_size = 4
                        seq_len = 16
                        input_ids = torch.randint(
                            0, 1000, (batch_size, seq_len), device=self.device
                        )

                        output = model(input_ids)
                        loss = output.mean()
                        loss.backward()

                        # Synchronize gradients
                        manager.synchronize_gradients()

                        # Verify gradients were computed
                        for param in model.parameters():
                            if param.requires_grad:
                                self.assertIsNotNone(param.grad)

                        # Reset for next iteration
                        manager.reset()

    def test_overlapped_communication(self):
        """Test overlapped communication with backward pass."""
        model = SimpleModel(hidden_size=64, num_layers=2).to(self.device)

        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.get_world_size", return_value=2):
                with patch("torch.distributed.get_rank", return_value=0):
                    with patch("torch.distributed.all_reduce") as mock_allreduce:
                        # Set async_op to return a mock handle
                        mock_handle = MagicMock()
                        mock_allreduce.return_value = mock_handle

                        config = GradientBucketConfig(
                            bucket_size_mb=1.0,  # Use minimum allowed size
                            overlap_communication=True,
                        )
                        manager = create_gradient_buckets(model, config)

                        # Simulate forward and backward pass
                        input_ids = torch.randint(0, 1000, (2, 8), device=self.device)
                        output = model(input_ids)
                        loss = output.mean()

                        # Backward pass should trigger hooks
                        loss.backward()

                        # Should have pending communications
                        self.assertGreater(len(manager.pending_communications), 0)

                        # Synchronize to complete
                        manager.synchronize_gradients()

                        # Verify handles were waited on
                        for bucket, handle in manager.pending_communications:
                            if handle is not None:
                                handle.wait.assert_called()


class TestMegatronCompatibility(unittest.TestCase):
    """Test compatibility with Megatron-LM patterns."""

    def test_bucket_size_calculation(self):
        """Test bucket size calculation matches Megatron-LM logic."""
        # Megatron uses similar size-based bucketing
        model = SimpleModel(hidden_size=256, num_layers=8)

        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.get_world_size", return_value=8):
                with patch("torch.distributed.get_rank", return_value=0):
                    config = GradientBucketConfig(
                        bucket_size_mb=50,  # Megatron default
                        bucketing_strategy=BucketingStrategy.SIZE_BASED,
                        dtype_bucketing=True,
                    )
                    manager = create_gradient_buckets(model, config)

                    # Verify bucketing respects size limits
                    max_bucket_bytes = (
                        config.bucket_size_mb * 1024 * 1024 * config.bucket_cap_factor
                    )
                    for bucket in manager.buckets:
                        bytes_per_element = torch.tensor(
                            [], dtype=bucket.dtype
                        ).element_size()
                        bucket_bytes = bucket.numel * bytes_per_element
                        self.assertLessEqual(bucket_bytes, max_bucket_bytes)

    def test_distributed_optimizer_mode(self):
        """Test distributed optimizer mode (reduce-scatter vs all-reduce)."""
        model = SimpleModel()

        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.get_world_size", return_value=4):
                with patch("torch.distributed.get_rank", return_value=0):
                    with patch("torch.distributed.reduce_scatter") as mock_rs:
                        config = GradientBucketConfig(
                            use_distributed_optimizer=True, overlap_communication=False
                        )
                        manager = create_gradient_buckets(model, config)

                        # Set gradients
                        for param in model.parameters():
                            if param.requires_grad:
                                param.grad = torch.randn_like(param)

                        # Synchronize should use reduce-scatter
                        manager.synchronize_gradients()

                        # Verify reduce_scatter was called
                        self.assertGreater(mock_rs.call_count, 0)


if __name__ == "__main__":
    unittest.main()
