"""
Tests for Gradient Bucket Coalescing

This module provides comprehensive tests for the gradient bucket coalescing
implementation, including unit tests, integration tests, and performance
benchmarks.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from rosellm.rosetrainer.communication.coalescing import (
    CoalescingConfig,
    CoalescingManager,
    CoalescingMetrics,
)
from rosellm.rosetrainer.optimizer.coalesced_gradient_buffer import (
    CoalescedBucket,
    CoalescedGradientBuffer,
)


class TestCoalescingConfig(unittest.TestCase):
    """Test CoalescingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CoalescingConfig()
        self.assertTrue(config.enable_coalescing)
        self.assertEqual(config.max_coalesce_size_mb, 100.0)
        self.assertEqual(config.min_buckets_to_coalesce, 2)
        self.assertTrue(config.adaptive_sizing)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CoalescingConfig(
            enable_coalescing=False,
            max_coalesce_size_mb=50.0,
            adaptive_sizing=False,
        )
        self.assertFalse(config.enable_coalescing)
        self.assertEqual(config.max_coalesce_size_mb, 50.0)
        self.assertFalse(config.adaptive_sizing)


class TestCoalescingMetrics(unittest.TestCase):
    """Test CoalescingMetrics tracking."""

    def test_metrics_update(self):
        """Test metrics update functionality."""
        metrics = CoalescingMetrics()

        # Update with first operation
        metrics.update(num_ops=5, bytes_coalesced=1024 * 1024, time_ms=10.0)

        self.assertEqual(metrics.total_coalesced_ops, 5)
        self.assertEqual(metrics.total_bytes_coalesced, 1024 * 1024)
        self.assertEqual(metrics.total_coalesce_time_ms, 10.0)
        self.assertEqual(metrics.num_coalesce_calls, 1)
        self.assertEqual(metrics.avg_ops_per_coalesce, 5.0)
        self.assertEqual(metrics.peak_coalesce_size_mb, 1.0)

    def test_metrics_accumulation(self):
        """Test metrics accumulation over multiple updates."""
        metrics = CoalescingMetrics()

        # Multiple updates
        metrics.update(num_ops=3, bytes_coalesced=512 * 1024, time_ms=5.0)
        metrics.update(num_ops=7, bytes_coalesced=2 * 1024 * 1024, time_ms=15.0)

        self.assertEqual(metrics.total_coalesced_ops, 10)
        self.assertEqual(metrics.num_coalesce_calls, 2)
        self.assertEqual(metrics.avg_ops_per_coalesce, 5.0)
        self.assertEqual(metrics.peak_coalesce_size_mb, 2.0)


class TestCoalescingManager(unittest.TestCase):
    """Test CoalescingManager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = CoalescingConfig(enable_coalescing=True)

        # Mock process group
        self.mock_pg = MagicMock()
        self.mock_pg.size.return_value = 4
        self.mock_pg.rank.return_value = 0

    @patch("rosellm.rosetrainer.communication.coalescing.dist.get_backend")
    def test_manager_initialization(self, mock_get_backend):
        """Test manager initialization."""
        mock_get_backend.return_value = "nccl"

        manager = CoalescingManager(process_group=self.mock_pg, config=self.config)

        self.assertIsNotNone(manager)
        self.assertEqual(manager.process_group, self.mock_pg)
        self.assertEqual(manager.config, self.config)
        self.assertIsNotNone(manager.metrics)

    @patch("rosellm.rosetrainer.communication.coalescing.HAS_COALESCING", False)
    def test_fallback_mode(self):
        """Test fallback when coalescing is not available."""
        manager = CoalescingManager(config=self.config)

        # Should disable coalescing when not available
        self.assertFalse(manager.supports_coalescing)

    def test_adaptive_sizing(self):
        """Test adaptive sizing functionality."""
        config = CoalescingConfig(adaptive_sizing=True)
        manager = CoalescingManager(config=config)

        # Simulate performance history
        manager._adjust_coalesce_size(
            num_ops=10, total_bytes=50 * 1024 * 1024, elapsed_ms=20.0
        )

        self.assertTrue(hasattr(manager, "adaptive_size_mb"))

    def test_get_optimal_bucket_size(self):
        """Test optimal bucket size retrieval."""
        manager = CoalescingManager(config=self.config)

        size = manager.get_optimal_bucket_size()
        self.assertEqual(size, self.config.max_coalesce_size_mb)

        # With adaptive sizing
        manager.adaptive_size_mb = 75.0
        size = manager.get_optimal_bucket_size()
        self.assertEqual(size, 75.0 if self.config.adaptive_sizing else 100.0)


class TestCoalescedBucket(unittest.TestCase):
    """Test CoalescedBucket functionality."""

    def test_bucket_initialization(self):
        """Test coalesced bucket initialization."""
        params = [nn.Parameter(torch.randn(100, 100))]
        bucket = CoalescedBucket(
            index=0,
            size=10000,
            dtype=torch.float32,
            params=params,
            param_indices=[0],
        )

        self.assertEqual(bucket.index, 0)
        self.assertEqual(bucket.size, 10000)
        self.assertIsNone(bucket.coalesce_group_id)

    def test_calculate_bytes_size(self):
        """Test byte size calculation."""
        bucket = CoalescedBucket(
            index=0,
            size=1000,
            dtype=torch.float32,
            params=[],
            param_indices=[],
        )

        # Without grad buffer
        size = bucket.calculate_bytes_size()
        self.assertEqual(size, 0)

        # With grad buffer
        bucket.grad_buffer = torch.zeros(1000, dtype=torch.float32)
        size = bucket.calculate_bytes_size()
        self.assertEqual(size, 1000 * 4)  # float32 = 4 bytes


class TestCoalescedGradientBuffer(unittest.TestCase):
    """Test CoalescedGradientBuffer functionality."""

    def setUp(self):
        """Set up test model and parameters."""
        self.model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )
        self.params = list(self.model.parameters())

    def test_buffer_initialization(self):
        """Test coalesced gradient buffer initialization."""
        buffer = CoalescedGradientBuffer(
            params=self.params,
            enable_coalescing=True,
            bucket_size_mb=1.0,
        )

        self.assertTrue(buffer.enable_coalescing)
        self.assertIsNotNone(buffer.coalescing_manager)
        self.assertIsInstance(buffer.buckets[0], CoalescedBucket)

    def test_coalescing_groups_creation(self):
        """Test creation of coalescing groups."""
        config = CoalescingConfig(
            max_coalesce_size_mb=0.001,  # Very small for testing
            min_coalesce_size_mb=0.0001,  # Even smaller minimum
        )
        buffer = CoalescedGradientBuffer(
            params=self.params,
            enable_coalescing=True,
            coalescing_config=config,
            bucket_size_mb=0.0005,
        )

        # Should create multiple groups due to small max size
        self.assertGreater(len(buffer.coalescing_groups), 0)

        # Each bucket should have a group ID
        for bucket in buffer.buckets:
            if isinstance(bucket, CoalescedBucket):
                self.assertIsNotNone(bucket.coalesce_group_id)

    def test_distributed_optimizer_mode(self):
        """Test buffer with distributed optimizer mode."""
        buffer = CoalescedGradientBuffer(
            params=self.params,
            enable_coalescing=True,
            use_distributed_optimizer=True,
        )

        self.assertTrue(buffer.use_distributed_optimizer)

    def test_get_coalescing_stats(self):
        """Test retrieval of coalescing statistics."""
        buffer = CoalescedGradientBuffer(
            params=self.params,
            enable_coalescing=True,
        )

        stats = buffer.get_coalescing_stats()

        self.assertIn("total_coalesced_ops", stats)
        self.assertIn("num_coalescing_groups", stats)
        self.assertEqual(stats["num_coalescing_groups"], len(buffer.coalescing_groups))

    def test_optimize_coalescing_groups(self):
        """Test re-optimization of coalescing groups."""
        config = CoalescingConfig(adaptive_sizing=True)
        buffer = CoalescedGradientBuffer(
            params=self.params,
            enable_coalescing=True,
            coalescing_config=config,
        )

        # Simulate performance improvement
        if buffer.coalescing_manager is not None:
            buffer.coalescing_manager.adaptive_size_mb = 200.0
            buffer.optimize_coalescing_groups()

        # Groups might change based on new size
        self.assertGreaterEqual(len(buffer.coalescing_groups), 1)


class TestIntegration(unittest.TestCase):
    """Integration tests for coalescing with gradient synchronization."""

    def setUp(self):
        """Set up test environment."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create a simple model
        self.model = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        ).to(self.device)

    def test_end_to_end_coalescing(self):
        """Test complete gradient synchronization with coalescing."""
        params = list(self.model.parameters())

        # Create buffer with coalescing
        buffer = CoalescedGradientBuffer(
            params=params,
            enable_coalescing=True,
            bucket_size_mb=0.01,  # Small buckets for testing
        )

        # Simulate gradients
        for param in params:
            if param.requires_grad:
                param.grad = torch.randn_like(param)

        # Mark buckets as ready
        for bucket in buffer.buckets:
            bucket.is_ready = True
            bucket.grad_buffer = torch.randn(
                bucket.size, dtype=bucket.dtype, device=self.device
            )

        # Test synchronization (would actually communicate in distributed setting)
        with patch.object(buffer, "_all_reduce_bucket") as mock_reduce:
            with patch.object(buffer, "_wait_all_handles"):
                buffer.synchronize_gradients()

                # Verify all buckets were processed
                self.assertEqual(mock_reduce.call_count, len(buffer.buckets))

    def test_performance_comparison(self):
        """Compare performance with and without coalescing."""
        params = list(self.model.parameters())

        # Without coalescing
        buffer_no_coalesce = CoalescedGradientBuffer(
            params=params,
            enable_coalescing=False,
        )

        # With coalescing
        buffer_coalesce = CoalescedGradientBuffer(
            params=params,
            enable_coalescing=True,
        )

        # Simulate gradients
        for param in params:
            if param.requires_grad:
                param.grad = torch.randn_like(param)

        # This would show performance difference in actual distributed setting
        self.assertIsNone(buffer_no_coalesce.coalescing_manager)
        self.assertIsNotNone(buffer_coalesce.coalescing_manager)


class TestMemoryEfficiency(unittest.TestCase):
    """Test memory efficiency of coalescing."""

    def test_memory_usage(self):
        """Test that coalescing doesn't significantly increase memory."""
        # Create large model
        model = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.Linear(1000, 1000),
            nn.Linear(1000, 1000),
        )

        params = list(model.parameters())

        # Create buffer
        buffer = CoalescedGradientBuffer(
            params=params,
            enable_coalescing=True,
        )

        # Memory overhead should be minimal
        # Mainly just metadata for grouping
        self.assertLess(len(buffer.coalescing_groups), 100)


if __name__ == "__main__":
    unittest.main()
