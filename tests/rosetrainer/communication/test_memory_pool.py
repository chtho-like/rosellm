"""
Tests for Coalescing Memory Pool
"""

import unittest

import torch

from rosellm.rosetrainer.communication.memory_pool import CoalescingMemoryPool


class TestCoalescingMemoryPool(unittest.TestCase):
    """Test cases for CoalescingMemoryPool."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.pool = CoalescingMemoryPool(
            initial_size_mb=10.0, max_size_mb=50.0, device=self.device
        )

    def test_initialization(self):
        """Test memory pool initialization."""
        self.assertIsNotNone(self.pool)
        self.assertEqual(self.pool.device, self.device)
        self.assertGreater(self.pool._allocated_bytes, 0)

        # Check pre-allocated buffers
        stats = self.pool.get_stats()
        self.assertGreater(stats["num_buffers"], 0)

    def test_get_buffer_exact_size(self):
        """Test getting buffer of exact size from pool."""
        # Pre-populate pool with specific buffer
        buffer1 = torch.empty(1000, dtype=torch.float32, device=self.device)
        self.pool.return_buffer(buffer1)

        # Get buffer of same size
        buffer2 = self.pool.get_buffer(1000, torch.float32)
        self.assertEqual(buffer2.numel(), 1000)
        self.assertEqual(buffer2.dtype, torch.float32)

        # Should be a hit
        self.assertGreater(self.pool.hits, 0)

    def test_get_buffer_larger_available(self):
        """Test getting buffer when larger buffer is available."""
        # Pre-populate with larger buffer
        large_buffer = torch.empty(2000, dtype=torch.float32, device=self.device)
        self.pool.return_buffer(large_buffer)

        # Request smaller buffer
        buffer = self.pool.get_buffer(1000, torch.float32)
        self.assertEqual(buffer.numel(), 1000)

        # Should still be a hit
        self.assertGreater(self.pool.hits, 0)

    def test_get_buffer_allocation(self):
        """Test new buffer allocation when none available."""
        # Clear pool
        self.pool.clear()

        # Request buffer (should allocate new)
        buffer = self.pool.get_buffer(5000, torch.float16)
        self.assertEqual(buffer.numel(), 5000)
        self.assertEqual(buffer.dtype, torch.float16)

        # Should be a miss
        self.assertGreater(self.pool.misses, 0)
        self.assertGreater(self.pool.allocations, 0)

    def test_zero_out_option(self):
        """Test zero_out option for buffers."""
        # Create buffer with non-zero values
        buffer = torch.ones(100, dtype=torch.float32, device=self.device)
        self.pool.return_buffer(buffer)

        # Get buffer with zero_out
        clean_buffer = self.pool.get_buffer(100, torch.float32, zero_out=True)
        self.assertTrue(torch.all(clean_buffer == 0))

    def test_eviction(self):
        """Test buffer eviction when pool is full."""
        # Set small pool for testing
        small_pool = CoalescingMemoryPool(
            initial_size_mb=0.001, max_size_mb=0.01, device=self.device
        )

        # Allocate buffers until we exceed limit
        buffers = []
        for i in range(10):
            buffer = small_pool.get_buffer(1000, torch.float32)
            buffers.append(buffer)

        # Return some buffers
        for buffer in buffers[:5]:
            small_pool.return_buffer(buffer)

        # Allocate large buffer (should trigger eviction)
        large_buffer = small_pool.get_buffer(10000, torch.float32)
        self.assertEqual(large_buffer.numel(), 10000)

    def test_statistics(self):
        """Test statistics tracking."""
        # Clear the pool first to ensure we get misses
        self.pool.clear()
        initial_stats = self.pool.get_stats()

        # Perform operations
        # Should be a miss (pool was cleared)
        buffer1 = self.pool.get_buffer(1000, torch.float32)
        self.pool.get_buffer(2000, torch.float32)  # Should be a miss
        self.pool.return_buffer(buffer1)
        self.pool.get_buffer(1000, torch.float32)  # Should be a hit

        final_stats = self.pool.get_stats()

        # Check stats updated
        self.assertGreater(final_stats["hits"], initial_stats["hits"])
        self.assertGreater(final_stats["misses"], initial_stats["misses"])
        self.assertGreater(final_stats["hit_rate"], 0)

    def test_clear(self):
        """Test clearing the pool."""
        # Add buffers to pool
        for i in range(5):
            buffer = torch.empty(
                1000 * (i + 1), dtype=torch.float32, device=self.device
            )
            self.pool.return_buffer(buffer)

        # Clear pool
        self.pool.clear()

        # Check pool is empty
        stats = self.pool.get_stats()
        self.assertEqual(stats["allocated_mb"], 0)

    def test_thread_safety(self):
        """Test thread-safe operations."""
        import threading
        import time

        errors = []

        def worker():
            try:
                for _ in range(10):
                    buffer = self.pool.get_buffer(1000, torch.float32)
                    time.sleep(0.001)  # Simulate work
                    self.pool.return_buffer(buffer)
            except Exception as e:
                errors.append(e)

        # Run multiple threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)

        # Wait for completion
        for t in threads:
            t.join()

        # Check no errors occurred
        self.assertEqual(len(errors), 0)

    def test_device_handling(self):
        """Test handling of different devices."""
        if torch.cuda.is_available():
            cuda_pool = CoalescingMemoryPool(device=torch.device("cuda:0"))

            buffer = cuda_pool.get_buffer(1000, torch.float32)
            self.assertEqual(buffer.device.type, "cuda")
            self.assertEqual(buffer.device.index, 0)


class TestMemoryPoolIntegration(unittest.TestCase):
    """Integration tests for memory pool with coalescing."""

    def test_with_coalescing_manager(self):
        """Test memory pool integration with CoalescingManager."""
        from rosellm.rosetrainer.communication.coalescing import (
            CoalescingConfig,
            CoalescingManager,
        )

        config = CoalescingConfig(
            use_memory_pool=True,
            memory_pool_size_mb=20.0,
        )

        manager = CoalescingManager(config=config)

        # Check memory pool was created
        self.assertIsNotNone(manager.memory_pool)

        # Get pool stats
        stats = manager.get_memory_pool_stats()
        self.assertIsNotNone(stats)
        if stats is not None:
            self.assertIn("allocated_mb", stats)

    def test_cleanup(self):
        """Test cleanup of memory pool resources."""
        from rosellm.rosetrainer.communication.coalescing import (
            CoalescingConfig,
            CoalescingManager,
        )

        config = CoalescingConfig(use_memory_pool=True)
        manager = CoalescingManager(config=config)

        # Use the pool
        if manager.memory_pool:
            buffer = manager.memory_pool.get_buffer(1000, torch.float32)
            manager.memory_pool.return_buffer(buffer)

        # Cleanup
        manager.cleanup()

        # Check pool is cleared
        if manager.memory_pool:
            stats = manager.memory_pool.get_stats()
            self.assertEqual(stats["allocated_mb"], 0)


if __name__ == "__main__":
    unittest.main()
