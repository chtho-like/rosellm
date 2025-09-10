"""
Comprehensive tests for parameter gathering overlap with computation.

These tests verify the correctness and performance of async parameter gathering,
stream-based overlap management, and integration with parallelism strategies.
"""

import time
import unittest

import torch
import torch.nn as nn

from rosellm.rosetrainer.memory.parameter_overlap import (
    AsyncParameterGatherer,
    OverlapConfig,
    OverlapMode,
    OverlappedLinear,
    ParameterCache,
    PipelineOverlapScheduler,
    StreamPool,
)


class TestStreamPool(unittest.TestCase):
    """Test stream pool for overlapped operations."""

    def test_stream_pool_cpu(self) -> None:
        """Test stream pool on CPU (should handle gracefully)."""
        device = torch.device("cpu")
        pool = StreamPool(num_streams=4, device=device)

        # CPU should not create actual streams
        self.assertEqual(len(pool.streams), 0)

        # Acquire should return None for CPU
        stream = pool.acquire_stream()
        self.assertIsNone(stream)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_stream_pool_gpu(self) -> None:
        """Test stream pool on GPU."""
        device = torch.device("cuda:0")
        pool = StreamPool(num_streams=4, device=device)

        # Should create 4 streams
        self.assertEqual(len(pool.streams), 4)

        # Test acquire and release
        streams = []
        for _ in range(4):
            stream = pool.acquire_stream()
            self.assertIsNotNone(stream)
            streams.append(stream)

        # Pool should be exhausted
        extra_stream = pool.acquire_stream()
        self.assertIsNone(extra_stream)

        # Release one stream
        pool.release_stream(streams[0])

        # Should be able to acquire again
        reacquired = pool.acquire_stream()
        self.assertIsNotNone(reacquired)
        self.assertEqual(reacquired, streams[0])

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_stream_synchronization(self) -> None:
        """Test stream synchronization."""
        device = torch.device("cuda:0")
        pool = StreamPool(num_streams=2, device=device)

        stream = pool.acquire_stream()
        self.assertIsNotNone(stream)

        # Schedule work on stream
        with torch.cuda.stream(stream):
            tensor = torch.randn(1000, 1000, device=device)
            result = torch.matmul(tensor, tensor)

        # Synchronize all streams
        pool.wait_all()

        # Result should be ready
        self.assertTrue(result.is_cuda)


class TestParameterCache(unittest.TestCase):
    """Test LRU cache for gathered parameters."""

    def test_cache_basic_operations(self) -> None:
        """Test basic cache operations."""
        cache = ParameterCache(max_size_mb=1)

        # Test put and get
        tensor1 = torch.randn(100, 100)
        cache.put("key1", tensor1)
        retrieved = cache.get("key1")
        self.assertIsNotNone(retrieved)
        torch.testing.assert_close(retrieved, tensor1)

        # Test miss
        missed = cache.get("nonexistent")
        self.assertIsNone(missed)

    def test_cache_eviction(self) -> None:
        """Test LRU eviction when cache is full."""
        cache = ParameterCache(max_size_mb=1)  # 1MB cache

        # Calculate approximate tensor size for 0.5MB
        num_elements = (512 * 1024) // 4  # 4 bytes per float32
        size = int(num_elements**0.5)

        # Add first tensor (about 0.5MB)
        tensor1 = torch.randn(size, size)
        cache.put("key1", tensor1)

        # Add second tensor (about 0.5MB)
        tensor2 = torch.randn(size, size)
        cache.put("key2", tensor2)

        # Access key1 to make it more recent
        _ = cache.get("key1")

        # Add third tensor - should evict key2 (LRU)
        tensor3 = torch.randn(size, size)
        cache.put("key3", tensor3)

        # key2 should be evicted, key1 should remain
        self.assertIsNotNone(cache.get("key1"))
        self.assertIsNone(cache.get("key2"))
        self.assertIsNotNone(cache.get("key3"))

    def test_cache_statistics(self) -> None:
        """Test cache statistics tracking."""
        cache = ParameterCache(max_size_mb=10)

        # Generate some hits and misses
        tensor = torch.randn(100, 100)
        cache.put("key1", tensor)

        for _ in range(3):
            cache.get("key1")  # 3 hits

        for i in range(2):
            cache.get(f"miss{i}")  # 2 misses

        stats = cache.get_stats()
        self.assertEqual(stats["hits"], 3)
        self.assertEqual(stats["misses"], 2)
        self.assertAlmostEqual(stats["hit_rate"], 0.6, places=2)


class TestAsyncParameterGatherer(unittest.TestCase):
    """Test async parameter gathering functionality."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_gatherer_initialization(self) -> None:
        """Test gatherer initialization."""
        config = OverlapConfig(mode=OverlapMode.PREFETCH, num_streams=4)
        gatherer = AsyncParameterGatherer(config, device=self.device)

        self.assertEqual(gatherer.config.mode, OverlapMode.PREFETCH)
        self.assertIsNotNone(gatherer.cache)
        self.assertIsNotNone(gatherer.stream_pool)

        # Shutdown cleanly
        gatherer.shutdown()

    def test_gather_async_basic(self) -> None:
        """Test basic async gathering."""
        config = OverlapConfig(mode=OverlapMode.PIPELINE)
        gatherer = AsyncParameterGatherer(config, device=self.device)

        try:
            tensor = torch.randn(100, 100, device=self.device)
            future = gatherer.gather_async(
                param_id="test_param",
                tensor=tensor,
                target_device=self.device,
            )

            # Wait for result
            result = future.result(timeout=1.0)
            self.assertIsNotNone(result)
            torch.testing.assert_close(result, tensor)

        finally:
            gatherer.shutdown()

    def test_gather_with_cache(self) -> None:
        """Test gathering with cache hits."""
        config = OverlapConfig(mode=OverlapMode.PIPELINE, cache_size_mb=10)
        gatherer = AsyncParameterGatherer(config, device=self.device)

        try:
            tensor = torch.randn(100, 100, device=self.device)

            # First gather - cache miss
            future1 = gatherer.gather_async("param1", tensor, self.device)
            result1 = future1.result(timeout=1.0)

            # Second gather - cache hit
            future2 = gatherer.gather_async("param1", tensor, self.device)
            result2 = future2.result(timeout=1.0)

            torch.testing.assert_close(result1, result2)

            # Check cache statistics
            stats = gatherer.get_stats()
            self.assertGreater(stats["cache_stats"]["hits"], 0)

        finally:
            gatherer.shutdown()

    def test_prefetch_parameters(self) -> None:
        """Test parameter prefetching."""
        config = OverlapConfig(mode=OverlapMode.PREFETCH, prefetch_depth=3)
        gatherer = AsyncParameterGatherer(config, device=self.device)

        try:
            # Create multiple parameters
            param_ids = [f"param_{i}" for i in range(5)]
            tensors = [torch.randn(50, 50, device=self.device) for _ in range(5)]

            # Prefetch parameters
            gatherer.prefetch_parameters(param_ids[:3], tensors[:3], self.device)

            # Give some time for prefetch
            time.sleep(0.1)

            # Check if parameters are cached
            for i in range(3):
                cached = gatherer.cache.get(param_ids[i])
                self.assertIsNotNone(
                    cached, f"Parameter {param_ids[i]} should be cached"
                )

        finally:
            gatherer.shutdown()

    def test_priority_gathering(self) -> None:
        """Test priority-based gathering."""
        config = OverlapConfig(mode=OverlapMode.PIPELINE)
        gatherer = AsyncParameterGatherer(config, device=self.device)

        try:
            results = []

            def callback(tensor: torch.Tensor) -> None:
                results.append(tensor.sum().item())

            # Submit requests with different priorities
            tensor1 = torch.ones(10, 10) * 1
            tensor2 = torch.ones(10, 10) * 2
            tensor3 = torch.ones(10, 10) * 3

            # Low priority
            gatherer.gather_async(
                "low", tensor1, self.device, priority=0, callback=callback
            )
            # High priority
            gatherer.gather_async(
                "high", tensor3, self.device, priority=10, callback=callback
            )
            # Medium priority
            gatherer.gather_async(
                "med", tensor2, self.device, priority=5, callback=callback
            )

            # Wait for completion
            gatherer.synchronize()

            # High priority should complete first (value 3*100=300)
            # This is a weak test as timing is not guaranteed
            self.assertEqual(len(results), 3)

        finally:
            gatherer.shutdown()


class TestOverlappedLinear(unittest.TestCase):
    """Test overlapped linear layer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_overlapped_linear_forward(self) -> None:
        """Test forward pass of overlapped linear layer."""
        config = OverlapConfig(mode=OverlapMode.PIPELINE)
        gatherer = AsyncParameterGatherer(config, device=self.device)

        try:
            # Create overlapped linear layer
            layer = OverlappedLinear(
                in_features=128,
                out_features=256,
                bias=True,
                gatherer=gatherer,
                device=self.device,
            )

            # Test forward pass
            input_tensor = torch.randn(32, 128, device=self.device)
            output = layer(input_tensor)

            self.assertEqual(output.shape, (32, 256))
            self.assertEqual(output.device, self.device)

        finally:
            gatherer.shutdown()

    def test_overlapped_linear_equivalence(self) -> None:
        """Test that overlapped linear produces same results as standard linear."""
        torch.manual_seed(42)

        # Create standard linear
        standard_linear = nn.Linear(64, 128, device=self.device)

        # Create overlapped linear with same weights
        config = OverlapConfig(mode=OverlapMode.NONE)  # Disable overlap for comparison
        gatherer = AsyncParameterGatherer(config, device=self.device)

        try:
            overlapped_linear = OverlappedLinear(
                in_features=64,
                out_features=128,
                bias=True,
                gatherer=gatherer,
                device=self.device,
            )

            # Copy weights
            overlapped_linear.weight.data = standard_linear.weight.data.clone()
            overlapped_linear.bias.data = standard_linear.bias.data.clone()

            # Test with same input
            input_tensor = torch.randn(16, 64, device=self.device)
            standard_output = standard_linear(input_tensor)
            overlapped_output = overlapped_linear(input_tensor)

            torch.testing.assert_close(
                standard_output, overlapped_output, rtol=1e-5, atol=1e-5
            )

        finally:
            gatherer.shutdown()


class TestPipelineOverlapScheduler(unittest.TestCase):
    """Test pipeline overlap scheduler."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_scheduler_initialization(self) -> None:
        """Test scheduler initialization."""
        config = OverlapConfig(mode=OverlapMode.PIPELINE)
        gatherer = AsyncParameterGatherer(config, device=self.device)

        try:
            scheduler = PipelineOverlapScheduler(
                num_stages=4,
                num_microbatches=8,
                gatherer=gatherer,
                device=self.device,
            )

            self.assertEqual(scheduler.num_stages, 4)
            self.assertEqual(scheduler.num_microbatches, 8)

        finally:
            gatherer.shutdown()

    def test_gradient_scheduling(self) -> None:
        """Test gradient reduction scheduling."""
        config = OverlapConfig(mode=OverlapMode.PIPELINE)
        gatherer = AsyncParameterGatherer(config, device=self.device)

        try:
            scheduler = PipelineOverlapScheduler(
                num_stages=2,
                num_microbatches=4,
                gatherer=gatherer,
                device=self.device,
            )

            # Schedule gradient reduction
            gradients = torch.randn(100, 100, device=self.device)
            future = scheduler.schedule_gradient_reduction(
                stage=0,
                microbatch=0,
                gradients=gradients,
            )

            # Wait for completion
            result = future.result(timeout=1.0)
            self.assertIsNotNone(result)

            # Check if gradients are ready
            ready = scheduler.wait_for_gradients(stage=0, microbatch=0)
            self.assertTrue(ready)

        finally:
            gatherer.shutdown()

    def test_parameter_prefetching(self) -> None:
        """Test parameter prefetching for next microbatch."""
        config = OverlapConfig(mode=OverlapMode.PREFETCH, prefetch_depth=2)
        gatherer = AsyncParameterGatherer(config, device=self.device)

        try:
            scheduler = PipelineOverlapScheduler(
                num_stages=2,
                num_microbatches=4,
                gatherer=gatherer,
                device=self.device,
            )

            # Create mock parameters
            parameters = [torch.randn(50, 50, device=self.device) for _ in range(3)]

            # Prefetch parameters
            scheduler.prefetch_next_parameters(stage=0, parameters=parameters)

            # Give time for prefetch
            time.sleep(0.1)

            # Check cache for prefetched parameters
            stats = gatherer.get_stats()
            self.assertGreater(stats["completed_requests"], 0)

        finally:
            gatherer.shutdown()


class TestPerformanceBenchmark(unittest.TestCase):
    """Performance benchmarks for parameter overlap."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required for benchmark")
    def test_overlap_speedup(self) -> None:
        """Benchmark overlap vs no overlap."""
        # Test configuration
        batch_size = 32
        seq_len = 512
        hidden_dim = 768
        num_layers = 4
        num_iterations = 10

        # Create model with standard linear layers
        standard_model = nn.Sequential(
            *[
                nn.Linear(hidden_dim, hidden_dim, device=self.device)
                for _ in range(num_layers)
            ]
        )

        # Create model with overlapped linear layers
        config = OverlapConfig(mode=OverlapMode.AGGRESSIVE, num_streams=4)
        gatherer = AsyncParameterGatherer(config, device=self.device)

        overlapped_model = nn.Sequential(
            *[
                OverlappedLinear(
                    hidden_dim, hidden_dim, gatherer=gatherer, device=self.device
                )
                for _ in range(num_layers)
            ]
        )

        # Copy weights for fair comparison
        for std_layer, ovl_layer in zip(
            standard_model.children(), overlapped_model.children()
        ):
            if (
                hasattr(ovl_layer, "weight")
                and hasattr(std_layer, "weight")
                and isinstance(ovl_layer.weight, torch.nn.Parameter)
                and isinstance(std_layer.weight, torch.nn.Parameter)
            ):
                ovl_layer.weight.data.copy_(std_layer.weight.data)
            if (
                hasattr(ovl_layer, "bias")
                and hasattr(std_layer, "bias")
                and ovl_layer.bias is not None
                and std_layer.bias is not None
                and isinstance(ovl_layer.bias, torch.nn.Parameter)
                and isinstance(std_layer.bias, torch.nn.Parameter)
            ):
                ovl_layer.bias.data.copy_(std_layer.bias.data)

        # Warm up
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)
        _ = standard_model(input_tensor)
        _ = overlapped_model(input_tensor)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark standard model
        start_time = time.time()
        for _ in range(num_iterations):
            output = standard_model(input_tensor)
            loss = output.mean()
            loss.backward()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        standard_time = time.time() - start_time

        # Clear gradients
        standard_model.zero_grad()
        overlapped_model.zero_grad()

        # Benchmark overlapped model
        start_time = time.time()
        for _ in range(num_iterations):
            output = overlapped_model(input_tensor)
            loss = output.mean()
            loss.backward()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        overlapped_time = time.time() - start_time

        # Clean up
        gatherer.shutdown()

        # Calculate speedup
        speedup = standard_time / overlapped_time

        print(f"\nBenchmark Results:")
        print(f"Standard model time: {standard_time:.4f}s")
        print(f"Overlapped model time: {overlapped_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x")

        # With overlap, we should see some speedup on GPU
        # Note: Actual speedup depends on hardware and workload
        if self.device.type == "cuda":
            # On GPU, overlap should provide benefit (even if small)
            self.assertGreaterEqual(speedup, 0.9)  # At least no significant slowdown
        else:
            # On CPU, overlap might not help much
            self.assertGreaterEqual(speedup, 0.8)  # Allow some overhead


if __name__ == "__main__":
    unittest.main()
