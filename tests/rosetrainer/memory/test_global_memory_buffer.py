"""
Comprehensive unit tests for GlobalMemoryBuffer

This module tests the global memory buffer system that prevents dynamic allocations
by reusing pre-allocated buffers. Tests cover:
- Basic allocation and deallocation
- Memory pooling and growth
- Thread safety
- Integration with parallel operations
- Memory leak detection
- Performance benchmarks
"""

import threading
import time
from typing import List

import pytest
import torch

from rosellm.rosetrainer.memory.global_memory_buffer import (
    BufferConfig,
    BufferContext,
    BufferType,
    GlobalMemoryBuffer,
    MemoryPool,
    allocate_tensor,
    get_global_memory_buffer,
    initialize_global_memory_buffer,
    release_tensor,
)


class TestMemoryPool:
    """Test the MemoryPool class"""

    def test_pool_initialization(self):
        """Test basic pool initialization"""
        config = BufferConfig(activation_buffer_size=10)  # 10MB
        pool = MemoryPool(
            buffer_type=BufferType.ACTIVATION,
            initial_size_bytes=10 * 1024 * 1024,
            dtype=torch.float32,
            device=torch.device("cpu"),
            config=config,
        )

        assert pool.buffer_type == BufferType.ACTIVATION
        assert pool.dtype == torch.float32
        assert pool.device == torch.device("cpu")
        assert pool.total_size == 10 * 1024 * 1024
        assert len(pool.free_blocks) == 1
        assert pool.free_blocks[0] == (0, 10 * 1024 * 1024)

    def test_allocation_and_deallocation(self):
        """Test allocation and deallocation from pool"""
        config = BufferConfig(activation_buffer_size=10)
        pool = MemoryPool(
            buffer_type=BufferType.ACTIVATION,
            initial_size_bytes=10 * 1024 * 1024,
            dtype=torch.float32,
            device=torch.device("cpu"),
            config=config,
        )

        # Allocate 1MB
        allocation = pool.allocate(1024 * 1024, "test_allocation")
        assert allocation is not None
        assert allocation.size >= 1024 * 1024  # May be aligned
        assert allocation.offset == 0
        assert len(pool.allocations) == 1

        # Check free blocks updated
        assert len(pool.free_blocks) == 1
        assert pool.free_blocks[0][0] > 0  # Offset moved

        # Deallocate
        pool.deallocate(allocation)
        assert len(pool.allocations) == 0
        assert len(pool.free_blocks) == 1
        assert pool.free_blocks[0] == (0, 10 * 1024 * 1024)  # Full block free again

    def test_best_fit_allocation(self):
        """Test best-fit allocation strategy"""
        config = BufferConfig(
            activation_buffer_size=10, alignment=1
        )  # No alignment for simplicity
        pool = MemoryPool(
            buffer_type=BufferType.ACTIVATION,
            initial_size_bytes=10 * 1024 * 1024,
            dtype=torch.float32,
            device=torch.device("cpu"),
            config=config,
        )

        # Create fragmentation: allocate and deallocate to create gaps
        alloc1 = pool.allocate(2 * 1024 * 1024, "alloc1")  # 2MB  # noqa: F841
        alloc2 = pool.allocate(1 * 1024 * 1024, "alloc2")  # 1MB
        alloc3 = pool.allocate(3 * 1024 * 1024, "alloc3")  # 3MB

        if alloc2:
            pool.deallocate(alloc2)  # Creates 1MB gap
        if alloc3:
            pool.deallocate(alloc3)  # Creates 3MB gap

        # Now we have: [2MB used][8MB free] (blocks were coalesced)
        # Best fit for 2.5MB should use part of the 8MB block
        alloc4 = pool.allocate(int(2.5 * 1024 * 1024), "alloc4")
        assert alloc4 is not None

        # Check that we have the right fragmentation pattern
        # Should have [2MB used][2.5MB used][5.5MB free]
        assert len(pool.free_blocks) >= 1  # At least 1 free block

    def test_pool_growth(self):
        """Test automatic pool growth when needed"""
        config = BufferConfig(
            activation_buffer_size=1,  # Start with 1MB
            enable_pooling=True,
            pool_growth_factor=2.0,
            max_pool_size_mb=10,
        )
        pool = MemoryPool(
            buffer_type=BufferType.ACTIVATION,
            initial_size_bytes=1 * 1024 * 1024,
            dtype=torch.float32,
            device=torch.device("cpu"),
            config=config,
        )

        initial_size = pool.total_size

        # Try to allocate more than available
        large_alloc = pool.allocate(2 * 1024 * 1024, "large_allocation")

        assert large_alloc is not None
        assert pool.total_size > initial_size  # Pool should have grown
        assert pool.total_size >= 2 * 1024 * 1024  # At least enough for allocation

    def test_alignment(self):
        """Test memory alignment"""
        config = BufferConfig(alignment=512)
        pool = MemoryPool(
            buffer_type=BufferType.ACTIVATION,
            initial_size_bytes=10 * 1024 * 1024,
            dtype=torch.float32,
            device=torch.device("cpu"),
            config=config,
        )

        # Allocate unaligned size
        allocation = pool.allocate(1000, "unaligned")
        assert allocation is not None
        assert allocation.size % 512 == 0  # Size should be aligned

    def test_statistics(self):
        """Test pool statistics tracking"""
        config = BufferConfig()
        pool = MemoryPool(
            buffer_type=BufferType.ACTIVATION,
            initial_size_bytes=10 * 1024 * 1024,
            dtype=torch.float32,
            device=torch.device("cpu"),
            config=config,
        )

        # Make some allocations
        alloc1 = pool.allocate(1 * 1024 * 1024, "alloc1")
        alloc2 = pool.allocate(2 * 1024 * 1024, "alloc2")  # noqa: F841

        stats = pool.get_stats()
        assert stats["total_size_mb"] == 10.0
        assert stats["current_usage_mb"] > 0
        assert stats["num_allocations"] == 2
        assert stats["total_allocations"] == 2
        assert stats["total_deallocations"] == 0

        if alloc1:
            pool.deallocate(alloc1)
        stats = pool.get_stats()
        assert stats["total_deallocations"] == 1
        assert stats["num_allocations"] == 1


class TestGlobalMemoryBuffer:
    """Test the GlobalMemoryBuffer class"""

    def setup_method(self):
        """Reset global state before each test"""
        # Clear any existing global buffer
        if hasattr(GlobalMemoryBuffer, "_instance"):
            GlobalMemoryBuffer._instance = None

    def test_singleton_pattern(self):
        """Test that GlobalMemoryBuffer follows singleton pattern"""
        buffer1 = GlobalMemoryBuffer()
        buffer2 = GlobalMemoryBuffer()
        assert buffer1 is buffer2

    def test_buffer_allocation(self):
        """Test basic buffer allocation and release"""
        config = BufferConfig(activation_buffer_size=10)
        buffer = GlobalMemoryBuffer(config)

        # Allocate a tensor
        tensor = buffer.get_buffer(
            BufferType.ACTIVATION,
            (1024, 1024),
            torch.float32,
            torch.device("cpu"),
            "test_allocation",
        )

        assert tensor is not None
        assert tensor.shape == (1024, 1024)
        assert tensor.dtype == torch.float32

        # Release the tensor
        buffer.release_buffer(tensor)

    def test_different_buffer_types(self):
        """Test allocation from different buffer types"""
        config = BufferConfig(
            activation_buffer_size=10,
            gradient_buffer_size=5,
            communication_buffer_size=3,
        )
        buffer = GlobalMemoryBuffer(config)

        # Allocate from different pools
        act_tensor = buffer.get_buffer(
            BufferType.ACTIVATION, (512, 512), torch.float32, torch.device("cpu")
        )
        grad_tensor = buffer.get_buffer(
            BufferType.GRADIENT, (256, 256), torch.float32, torch.device("cpu")
        )
        comm_tensor = buffer.get_buffer(
            BufferType.COMMUNICATION, (128, 128), torch.float32, torch.device("cpu")
        )

        assert act_tensor.shape == (512, 512)
        assert grad_tensor.shape == (256, 256)
        assert comm_tensor.shape == (128, 128)

        # Release all
        buffer.release_buffer(act_tensor)
        buffer.release_buffer(grad_tensor)
        buffer.release_buffer(comm_tensor)

    def test_multiple_dtypes_and_devices(self):
        """Test allocation with different data types and devices"""
        config = BufferConfig()
        buffer = GlobalMemoryBuffer(config)

        # Different dtypes
        float_tensor = buffer.get_buffer(
            BufferType.TEMPORARY, 1024, torch.float32, torch.device("cpu")
        )
        int_tensor = buffer.get_buffer(
            BufferType.TEMPORARY, 1024, torch.int64, torch.device("cpu")
        )
        half_tensor = buffer.get_buffer(
            BufferType.TEMPORARY, 1024, torch.float16, torch.device("cpu")
        )

        assert float_tensor.dtype == torch.float32
        assert int_tensor.dtype == torch.int64
        assert half_tensor.dtype == torch.float16

        # Each should come from a different pool
        assert len(buffer.pools) == 3

        buffer.release_buffer(float_tensor)
        buffer.release_buffer(int_tensor)
        buffer.release_buffer(half_tensor)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_allocation(self):
        """Test allocation on GPU"""
        config = BufferConfig()
        buffer = GlobalMemoryBuffer(config)

        gpu_tensor = buffer.get_buffer(
            BufferType.ACTIVATION,
            (256, 256),
            torch.float32,
            torch.device("cuda:0"),
        )

        assert gpu_tensor.device.type == "cuda"
        assert gpu_tensor.shape == (256, 256)

        buffer.release_buffer(gpu_tensor)

    def test_memory_leak_detection(self):
        """Test memory leak detection"""
        config = BufferConfig(track_allocations=True, check_memory_leaks=True)
        buffer = GlobalMemoryBuffer(config)

        # Allocate without releasing
        tensor1 = buffer.get_buffer(
            BufferType.TEMPORARY, 1024, torch.float32, torch.device("cpu"), "leak1"
        )
        tensor2 = buffer.get_buffer(
            BufferType.TEMPORARY, 2048, torch.float32, torch.device("cpu"), "leak2"
        )

        # Check for leaks
        warnings = buffer.check_memory_leaks()
        assert len(warnings) > 0
        assert "unreleased allocations" in warnings[0]

        # Clean up
        buffer.release_buffer(tensor1)
        buffer.release_buffer(tensor2)

        warnings = buffer.check_memory_leaks()
        # Should have fewer warnings now
        assert len([w for w in warnings if "unreleased" in w]) == 0

    def test_reset(self):
        """Test resetting the buffer system"""
        config = BufferConfig()
        buffer = GlobalMemoryBuffer(config)

        # Allocate some tensors
        tensor1 = buffer.get_buffer(  # noqa: F841
            BufferType.ACTIVATION, 1024, torch.float32, torch.device("cpu")
        )
        tensor2 = buffer.get_buffer(  # noqa: F841
            BufferType.GRADIENT, 2048, torch.float32, torch.device("cpu")
        )

        assert len(buffer.pools) > 0
        assert len(buffer.allocation_tracking) > 0

        # Reset
        buffer.reset()

        assert len(buffer.pools) == 0
        assert len(buffer.allocation_tracking) == 0

    def test_statistics(self):
        """Test statistics collection"""
        config = BufferConfig()
        buffer = GlobalMemoryBuffer(config)

        # Make allocations
        tensors = []
        for i in range(5):
            t = buffer.get_buffer(
                BufferType.ACTIVATION,
                (256, 256),
                torch.float32,
                torch.device("cpu"),
                f"alloc_{i}",
            )
            tensors.append(t)

        stats = buffer.get_stats()
        assert len(stats) > 0

        # Check one pool's stats
        for key, pool_stats in stats.items():
            if "activation" in key:
                assert pool_stats["num_allocations"] == 5
                assert pool_stats["current_usage_mb"] > 0

        # Release some
        for t in tensors[:3]:
            buffer.release_buffer(t)

        stats = buffer.get_stats()
        for key, pool_stats in stats.items():
            if "activation" in key:
                assert pool_stats["num_allocations"] == 2  # 2 still allocated


class TestHelperFunctions:
    """Test helper functions and context managers"""

    def setup_method(self):
        """Reset global state before each test"""
        if hasattr(GlobalMemoryBuffer, "_instance"):
            GlobalMemoryBuffer._instance = None

    def test_initialize_global_memory_buffer(self):
        """Test global initialization function"""
        config = BufferConfig(activation_buffer_size=20)
        initialize_global_memory_buffer(config)

        buffer = get_global_memory_buffer()
        assert buffer is not None
        assert buffer.config.activation_buffer_size == 20

    def test_allocate_and_release_tensor(self):
        """Test convenience allocation functions"""
        initialize_global_memory_buffer()

        # Allocate
        tensor = allocate_tensor(
            (512, 512),
            dtype=torch.float32,
            device=torch.device("cpu"),
            buffer_type=BufferType.TEMPORARY,
        )

        assert tensor is not None
        assert tensor.shape == (512, 512)

        # Release
        release_tensor(tensor)

    def test_buffer_context_manager(self):
        """Test BufferContext context manager"""
        initialize_global_memory_buffer()

        # Use context manager
        with BufferContext(
            shape=(256, 256),
            dtype=torch.float32,
            device=torch.device("cpu"),
            buffer_type=BufferType.TEMPORARY,
        ) as tensor:
            assert tensor is not None
            assert tensor.shape == (256, 256)
            # Tensor should be valid here
            tensor.fill_(1.0)

        # Tensor should be released after exiting context
        buffer = get_global_memory_buffer()
        # Check that allocation was released
        stats = buffer.get_stats()
        for key, pool_stats in stats.items():
            if "temporary" in key:
                assert pool_stats["num_allocations"] == 0

    def test_buffer_context_with_exception(self):
        """Test that BufferContext releases even with exception"""
        initialize_global_memory_buffer()

        try:
            with BufferContext(
                shape=(128, 128),
                dtype=torch.float32,
            ) as tensor:
                assert tensor is not None
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Buffer should still be released
        buffer = get_global_memory_buffer()
        stats = buffer.get_stats()
        for key, pool_stats in stats.items():
            if "temporary" in key:
                assert pool_stats["num_allocations"] == 0


class TestThreadSafety:
    """Test thread safety of the buffer system"""

    def setup_method(self):
        """Reset global state before each test"""
        if hasattr(GlobalMemoryBuffer, "_instance"):
            GlobalMemoryBuffer._instance = None

    def test_concurrent_allocations(self):
        """Test concurrent allocations from multiple threads"""
        config = BufferConfig(activation_buffer_size=100)
        buffer = GlobalMemoryBuffer(config)

        num_threads = 10
        allocations_per_thread = 20
        all_tensors: List[List[torch.Tensor]] = [[] for _ in range(num_threads)]

        def allocate_tensors(thread_id: int):
            """Allocate tensors in a thread"""
            for i in range(allocations_per_thread):
                tensor = buffer.get_buffer(
                    BufferType.ACTIVATION,
                    (128, 128),
                    torch.float32,
                    torch.device("cpu"),
                    f"thread_{thread_id}_alloc_{i}",
                )
                all_tensors[thread_id].append(tensor)
                # Small delay to increase chance of contention
                time.sleep(0.001)

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=allocate_tensors, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Check all allocations succeeded
        total_tensors = sum(len(tensors) for tensors in all_tensors)
        assert total_tensors == num_threads * allocations_per_thread

        # Release all tensors
        for tensor_list in all_tensors:
            for tensor in tensor_list:
                buffer.release_buffer(tensor)

    def test_concurrent_alloc_dealloc(self):
        """Test concurrent allocation and deallocation"""
        config = BufferConfig(activation_buffer_size=50)
        buffer = GlobalMemoryBuffer(config)

        stop_flag = threading.Event()
        errors = []

        def alloc_dealloc_loop():
            """Continuously allocate and deallocate"""
            try:
                while not stop_flag.is_set():
                    tensor = buffer.get_buffer(
                        BufferType.ACTIVATION,
                        (64, 64),
                        torch.float32,
                        torch.device("cpu"),
                    )
                    time.sleep(0.001)
                    buffer.release_buffer(tensor)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=alloc_dealloc_loop)
            threads.append(t)
            t.start()

        # Let them run for a bit
        time.sleep(1.0)
        stop_flag.set()

        for t in threads:
            t.join()

        # Check no errors occurred
        assert len(errors) == 0


class TestIntegrationWithParallelState:
    """Test integration with parallel state management"""

    def test_initialization_with_parallel_state(self):
        """Test buffer initialization through parallel_state"""
        # This test would normally require distributed setup
        # For unit testing, we just verify the interface exists
        from rosellm.rosetrainer.parallelism.parallel_state import (
            get_global_memory_buffer,
            is_global_memory_buffer_initialized,
        )

        # Initially not initialized
        assert not is_global_memory_buffer_initialized()
        assert get_global_memory_buffer() is None

        # After manual initialization
        initialize_global_memory_buffer(BufferConfig())
        # Note: parallel_state won't know about this direct initialization
        # This would normally be done through initialize_model_parallel


class TestPerformanceBenchmarks:
    """Performance benchmarks for the buffer system"""

    @pytest.mark.skipif(
        "benchmark" not in dir(pytest), reason="pytest-benchmark not installed"
    )
    def test_allocation_performance(self, benchmark):
        """Benchmark allocation performance vs regular allocation"""
        config = BufferConfig(temporary_buffer_size=100)
        buffer = GlobalMemoryBuffer(config)

        def allocate_with_buffer():
            tensor = buffer.get_buffer(
                BufferType.TEMPORARY,
                (256, 256),
                torch.float32,
                torch.device("cpu"),
            )
            buffer.release_buffer(tensor)
            return tensor

        def allocate_regular():
            return torch.zeros((256, 256), dtype=torch.float32)

        # Warm up the buffer pool
        for _ in range(10):
            allocate_with_buffer()

        # Benchmark buffer allocation
        benchmark(allocate_with_buffer)  # Run benchmark

        # Compare with regular allocation
        # Note: This is just for reference, not part of the benchmark
        import timeit

        _ = timeit.timeit(allocate_regular, number=1000) / 1000  # Reference timing

        # Buffer allocation should be competitive after warm-up
        # (May be slightly slower due to bookkeeping, but reduces fragmentation)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(
        "benchmark" not in dir(pytest), reason="pytest-benchmark not installed"
    )
    def test_gpu_allocation_performance(self, benchmark):
        """Benchmark GPU allocation performance"""
        config = BufferConfig(activation_buffer_size=1024)  # 1GB
        buffer = GlobalMemoryBuffer(config)

        def allocate_gpu_buffer():
            tensor = buffer.get_buffer(
                BufferType.ACTIVATION,
                (1024, 1024),
                torch.float32,
                torch.device("cuda:0"),
            )
            buffer.release_buffer(tensor)
            return tensor

        # Warm up
        for _ in range(10):
            allocate_gpu_buffer()

        # Benchmark
        benchmark(allocate_gpu_buffer)  # Run GPU benchmark

        # GPU allocation through buffer should be fast after pool is warmed up

    def test_fragmentation_handling(self):
        """Test that buffer handles fragmentation well"""
        config = BufferConfig(
            temporary_buffer_size=10,
            warn_on_reallocation=False,
        )
        buffer = GlobalMemoryBuffer(config)

        # Create fragmentation pattern
        allocations = []
        for i in range(100):
            size = (i % 4 + 1) * 64 * 64  # Varying sizes
            tensor = buffer.get_buffer(
                BufferType.TEMPORARY,
                size,
                torch.float32,
                torch.device("cpu"),
            )
            allocations.append(tensor)

            # Release every other allocation to create gaps
            if i % 2 == 1 and len(allocations) > 2:
                if allocations[-2] is not None:
                    buffer.release_buffer(allocations[-2])
                    allocations[-2] = None

        # Clean up remaining allocations
        for tensor in allocations:
            if tensor is not None:
                buffer.release_buffer(tensor)

        # Check fragmentation stats
        stats = buffer.get_stats()
        for key, pool_stats in stats.items():
            if "temporary" in key:
                # Even with fragmentation, pool should handle it
                assert pool_stats["num_allocations"] == 0  # All released
                # Fragmentation metric shows number of free blocks
                print(f"Fragmentation after test: {pool_stats['fragmentation']} blocks")


class TestRealWorldScenarios:
    """Test real-world usage scenarios"""

    def test_gradient_accumulation_scenario(self):
        """Test buffer usage for gradient accumulation"""
        config = BufferConfig(
            gradient_buffer_size=50,
            activation_buffer_size=100,
        )
        buffer = GlobalMemoryBuffer(config)  # noqa: F841  # Initializes global buffer

        batch_size = 32
        hidden_size = 768
        num_accumulation_steps = 4

        # Simulate gradient accumulation
        accumulated_grad = None

        for step in range(num_accumulation_steps):
            # Get activation buffer for forward pass
            with BufferContext(
                shape=(batch_size, hidden_size),
                dtype=torch.float32,
                buffer_type=BufferType.ACTIVATION,
            ) as activation:
                # Simulate forward pass
                activation.normal_()

                # Get gradient buffer
                with BufferContext(
                    shape=(batch_size, hidden_size),
                    dtype=torch.float32,
                    buffer_type=BufferType.GRADIENT,
                ) as gradient:
                    # Simulate backward pass
                    gradient.copy_(activation * 0.1)  # Fake gradient

                    # Accumulate
                    if accumulated_grad is None:
                        accumulated_grad = gradient.clone()
                    else:
                        accumulated_grad.add_(gradient)

        # After accumulation, gradients would be applied
        assert accumulated_grad is not None
        assert accumulated_grad.shape == (batch_size, hidden_size)

    def test_activation_checkpointing_scenario(self):
        """Test buffer usage with activation checkpointing"""
        config = BufferConfig(
            activation_buffer_size=200,
            temporary_buffer_size=50,
        )
        buffer = GlobalMemoryBuffer(config)  # noqa: F841  # Initializes global buffer

        num_layers = 12
        batch_size = 16
        seq_len = 512
        hidden_size = 768

        # Simulate transformer layers with checkpointing
        checkpointed_activations = []

        for layer in range(num_layers):
            # Allocate activation buffer
            with BufferContext(
                shape=(batch_size, seq_len, hidden_size),
                dtype=torch.float16,  # Use FP16 for memory efficiency
                buffer_type=BufferType.ACTIVATION,
            ) as activation:
                # Simulate layer computation
                activation.normal_()

                # Checkpoint every 3rd layer
                if layer % 3 == 0:
                    # Store checkpointed activation (would normally save to CPU or disk)
                    checkpointed_activations.append(activation.clone())

        assert len(checkpointed_activations) == 4  # Layers 0, 3, 6, 9

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="Multi-GPU not available",
    )
    def test_multi_gpu_scenario(self):
        """Test buffer usage across multiple GPUs"""
        config = BufferConfig(
            communication_buffer_size=100,
            activation_buffer_size=200,
        )
        buffer = GlobalMemoryBuffer(config)

        # Allocate on different GPUs
        gpu0_tensor = buffer.get_buffer(
            BufferType.ACTIVATION,
            (256, 256),
            torch.float32,
            torch.device("cuda:0"),
        )

        gpu1_tensor = buffer.get_buffer(
            BufferType.ACTIVATION,
            (256, 256),
            torch.float32,
            torch.device("cuda:1"),
        )

        assert gpu0_tensor.device.index == 0
        assert gpu1_tensor.device.index == 1

        # Communication buffer for all-reduce
        comm_buffer = buffer.get_buffer(
            BufferType.COMMUNICATION,
            (256, 256),
            torch.float32,
            torch.device("cuda:0"),
        )

        # Simulate all-reduce operation
        comm_buffer.copy_(gpu0_tensor)
        # In real scenario, would do: dist.all_reduce(comm_buffer)

        # Clean up
        buffer.release_buffer(gpu0_tensor)
        buffer.release_buffer(gpu1_tensor)
        buffer.release_buffer(comm_buffer)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
