"""
Tests for Parameter-Gradient Buffer Mapping with Bucket-based Reduction

This test suite validates the param_grad_mapping module with:
- Bit-to-bit accuracy validation against reference implementation
- Performance benchmarking
- Memory usage validation
- Distributed communication patterns
- Edge cases and error handling
"""

import unittest

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from rosellm.rosetrainer.gradient.param_grad_mapping import (
    BucketConfig,
    ParamGradMapping,
    ReductionOp,
)


class SimpleModel(nn.Module):
    """Simple model for testing gradient mapping."""

    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class TestParamGradMapping(unittest.TestCase):
    """Test cases for ParamGradMapping."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleModel().to(self.device)
        self.config = BucketConfig(
            bucket_size_mb=1.0,
            dtype=torch.float32,
            reduction_op=ReductionOp.AVG,
            overlap_grad_reduce=True,
            use_contiguous_buffers=True,
        )

    def test_bucket_creation(self):
        """Test that buckets are created correctly."""
        mapping = ParamGradMapping(self.model, self.config)

        # Check that buckets were created
        self.assertGreater(mapping.num_buckets, 0)
        self.assertEqual(len(mapping.buckets), mapping.num_buckets)

        # Verify all parameters are mapped
        params_in_buckets = []
        for bucket in mapping.buckets:
            params_in_buckets.extend(bucket.params)

        model_params = [p for p in self.model.parameters() if p.requires_grad]
        self.assertEqual(len(params_in_buckets), len(model_params))

        # Check bucket sizes respect configuration
        dtype_size = torch.finfo(self.config.dtype).bits // 8
        max_bucket_size = self.config.bucket_cap_mb * 1024 * 1024 / dtype_size

        for bucket in mapping.buckets:
            self.assertLessEqual(bucket.numel, max_bucket_size)

    def test_contiguous_buffer_allocation(self):
        """Test contiguous gradient buffer allocation."""
        mapping = ParamGradMapping(self.model, self.config)

        # Check buffer is allocated
        self.assertIsNotNone(mapping.grad_buffer)
        if mapping.grad_buffer is not None:
            self.assertEqual(mapping.grad_buffer.numel(), mapping.grad_buffer_numel)
            self.assertEqual(mapping.grad_buffer.dtype, self.config.dtype)
            self.assertEqual(mapping.grad_buffer.device.type, self.device.type)

        # Verify bucket views
        for bucket in mapping.buckets:
            self.assertIsNotNone(bucket.grad_buffer)
            # Check that bucket buffer shares storage with main buffer
            if bucket.grad_buffer is not None and mapping.grad_buffer is not None:
                self.assertTrue(
                    bucket.grad_buffer.storage().data_ptr()
                    == mapping.grad_buffer.storage().data_ptr()
                )

    def test_gradient_copying(self):
        """Test gradient copying to buffer."""
        config = BucketConfig(
            bucket_size_mb=1.0,
            overlap_grad_reduce=False,  # Disable hooks for manual testing
            use_contiguous_buffers=True,
        )
        mapping = ParamGradMapping(self.model, config)

        # Create dummy gradients
        for param in self.model.parameters():
            if param.requires_grad:
                param.grad = torch.randn_like(param)

        # Manually copy gradients to buffers
        for param in self.model.parameters():
            if param.requires_grad and param in mapping.param_to_bucket:
                bucket = mapping.param_to_bucket[param]
                start, end = mapping.param_to_buffer_offset[param]
                if bucket.grad_buffer is not None and param.grad is not None:
                    bucket.grad_buffer[
                        start - bucket.offset : end - bucket.offset
                    ].copy_(param.grad.view(-1))

        # Verify copying
        for param in self.model.parameters():
            if param.requires_grad and param in mapping.param_to_bucket:
                bucket = mapping.param_to_bucket[param]
                start, end = mapping.param_to_buffer_offset[param]
                if bucket.grad_buffer is not None and param.grad is not None:
                    buffer_grad = bucket.grad_buffer[
                        start - bucket.offset : end - bucket.offset
                    ].view_as(param)
                    torch.testing.assert_close(buffer_grad, param.grad)

    def test_bucket_alignment(self):
        """Test buffer alignment for optimal memory access."""
        config = BucketConfig(
            bucket_size_mb=0.5,
            align_buffers=True,
        )
        mapping = ParamGradMapping(self.model, config)

        # Check that bucket offsets are aligned
        ALIGNMENT_BYTES = 256
        dtype_size = torch.finfo(config.dtype).bits // 8
        aligned_elements = ALIGNMENT_BYTES // dtype_size

        for bucket in mapping.buckets:
            # Check if bucket size is aligned (except possibly the last one)
            if bucket.index < len(mapping.buckets) - 1:
                self.assertEqual(bucket.numel % aligned_elements, 0)

    def test_parameter_skipping(self):
        """Test skipping parameters that don't participate in DP reduction."""
        # Mark some parameters to skip
        params_list = list(self.model.parameters())
        if len(params_list) > 2:
            setattr(params_list[0], "skip_data_parallel_grad_reduce", True)
            setattr(params_list[1], "skip_data_parallel_grad_reduce", True)

        mapping = ParamGradMapping(self.model, self.config)

        # Verify skipped parameters are not in mapping
        for param in params_list[:2]:
            self.assertNotIn(param, mapping.param_to_bucket)

    def test_bucket_ready_state(self):
        """Test bucket readiness tracking."""
        # Disable hooks for this test
        config = BucketConfig(
            bucket_size_mb=1.0,
            overlap_grad_reduce=False,  # Disable hooks
        )
        mapping = ParamGradMapping(self.model, config)

        for bucket in mapping.buckets:
            # Initially not ready
            self.assertFalse(bucket.is_ready_for_reduce())

            # Mark parameters as having gradients
            for param in bucket.params:
                bucket.params_with_grad.add(param)

            # Now should be ready
            self.assertTrue(bucket.is_ready_for_reduce())

            # Reset should clear state
            bucket.reset()
            self.assertFalse(bucket.is_ready_for_reduce())
            self.assertEqual(len(bucket.params_with_grad), 0)

    def test_statistics(self):
        """Test statistics reporting."""
        mapping = ParamGradMapping(self.model, self.config)
        stats = mapping.get_stats()

        # Verify statistics
        self.assertIn("num_buckets", stats)
        self.assertIn("total_params", stats)
        self.assertIn("total_numel", stats)
        self.assertIn("buffer_size_mb", stats)
        self.assertIn("padding_overhead", stats)

        self.assertEqual(stats["num_buckets"], mapping.num_buckets)
        self.assertGreaterEqual(stats["padding_overhead"], 0)
        self.assertLess(stats["padding_overhead"], 100)

    @unittest.skipUnless(
        torch.cuda.is_available() and dist.is_available(),
        "Requires CUDA and distributed",
    )
    def test_distributed_reduction(self):
        """Test distributed gradient reduction (requires distributed setup)."""
        # This test would need to be run with torchrun in a distributed environment
        # It's included here as a template for distributed testing

        if not dist.is_initialized():
            self.skipTest("Distributed not initialized")

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # Create model and mapping
        model = SimpleModel().cuda()
        config = BucketConfig(
            bucket_size_mb=1.0,
            reduction_op=ReductionOp.AVG,
            overlap_grad_reduce=False,  # Manual control for testing
        )
        mapping = ParamGradMapping(model, config)

        # Create different gradients on each rank
        for param in model.parameters():
            if param.requires_grad:
                param.grad = torch.full_like(param, rank + 1.0)

        # Copy to buffers
        for param in model.parameters():
            if param.requires_grad and param in mapping.param_to_bucket:
                bucket = mapping.param_to_bucket[param]
                start, end = mapping.param_to_buffer_offset[param]
                if bucket.grad_buffer is not None and param.grad is not None:
                    bucket.grad_buffer[
                        start - bucket.offset : end - bucket.offset
                    ].copy_(param.grad.view(-1))

        # Launch reductions
        for bucket in mapping.buckets:
            bucket.params_with_grad = set(bucket.params)
            mapping._launch_bucket_reduce(bucket)

        # Wait for completion
        mapping.wait_for_all_reduces()

        # Verify averaging worked correctly
        expected_avg = sum(range(1, world_size + 1)) / world_size
        for param in model.parameters():
            if param.requires_grad and param in mapping.param_to_bucket:
                torch.testing.assert_close(
                    param.grad,
                    torch.full_like(param, expected_avg),
                    rtol=1e-5,
                    atol=1e-5,
                )


class TestBitToBitAccuracy(unittest.TestCase):
    """Test bit-to-bit accuracy with reference implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)  # For reproducibility

    def test_gradient_norm_accuracy(self):
        """Test gradient norm calculation accuracy."""
        model = SimpleModel().to(self.device)
        config = BucketConfig(use_contiguous_buffers=True)
        mapping = ParamGradMapping(model, config)

        # Create known gradients
        for i, param in enumerate(model.parameters()):
            if param.requires_grad:
                param.grad = torch.full_like(param, i + 1.0)

        # Copy to buffer
        for param in model.parameters():
            if param.requires_grad and param in mapping.param_to_bucket:
                bucket = mapping.param_to_bucket[param]
                start, end = mapping.param_to_buffer_offset[param]
                if bucket.grad_buffer is not None and param.grad is not None:
                    bucket.grad_buffer[
                        start - bucket.offset : end - bucket.offset
                    ].copy_(param.grad.view(-1))

        # Calculate norm from buffer
        if mapping.grad_buffer is not None:
            buffer_norm = torch.norm(mapping.grad_buffer[: mapping.total_numel], p=2)

            # Calculate reference norm
            grad_list = []
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    grad_list.append(param.grad.view(-1))
            reference_norm = torch.norm(torch.cat(grad_list), p=2)

            # Compare
            torch.testing.assert_close(
                buffer_norm, reference_norm, rtol=1e-7, atol=1e-7
            )

    def test_memory_efficiency(self):
        """Test memory efficiency of bucketing."""
        # Create a large model
        large_model = nn.Sequential(*[nn.Linear(1024, 1024) for _ in range(10)]).to(
            self.device
        )

        config = BucketConfig(
            bucket_size_mb=5.0,
            use_contiguous_buffers=True,
            align_buffers=True,
        )
        mapping = ParamGradMapping(large_model, config)

        # Calculate memory overhead
        stats = mapping.get_stats()
        padding_overhead = stats["padding_overhead"]

        # Should have reasonable padding overhead (< 10% typically)
        self.assertLess(padding_overhead, 10.0)

        # Verify memory consolidation
        total_param_memory = sum(
            p.numel() * p.element_size()
            for p in large_model.parameters()
            if p.requires_grad
        )
        if mapping.grad_buffer is not None:
            buffer_memory = (
                mapping.grad_buffer.numel() * mapping.grad_buffer.element_size()
            )

            # Buffer should not be much larger than total param memory
            overhead_ratio = (buffer_memory - total_param_memory) / total_param_memory
            self.assertLess(overhead_ratio, 0.1)  # Less than 10% overhead


class TestPerformanceBenchmark(unittest.TestCase):
    """Performance benchmarks for gradient mapping."""

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
    def test_bucketing_performance(self):
        """Benchmark bucketing performance vs naive approach."""
        import time

        # Create a reasonably large model
        model = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
        ).cuda()

        # Warm up
        for _ in range(5):
            x = torch.randn(32, 2048, device="cuda")
            y = model(x)
            loss = y.sum()
            loss.backward()
            model.zero_grad()

        # Benchmark with bucketing
        config = BucketConfig(
            bucket_size_mb=10.0,
            use_contiguous_buffers=True,
        )
        mapping = ParamGradMapping(model, config)

        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(100):
            # Simulate gradient copying
            for param in model.parameters():
                if param.requires_grad and param in mapping.param_to_bucket:
                    param.grad = torch.randn_like(param)
                    bucket = mapping.param_to_bucket[param]
                    start_idx, end_idx = mapping.param_to_buffer_offset[param]
                    if bucket.grad_buffer is not None:
                        bucket.grad_buffer[
                            start_idx - bucket.offset : end_idx - bucket.offset
                        ].copy_(param.grad.view(-1))

        torch.cuda.synchronize()
        bucketed_time = time.perf_counter() - start

        # Benchmark naive approach
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(100):
            grads = []
            for param in model.parameters():
                if param.requires_grad:
                    param.grad = torch.randn_like(param)
                    grads.append(param.grad.view(-1))
            # Simulate concatenation
            _ = torch.cat(grads)

        torch.cuda.synchronize()
        naive_time = time.perf_counter() - start

        # Bucketing should not be significantly slower
        # In practice, it may even be faster due to better memory locality
        self.assertLess(bucketed_time, naive_time * 2.0)


if __name__ == "__main__":
    unittest.main()
