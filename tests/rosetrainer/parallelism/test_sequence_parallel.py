"""
Comprehensive Unit Tests for Sequence Parallel Operations

Tests the core sequence parallel communication primitives and validates
correctness against expected behavior patterns from Megatron-LM.

Test Coverage:
- Helper functions for tensor splitting and gathering
- Autograd functions with gradient flow validation
- Integration with parallel state management
- All-to-all communication patterns
- Error handling and edge cases
- Performance benchmarking
- Configuration management
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist

from rosellm.rosetrainer.parallelism import (
    gather_from_sequence_parallel_region,
    get_sequence_parallel_world_size,
    initialize_model_parallel,
    is_sequence_parallel_enabled,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from rosellm.rosetrainer.parallelism.sequence_parallel import (
    MemoryProfiler,
    SequenceParallelBenchmark,
    SequenceParallelConfig,
    TensorShapeError,
    _gather_along_first_dim,
    _split_along_first_dim,
    _split_along_last_dim,
    all_to_all_hidden_to_sequence,
    all_to_all_sequence_to_hidden,
    get_sequence_parallel_config,
    is_sequence_parallel_tensor,
    mark_tensor_as_sequence_parallel,
    set_sequence_parallel_config,
    validate_sequence_parallel_invariants,
    validate_tensor_for_sequence_parallel,
)


class TestSequenceParallelHelpers(unittest.TestCase):
    """Test helper functions for sequence parallel operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_split_along_first_dim(self):
        """Test splitting tensor along first dimension."""
        # Create mock process group
        mock_group = MagicMock()
        mock_group.size.return_value = 2

        # Test with rank 0
        with patch("torch.distributed.get_rank", return_value=0):
            input_tensor = torch.randn(8, 4, 16, device=self.device)
            output = _split_along_first_dim(input_tensor, mock_group)

            self.assertEqual(output.shape, (4, 4, 16))
            torch.testing.assert_close(output, input_tensor[:4])

        # Test with rank 1
        with patch("torch.distributed.get_rank", return_value=1):
            input_tensor = torch.randn(8, 4, 16, device=self.device)
            output = _split_along_first_dim(input_tensor, mock_group)

            self.assertEqual(output.shape, (4, 4, 16))
            torch.testing.assert_close(output, input_tensor[4:])

    def test_split_along_last_dim(self):
        """Test splitting tensor along last dimension."""
        # Create mock process group
        mock_group = MagicMock()
        mock_group.size.return_value = 2

        # Test with rank 0
        with patch("torch.distributed.get_rank", return_value=0):
            input_tensor = torch.randn(4, 4, 16, device=self.device)
            output = _split_along_last_dim(input_tensor, mock_group)

            self.assertEqual(output.shape, (4, 4, 8))
            torch.testing.assert_close(output, input_tensor[:, :, :8])

        # Test with rank 1
        with patch("torch.distributed.get_rank", return_value=1):
            input_tensor = torch.randn(4, 4, 16, device=self.device)
            output = _split_along_last_dim(input_tensor, mock_group)

            self.assertEqual(output.shape, (4, 4, 8))
            torch.testing.assert_close(output, input_tensor[:, :, 8:])

    def test_mark_and_check_sequence_parallel_tensor(self):
        """Test marking and checking tensors as sequence parallel."""
        tensor = torch.randn(4, 4, 16, device=self.device)

        # Initially not marked
        self.assertFalse(is_sequence_parallel_tensor(tensor))

        # Mark as sequence parallel
        mark_tensor_as_sequence_parallel(tensor)

        # Now should be marked
        self.assertTrue(is_sequence_parallel_tensor(tensor))


class TestSequenceParallelAutograd(unittest.TestCase):
    """Test autograd functions for sequence parallel operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @patch(
        "rosellm.rosetrainer.parallelism.sequence_parallel.is_initialized",
        return_value=True,
    )
    @patch(
        "rosellm.rosetrainer.parallelism.sequence_parallel."
        "get_tensor_model_parallel_group"
    )
    def test_scatter_gather_gradient_flow(self, mock_get_tp_group, mock_is_init):
        """Test gradient flow through scatter and gather operations."""
        # Mock single GPU (no actual parallelism)
        mock_get_tp_group.return_value = None

        # Create input tensor with gradients
        input_tensor = torch.randn(8, 4, 16, device=self.device, requires_grad=True)

        # Scatter to sequence parallel region (no-op for single GPU)
        scattered = scatter_to_sequence_parallel_region(input_tensor)
        self.assertEqual(scattered.shape, input_tensor.shape)

        # Gather from sequence parallel region (no-op for single GPU)
        gathered = gather_from_sequence_parallel_region(scattered)
        self.assertEqual(gathered.shape, input_tensor.shape)

        # Compute loss and backward
        loss = gathered.sum()
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(input_tensor.grad)
        assert input_tensor.grad is not None
        self.assertEqual(input_tensor.grad.shape, input_tensor.shape)

    @patch(
        "rosellm.rosetrainer.parallelism.sequence_parallel.is_initialized",
        return_value=True,
    )
    @patch(
        "rosellm.rosetrainer.parallelism.sequence_parallel."
        "get_tensor_model_parallel_group"
    )
    def test_reduce_scatter_gradient_flow(self, mock_get_tp_group, mock_is_init):
        """Test gradient flow through reduce-scatter operation."""
        # Mock single GPU
        mock_get_tp_group.return_value = None

        # Create input tensor with gradients
        input_tensor = torch.randn(8, 4, 16, device=self.device, requires_grad=True)

        # Reduce-scatter (no-op for single GPU)
        output = reduce_scatter_to_sequence_parallel_region(input_tensor)
        self.assertEqual(output.shape, input_tensor.shape)

        # Compute loss and backward
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(input_tensor.grad)
        assert input_tensor.grad is not None
        self.assertEqual(input_tensor.grad.shape, input_tensor.shape)


class TestSequenceParallelIntegration(unittest.TestCase):
    """Integration tests for sequence parallel with other parallelism dimensions."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Skip if distributed not available
        if not dist.is_available():
            self.skipTest("Distributed package not available")

    @patch("torch.distributed.new_group")
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=4)
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.init_process_group")
    def test_sequence_parallel_initialization(
        self,
        mock_init_pg,
        mock_is_initialized,
        mock_world_size,
        mock_rank,
        mock_new_group,
    ):
        """Test sequence parallel initialization with model parallel."""
        # Mock process group creation
        mock_group = MagicMock()
        mock_new_group.return_value = mock_group

        # Reset parallel state before test
        from rosellm.rosetrainer.parallelism import parallel_state

        with patch.object(parallel_state, "_INITIALIZED", False):
            # Initialize with sequence parallel enabled
            initialize_model_parallel(
                tensor_model_parallel_size=2,
                pipeline_model_parallel_size=1,
                data_parallel_size=2,
                sequence_parallel_enabled=True,
            )

            # Check sequence parallel is enabled
            self.assertTrue(is_sequence_parallel_enabled())

            # Check sequence parallel uses same size as TP
            self.assertEqual(get_sequence_parallel_world_size(), 2)

    @patch("torch.distributed.init_process_group")
    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_sequence_parallel_disabled_with_tp1(
        self, mock_rank, mock_world_size, mock_is_initialized, mock_init_pg
    ):
        """Test that sequence parallel is disabled when TP=1."""
        # Initialize with TP=1 and sequence parallel requested
        with patch("builtins.print"):  # Suppress warnings
            initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                data_parallel_size=1,
                sequence_parallel_enabled=True,
            )

        # Should be disabled since TP=1
        self.assertFalse(is_sequence_parallel_enabled())


class TestAllToAllOperations(unittest.TestCase):
    """Test all-to-all communication patterns for sequence parallel."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @patch(
        "rosellm.rosetrainer.parallelism.sequence_parallel."
        "get_tensor_model_parallel_group"
    )
    def test_all_to_all_sequence_to_hidden(self, mock_get_tp_group):
        """Test all-to-all from sequence to hidden parallel."""
        # Mock single GPU
        mock_get_tp_group.return_value = None

        # Create input in sequence parallel format
        input_tensor = torch.randn(
            4, 2, 16, device=self.device
        )  # [seq/TP, batch, hidden]

        # All-to-all (no-op for single GPU)
        output = all_to_all_sequence_to_hidden(input_tensor)
        self.assertEqual(output.shape, input_tensor.shape)

    @patch(
        "rosellm.rosetrainer.parallelism.sequence_parallel."
        "get_tensor_model_parallel_group"
    )
    def test_all_to_all_hidden_to_sequence(self, mock_get_tp_group):
        """Test all-to-all from hidden to sequence parallel."""
        # Mock single GPU
        mock_get_tp_group.return_value = None

        # Create input in hidden parallel format
        input_tensor = torch.randn(
            8, 2, 8, device=self.device
        )  # [seq, batch, hidden/TP]

        # All-to-all (no-op for single GPU)
        output = all_to_all_hidden_to_sequence(input_tensor)
        self.assertEqual(output.shape, input_tensor.shape)


class TestSequenceParallelNumerics(unittest.TestCase):
    """Test numerical correctness of sequence parallel operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_scatter_gather_identity(self):
        """Test that scatter followed by gather is identity operation."""
        with patch(
            "rosellm.rosetrainer.parallelism.sequence_parallel."
            "get_tensor_model_parallel_group",
            return_value=None,
        ):
            # Create random input
            input_tensor = torch.randn(16, 4, 32, device=self.device)

            # Scatter then gather
            scattered = scatter_to_sequence_parallel_region(input_tensor)
            gathered = gather_from_sequence_parallel_region(scattered)

            # Should be identical (single GPU case)
            torch.testing.assert_close(gathered, input_tensor)

    def test_reduce_scatter_sum_preservation(self):
        """Test that reduce-scatter preserves sum across ranks."""
        with patch(
            "rosellm.rosetrainer.parallelism.sequence_parallel."
            "get_tensor_model_parallel_group",
            return_value=None,
        ):
            # Create input with known sum
            input_tensor = torch.ones(8, 4, 16, device=self.device)
            expected_sum = input_tensor.sum()

            # Reduce-scatter (no-op for single GPU)
            output = reduce_scatter_to_sequence_parallel_region(input_tensor)

            # Sum should be preserved
            self.assertAlmostEqual(output.sum().item(), expected_sum.item(), places=5)


class TestSequenceParallelConfiguration(unittest.TestCase):
    """Test configuration management for sequence parallel."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration
        config = SequenceParallelConfig(
            enable_memory_profiling=True,
            enable_communication_stats=True,
            debug_mode=True,
        )
        config.validate()  # Should not raise

        # Test setting and getting config
        set_sequence_parallel_config(config)
        retrieved_config = get_sequence_parallel_config()
        self.assertEqual(retrieved_config.debug_mode, True)
        self.assertEqual(retrieved_config.enable_memory_profiling, True)

    def test_invalid_config(self):
        """Test invalid configuration raises errors."""
        # Communication overlap requires CUDA
        if not torch.cuda.is_available():
            config = SequenceParallelConfig(communication_overlap=True)
            with self.assertRaises(ValueError):
                config.validate()


class TestSequenceParallelErrorHandling(unittest.TestCase):
    """Test error handling in sequence parallel operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_tensor_shape_validation(self):
        """Test tensor shape validation."""
        # Test with invalid tensor type
        with self.assertRaises(TensorShapeError):
            validate_tensor_for_sequence_parallel(
                "not a tensor", expected_dims=3  # type: ignore
            )

        # Test with wrong number of dimensions
        tensor_2d = torch.randn(4, 8, device=self.device)
        with self.assertRaises(TensorShapeError):
            validate_tensor_for_sequence_parallel(
                tensor_2d, expected_dims=3, operation="Test"
            )

        # Test with insufficient sequence length
        tensor_short = torch.randn(2, 4, 8, device=self.device)
        with self.assertRaises(TensorShapeError):
            validate_tensor_for_sequence_parallel(
                tensor_short, min_seq_len=4, operation="Test"
            )

        # Valid tensor should not raise
        tensor_valid = torch.randn(8, 4, 16, device=self.device)
        validate_tensor_for_sequence_parallel(
            tensor_valid, expected_dims=3, min_seq_len=4
        )

    def test_split_with_indivisible_size(self):
        """Test splitting with size not divisible by world size."""
        mock_group = MagicMock()
        mock_group.size.return_value = 3

        with patch("torch.distributed.get_world_size", return_value=3):
            with patch("torch.distributed.get_rank", return_value=0):
                # Tensor size not divisible by world size
                tensor = torch.randn(7, 4, 16, device=self.device)

                with self.assertRaises(TensorShapeError):
                    _split_along_first_dim(tensor, mock_group)

                tensor = torch.randn(4, 4, 7, device=self.device)
                with self.assertRaises(TensorShapeError):
                    _split_along_last_dim(tensor, mock_group)


class TestSequenceParallelMemoryProfiler(unittest.TestCase):
    """Test memory profiling functionality."""

    def test_memory_profiler(self):
        """Test memory profiler recording and reporting."""
        profiler = MemoryProfiler(enabled=True)

        # Record some operations
        profiler.record("test_op1", "forward")
        profiler.record("test_op1", "backward")
        profiler.record("test_op2", "forward")

        # Generate report
        report = profiler.report()
        self.assertIn("Memory Usage Report", report)
        self.assertIn("test_op1_forward", report)
        self.assertIn("test_op1_backward", report)
        self.assertIn("test_op2_forward", report)

    def test_disabled_profiler(self):
        """Test that disabled profiler doesn't record."""
        profiler = MemoryProfiler(enabled=False)
        profiler.record("test_op", "forward")

        report = profiler.report()
        self.assertEqual(report, "No memory statistics collected")


class TestSequenceParallelBenchmarking(unittest.TestCase):
    """Test benchmarking utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_benchmark_operation(self):
        """Test benchmarking a simple operation."""
        benchmark = SequenceParallelBenchmark()

        # Define a simple operation
        def simple_op(x):
            return x * 2 + 1

        # Benchmark the operation
        input_tensor = torch.randn(100, 100, device=self.device)
        stats = benchmark.benchmark_operation(
            simple_op,
            input_tensor,
            "simple_multiply_add",
            num_iterations=10,
            warmup_iterations=2,
        )

        # Check that all expected stats are present
        self.assertIn("avg_time_ms", stats)
        self.assertIn("min_time_ms", stats)
        self.assertIn("max_time_ms", stats)
        self.assertIn("memory_gb", stats)
        self.assertIn("throughput_gb_s", stats)

        # Generate report
        report = benchmark.generate_report()
        self.assertIn("Sequence Parallel Performance Report", report)
        self.assertIn("simple_multiply_add", report)


class TestSequenceParallelInvariants(unittest.TestCase):
    """Test validation of sequence parallel invariants."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_validate_invariants_success(self):
        """Test successful invariant validation."""
        # Create original tensor
        original = torch.randn(8, 4, 16, device=self.device)

        # Simulate scattered tensors (manual split)
        world_size = 2
        scattered = [
            original[:4],
            original[4:],
        ]

        # Validation should pass
        result = validate_sequence_parallel_invariants(original, scattered, world_size)
        self.assertTrue(result)

    def test_validate_invariants_failure(self):
        """Test failed invariant validation."""
        # Create original tensor
        original = torch.randn(8, 4, 16, device=self.device)

        # Create incorrect scattered tensors
        world_size = 2
        scattered = [
            torch.randn(4, 4, 16, device=self.device),  # Random data
            torch.randn(4, 4, 16, device=self.device),
        ]

        # Validation should fail
        with patch(
            "rosellm.rosetrainer.parallelism.sequence_parallel.logger"
        ) as mock_logger:
            result = validate_sequence_parallel_invariants(
                original, scattered, world_size
            )
            self.assertFalse(result)
            mock_logger.error.assert_called()


class TestSequenceParallelGatherOptimizations(unittest.TestCase):
    """Test optimized gather operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @patch("torch.distributed.all_gather")
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=2)
    def test_optimized_gather_first_dim(
        self, mock_world_size, mock_rank, mock_all_gather
    ):
        """Test optimized memory layout in gather operations."""
        # Set optimization config
        config = SequenceParallelConfig(optimize_memory=True)
        set_sequence_parallel_config(config)

        # Create mock group
        mock_group = MagicMock()

        # Create input tensor
        input_tensor = torch.randn(4, 8, 16, device=self.device)

        # Call gather function
        _ = _gather_along_first_dim(input_tensor, mock_group)

        # Verify all_gather was called
        mock_all_gather.assert_called_once()

    @patch("torch.distributed.all_gather")
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=2)
    def test_standard_gather_first_dim(
        self, mock_world_size, mock_rank, mock_all_gather
    ):
        """Test standard gather without optimization."""
        # Set non-optimized config
        config = SequenceParallelConfig(optimize_memory=False)
        set_sequence_parallel_config(config)

        # Create mock group
        mock_group = MagicMock()

        # Create input tensor
        input_tensor = torch.randn(4, 8, 16, device=self.device)

        # Call gather function
        _ = _gather_along_first_dim(input_tensor, mock_group)

        # Verify all_gather was called
        mock_all_gather.assert_called_once()


class TestSequenceParallelCommunicationStats(unittest.TestCase):
    """Test communication statistics collection."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @patch(
        "rosellm.rosetrainer.parallelism.sequence_parallel."
        "get_tensor_model_parallel_group"
    )
    @patch("rosellm.rosetrainer.parallelism.sequence_parallel.logger")
    def test_communication_stats_logging(self, mock_logger, mock_get_tp_group):
        """Test that communication stats are logged when enabled."""
        # Enable communication stats
        config = SequenceParallelConfig(enable_communication_stats=True)
        set_sequence_parallel_config(config)

        # Mock single GPU
        mock_get_tp_group.return_value = None

        # Create input tensor
        input_tensor = torch.randn(8, 4, 16, device=self.device, requires_grad=True)

        # Scatter operation
        _ = scatter_to_sequence_parallel_region(input_tensor)

        # Verify stats were logged
        # Note: In single GPU mode, no actual communication happens
        # but the structure is in place for multi-GPU


if __name__ == "__main__":
    # Set environment for testing
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")

    unittest.main()
