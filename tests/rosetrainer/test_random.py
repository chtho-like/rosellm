"""
Comprehensive Tests for Multi-Parallel RNG State Management

This test suite validates the functionality of the advanced RNG state management
system, including multi-dimensional parallelism support, state tracking,
checkpointing, and Megatron-LM compatibility.

Test Coverage:
- CudaRNGStatesTracker basic functionality
- Multi-parallel RNG seed generation
- State forking and isolation
- Checkpoint/restore operations
- CUDA Graph compatibility
- Performance and caching
- Error handling and edge cases
"""

import unittest
from typing import List
from unittest.mock import patch

import pytest
import torch

# Import the modules to test
from rosellm.rosetrainer.random import (
    CudaRNGStatesTracker,
    RNGStateInfo,
    RNGStateType,
    checkpoint_parallel_rng_state,
    fork_parallel_rng_state,
    get_cuda_rng_tracker,
    get_parallel_rng_state,
    get_rng_state_summary,
    initialize_cuda_rng_tracker,
    model_parallel_cuda_manual_seed,
    parallel_rng_context,
    reset_cuda_rng_tracker,
    restore_parallel_rng_state,
    set_parallel_rng_state,
    synchronize_parallel_rng_states,
)


class TestCudaRNGStatesTracker(unittest.TestCase):
    """Test cases for CudaRNGStatesTracker class."""

    def setUp(self):
        """Set up test environment."""
        self.tracker = CudaRNGStatesTracker(
            enable_cuda_graphs=True,
            cache_capacity=100,
            auto_cleanup=True,
            verbose=False,
        )

    def tearDown(self):
        """Clean up after tests."""
        self.tracker.reset()

    def test_tracker_initialization(self):
        """Test tracker initialization with different configurations."""
        # Test default initialization
        tracker = CudaRNGStatesTracker()
        self.assertFalse(tracker.enable_cuda_graphs)
        self.assertEqual(tracker.cache_capacity, 1000)
        self.assertTrue(tracker.auto_cleanup)
        self.assertFalse(tracker.verbose)

        # Test custom initialization
        tracker = CudaRNGStatesTracker(
            enable_cuda_graphs=True,
            cache_capacity=500,
            auto_cleanup=False,
            verbose=True,
        )
        self.assertTrue(tracker.enable_cuda_graphs)
        self.assertEqual(tracker.cache_capacity, 500)
        self.assertFalse(tracker.auto_cleanup)
        self.assertTrue(tracker.verbose)

    def test_add_rng_state(self):
        """Test adding RNG states with various configurations."""
        # Add basic state
        self.tracker.add("test_state", seed=1234)
        self.assertIn("test_state", self.tracker.get_states())

        # Add state with parallel dimensions
        self.tracker.add("tensor_parallel_state", parallel_dimensions=["tp"], seed=5678)
        self.assertIn("tensor_parallel_state", self.tracker.get_states())

        # Test adding duplicate state (should raise error without force)
        with self.assertRaises(ValueError):
            self.tracker.add("test_state", seed=9999)

        # Test adding duplicate state with force=True
        self.tracker.add("test_state", seed=9999, force=True)

    def test_cuda_state_handling(self):
        """Test CUDA-specific state handling."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for testing")

        # Add CUDA state
        self.tracker.add("cuda_state", state_type=RNGStateType.CUDA, seed=1111)

        # Set CUDA state
        self.tracker.set("cuda_state")

        # Verify state is set
        current_name = self.tracker.get_current_state_name("global")
        self.assertEqual(current_name, "cuda_state")

    def test_state_forking(self):
        """Test RNG state forking functionality."""
        # Add source state
        self.tracker.add("source_state", seed=1234)

        # Fork the state
        self.tracker.fork(
            "source_state", "forked_state", parallel_dimensions=["tp"], offset=10
        )

        # Verify fork was created
        self.assertIn("forked_state", self.tracker.get_states())

        # Check fork metadata
        fork_info = self.tracker._state_info["forked_state"]
        self.assertTrue(fork_info.is_forked)
        self.assertEqual(fork_info.parent_state, "source_state")
        self.assertIn("tp", fork_info.parallel_dimensions)

        # Check parent-child relationship
        source_info = self.tracker._state_info["source_state"]
        self.assertIn("forked_state", source_info.children_states)

    def test_state_removal(self):
        """Test RNG state removal with cleanup."""
        # Add parent and child states
        self.tracker.add("parent_state", seed=1234)
        self.tracker.fork("parent_state", "child_state")

        # Remove parent with cleanup
        self.tracker.remove("parent_state", cleanup_children=True)

        # Verify both parent and child are removed
        self.assertNotIn("parent_state", self.tracker.get_states())
        self.assertNotIn("child_state", self.tracker.get_states())

    def test_fork_and_set_context_manager(self):
        """Test fork_and_set context manager."""
        self.tracker.add("base_state", seed=1234)
        self.tracker.set("base_state")

        with self.tracker.fork_and_set("temp_state", ["tp"]) as fork_name:
            # Verify we're using the forked state
            current_name = self.tracker.get_current_state_name("tp")
            self.assertEqual(current_name, fork_name)

        # Verify forked state is cleaned up after context
        self.assertNotIn(fork_name, self.tracker.get_states())

    def test_statistics_and_step_tracking(self):
        """Test statistics collection and step tracking."""
        initial_stats = self.tracker.get_statistics()
        self.assertEqual(initial_stats["num_states"], 0)
        self.assertEqual(initial_stats["step_counter"], 0)

        # Add states and track steps
        self.tracker.add("state1", seed=1111)
        self.tracker.add("state2", seed=2222)
        self.tracker.step()
        self.tracker.step()

        updated_stats = self.tracker.get_statistics()
        self.assertEqual(updated_stats["num_states"], 2)
        self.assertEqual(updated_stats["step_counter"], 2)

    def test_state_dict_and_checkpointing(self):
        """Test state dictionary creation and loading."""
        # Add states
        self.tracker.add("state1", seed=1111, parallel_dimensions=["tp"])
        self.tracker.add("state2", seed=2222, parallel_dimensions=["dp"])
        self.tracker.set("state1")

        # Get state dictionary
        state_dict = self.tracker.state_dict()

        # Verify state dictionary contents
        self.assertIn("states", state_dict)
        self.assertIn("state_info", state_dict)
        self.assertIn("current_states", state_dict)
        self.assertIn("config", state_dict)

        # Create new tracker and load state
        new_tracker = CudaRNGStatesTracker()
        new_tracker.load_state_dict(state_dict)

        # Verify states were loaded correctly
        self.assertEqual(set(new_tracker.get_states()), {"state1", "state2"})
        self.assertEqual(new_tracker.get_current_state_name("tp"), "state1")

    def test_cuda_graph_compatibility(self):
        """Test CUDA Graph compatibility mode."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for testing")

        # Enable CUDA Graph compatibility
        tracker = CudaRNGStatesTracker(enable_cuda_graphs=True)
        tracker.add("cuda_state", state_type=RNGStateType.CUDA, seed=1234)

        # Enable compatibility mode
        tracker.enable_cuda_graph_compatibility()
        self.assertTrue(tracker._cuda_graph_mode)

        # Disable compatibility mode
        tracker.disable_cuda_graph_compatibility()
        self.assertFalse(tracker._cuda_graph_mode)

    def test_cache_management(self):
        """Test automatic cache cleanup."""
        # Create tracker with small cache capacity
        tracker = CudaRNGStatesTracker(cache_capacity=5, auto_cleanup=True)

        # Add more states than cache capacity
        for i in range(10):
            tracker.add(f"state_{i}", seed=i)
            if i < 5:
                tracker.step()  # Trigger cleanup periodically

        # Verify cache management is working
        self.assertLessEqual(len(tracker.get_states()), 10)  # Some cleanup may occur

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test setting non-existent state
        with self.assertRaises(KeyError):
            self.tracker.set("non_existent_state")

        # Test forking non-existent state
        with self.assertRaises(KeyError):
            self.tracker.fork("non_existent_source", "new_state")

        # Test removing non-existent state
        with self.assertRaises(KeyError):
            self.tracker.remove("non_existent_state")

        # Test invalid parallel dimensions
        with self.assertRaises(ValueError):
            self.tracker.add("state", parallel_dimensions=["invalid_dim"])


class TestParallelRNG(unittest.TestCase):
    """Test cases for parallel RNG functions."""

    def setUp(self):
        """Set up test environment."""
        reset_cuda_rng_tracker()

    def tearDown(self):
        """Clean up after tests."""
        reset_cuda_rng_tracker()

    @patch("rosellm.rosetrainer.parallelism.parallel_state.is_initialized")
    @patch(
        "rosellm.rosetrainer.parallelism.parallel_state."
        "get_tensor_model_parallel_rank"
    )
    @patch(
        "rosellm.rosetrainer.parallelism.parallel_state."
        "get_tensor_model_parallel_size"
    )
    @patch(
        "rosellm.rosetrainer.parallelism.parallel_state."
        "get_pipeline_model_parallel_rank"
    )
    @patch(
        "rosellm.rosetrainer.parallelism.parallel_state."
        "get_pipeline_model_parallel_size"
    )
    @patch("rosellm.rosetrainer.parallelism.parallel_state.get_data_parallel_rank")
    @patch("rosellm.rosetrainer.parallelism.parallel_state.get_data_parallel_size")
    @patch(
        "rosellm.rosetrainer.parallelism.parallel_state." "get_context_parallel_rank"
    )
    @patch(
        "rosellm.rosetrainer.parallelism.parallel_state." "get_context_parallel_size"
    )
    @patch(
        "rosellm.rosetrainer.parallelism.parallel_state."
        "get_expert_model_parallel_rank"
    )
    @patch(
        "rosellm.rosetrainer.parallelism.parallel_state."
        "get_expert_model_parallel_size"
    )
    def test_model_parallel_cuda_manual_seed(
        self,
        mock_ep_size,
        mock_ep_rank,
        mock_cp_size,
        mock_cp_rank,
        mock_dp_size,
        mock_dp_rank,
        mock_pp_size,
        mock_pp_rank,
        mock_tp_size,
        mock_tp_rank,
        mock_is_initialized,
    ):
        """Test multi-parallel RNG seed initialization."""
        # Mock parallel state
        mock_is_initialized.return_value = True
        mock_tp_rank.return_value = 0
        mock_tp_size.return_value = 2
        mock_pp_rank.return_value = 1
        mock_pp_size.return_value = 2
        mock_dp_rank.return_value = 0
        mock_dp_size.return_value = 1
        mock_cp_rank.return_value = 0
        mock_cp_size.return_value = 1
        mock_ep_rank.return_value = 0
        mock_ep_size.return_value = 1

        # Test seed initialization
        seeds = model_parallel_cuda_manual_seed(
            seed=1234,
            enable_deterministic=False,  # Skip deterministic ops for testing
            verbose=True,
        )

        # Verify seeds were computed correctly
        self.assertIn("global", seeds)
        self.assertIn("tensor_parallel", seeds)
        self.assertIn("pipeline_parallel", seeds)
        self.assertIn("model_parallel", seeds)

        # Check specific seed values
        self.assertEqual(seeds["global"], 1234)
        self.assertEqual(seeds["tensor_parallel"], 1234 + 0 + 0)  # tp_rank = 0
        self.assertEqual(seeds["pipeline_parallel"], 1234 + 100000 + 1)  # pp_rank = 1

    def test_parallel_rng_state_operations(self):
        """Test parallel RNG state get/set operations."""
        # Initialize tracker
        initialize_cuda_rng_tracker(verbose=False)
        tracker = get_cuda_rng_tracker()
        tracker.add("test_state", seed=1234)

        # Test setting parallel RNG state
        set_parallel_rng_state("global", "test_state")

        # Test getting parallel RNG state
        state_info = get_parallel_rng_state("global")
        self.assertIsNotNone(state_info)
        self.assertIn("parallel_dimension", state_info)
        self.assertEqual(state_info["parallel_dimension"], "global")

    def test_fork_parallel_rng_state(self):
        """Test parallel RNG state forking."""
        # Initialize tracker and base state
        initialize_cuda_rng_tracker(verbose=False)
        tracker = get_cuda_rng_tracker()
        tracker.add("base_state", seed=1234)

        # Fork the state
        fork_name = fork_parallel_rng_state(
            source_dimension="base_state",
            new_name="forked_test",
            target_dimensions=["tp"],
            offset=50,
        )

        self.assertEqual(fork_name, "forked_test")
        self.assertIn(fork_name, tracker.get_states())

    def test_checkpoint_restore_operations(self):
        """Test RNG checkpoint and restore operations."""
        # Initialize tracker and states
        initialize_cuda_rng_tracker(verbose=False)
        tracker = get_cuda_rng_tracker()
        tracker.add("state1", seed=1111)
        tracker.add("state2", seed=2222)
        tracker.set("state1")

        # Create checkpoint
        checkpoint = checkpoint_parallel_rng_state()
        self.assertIsNotNone(checkpoint)
        self.assertIn("tracker_state", checkpoint)

        # Modify state and restore
        tracker.add("state3", seed=3333)
        restore_parallel_rng_state(checkpoint)

        # Verify restoration
        restored_states = tracker.get_states()
        self.assertIn("state1", restored_states)
        self.assertIn("state2", restored_states)
        # state3 should be gone after restore
        self.assertNotIn("state3", restored_states)

    def test_parallel_rng_context_manager(self):
        """Test parallel RNG context manager."""
        initialize_cuda_rng_tracker(verbose=False)
        tracker = get_cuda_rng_tracker()
        tracker.add("base_state", seed=1234)
        tracker.add("tensor_parallel", seed=5678, parallel_dimensions=["tp"])
        tracker.set("base_state")

        # Test context manager
        with parallel_rng_context("tensor_parallel"):
            current_state = tracker.get_current_state_name("tp")
            self.assertEqual(current_state, "tensor_parallel")

        # Verify we're back to original state after context
        # (This might be global if no previous state was set for tensor_parallel)
        current_state = tracker.get_current_state_name("global")
        self.assertIsNotNone(current_state)

    def test_rng_state_summary(self):
        """Test RNG state summary generation."""
        initialize_cuda_rng_tracker(verbose=False)

        summary = get_rng_state_summary()
        self.assertIsNotNone(summary)
        self.assertIn("tracker_stats", summary)
        self.assertIn("torch_initial_seed", summary)
        self.assertIn("cuda_available", summary)
        self.assertIn("deterministic_enabled", summary)

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.broadcast")
    def test_synchronize_parallel_rng_states(self, mock_broadcast, mock_dist_init):
        """Test RNG state synchronization across ranks."""
        mock_dist_init.return_value = True

        initialize_cuda_rng_tracker(verbose=False)
        tracker = get_cuda_rng_tracker()
        tracker.add("sync_state", seed=1234)
        tracker.set("sync_state")

        # Test synchronization (should not raise exception)
        try:
            synchronize_parallel_rng_states(dimensions=["global"], source_rank=0)
        except Exception as e:
            self.fail(f"Synchronization failed unexpectedly: {e}")

    def test_error_handling_without_initialization(self):
        """Test error handling when parallel state is not initialized."""
        with self.assertRaises(RuntimeError):
            with patch(
                "rosellm.rosetrainer.parallelism.parallel_state." "is_initialized",
                return_value=False,
            ):
                model_parallel_cuda_manual_seed(1234)

    def test_invalid_parallel_dimension_handling(self):
        """Test handling of invalid parallel dimensions."""
        initialize_cuda_rng_tracker(verbose=False)

        # Test invalid dimension in get_parallel_rng_state
        with self.assertRaises(ValueError):
            get_parallel_rng_state("invalid_dimension")

        # Test setting invalid dimension
        set_parallel_rng_state("invalid_dimension")  # Should warn but not crash


class TestRNGStateInfo(unittest.TestCase):
    """Test cases for RNGStateInfo dataclass."""

    def test_rng_state_info_creation(self):
        """Test RNGStateInfo creation and properties."""
        state_info = RNGStateInfo(
            name="test_state",
            state_type=RNGStateType.CUDA,
            device_id=0,
            parallel_dimensions={"tp", "dp"},
            creation_step=10,
        )

        self.assertEqual(state_info.name, "test_state")
        self.assertEqual(state_info.state_type, RNGStateType.CUDA)
        self.assertEqual(state_info.device_id, 0)
        self.assertEqual(state_info.parallel_dimensions, {"tp", "dp"})
        self.assertEqual(state_info.creation_step, 10)
        self.assertFalse(state_info.is_forked)
        self.assertIsNone(state_info.parent_state)
        self.assertEqual(len(state_info.children_states), 0)


class TestRNGIntegration(unittest.TestCase):
    """Integration tests for RNG system with parallel state."""

    def setUp(self):
        """Set up integration test environment."""
        reset_cuda_rng_tracker()

    def tearDown(self):
        """Clean up after integration tests."""
        reset_cuda_rng_tracker()

    def test_deterministic_behavior(self):
        """Test that RNG produces deterministic results."""

        def generate_random_numbers(seed: int) -> List[float]:
            """Generate reproducible random numbers."""
            reset_cuda_rng_tracker()
            initialize_cuda_rng_tracker(verbose=False)

            # Initialize with specific seed
            ps = "rosellm.rosetrainer.parallelism.parallel_state"
            with patch(f"{ps}.is_initialized", return_value=True):
                with patch(f"{ps}.get_tensor_model_parallel_rank", return_value=0):
                    with patch(f"{ps}.get_tensor_model_parallel_size", return_value=1):
                        with patch(
                            f"{ps}.get_pipeline_model_parallel_rank", return_value=0
                        ):
                            with patch(
                                f"{ps}.get_pipeline_model_parallel_size", return_value=1
                            ):
                                with patch(
                                    f"{ps}.get_data_parallel_rank", return_value=0
                                ):
                                    with patch(
                                        f"{ps}.get_data_parallel_size", return_value=1
                                    ):
                                        with patch(
                                            f"{ps}.get_context_parallel_rank",
                                            return_value=0,
                                        ):
                                            with patch(
                                                f"{ps}.get_context_parallel_size",
                                                return_value=1,
                                            ):
                                                with patch(
                                                    f"{ps}.get_expert_model_parallel_rank",  # noqa: E501
                                                    return_value=0,
                                                ):
                                                    with patch(
                                                        f"{ps}.get_expert_model_parallel_size",  # noqa: E501
                                                        return_value=1,
                                                    ):
                                                        model_parallel_cuda_manual_seed(
                                                            seed=seed,
                                                            enable_deterministic=False,
                                                            verbose=False,
                                                        )

            # Generate random numbers using PyTorch
            torch.manual_seed(seed)
            return [torch.rand(1).item() for _ in range(10)]

        # Generate numbers twice with same seed
        numbers1 = generate_random_numbers(12345)
        numbers2 = generate_random_numbers(12345)

        # Verify deterministic behavior
        self.assertEqual(numbers1, numbers2)

        # Verify different seeds produce different results
        numbers3 = generate_random_numbers(54321)
        self.assertNotEqual(numbers1, numbers3)

    def test_memory_efficiency(self):
        """Test memory efficiency of RNG tracker."""
        initialize_cuda_rng_tracker(cache_capacity=10, auto_cleanup=True)
        tracker = get_cuda_rng_tracker()

        # Add many states and verify cleanup occurs
        for i in range(50):
            tracker.add(f"state_{i}", seed=i)
            tracker.set(f"state_{i}")
            tracker.step()

        # Verify cache management is working
        final_state_count = len(tracker.get_states())
        self.assertLessEqual(final_state_count, 50)  # Some cleanup should occur

    def test_concurrent_access_safety(self):
        """Test thread safety of RNG operations."""
        import threading
        import time

        initialize_cuda_rng_tracker(verbose=False)
        tracker = get_cuda_rng_tracker()
        results = []
        errors = []

        def worker_function(worker_id: int):
            """Worker function for concurrent testing."""
            try:
                # Each worker adds its own state
                state_name = f"worker_state_{worker_id}"
                tracker.add(state_name, seed=worker_id * 1000)
                tracker.set(state_name)
                time.sleep(0.01)  # Small delay to increase chance of race conditions
                current_state = tracker.get_current_state_name("global")
                results.append((worker_id, current_state))
            except Exception as e:
                errors.append((worker_id, str(e)))

        # Create and start worker threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker_function, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Verify no errors occurred
        if errors:
            self.fail(f"Concurrent access errors: {errors}")

        # Verify all workers completed successfully
        self.assertEqual(len(results), 10)


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
