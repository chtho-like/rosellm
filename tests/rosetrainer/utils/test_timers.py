"""Comprehensive tests for the Timers system."""

import os
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from rosellm.rosetrainer.utils import (
    Timer,
    TimerConfig,
    TimerLogLevel,
    Timers,
    get_timers,
    log_timers,
    reset_timers,
    set_timers,
)


class TestTimer(unittest.TestCase):
    """Test individual Timer functionality."""

    def test_timer_basic_timing(self):
        """Test basic start/stop functionality."""
        timer = Timer("test", synchronize_cuda=False, use_barrier=False)

        # Start timer
        timer.start()
        time.sleep(0.1)
        elapsed = timer.stop()

        # Check timing
        self.assertGreater(elapsed, 0.09)
        self.assertLess(elapsed, 0.15)
        self.assertEqual(timer.count, 1)
        self.assertAlmostEqual(timer.elapsed_time, elapsed, places=5)

    def test_timer_multiple_runs(self):
        """Test accumulation over multiple runs."""
        timer = Timer("test", synchronize_cuda=False, use_barrier=False)

        total_elapsed = 0
        for _ in range(3):
            timer.start()
            time.sleep(0.05)
            elapsed = timer.stop()
            total_elapsed += elapsed

        self.assertEqual(timer.count, 3)
        self.assertAlmostEqual(timer.elapsed_time, total_elapsed, places=3)

        stats = timer.get_stats()
        self.assertEqual(stats["count"], 3)
        self.assertAlmostEqual(stats["total"], total_elapsed, places=3)
        self.assertAlmostEqual(stats["mean"], total_elapsed / 3, places=3)

    def test_timer_context_manager(self):
        """Test timer as context manager."""
        timer = Timer("test", synchronize_cuda=False, use_barrier=False)

        with timer():
            time.sleep(0.1)

        self.assertEqual(timer.count, 1)
        self.assertGreater(timer.elapsed_time, 0.09)

    def test_timer_reset(self):
        """Test timer reset functionality."""
        timer = Timer("test", synchronize_cuda=False, use_barrier=False)

        timer.start()
        time.sleep(0.05)
        timer.stop()

        self.assertEqual(timer.count, 1)
        self.assertGreater(timer.elapsed_time, 0)

        timer.reset()

        self.assertEqual(timer.count, 0)
        self.assertEqual(timer.elapsed_time, 0.0)
        self.assertEqual(len(timer.history), 0)

    def test_timer_statistics(self):
        """Test timer statistics calculation."""
        timer = Timer("test", synchronize_cuda=False, use_barrier=False)

        # Run timer with different durations
        durations = [0.01, 0.02, 0.03, 0.04, 0.05]
        for duration in durations:
            timer.start()
            time.sleep(duration)
            timer.stop()

        stats = timer.get_stats()

        self.assertEqual(stats["count"], 5)
        self.assertAlmostEqual(stats["mean"], stats["total"] / 5, places=5)
        self.assertLessEqual(stats["min"], 0.015)
        self.assertGreaterEqual(stats["max"], 0.045)

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.synchronize")
    def test_timer_cuda_sync(self, mock_sync, mock_available):
        """Test CUDA synchronization."""
        mock_available.return_value = True

        timer = Timer("test", synchronize_cuda=True, use_barrier=False)

        timer.start()
        timer.stop()

        # CUDA sync should be called twice (start and stop)
        self.assertEqual(mock_sync.call_count, 2)

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.barrier")
    def test_timer_distributed_barrier(self, mock_barrier, mock_initialized):
        """Test distributed barrier functionality."""
        mock_initialized.return_value = True

        timer = Timer("test", synchronize_cuda=False, use_barrier=True)

        timer.start()
        timer.stop()

        # Barrier should be called twice (start and stop)
        self.assertEqual(mock_barrier.call_count, 2)

    def test_timer_history_limit(self):
        """Test that history is bounded by max_history."""
        timer = Timer("test", synchronize_cuda=False, use_barrier=False, max_history=5)

        # Run timer 10 times
        for i in range(10):
            timer.start()
            time.sleep(0.01 * (i + 1))
            timer.stop()

        # History should only keep last 5
        self.assertEqual(len(timer.history), 5)
        self.assertEqual(timer.count, 10)


class TestTimers(unittest.TestCase):
    """Test Timers collection functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = TimerConfig(
            enabled=True,
            log_level=TimerLogLevel.INTERVAL,
            log_interval=10,
            synchronize_cuda=False,
            use_barrier=False,
        )

    def test_timers_creation_and_access(self):
        """Test creating and accessing timers."""
        timers = Timers(self.config)

        # Get or create timer
        timer1 = timers("test1")
        self.assertIsInstance(timer1, Timer)

        # Get same timer again
        timer2 = timers("test1")
        self.assertIs(timer1, timer2)

        # Create different timer
        timer3 = timers("test2")
        self.assertIsNot(timer1, timer3)

    def test_timers_start_stop(self):
        """Test start/stop methods."""
        timers = Timers(self.config)

        timers.start("test")
        time.sleep(0.05)
        elapsed = timers.stop("test")

        self.assertGreater(elapsed, 0.04)
        self.assertIn("test", timers.timers)

    def test_timers_disabled(self):
        """Test disabled timers."""
        config = TimerConfig(enabled=False)
        timers = Timers(config)

        timer = timers("test")
        timer.start()
        elapsed = timer.stop()

        # Should return no-op timer
        self.assertEqual(elapsed, 0.0)

    def test_timers_specific_enabled(self):
        """Test specific timer enabling."""
        config = TimerConfig(
            enabled=True,
            enabled_timers=["allowed"],
            synchronize_cuda=False,
        )
        timers = Timers(config)

        # Allowed timer should work
        allowed = timers("allowed")
        allowed.start()
        time.sleep(0.01)
        elapsed = allowed.stop()
        self.assertGreater(elapsed, 0)

        # Disallowed timer should be no-op
        disallowed = timers("notallowed")
        disallowed.start()
        time.sleep(0.01)
        elapsed = disallowed.stop()
        self.assertEqual(elapsed, 0.0)

    def test_timers_reset(self):
        """Test reset functionality."""
        timers = Timers(self.config)

        # Create and run some timers
        timers.start("test1")
        time.sleep(0.01)
        timers.stop("test1")

        timers.start("test2")
        time.sleep(0.01)
        timers.stop("test2")

        # Reset specific timer
        timers.reset("test1")
        self.assertEqual(timers.timers["test1"].count, 0)
        self.assertGreater(timers.timers["test2"].count, 0)

        # Reset all timers
        timers.reset()
        self.assertEqual(timers.timers["test2"].count, 0)

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    def test_timers_distributed_init(
        self, mock_world_size, mock_rank, mock_initialized
    ):
        """Test distributed initialization."""
        mock_initialized.return_value = True
        mock_rank.return_value = 2
        mock_world_size.return_value = 4

        timers = Timers(self.config)

        self.assertEqual(timers.rank, 2)
        self.assertEqual(timers.world_size, 4)

    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.all_reduce")
    def test_timers_aggregation(
        self, mock_all_reduce, mock_initialized, mock_rank, mock_world_size
    ):
        """Test distributed aggregation."""
        mock_initialized.return_value = True
        mock_rank.return_value = 0
        mock_world_size.return_value = 4

        # Mock all_reduce to simulate aggregation
        def all_reduce_side_effect(tensor, op):
            # Simulate summing across 4 ranks
            tensor.mul_(4)

        mock_all_reduce.side_effect = all_reduce_side_effect

        timers = Timers(self.config)

        # Create some timing data
        timer = timers.timers["test"] = Timer("test", synchronize_cuda=False)
        timer.elapsed_time = 1.0
        timer.count = 10
        timer.history.append(0.1)

        # Aggregate stats
        stats = timers._aggregate_stats()

        # Check aggregation results
        self.assertEqual(stats["test"]["count"], 40)  # Sum across ranks
        self.assertEqual(stats["test"]["total"], 4.0)  # Sum across ranks

    def test_timers_logging(self):
        """Test timer logging output."""
        timers = Timers(self.config)

        # Create some timers
        timer = timers("forward-compute")
        with timer():
            time.sleep(0.01)

        timer = timers("backward-compute")
        with timer():
            time.sleep(0.02)

        # Get summary
        summary = timers.summary()

        self.assertIn("forward-compute", summary)
        self.assertIn("backward-compute", summary)
        self.assertIn("Performance Timers", summary)

    def test_timers_categories(self):
        """Test timer categorization."""
        timers = Timers(self.config)

        # Test default categories
        self.assertEqual(timers.config.get_category("forward-compute"), "forward")
        self.assertEqual(timers.config.get_category("backward-comm"), "backward")
        self.assertEqual(timers.config.get_category("optimizer-step"), "optimizer")
        self.assertEqual(timers.config.get_category("unknown-timer"), "misc")

    def test_timers_output_file(self):
        """Test writing timer output to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "timers.txt")

            config = TimerConfig(
                enabled=True,
                output_file=output_file,
                synchronize_cuda=False,
            )
            timers = Timers(config)

            # Create some timings
            timer = timers("test")
            with timer():
                time.sleep(0.01)

            # Write summary
            timers.write_summary(output_file)

            # Check file exists and contains expected content
            self.assertTrue(os.path.exists(output_file))
            with open(output_file, "r") as f:
                content = f.read()
                self.assertIn("Performance Timers", content)
                self.assertIn("test", content)


class TestTimerConfig(unittest.TestCase):
    """Test TimerConfig functionality."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = TimerConfig()

        self.assertTrue(config.enabled)
        self.assertEqual(config.log_level, TimerLogLevel.INTERVAL)
        self.assertEqual(config.log_interval, 100)
        self.assertTrue(config.synchronize_cuda)
        self.assertFalse(config.use_barrier)

    def test_config_should_log(self):
        """Test should_log logic."""
        config = TimerConfig(
            log_level=TimerLogLevel.INTERVAL,
            log_interval=10,
            warmup_steps=5,
        )

        # Should not log during warmup
        self.assertFalse(config.should_log(3))

        # Should not log on non-interval steps
        self.assertFalse(config.should_log(7))

        # Should log on interval steps after warmup
        self.assertTrue(config.should_log(10))
        self.assertTrue(config.should_log(20))

        # Test with OFF level
        config.log_level = TimerLogLevel.OFF
        self.assertFalse(config.should_log(100))

        # Test with SUMMARY level
        config.log_level = TimerLogLevel.SUMMARY
        self.assertFalse(config.should_log(100))

    def test_config_timer_enabled(self):
        """Test is_timer_enabled logic."""
        # Test with all timers enabled
        config = TimerConfig(enabled=True)
        self.assertTrue(config.is_timer_enabled("any-timer"))

        # Test with disabled timers list
        config = TimerConfig(
            enabled=True,
            disabled_timers=["disabled1", "disabled2"],
        )
        self.assertTrue(config.is_timer_enabled("enabled"))
        self.assertFalse(config.is_timer_enabled("disabled1"))

        # Test with enabled timers list
        config = TimerConfig(
            enabled=True,
            enabled_timers=["allowed1", "allowed2"],
        )
        self.assertTrue(config.is_timer_enabled("allowed1"))
        self.assertFalse(config.is_timer_enabled("notallowed"))

        # Test with globally disabled
        config = TimerConfig(enabled=False)
        self.assertFalse(config.is_timer_enabled("any-timer"))


class TestGlobalTimers(unittest.TestCase):
    """Test global timer functions."""

    def test_global_timers_management(self):
        """Test global timers get/set/reset."""
        # Get default global timers
        timers1 = get_timers()
        self.assertIsInstance(timers1, Timers)

        # Get again - should be same instance
        timers2 = get_timers()
        self.assertIs(timers1, timers2)

        # Set new timers
        new_timers = Timers(TimerConfig(enabled=False))
        set_timers(new_timers)

        timers3 = get_timers()
        self.assertIs(timers3, new_timers)
        self.assertIsNot(timers3, timers1)

        # Reset timers
        reset_timers()
        # Should clear all timer data but keep same instance
        self.assertEqual(len(timers3.timers), 0)

    def test_global_log_timers(self):
        """Test global log_timers function."""
        # Set up a real timers instance with appropriate config
        config = TimerConfig(
            synchronize_cuda=False,
            log_level=TimerLogLevel.INTERVAL,
            log_interval=1,  # Log every step
            warmup_steps=0,  # No warmup
        )
        timers = Timers(config)
        set_timers(timers)

        # Add some timing data
        with timers("test")():
            time.sleep(0.01)

        # Should not raise any errors
        log_timers(step=1, reset=False)

        # Check timer exists and has data
        self.assertIn("test", timers.timers)
        self.assertEqual(timers.timers["test"].count, 1)

        # After reset, timer count should be 0
        log_timers(step=2, reset=True)
        self.assertEqual(timers.timers["test"].count, 0)


class TestTimerIntegration(unittest.TestCase):
    """Integration tests for timers with real-world scenarios."""

    def test_nested_timers(self):
        """Test nested timer usage."""
        timers = Timers(TimerConfig(synchronize_cuda=False))

        with timers("outer")():
            time.sleep(0.01)

            with timers("inner1")():
                time.sleep(0.01)

            with timers("inner2")():
                time.sleep(0.01)

        stats = timers.get_all_stats()

        # Outer should be approximately sum of all
        self.assertGreater(stats["outer"]["total"], 0.025)
        self.assertGreater(stats["inner1"]["total"], 0.008)
        self.assertGreater(stats["inner2"]["total"], 0.008)

    def test_training_simulation(self):
        """Simulate a training loop with timers."""
        config = TimerConfig(
            enabled=True,
            log_level=TimerLogLevel.INTERVAL,
            log_interval=5,
            warmup_steps=2,
            synchronize_cuda=False,
        )
        timers = Timers(config)

        # Simulate training steps
        for step in range(10):
            with timers("training-step")():
                # Forward pass
                with timers("forward-compute")():
                    time.sleep(0.001)

                # Backward pass
                with timers("backward-compute")():
                    time.sleep(0.002)

                # Optimizer step
                with timers("optimizer-step")():
                    time.sleep(0.001)

        stats = timers.get_all_stats()

        # Check all timers ran 10 times
        self.assertEqual(stats["training-step"]["count"], 10)
        self.assertEqual(stats["forward-compute"]["count"], 10)
        self.assertEqual(stats["backward-compute"]["count"], 10)
        self.assertEqual(stats["optimizer-step"]["count"], 10)

        # Check timing hierarchy
        self.assertGreater(
            stats["training-step"]["total"],
            stats["forward-compute"]["total"]
            + stats["backward-compute"]["total"]
            + stats["optimizer-step"]["total"],
        )

    def test_memory_tracking(self):
        """Test memory tracking functionality."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        config = TimerConfig(
            enabled=True,
            track_memory=True,
            synchronize_cuda=True,
        )
        timers = Timers(config)

        with timers("memory-test")():
            # Allocate some GPU memory
            x = torch.randn(1000, 1000, device="cuda")
            _ = x @ x.T  # Allocate memory for test

        stats = timers.get_all_stats()

        # Check memory tracking
        self.assertIn("memory_used_mb", stats["memory-test"])
        self.assertIn("peak_memory_mb", stats["memory-test"])
        self.assertGreater(stats["memory-test"]["memory_used_mb"], 0)


if __name__ == "__main__":
    unittest.main()
