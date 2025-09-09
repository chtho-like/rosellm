"""Performance timing utilities for distributed training.

This module provides comprehensive timing utilities compatible with Megatron-LM's
timing infrastructure, supporting CUDA synchronization, distributed aggregation,
and minimal overhead profiling.

Features:
    - Thread-safe timer operations with minimal lock contention
    - Zero-overhead when disabled via configuration
    - Memory-efficient statistics tracking with bounded history
    - Distributed aggregation with optimized tensor operations
    - CUDA and distributed synchronization support
    - Hierarchical timer categorization for organized reporting

Examples:
    Basic usage::

        timers = Timers(TimerConfig(enabled=True))
        with timers("forward-pass")():
            # Your forward pass code here
            pass

    Distributed training::

        config = TimerConfig(
            aggregation_method=TimerAggregation.MEAN,
            use_barrier=True
        )
        timers = Timers(config)
        # Timers will automatically aggregate across ranks
"""

import contextlib
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from .timer_config import TimerAggregation, TimerConfig, TimerLogLevel

# Constants
BYTES_PER_MB = 1024 * 1024
DEFAULT_PRECISION = 3
MIN_ELAPSED_TIME = 0.0
MAX_PRECISION = 10


class TimerError(Exception):
    """Base exception for timer-related errors."""

    pass


class TimerNotStartedError(TimerError):
    """Raised when attempting to stop a timer that wasn't started."""

    pass


class TimerAlreadyStartedError(TimerError):
    """Raised when attempting to start an already running timer."""

    pass


class TimerConfigurationError(TimerError):
    """Raised when timer configuration is invalid."""

    pass


class Timer:
    """Single timer for tracking elapsed time.

    Features:
    - CUDA synchronization support
    - Distributed barrier support
    - Memory tracking
    - Minimal overhead design
    """

    def __init__(
        self,
        name: str,
        synchronize_cuda: bool = True,
        use_barrier: bool = False,
        track_memory: bool = False,
        max_history: int = 10000,
    ) -> None:
        """Initialize timer.

        Args:
            name: Timer name for identification
            synchronize_cuda: Whether to sync CUDA before timing
            use_barrier: Whether to use distributed barrier
            track_memory: Whether to track GPU memory usage
            max_history: Maximum history entries to keep (for statistics)

        Raises:
            TimerConfigurationError: If max_history is negative
        """
        if max_history < 0:
            raise TimerConfigurationError(
                f"max_history must be non-negative, got {max_history}"
            )

        self.name = name
        self.synchronize_cuda = synchronize_cuda
        self.use_barrier = use_barrier
        self.track_memory = track_memory

        # Thread safety
        self._lock = threading.RLock()

        # Timing state
        self.start_time: Optional[float] = None
        self.elapsed_time: float = 0.0
        self.count: int = 0

        # History tracking with bounded deque
        self.history: Deque[float] = deque(maxlen=max_history)

        # Memory tracking
        self.start_memory: Optional[int] = None
        self.memory_used: int = 0
        self.peak_memory: int = 0

        # Statistics cache
        self._stats_cache: Optional[Dict[str, float]] = None
        self._stats_cache_count: int = -1

    def _sync_if_needed(self) -> None:
        """Synchronize CUDA and/or distributed if configured."""
        if self.synchronize_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        if self.use_barrier and dist.is_initialized():
            dist.barrier()

    def _get_memory(self) -> int:
        """Get current GPU memory usage."""
        if not self.track_memory or not torch.cuda.is_available():
            return 0
        return int(torch.cuda.memory_allocated())

    def start(self) -> None:
        """Start the timer.

        Raises:
            TimerAlreadyStartedError: If timer is already running
        """
        with self._lock:
            if self.start_time is not None:
                raise TimerAlreadyStartedError(
                    f"Timer '{self.name}' is already running. Call stop() first."
                )

            self._sync_if_needed()

            if self.track_memory:
                self.start_memory = self._get_memory()

            self.start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer and return elapsed time.

        Returns:
            Elapsed time in seconds

        Raises:
            TimerNotStartedError: If timer wasn't started
        """
        with self._lock:
            if self.start_time is None:
                raise TimerNotStartedError(
                    f"Timer '{self.name}' was not started. Call start() first."
                )

            self._sync_if_needed()

            elapsed = max(time.perf_counter() - self.start_time, MIN_ELAPSED_TIME)
            self.elapsed_time += elapsed
            self.count += 1
            self.history.append(elapsed)

            if self.track_memory and self.start_memory is not None:
                current_memory = self._get_memory()
                memory_delta = max(current_memory - self.start_memory, 0)
                self.memory_used += memory_delta
                self.peak_memory = max(self.peak_memory, current_memory)

            self.start_time = None
            self.start_memory = None

            # Invalidate cache
            self._stats_cache = None

            return elapsed

    def reset(self) -> None:
        """Reset the timer to initial state.

        This will clear all timing history and statistics.
        Safe to call even if timer is currently running.
        """
        with self._lock:
            self.start_time = None
            self.elapsed_time = 0.0
            self.count = 0
            self.history.clear()
            self.start_memory = None
            self.memory_used = 0
            self.peak_memory = 0
            self._stats_cache = None

    def get_stats(self) -> Dict[str, float]:
        """Get timer statistics.

        Returns:
            Dictionary containing:
                - count: Number of timing measurements
                - total: Total elapsed time in seconds
                - mean: Average time per measurement
                - min: Minimum time recorded
                - max: Maximum time recorded
                - last: Most recent measurement
                - std: Standard deviation of measurements
                - memory_used_mb: Total memory used (if tracking)
                - peak_memory_mb: Peak memory usage (if tracking)
        """
        with self._lock:
            # Use cache if valid
            if self._stats_cache is not None and self._stats_cache_count == self.count:
                return self._stats_cache.copy()

            stats: Dict[str, float] = {
                "count": float(self.count),
                "total": self.elapsed_time,
                "mean": self.elapsed_time / max(1, self.count),
            }

            if self.history:
                history_list = list(self.history)
                stats.update(
                    {
                        "min": min(history_list),
                        "max": max(history_list),
                        "last": history_list[-1],
                        "std": (
                            float(torch.std(torch.tensor(history_list)).item())
                            if len(history_list) > 1
                            else 0.0
                        ),
                    }
                )
            else:
                stats.update({"min": 0.0, "max": 0.0, "last": 0.0, "std": 0.0})

            if self.track_memory:
                stats.update(
                    {
                        "memory_used_mb": self.memory_used / BYTES_PER_MB,
                        "peak_memory_mb": self.peak_memory / BYTES_PER_MB,
                    }
                )

            # Cache results
            self._stats_cache = stats
            self._stats_cache_count = self.count

            return stats.copy()

    @contextlib.contextmanager
    def __call__(self) -> Iterator[None]:
        """Context manager for timing.

        Yields:
            None

        Raises:
            TimerAlreadyStartedError: If timer is already running
            TimerNotStartedError: If timer fails to start properly
        """
        try:
            self.start()
            yield
        finally:
            try:
                self.stop()
            except TimerNotStartedError:
                # Handle case where start failed
                pass


class Timers:
    """Collection of timers with log levels and distributed support.

    This class manages multiple timers and provides aggregation,
    logging, and reporting capabilities. Thread-safe for concurrent access.

    Attributes:
        config: Timer configuration settings
        timers: Dictionary of named timers
        global_step: Current global training step
        rank: Process rank in distributed training
        world_size: Total number of processes
    """

    def __init__(self, config: Optional[TimerConfig] = None) -> None:
        """Initialize timers collection.

        Args:
            config: Timer configuration (uses defaults if None)
        """
        self.config = config or TimerConfig()
        self.timers: Dict[str, Timer] = {}
        self.global_step: int = 0

        # Thread safety
        self._lock = threading.RLock()

        # Distributed state
        self.rank: int = dist.get_rank() if dist.is_initialized() else 0
        self.world_size: int = dist.get_world_size() if dist.is_initialized() else 1

        # Output buffer for batched logging
        self.output_buffer: List[str] = []

        # Pre-allocate tensors for distributed ops to reduce overhead
        self._aggregation_tensors: Optional[Dict[str, torch.Tensor]] = None
        if self.world_size > 1:
            self._initialize_aggregation_tensors()

    def _initialize_aggregation_tensors(self) -> None:
        """Pre-allocate tensors for efficient distributed aggregation."""
        # Pre-allocate reusable tensors to avoid repeated allocations
        self._aggregation_tensors = {
            "scalar": torch.zeros(1, dtype=torch.float32),
            "batch": torch.zeros(10, dtype=torch.float32),  # For batched operations
        }

    def _get_or_create_timer(self, name: str) -> Timer:
        """Get existing timer or create new one.

        Args:
            name: Timer name

        Returns:
            Timer instance
        """
        with self._lock:
            if name not in self.timers:
                self.timers[name] = Timer(
                    name=name,
                    synchronize_cuda=self.config.synchronize_cuda,
                    use_barrier=self.config.use_barrier,
                    track_memory=self.config.track_memory,
                    max_history=self.config.max_history,
                )
            return self.timers[name]

    def __call__(self, name: str) -> Union[Timer, "_NoOpTimer"]:
        """Get timer by name (creates if not exists).

        Args:
            name: Timer name

        Returns:
            Timer instance (can be used as context manager) or no-op timer if disabled
        """
        if not self.config.is_timer_enabled(name):
            # Return singleton no-op timer to avoid allocations
            return _NoOpTimer.get_instance()

        return self._get_or_create_timer(name)

    def start(self, name: str) -> None:
        """Start a timer.

        Args:
            name: Timer name

        Raises:
            TimerAlreadyStartedError: If timer is already running
        """
        if self.config.is_timer_enabled(name):
            self._get_or_create_timer(name).start()

    def stop(self, name: str) -> float:
        """Stop a timer.

        Args:
            name: Timer name

        Returns:
            Elapsed time in seconds (0.0 if timer disabled)

        Raises:
            TimerNotStartedError: If timer wasn't started
        """
        if not self.config.is_timer_enabled(name):
            return 0.0

        return self._get_or_create_timer(name).stop()

    def reset(self, name: Optional[str] = None) -> None:
        """Reset timer(s).

        Args:
            name: Timer name to reset (None = reset all)
        """
        with self._lock:
            if name is None:
                for timer in self.timers.values():
                    timer.reset()
            elif name in self.timers:
                self.timers[name].reset()

    def log(self, step: Optional[int] = None, reset: bool = False) -> None:
        """Log timer statistics.

        Args:
            step: Current step (for interval logging)
            reset: Whether to reset timers after logging

        Note:
            Only rank 0 prints output in distributed mode.
            Output is appended to file if config.output_file is set.
        """
        if step is not None:
            self.global_step = step

        if self.config.log_level == TimerLogLevel.OFF:
            return

        if not self.config.should_log(self.global_step):
            return

        try:
            # Aggregate statistics across ranks if distributed
            stats = self._aggregate_stats()

            # Format and log output
            output = self._format_output(stats)

            if self.rank == 0:
                print(output)

                if self.config.output_file:
                    try:
                        with open(self.config.output_file, "a", encoding="utf-8") as f:
                            f.write(output + "\n")
                    except IOError as e:
                        # Log warning but don't fail
                        print(f"Warning: Failed to write to timer log file: {e}")

            if reset:
                self.reset()

        except Exception as e:
            # Don't let timer logging crash the training
            if self.rank == 0:
                print(f"Warning: Timer logging failed: {e}")

    def _aggregate_stats(self) -> Dict[str, Dict[str, float]]:
        """Aggregate statistics across distributed ranks.

        Uses optimized batched tensor operations to minimize communication overhead.

        Returns:
            Aggregated statistics dictionary
        """
        with self._lock:
            local_stats = {
                name: timer.get_stats() for name, timer in self.timers.items()
            }

            if self.world_size == 1:
                return local_stats

            # Batch statistics for efficient aggregation
            return self._batched_aggregation(local_stats)

    def _batched_aggregation(
        self, local_stats: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Perform batched aggregation to reduce communication overhead.

        Args:
            local_stats: Local timer statistics

        Returns:
            Aggregated statistics
        """
        if not dist.is_initialized():
            return local_stats

        aggregated_stats: Dict[str, Dict[str, float]] = {}

        # Group statistics by aggregation type for batched operations
        sum_values: List[float] = []
        mean_values: List[float] = []
        min_values: List[float] = []
        max_values: List[float] = []

        stat_mapping: List[
            Tuple[str, str, str]
        ] = []  # (timer_name, stat_key, agg_type)

        for name, stats in local_stats.items():
            for key, value in stats.items():
                if key == "count" or key in ["total", "memory_used_mb"]:
                    sum_values.append(value)
                    stat_mapping.append((name, key, "sum"))
                elif key in ["mean", "last"]:
                    mean_values.append(value)
                    stat_mapping.append((name, key, "mean"))
                elif key == "min":
                    min_values.append(value)
                    stat_mapping.append((name, key, "min"))
                elif key in ["max", "peak_memory_mb"]:
                    max_values.append(value)
                    stat_mapping.append((name, key, "max"))
                else:
                    # Use default aggregation method
                    if self.config.aggregation_method == TimerAggregation.SUM:
                        sum_values.append(value)
                        stat_mapping.append((name, key, "sum"))
                    elif self.config.aggregation_method == TimerAggregation.MAX:
                        max_values.append(value)
                        stat_mapping.append((name, key, "max"))
                    elif self.config.aggregation_method == TimerAggregation.MIN:
                        min_values.append(value)
                        stat_mapping.append((name, key, "min"))
                    else:  # MEAN
                        mean_values.append(value)
                        stat_mapping.append((name, key, "mean"))

        # Perform batched all-reduce operations
        results: Dict[str, List[float]] = {}

        if sum_values:
            tensor = torch.tensor(sum_values, dtype=torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            results["sum"] = tensor.tolist()

        if mean_values:
            tensor = torch.tensor(mean_values, dtype=torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= self.world_size
            results["mean"] = tensor.tolist()

        if min_values:
            tensor = torch.tensor(min_values, dtype=torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
            results["min"] = tensor.tolist()

        if max_values:
            tensor = torch.tensor(max_values, dtype=torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            results["max"] = tensor.tolist()

        # Map results back to statistics
        counters: Dict[str, int] = {"sum": 0, "mean": 0, "min": 0, "max": 0}

        for timer_name, stat_key, agg_type in stat_mapping:
            if timer_name not in aggregated_stats:
                aggregated_stats[timer_name] = {}

            idx = counters[agg_type]
            aggregated_stats[timer_name][stat_key] = results[agg_type][idx]
            counters[agg_type] += 1

        return aggregated_stats

    def _format_output(self, stats: Dict[str, Dict[str, float]]) -> str:
        """Format statistics for output.

        Args:
            stats: Timer statistics

        Returns:
            Formatted string for display
        """
        if not stats:
            return "No timers recorded."

        lines: List[str] = [f"\n{'='*80}"]
        lines.append(f"Performance Timers (step {self.global_step})")
        lines.append(f"{'='*80}")

        # Group timers by category
        categories = defaultdict(list)
        for name in stats.keys():
            category = self.config.get_category(name)
            categories[category].append(name)

        # Sort categories
        category_order = [
            "forward",
            "backward",
            "optimizer",
            "data",
            "checkpoint",
            "misc",
        ]
        sorted_categories = [c for c in category_order if c in categories]
        sorted_categories.extend([c for c in categories if c not in category_order])

        for category in sorted_categories:
            if not categories[category]:
                continue

            lines.append(f"\n{category.upper()}:")
            lines.append("-" * 40)

            # Header
            if self.config.track_memory:
                header = (
                    f"{'Timer':<25} {'Count':>8} {'Mean':>10} "
                    f"{'Min':>10} {'Max':>10} {'Total':>10} {'Mem(MB)':>10}"
                )
                lines.append(header)
            else:
                header = (
                    f"{'Timer':<25} {'Count':>8} {'Mean':>10} "
                    f"{'Min':>10} {'Max':>10} {'Total':>10}"
                )
                lines.append(header)

            # Timer rows
            for name in sorted(categories[category]):
                s = stats[name]

                # Ensure precision is within bounds
                precision = min(max(self.config.precision, 0), MAX_PRECISION)

                row = f"{name:<25} {int(s.get('count', 0)):>8} "
                row += f"{s.get('mean', 0):.{precision}f}s".rjust(10) + " "
                row += f"{s.get('min', 0):.{precision}f}s".rjust(10) + " "
                row += f"{s.get('max', 0):.{precision}f}s".rjust(10) + " "
                row += f"{s.get('total', 0):.{precision}f}s".rjust(10)

                if self.config.track_memory and "memory_used_mb" in s:
                    row += f" {s['memory_used_mb']:.1f}".rjust(10)

                lines.append(row)

        # Summary
        precision = min(max(self.config.precision, 0), MAX_PRECISION)
        total_time = sum(s.get("total", 0) for s in stats.values())
        lines.append(f"\n{'='*80}")
        lines.append(f"Total time: {total_time:.{precision}f}s")

        if self.config.track_memory:
            total_memory = sum(s.get("memory_used_mb", 0) for s in stats.values())
            peak_memory = max(
                (s.get("peak_memory_mb", 0) for s in stats.values()), default=0
            )
            lines.append(
                f"Total memory: {total_memory:.1f} MB, Peak: {peak_memory:.1f} MB"
            )

        lines.append(f"{'='*80}")

        return "\n".join(lines)

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get all timer statistics.

        Returns:
            Dictionary mapping timer names to their statistics
        """
        return self._aggregate_stats()

    def summary(self) -> str:
        """Get formatted summary of all timers.

        Returns:
            Formatted summary string
        """
        stats = self._aggregate_stats()
        return self._format_output(stats)

    def write_summary(self, filename: str) -> None:
        """Write timer summary to file.

        Args:
            filename: Output filename

        Raises:
            IOError: If file cannot be written
        """
        try:
            summary = self.summary()
            with open(filename, "w", encoding="utf-8") as f:
                f.write(summary)

            if self.rank == 0:
                print(f"Timer summary written to {filename}")
        except IOError as e:
            raise IOError(f"Failed to write timer summary to {filename}: {e}") from e


class _NoOpTimer:
    """No-op timer for disabled timers.

    Uses singleton pattern to avoid repeated instantiation overhead.
    All methods are no-ops returning appropriate default values.
    """

    _instance: Optional["_NoOpTimer"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "_NoOpTimer":
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "_NoOpTimer":
        """Get singleton instance.

        Returns:
            Singleton _NoOpTimer instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start(self) -> None:
        """No-op start."""
        pass

    def stop(self) -> float:
        """No-op stop.

        Returns:
            Always returns 0.0
        """
        return 0.0

    def reset(self) -> None:
        """No-op reset."""
        pass

    def get_stats(self) -> Dict[str, float]:
        """Return empty statistics.

        Returns:
            Dictionary with all zero values
        """
        return {
            "count": 0.0,
            "total": 0.0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "last": 0.0,
            "std": 0.0,
        }

    @contextlib.contextmanager
    def __call__(self) -> Iterator[None]:
        """No-op context manager.

        Yields:
            None
        """
        yield


# Global timers instance (for compatibility with Megatron-LM)
_GLOBAL_TIMERS: Optional[Timers] = None


def get_timers() -> Timers:
    """Get global timers instance.

    Creates a default instance if none exists.

    Returns:
        Global Timers instance
    """
    global _GLOBAL_TIMERS
    if _GLOBAL_TIMERS is None:
        _GLOBAL_TIMERS = Timers()
    return _GLOBAL_TIMERS


def set_timers(timers: Timers) -> None:
    """Set global timers instance.

    Args:
        timers: Timers instance to set as global
    """
    global _GLOBAL_TIMERS
    _GLOBAL_TIMERS = timers


def reset_timers() -> None:
    """Reset all global timers.

    No-op if global timers haven't been initialized.
    """
    if _GLOBAL_TIMERS is not None:
        _GLOBAL_TIMERS.reset()


def log_timers(step: Optional[int] = None, reset: bool = False) -> None:
    """Log global timers.

    Args:
        step: Current step number
        reset: Whether to reset timers after logging

    Note:
        No-op if global timers haven't been initialized.
    """
    if _GLOBAL_TIMERS is not None:
        _GLOBAL_TIMERS.log(step, reset)
