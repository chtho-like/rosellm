"""Performance timing utilities for distributed training.

This module provides comprehensive timing utilities compatible with Megatron-LM's
timing infrastructure, supporting CUDA synchronization, distributed aggregation,
and minimal overhead profiling.
"""

import contextlib
import time
import warnings
from collections import defaultdict, deque
from typing import Dict, Iterator, List, Optional, Union

import torch
import torch.distributed as dist

from .timer_config import TimerAggregation, TimerConfig, TimerLogLevel


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
    ):
        """Initialize timer.

        Args:
            name: Timer name
            synchronize_cuda: Whether to sync CUDA before timing
            use_barrier: Whether to use distributed barrier
            track_memory: Whether to track memory usage
            max_history: Maximum history to keep
        """
        self.name = name
        self.synchronize_cuda = synchronize_cuda
        self.use_barrier = use_barrier
        self.track_memory = track_memory

        # Timing state
        self.start_time: Optional[float] = None
        self.elapsed_time: float = 0.0
        self.count: int = 0

        # History tracking with bounded deque
        self.history: deque = deque(maxlen=max_history)

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
        """Start the timer."""
        if self.start_time is not None:
            warnings.warn(f"Timer '{self.name}' already started")
            return

        self._sync_if_needed()

        if self.track_memory:
            self.start_memory = self._get_memory()

        self.start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer and return elapsed time.

        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            warnings.warn(f"Timer '{self.name}' not started")
            return 0.0

        self._sync_if_needed()

        elapsed = time.perf_counter() - self.start_time
        self.elapsed_time += elapsed
        self.count += 1
        self.history.append(elapsed)

        if self.track_memory and self.start_memory is not None:
            current_memory = self._get_memory()
            memory_delta = current_memory - self.start_memory
            self.memory_used += memory_delta
            self.peak_memory = max(self.peak_memory, current_memory)

        self.start_time = None
        self.start_memory = None

        # Invalidate cache
        self._stats_cache = None

        return elapsed

    def reset(self) -> None:
        """Reset the timer."""
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
            Dictionary with statistics
        """
        # Use cache if valid
        if self._stats_cache is not None and self._stats_cache_count == self.count:
            return self._stats_cache

        stats = {
            "count": self.count,
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
                    "std": torch.std(torch.tensor(history_list)).item()
                    if len(history_list) > 1
                    else 0.0,
                }
            )
        else:
            stats.update({"min": 0.0, "max": 0.0, "last": 0.0, "std": 0.0})

        if self.track_memory:
            stats.update(
                {
                    "memory_used_mb": self.memory_used / (1024 * 1024),
                    "peak_memory_mb": self.peak_memory / (1024 * 1024),
                }
            )

        # Cache results
        self._stats_cache = stats
        self._stats_cache_count = self.count

        return stats

    @contextlib.contextmanager
    def __call__(self) -> Iterator[None]:
        """Context manager for timing."""
        self.start()
        try:
            yield
        finally:
            self.stop()


class Timers:
    """Collection of timers with log levels and distributed support.

    This class manages multiple timers and provides aggregation,
    logging, and reporting capabilities.
    """

    def __init__(self, config: Optional[TimerConfig] = None):
        """Initialize timers collection.

        Args:
            config: Timer configuration
        """
        self.config = config or TimerConfig()
        self.timers: Dict[str, Timer] = {}
        self.global_step = 0

        # Distributed state
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Output buffer for batched logging
        self.output_buffer: List[str] = []

    def _get_or_create_timer(self, name: str) -> Timer:
        """Get existing timer or create new one."""
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
            Timer instance (can be used as context manager)
        """
        if not self.config.is_timer_enabled(name):
            # Return a no-op timer
            return _NoOpTimer()

        return self._get_or_create_timer(name)

    def start(self, name: str) -> None:
        """Start a timer.

        Args:
            name: Timer name
        """
        if self.config.is_timer_enabled(name):
            self._get_or_create_timer(name).start()

    def stop(self, name: str) -> float:
        """Stop a timer.

        Args:
            name: Timer name

        Returns:
            Elapsed time
        """
        if not self.config.is_timer_enabled(name):
            return 0.0

        return self._get_or_create_timer(name).stop()

    def reset(self, name: Optional[str] = None) -> None:
        """Reset timer(s).

        Args:
            name: Timer name to reset (None = reset all)
        """
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
        """
        if step is not None:
            self.global_step = step

        if self.config.log_level == TimerLogLevel.OFF:
            return

        if not self.config.should_log(self.global_step):
            return

        # Aggregate statistics across ranks if distributed
        stats = self._aggregate_stats()

        # Format and log output
        output = self._format_output(stats)

        if self.rank == 0:
            print(output)

            if self.config.output_file:
                with open(self.config.output_file, "a") as f:
                    f.write(output + "\n")

        if reset:
            self.reset()

    def _aggregate_stats(self) -> Dict[str, Dict[str, float]]:
        """Aggregate statistics across distributed ranks.

        Returns:
            Aggregated statistics
        """
        local_stats = {name: timer.get_stats() for name, timer in self.timers.items()}

        if self.world_size == 1:
            return local_stats

        # Prepare tensors for all-reduce
        aggregated_stats = {}

        for name, stats in local_stats.items():
            aggregated = {}

            for key, value in stats.items():
                if key == "count":
                    # Sum counts
                    tensor = torch.tensor([value], dtype=torch.float32)
                    if dist.is_initialized():
                        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                    aggregated[key] = tensor.item()

                elif key in ["total", "memory_used_mb"]:
                    # Sum totals
                    tensor = torch.tensor([value], dtype=torch.float32)
                    if dist.is_initialized():
                        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                    aggregated[key] = tensor.item()

                elif key in ["mean", "last"]:
                    # Average across ranks
                    tensor = torch.tensor([value], dtype=torch.float32)
                    if dist.is_initialized():
                        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                    aggregated[key] = tensor.item() / self.world_size

                elif key == "min":
                    # Global minimum
                    tensor = torch.tensor([value], dtype=torch.float32)
                    if dist.is_initialized():
                        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
                    aggregated[key] = tensor.item()

                elif key in ["max", "peak_memory_mb"]:
                    # Global maximum
                    tensor = torch.tensor([value], dtype=torch.float32)
                    if dist.is_initialized():
                        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
                    aggregated[key] = tensor.item()

                else:
                    # Default: use aggregation method from config
                    tensor = torch.tensor([value], dtype=torch.float32)
                    if dist.is_initialized():
                        if self.config.aggregation_method == TimerAggregation.SUM:
                            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                        elif self.config.aggregation_method == TimerAggregation.MAX:
                            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
                        elif self.config.aggregation_method == TimerAggregation.MIN:
                            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
                        else:  # MEAN
                            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                            tensor /= self.world_size
                    aggregated[key] = tensor.item()

            aggregated_stats[name] = aggregated

        return aggregated_stats

    def _format_output(self, stats: Dict[str, Dict[str, float]]) -> str:
        """Format statistics for output.

        Args:
            stats: Timer statistics

        Returns:
            Formatted string
        """
        lines = [f"\n{'='*80}"]
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
                precision = self.config.precision

                row = f"{name:<25} {int(s['count']):>8} "
                row += f"{s['mean']:.{precision}f}s".rjust(10) + " "
                row += f"{s['min']:.{precision}f}s".rjust(10) + " "
                row += f"{s['max']:.{precision}f}s".rjust(10) + " "
                row += f"{s['total']:.{precision}f}s".rjust(10)

                if self.config.track_memory and "memory_used_mb" in s:
                    row += f" {s['memory_used_mb']:.1f}".rjust(10)

                lines.append(row)

        # Summary
        total_time = sum(s["total"] for s in stats.values())
        lines.append(f"\n{'='*80}")
        lines.append(f"Total time: {total_time:.{self.config.precision}f}s")

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
            Dictionary of all timer statistics
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
        """
        summary = self.summary()
        with open(filename, "w") as f:
            f.write(summary)

        if self.rank == 0:
            print(f"Timer summary written to {filename}")


class _NoOpTimer:
    """No-op timer for disabled timers."""

    def start(self) -> None:
        pass

    def stop(self) -> float:
        return 0.0

    def reset(self) -> None:
        pass

    def get_stats(self) -> Dict[str, float]:
        return {"count": 0, "total": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}

    @contextlib.contextmanager
    def __call__(self) -> Iterator[None]:
        yield


# Global timers instance (for compatibility with Megatron-LM)
_GLOBAL_TIMERS: Optional[Timers] = None


def get_timers() -> Timers:
    """Get global timers instance.

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
    """Reset all global timers."""
    if _GLOBAL_TIMERS is not None:
        _GLOBAL_TIMERS.reset()


def log_timers(step: Optional[int] = None, reset: bool = False) -> None:
    """Log global timers.

    Args:
        step: Current step
        reset: Whether to reset after logging
    """
    if _GLOBAL_TIMERS is not None:
        _GLOBAL_TIMERS.log(step, reset)
