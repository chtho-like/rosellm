"""
Performance Optimization Utilities for Distributed Checkpointing

This module provides advanced performance optimization utilities for
distributed activation checkpointing, including memory pooling,
prefetching, and performance monitoring.
"""

import functools
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Generator, List, Optional, Tuple

import torch

# import torch.distributed as dist  # Unused


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    checkpoint_time_ms: float = 0.0
    recompute_time_ms: float = 0.0
    memory_saved_mb: float = 0.0
    memory_peak_mb: float = 0.0
    communication_time_ms: float = 0.0
    total_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def average_checkpoint_time_ms(self) -> float:
        """Calculate average checkpoint time."""
        return (
            self.checkpoint_time_ms / self.total_operations
            if self.total_operations > 0
            else 0.0
        )

    def update(self, other: "PerformanceMetrics") -> None:
        """Update metrics with values from another instance."""
        self.checkpoint_time_ms += other.checkpoint_time_ms
        self.recompute_time_ms += other.recompute_time_ms
        self.memory_saved_mb += other.memory_saved_mb
        self.memory_peak_mb = max(self.memory_peak_mb, other.memory_peak_mb)
        self.communication_time_ms += other.communication_time_ms
        self.total_operations += other.total_operations
        self.cache_hits += other.cache_hits
        self.cache_misses += other.cache_misses

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "checkpoint_time_ms": self.checkpoint_time_ms,
            "recompute_time_ms": self.recompute_time_ms,
            "memory_saved_mb": self.memory_saved_mb,
            "memory_peak_mb": self.memory_peak_mb,
            "communication_time_ms": self.communication_time_ms,
            "total_operations": self.total_operations,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hit_rate,
            "avg_checkpoint_time_ms": self.average_checkpoint_time_ms,
        }


class MemoryPool:
    """Memory pool for efficient tensor allocation and reuse."""

    def __init__(
        self,
        initial_size_mb: float = 100.0,
        growth_factor: float = 1.5,
        max_size_mb: float = 1000.0,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize memory pool.

        Args:
            initial_size_mb: Initial pool size in MB
            growth_factor: Factor to grow pool when needed
            max_size_mb: Maximum pool size in MB
            device: Device for tensor allocation
        """
        self.initial_size_mb = initial_size_mb
        self.growth_factor = growth_factor
        self.max_size_mb = max_size_mb
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Pool state
        self.pool: List[torch.Tensor] = []
        self.free_list: Deque[torch.Tensor] = deque()
        self.allocated: Dict[int, torch.Tensor] = {}
        self.current_size_mb = 0.0

        # Statistics
        self.allocations = 0
        self.reuses = 0
        self.growths = 0

        # Initialize pool
        self._grow_pool(initial_size_mb)

    def _grow_pool(self, size_mb: float) -> None:
        """Grow the memory pool."""
        if self.current_size_mb + size_mb > self.max_size_mb:
            size_mb = self.max_size_mb - self.current_size_mb

        if size_mb <= 0:
            return

        # Allocate new buffer
        num_elements = int(size_mb * 1024 * 1024 / 4)  # Assuming float32
        buffer = torch.empty(num_elements, dtype=torch.float32, device=self.device)

        self.pool.append(buffer)
        self.free_list.append(buffer)
        self.current_size_mb += size_mb
        self.growths += 1

    def allocate(
        self, size: torch.Size, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Allocate tensor from pool.

        Args:
            size: Size of tensor to allocate
            dtype: Data type of tensor

        Returns:
            Allocated tensor
        """
        num_elements = torch.prod(torch.tensor(size)).item()
        bytes_needed = num_elements * torch.finfo(dtype).bits // 8

        # Try to reuse from free list
        for i, buffer in enumerate(self.free_list):
            if buffer.numel() * 4 >= bytes_needed:  # 4 bytes for float32
                self.free_list.remove(buffer)
                # Slice and reshape buffer
                tensor = buffer[: int(num_elements)].view(size).to(dtype)
                self.allocated[id(tensor)] = tensor
                self.reuses += 1
                return tensor

        # Need to grow pool
        grow_size = max(self.initial_size_mb, bytes_needed / (1024 * 1024))
        grow_size = min(
            grow_size * self.growth_factor, self.max_size_mb - self.current_size_mb
        )

        if grow_size > 0:
            self._grow_pool(grow_size)
            return self.allocate(size, dtype)  # Retry after growing

        # Fall back to regular allocation
        self.allocations += 1
        tensor = torch.empty(size, dtype=dtype, device=self.device)
        self.allocated[id(tensor)] = tensor
        return tensor

    def free(self, tensor: torch.Tensor) -> None:
        """
        Return tensor to pool.

        Args:
            tensor: Tensor to free
        """
        tensor_id = id(tensor)
        if tensor_id in self.allocated:
            del self.allocated[tensor_id]
            # Try to return to free list if it's from our pool
            for buffer in self.pool:
                if (
                    tensor.data_ptr() >= buffer.data_ptr()
                    and tensor.data_ptr() < buffer.data_ptr() + buffer.numel() * 4
                ):
                    # This tensor is from our pool
                    self.free_list.append(buffer)
                    break

    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "current_size_mb": self.current_size_mb,
            "max_size_mb": self.max_size_mb,
            "allocated_tensors": len(self.allocated),
            "free_buffers": len(self.free_list),
            "total_allocations": self.allocations,
            "total_reuses": self.reuses,
            "pool_growths": self.growths,
            "reuse_rate": (
                self.reuses / (self.allocations + self.reuses)
                if (self.allocations + self.reuses) > 0
                else 0.0
            ),
        }

    def clear(self) -> None:
        """Clear the memory pool."""
        self.pool.clear()
        self.free_list.clear()
        self.allocated.clear()
        self.current_size_mb = 0.0


class PerformanceMonitor:
    """Monitor and analyze checkpoint performance."""

    def __init__(self, window_size: int = 100):
        """
        Initialize performance monitor.

        Args:
            window_size: Size of sliding window for metrics
        """
        self.window_size = window_size
        self.metrics_history: Deque[PerformanceMetrics] = deque(maxlen=window_size)
        self.current_metrics = PerformanceMetrics()
        self.global_metrics = PerformanceMetrics()

    @contextmanager
    def measure_checkpoint(self) -> Generator[None, None, None]:
        """Context manager to measure checkpoint operation."""
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage_mb()

        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            memory_saved = start_memory - self._get_memory_usage_mb()

            self.current_metrics.checkpoint_time_ms += elapsed_ms
            self.current_metrics.memory_saved_mb += max(0, memory_saved)
            self.current_metrics.total_operations += 1

    @contextmanager
    def measure_recompute(self) -> Generator[None, None, None]:
        """Context manager to measure recompute operation."""
        start_time = time.perf_counter()

        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.current_metrics.recompute_time_ms += elapsed_ms

    @contextmanager
    def measure_communication(self) -> Generator[None, None, None]:
        """Context manager to measure communication operation."""
        start_time = time.perf_counter()

        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.current_metrics.communication_time_ms += elapsed_ms

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        self.current_metrics.cache_hits += 1

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        self.current_metrics.cache_misses += 1

    def commit_metrics(self) -> None:
        """Commit current metrics to history."""
        self.metrics_history.append(self.current_metrics)
        self.global_metrics.update(self.current_metrics)
        self.current_metrics = PerformanceMetrics()

    def get_recent_metrics(self) -> PerformanceMetrics:
        """Get aggregated recent metrics."""
        if not self.metrics_history:
            return PerformanceMetrics()

        aggregated = PerformanceMetrics()
        for metrics in self.metrics_history:
            aggregated.update(metrics)

        return aggregated

    def get_global_metrics(self) -> PerformanceMetrics:
        """Get global aggregated metrics."""
        return self.global_metrics

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return float(torch.cuda.memory_allocated() / (1024 * 1024))
        else:
            # CPU memory tracking is more complex, return 0 for now
            return 0.0

    def should_adjust_strategy(self, threshold: float = 0.8) -> bool:
        """
        Determine if strategy should be adjusted based on performance.

        Args:
            threshold: Performance threshold for adjustment

        Returns:
            True if strategy should be adjusted
        """
        recent = self.get_recent_metrics()

        # Check cache hit rate
        if recent.cache_hit_rate < threshold:
            return True

        # Check memory efficiency
        if recent.memory_saved_mb < 100:  # Less than 100MB saved
            return True

        # Check time efficiency
        checkpoint_ratio = recent.checkpoint_time_ms / (recent.recompute_time_ms + 1e-6)
        if checkpoint_ratio > 2.0:  # Checkpointing takes 2x more than recompute
            return True

        return False


def optimize_checkpoint_schedule(
    layers: List[str],
    memory_costs: List[float],
    compute_costs: List[float],
    memory_budget: float,
) -> List[bool]:
    """
    Optimize checkpoint schedule using dynamic programming.

    Args:
        layers: List of layer identifiers
        memory_costs: Memory cost for each layer in MB
        compute_costs: Compute cost for each layer (relative)
        memory_budget: Total memory budget in MB

    Returns:
        List of booleans indicating which layers to checkpoint
    """
    n = len(layers)
    if n == 0:
        return []

    # Dynamic programming approach
    # dp[i][m] = minimum compute cost for layers 0..i with memory budget m
    # We discretize memory budget for practical computation
    memory_steps = 100
    memory_unit = memory_budget / memory_steps

    # Initialize DP table
    dp = [[float("inf")] * (memory_steps + 1) for _ in range(n + 1)]
    dp[0][0] = 0

    # Track decisions
    decisions = [[False] * (memory_steps + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        layer_memory = int(memory_costs[i - 1] / memory_unit)
        layer_compute = compute_costs[i - 1]

        for m in range(memory_steps + 1):
            # Option 1: Don't checkpoint (save memory)
            if m >= layer_memory:
                cost_no_checkpoint = dp[i - 1][m - layer_memory]
                if cost_no_checkpoint < dp[i][m]:
                    dp[i][m] = cost_no_checkpoint
                    decisions[i][m] = False

            # Option 2: Checkpoint (pay compute cost)
            cost_checkpoint = dp[i - 1][m] + layer_compute
            if cost_checkpoint < dp[i][m]:
                dp[i][m] = cost_checkpoint
                decisions[i][m] = True

    # Find optimal solution
    best_memory = 0
    best_cost = float("inf")
    for m in range(memory_steps + 1):
        if dp[n][m] < best_cost:
            best_cost = dp[n][m]
            best_memory = m

    # Reconstruct solution
    checkpoint_decisions = []
    current_memory = best_memory

    for i in range(n, 0, -1):
        should_checkpoint = decisions[i][current_memory]
        checkpoint_decisions.append(should_checkpoint)

        if not should_checkpoint:
            layer_memory = int(memory_costs[i - 1] / memory_unit)
            current_memory -= layer_memory

    checkpoint_decisions.reverse()
    return checkpoint_decisions


class CheckpointPrefetcher:
    """Prefetch checkpoints for improved performance."""

    def __init__(self, prefetch_depth: int = 2):
        """
        Initialize checkpoint prefetcher.

        Args:
            prefetch_depth: Number of checkpoints to prefetch
        """
        self.prefetch_depth = prefetch_depth
        self.prefetch_queue: Deque[Tuple[str, torch.Tensor]] = deque(
            maxlen=prefetch_depth
        )
        self.pending_prefetches: Dict[str, torch.cuda.Stream] = {}

    def prefetch(
        self, checkpoint_id: str, checkpoint_fn: Callable[[], torch.Tensor]
    ) -> None:
        """
        Prefetch a checkpoint asynchronously.

        Args:
            checkpoint_id: Unique identifier for checkpoint
            checkpoint_fn: Function to generate checkpoint
        """
        if torch.cuda.is_available():
            # Create a new stream for prefetching
            stream = torch.cuda.Stream()

            with torch.cuda.stream(stream):  # type: ignore
                checkpoint = checkpoint_fn()
                self.prefetch_queue.append((checkpoint_id, checkpoint))

            self.pending_prefetches[checkpoint_id] = stream  # type: ignore

    def get(self, checkpoint_id: str) -> Optional[torch.Tensor]:
        """
        Get a prefetched checkpoint.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            Prefetched checkpoint tensor or None if not available
        """
        # Wait for prefetch to complete if pending
        if checkpoint_id in self.pending_prefetches:
            stream = self.pending_prefetches[checkpoint_id]
            torch.cuda.current_stream().wait_stream(stream)
            del self.pending_prefetches[checkpoint_id]

        # Search in prefetch queue
        for idx, (cid, checkpoint) in enumerate(self.prefetch_queue):
            if cid == checkpoint_id:
                # Remove and return
                self.prefetch_queue.remove((cid, checkpoint))
                return checkpoint

        return None

    def clear(self) -> None:
        """Clear prefetch queue and pending operations."""
        self.prefetch_queue.clear()

        # Wait for all pending prefetches
        if torch.cuda.is_available():
            for stream in self.pending_prefetches.values():
                torch.cuda.current_stream().wait_stream(stream)

        self.pending_prefetches.clear()


def profile_memory_usage(
    func: Callable, *args, **kwargs
) -> Tuple[Any, Dict[str, float]]:
    """
    Profile memory usage of a function.

    Args:
        func: Function to profile
        *args: Positional arguments for function
        **kwargs: Keyword arguments for function

    Returns:
        Tuple of (function result, memory statistics)
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start_memory = torch.cuda.memory_allocated()

        result = func(*args, **kwargs)

        torch.cuda.synchronize()

        end_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()

        stats = {
            "start_memory_mb": start_memory / (1024 * 1024),
            "end_memory_mb": end_memory / (1024 * 1024),
            "peak_memory_mb": peak_memory / (1024 * 1024),
            "allocated_mb": (end_memory - start_memory) / (1024 * 1024),
            "peak_allocated_mb": (peak_memory - start_memory) / (1024 * 1024),
        }
    else:
        # CPU profiling - basic implementation
        result = func(*args, **kwargs)
        stats = {
            "start_memory_mb": 0.0,
            "end_memory_mb": 0.0,
            "peak_memory_mb": 0.0,
            "allocated_mb": 0.0,
            "peak_allocated_mb": 0.0,
        }

    return result, stats


@functools.lru_cache(maxsize=128)
def estimate_memory_cost(
    tensor_shape: Tuple[int, ...], dtype: torch.dtype = torch.float32
) -> float:
    """
    Estimate memory cost for a tensor shape.

    Args:
        tensor_shape: Shape of tensor
        dtype: Data type of tensor

    Returns:
        Estimated memory cost in MB
    """
    num_elements = 1
    for dim in tensor_shape:
        num_elements *= dim

    bytes_per_element = torch.finfo(dtype).bits // 8
    memory_mb = (num_elements * bytes_per_element) / (1024 * 1024)

    return float(memory_mb)
