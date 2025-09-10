"""
Memory Pool Management for Gradient Coalescing

This module provides efficient memory pool management for gradient coalescing
operations, reducing allocation overhead and improving cache locality.
"""

import logging
import threading
from collections import deque
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class CoalescingMemoryPool:
    """
    Memory pool for efficient buffer allocation during gradient coalescing.

    This class manages a pool of pre-allocated tensors that can be reused
    across coalescing operations, reducing allocation overhead and memory
    fragmentation.

    Args:
        initial_size_mb: Initial pool size in MB
        growth_factor: Factor to grow pool when needed
        max_size_mb: Maximum pool size in MB
        device: Device for tensor allocation
    """

    def __init__(
        self,
        initial_size_mb: float = 50.0,
        growth_factor: float = 1.5,
        max_size_mb: float = 500.0,
        device: Optional[torch.device] = None,
    ):
        self.initial_size_mb = initial_size_mb
        self.growth_factor = growth_factor
        self.max_size_mb = max_size_mb
        self.device = device or torch.device("cpu")

        # Thread safety
        self._lock = threading.RLock()

        # Pool storage by dtype and size
        self._pools: Dict[Tuple[torch.dtype, int], deque] = {}
        self._allocated_bytes = 0
        self._peak_allocated_bytes = 0

        # Statistics
        self.hits = 0
        self.misses = 0
        self.allocations = 0

        # Pre-allocate initial buffers
        self._preallocate_buffers()

    def _preallocate_buffers(self):
        """Pre-allocate initial set of buffers."""
        # Common buffer sizes (in elements)
        common_sizes = [
            1024,  # 1K elements
            10240,  # 10K elements
            102400,  # 100K elements
            1024000,  # 1M elements
        ]

        # Common dtypes
        common_dtypes = [torch.float32, torch.float16]

        initial_bytes = int(self.initial_size_mb * 1024 * 1024)
        bytes_per_dtype = initial_bytes // len(common_dtypes)

        with self._lock:
            for dtype in common_dtypes:
                bytes_allocated = 0
                for size in common_sizes:
                    if bytes_allocated >= bytes_per_dtype:
                        break

                    # Allocate a few buffers of each size
                    num_buffers = min(3, bytes_per_dtype // (size * dtype.itemsize))
                    for _ in range(num_buffers):
                        buffer = torch.empty(size, dtype=dtype, device=self.device)
                        self._return_buffer(buffer)
                        bytes_allocated += buffer.numel() * buffer.element_size()
                        self._allocated_bytes += buffer.numel() * buffer.element_size()

        logger.debug(
            f"Pre-allocated {self._allocated_bytes / 1e6:.1f}MB in memory pool"
        )

    def get_buffer(
        self, size: int, dtype: torch.dtype, zero_out: bool = False
    ) -> torch.Tensor:
        """
        Get a buffer from the pool or allocate a new one.

        Args:
            size: Number of elements needed
            dtype: Data type of the buffer
            zero_out: Whether to zero the buffer before returning

        Returns:
            Tensor buffer of requested size and dtype
        """
        key = (dtype, size)

        with self._lock:
            # Check if we have a buffer of exact size
            if key in self._pools and self._pools[key]:
                buffer = self._pools[key].popleft()
                self.hits += 1

                if zero_out:
                    buffer.zero_()
                return buffer  # type: ignore[no-any-return]

            # Check for larger buffers we can use
            for (pool_dtype, pool_size), pool in self._pools.items():
                if pool_dtype == dtype and pool_size >= size and pool:
                    buffer = pool.popleft()
                    # Return the view of required size
                    buffer = buffer[:size]
                    self.hits += 1

                    if zero_out:
                        buffer.zero_()
                    return buffer  # type: ignore[no-any-return]

            # Allocate new buffer
            self.misses += 1
            self.allocations += 1

            # Check if we're within limits
            new_bytes = size * dtype.itemsize
            if self._allocated_bytes + new_bytes > self.max_size_mb * 1024 * 1024:
                # Try to free some memory
                self._evict_buffers(new_bytes)

            buffer = torch.empty(size, dtype=dtype, device=self.device)
            if zero_out:
                buffer.zero_()

            self._allocated_bytes += new_bytes
            self._peak_allocated_bytes = max(
                self._peak_allocated_bytes, self._allocated_bytes
            )

            return buffer  # type: ignore[no-any-return]

    def return_buffer(self, buffer: torch.Tensor):
        """
        Return a buffer to the pool for reuse.

        Args:
            buffer: Tensor to return to the pool
        """
        if buffer is None:
            return

        self._return_buffer(buffer)

    def _return_buffer(self, buffer: torch.Tensor):
        """Internal method to return buffer to pool."""
        key = (buffer.dtype, buffer.numel())

        with self._lock:
            if key not in self._pools:
                self._pools[key] = deque()

            # Limit pool size per key to prevent unbounded growth
            if len(self._pools[key]) < 10:
                self._pools[key].append(buffer)

    def _evict_buffers(self, bytes_needed: int):
        """
        Evict buffers from pool to make room for new allocation.

        Args:
            bytes_needed: Number of bytes needed
        """
        bytes_freed = 0

        with self._lock:
            # Evict largest buffers first
            sorted_pools = sorted(
                self._pools.items(),
                key=lambda x: x[0][1] * x[0][0].itemsize,
                reverse=True,
            )

            for (dtype, size), pool in sorted_pools:
                while pool and bytes_freed < bytes_needed:
                    buffer = pool.popleft()
                    bytes_freed += buffer.numel() * buffer.element_size()
                    self._allocated_bytes -= buffer.numel() * buffer.element_size()
                    del buffer  # Explicitly delete to free memory

                if bytes_freed >= bytes_needed:
                    break

        if bytes_freed > 0:
            logger.debug(f"Evicted {bytes_freed / 1e6:.1f}MB from memory pool")

    def clear(self):
        """Clear all buffers from the pool."""
        with self._lock:
            self._pools.clear()
            self._allocated_bytes = 0
            logger.debug("Cleared memory pool")

    def get_stats(self) -> Dict[str, float]:
        """
        Get memory pool statistics.

        Returns:
            Dictionary with pool statistics
        """
        with self._lock:
            total_buffers = sum(len(pool) for pool in self._pools.values())
            hit_rate = (
                self.hits / (self.hits + self.misses)
                if (self.hits + self.misses) > 0
                else 0.0
            )

            return {
                "allocated_mb": self._allocated_bytes / 1e6,
                "peak_allocated_mb": self._peak_allocated_bytes / 1e6,
                "num_buffers": total_buffers,
                "hit_rate": hit_rate,
                "hits": self.hits,
                "misses": self.misses,
                "allocations": self.allocations,
            }

    def log_stats(self):
        """Log memory pool statistics."""
        stats = self.get_stats()
        logger.info(
            f"Memory Pool Stats: "
            f"allocated={stats['allocated_mb']:.1f}MB, "
            f"peak={stats['peak_allocated_mb']:.1f}MB, "
            f"buffers={stats['num_buffers']}, "
            f"hit_rate={stats['hit_rate']:.2%}, "
            f"allocations={stats['allocations']}"
        )
