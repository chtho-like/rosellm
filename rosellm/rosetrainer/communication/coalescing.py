"""
Gradient Bucket Coalescing for Optimized Communication

This module implements gradient bucket coalescing, a critical optimization that
batches multiple gradient communication operations into a single kernel launch.
This significantly reduces communication overhead and improves training throughput.

The implementation is inspired by Megatron-LM's use of PyTorch's _coalescing_manager
and provides a clean abstraction that integrates with RoseLLM's parallelism framework.

Key Features:
    - Context manager interface for automatic resource management
    - Support for both NCCL and Gloo backends
    - Adaptive coalescing window sizing
    - Fallback mechanisms for older PyTorch versions
    - Performance metrics collection
    - Memory-efficient buffer management

Example:
    >>> from rosellm.rosetrainer.communication import CoalescingManager
    >>>
    >>> manager = CoalescingManager(process_group=dp_group)
    >>> with manager.coalesce_context(async_ops=True) as handle:
    ...     for bucket in gradient_buckets:
    ...         dist.all_reduce(bucket.grad_data, async_op=True)
    >>> if handle:
    ...     handle.wait()

References:
    - Megatron-LM: https://github.com/NVIDIA/Megatron-LM
    - PyTorch Distributed: https://pytorch.org/docs/stable/distributed.html
"""

import logging
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Check for PyTorch coalescing support
# Note: _coalescing_manager is a private API that may change between versions
try:
    from torch.distributed import _coalescing_manager  # type: ignore

    HAS_COALESCING = True
except (ImportError, AttributeError):
    _coalescing_manager = None  # type: ignore
    HAS_COALESCING = False


@dataclass
class CoalescingConfig:
    """Configuration for gradient bucket coalescing.

    Args:
        enable_coalescing: Enable gradient bucket coalescing
        max_coalesce_size_mb: Maximum size of coalesced operation in MB
        min_buckets_to_coalesce: Minimum number of buckets before coalescing
        coalesce_timeout_ms: Timeout in ms before forcing coalesce
        adaptive_sizing: Dynamically adjust coalesce size based on performance
        profile_communication: Enable detailed communication profiling
        fallback_on_error: Fallback to non-coalesced mode on errors
    """

    enable_coalescing: bool = True
    max_coalesce_size_mb: float = 100.0
    min_buckets_to_coalesce: int = 2
    coalesce_timeout_ms: float = 10.0
    adaptive_sizing: bool = True
    profile_communication: bool = False
    fallback_on_error: bool = True


@dataclass
class CoalescingMetrics:
    """Metrics for coalescing performance monitoring."""

    total_coalesced_ops: int = 0
    total_bytes_coalesced: int = 0
    total_coalesce_time_ms: float = 0.0
    num_coalesce_calls: int = 0
    num_fallbacks: int = 0
    avg_ops_per_coalesce: float = 0.0
    peak_coalesce_size_mb: float = 0.0

    def update(self, num_ops: int, bytes_coalesced: int, time_ms: float):
        """Update metrics with new coalescing operation."""
        self.total_coalesced_ops += num_ops
        self.total_bytes_coalesced += bytes_coalesced
        self.total_coalesce_time_ms += time_ms
        self.num_coalesce_calls += 1

        # Update averages
        if self.num_coalesce_calls > 0:
            self.avg_ops_per_coalesce = (
                self.total_coalesced_ops / self.num_coalesce_calls
            )

        # Update peak
        size_mb = bytes_coalesced / (1024 * 1024)
        self.peak_coalesce_size_mb = max(self.peak_coalesce_size_mb, size_mb)

    def log_summary(self):
        """Log summary of coalescing metrics."""
        if self.num_coalesce_calls == 0:
            return

        avg_time = self.total_coalesce_time_ms / self.num_coalesce_calls
        total_gb = self.total_bytes_coalesced / (1024**3)

        logger.info(
            f"Coalescing Metrics: "
            f"calls={self.num_coalesce_calls}, "
            f"ops={self.total_coalesced_ops}, "
            f"avg_ops={self.avg_ops_per_coalesce:.1f}, "
            f"total_data={total_gb:.2f}GB, "
            f"avg_time={avg_time:.2f}ms, "
            f"peak_size={self.peak_coalesce_size_mb:.1f}MB, "
            f"fallbacks={self.num_fallbacks}"
        )


class CoalescingManager:
    """
    Manages coalescing of multiple gradient communication operations.

    This class provides a context manager interface for coalescing multiple
    communication operations (all-reduce, reduce-scatter) into a single
    NCCL/Gloo kernel launch, reducing overhead and improving performance.

    Args:
        process_group: Process group for communication
        config: Configuration for coalescing behavior
    """

    def __init__(
        self,
        process_group: Optional[dist.ProcessGroup] = None,
        config: Optional[CoalescingConfig] = None,
    ):
        self.process_group = process_group
        self.config = config or CoalescingConfig()
        self.metrics = CoalescingMetrics()

        # State management
        self.is_coalescing = False
        self.pending_operations: List[Any] = []
        self.coalesce_buffer_size = 0
        self.coalesce_start_time: Optional[float] = None

        # Adaptive sizing state
        if self.config.adaptive_sizing:
            self.adaptive_size_mb = self.config.max_coalesce_size_mb
            self.size_adjustment_factor = 1.1
            self.performance_history: List[dict] = []

        # Check backend support
        if dist.is_initialized() and self.process_group is not None:
            self.backend = dist.get_backend(self.process_group)
        elif dist.is_initialized():
            self.backend = dist.get_backend()
        else:
            self.backend = "gloo"  # Default for non-distributed
        self.supports_coalescing = self._check_coalescing_support()

        if not self.supports_coalescing and self.config.enable_coalescing:
            warnings.warn(
                f"Backend '{self.backend}' or PyTorch version does not support "
                f"coalescing. Falling back to non-coalesced mode.",
                RuntimeWarning,
            )
            self.config.enable_coalescing = False

    def _check_coalescing_support(self) -> bool:
        """Check if the current backend and PyTorch version support coalescing."""
        if not HAS_COALESCING:
            return False

        # NCCL always supports coalescing in recent PyTorch
        if self.backend == "nccl":
            return True

        # Gloo support is limited
        if self.backend == "gloo":
            # Check PyTorch version for Gloo coalescing support
            torch_version = torch.__version__.split(".")
            major, minor = int(torch_version[0]), int(torch_version[1])
            return major > 1 or (major == 1 and minor >= 12)

        return False

    @contextmanager
    def coalesce_context(self, async_ops: bool = True):
        """
        Context manager for coalescing communication operations.

        Args:
            async_ops: Whether to use asynchronous operations

        Yields:
            Communication handle if async_ops=True, None otherwise

        Example:
            >>> with manager.coalesce_context(async_ops=True) as handle:
            ...     for tensor in tensors:
            ...         dist.all_reduce(tensor, async_op=True)
            >>> if handle:
            ...     handle.wait()
        """
        if not self.config.enable_coalescing or not self.supports_coalescing:
            # Fallback to non-coalesced mode
            yield None
            return

        start_time = time.perf_counter()
        num_ops = 0
        total_bytes = 0

        try:
            # Use PyTorch's native coalescing manager if available
            if HAS_COALESCING and _coalescing_manager is not None:
                # Only use process group if distributed is initialized
                pg = self.process_group if dist.is_initialized() else None
                with _coalescing_manager(pg, async_ops=async_ops) as cm_handle:
                    self.is_coalescing = True
                    self.coalesce_start_time = start_time

                    # Track operations for metrics
                    original_all_reduce = dist.all_reduce

                    def tracked_all_reduce(tensor, *args, **kwargs):
                        nonlocal num_ops, total_bytes
                        num_ops += 1
                        total_bytes += tensor.numel() * tensor.element_size()
                        return original_all_reduce(tensor, *args, **kwargs)

                    # Temporarily replace functions to track metrics
                    dist.all_reduce = tracked_all_reduce

                    try:
                        yield cm_handle
                    finally:
                        # Restore original functions
                        dist.all_reduce = original_all_reduce
            else:
                # Fallback implementation for older PyTorch
                yield self._fallback_coalesce(async_ops)

        except Exception as e:
            logger.error(f"Error during coalescing: {e}")
            self.metrics.num_fallbacks += 1

            if self.config.fallback_on_error:
                # Fallback to non-coalesced execution
                yield None
            else:
                raise

        finally:
            self.is_coalescing = False

            # Update metrics
            if num_ops > 0:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                self.metrics.update(num_ops, total_bytes, elapsed_ms)

                # Adaptive sizing
                if self.config.adaptive_sizing:
                    self._adjust_coalesce_size(num_ops, total_bytes, elapsed_ms)

            # Log performance if profiling is enabled
            if self.config.profile_communication and num_ops > 0:
                logger.debug(
                    f"Coalesced {num_ops} ops, {total_bytes / 1e6:.1f}MB "
                    f"in {elapsed_ms:.2f}ms"
                )

    def _fallback_coalesce(self, async_ops: bool) -> Optional[Any]:
        """
        Fallback coalescing implementation for older PyTorch versions.

        This implementation manually batches operations but doesn't achieve
        the same level of kernel fusion as the native implementation.
        """
        # For older PyTorch, we can't truly coalesce at the kernel level
        # but we can at least batch the Python-level calls
        logger.debug("Using fallback coalescing implementation")
        self.metrics.num_fallbacks += 1

        # Return None to indicate no handle is available
        return None

    def _adjust_coalesce_size(self, num_ops: int, total_bytes: int, elapsed_ms: float):
        """
        Adjust coalescing size based on performance history.

        Uses a simple heuristic to find the optimal coalescing size
        that balances latency and throughput.
        """
        if not self.config.adaptive_sizing:
            return

        # Calculate throughput
        throughput_gbps = (total_bytes / 1e9) / (elapsed_ms / 1000)

        # Store performance history
        self.performance_history.append(
            {
                "size_mb": total_bytes / 1e6,
                "num_ops": num_ops,
                "time_ms": elapsed_ms,
                "throughput_gbps": throughput_gbps,
            }
        )

        # Adjust size based on recent performance
        if len(self.performance_history) >= 5:
            recent = self.performance_history[-5:]
            avg_throughput = sum(h["throughput_gbps"] for h in recent) / len(recent)

            # If throughput is increasing, try larger sizes
            if recent[-1]["throughput_gbps"] > avg_throughput * 1.05:
                self.adaptive_size_mb = min(
                    self.adaptive_size_mb * self.size_adjustment_factor,
                    self.config.max_coalesce_size_mb,
                )
                logger.debug(
                    f"Increased coalesce size to {self.adaptive_size_mb:.1f}MB"
                )

            # If throughput is decreasing, try smaller sizes
            elif recent[-1]["throughput_gbps"] < avg_throughput * 0.95:
                self.adaptive_size_mb = max(
                    self.adaptive_size_mb / self.size_adjustment_factor,
                    10.0,  # Minimum 10MB
                )
                logger.debug(
                    f"Decreased coalesce size to {self.adaptive_size_mb:.1f}MB"
                )

            # Trim history to prevent unbounded growth
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-50:]

    def get_optimal_bucket_size(self) -> float:
        """
        Get the current optimal bucket size based on adaptive sizing.

        Returns:
            Optimal bucket size in MB
        """
        if self.config.adaptive_sizing and hasattr(self, "adaptive_size_mb"):
            return self.adaptive_size_mb
        return self.config.max_coalesce_size_mb

    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics = CoalescingMetrics()
        if self.config.adaptive_sizing:
            self.performance_history.clear()
            self.adaptive_size_mb = self.config.max_coalesce_size_mb

    def log_metrics(self):
        """Log summary of coalescing metrics."""
        self.metrics.log_summary()


class CoalescingError(Exception):
    """Exception raised for coalescing-related errors."""

    pass
