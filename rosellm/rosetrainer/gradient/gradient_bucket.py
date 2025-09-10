"""Gradient bucketing for efficient communication.

This module implements gradient bucketing to coalesce small tensors
for more efficient distributed communication.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass
class GradientBucket:
    """Container for gradients to be reduced together.

    Attributes:
        tensors: List of gradient tensors in this bucket.
        size_bytes: Total size of tensors in bytes.
        is_full: Whether the bucket is full.
        reduction_handle: Async reduction handle if applicable.
    """

    tensors: List[torch.Tensor] = field(default_factory=list)
    size_bytes: int = 0
    is_full: bool = False
    reduction_handle: Optional[dist.Work] = None

    def add_tensor(self, tensor: torch.Tensor, max_size_bytes: int) -> bool:
        """Add a tensor to the bucket.

        Args:
            tensor: Tensor to add.
            max_size_bytes: Maximum bucket size.

        Returns:
            True if tensor was added, False if bucket would overflow.
        """
        tensor_size = tensor.numel() * tensor.element_size()

        if self.size_bytes + tensor_size > max_size_bytes and self.tensors:
            # Would overflow and not empty
            self.is_full = True
            return False

        self.tensors.append(tensor)
        self.size_bytes += tensor_size

        if self.size_bytes >= max_size_bytes:
            self.is_full = True

        return True

    def clear(self) -> None:
        """Clear the bucket."""
        self.tensors.clear()
        self.size_bytes = 0
        self.is_full = False
        self.reduction_handle = None


class GradientBucketer:
    """Manages gradient bucketing for efficient communication.

    This class groups small gradient tensors into buckets for
    coalesced communication, reducing the overhead of many small
    all-reduce operations.
    """

    def __init__(
        self,
        bucket_size_mb: float = 25.0,
        dtype_buckets: bool = True,
        device_buckets: bool = True,
    ):
        """Initialize gradient bucketer.

        Args:
            bucket_size_mb: Maximum bucket size in megabytes.
            dtype_buckets: Whether to create separate buckets per dtype.
            device_buckets: Whether to create separate buckets per device.
        """
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        self.dtype_buckets = dtype_buckets
        self.device_buckets = device_buckets

        # Buckets organized by (device, dtype)
        self.buckets: Dict[Tuple[torch.device, torch.dtype], GradientBucket] = {}
        self.pending_reductions: List[dist.Work] = []

        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration."""
        if self.bucket_size_bytes <= 0:
            raise ValueError(
                f"Bucket size must be positive, got {self.bucket_size_bytes}"
            )

    def _get_bucket_key(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.device, torch.dtype]:
        """Get bucket key for a tensor.

        Args:
            tensor: Tensor to get key for.

        Returns:
            Tuple of (device, dtype) for bucketing.
        """
        device = tensor.device if self.device_buckets else torch.device("cpu")
        dtype = tensor.dtype if self.dtype_buckets else torch.float32
        return (device, dtype)

    def add_gradient(
        self,
        gradient: torch.Tensor,
        force_new_bucket: bool = False,
    ) -> Optional[GradientBucket]:
        """Add a gradient to the bucketing system.

        Args:
            gradient: Gradient tensor to add.
            force_new_bucket: Whether to force a new bucket.

        Returns:
            The bucket that became full, if any.
        """
        key = self._get_bucket_key(gradient)

        # Get or create bucket
        if key not in self.buckets or force_new_bucket:
            self.buckets[key] = GradientBucket()

        bucket = self.buckets[key]

        # Try to add to bucket
        if not bucket.add_tensor(gradient, self.bucket_size_bytes):
            # Bucket is full, create new one
            full_bucket = bucket
            self.buckets[key] = GradientBucket()
            self.buckets[key].add_tensor(gradient, self.bucket_size_bytes)
            return full_bucket

        # Check if bucket became full
        if bucket.is_full:
            return bucket

        return None

    def flush_bucket(
        self,
        bucket: GradientBucket,
        process_group: Optional[dist.ProcessGroup] = None,
        async_op: bool = False,
    ) -> Optional[dist.Work]:
        """Flush a bucket by performing reduction.

        Args:
            bucket: Bucket to flush.
            process_group: Process group for communication.
            async_op: Whether to perform async reduction.

        Returns:
            Async work handle if async_op is True.
        """
        if not bucket.tensors:
            return None

        # Coalesce tensors in bucket
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

        coalesced = _flatten_dense_tensors(bucket.tensors)

        # Perform reduction
        handle = dist.all_reduce(
            coalesced,
            group=process_group,
            async_op=async_op,
        )

        if async_op and handle is not None:
            # Store handle for later synchronization
            bucket.reduction_handle = handle
            self.pending_reductions.append(handle)

        # Unflatten if synchronous
        if not async_op:
            unflattened = _unflatten_dense_tensors(coalesced, bucket.tensors)
            for i, tensor in enumerate(bucket.tensors):
                tensor.copy_(unflattened[i])

        return handle  # type: ignore[no-any-return]

    def flush_all_buckets(
        self,
        process_group: Optional[dist.ProcessGroup] = None,
        async_op: bool = False,
    ) -> List[dist.Work]:
        """Flush all remaining buckets.

        Args:
            process_group: Process group for communication.
            async_op: Whether to perform async reduction.

        Returns:
            List of async work handles if async_op is True.
        """
        handles = []

        for bucket in self.buckets.values():
            if bucket.tensors:
                handle = self.flush_bucket(bucket, process_group, async_op)
                if handle is not None:
                    handles.append(handle)

        # Clear buckets after flushing
        self.buckets.clear()

        return handles

    def synchronize_reductions(self) -> None:
        """Wait for all pending async reductions to complete."""
        for handle in self.pending_reductions:
            handle.wait()

        self.pending_reductions.clear()

    def get_statistics(self) -> Dict[str, float]:
        """Get bucketing statistics.

        Returns:
            Dictionary with statistics about bucketing.
        """
        total_buckets = len(self.buckets)
        total_tensors = sum(len(b.tensors) for b in self.buckets.values())
        total_bytes = sum(b.size_bytes for b in self.buckets.values())

        return {
            "num_buckets": total_buckets,
            "num_tensors": total_tensors,
            "total_bytes": total_bytes,
            "avg_bucket_size_mb": (total_bytes / max(1, total_buckets)) / (1024 * 1024),
            "pending_reductions": len(self.pending_reductions),
        }


class SmartGradientBucketer(GradientBucketer):
    """Enhanced gradient bucketer with adaptive strategies.

    This bucketer analyzes gradient patterns and network conditions
    to optimize bucketing strategies dynamically.
    """

    def __init__(
        self,
        bucket_size_mb: float = 25.0,
        dtype_buckets: bool = True,
        device_buckets: bool = True,
        adaptive_sizing: bool = True,
        profile_communication: bool = False,
    ):
        """Initialize smart gradient bucketer.

        Args:
            bucket_size_mb: Initial bucket size in megabytes.
            dtype_buckets: Whether to create separate buckets per dtype.
            device_buckets: Whether to create separate buckets per device.
            adaptive_sizing: Whether to adapt bucket size based on patterns.
            profile_communication: Whether to profile communication times.
        """
        super().__init__(bucket_size_mb, dtype_buckets, device_buckets)

        self.adaptive_sizing = adaptive_sizing
        self.profile_communication = profile_communication

        # Adaptive sizing state
        self.communication_times: List[float] = []
        self.bucket_sizes: List[int] = []
        self.optimal_bucket_size: Optional[int] = None

    def _adapt_bucket_size(self) -> None:
        """Adapt bucket size based on observed patterns."""
        if not self.adaptive_sizing or len(self.communication_times) < 10:
            return

        # Analyze communication efficiency
        import numpy as np

        times = np.array(self.communication_times[-100:])
        sizes = np.array(self.bucket_sizes[-100:])

        # Find size with best throughput
        unique_sizes = np.unique(sizes)
        throughputs = []

        for size in unique_sizes:
            mask = sizes == size
            if mask.sum() > 0:
                avg_time = times[mask].mean()
                throughput = size / max(avg_time, 1e-6)
                throughputs.append(throughput)

        if throughputs:
            best_idx = np.argmax(throughputs)
            self.optimal_bucket_size = int(unique_sizes[best_idx])
            self.bucket_size_bytes = self.optimal_bucket_size

            logger.info(
                f"Adapted bucket size to {self.bucket_size_bytes / (1024*1024):.1f} MB"
            )

    def flush_bucket(
        self,
        bucket: GradientBucket,
        process_group: Optional[dist.ProcessGroup] = None,
        async_op: bool = False,
    ) -> Optional[dist.Work]:
        """Flush bucket with profiling if enabled.

        Args:
            bucket: Bucket to flush.
            process_group: Process group for communication.
            async_op: Whether to perform async reduction.

        Returns:
            Async work handle if async_op is True.
        """
        if self.profile_communication and bucket.tensors:
            import time

            start_time = time.perf_counter()

        handle = super().flush_bucket(bucket, process_group, async_op)

        if self.profile_communication and bucket.tensors:
            if not async_op:
                # Synchronous operation, measure time directly
                comm_time = time.perf_counter() - start_time
                self.communication_times.append(comm_time)
                self.bucket_sizes.append(bucket.size_bytes)

                # Periodically adapt bucket size
                if len(self.communication_times) % 50 == 0:
                    self._adapt_bucket_size()

        return handle
