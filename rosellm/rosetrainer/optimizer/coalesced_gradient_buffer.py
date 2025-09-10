"""
Coalesced Gradient Buffer for Optimized Distributed Training

This module extends the base GradientBuffer with coalescing support,
enabling multiple gradient reduction operations to be batched into
single kernel launches for improved communication efficiency.

The implementation follows Megatron-LM's approach while integrating
seamlessly with RoseLLM's existing gradient management infrastructure.

Key Features:
    - Automatic bucket coalescing for gradient reductions
    - Support for both all-reduce and reduce-scatter operations
    - Adaptive bucket sizing based on coalescing efficiency
    - Backward compatibility with non-coalesced mode
    - Integration with distributed optimizer

Example:
    >>> from rosellm.rosetrainer.optimizer import CoalescedGradientBuffer
    >>>
    >>> buffer = CoalescedGradientBuffer(
    ...     params=model.parameters(),
    ...     enable_coalescing=True,
    ...     bucket_size_mb=25.0,
    ... )
    >>>
    >>> # During backward pass
    >>> loss.backward()
    >>> buffer.synchronize_gradients()  # Uses coalescing automatically
"""

import logging
import threading
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.nn import Parameter

from rosellm.rosetrainer.communication.coalescing import (
    CoalescingConfig,
    CoalescingManager,
)
from rosellm.rosetrainer.optimizer.exceptions import (
    CommunicationError,
    GradientBufferError,
)
from rosellm.rosetrainer.optimizer.gradient_buffer import Bucket, GradientBuffer

logger = logging.getLogger(__name__)


class CoalescedBucket(Bucket):
    """Extended bucket with coalescing-specific metadata."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coalesce_group_id: Optional[int] = None
        self.bytes_size: int = 0

    def calculate_bytes_size(self) -> int:
        """Calculate the size of this bucket in bytes."""
        if self.grad_buffer is not None:
            self.bytes_size = self.grad_buffer.numel() * self.grad_buffer.element_size()
        return self.bytes_size


class CoalescedGradientBuffer(GradientBuffer):
    """
    Gradient buffer with coalescing support for optimized communication.

    This class extends GradientBuffer to support coalescing of multiple
    gradient reduction operations into single kernel launches, following
    the patterns established by Megatron-LM.

    Args:
        params: List of model parameters to manage
        enable_coalescing: Enable gradient bucket coalescing
        coalescing_config: Configuration for coalescing behavior
        use_distributed_optimizer: Use reduce-scatter instead of all-reduce
        **kwargs: Additional arguments passed to GradientBuffer
    """

    def __init__(
        self,
        params: List[Parameter],
        enable_coalescing: bool = True,
        coalescing_config: Optional[CoalescingConfig] = None,
        use_distributed_optimizer: bool = False,
        **kwargs,
    ):
        # Initialize base gradient buffer
        super().__init__(params, **kwargs)

        self.enable_coalescing = enable_coalescing
        self.use_distributed_optimizer = use_distributed_optimizer

        # Setup coalescing manager
        self.coalescing_manager: Optional[CoalescingManager]
        self.coalescing_groups: List[List[CoalescedBucket]]

        if enable_coalescing:
            config = coalescing_config or CoalescingConfig()
            self.coalescing_manager = CoalescingManager(
                process_group=self.process_group, config=config
            )

            # Convert buckets to CoalescedBucket type
            self._upgrade_buckets()

            # Organize buckets into coalescing groups
            self._create_coalescing_groups()
        else:
            self.coalescing_manager = None
            self.coalescing_groups = []

        # Communication handles
        self.active_handles: Dict[int, Any] = {}
        self.handle_lock = threading.Lock()

        logger.info(
            f"Created CoalescedGradientBuffer with {len(self.buckets)} buckets, "
            f"coalescing={'enabled' if enable_coalescing else 'disabled'}, "
            f"distributed_optimizer={use_distributed_optimizer}"
        )

    def _upgrade_buckets(self):
        """Convert existing buckets to CoalescedBucket type."""
        upgraded_buckets = []
        for bucket in self.buckets:
            coalesced_bucket = CoalescedBucket(
                index=bucket.index,
                size=bucket.size,
                dtype=bucket.dtype,
                params=bucket.params,
                param_indices=bucket.param_indices,
                grad_buffer=bucket.grad_buffer,
                all_reduce_handle=bucket.all_reduce_handle,
                is_ready=bucket.is_ready,
                is_reduced=bucket.is_reduced,
            )
            coalesced_bucket.calculate_bytes_size()
            upgraded_buckets.append(coalesced_bucket)

        self.buckets = upgraded_buckets

    def _create_coalescing_groups(self):
        """
        Organize buckets into groups for coalescing.

        Groups buckets based on the configured max coalesce size
        to optimize kernel launches while respecting memory constraints.
        """
        if not self.coalescing_manager:
            return

        self.coalescing_groups = []
        current_group = []
        current_group_size = 0
        max_size_bytes = int(
            self.coalescing_manager.config.max_coalesce_size_mb * 1024 * 1024
        )

        for bucket in self.buckets:
            bucket_size = (
                bucket.calculate_bytes_size()
                if isinstance(bucket, CoalescedBucket)
                else 0
            )

            # Check if adding this bucket exceeds max coalesce size
            if current_group and current_group_size + bucket_size > max_size_bytes:
                # Save current group and start new one
                self.coalescing_groups.append(current_group)
                current_group = []
                current_group_size = 0

            # Add bucket to current group
            current_group.append(bucket)
            current_group_size += bucket_size
            if isinstance(bucket, CoalescedBucket):
                bucket.coalesce_group_id = len(self.coalescing_groups)

        # Add final group
        if current_group:
            self.coalescing_groups.append(current_group)

        logger.debug(
            f"Created {len(self.coalescing_groups)} coalescing groups from "
            f"{len(self.buckets)} buckets"
        )

    def synchronize_gradients(self) -> None:
        """
        Synchronize gradients across data parallel ranks using coalescing.

        This method overrides the base implementation to use coalesced
        communication when enabled, falling back to standard behavior
        when coalescing is disabled or unavailable.
        """
        if not self.enable_coalescing or not self.coalescing_manager:
            # Fall back to standard synchronization
            self.synchronize_all_buckets()
            return

        # Process each coalescing group
        for group_id, bucket_group in enumerate(self.coalescing_groups):
            self._synchronize_coalesced_group(group_id, bucket_group)

        # Wait for all handles
        self._wait_all_handles()

        # Log metrics if profiling is enabled
        if self.coalescing_manager.config.profile_communication:
            self.coalescing_manager.log_metrics()

    def _synchronize_coalesced_group(
        self, group_id: int, buckets: List[CoalescedBucket]
    ) -> None:
        """
        Synchronize a group of buckets using coalescing.

        Args:
            group_id: Identifier for this coalescing group
            buckets: List of buckets to synchronize together
        """
        if not buckets:
            return

        # Check if all buckets in group are ready
        if not all(b.is_ready for b in buckets):
            logger.warning(
                f"Coalescing group {group_id} has unready buckets, "
                f"falling back to individual reductions"
            )
            for bucket in buckets:
                if bucket.is_ready:
                    self._reduce_bucket_standard(bucket)
            return

        # Perform coalesced reduction
        if self.coalescing_manager is None:
            return
        with self.coalescing_manager.coalesce_context(async_ops=True) as handle:
            for bucket in buckets:
                if self.use_distributed_optimizer:
                    self._reduce_scatter_bucket(bucket, async_op=True)
                else:
                    self._all_reduce_bucket(bucket, async_op=True)

                bucket.is_reduced = True

        # Store handle for later synchronization
        if handle and hasattr(handle, "wait"):
            with self.handle_lock:
                self.active_handles[group_id] = handle

    def _all_reduce_bucket(self, bucket: CoalescedBucket, async_op: bool = True):
        """
        Perform all-reduce on a bucket's gradient buffer.

        Args:
            bucket: Bucket to reduce
            async_op: Whether to perform asynchronous operation
        """
        if bucket.grad_buffer is None:
            raise GradientBufferError(f"Bucket {bucket.index} has no gradient buffer")

        # Scale gradients if needed (e.g., for gradient averaging)
        world_size = dist.get_world_size(self.process_group)
        if world_size > 1:
            bucket.grad_buffer.div_(world_size)

        # Perform all-reduce
        handle = dist.all_reduce(
            bucket.grad_buffer,
            op=dist.ReduceOp.SUM,
            group=self.process_group,
            async_op=async_op,
        )

        if async_op and handle:
            bucket.all_reduce_handle = handle

    def _reduce_scatter_bucket(self, bucket: CoalescedBucket, async_op: bool = True):
        """
        Perform reduce-scatter on a bucket's gradient buffer.

        This is used with distributed optimizer to partition gradients
        across ranks, following Megatron-LM's approach.

        Args:
            bucket: Bucket to reduce-scatter
            async_op: Whether to perform asynchronous operation
        """
        if bucket.grad_buffer is None:
            raise GradientBufferError(f"Bucket {bucket.index} has no gradient buffer")

        world_size = dist.get_world_size(self.process_group)
        if world_size == 1:
            return

        # Ensure buffer is evenly divisible
        if bucket.grad_buffer.numel() % world_size != 0:
            raise GradientBufferError(
                f"Buffer size {bucket.grad_buffer.numel()} not divisible by "
                f"world size {world_size} for reduce-scatter"
            )

        # Create output buffer for local shard
        shard_size = bucket.grad_buffer.numel() // world_size
        output_buffer = torch.zeros(
            shard_size, dtype=bucket.grad_buffer.dtype, device=bucket.grad_buffer.device
        )

        # Perform reduce-scatter
        if hasattr(dist, "reduce_scatter_tensor"):
            handle = dist.reduce_scatter_tensor(
                output_buffer,
                bucket.grad_buffer,
                op=dist.ReduceOp.SUM,
                group=self.process_group,
                async_op=async_op,
            )
        else:
            # Fallback for older PyTorch versions
            input_list = list(bucket.grad_buffer.chunk(world_size))
            handle = dist.reduce_scatter(
                output_buffer,
                input_list,
                op=dist.ReduceOp.SUM,
                group=self.process_group,
                async_op=async_op,
            )

        if async_op and handle:
            bucket.all_reduce_handle = handle

        # Store the reduced shard
        bucket.grad_buffer = output_buffer

    def _reduce_bucket_standard(self, bucket: CoalescedBucket):
        """Fallback to standard (non-coalesced) bucket reduction."""
        if self.use_distributed_optimizer:
            self._reduce_scatter_bucket(bucket, async_op=False)
        else:
            self._all_reduce_bucket(bucket, async_op=False)
        bucket.is_reduced = True

    def _wait_all_handles(self):
        """Wait for all active communication handles to complete."""
        with self.handle_lock:
            for group_id, handle in self.active_handles.items():
                try:
                    handle.wait()
                except Exception as e:
                    raise CommunicationError(
                        f"Failed to complete coalesced communication for "
                        f"group {group_id}: {e}"
                    )
            self.active_handles.clear()

    def get_coalescing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about coalescing performance.

        Returns:
            Dictionary containing coalescing metrics
        """
        if not self.coalescing_manager:
            return {}

        metrics = self.coalescing_manager.metrics
        return {
            "total_coalesced_ops": metrics.total_coalesced_ops,
            "total_bytes_coalesced": metrics.total_bytes_coalesced,
            "avg_ops_per_coalesce": metrics.avg_ops_per_coalesce,
            "peak_coalesce_size_mb": metrics.peak_coalesce_size_mb,
            "num_coalesce_calls": metrics.num_coalesce_calls,
            "num_fallbacks": metrics.num_fallbacks,
            "num_coalescing_groups": len(self.coalescing_groups),
        }

    def reset_coalescing_stats(self):
        """Reset coalescing performance statistics."""
        if self.coalescing_manager:
            self.coalescing_manager.reset_metrics()

    def optimize_coalescing_groups(self):
        """
        Re-optimize coalescing groups based on observed performance.

        This method can be called periodically to adjust grouping
        based on the adaptive sizing in the coalescing manager.
        """
        if (
            not self.coalescing_manager
            or not self.coalescing_manager.config.adaptive_sizing
        ):
            return

        # Get updated optimal size
        optimal_size_mb = self.coalescing_manager.get_optimal_bucket_size()

        # Update config and recreate groups
        self.coalescing_manager.config.max_coalesce_size_mb = optimal_size_mb
        self._create_coalescing_groups()

        logger.info(
            f"Re-optimized coalescing groups with size {optimal_size_mb:.1f}MB, "
            f"created {len(self.coalescing_groups)} groups"
        )
