"""Range-Aware Gradient Buffer with Advanced Memory Management.

This module extends the standard gradient buffer with range-based parameter mapping,
providing more efficient memory utilization and better communication patterns for
distributed training scenarios.

Key Features:
- Integration with range-based parameter buffer mapping
- Optimized gradient reduction patterns
- Memory-efficient buffer management with compaction
- Support for mixed precision and multi-tensor operations
- Advanced profiling and monitoring capabilities
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Parameter

from .exceptions import CommunicationError, ConfigurationError, GradientBufferError
from .range_buffer_mapping import (
    BufferRange,
    RangeBufferConfig,
    RangeBufferMapper,
    RangeBufferStrategy,
)

logger = logging.getLogger(__name__)

# Constants
BYTES_PER_MB = 1024 * 1024
DEFAULT_BUCKET_SIZE_MB = 25.0
GRADIENT_SYNC_TIMEOUT_SEC = 30.0
MAX_REDUCTION_RETRIES = 3


@dataclass
class RangeAwareBucket:
    """Enhanced bucket with range-based parameter organization.

    Attributes:
        index: Bucket index for identification.
        buffer_range: Associated buffer range from range mapper.
        params: List of parameters in this bucket.
        param_indices: Parameter indices in original parameter list.
        grad_buffer: Gradient buffer tensor.
        reduction_handle: Handle for asynchronous reduction operation.
        is_ready: Whether all gradients are ready for reduction.
        is_reduced: Whether reduction has been completed.
        last_sync_time: Timestamp of last synchronization.
        reduction_count: Number of reductions performed.
    """

    index: int
    buffer_range: BufferRange
    params: List[Parameter]
    param_indices: List[int]
    grad_buffer: Optional[Tensor] = None
    reduction_handle: Optional[dist.Work] = None
    is_ready: bool = False
    is_reduced: bool = False
    last_sync_time: float = 0.0
    reduction_count: int = 0


@dataclass
class GradientReductionStats:
    """Statistics for gradient reduction operations.

    Attributes:
        total_reductions: Total number of reductions performed.
        average_reduction_time: Average time per reduction in seconds.
        bytes_reduced: Total bytes reduced across all operations.
        failed_reductions: Number of failed reduction attempts.
        timeout_count: Number of reduction timeouts.
        compaction_triggers: Number of times buffer compaction was triggered.
    """

    total_reductions: int = 0
    average_reduction_time: float = 0.0
    bytes_reduced: int = 0
    failed_reductions: int = 0
    timeout_count: int = 0
    compaction_triggers: int = 0


class RangeAwareGradientBuffer:
    """Advanced gradient buffer with range-based parameter mapping.

    This class provides enhanced gradient buffering by leveraging range-based
    parameter organization for better memory efficiency and communication
    patterns in distributed training.

    Args:
        params: List of model parameters to manage.
        range_buffer_config: Configuration for range-based buffer mapping.
        bucket_size_mb: Target bucket size in megabytes.
        world_size: Number of ranks in distributed training.
        rank: Current rank identifier.
        process_group: Process group for gradient reduction.
        enable_profiling: Whether to enable detailed profiling.
    """

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        self.cleanup()
        return False

    def __init__(
        self,
        params: List[Parameter],
        range_buffer_config: Optional[RangeBufferConfig] = None,
        bucket_size_mb: float = DEFAULT_BUCKET_SIZE_MB,
        world_size: int = 1,
        rank: int = 0,
        process_group: Optional[dist.ProcessGroup] = None,
        enable_profiling: bool = False,
    ):
        # Validate inputs
        if not params:
            raise ConfigurationError(
                "No parameters provided to RangeAwareGradientBuffer"
            )
        if bucket_size_mb <= 0:
            raise ConfigurationError(
                f"bucket_size_mb must be positive, got {bucket_size_mb}"
            )

        # Filter to only trainable parameters
        self.params = [p for p in params if p.requires_grad]
        if not self.params:
            raise ConfigurationError(
                "No trainable parameters (requires_grad=True) provided to "
                "RangeAwareGradientBuffer"
            )

        # Check parameter consistency
        if self.params:
            first_device = self.params[0].device
            for i, param in enumerate(self.params):
                if param.device != first_device:
                    raise ConfigurationError(
                        f"Parameter {i} is on device {param.device}, "
                        f"expected {first_device}"
                    )

        self.bucket_size_mb = bucket_size_mb
        self.world_size = world_size
        self.rank = rank
        self.process_group = process_group
        # Use config's profiling setting if provided, otherwise use explicit parameter
        self.enable_profiling = (
            range_buffer_config.enable_profiling
            if range_buffer_config is not None
            else enable_profiling
        )

        # Device configuration
        self.device = self.params[0].device if self.params else torch.device("cpu")

        # Thread safety
        self._lock = threading.RLock()
        self._reduction_lock = threading.Lock()

        # Initialize range buffer mapper
        if range_buffer_config is None:
            range_buffer_config = RangeBufferConfig(
                strategy=RangeBufferStrategy.CONTIGUOUS,
                device=self.device,
                enable_profiling=enable_profiling,
            )

        try:
            self.range_mapper = RangeBufferMapper(
                parameters=self.params,
                config=range_buffer_config,
                world_size=world_size,
                rank=rank,
                process_group=process_group,
            )
        except Exception as e:
            logger.error(f"Failed to initialize range buffer mapper: {e}")
            raise ConfigurationError(
                f"Range buffer mapper initialization failed: {e}"
            ) from e

        # Initialize buckets and mappings
        self.buckets: List[RangeAwareBucket] = []
        self.param_to_bucket: Dict[Parameter, int] = {}
        self.param_to_buffer_offset: Dict[Parameter, Tuple[int, int]] = {}
        self._params_with_grads: Set[Parameter] = set()

        # Statistics and profiling
        self.reduction_stats = GradientReductionStats()
        self._profiling_data: List[Dict[str, float]] = []

        # Create buckets from range mapper
        self._create_buckets_from_ranges()

        # Register gradient hooks
        self._register_grad_hooks()

        logger.info(
            f"Initialized RangeAwareGradientBuffer with {len(self.buckets)} buckets "
            f"on rank {rank}/{world_size}"
        )

    def _create_buckets_from_ranges(self) -> None:
        """Create gradient buckets based on range buffer mapping.

        Bucket Creation Algorithm:
        1. Convert target bucket size from MB to bytes
        2. Iterate through all active buffer ranges from the range mapper
        3. Group ranges into buckets respecting size constraints
        4. Create gradient buffers for each bucket
        5. Establish parameter-to-bucket mappings

        Optimization Strategy:
        - Minimize communication overhead by grouping related parameters
        - Respect memory alignment requirements from range mapper
        - Balance bucket sizes for optimal reduction performance
        - Maintain dtype consistency within buckets

        Memory Layout:
        Each bucket contains a contiguous gradient buffer that aggregates
        gradients from multiple parameter ranges, enabling efficient
        collective communication operations.
        """
        bucket_size_bytes = int(self.bucket_size_mb * BYTES_PER_MB)

        # Group buffer ranges into buckets based on size constraints
        current_bucket_ranges: List[BufferRange] = []
        current_bucket_size = 0
        bucket_index = 0

        for buffer_range in self.range_mapper.buffer_ranges:
            if not buffer_range.is_active:
                continue

            range_size_bytes = buffer_range.size_bytes

            # Check if adding this range exceeds bucket size
            if (
                current_bucket_size > 0
                and current_bucket_size + range_size_bytes > bucket_size_bytes
            ):
                # Create bucket with current ranges
                if current_bucket_ranges:
                    self._create_bucket_from_ranges(bucket_index, current_bucket_ranges)
                    bucket_index += 1

                # Reset for next bucket
                current_bucket_ranges = []
                current_bucket_size = 0

            # Add range to current bucket
            current_bucket_ranges.append(buffer_range)
            current_bucket_size += range_size_bytes

        # Create final bucket if needed
        if current_bucket_ranges:
            self._create_bucket_from_ranges(bucket_index, current_bucket_ranges)

    def _create_bucket_from_ranges(
        self, bucket_index: int, buffer_ranges: List[BufferRange]
    ) -> None:
        """Create a bucket from a list of buffer ranges."""
        if not buffer_ranges:
            return

        # Collect parameters from all ranges
        all_params: List[Parameter] = []
        all_param_indices: List[int] = []
        total_elements = 0

        # Use the first range's dtype as the bucket dtype
        bucket_dtype = buffer_ranges[0].dtype

        for buffer_range in buffer_ranges:
            # Validate dtype consistency
            if buffer_range.dtype != bucket_dtype:
                logger.warning(
                    f"Mixed dtypes in bucket {bucket_index}: "
                    f"{bucket_dtype} and {buffer_range.dtype}"
                )
                continue

            # Add parameters from this range
            for param_idx in buffer_range.param_indices:
                if param_idx < len(self.params):
                    param = self.params[param_idx]
                    all_params.append(param)
                    all_param_indices.append(param_idx)

                    # Get parameter slice size for this rank
                    local_range = self.range_mapper.partitioner.get_local_param_range()
                    if local_range is not None:
                        param_slice = local_range.get_param_slice(
                            param_idx, param.numel()
                        )
                        if param_slice is not None:
                            start, end = param_slice
                            total_elements += end - start

        if not all_params or total_elements == 0:
            return

        # Create gradient buffer
        grad_buffer = torch.zeros(
            total_elements, dtype=bucket_dtype, device=self.device
        )

        # Create bucket with the first buffer range as representative
        bucket = RangeAwareBucket(
            index=bucket_index,
            buffer_range=buffer_ranges[0],
            params=all_params,
            param_indices=all_param_indices,
            grad_buffer=grad_buffer,
        )

        self.buckets.append(bucket)

        # Map parameters to this bucket and compute buffer offsets
        buffer_offset = 0
        for i, param in enumerate(all_params):
            param_idx = all_param_indices[i]
            self.param_to_bucket[param] = len(self.buckets) - 1

            # Get parameter slice size
            local_range = self.range_mapper.partitioner.get_local_param_range()
            if local_range is not None:
                param_slice = local_range.get_param_slice(param_idx, param.numel())
                if param_slice is not None:
                    start, end = param_slice
                    slice_size = end - start
                    self.param_to_buffer_offset[param] = (
                        buffer_offset,
                        buffer_offset + slice_size,
                    )
                    buffer_offset += slice_size

    def _register_grad_hooks(self) -> None:
        """Register gradient hooks for automatic bucket management."""
        for param in self.params:
            if not param.requires_grad:
                continue

            def make_hook(p: Parameter):
                def grad_hook(grad: Tensor) -> Tensor:
                    with self._lock:
                        # Copy gradient to bucket buffer
                        self._copy_grad_to_bucket(p, grad)
                        # Check if bucket is ready for reduction
                        self._check_bucket_ready(p)
                    return grad

                return grad_hook

            hook = make_hook(param)
            param.register_hook(hook)

    def _copy_grad_to_bucket(self, param: Parameter, grad: Tensor) -> None:
        """Copy parameter gradient to its bucket buffer."""
        if param not in self.param_to_bucket:
            logger.debug(f"Parameter not in bucket mapping, skipping")
            return

        try:
            bucket_idx = self.param_to_bucket[param]
            bucket = self.buckets[bucket_idx]
            start, end = self.param_to_buffer_offset[param]

            # Validate gradient shape
            expected_size = end - start
            if grad.numel() != expected_size:
                # Handle parameter slicing
                param_idx = self.params.index(param)
                local_range = self.range_mapper.partitioner.get_local_param_range()
                if local_range is not None:
                    param_slice = local_range.get_param_slice(param_idx, param.numel())
                    if param_slice is not None:
                        slice_start, slice_end = param_slice
                        grad_slice = grad.view(-1)[slice_start:slice_end]

                        if bucket.grad_buffer is not None:
                            bucket.grad_buffer[start:end].copy_(grad_slice)
                        else:
                            raise GradientBufferError(
                                f"Bucket {bucket_idx} has no allocated gradient buffer"
                            )
                    else:
                        logger.debug(f"No parameter slice found for param {param_idx}")
                else:
                    logger.debug(f"No local range found for current rank")
            else:
                # Direct copy for full parameter
                if bucket.grad_buffer is not None:
                    bucket.grad_buffer[start:end].copy_(grad.flatten())
                else:
                    raise GradientBufferError(
                        f"Bucket {bucket_idx} has no allocated gradient buffer"
                    )

        except Exception as e:
            logger.error(f"Failed to copy gradient to bucket: {e}")
            raise GradientBufferError(f"Gradient copy failed: {e}") from e

    def _check_bucket_ready(self, param: Parameter) -> None:
        """Check if all gradients in a bucket are ready for reduction."""
        if param not in self.param_to_bucket:
            return

        # Track this parameter as having gradient processed
        self._params_with_grads.add(param)

        bucket_idx = self.param_to_bucket[param]
        bucket = self.buckets[bucket_idx]

        # Check if all parameters in bucket have been processed
        bucket_params_processed = all(
            p in self._params_with_grads for p in bucket.params
        )
        # Also check if all parameters in bucket have gradients (double check)
        all_grads_ready = all(p.grad is not None for p in bucket.params)

        if bucket_params_processed and all_grads_ready and not bucket.is_ready:
            bucket.is_ready = True
            # Start asynchronous reduction
            self._start_bucket_reduction(bucket)

    def _start_bucket_reduction(self, bucket: RangeAwareBucket) -> None:
        """Start asynchronous gradient reduction for a bucket."""
        if bucket.grad_buffer is None or bucket.is_reduced:
            return

        start_time = time.time()

        try:
            with self._reduction_lock:
                if (
                    self.world_size > 1
                    and self.process_group is not None
                    and dist.is_initialized()
                ):
                    # Start async all-reduce
                    bucket.reduction_handle = dist.all_reduce(
                        bucket.grad_buffer,
                        op=dist.ReduceOp.SUM,
                        group=self.process_group,
                        async_op=True,
                    )
                    logger.debug(
                        f"Started async reduction for bucket {bucket.index} "
                        f"({bucket.grad_buffer.numel()} elements)"
                    )
                else:
                    # No reduction needed for single process
                    bucket.is_reduced = True

                bucket.last_sync_time = start_time

                # Update statistics
                if self.enable_profiling:
                    bytes_reduced = (
                        bucket.grad_buffer.numel() * bucket.grad_buffer.element_size()
                    )
                    self.reduction_stats.bytes_reduced += bytes_reduced
                    self.reduction_stats.total_reductions += 1

        except Exception as e:
            # Clean up reduction handle on failure to prevent resource leaks
            if bucket.reduction_handle is not None:
                try:
                    bucket.reduction_handle.wait()
                except Exception:
                    pass  # Ignore cleanup errors
                finally:
                    bucket.reduction_handle = None

            self.reduction_stats.failed_reductions += 1
            logger.error(f"Failed to start bucket reduction: {e}")
            raise CommunicationError(f"All-reduce initialization failed: {e}") from e

    def check_all_buckets_ready(self) -> None:
        """Check and update readiness status for all buckets.

        This should be called after the backward pass completes to ensure
        that all buckets that are actually ready get marked as such.
        The gradient hooks may not catch all ready buckets due to timing
        issues during the backward pass.
        """
        with self._lock:
            for bucket in self.buckets:
                if bucket.is_ready:
                    continue  # Already ready

                # Check if all parameters in bucket have been processed
                bucket_params_processed = all(
                    p in self._params_with_grads for p in bucket.params
                )

                # Check if all parameters in bucket have gradients
                all_grads_ready = all(p.grad is not None for p in bucket.params)
                if bucket_params_processed and all_grads_ready:
                    bucket.is_ready = True
                    # Start asynchronous reduction
                    self._start_bucket_reduction(bucket)

    def synchronize_bucket(
        self, bucket_idx: int, timeout_sec: float = GRADIENT_SYNC_TIMEOUT_SEC
    ) -> bool:
        """Synchronize a specific bucket and copy gradients back to parameters.

        Args:
            bucket_idx: Index of the bucket to synchronize.
            timeout_sec: Timeout for synchronization in seconds.

        Returns:
            True if synchronization succeeded, False otherwise.
        """
        if bucket_idx < 0 or bucket_idx >= len(self.buckets):
            logger.warning(f"Invalid bucket index {bucket_idx}")
            return False

        bucket = self.buckets[bucket_idx]
        start_time = time.time()

        # Wait for reduction to complete with proper resource cleanup
        if bucket.reduction_handle is not None:
            try:
                # Set timeout for wait operation
                if hasattr(bucket.reduction_handle, "wait"):
                    bucket.reduction_handle.wait()
            except Exception as e:
                elapsed = time.time() - start_time
                if elapsed > timeout_sec:
                    self.reduction_stats.timeout_count += 1
                    logger.error(
                        f"Bucket {bucket_idx} reduction timeout after {elapsed:.2f}s"
                    )
                    return False
                else:
                    self.reduction_stats.failed_reductions += 1
                    logger.error(f"Bucket {bucket_idx} reduction failed: {e}")
                    raise CommunicationError(
                        f"Reduction synchronization failed: {e}"
                    ) from e
            finally:
                # Always clean up the handle to prevent resource leaks
                bucket.reduction_handle = None

        # Get world size for gradient averaging
        world_size = (
            dist.get_world_size(self.process_group)
            if self.process_group and self.world_size > 1
            else 1
        )

        # Copy reduced gradients back to parameters
        if bucket.grad_buffer is not None:
            for param in bucket.params:
                if param in self.param_to_buffer_offset:
                    start, end = self.param_to_buffer_offset[param]

                    # Get gradient slice from buffer
                    grad_slice = bucket.grad_buffer[start:end]

                    # Average gradients if distributed
                    if world_size > 1:
                        grad_slice = grad_slice / world_size

                    # Get parameter slice information
                    param_idx = self.params.index(param)
                    local_range = self.range_mapper.partitioner.get_local_param_range()
                    if local_range is not None:
                        param_slice_info = local_range.get_param_slice(
                            param_idx, param.numel()
                        )
                        if param_slice_info is not None:
                            slice_start, slice_end = param_slice_info
                            grad_view = grad_slice.view_as(
                                param.view(-1)[slice_start:slice_end]
                            )

                            # Copy back to parameter gradient
                            if param.grad is None:
                                param.grad = torch.zeros_like(param)
                            param.grad.view(-1)[slice_start:slice_end].copy_(grad_view)

        bucket.is_reduced = True
        bucket.reduction_count += 1

        # Update timing statistics
        if self.enable_profiling:
            reduction_time = time.time() - start_time
            # Update running average
            total_reductions = self.reduction_stats.total_reductions
            if total_reductions > 0:
                current_avg = self.reduction_stats.average_reduction_time
                self.reduction_stats.average_reduction_time = (
                    current_avg * (total_reductions - 1) + reduction_time
                ) / total_reductions

        return True

    def synchronize_all_buckets(
        self, timeout_sec: float = GRADIENT_SYNC_TIMEOUT_SEC
    ) -> int:
        """Synchronize all ready buckets with automatic memory optimization.

        Synchronization Algorithm:
        1. Iterate through all buckets in dependency order
        2. For each ready bucket, wait for reduction completion
        3. Copy reduced gradients back to parameter tensors
        4. Apply gradient averaging based on world size
        5. Monitor memory fragmentation and trigger compaction if needed

        Memory Management:
        - Tracks fragmentation ratio during synchronization
        - Automatically triggers buffer compaction when threshold exceeded
        - Recreates bucket mappings after compaction for optimal layout

        Performance Optimization:
        - Overlaps communication with computation where possible
        - Batches gradient copying operations for cache efficiency
        - Uses timeout handling to prevent hanging on failed reductions

        Args:
            timeout_sec: Timeout for synchronization operations.

        Returns:
            Number of successfully synchronized buckets.
        """
        synchronized_count = 0

        for i, bucket in enumerate(self.buckets):
            if bucket.is_ready and not bucket.is_reduced:
                try:
                    if self.synchronize_bucket(i, timeout_sec):
                        synchronized_count += 1
                except Exception as e:
                    logger.error(f"Failed to synchronize bucket {i}: {e}")
                    continue

        # Adaptive memory management: check if buffer compaction is needed
        memory_stats = self.range_mapper.get_memory_stats()
        if (
            memory_stats.fragmentation_ratio
            > self.range_mapper.config.max_fragmentation
            and self.range_mapper.config.enable_compaction
        ):
            logger.debug(
                f"Triggering buffer compaction "
                f"(fragmentation: {memory_stats.fragmentation_ratio:.2%})"
            )
            if self.range_mapper.compact_buffers():
                self.reduction_stats.compaction_triggers += 1
                # Recreate buckets after compaction for optimal memory layout
                self._recreate_buckets_after_compaction()

        return synchronized_count

    def _recreate_buckets_after_compaction(self) -> None:
        """Recreate buckets after buffer compaction."""
        with self._lock:
            # Clear existing buckets
            self.buckets.clear()
            self.param_to_bucket.clear()
            self.param_to_buffer_offset.clear()
            self._params_with_grads.clear()

            # Recreate buckets from updated range mapping
            self._create_buckets_from_ranges()

            logger.debug(f"Recreated {len(self.buckets)} buckets after compaction")

    def reset(self) -> None:
        """Reset all buckets for next iteration."""
        with self._lock:
            for bucket in self.buckets:
                # Clean up any pending reduction handles before reset
                if bucket.reduction_handle is not None:
                    try:
                        bucket.reduction_handle.wait()
                    except Exception as e:
                        logger.debug(
                            f"Failed to wait for reduction handle during reset: {e}"
                        )
                    finally:
                        bucket.reduction_handle = None

                bucket.is_ready = False
                bucket.is_reduced = False
                if bucket.grad_buffer is not None:
                    bucket.grad_buffer.zero_()

            # Clear the tracking set for processed parameters
            self._params_with_grads.clear()

    def get_memory_usage(self) -> Dict[str, Union[float, int]]:
        """Get comprehensive memory usage statistics.

        Returns:
            Dictionary with memory usage information.
        """
        # Get base memory stats from range mapper
        range_stats = self.range_mapper.get_memory_stats()

        # Calculate gradient buffer memory
        grad_buffer_bytes = sum(
            bucket.grad_buffer.numel() * bucket.grad_buffer.element_size()
            for bucket in self.buckets
            if bucket.grad_buffer is not None
        )

        return {
            # Range mapper statistics
            "total_allocated_mb": range_stats.total_allocated_bytes / (1024**2),
            "total_used_mb": range_stats.total_used_bytes / (1024**2),
            "fragmentation_ratio": range_stats.fragmentation_ratio,
            "alignment_waste_mb": range_stats.alignment_waste_bytes / (1024**2),
            "peak_allocated_mb": range_stats.peak_allocated_bytes / (1024**2),
            # Gradient buffer statistics
            "gradient_buffers_mb": grad_buffer_bytes / (1024**2),
            "num_buckets": len(self.buckets),
            "num_ranges": range_stats.num_ranges,
            # Reduction statistics
            "total_reductions": self.reduction_stats.total_reductions,
            "failed_reductions": self.reduction_stats.failed_reductions,
            "average_reduction_time_ms": self.reduction_stats.average_reduction_time
            * 1000,
            "bytes_reduced_mb": self.reduction_stats.bytes_reduced / (1024**2),
            "compaction_triggers": self.reduction_stats.compaction_triggers,
        }

    def get_bucket_info(self) -> Dict[str, object]:
        """Get detailed bucket information for debugging."""
        return {
            "num_buckets": len(self.buckets),
            "bucket_size_mb": self.bucket_size_mb,
            "bucket_details": [
                {
                    "index": bucket.index,
                    "num_params": len(bucket.params),
                    "buffer_size_elements": (
                        bucket.grad_buffer.numel()
                        if bucket.grad_buffer is not None
                        else 0
                    ),
                    "buffer_dtype": str(bucket.buffer_range.dtype),
                    "is_ready": bucket.is_ready,
                    "is_reduced": bucket.is_reduced,
                    "reduction_count": bucket.reduction_count,
                    "last_sync_time": bucket.last_sync_time,
                }
                for bucket in self.buckets
            ],
            "range_mapper_info": self.range_mapper.get_buffer_info(),
        }

    def cleanup(self) -> None:
        """Clean up resources and pending operations."""
        with self._lock:
            # Wait for and clean up all pending reduction handles
            for bucket in self.buckets:
                if bucket.reduction_handle is not None:
                    try:
                        bucket.reduction_handle.wait()
                    except Exception as e:
                        logger.debug(
                            f"Error waiting for reduction handle during cleanup: {e}"
                        )
                    finally:
                        bucket.reduction_handle = None

            # Clear buckets and mappings
            self.buckets.clear()
            self.param_to_bucket.clear()
            self.param_to_buffer_offset.clear()

    def __repr__(self) -> str:
        """String representation of the range-aware gradient buffer."""
        stats = self.get_memory_usage()
        return (
            f"RangeAwareGradientBuffer("
            f"num_buckets={len(self.buckets)}, "
            f"total_mb={stats['total_allocated_mb']:.2f}, "
            f"fragmentation={stats['fragmentation_ratio']:.2%})"
        )
