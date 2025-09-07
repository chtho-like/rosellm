"""
Gradient Buffer Management for Distributed Training

This module implements an efficient gradient buffering system for distributed
training with gradient bucketing and asynchronous reduction capabilities.
It organizes parameters into buckets for optimized all-reduce operations,
enabling communication-computation overlap during backward pass.

Key Features:
- Dynamic gradient bucketing based on parameter sizes
- Asynchronous gradient reduction with NCCL
- Memory-efficient buffer management
- Support for mixed precision training
"""

import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Parameter

from rosellm.rosetrainer.optimizer.exceptions import (
    CommunicationError,
    ConfigurationError,
    GradientBufferError,
)

logger = logging.getLogger(__name__)

# Constants
BYTES_PER_MB = 1024 * 1024
DEFAULT_BUCKET_SIZE_MB = 25.0


@dataclass
class Bucket:
    """Container for a group of parameters sharing a gradient buffer"""

    index: int
    size: int
    dtype: torch.dtype
    params: List[Parameter]
    param_indices: List[int]  # Indices in the original parameter list
    grad_buffer: Optional[Tensor] = None
    all_reduce_handle: Optional[dist.Work] = None
    is_ready: bool = False
    is_reduced: bool = False


class GradientBuffer:
    """
    Manages gradient buffers for efficient distributed training.

    This class organizes model parameters into buckets for optimized
    gradient reduction across data parallel ranks. It supports asynchronous
    communication to overlap gradient reduction with backward computation.

    Args:
        params: List of model parameters to manage
        bucket_size_mb: Target bucket size in megabytes (default: 25MB)
        dtype: Data type for gradient buffers (default: torch.float32)
        device: Device for gradient buffers (default: cuda)
        process_group: Process group for gradient reduction
    """

    def __init__(
        self,
        params: List[Parameter],
        bucket_size_mb: float = DEFAULT_BUCKET_SIZE_MB,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        # Validate inputs
        if not params:
            raise ConfigurationError("No parameters provided to GradientBuffer")
        if bucket_size_mb <= 0:
            raise ConfigurationError(
                f"bucket_size_mb must be positive, got {bucket_size_mb}"
            )

        # Check parameter consistency
        if params:
            first_device = params[0].device
            for p in params:
                if p.device != first_device:
                    raise ConfigurationError(
                        f"All parameters must be on the same device. "
                        f"Found devices: {first_device} and {p.device}"
                    )

        self.params = list(params)
        self.bucket_size_mb = bucket_size_mb
        self.dtype = dtype
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.process_group = process_group

        # Thread safety
        self._lock = threading.Lock()

        # Calculate bucket size in elements
        self.bucket_size_bytes = int(bucket_size_mb * BYTES_PER_MB)
        self.element_size = torch.tensor([], dtype=dtype).element_size()
        self.bucket_size_elements = self.bucket_size_bytes // self.element_size

        # Initialize buckets and mappings
        self.buckets: List[Bucket] = []
        self.param_to_bucket: Dict[Parameter, int] = {}
        self.param_to_offset: Dict[Parameter, Tuple[int, int]] = {}

        # Create buckets
        self._create_buckets()

        # Register gradient hooks for automatic bucketing
        self._register_grad_hooks()

        logger.info(
            f"Created GradientBuffer with {len(self.buckets)} buckets, "
            f"bucket_size={bucket_size_mb}MB, dtype={dtype}"
        )

    def _create_buckets(self) -> None:
        """Create buckets by grouping parameters based on size constraints"""
        current_bucket_params: List[Parameter] = []
        current_bucket_indices: List[int] = []
        current_bucket_size = 0
        bucket_index = 0

        for param_idx, param in enumerate(self.params):
            if not param.requires_grad:
                continue

            param_size = param.numel()

            # Check if adding this param exceeds bucket size
            if (
                current_bucket_size > 0
                and current_bucket_size + param_size > self.bucket_size_elements
            ):
                # Create bucket with current params
                if current_bucket_params:
                    bucket = Bucket(
                        index=bucket_index,
                        size=current_bucket_size,
                        dtype=self.dtype,
                        params=current_bucket_params.copy(),
                        param_indices=current_bucket_indices.copy(),
                    )
                    self.buckets.append(bucket)
                    bucket_index += 1

                    # Map params to bucket
                    for p in current_bucket_params:
                        self.param_to_bucket[p] = len(self.buckets) - 1

                # Reset for next bucket
                current_bucket_params = []
                current_bucket_indices = []
                current_bucket_size = 0

            # Add param to current bucket
            current_bucket_params.append(param)
            current_bucket_indices.append(param_idx)
            current_bucket_size += param_size

        # Create final bucket if needed
        if current_bucket_params:
            bucket = Bucket(
                index=bucket_index,
                size=current_bucket_size,
                dtype=self.dtype,
                params=current_bucket_params.copy(),
                param_indices=current_bucket_indices.copy(),
            )
            self.buckets.append(bucket)

            # Map params to bucket
            for p in current_bucket_params:
                self.param_to_bucket[p] = len(self.buckets) - 1

        # Allocate gradient buffers and compute offsets
        for bucket in self.buckets:
            # Check for existing buffer to prevent memory leak
            if bucket.grad_buffer is None:
                bucket.grad_buffer = torch.zeros(
                    bucket.size, dtype=self.dtype, device=self.device
                )

            offset = 0
            for param in bucket.params:
                param_size = param.numel()
                self.param_to_offset[param] = (offset, offset + param_size)
                offset += param_size

    def _register_grad_hooks(self) -> None:
        """Register gradient accumulation hooks on parameters"""
        for param in self.params:
            if not param.requires_grad:
                continue

            # Fix race condition by capturing param in closure properly
            def make_hook(p: Parameter):
                def grad_hook(grad: Tensor) -> Tensor:
                    with self._lock:
                        # Copy gradient to bucket buffer
                        self._copy_grad_to_bucket(p, grad)
                        # Mark bucket as ready if all gradients received
                        self._check_bucket_ready(p)
                    return grad

                return grad_hook

            # Ensure param is captured correctly
            hook = make_hook(param)
            param.register_hook(hook)

    def _copy_grad_to_bucket(self, param: Parameter, grad: Tensor) -> None:
        """Copy parameter gradient to its bucket buffer with error handling."""
        if param not in self.param_to_bucket:
            logger.debug(f"Parameter not in bucket mapping, skipping")
            return

        try:
            bucket_idx = self.param_to_bucket[param]
            bucket = self.buckets[bucket_idx]
            start, end = self.param_to_offset[param]

            # Validate gradient shape
            if grad.numel() != param.numel():
                raise GradientBufferError(
                    f"Gradient size mismatch: expected {param.numel()}, "
                    f"got {grad.numel()}"
                )

            # Flatten and copy gradient to buffer
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
        """Check if all gradients in a bucket are ready for reduction"""
        if param not in self.param_to_bucket:
            return

        bucket_idx = self.param_to_bucket[param]
        bucket = self.buckets[bucket_idx]

        # Check if all parameters in bucket have gradients
        all_grads_ready = all(p.grad is not None for p in bucket.params)

        if all_grads_ready and not bucket.is_ready:
            bucket.is_ready = True
            # Start asynchronous all-reduce
            self._start_bucket_reduction(bucket)

    def _start_bucket_reduction(self, bucket: Bucket) -> None:
        """Start asynchronous all-reduce for a bucket with error handling."""
        if bucket.grad_buffer is None or bucket.is_reduced:
            return

        try:
            if self.process_group is not None and dist.is_initialized():
                # Start async all-reduce
                bucket.all_reduce_handle = dist.all_reduce(
                    bucket.grad_buffer, group=self.process_group, async_op=True
                )
                logger.debug(f"Started async all-reduce for bucket {bucket.index}")
            else:
                # No reduction needed for single process
                bucket.is_reduced = True

        except Exception as e:
            logger.error(f"Failed to start bucket reduction: {e}")
            raise CommunicationError(f"All-reduce initialization failed: {e}") from e

    def synchronize_bucket(self, bucket_idx: int) -> None:
        """Wait for bucket reduction to complete and copy gradients back"""
        if bucket_idx < 0 or bucket_idx >= len(self.buckets):
            logger.warning(
                f"Invalid bucket index {bucket_idx}, skipping synchronization"
            )
            return

        bucket = self.buckets[bucket_idx]

        # Wait for all-reduce to complete with timeout
        if bucket.all_reduce_handle is not None:
            try:
                bucket.all_reduce_handle.wait()
                bucket.all_reduce_handle = None
            except Exception as e:
                logger.error(f"All-reduce wait failed for bucket {bucket_idx}: {e}")
                raise CommunicationError(
                    f"All-reduce synchronization failed: {e}"
                ) from e

        # Get world size for averaging
        world_size = (
            dist.get_world_size(self.process_group) if self.process_group else 1
        )

        # Copy reduced gradients back to parameters
        if bucket.grad_buffer is not None:
            for param in bucket.params:
                start, end = self.param_to_offset[param]
                grad_view = bucket.grad_buffer[start:end].view_as(param)

                # Average gradients
                if world_size > 1:
                    grad_view.div_(world_size)

                # Copy back to parameter gradient
                if param.grad is None:
                    param.grad = grad_view.clone()
                else:
                    param.grad.copy_(grad_view)

        bucket.is_reduced = True

    def synchronize_all_buckets(self) -> None:
        """Synchronize all buckets and finalize gradient reduction"""
        for i, bucket in enumerate(self.buckets):
            if bucket.is_ready and not bucket.is_reduced:
                self.synchronize_bucket(i)

    def reset(self) -> None:
        """Reset all buckets for next iteration"""
        for bucket in self.buckets:
            bucket.is_ready = False
            bucket.is_reduced = False
            bucket.all_reduce_handle = None
            if bucket.grad_buffer is not None:
                bucket.grad_buffer.zero_()

    def get_bucket_info(self) -> Dict[str, object]:
        """Get information about bucket configuration"""
        return {
            "num_buckets": len(self.buckets),
            "bucket_size_mb": self.bucket_size_mb,
            "bucket_sizes": [b.size for b in self.buckets],
            "num_params_per_bucket": [len(b.params) for b in self.buckets],
            "total_buffer_size_mb": sum(b.size for b in self.buckets)
            * self.element_size
            / (1024 * 1024),
        }

    def __repr__(self) -> str:
        info = self.get_bucket_info()
        return (
            f"GradientBuffer(num_buckets={info['num_buckets']}, "
            f"bucket_size_mb={info['bucket_size_mb']}, "
            f"total_buffer_size_mb={info['total_buffer_size_mb']:.2f})"
        )
