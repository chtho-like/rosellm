"""
Parameter and Gradient Buffer System with Bucketing

This module implements a high-performance parameter and gradient buffer system
inspired by Megatron-LM's approach. It provides:
- Efficient memory management with contiguous buffers
- Gradient bucketing for optimized all-reduce operations
- Support for multiple precision types (fp32, fp16, bf16)
- Integration with distributed data parallel training
- Memory alignment for optimal CUDA performance

References:
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- PyTorch DDP: https://pytorch.org/docs/stable/notes/ddp.html
- NVIDIA Apex: https://github.com/NVIDIA/apex
"""

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Constants
DEFAULT_BUCKET_SIZE_MB = 40.0
DEFAULT_ALIGNMENT = 128
BUCKET_FILL_THRESHOLD = 0.9  # Consider bucket full at 90% capacity
MIN_BUCKET_SIZE_MB = 1.0
MAX_BUCKET_SIZE_MB = 100.0


# Custom Exceptions
class BufferError(Exception):
    """Base exception for buffer-related errors."""

    pass


class BucketConfigError(BufferError):
    """Exception raised for invalid bucket configuration."""

    pass


class BucketCapacityError(BufferError):
    """Exception raised when bucket capacity is exceeded."""

    pass


class CommunicationError(BufferError):
    """Exception raised for communication-related errors."""

    pass


class ParameterMappingError(BufferError):
    """Exception raised for parameter mapping errors."""

    pass


@dataclass
class BucketConfig:
    """Configuration for gradient bucketing."""

    # Bucket size in MB (recommended: 25-50 MB for good performance)
    bucket_size_mb: float = DEFAULT_BUCKET_SIZE_MB
    # Whether to use constant bucket size or adaptive sizing
    use_constant_size: bool = False
    # Alignment for CUDA operations (should be power of 2)
    alignment: int = DEFAULT_ALIGNMENT
    # Whether to overlap communication with computation
    overlap_comm: bool = True
    # NCCL communication hints
    nccl_hints: Dict[str, bool] = field(default_factory=lambda: {"use_lla": True})

    def __post_init__(self):
        """Validate configuration parameters."""
        if (
            self.bucket_size_mb < MIN_BUCKET_SIZE_MB
            or self.bucket_size_mb > MAX_BUCKET_SIZE_MB
        ):
            raise BucketConfigError(
                f"bucket_size_mb must be between {MIN_BUCKET_SIZE_MB} and "
                f"{MAX_BUCKET_SIZE_MB}, got {self.bucket_size_mb}"
            )
        if self.alignment <= 0 or (self.alignment & (self.alignment - 1)) != 0:
            raise BucketConfigError(
                f"alignment must be a positive power of 2, got {self.alignment}"
            )


class GradientBucket:
    """
    Manages a single gradient bucket for efficient all-reduce operations.
    A bucket contains gradients from multiple parameters that are communicated
    together to amortize communication overhead.
    """

    def __init__(
        self,
        bucket_id: int,
        dtype: torch.dtype,
        device: torch.device,
        bucket_size_bytes: int,
        alignment: int = 128,
    ) -> None:
        """
        Initialize a gradient bucket.

        Args:
            bucket_id: Unique identifier for this bucket
            dtype: Data type for the bucket buffer
            bucket_size_bytes: Size of the bucket in bytes
            device: Device to allocate the bucket on
            alignment: Memory alignment for CUDA operations
        """
        self.bucket_id = bucket_id
        self.dtype = dtype
        self.device = device
        self.alignment = alignment

        # Calculate number of elements based on dtype
        if dtype.is_floating_point:
            self.element_size = torch.finfo(dtype).bits // 8
        else:
            # For integer types, use the actual itemsize
            self.element_size = torch.empty(0, dtype=dtype).element_size()
        self.numel = bucket_size_bytes // self.element_size

        # Ensure alignment
        if self.numel % alignment != 0:
            self.numel = ((self.numel // alignment) + 1) * alignment

        # Allocate the bucket buffer
        self.data = torch.zeros(self.numel, dtype=dtype, device=device)
        self.grad_data = torch.zeros_like(self.data)

        # Track parameters in this bucket - use deque for efficient append
        self.params: List[torch.nn.Parameter] = []
        self.param_offsets: List[Tuple[int, int]] = []  # (start, end) offsets
        self.current_offset = 0

        # Pre-allocate space for better performance
        self.params_capacity = 100  # Initial capacity
        self.params = []
        self.param_offsets = []

        # Communication handle for async operations
        self.comm_handle: Optional[dist.Work] = None

        # State tracking
        self.is_full = False
        self.ready_for_comm = False

    def can_add_param(self, param: torch.nn.Parameter) -> bool:
        """Check if a parameter can fit in this bucket."""
        param_numel = param.numel()
        return (self.current_offset + param_numel) <= self.numel and not self.is_full

    def add_param(self, param: torch.nn.Parameter) -> Tuple[int, int]:
        """
        Add a parameter to this bucket.

        Args:
            param: Parameter to add

        Returns:
            Tuple of (start_offset, end_offset) for this parameter
        """
        if not self.can_add_param(param):
            raise BucketCapacityError(
                f"Parameter with {param.numel()} elements cannot fit in bucket"
            )

        param_numel = param.numel()
        start_offset = self.current_offset
        end_offset = start_offset + param_numel

        self.params.append(param)
        self.param_offsets.append((start_offset, end_offset))
        self.current_offset = end_offset

        # Check if bucket is nearly full
        if self.current_offset > BUCKET_FILL_THRESHOLD * self.numel:
            self.is_full = True

        return start_offset, end_offset

    def pack_gradients(self) -> None:
        """Pack parameter gradients into the bucket buffer.

        Optimized version using batched operations where possible.

        Raises:
            ValueError: If gradient shape doesn't match parameter shape
        """
        # Pack parameters with gradients
        for param, (start, end) in zip(self.params, self.param_offsets):
            if param.grad is not None:
                # Validate gradient shape matches parameter shape
                if param.grad.shape != param.shape:
                    raise ValueError(
                        f"Gradient shape {param.grad.shape} doesn't match "
                        f"parameter shape {param.shape}. This may indicate "
                        f"a model architecture change during training."
                    )

                # Validate buffer segment size matches parameter size
                expected_numel = end - start
                if param.grad.numel() != expected_numel:
                    raise ValueError(
                        f"Parameter gradient has {param.grad.numel()} elements but "
                        f"buffer segment expects {expected_numel} elements. "
                        f"This indicates a buffer allocation mismatch."
                    )

                # Use no_grad context to avoid unnecessary gradient tracking
                with torch.no_grad():
                    self.grad_data[start:end].copy_(param.grad.data.view(-1))
            else:
                self.grad_data[start:end].zero_()

        self.ready_for_comm = True

    def unpack_gradients(self, scale: float = 1.0) -> None:
        """
        Unpack gradients from the bucket buffer back to parameters.

        Args:
            scale: Scaling factor to apply to gradients (e.g., 1/world_size)
        """
        for param, (start, end) in zip(self.params, self.param_offsets):
            if param.grad is not None:
                param.grad.data = self.grad_data[start:end].view_as(param.grad) * scale

    def start_all_reduce(
        self, process_group: Optional[dist.ProcessGroup] = None
    ) -> None:
        """
        Start asynchronous all-reduce operation on this bucket.

        Args:
            process_group: Process group for communication
        """
        if not self.ready_for_comm:
            raise CommunicationError(
                "Bucket not ready for communication. Call pack_gradients first."
            )

        self.comm_handle = dist.all_reduce(
            self.grad_data,
            op=dist.ReduceOp.SUM,
            group=process_group,
            async_op=True,
        )

    def finish_all_reduce(self) -> None:
        """Wait for all-reduce operation to complete."""
        if self.comm_handle is not None:
            self.comm_handle.wait()
            self.comm_handle = None

    def reset(self) -> None:
        """Reset bucket state for next iteration."""
        self.grad_data.zero_()
        self.ready_for_comm = False


class ParamAndGradBuffer:
    """
    Manages parameter and gradient buffers with efficient memory layout.

    This class creates contiguous buffers for parameters and their gradients,
    enabling efficient communication and memory access patterns. It supports:
    - Multiple data types (fp32, fp16, bf16)
    - Gradient bucketing for optimized all-reduce
    - Memory alignment for CUDA operations
    - Integration with mixed precision training

    IMPORTANT: Buffer Lifetime Management
    -------------------------------------
    This class modifies parameter tensors to point to views of the internal buffer.
    The buffer MUST outlive all parameters that reference it. Deallocating the buffer
    while parameters still hold references will cause undefined behavior and crashes.

    Best Practices:
    - Use BufferManager's context manager for automatic cleanup
    - Never manually delete buffers while model is in use
    - Call restore_params() before buffer deallocation if needed
    """

    def __init__(
        self,
        dtype: torch.dtype,
        params: List[torch.nn.Parameter],
        data_parallel_group: Optional[dist.ProcessGroup] = None,
        bucket_config: Optional[BucketConfig] = None,
        grad_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Initialize parameter and gradient buffer.

        Args:
            dtype: Data type for parameters
            params: List of model parameters to manage
            data_parallel_group: Process group for gradient all-reduce
            bucket_config: Configuration for gradient bucketing
            grad_dtype: Data type for gradients (defaults to dtype)
        """
        self.dtype = dtype
        self.grad_dtype = grad_dtype or dtype
        self.data_parallel_group = data_parallel_group
        self.bucket_config = bucket_config or BucketConfig()

        # Validate inputs
        if not params:
            raise ParameterMappingError("params list cannot be empty")

        # Filter and sort parameters
        self.params = [p for p in params if p.requires_grad]
        # Sort by size for better bucketing (largest first)
        self.params.sort(key=lambda p: p.numel(), reverse=True)
        # Calculate total size
        self.numel = sum(p.numel() for p in self.params)
        self.numel_per_dtype: Dict[torch.dtype, int] = {}

        # Build parameter mapping
        self.param_to_buffer_index: Dict[torch.nn.Parameter, int] = {}
        self.param_offsets: List[Tuple[int, int]] = []

        # Initialize buffers
        self._allocate_buffers()

        # Create gradient buckets
        self.buckets: List[GradientBucket] = []
        self._create_buckets()

        # Statistics
        self.stats = {
            "num_params": len(self.params),
            "total_numel": self.numel,
            "num_buckets": len(self.buckets),
            "buffer_memory_mb": (self.numel * self.dtype.itemsize) / (1024 * 1024),
        }

        logger.info(f"Created ParamAndGradBuffer: {self.stats}")

    def _allocate_buffers(self) -> None:
        """Allocate contiguous buffers for parameters and gradients."""
        if self.numel == 0:
            self.param_data = torch.empty(0, dtype=self.dtype)
            self.grad_data = torch.empty(0, dtype=self.grad_dtype)
            return

        # Get device from first parameter
        device = self.params[0].device

        # Allocate parameter buffer
        self.param_data = torch.zeros(self.numel, dtype=self.dtype, device=device)

        # Allocate gradient buffer
        self.grad_data = torch.zeros(self.numel, dtype=self.grad_dtype, device=device)

        # Map parameters to buffer locations
        offset = 0
        for i, param in enumerate(self.params):
            param_numel = param.numel()
            self.param_to_buffer_index[param] = i
            self.param_offsets.append((offset, offset + param_numel))

            # Copy parameter data to buffer
            self.param_data[offset : offset + param_numel].copy_(param.data.view(-1))

            # Update parameter to point to buffer slice
            # WARNING: This creates a view into the buffer. The buffer must outlive
            # all parameters to avoid dangling references. Ensure proper cleanup
            # order and never deallocate the buffer while parameters are still in use.
            param.data = self.param_data[offset : offset + param_numel].view_as(
                param.data
            )

            offset += param_numel

    def _create_buckets(self) -> None:
        """Create gradient buckets for efficient all-reduce."""
        if self.numel == 0 or self.data_parallel_group is None:
            return

        # Convert bucket size from MB to bytes
        bucket_size_bytes = int(self.bucket_config.bucket_size_mb * 1024 * 1024)

        # Calculate elements per bucket (not currently used)
        # element_size = self.grad_dtype.itemsize
        # elements_per_bucket = bucket_size_bytes // element_size

        # Create buckets and assign parameters
        device = self.params[0].device
        bucket_id = 0

        current_bucket = GradientBucket(
            bucket_id=bucket_id,
            dtype=self.grad_dtype,
            device=device,
            bucket_size_bytes=bucket_size_bytes,
            alignment=self.bucket_config.alignment,
        )

        for param in self.params:
            if not current_bucket.can_add_param(param):
                # Finalize current bucket and create new one
                self.buckets.append(current_bucket)
                bucket_id += 1
                current_bucket = GradientBucket(
                    bucket_id=bucket_id,
                    dtype=self.grad_dtype,
                    device=device,
                    bucket_size_bytes=bucket_size_bytes,
                    alignment=self.bucket_config.alignment,
                )

            current_bucket.add_param(param)

        # Add the last bucket
        if current_bucket.params:
            self.buckets.append(current_bucket)

    def sync_params_to_buffer(self) -> None:
        """Synchronize parameters from model to buffer."""
        for param, (start, end) in zip(self.params, self.param_offsets):
            self.param_data[start:end].copy_(param.data.view(-1))

    def sync_buffer_to_params(self) -> None:
        """Synchronize parameters from buffer to model."""
        for param, (start, end) in zip(self.params, self.param_offsets):
            param.data = self.param_data[start:end].view_as(param.data)

    def sync_gradients_to_buffer(self) -> None:
        """Synchronize gradients from parameters to buffer."""
        for param, (start, end) in zip(self.params, self.param_offsets):
            if param.grad is not None:
                self.grad_data[start:end].copy_(param.grad.data.view(-1))
            else:
                self.grad_data[start:end].zero_()

    def sync_buffer_to_gradients(self, scale: float = 1.0) -> None:
        """
        Synchronize gradients from buffer to parameters.

        Args:
            scale: Scaling factor to apply to gradients
        """
        for param, (start, end) in zip(self.params, self.param_offsets):
            if param.grad is None:
                param.grad = torch.zeros_like(param.data)
            param.grad.data = self.grad_data[start:end].view_as(param.grad) * scale

    def all_reduce_gradients(
        self, async_op: bool = False
    ) -> Optional[Union[dist.Work, List[dist.Work]]]:
        """
        Perform all-reduce on gradients.

        Args:
            async_op: Whether to perform asynchronous operation

        Returns:
            Communication handle(s) if async_op=True, None otherwise
        """
        if self.data_parallel_group is None:
            return None

        # Use bucketed communication if configured
        if self.buckets and self.bucket_config.overlap_comm:
            return self._all_reduce_bucketed(async_op)
        else:
            return self._all_reduce_flat(async_op)

    def _all_reduce_flat(self, async_op: bool = False) -> Optional[dist.Work]:
        """Perform flat all-reduce on entire gradient buffer."""
        self.sync_gradients_to_buffer()

        handle = dist.all_reduce(
            self.grad_data,
            op=dist.ReduceOp.SUM,
            group=self.data_parallel_group,
            async_op=async_op,
        )

        if not async_op:
            world_size = dist.get_world_size(self.data_parallel_group)
            self.sync_buffer_to_gradients(scale=1.0 / world_size)

        return handle  # type: ignore[no-any-return]

    def _all_reduce_bucketed(self, async_op: bool = False) -> Optional[List[dist.Work]]:
        """Perform bucketed all-reduce with overlapping communication."""
        handles: List[dist.Work] = []

        # Pack gradients into buckets
        for bucket in self.buckets:
            bucket.pack_gradients()

        # Start all-reduce for each bucket
        for bucket in self.buckets:
            bucket.start_all_reduce(self.data_parallel_group)
            if async_op and bucket.comm_handle is not None:
                handles.append(bucket.comm_handle)

        # If synchronous, wait and unpack
        if not async_op:
            world_size = dist.get_world_size(self.data_parallel_group)
            for bucket in self.buckets:
                bucket.finish_all_reduce()
                bucket.unpack_gradients(scale=1.0 / world_size)
                bucket.reset()

        return handles if async_op else None

    def finish_bucketed_all_reduce(self) -> None:
        """Finish async bucketed all-reduce by unpacking gradients from buckets."""
        if not self.buckets:
            return

        world_size = dist.get_world_size(self.data_parallel_group)
        for bucket in self.buckets:
            # Wait for any pending async operations
            bucket.finish_all_reduce()
            # Unpack gradients with proper scaling
            bucket.unpack_gradients(scale=1.0 / world_size)
            # Reset bucket for next iteration
            bucket.reset()

    def zero_gradients(self) -> None:
        """Zero out all gradients in the buffer."""
        self.grad_data.zero_()
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def clip_gradients(self, max_norm: float) -> float:
        """
        Clip gradients by norm.

        Args:
            max_norm: Maximum norm value

        Returns:
            Total norm of gradients before clipping
        """
        self.sync_gradients_to_buffer()
        total_norm = torch.norm(self.grad_data, p=2).item()

        if total_norm > max_norm:
            clip_scale = max_norm / total_norm
            self.grad_data.mul_(clip_scale)
            self.sync_buffer_to_gradients()

        return float(total_norm)

    def restore_params(self) -> None:
        """
        Restore parameters to independent memory allocations.

        This method creates new independent tensors for each parameter, breaking
        the dependency on the buffer. Use this before deallocating the buffer
        to prevent dangling references.
        """
        for param, (start, end) in zip(self.params, self.param_offsets):
            # Create a new tensor with the current data
            param.data = self.param_data[start:end].view_as(param.data).clone()

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics in MB."""
        element_size = self.dtype.itemsize
        grad_element_size = self.grad_dtype.itemsize

        param_memory = (self.numel * element_size) / (1024 * 1024)
        grad_memory = (self.numel * grad_element_size) / (1024 * 1024)

        bucket_memory = 0.0
        if self.buckets:
            for bucket in self.buckets:
                bucket_memory += (bucket.numel * bucket.element_size) / (1024 * 1024)

        return {
            "param_buffer_mb": param_memory,
            "grad_buffer_mb": grad_memory,
            "bucket_memory_mb": bucket_memory,
            "total_mb": param_memory + grad_memory + bucket_memory,
        }


class BufferManager:
    """
    Manages multiple parameter and gradient buffers for different parameter groups.

    This class orchestrates multiple buffers, typically one per data type or
    parameter group, and provides a unified interface for operations.

    Supports context manager protocol for automatic resource cleanup.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        data_parallel_group: Optional[dist.ProcessGroup] = None,
        bucket_config: Optional[BucketConfig] = None,
        create_per_dtype_buffers: bool = True,
        overlap_comm: Optional[bool] = None,
    ) -> None:
        """
        Initialize buffer manager.

        Args:
            model: Model whose parameters to manage
            data_parallel_group: Process group for gradient all-reduce
            bucket_config: Configuration for gradient bucketing
            create_per_dtype_buffers: Whether to create separate buffers per dtype
            overlap_comm: Optional override for communication overlap. If None,
                preserves the value from the provided BucketConfig (or its default).
        """
        self.model = model
        self.data_parallel_group = data_parallel_group
        self.bucket_config = bucket_config or BucketConfig()
        # Respect user-provided BucketConfig.overlap_comm by only overriding
        # when an explicit override is provided via the constructor
        if overlap_comm is not None:
            self.bucket_config.overlap_comm = overlap_comm

        # Group parameters by dtype if requested
        self.buffers: Dict[str, ParamAndGradBuffer] = {}

        if create_per_dtype_buffers:
            self._create_per_dtype_buffers()
        else:
            self._create_single_buffer()

        # Communication handles for async operations
        self.comm_handles: List[dist.Work] = []

        # Statistics
        self.total_params = sum(b.stats["num_params"] for b in self.buffers.values())
        self.total_memory_mb = sum(
            b.get_memory_usage()["total_mb"] for b in self.buffers.values()
        )

        logger.info(
            f"BufferManager initialized: {len(self.buffers)} buffers, "
            f"{self.total_params} params, {self.total_memory_mb:.2f} MB"
        )

    def _create_per_dtype_buffers(self) -> None:
        """Create separate buffers for each parameter dtype."""
        dtype_params: Dict[torch.dtype, List[torch.nn.Parameter]] = {}

        # Group parameters by dtype
        for param in self.model.parameters():
            if param.requires_grad:
                dtype = param.dtype
                if dtype not in dtype_params:
                    dtype_params[dtype] = []
                dtype_params[dtype].append(param)

        # Create buffer for each dtype
        for dtype, params in dtype_params.items():
            buffer_name = f"{dtype}"
            self.buffers[buffer_name] = ParamAndGradBuffer(
                dtype=dtype,
                params=params,
                data_parallel_group=self.data_parallel_group,
                bucket_config=self.bucket_config,
            )

    def _create_single_buffer(self) -> None:
        """Create a single buffer for all parameters."""
        params = [p for p in self.model.parameters() if p.requires_grad]

        if params:
            # Use dtype of first parameter
            dtype = params[0].dtype
            self.buffers["main"] = ParamAndGradBuffer(
                dtype=dtype,
                params=params,
                data_parallel_group=self.data_parallel_group,
                bucket_config=self.bucket_config,
            )

    def sync_params_to_buffers(self) -> None:
        """Synchronize all parameters to their buffers."""
        for buffer in self.buffers.values():
            buffer.sync_params_to_buffer()

    def sync_buffers_to_params(self) -> None:
        """Synchronize all buffers to their parameters."""
        for buffer in self.buffers.values():
            buffer.sync_buffer_to_params()

    def all_reduce_gradients(self, async_op: bool = False) -> None:
        """
        Perform all-reduce on gradients across all buffers.

        Args:
            async_op: Whether to perform asynchronous operation
        """
        self.comm_handles.clear()

        for buffer in self.buffers.values():
            handle = buffer.all_reduce_gradients(async_op=async_op)
            if handle is not None:
                if isinstance(handle, list):
                    self.comm_handles.extend(handle)
                else:
                    self.comm_handles.append(handle)

    def wait_for_all_reduce(self) -> None:
        """Wait for all asynchronous all-reduce operations to complete."""
        for handle in self.comm_handles:
            if handle is not None:
                handle.wait()

        # Handle bucketed and flat all-reduce differently
        if self.comm_handles:
            for buffer in self.buffers.values():
                # If using bucketed all-reduce, finish bucket operations
                if buffer.buckets:
                    buffer.finish_bucketed_all_reduce()
                else:
                    # For flat all-reduce, sync gradients back with scaling
                    world_size = dist.get_world_size(self.data_parallel_group)
                    buffer.sync_buffer_to_gradients(scale=1.0 / world_size)

        self.comm_handles.clear()

    def zero_gradients(self) -> None:
        """Zero all gradients."""
        for buffer in self.buffers.values():
            buffer.zero_gradients()

    def clip_gradients(self, max_norm: float) -> float:
        """
        Clip gradients by global norm.

        Args:
            max_norm: Maximum norm value

        Returns:
            Total norm of gradients before clipping
        """
        # Calculate global norm across all buffers
        total_norm_sq = 0.0
        for buffer in self.buffers.values():
            buffer.sync_gradients_to_buffer()
            norm_sq = torch.norm(buffer.grad_data, p=2).item() ** 2
            total_norm_sq += norm_sq

        total_norm = math.sqrt(total_norm_sq)

        # Clip if necessary
        if total_norm > max_norm:
            clip_scale = max_norm / total_norm
            for buffer in self.buffers.values():
                buffer.grad_data.mul_(clip_scale)
                buffer.sync_buffer_to_gradients()

        return total_norm

    def get_memory_usage(self) -> Dict[str, Union[int, float, Dict]]:
        """Get memory usage statistics for all buffers."""
        stats: Dict[str, Union[int, float, Dict]] = {
            "num_buffers": len(self.buffers),
            "total_params": self.total_params,
            "total_memory_mb": self.total_memory_mb,
            "buffers": {},
        }

        buffer_stats = {}
        for name, buffer in self.buffers.items():
            buffer_stats[name] = buffer.get_memory_usage()
        stats["buffers"] = buffer_stats

        return stats

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and cleanup resources."""
        # Clear communication handles
        self.comm_handles.clear()

        # Zero gradients to free memory
        self.zero_gradients()

        # Log exit
        if exc_type is not None:
            logger.error(
                f"BufferManager exiting with exception: "
                f"{exc_type.__name__}: {exc_val}"
            )

        return False  # Don't suppress exceptions

    @contextmanager
    def gradient_accumulation_context(self, num_steps: int):
        """
        Context manager for gradient accumulation.

        Args:
            num_steps: Number of accumulation steps

        Yields:
            None (auto-scales gradients on exit). To avoid double-scaling,
            do not additionally divide the loss by num_steps when using this context.
        """
        scale = 1.0 / num_steps
        try:
            # Yield nothing (value ignored) to discourage external scaling and
            # avoid double-scaling
            yield None
        finally:
            # Ensure gradients are properly scaled after accumulation
            for buffer in self.buffers.values():
                buffer.sync_gradients_to_buffer()
                buffer.grad_data.mul_(scale)
                buffer.sync_buffer_to_gradients()
