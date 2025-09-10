"""
Contiguous Parameter-Gradient Buffer System

This module implements a high-performance contiguous parameter-gradient buffer system
inspired by Megatron-LM's approach. It provides:
- Unified memory management for parameters and gradients
- Automatic gradient accumulation with hooks
- Efficient memory layout with proper alignment
- Integration with async gradient allreduce
- Support for multiple precision types (fp32, fp16, bf16)

Key Features:
- Zero-copy gradient accumulation through buffer views
- Automatic gradient synchronization with backward hooks
- Memory-efficient bucketing for distributed communication
- Support for gradient clipping and scaling
- Integration with mixed precision training

References:
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- PyTorch DDP: https://pytorch.org/docs/stable/notes/ddp.html
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

logger = logging.getLogger(__name__)

# Constants
DEFAULT_BUCKET_SIZE_MB = 25.0
DEFAULT_ALIGNMENT = 128
MIN_BUCKET_SIZE_MB = 1.0
MAX_BUCKET_SIZE_MB = 100.0
BUCKET_FILL_THRESHOLD = 0.9


class BufferType(Enum):
    """Type of buffer for parameter-gradient storage."""

    PARAMETER = "parameter"
    GRADIENT = "gradient"
    OPTIMIZER_STATE = "optimizer_state"  # For future ZeRO integration
    MIXED = "mixed"  # Combined parameter and gradient


@dataclass
class BucketConfig:
    """Configuration for gradient bucketing in contiguous buffers."""

    # Bucket size in MB (recommended: 25-50 MB for good performance)
    bucket_size_mb: float = DEFAULT_BUCKET_SIZE_MB

    # Memory alignment for CUDA operations (should be power of 2)
    alignment: int = DEFAULT_ALIGNMENT

    # Whether to overlap communication with computation
    overlap_comm: bool = True

    # Whether to use separate buckets per data type
    dtype_buckets: bool = True

    # Whether to use separate buckets per device
    device_buckets: bool = True

    # Maximum number of parameters per bucket
    max_params_per_bucket: int = 100

    # Whether to enable gradient accumulation hooks
    use_gradient_hooks: bool = True

    # Whether to automatically clip gradients
    auto_clip_gradients: bool = False

    # Maximum gradient norm for clipping (if auto_clip_gradients is True)
    max_gradient_norm: float = 1.0

    def __post_init__(self):
        """Validate configuration parameters."""
        if (
            self.bucket_size_mb < MIN_BUCKET_SIZE_MB
            or self.bucket_size_mb > MAX_BUCKET_SIZE_MB
        ):
            raise ValueError(
                f"bucket_size_mb must be between {MIN_BUCKET_SIZE_MB} and "
                f"{MAX_BUCKET_SIZE_MB}, got {self.bucket_size_mb}"
            )

        if self.alignment <= 0 or (self.alignment & (self.alignment - 1)) != 0:
            raise ValueError(
                f"alignment must be a positive power of 2, got {self.alignment}"
            )

        if self.max_params_per_bucket <= 0:
            raise ValueError(
                f"max_params_per_bucket must be positive, "
                f"got {self.max_params_per_bucket}"
            )

        if self.auto_clip_gradients and self.max_gradient_norm <= 0:
            raise ValueError(
                f"max_gradient_norm must be positive when auto_clip_gradients is True, "
                f"got {self.max_gradient_norm}"
            )


class ParamGradBucket:
    """
    Manages a single parameter-gradient bucket with contiguous memory.

    This class provides a unified view of parameters and their gradients,
    enabling efficient memory access patterns and communication.
    """

    def __init__(
        self,
        bucket_id: int,
        dtype: torch.dtype,
        device: torch.device,
        bucket_size_bytes: int,
        alignment: int = DEFAULT_ALIGNMENT,
        grad_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Initialize a parameter-gradient bucket.

        Args:
            bucket_id: Unique identifier for this bucket
            dtype: Data type for parameters
            device: Device to allocate the bucket on
            bucket_size_bytes: Size of the bucket in bytes
            alignment: Memory alignment for CUDA operations
            grad_dtype: Data type for gradients (defaults to dtype)
        """
        self.bucket_id = bucket_id
        self.dtype = dtype
        self.grad_dtype = grad_dtype or dtype
        self.device = device
        self.alignment = alignment

        # Calculate number of elements based on dtype
        self.param_element_size = self._get_element_size(dtype)
        self.grad_element_size = self._get_element_size(self.grad_dtype)

        # Calculate bucket capacity
        self.param_numel = bucket_size_bytes // self.param_element_size
        self.grad_numel = bucket_size_bytes // self.grad_element_size

        # Ensure alignment
        if self.param_numel % alignment != 0:
            self.param_numel = ((self.param_numel // alignment) + 1) * alignment
        if self.grad_numel % alignment != 0:
            self.grad_numel = ((self.grad_numel // alignment) + 1) * alignment

        # Allocate contiguous buffers
        self.param_data = torch.zeros(self.param_numel, dtype=dtype, device=device)
        self.grad_data = torch.zeros(
            self.grad_numel, dtype=self.grad_dtype, device=device
        )

        # Track parameters in this bucket
        self.params: List[nn.Parameter] = []
        self.param_offsets: List[Tuple[int, int]] = []  # (start, end) offsets
        self.grad_offsets: List[Tuple[int, int]] = []  # Gradient offsets
        self.param_shapes: List[torch.Size] = []  # Original shapes

        # Current fill level
        self.param_offset = 0
        self.grad_offset = 0

        # State tracking
        self.is_full = False
        self.ready_for_comm = False

        # Communication handle for async operations
        self.comm_handle: Optional[dist.Work] = None

        # Gradient accumulation state
        self.accumulation_count = 0
        self.requires_grad_sync = False

    def _get_element_size(self, dtype: torch.dtype) -> int:
        """Get element size in bytes for a given dtype."""
        if dtype.is_floating_point:
            return int(torch.finfo(dtype).bits // 8)
        else:
            return int(torch.empty(0, dtype=dtype).element_size())

    def can_add_param(self, param: nn.Parameter) -> bool:
        """Check if a parameter can fit in this bucket."""
        if self.is_full:
            return False

        param_numel = param.numel()
        grad_numel = param.numel()  # Gradient has same shape as parameter

        # Check both parameter and gradient space
        param_fits = (self.param_offset + param_numel) <= self.param_numel
        grad_fits = (self.grad_offset + grad_numel) <= self.grad_numel

        # Check parameter count limit
        param_count_ok = len(self.params) < 100  # Reasonable limit

        return param_fits and grad_fits and param_count_ok

    def add_param(self, param: nn.Parameter) -> Tuple[int, int]:
        """
        Add a parameter to this bucket and set up gradient hook.

        Args:
            param: Parameter to add

        Returns:
            Tuple of (param_start_offset, param_end_offset)
        """
        if not self.can_add_param(param):
            raise ValueError(
                f"Parameter with {param.numel()} elements cannot fit in bucket"
            )

        param_numel = param.numel()

        # Record parameter offsets
        param_start = self.param_offset
        param_end = param_start + param_numel
        self.param_offsets.append((param_start, param_end))

        # Record gradient offsets
        grad_start = self.grad_offset
        grad_end = grad_start + param_numel
        self.grad_offsets.append((grad_start, grad_end))

        # Store original shape
        self.param_shapes.append(param.shape)

        # Copy parameter data to buffer
        with torch.no_grad():
            self.param_data[param_start:param_end].copy_(param.data.view(-1))

        # Create view for parameter (this replaces the original parameter storage)
        param.data = self.param_data[param_start:param_end].view(param.shape)

        # Add to tracked parameters
        self.params.append(param)

        # Update offsets
        self.param_offset = param_end
        self.grad_offset = grad_end

        # Check if bucket is nearly full
        param_fill_ratio = self.param_offset / self.param_numel
        grad_fill_ratio = self.grad_offset / self.grad_numel
        if max(param_fill_ratio, grad_fill_ratio) > BUCKET_FILL_THRESHOLD:
            self.is_full = True

        return param_start, param_end

    def register_gradient_hook(self, param: nn.Parameter, param_index: int) -> None:
        """
        Register a gradient accumulation hook for a parameter.

        Args:
            param: Parameter to register hook for
            param_index: Index of the parameter in this bucket
        """

        def gradient_accumulation_hook(grad: torch.Tensor) -> torch.Tensor:
            """Hook that accumulates gradients directly into the buffer."""
            if grad is None:
                return grad

            # Get gradient offsets for this parameter
            grad_start, grad_end = self.grad_offsets[param_index]

            # Accumulate gradient into buffer
            with torch.no_grad():
                if self.accumulation_count == 0:
                    # First accumulation, copy gradient
                    self.grad_data[grad_start:grad_end].copy_(grad.view(-1))
                else:
                    # Subsequent accumulation, add gradient
                    self.grad_data[grad_start:grad_end].add_(grad.view(-1))

            # Mark that gradient sync is required
            self.requires_grad_sync = True

            # Return the gradient unchanged (for other hooks)
            return grad

        # Register the hook
        param.register_hook(gradient_accumulation_hook)

    def sync_gradients_to_params(self, scale: float = 1.0) -> None:
        """
        Synchronize gradients from buffer to parameters.

        Args:
            scale: Scaling factor to apply to gradients
        """
        for i, param in enumerate(self.params):
            grad_start, grad_end = self.grad_offsets[i]
            grad_view = self.grad_data[grad_start:grad_end].view(param.shape)

            if scale != 1.0:
                grad_view = grad_view * scale

            # Convert dtype if necessary
            if grad_view.dtype != param.dtype:
                grad_view = grad_view.to(param.dtype)

            # Set or update parameter gradient
            if param.grad is None:
                param.grad = grad_view.clone()
            else:
                param.grad.copy_(grad_view)

    def zero_gradients(self) -> None:
        """Zero out all gradients in the bucket."""
        self.grad_data.zero_()
        self.accumulation_count = 0
        self.requires_grad_sync = False

    def start_all_reduce(
        self, process_group: Optional[dist.ProcessGroup] = None
    ) -> Optional[dist.Work]:
        """
        Start asynchronous all-reduce operation on gradient buffer.

        Args:
            process_group: Process group for communication

        Returns:
            Communication handle if async operation started
        """
        if not self.requires_grad_sync:
            return None

        self.comm_handle = dist.all_reduce(
            self.grad_data,
            op=dist.ReduceOp.SUM,
            group=process_group,
            async_op=True,
        )

        self.ready_for_comm = True
        return self.comm_handle

    def finish_all_reduce(self, world_size: int = 1) -> None:
        """
        Wait for all-reduce operation to complete and scale gradients.

        Args:
            world_size: Number of processes for gradient averaging
        """
        if self.comm_handle is not None:
            self.comm_handle.wait()
            self.comm_handle = None

            # Average gradients across processes
            if world_size > 1:
                self.grad_data.div_(world_size)

            self.ready_for_comm = False

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics in MB."""
        param_memory = (self.param_numel * self.param_element_size) / (1024 * 1024)
        grad_memory = (self.grad_numel * self.grad_element_size) / (1024 * 1024)

        return {
            "param_memory_mb": param_memory,
            "grad_memory_mb": grad_memory,
            "total_memory_mb": param_memory + grad_memory,
            "param_fill_ratio": self.param_offset / max(1, self.param_numel),
            "grad_fill_ratio": self.grad_offset / max(1, self.grad_numel),
        }


class ContiguousParamGradBuffer:
    """
    Manages contiguous parameter-gradient buffers for a model.

    This class orchestrates multiple buckets, handles gradient accumulation,
    and provides efficient communication for distributed training.
    """

    def __init__(
        self,
        model: nn.Module,
        bucket_config: Optional[BucketConfig] = None,
        data_parallel_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        """
        Initialize contiguous parameter-gradient buffer manager.

        Args:
            model: Model whose parameters to manage
            bucket_config: Configuration for bucketing
            data_parallel_group: Process group for gradient all-reduce
        """
        self.model = model
        self.bucket_config = bucket_config or BucketConfig()
        self.data_parallel_group = data_parallel_group

        # Get world size for gradient averaging
        self.world_size = 1
        if data_parallel_group is not None:
            self.world_size = dist.get_world_size(data_parallel_group)

        # Buckets organized by (device, dtype)
        self.buckets: Dict[Tuple[torch.device, torch.dtype], List[ParamGradBucket]] = {}

        # Parameter to bucket mapping
        self.param_to_bucket: Dict[int, ParamGradBucket] = {}  # param_id -> bucket
        self.param_to_index: Dict[int, int] = {}  # param_id -> index in bucket

        # Gradient hooks registered
        self.hooks_registered = False

        # Communication handles for async operations
        self.comm_handles: List[dist.Work] = []

        # Statistics
        self.total_params = 0
        self.total_buckets = 0

        # Initialize buffers
        self._create_buckets()

        # Register gradient hooks if configured
        if self.bucket_config.use_gradient_hooks:
            self._register_gradient_hooks()

        logger.info(
            f"ContiguousParamGradBuffer initialized: "
            f"{self.total_buckets} buckets, {self.total_params} parameters"
        )

    def _get_bucket_key(self, param: nn.Parameter) -> Tuple[torch.device, torch.dtype]:
        """Get bucket key for a parameter."""
        device = (
            param.device if self.bucket_config.device_buckets else torch.device("cpu")
        )
        dtype = param.dtype if self.bucket_config.dtype_buckets else torch.float32
        return (device, dtype)

    def _create_buckets(self) -> None:
        """Create buckets and assign parameters to them."""
        # Convert bucket size from MB to bytes
        bucket_size_bytes = int(self.bucket_config.bucket_size_mb * 1024 * 1024)

        # Group parameters by (device, dtype)
        param_groups: Dict[Tuple[torch.device, torch.dtype], List[nn.Parameter]] = {}

        for param in self.model.parameters():
            if not param.requires_grad:
                continue

            key = self._get_bucket_key(param)
            if key not in param_groups:
                param_groups[key] = []
            param_groups[key].append(param)
            self.total_params += 1

        # Create buckets for each group
        bucket_id = 0
        for key, params in param_groups.items():
            device, dtype = key
            buckets_for_key = []

            # Sort parameters by size (largest first for better packing)
            params.sort(key=lambda p: p.numel(), reverse=True)

            # Create first bucket
            current_bucket = ParamGradBucket(
                bucket_id=bucket_id,
                dtype=dtype,
                device=device,
                bucket_size_bytes=bucket_size_bytes,
                alignment=self.bucket_config.alignment,
            )
            bucket_id += 1

            # Assign parameters to buckets
            for param in params:
                # Try to add to current bucket
                if not current_bucket.can_add_param(param):
                    # Current bucket is full, save it and create new one
                    if current_bucket.params:
                        buckets_for_key.append(current_bucket)
                        self.total_buckets += 1

                    # Create new bucket
                    current_bucket = ParamGradBucket(
                        bucket_id=bucket_id,
                        dtype=dtype,
                        device=device,
                        bucket_size_bytes=bucket_size_bytes,
                        alignment=self.bucket_config.alignment,
                    )
                    bucket_id += 1

                # Add parameter to bucket
                param_index = len(current_bucket.params)
                current_bucket.add_param(param)

                # Update mappings
                param_id = id(param)
                self.param_to_bucket[param_id] = current_bucket
                self.param_to_index[param_id] = param_index

            # Add last bucket if it has parameters
            if current_bucket.params:
                buckets_for_key.append(current_bucket)
                self.total_buckets += 1

            # Store buckets for this key
            if buckets_for_key:
                self.buckets[key] = buckets_for_key

    def _register_gradient_hooks(self) -> None:
        """Register gradient accumulation hooks for all parameters."""
        if self.hooks_registered:
            return

        for param in self.model.parameters():
            if not param.requires_grad:
                continue

            param_id = id(param)
            if param_id in self.param_to_bucket:
                bucket = self.param_to_bucket[param_id]
                param_index = self.param_to_index[param_id]
                bucket.register_gradient_hook(param, param_index)

        self.hooks_registered = True
        logger.debug("Registered gradient accumulation hooks")

    def zero_gradients(self) -> None:
        """Zero all gradients in all buckets."""
        for bucket_list in self.buckets.values():
            for bucket in bucket_list:
                bucket.zero_gradients()

    def sync_gradients_to_params(self, scale: float = 1.0) -> None:
        """
        Synchronize gradients from buffers to parameters.

        Args:
            scale: Scaling factor to apply to gradients
        """
        for bucket_list in self.buckets.values():
            for bucket in bucket_list:
                bucket.sync_gradients_to_params(scale)

    def all_reduce_gradients(self, async_op: bool = True) -> Optional[List[dist.Work]]:
        """
        Perform all-reduce on gradients across all buckets.

        Args:
            async_op: Whether to perform asynchronous operation

        Returns:
            List of communication handles if async_op is True
        """
        if self.data_parallel_group is None:
            return None

        self.comm_handles.clear()

        # Start all-reduce for each bucket
        for bucket_list in self.buckets.values():
            for bucket in bucket_list:
                handle = bucket.start_all_reduce(self.data_parallel_group)
                if handle is not None:
                    self.comm_handles.append(handle)

        # If synchronous, wait for completion
        if not async_op:
            self.finish_all_reduce()
            return None

        return self.comm_handles if self.comm_handles else None

    def finish_all_reduce(self) -> None:
        """Wait for all async all-reduce operations to complete."""
        for bucket_list in self.buckets.values():
            for bucket in bucket_list:
                bucket.finish_all_reduce(self.world_size)

        # Sync gradients back to parameters
        self.sync_gradients_to_params()

        self.comm_handles.clear()

    def clip_gradients(self, max_norm: float) -> float:
        """
        Clip gradients by global norm.

        Args:
            max_norm: Maximum norm value

        Returns:
            Total norm of gradients before clipping
        """
        # Calculate global norm across all buckets
        total_norm_sq = 0.0

        for bucket_list in self.buckets.values():
            for bucket in bucket_list:
                # Compute squared L2 norm in FP32
                grad_float = bucket.grad_data.float()
                norm_sq = (grad_float * grad_float).sum().item()
                total_norm_sq += norm_sq

        total_norm = math.sqrt(total_norm_sq)

        # Clip if necessary
        if total_norm > max_norm:
            clip_scale = max_norm / total_norm
            for bucket_list in self.buckets.values():
                for bucket in bucket_list:
                    bucket.grad_data.mul_(clip_scale)

        return total_norm

    def get_memory_usage(self) -> Dict[str, Union[float, int, Dict]]:
        """Get memory usage statistics for all buckets."""
        total_param_memory = 0.0
        total_grad_memory = 0.0
        bucket_stats: Dict[str, Any] = {}

        for key, bucket_list in self.buckets.items():
            device, dtype = key
            key_str = f"{device}_{dtype}"
            bucket_stats[key_str] = []

            for bucket in bucket_list:
                stats = bucket.get_memory_usage()
                total_param_memory += stats["param_memory_mb"]
                total_grad_memory += stats["grad_memory_mb"]
                bucket_stats[key_str].append(stats)

        return {
            "total_params": self.total_params,
            "total_buckets": self.total_buckets,
            "total_param_memory_mb": total_param_memory,
            "total_grad_memory_mb": total_grad_memory,
            "total_memory_mb": total_param_memory + total_grad_memory,
            "bucket_stats": bucket_stats,
        }

    def restore_params(self) -> None:
        """
        Restore parameters to independent memory allocations.

        This breaks the dependency on the buffers and should be called
        before deallocating the buffer manager.
        """
        for param in self.model.parameters():
            if not param.requires_grad:
                continue

            param_id = id(param)
            if param_id in self.param_to_bucket:
                # Create a new tensor with the current data
                param.data = param.data.clone()

    def __del__(self):
        """Cleanup when the buffer manager is destroyed."""
        # Restore parameters to avoid dangling references
        try:
            self.restore_params()
        except:
            pass  # Ignore errors during cleanup
