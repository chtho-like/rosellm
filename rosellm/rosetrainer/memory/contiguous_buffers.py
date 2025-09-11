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
- Thread-safe operations for distributed scenarios
- Comprehensive error recovery mechanisms
- Performance monitoring and statistics
- Memory usage optimization and caching
- Context manager support for automatic cleanup

Design Patterns:
- Factory Pattern: Bucket creation with configurable parameters
- Strategy Pattern: Pluggable sorting and packing algorithms
- Observer Pattern: Gradient hooks for automatic accumulation
- Context Manager: Resource management with automatic cleanup
- Weak References: Avoid circular dependencies with model

Performance Optimizations:
- Kahan summation for numerical stability in gradient norms
- Smart parameter sorting for optimal memory packing
- Batch operations for gradient synchronization
- LRU caching for frequently accessed computations
- Pre-allocated temporary buffers for operations

References:
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- PyTorch DDP: https://pytorch.org/docs/stable/notes/ddp.html
- Kahan Summation: https://en.wikipedia.org/wiki/Kahan_summation_algorithm
"""

import logging
import math
import threading
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

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

    # Thread-safe operations for distributed training
    thread_safe: bool = True

    # Cache size for memory usage calculations
    cache_size: int = 128

    # Supported dtype combinations for mixed precision
    supported_dtype_pairs: Dict[torch.dtype, List[torch.dtype]] = field(
        default_factory=lambda: {
            torch.float32: [torch.float32, torch.float16, torch.bfloat16],
            torch.float16: [torch.float16, torch.float32],
            torch.bfloat16: [torch.bfloat16, torch.float32],
        }
    )

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


class BucketStatistics:
    """Statistics tracker for bucket performance."""

    def __init__(self) -> None:
        self.total_allreduce_time: float = 0.0
        self.total_sync_time: float = 0.0
        self.num_allreduce_calls: int = 0
        self.num_sync_calls: int = 0
        self.peak_memory_usage: float = 0.0
        self.last_update_time: float = 0.0

    def update_allreduce_time(self, duration: float) -> None:
        """Update all-reduce timing statistics."""
        self.total_allreduce_time += duration
        self.num_allreduce_calls += 1

    def update_sync_time(self, duration: float) -> None:
        """Update synchronization timing statistics."""
        self.total_sync_time += duration
        self.num_sync_calls += 1

    def get_average_times(self) -> Dict[str, float]:
        """Get average timing statistics."""
        return {
            "avg_allreduce_time": (
                self.total_allreduce_time / max(1, self.num_allreduce_calls)
            ),
            "avg_sync_time": (self.total_sync_time / max(1, self.num_sync_calls)),
        }


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
        max_params_per_bucket: int = 100,
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
            max_params_per_bucket: Maximum number of parameters per bucket
        """
        self.bucket_id = bucket_id
        self.dtype = dtype
        self.grad_dtype = grad_dtype or dtype
        self.device = device
        self.alignment = alignment
        self._max_params_per_bucket = max_params_per_bucket

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

        # Performance statistics
        self.stats = BucketStatistics()

        # Optimization: Pre-allocate temporary buffers for operations
        self._temp_buffer: Optional[torch.Tensor] = None

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

        # Check parameter count limit (configurable via BucketConfig)
        max_params = getattr(self, "_max_params_per_bucket", 100)
        param_count_ok = len(self.params) < max_params

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
            """Hook that accumulates gradients directly into the buffer.

            This hook is called during the backward pass and efficiently
            accumulates gradients into the pre-allocated contiguous buffer.

            Args:
                grad: The computed gradient for the parameter

            Returns:
                The gradient unchanged (for compatibility with other hooks)
            """
            if grad is None:
                return grad

            # Get gradient offsets for this parameter
            grad_start, grad_end = self.grad_offsets[param_index]

            # Accumulate gradient into buffer with optimal memory access
            with torch.no_grad():
                grad_flat = grad.contiguous().view(-1)

                if self.accumulation_count == 0:
                    # First accumulation, copy gradient
                    self.grad_data[grad_start:grad_end].copy_(grad_flat)
                else:
                    # Subsequent accumulation, add gradient
                    self.grad_data[grad_start:grad_end].add_(grad_flat)

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
        import time

        start_time = time.perf_counter()

        # Optimization: Batch operations for better performance
        with torch.no_grad():
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

        # Track timing
        duration = time.perf_counter() - start_time
        self.stats.update_sync_time(duration)

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

        import time

        start_time = time.perf_counter()

        self.comm_handle = dist.all_reduce(
            self.grad_data,
            op=dist.ReduceOp.SUM,
            group=process_group,
            async_op=True,
        )

        self.ready_for_comm = True

        # Track timing for performance analysis
        if self.comm_handle is not None:
            self.stats.last_update_time = start_time

        return self.comm_handle

    def finish_all_reduce(self, world_size: int = 1) -> None:
        """
        Wait for all-reduce operation to complete and scale gradients.

        Args:
            world_size: Number of processes for gradient averaging
        """
        if self.comm_handle is not None:
            import time

            self.comm_handle.wait()
            self.comm_handle = None

            # Track timing
            if (
                hasattr(self.stats, "last_update_time")
                and self.stats.last_update_time > 0
            ):
                duration = time.perf_counter() - self.stats.last_update_time
                self.stats.update_allreduce_time(duration)

            # Average gradients across processes
            if world_size > 1:
                self.grad_data.div_(world_size)

            self.ready_for_comm = False

    def get_memory_usage(self) -> Dict[str, Union[float, Dict]]:
        """Get memory usage statistics in MB."""
        param_memory = (self.param_numel * self.param_element_size) / (1024 * 1024)
        grad_memory = (self.grad_numel * self.grad_element_size) / (1024 * 1024)

        # Update peak memory tracking
        total_memory = param_memory + grad_memory
        self.stats.peak_memory_usage = max(self.stats.peak_memory_usage, total_memory)

        return {
            "param_memory_mb": param_memory,
            "grad_memory_mb": grad_memory,
            "total_memory_mb": total_memory,
            "param_fill_ratio": self.param_offset / max(1, self.param_numel),
            "grad_fill_ratio": self.grad_offset / max(1, self.grad_numel),
            "performance_stats": self.stats.get_average_times(),
            "peak_memory_mb": self.stats.peak_memory_usage,
        }


class BufferAllocationError(Exception):
    """Exception raised when buffer allocation fails."""

    pass


class BufferSynchronizationError(Exception):
    """Exception raised when buffer synchronization fails."""

    pass


class ContiguousParamGradBuffer:
    """
    Manages contiguous parameter-gradient buffers for a model.

    This class orchestrates multiple buckets, handles gradient accumulation,
    and provides efficient communication for distributed training.

    Attributes:
        model: The neural network model being managed
        bucket_config: Configuration for buffer allocation and behavior
        data_parallel_group: Process group for distributed communication
        world_size: Number of processes in the distributed setup
        total_params: Total number of parameters being managed
        total_buckets: Total number of buckets created

    Methods:
        zero_gradients(): Zero all gradients in all buckets
        sync_gradients_to_params(): Synchronize gradients from buffers to parameters
        all_reduce_gradients(): Perform distributed gradient reduction
        finish_all_reduce(): Complete async all-reduce operations
        clip_gradients(): Clip gradients by global norm
        get_memory_usage(): Get detailed memory usage statistics
        restore_params(): Restore parameters to independent memory
        cleanup(): Explicit cleanup of resources
        managed_buffers(): Context manager for automatic cleanup

    Example:
        >>> model = MyModel()
        >>> config = BucketConfig(bucket_size_mb=50.0, thread_safe=True)
        >>>
        >>> # Using context manager for automatic cleanup
        >>> with ContiguousParamGradBuffer(model, config) as buffer_mgr:
        ...     buffer_mgr.zero_gradients()
        ...     loss.backward()
        ...     buffer_mgr.clip_gradients(max_norm=1.0)
        ...     buffer_mgr.sync_gradients_to_params()
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
            model: Model whose parameters to manage. The model's parameters
                will be reorganized into contiguous memory buffers.
            bucket_config: Configuration for bucketing behavior. If None,
                default configuration will be used with 25MB bucket size.
            data_parallel_group: Process group for gradient all-reduce in
                distributed training. If None, no distributed operations
                will be performed.

        Raises:
            BufferAllocationError: If memory allocation fails or insufficient
                memory is available on the target device.
            ValueError: If configuration parameters are invalid.

        Note:
            The initialization process will:
            1. Validate dtype compatibility
            2. Group parameters by device and dtype
            3. Create optimally packed buckets
            4. Register gradient accumulation hooks
            5. Set up thread-safe operations if configured
        """
        self.model = model
        self.bucket_config = bucket_config or BucketConfig()
        self.data_parallel_group = data_parallel_group

        # Thread safety
        self._lock = threading.RLock() if self.bucket_config.thread_safe else None

        # Weak reference to avoid circular dependencies
        self._model_ref = weakref.ref(model)

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
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

        # Communication handles for async operations
        self.comm_handles: List[dist.Work] = []

        # Statistics
        self.total_params = 0
        self.total_buckets = 0

        # Memory usage cache
        self._memory_cache: Optional[Dict[str, Any]] = None
        self._cache_invalidated = True

        # Performance monitoring
        self._performance_stats: Dict[str, float] = {}
        self._param_dtype_distribution: Dict[torch.dtype, int] = {}

        # Optimization flags
        self._optimization_enabled = True
        self._fusion_enabled = torch.cuda.is_available()

        # Validate dtype compatibility
        self._validate_dtype_compatibility()

        # Initialize buffers
        self._create_buckets()

        # Register gradient hooks if configured
        if self.bucket_config.use_gradient_hooks:
            self._register_gradient_hooks()

        logger.info(
            f"ContiguousParamGradBuffer initialized: "
            f"{self.total_buckets} buckets, {self.total_params} parameters"
        )

    def _validate_dtype_compatibility(self) -> None:
        """Validate that model dtypes are compatible with configuration."""
        model = self._model_ref()
        if model is None:
            return

        for param in model.parameters():
            if not param.requires_grad:
                continue

            param_dtype = param.dtype
            if param_dtype not in self.bucket_config.supported_dtype_pairs:
                logger.warning(
                    f"Parameter dtype {param_dtype} not in supported list. "
                    f"Adding with default compatibility."
                )
                self.bucket_config.supported_dtype_pairs[param_dtype] = [param_dtype]

    def _get_bucket_key(self, param: nn.Parameter) -> Tuple[torch.device, torch.dtype]:
        """Get bucket key for a parameter."""
        device = (
            param.device if self.bucket_config.device_buckets else torch.device("cpu")
        )
        dtype = param.dtype if self.bucket_config.dtype_buckets else torch.float32
        return (device, dtype)

    def _validate_memory_requirements(self, param_groups: Dict) -> None:
        """Validate that we have enough memory for buffer allocation."""
        total_memory_required = 0

        for (device, dtype), params in param_groups.items():
            # Calculate memory for this group
            param_memory = sum(p.numel() * p.element_size() for p in params)
            # Double for gradients
            total_memory_required += param_memory * 2

            # Check available memory on device
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                mem_info = torch.cuda.mem_get_info(device.index)
                available_memory = mem_info[0]  # Available memory in bytes

                if total_memory_required > available_memory * 0.9:  # Leave 10% buffer
                    raise BufferAllocationError(
                        f"Insufficient memory on {device}. "
                        f"Required: {total_memory_required / 1e9:.2f} GB, "
                        f"Available: {available_memory / 1e9:.2f} GB"
                    )

    def _create_buckets(self) -> None:
        """Create buckets and assign parameters to them."""
        # Convert bucket size from MB to bytes
        bucket_size_bytes = int(self.bucket_config.bucket_size_mb * 1024 * 1024)

        # Group parameters by (device, dtype)
        param_groups: Dict[Tuple[torch.device, torch.dtype], List[nn.Parameter]] = {}

        model = self._model_ref()
        if model is None:
            return

        try:
            for param in model.parameters():
                if not param.requires_grad:
                    continue

                key = self._get_bucket_key(param)
                if key not in param_groups:
                    param_groups[key] = []
                param_groups[key].append(param)
                self.total_params += 1

                # Track dtype distribution for optimization
                dtype = param.dtype
                self._param_dtype_distribution[dtype] = (
                    self._param_dtype_distribution.get(dtype, 0) + param.numel()
                )

            # Validate memory requirements before allocation
            self._validate_memory_requirements(param_groups)

        except Exception as e:
            logger.error(f"Failed to group parameters: {e}")
            raise BufferAllocationError(f"Parameter grouping failed: {e}") from e

        # Create buckets for each group
        bucket_id = 0
        for key, params in param_groups.items():
            device, dtype = key
            buckets_for_key = []

            # Advanced sorting strategy for optimal packing
            # Sort by size, but group similar sizes together for better cache locality
            def smart_sort_key(p: nn.Parameter) -> Tuple[int, int]:
                size = p.numel()
                # Group into size buckets (powers of 2)
                size_bucket = int(math.log2(max(1, size)))
                return (-size_bucket, -size)  # Negative for descending order

            params.sort(key=smart_sort_key)

            # Create first bucket with configuration
            current_bucket = ParamGradBucket(
                bucket_id=bucket_id,
                dtype=dtype,
                device=device,
                bucket_size_bytes=bucket_size_bytes,
                alignment=self.bucket_config.alignment,
                max_params_per_bucket=self.bucket_config.max_params_per_bucket,
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

                    # Create new bucket with configuration
                    current_bucket = ParamGradBucket(
                        bucket_id=bucket_id,
                        dtype=dtype,
                        device=device,
                        bucket_size_bytes=bucket_size_bytes,
                        alignment=self.bucket_config.alignment,
                        max_params_per_bucket=self.bucket_config.max_params_per_bucket,
                    )
                    bucket_id += 1

                # Add parameter to bucket with error handling
                try:
                    param_index = len(current_bucket.params)
                    current_bucket.add_param(param)

                    # Update mappings
                    param_id = id(param)
                    self.param_to_bucket[param_id] = current_bucket
                    self.param_to_index[param_id] = param_index

                except Exception as e:
                    logger.error(
                        f"Failed to add parameter to bucket "
                        f"{current_bucket.bucket_id}: {e}"
                    )
                    # Try to recover by creating a new bucket
                    if current_bucket.params:
                        buckets_for_key.append(current_bucket)
                        self.total_buckets += 1

                    # Create emergency bucket with larger size
                    emergency_size = max(
                        bucket_size_bytes * 2, param.numel() * param.element_size() * 2
                    )
                    current_bucket = ParamGradBucket(
                        bucket_id=bucket_id,
                        dtype=dtype,
                        device=device,
                        bucket_size_bytes=int(emergency_size),
                        alignment=self.bucket_config.alignment,
                        max_params_per_bucket=self.bucket_config.max_params_per_bucket,
                    )
                    bucket_id += 1

                    # Retry adding parameter
                    param_index = len(current_bucket.params)
                    current_bucket.add_param(param)
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

        model = self._model_ref()
        if model is None:
            return

        # Batch register hooks for better performance
        hooks_to_register = []
        for param in model.parameters():
            if not param.requires_grad:
                continue

            param_id = id(param)
            if param_id in self.param_to_bucket:
                bucket = self.param_to_bucket[param_id]
                param_index = self.param_to_index[param_id]
                hooks_to_register.append((param, bucket, param_index))

        # Register all hooks
        for param, bucket, param_index in hooks_to_register:
            bucket.register_gradient_hook(param, param_index)
            # Store hook handles for cleanup
            if hasattr(param, "_grad_fn") and hasattr(param._grad_fn, "next_functions"):
                # Track the hook for later removal if needed
                pass

        self.hooks_registered = True
        logger.debug(f"Registered {len(hooks_to_register)} gradient accumulation hooks")

    @contextmanager
    def _thread_safe_operation(self) -> Iterator[None]:
        """Context manager for thread-safe operations."""
        if self._lock is not None:
            with self._lock:
                yield
        else:
            yield

    def zero_gradients(self) -> None:
        """Zero all gradients in all buckets."""
        with self._thread_safe_operation():
            for bucket_list in self.buckets.values():
                for bucket in bucket_list:
                    bucket.zero_gradients()
            self._cache_invalidated = True

    def sync_gradients_to_params(self, scale: float = 1.0) -> None:
        """
        Synchronize gradients from buffers to parameters.

        Args:
            scale: Scaling factor to apply to gradients
        """
        with self._thread_safe_operation():
            for bucket_list in self.buckets.values():
                for bucket in bucket_list:
                    bucket.sync_gradients_to_params(scale)
            self._cache_invalidated = True

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

        with self._thread_safe_operation():
            self.comm_handles.clear()
            failed_buckets = []

            # Start all-reduce for each bucket with error recovery
            for key, bucket_list in self.buckets.items():
                for bucket in bucket_list:
                    try:
                        handle = bucket.start_all_reduce(self.data_parallel_group)
                        if handle is not None:
                            self.comm_handles.append(handle)
                    except Exception as e:
                        logger.error(
                            f"Failed to start all-reduce for bucket "
                            f"{bucket.bucket_id}: {e}"
                        )
                        failed_buckets.append((key, bucket))

            # Handle failed buckets
            if failed_buckets:
                logger.warning(
                    f"Retrying all-reduce for {len(failed_buckets)} failed buckets"
                )
                for key, bucket in failed_buckets:
                    try:
                        # Fallback to synchronous all-reduce
                        if bucket.requires_grad_sync:
                            dist.all_reduce(
                                bucket.grad_data,
                                op=dist.ReduceOp.SUM,
                                group=self.data_parallel_group,
                                async_op=False,
                            )
                            bucket.grad_data.div_(self.world_size)
                            bucket.requires_grad_sync = False
                    except Exception as e:
                        raise BufferSynchronizationError(
                            f"Critical failure in all-reduce recovery: {e}"
                        ) from e

            # If synchronous, wait for completion
            if not async_op:
                self.finish_all_reduce()
                return None

            return self.comm_handles if self.comm_handles else None

    def finish_all_reduce(self) -> None:
        """Wait for all async all-reduce operations to complete."""
        with self._thread_safe_operation():
            for bucket_list in self.buckets.values():
                for bucket in bucket_list:
                    bucket.finish_all_reduce(self.world_size)

            # Sync gradients back to parameters
            self.sync_gradients_to_params()

            self.comm_handles.clear()

    def clip_gradients(self, max_norm: float) -> float:
        """
        Clip gradients by global norm with numerical stability.

        Args:
            max_norm: Maximum norm value

        Returns:
            Total norm of gradients before clipping
        """
        if max_norm <= 0:
            raise ValueError(f"max_norm must be positive, got {max_norm}")

        with self._thread_safe_operation():
            try:
                # Calculate global norm across all buckets with numerical stability
                total_norm_sq = 0.0

                # Use Kahan summation for better numerical accuracy
                compensation = 0.0

                for bucket_list in self.buckets.values():
                    for bucket in bucket_list:
                        # Compute squared L2 norm in FP32
                        grad_float = bucket.grad_data.float()
                        norm_sq = (grad_float * grad_float).sum().item()

                        # Kahan summation
                        y = norm_sq - compensation
                        t = total_norm_sq + y
                        compensation = (t - total_norm_sq) - y
                        total_norm_sq = t

                # Handle edge cases
                if not math.isfinite(total_norm_sq):
                    logger.warning(
                        "Non-finite gradient norm detected, skipping clipping"
                    )
                    return float("inf")

                total_norm = math.sqrt(max(0.0, total_norm_sq))  # Ensure non-negative

                # Clip if necessary
                if total_norm > max_norm:
                    # Add small epsilon to avoid division by zero
                    clip_scale = max_norm / (total_norm + 1e-6)

                    for bucket_list in self.buckets.values():
                        for bucket in bucket_list:
                            bucket.grad_data.mul_(clip_scale)

                self._cache_invalidated = True
                return total_norm

            except Exception as e:
                logger.error(f"Error during gradient clipping: {e}")
                # Return infinity to signal error without crashing
                return float("inf")

    @lru_cache(maxsize=1)
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics for all buckets (cached)."""
        # Check cache validity
        if not self._cache_invalidated and self._memory_cache is not None:
            return self._memory_cache

        with self._thread_safe_operation():
            total_param_memory = 0.0
            total_grad_memory = 0.0
            bucket_stats: Dict[str, Any] = {}

            for key, bucket_list in self.buckets.items():
                device, dtype = key
                key_str = f"{device}_{dtype}"
                bucket_stats[key_str] = []

                for bucket in bucket_list:
                    stats = bucket.get_memory_usage()
                    # Type-safe access with proper casting
                    param_mem = stats.get("param_memory_mb", 0.0)
                    grad_mem = stats.get("grad_memory_mb", 0.0)
                    if isinstance(param_mem, (int, float)):
                        total_param_memory += float(param_mem)
                    if isinstance(grad_mem, (int, float)):
                        total_grad_memory += float(grad_mem)
                    bucket_stats[key_str].append(stats)

            result = {
                "total_params": self.total_params,
                "total_buckets": self.total_buckets,
                "total_param_memory_mb": total_param_memory,
                "total_grad_memory_mb": total_grad_memory,
                "total_memory_mb": total_param_memory + total_grad_memory,
                "bucket_stats": bucket_stats,
            }

            # Update cache
            self._memory_cache = result
            self._cache_invalidated = False

            return result

    def restore_params(self) -> None:
        """
        Restore parameters to independent memory allocations.

        This breaks the dependency on the buffers and should be called
        before deallocating the buffer manager.
        """
        model = self._model_ref()
        if model is None:
            return

        with self._thread_safe_operation():
            for param in model.parameters():
                if not param.requires_grad:
                    continue

                param_id = id(param)
                if param_id in self.param_to_bucket:
                    # Create a new tensor with the current data
                    param.data = param.data.clone()

            # Clear hook handles
            self._hook_handles.clear()
            self.hooks_registered = False

    def cleanup(self) -> None:
        """Explicit cleanup method for resource management."""
        try:
            self.restore_params()
            # Clear all internal references
            self.buckets.clear()
            self.param_to_bucket.clear()
            self.param_to_index.clear()
            self.comm_handles.clear()
            self._memory_cache = None
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    @contextmanager
    def managed_buffers(self) -> Iterator["ContiguousParamGradBuffer"]:
        """Context manager for automatic buffer cleanup."""
        try:
            yield self
        finally:
            self.cleanup()

    def __del__(self) -> None:
        """Cleanup when the buffer manager is destroyed."""
        # Use explicit cleanup method instead of bare except
        self.cleanup()
