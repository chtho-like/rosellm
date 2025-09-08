"""Decoupled gradient storage for memory-efficient training.

This module provides gradient storage decoupled from parameters, enabling:
- Memory-efficient gradient accumulation
- Delayed gradient updates for large models
- Integration with distributed optimizers
- Support for mixed precision training
"""

import logging
import threading
import warnings
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

# Constants for memory management
DEFAULT_MIN_BUFFER_MB = 1.0
DEFAULT_MAX_BUFFER_MB = 1024.0
DEFAULT_GROWTH_FACTOR = 1.5
BYTES_PER_MB = 1024 * 1024


class StorageMode(Enum):
    """Storage mode for gradient buffers."""

    CONTIGUOUS = "contiguous"  # Single contiguous buffer
    INDIVIDUAL = "individual"  # Separate buffers per parameter
    CHUNKED = "chunked"  # Chunked allocation for large models


@dataclass(init=False)
class DecoupledGradientConfig:
    """Configuration for decoupled gradient storage.

    Attributes:
        enabled: Whether to use decoupled gradient storage.
        dtype: Data type for gradient storage (default: torch.float32).
        device: Device for gradient storage ('cuda' or 'cpu').
        storage_mode: Storage mode for gradient buffers.
        persistent_storage: Keep gradients in storage across steps.
        lazy_init: Initialize storage only when gradients are computed.
        buffer_growth_factor: Growth factor for dynamic buffer resizing.
        min_buffer_size_mb: Minimum buffer size in MB.
        max_buffer_size_mb: Maximum buffer size in MB.
        enable_profiling: Enable memory profiling for gradients.
        use_pinned_memory: Use pinned memory for CPU storage.
        async_gpu_transfer: Enable async GPU transfers.
        thread_safe: Enable thread-safe operations.
        debug_mode: Enable debug mode with additional checks.
        memory_efficient_hooks: Use memory-efficient gradient hooks.
        validate_dtypes: Validate gradient dtype compatibility.
    """

    enabled: bool = True
    dtype: torch.dtype = torch.float32
    device: str = "cuda"
    storage_mode: StorageMode = StorageMode.CONTIGUOUS
    persistent_storage: bool = True
    lazy_init: bool = False
    buffer_growth_factor: float = DEFAULT_GROWTH_FACTOR
    min_buffer_size_mb: float = DEFAULT_MIN_BUFFER_MB
    max_buffer_size_mb: float = DEFAULT_MAX_BUFFER_MB
    enable_profiling: bool = False
    use_pinned_memory: bool = False
    async_gpu_transfer: bool = True
    thread_safe: bool = True
    debug_mode: bool = False
    memory_efficient_hooks: bool = True
    validate_dtypes: bool = True

    def __init__(self, **kwargs: Any) -> None:
        """Initialize config with backward compatibility."""
        # Handle deprecated contiguous_buffer parameter
        if "contiguous_buffer" in kwargs:
            contiguous_buffer = kwargs.pop("contiguous_buffer")
            if "storage_mode" not in kwargs:
                kwargs["storage_mode"] = (
                    StorageMode.CONTIGUOUS
                    if contiguous_buffer
                    else StorageMode.INDIVIDUAL
                )
            warnings.warn(
                "contiguous_buffer is deprecated, use storage_mode instead",
                DeprecationWarning,
                stacklevel=2,
            )

        # Set defaults for dataclass fields
        for field_name, field_def in self.__dataclass_fields__.items():
            if field_name in kwargs:
                setattr(self, field_name, kwargs[field_name])
            elif hasattr(field_def, "default"):
                setattr(self, field_name, field_def.default)
            elif hasattr(field_def, "default_factory"):
                factory = field_def.default_factory
                if callable(factory):
                    setattr(self, field_name, factory())

        # Call post_init for validation
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""

        # Validate configuration values
        if self.buffer_growth_factor <= 1.0:
            raise ValueError(
                f"buffer_growth_factor must be > 1.0, got {self.buffer_growth_factor}"
            )

        if self.min_buffer_size_mb <= 0:
            raise ValueError(
                f"min_buffer_size_mb must be > 0, got {self.min_buffer_size_mb}"
            )

        if self.max_buffer_size_mb < self.min_buffer_size_mb:
            raise ValueError(
                f"max_buffer_size_mb ({self.max_buffer_size_mb}) must be >= "
                f"min_buffer_size_mb ({self.min_buffer_size_mb})"
            )

        if self.device not in ["cuda", "cpu"]:
            raise ValueError(f"device must be 'cuda' or 'cpu', got {self.device}")

        # Validate dtype is supported
        supported_dtypes = {torch.float16, torch.float32, torch.float64, torch.bfloat16}
        if self.dtype not in supported_dtypes:
            raise ValueError(
                f"Unsupported dtype {self.dtype}. Supported: {supported_dtypes}"
            )

        # Warn about potential issues
        if self.async_gpu_transfer and not torch.cuda.is_available():
            warnings.warn(
                "async_gpu_transfer is enabled but CUDA is not available",
                RuntimeWarning,
            )

        if self.use_pinned_memory and self.device == "cuda":
            warnings.warn(
                "use_pinned_memory is typically used with CPU storage", UserWarning
            )

    @property
    def contiguous_buffer(self) -> bool:
        """Backward compatibility property for contiguous_buffer."""
        return self.storage_mode == StorageMode.CONTIGUOUS


class DecoupledGradientBuffer:
    """Manages decoupled gradient storage for a set of parameters.

    This class provides:
    - Gradient storage separate from parameters
    - Automatic backward hooks for gradient capture
    - Memory-efficient buffer management
    - Support for gradient accumulation
    - Thread-safe operations when enabled
    - Optimized memory transfers
    """

    def __init__(
        self,
        parameters: List[nn.Parameter],
        config: DecoupledGradientConfig,
        param_group_id: Optional[int] = None,
    ):
        """Initialize decoupled gradient buffer.

        Args:
            parameters: List of parameters to manage.
            config: Configuration for gradient storage.
            param_group_id: Optional ID for parameter group.

        Raises:
            ValueError: If parameters list is empty.
            RuntimeError: If memory allocation fails.
        """
        if not parameters:
            raise ValueError("Parameters list cannot be empty")

        self.config = config
        self.param_group_id = param_group_id
        self.parameters = parameters
        self.param_to_index: Dict[nn.Parameter, int] = {}
        self.hooks: List[Any] = []
        self._lock = threading.RLock() if config.thread_safe else None
        self._initialized = False
        self._released = False

        # Calculate total number of elements and validate parameters
        self.total_numel = 0
        self.param_shapes: List[torch.Size] = []
        self.param_offsets: List[int] = []
        self.param_dtypes: List[torch.dtype] = []

        for i, param in enumerate(parameters):
            if not isinstance(param, nn.Parameter):
                raise TypeError(f"Expected nn.Parameter, got {type(param)}")

            self.param_to_index[param] = i
            self.param_shapes.append(param.shape)
            self.param_dtypes.append(param.dtype)
            self.param_offsets.append(self.total_numel)
            self.total_numel += param.numel()

        # Initialize gradient storage
        self.gradient_buffer: Optional[Tensor] = None
        self.buffer_device: Optional[torch.device] = None

        # Hook optimization: pre-create hook functions if memory-efficient mode
        self._hook_functions: Dict[int, Callable] = {}
        if config.memory_efficient_hooks:
            self._prepare_hook_functions()

        # Statistics
        self.accumulation_count = 0
        self.bytes_allocated = 0
        self.peak_memory_mb = 0.0
        self.total_gradient_updates = 0
        self.cache_hits = 0
        self.cache_misses = 0

        # Initialize buffer if not lazy
        if not config.lazy_init:
            self._initialize_buffer()

        # Register backward hooks
        self._register_hooks()

    @contextmanager
    def _thread_safe_context(self) -> Iterator[None]:
        """Context manager for thread-safe operations."""
        if self._lock:
            self._lock.acquire()
        try:
            yield
        finally:
            if self._lock:
                self._lock.release()

    def _prepare_hook_functions(self) -> None:
        """Pre-create hook functions for memory efficiency."""
        weak_self = weakref.ref(self)

        for idx in range(len(self.parameters)):

            def make_hook(param_idx: int) -> Callable[[Tensor], Tensor]:
                def hook(grad: Tensor) -> Tensor:
                    self_ref = weak_self()
                    if self_ref is not None and not self_ref._released:
                        return self_ref._gradient_hook(grad, param_idx)
                    return grad

                return hook

            self._hook_functions[idx] = make_hook(idx)

    def _initialize_buffer(self) -> None:
        """Initialize the gradient buffer with proper error handling.

        Raises:
            RuntimeError: If buffer allocation fails.
        """
        with self._thread_safe_context():
            if self.gradient_buffer is not None or self._released:
                return

            try:
                # Determine device
                device = self._determine_device()
                self.buffer_device = device

                # Calculate required memory
                required_bytes = (
                    self.total_numel * torch.finfo(self.config.dtype).bits // 8
                )
                required_mb = required_bytes / BYTES_PER_MB

                if required_mb > self.config.max_buffer_size_mb:
                    raise RuntimeError(
                        f"Required buffer size ({required_mb:.2f} MB) exceeds "
                        f"maximum allowed ({self.config.max_buffer_size_mb:.2f} MB)"
                    )

                if self.config.debug_mode:
                    logger.debug(
                        f"Allocating gradient buffer: {required_mb:.2f} MB on {device}"
                    )

                # Allocate buffer based on storage mode
                if self.config.storage_mode == StorageMode.CONTIGUOUS:
                    self._allocate_contiguous_buffer(device)
                elif self.config.storage_mode == StorageMode.INDIVIDUAL:
                    self._allocate_individual_buffers(device)
                elif self.config.storage_mode == StorageMode.CHUNKED:
                    self._allocate_chunked_buffer(device)
                else:
                    raise ValueError(
                        f"Unknown storage mode: {self.config.storage_mode}"
                    )

                self._initialized = True

                if self.config.enable_profiling:
                    self.peak_memory_mb = max(
                        self.peak_memory_mb, self.bytes_allocated / BYTES_PER_MB
                    )
                    logger.info(
                        f"Initialized gradient buffer for group {self.param_group_id}: "
                        f"{self.bytes_allocated / BYTES_PER_MB:.2f} MB on {device}"
                    )

            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                self._cleanup_failed_allocation()
                error_msg = (
                    f"Failed to allocate gradient buffer: {e}. "
                    f"Required: {required_mb:.2f} MB on {device}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

    def _determine_device(self) -> torch.device:
        """Determine the appropriate device for gradient storage."""
        device = self.config.device

        if device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                return torch.device("cpu")

            # Use same device as first parameter if on CUDA
            if self.parameters and self.parameters[0].is_cuda:
                return self.parameters[0].device
            else:
                return torch.device("cuda:0")

        return torch.device("cpu")

    def _allocate_contiguous_buffer(self, device: torch.device) -> None:
        """Allocate a single contiguous buffer."""
        pin_memory = self.config.use_pinned_memory and device.type == "cpu"

        self.gradient_buffer = torch.zeros(
            self.total_numel,
            dtype=self.config.dtype,
            device=device,
            pin_memory=pin_memory,
        )

        self.bytes_allocated = (
            self.gradient_buffer.numel() * self.gradient_buffer.element_size()
        )

    def _allocate_individual_buffers(self, device: torch.device) -> None:
        """Allocate individual buffers with view-based optimization.

        Uses a single contiguous buffer with views for efficiency while
        maintaining logical separation of parameter gradients.
        """
        # Use contiguous buffer with views for cache efficiency
        # This provides the benefits of individual buffers (logical separation)
        # with the performance benefits of contiguous memory (cache locality)
        self._allocate_contiguous_buffer(device)

    def _allocate_chunked_buffer(self, device: torch.device) -> None:
        """Allocate chunked buffer for very large models.

        Uses a chunked allocation strategy to handle very large models
        that might exceed contiguous memory limits.
        """
        # Calculate optimal chunk size based on available memory
        # chunk_size_mb = min(256.0, self.config.max_buffer_size_mb / 4)
        # chunk_size_elements for future chunked implementation
        # int(chunk_size_mb * BYTES_PER_MB / torch.finfo(self.config.dtype).bits * 8)

        # For now, use contiguous allocation with memory-aware sizing
        # Future: implement true chunked storage with scatter-gather
        try:
            self._allocate_contiguous_buffer(device)
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            # If contiguous allocation fails, could implement fallback
            logger.warning(
                f"Contiguous allocation failed, falling back to standard allocation. "
                f"Consider reducing buffer size or using CPU storage."
            )
            raise

    def _cleanup_failed_allocation(self) -> None:
        """Clean up after failed buffer allocation."""
        self.gradient_buffer = None
        self.buffer_device = None
        self.bytes_allocated = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _register_hooks(self) -> None:
        """Register backward hooks to capture gradients efficiently."""
        with self._thread_safe_context():
            if self._released:
                return

            for param in self.parameters:
                if param.requires_grad:
                    param_idx = self.param_to_index[param]

                    # Use pre-created hook function if available
                    if (
                        self.config.memory_efficient_hooks
                        and param_idx in self._hook_functions
                    ):
                        hook_fn = self._hook_functions[param_idx]
                    else:
                        # Fallback to creating hook on-the-fly
                        weak_self = weakref.ref(self)

                        def make_hook(idx: int) -> Callable[[Tensor], Tensor]:
                            def hook(grad: Tensor) -> Tensor:
                                self_ref = weak_self()
                                if self_ref is not None and not self_ref._released:
                                    return self_ref._gradient_hook(grad, idx)
                                return grad

                            return hook

                        hook_fn = make_hook(param_idx)

                    try:
                        handle = param.register_hook(hook_fn)
                        self.hooks.append(handle)
                    except RuntimeError as e:
                        logger.warning(f"Failed to register hook for parameter: {e}")

    def _gradient_hook(self, grad: Tensor, param_idx: int) -> Tensor:
        """Hook to capture and store gradients with optimizations.

        Args:
            grad: Gradient tensor from backward pass.
            param_idx: Index of parameter in the buffer.

        Returns:
            Modified gradient (or original if not storing).
        """
        if not self.config.enabled or self._released:
            return grad

        with self._thread_safe_context():
            # Initialize buffer if needed (lazy initialization)
            if self.gradient_buffer is None:
                self._initialize_buffer()

            # Validate gradient dtype if enabled
            if self.config.validate_dtypes:
                # param = self.parameters[param_idx]  # for future dtype validation
                expected_dtype = self.param_dtypes[param_idx]
                if grad.dtype != expected_dtype and self.config.debug_mode:
                    logger.warning(
                        f"Gradient dtype mismatch for param {param_idx}: "
                        f"expected {expected_dtype}, got {grad.dtype}"
                    )

            # Get parameter info
            offset = self.param_offsets[param_idx]
            numel = self.parameters[param_idx].numel()

            # Store gradient in buffer
            if self.gradient_buffer is not None:
                try:
                    grad_view = self.gradient_buffer[offset : offset + numel].view_as(
                        self.parameters[param_idx]
                    )

                    if self.config.persistent_storage and self.accumulation_count > 0:
                        # Accumulate gradients
                        grad_view.add_(grad)
                    else:
                        # Optimize transfer based on device configuration
                        self._optimized_gradient_copy(grad_view, grad)

                    self.total_gradient_updates += 1

                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    if self.config.debug_mode:
                        logger.error(f"Failed to store gradient: {e}")
                    return grad  # Return original gradient on failure

            # Return zero tensor to free original gradient memory
            # Use zeros_like with correct device to avoid transfers
            if self.config.enabled:
                return torch.zeros_like(
                    grad, device=grad.device, memory_format=torch.preserve_format
                )

            return grad

    def _optimized_gradient_copy(self, dest: Tensor, src: Tensor) -> None:
        """Perform optimized gradient copy based on device configuration.

        Args:
            dest: Destination tensor (in buffer).
            src: Source tensor (gradient).
        """
        if (
            self.config.async_gpu_transfer
            and src.is_cuda
            and self.buffer_device
            and self.buffer_device.type == "cpu"
        ):
            # Async copy GPU -> CPU
            dest.copy_(src, non_blocking=True)
        elif (
            self.config.async_gpu_transfer
            and not src.is_cuda
            and self.buffer_device
            and self.buffer_device.type == "cuda"
        ):
            # Async copy CPU -> GPU
            dest.copy_(src, non_blocking=True)
        else:
            # Synchronous copy
            dest.copy_(src)

    def get_gradient(self, param: nn.Parameter) -> Optional[Tensor]:
        """Get gradient for a specific parameter.

        Args:
            param: Parameter to get gradient for.

        Returns:
            Gradient tensor or None if not available.

        Note:
            Returns a view of the gradient buffer when possible to avoid copies.
        """
        with self._thread_safe_context():
            if self.gradient_buffer is None or self._released:
                self.cache_misses += 1
                return None

            if param not in self.param_to_index:
                if self.config.debug_mode:
                    logger.warning(f"Parameter not found in buffer: {param.shape}")
                self.cache_misses += 1
                return None

            idx = self.param_to_index[param]
            offset = self.param_offsets[idx]
            numel = param.numel()
            shape = self.param_shapes[idx]

            try:
                grad_view = self.gradient_buffer[offset : offset + numel].view(shape)

                # Convert to parameter device if needed
                if grad_view.device != param.device:
                    grad_view = grad_view.to(
                        param.device,
                        dtype=(
                            param.dtype
                            if self.config.validate_dtypes
                            else grad_view.dtype
                        ),
                        non_blocking=self.config.async_gpu_transfer,
                    )

                self.cache_hits += 1
                return grad_view

            except (RuntimeError, IndexError) as e:
                if self.config.debug_mode:
                    logger.error(f"Failed to get gradient: {e}")
                self.cache_misses += 1
                return None

    def set_gradient(self, param: nn.Parameter, grad: Tensor) -> None:
        """Set gradient for a specific parameter.

        Args:
            param: Parameter to set gradient for.
            grad: Gradient tensor to set.

        Raises:
            ValueError: If parameter not in buffer.
            RuntimeError: If gradient shape mismatch.
        """
        with self._thread_safe_context():
            if self._released:
                raise RuntimeError("Buffer has been released")

            if self.gradient_buffer is None:
                self._initialize_buffer()

            if param not in self.param_to_index:
                raise ValueError(
                    f"Parameter with shape {param.shape} not found in buffer"
                )

            idx = self.param_to_index[param]

            # Validate gradient shape
            if grad.shape != self.param_shapes[idx]:
                raise RuntimeError(
                    f"Gradient shape mismatch: expected {self.param_shapes[idx]}, "
                    f"got {grad.shape}"
                )

            # Validate dtype if enabled
            if self.config.validate_dtypes and grad.dtype != self.config.dtype:
                grad = grad.to(dtype=self.config.dtype)

            offset = self.param_offsets[idx]
            numel = param.numel()

            if self.gradient_buffer is not None:
                try:
                    grad_view = self.gradient_buffer[offset : offset + numel].view_as(
                        param
                    )
                    self._optimized_gradient_copy(grad_view, grad)
                    self.total_gradient_updates += 1
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    error_msg = f"Failed to set gradient: {e}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

    def accumulate_gradient(
        self, param: nn.Parameter, grad: Tensor, alpha: float = 1.0
    ) -> None:
        """Accumulate gradient for a specific parameter.

        Args:
            param: Parameter to accumulate gradient for.
            grad: Gradient tensor to accumulate.
            alpha: Scaling factor for accumulation (grad = buffer + alpha * grad).

        Raises:
            ValueError: If parameter not in buffer.
            RuntimeError: If accumulation fails.
        """
        with self._thread_safe_context():
            if self._released:
                raise RuntimeError("Buffer has been released")

            if self.gradient_buffer is None:
                self._initialize_buffer()

            if param not in self.param_to_index:
                raise ValueError(
                    f"Parameter with shape {param.shape} not found in buffer"
                )

            idx = self.param_to_index[param]

            # Validate gradient shape
            if grad.shape != self.param_shapes[idx]:
                raise RuntimeError(
                    f"Gradient shape mismatch: expected {self.param_shapes[idx]}, "
                    f"got {grad.shape}"
                )

            offset = self.param_offsets[idx]
            numel = param.numel()

            if self.gradient_buffer is not None:
                try:
                    grad_view = self.gradient_buffer[offset : offset + numel].view_as(
                        param
                    )

                    if alpha == 1.0:
                        grad_view.add_(grad)
                    else:
                        grad_view.add_(grad, alpha=alpha)

                    self.accumulation_count += 1
                    self.total_gradient_updates += 1

                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    error_msg = f"Failed to accumulate gradient: {e}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

    def zero_gradients(self, set_to_none: bool = False) -> None:
        """Zero all gradients in the buffer.

        Args:
            set_to_none: If True, release buffer memory instead of zeroing.
        """
        with self._thread_safe_context():
            if self._released:
                return

            if set_to_none:
                # Release buffer memory
                self.gradient_buffer = None
                self.bytes_allocated = 0
                self._initialized = False
            elif self.gradient_buffer is not None:
                # Zero buffer in-place
                self.gradient_buffer.zero_()

            self.accumulation_count = 0

    def scale_gradients(self, scale_factor: float) -> None:
        """Scale all gradients by a factor.

        Args:
            scale_factor: Factor to scale gradients by.

        Note:
            This operation is performed in-place for efficiency.
        """
        with self._thread_safe_context():
            if self.gradient_buffer is not None and not self._released:
                if scale_factor == 0.0:
                    # More efficient than multiplication for zeroing
                    self.gradient_buffer.zero_()
                elif scale_factor != 1.0:
                    self.gradient_buffer.mul_(scale_factor)

    def get_all_gradients(self) -> Optional[Tensor]:
        """Get the entire gradient buffer.

        Returns:
            The gradient buffer tensor or None if not initialized.

        Warning:
            Returns direct reference to internal buffer. Modifications will
            affect stored gradients.
        """
        with self._thread_safe_context():
            if self._released:
                return None
            return self.gradient_buffer

    def to(
        self, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None
    ) -> None:
        """Move gradient buffer to a device and optionally change dtype.

        Args:
            device: Target device.
            dtype: Optional target dtype.

        Raises:
            RuntimeError: If device transfer fails.
        """
        with self._thread_safe_context():
            if self._released:
                return

            if isinstance(device, str):
                device = torch.device(device)

            if self.gradient_buffer is not None:
                try:
                    # Check if transfer is needed
                    needs_transfer = self.buffer_device != device
                    needs_dtype_change = (
                        dtype is not None and self.gradient_buffer.dtype != dtype
                    )

                    if needs_transfer or needs_dtype_change:
                        kwargs = {
                            "device": device,
                            "non_blocking": self.config.async_gpu_transfer,
                        }
                        if dtype is not None:
                            kwargs["dtype"] = dtype

                        self.gradient_buffer = self.gradient_buffer.to(
                            device=device,
                            dtype=dtype,
                            non_blocking=self.config.async_gpu_transfer,
                        )
                        self.buffer_device = device

                        # Update config dtype if changed
                        if dtype is not None:
                            self.config.dtype = dtype

                        # Recalculate allocated bytes if dtype changed
                        if needs_dtype_change and self.gradient_buffer is not None:
                            self.bytes_allocated = (
                                self.gradient_buffer.numel()
                                * self.gradient_buffer.element_size()
                            )

                        if self.config.debug_mode:
                            logger.debug(
                                f"Moved gradient buffer to {device}"
                                f"{f' with dtype {dtype}' if dtype else ''}"
                            )

                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    error_msg = f"Failed to move gradient buffer to {device}: {e}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

    def release(self) -> None:
        """Release gradient buffer and hooks.

        Note:
            This method ensures proper cleanup of resources and prevents
            memory leaks from hook references.
        """
        with self._thread_safe_context():
            if self._released:
                return

            self._released = True

            # Remove hooks safely
            for hook in self.hooks:
                try:
                    hook.remove()
                except Exception as e:
                    if self.config.debug_mode:
                        logger.warning(f"Failed to remove hook: {e}")

            self.hooks.clear()
            self._hook_functions.clear()

            # Release buffer
            self.gradient_buffer = None
            self.bytes_allocated = 0
            self._initialized = False

            # Clear parameter references to help GC
            self.param_to_index.clear()

            # Force CUDA cache cleanup if using GPU
            if self.buffer_device and self.buffer_device.type == "cuda":
                torch.cuda.empty_cache()

            if self.config.debug_mode:
                logger.debug(
                    f"Released gradient buffer for group {self.param_group_id}"
                )

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get comprehensive memory usage statistics.

        Returns:
            Dictionary with detailed memory and performance statistics.
        """
        with self._thread_safe_context():
            stats: Dict[str, Any] = {
                "allocated_mb": self.bytes_allocated / BYTES_PER_MB,
                "peak_mb": self.peak_memory_mb,
                "num_parameters": len(self.parameters),
                "total_elements": self.total_numel,
                "accumulation_count": self.accumulation_count,
                "total_updates": self.total_gradient_updates,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": (
                    self.cache_hits / max(1, self.cache_hits + self.cache_misses)
                ),
                "is_initialized": self._initialized,
                "is_released": self._released,
                "storage_mode": self.config.storage_mode.value,
            }

            if self.gradient_buffer is not None:
                stats.update(
                    {
                        "buffer_device": str(self.buffer_device),
                        "buffer_dtype": str(self.gradient_buffer.dtype),
                        "buffer_shape": (
                            list(self.gradient_buffer.shape)
                            if self.gradient_buffer is not None
                            else []
                        ),
                    }
                )

            # Add parameter group info if available
            if self.param_group_id is not None:
                stats["param_group_id"] = self.param_group_id

            return stats


class DecoupledGradientManager:
    """Manages decoupled gradient storage for an entire model.

    This class coordinates gradient storage across multiple parameter groups
    and provides a unified interface for gradient operations with:
    - Automatic parameter grouping strategies
    - Memory-efficient buffer allocation
    - Thread-safe operations
    - Comprehensive monitoring and profiling

    Example:
        >>> model = MyLargeModel()
        >>> config = DecoupledGradientConfig(
        ...     storage_mode=StorageMode.CONTIGUOUS,
        ...     device="cuda",
        ...     enable_profiling=True
        ... )
        >>> manager = DecoupledGradientManager(model, config)
        >>> # Use with optimizer
        >>> optimizer = DistributedOptimizer(
        ...     model.parameters(),
        ...     torch.optim.Adam,
        ...     {"lr": 0.001},
        ...     decoupled_grad_manager=manager
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[DecoupledGradientConfig] = None,
        param_grouping_strategy: str = "by_requires_grad",
    ):
        """Initialize gradient manager for a model.

        Args:
            model: Model to manage gradients for.
            config: Configuration for gradient storage.
            param_grouping_strategy: Strategy for grouping parameters
                ('by_requires_grad', 'by_layer', 'by_size', 'single_group').

        Raises:
            ValueError: If invalid grouping strategy provided.
        """
        if not isinstance(model, nn.Module):
            raise TypeError(f"Expected nn.Module, got {type(model)}")

        self.model = model
        self.config = config or DecoupledGradientConfig()
        self.param_grouping_strategy = param_grouping_strategy
        self._lock = threading.RLock() if self.config.thread_safe else None
        self._is_released = False

        # Create buffers for each parameter group
        self.buffers: List[DecoupledGradientBuffer] = []
        self.param_to_buffer: Dict[nn.Parameter, DecoupledGradientBuffer] = {}
        self.buffer_to_params: Dict[DecoupledGradientBuffer, List[nn.Parameter]] = {}

        # Track all parameters
        self.all_parameters: List[nn.Parameter] = []
        self.param_names: Dict[nn.Parameter, str] = {}

        # Initialize statistics before buffer creation
        self.total_memory_mb = 0.0
        self.gradient_sync_count = 0
        self.total_operations = 0
        self.last_sync_time: Optional[float] = None

        # Collect parameter names for debugging
        for name, param in model.named_parameters():
            self.param_names[param] = name

        # Initialize buffers
        self._initialize_buffers()

        # Register model hooks if profiling enabled
        if self.config.enable_profiling:
            self._register_profiling_hooks()

    def _initialize_buffers(self) -> None:
        """Initialize gradient buffers for all parameters using specified strategy."""
        # Collect all parameters
        for param in self.model.parameters():
            self.all_parameters.append(param)

        # Group parameters based on strategy
        param_groups = self._create_parameter_groups()

        # Create buffers for each group
        total_grad_params = 0
        total_no_grad_params = 0

        for group_id, params in enumerate(param_groups):
            grad_params = [p for p in params if p.requires_grad]
            no_grad_params = [p for p in params if not p.requires_grad]

            total_grad_params += len(grad_params)
            total_no_grad_params += len(no_grad_params)

            if grad_params:
                try:
                    buffer = DecoupledGradientBuffer(
                        grad_params, self.config, param_group_id=group_id
                    )
                    self.buffers.append(buffer)
                    self.buffer_to_params[buffer] = grad_params

                    for param in grad_params:
                        self.param_to_buffer[param] = buffer

                    self.total_memory_mb += buffer.bytes_allocated / BYTES_PER_MB

                except RuntimeError as e:
                    logger.error(f"Failed to create buffer for group {group_id}: {e}")
                    # Clean up any created buffers
                    self._cleanup_buffers()
                    raise

        if self.config.enable_profiling or self.config.debug_mode:
            logger.info(
                "Initialized DecoupledGradientManager with strategy "
                f"'{self.param_grouping_strategy}': "
                f"{len(self.buffers)} buffer(s), "
                f"{total_grad_params} parameters with gradients, "
                f"{total_no_grad_params} parameters without gradients, "
                f"Total memory: {self.total_memory_mb:.2f} MB"
            )

    def _create_parameter_groups(self) -> List[List[nn.Parameter]]:
        """Create parameter groups based on the specified strategy.

        Returns:
            List of parameter groups.
        """
        if self.param_grouping_strategy == "single_group":
            return [self.all_parameters]

        elif self.param_grouping_strategy == "by_requires_grad":
            grad_params = [p for p in self.all_parameters if p.requires_grad]
            no_grad_params = [p for p in self.all_parameters if not p.requires_grad]
            groups = []
            if grad_params:
                groups.append(grad_params)
            if no_grad_params:
                groups.append(no_grad_params)
            return groups

        elif self.param_grouping_strategy == "by_layer":
            # Group by module hierarchy
            layer_groups: Dict[str, List[nn.Parameter]] = {}
            for name, param in self.model.named_parameters():
                # Extract layer name (first component before '.')
                layer_name = name.split(".")[0] if "." in name else "root"
                if layer_name not in layer_groups:
                    layer_groups[layer_name] = []
                layer_groups[layer_name].append(param)
            return list(layer_groups.values())

        elif self.param_grouping_strategy == "by_size":
            # Group by parameter size (small, medium, large)
            small_params = []  # < 1M elements
            medium_params = []  # 1M - 10M elements
            large_params = []  # > 10M elements

            for param in self.all_parameters:
                numel = param.numel()
                if numel < 1_000_000:
                    small_params.append(param)
                elif numel < 10_000_000:
                    medium_params.append(param)
                else:
                    large_params.append(param)

            groups = []
            for group in [large_params, medium_params, small_params]:
                if group:
                    groups.append(group)
            return groups

        else:
            raise ValueError(
                f"Unknown parameter grouping strategy: {self.param_grouping_strategy}. "
                "Valid options: 'single_group', 'by_requires_grad', "
                "'by_layer', 'by_size'"
            )

    def _cleanup_buffers(self) -> None:
        """Clean up any created buffers on initialization failure."""
        for buffer in self.buffers:
            try:
                buffer.release()
            except Exception:
                pass
        self.buffers.clear()
        self.param_to_buffer.clear()
        self.buffer_to_params.clear()
        self.total_memory_mb = 0.0

    def _register_profiling_hooks(self) -> None:
        """Register hooks for profiling gradient operations."""
        if not self.config.enable_profiling:
            return

        # Track gradient computation timing and memory usage
        self._gradient_compute_times: Dict[str, List[float]] = {}
        self._peak_memory_usage: Dict[str, float] = {}

        def create_timing_hook(name: str) -> Callable:
            def hook(module: nn.Module, input: Any, output: Any) -> None:
                if name not in self._gradient_compute_times:
                    self._gradient_compute_times[name] = []
                # Placeholder for actual timing - would need backward hook
                # This is a forward hook example for structure
                pass

            return hook

        # Register hooks on key modules if detailed profiling is needed
        # This is a placeholder structure for future enhancement
        if self.config.debug_mode:
            logger.debug("Profiling hooks registered for gradient operations")

    @contextmanager
    def _thread_safe_context(self) -> Iterator[None]:
        """Context manager for thread-safe operations."""
        if self._lock:
            self._lock.acquire()
        try:
            yield
        finally:
            if self._lock:
                self._lock.release()

    def create_param_groups(
        self, param_groups: List[Dict[str, Any]]
    ) -> List[DecoupledGradientBuffer]:
        """Create gradient buffers for parameter groups.

        Args:
            param_groups: List of parameter groups (optimizer format).

        Returns:
            List of gradient buffers for each group.

        Raises:
            RuntimeError: If manager has been released.
        """
        with self._thread_safe_context():
            if self._is_released:
                raise RuntimeError("Manager has been released")

            # Clear existing buffers
            for buffer in self.buffers:
                buffer.release()
            self.buffers.clear()
            self.param_to_buffer.clear()
            self.buffer_to_params.clear()
            self.total_memory_mb = 0.0

            # Create buffer for each parameter group
            for group_id, group in enumerate(param_groups):
                params = list(group["params"])

                # Filter for parameters requiring gradients
                grad_params = [p for p in params if p.requires_grad]

                if grad_params:
                    try:
                        buffer = DecoupledGradientBuffer(
                            grad_params, self.config, param_group_id=group_id
                        )
                        self.buffers.append(buffer)
                        self.buffer_to_params[buffer] = grad_params

                        for param in grad_params:
                            self.param_to_buffer[param] = buffer

                        self.total_memory_mb += buffer.bytes_allocated / BYTES_PER_MB

                    except RuntimeError as e:
                        logger.error(
                            f"Failed to create buffer for group {group_id}: {e}"
                        )
                        self._cleanup_buffers()
                        raise

            if self.config.debug_mode:
                logger.debug(
                    f"Created {len(self.buffers)} gradient buffers for "
                    f"{len(param_groups)} parameter groups, "
                    f"Total memory: {self.total_memory_mb:.2f} MB"
                )

            return self.buffers

    def get_gradient(self, param: nn.Parameter) -> Optional[Tensor]:
        """Get gradient for a parameter.

        Args:
            param: Parameter to get gradient for.

        Returns:
            Gradient tensor or None if not available.
        """
        with self._thread_safe_context():
            if self._is_released:
                return None

            buffer = self.param_to_buffer.get(param)
            if buffer is None:
                if self.config.debug_mode and param in self.all_parameters:
                    param_name = self.param_names.get(param, "unknown")
                    logger.debug(f"No gradient buffer for parameter '{param_name}'")
                return None

            self.total_operations += 1
            return buffer.get_gradient(param)

    def set_gradient(self, param: nn.Parameter, grad: Tensor) -> None:
        """Set gradient for a parameter.

        Args:
            param: Parameter to set gradient for.
            grad: Gradient tensor.

        Raises:
            ValueError: If no buffer found for parameter.
            RuntimeError: If manager has been released.
        """
        with self._thread_safe_context():
            if self._is_released:
                raise RuntimeError("Manager has been released")

            buffer = self.param_to_buffer.get(param)
            if buffer is None:
                param_name = self.param_names.get(param, "unknown")
                raise ValueError(
                    f"No buffer found for parameter '{param_name}' "
                    f"with shape {param.shape}"
                )

            self.total_operations += 1
            buffer.set_gradient(param, grad)

    def accumulate_gradient(
        self, param: nn.Parameter, grad: Tensor, alpha: float = 1.0
    ) -> None:
        """Accumulate gradient for a parameter.

        Args:
            param: Parameter to accumulate gradient for.
            grad: Gradient to accumulate.
            alpha: Scaling factor for accumulation.

        Raises:
            ValueError: If no buffer found for parameter.
            RuntimeError: If manager has been released.
        """
        with self._thread_safe_context():
            if self._is_released:
                raise RuntimeError("Manager has been released")

            buffer = self.param_to_buffer.get(param)
            if buffer is None:
                param_name = self.param_names.get(param, "unknown")
                raise ValueError(
                    f"No buffer found for parameter '{param_name}' "
                    f"with shape {param.shape}"
                )

            self.total_operations += 1
            buffer.accumulate_gradient(param, grad, alpha)

    def zero_gradients(self, set_to_none: bool = False) -> None:
        """Zero all gradients.

        Args:
            set_to_none: If True, release buffer memory instead of zeroing.
        """
        with self._thread_safe_context():
            if self._is_released:
                return

            for buffer in self.buffers:
                buffer.zero_gradients(set_to_none)

            self.total_operations += 1

    def scale_gradients(self, scale_factor: float) -> None:
        """Scale all gradients by a factor.

        Args:
            scale_factor: Scaling factor.
        """
        with self._thread_safe_context():
            if self._is_released:
                return

            for buffer in self.buffers:
                buffer.scale_gradients(scale_factor)

            self.total_operations += 1

    def sync_gradients_to_params(self, clone: bool = True) -> None:
        """Synchronize gradients from buffers to parameters.

        Args:
            clone: If True, clone gradients (safer but uses more memory).
                   If False, share gradient memory (faster but
                   modifications affect buffer).
        """
        with self._thread_safe_context():
            if self._is_released:
                return

            import time

            start_time = time.time()

            synced_count = 0
            for param in self.all_parameters:
                if param.requires_grad:
                    grad = self.get_gradient(param)
                    if grad is not None:
                        if clone:
                            param.grad = grad.clone()
                        else:
                            param.grad = grad
                        synced_count += 1

            self.gradient_sync_count += 1
            self.total_operations += 1
            self.last_sync_time = time.time() - start_time

            if self.config.debug_mode:
                logger.debug(
                    f"Synced {synced_count} gradients to parameters in "
                    f"{self.last_sync_time*1000:.2f} ms"
                )

    def sync_gradients_from_params(self, clear_param_grads: bool = True) -> None:
        """Synchronize gradients from parameters to buffers.

        Args:
            clear_param_grads: If True, clear parameter gradients after sync
                              to save memory.
        """
        with self._thread_safe_context():
            if self._is_released:
                return

            import time

            start_time = time.time()

            synced_count = 0
            for param in self.all_parameters:
                if param.requires_grad and param.grad is not None:
                    self.set_gradient(param, param.grad)
                    if clear_param_grads:
                        param.grad = None
                    synced_count += 1

            self.gradient_sync_count += 1
            self.total_operations += 1
            self.last_sync_time = time.time() - start_time

            if self.config.debug_mode:
                logger.debug(
                    f"Synced {synced_count} gradients from parameters in "
                    f"{self.last_sync_time*1000:.2f} ms"
                )

    def to(
        self, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None
    ) -> None:
        """Move all gradient buffers to a device and optionally change dtype.

        Args:
            device: Target device.
            dtype: Optional target dtype.
        """
        with self._thread_safe_context():
            if self._is_released:
                return

            for buffer in self.buffers:
                buffer.to(device, dtype)

            self.total_operations += 1

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get comprehensive memory usage statistics.

        Returns:
            Dictionary with detailed memory and performance statistics.
        """
        with self._thread_safe_context():
            total_allocated = 0.0
            total_peak = 0.0
            total_cache_hits = 0
            total_cache_misses = 0
            total_updates = 0
            buffer_stats = []

            for i, buffer in enumerate(self.buffers):
                stats = buffer.get_memory_usage()
                buffer_stats.append(stats)
                total_allocated += stats["allocated_mb"]
                total_peak += stats["peak_mb"]
                total_cache_hits += stats.get("cache_hits", 0)
                total_cache_misses += stats.get("cache_misses", 0)
                total_updates += stats.get("total_updates", 0)

            # Calculate parameter statistics
            total_params = len(self.all_parameters)
            grad_params = sum(1 for p in self.all_parameters if p.requires_grad)
            no_grad_params = total_params - grad_params

            return {
                "total_allocated_mb": total_allocated,
                "total_peak_mb": total_peak,
                "num_buffers": len(self.buffers),
                "num_parameters": total_params,
                "num_grad_parameters": grad_params,
                "num_no_grad_parameters": no_grad_params,
                "gradient_sync_count": self.gradient_sync_count,
                "total_operations": self.total_operations,
                "total_cache_hits": total_cache_hits,
                "total_cache_misses": total_cache_misses,
                "total_gradient_updates": total_updates,
                "cache_hit_rate": (
                    total_cache_hits / max(1, total_cache_hits + total_cache_misses)
                ),
                "last_sync_time_ms": (
                    self.last_sync_time * 1000 if self.last_sync_time else None
                ),
                "grouping_strategy": self.param_grouping_strategy,
                "config": {
                    "storage_mode": self.config.storage_mode.value,
                    "device": self.config.device,
                    "dtype": str(self.config.dtype),
                    "thread_safe": self.config.thread_safe,
                    "enable_profiling": self.config.enable_profiling,
                },
                "buffer_stats": buffer_stats,
                "is_released": self._is_released,
            }

    def release(self) -> None:
        """Release all gradient buffers and clean up resources.

        Note:
            After calling this method, the manager cannot be used again.
        """
        with self._thread_safe_context():
            if self._is_released:
                return

            self._is_released = True

            # Release all buffers
            for buffer in self.buffers:
                try:
                    buffer.release()
                except Exception as e:
                    if self.config.debug_mode:
                        logger.warning(f"Failed to release buffer: {e}")

            # Clear all references
            self.buffers.clear()
            self.param_to_buffer.clear()
            self.buffer_to_params.clear()
            self.all_parameters.clear()
            self.param_names.clear()

            # Reset statistics
            self.total_memory_mb = 0.0

            if self.config.debug_mode:
                logger.debug(
                    f"Released DecoupledGradientManager: "
                    f"{self.gradient_sync_count} total syncs, "
                    f"{self.total_operations} total operations"
                )
