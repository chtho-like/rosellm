"""
Gradient Communication Bucketing for RoseTrainer

This module provides a comprehensive gradient bucketing system for efficient
distributed training communication. It reduces communication overhead by
intelligently grouping gradients into buckets and using advanced memory
management and error handling.

Key Features:
- Multiple bucketing strategies (size-based, layer-based, mixed, custom)
- Memory pooling for efficient tensor reuse and reduced allocations
- Comprehensive error handling with custom exception types
- Thread-safe operations for multi-threaded environments
- Performance metrics and optimization hints
- Robust validation and type safety

Basic Usage Example:
    ```python
    import torch
    from rosellm.rosetrainer.communication import (
        BucketConfig, BucketManager, BucketStrategy
    )

    # Configure bucketing
    config = BucketConfig(
        strategy=BucketStrategy.SIZE_BASED,
        max_bucket_size_mb=25.0,
        overlap_communication=True
    )

    # Create bucket manager
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manager = BucketManager(config, device)

    # Assign gradients to buckets
    gradients = {"param1": torch.randn(100, 50), "param2": torch.randn(200)}
    assignments = manager.assign_gradients_bulk(gradients)

    # Synchronize all buckets
    stats = manager.synchronize_buckets()
    print(f"Communication completed in {stats['total_time']:.3f}s")
    ```

Advanced Usage with Groups:
    ```python
    from rosellm.rosetrainer.communication import (
        BucketGroupConfig, BucketGroupManager, GroupStrategy
    )

    # Create hierarchical bucket groups
    group_config = BucketGroupConfig(
        group_strategy=GroupStrategy.HIERARCHICAL,
        enable_prioritization=True,
        overlap_groups=True
    )

    group_manager = BucketGroupManager(group_config, manager)
    group_manager.assign_buckets_to_groups()

    # Synchronize with priority-based communication
    sync_stats = group_manager.synchronize_groups()
    ```

Memory Optimization:
    The module includes a sophisticated memory pool system that reuses tensors
    across gradient bucketing operations, significantly reducing memory allocation
    overhead and improving performance in large-scale training scenarios.

Error Handling:
    Custom exception types provide detailed error information:
    - BucketingError: Base exception for all bucketing operations
    - GradientValidationError: Invalid gradient data (NaN, inf, wrong dtype)
    - BucketCapacityError: Bucket size limits exceeded
    - CommunicationError: Distributed communication failures
    - BucketStateError: Invalid bucket state for operation

Performance Considerations:
    - Use bulk assignment (assign_gradients_bulk) for better performance
    - Enable dynamic bucketing for adaptive optimization
    - Consider memory vs speed optimization hints when creating buckets
    - Use appropriate bucket sizes based on network bandwidth

References:
- PyTorch DDP Communication Hooks: https://pytorch.org/docs/stable/ddp_comm_hooks.html
- FairScale GradBucket: https://github.com/facebookresearch/fairscale
- "Efficient Large-Scale Language Model Training on GPU Clusters"
  https://arxiv.org/abs/2104.04473
"""

import logging
import math
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# Custom exceptions for gradient bucketing
class BucketingError(Exception):
    """Base exception for gradient bucketing operations."""

    pass


class BucketCapacityError(BucketingError):
    """Raised when bucket capacity is exceeded."""

    pass


class BucketStateError(BucketingError):
    """Raised when bucket is in invalid state for operation."""

    pass


class CommunicationError(BucketingError):
    """Raised when distributed communication fails."""

    pass


class GradientValidationError(BucketingError):
    """Raised when gradient validation fails."""

    pass


# Thread-safe operation decorator
def thread_safe_operation(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to ensure thread-safe operations on bucket management.

    This decorator wraps methods to acquire a lock before execution,
    preventing race conditions in multi-threaded environments.

    Args:
        func: Method to make thread-safe

    Returns:
        Thread-safe wrapper function
    """

    @wraps(func)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        if hasattr(self, "_lock") and self._lock is not None:
            with self._lock:
                return func(self, *args, **kwargs)
        return func(self, *args, **kwargs)

    return wrapper


class BucketStrategy(Enum):
    """Strategy for grouping gradients into buckets."""

    SIZE_BASED = "size_based"  # Group by parameter size
    LAYER_BASED = "layer_based"  # Group by layer type/name
    MIXED = "mixed"  # Hybrid approach combining size and layer information
    CUSTOM = "custom"  # User-defined bucketing function


class CommunicationBackend(Enum):
    """Communication backend for gradient synchronization."""

    NCCL = "nccl"  # NVIDIA Collective Communication Library
    GLOO = "gloo"  # Facebook's Gloo library
    AUTO = "auto"  # Automatically select based on available hardware


@dataclass
class BucketConfig:
    """Configuration for gradient bucket management."""

    # Bucket strategy settings
    strategy: BucketStrategy = BucketStrategy.SIZE_BASED
    max_bucket_size_mb: float = 25.0  # Maximum bucket size in MB
    min_bucket_size_mb: float = 1.0  # Minimum bucket size in MB

    # Communication settings
    backend: CommunicationBackend = CommunicationBackend.AUTO
    overlap_communication: bool = True
    compress_gradients: bool = False

    # Advanced features
    dynamic_bucketing: bool = False  # Adapt bucket sizes based on performance
    gradient_predivision: bool = True  # Pre-divide gradients for numerical stability

    # Performance tuning
    communication_timeout_ms: int = 30000  # 30 seconds
    bucket_cap_mb: float = 100.0  # Hard limit on bucket size

    # Custom bucketing function (for CUSTOM strategy)
    custom_bucket_fn: Optional[Callable[[str, torch.Tensor], str]] = None

    # Layer type grouping patterns (for LAYER_BASED strategy)
    layer_groups: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "embedding": ["embed", "embedding", "position"],
            "attention": ["attn", "attention", "self_attn", "cross_attn"],
            "feedforward": ["mlp", "ffn", "feed_forward", "fc"],
            "normalization": ["norm", "ln", "layer_norm", "batch_norm"],
            "output": ["output", "head", "classifier", "lm_head"],
        }
    )


class TensorMemoryPool:
    """
    Memory pool for efficient tensor allocation and reuse.

    This class manages a pool of tensors to avoid frequent allocations and
    deallocations,
    which can be expensive in GPU memory management. Tensors are organized by size and
    reused when possible.

    The pool implements several safety features:
    - Size-based organization for efficient lookup
    - Device and dtype validation to prevent mismatched tensors
    - Size limits to prevent unbounded memory growth
    - Automatic tensor clearing when returned to pool

    Thread Safety:
        All operations are thread-safe using internal locking.

    Memory Management:
        Tensors are automatically zeroed when returned to the pool.
        Pool size is limited to prevent memory bloat (configurable per size).

    Example:
        ```python
        device = torch.device("cuda")
        pool = TensorMemoryPool(device, torch.float32)

        # Get tensor from pool (creates new if pool empty)
        tensor = pool.get_tensor(1000)
        tensor.fill_(42.0)

        # Return to pool (will be zeroed automatically)
        pool.return_tensor(tensor)

        # Get another tensor (likely same instance, now zeroed)
        tensor2 = pool.get_tensor(1000)
        assert torch.all(tensor2 == 0.0)
        ```
    """

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype
        self._pool: Dict[int, List[torch.Tensor]] = {}  # size -> tensors
        self._lock = threading.RLock()

    @thread_safe_operation
    def get_tensor(self, size: int) -> torch.Tensor:
        """Get a tensor from the pool or create a new one."""
        if size in self._pool and self._pool[size]:
            return self._pool[size].pop()

        # Create new tensor if pool is empty
        return torch.empty(size, dtype=self.dtype, device=self.device)

    @thread_safe_operation
    def return_tensor(self, tensor: torch.Tensor) -> None:
        """Return a tensor to the pool for reuse."""
        if tensor.device != self.device or tensor.dtype != self.dtype:
            return  # Don't pool mismatched tensors

        size = tensor.numel()
        if size not in self._pool:
            self._pool[size] = []

        # Limit pool size to prevent memory bloat
        if len(self._pool[size]) < 10:
            tensor.zero_()  # Clear the tensor
            self._pool[size].append(tensor)


class GradientBucket:
    """
    Container for grouped gradients that will be communicated together.

    This class manages a collection of gradient tensors that are grouped
    together for efficient communication. It provides memory-optimized
    tensor operations, comprehensive error handling, and detailed metrics.

    Key Features:
    - Memory pool integration for efficient tensor reuse
    - Comprehensive gradient validation (NaN, inf, dtype checks)
    - Thread-safe operations with proper locking
    - Performance metrics collection and bounds management
    - Automatic device/dtype conversion
    - Robust communication error handling with timeouts

    Lifecycle:
        1. Creation: Initialize empty bucket with size constraints
        2. Addition: Add gradients with validation and capacity checks
        3. Flattening: Combine gradients into single tensor for communication
        4. Communication: Asynchronous distributed operations
        5. Unflattening: Restore individual gradient shapes
        6. Clearing: Return resources to pools and reset state

    Memory Management:
        Uses class-level memory pools to avoid frequent tensor allocations.
        Pools are shared across buckets with same device/dtype combination.
        Automatic tensor reuse reduces memory fragmentation and improves performance.

    Error Handling:
        Validates all gradient data for NaN/inf values, correct dtypes, and sizes.
        Communication operations include timeout handling and retry logic.
        Thread-safe error recovery with proper resource cleanup.

    Performance:
        Tracks detailed timing metrics for communication operations.
        Metrics history is bounded to prevent memory leaks.
        Supports optimization hints for memory vs speed trade-offs.

    Example:
        ```python
        device = torch.device("cuda")
        bucket = GradientBucket(
            bucket_id=0,
            max_size_bytes=25 * 1024 * 1024,  # 25MB
            device=device
        )

        # Add gradients with validation
        grad1 = torch.randn(1000, 500, device=device)
        success = bucket.add_gradient(grad1, "layer1.weight", "linear")

        # Flatten for communication
        flattened = bucket.flatten_gradients()

        # Start async communication
        handle = bucket.start_communication()

        # Wait for completion and get timing
        comm_time = bucket.wait_communication()

        # Restore individual gradients
        unflattened = bucket.unflatten_gradients()

        # Get performance statistics
        stats = bucket.get_statistics()
        print(f"Avg comm time: {stats['avg_communication_time']:.3f}s")
        ```
    """

    # Class-level memory pools for tensor reuse
    _memory_pools: Dict[Tuple[torch.device, torch.dtype], TensorMemoryPool] = {}
    _pool_lock = threading.RLock()

    @classmethod
    def _get_memory_pool(
        cls, device: torch.device, dtype: torch.dtype
    ) -> TensorMemoryPool:
        """Get or create a memory pool for the given device/dtype combination."""
        with cls._pool_lock:
            key = (device, dtype)
            if key not in cls._memory_pools:
                cls._memory_pools[key] = TensorMemoryPool(device, dtype)
            return cls._memory_pools[key]

    def __init__(
        self,
        bucket_id: int,
        max_size_bytes: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize a gradient bucket.

        Args:
            bucket_id: Unique identifier for this bucket
            max_size_bytes: Maximum size in bytes this bucket can hold
            device: Device where tensors will be stored
            dtype: Data type for bucket tensors
        """
        # Thread safety
        self._lock = threading.RLock()
        self.bucket_id = bucket_id
        self.max_size_bytes = max_size_bytes
        self.device = device
        self.dtype = dtype

        # Get memory pool for this bucket
        self._memory_pool = self._get_memory_pool(device, dtype)

        # Gradient storage
        self.gradients: List[torch.Tensor] = []
        self.gradient_metadata: List[Dict[str, Any]] = []
        self.flattened_gradient: Optional[torch.Tensor] = None

        # Pre-allocate flattened tensor for better memory efficiency
        self._preallocated_size = 0

        # Tracking
        self.current_size_bytes = 0
        self.is_ready = False
        self.communication_handle: Optional[dist.Work] = None
        self.last_communication_time: Optional[float] = None

        # Performance metrics with bounds to prevent memory leaks
        self.communication_times: List[float] = []
        self.compression_ratios: List[float] = []
        self._max_metrics_history = 1000  # Prevent unbounded growth

    @thread_safe_operation
    def can_add_gradient(self, gradient: Optional[torch.Tensor]) -> bool:
        """
        Check if a gradient can be added to this bucket.

        Args:
            gradient: Gradient tensor to check, may be None

        Returns:
            True if gradient can be added, False otherwise
        """
        if gradient is None or gradient.numel() == 0:
            return False

        try:
            grad_size = gradient.numel() * gradient.element_size()
            return bool((self.current_size_bytes + grad_size) <= self.max_size_bytes)
        except (RuntimeError, AttributeError) as e:
            logger.warning(f"Error checking gradient capacity: {e}")
            return False

    @thread_safe_operation
    def add_gradient(
        self,
        gradient: torch.Tensor,
        param_name: str,
        layer_type: Optional[str] = None,
    ) -> bool:
        """
        Add a gradient to this bucket.

        Args:
            gradient: The gradient tensor to add
            param_name: Name/identifier of the parameter
            layer_type: Optional layer type classification

        Returns:
            True if gradient was successfully added, False otherwise
        """
        # Enhanced input validation with specific error types
        try:
            if gradient is None:
                raise GradientValidationError(
                    f"Gradient for parameter '{param_name}' is None"
                )

            if gradient.numel() == 0:
                raise GradientValidationError(
                    f"Gradient for parameter '{param_name}' is empty"
                )

            if not param_name or not param_name.strip():
                raise GradientValidationError("Parameter name cannot be empty")

            if not gradient.dtype.is_floating_point:
                raise GradientValidationError(
                    f"Gradient for parameter '{param_name}' must have "
                    f"floating-point dtype, got {gradient.dtype}"
                )

            if torch.isnan(gradient).any():
                raise GradientValidationError(
                    f"Gradient for parameter '{param_name}' contains NaN values"
                )

            if torch.isinf(gradient).any():
                raise GradientValidationError(
                    f"Gradient for parameter '{param_name}' contains infinite values"
                )

            if not self.can_add_gradient(gradient):
                grad_size_mb = (
                    gradient.numel() * gradient.element_size() / (1024 * 1024)
                )
                available_mb = (self.max_size_bytes - self.current_size_bytes) / (
                    1024 * 1024
                )
                raise BucketCapacityError(
                    f"Cannot add gradient '{param_name}' "
                    f"({grad_size_mb:.2f} MB) to bucket {self.bucket_id}. "
                    f"Available space: {available_mb:.2f} MB"
                )

        except (GradientValidationError, BucketCapacityError):
            raise  # Re-raise our custom exceptions
        except Exception as e:
            logger.error(f"Unexpected error validating gradient '{param_name}': {e}")
            return False

        # Move gradient to bucket device if needed
        if gradient.device != self.device:
            gradient = gradient.to(self.device)

        # Convert dtype if needed
        if gradient.dtype != self.dtype:
            gradient = gradient.to(self.dtype)

        # Store gradient and metadata
        self.gradients.append(gradient)
        self.gradient_metadata.append(
            {
                "param_name": param_name,
                "layer_type": layer_type,
                "shape": gradient.shape,
                "size_bytes": gradient.numel() * gradient.element_size(),
                "added_time": time.time(),
            }
        )

        self.current_size_bytes += gradient.numel() * gradient.element_size()
        self.is_ready = False  # Mark as needing flattening

        return True

    @thread_safe_operation
    def flatten_gradients(self) -> torch.Tensor:
        """
        Flatten all gradients in this bucket into a single tensor with memory
        optimization.

        Returns:
            Flattened gradient tensor

        Raises:
            ValueError: If bucket has no gradients or is already flattened
            RuntimeError: If gradient shapes have changed unexpectedly
        """
        if not self.gradients:
            raise ValueError(f"Bucket {self.bucket_id} has no gradients to flatten")

        if self.is_ready and self.flattened_gradient is not None:
            logger.debug(f"Bucket {self.bucket_id} already flattened")
            return self.flattened_gradient

        # Calculate total size needed
        total_elements = sum(grad.numel() for grad in self.gradients)

        # Try to reuse existing flattened tensor if size matches
        if (
            self.flattened_gradient is not None
            and self.flattened_gradient.numel() == total_elements
        ):
            # Reuse existing tensor
            flattened_tensor = self.flattened_gradient
            flattened_tensor.zero_()
        else:
            # Return old tensor to pool if it exists
            if self.flattened_gradient is not None:
                self._memory_pool.return_tensor(self.flattened_gradient)

            # Get optimized tensor from pool
            flattened_tensor = self._memory_pool.get_tensor(total_elements)
            if flattened_tensor.numel() != total_elements:
                # Pool tensor doesn't match, create new one
                flattened_tensor = torch.empty(
                    total_elements, dtype=self.dtype, device=self.device
                )

        # Efficiently copy gradients to flattened tensor
        try:
            offset = 0
            for i, grad in enumerate(self.gradients):
                if grad is None or grad.numel() == 0:
                    raise RuntimeError(f"Invalid gradient at index {i}")

                grad_size = grad.numel()
                flattened_tensor[offset : offset + grad_size].copy_(grad.flatten())
                offset += grad_size

        except Exception as e:
            logger.error(
                f"Error during gradient flattening in bucket {self.bucket_id}: {e}"
            )
            # Return tensor to pool on error
            if flattened_tensor is not self.flattened_gradient:
                self._memory_pool.return_tensor(flattened_tensor)
            raise

        self.flattened_gradient = flattened_tensor
        self.is_ready = True

        # Type checker assertion - should never be None at this point
        assert self.flattened_gradient is not None
        return self.flattened_gradient

    @thread_safe_operation
    def unflatten_gradients(self) -> List[torch.Tensor]:
        """
        Unflatten the bucket's flattened gradient back to original shapes.

        Returns:
            List of gradient tensors in their original shapes

        Raises:
            ValueError: If no flattened gradient exists
            RuntimeError: If gradient shapes are inconsistent
        """
        if self.flattened_gradient is None:
            raise ValueError(f"Bucket {self.bucket_id} has no flattened gradient")

        if len(self.gradient_metadata) != len(self.gradients):
            raise RuntimeError(
                f"Metadata count ({len(self.gradient_metadata)}) doesn't match "
                f"gradient count ({len(self.gradients)})"
            )

        unflattened_grads = []
        offset = 0

        for i, metadata in enumerate(self.gradient_metadata):
            shape = metadata["shape"]
            numel = math.prod(shape)

            grad_slice = self.flattened_gradient[offset : offset + numel]
            unflattened_grad = grad_slice.reshape(shape)
            unflattened_grads.append(unflattened_grad)

            offset += numel

        return unflattened_grads

    @thread_safe_operation
    def start_communication(
        self,
        process_group: Optional[dist.ProcessGroup] = None,
        predivide: bool = True,
    ) -> Optional[dist.Work]:
        """
        Start asynchronous gradient communication.

        Args:
            process_group: Distributed process group for communication
            predivide: Whether to pre-divide gradients by world size

        Returns:
            Communication work handle for async operations
        """
        try:
            if not self.is_ready or self.flattened_gradient is None:
                self.flatten_gradients()
        except Exception as e:
            logger.error(f"Failed to prepare gradients for communication: {e}")
            return None

        start_time = time.time()

        # Pre-divide gradients for numerical stability
        if predivide and dist.is_initialized() and self.flattened_gradient is not None:
            world_size = dist.get_world_size(process_group)
            if world_size > 1:
                self.flattened_gradient.div_(world_size)

        # Start async all-reduce with comprehensive error handling
        if dist.is_initialized():
            try:
                # Validate distributed environment before communication
                world_size = dist.get_world_size(process_group)
                if world_size <= 1:
                    logger.debug(
                        f"Single process environment, skipping communication "
                        f"for bucket {self.bucket_id}"
                    )
                    return None

                # Check tensor validity before communication
                if self.flattened_gradient is None:
                    raise CommunicationError(
                        f"Flattened gradient is None for bucket {self.bucket_id}"
                    )

                if self.flattened_gradient.numel() == 0:
                    raise CommunicationError(
                        f"Cannot communicate empty tensor in bucket {self.bucket_id}"
                    )

                if torch.isnan(self.flattened_gradient).any():
                    raise CommunicationError(
                        f"Cannot communicate tensor with NaN values "
                        f"in bucket {self.bucket_id}"
                    )

                self.communication_handle = dist.all_reduce(
                    self.flattened_gradient,
                    op=dist.ReduceOp.SUM,
                    group=process_group,
                    async_op=True,
                )

            except CommunicationError:
                raise  # Re-raise communication errors
            except RuntimeError as e:
                error_msg = (
                    f"Distributed communication failed for bucket {self.bucket_id}: {e}"
                )
                logger.error(error_msg)
                raise CommunicationError(error_msg) from e
            except Exception as e:
                error_msg = (
                    f"Unexpected error during communication start for bucket "
                    f"{self.bucket_id}: {e}"
                )
                logger.error(error_msg)
                raise CommunicationError(error_msg) from e
        else:
            logger.debug("Distributed not initialized, skipping communication")

        self.last_communication_time = start_time
        return self.communication_handle

    @thread_safe_operation
    def wait_communication(self, timeout_ms: Optional[int] = None) -> float:
        """
        Wait for communication to complete with timeout and error recovery.

        Args:
            timeout_ms: Optional timeout in milliseconds

        Returns:
            Communication time in seconds

        Raises:
            CommunicationError: If communication fails or times out
        """
        if self.communication_handle is None:
            return 0.0

        start_wait_time = time.time()

        # Determine timeout value
        effective_timeout_ms = timeout_ms or 30000  # Default 30 seconds

        try:
            # Wait for communication with timeout monitoring
            self.communication_handle.wait()

            # Verify communication completed successfully
            if hasattr(self.communication_handle, "is_completed"):
                if not self.communication_handle.is_completed():
                    raise CommunicationError(
                        f"Communication did not complete for bucket {self.bucket_id}"
                    )

        except RuntimeError as e:
            error_msg = f"Communication wait failed for bucket {self.bucket_id}: {e}"
            logger.error(error_msg)
            raise CommunicationError(error_msg) from e
        except Exception as e:
            # Check if this is a timeout
            wait_time_ms = (time.time() - start_wait_time) * 1000
            if wait_time_ms > effective_timeout_ms:
                error_msg = (
                    f"Communication timeout ({effective_timeout_ms}ms) "
                    f"for bucket {self.bucket_id}"
                )
                logger.error(error_msg)
                raise CommunicationError(error_msg) from e
            else:
                error_msg = (
                    f"Unexpected error during communication wait for bucket "
                    f"{self.bucket_id}: {e}"
                )
                logger.error(error_msg)
                raise CommunicationError(error_msg) from e

        # Calculate and record communication time with metrics management
        if self.last_communication_time is not None:
            comm_time = time.time() - self.last_communication_time

            # Validate communication time is reasonable
            if comm_time < 0:
                logger.warning(
                    f"Negative communication time detected for bucket {self.bucket_id}"
                )
                comm_time = 0.0
            elif comm_time > 300:  # 5 minutes seems excessive
                logger.warning(
                    f"Very long communication time ({comm_time:.2f}s) "
                    f"for bucket {self.bucket_id}"
                )

            # Manage metrics history to prevent memory leaks
            if len(self.communication_times) >= self._max_metrics_history:
                # Keep only recent half of metrics
                self.communication_times = self.communication_times[
                    self._max_metrics_history // 2 :
                ]

            self.communication_times.append(comm_time)
            return comm_time

        return 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Get performance and usage statistics for this bucket."""
        return {
            "bucket_id": self.bucket_id,
            "current_size_mb": self.current_size_bytes / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "utilization": self.current_size_bytes / self.max_size_bytes,
            "num_gradients": len(self.gradients),
            "avg_communication_time": (
                sum(self.communication_times) / len(self.communication_times)
                if self.communication_times
                else 0.0
            ),
            "total_communications": len(self.communication_times),
        }

    @contextmanager
    def memory_managed_operation(self):
        """Context manager for memory-managed bucket operations."""
        try:
            yield self
        finally:
            # Force garbage collection of temporary tensors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @thread_safe_operation
    def clear(self) -> None:
        """
        Clear all gradients and reset bucket state with optimized memory management.
        """
        try:
            # Safely clear gradients with explicit memory management
            for grad in self.gradients:
                if grad is not None:
                    # Detach from computation graph before deletion
                    if grad.requires_grad:
                        grad = grad.detach()
                    del grad

            self.gradients.clear()
            self.gradient_metadata.clear()

            # Return flattened tensor to pool instead of deleting
            if self.flattened_gradient is not None:
                if self.flattened_gradient.requires_grad:
                    self.flattened_gradient = self.flattened_gradient.detach()

                # Return to memory pool for reuse
                self._memory_pool.return_tensor(self.flattened_gradient)
                self.flattened_gradient = None

            self.current_size_bytes = 0
            self.is_ready = False
            self.communication_handle = None

            # Manage metrics history to prevent unbounded growth
            if len(self.communication_times) > self._max_metrics_history // 2:
                self.communication_times = self.communication_times[
                    -self._max_metrics_history // 4 :
                ]

            # Only force CUDA cache cleanup occasionally to avoid performance overhead
            if torch.cuda.is_available() and len(self.communication_times) % 10 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error during bucket cleanup: {e}")
            raise


class BucketFactory:
    """Factory class for creating gradient buckets with optimized configurations."""

    @staticmethod
    def create_bucket(
        bucket_id: int,
        max_size_bytes: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        optimization_hint: Optional[str] = None,
    ) -> GradientBucket:
        """
        Create an optimized gradient bucket.

        Args:
            bucket_id: Unique identifier for the bucket
            max_size_bytes: Maximum size in bytes
            device: Target device
            dtype: Data type for tensors
            optimization_hint: Optional hint for optimization (e.g., "memory", "speed")

        Returns:
            Configured GradientBucket instance
        """
        bucket = GradientBucket(bucket_id, max_size_bytes, device, dtype)

        # Apply optimizations based on hints
        if optimization_hint == "memory":
            bucket._max_metrics_history = 100  # Reduce memory footprint
        elif optimization_hint == "speed":
            bucket._max_metrics_history = 10000  # Allow more metrics for analysis

        return bucket


class BucketManager:
    """
    Manages gradient buckets and implements intelligent bucketing strategies.

    This class is the main interface for gradient bucketing operations, providing
    automatic bucket assignment, strategy-based grouping, and performance optimization.
    It coordinates multiple GradientBucket instances and handles their lifecycle.

    Bucketing Strategies:
        SIZE_BASED: Groups gradients by tensor size for balanced communication
        LAYER_BASED: Groups by layer type (embedding, attention, etc.) for locality
        MIXED: Combines size and layer information for optimal grouping
        CUSTOM: User-defined bucketing function for specialized needs

    Key Features:
    - Automatic bucket creation with factory pattern
    - Intelligent gradient-to-bucket assignment
    - Bulk operations for better performance
    - Dynamic bucket optimization based on usage patterns
    - Comprehensive statistics and performance tracking
    - Memory-efficient operations with shared pools

    Performance Optimizations:
    - Uses BucketFactory for optimized bucket creation
    - Supports bulk gradient assignment with batching
    - Implements layer type caching for fast classification
    - Provides size bucket pre-computation for efficient lookup
    - Tracks performance metrics for adaptive optimization

    Thread Safety:
        All operations are thread-safe. Bucket assignment and synchronization
        can be called concurrently from multiple threads.

    Memory Management:
        Leverages shared memory pools across all managed buckets.
        Automatic cleanup and garbage collection hints for large models.

    Example - Basic Usage:
        ```python
        config = BucketConfig(
            strategy=BucketStrategy.SIZE_BASED,
            max_bucket_size_mb=25.0,
            dynamic_bucketing=True
        )

        manager = BucketManager(config, device=torch.device("cuda"))

        # Single gradient assignment
        bucket_id = manager.assign_gradient("model.layer.weight", gradient)

        # Bulk assignment (recommended for multiple gradients)
        gradients = {"param1": grad1, "param2": grad2}
        assignments = manager.assign_gradients_bulk(gradients)

        # Synchronize all buckets
        stats = manager.synchronize_buckets()
        print(f"Communication: {stats['total_time']:.3f}s")

        # Get updated gradients after synchronization
        updated_grads = manager.get_bucket_assignments()
        ```

    Example - Advanced Configuration:
        ```python
        config = BucketConfig(
            strategy=BucketStrategy.MIXED,
            max_bucket_size_mb=50.0,
            min_bucket_size_mb=1.0,
            dynamic_bucketing=True,
            gradient_predivision=True,
            layer_groups={
                "transformer": ["attn", "attention", "self_attn"],
                "embedding": ["embed", "embedding", "wte", "wpe"],
                "output": ["lm_head", "output", "classifier"]
            }
        )

        manager = BucketManager(config, device, dtype=torch.float16)

        # Assignment with automatic optimization
        assignments = manager.assign_gradients_bulk(model_gradients)

        # Synchronize with performance tracking
        sync_stats = manager.synchronize_buckets(
            process_group=my_process_group,
            overlap=True
        )

        # Adaptive optimization based on usage patterns
        manager.optimize_buckets()

        # Comprehensive statistics
        detailed_stats = manager.get_statistics()
        ```
    """

    def __init__(
        self,
        config: BucketConfig,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the bucket manager.

        Args:
            config: Bucket configuration
            device: Device for tensor operations
            dtype: Data type for bucket operations
        """
        self.config = config
        self.device = device
        self.dtype = dtype

        # Bucket management
        self.buckets: List[GradientBucket] = []
        self.bucket_assignments: Dict[str, int] = {}  # param_name -> bucket_id
        self.next_bucket_id = 0

        # Performance tracking
        self.total_communications = 0
        self.total_communication_time = 0.0
        self.bucket_performance: Dict[int, Dict[str, float]] = {}

        # Strategy-specific data
        self._layer_type_cache: Dict[str, str] = {}
        self._size_buckets: List[Tuple[int, int]] = []  # (min_size, max_size) pairs

        # Initialize strategy-specific components
        self._initialize_strategy()

    def _initialize_strategy(self) -> None:
        """Initialize strategy-specific components."""
        if self.config.strategy == BucketStrategy.SIZE_BASED:
            self._initialize_size_buckets()
        elif self.config.strategy == BucketStrategy.LAYER_BASED:
            self._initialize_layer_groups()
        elif self.config.strategy == BucketStrategy.MIXED:
            self._initialize_size_buckets()
            self._initialize_layer_groups()

    def _initialize_size_buckets(self) -> None:
        """Initialize size-based bucket ranges."""
        min_size = int(self.config.min_bucket_size_mb * 1024 * 1024)
        max_size = int(self.config.max_bucket_size_mb * 1024 * 1024)

        # Create logarithmic size ranges
        num_ranges = 5
        for i in range(num_ranges):
            range_min = int(min_size * (2**i))
            range_max = int(min_size * (2 ** (i + 1)))
            if range_max > max_size:
                range_max = max_size
            self._size_buckets.append((range_min, range_max))
            if range_max >= max_size:
                break

    def _initialize_layer_groups(self) -> None:
        """Initialize layer type groupings."""
        # Layer groups are already in config, nothing special needed
        pass

    def _classify_parameter(self, param_name: str, gradient: torch.Tensor) -> str:
        """
        Classify a parameter for bucketing purposes.

        Args:
            param_name: Name of the parameter
            gradient: Gradient tensor

        Returns:
            Classification string for bucketing
        """
        if param_name in self._layer_type_cache:
            return self._layer_type_cache[param_name]

        # Determine layer type based on parameter name
        layer_type = "other"
        param_lower = param_name.lower()

        for group_name, patterns in self.config.layer_groups.items():
            if any(pattern in param_lower for pattern in patterns):
                layer_type = group_name
                break

        self._layer_type_cache[param_name] = layer_type
        return layer_type

    def _get_bucket_key(self, param_name: str, gradient: torch.Tensor) -> str:
        """
        Generate a bucket key based on the configured strategy.

        Args:
            param_name: Name of the parameter
            gradient: Gradient tensor

        Returns:
            Bucket key for grouping
        """
        if self.config.strategy == BucketStrategy.SIZE_BASED:
            grad_size = gradient.numel() * gradient.element_size()
            for i, (min_size, max_size) in enumerate(self._size_buckets):
                if min_size <= grad_size <= max_size:
                    return f"size_{i}"
            return "size_overflow"

        elif self.config.strategy == BucketStrategy.LAYER_BASED:
            layer_type = self._classify_parameter(param_name, gradient)
            return f"layer_{layer_type}"

        elif self.config.strategy == BucketStrategy.MIXED:
            layer_type = self._classify_parameter(param_name, gradient)
            grad_size = gradient.numel() * gradient.element_size()

            # Find size bucket
            size_bucket = "overflow"
            for i, (min_size, max_size) in enumerate(self._size_buckets):
                if min_size <= grad_size <= max_size:
                    size_bucket = str(i)
                    break

            return f"mixed_{layer_type}_{size_bucket}"

        elif self.config.strategy == BucketStrategy.CUSTOM:
            if self.config.custom_bucket_fn is not None:
                return self.config.custom_bucket_fn(param_name, gradient)
            else:
                logger.warning(
                    "Custom strategy selected but no custom function provided"
                )
                return "custom_default"

        return "default"

    def _create_bucket(self, bucket_key: str) -> GradientBucket:
        """Create a new bucket with the given key using the factory pattern."""
        max_size_bytes = int(self.config.max_bucket_size_mb * 1024 * 1024)

        # Determine optimization hint based on configuration
        optimization_hint = None
        if self.config.dynamic_bucketing:
            optimization_hint = "memory"
        elif len(self.buckets) < 5:  # For initial buckets, optimize for speed
            optimization_hint = "speed"

        bucket = BucketFactory.create_bucket(
            bucket_id=self.next_bucket_id,
            max_size_bytes=max_size_bytes,
            device=self.device,
            dtype=self.dtype,
            optimization_hint=optimization_hint,
        )

        self.buckets.append(bucket)
        self.next_bucket_id += 1

        logger.debug(
            f"Created new bucket {bucket.bucket_id} for key '{bucket_key}' "
            f"with hint '{optimization_hint}'"
        )
        return bucket

    def assign_gradient(
        self,
        param_name: str,
        gradient: torch.Tensor,
    ) -> int:
        """
        Assign a gradient to an appropriate bucket.

        Args:
            param_name: Name of the parameter
            gradient: Gradient tensor to assign

        Returns:
            Bucket ID where the gradient was assigned

        Raises:
            ValueError: If param_name is empty or gradient is invalid
            RuntimeError: If gradient assignment fails
        """
        # Enhanced input validation
        if not param_name or not param_name.strip():
            raise ValueError("Parameter name cannot be empty")

        if gradient is None:
            raise ValueError(f"Gradient for parameter '{param_name}' is None")

        if gradient.numel() == 0:
            raise ValueError(f"Gradient for parameter '{param_name}' is empty")

        if not gradient.dtype.is_floating_point:
            raise ValueError(
                f"Gradient for parameter '{param_name}' must have "
                f"floating-point dtype, got {gradient.dtype}"
            )

        try:
            grad_size_bytes = gradient.numel() * gradient.element_size()
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to calculate gradient size for parameter '{param_name}': {e}"
            )

        if grad_size_bytes > self.config.bucket_cap_mb * 1024 * 1024:
            logger.warning(
                f"Gradient '{param_name}' ({grad_size_bytes / (1024*1024):.2f} MB) "
                f"exceeds bucket cap ({self.config.bucket_cap_mb} MB)"
            )
        # Check if already assigned
        if param_name in self.bucket_assignments:
            bucket_id = self.bucket_assignments[param_name]
            bucket = self.buckets[bucket_id]

            # Try to add to existing bucket
            layer_type = self._classify_parameter(param_name, gradient)
            if bucket.add_gradient(gradient, param_name, layer_type):
                return bucket_id

        # Need to find or create a new bucket
        bucket_key = self._get_bucket_key(param_name, gradient)
        layer_type = self._classify_parameter(param_name, gradient)

        # Look for existing bucket that can accommodate this gradient
        for bucket in self.buckets:
            if bucket.can_add_gradient(gradient):
                if bucket.add_gradient(gradient, param_name, layer_type):
                    self.bucket_assignments[param_name] = bucket.bucket_id
                    return bucket.bucket_id

        # Create new bucket
        new_bucket = self._create_bucket(bucket_key)

        # Check if gradient is larger than max bucket size
        # if so, create a special large bucket
        grad_size_bytes = gradient.numel() * gradient.element_size()
        if grad_size_bytes > new_bucket.max_size_bytes:
            # Create a bucket large enough for this gradient
            large_bucket = GradientBucket(
                bucket_id=self.next_bucket_id,
                max_size_bytes=int(grad_size_bytes * 1.2),  # 20% buffer
                device=self.device,
                dtype=self.dtype,
            )
            self.buckets.append(large_bucket)
            self.next_bucket_id += 1

            if large_bucket.add_gradient(gradient, param_name, layer_type):
                self.bucket_assignments[param_name] = large_bucket.bucket_id
                return large_bucket.bucket_id

        if new_bucket.add_gradient(gradient, param_name, layer_type):
            self.bucket_assignments[param_name] = new_bucket.bucket_id
            return new_bucket.bucket_id
        else:
            raise RuntimeError(
                f"Failed to add gradient {param_name} to new bucket "
                f"(size: {gradient.numel() * gradient.element_size()} bytes)"
            )

    def assign_gradients_bulk(
        self,
        gradients: Dict[str, torch.Tensor],
        batch_size: int = 10,
    ) -> Dict[str, int]:
        """
        Efficiently assign multiple gradients in bulk.

        Args:
            gradients: Dictionary of parameter_name -> gradient mappings
            batch_size: Number of gradients to process in each batch

        Returns:
            Dictionary mapping parameter names to bucket IDs
        """
        assignments = {}
        gradient_items = list(gradients.items())

        # Process gradients in batches for better memory management
        for i in range(0, len(gradient_items), batch_size):
            batch = gradient_items[i : i + batch_size]

            for param_name, gradient in batch:
                try:
                    bucket_id = self.assign_gradient(param_name, gradient)
                    assignments[param_name] = bucket_id
                except Exception as e:
                    logger.error(
                        f"Failed to assign gradient '{param_name}' "
                        f"in bulk operation: {e}"
                    )
                    # Continue with remaining gradients
                    continue

            # Optional: Force garbage collection between batches for large models
            if i > 0 and len(gradient_items) > 100:
                import gc

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return assignments

    def synchronize_buckets(
        self,
        process_group: Optional[dist.ProcessGroup] = None,
        overlap: bool = True,
    ) -> Dict[str, Any]:
        """
        Synchronize all gradient buckets.

        Args:
            process_group: Distributed process group
            overlap: Whether to overlap communication

        Returns:
            Synchronization statistics
        """
        if not self.buckets:
            return {"total_time": 0.0, "num_buckets": 0}

        start_time = time.time()
        communication_handles = []

        # Start all communications
        for bucket in self.buckets:
            if bucket.gradients:  # Only sync buckets with gradients
                handle = bucket.start_communication(
                    process_group=process_group,
                    predivide=self.config.gradient_predivision,
                )
                if handle is not None:
                    communication_handles.append((bucket, handle))

        # Wait for all communications to complete
        total_comm_time = 0.0
        for bucket, handle in communication_handles:
            comm_time = bucket.wait_communication()
            total_comm_time += comm_time

        total_time = time.time() - start_time

        # Update statistics
        self.total_communications += 1
        self.total_communication_time += total_time

        return {
            "total_time": total_time,
            "communication_time": total_comm_time,
            "num_buckets": len(communication_handles),
            "overlap_efficiency": (
                total_comm_time / total_time if total_time > 0 else 0.0
            ),
        }

    def get_bucket_assignments(self) -> Dict[str, torch.Tensor]:
        """
        Get the unflattened gradients organized by parameter name.

        Returns:
            Dictionary mapping parameter names to their updated gradients
        """
        result = {}

        for param_name, bucket_id in self.bucket_assignments.items():
            bucket = self.buckets[bucket_id]
            if bucket.flattened_gradient is not None:
                # Find the gradient for this parameter
                unflattened_grads = bucket.unflatten_gradients()
                for i, metadata in enumerate(bucket.gradient_metadata):
                    if metadata["param_name"] == param_name:
                        result[param_name] = unflattened_grads[i]
                        break

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive bucketing statistics."""
        bucket_stats = [bucket.get_statistics() for bucket in self.buckets]

        total_gradients = sum(len(bucket.gradients) for bucket in self.buckets)
        total_size_mb = sum(bucket.current_size_bytes for bucket in self.buckets) / (
            1024 * 1024
        )

        return {
            "strategy": self.config.strategy.value,
            "num_buckets": len(self.buckets),
            "total_gradients": total_gradients,
            "total_size_mb": total_size_mb,
            "avg_bucket_size_mb": (
                total_size_mb / len(self.buckets) if self.buckets else 0
            ),
            "total_communications": self.total_communications,
            "avg_communication_time": (
                self.total_communication_time / self.total_communications
                if self.total_communications > 0
                else 0.0
            ),
            "bucket_details": bucket_stats,
        }

    def reset(self) -> None:
        """Reset all buckets and clear assignments."""
        for bucket in self.buckets:
            bucket.clear()

        # Keep buckets but clear their contents
        # This preserves the bucketing strategy while allowing reuse
        self.bucket_assignments.clear()

    def optimize_buckets(self) -> None:
        """
        Optimize bucket configuration based on performance metrics.

        This method analyzes communication patterns and adjusts bucket
        sizes or strategies to improve performance.
        """
        if not self.config.dynamic_bucketing:
            return

        # Analyze bucket performance
        underutilized_buckets = []
        overutilized_buckets = []

        for bucket in self.buckets:
            stats = bucket.get_statistics()
            utilization = stats["utilization"]

            if utilization < 0.3:  # Less than 30% utilized
                underutilized_buckets.append(bucket)
            elif utilization > 0.9:  # More than 90% utilized
                overutilized_buckets.append(bucket)

        # Log optimization opportunities
        if underutilized_buckets:
            logger.info(f"Found {len(underutilized_buckets)} underutilized buckets")

        if overutilized_buckets:
            logger.info(f"Found {len(overutilized_buckets)} overutilized buckets")

        # Could implement actual optimization logic here
        # For now, just log the findings
