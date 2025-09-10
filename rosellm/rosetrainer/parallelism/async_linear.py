"""
Async Gradient Allreduce Implementation

This module implements asynchronous gradient allreduce operations for distributed
training.
The implementation overlaps gradient computation and communication to improve
training throughput.

Key features:
- Gradient bucketing for efficient batched communication
- Async operations with proper synchronization
- Memory buffer management for communication
- Support for both immediate and bucketed strategies
"""

import logging
import threading
import time
import warnings
from collections import deque
from typing import Any, Dict, List, Optional, Set

import torch
import torch.distributed as dist

from ..memory.global_memory_buffer import BufferType, allocate_tensor, release_tensor
from .async_config import AsyncAllreduceConfig, AsyncAllreduceStrategy
from .parallel_state import get_data_parallel_group, get_data_parallel_size
from .parallel_state import get_global_memory_buffer as get_buffer_from_state

logger = logging.getLogger(__name__)


class GradientBucket:
    """
    Manages a bucket of gradients for batched allreduce operations.

    Each bucket accumulates gradients until it reaches the configured size,
    then triggers an async allreduce operation.
    """

    def __init__(
        self,
        bucket_id: int,
        max_size: int,
        device: torch.device,
        dtype: torch.dtype,
        process_group: dist.ProcessGroup,
    ) -> None:
        """
        Initialize gradient bucket.

        Args:
            bucket_id: Unique identifier for this bucket
            max_size: Maximum size of bucket in bytes
            device: Device to allocate buffers on
            dtype: Data type for gradient tensors
            process_group: Process group for allreduce operations
        """
        self.bucket_id = bucket_id
        self.max_size = max_size
        self.device = device
        self.dtype = dtype
        self.process_group = process_group

        # Bucket state
        self.gradients: List[torch.Tensor] = []
        self.gradient_views: List[torch.Tensor] = []
        self.current_size = 0
        self.is_ready = False
        self.allreduce_handle: Optional[dist.Work] = None

        # Communication buffer
        self.buffer: Optional[torch.Tensor] = None
        self.buffer_offset = 0

        # Synchronization
        self._lock = threading.Lock()

    def add_gradient(self, gradient: torch.Tensor) -> bool:
        """
        Add a gradient tensor to the bucket.

        Args:
            gradient: Gradient tensor to add

        Returns:
            True if gradient was added, False if bucket is full
        """
        with self._lock:
            grad_size = gradient.numel() * gradient.element_size()

            if self.current_size + grad_size > self.max_size and self.gradients:
                return False  # Bucket is full

            # Create a view of the gradient for this bucket
            grad_view = gradient.view(-1)
            self.gradients.append(gradient)
            self.gradient_views.append(grad_view)
            self.current_size += grad_size

            return True

    def prepare_buffer(self) -> None:
        """Prepare the communication buffer for allreduce."""
        if not self.gradient_views:
            return

        with self._lock:
            total_elements = sum(grad.numel() for grad in self.gradient_views)

            # Allocate buffer from global memory manager if available
            buffer_manager = get_buffer_from_state()
            if buffer_manager is not None:
                self.buffer = allocate_tensor(
                    (total_elements,),
                    dtype=self.dtype,
                    device=self.device,
                    buffer_type=BufferType.COMMUNICATION,
                    caller_info=f"GradientBucket.{self.bucket_id}",
                )
            else:
                self.buffer = torch.zeros(
                    total_elements, dtype=self.dtype, device=self.device
                )

            # Copy gradients into buffer
            offset = 0
            for grad_view in self.gradient_views:
                self.buffer[offset : offset + grad_view.numel()].copy_(grad_view)
                offset += grad_view.numel()

    def start_allreduce(self) -> Optional[dist.Work]:
        """
        Start async allreduce operation.

        Returns:
            Work handle for the async operation, or None if already started
        """
        with self._lock:
            if self.allreduce_handle is not None:
                return self.allreduce_handle

            if self.buffer is None:
                self.prepare_buffer()

            if self.buffer is not None and self.buffer.numel() > 0:
                self.allreduce_handle = dist.all_reduce(
                    self.buffer, group=self.process_group, async_op=True
                )
                return self.allreduce_handle

        return None

    def wait_and_copy_back(self) -> None:
        """Wait for allreduce to complete and copy results back to gradients."""
        with self._lock:
            if self.allreduce_handle is None:
                return

            # Wait for allreduce to complete
            self.allreduce_handle.wait()

            if self.buffer is None:
                return

            # Copy results back to original gradients
            offset = 0
            for grad_view in self.gradient_views:
                grad_view.copy_(self.buffer[offset : offset + grad_view.numel()])
                offset += grad_view.numel()

            # Cleanup
            self.allreduce_handle = None
            if self.buffer is not None:
                buffer_manager = get_buffer_from_state()
                if buffer_manager is not None:
                    release_tensor(self.buffer)
                self.buffer = None

    def reset(self) -> None:
        """Reset the bucket for the next iteration."""
        with self._lock:
            self.gradients.clear()
            self.gradient_views.clear()
            self.current_size = 0
            self.buffer_offset = 0
            self.is_ready = False

            if self.buffer is not None:
                buffer_manager = get_buffer_from_state()
                if buffer_manager is not None:
                    release_tensor(self.buffer)
                self.buffer = None

            self.allreduce_handle = None


class AsyncGradientAllreduce:
    """
    Manager for asynchronous gradient allreduce operations.

    This class implements various strategies for overlapping gradient computation
    and communication in distributed training scenarios.
    """

    def __init__(
        self,
        config: AsyncAllreduceConfig,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        """
        Initialize async gradient allreduce manager.

        Args:
            config: Configuration for async allreduce operations
            process_group: Process group for communication (defaults to data
                           parallel group)
        """
        self.config = config
        self.process_group = process_group or get_data_parallel_group()
        self.world_size = get_data_parallel_size()

        if self.process_group is None:
            warnings.warn(
                "No process group available for async allreduce. "
                "Operations will be performed synchronously."
            )

        # Validate configuration
        self.config.validate_for_world_size(self.world_size)

        # State management
        self.step_count = 0
        self.buckets: List[GradientBucket] = []
        self.current_bucket_idx = 0
        self.pending_handles: Set[dist.Work] = set()

        # Performance tracking
        self.communication_times: deque = deque(maxlen=100)
        self.overlap_ratios: deque = deque(maxlen=100)

        # Thread safety
        self._lock = threading.Lock()

        # Initialize buckets
        self._initialize_buckets()

        logger.info(
            f"Initialized AsyncGradientAllreduce with {len(self.buckets)} "
            f"buckets, strategy={self.config.strategy.value}, "
            f"world_size={self.world_size}"
        )

    def _initialize_buckets(self) -> None:
        """Initialize gradient buckets based on configuration."""
        if not self.config.enabled or self.world_size <= 1:
            return

        if torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            device = torch.device("cpu")
        dtype = torch.float32  # Default dtype for gradients

        for i in range(self.config.max_buckets):
            if self.process_group is not None:
                bucket = GradientBucket(
                    bucket_id=i,
                    max_size=self.config.bucket_size,
                    device=device,
                    dtype=dtype,
                    process_group=self.process_group,
                )
                self.buckets.append(bucket)

    def register_gradient_hook(
        self,
        parameter: torch.nn.Parameter,
        layer_name: Optional[str] = None,
    ) -> None:
        """
        Register a gradient hook for asynchronous allreduce.

        Args:
            parameter: Model parameter to register hook for
            layer_name: Optional name of the layer (for priority/skip lists)
        """
        if not self.config.enabled or self.world_size <= 1:
            return

        if (
            layer_name
            and self.config.skip_layers
            and layer_name in self.config.skip_layers
        ):
            return

        def gradient_hook(grad: torch.Tensor) -> Optional[torch.Tensor]:
            if self.step_count < self.config.warmup_steps:
                return None  # Skip async allreduce during warmup

            self._handle_gradient(grad, layer_name)
            return None

        parameter.register_hook(gradient_hook)

    def _handle_gradient(
        self,
        gradient: torch.Tensor,
        layer_name: Optional[str] = None,
    ) -> None:
        """
        Handle a gradient tensor based on the configured strategy.

        Args:
            gradient: Gradient tensor to process
            layer_name: Optional layer name for priority handling
        """
        if self.config.strategy == AsyncAllreduceStrategy.IMMEDIATE:
            self._handle_immediate_allreduce(gradient)
        elif self.config.strategy == AsyncAllreduceStrategy.BUCKETED:
            self._handle_bucketed_allreduce(gradient)
        elif self.config.strategy == AsyncAllreduceStrategy.LAYERWISE:
            self._handle_layerwise_allreduce(gradient, layer_name)
        elif self.config.strategy == AsyncAllreduceStrategy.PRIORITY_BASED:
            self._handle_priority_allreduce(gradient, layer_name)
        else:
            logger.warning(f"Unknown async allreduce strategy: {self.config.strategy}")

    def _handle_immediate_allreduce(self, gradient: torch.Tensor) -> None:
        """Handle immediate async allreduce for a single gradient."""
        if self.process_group is None:
            return

        start_time = time.time()

        # Pre-divide gradient if configured
        if self.config.gradient_predivision:
            gradient.div_(self.world_size)

        # Start async allreduce
        handle = dist.all_reduce(gradient, group=self.process_group, async_op=True)

        with self._lock:
            if handle is not None:
                self.pending_handles.add(handle)

        if self.config.log_communication_stats:
            comm_time = time.time() - start_time
            self.communication_times.append(comm_time)

    def _handle_bucketed_allreduce(self, gradient: torch.Tensor) -> None:
        """Handle bucketed async allreduce."""
        if not self.buckets:
            return

        with self._lock:
            # Try to add gradient to current bucket
            current_bucket = self.buckets[self.current_bucket_idx]

            if not current_bucket.add_gradient(gradient):
                # Current bucket is full, start allreduce and move to next bucket
                current_bucket.start_allreduce()

                # Move to next bucket
                self.current_bucket_idx = (self.current_bucket_idx + 1) % len(
                    self.buckets
                )
                next_bucket = self.buckets[self.current_bucket_idx]

                # Wait for next bucket if it's still processing
                if next_bucket.allreduce_handle is not None:
                    next_bucket.wait_and_copy_back()
                    next_bucket.reset()

                # Add gradient to next bucket
                next_bucket.add_gradient(gradient)

    def _handle_layerwise_allreduce(
        self,
        gradient: torch.Tensor,
        layer_name: Optional[str] = None,
    ) -> None:
        """Handle layer-wise async allreduce."""
        # For now, delegate to immediate allreduce
        # TODO: Implement proper layer-wise batching
        self._handle_immediate_allreduce(gradient)

    def _handle_priority_allreduce(
        self,
        gradient: torch.Tensor,
        layer_name: Optional[str] = None,
    ) -> None:
        """Handle priority-based async allreduce."""
        is_priority = (
            layer_name is not None
            and self.config.priority_layers is not None
            and layer_name in self.config.priority_layers
        )

        if is_priority:
            # Process priority layers immediately
            self._handle_immediate_allreduce(gradient)
        else:
            # Use bucketed approach for non-priority layers
            self._handle_bucketed_allreduce(gradient)

    def synchronize(self) -> None:
        """Synchronize all pending async operations."""
        with self._lock:
            # Wait for all pending immediate allreduce operations
            for handle in self.pending_handles:
                if not handle.is_completed():
                    handle.wait()
            self.pending_handles.clear()

            # Synchronize all buckets
            for bucket in self.buckets:
                if bucket.allreduce_handle is not None:
                    bucket.wait_and_copy_back()
                    bucket.reset()

    def step(self) -> None:
        """
        Complete one training step for async allreduce.

        This should be called at the end of each training step to ensure
        all gradients are properly synchronized.
        """
        # Synchronize any remaining operations
        self.synchronize()

        # Increment step counter
        self.step_count += 1

        # Log statistics if enabled
        if self.config.log_communication_stats and self.communication_times:
            avg_comm_time = sum(self.communication_times) / len(
                self.communication_times
            )
            logger.info(
                f"Step {self.step_count}: avg_comm_time={avg_comm_time:.4f}s, "
                f"pending_ops={len(self.pending_handles)}"
            )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get performance statistics for the async allreduce manager.

        Returns:
            Dictionary containing performance metrics
        """
        stats = {
            "step_count": self.step_count,
            "world_size": self.world_size,
            "num_buckets": len(self.buckets),
            "config": self.config,
        }

        if self.communication_times:
            stats.update(
                {
                    "avg_comm_time": sum(self.communication_times)
                    / len(self.communication_times),
                    "min_comm_time": min(self.communication_times),
                    "max_comm_time": max(self.communication_times),
                }
            )

        if self.overlap_ratios:
            stats.update(
                {
                    "avg_overlap_ratio": sum(self.overlap_ratios)
                    / len(self.overlap_ratios),
                    "min_overlap_ratio": min(self.overlap_ratios),
                    "max_overlap_ratio": max(self.overlap_ratios),
                }
            )

        return stats

    def reset_statistics(self) -> None:
        """Reset performance statistics."""
        self.communication_times.clear()
        self.overlap_ratios.clear()
        self.step_count = 0

    def __del__(self) -> None:
        """Cleanup resources when object is destroyed."""
        try:
            self.synchronize()
        except Exception:
            # Ignore errors during cleanup
            pass


# Global async allreduce manager instance
_global_async_manager: Optional[AsyncGradientAllreduce] = None
_manager_lock = threading.Lock()


def get_async_allreduce_manager() -> Optional[AsyncGradientAllreduce]:
    """Get the global async allreduce manager instance."""
    global _global_async_manager
    with _manager_lock:
        return _global_async_manager


def initialize_async_allreduce(
    config: AsyncAllreduceConfig,
    process_group: Optional[dist.ProcessGroup] = None,
) -> AsyncGradientAllreduce:
    """
    Initialize the global async allreduce manager.

    Args:
        config: Configuration for async allreduce operations
        process_group: Process group for communication

    Returns:
        Initialized AsyncGradientAllreduce instance
    """
    global _global_async_manager
    with _manager_lock:
        if _global_async_manager is not None:
            logger.warning(
                "Async allreduce manager already initialized. "
                "Replacing existing instance."
            )

        _global_async_manager = AsyncGradientAllreduce(config, process_group)
        return _global_async_manager


def destroy_async_allreduce() -> None:
    """Destroy the global async allreduce manager."""
    global _global_async_manager
    with _manager_lock:
        if _global_async_manager is not None:
            _global_async_manager.synchronize()
            _global_async_manager = None


def register_parameter_for_async_allreduce(
    parameter: torch.nn.Parameter,
    layer_name: Optional[str] = None,
) -> None:
    """
    Register a parameter for async gradient allreduce.

    Args:
        parameter: Model parameter to register
        layer_name: Optional name of the layer
    """
    manager = get_async_allreduce_manager()
    if manager is not None:
        manager.register_gradient_hook(parameter, layer_name)


def async_allreduce_step() -> None:
    """Complete one training step for async allreduce."""
    manager = get_async_allreduce_manager()
    if manager is not None:
        manager.step()


def sync_async_allreduce() -> None:
    """Synchronize all pending async allreduce operations."""
    manager = get_async_allreduce_manager()
    if manager is not None:
        manager.synchronize()
