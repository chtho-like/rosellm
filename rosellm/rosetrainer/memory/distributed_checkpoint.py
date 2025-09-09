"""
Advanced Distributed Activation Checkpointing for RoseTrainer

This module provides comprehensive distributed activation checkpointing that coordinates
checkpoint decisions across multiple parallel dimensions (TP, PP, DP, CP, EP).
It extends the existing checkpointing infrastructure with distributed coordination,
memory profiling across ranks, and model parallel activation management.

Key Features:
- Cross-rank checkpoint coordination and synchronization
- Model parallel activation management for tensor parallel layers
- Distributed memory profiling and optimization
- Advanced selective checkpointing strategies for distributed training
- Integration with all parallelism dimensions (TP, PP, DP, CP, EP)
- CUDA Graph compatibility and optimization
- Dynamic load balancing across distributed ranks

References:
[1] Chen et al., "Training Deep Nets with Sublinear Memory Cost" (2016)
[2] Kirisame et al., "Dynamic Tensor Rematerialization" (2021)
[3] NVIDIA Megatron-LM Distributed Training Architecture
[4] PyTorch Distributed Communication Primitives
"""

import enum
import logging
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Generator, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.autograd import Function

from ..parallelism import parallel_state
from .activation_checkpoint import ActivationCheckpointing, MemoryProfiler
from .selective_recompute import SelectiveCheckpointConfig, SelectiveRecomputeManager

logger = logging.getLogger(__name__)


class DistributedCheckpointError(Exception):
    """Base exception for distributed checkpointing errors."""

    pass


class CoordinationError(DistributedCheckpointError):
    """Exception raised when distributed coordination fails."""

    pass


class MemoryProfilingError(DistributedCheckpointError):
    """Exception raised when distributed memory profiling fails."""

    pass


class NetworkTimeoutError(DistributedCheckpointError):
    """Exception raised when network operations timeout."""

    pass


@dataclass
class ErrorRecoveryState:
    """State management for distributed operation error recovery.

    This class implements an exponential backoff strategy for recovering from
    distributed operation failures. It tracks failure history, timing, and
    provides intelligent retry decisions to prevent overwhelming failing systems
    while maintaining reasonable responsiveness.

    The exponential backoff formula used is:
        wait_time = base_backoff * (2 ** attempt_number)

    Attributes:
        failed_operations: List of operation names that have failed, used for
                         debugging and analysis of failure patterns
        recovery_attempts: Current number of recovery attempts for the active operation
        max_recovery_attempts: Maximum number of retry attempts before giving up
        last_error_time: Unix timestamp of the last error occurrence
        error_backoff_seconds: Base backoff time in seconds for exponential backoff
    """

    failed_operations: List[str] = field(default_factory=list)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    last_error_time: float = 0.0
    error_backoff_seconds: float = 1.0

    def should_retry(self) -> bool:
        """Determine if operation should be retried based on backoff policy.

        Uses exponential backoff to determine if enough time has passed since the
        last error to warrant another retry attempt. This prevents overwhelming
        failing systems while providing reasonable retry intervals.

        Returns:
            True if the operation should be retried, False if max attempts reached
            or insufficient time has passed since last error

        Example:
            With default settings:
            - Attempt 1: wait 1 second
            - Attempt 2: wait 2 seconds
            - Attempt 3: wait 4 seconds
            - Attempt 4+: give up (max_recovery_attempts=3)
        """
        if self.recovery_attempts >= self.max_recovery_attempts:
            return False

        current_time = time.time()
        time_since_error = current_time - self.last_error_time

        # Exponential backoff: base_time * 2^attempts
        required_wait = self.error_backoff_seconds * (2**self.recovery_attempts)
        return bool(time_since_error >= required_wait)

    def record_error(self, operation: str) -> None:
        """Record a failed operation and update recovery state.

        This method should be called whenever an operation fails to update
        the internal state for backoff calculations and failure tracking.

        Args:
            operation: Name of the failed operation for debugging purposes
        """
        self.failed_operations.append(operation)
        self.recovery_attempts += 1
        self.last_error_time = time.time()

    def reset(self) -> None:
        """Reset error recovery state after successful operation.

        This method should be called when an operation succeeds to reset
        the recovery state for future operations. The failed_operations
        history is cleared to prevent unbounded growth.
        """
        self.failed_operations.clear()
        self.recovery_attempts = 0
        self.last_error_time = 0.0


@contextmanager
def distributed_error_recovery(
    operation_name: str,
    recovery_state: ErrorRecoveryState,
    cleanup_fn: Optional[Callable[[], None]] = None,
) -> Generator[None, None, None]:
    """Context manager for distributed error recovery with exponential backoff.

    This context manager provides robust error handling for distributed operations
    with automatic retry logic, exponential backoff, and cleanup capabilities.
    It tracks failure patterns and provides intelligent retry decisions based on
    the operation's failure history.

    Args:
        operation_name: Name of the operation being performed (used for logging)
        recovery_state: Error recovery state tracker that maintains failure counts
                       and timing information for backoff calculations
        cleanup_fn: Optional cleanup function to call on error. This function
                   should be idempotent as it may be called multiple times.

    Yields:
        None: Control is yielded to the wrapped operation

    Raises:
        Exception: The original exception is re-raised after recovery attempts
                  are exhausted or if the operation succeeds but cleanup fails

    Example:
        >>> recovery_state = ErrorRecoveryState(max_recovery_attempts=3)
        >>> with distributed_error_recovery("memory_sync", recovery_state):
        ...     perform_distributed_operation()
    """
    try:
        yield
        recovery_state.reset()  # Reset on success
    except Exception as e:
        recovery_state.record_error(operation_name)

        if cleanup_fn:
            try:
                cleanup_fn()
            except Exception as cleanup_error:
                logger.warning(f"Cleanup failed for {operation_name}: {cleanup_error}")

        # Determine if we should retry or propagate the error
        if recovery_state.should_retry():
            logger.warning(
                f"Operation {operation_name} failed "
                f"(attempt {recovery_state.recovery_attempts}), will retry: {e}"
            )
        else:
            logger.error(
                f"Operation {operation_name} failed permanently after "
                f"{recovery_state.recovery_attempts} attempts: {e}"
            )

        raise


# Constants for distributed checkpointing
DEFAULT_MEMORY_SYNC_INTERVAL_STEPS = 50
DEFAULT_REBALANCE_INTERVAL_STEPS = 100
DEFAULT_LOAD_BALANCE_THRESHOLD = 0.8
DEFAULT_MAX_MEMORY_IMBALANCE_RATIO = 1.5
DEFAULT_COMMUNICATION_TIMEOUT_SEC = 30.0
DEFAULT_ERROR_RECOVERY_ATTEMPTS = 3
DEFAULT_COORDINATION_CACHE_SIZE = 1000

# Performance constants
MEMORY_STATS_HISTORY_SIZE = 100
STRATEGY_PERFORMANCE_WINDOW = 50
CHECKPOINT_FUSION_MIN_LAYERS = 4


class DistributedCheckpointStrategy(enum.Enum):
    """Strategies for distributed checkpointing decisions.

    Available strategies:
        COORDINATED: All ranks make the same checkpoint decisions
        LOAD_BALANCED: Distribute checkpoints across ranks for memory balancing
        HIERARCHICAL: Different strategies for different parallel dimensions
        ADAPTIVE: Dynamic strategy selection based on runtime conditions
        EXPERT_AWARE: Specialized strategy for MoE models with expert parallelism
        PIPELINE_AWARE: Pipeline-specific checkpointing strategy
    """

    COORDINATED = "coordinated"  # Synchronized decisions across all ranks
    LOAD_BALANCED = "load_balanced"  # Distribute memory load across ranks
    HIERARCHICAL = "hierarchical"  # Different strategies per parallel dimension
    ADAPTIVE = "adaptive"  # Dynamic strategy selection
    EXPERT_AWARE = "expert_aware"  # MoE-specific checkpointing
    PIPELINE_AWARE = "pipeline_aware"  # Pipeline-optimized checkpointing


@dataclass
class DistributedCheckpointConfig:
    """Configuration for distributed activation checkpointing."""

    # Core distributed strategy
    strategy: DistributedCheckpointStrategy = DistributedCheckpointStrategy.COORDINATED

    # Parallelism coordination settings
    coordinate_across_tp: bool = True  # Coordinate tensor parallel ranks
    coordinate_across_pp: bool = False  # Coordinate pipeline parallel ranks
    coordinate_across_dp: bool = False  # Coordinate data parallel ranks
    coordinate_across_cp: bool = True  # Coordinate context parallel ranks
    coordinate_across_ep: bool = True  # Coordinate expert parallel ranks

    # Load balancing parameters
    enable_load_balancing: bool = True
    load_balance_threshold: float = DEFAULT_LOAD_BALANCE_THRESHOLD
    rebalance_interval: int = DEFAULT_REBALANCE_INTERVAL_STEPS

    # Memory optimization settings
    enable_cross_rank_profiling: bool = True
    memory_sync_interval: int = DEFAULT_MEMORY_SYNC_INTERVAL_STEPS
    max_memory_imbalance_ratio: float = DEFAULT_MAX_MEMORY_IMBALANCE_RATIO

    # Pipeline parallel optimization
    pipeline_bubble_optimization: bool = True
    pipeline_checkpoint_stages: Optional[
        List[int]
    ] = None  # Specific stages to checkpoint

    # Expert parallel optimization
    expert_load_balancing: bool = True
    expert_memory_threshold_mb: float = 2048.0

    # Communication optimization
    use_async_communication: bool = True
    communication_backend: str = "nccl"  # Communication backend
    communication_timeout_sec: float = DEFAULT_COMMUNICATION_TIMEOUT_SEC

    # Error recovery settings
    max_recovery_attempts: int = DEFAULT_ERROR_RECOVERY_ATTEMPTS
    error_backoff_seconds: float = 1.0
    enable_error_recovery: bool = True

    # Resource management
    coordination_cache_max_size: int = DEFAULT_COORDINATION_CACHE_SIZE
    memory_stats_history_size: int = MEMORY_STATS_HISTORY_SIZE
    enable_resource_cleanup: bool = True

    # CUDA Graph compatibility
    cuda_graph_compatible: bool = False

    # Base selective checkpointing config
    base_config: SelectiveCheckpointConfig = field(
        default_factory=SelectiveCheckpointConfig
    )

    # Advanced features
    enable_gradient_checkpointing_fusion: bool = False
    enable_dynamic_recomputation: bool = True
    checkpoint_fusion_threshold: int = CHECKPOINT_FUSION_MIN_LAYERS

    # Debugging and monitoring
    verbose_distributed: bool = False
    collect_distributed_metrics: bool = True
    log_memory_imbalances: bool = True

    def validate(self) -> None:
        """Validate distributed configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid
        """
        errors = []

        # Basic parameter validation
        if not 0 < self.load_balance_threshold <= 1.0:
            errors.append("load_balance_threshold must be between 0 and 1")
        if self.rebalance_interval <= 0:
            errors.append("rebalance_interval must be positive")
        if self.memory_sync_interval <= 0:
            errors.append("memory_sync_interval must be positive")
        if self.max_memory_imbalance_ratio < 1.0:
            errors.append("max_memory_imbalance_ratio must be >= 1.0")
        if self.expert_memory_threshold_mb <= 0:
            errors.append("expert_memory_threshold_mb must be positive")
        if self.communication_timeout_sec <= 0:
            errors.append("communication_timeout_sec must be positive")
        if self.checkpoint_fusion_threshold < 1:
            errors.append("checkpoint_fusion_threshold must be >= 1")

        # Error recovery validation
        if self.max_recovery_attempts < 1:
            errors.append("max_recovery_attempts must be >= 1")
        if self.error_backoff_seconds <= 0:
            errors.append("error_backoff_seconds must be positive")

        # Resource management validation
        if self.coordination_cache_max_size <= 0:
            errors.append("coordination_cache_max_size must be positive")
        if self.memory_stats_history_size <= 0:
            errors.append("memory_stats_history_size must be positive")

        # Communication backend validation
        valid_backends = {"nccl", "gloo", "mpi"}
        if self.communication_backend not in valid_backends:
            errors.append(f"communication_backend must be one of {valid_backends}")

        # Strategy-specific validation
        if self.strategy == DistributedCheckpointStrategy.PIPELINE_AWARE:
            if self.pipeline_checkpoint_stages is not None:
                if not all(
                    isinstance(stage, int) and stage >= 0
                    for stage in self.pipeline_checkpoint_stages
                ):
                    errors.append(
                        "pipeline_checkpoint_stages must contain non-negative integers"
                    )

        if errors:
            raise ValueError(
                f"Distributed configuration validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        # Validate base config
        self.base_config.validate()

        # Log warnings for potentially suboptimal configurations
        self._log_configuration_warnings()

    def _log_configuration_warnings(self) -> None:
        """Log warnings for potentially suboptimal configurations."""
        if self.memory_sync_interval > self.rebalance_interval:
            logger.warning(
                "memory_sync_interval > rebalance_interval may lead to "
                "suboptimal load balancing"
            )

        if self.communication_timeout_sec < 10.0:
            logger.warning(
                "communication_timeout_sec < 10s may be too short for " "large clusters"
            )

        if not self.enable_error_recovery:
            logger.warning(
                "Error recovery is disabled - distributed operations may "
                "fail permanently"
            )


@dataclass
class DistributedMemoryStats:
    """Memory statistics across distributed ranks."""

    rank: int
    tensor_parallel_rank: int
    pipeline_parallel_rank: int
    data_parallel_rank: int
    context_parallel_rank: int
    expert_parallel_rank: int

    allocated_memory_mb: float = 0.0
    reserved_memory_mb: float = 0.0
    max_allocated_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0

    active_checkpoints: int = 0
    checkpoint_memory_saved_mb: float = 0.0
    recomputation_overhead_ratio: float = 0.0

    timestamp: float = field(default_factory=time.time)


class DistributedMemoryProfiler:
    """Distributed memory profiler for cross-rank memory tracking."""

    def __init__(self, config: DistributedCheckpointConfig) -> None:
        """Initialize distributed memory profiler.

        Args:
            config: Distributed checkpointing configuration
        """
        self.config = config
        self.local_profiler = MemoryProfiler()

        # Distributed state
        self.is_distributed = parallel_state.is_initialized()
        if self.is_distributed:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()

            # Get parallel ranks
            self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
            self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
            self.dp_rank = parallel_state.get_data_parallel_rank()
            self.cp_rank = parallel_state.get_context_parallel_rank()
            self.ep_rank = parallel_state.get_expert_model_parallel_rank()
        else:
            self.world_size = 1
            self.rank = 0
            self.tp_rank = 0
            self.pp_rank = 0
            self.dp_rank = 0
            self.cp_rank = 0
            self.ep_rank = 0

        # Memory tracking with resource management
        self.local_stats: Dict[str, DistributedMemoryStats] = {}
        self.global_stats: Dict[int, DistributedMemoryStats] = {}
        self.memory_history: Deque[Dict[int, DistributedMemoryStats]] = deque(
            maxlen=config.memory_stats_history_size
        )

        # Synchronization
        self.last_sync_time = time.time()
        self._lock = threading.RLock() if config.base_config.thread_safe else None

        # Error recovery
        self.error_recovery_state = ErrorRecoveryState(
            max_recovery_attempts=config.max_recovery_attempts,
            error_backoff_seconds=config.error_backoff_seconds,
        )

        # Resource cleanup
        self._cleanup_enabled = config.enable_resource_cleanup
        self._max_local_stats_size = config.coordination_cache_max_size

    def profile_memory_distributed(
        self, layer_id: str, phase: str = "before"
    ) -> Dict[str, Any]:
        """Profile memory usage across distributed ranks.

        Args:
            layer_id: Identifier for the layer being profiled
            phase: Phase of profiling ("before" or "after")

        Returns:
            Dictionary containing local and global memory statistics
        """
        # Get local memory stats
        local_stats = self.local_profiler.profile_memory(layer_id, phase)

        # Create distributed stats entry
        if layer_id not in self.local_stats:
            self.local_stats[layer_id] = DistributedMemoryStats(
                rank=self.rank,
                tensor_parallel_rank=self.tp_rank,
                pipeline_parallel_rank=self.pp_rank,
                data_parallel_rank=self.dp_rank,
                context_parallel_rank=self.cp_rank,
                expert_parallel_rank=self.ep_rank,
            )

        stats = self.local_stats[layer_id]
        stats.allocated_memory_mb = local_stats.get("allocated_gb", 0.0) * 1024
        stats.reserved_memory_mb = local_stats.get("reserved_gb", 0.0) * 1024
        stats.max_allocated_memory_mb = local_stats.get("max_allocated_gb", 0.0) * 1024
        stats.peak_memory_mb = max(stats.peak_memory_mb, stats.allocated_memory_mb)
        stats.timestamp = time.time()

        return {
            "local_stats": local_stats,
            "distributed_stats": stats,
            "rank": self.rank,
            "parallel_ranks": {
                "tp": self.tp_rank,
                "pp": self.pp_rank,
                "dp": self.dp_rank,
                "cp": self.cp_rank,
                "ep": self.ep_rank,
            },
        }

    def should_sync_memory_stats(self) -> bool:
        """Determine if memory stats should be synchronized across ranks."""
        if not self.config.enable_cross_rank_profiling or not self.is_distributed:
            return False

        current_time = time.time()
        time_since_sync = current_time - self.last_sync_time
        return time_since_sync >= (
            self.config.memory_sync_interval * 0.1
        )  # Convert to seconds

    def sync_memory_stats(self) -> Optional[Dict[int, DistributedMemoryStats]]:
        """Synchronize memory statistics across all ranks with error recovery.

        Returns:
            Dictionary mapping rank to memory stats, or None if not distributed
        """
        if not self.is_distributed or not self.config.enable_cross_rank_profiling:
            return None

        def cleanup_on_error() -> None:
            """Cleanup function called on synchronization errors."""
            if self._cleanup_enabled and self._lock:
                with self._lock:
                    # Clear potentially corrupted stats
                    if len(self.global_stats) > self.world_size * 2:
                        self.global_stats.clear()
                        logger.info("Cleared corrupted global stats after sync error")

        if self.config.enable_error_recovery:
            with distributed_error_recovery(
                "memory_sync", self.error_recovery_state, cleanup_on_error
            ):
                return self._perform_memory_sync()
        else:
            try:
                return self._perform_memory_sync()
            except Exception as e:
                logger.error(f"Failed to sync memory stats: {e}")
                cleanup_on_error()
                return None

    def _perform_memory_sync(self) -> Optional[Dict[int, DistributedMemoryStats]]:
        """Perform the actual memory synchronization operation with optimizations.

        This method implements a three-phase synchronization process:
        1. Local stats aggregation and summarization
        2. Distributed all-gather with timeout protection
        3. Atomic global state update with resource cleanup

        The operation uses all_gather_object which is suitable for variable-sized
        data but may have higher overhead than tensor-based operations. For
        large-scale deployments, consider implementing tensor-based variants.

        Returns:
            Dictionary mapping rank to memory stats, or None on failure

        Raises:
            NetworkTimeoutError: If the distributed operation exceeds timeout
            DistributedCheckpointError: For other synchronization failures
        """
        # Phase 1: Gather local stats summary with resource cleanup
        # This aggregates all per-layer statistics into a single summary
        # to minimize communication overhead
        local_summary = self._create_local_summary()

        # Phase 2: All-gather statistics from all ranks with timeout protection
        # Pre-allocate list to avoid dynamic allocation during critical section
        gathered_stats = [None] * self.world_size

        # Set up timeout protection to prevent hanging on network failures
        # This is critical in distributed environments where network partitions
        # or node failures can cause indefinite blocks
        timeout_handle = None
        if self.config.communication_timeout_sec > 0:

            def timeout_handler():
                raise NetworkTimeoutError(
                    f"Memory sync timed out after "
                    f"{self.config.communication_timeout_sec}s"
                )

            timeout_handle = threading.Timer(
                self.config.communication_timeout_sec, timeout_handler
            )
            timeout_handle.start()

        try:
            # Perform distributed all-gather operation
            # This collects memory statistics from all participating ranks
            dist.all_gather_object(gathered_stats, local_summary)
        finally:
            # Always cancel timeout to prevent resource leaks
            if timeout_handle:
                timeout_handle.cancel()

        # Phase 3: Update global stats atomically with resource cleanup
        # Use locking to ensure thread-safe updates to shared state
        if self._lock:
            with self._lock:
                self._update_global_stats(gathered_stats)
                self._cleanup_local_stats_if_needed()
        else:
            # Non-threaded path for performance in single-threaded scenarios
            self._update_global_stats(gathered_stats)
            self._cleanup_local_stats_if_needed()

        # Update synchronization timestamp for future decisions
        self.last_sync_time = time.time()

        # Optional verbose logging for debugging distributed coordination
        if self.config.verbose_distributed:
            logger.info(
                f"Rank {self.rank}: Synchronized memory stats across "
                f"{self.world_size} ranks"
            )

        # Return a copy to prevent external mutation of internal state
        return self.global_stats.copy()

    def _cleanup_local_stats_if_needed(self) -> None:
        """Clean up local stats if they exceed maximum size."""
        if not self._cleanup_enabled:
            return

        if len(self.local_stats) > self._max_local_stats_size:
            # Remove oldest entries (simple FIFO cleanup)
            items_to_remove = len(self.local_stats) - (self._max_local_stats_size // 2)
            oldest_keys = list(self.local_stats.keys())[:items_to_remove]

            for key in oldest_keys:
                del self.local_stats[key]

            logger.debug(
                f"Cleaned up {items_to_remove} old memory stat entries "
                f"(new size: {len(self.local_stats)})"
            )

    def _create_local_summary(self) -> DistributedMemoryStats:
        """Create a summary of local memory statistics."""
        total_allocated = sum(s.allocated_memory_mb for s in self.local_stats.values())
        total_reserved = sum(s.reserved_memory_mb for s in self.local_stats.values())
        max_allocated = max(
            (s.max_allocated_memory_mb for s in self.local_stats.values()), default=0.0
        )
        peak_memory = max(
            (s.peak_memory_mb for s in self.local_stats.values()), default=0.0
        )
        active_checkpoints = sum(
            s.active_checkpoints for s in self.local_stats.values()
        )

        return DistributedMemoryStats(
            rank=self.rank,
            tensor_parallel_rank=self.tp_rank,
            pipeline_parallel_rank=self.pp_rank,
            data_parallel_rank=self.dp_rank,
            context_parallel_rank=self.cp_rank,
            expert_parallel_rank=self.ep_rank,
            allocated_memory_mb=total_allocated,
            reserved_memory_mb=total_reserved,
            max_allocated_memory_mb=max_allocated,
            peak_memory_mb=peak_memory,
            active_checkpoints=active_checkpoints,
        )

    def _update_global_stats(self, gathered_stats: List[Any]) -> None:
        """Update global statistics from gathered data."""
        self.global_stats.clear()

        for rank, stats in enumerate(gathered_stats):
            if stats is not None and isinstance(stats, DistributedMemoryStats):
                self.global_stats[rank] = stats

        # Add to history
        self.memory_history.append(self.global_stats.copy())

        # Check for memory imbalances
        if self.config.log_memory_imbalances:
            self._check_memory_imbalances()

    def _check_memory_imbalances(self) -> None:
        """Check for memory imbalances across ranks and log warnings."""
        if not self.global_stats:
            return

        memory_usage = [
            stats.allocated_memory_mb for stats in self.global_stats.values()
        ]

        if not memory_usage:
            return

        min_memory = min(memory_usage)
        max_memory = max(memory_usage)

        if min_memory > 0:
            imbalance_ratio = max_memory / min_memory

            if imbalance_ratio > self.config.max_memory_imbalance_ratio:
                logger.warning(
                    f"Memory imbalance detected: ratio {imbalance_ratio:.2f} "
                    f"(max: {max_memory:.1f} MB, min: {min_memory:.1f} MB)"
                )

                # Log per-rank details
                for rank, stats in self.global_stats.items():
                    logger.debug(
                        f"Rank {rank} (TP:{stats.tensor_parallel_rank}, "
                        f"PP:{stats.pipeline_parallel_rank}): "
                        f"{stats.allocated_memory_mb:.1f} MB"
                    )

    def get_memory_imbalance_ratio(self) -> float:
        """Get current memory imbalance ratio across ranks.

        Returns:
            Ratio of max to min memory usage, or 1.0 if no imbalance
        """
        if not self.global_stats:
            return 1.0

        memory_usage = [
            stats.allocated_memory_mb for stats in self.global_stats.values()
        ]

        if not memory_usage:
            return 1.0

        min_memory = min(memory_usage)
        max_memory = max(memory_usage)

        return max_memory / min_memory if min_memory > 0 else 1.0

    def get_distributed_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive distributed memory report.

        Returns:
            Dictionary containing distributed memory statistics and analysis
        """
        local_report = self.local_profiler.get_memory_report()

        if not self.is_distributed:
            return {
                "local_report": local_report,
                "distributed_enabled": False,
                "rank": self.rank,
            }

        # Global statistics
        global_memory_usage = []
        global_active_checkpoints = 0

        for stats in self.global_stats.values():
            global_memory_usage.append(stats.allocated_memory_mb)
            global_active_checkpoints += stats.active_checkpoints

        # Calculate distribution statistics
        if global_memory_usage:
            mean_memory = sum(global_memory_usage) / len(global_memory_usage)
            min_memory = min(global_memory_usage)
            max_memory = max(global_memory_usage)
            std_memory = (
                sum((x - mean_memory) ** 2 for x in global_memory_usage)
                / len(global_memory_usage)
            ) ** 0.5
        else:
            mean_memory = min_memory = max_memory = std_memory = 0.0

        return {
            "local_report": local_report,
            "distributed_enabled": True,
            "world_size": self.world_size,
            "rank": self.rank,
            "parallel_ranks": {
                "tensor_parallel": self.tp_rank,
                "pipeline_parallel": self.pp_rank,
                "data_parallel": self.dp_rank,
                "context_parallel": self.cp_rank,
                "expert_parallel": self.ep_rank,
            },
            "global_memory_stats": {
                "mean_memory_mb": mean_memory,
                "min_memory_mb": min_memory,
                "max_memory_mb": max_memory,
                "std_memory_mb": std_memory,
                "imbalance_ratio": self.get_memory_imbalance_ratio(),
                "total_active_checkpoints": global_active_checkpoints,
            },
            "per_rank_stats": {
                rank: {
                    "allocated_memory_mb": stats.allocated_memory_mb,
                    "active_checkpoints": stats.active_checkpoints,
                    "parallel_coords": (
                        stats.tensor_parallel_rank,
                        stats.pipeline_parallel_rank,
                        stats.data_parallel_rank,
                        stats.context_parallel_rank,
                        stats.expert_parallel_rank,
                    ),
                }
                for rank, stats in self.global_stats.items()
            },
            "memory_history_length": len(self.memory_history),
        }

    def reset_distributed_stats(self) -> None:
        """Reset all distributed memory statistics."""
        if self._lock:
            with self._lock:
                self.local_stats.clear()
                self.global_stats.clear()
                self.memory_history.clear()
        else:
            self.local_stats.clear()
            self.global_stats.clear()
            self.memory_history.clear()

        self.local_profiler.reset()
        self.last_sync_time = time.time()


class DistributedCheckpointFunction(Function):
    """Custom autograd function for distributed checkpointing with cross-rank
    coordination.
    """

    @staticmethod
    def forward(
        ctx: Any,
        run_function: Callable[..., Any],
        preserve_rng_state: bool,
        layer_id: str,
        profiler: Optional[DistributedMemoryProfiler],
        coordinator: Optional["DistributedCheckpointCoordinator"],
        *args: Any,
    ) -> Any:
        """Forward pass with distributed profiling and coordination.

        Args:
            ctx: Context for saving tensors
            run_function: Function to run and potentially checkpoint
            preserve_rng_state: Whether to preserve RNG state
            layer_id: Identifier for the layer
            profiler: Optional distributed profiler
            coordinator: Optional distributed coordinator
            *args: Arguments to the function

        Returns:
            Output of the function
        """
        # Validate inputs
        if run_function is None:
            raise ValueError("run_function cannot be None")
        if not layer_id:
            raise ValueError("layer_id cannot be empty")

        # Store metadata in context
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        ctx.layer_id = layer_id
        ctx.profiler = profiler
        ctx.coordinator = coordinator

        # Save RNG state if needed (distributed-aware)
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            if torch.cuda.is_available():
                ctx.fwd_gpu_devices = list(range(torch.cuda.device_count()))
                ctx.fwd_gpu_states = [
                    torch.cuda.get_rng_state(device) for device in ctx.fwd_gpu_devices
                ]

            # Save parallel RNG states if available
            if parallel_state.is_initialized():
                try:
                    ctx.parallel_rng_checkpoint = (
                        parallel_state.checkpoint_parallel_rng()
                    )
                except Exception as e:
                    logger.warning(f"Failed to checkpoint parallel RNG state: {e}")
                    ctx.parallel_rng_checkpoint = None
            else:
                ctx.parallel_rng_checkpoint = None

        # Profile distributed memory before execution
        start_time = time.time()
        if profiler is not None:
            profiler.profile_memory_distributed(layer_id, "before")

        # Coordinate checkpoint decision across ranks if needed
        if coordinator is not None:
            coordinator.coordinate_checkpoint_decision(layer_id)

        # Run forward function
        try:
            with torch.no_grad():
                outputs = run_function(*args)
        except Exception as e:
            logger.error(f"Forward execution failed for layer {layer_id}: {e}")
            raise RuntimeError(f"Forward execution failed for layer {layer_id}") from e

        # Profile distributed memory after execution
        if profiler is not None:
            profiler.profile_memory_distributed(layer_id, "after")

            # Update checkpoint statistics
            if layer_id in profiler.local_stats:
                profiler.local_stats[layer_id].active_checkpoints += 1

        # Save tensors for backward
        ctx.save_for_backward(*args)

        # Record execution time
        ctx.forward_time = time.time() - start_time

        return outputs

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Tuple[Optional[torch.Tensor], ...]:
        """Backward pass with distributed recomputation and profiling.

        Args:
            ctx: Context with saved tensors and metadata
            *grad_outputs: Gradients from the next layer

        Returns:
            Tuple of gradients for each input
        """
        # Restore RNG state if needed (distributed-aware)
        if ctx.preserve_rng_state:
            rng_devices: List[int] = []
            if torch.cuda.is_available():
                rng_devices = ctx.fwd_gpu_devices

            with torch.random.fork_rng(devices=rng_devices):
                if ctx.preserve_rng_state:
                    torch.set_rng_state(ctx.fwd_cpu_state)
                    if torch.cuda.is_available():
                        for device, state in zip(
                            ctx.fwd_gpu_devices, ctx.fwd_gpu_states
                        ):
                            torch.cuda.set_rng_state(state, device)

                    # Restore parallel RNG states if available
                    if ctx.parallel_rng_checkpoint is not None:
                        try:
                            parallel_state.restore_parallel_rng(
                                ctx.parallel_rng_checkpoint
                            )
                        except Exception as e:
                            logger.warning(f"Failed to restore parallel RNG state: {e}")

                # Profile recomputation time
                start_time = time.time()

                # Recompute forward pass
                try:
                    with torch.enable_grad():
                        inputs = ctx.saved_tensors
                        outputs = ctx.run_function(*inputs)
                except Exception as e:
                    logger.error(f"Recomputation failed for layer {ctx.layer_id}: {e}")
                    raise RuntimeError(
                        f"Recomputation failed for layer {ctx.layer_id}"
                    ) from e

                # Record recomputation statistics
                recompute_time = time.time() - start_time
                if (
                    ctx.profiler is not None
                    and ctx.layer_id in ctx.profiler.local_stats
                ):
                    stats = ctx.profiler.local_stats[ctx.layer_id]
                    if ctx.forward_time > 0:
                        stats.recomputation_overhead_ratio = (
                            recompute_time / ctx.forward_time
                        )

        # Ensure outputs is a tuple
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        # Compute gradients for inputs that require them
        gradients: List[Optional[torch.Tensor]] = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor) and inp.requires_grad:
                grad_list = []
                for out, grad_out in zip(outputs, grad_outputs):
                    if grad_out is not None:
                        grad = torch.autograd.grad(
                            outputs=out,
                            inputs=inp,
                            grad_outputs=grad_out,
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                        if grad is not None:
                            grad_list.append(grad)

                if grad_list:
                    total_grad = grad_list[0]
                    for g in grad_list[1:]:
                        total_grad = total_grad + g
                    gradients.append(total_grad)
                else:
                    gradients.append(None)
            else:
                gradients.append(None)

        # Return gradients (None for metadata arguments)
        return (None, None, None, None, None) + tuple(gradients)


class DistributedCheckpointCoordinator:
    """Coordinator for distributed checkpointing decisions across ranks."""

    def __init__(self, config: DistributedCheckpointConfig) -> None:
        """Initialize distributed checkpoint coordinator.

        Args:
            config: Distributed checkpointing configuration
        """
        self.config = config

        # Distributed state
        self.is_distributed = parallel_state.is_initialized()
        if self.is_distributed:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()

            # Get parallel groups and sizes
            self.tp_group = parallel_state.get_tensor_model_parallel_group()
            self.pp_group = parallel_state.get_pipeline_model_parallel_group()
            self.dp_group = parallel_state.get_data_parallel_group()
            self.cp_group = parallel_state.get_context_parallel_group()
            self.ep_group = parallel_state.get_expert_model_parallel_group()

            self.tp_size = parallel_state.get_tensor_model_parallel_size()
            self.pp_size = parallel_state.get_pipeline_model_parallel_size()
            self.dp_size = parallel_state.get_data_parallel_size()
            self.cp_size = parallel_state.get_context_parallel_size()
            self.ep_size = parallel_state.get_expert_model_parallel_size()

            self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
            self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
            self.dp_rank = parallel_state.get_data_parallel_rank()
            self.cp_rank = parallel_state.get_context_parallel_rank()
            self.ep_rank = parallel_state.get_expert_model_parallel_rank()
        else:
            # Single rank setup
            self.world_size = 1
            self.rank = 0
            self.tp_group = (
                self.pp_group
            ) = self.dp_group = self.cp_group = self.ep_group = None
            self.tp_size = self.pp_size = self.dp_size = self.cp_size = self.ep_size = 1
            self.tp_rank = self.pp_rank = self.dp_rank = self.cp_rank = self.ep_rank = 0

        # Coordination state with resource management
        self.checkpoint_decisions: Dict[str, bool] = {}
        self.load_balance_state: Dict[int, float] = {}  # rank -> memory usage
        self.last_rebalance_time = time.time()

        # Resource management
        self._max_cache_size = config.coordination_cache_max_size
        self._cleanup_enabled = config.enable_resource_cleanup

        # Error recovery
        self.error_recovery_state = ErrorRecoveryState(
            max_recovery_attempts=config.max_recovery_attempts,
            error_backoff_seconds=config.error_backoff_seconds,
        )

        # Thread safety
        self._lock = threading.RLock() if config.base_config.thread_safe else None

        if config.verbose_distributed:
            logger.info(
                f"Initialized DistributedCheckpointCoordinator for rank {self.rank} "
                f"with strategy: {config.strategy.value}"
            )

    def coordinate_checkpoint_decision(self, layer_id: str) -> bool:
        """Coordinate checkpointing decision across relevant ranks with error recovery.

        Args:
            layer_id: Identifier for the layer

        Returns:
            True if the layer should be checkpointed on this rank
        """
        if not self.is_distributed:
            return True  # Default to checkpointing in single-rank setup

        # Clean up coordination cache if needed
        self._cleanup_coordination_cache_if_needed()

        def cleanup_on_error() -> None:
            """Cleanup coordination state on error."""
            if self._cleanup_enabled and layer_id in self.checkpoint_decisions:
                if self._lock:
                    with self._lock:
                        self.checkpoint_decisions.pop(layer_id, None)
                else:
                    self.checkpoint_decisions.pop(layer_id, None)

        if self.config.enable_error_recovery:
            with distributed_error_recovery(
                f"coordination_decision_{layer_id}",
                self.error_recovery_state,
                cleanup_on_error,
            ):
                return self._make_strategy_decision(layer_id)
        else:
            try:
                return self._make_strategy_decision(layer_id)
            except Exception as e:
                logger.error(f"Coordination decision failed for {layer_id}: {e}")
                cleanup_on_error()
                return True  # Default to checkpointing on error

    def _make_strategy_decision(self, layer_id: str) -> bool:
        """Make the actual strategy-specific decision."""
        if self.config.strategy == DistributedCheckpointStrategy.COORDINATED:
            return self._coordinated_decision(layer_id)
        elif self.config.strategy == DistributedCheckpointStrategy.LOAD_BALANCED:
            return self._load_balanced_decision(layer_id)
        elif self.config.strategy == DistributedCheckpointStrategy.HIERARCHICAL:
            return self._hierarchical_decision(layer_id)
        elif self.config.strategy == DistributedCheckpointStrategy.ADAPTIVE:
            return self._adaptive_decision(layer_id)
        elif self.config.strategy == DistributedCheckpointStrategy.EXPERT_AWARE:
            return self._expert_aware_decision(layer_id)
        elif self.config.strategy == DistributedCheckpointStrategy.PIPELINE_AWARE:
            return self._pipeline_aware_decision(layer_id)
        else:
            logger.warning(f"Unknown distributed strategy: {self.config.strategy}")
            return True

    def _cleanup_coordination_cache_if_needed(self) -> None:
        """Clean up coordination cache if it exceeds maximum size."""
        if not self._cleanup_enabled:
            return

        if len(self.checkpoint_decisions) > self._max_cache_size:
            # Remove half the entries (simple cleanup strategy)
            items_to_remove = len(self.checkpoint_decisions) - (
                self._max_cache_size // 2
            )
            oldest_keys = list(self.checkpoint_decisions.keys())[:items_to_remove]

            if self._lock:
                with self._lock:
                    for key in oldest_keys:
                        self.checkpoint_decisions.pop(key, None)
            else:
                for key in oldest_keys:
                    self.checkpoint_decisions.pop(key, None)

            logger.debug(
                f"Cleaned up {items_to_remove} coordination cache entries "
                f"(new size: {len(self.checkpoint_decisions)})"
            )

    def _coordinated_decision(self, layer_id: str) -> bool:
        """Make coordinated checkpointing decision across all relevant ranks."""
        # Determine which groups need coordination
        groups_to_sync = []

        if (
            self.config.coordinate_across_tp
            and self.tp_group is not None
            and self.tp_size > 1
        ):
            groups_to_sync.append(("tp", self.tp_group))
        if (
            self.config.coordinate_across_pp
            and self.pp_group is not None
            and self.pp_size > 1
        ):
            groups_to_sync.append(("pp", self.pp_group))
        if (
            self.config.coordinate_across_dp
            and self.dp_group is not None
            and self.dp_size > 1
        ):
            groups_to_sync.append(("dp", self.dp_group))
        if (
            self.config.coordinate_across_cp
            and self.cp_group is not None
            and self.cp_size > 1
        ):
            groups_to_sync.append(("cp", self.cp_group))
        if (
            self.config.coordinate_across_ep
            and self.ep_group is not None
            and self.ep_size > 1
        ):
            groups_to_sync.append(("ep", self.ep_group))

        if not groups_to_sync:
            return True  # No coordination needed

        # Use the first coordinating rank in the first group to make the decision
        primary_group_name, primary_group = groups_to_sync[0]

        # Simple hash-based decision for consistency
        decision = hash(layer_id) % 2 == 0

        # Broadcast decision from rank 0 in the primary group with optimizations
        try:
            decision_tensor = torch.tensor(int(decision), dtype=torch.int32)
            if torch.cuda.is_available():
                decision_tensor = decision_tensor.cuda()

            # Use async broadcast if enabled and supported
            if self.config.use_async_communication:
                # For simple broadcast, async doesn't provide much benefit
                # but we keep the option for future enhancements
                dist.broadcast(decision_tensor, src=0, group=primary_group)
            else:
                dist.broadcast(decision_tensor, src=0, group=primary_group)

            final_decision = bool(decision_tensor.item())

            # Cache the decision atomically
            if self._lock:
                with self._lock:
                    self.checkpoint_decisions[layer_id] = final_decision
            else:
                self.checkpoint_decisions[layer_id] = final_decision

            return final_decision

        except Exception as e:
            raise CoordinationError(
                f"Failed to coordinate checkpoint decision for {layer_id}: {e}"
            ) from e

    def _load_balanced_decision(self, layer_id: str) -> bool:
        """Make load-balanced checkpointing decision to distribute memory usage."""
        current_time = time.time()

        # Check if it's time to rebalance
        if (current_time - self.last_rebalance_time) > (
            self.config.rebalance_interval * 0.1
        ):
            self._update_load_balance_state()
            self.last_rebalance_time = current_time

        # Get current rank's memory usage (estimated)
        current_memory = self.load_balance_state.get(self.rank, 0.0)

        # Calculate load balancing probability based on inverse memory usage
        if self.load_balance_state:
            total_memory = sum(self.load_balance_state.values())
            avg_memory = total_memory / len(self.load_balance_state)

            if avg_memory > 0:
                # Higher probability for ranks with lower memory usage
                probability = min(
                    1.0, avg_memory / max(current_memory, avg_memory * 0.1)
                )
            else:
                probability = 0.5  # Default probability
        else:
            probability = 0.5

        # Apply threshold
        should_checkpoint = probability > (1.0 - self.config.load_balance_threshold)

        # Simple hash-based consistency for same layer_id
        layer_hash = hash(layer_id) / float(2**31)  # Normalize to [0, 1)
        should_checkpoint = should_checkpoint or (layer_hash < probability)

        return should_checkpoint

    def _update_load_balance_state(self) -> None:
        """Update load balancing state with current memory usage and optimizations."""
        if not self.is_distributed:
            return

        try:
            # Get current memory usage with error handling
            current_memory = 0.0
            try:
                if torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
            except Exception as e:
                logger.warning(f"Failed to get CUDA memory usage: {e}")
                current_memory = 0.0

            # Create memory tensor with proper device placement
            memory_tensor = torch.tensor(current_memory, dtype=torch.float32)
            if torch.cuda.is_available():
                memory_tensor = memory_tensor.cuda()

            # Use optimized all-gather with pre-allocated buffers
            if not hasattr(self, "_memory_gather_buffers"):
                self._memory_gather_buffers = [
                    torch.zeros_like(memory_tensor) for _ in range(self.world_size)
                ]

            # Reuse pre-allocated buffers for better performance
            gathered_memory = self._memory_gather_buffers
            for buf in gathered_memory:
                buf.zero_()

            dist.all_gather(gathered_memory, memory_tensor)

            # Update load balance state atomically with batched updates
            new_state = {}
            for rank, mem_tensor in enumerate(gathered_memory):
                new_state[rank] = mem_tensor.item()

            if self._lock:
                with self._lock:
                    self.load_balance_state.update(new_state)
            else:
                self.load_balance_state.update(new_state)

            if self.config.verbose_distributed:
                memory_range = (
                    f"{min(new_state.values()):.2f}-{max(new_state.values()):.2f}GB"
                )
                logger.debug(f"Updated load balance state: {memory_range}")

        except Exception as e:
            raise CoordinationError(f"Failed to update load balance state: {e}") from e

    def _hierarchical_decision(self, layer_id: str) -> bool:
        """Make hierarchical checkpointing decision based on parallel dimension."""
        # Different strategies for different parallel dimensions

        # Pipeline parallel: checkpoint different stages on different ranks
        if self.pp_size > 1:
            stage_hash = hash(layer_id) % self.pp_size
            return stage_hash == self.pp_rank

        # Tensor parallel: coordinate checkpointing within TP group
        if self.tp_size > 1:
            return self.tp_rank == 0  # Only first TP rank checkpoints

        # Expert parallel: balance checkpointing across experts
        if self.ep_size > 1:
            expert_hash = hash(layer_id) % self.ep_size
            return expert_hash == self.ep_rank

        # Default: checkpoint on rank 0 of remaining dimensions
        return self.dp_rank == 0 and self.cp_rank == 0

    def _adaptive_decision(self, layer_id: str) -> bool:
        """Make adaptive checkpointing decision based on runtime conditions."""
        # This is a simplified adaptive strategy
        # In practice, this would use more sophisticated ML-based decisions

        # Factor in memory pressure
        memory_pressure = 0.0
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            memory_pressure = allocated / max(reserved, 1)

        # Factor in parallel configuration
        parallel_factor = 1.0 / max(self.tp_size * self.pp_size, 1)

        # Adaptive threshold based on conditions
        threshold = 0.5 * (1 + memory_pressure) * (1 + parallel_factor)

        # Hash-based decision with adaptive threshold
        layer_hash = (hash(layer_id) % 1000) / 1000.0  # Normalize to [0, 1)

        return layer_hash < threshold

    def _expert_aware_decision(self, layer_id: str) -> bool:
        """Make expert-aware checkpointing decision for MoE models."""
        if self.ep_size <= 1:
            return True  # Not an MoE model

        # Check if this is an expert layer
        is_expert_layer = "expert" in layer_id.lower() or "moe" in layer_id.lower()

        if is_expert_layer:
            # Balance checkpointing across experts
            expert_id = hash(layer_id) % self.ep_size
            return expert_id == self.ep_rank
        else:
            # Non-expert layers: checkpoint on all ranks or coordinate
            if self.config.coordinate_across_tp and self.tp_size > 1:
                return self.tp_rank == 0
            return True

    def _pipeline_aware_decision(self, layer_id: str) -> bool:
        """Make pipeline-aware checkpointing decision optimized for pipeline
        parallelism.
        """
        if self.pp_size <= 1:
            return True  # Not using pipeline parallelism

        # Use explicit pipeline checkpoint stages if configured
        if self.config.pipeline_checkpoint_stages:
            return self.pp_rank in self.config.pipeline_checkpoint_stages

        # Optimize for pipeline bubble reduction
        if self.config.pipeline_bubble_optimization:
            # Checkpoint on middle stages to reduce bubble overhead
            middle_stages = set(range(self.pp_size // 4, 3 * self.pp_size // 4))
            return self.pp_rank in middle_stages

        # Default: checkpoint on even pipeline stages
        return self.pp_rank % 2 == 0

    def get_coordination_stats(self) -> Dict[str, Any]:
        """Get coordination statistics and state information.

        Returns:
            Dictionary containing coordination statistics
        """
        stats = {
            "strategy": self.config.strategy.value,
            "is_distributed": self.is_distributed,
            "world_size": self.world_size,
            "rank": self.rank,
            "parallel_config": {
                "tp_size": self.tp_size,
                "pp_size": self.pp_size,
                "dp_size": self.dp_size,
                "cp_size": self.cp_size,
                "ep_size": self.ep_size,
                "tp_rank": self.tp_rank,
                "pp_rank": self.pp_rank,
                "dp_rank": self.dp_rank,
                "cp_rank": self.cp_rank,
                "ep_rank": self.ep_rank,
            },
            "coordination_settings": {
                "coordinate_across_tp": self.config.coordinate_across_tp,
                "coordinate_across_pp": self.config.coordinate_across_pp,
                "coordinate_across_dp": self.config.coordinate_across_dp,
                "coordinate_across_cp": self.config.coordinate_across_cp,
                "coordinate_across_ep": self.config.coordinate_across_ep,
            },
        }

        if self._lock:
            with self._lock:
                stats["checkpoint_decisions_count"] = len(self.checkpoint_decisions)
                balance_dict_locked: Dict[str, Any] = {
                    str(k): v for k, v in self.load_balance_state.items()
                }
                stats["load_balance_state"] = balance_dict_locked
        else:
            stats["checkpoint_decisions_count"] = len(self.checkpoint_decisions)
            balance_dict_unlocked: Dict[str, Any] = {
                str(k): v for k, v in self.load_balance_state.items()
            }
            stats["load_balance_state"] = balance_dict_unlocked

        return stats

    def reset_coordination_state(self) -> None:
        """Reset coordination state and statistics."""
        if self._lock:
            with self._lock:
                self.checkpoint_decisions.clear()
                self.load_balance_state.clear()
        else:
            self.checkpoint_decisions.clear()
            self.load_balance_state.clear()

        self.last_rebalance_time = time.time()


class DistributedActivationCheckpointing:
    """Main class for distributed activation checkpointing with cross-rank
    coordination.
    """

    def __init__(self, config: DistributedCheckpointConfig) -> None:
        """Initialize distributed activation checkpointing.

        Args:
            config: Distributed checkpointing configuration
        """
        config.validate()
        self.config = config

        # Initialize components
        self.profiler = DistributedMemoryProfiler(config)
        self.coordinator = DistributedCheckpointCoordinator(config)

        # Initialize base selective checkpointing
        self.selective_manager = SelectiveRecomputeManager(config.base_config)

        # Initialize traditional checkpointing for compatibility
        self.activation_checkpoint = ActivationCheckpointing(config.base_config)

        # State tracking with optimizations
        self.step_count = 0
        self.last_memory_sync = time.time()

        # Performance monitoring and health checks
        self._performance_metrics = {
            "checkpointed_layers": 0,
            "non_checkpointed_layers": 0,
            "coordination_failures": 0,
            "memory_sync_failures": 0,
            "avg_coordination_time": 0.0,
            "avg_memory_sync_time": 0.0,
        }

        # Health monitoring
        self._health_check_interval = 100  # steps
        self._last_health_check = 0
        self._system_healthy = True

        # Resource optimization
        self._tensor_cache: Dict[
            str, torch.Tensor
        ] = {}  # Cache for frequently used tensors
        self._enable_caching = config.collect_distributed_metrics

        if config.verbose_distributed:
            logger.info(
                f"Initialized DistributedActivationCheckpointing with "
                f"strategy: {config.strategy.value}, "
                f"cache_enabled: {self._enable_caching}, "
                f"error_recovery: {config.enable_error_recovery}"
            )

    def checkpoint_layer_distributed(
        self,
        layer: Union[nn.Module, Callable[..., Any]],
        *args: Any,
        layer_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Checkpoint a layer with distributed coordination.

        Args:
            layer: The layer module or callable to potentially checkpoint
            *args: Positional arguments to the layer
            layer_id: Optional identifier for the layer
            **kwargs: Keyword arguments to the layer

        Returns:
            Output of the layer (checkpointed or not based on distributed strategy)
        """
        if layer is None:
            raise ValueError("Layer cannot be None")

        # Generate layer ID if needed
        if layer_id is None:
            layer_id = f"distributed_layer_{self.step_count}"

        self.step_count += 1

        # Periodic health check and system optimization
        self._perform_health_check_if_needed()

        # Sync memory stats periodically with timing
        sync_start_time = time.time()
        if self.profiler.should_sync_memory_stats():
            try:
                self.profiler.sync_memory_stats()
                sync_time = time.time() - sync_start_time
                self._update_performance_metric("avg_memory_sync_time", sync_time)
            except Exception as e:
                self._performance_metrics["memory_sync_failures"] += 1
                logger.warning(f"Memory sync failed for {layer_id}: {e}")

        # Get distributed checkpoint decision with timing
        coord_start_time = time.time()
        try:
            should_checkpoint = self.coordinator.coordinate_checkpoint_decision(
                layer_id
            )
            coord_time = time.time() - coord_start_time
            self._update_performance_metric("avg_coordination_time", coord_time)
        except Exception as e:
            self._performance_metrics["coordination_failures"] += 1
            logger.warning(f"Coordination failed for {layer_id}: {e}")
            should_checkpoint = True  # Default to checkpointing on coordination failure

        # Execute with performance tracking
        try:
            if should_checkpoint:
                self._performance_metrics["checkpointed_layers"] += 1
                # Use distributed checkpointing function
                return DistributedCheckpointFunction.apply(
                    layer,
                    self.config.base_config.preserve_rng_state,
                    layer_id,
                    self.profiler,
                    self.coordinator,
                    *args,
                )
            else:
                self._performance_metrics["non_checkpointed_layers"] += 1
                # Execute without checkpointing but with profiling
                return self._execute_with_profiling(layer, layer_id, *args, **kwargs)

        except Exception as e:
            logger.error(
                f"Error in distributed checkpointing for layer {layer_id}: {e}"
            )
            raise RuntimeError(
                f"Distributed checkpointing failed for {layer_id}"
            ) from e

    def _perform_health_check_if_needed(self) -> None:
        """Perform system health check and optimization if needed."""
        if self.step_count - self._last_health_check < self._health_check_interval:
            return

        self._last_health_check = self.step_count

        # Check coordination failure rate
        total_decisions = (
            self._performance_metrics["checkpointed_layers"]
            + self._performance_metrics["non_checkpointed_layers"]
        )

        if total_decisions > 0:
            failure_rate = (
                self._performance_metrics["coordination_failures"] / total_decisions
            )
            if failure_rate > 0.1:  # More than 10% failure rate
                self._system_healthy = False
                logger.warning(
                    f"High coordination failure rate: {failure_rate:.2%} "
                    f"({self._performance_metrics['coordination_failures']}/"
                    f"{total_decisions})"
                )
            else:
                self._system_healthy = True

        # Cleanup caches if needed
        if self._enable_caching and len(self._tensor_cache) > 100:
            self._tensor_cache.clear()
            logger.debug("Cleared tensor cache for memory optimization")

        # Log performance metrics
        if self.config.verbose_distributed and total_decisions > 0:
            checkpoint_rate = (
                self._performance_metrics["checkpointed_layers"] / total_decisions
            )
            logger.info(
                f"Checkpointing health: rate={checkpoint_rate:.2%}, "
                f"coord_time="
                f"{self._performance_metrics['avg_coordination_time']:.3f}s, "
                f"sync_time={self._performance_metrics['avg_memory_sync_time']:.3f}s"
            )

    def _update_performance_metric(self, metric_name: str, new_value: float) -> None:
        """Update performance metric with exponential moving average."""
        alpha = 0.1  # Smoothing factor
        current_value = self._performance_metrics.get(metric_name, 0.0)
        self._performance_metrics[metric_name] = (
            1 - alpha
        ) * current_value + alpha * new_value

    def _execute_with_profiling(
        self,
        layer: Callable[..., Any],
        layer_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute layer without checkpointing but with distributed profiling."""
        # Profile before execution
        self.profiler.profile_memory_distributed(layer_id, "before")

        try:
            result = layer(*args, **kwargs)
        except Exception as e:
            logger.error(f"Layer execution failed for {layer_id}: {e}")
            raise

        # Profile after execution
        self.profiler.profile_memory_distributed(layer_id, "after")

        return result

    def apply_to_transformer_layers_distributed(
        self,
        model: nn.Module,
        layer_attr: str = "transformer.h",
        **kwargs: Any,
    ) -> nn.Module:
        """Apply distributed checkpointing to transformer layers.

        Args:
            model: PyTorch model
            layer_attr: Attribute path to transformer layers
            **kwargs: Additional arguments passed to base checkpointing

        Returns:
            Model with distributed checkpointing applied
        """
        # First apply base checkpointing
        model = self.activation_checkpoint.apply_to_transformer_layers(
            model, layer_attr=layer_attr, **kwargs
        )

        # Then wrap with distributed coordination
        return self._wrap_model_with_distributed_checkpointing(model, layer_attr)

    def _wrap_model_with_distributed_checkpointing(
        self,
        model: nn.Module,
        layer_attr: str,
    ) -> nn.Module:
        """Wrap model layers with distributed checkpointing coordination."""
        # Get layers
        current = model
        for attr in layer_attr.split("."):
            if hasattr(current, attr):
                current = getattr(current, attr)
            else:
                logger.warning(f"Model doesn't have attribute {attr}")
                return model

        if not isinstance(current, (nn.ModuleList, list)):
            logger.warning(f"{layer_attr} is not a ModuleList or list")
            return model

        layers = current

        # Wrap each layer's forward method with distributed coordination
        for i, layer in enumerate(layers):
            original_forward = layer.forward
            layer_id = f"{layer_attr}.{i}"

            def create_distributed_forward(orig_forward, lid):
                def distributed_forward(*args, **kwargs):
                    return self.checkpoint_layer_distributed(
                        orig_forward, *args, layer_id=lid, **kwargs
                    )

                return distributed_forward

            layer.forward = create_distributed_forward(original_forward, layer_id)

        return model

    def get_distributed_profiling_report(self) -> Dict[str, Any]:
        """Get comprehensive distributed profiling report.

        Returns:
            Dictionary containing all distributed profiling information
        """
        # Get base reports
        distributed_memory_report = self.profiler.get_distributed_memory_report()
        coordination_stats = self.coordinator.get_coordination_stats()
        selective_report = self.selective_manager.get_profiling_report()
        activation_report = self.activation_checkpoint.get_profiling_report()

        return {
            "distributed_checkpointing": {
                "config": {
                    "strategy": self.config.strategy.value,
                    "coordination_settings": {
                        "coordinate_across_tp": self.config.coordinate_across_tp,
                        "coordinate_across_pp": self.config.coordinate_across_pp,
                        "coordinate_across_dp": self.config.coordinate_across_dp,
                        "coordinate_across_cp": self.config.coordinate_across_cp,
                        "coordinate_across_ep": self.config.coordinate_across_ep,
                    },
                    "load_balancing": {
                        "enabled": self.config.enable_load_balancing,
                        "threshold": self.config.load_balance_threshold,
                        "rebalance_interval": self.config.rebalance_interval,
                    },
                    "error_recovery": {
                        "enabled": self.config.enable_error_recovery,
                        "max_attempts": self.config.max_recovery_attempts,
                        "backoff_seconds": self.config.error_backoff_seconds,
                    },
                    "resource_management": {
                        "cache_max_size": self.config.coordination_cache_max_size,
                        "cleanup_enabled": self.config.enable_resource_cleanup,
                        "stats_history_size": self.config.memory_stats_history_size,
                    },
                },
                "memory_profiling": distributed_memory_report,
                "coordination": coordination_stats,
                "performance_metrics": self._performance_metrics,
                "system_health": {
                    "healthy": self._system_healthy,
                    "last_health_check": self._last_health_check,
                    "cache_size": len(self._tensor_cache),
                },
                "step_count": self.step_count,
                "last_memory_sync": self.last_memory_sync,
            },
            "selective_checkpointing": selective_report,
            "activation_checkpointing": activation_report,
        }

    def reset_distributed_profiling(self) -> None:
        """Reset all distributed profiling statistics and performance metrics."""
        self.profiler.reset_distributed_stats()
        self.coordinator.reset_coordination_state()
        self.selective_manager.reset_profiling()
        self.activation_checkpoint.profiler.reset()

        # Reset performance metrics and state
        self._performance_metrics = {
            "checkpointed_layers": 0,
            "non_checkpointed_layers": 0,
            "coordination_failures": 0,
            "memory_sync_failures": 0,
            "avg_coordination_time": 0.0,
            "avg_memory_sync_time": 0.0,
        }

        # Reset state tracking
        self.step_count = 0
        self.last_memory_sync = time.time()
        self._last_health_check = 0
        self._system_healthy = True

        # Clear caches
        self._tensor_cache.clear()

        if self.config.verbose_distributed:
            logger.info(
                "Reset all distributed profiling statistics and performance metrics"
            )

    def get_system_health_status(self) -> Dict[str, Any]:
        """Get detailed system health status for monitoring.

        Returns:
            Dictionary containing system health information
        """
        total_decisions = (
            self._performance_metrics["checkpointed_layers"]
            + self._performance_metrics["non_checkpointed_layers"]
        )

        coordination_failure_rate = 0.0
        memory_sync_failure_rate = 0.0
        checkpoint_rate = 0.0

        if total_decisions > 0:
            coordination_failure_rate = (
                self._performance_metrics["coordination_failures"] / total_decisions
            )
            memory_sync_failure_rate = (
                self._performance_metrics["memory_sync_failures"] / total_decisions
            )
            checkpoint_rate = (
                self._performance_metrics["checkpointed_layers"] / total_decisions
            )

        return {
            "overall_healthy": self._system_healthy,
            "total_layer_decisions": total_decisions,
            "checkpoint_rate": checkpoint_rate,
            "failure_rates": {
                "coordination": coordination_failure_rate,
                "memory_sync": memory_sync_failure_rate,
            },
            "performance": {
                "avg_coordination_time_ms": self._performance_metrics[
                    "avg_coordination_time"
                ]
                * 1000,
                "avg_memory_sync_time_ms": self._performance_metrics[
                    "avg_memory_sync_time"
                ]
                * 1000,
            },
            "resource_usage": {
                "tensor_cache_size": len(self._tensor_cache),
                "cache_enabled": self._enable_caching,
            },
            "uptime_steps": self.step_count,
            "last_health_check_steps_ago": self.step_count - self._last_health_check,
        }


# Factory function for easy creation
def create_distributed_checkpointing(
    strategy: DistributedCheckpointStrategy = DistributedCheckpointStrategy.COORDINATED,
    coordinate_tp: bool = True,
    coordinate_cp: bool = True,
    coordinate_ep: bool = True,
    enable_load_balancing: bool = True,
    base_selective_config: Optional[SelectiveCheckpointConfig] = None,
    **kwargs: Any,
) -> DistributedActivationCheckpointing:
    """Create distributed activation checkpointing with common settings.

    Args:
        strategy: Distributed checkpointing strategy
        coordinate_tp: Whether to coordinate across tensor parallel ranks
        coordinate_cp: Whether to coordinate across context parallel ranks
        coordinate_ep: Whether to coordinate across expert parallel ranks
        enable_load_balancing: Whether to enable load balancing
        base_selective_config: Base selective checkpointing configuration
        **kwargs: Additional configuration parameters

    Returns:
        Configured DistributedActivationCheckpointing instance
    """
    if base_selective_config is None:
        base_selective_config = SelectiveCheckpointConfig()

    config = DistributedCheckpointConfig(
        strategy=strategy,
        coordinate_across_tp=coordinate_tp,
        coordinate_across_cp=coordinate_cp,
        coordinate_across_ep=coordinate_ep,
        enable_load_balancing=enable_load_balancing,
        base_config=base_selective_config,
        **kwargs,
    )

    return DistributedActivationCheckpointing(config)
