"""Main gradient finalizer class for multi-dimensional gradient synchronization."""

import logging
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import torch
import torch.distributed as dist
import torch.nn as nn

# Optional profiling support
try:
    from torch.profiler import ProfilerActivity, profile, record_function

    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False
    profile = None  # type: ignore
    ProfilerActivity = None  # type: ignore
    record_function = None  # type: ignore

from ..optimizer.distributed_optimizer import DistributedOptimizer
from ..parallelism import parallel_state
from ..utils.gradient_utils import (
    GradientClipConfig,
    apply_gradient_clipping,
    check_gradient_finite,
    get_gradient_stats,
)
from ..utils.multi_tensor_ops import MultiTensorOperator, multi_tensor_clip_grad_norm
from .config import GradientFinalizationConfig
from .strategies import (
    BucketedGradientSync,
    GradientSyncStrategy,
    HierarchicalGradientSync,
    SimpleGradientSync,
)

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Track performance metrics for gradient finalization."""

    total_sync_time: float = 0.0
    total_clip_time: float = 0.0
    total_stats_time: float = 0.0
    num_syncs: int = 0
    num_clips: int = 0
    bytes_communicated: int = 0

    def update_sync(self, time_taken: float, bytes_sent: int = 0) -> None:
        """Update synchronization metrics."""
        self.total_sync_time += time_taken
        self.num_syncs += 1
        self.bytes_communicated += bytes_sent

    def update_clip(self, time_taken: float) -> None:
        """Update clipping metrics."""
        self.total_clip_time += time_taken
        self.num_clips += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            "avg_sync_time": self.total_sync_time / max(1, self.num_syncs),
            "avg_clip_time": self.total_clip_time / max(1, self.num_clips),
            "total_bytes_communicated": self.bytes_communicated,
            "num_syncs": self.num_syncs,
            "num_clips": self.num_clips,
        }


class GradientBufferPool:
    """Efficient pool for gradient buffer management."""

    def __init__(self, max_buffers: int = 10):
        """Initialize buffer pool.

        Args:
            max_buffers: Maximum number of buffers to keep in pool.
        """
        self.max_buffers = max_buffers
        self.free_buffers: Dict[
            Tuple[int, torch.dtype, torch.device], List[torch.Tensor]
        ] = {}
        self.allocated_buffers: List[torch.Tensor] = []
        self._lock = threading.Lock()

    def acquire(
        self, size: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Acquire a buffer from the pool or create a new one.

        Args:
            size: Size of the buffer needed.
            dtype: Data type of the buffer.
            device: Device for the buffer.

        Returns:
            Tensor buffer.
        """
        key = (size, dtype, device)

        with self._lock:
            if key in self.free_buffers and self.free_buffers[key]:
                buffer = self.free_buffers[key].pop()
                buffer.zero_()  # Clear the buffer
            else:
                buffer = torch.zeros(size, dtype=dtype, device=device)

            self.allocated_buffers.append(buffer)
            return buffer

    def release(self, buffer: torch.Tensor) -> None:
        """Release a buffer back to the pool.

        Args:
            buffer: Buffer to release.
        """
        key = (buffer.numel(), buffer.dtype, buffer.device)

        with self._lock:
            if buffer in self.allocated_buffers:
                self.allocated_buffers.remove(buffer)

            if key not in self.free_buffers:
                self.free_buffers[key] = []

            if len(self.free_buffers[key]) < self.max_buffers:
                self.free_buffers[key].append(buffer)

    def clear(self) -> None:
        """Clear all buffers from the pool."""
        with self._lock:
            self.free_buffers.clear()
            self.allocated_buffers.clear()


# Check for DTensor support (PyTorch 2.0+)
try:
    from torch.distributed.device_mesh import DeviceMesh as _DeviceMesh
    from torch.distributed.tensor._api import DTensor as _DTensor

    DTensor = _DTensor
    DeviceMesh = _DeviceMesh
    DTENSOR_AVAILABLE = True
except ImportError:
    # Proper type stubs for unavailable modules
    DTensor = type("DTensor", (), {})  # type: ignore
    DeviceMesh = type("DeviceMesh", (), {})  # type: ignore
    DTENSOR_AVAILABLE = False


class TelemetryHooks:
    """Hooks for telemetry and monitoring."""

    def __init__(self) -> None:
        """Initialize telemetry hooks."""
        self.pre_sync_hooks: List[Callable[[Dict[str, Any]], None]] = []
        self.post_sync_hooks: List[Callable[[Dict[str, Any]], None]] = []
        self.pre_clip_hooks: List[Callable[[Dict[str, Any]], None]] = []
        self.post_clip_hooks: List[Callable[[Dict[str, Any]], None]] = []

    def register_pre_sync(self, hook: Callable[[Dict[str, Any]], None]) -> None:
        """Register a pre-synchronization hook."""
        self.pre_sync_hooks.append(hook)

    def register_post_sync(self, hook: Callable[[Dict[str, Any]], None]) -> None:
        """Register a post-synchronization hook."""
        self.post_sync_hooks.append(hook)

    def register_pre_clip(self, hook: Callable[[Dict[str, Any]], None]) -> None:
        """Register a pre-clipping hook."""
        self.pre_clip_hooks.append(hook)

    def register_post_clip(self, hook: Callable[[Dict[str, Any]], None]) -> None:
        """Register a post-clipping hook."""
        self.post_clip_hooks.append(hook)

    def fire_pre_sync(self, stats: Dict[str, Any]) -> None:
        """Fire pre-synchronization hooks."""
        for hook in self.pre_sync_hooks:
            try:
                hook(stats)
            except Exception as e:
                logger.warning(f"Pre-sync hook failed: {e}")

    def fire_post_sync(self, stats: Dict[str, Any]) -> None:
        """Fire post-synchronization hooks."""
        for hook in self.post_sync_hooks:
            try:
                hook(stats)
            except Exception as e:
                logger.warning(f"Post-sync hook failed: {e}")

    def fire_pre_clip(self, stats: Dict[str, Any]) -> None:
        """Fire pre-clipping hooks."""
        for hook in self.pre_clip_hooks:
            try:
                hook(stats)
            except Exception as e:
                logger.warning(f"Pre-clip hook failed: {e}")

    def fire_post_clip(self, stats: Dict[str, Any]) -> None:
        """Fire post-clipping hooks."""
        for hook in self.post_clip_hooks:
            try:
                hook(stats)
            except Exception as e:
                logger.warning(f"Post-clip hook failed: {e}")


class GradientFinalizer:
    """Gradient finalizer for multi-dimensional parallelism.

    This class handles gradient finalization and synchronization across
    multiple parallelism dimensions (TP, PP, DP, CP, EP) with support for:
    - Various synchronization strategies (simple, bucketed, hierarchical)
    - Virtual pipeline parallel ranks
    - DTensor integration for PyTorch 2.0+
    - Integration with distributed optimizer
    - Gradient clipping and normalization
    - Comprehensive statistics collection
    """

    def __init__(
        self,
        model: nn.Module,
        config: GradientFinalizationConfig,
        distributed_optimizer: Optional[DistributedOptimizer] = None,
    ):
        """Initialize gradient finalizer.

        Args:
            model: Model to finalize gradients for.
            config: Configuration for gradient finalization.
            distributed_optimizer: Optional distributed optimizer for integration.
        """
        self.model = model
        self.config = config
        self.distributed_optimizer = distributed_optimizer

        # Thread safety with fine-grained locking
        self._lock = threading.RLock()
        self._stats_lock = threading.Lock()
        self._buffer_lock = threading.Lock()

        # Initialize process groups
        self._init_process_groups()

        # Create synchronization strategy
        self.sync_strategy = self._create_sync_strategy()

        # Statistics tracking with circular buffer for memory efficiency
        self._max_stats_history = (
            config.max_stats_history if hasattr(config, "max_stats_history") else 100
        )
        self.stats_history: Deque[Dict[str, Any]] = deque(
            maxlen=self._max_stats_history
        )
        self.finalization_count = 0

        # Performance metrics
        self._perf_metrics = PerformanceMetrics()

        # Telemetry hooks
        self.telemetry = TelemetryHooks()

        # Profiling support
        self.profiler_enabled = PROFILER_AVAILABLE and config.enable_profiling
        self.profiler: Optional[Any] = None
        if self.profiler_enabled:
            self._init_profiler()

        # DTensor support
        self.dtensor_enabled = DTENSOR_AVAILABLE and config.dtensor_enabled
        if self.dtensor_enabled:
            self._init_dtensor_support()

        # Virtual pipeline parallel support
        self.virtual_pp_rank: Optional[int] = None
        self.virtual_pp_size: Optional[int] = None
        if config.virtual_pipeline_aware:
            self._init_virtual_pipeline_support()

        # Gradient buffer pool for efficient memory reuse
        self.gradient_buffer_pool = GradientBufferPool()
        self.gradient_buffers: Dict[str, torch.Tensor] = {}
        self._init_gradient_buffers()

        # Initialize multi-tensor operator for optimized operations
        try:
            first_param = next(self.model.parameters())
            device = first_param.device
        except StopIteration:
            device = None

        self.multi_tensor_operator = MultiTensorOperator(
            device=device,
            enable_benchmarking=(
                config.enable_timing_stats
                if hasattr(config, "enable_timing_stats")
                else False
            ),
        )

        if config.verbose:
            logger.info(
                f"Initialized GradientFinalizer with strategy: "
                f"{config.sync_strategy}, DTensor: {self.dtensor_enabled}, "
                f"Virtual PP: {self.virtual_pp_rank is not None}"
            )

    def _init_process_groups(self) -> None:
        """Initialize process groups for each parallelism dimension."""
        self.process_groups: Dict[str, Optional[dist.ProcessGroup]] = {}

        # Get process groups from parallel state
        if parallel_state.is_initialized():
            self.process_groups["tp"] = parallel_state.get_tensor_model_parallel_group()
            self.process_groups["pp"] = (
                parallel_state.get_pipeline_model_parallel_group()
            )
            self.process_groups["dp"] = parallel_state.get_data_parallel_group()
            self.process_groups["cp"] = parallel_state.get_context_parallel_group()
            self.process_groups["ep"] = parallel_state.get_expert_model_parallel_group()

            # Combined groups for optimized communication
            self.process_groups["tp_dp"] = (
                parallel_state.get_tensor_and_data_parallel_group()
            )
            self.process_groups["tp_dp_cp"] = (
                parallel_state.get_tensor_and_data_parallel_group_with_cp()
            )
            self.process_groups["model"] = parallel_state.get_model_parallel_group()
        else:
            # Fallback to default world group
            if dist.is_initialized():
                self.process_groups["dp"] = dist.group.WORLD
            logger.warning(
                "Parallel state not initialized, using default process groups"
            )

        # Store world sizes and ranks for each dimension
        self.world_sizes: Dict[str, int] = {}
        self.ranks: Dict[str, int] = {}

        for dim, group in self.process_groups.items():
            if group is not None:
                self.world_sizes[dim] = dist.get_world_size(group)
                self.ranks[dim] = dist.get_rank(group)
            else:
                self.world_sizes[dim] = 1
                self.ranks[dim] = 0

    def _create_sync_strategy(self) -> GradientSyncStrategy:
        """Create gradient synchronization strategy based on configuration.

        Returns:
            Gradient sync strategy instance.
        """
        strategy_map = {
            "simple": SimpleGradientSync,
            "bucketed": BucketedGradientSync,
            "hierarchical": HierarchicalGradientSync,
        }

        strategy_class = strategy_map.get(self.config.sync_strategy)
        if strategy_class is None:
            logger.warning(
                f"Unknown sync strategy: {self.config.sync_strategy}, using simple"
            )
            strategy_class = SimpleGradientSync

        return strategy_class(self.config)  # type: ignore

    def _init_dtensor_support(self) -> None:
        """Initialize DTensor support for PyTorch 2.0+."""
        if not DTENSOR_AVAILABLE:
            self.dtensor_enabled = False
            return

        try:
            # Create device mesh for multi-dimensional parallelism
            # This is a simplified example - actual implementation would be more complex
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if world_size > 1:
                # Create device mesh based on parallelism dimensions
                tp_size = self.world_sizes.get("tp", 1)
                dp_size = self.world_sizes.get("dp", 1)

                if tp_size > 1 and dp_size > 1 and DeviceMesh is not None:
                    # 2D mesh for TP and DP
                    self.device_mesh = DeviceMesh(
                        "cuda",
                        torch.arange(world_size).reshape(dp_size, tp_size),
                    )
                elif DeviceMesh is not None:
                    # 1D mesh
                    self.device_mesh = DeviceMesh("cuda", torch.arange(world_size))
                else:
                    self.dtensor_enabled = False

                if self.config.verbose:
                    logger.info(f"Initialized DTensor device mesh: {self.device_mesh}")
            else:
                self.dtensor_enabled = False

        except Exception as e:
            logger.warning(f"Failed to initialize DTensor support: {e}")
            self.dtensor_enabled = False

    def _init_virtual_pipeline_support(self) -> None:
        """Initialize support for virtual pipeline parallel ranks."""
        if parallel_state.is_initialized():
            # Check if virtual pipeline functions exist
            if hasattr(parallel_state, "get_virtual_pipeline_model_parallel_rank"):
                self.virtual_pp_rank = (
                    parallel_state.get_virtual_pipeline_model_parallel_rank()
                )
            else:
                self.virtual_pp_rank = None

            if hasattr(
                parallel_state, "get_virtual_pipeline_model_parallel_world_size"
            ):
                self.virtual_pp_size = getattr(
                    parallel_state,
                    "get_virtual_pipeline_model_parallel_world_size",
                    lambda: None,
                )()
            else:
                self.virtual_pp_size = None

            if self.virtual_pp_rank is not None and self.config.verbose:
                logger.info(
                    f"Virtual pipeline parallel: rank {self.virtual_pp_rank} "
                    f"of {self.virtual_pp_size}"
                )

    def _init_profiler(self) -> None:
        """Initialize PyTorch profiler for performance analysis."""
        if not PROFILER_AVAILABLE:
            return

        try:
            self.profiler = profile(  # type: ignore
                activities=[  # type: ignore
                    ProfilerActivity.CPU,  # type: ignore
                    ProfilerActivity.CUDA,  # type: ignore
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,  # Avoid overhead
            )
            logger.info("Profiler initialized for gradient finalization")
        except Exception as e:
            logger.warning(f"Failed to initialize profiler: {e}")
            self.profiler_enabled = False

    def _init_gradient_buffers(self) -> None:
        """Initialize gradient buffers for efficient communication."""
        if not self.config.use_contiguous_buffers:
            return

        with self._buffer_lock:
            # Calculate total gradient size
            total_grad_size = 0
            dtype_groups: Dict[torch.dtype, int] = {}

            for param in self.model.parameters():
                if param.requires_grad:
                    param_size = param.numel()
                    total_grad_size += param_size

                    # Track sizes by dtype for mixed precision
                    if param.dtype not in dtype_groups:
                        dtype_groups[param.dtype] = 0
                    dtype_groups[param.dtype] += param_size

            if total_grad_size == 0:
                return

            # Get device from model
            device = next(self.model.parameters()).device

            # Allocate buffers from pool for each dtype group
            for dtype, size in dtype_groups.items():
                buffer_key = f"main_{dtype}"
                self.gradient_buffers[buffer_key] = self.gradient_buffer_pool.acquire(
                    size, dtype, device
                )

                # Create FP16 compression buffer if needed
                if self.config.fp16_compression and dtype == torch.float32:
                    fp16_key = f"fp16_{dtype}"
                    self.gradient_buffers[fp16_key] = self.gradient_buffer_pool.acquire(
                        size, torch.float16, device
                    )

            if self.config.verbose:
                total_buffer_size_mb = sum(
                    buffer.numel() * buffer.element_size()
                    for buffer in self.gradient_buffers.values()
                ) / (1024 * 1024)
                logger.info(
                    f"Allocated gradient buffers: {total_buffer_size_mb:.2f} MB "
                    f"across {len(dtype_groups)} dtype groups"
                )

    def finalize_gradients(
        self,
        clip_gradients: bool = True,
        check_finite: bool = True,
        collect_stats: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Finalize gradients with synchronization and optional operations.

        This is the main entry point for gradient finalization. It performs:
        1. Gradient validity checking (NaN/Inf detection)
        2. Gradient synchronization across process groups
        3. Gradient clipping (if enabled)
        4. Statistics collection (if enabled)
        5. Integration with distributed optimizer (if available)

        Args:
            clip_gradients: Whether to apply gradient clipping.
            check_finite: Whether to check for finite gradients.
            collect_stats: Whether to collect gradient statistics.
                          If None, uses config.enable_gradient_stats.

        Returns:
            Dictionary with finalization statistics.

        Raises:
            RuntimeError: If gradients are non-finite and check_finite is True.
        """
        # Use profiler context if enabled
        profiler_context = (
            self.profiler if self.profiler_enabled and self.profiler else None
        )

        with self._lock:
            stats: Dict[str, Any] = {
                "finalization_time": 0.0,
                "sync_stats": {},
                "gradient_norm": 0.0,
                "clipped": False,
                "finite": True,
                "step": self.finalization_count,
            }

            start_time = time.perf_counter()

            # Start profiling if enabled
            if profiler_context:
                profiler_context.__enter__()

            # Check for finite gradients if requested
            if check_finite:
                finite, finite_stats = self._check_finite_gradients()
                stats["finite"] = finite
                stats["finite_stats"] = finite_stats

                if not finite:
                    logger.warning(
                        f"Non-finite gradients detected: "
                        f"{finite_stats['nan_parameters']} NaN, "
                        f"{finite_stats['inf_parameters']} Inf"
                    )
                    if self.config.verbose:
                        stats["finalization_time"] = time.perf_counter() - start_time
                        return stats

            # Synchronize gradients before clipping if configured
            if self.config.sync_grad_before_clip:
                # Fire pre-sync telemetry hooks
                self.telemetry.fire_pre_sync(stats)

                sync_stats = self._synchronize_gradients()
                stats["sync_stats"] = sync_stats
                stats["gradient_norm"] = sync_stats.get("total_gradient_norm", 0.0)

                # Fire post-sync telemetry hooks
                self.telemetry.fire_post_sync(stats)

            # Apply gradient clipping if requested
            if clip_gradients and self.distributed_optimizer is not None:
                # Fire pre-clip telemetry hooks
                self.telemetry.fire_pre_clip(stats)

                clip_stats = self._clip_gradients()
                stats["clipped"] = clip_stats.get("clipped", False)
                stats["clip_stats"] = clip_stats

                # Update gradient norm if clipped
                if stats["clipped"]:
                    stats["gradient_norm"] = clip_stats.get(
                        "grad_norm", stats["gradient_norm"]
                    )

                # Fire post-clip telemetry hooks
                self.telemetry.fire_post_clip(stats)

            # Synchronize gradients after clipping if configured
            if not self.config.sync_grad_before_clip:
                # Fire pre-sync telemetry hooks
                self.telemetry.fire_pre_sync(stats)

                sync_stats = self._synchronize_gradients()
                stats["sync_stats"] = sync_stats
                stats["gradient_norm"] = sync_stats.get("total_gradient_norm", 0.0)

                # Fire post-sync telemetry hooks
                self.telemetry.fire_post_sync(stats)

            # Collect gradient statistics if requested
            if collect_stats or (
                collect_stats is None and self.config.enable_gradient_stats
            ):
                grad_stats = self._collect_gradient_stats()
                stats["gradient_stats"] = grad_stats

            # Handle virtual pipeline parallel ranks
            if self.virtual_pp_rank is not None:
                self._handle_virtual_pipeline_gradients()

            # Update finalization count
            self.finalization_count += 1

            # Store stats in history (circular buffer handles size limit automatically)
            if self.config.enable_gradient_stats:
                with self._stats_lock:
                    self.stats_history.append(stats)

            stats["finalization_time"] = time.perf_counter() - start_time

            # Stop profiling if enabled
            if profiler_context:
                profiler_context.__exit__(None, None, None)

                # Export profiler results periodically
                if self.finalization_count % 1000 == 0:
                    try:
                        profiler_context.export_chrome_trace(
                            f"gradient_finalizer_trace_{self.finalization_count}.json"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to export profiler trace: {e}")

            if self.config.verbose and self.finalization_count % 100 == 0:
                logger.info(
                    f"Gradient finalization step {self.finalization_count}: "
                    f"norm={stats['gradient_norm']:.4f}, "
                    f"time={stats['finalization_time']:.3f}s"
                )

            return stats

    def _check_finite_gradients(self) -> Tuple[bool, Dict[str, int]]:
        """Check if all gradients are finite.

        Returns:
            Tuple of (all_finite, statistics).
        """
        params = list(self.model.parameters())
        return check_gradient_finite(params, raise_on_nonfinite=False)

    def _synchronize_gradients(self) -> Dict[str, Any]:
        """Synchronize gradients across process groups.

        Returns:
            Dictionary with synchronization statistics.

        Raises:
            RuntimeError: If synchronization fails.
        """
        sync_start = time.perf_counter()
        sync_stats: Dict[str, Any] = {}

        try:
            # Handle batch norm synchronization if configured
            if self.config.sync_batch_norm:
                with self._measure_time("batch_norm_sync"):
                    self._sync_batch_norm_stats()

            # Handle layer norm synchronization if configured
            if self.config.sync_layer_norm:
                with self._measure_time("layer_norm_sync"):
                    self._sync_layer_norm_stats()

            # Use the configured sync strategy
            sync_stats = self.sync_strategy.sync_gradients(
                self.model, self.process_groups
            )

            # Track performance metrics
            sync_time = time.perf_counter() - sync_start
            bytes_sent = self._estimate_gradient_bytes()
            self._perf_metrics.update_sync(sync_time, bytes_sent)

            sync_stats["sync_time_total"] = sync_time

        except Exception as e:
            logger.error(f"Gradient synchronization failed: {e}")
            # Attempt recovery
            if self.config.enable_error_recovery:
                sync_stats = self._recover_from_sync_error(e)
            else:
                raise RuntimeError(f"Gradient synchronization failed: {e}") from e

        return sync_stats

    def _clip_gradients(self) -> Dict[str, float]:
        """Apply gradient clipping based on configuration.

        Returns:
            Dictionary with clipping statistics.

        Raises:
            ValueError: If clipping configuration is invalid.
        """
        clip_start = time.perf_counter()

        if self.distributed_optimizer is None:
            return {"clipped": False, "grad_norm": 0.0, "clip_time": 0.0}

        try:
            # Get clip configuration from distributed optimizer
            if hasattr(self.distributed_optimizer, "grad_clip_config"):
                clip_config = self.distributed_optimizer.grad_clip_config
            else:
                # Create default config
                clip_config = GradientClipConfig(
                    clip_type="norm",
                    max_norm=1.0,
                    norm_type=self.config.gradient_norm_type,
                )

            # Apply gradient clipping
            if clip_config is not None:
                # Get parameters with gradients for efficiency
                params_with_grad = [
                    p for p in self.model.parameters() if p.grad is not None
                ]

                if params_with_grad:
                    # Use multi-tensor operations for optimized clipping
                    use_multi_tensor = getattr(
                        self.config, "use_multi_tensor_ops", True
                    )
                    if use_multi_tensor:
                        # Use optimized multi-tensor clipping
                        clip_stats = multi_tensor_clip_grad_norm(
                            params_with_grad,
                            clip_config.max_norm,
                            clip_config.norm_type,
                            self.multi_tensor_operator,
                        )
                        # Convert to expected format
                        clip_stats = {
                            "clipped": clip_stats["was_clipped"],
                            "grad_norm": clip_stats["total_norm"],
                            "clip_coeff": clip_stats["clip_coeff"],
                        }
                    else:
                        # Fall back to standard clipping
                        params_or_tensors: Union[List[torch.Tensor], nn.Module] = cast(
                            List[torch.Tensor], [p for p in params_with_grad]
                        )
                        clip_stats = apply_gradient_clipping(
                            params_or_tensors, clip_config
                        )
                else:
                    clip_stats = {"clipped": False, "grad_norm": 0.0}
            else:
                clip_stats = {"clipped": False, "grad_norm": 0.0}

            # Track performance
            clip_time = time.perf_counter() - clip_start
            clip_stats["clip_time"] = clip_time
            self._perf_metrics.update_clip(clip_time)

        except Exception as e:
            logger.error(f"Gradient clipping failed: {e}")
            raise ValueError(f"Invalid gradient clipping configuration: {e}") from e

        return clip_stats

    def _collect_gradient_stats(self) -> Dict[str, Any]:
        """Collect comprehensive gradient statistics.

        Returns:
            Dictionary with gradient statistics.
        """
        params = list(self.model.parameters())
        return get_gradient_stats(
            params,
            include_histograms=False,  # Too expensive for regular use
            compute_percentiles=self.finalization_count % 100 == 0,  # Every 100 steps
        )

    @contextmanager
    def _measure_time(self, operation: str) -> Generator[None, None, None]:
        """Context manager for measuring operation time.

        Args:
            operation: Name of the operation being measured.
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            if self.config.enable_timing_stats:
                logger.debug(f"{operation} took {elapsed:.3f}s")

    def _estimate_gradient_bytes(self) -> int:
        """Estimate the number of bytes in gradients.

        Returns:
            Estimated number of bytes.
        """
        total_bytes = 0
        for param in self.model.parameters():
            if param.grad is not None:
                total_bytes += param.grad.numel() * param.grad.element_size()
        return total_bytes

    def _recover_from_sync_error(self, error: Exception) -> Dict[str, Any]:
        """Attempt to recover from synchronization error.

        Args:
            error: The error that occurred.

        Returns:
            Recovery statistics.
        """
        logger.warning(f"Attempting recovery from sync error: {error}")

        # Clear any pending async operations
        if dist.is_initialized():
            dist.barrier()

        # Return minimal stats to continue
        return {
            "sync_error": str(error),
            "recovered": True,
            "total_gradient_norm": 0.0,
        }

    def _handle_virtual_pipeline_gradients(self) -> None:
        """Handle gradient synchronization for virtual pipeline parallel ranks."""
        if self.virtual_pp_rank is None or self.virtual_pp_size is None:
            return

        # Virtual pipeline parallel requires special handling
        # as multiple virtual ranks may map to the same physical rank
        pp_group = self.process_groups.get("pp")
        if pp_group is None:
            return

        # Get physical pipeline parallel rank
        pp_rank = self.ranks.get("pp", 0)
        pp_size = self.world_sizes.get("pp", 1)

        if pp_size <= 1:
            return

        # Calculate which virtual ranks are on this physical rank
        virtual_ranks_per_physical = self.virtual_pp_size // pp_size
        my_virtual_ranks = list(
            range(
                pp_rank * virtual_ranks_per_physical,
                (pp_rank + 1) * virtual_ranks_per_physical,
            )
        )

        # If this physical rank handles multiple virtual ranks,
        # we need to aggregate gradients across virtual pipeline stages
        if len(my_virtual_ranks) > 1:
            # This would require model-specific logic to identify
            # which parameters belong to which virtual stage
            # For now, we log a warning
            if self.config.verbose:
                logger.debug(
                    f"Physical PP rank {pp_rank} handles virtual "
                    f"ranks {my_virtual_ranks}"
                )

    def _sync_batch_norm_stats(self) -> None:
        """Synchronize batch normalization statistics across data parallel ranks."""
        dp_group = self.process_groups.get("dp")
        if dp_group is None or self.world_sizes.get("dp", 1) <= 1:
            return

        # Find all batch norm layers
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # Synchronize running mean and variance
                if module.running_mean is not None:
                    dist.all_reduce(
                        module.running_mean,
                        op=dist.ReduceOp.SUM,
                        group=dp_group,
                    )
                    module.running_mean.div_(self.world_sizes["dp"])

                if module.running_var is not None:
                    dist.all_reduce(
                        module.running_var,
                        op=dist.ReduceOp.SUM,
                        group=dp_group,
                    )
                    module.running_var.div_(self.world_sizes["dp"])

    def _sync_layer_norm_stats(self) -> None:
        """Synchronize layer normalization statistics if needed."""
        # Layer norm typically doesn't need synchronization as it normalizes
        # within each sample, but this is here for completeness
        pass

    def cleanup(self) -> None:
        """Clean up resources used by the gradient finalizer."""
        # Release gradient buffers back to pool
        with self._buffer_lock:
            for buffer in self.gradient_buffers.values():
                self.gradient_buffer_pool.release(buffer)
            self.gradient_buffers.clear()

        # Clear buffer pool
        self.gradient_buffer_pool.clear()

        # Clear stats history
        with self._stats_lock:
            self.stats_history.clear()

        logger.info("Gradient finalizer resources cleaned up")

    def __enter__(self) -> "GradientFinalizer":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - cleanup resources."""
        self.cleanup()

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary.

        Returns:
            Dictionary with performance metrics.
        """
        return self._perf_metrics.get_summary()

    def get_statistics_summary(self) -> Dict[str, Any]:
        """Get summary of gradient finalization statistics.

        Returns:
            Dictionary with summary statistics.
        """
        if not self.stats_history:
            return {
                "total_finalizations": self.finalization_count,
                "no_statistics_collected": True,
            }

        # Compute summary statistics
        summary = {
            "total_finalizations": self.finalization_count,
            "avg_finalization_time": 0.0,
            "avg_gradient_norm": 0.0,
            "max_gradient_norm": 0.0,
            "min_gradient_norm": float("inf"),
            "num_clipped": 0,
            "num_non_finite": 0,
        }

        for stats in self.stats_history:
            summary["avg_finalization_time"] += stats.get("finalization_time", 0.0)
            grad_norm = stats.get("gradient_norm", 0.0)
            summary["avg_gradient_norm"] += grad_norm
            summary["max_gradient_norm"] = max(summary["max_gradient_norm"], grad_norm)
            summary["min_gradient_norm"] = min(summary["min_gradient_norm"], grad_norm)

            if stats.get("clipped", False):
                summary["num_clipped"] += 1

            if not stats.get("finite", True):
                summary["num_non_finite"] += 1

        num_stats = len(self.stats_history)
        if num_stats > 0:
            summary["avg_finalization_time"] /= num_stats
            summary["avg_gradient_norm"] /= num_stats

        return summary

    def reset_statistics(self) -> None:
        """Reset collected statistics."""
        self.stats_history.clear()
        self.finalization_count = 0

    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for checkpointing.

        Returns:
            State dictionary.
        """
        return {
            "finalization_count": self.finalization_count,
            "config": self.config.to_dict(),
            "statistics_summary": self.get_statistics_summary(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint.

        Args:
            state_dict: State dictionary to load.
        """
        self.finalization_count = state_dict.get("finalization_count", 0)
        # Config is immutable after initialization
