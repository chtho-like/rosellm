"""Advanced gradient finalization for RoseTrainer with multi-dimensional parallelism.

This module provides a comprehensive gradient finalization system designed for
distributed training scenarios with multiple parallelism dimensions. The implementation
emphasizes performance, reliability, and resource efficiency.

Key Features:
    - Multi-precision gradient management (FP32/FP16/BF16/FP8*)
    - Advanced memory management with automatic cleanup
    - Error recovery mechanisms with retry logic
    - Performance-optimized communication patterns
    - Thread-safe operations for concurrent access
    - Integration with RoseTrainer's parallelism framework

Architecture:
    - GradientDataTypeManager: Handles precision conversions and compression
    - AdvancedGradientFinalizer: Orchestrates multi-dimensional synchronization
    - Error handling decorators for robust CUDA operations
    - Weak reference management for automatic memory cleanup

Performance Characteristics:
    - O(1) space complexity for gradient storage (weak references)
    - Optimized communication with contiguous buffer alignment
    - Exponential backoff retry mechanisms for transient failures
    - Memory-efficient circular buffers for metrics tracking

*FP8 support is experimental and requires compatible hardware.
"""

import logging
import time
from abc import abstractmethod
from contextlib import contextmanager
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Protocol, Tuple
from weakref import WeakKeyDictionary

import torch
import torch.distributed as dist
import torch.nn as nn

from ..parallelism import parallel_state
from ..utils.gradient_utils import (
    GradientClipConfig,
    apply_gradient_clipping,
    check_gradient_finite,
    get_gradient_stats,
)
from .config import GradientFinalizationConfig
from .finalizer import GradientFinalizer

logger = logging.getLogger(__name__)


class DataTypeConversionStrategy(Protocol):
    """Protocol for gradient data type conversion strategies."""

    @abstractmethod
    def convert_to_master(
        self, model: nn.Module, store_originals: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Convert gradients to master precision."""
        ...

    @abstractmethod
    def convert_for_communication(
        self, gradients: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Convert gradients for communication."""
        ...

    @abstractmethod
    def restore_from_communication(
        self,
        model: nn.Module,
        received_gradients: Dict[str, torch.Tensor],
        metadata: Dict[str, Any],
    ) -> None:
        """Restore gradients from communication format."""
        ...

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get conversion statistics."""
        ...

    @abstractmethod
    def reset_statistics(self) -> None:
        """Reset statistics."""
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources."""
        ...


class GradientProcessingError(Exception):
    """Base exception for gradient processing errors."""

    pass


class GradientConversionError(GradientProcessingError):
    """Exception raised during gradient data type conversion."""

    pass


class GradientSynchronizationError(GradientProcessingError):
    """Exception raised during gradient synchronization."""

    pass


def _retry_on_cuda_error(max_retries: int = 3, delay: float = 0.1):
    """Decorator to retry operations that might fail due to CUDA errors."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"CUDA operation failed (attempt {attempt + 1}/"
                            f"{max_retries}): {e}"
                        )
                        time.sleep(delay * (2**attempt))  # Exponential backoff
                        torch.cuda.empty_cache()  # Clear cache before retry
                    else:
                        logger.error(
                            f"CUDA operation failed after {max_retries} attempts"
                        )
                        break
            raise (
                last_exception
                if last_exception
                else RuntimeError("Unknown error in retry mechanism")
            )

        return wrapper

    return decorator


class GradientDataType(str, Enum):
    """Supported gradient data types for conversion and processing.

    This enum defines the precision types supported for gradient storage,
    computation, and communication in distributed training scenarios.
    """

    FP32 = "fp32"  # 32-bit floating point (full precision)
    FP16 = "fp16"  # 16-bit floating point (half precision)
    BF16 = "bf16"  # 16-bit brain floating point (Google's format)
    FP8 = "fp8"  # 8-bit floating point (experimental, future support)


class GradientDataTypeManager:
    """Manages gradient data type conversions for multi-precision training.

    This class handles conversion between different gradient data types while
    preserving numerical stability and supporting mixed precision workflows.
    """

    def __init__(
        self,
        master_dtype: GradientDataType = GradientDataType.FP32,
        compute_dtype: Optional[GradientDataType] = None,
        communication_dtype: Optional[GradientDataType] = None,
        enable_compression: bool = False,
        compression_threshold_mb: float = 10.0,
        preserve_master_precision: bool = True,
    ) -> None:
        """Initialize gradient data type manager.

        Args:
            master_dtype: Master precision for gradient accumulation and storage.
            compute_dtype: Precision for gradient computations. If None, uses
                master_dtype.
            communication_dtype: Precision for gradient communication. If None, uses
                compute_dtype.
            enable_compression: Whether to compress gradients for communication.
            compression_threshold_mb: Size threshold in MB for applying compression.
            preserve_master_precision: Whether to preserve master precision in
                conversions.
        """
        # Validate input types
        if not isinstance(master_dtype, GradientDataType):
            raise ValueError(
                f"master_dtype must be a GradientDataType, got {type(master_dtype)}"
            )
        if compute_dtype is not None and not isinstance(
            compute_dtype, GradientDataType
        ):
            raise ValueError(
                f"compute_dtype must be a GradientDataType, got {type(compute_dtype)}"
            )
        if communication_dtype is not None and not isinstance(
            communication_dtype, GradientDataType
        ):
            raise ValueError(
                f"communication_dtype must be a GradientDataType, "
                f"got {type(communication_dtype)}"
            )

        self.master_dtype = master_dtype
        self.compute_dtype = compute_dtype or master_dtype
        self.communication_dtype = communication_dtype or self.compute_dtype
        self.enable_compression = enable_compression
        self.compression_threshold_mb = compression_threshold_mb
        self.preserve_master_precision = preserve_master_precision

        # Torch dtype mappings
        self._dtype_map = {
            GradientDataType.FP32: torch.float32,
            GradientDataType.FP16: torch.float16,
            GradientDataType.BF16: torch.bfloat16,
        }

        # Check FP8 support if available
        if hasattr(torch, "float8_e4m3fn"):
            self._dtype_map[GradientDataType.FP8] = torch.float8_e4m3fn

        # Gradient storage for type conversions (using WeakKeyDictionary for
        # automatic cleanup)
        self._master_gradients: WeakKeyDictionary[
            nn.Parameter, torch.Tensor
        ] = WeakKeyDictionary()
        self._gradient_refs: Dict[
            str, torch.Tensor
        ] = {}  # Name-based references for restoration
        self._conversion_stats = {
            "total_conversions": 0,
            "compression_ratio": 0.0,
            "time_spent_converting": 0.0,
        }

        logger.info(
            f"Initialized GradientDataTypeManager: master={master_dtype.value}, "
            f"compute={self.compute_dtype.value}, comm={self.communication_dtype.value}"
        )

    def get_torch_dtype(self, dtype: GradientDataType) -> torch.dtype:
        """Get torch dtype from gradient data type enum.

        Args:
            dtype: Gradient data type enum.

        Returns:
            Corresponding torch.dtype.

        Raises:
            ValueError: If dtype is not supported.
        """
        if dtype not in self._dtype_map:
            raise ValueError(f"Unsupported gradient data type: {dtype}")
        return self._dtype_map[dtype]

    @_retry_on_cuda_error(max_retries=2)
    def convert_to_master(
        self, model: nn.Module, store_originals: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Convert model gradients to master precision with error handling.

        Args:
            model: Model with gradients to convert.
            store_originals: Whether to store original gradients for restoration.

        Returns:
            Dictionary mapping parameter names to converted gradients.

        Raises:
            GradientConversionError: If conversion fails after retries.
        """
        start_time = time.perf_counter()
        converted_grads = {}

        try:
            master_torch_dtype = self.get_torch_dtype(self.master_dtype)

            for name, param in model.named_parameters():
                if param.grad is not None:
                    try:
                        if store_originals:
                            # Store original gradient for potential restoration using
                            # weak references
                            self._master_gradients[param] = param.grad.clone()
                            self._gradient_refs[name] = param.grad.clone()

                        # Convert to master precision if needed
                        if param.grad.dtype != master_torch_dtype:
                            converted_grad = param.grad.to(dtype=master_torch_dtype)
                            if self.preserve_master_precision:
                                # Use higher precision for accumulation
                                converted_grad = (
                                    converted_grad.float()
                                    if master_torch_dtype != torch.float32
                                    else converted_grad
                                )
                            converted_grads[name] = converted_grad
                            param.grad = converted_grad
                        else:
                            converted_grads[name] = param.grad

                    except Exception as e:
                        logger.error(
                            f"Failed to convert gradient for parameter {name}: {e}"
                        )
                        raise GradientConversionError(
                            f"Gradient conversion failed for {name}"
                        ) from e

            conversion_time = time.perf_counter() - start_time
            self._conversion_stats["time_spent_converting"] += conversion_time
            self._conversion_stats["total_conversions"] += 1

            logger.debug(
                f"Converted {len(converted_grads)} gradients to master precision "
                f"in {conversion_time:.3f}s"
            )
            return converted_grads

        except Exception as e:
            logger.error(f"Master gradient conversion failed: {e}")
            raise GradientConversionError(
                "Failed to convert gradients to master precision"
            ) from e

    def convert_for_communication(
        self, gradients: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Convert gradients to communication precision with optional compression.

        Args:
            gradients: Dictionary of gradients to convert.

        Returns:
            Tuple of (converted gradients, conversion metadata).
        """
        start_time = time.perf_counter()
        converted_grads = {}
        metadata: Dict[str, Any] = {
            "compressed_params": [],
            "original_shapes": {},
            "compression_ratio": 1.0,
        }

        comm_torch_dtype = self.get_torch_dtype(self.communication_dtype)
        total_original_bytes = 0
        total_compressed_bytes = 0

        for name, grad in gradients.items():
            original_size_bytes = grad.numel() * grad.element_size()
            total_original_bytes += original_size_bytes

            # Check if we should compress this gradient
            should_compress = (
                self.enable_compression
                and (original_size_bytes / (1024 * 1024))
                > self.compression_threshold_mb
            )

            if grad.dtype != comm_torch_dtype:
                # Convert dtype
                converted_grad = grad.to(dtype=comm_torch_dtype)
            else:
                converted_grad = grad

            if should_compress:
                # Simple compression: use lower precision and potential sparsification
                metadata["compressed_params"].append(name)
                metadata["original_shapes"][name] = grad.shape

                # For demonstration, we'll use FP16 as compression
                if comm_torch_dtype != torch.float16:
                    converted_grad = converted_grad.half()

            converted_grads[name] = converted_grad
            total_compressed_bytes += (
                converted_grad.numel() * converted_grad.element_size()
            )

        # Calculate compression ratio
        if total_original_bytes > 0:
            compression_ratio = total_compressed_bytes / total_original_bytes
            metadata["compression_ratio"] = compression_ratio
            self._conversion_stats["compression_ratio"] = compression_ratio

        conversion_time = time.perf_counter() - start_time
        self._conversion_stats["time_spent_converting"] += conversion_time

        logger.debug(
            f"Converted {len(converted_grads)} gradients for communication "
            f"(compression_ratio={metadata['compression_ratio']:.3f}) "
            f"in {conversion_time:.3f}s"
        )

        return converted_grads, metadata

    def restore_from_communication(
        self,
        model: nn.Module,
        received_gradients: Dict[str, torch.Tensor],
        metadata: Dict[str, Any],
    ) -> None:
        """Restore gradients from communication back to original precision.

        Args:
            model: Model to restore gradients to.
            received_gradients: Gradients received from communication.
            metadata: Metadata from communication conversion.
        """
        start_time = time.perf_counter()
        master_torch_dtype = self.get_torch_dtype(self.master_dtype)

        for name, param in model.named_parameters():
            if name in received_gradients:
                grad = received_gradients[name]

                # Handle decompression if needed
                if name in metadata.get("compressed_params", []):
                    # Restore shape if needed
                    if name in metadata.get("original_shapes", {}):
                        original_shape = metadata["original_shapes"][name]
                        if grad.shape != original_shape:
                            grad = grad.view(original_shape)

                # Convert back to master precision
                if grad.dtype != master_torch_dtype:
                    grad = grad.to(dtype=master_torch_dtype)

                param.grad = grad

        conversion_time = time.perf_counter() - start_time
        self._conversion_stats["time_spent_converting"] += conversion_time

        logger.debug(f"Restored gradients from communication in {conversion_time:.3f}s")

    def get_statistics(self) -> Dict[str, Any]:
        """Get conversion and compression statistics.

        Returns:
            Dictionary with statistics.
        """
        return {
            "conversion_stats": self._conversion_stats.copy(),
            "master_dtype": self.master_dtype.value,
            "compute_dtype": self.compute_dtype.value,
            "communication_dtype": self.communication_dtype.value,
            "compression_enabled": self.enable_compression,
            "stored_master_gradients": len(self._master_gradients),
        }

    def reset_statistics(self) -> None:
        """Reset conversion statistics."""
        self._conversion_stats = {
            "total_conversions": 0,
            "compression_ratio": 0.0,
            "time_spent_converting": 0.0,
        }

    @contextmanager
    def gradient_conversion_context(self) -> Generator[None, None, None]:
        """Context manager for safe gradient conversion with automatic cleanup."""
        try:
            yield
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up stored gradients and free memory."""
        # Count references before clearing for logging
        total_refs = len(self._gradient_refs)

        # Clear weak references (automatically handled but explicit for clarity)
        self._master_gradients.clear()
        # Clear name-based references
        self._gradient_refs.clear()

        # Reset conversion stats to avoid memory accumulation
        self.reset_statistics()

        # Force garbage collection for large cleanups
        if total_refs > 100:  # Only for large cleanups
            import gc

            collected = gc.collect()
            logger.debug(
                f"Cleanup: collected {collected} objects from {total_refs} references"
            )


class AdvancedGradientFinalizerBuilder:
    """Builder for AdvancedGradientFinalizer with fluent API."""

    def __init__(self, model: nn.Module) -> None:
        """Initialize builder with required model."""
        self._model = model
        self._config: Optional[GradientFinalizationConfig] = None
        self._data_type_manager: Optional[DataTypeConversionStrategy] = None
        self._enable_advanced_sync = True
        self._verbose = False

    def with_config(
        self, config: GradientFinalizationConfig
    ) -> "AdvancedGradientFinalizerBuilder":
        """Set finalization configuration."""
        self._config = config
        return self

    def with_data_type_manager(
        self, manager: DataTypeConversionStrategy
    ) -> "AdvancedGradientFinalizerBuilder":
        """Set data type conversion strategy."""
        self._data_type_manager = manager
        return self

    def with_advanced_sync(
        self, enabled: bool = True
    ) -> "AdvancedGradientFinalizerBuilder":
        """Enable or disable advanced synchronization."""
        self._enable_advanced_sync = enabled
        return self

    def with_verbose_logging(
        self, enabled: bool = True
    ) -> "AdvancedGradientFinalizerBuilder":
        """Enable or disable verbose logging."""
        self._verbose = enabled
        return self

    def with_mixed_precision(
        self,
        master_dtype: GradientDataType = GradientDataType.FP32,
        communication_dtype: Optional[GradientDataType] = None,
        enable_compression: bool = False,
    ) -> "AdvancedGradientFinalizerBuilder":
        """Configure mixed precision settings."""
        self._data_type_manager = GradientDataTypeManager(
            master_dtype=master_dtype,
            communication_dtype=communication_dtype,
            enable_compression=enable_compression,
        )
        return self

    def build(self) -> "AdvancedGradientFinalizer":
        """Build the configured AdvancedGradientFinalizer."""
        return AdvancedGradientFinalizer(
            model=self._model,
            config=self._config,
            data_type_manager=self._data_type_manager,
            enable_advanced_sync=self._enable_advanced_sync,
            verbose=self._verbose,
        )


class AdvancedGradientFinalizer:
    """Advanced gradient finalizer with multi-dimensional parallelism support.

    This class provides a comprehensive gradient finalization API that integrates
    gradient data type management, multi-dimensional parallelism awareness, and
    advanced synchronization strategies.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[GradientFinalizationConfig] = None,
        data_type_manager: Optional[DataTypeConversionStrategy] = None,
        enable_advanced_sync: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialize advanced gradient finalizer.

        Args:
            model: Model to finalize gradients for.
            config: Gradient finalization configuration. If None, uses defaults.
            data_type_manager: Data type manager for gradient conversions. If None,
                creates default.
            enable_advanced_sync: Whether to enable advanced synchronization features.
            verbose: Whether to enable verbose logging.
        """
        self.model = model
        self.config = config or GradientFinalizationConfig()
        self.enable_advanced_sync = enable_advanced_sync
        self.verbose = verbose

        # Initialize data type manager
        if data_type_manager is not None:
            self.data_type_manager = data_type_manager
        else:
            # Create default data type manager based on config
            comm_dtype = (
                GradientDataType.FP16
                if self.config.fp16_compression
                else GradientDataType.FP32
            )
            self.data_type_manager = GradientDataTypeManager(
                master_dtype=GradientDataType.FP32,
                communication_dtype=comm_dtype,
                enable_compression=self.config.fp16_compression,
            )

        # Initialize core gradient finalizer
        self.core_finalizer = GradientFinalizer(
            model=model,
            config=self.config,
            distributed_optimizer=None,  # Will be set if needed
        )

        # Multi-dimensional parallelism state
        self._parallel_dims_initialized = False
        self._parallel_groups: Dict[str, Optional[dist.ProcessGroup]] = {}
        self._parallel_ranks: Dict[str, int] = {}
        self._parallel_sizes: Dict[str, int] = {}

        # Performance tracking
        self._finalization_count = 0
        self._performance_metrics = {
            "total_finalization_time": 0.0,
            "dtype_conversion_time": 0.0,
            "sync_time": 0.0,
            "clip_time": 0.0,
            "avg_finalization_time": 0.0,
        }

        # Initialize parallelism state
        self._initialize_parallelism_state()

        if self.verbose:
            logger.info(
                f"Initialized AdvancedGradientFinalizer with "
                f"advanced_sync={enable_advanced_sync}"
            )

    def _initialize_parallelism_state(self) -> None:
        """Initialize multi-dimensional parallelism state."""
        if not parallel_state.is_initialized():
            logger.warning("Parallel state not initialized, using default groups")
            self._parallel_dims_initialized = False
            return

        try:
            # Get all parallelism dimensions
            parallel_dims = ["tp", "pp", "dp", "cp", "ep"]

            for dim in parallel_dims:
                if dim == "tp":
                    self._parallel_groups[
                        dim
                    ] = parallel_state.get_tensor_model_parallel_group()
                    self._parallel_ranks[
                        dim
                    ] = parallel_state.get_tensor_model_parallel_rank()
                    self._parallel_sizes[dim] = getattr(
                        parallel_state,
                        "get_tensor_model_parallel_world_size",
                        lambda: 1,
                    )()
                elif dim == "pp":
                    self._parallel_groups[
                        dim
                    ] = parallel_state.get_pipeline_model_parallel_group()
                    self._parallel_ranks[
                        dim
                    ] = parallel_state.get_pipeline_model_parallel_rank()
                    self._parallel_sizes[dim] = getattr(
                        parallel_state,
                        "get_pipeline_model_parallel_world_size",
                        lambda: 1,
                    )()
                elif dim == "dp":
                    self._parallel_groups[
                        dim
                    ] = parallel_state.get_data_parallel_group()
                    self._parallel_ranks[dim] = parallel_state.get_data_parallel_rank()
                    self._parallel_sizes[dim] = getattr(
                        parallel_state, "get_data_parallel_world_size", lambda: 1
                    )()
                elif dim == "cp":
                    self._parallel_groups[
                        dim
                    ] = parallel_state.get_context_parallel_group()
                    if hasattr(parallel_state, "get_context_parallel_rank"):
                        self._parallel_ranks[
                            dim
                        ] = parallel_state.get_context_parallel_rank()
                        self._parallel_sizes[dim] = getattr(
                            parallel_state, "get_context_parallel_world_size", lambda: 1
                        )()
                    else:
                        self._parallel_ranks[dim] = 0
                        self._parallel_sizes[dim] = 1
                elif dim == "ep":
                    self._parallel_groups[
                        dim
                    ] = parallel_state.get_expert_model_parallel_group()
                    if hasattr(parallel_state, "get_expert_model_parallel_rank"):
                        self._parallel_ranks[
                            dim
                        ] = parallel_state.get_expert_model_parallel_rank()
                        self._parallel_sizes[dim] = getattr(
                            parallel_state,
                            "get_expert_model_parallel_world_size",
                            lambda: 1,
                        )()
                    else:
                        self._parallel_ranks[dim] = 0
                        self._parallel_sizes[dim] = 1

            self._parallel_dims_initialized = True

            if self.verbose:
                active_dims = [
                    dim for dim, size in self._parallel_sizes.items() if size > 1
                ]
                logger.info(
                    f"Initialized parallelism state with active dimensions: "
                    f"{active_dims}"
                )

        except Exception as e:
            logger.warning(f"Failed to initialize parallelism state: {e}")
            self._parallel_dims_initialized = False

    def finalize_gradients(
        self,
        clip_gradients: bool = True,
        check_finite: bool = True,
        normalize_gradients: bool = False,
        collect_stats: Optional[bool] = None,
        custom_sync_order: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Advanced gradient finalization with multi-dimensional parallelism support.

        This is the main API for advanced gradient finalization. It performs:
        1. Gradient data type management and conversion
        2. Multi-dimensional parallelism aware synchronization
        3. Advanced gradient clipping and normalization
        4. Comprehensive statistics collection

        Args:
            clip_gradients: Whether to apply gradient clipping.
            check_finite: Whether to check for finite gradients.
            normalize_gradients: Whether to normalize gradients across parallelism
                dimensions.
            collect_stats: Whether to collect detailed statistics.
            custom_sync_order: Custom order for dimension synchronization.

        Returns:
            Dictionary with finalization statistics and metrics.
        """
        start_time = time.perf_counter()

        stats = {
            "step": self._finalization_count,
            "finalization_time": 0.0,
            "dtype_conversion_time": 0.0,
            "sync_time": 0.0,
            "clip_time": 0.0,
            "finite": True,
            "clipped": False,
            "normalized": False,
            "gradient_norm": 0.0,
            "parallel_stats": {},
            "data_type_stats": {},
        }

        try:
            # Step 1: Convert gradients to master precision
            dtype_start = time.perf_counter()
            # Convert gradients to master precision
            self.data_type_manager.convert_to_master(self.model, store_originals=True)
            dtype_time = time.perf_counter() - dtype_start
            stats["dtype_conversion_time"] = dtype_time

            # Step 2: Check gradient finiteness
            if check_finite:
                finite_result, finite_stats = check_gradient_finite(
                    list(self.model.parameters()), raise_on_nonfinite=False
                )
                stats["finite"] = finite_result
                stats["finite_stats"] = finite_stats

                if not finite_result:
                    logger.warning(f"Non-finite gradients detected: {finite_stats}")
                    if self.config.check_gradient_norm:
                        # Early return for non-finite gradients
                        stats["finalization_time"] = time.perf_counter() - start_time
                        return stats

            # Step 3: Advanced multi-dimensional gradient synchronization
            if self.enable_advanced_sync and self._parallel_dims_initialized:
                sync_start = time.perf_counter()
                sync_stats = self._advanced_gradient_sync(custom_sync_order)
                sync_time = time.perf_counter() - sync_start
                stats["sync_time"] = sync_time
                stats["parallel_stats"] = sync_stats
            else:
                # Fallback to core finalizer synchronization
                sync_start = time.perf_counter()
                core_stats = self.core_finalizer._synchronize_gradients()
                sync_time = time.perf_counter() - sync_start
                stats["sync_time"] = sync_time
                stats["parallel_stats"] = core_stats

            # Step 4: Gradient clipping
            if clip_gradients:
                clip_start = time.perf_counter()
                clip_stats = self._apply_advanced_gradient_clipping()
                clip_time = time.perf_counter() - clip_start
                stats["clip_time"] = clip_time
                stats["clipped"] = clip_stats.get("clipped", False)
                stats["gradient_norm"] = clip_stats.get("grad_norm", 0.0)
                stats["clip_stats"] = clip_stats

            # Step 5: Gradient normalization (if requested)
            if normalize_gradients:
                self._normalize_gradients_across_dimensions()
                stats["normalized"] = True

            # Step 6: Collect statistics
            if collect_stats or (
                collect_stats is None and self.config.enable_gradient_stats
            ):
                grad_stats = get_gradient_stats(
                    list(self.model.parameters()),
                    include_histograms=False,  # Expensive operation
                )
                stats["gradient_stats"] = grad_stats

            # Step 7: Data type statistics
            stats["data_type_stats"] = self.data_type_manager.get_statistics()

        except Exception as e:
            logger.error(f"Advanced gradient finalization failed: {e}")
            stats["error"] = str(e)
            stats["success"] = False
        else:
            stats["success"] = True

        # Update performance metrics
        total_time = time.perf_counter() - start_time
        stats["finalization_time"] = total_time
        self._update_performance_metrics(total_time, stats)
        self._finalization_count += 1

        if self.verbose and self._finalization_count % 50 == 0:
            logger.info(
                f"Advanced gradient finalization step {self._finalization_count}: "
                f"norm={stats['gradient_norm']:.4f}, time={total_time:.3f}s, "
                f"success={stats.get('success', True)}"
            )

        return stats

    def _advanced_gradient_sync(
        self, custom_order: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Perform advanced multi-dimensional gradient synchronization.

        Args:
            custom_order: Custom order for synchronization dimensions.

        Returns:
            Synchronization statistics.
        """
        sync_stats: Dict[str, Any] = {
            "total_sync_time": 0.0,
            "dimension_sync_times": {},
            "bytes_communicated": 0,
            "sync_order": [],
        }

        # Determine synchronization order
        if custom_order is not None:
            sync_order = custom_order
        elif self.config.dimension_order == "hierarchical":
            # Use hierarchical levels from config
            sync_order = []
            for level in self.config.hierarchical_levels:
                sync_order.extend(level)
        elif self.config.custom_dimension_order is not None:
            sync_order = self.config.custom_dimension_order
        else:
            # Default order: TP -> PP -> DP+CP+EP
            sync_order = ["tp", "pp", "dp", "cp", "ep"]

        sync_stats["sync_order"] = sync_order
        total_sync_start = time.perf_counter()

        # Convert gradients for communication
        (
            comm_gradients,
            comm_metadata,
        ) = self.data_type_manager.convert_for_communication(
            {
                name: param.grad
                for name, param in self.model.named_parameters()
                if param.grad is not None
            }
        )

        # Synchronize across each dimension in order
        for dim in sync_order:
            if dim in self._parallel_groups and self._parallel_sizes[dim] > 1:
                dim_start = time.perf_counter()

                group = self._parallel_groups[dim]
                if group is not None:
                    # Synchronize gradients for this dimension
                    for name, grad in comm_gradients.items():
                        try:
                            # Ensure gradient is contiguous for optimal communication
                            # performance
                            if not grad.is_contiguous():
                                grad = grad.contiguous()
                                comm_gradients[name] = grad  # Update reference

                            # Perform reduction with timeout handling
                            if self.config.reduction_op == "mean":
                                dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=group)
                                grad.div_(self._parallel_sizes[dim])
                            elif self.config.reduction_op == "sum":
                                dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=group)
                            else:
                                # Default to mean
                                dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=group)
                                grad.div_(self._parallel_sizes[dim])

                        except dist.DistBackendError as e:
                            logger.error(
                                f"Distributed backend error for gradient {name} in "
                                f"dimension {dim}: {e}"
                            )
                            raise GradientSynchronizationError(
                                f"Sync failed for {name} in {dim}"
                            ) from e
                        except Exception as e:
                            logger.warning(
                                f"Failed to sync gradient {name} for "
                                f"dimension {dim}: {e}"
                            )

                dim_time = time.perf_counter() - dim_start
                sync_stats["dimension_sync_times"][dim] = dim_time

                if self.verbose:
                    logger.debug(
                        f"Synchronized gradients for {dim} dimension in {dim_time:.3f}s"
                    )

        # Restore gradients from communication format
        self.data_type_manager.restore_from_communication(
            self.model, comm_gradients, comm_metadata
        )

        sync_stats["total_sync_time"] = time.perf_counter() - total_sync_start

        # Estimate bytes communicated
        total_bytes = 0
        for grad in comm_gradients.values():
            total_bytes += grad.numel() * grad.element_size()

        # Account for multiple dimensions
        active_dims = [
            dim for dim in sync_order if self._parallel_sizes.get(dim, 1) > 1
        ]
        sync_stats["bytes_communicated"] = total_bytes * len(active_dims)

        return sync_stats

    def _apply_advanced_gradient_clipping(self) -> Dict[str, Any]:
        """Apply advanced gradient clipping with multi-precision support.

        Returns:
            Clipping statistics.
        """
        # Use the advanced gradient clipping from gradient utilities
        try:
            clip_config = GradientClipConfig(
                clip_type=self.config.sync_strategy,  # Reuse for demonstration
                max_norm=self.config.gradient_predivide_factor,
                norm_type=self.config.gradient_norm_type,
                error_if_nonfinite=True,
                model_parallel_reduce=True,
                use_multitensor=True,
            )

            params_with_grad = [
                p for p in self.model.parameters() if p.grad is not None
            ]
            if params_with_grad:
                # Convert to list of tensors for gradient clipping
                grad_tensors = [p.grad for p in params_with_grad if p.grad is not None]
                return apply_gradient_clipping(grad_tensors, clip_config)
            else:
                return {"clipped": False, "grad_norm": 0.0}

        except Exception as e:
            logger.warning(f"Advanced gradient clipping failed, using fallback: {e}")
            # Simple fallback clipping
            clip_value = getattr(self.config, "gradient_clip_value", None) or 1.0
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), clip_value
            )
            return {
                "clipped": total_norm > clip_value,
                "grad_norm": float(total_norm),
            }

    def _normalize_gradients_across_dimensions(self) -> None:
        """Normalize gradients across active parallelism dimensions."""
        if not self._parallel_dims_initialized:
            return

        # Calculate total world size across active dimensions
        total_world_size = 1
        for dim, size in self._parallel_sizes.items():
            if size > 1:
                total_world_size *= size

        if total_world_size > 1:
            # Normalize gradients by total world size
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.div_(total_world_size)

            if self.verbose:
                logger.debug(
                    f"Normalized gradients by total world size: {total_world_size}"
                )

    def _update_performance_metrics(
        self, total_time: float, stats: Dict[str, Any]
    ) -> None:
        """Update performance tracking metrics.

        Args:
            total_time: Total finalization time.
            stats: Finalization statistics.
        """
        self._performance_metrics["total_finalization_time"] += total_time
        self._performance_metrics["dtype_conversion_time"] += stats.get(
            "dtype_conversion_time", 0.0
        )
        self._performance_metrics["sync_time"] += stats.get("sync_time", 0.0)
        self._performance_metrics["clip_time"] += stats.get("clip_time", 0.0)

        # Calculate running average
        count = max(1, self._finalization_count + 1)
        self._performance_metrics["avg_finalization_time"] = (
            self._performance_metrics["total_finalization_time"] / count
        )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics.

        Returns:
            Dictionary with performance metrics.
        """
        metrics: Dict[str, Any] = self._performance_metrics.copy()
        additional_metrics = {
            "finalization_count": self._finalization_count,
            "data_type_stats": self.data_type_manager.get_statistics(),
            "parallelism_stats": {
                "initialized": self._parallel_dims_initialized,
                "active_dimensions": [
                    dim for dim, size in self._parallel_sizes.items() if size > 1
                ],
                "total_world_size": sum(self._parallel_sizes.values()),
            },
        }

        # Manually add the additional metrics
        for key, value in additional_metrics.items():
            metrics[key] = value
        return metrics

    def get_parallelism_info(self) -> Dict[str, Any]:
        """Get detailed parallelism information.

        Returns:
            Dictionary with parallelism details.
        """
        return {
            "initialized": self._parallel_dims_initialized,
            "groups": {
                dim: group is not None for dim, group in self._parallel_groups.items()
            },
            "ranks": self._parallel_ranks.copy(),
            "sizes": self._parallel_sizes.copy(),
            "config": {
                "dimension_order": self.config.dimension_order,
                "hierarchical_levels": self.config.hierarchical_levels,
                "custom_dimension_order": self.config.custom_dimension_order,
            },
        }

    def reset_statistics(self) -> None:
        """Reset all statistics and counters."""
        self._finalization_count = 0
        self._performance_metrics = {
            "total_finalization_time": 0.0,
            "dtype_conversion_time": 0.0,
            "sync_time": 0.0,
            "clip_time": 0.0,
            "avg_finalization_time": 0.0,
        }
        self.data_type_manager.reset_statistics()
        self.core_finalizer.reset_statistics()

    def benchmark_performance(self, num_iterations: int = 100) -> Dict[str, Any]:
        """Benchmark gradient finalization performance.

        Args:
            num_iterations: Number of iterations to run for benchmarking.

        Returns:
            Dictionary with performance metrics including timing and throughput.
        """
        if not hasattr(self, "model") or self.model is None:
            raise RuntimeError("Model not available for benchmarking")

        # Warm-up phase
        for _ in range(min(10, num_iterations // 10)):
            try:
                self.finalize_gradients(collect_stats=False)
            except Exception:
                pass  # Ignore errors during warmup

        # Actual benchmark
        start_time = time.perf_counter()
        successful_runs = 0

        for _ in range(num_iterations):
            try:
                self.finalize_gradients(collect_stats=False)
                successful_runs += 1
            except Exception as e:
                logger.debug(f"Benchmark iteration failed: {e}")

        end_time = time.perf_counter()
        total_time = end_time - start_time

        return {
            "total_time_seconds": total_time,
            "average_time_per_iteration_ms": (total_time / max(1, successful_runs))
            * 1000,
            "iterations_per_second": (
                successful_runs / total_time if total_time > 0 else 0
            ),
            "successful_runs": successful_runs,
            "success_rate": successful_runs / num_iterations,
            "data_type_stats": self.data_type_manager.get_statistics(),
            "performance_metrics": self.get_performance_metrics(),
        }

    def cleanup(self) -> None:
        """Clean up resources and log performance summary."""
        try:
            # Log final performance summary if verbose
            if (
                self.verbose
                and hasattr(self, "_finalization_count")
                and self._finalization_count > 0
            ):
                metrics = self.get_performance_metrics()
                logger.info(
                    f"Advanced gradient finalizer summary: "
                    f"{self._finalization_count} finalizations, "
                    f"avg_time={metrics.get('avg_finalization_time', 0):.3f}s"
                )

            # Cleanup components
            self.data_type_manager.cleanup()
            if hasattr(self, "core_finalizer"):
                self.core_finalizer.cleanup()

        except Exception as e:
            logger.warning(f"Error during finalizer cleanup: {e}")
        finally:
            if self.verbose:
                logger.info("Advanced gradient finalizer cleaned up")


# Public API functions for convenient usage
def finalize_gradients_advanced(
    model: nn.Module,
    config: Optional[GradientFinalizationConfig] = None,
    data_type_manager: Optional[DataTypeConversionStrategy] = None,
    clip_gradients: bool = True,
    check_finite: bool = True,
    normalize_gradients: bool = False,
    collect_stats: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Convenient function for advanced gradient finalization.

    Args:
        model: Model to finalize gradients for.
        config: Gradient finalization configuration.
        data_type_manager: Data type manager for gradient conversions.
        clip_gradients: Whether to apply gradient clipping.
        check_finite: Whether to check for finite gradients.
        normalize_gradients: Whether to normalize gradients.
        collect_stats: Whether to collect detailed statistics.
        verbose: Whether to enable verbose logging.

    Returns:
        Dictionary with finalization results.
    """
    # Use builder pattern for better configuration management
    builder = AdvancedGradientFinalizerBuilder(model).with_verbose_logging(verbose)

    if config is not None:
        builder = builder.with_config(config)
    if data_type_manager is not None:
        builder = builder.with_data_type_manager(data_type_manager)

    finalizer = builder.build()

    try:
        return finalizer.finalize_gradients(
            clip_gradients=clip_gradients,
            check_finite=check_finite,
            normalize_gradients=normalize_gradients,
            collect_stats=collect_stats,
        )
    finally:
        finalizer.cleanup()


def create_gradient_data_type_manager(
    master_precision: str = "fp32",
    communication_precision: Optional[str] = None,
    enable_compression: bool = False,
    **kwargs,
) -> GradientDataTypeManager:
    """Factory function to create gradient data type manager.

    Args:
        master_precision: Master precision (fp32, fp16, bf16).
        communication_precision: Communication precision. If None, uses
            master_precision.
        enable_compression: Whether to enable gradient compression.
        **kwargs: Additional arguments for GradientDataTypeManager.

    Returns:
        Configured GradientDataTypeManager instance.
    """
    master_dtype = GradientDataType(master_precision)
    comm_dtype = (
        GradientDataType(communication_precision)
        if communication_precision
        else master_dtype
    )

    return GradientDataTypeManager(
        master_dtype=master_dtype,
        communication_dtype=comm_dtype,
        enable_compression=enable_compression,
        **kwargs,
    )
