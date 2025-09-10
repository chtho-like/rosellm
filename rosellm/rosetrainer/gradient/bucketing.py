"""
Gradient Bucketing with Communication Overlap

This module implements an efficient gradient bucketing system inspired by Megatron-LM's
distributed training optimizations. It groups parameters into buckets for efficient
communication and enables overlapping of gradient reduction with backward computation.

Key Features:
- Automatic parameter grouping into communication-efficient buckets
- Overlapped gradient all-reduce/reduce-scatter with backward pass
- Support for both data-parallel and distributed optimizer modes
- Configurable bucket sizes and communication strategies
- Memory-efficient gradient accumulation

References:
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)

# Constants for bucketing
DEFAULT_BUCKET_SIZE_MB = 50  # Default bucket size in MB
MIN_BUCKET_SIZE_MB = 1  # Minimum bucket size in MB
MAX_BUCKET_SIZE_MB = 500  # Maximum bucket size in MB
ALIGNMENT_PADDING = 128  # Alignment for efficient memory access


class BucketingStrategy(str, Enum):
    """Strategies for grouping parameters into buckets."""

    SIZE_BASED = "size_based"  # Group by total size threshold
    TYPE_BASED = "type_based"  # Group by parameter type/dtype
    LAYER_BASED = "layer_based"  # Group by model layers
    HYBRID = "hybrid"  # Combination of strategies


@dataclass
class GradientBucketConfig:
    """Configuration for gradient bucketing.

    Attributes:
        bucket_size_mb: Target bucket size in megabytes
        bucketing_strategy: Strategy for grouping parameters
        overlap_communication: Enable communication/computation overlap
        use_distributed_optimizer: Use distributed optimizer mode
        alignment_padding: Memory alignment for efficient access
        bucket_cap_factor: Factor to cap maximum bucket size
        dtype_bucketing: Group parameters by dtype
        param_to_name_fn: Function to get parameter names for debugging
    """

    bucket_size_mb: float = DEFAULT_BUCKET_SIZE_MB
    bucketing_strategy: str = BucketingStrategy.SIZE_BASED.value
    overlap_communication: bool = True
    use_distributed_optimizer: bool = False
    alignment_padding: int = ALIGNMENT_PADDING
    bucket_cap_factor: float = 1.5
    dtype_bucketing: bool = True
    param_to_name_fn: Optional[Callable[[nn.Parameter], str]] = None

    def __post_init__(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        if not isinstance(self.bucket_size_mb, (int, float)):
            raise ValueError(
                f"bucket_size_mb must be a number, got "
                f"{type(self.bucket_size_mb).__name__}"
            )

        if not MIN_BUCKET_SIZE_MB <= self.bucket_size_mb <= MAX_BUCKET_SIZE_MB:
            raise ValueError(
                f"bucket_size_mb must be between {MIN_BUCKET_SIZE_MB} "
                f"and {MAX_BUCKET_SIZE_MB}, got {self.bucket_size_mb}"
            )

        valid_strategies = {s.value for s in BucketingStrategy}
        if self.bucketing_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid bucketing_strategy: '{self.bucketing_strategy}'. "
                f"Valid options are: {', '.join(sorted(valid_strategies))}"
            )

        if not isinstance(self.bucket_cap_factor, (int, float)):
            raise ValueError(
                f"bucket_cap_factor must be a number, got "
                f"{type(self.bucket_cap_factor).__name__}"
            )

        if self.bucket_cap_factor < 1.0:
            raise ValueError(
                f"bucket_cap_factor must be >= 1.0, got {self.bucket_cap_factor}"
            )

        if self.alignment_padding < 0:
            raise ValueError(
                f"alignment_padding must be non-negative, got {self.alignment_padding}"
            )


class GradientBucket:
    """
    Container for a group of parameters whose gradients are communicated together.

    This class manages a subset of model parameters and their gradients,
    providing efficient batched communication operations.

    Args:
        bucket_id: Unique identifier for this bucket
        params: List of parameters in this bucket
        numel: Total number of elements in bucket
        dtype: Data type of parameters in bucket
        device: Device where bucket resides
    """

    def __init__(
        self,
        bucket_id: int,
        params: List[nn.Parameter],
        numel: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.bucket_id = bucket_id
        self.params = params
        self.param_set = set(params)  # For fast membership checking
        self.numel = numel
        self.dtype = dtype
        self.device = device

        # Create gradient buffer for this bucket
        self.grad_buffer: Optional[torch.Tensor] = None
        self.param_to_buffer_offset: Dict[nn.Parameter, Tuple[int, int]] = {}

        # Track gradients ready for reduction
        self.params_with_grad: Set[nn.Parameter] = set()
        self.all_gradients_ready = False

        # Communication handle for async operations
        self.communication_handle: Optional[Any] = None

        # Initialize parameter offsets
        self._initialize_offsets()

    def _initialize_offsets(self) -> None:
        """Initialize mapping from parameters to buffer offsets."""
        offset = 0
        for param in self.params:
            param_numel = param.numel()
            self.param_to_buffer_offset[param] = (offset, offset + param_numel)
            offset += param_numel

    def allocate_grad_buffer(self) -> None:
        """Allocate gradient buffer for this bucket."""
        if self.grad_buffer is None:
            self.grad_buffer = torch.zeros(
                self.numel, dtype=self.dtype, device=self.device, requires_grad=False
            )

    def register_grad_ready(self, param: nn.Parameter) -> bool:
        """
        Register that a parameter's gradient is ready.

        Args:
            param: Parameter whose gradient is ready

        Returns:
            True if all gradients in bucket are now ready

        Raises:
            ValueError: If the parameter is not in this bucket.
        """
        if param not in self.param_set:
            param_info = (
                f"shape={param.shape}, dtype={param.dtype}"
                if param is not None
                else "None"
            )
            raise ValueError(
                f"Parameter ({param_info}) not found in bucket {self.bucket_id}. "
                f"This bucket contains {len(self.params)} parameters."
            )

        self.params_with_grad.add(param)

        # Check if all gradients are ready
        if len(self.params_with_grad) == len(self.params):
            self.all_gradients_ready = True
            return True

        return False

    def copy_gradients_to_buffer(self) -> None:
        """Copy parameter gradients to the bucket's gradient buffer.

        Note:
            This method allocates the gradient buffer if it doesn't exist.
            Parameters without gradients are skipped silently.
        """
        if self.grad_buffer is None:
            self.allocate_grad_buffer()

        if self.grad_buffer is not None:  # Check again after allocation
            for param in self.params:
                if param.grad is not None:
                    try:
                        start, end = self.param_to_buffer_offset[param]
                        grad_view = self.grad_buffer[start:end].view_as(param.grad)
                        grad_view.copy_(param.grad)
                    except RuntimeError as e:
                        raise RuntimeError(
                            f"Failed to copy gradient to buffer for parameter "
                            f"(shape={param.shape}, dtype={param.dtype}): {e}"
                        ) from e

    def copy_buffer_to_gradients(self) -> None:
        """Copy gradient buffer back to parameter gradients.

        Note:
            Creates new gradient tensors for parameters that don't have them.
            If the buffer is None, this method returns early without error.
        """
        if self.grad_buffer is None:
            logger.debug(f"Bucket {self.bucket_id} has no gradient buffer to copy from")
            return

        for param in self.params:
            try:
                start, end = self.param_to_buffer_offset[param]
                grad_view = self.grad_buffer[start:end].view_as(param)

                if param.grad is None:
                    param.grad = grad_view.clone()
                else:
                    param.grad.copy_(grad_view)
            except (KeyError, RuntimeError) as e:
                raise RuntimeError(
                    f"Failed to copy buffer to gradient for parameter "
                    f"(shape={param.shape}, dtype={param.dtype}): {e}"
                ) from e

    def reset(self) -> None:
        """Reset bucket state for next iteration."""
        self.params_with_grad.clear()
        self.all_gradients_ready = False
        self.communication_handle = None

        # Optionally clear gradient buffer
        if self.grad_buffer is not None:
            self.grad_buffer.zero_()


class GradientBucketManager:
    """
    Manages gradient bucketing and communication for distributed training.

    This class organizes model parameters into buckets and orchestrates
    efficient gradient communication with computation overlap.

    Args:
        model: PyTorch model
        config: Gradient bucketing configuration
        process_group: Process group for communication
    """

    def __init__(
        self,
        model: nn.Module,
        config: GradientBucketConfig,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        self.model = model
        self.config = config
        self.process_group = process_group or dist.group.WORLD

        # Get world size and rank
        self.world_size = dist.get_world_size(self.process_group)
        self.rank = dist.get_rank(self.process_group)

        # Storage for buckets
        self.buckets: List[GradientBucket] = []
        self.param_to_bucket: Dict[nn.Parameter, GradientBucket] = {}

        # Communication state
        self.pending_communications: List[Tuple[GradientBucket, Any]] = []

        # Create buckets based on model parameters
        self._create_buckets()

        # Register gradient hooks if overlapping is enabled
        if self.config.overlap_communication:
            self._register_gradient_hooks()

        logger.info(
            f"Created {len(self.buckets)} gradient buckets with "
            f"strategy {self.config.bucketing_strategy}"
        )

    def _create_buckets(self) -> None:
        """Create buckets based on configured strategy."""
        if self.config.bucketing_strategy == BucketingStrategy.SIZE_BASED:
            self._create_size_based_buckets()
        elif self.config.bucketing_strategy == BucketingStrategy.TYPE_BASED:
            self._create_type_based_buckets()
        elif self.config.bucketing_strategy == BucketingStrategy.LAYER_BASED:
            self._create_layer_based_buckets()
        else:  # HYBRID
            self._create_hybrid_buckets()

    def _create_size_based_buckets(self) -> None:
        """Create buckets based on size threshold."""
        # Convert MB to number of elements
        bucket_size_bytes = self.config.bucket_size_mb * 1024 * 1024

        # Group parameters by dtype if requested
        dtype_groups: Dict[torch.dtype, List[nn.Parameter]] = {}

        for param in self.model.parameters():
            if not param.requires_grad:
                continue

            dtype = param.dtype
            if self.config.dtype_bucketing:
                if dtype not in dtype_groups:
                    dtype_groups[dtype] = []
                dtype_groups[dtype].append(param)
            else:
                # Use a single group for all dtypes
                if torch.float32 not in dtype_groups:
                    dtype_groups[torch.float32] = []
                dtype_groups[torch.float32].append(param)

        # Create buckets for each dtype group
        bucket_id = 0
        for dtype, params in dtype_groups.items():
            # Sort parameters by size (largest first) for better packing
            params_sorted = sorted(params, key=lambda p: p.numel(), reverse=True)

            # Calculate elements per bucket based on dtype
            bytes_per_element = torch.tensor([], dtype=dtype).element_size()
            bucket_numel_threshold = bucket_size_bytes // bytes_per_element
            bucket_numel_cap = int(
                bucket_numel_threshold * self.config.bucket_cap_factor
            )

            current_bucket_params: List[nn.Parameter] = []
            current_bucket_numel = 0

            for param in params_sorted:
                param_numel = param.numel()

                # Check if adding this param would exceed bucket size
                if (
                    current_bucket_numel > 0
                    and current_bucket_numel + param_numel > bucket_numel_cap
                ):
                    # Create bucket with current parameters
                    self._add_bucket(
                        bucket_id,
                        current_bucket_params,
                        current_bucket_numel,
                        dtype,
                        param.device,
                    )
                    bucket_id += 1

                    # Start new bucket
                    current_bucket_params = [param]
                    current_bucket_numel = param_numel
                else:
                    # Add to current bucket
                    current_bucket_params.append(param)
                    current_bucket_numel += param_numel

            # Create final bucket if there are remaining parameters
            if current_bucket_params:
                self._add_bucket(
                    bucket_id,
                    current_bucket_params,
                    current_bucket_numel,
                    dtype,
                    current_bucket_params[0].device,
                )
                bucket_id += 1

    def _create_type_based_buckets(self) -> None:
        """Create buckets based on parameter types."""
        # Group parameters by their module type
        module_params: Dict[str, List[nn.Parameter]] = {}

        for name, module in self.model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue

                module_type = type(module).__name__
                if module_type not in module_params:
                    module_params[module_type] = []
                module_params[module_type].append(param)

        # Create buckets for each module type
        bucket_id = 0
        for module_type, params in module_params.items():
            if not params:
                continue

            # Calculate total size
            total_numel = sum(p.numel() for p in params)
            dtype = params[0].dtype  # Assume same dtype within module type
            device = params[0].device

            self._add_bucket(bucket_id, params, total_numel, dtype, device)
            bucket_id += 1

    def _create_layer_based_buckets(self) -> None:
        """Create buckets based on model layers."""
        # Simple layer-based bucketing - group consecutive parameters
        bucket_size_bytes = self.config.bucket_size_mb * 1024 * 1024

        bucket_id = 0
        current_bucket_params: List[nn.Parameter] = []
        current_bucket_numel = 0
        current_dtype: Optional[torch.dtype] = None
        current_device: Optional[torch.device] = None

        for param in self.model.parameters():
            if not param.requires_grad:
                continue

            param_numel = param.numel()
            param_bytes = param_numel * param.element_size()

            # Check if we need to start a new bucket
            need_new_bucket = (
                current_dtype is not None and param.dtype != current_dtype
            ) or (
                current_bucket_numel > 0
                and (current_bucket_numel * param.element_size()) + param_bytes
                > bucket_size_bytes
            )

            if (
                need_new_bucket
                and current_bucket_params
                and current_dtype is not None
                and current_device is not None
            ):
                # Create bucket with current parameters
                self._add_bucket(
                    bucket_id,
                    current_bucket_params,
                    current_bucket_numel,
                    current_dtype,
                    current_device,
                )
                bucket_id += 1

                # Start new bucket
                current_bucket_params = [param]
                current_bucket_numel = param_numel
                current_dtype = param.dtype
                current_device = param.device
            else:
                # Add to current bucket
                current_bucket_params.append(param)
                current_bucket_numel += param_numel
                if current_dtype is None:
                    current_dtype = param.dtype
                    current_device = param.device

        # Create final bucket
        if (
            current_bucket_params
            and current_dtype is not None
            and current_device is not None
        ):
            self._add_bucket(
                bucket_id,
                current_bucket_params,
                current_bucket_numel,
                current_dtype,
                current_device,
            )

    def _create_hybrid_buckets(self) -> None:
        """Create buckets using a hybrid strategy."""
        # Combine type-based and size-based strategies
        # First group by type, then apply size limits

        module_params: Dict[str, List[nn.Parameter]] = {}

        for name, module in self.model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue

                module_type = type(module).__name__
                if module_type not in module_params:
                    module_params[module_type] = []
                module_params[module_type].append(param)

        # Apply size-based bucketing within each module type
        bucket_size_bytes = self.config.bucket_size_mb * 1024 * 1024
        bucket_id = 0

        for module_type, params in module_params.items():
            if not params:
                continue

            # Sort by size for better packing
            params_sorted = sorted(params, key=lambda p: p.numel(), reverse=True)

            current_bucket_params: List[nn.Parameter] = []
            current_bucket_numel = 0
            dtype = params_sorted[0].dtype
            device = params_sorted[0].device

            bytes_per_element = torch.tensor([], dtype=dtype).element_size()
            bucket_numel_threshold = bucket_size_bytes // bytes_per_element

            for param in params_sorted:
                param_numel = param.numel()

                if (
                    current_bucket_numel > 0
                    and current_bucket_numel + param_numel > bucket_numel_threshold
                ):
                    # Create bucket
                    self._add_bucket(
                        bucket_id,
                        current_bucket_params,
                        current_bucket_numel,
                        dtype,
                        device,
                    )
                    bucket_id += 1

                    current_bucket_params = [param]
                    current_bucket_numel = param_numel
                else:
                    current_bucket_params.append(param)
                    current_bucket_numel += param_numel

            # Create final bucket for this module type
            if current_bucket_params:
                self._add_bucket(
                    bucket_id,
                    current_bucket_params,
                    current_bucket_numel,
                    dtype,
                    device,
                )
                bucket_id += 1

    def _add_bucket(
        self,
        bucket_id: int,
        params: List[nn.Parameter],
        numel: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Add a new bucket to the manager."""
        bucket = GradientBucket(bucket_id, params, numel, dtype, device)
        self.buckets.append(bucket)

        # Update parameter to bucket mapping
        for param in params:
            self.param_to_bucket[param] = bucket

        # Pre-allocate gradient buffer if not using overlap
        if not self.config.overlap_communication:
            bucket.allocate_grad_buffer()

    def _register_gradient_hooks(self) -> None:
        """Register hooks to trigger communication when gradients are ready."""
        for param in self.model.parameters():
            if not param.requires_grad:
                continue

            if param not in self.param_to_bucket:
                logger.warning(f"Parameter not in any bucket, skipping hook")
                continue

            def make_hook(p: nn.Parameter, bucket: GradientBucket):
                def hook(grad: torch.Tensor) -> torch.Tensor:
                    # Register gradient as ready
                    all_ready = bucket.register_grad_ready(p)

                    if all_ready and self.config.overlap_communication:
                        # Launch async communication for this bucket
                        self._launch_bucket_communication(bucket)

                    return grad

                return hook

            bucket = self.param_to_bucket[param]
            param.register_hook(make_hook(param, bucket))

    def _launch_bucket_communication(self, bucket: GradientBucket) -> None:
        """Launch asynchronous gradient communication for a bucket."""
        # Copy gradients to buffer
        bucket.copy_gradients_to_buffer()

        if bucket.grad_buffer is None:
            logger.error(f"Gradient buffer not allocated for bucket {bucket.bucket_id}")
            return

        # Scale gradients if using averaging
        if self.world_size > 1:
            bucket.grad_buffer.div_(self.world_size)

        # Launch appropriate communication
        if self.config.use_distributed_optimizer:
            # Reduce-scatter for distributed optimizer
            output_tensor = torch.empty_like(
                bucket.grad_buffer[: bucket.numel // self.world_size]
            )
            handle = dist.reduce_scatter(
                output_tensor,
                list(bucket.grad_buffer.chunk(self.world_size)),
                group=self.process_group,
                async_op=True,
            )
        else:
            # All-reduce for standard data parallel
            handle = dist.all_reduce(
                bucket.grad_buffer, group=self.process_group, async_op=True
            )

        bucket.communication_handle = handle
        self.pending_communications.append((bucket, handle))

    def synchronize_gradients(self) -> None:
        """
        Synchronize all gradients across processes.

        This method ensures all gradient communications are complete
        and copies synchronized gradients back to parameters.
        """
        if self.config.overlap_communication:
            # Wait for all pending communications
            for bucket, handle in self.pending_communications:
                handle.wait()

                # Copy buffer back to gradients if not using distributed optimizer
                if not self.config.use_distributed_optimizer:
                    bucket.copy_buffer_to_gradients()

            self.pending_communications.clear()
        else:
            # Synchronous communication for all buckets
            for bucket in self.buckets:
                # Copy gradients to buffer
                bucket.copy_gradients_to_buffer()

                if bucket.grad_buffer is None:
                    continue

                # Scale gradients
                if self.world_size > 1:
                    bucket.grad_buffer.div_(self.world_size)

                # Perform communication
                if self.config.use_distributed_optimizer:
                    # Reduce-scatter
                    output_tensor = torch.empty_like(
                        bucket.grad_buffer[: bucket.numel // self.world_size]
                    )
                    dist.reduce_scatter(
                        output_tensor,
                        list(bucket.grad_buffer.chunk(self.world_size)),
                        group=self.process_group,
                    )
                    # Store reduced portion back
                    bucket.grad_buffer[
                        : bucket.numel // self.world_size
                    ] = output_tensor
                else:
                    # All-reduce
                    dist.all_reduce(bucket.grad_buffer, group=self.process_group)

                    # Copy back to gradients
                    bucket.copy_buffer_to_gradients()

    def reset(self) -> None:
        """Reset all buckets for next iteration."""
        for bucket in self.buckets:
            bucket.reset()

        self.pending_communications.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about bucketing."""
        stats = {
            "num_buckets": len(self.buckets),
            "total_parameters": sum(len(b.params) for b in self.buckets),
            "total_numel": sum(b.numel for b in self.buckets),
            "bucket_sizes": [b.numel for b in self.buckets],
            "avg_bucket_size": (
                sum(b.numel for b in self.buckets) / len(self.buckets)
                if self.buckets
                else 0
            ),
            "strategy": self.config.bucketing_strategy,
            "overlap_enabled": self.config.overlap_communication,
        }

        # Add per-dtype statistics
        dtype_stats: Dict[str, int] = {}
        for bucket in self.buckets:
            dtype_name = str(bucket.dtype)
            if dtype_name not in dtype_stats:
                dtype_stats[dtype_name] = 0
            dtype_stats[dtype_name] += bucket.numel
        stats["dtype_distribution"] = dtype_stats

        return stats


def create_gradient_buckets(
    model: nn.Module,
    config: Optional[GradientBucketConfig] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> GradientBucketManager:
    """
    Create and configure gradient bucket manager for a model.

    This is the main entry point for setting up gradient bucketing in distributed
    training. It analyzes the model parameters and creates efficient communication
    buckets based on the specified configuration.

    Args:
        model: PyTorch model to bucket. Must have parameters that require gradients.
        config: Bucketing configuration (uses optimized defaults if None)
        process_group: Process group for communication (uses WORLD group if None)

    Returns:
        Configured GradientBucketManager instance ready for training

    Raises:
        ValueError: If the model has no parameters requiring gradients
        RuntimeError: If distributed is not initialized when needed

    Example:
        >>> # Basic usage with defaults
        >>> manager = create_gradient_buckets(model)
        >>>
        >>> # Custom configuration for large models
        >>> config = GradientBucketConfig(
        ...     bucket_size_mb=100,
        ...     bucketing_strategy="hybrid",
        ...     overlap_communication=True
        ... )
        >>> bucket_manager = create_gradient_buckets(model, config)
        >>>
        >>> # In training loop
        >>> loss.backward()
        >>> bucket_manager.synchronize_gradients()
        >>> optimizer.step()
        >>> bucket_manager.reset()
    """
    # Validate inputs
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected model to be nn.Module, got {type(model).__name__}")

    # Check if model has parameters requiring gradients
    params_requiring_grad = sum(1 for p in model.parameters() if p.requires_grad)
    if params_requiring_grad == 0:
        raise ValueError(
            "Model has no parameters requiring gradients. "
            "Gradient bucketing is not applicable."
        )

    # Use default config if not provided
    if config is None:
        config = GradientBucketConfig()
        logger.info("Using default gradient bucketing configuration")

    # Validate distributed setup if needed
    if config.overlap_communication or process_group is not None:
        if not dist.is_initialized():
            raise RuntimeError(
                "Distributed not initialized but communication overlap or "
                "process group was requested. Call "
                "torch.distributed.init_process_group() first."
            )

    # Create and return the manager
    manager = GradientBucketManager(model, config, process_group)

    # Log statistics
    stats = manager.get_statistics()
    logger.info(
        f"Created gradient bucket manager: {stats['num_buckets']} buckets, "
        f"{stats['total_parameters']} parameters, "
        f"strategy={stats['strategy']}"
    )

    return manager
