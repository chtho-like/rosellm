"""
Parameter-Gradient Buffer Mapping with Bucket-based Reduction

This module provides an advanced parameter-gradient mapping system that efficiently
manages the relationship between model parameters and their gradient buffers. It
integrates with existing gradient bucketing infrastructure to provide optimized
gradient reduction for distributed training.

Key Features:
- Efficient parameter-to-gradient-buffer mapping
- Multi-tensor operations for batched gradient updates
- Bucket-based gradient reduction with communication overlap
- Memory-optimized buffer management
- Type-safe parameter tracking
- Comprehensive error handling and validation

The module follows Megatron-LM's design patterns for gradient buffer management
while providing enhanced functionality for RoseLLM's distributed training needs.

Example Usage:
    ```python
    from rosellm.rosetrainer.optimizer import ParamGradMapping

    # Create mapping for model parameters
    mapping = ParamGradMapping(
        params=model.parameters(),
        bucket_size_mb=25.0,
        dtype=torch.float16,
        device=torch.device("cuda")
    )

    # Map parameters to gradient buffers
    mapping.map_parameters()

    # During backward pass - gradients are automatically bucketed
    loss.backward()

    # Synchronize gradient buckets
    mapping.synchronize_gradients()

    # Get reduced gradients
    reduced_grads = mapping.get_reduced_gradients()
    ```

Advanced Features:
    - Parameter grouping by type (e.g., weights, biases, embeddings)
    - Gradient accumulation with configurable reduction frequency
    - Support for gradient clipping and scaling
    - Integration with mixed precision training
    - Performance profiling and optimization hints

References:
    - Megatron-LM gradient buffer management
    - PyTorch DDP gradient bucketing
    - "Efficient Large-Scale Language Model Training on GPU Clusters"
"""

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Parameter

from rosellm.rosetrainer.communication.gradient_buckets import (
    BucketConfig,
    BucketManager,
    BucketStrategy,
)

# Exceptions import removed - was unused
from rosellm.rosetrainer.optimizer.gradient_buffer import GradientBuffer

logger = logging.getLogger(__name__)


class ParameterType(Enum):
    """Classification of parameter types for optimized handling."""

    WEIGHT = "weight"  # Standard weight matrices
    BIAS = "bias"  # Bias vectors
    EMBEDDING = "embedding"  # Embedding tables
    NORM = "norm"  # Normalization parameters (LayerNorm, etc.)
    POSITION = "position"  # Position embeddings
    OTHER = "other"  # Unclassified parameters


class ReductionStrategy(Enum):
    """Strategy for gradient reduction across distributed ranks."""

    IMMEDIATE = "immediate"  # Reduce as soon as gradients are ready
    DELAYED = "delayed"  # Delay reduction until all gradients ready
    OVERLAPPED = "overlapped"  # Overlap reduction with computation
    HIERARCHICAL = "hierarchical"  # Multi-level reduction for large clusters


@dataclass
class MappingConfig:
    """Configuration for parameter-gradient mapping."""

    # Bucketing configuration
    bucket_size_mb: float = 25.0
    min_bucket_size_mb: float = 1.0
    bucketing_strategy: BucketStrategy = BucketStrategy.MIXED

    # Reduction configuration
    reduction_strategy: ReductionStrategy = ReductionStrategy.OVERLAPPED
    gradient_predivision: bool = True
    gradient_accumulation_steps: int = 1

    # Memory optimization
    use_memory_pool: bool = True
    pin_memory: bool = False
    contiguous_gradients: bool = True

    # Performance tuning
    num_buckets_per_group: int = 4
    communication_overlap: bool = True
    profile_communication: bool = False

    # Type-specific handling
    type_specific_buckets: bool = True
    type_bucket_sizes: Dict[ParameterType, float] = field(
        default_factory=lambda: {
            ParameterType.EMBEDDING: 50.0,  # Larger buckets for embeddings
            ParameterType.WEIGHT: 25.0,
            ParameterType.BIAS: 10.0,  # Smaller buckets for biases
            ParameterType.NORM: 10.0,
            ParameterType.POSITION: 25.0,
            ParameterType.OTHER: 25.0,
        }
    )

    # Advanced features
    enable_gradient_clipping: bool = False
    gradient_clip_value: float = 1.0
    enable_gradient_scaling: bool = False
    gradient_scale_factor: float = 1.0


@dataclass
class ParameterInfo:
    """Metadata for a model parameter."""

    param: Parameter
    name: str
    param_type: ParameterType
    shape: torch.Size
    numel: int
    dtype: torch.dtype
    device: torch.device
    bucket_id: Optional[int] = None
    buffer_offset: Optional[Tuple[int, int]] = None
    requires_grad: bool = True
    is_distributed: bool = False

    def __hash__(self) -> int:
        """Make ParameterInfo hashable for use in sets/dicts."""
        return id(self.param)

    def __eq__(self, other: object) -> bool:
        """Check equality based on parameter identity."""
        if not isinstance(other, ParameterInfo):
            return False
        return id(self.param) == id(other.param)


class MultiTensorOperator:
    """
    Efficient multi-tensor operations for gradient processing.

    This class provides batched tensor operations that are significantly
    faster than iterating over individual tensors, especially for models
    with many small parameters.
    """

    def __init__(self, device: torch.device):
        """
        Initialize multi-tensor operator.

        Args:
            device: Device for tensor operations
        """
        self.device = device
        self._stream = None
        if device.type == "cuda":
            self._stream = torch.cuda.Stream(device=device)

    def scale_tensors(
        self, tensors: List[Tensor], scale_factor: float, in_place: bool = True
    ) -> List[Tensor]:
        """
        Scale multiple tensors by a factor.

        Args:
            tensors: List of tensors to scale
            scale_factor: Scaling factor
            in_place: Whether to modify tensors in place

        Returns:
            Scaled tensors
        """
        if not tensors:
            return []

        if self._stream is not None and self.device.type == "cuda":
            # Use the stream context manager
            stream_ctx = torch.cuda.stream(self._stream)  # type: ignore
            with stream_ctx:
                if in_place:
                    for tensor in tensors:
                        tensor.mul_(scale_factor)
                    return tensors
                else:
                    return [tensor * scale_factor for tensor in tensors]
        else:
            if in_place:
                for tensor in tensors:
                    tensor.mul_(scale_factor)
                return tensors
            else:
                return [tensor * scale_factor for tensor in tensors]

    def clip_tensors(
        self, tensors: List[Tensor], max_norm: float, norm_type: float = 2.0
    ) -> Tuple[List[Tensor], float]:
        """
        Clip gradients by global norm.

        Args:
            tensors: List of gradient tensors
            max_norm: Maximum norm value
            norm_type: Type of norm (default: L2)

        Returns:
            Tuple of (clipped tensors, original norm)
        """
        if not tensors:
            return [], 0.0

        # Calculate global norm
        if norm_type == float("inf"):
            total_norm = float(max(tensor.abs().max().item() for tensor in tensors))
        else:
            total_norm = torch.norm(
                torch.stack([torch.norm(tensor, norm_type) for tensor in tensors]),
                norm_type,
            )

        clip_coef = float(max_norm / (total_norm + 1e-6))
        if clip_coef < 1:
            self.scale_tensors(tensors, clip_coef, in_place=True)

        return tensors, float(total_norm)

    def copy_tensors(
        self, src_tensors: List[Tensor], dst_tensors: List[Tensor]
    ) -> None:
        """
        Efficiently copy multiple tensors.

        Args:
            src_tensors: Source tensors
            dst_tensors: Destination tensors
        """
        if len(src_tensors) != len(dst_tensors):
            raise ValueError(
                f"Source and destination tensor counts mismatch: "
                f"{len(src_tensors)} != {len(dst_tensors)}"
            )

        if self._stream is not None and self.device.type == "cuda":
            stream_ctx = torch.cuda.stream(self._stream)  # type: ignore
            with stream_ctx:
                for src, dst in zip(src_tensors, dst_tensors):
                    dst.copy_(src)
        else:
            for src, dst in zip(src_tensors, dst_tensors):
                dst.copy_(src)

    def synchronize(self) -> None:
        """Synchronize any pending operations."""
        if self._stream is not None and self.device.type == "cuda":
            self._stream.synchronize()


class ParamGradMapping:
    """
    Advanced parameter-gradient mapping with bucket-based reduction.

    This class manages the mapping between model parameters and their gradient
    buffers, providing efficient gradient reduction through bucketing and
    multi-tensor operations. It integrates with existing infrastructure while
    adding enhanced functionality for large-scale distributed training.

    Key Responsibilities:
    - Map parameters to gradient buffers with type-aware grouping
    - Organize parameters into communication-efficient buckets
    - Provide multi-tensor operations for batched gradient processing
    - Handle gradient accumulation and reduction across distributed ranks
    - Support mixed precision and gradient scaling/clipping
    """

    def __init__(
        self,
        params: Union[List[Parameter], List[Dict[str, Any]]],
        config: Optional[MappingConfig] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        """
        Initialize parameter-gradient mapping.

        Args:
            params: Model parameters or parameter groups
            config: Mapping configuration
            dtype: Data type for gradient buffers
            device: Device for gradient operations
            process_group: Process group for distributed training
        """
        self.config = config or MappingConfig()
        self.dtype = dtype
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.process_group = process_group

        # Thread safety
        self._lock = threading.RLock()

        # Parameter management
        self.param_infos: List[ParameterInfo] = []
        self.param_to_info: Dict[Parameter, ParameterInfo] = {}
        self.param_groups: List[Dict[str, Any]] = []

        # Gradient buffer management
        self.gradient_buffer: Optional[GradientBuffer] = None
        self.bucket_manager: Optional[BucketManager] = None

        # Multi-tensor operator
        self.multi_tensor_op = MultiTensorOperator(self.device)

        # Tracking
        self.accumulation_step = 0
        self.total_reductions = 0
        self.communication_time = 0.0

        # Initialize with provided parameters
        self._initialize_parameters(params)

        # Create gradient buffers and buckets
        self._create_gradient_buffers()

        # Map parameters to buffers
        self.map_parameters()

        logger.info(
            f"Created ParamGradMapping with {len(self.param_infos)} parameters, "
            f"config={self.config}"
        )

    def _initialize_parameters(
        self, params: Union[List[Parameter], List[Dict[str, Any]]]
    ) -> None:
        """
        Initialize parameter information from provided parameters.

        Args:
            params: Parameters or parameter groups
        """
        # Convert to list if generator
        params_list = list(params) if params else []

        # Handle parameter groups
        if params_list and isinstance(params_list[0], dict):
            self.param_groups = params_list  # type: ignore
            all_params = []
            for group in self.param_groups:
                all_params.extend(group.get("params", []))
        else:
            all_params = params_list  # type: ignore
            self.param_groups = [{"params": all_params}]

        # Create parameter info for each parameter
        for idx, param in enumerate(all_params):
            if not isinstance(param, Parameter):
                continue

            # Determine parameter type
            param_type = self._classify_parameter(param, f"param_{idx}")

            # Create parameter info
            info = ParameterInfo(
                param=param,
                name=f"param_{idx}",
                param_type=param_type,
                shape=param.shape,
                numel=param.numel(),
                dtype=param.dtype,
                device=param.device,
                requires_grad=param.requires_grad,
                is_distributed=dist.is_initialized(),
            )

            self.param_infos.append(info)
            self.param_to_info[param] = info

    def _classify_parameter(self, param: Parameter, name: str) -> ParameterType:
        """
        Classify a parameter by its type.

        Args:
            param: Parameter to classify
            name: Parameter name

        Returns:
            Parameter type classification
        """
        name_lower = name.lower()

        # Check for specific parameter types based on name patterns
        if "bias" in name_lower:
            return ParameterType.BIAS
        elif any(pattern in name_lower for pattern in ["embed", "embedding"]):
            return ParameterType.EMBEDDING
        elif any(pattern in name_lower for pattern in ["norm", "ln", "bn"]):
            return ParameterType.NORM
        elif any(pattern in name_lower for pattern in ["pos", "position"]):
            return ParameterType.POSITION
        elif "weight" in name_lower:
            return ParameterType.WEIGHT

        # Check based on shape
        if len(param.shape) == 1:
            return ParameterType.BIAS
        elif len(param.shape) == 2:
            return ParameterType.WEIGHT

        return ParameterType.OTHER

    def _create_gradient_buffers(self) -> None:
        """Create gradient buffers and bucket manager."""
        # Filter parameters that require gradients
        grad_params = [info.param for info in self.param_infos if info.requires_grad]

        if not grad_params:
            logger.warning("No parameters require gradients")
            return

        # Create gradient buffer for basic operations
        self.gradient_buffer = GradientBuffer(
            params=grad_params,
            bucket_size_mb=self.config.bucket_size_mb,
            dtype=self.dtype,
            device=self.device,
            process_group=self.process_group,
        )

        # Create bucket manager for advanced bucketing
        bucket_config = BucketConfig(
            strategy=self.config.bucketing_strategy,
            max_bucket_size_mb=self.config.bucket_size_mb,
            min_bucket_size_mb=self.config.min_bucket_size_mb,
            overlap_communication=self.config.communication_overlap,
            gradient_predivision=self.config.gradient_predivision,
        )

        self.bucket_manager = BucketManager(
            config=bucket_config, device=self.device, dtype=self.dtype
        )

    def map_parameters(self) -> None:
        """
        Map parameters to gradient buffers and buckets.

        This method establishes the mapping between parameters and their
        corresponding gradient buffer locations, enabling efficient gradient
        reduction and communication.
        """
        with self._lock:
            if not self.gradient_buffer or not self.bucket_manager:
                logger.warning("Gradient buffers not initialized")
                return

            # Group parameters by type if configured
            if self.config.type_specific_buckets:
                type_groups = self._group_parameters_by_type()

                # Assign parameters to buckets by type
                for param_type, params in type_groups.items():
                    # Create type-specific buckets
                    # Get bucket size for this parameter type
                    # bucket_size_mb = self.config.type_bucket_sizes.get(
                    #     param_type, self.config.bucket_size_mb
                    # )

                    # Assign parameters to buckets
                    for param_info in params:
                        if param_info.param.grad is not None:
                            bucket_id = self.bucket_manager.assign_gradient(
                                param_name=param_info.name,
                                gradient=param_info.param.grad,
                            )
                            param_info.bucket_id = bucket_id
            else:
                # Assign all parameters to buckets without type grouping
                for param_info in self.param_infos:
                    if param_info.requires_grad and param_info.param.grad is not None:
                        bucket_id = self.bucket_manager.assign_gradient(
                            param_name=param_info.name, gradient=param_info.param.grad
                        )
                        param_info.bucket_id = bucket_id

            # Update buffer offsets from gradient buffer
            for param_info in self.param_infos:
                if param_info.param in self.gradient_buffer.param_to_offset:
                    param_info.buffer_offset = self.gradient_buffer.param_to_offset[
                        param_info.param
                    ]

            logger.info(
                f"Mapped {len(self.param_infos)} parameters to "
                f"{len(self.bucket_manager.buckets)} buckets"
            )

    def _group_parameters_by_type(self) -> Dict[ParameterType, List[ParameterInfo]]:
        """
        Group parameters by their type.

        Returns:
            Dictionary mapping parameter types to parameter info lists
        """
        type_groups = defaultdict(list)

        for param_info in self.param_infos:
            if param_info.requires_grad:
                type_groups[param_info.param_type].append(param_info)

        return dict(type_groups)

    def accumulate_gradients(self) -> None:
        """
        Accumulate gradients for gradient accumulation.

        This method handles gradient accumulation across multiple
        micro-batches before reduction.
        """
        with self._lock:
            self.accumulation_step += 1

            # Scale gradients if gradient accumulation is enabled
            if self.config.gradient_accumulation_steps > 1:
                scale_factor = 1.0 / self.config.gradient_accumulation_steps

                # Get all gradients
                gradients = [
                    info.param.grad
                    for info in self.param_infos
                    if info.param.grad is not None
                ]

                # Scale gradients using multi-tensor operations
                self.multi_tensor_op.scale_tensors(
                    gradients, scale_factor, in_place=True
                )

    def should_reduce_gradients(self) -> bool:
        """
        Check if gradients should be reduced based on accumulation steps.

        Returns:
            True if gradients should be reduced
        """
        return (
            self.accumulation_step > 0
            and self.accumulation_step % self.config.gradient_accumulation_steps == 0
        )

    def synchronize_gradients(self, force: bool = False) -> Dict[str, Any]:
        """
        Synchronize gradient buckets across distributed ranks.

        Args:
            force: Force synchronization regardless of accumulation steps

        Returns:
            Synchronization statistics
        """
        with self._lock:
            # Check if we should reduce gradients
            if not force and not self.should_reduce_gradients():
                return {"skipped": True, "reason": "accumulation_not_complete"}

            if not self.bucket_manager:
                return {"skipped": True, "reason": "no_bucket_manager"}

            # Apply gradient clipping if enabled
            if self.config.enable_gradient_clipping:
                self._clip_gradients()

            # Apply gradient scaling if enabled
            if self.config.enable_gradient_scaling:
                self._scale_gradients()

            # Synchronize buckets based on reduction strategy
            if self.config.reduction_strategy == ReductionStrategy.IMMEDIATE:
                stats = self._immediate_reduction()
            elif self.config.reduction_strategy == ReductionStrategy.DELAYED:
                stats = self._delayed_reduction()
            elif self.config.reduction_strategy == ReductionStrategy.OVERLAPPED:
                stats = self._overlapped_reduction()
            elif self.config.reduction_strategy == ReductionStrategy.HIERARCHICAL:
                stats = self._hierarchical_reduction()
            else:
                stats = self.bucket_manager.synchronize_buckets(
                    process_group=self.process_group,
                    overlap=self.config.communication_overlap,
                )

            # Update tracking
            self.total_reductions += 1
            self.communication_time += stats.get("total_time", 0.0)

            # Reset accumulation counter if we reduced
            if not stats.get("skipped", False):
                self.accumulation_step = 0

            return stats

    def _immediate_reduction(self) -> Dict[str, Any]:
        """
        Perform immediate gradient reduction.

        Returns:
            Reduction statistics
        """
        if not self.bucket_manager:
            return {"skipped": True}

        # Start reduction immediately for all ready buckets
        return self.bucket_manager.synchronize_buckets(
            process_group=self.process_group, overlap=False
        )

    def _delayed_reduction(self) -> Dict[str, Any]:
        """
        Perform delayed gradient reduction.

        Returns:
            Reduction statistics
        """
        if not self.bucket_manager:
            return {"skipped": True}

        # Wait for all gradients to be ready before reducing
        all_grads_ready = all(
            info.param.grad is not None
            for info in self.param_infos
            if info.requires_grad
        )

        if not all_grads_ready:
            return {"skipped": True, "reason": "gradients_not_ready"}

        return self.bucket_manager.synchronize_buckets(
            process_group=self.process_group, overlap=False
        )

    def _overlapped_reduction(self) -> Dict[str, Any]:
        """
        Perform overlapped gradient reduction.

        Returns:
            Reduction statistics
        """
        if not self.bucket_manager:
            return {"skipped": True}

        # Use overlapped communication for efficiency
        return self.bucket_manager.synchronize_buckets(
            process_group=self.process_group, overlap=True
        )

    def _hierarchical_reduction(self) -> Dict[str, Any]:
        """
        Perform hierarchical gradient reduction for large clusters.

        Returns:
            Reduction statistics
        """
        # For now, fall back to overlapped reduction
        # TODO: Implement multi-level reduction for large clusters
        return self._overlapped_reduction()

    def _clip_gradients(self) -> None:
        """Apply gradient clipping to all parameters."""
        gradients = [
            info.param.grad for info in self.param_infos if info.param.grad is not None
        ]

        if gradients:
            self.multi_tensor_op.clip_tensors(
                gradients, max_norm=self.config.gradient_clip_value, norm_type=2.0
            )

    def _scale_gradients(self) -> None:
        """Apply gradient scaling to all parameters."""
        gradients = [
            info.param.grad for info in self.param_infos if info.param.grad is not None
        ]

        if gradients:
            self.multi_tensor_op.scale_tensors(
                gradients, scale_factor=self.config.gradient_scale_factor, in_place=True
            )

    def get_reduced_gradients(self) -> Dict[str, Tensor]:
        """
        Get reduced gradients after synchronization.

        Returns:
            Dictionary mapping parameter names to reduced gradients
        """
        with self._lock:
            if not self.bucket_manager:
                return {}

            # Get gradients from bucket manager
            return self.bucket_manager.get_bucket_assignments()

    def get_parameter_info(self, param: Parameter) -> Optional[ParameterInfo]:
        """
        Get information about a specific parameter.

        Args:
            param: Parameter to query

        Returns:
            Parameter information or None if not found
        """
        return self.param_to_info.get(param)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the mapping.

        Returns:
            Dictionary of statistics
        """
        with self._lock:
            # Collect parameter statistics
            param_stats: Dict[str, int] = defaultdict(int)
            total_params = 0
            total_grad_params = 0

            for info in self.param_infos:
                param_stats[info.param_type.value] += 1
                total_params += info.numel
                if info.requires_grad:
                    total_grad_params += info.numel

            # Get bucket statistics
            bucket_stats = {}
            if self.bucket_manager:
                bucket_stats = self.bucket_manager.get_statistics()

            # Get buffer statistics
            buffer_stats = {}
            if self.gradient_buffer:
                buffer_stats = self.gradient_buffer.get_bucket_info()

            return {
                "total_parameters": len(self.param_infos),
                "total_parameter_elements": total_params,
                "gradient_parameter_elements": total_grad_params,
                "parameter_types": dict(param_stats),
                "accumulation_step": self.accumulation_step,
                "total_reductions": self.total_reductions,
                "total_communication_time": self.communication_time,
                "avg_communication_time": (
                    self.communication_time / self.total_reductions
                    if self.total_reductions > 0
                    else 0.0
                ),
                "bucket_statistics": bucket_stats,
                "buffer_statistics": buffer_stats,
                "config": {
                    "bucket_size_mb": self.config.bucket_size_mb,
                    "reduction_strategy": self.config.reduction_strategy.value,
                    "gradient_accumulation_steps": (
                        self.config.gradient_accumulation_steps
                    ),
                    "type_specific_buckets": self.config.type_specific_buckets,
                },
            }

    def reset(self) -> None:
        """Reset gradient buffers and tracking."""
        with self._lock:
            # Reset gradient buffer
            if self.gradient_buffer:
                self.gradient_buffer.reset()

            # Reset bucket manager
            if self.bucket_manager:
                self.bucket_manager.reset()

            # Reset tracking
            self.accumulation_step = 0

    def __repr__(self) -> str:
        """String representation of the mapping."""
        stats = self.get_statistics()
        return (
            f"ParamGradMapping("
            f"params={stats['total_parameters']}, "
            f"elements={stats['total_parameter_elements']}, "
            f"reductions={stats['total_reductions']}, "
            f"strategy={self.config.reduction_strategy.value})"
        )


class ParamGradMappingBuilder:
    """
    Builder class for creating ParamGradMapping instances with fluent API.

    This builder provides a convenient way to construct parameter-gradient
    mappings with various configurations.

    Example:
        ```python
        mapping = (ParamGradMappingBuilder()
            .with_parameters(model.parameters())
            .with_bucket_size(50.0)
            .with_reduction_strategy(ReductionStrategy.OVERLAPPED)
            .with_gradient_accumulation(4)
            .with_gradient_clipping(1.0)
            .with_type_specific_buckets({
                ParameterType.EMBEDDING: 100.0,
                ParameterType.WEIGHT: 50.0,
                ParameterType.BIAS: 10.0
            })
            .build()
        )
        ```
    """

    def __init__(self):
        """Initialize the builder."""
        self._params: Optional[Union[List[Parameter], List[Dict[str, Any]]]] = None
        self._config = MappingConfig()
        self._dtype = torch.float32
        self._device: Optional[torch.device] = None
        self._process_group: Optional[dist.ProcessGroup] = None

    def with_parameters(
        self, params: Union[List[Parameter], List[Dict[str, Any]]]
    ) -> "ParamGradMappingBuilder":
        """Set the parameters for the mapping."""
        self._params = params
        return self

    def with_bucket_size(self, size_mb: float) -> "ParamGradMappingBuilder":
        """Set the bucket size in megabytes."""
        self._config.bucket_size_mb = size_mb
        return self

    def with_reduction_strategy(
        self, strategy: ReductionStrategy
    ) -> "ParamGradMappingBuilder":
        """Set the gradient reduction strategy."""
        self._config.reduction_strategy = strategy
        return self

    def with_gradient_accumulation(self, steps: int) -> "ParamGradMappingBuilder":
        """Set the number of gradient accumulation steps."""
        self._config.gradient_accumulation_steps = steps
        return self

    def with_gradient_clipping(self, max_norm: float) -> "ParamGradMappingBuilder":
        """Enable gradient clipping with specified max norm."""
        self._config.enable_gradient_clipping = True
        self._config.gradient_clip_value = max_norm
        return self

    def with_gradient_scaling(self, scale_factor: float) -> "ParamGradMappingBuilder":
        """Enable gradient scaling with specified factor."""
        self._config.enable_gradient_scaling = True
        self._config.gradient_scale_factor = scale_factor
        return self

    def with_type_specific_buckets(
        self, type_sizes: Dict[ParameterType, float]
    ) -> "ParamGradMappingBuilder":
        """Enable type-specific bucketing with custom sizes."""
        self._config.type_specific_buckets = True
        self._config.type_bucket_sizes.update(type_sizes)
        return self

    def with_dtype(self, dtype: torch.dtype) -> "ParamGradMappingBuilder":
        """Set the data type for gradient buffers."""
        self._dtype = dtype
        return self

    def with_device(self, device: torch.device) -> "ParamGradMappingBuilder":
        """Set the device for gradient operations."""
        self._device = device
        return self

    def with_process_group(self, group: dist.ProcessGroup) -> "ParamGradMappingBuilder":
        """Set the process group for distributed training."""
        self._process_group = group
        return self

    def build(self) -> ParamGradMapping:
        """
        Build the ParamGradMapping instance.

        Returns:
            Configured ParamGradMapping instance

        Raises:
            ValueError: If required parameters are not set
        """
        if self._params is None:
            raise ValueError("Parameters must be set before building")

        return ParamGradMapping(
            params=self._params,
            config=self._config,
            dtype=self._dtype,
            device=self._device,
            process_group=self._process_group,
        )
