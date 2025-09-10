"""Configuration for gradient finalization and synchronization."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class GradientSyncStrategy(str, Enum):
    """Gradient synchronization strategies."""

    SIMPLE = "simple"  # Basic all-reduce
    BUCKETED = "bucketed"  # Bucketed all-reduce for efficiency
    HIERARCHICAL = "hierarchical"  # Hierarchical reduction across dimensions
    OVERLAPPED = "overlapped"  # Overlap comm with computation
    DELAYED = "delayed"  # Delay sync until optimizer step


class ReductionOp(str, Enum):
    """Reduction operations for gradient synchronization."""

    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"


class DimensionOrder(str, Enum):
    """Order of dimensions for gradient reduction."""

    TP_FIRST = "tp-pp-dp-cp-ep"  # Tensor parallel first
    DP_FIRST = "dp-tp-pp-cp-ep"  # Data parallel first
    HIERARCHICAL = "hierarchical"  # Hierarchical (TP -> PP -> DP+CP+EP)
    CUSTOM = "custom"  # Custom ordering


@dataclass
class GradientFinalizationConfig:
    """Configuration for gradient finalization and synchronization.

    This configuration controls how gradients are synchronized across
    different parallelism dimensions and process groups.

    Attributes:
        sync_strategy: Strategy for gradient synchronization
        reduction_op: Operation to use for gradient reduction
        dimension_order: Order in which to reduce across dimensions
        bucket_size_mb: Bucket size in MB for bucketed strategies
        num_buckets: Number of buckets for bucketed strategies
        overlap_grad_sync: Whether to overlap gradient sync with backward pass
        sync_grad_before_clip: Whether to sync before gradient clipping
        use_contiguous_buffers: Whether to use contiguous gradient buffers
        check_gradient_norm: Whether to check gradient norm before/after sync
        sync_batch_norm: Whether to sync batch norm statistics
        sync_layer_norm: Whether to sync layer norm statistics
        fp16_compression: Whether to compress gradients to FP16 for communication
        gradient_predivide_factor: Factor to predivide gradients before all-reduce
        gradient_postdivide_factor: Factor to postdivide gradients after all-reduce
        enable_async_grad_sync: Whether to enable asynchronous gradient sync
        hierarchical_levels: Levels for hierarchical reduction
        custom_dimension_order: Custom order for dimension reduction
        dtensor_enabled: Whether to use DTensor for PyTorch 2.0+
        virtual_pipeline_aware: Whether to handle virtual pipeline parallel ranks
        expert_parallel_sync_type: Type of sync for expert parallelism
        context_parallel_sync_type: Type of sync for context parallelism
        enable_gradient_stats: Whether to collect gradient statistics
        gradient_norm_type: Type of norm to use for gradient norm calculation
        verbose: Whether to enable verbose logging
        share_embeddings_and_output_weights: Whether input/output weights are tied
        share_position_embeddings: Whether position embeddings are shared
        embedding_reduce_group_size: Size of embedding reduction group
        position_embedding_reduce_group_size: Size of position embedding group
    """

    # Basic configuration
    sync_strategy: str = GradientSyncStrategy.BUCKETED.value
    reduction_op: str = ReductionOp.MEAN.value
    dimension_order: str = DimensionOrder.HIERARCHICAL.value

    # Bucketing configuration
    bucket_size_mb: float = 25.0
    num_buckets: Optional[int] = None
    bucket_cap_mb: float = 100.0

    # Synchronization control
    overlap_grad_sync: bool = True
    sync_grad_before_clip: bool = True
    use_contiguous_buffers: bool = True
    check_gradient_norm: bool = True
    sync_batch_norm: bool = True
    sync_layer_norm: bool = False

    # Communication optimization
    fp16_compression: bool = False
    gradient_predivide_factor: float = 1.0
    gradient_postdivide_factor: float = 1.0
    enable_async_grad_sync: bool = False

    # Hierarchical configuration
    hierarchical_levels: List[List[str]] = field(
        default_factory=lambda: [["tp"], ["pp"], ["dp", "cp", "ep"]]
    )
    custom_dimension_order: Optional[List[str]] = None

    # Advanced features
    dtensor_enabled: bool = False
    virtual_pipeline_aware: bool = True
    expert_parallel_sync_type: str = "all_to_all"  # or "all_reduce"
    context_parallel_sync_type: str = "ring"  # or "tree"

    # Statistics and debugging
    enable_gradient_stats: bool = False
    gradient_norm_type: float = 2.0
    verbose: bool = False

    # Timing configuration
    sync_timeout_seconds: float = 30.0
    enable_timing_stats: bool = False

    # Error recovery
    enable_error_recovery: bool = True
    max_recovery_attempts: int = 3

    # Performance optimization
    max_stats_history: int = 100
    enable_profiling: bool = False

    # Shared weight configuration (for tied embeddings)
    share_embeddings_and_output_weights: bool = False
    share_position_embeddings: bool = False
    embedding_reduce_group_size: Optional[int] = None
    position_embedding_reduce_group_size: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate sync strategy
        if self.sync_strategy not in [s.value for s in GradientSyncStrategy]:
            raise ValueError(
                f"Invalid sync_strategy: {self.sync_strategy}. "
                f"Must be one of {[s.value for s in GradientSyncStrategy]}"
            )

        # Validate reduction operation
        if self.reduction_op not in [op.value for op in ReductionOp]:
            raise ValueError(
                f"Invalid reduction_op: {self.reduction_op}. "
                f"Must be one of {[op.value for op in ReductionOp]}"
            )

        # Validate dimension order
        if self.dimension_order not in [d.value for d in DimensionOrder]:
            raise ValueError(
                f"Invalid dimension_order: {self.dimension_order}. "
                f"Must be one of {[d.value for d in DimensionOrder]}"
            )

        # Validate bucket size
        if self.bucket_size_mb <= 0:
            raise ValueError(
                f"bucket_size_mb must be positive, got {self.bucket_size_mb}"
            )

        if self.bucket_cap_mb < self.bucket_size_mb:
            raise ValueError(
                f"bucket_cap_mb ({self.bucket_cap_mb}) must be >= "
                f"bucket_size_mb ({self.bucket_size_mb})"
            )

        # Validate gradient norm type
        if self.gradient_norm_type <= 0 and self.gradient_norm_type not in [
            float("inf"),
            float("-inf"),
        ]:
            raise ValueError(
                "gradient_norm_type must be positive or inf, "
                f"got {self.gradient_norm_type}"
            )

        # Validate max_stats_history
        if self.max_stats_history <= 0:
            raise ValueError(
                f"max_stats_history must be positive, got {self.max_stats_history}"
            )

        # Validate max_recovery_attempts
        if self.max_recovery_attempts < 0:
            raise ValueError(
                "max_recovery_attempts must be non-negative, "
                f"got {self.max_recovery_attempts}"
            )

        # Validate hierarchical levels
        if self.dimension_order == DimensionOrder.HIERARCHICAL.value:
            if not self.hierarchical_levels:
                raise ValueError(
                    "hierarchical_levels must be specified for hierarchical ordering"
                )

            valid_dims = {"tp", "pp", "dp", "cp", "ep"}
            all_dims = set()
            for level in self.hierarchical_levels:
                for dim in level:
                    if dim not in valid_dims:
                        raise ValueError(
                            f"Invalid dimension in hierarchical_levels: {dim}"
                        )
                    if dim in all_dims:
                        raise ValueError(
                            f"Dimension {dim} appears multiple times in "
                            "hierarchical_levels"
                        )
                    all_dims.add(dim)

        # Validate custom dimension order
        if self.dimension_order == DimensionOrder.CUSTOM.value:
            if not self.custom_dimension_order:
                raise ValueError(
                    "custom_dimension_order must be specified for custom ordering"
                )

            valid_dims = {"tp", "pp", "dp", "cp", "ep"}
            for dim in self.custom_dimension_order:
                if dim not in valid_dims:
                    raise ValueError(
                        f"Invalid dimension in custom_dimension_order: {dim}"
                    )

        # Validate timeout
        if self.sync_timeout_seconds <= 0:
            raise ValueError(
                "sync_timeout_seconds must be positive, "
                f"got {self.sync_timeout_seconds}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration.
        """
        return {
            "sync_strategy": self.sync_strategy,
            "reduction_op": self.reduction_op,
            "dimension_order": self.dimension_order,
            "bucket_size_mb": self.bucket_size_mb,
            "num_buckets": self.num_buckets,
            "bucket_cap_mb": self.bucket_cap_mb,
            "overlap_grad_sync": self.overlap_grad_sync,
            "sync_grad_before_clip": self.sync_grad_before_clip,
            "use_contiguous_buffers": self.use_contiguous_buffers,
            "check_gradient_norm": self.check_gradient_norm,
            "sync_batch_norm": self.sync_batch_norm,
            "sync_layer_norm": self.sync_layer_norm,
            "fp16_compression": self.fp16_compression,
            "gradient_predivide_factor": self.gradient_predivide_factor,
            "gradient_postdivide_factor": self.gradient_postdivide_factor,
            "enable_async_grad_sync": self.enable_async_grad_sync,
            "hierarchical_levels": self.hierarchical_levels,
            "custom_dimension_order": self.custom_dimension_order,
            "dtensor_enabled": self.dtensor_enabled,
            "virtual_pipeline_aware": self.virtual_pipeline_aware,
            "expert_parallel_sync_type": self.expert_parallel_sync_type,
            "context_parallel_sync_type": self.context_parallel_sync_type,
            "enable_gradient_stats": self.enable_gradient_stats,
            "gradient_norm_type": self.gradient_norm_type,
            "verbose": self.verbose,
            "sync_timeout_seconds": self.sync_timeout_seconds,
            "enable_timing_stats": self.enable_timing_stats,
            "share_embeddings_and_output_weights": (
                self.share_embeddings_and_output_weights
            ),
            "share_position_embeddings": self.share_position_embeddings,
            "embedding_reduce_group_size": self.embedding_reduce_group_size,
            "position_embedding_reduce_group_size": (
                self.position_embedding_reduce_group_size
            ),
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GradientFinalizationConfig":
        """Create configuration from dictionary.

        Args:
            config_dict: Dictionary with configuration values.

        Returns:
            GradientFinalizationConfig instance.
        """
        return cls(**config_dict)
