"""Gradient finalization and synchronization module for RoseLLM.

Advanced gradient handling with multi-dimensional parallelism support.
"""

from .accumulation_fusion import (
    AccumulationState,
    AsyncReductionOrchestrator,
    FusedParamGradMapping,
    FusionConfig,
    FusionMetrics,
    FusionStrategy,
    GradientAccumulationFusion,
    GradientFusionBuffer,
    OverlapStrategy,
)
from .bucketing import (
    BucketingStrategy,
    GradientBucket,
    GradientBucketConfig,
    GradientBucketManager,
    create_gradient_buckets,
)
from .clip_grads import ClipType, GradientClipper, clip_grad_norm, clip_grad_value
from .config import GradientFinalizationConfig
from .decoupled_grad import (
    DecoupledGradientBuffer,
    DecoupledGradientConfig,
    DecoupledGradientManager,
    StorageMode,
)
from .finalization import (
    AdvancedGradientFinalizer,
    GradientDataType,
    GradientDataTypeManager,
    create_gradient_data_type_manager,
    finalize_gradients_advanced,
)
from .finalizer import GradientFinalizer
from .shared_weight_reducer import SharedWeightConfig, SharedWeightGradientReducer
from .strategies import (
    BucketedGradientSync,
    GradientSyncStrategy,
    HierarchicalGradientSync,
    SimpleGradientSync,
)

__all__ = [
    # Gradient Accumulation Fusion
    "AccumulationState",
    "AsyncReductionOrchestrator",
    "FusedParamGradMapping",
    "FusionConfig",
    "FusionMetrics",
    "FusionStrategy",
    "GradientAccumulationFusion",
    "GradientFusionBuffer",
    "OverlapStrategy",
    # Gradient Bucketing
    "BucketingStrategy",
    "GradientBucket",
    "GradientBucketConfig",
    "GradientBucketManager",
    "create_gradient_buckets",
    # Gradient Clipping
    "ClipType",
    "GradientClipper",
    "clip_grad_norm",
    "clip_grad_value",
    # Gradient Finalization
    "GradientFinalizationConfig",
    "GradientFinalizer",
    "GradientSyncStrategy",
    "SimpleGradientSync",
    "BucketedGradientSync",
    "HierarchicalGradientSync",
    # Advanced Gradient Finalization
    "AdvancedGradientFinalizer",
    "GradientDataType",
    "GradientDataTypeManager",
    "create_gradient_data_type_manager",
    "finalize_gradients_advanced",
    # Decoupled Gradients
    "DecoupledGradientBuffer",
    "DecoupledGradientConfig",
    "DecoupledGradientManager",
    "StorageMode",
    # Shared Weight Gradient Reduction
    "SharedWeightConfig",
    "SharedWeightGradientReducer",
]
