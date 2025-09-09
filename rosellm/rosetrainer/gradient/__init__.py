"""Gradient finalization and synchronization module for RoseLLM.

Advanced gradient handling with multi-dimensional parallelism support.
"""

from .bucketing import (
    BucketingStrategy,
    GradientBucket,
    GradientBucketConfig,
    GradientBucketManager,
    create_gradient_buckets,
)
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
from .strategies import (
    BucketedGradientSync,
    GradientSyncStrategy,
    HierarchicalGradientSync,
    SimpleGradientSync,
)

__all__ = [
    # Gradient Bucketing
    "BucketingStrategy",
    "GradientBucket",
    "GradientBucketConfig",
    "GradientBucketManager",
    "create_gradient_buckets",
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
]
