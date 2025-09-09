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
    # Decoupled Gradients
    "DecoupledGradientBuffer",
    "DecoupledGradientConfig",
    "DecoupledGradientManager",
    "StorageMode",
]
