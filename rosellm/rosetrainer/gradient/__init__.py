"""Gradient finalization and synchronization module for RoseLLM.

Advanced gradient handling with multi-dimensional parallelism support.
"""

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
    "GradientFinalizationConfig",
    "GradientFinalizer",
    "GradientSyncStrategy",
    "SimpleGradientSync",
    "BucketedGradientSync",
    "HierarchicalGradientSync",
    "DecoupledGradientBuffer",
    "DecoupledGradientConfig",
    "DecoupledGradientManager",
    "StorageMode",
]
