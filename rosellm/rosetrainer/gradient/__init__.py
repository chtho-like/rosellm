"""Gradient finalization and synchronization module for RoseLLM.

Advanced gradient handling with multi-dimensional parallelism support.
"""

from .config import GradientFinalizationConfig
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
]
