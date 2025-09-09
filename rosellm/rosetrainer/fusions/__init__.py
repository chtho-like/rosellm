"""
Fused Operations for RoseTrainer

This module provides optimized fused operations that combine multiple
computations into single kernel launches for improved performance.
"""

from .fused_layer_norm import (
    FusedLayerNorm,
    InvalidConfigurationError,
    KernelNotAvailableError,
    LayerNormConfig,
    LayerNormException,
    LayerNormKernelType,
)

__all__ = [
    "FusedLayerNorm",
    "LayerNormConfig",
    "LayerNormKernelType",
    "LayerNormException",
    "KernelNotAvailableError",
    "InvalidConfigurationError",
]
