"""
RoseTrainer Utilities Package

This package provides utility functions for gradient handling, optimization, and other
common operations in distributed training.
"""

from .gradient_utils import (
    GradientClipConfig,
    apply_gradient_clipping,
    calculate_gradient_norm_multitensor,
    check_gradient_finite,
    get_gradient_stats,
    gradient_accumulation_context,
    sync_gradients,
)

__all__ = [
    "GradientClipConfig",
    "apply_gradient_clipping",
    "calculate_gradient_norm_multitensor",
    "check_gradient_finite",
    "get_gradient_stats",
    "gradient_accumulation_context",
    "sync_gradients",
]
