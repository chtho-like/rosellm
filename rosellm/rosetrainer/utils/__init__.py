"""
RoseTrainer Utilities Package

This package provides utility functions for gradient handling, optimization, and other
common operations in distributed training.
"""

from .gradient_scaler import CustomGradientScaler
from .gradient_utils import (
    GradientClipConfig,
    apply_gradient_clipping,
    calculate_gradient_norm_multitensor,
    check_for_inf_and_nan_with_scaler,
    check_gradient_finite,
    get_gradient_stats,
    gradient_accumulation_context,
    sync_gradients,
)
from .timer_config import TimerAggregation, TimerConfig, TimerLogLevel
from .timers import Timer, Timers, get_timers, log_timers, reset_timers, set_timers

__all__ = [
    "GradientClipConfig",
    "apply_gradient_clipping",
    "calculate_gradient_norm_multitensor",
    "check_gradient_finite",
    "check_for_inf_and_nan_with_scaler",
    "get_gradient_stats",
    "gradient_accumulation_context",
    "sync_gradients",
    "CustomGradientScaler",
    "Timer",
    "Timers",
    "TimerConfig",
    "TimerLogLevel",
    "TimerAggregation",
    "get_timers",
    "set_timers",
    "reset_timers",
    "log_timers",
]
