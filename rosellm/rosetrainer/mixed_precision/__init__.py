"""
Mixed Precision Training Utilities for RoseLLM

This module provides advanced mixed precision training utilities including:
- Dynamic gradient scaling with multi-tensor operations
- Comprehensive mixed precision management
- APEX integration for enhanced performance
- Overflow detection and handling
- Comprehensive monitoring and logging
"""

from .dynamic_scaler import DynamicGradScaler as EnhancedDynamicGradScaler
from .dynamic_scaler import (
    DynamicScalerConfig,
    MultiTensorOverflowDetector,
    create_dynamic_scaler,
    get_recommended_config,
    is_apex_available,
)
from .gradient_scaler import (
    AbstractGradScaler,
    ConstantGradScaler,
)
from .gradient_scaler import (
    DynamicGradScaler as LegacyDynamicGradScaler,  # Legacy version
)
from .gradient_scaler import (
    GradScalerConfig,
    check_for_inf_and_nan,
)
from .mixed_precision import (
    MixedPrecisionConfig,
    MixedPrecisionManager,
    PrecisionType,
    create_mixed_precision_manager,
    get_precision_context,
)

# Default to enhanced version for new code
DynamicGradScaler = EnhancedDynamicGradScaler

__all__ = [
    # Enhanced dynamic scaling
    "DynamicGradScaler",
    "EnhancedDynamicGradScaler",
    "DynamicScalerConfig",
    "MultiTensorOverflowDetector",
    "create_dynamic_scaler",
    "get_recommended_config",
    "is_apex_available",
    # Mixed precision management
    "MixedPrecisionManager",
    "MixedPrecisionConfig",
    "PrecisionType",
    "create_mixed_precision_manager",
    "get_precision_context",
    # Legacy gradient scalers
    "AbstractGradScaler",
    "ConstantGradScaler",
    "LegacyDynamicGradScaler",
    "GradScalerConfig",
    "check_for_inf_and_nan",
]
