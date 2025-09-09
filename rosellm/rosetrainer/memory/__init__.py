"""
Memory Optimization Module for RoseTrainer

Provides memory optimization techniques:
- Activation checkpointing
- Selective activation recomputation
- CPU offloading
- Mixed precision training
- Memory profiling
- Parameter and gradient buffering with bucketing
- Parameter gathering overlap with computation
"""

from .activation_checkpoint import ActivationCheckpointing, MemoryProfiler
from .cpu_offload import CPUOffloadOptimizer, ParameterOffloader
from .param_grad_buffer import (
    BucketConfig,
    BufferManager,
    GradientBucket,
    ParamAndGradBuffer,
)
from .parameter_overlap import (
    AsyncParameterGatherer,
    GatherRequest,
    OverlapConfig,
    OverlapMode,
    OverlappedLinear,
    ParameterCache,
    PipelineOverlapScheduler,
    StreamPool,
)
from .selective_recompute import (
    LayerProfile,
    LayerProfiler,
    SelectionStrategy,
    SelectiveCheckpointConfig,
    SelectiveCheckpointFunction,
    SelectiveRecomputeManager,
    create_selective_checkpoint_wrapper,
    selective_checkpoint,
)

# Conditional imports to avoid circular dependencies
try:
    from .mixed_precision import (
        DynamicLossScaler,
        MixedPrecisionManager,
        PrecisionType,
        check_overflow,
        convert_model_to_bf16,
        convert_model_to_fp16,
    )
except ImportError:
    # Mixed precision imports failed, likely due to circular import
    # These will be available after full module initialization
    pass

__all__ = [
    # Activation Checkpointing
    "ActivationCheckpointing",
    "MemoryProfiler",
    # Selective Recomputation
    "SelectiveCheckpointConfig",
    "SelectiveRecomputeManager",
    "SelectionStrategy",
    "LayerProfile",
    "LayerProfiler",
    "SelectiveCheckpointFunction",
    "selective_checkpoint",
    "create_selective_checkpoint_wrapper",
    # CPU Offloading
    "CPUOffloadOptimizer",
    "ParameterOffloader",
    # Parameter and Gradient Buffering
    "ParamAndGradBuffer",
    "GradientBucket",
    "BufferManager",
    "BucketConfig",
    # Parameter Overlap
    "AsyncParameterGatherer",
    "GatherRequest",
    "OverlapConfig",
    "OverlapMode",
    "OverlappedLinear",
    "ParameterCache",
    "PipelineOverlapScheduler",
    "StreamPool",
    # Mixed Precision (conditionally available)
    "MixedPrecisionManager",
    "PrecisionType",
    "DynamicLossScaler",
    "check_overflow",
    "convert_model_to_fp16",
    "convert_model_to_bf16",
]
