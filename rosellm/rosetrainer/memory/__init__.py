"""
Memory Optimization Module for RoseTrainer

Provides memory optimization techniques:
- Activation checkpointing
- CPU offloading
- Mixed precision training
- Memory profiling
"""

from .activation_checkpoint import ActivationCheckpointing, MemoryProfiler
from .cpu_offload import CPUOffloadOptimizer, ParameterOffloader
from .mixed_precision import (DynamicLossScaler, MixedPrecisionManager,
                              PrecisionType, check_overflow,
                              convert_model_to_bf16, convert_model_to_fp16)

__all__ = [
    # Activation Checkpointing
    "ActivationCheckpointing",
    "MemoryProfiler",
    # CPU Offloading
    "CPUOffloadOptimizer",
    "ParameterOffloader",
    # Mixed Precision
    "MixedPrecisionManager",
    "PrecisionType",
    "DynamicLossScaler",
    "check_overflow",
    "convert_model_to_fp16",
    "convert_model_to_bf16",
]
