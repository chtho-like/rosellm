"""Distributed optimizer with parameter partitioning for memory efficiency.

This package provides a comprehensive distributed optimizer implementation with:
- Parameter, gradient, and optimizer state partitioning
- Mixed precision training support
- Memory profiling and optimization
- Factory pattern for easy configuration
- Parameter-gradient buffer mapping with bucket-based reduction
"""

from .config import DistributedOptimizerConfig, PartitioningStrategy
from .distributed_optimizer import DistributedOptimizer
from .factory import OptimizerFactory
from .memory_profiler import MemoryProfiler, MemoryStats
from .param_grad_mapping import (
    MappingConfig,
    MultiTensorOperator,
    ParameterInfo,
    ParameterType,
    ParamGradMapping,
    ParamGradMappingBuilder,
    ReductionStrategy,
)
from .param_range import ParameterPartitioner, ParameterRange

__all__ = [
    # Core classes
    "DistributedOptimizer",
    "DistributedOptimizerConfig",
    "ParameterPartitioner",
    "ParameterRange",
    # Parameter-gradient mapping
    "ParamGradMapping",
    "ParamGradMappingBuilder",
    "MappingConfig",
    "ParameterInfo",
    "ParameterType",
    "ReductionStrategy",
    "MultiTensorOperator",
    # Factory and utilities
    "OptimizerFactory",
    "MemoryProfiler",
    "MemoryStats",
    # Enums
    "PartitioningStrategy",
]
