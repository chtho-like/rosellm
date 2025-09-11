"""Distributed optimizer with parameter partitioning for memory efficiency.

This package provides a comprehensive distributed optimizer implementation with:
- Parameter, gradient, and optimizer state partitioning
- Mixed precision training support
- Memory profiling and optimization
- Factory pattern for easy configuration
- Parameter-gradient buffer mapping with bucket-based reduction
- Gradient bucket coalescing for optimized communication
"""

# Import standard PyTorch optimizers for convenience
from torch.optim import SGD, Adagrad, Adam, AdamW, RMSprop

from .chained_optimizer import ChainedOptimizer
from .coalesced_gradient_buffer import CoalescedBucket, CoalescedGradientBuffer
from .config import DistributedOptimizerConfig, PartitioningStrategy
from .distributed_optimizer import DistributedOptimizer
from .factory import OptimizerFactory
from .gradient_buffer import Bucket, GradientBuffer
from .memory_profiler import MemoryProfiler, MemoryStats
from .multi_tensor_adam import (
    MultiTensorAdam,
    MultiTensorAdamConfig,
    OverflowAction,
    WeightDecayMode,
    create_multi_tensor_adam,
    create_multi_tensor_adamw,
)
from .optimizer_factory import OptimFactory
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
from .range_aware_gradient_buffer import (
    GradientReductionStats,
    RangeAwareBucket,
    RangeAwareGradientBuffer,
)
from .range_buffer_mapping import (
    BufferAllocationMode,
    BufferRange,
    RangeBufferConfig,
    RangeBufferMapper,
    RangeBufferStrategy,
)
from .range_multi_tensor_ops import (
    RangeMultiTensorOperator,
    create_range_multi_tensor_operator,
    multi_tensor_range_scale,
)

__all__ = [
    # Core classes
    "DistributedOptimizer",
    "DistributedOptimizerConfig",
    "ParameterPartitioner",
    "ParameterRange",
    # Gradient buffer classes
    "GradientBuffer",
    "Bucket",
    "CoalescedGradientBuffer",
    "CoalescedBucket",
    # Range-based buffer mapping
    "RangeBufferConfig",
    "RangeBufferMapper",
    "RangeBufferStrategy",
    "BufferAllocationMode",
    "BufferRange",
    # Range-aware gradient buffer
    "RangeAwareGradientBuffer",
    "RangeAwareBucket",
    "GradientReductionStats",
    # Range multi-tensor operations
    "RangeMultiTensorOperator",
    "create_range_multi_tensor_operator",
    "multi_tensor_range_scale",
    # Multi-Tensor Adam Optimizer
    "MultiTensorAdam",
    "MultiTensorAdamConfig",
    "WeightDecayMode",
    "OverflowAction",
    "create_multi_tensor_adam",
    "create_multi_tensor_adamw",
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
    "OptimFactory",
    "MemoryProfiler",
    # Enums
    "PartitioningStrategy",
    # ChainedOptimizer
    "ChainedOptimizer",
    # Standard PyTorch optimizers
    "Adam",
    "AdamW",
    "SGD",
    "RMSprop",
    "Adagrad",
]
