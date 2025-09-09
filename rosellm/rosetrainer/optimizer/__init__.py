"""
RoseLLM Distributed Optimizer Module

This module provides optimized distributed optimizers with gradient bucketing
and communication-computation overlap for efficient large-scale training.
"""

from dataclasses import dataclass
from typing import List

from rosellm.rosetrainer.optimizer.distributed_optimizer import DistributedOptimizer
from rosellm.rosetrainer.optimizer.exceptions import (
    CommunicationError,
    ConfigurationError,
    GradientBufferError,
    OptimizerError,
    PartitioningError,
    SynchronizationError,
)
from rosellm.rosetrainer.optimizer.gradient_buffer import Bucket, GradientBuffer
from rosellm.rosetrainer.optimizer.metrics import OptimizerMetrics, PerformanceMonitor
from rosellm.rosetrainer.optimizer.optimizer_utils import (
    async_all_reduce_buckets,
    compute_bucket_assignment,
    create_parameter_buckets,
    estimate_memory_savings,
    flatten_dense_tensors,
    get_optimizer_memory_usage,
    partition_parameters_by_size,
    partition_parameters_round_robin,
    synchronize_bucket_gradients,
    unflatten_dense_tensors,
    validate_bucket_configuration,
)
from rosellm.rosetrainer.optimizer.partitioning_strategies import (
    LayerWisePartitioning,
    PartitioningStrategy,
    PartitioningStrategyFactory,
    RoundRobinPartitioning,
    SizeBalancedPartitioning,
)


# Configuration classes
@dataclass
class DistributedOptimizerConfig:
    """Configuration for distributed optimizer."""

    bucket_size_mb: float = 25.0
    overlap_grad_reduce: bool = True
    partition_optimizer_states: bool = True


@dataclass
class ParameterRange:
    """Range of parameters for partitioning."""

    start_idx: int
    end_idx: int


class ParameterPartitioner:
    """Base class for parameter partitioning strategies."""

    pass


__all__ = [
    # Main classes
    "DistributedOptimizer",
    "DistributedOptimizerConfig",
    "ParameterPartitioner",
    "ParameterRange",
    "GradientBuffer",
    "Bucket",
    # Partitioning strategies
    "PartitioningStrategy",
    "PartitioningStrategyFactory",
    "RoundRobinPartitioning",
    "SizeBalancedPartitioning",
    "LayerWisePartitioning",
    # Exceptions
    "OptimizerError",
    "GradientBufferError",
    "PartitioningError",
    "CommunicationError",
    "ConfigurationError",
    "SynchronizationError",
    # Metrics
    "OptimizerMetrics",
    "PerformanceMonitor",
    # Utility functions
    "create_parameter_buckets",
    "partition_parameters_round_robin",
    "partition_parameters_by_size",
    "compute_bucket_assignment",
    "flatten_dense_tensors",
    "unflatten_dense_tensors",
    "async_all_reduce_buckets",
    "synchronize_bucket_gradients",
    "estimate_memory_savings",
    "get_optimizer_memory_usage",
    "validate_bucket_configuration",
]
