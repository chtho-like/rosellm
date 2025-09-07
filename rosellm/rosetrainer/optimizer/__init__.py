"""
RoseLLM Distributed Optimizer Module

This module provides optimized distributed optimizers with gradient bucketing
and communication-computation overlap for efficient large-scale training.
"""

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

__all__ = [
    # Main classes
    "DistributedOptimizer",
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
