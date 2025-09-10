"""
Async Gradient Allreduce Configuration

This module provides configuration classes for asynchronous gradient allreduce
operations, enabling overlapped computation and communication in distributed training.

The async gradient allreduce technique reduces training time by overlapping
backward pass gradient computation with gradient communication across processes.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class AsyncAllreduceStrategy(Enum):
    """Strategy for async gradient allreduce operations."""

    BUCKETED = "bucketed"  # Group gradients into buckets for efficient communication
    IMMEDIATE = "immediate"  # Start allreduce immediately when gradient is ready
    LAYERWISE = "layerwise"  # Allreduce gradients layer by layer
    PRIORITY_BASED = "priority"  # Prioritize critical layers first


class GradientBucketingStrategy(Enum):
    """Strategy for gradient bucketing in async allreduce."""

    SIZE_BASED = "size_based"  # Bucket by gradient tensor size
    LAYER_BASED = "layer_based"  # Bucket by model layer
    MIXED = "mixed"  # Combine size and layer based strategies


@dataclass
class AsyncAllreduceConfig:
    """
    Configuration for asynchronous gradient allreduce operations.

    This configuration enables fine-tuning of async gradient communication
    to optimize training throughput and convergence.
    """

    # Core async allreduce settings
    enabled: bool = True
    """Enable asynchronous gradient allreduce."""

    strategy: AsyncAllreduceStrategy = AsyncAllreduceStrategy.BUCKETED
    """Strategy for async gradient allreduce operations."""

    # Gradient bucketing configuration
    bucket_size: int = 25 * 1024 * 1024  # 25MB default bucket size
    """Maximum size of gradient buckets in bytes."""

    max_buckets: int = 4
    """Maximum number of gradient buckets to use."""

    bucketing_strategy: GradientBucketingStrategy = GradientBucketingStrategy.SIZE_BASED
    """Strategy for organizing gradients into buckets."""

    # Communication optimization
    overlap_threshold: float = 0.1
    """Minimum computation time (seconds) to overlap with communication."""

    warmup_steps: int = 10
    """Number of warmup steps before enabling async allreduce."""

    # Buffer management
    buffer_growth_factor: float = 1.25
    """Growth factor for communication buffers when resizing."""

    max_buffer_size: int = 100 * 1024 * 1024  # 100MB
    """Maximum buffer size per process in bytes."""

    # Advanced async settings
    enable_async_param_sync: bool = True
    """Enable asynchronous parameter synchronization after allreduce."""

    async_op_timeout: Optional[float] = 30.0
    """Timeout for async operations in seconds. None for no timeout."""

    gradient_predivision: bool = True
    """Divide gradients by world size before allreduce for numerical stability."""

    # Debugging and monitoring
    enable_gradient_monitoring: bool = False
    """Enable detailed gradient monitoring and statistics."""

    log_communication_stats: bool = False
    """Log communication timing and throughput statistics."""

    # Layer-wise configuration
    priority_layers: Optional[List[str]] = None
    """List of layer names to prioritize in async allreduce."""

    skip_layers: Optional[List[str]] = None
    """List of layer names to skip in async allreduce."""

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        if self.bucket_size <= 0:
            raise ValueError("bucket_size must be positive")

        if self.max_buckets <= 0:
            raise ValueError("max_buckets must be positive")

        if self.overlap_threshold < 0:
            raise ValueError("overlap_threshold must be non-negative")

        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")

        if self.buffer_growth_factor <= 1.0:
            raise ValueError("buffer_growth_factor must be greater than 1.0")

        if self.max_buffer_size <= 0:
            raise ValueError("max_buffer_size must be positive")

        if self.async_op_timeout is not None and self.async_op_timeout <= 0:
            raise ValueError("async_op_timeout must be positive or None")

        # Ensure bucket size doesn't exceed max buffer size
        if self.bucket_size > self.max_buffer_size:
            raise ValueError("bucket_size cannot exceed max_buffer_size")

    @classmethod
    def create_optimized_config(
        cls,
        world_size: int,
        model_size_mb: float,
        gpu_memory_gb: float = 16.0,
    ) -> "AsyncAllreduceConfig":
        """
        Create an optimized async allreduce configuration based on system parameters.

        Args:
            world_size: Number of processes in distributed training
            model_size_mb: Model size in megabytes
            gpu_memory_gb: Available GPU memory in gigabytes

        Returns:
            Optimized AsyncAllreduceConfig instance
        """
        # Calculate optimal bucket size based on model size and world size
        optimal_bucket_size = int(
            min(
                model_size_mb
                * 1024
                * 1024
                // (world_size * 4),  # 1/4 model per process
                25 * 1024 * 1024,  # Cap at 25MB
            )
        )
        optimal_bucket_size = max(optimal_bucket_size, 1024 * 1024)  # Min 1MB

        # Adjust max buckets based on available memory
        max_buckets = min(
            int(
                gpu_memory_gb * 1024 * 1024 * 1024 * 0.02 / optimal_bucket_size
            ),  # 2% of GPU memory
            8,
        )
        max_buckets = max(max_buckets, 2)  # Minimum 2 buckets

        # Set buffer size based on GPU memory
        max_buffer_size = int(
            gpu_memory_gb * 1024 * 1024 * 1024 * 0.05
        )  # 5% of GPU memory

        return cls(
            enabled=True,
            strategy=AsyncAllreduceStrategy.BUCKETED,
            bucket_size=optimal_bucket_size,
            max_buckets=max_buckets,
            max_buffer_size=max_buffer_size,
            warmup_steps=max(10, world_size),  # More warmup for larger world sizes
            overlap_threshold=0.05
            if world_size >= 8
            else 0.1,  # Less threshold for more processes
        )

    def validate_for_world_size(self, world_size: int) -> None:
        """
        Validate configuration for a specific world size.

        Args:
            world_size: Number of processes in distributed training

        Raises:
            ValueError: If configuration is invalid for the given world size
        """
        if world_size <= 1 and self.enabled:
            raise ValueError("Async allreduce cannot be enabled with world_size <= 1")

        # For small world sizes, ensure we don't over-optimize
        if world_size < 4 and self.max_buckets > world_size:
            import warnings

            warnings.warn(
                f"max_buckets ({self.max_buckets}) is larger than "
                f"world_size ({world_size}). Consider reducing max_buckets "
                f"for better performance."
            )
