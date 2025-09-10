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
        # Validate basic numeric constraints
        if self.bucket_size <= 0:
            raise ValueError(f"bucket_size must be positive, got {self.bucket_size}")

        if self.bucket_size < 512:  # Minimum 512 bytes for testing
            raise ValueError(
                f"bucket_size too small, minimum 512 bytes required, "
                f"got {self.bucket_size}"
            )

        if self.bucket_size > 1024 * 1024 * 1024:  # Maximum 1GB
            raise ValueError(
                f"bucket_size too large, maximum 1GB allowed, "
                f"got {self.bucket_size}"
            )

        if self.max_buckets <= 0:
            raise ValueError(f"max_buckets must be positive, got {self.max_buckets}")

        if self.max_buckets > 32:  # Reasonable upper limit
            raise ValueError(
                f"max_buckets too large, maximum 32 allowed, " f"got {self.max_buckets}"
            )

        if self.overlap_threshold < 0:
            raise ValueError(
                f"overlap_threshold must be non-negative, "
                f"got {self.overlap_threshold}"
            )

        if self.overlap_threshold > 10.0:  # 10 seconds reasonable upper bound
            raise ValueError(
                f"overlap_threshold too large, maximum 10.0s allowed, "
                f"got {self.overlap_threshold}"
            )

        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be non-negative, got {self.warmup_steps}"
            )

        if self.warmup_steps > 1000:  # Reasonable upper limit
            raise ValueError(
                f"warmup_steps too large, maximum 1000 allowed, "
                f"got {self.warmup_steps}"
            )

        if self.buffer_growth_factor <= 1.0:
            raise ValueError(
                f"buffer_growth_factor must be greater than 1.0, "
                f"got {self.buffer_growth_factor}"
            )

        if self.buffer_growth_factor > 2.0:
            raise ValueError(
                f"buffer_growth_factor too large, maximum 2.0 allowed, "
                f"got {self.buffer_growth_factor}"
            )

        if self.max_buffer_size <= 0:
            raise ValueError(
                f"max_buffer_size must be positive, got {self.max_buffer_size}"
            )

        if self.max_buffer_size > 10 * 1024 * 1024 * 1024:  # 10GB limit
            raise ValueError(
                f"max_buffer_size too large, maximum 10GB allowed, "
                f"got {self.max_buffer_size}"
            )

        if self.async_op_timeout is not None:
            if self.async_op_timeout <= 0:
                raise ValueError(
                    f"async_op_timeout must be positive or None, "
                    f"got {self.async_op_timeout}"
                )
            if self.async_op_timeout > 3600:  # 1 hour limit
                raise ValueError(
                    f"async_op_timeout too large, maximum 3600s allowed, "
                    f"got {self.async_op_timeout}"
                )

        # Ensure bucket size doesn't exceed max buffer size
        if self.bucket_size > self.max_buffer_size:
            raise ValueError(f"bucket_size cannot exceed max_buffer_size")
        # Validate list parameters
        if self.priority_layers is not None:
            if not isinstance(self.priority_layers, list):
                raise TypeError(
                    f"priority_layers must be a list or None, "
                    f"got {type(self.priority_layers)}"
                )
            if not all(isinstance(layer, str) for layer in self.priority_layers):
                raise TypeError("All priority_layers elements must be strings")

        if self.skip_layers is not None:
            if not isinstance(self.skip_layers, list):
                raise TypeError(
                    f"skip_layers must be a list or None, "
                    f"got {type(self.skip_layers)}"
                )
            if not all(isinstance(layer, str) for layer in self.skip_layers):
                raise TypeError("All skip_layers elements must be strings")
        # Validate enum types
        if not isinstance(self.strategy, AsyncAllreduceStrategy):
            raise TypeError(
                f"strategy must be AsyncAllreduceStrategy, "
                f"got {type(self.strategy)}"
            )

        if not isinstance(self.bucketing_strategy, GradientBucketingStrategy):
            raise TypeError(
                f"bucketing_strategy must be GradientBucketingStrategy, "
                f"got {type(self.bucketing_strategy)}"
            )

    @classmethod
    def create_optimized_config(
        cls,
        world_size: int,
        model_size_mb: float,
        gpu_memory_gb: float = 16.0,
        target_overlap_ratio: float = 0.8,
    ) -> "AsyncAllreduceConfig":
        """
        Create optimized async allreduce config based on system parameters.

        Args:
            world_size: Number of processes in distributed training
            model_size_mb: Model size in megabytes
            gpu_memory_gb: Available GPU memory in gigabytes
            target_overlap_ratio: Target computation/communication overlap
                ratio (0.0-1.0)

        Returns:
            Optimized AsyncAllreduceConfig instance

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate inputs
        if not isinstance(world_size, int) or world_size < 1:
            raise ValueError(f"world_size must be a positive integer, got {world_size}")
        if not isinstance(model_size_mb, (int, float)) or model_size_mb <= 0:
            raise ValueError(f"model_size_mb must be positive, got {model_size_mb}")
        if not isinstance(gpu_memory_gb, (int, float)) or gpu_memory_gb <= 0:
            raise ValueError(f"gpu_memory_gb must be positive, got {gpu_memory_gb}")
        if not isinstance(target_overlap_ratio, (int, float)) or not (
            0 <= target_overlap_ratio <= 1
        ):
            raise ValueError(
                f"target_overlap_ratio must be between 0 and 1, "
                f"got {target_overlap_ratio}"
            )
        # Calculate optimal bucket size based on model size and world size
        # Target: each bucket contains roughly 1/8 of model gradients per process
        base_bucket_size = model_size_mb * 1024 * 1024 // (world_size * 8)

        # Apply constraints based on world size and memory
        if world_size <= 4:
            # Small world size: use larger buckets for efficiency
            optimal_bucket_size = min(
                base_bucket_size * 2, 50 * 1024 * 1024
            )  # Max 50MB
        elif world_size <= 16:
            # Medium world size: balanced approach
            optimal_bucket_size = min(base_bucket_size, 25 * 1024 * 1024)  # Max 25MB
        else:
            # Large world size: smaller buckets for better parallelization
            optimal_bucket_size = min(base_bucket_size, 10 * 1024 * 1024)  # Max 10MB

        # Ensure minimum bucket size
        optimal_bucket_size = max(int(optimal_bucket_size), 1024 * 1024)  # Min 1MB

        # Adjust max buckets based on available memory and world size
        memory_based_buckets = int(
            gpu_memory_gb * 1024 * 1024 * 1024 * 0.02 / optimal_bucket_size
        )  # 2% of GPU memory

        # Scale buckets with world size but cap appropriately
        if world_size <= 4:
            max_buckets = min(memory_based_buckets, 4)
        elif world_size <= 16:
            max_buckets = min(memory_based_buckets, 8)
        else:
            max_buckets = min(memory_based_buckets, 16)

        max_buckets = max(max_buckets, 2)  # Minimum 2 buckets

        # Set buffer size based on GPU memory (more conservative)
        max_buffer_size = int(
            gpu_memory_gb * 1024 * 1024 * 1024 * 0.03
        )  # 3% of GPU memory
        max_buffer_size = max(
            max_buffer_size, int(optimal_bucket_size * max_buckets * 1.2)
        )  # 20% safety margin

        # Determine strategy based on world size and target overlap
        if world_size <= 2:
            strategy = AsyncAllreduceStrategy.IMMEDIATE
        elif world_size >= 16 and target_overlap_ratio >= 0.9:
            strategy = AsyncAllreduceStrategy.PRIORITY_BASED
        else:
            strategy = AsyncAllreduceStrategy.BUCKETED

        # Calculate optimal warmup and threshold based on world size
        warmup_steps = min(
            max(10, world_size // 2), 50
        )  # Scale with world size, cap at 50
        overlap_threshold = (
            0.05 if world_size >= 8 else 0.1
        )  # Less threshold for more processes

        # Adjust timeout based on world size
        timeout = min(
            30.0 + world_size * 0.5, 120.0
        )  # Scale with world size, cap at 2 minutes

        config = cls(
            enabled=True,
            strategy=strategy,
            bucket_size=optimal_bucket_size,
            max_buckets=max_buckets,
            max_buffer_size=max_buffer_size,
            warmup_steps=warmup_steps,
            overlap_threshold=overlap_threshold,
            async_op_timeout=timeout,
            gradient_predivision=True,  # Enable for numerical stability
            enable_gradient_monitoring=world_size <= 8,  # Smaller setups
            log_communication_stats=world_size <= 16,  # Smaller setups
        )

        # Validate the generated config
        config.validate_for_world_size(world_size)

        return config

    def validate_for_world_size(self, world_size: int) -> None:
        """
        Validate configuration for a specific world size.

        Args:
            world_size: Number of processes in distributed training

        Raises:
            ValueError: If configuration is invalid for the given world size
        """
        if not isinstance(world_size, int) or world_size < 1:
            raise ValueError(f"world_size must be a positive integer, got {world_size}")

        if world_size > 1024:  # Reasonable upper limit
            raise ValueError(
                f"world_size too large, maximum 1024 supported, " f"got {world_size}"
            )

        if world_size <= 1 and self.enabled:
            raise ValueError(
                f"Async allreduce cannot be enabled with world_size <= 1, "
                f"got {world_size}"
            )

        # For small world sizes, ensure we don't over-optimize
        if world_size < 4 and self.max_buckets > world_size * 2:
            import warnings

            warnings.warn(
                f"max_buckets ({self.max_buckets}) is larger than "
                f"world_size ({world_size}). Consider reducing max_buckets to "
                f"{world_size} for better performance."
            )

        # For very large world sizes, ensure enough buckets for efficiency
        if world_size > 16 and self.max_buckets < 4:
            import warnings

            warnings.warn(
                f"max_buckets ({self.max_buckets}) may be too small for "
                f"world_size ({world_size}). Consider increasing max_buckets "
                f"for better parallelization."
            )

        # Validate bucket size relative to world size
        if self.bucket_size < world_size * 1024:  # At least 1KB per process
            import warnings

            warnings.warn(
                f"bucket_size ({self.bucket_size}) may be too small for "
                f"world_size ({world_size}). Consider increasing bucket_size "
                f"to at least {world_size * 1024} bytes."
            )
