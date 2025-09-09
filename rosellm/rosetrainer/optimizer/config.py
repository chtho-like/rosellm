"""Configuration for distributed optimizer with parameter partitioning.

Provides comprehensive configuration options for memory-efficient distributed
training with parameter, gradient, and optimizer state partitioning.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import torch


class PartitioningStrategy(Enum):
    """Strategy for partitioning parameters across ranks."""

    NONE = "none"  # No partitioning
    GREEDY = "greedy"  # Greedy assignment (default)
    BALANCED = "balanced"  # Balanced load distribution
    MEMORY_AWARE = "memory_aware"  # Consider memory constraints


@dataclass
class DistributedOptimizerConfig:
    """Configuration for distributed optimizer with parameter partitioning.

    This configuration controls how parameters and optimizer states are
    partitioned across data parallel ranks for memory efficiency.

    Attributes:
        partition_parameters: Whether to partition parameters across DP ranks.
        partition_gradients: Whether to partition gradients
            (always True if partitioning).
        partition_optimizer_states: Whether to partition optimizer states.
        contiguous_gradients: Use contiguous gradient buffer for efficiency.
        overlap_grad_reduce: Overlap gradient reduction with backward pass.
        cpu_offload: Offload optimizer states to CPU memory.
        mixed_precision: Use mixed precision with FP32 main parameters.
        grad_clip_value: Maximum gradient norm for clipping.
        use_multi_tensor_apply: Use multi-tensor operations for efficiency.
        bucket_size_mb: Size of gradient reduction buckets in MB.
        allgather_bucket_size_mb: Size of parameter allgather buckets in MB.
        reduce_bucket_size_mb: Size of gradient reduce buckets in MB.
        param_group_configs: Per-parameter-group optimizer configs.
        verbose: Enable verbose logging for debugging.
        profile_memory: Profile memory usage during optimization.
        check_gradients: Check for NaN/Inf gradients before optimization.
        gradient_predivide_factor: Factor to predivide gradients by.
        gradient_postdivide_factor: Factor to postdivide gradients by.
        dtype: Data type for computations (default: torch.float32).
        grad_scaler_config: Configuration for gradient scaler in mixed precision.
    """

    # Core partitioning settings
    partition_parameters: bool = True
    partition_gradients: bool = True
    partition_optimizer_states: bool = True
    partitioning_strategy: PartitioningStrategy = PartitioningStrategy.GREEDY

    # Memory optimization
    contiguous_gradients: bool = True
    overlap_grad_reduce: bool = False
    cpu_offload: bool = False
    mixed_precision: bool = False

    # Gradient handling
    grad_clip_value: Optional[float] = None
    use_multi_tensor_apply: bool = True

    # Communication settings
    bucket_size_mb: int = 25
    allgather_bucket_size_mb: int = 25
    reduce_bucket_size_mb: int = 25

    # Per-parameter-group configs
    param_group_configs: Optional[List[Dict[str, Any]]] = None

    # Debugging and profiling
    verbose: bool = False
    profile_memory: bool = False
    check_gradients: bool = False

    # Gradient scaling
    gradient_predivide_factor: float = 1.0
    gradient_postdivide_factor: float = 1.0

    # Data type
    dtype: torch.dtype = torch.float32

    # Gradient scaler config for mixed precision
    grad_scaler_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "init_scale": 2**16,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "enabled": True,
            "min_scale": 1.0,
            "max_scale": 2**24,
        }
    )

    # Advanced options
    gradient_accumulation_steps: int = 1
    delay_allgather: bool = False  # Delay parameter allgather until needed
    use_hierarchical_allreduce: bool = False  # Use hierarchical communication
    memory_efficient_fp16: bool = True  # Store FP16 grads, compute in FP32

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Dependency validations
        if self.partition_parameters:
            # If partitioning parameters, must partition gradients
            self.partition_gradients = True

        if self.overlap_grad_reduce and not self.contiguous_gradients:
            raise ValueError("overlap_grad_reduce requires contiguous_gradients=True")

        if self.cpu_offload and not self.partition_optimizer_states:
            raise ValueError("cpu_offload requires partition_optimizer_states=True")

        if self.delay_allgather and not self.partition_parameters:
            raise ValueError("delay_allgather requires partition_parameters=True")

        # Mixed precision configuration
        if self.mixed_precision:
            if self.dtype not in (torch.float16, torch.bfloat16):
                # Mixed precision requires half precision dtype
                self.dtype = torch.float16

            # Validate gradient scaler config
            if self.grad_scaler_config.get("min_scale", 1.0) <= 0:
                raise ValueError("grad_scaler_config.min_scale must be positive")
            max_scale = self.grad_scaler_config.get("max_scale", 2**24)
            min_scale = self.grad_scaler_config.get("min_scale", 1.0)
            if max_scale < min_scale:
                raise ValueError("grad_scaler_config.max_scale must be >= min_scale")

        # Validate bucket sizes
        for field_name, value in [
            ("bucket_size_mb", self.bucket_size_mb),
            ("allgather_bucket_size_mb", self.allgather_bucket_size_mb),
            ("reduce_bucket_size_mb", self.reduce_bucket_size_mb),
        ]:
            if value <= 0 or value > 1000:  # Max 1GB buckets
                raise ValueError(
                    f"{field_name} must be between 0 and 1000 MB, got {value}"
                )

        # Validate gradient accumulation
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                "gradient_accumulation_steps must be >= 1, "
                f"got {self.gradient_accumulation_steps}"
            )

        # Validate scaling factors
        if self.gradient_predivide_factor <= 0 or self.gradient_postdivide_factor <= 0:
            raise ValueError("Gradient scaling factors must be positive")

    def get_memory_usage_gb(
        self, num_params: int, optimizer_state_size: int = 2
    ) -> float:
        """Estimate memory usage in GB.

        Args:
            num_params: Total number of parameters.
            optimizer_state_size: Number of optimizer state tensors per parameter
                                (e.g., 2 for Adam: momentum and variance).

        Returns:
            Estimated memory usage in GB.
        """
        bytes_per_param = 4 if self.dtype == torch.float32 else 2

        # Parameter memory
        param_memory = num_params * bytes_per_param

        # Gradient memory
        grad_memory = (
            num_params * bytes_per_param if not self.partition_gradients else 0
        )

        # Optimizer state memory
        if self.partition_optimizer_states:
            # Only store states for local partition
            state_memory = 0
        else:
            # States typically FP32
            state_memory = num_params * optimizer_state_size * 4

        # Mixed precision main parameters
        if self.mixed_precision:
            # Need FP32 copy of parameters
            param_memory += num_params * 4

        # Account for communication buffers
        comm_buffer_memory = (
            self.bucket_size_mb
            + self.allgather_bucket_size_mb
            + self.reduce_bucket_size_mb
        ) * (
            1024 * 1024
        )  # Convert MB to bytes

        total_bytes = param_memory + grad_memory + state_memory + comm_buffer_memory
        return total_bytes / (1024**3)  # Convert to GB

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration.
        """
        config_dict = {
            "partition_parameters": self.partition_parameters,
            "partition_gradients": self.partition_gradients,
            "partition_optimizer_states": self.partition_optimizer_states,
            "partitioning_strategy": self.partitioning_strategy.value,
            "contiguous_gradients": self.contiguous_gradients,
            "overlap_grad_reduce": self.overlap_grad_reduce,
            "cpu_offload": self.cpu_offload,
            "mixed_precision": self.mixed_precision,
            "grad_clip_value": self.grad_clip_value,
            "use_multi_tensor_apply": self.use_multi_tensor_apply,
            "bucket_size_mb": self.bucket_size_mb,
            "dtype": str(self.dtype),
            "grad_scaler_config": self.grad_scaler_config,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }
        return config_dict
