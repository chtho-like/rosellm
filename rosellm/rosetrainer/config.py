"""
Configuration Schema for RoseTrainer

Provides validated configuration with Pydantic models for type safety
and automatic validation of training parameters.
"""

from enum import Enum
from typing import Literal, Optional, Union

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_CHECKPOINT_INTERVAL,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_GRADIENT_CLIP_VALUE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    DEFAULT_WARMUP_STEPS,
    DEFAULT_WEIGHT_DECAY,
)


class PrecisionType(str, Enum):
    """Supported precision types for training."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    MIXED = "mixed"


class GradientClipType(str, Enum):
    """Gradient clipping strategies."""

    NORM = "norm"
    VALUE = "value"
    NONE = "none"


class OptimizerConfig(BaseModel):
    """Optimizer configuration."""

    model_config = ConfigDict(use_enum_values=True)

    name: Literal["adam", "adamw", "sgd", "rmsprop"] = "adamw"
    learning_rate: float = Field(
        DEFAULT_LEARNING_RATE, gt=0, le=1.0, description="Learning rate"
    )
    weight_decay: float = Field(DEFAULT_WEIGHT_DECAY, ge=0, le=1.0)
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = Field(1e-8, gt=0)

    @field_validator("betas")
    def validate_betas(cls, v):
        beta1, beta2 = v
        if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
            raise ValueError("Betas must be in range [0, 1)")
        return v


class GradientConfig(BaseModel):
    """Gradient handling configuration."""

    model_config = ConfigDict(use_enum_values=True)

    clip_type: GradientClipType = GradientClipType.NORM
    clip_value: Optional[float] = Field(
        DEFAULT_GRADIENT_CLIP_VALUE, gt=0, description="Gradient clip value"
    )
    accumulation_steps: int = Field(
        DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        ge=1,
        description="Gradient accumulation steps",
    )

    @field_validator("clip_value")
    def validate_clip_value(cls, v, info):
        if info.data.get("clip_type") != GradientClipType.NONE and v is None:
            raise ValueError("clip_value required when clip_type is not 'none'")
        return v


class MemoryConfig(BaseModel):
    """Memory optimization configuration."""

    model_config = ConfigDict(use_enum_values=True)

    activation_checkpointing: bool = False
    cpu_offload: bool = False
    zero_optimization_stage: Literal[0, 1, 2, 3] = 0
    gradient_checkpointing_ratio: float = Field(0.0, ge=0, le=1.0)
    memory_efficient_attention: bool = True


class ParallelismConfig(BaseModel):
    """Parallelism configuration."""

    tensor_parallel_size: int = Field(1, ge=1)
    pipeline_parallel_size: int = Field(1, ge=1)
    data_parallel_size: Optional[int] = Field(None, ge=1)
    context_parallel_size: int = Field(1, ge=1)
    expert_parallel_size: int = Field(1, ge=1)

    @field_validator(
        "data_parallel_size", "tensor_parallel_size", "pipeline_parallel_size"
    )
    def validate_power_of_two(cls, v):
        if v is not None and v > 1 and (v & (v - 1)) != 0:
            raise ValueError(f"Parallel size {v} should be a power of 2")
        return v


# Factory functions for cleaner default configurations
def _default_optimizer() -> OptimizerConfig:
    """Create default optimizer configuration.

    Returns:
        OptimizerConfig: Default optimizer settings with learning_rate=1e-4,
                        weight_decay=0.01, eps=1e-8, betas=(0.9, 0.999)
    """
    return OptimizerConfig()  # type: ignore[call-arg]


def _default_gradient() -> GradientConfig:
    """Create default gradient configuration.

    Returns:
        GradientConfig: Default gradient settings with clip_type='norm',
                       clip_value=1.0, accumulation_steps=1
    """
    return GradientConfig()  # type: ignore[call-arg]


def _default_memory() -> MemoryConfig:
    """Create default memory configuration.

    Returns:
        MemoryConfig: Default memory settings with all optimizations disabled
    """
    return MemoryConfig()  # type: ignore[call-arg]


def _default_parallelism() -> ParallelismConfig:
    """Create default parallelism configuration.

    Returns:
        ParallelismConfig: Default parallelism settings with no model parallelism
    """
    return ParallelismConfig()  # type: ignore[call-arg]


class TrainingConfig(BaseModel):
    """Main training configuration with validation."""

    model_config = ConfigDict(use_enum_values=True, arbitrary_types_allowed=True)

    # Basic training parameters
    batch_size: int = Field(
        DEFAULT_BATCH_SIZE, ge=1, description="Batch size per device"
    )
    num_epochs: Optional[int] = Field(None, ge=1)
    max_steps: Optional[int] = Field(None, ge=1)
    warmup_steps: int = Field(DEFAULT_WARMUP_STEPS, ge=0)
    seed: int = Field(DEFAULT_SEED, ge=0)

    # Precision and performance
    precision: PrecisionType = PrecisionType.FP32
    compile_model: bool = False
    use_cuda_graphs: bool = False

    # Sub-configurations with clean factory functions
    optimizer: OptimizerConfig = Field(default_factory=_default_optimizer)
    gradient: GradientConfig = Field(default_factory=_default_gradient)
    memory: MemoryConfig = Field(default_factory=_default_memory)
    parallelism: ParallelismConfig = Field(default_factory=_default_parallelism)

    # Checkpointing
    checkpoint_interval: int = Field(DEFAULT_CHECKPOINT_INTERVAL, ge=1)
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR
    resume_from_checkpoint: Optional[str] = None

    # Logging and monitoring
    log_interval: int = Field(10, ge=1)
    eval_interval: int = Field(100, ge=1)
    tensorboard_dir: Optional[str] = None
    wandb_project: Optional[str] = None

    # Performance tracking
    track_memory: bool = True
    track_throughput: bool = True
    profile_kernel: bool = False

    @field_validator("max_steps")
    def validate_max_steps(cls, v, info):
        if v is not None and "num_epochs" in info.data:
            num_epochs = info.data.get("num_epochs")
            if v > 0 and num_epochs is not None and num_epochs > 0:
                raise ValueError("Cannot specify both max_steps and num_epochs")
        return v

    @model_validator(mode="after")
    def validate_training_duration(self):
        """Ensure at least one of max_steps or num_epochs is specified."""
        if self.max_steps is None and self.num_epochs is None:
            # Set default num_epochs if neither is specified
            self.num_epochs = 3
        return self

    @field_validator("precision")
    def validate_precision_support(cls, v):
        if v == PrecisionType.FP8:
            if not hasattr(torch, "float8_e4m3fn"):
                raise ValueError("FP8 not supported in current PyTorch version")
        elif v == PrecisionType.BF16:
            # Only validate BF16 support when CUDA is available
            if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
                raise ValueError("BF16 not supported on current hardware")
            elif not torch.cuda.is_available():
                # On CPU, BF16 can still be used (though may be emulated)
                import warnings
                warnings.warn(
                    "BF16 validation skipped - no CUDA available. "
                    "BF16 may be emulated on CPU with potential performance impact."
                )
        return v

    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility."""
        config_dict = self.model_dump(mode="json")  # Serialize enums properly

        # Flatten nested configs for backward compatibility
        if "optimizer" in config_dict:
            config_dict["learning_rate"] = config_dict["optimizer"]["learning_rate"]
            config_dict["weight_decay"] = config_dict["optimizer"]["weight_decay"]

        if "gradient" in config_dict:
            config_dict["gradient_clip_type"] = config_dict["gradient"]["clip_type"]
            config_dict["gradient_clip_value"] = config_dict["gradient"]["clip_value"]

        return config_dict

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainingConfig":
        """Create from dictionary, handling legacy configs."""
        # Handle legacy flat structure
        if "learning_rate" in config_dict and "optimizer" not in config_dict:
            config_dict["optimizer"] = {
                "learning_rate": config_dict.pop("learning_rate", 1e-4),
                "weight_decay": config_dict.pop("weight_decay", 0.01),
            }

        if "gradient_clip_value" in config_dict and "gradient" not in config_dict:
            config_dict["gradient"] = {
                "clip_type": config_dict.pop("gradient_clip_type", "norm"),
                "clip_value": config_dict.pop("gradient_clip_value", None),
            }

        # Handle old max_grad_norm parameter
        if "max_grad_norm" in config_dict:
            if "gradient" not in config_dict:
                config_dict["gradient"] = {}
            config_dict["gradient"]["clip_value"] = config_dict.pop("max_grad_norm")
            config_dict["gradient"]["clip_type"] = "norm"

        return cls(**config_dict)


def validate_config(config: Union[dict, TrainingConfig]) -> TrainingConfig:
    """
    Validate and convert configuration to TrainingConfig with recursive validation.

    Args:
        config: Dictionary or TrainingConfig object

    Returns:
        Validated TrainingConfig object with all sub-configs validated

    Raises:
        ValidationError: If configuration is invalid
        TypeError: If config is not dict or TrainingConfig
    """
    # Convert to TrainingConfig if needed
    if isinstance(config, dict):
        validated = TrainingConfig.from_dict(config)
    elif isinstance(config, TrainingConfig):
        validated = config
    else:
        raise TypeError(f"Config must be dict or TrainingConfig, got {type(config)}")

    # Recursively validate sub-configurations
    try:
        # Validate optimizer config
        if hasattr(validated, "optimizer"):
            validated.optimizer = OptimizerConfig(**validated.optimizer.model_dump())

        # Validate gradient config
        if hasattr(validated, "gradient"):
            validated.gradient = GradientConfig(**validated.gradient.model_dump())

        # Validate memory config
        if hasattr(validated, "memory"):
            validated.memory = MemoryConfig(**validated.memory.model_dump())

        # Validate parallelism config
        if hasattr(validated, "parallelism"):
            validated.parallelism = ParallelismConfig(
                **validated.parallelism.model_dump()
            )

    except Exception as e:
        raise ValueError(f"Sub-configuration validation failed: {e}")

    return validated
