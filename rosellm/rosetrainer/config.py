"""
Configuration Schema for RoseTrainer

Provides validated configuration with Pydantic models for type safety
and automatic validation of training parameters.
"""

from enum import Enum
from typing import Literal, Optional, Union

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    learning_rate: float = Field(1e-4, gt=0, le=1.0, description="Learning rate")
    weight_decay: float = Field(0.01, ge=0, le=1.0)
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
    clip_value: Optional[float] = Field(None, gt=0, description="Gradient clip value")
    accumulation_steps: int = Field(1, ge=1, description="Gradient accumulation steps")

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


class TrainingConfig(BaseModel):
    """Main training configuration with validation."""

    model_config = ConfigDict(use_enum_values=True, arbitrary_types_allowed=True)

    # Basic training parameters
    batch_size: int = Field(32, ge=1, description="Batch size per device")
    num_epochs: int = Field(3, ge=1)
    max_steps: Optional[int] = Field(None, ge=1)
    warmup_steps: int = Field(0, ge=0)
    seed: int = Field(42, ge=0)

    # Precision and performance
    precision: PrecisionType = PrecisionType.FP32
    compile_model: bool = False
    use_cuda_graphs: bool = False

    # Sub-configurations
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    gradient: GradientConfig = Field(default_factory=GradientConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    parallelism: ParallelismConfig = Field(default_factory=ParallelismConfig)

    # Checkpointing
    checkpoint_interval: int = Field(1000, ge=1)
    checkpoint_dir: str = "./checkpoints"
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
            if v > 0 and info.data["num_epochs"] > 0:
                raise ValueError("Cannot specify both max_steps and num_epochs")
        return v

    @field_validator("precision")
    def validate_precision_support(cls, v):
        if v == PrecisionType.FP8:
            if not hasattr(torch, "float8_e4m3fn"):
                raise ValueError("FP8 not supported in current PyTorch version")
        elif v == PrecisionType.BF16:
            if not torch.cuda.is_bf16_supported():
                raise ValueError("BF16 not supported on current hardware")
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
    Validate and convert configuration to TrainingConfig.

    Args:
        config: Dictionary or TrainingConfig object

    Returns:
        Validated TrainingConfig object

    Raises:
        ValidationError: If configuration is invalid
    """
    if isinstance(config, dict):
        return TrainingConfig.from_dict(config)
    elif isinstance(config, TrainingConfig):
        return config
    else:
        raise TypeError(f"Config must be dict or TrainingConfig, got {type(config)}")
