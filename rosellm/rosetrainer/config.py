"""
Configuration Schema for RoseTrainer

Provides validated configuration with Pydantic models for type safety
and automatic validation of training parameters.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

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
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = Field(1e-8, gt=0)

    @field_validator("betas")
    def validate_betas(cls, v):
        beta1, beta2 = v
        if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
            raise ValueError("Betas must be in range [0, 1)")
        return v


class GradientConfig(BaseModel):
    """Gradient handling configuration with advanced utilities support."""

    model_config = ConfigDict(use_enum_values=True)

    # Basic gradient clipping
    clip_type: GradientClipType = GradientClipType.NORM
    clip_value: Optional[float] = Field(
        DEFAULT_GRADIENT_CLIP_VALUE, gt=0, description="Gradient clip value"
    )
    accumulation_steps: int = Field(
        DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        ge=1,
        description="Gradient accumulation steps",
    )

    # Advanced gradient utilities configuration
    norm_type: float = Field(2.0, gt=0, description="Norm type for gradient clipping")
    use_multitensor: bool = Field(
        True, description="Use multi-tensor operations when available"
    )
    model_parallel_reduce: bool = Field(
        True, description="Reduce gradients across model parallel groups"
    )
    error_if_nonfinite: bool = Field(
        True, description="Raise error if non-finite gradients detected"
    )

    # Gradient synchronization
    sync_on_accumulation: bool = Field(
        False, description="Sync gradients on every accumulation step"
    )

    # Gradient monitoring
    track_gradient_stats: bool = Field(
        False, description="Track detailed gradient statistics"
    )
    gradient_stats_interval: int = Field(
        100, ge=1, description="Interval for gradient statistics logging"
    )
    include_gradient_histograms: bool = Field(
        False, description="Include gradient histograms in stats (expensive)"
    )

    # Advanced gradient finalization
    enable_advanced_finalization: bool = Field(
        False,
        description="Enable advanced gradient finalization with multi-precision",
    )
    master_precision: Literal["fp32", "fp16", "bf16"] = Field(
        "fp32", description="Master precision for gradient storage and accumulation"
    )
    communication_precision: Optional[Literal["fp32", "fp16", "bf16"]] = Field(
        None,
        description="Precision for gradient communication (None uses master_precision)",
    )
    enable_gradient_compression: bool = Field(
        False, description="Enable gradient compression for communication"
    )
    compression_threshold_mb: float = Field(
        10.0, gt=0, description="Size threshold in MB for applying compression"
    )
    normalize_gradients: bool = Field(
        False, description="Normalize gradients across parallelism dimensions"
    )
    advanced_sync_order: Optional[List[str]] = Field(
        None, description="Custom synchronization order for parallelism dimensions"
    )
    finalization_verbose: bool = Field(
        False, description="Enable verbose logging for gradient finalization"
    )

    @field_validator("clip_value")
    def validate_clip_value(cls, v, info):
        if info.data.get("clip_type") != GradientClipType.NONE and v is None:
            raise ValueError("clip_value required when clip_type is not 'none'")
        return v

    @field_validator("norm_type")
    def validate_norm_type(cls, v):
        if v != float("inf") and v <= 0:
            raise ValueError("norm_type must be positive or inf")
        return v


class MixedPrecisionConfig(BaseModel):
    """Mixed precision training configuration with dynamic loss scaling support."""

    model_config = ConfigDict(use_enum_values=True)

    # Precision settings
    enabled: bool = Field(True, description="Enable mixed precision training")
    precision_type: Literal["fp16", "bf16", "fp32", "mixed"] = Field(
        "fp16", description="Precision type for mixed precision training"
    )
    autocast_enabled: bool = Field(True, description="Enable PyTorch autocast")

    # Dynamic loss scaling configuration
    use_dynamic_scaling: bool = Field(True, description="Use dynamic loss scaling")
    initial_scale: float = Field(2**16, gt=0, description="Initial loss scale")
    min_scale: float = Field(1.0, gt=0, description="Minimum loss scale")
    max_scale: float = Field(2**24, gt=0, description="Maximum loss scale")
    growth_factor: float = Field(
        2.0, gt=1.0, le=10.0, description="Scale growth factor"
    )
    backoff_factor: float = Field(0.5, gt=0, lt=1.0, description="Scale backoff factor")
    growth_interval: int = Field(2000, ge=1, description="Steps before scale growth")
    hysteresis: int = Field(2, ge=1, description="Consecutive overflows before backoff")

    # Multi-tensor optimization
    use_multi_tensor: bool = Field(
        True, description="Use multi-tensor operations when available"
    )
    chunk_size: int = Field(
        2**20, gt=0, description="Chunk size for multi-tensor ops"
    )

    # Overflow detection
    enable_inf_nan_check: bool = Field(
        True, description="Check for inf/nan in gradients"
    )
    check_frequency: int = Field(1, ge=1, description="Overflow check frequency")
    skip_first_n_steps: int = Field(
        10, ge=0, description="Skip overflow check for first N steps"
    )

    # Performance optimizations
    use_fused_kernels: bool = Field(
        True, description="Use fused CUDA kernels when available"
    )
    cache_inv_scale: bool = Field(
        True, description="Cache inverse scale for efficiency"
    )

    # Monitoring
    log_scale_changes: bool = Field(True, description="Log scale changes")
    detailed_overflow_info: bool = Field(
        False, description="Log detailed overflow info"
    )
    track_overflow_history: int = Field(
        100, ge=0, description="Track N recent overflow events"
    )

    @field_validator("max_scale")
    def validate_max_scale(cls, v, info):
        if "min_scale" in info.data and v < info.data["min_scale"]:
            raise ValueError("max_scale must be >= min_scale")
        return v

    @field_validator("initial_scale")
    def validate_initial_scale(cls, v, info):
        data = info.data
        if "min_scale" in data and v < data["min_scale"]:
            raise ValueError("initial_scale must be >= min_scale")
        if "max_scale" in data and v > data["max_scale"]:
            raise ValueError("initial_scale must be <= max_scale")
        return v


class PositionEmbeddingConfig(BaseModel):
    """Position embedding configuration."""

    model_config = ConfigDict(use_enum_values=True)

    # Position embedding type
    embedding_type: Literal["none", "learned", "sinusoidal", "rotary", "alibi"] = Field(
        "none", description="Type of position embedding to use"
    )

    # Common parameters
    max_position_embeddings: int = Field(
        2048, ge=1, description="Maximum sequence length for position embeddings"
    )
    hidden_size: Optional[int] = Field(
        None, ge=1, description="Hidden size for learned/sinusoidal embeddings"
    )

    # RoPE specific configuration
    rope_dim: Optional[int] = Field(
        None, ge=2, description="Dimension for RoPE (must be even)"
    )
    rope_base: float = Field(
        10000.0, gt=0, description="Base for RoPE frequency computation"
    )
    rope_scaling: Optional[Dict[str, Any]] = Field(
        None, description="RoPE scaling configuration for context extension"
    )
    rope_interpolation_type: Literal[
        "none", "linear", "ntk", "dynamic_ntk", "yarn"
    ] = Field("none", description="RoPE interpolation method")
    rope_scaling_factor: float = Field(
        1.0, gt=0, description="Scaling factor for RoPE positions"
    )
    rope_partial_factor: float = Field(
        1.0, gt=0, le=1.0, description="Fraction of dimensions to apply RoPE to"
    )
    rope_use_fused: bool = Field(
        True, description="Use fused RoPE operations for better performance"
    )

    # ALiBi specific
    alibi_num_heads: Optional[int] = Field(
        None, ge=1, description="Number of heads for ALiBi"
    )

    # Learned embeddings specific
    learned_dropout: float = Field(
        0.0, ge=0, le=1.0, description="Dropout for learned embeddings"
    )

    @field_validator("rope_dim")
    def validate_rope_dim(cls, v):
        if v is not None and v % 2 != 0:
            raise ValueError(f"rope_dim must be even, got {v}")
        return v

    @model_validator(mode="after")
    def validate_embedding_params(self):
        """Validate embedding-specific parameters."""
        if (
            self.embedding_type in ["learned", "sinusoidal"]
            and self.hidden_size is None
        ):
            raise ValueError(f"{self.embedding_type} embeddings require hidden_size")

        if self.embedding_type == "rotary" and self.rope_dim is None:
            raise ValueError("Rotary embeddings require rope_dim")

        if self.embedding_type == "alibi" and self.alibi_num_heads is None:
            raise ValueError("ALiBi embeddings require alibi_num_heads")

        return self


class MemoryConfig(BaseModel):
    """Memory optimization configuration."""

    model_config = ConfigDict(use_enum_values=True)

    activation_checkpointing: bool = False
    cpu_offload: bool = False
    zero_optimization_stage: Literal[0, 1, 2, 3] = 0
    gradient_checkpointing_ratio: float = Field(0.0, ge=0, le=1.0)
    memory_efficient_attention: bool = True


class BucketingConfig(BaseModel):
    """Gradient bucketing configuration."""

    model_config = ConfigDict(use_enum_values=True)

    # Enable gradient bucketing
    enabled: bool = Field(False, description="Enable gradient bucketing")

    # Bucket strategy
    strategy: Literal["size_based", "layer_based", "mixed", "custom"] = Field(
        "size_based", description="Bucketing strategy"
    )

    # Bucket size limits
    max_bucket_size_mb: float = Field(
        25.0, gt=0, le=200.0, description="Maximum bucket size in MB"
    )
    min_bucket_size_mb: float = Field(
        1.0, gt=0, description="Minimum bucket size in MB"
    )

    # Communication settings
    backend: Literal["nccl", "gloo", "auto"] = Field(
        "auto", description="Communication backend"
    )
    overlap_communication: bool = Field(
        True, description="Overlap communication with computation"
    )
    compress_gradients: bool = Field(False, description="Enable gradient compression")

    # Advanced features
    dynamic_bucketing: bool = Field(False, description="Dynamically adapt bucket sizes")
    gradient_predivision: bool = Field(
        True, description="Pre-divide gradients for numerical stability"
    )

    # Group management
    enable_groups: bool = Field(False, description="Enable bucket grouping")
    group_strategy: Literal[
        "parallel", "sequential", "hierarchical", "adaptive"
    ] = Field("adaptive", description="Group communication strategy")
    max_groups: int = Field(8, ge=1, le=32, description="Maximum number of groups")

    # Performance tuning
    communication_timeout_ms: int = Field(
        30000, ge=1000, description="Communication timeout in milliseconds"
    )
    bucket_cap_mb: float = Field(
        100.0, gt=0, description="Hard limit on bucket size in MB"
    )

    @field_validator("max_bucket_size_mb")
    def validate_max_bucket_size(cls, v, info):
        if "min_bucket_size_mb" in info.data and v < info.data["min_bucket_size_mb"]:
            raise ValueError("max_bucket_size_mb must be >= min_bucket_size_mb")
        return v

    @field_validator("bucket_cap_mb")
    def validate_bucket_cap(cls, v, info):
        if "max_bucket_size_mb" in info.data and v < info.data["max_bucket_size_mb"]:
            raise ValueError("bucket_cap_mb must be >= max_bucket_size_mb")
        return v


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


class SchedulerConfig(BaseModel):
    """Learning rate and weight decay scheduler configuration."""

    model_config = ConfigDict(use_enum_values=True)

    # Enable scheduler
    enabled: bool = Field(True, description="Enable learning rate scheduling")

    # Learning rate bounds
    init_lr: float = Field(0.0, ge=0, description="Initial LR for warmup start")
    max_lr: float = Field(1e-3, gt=0, description="Maximum LR after warmup")
    min_lr: float = Field(1e-5, ge=0, description="Minimum LR after decay")

    # Warmup and decay steps
    lr_warmup_steps: int = Field(0, ge=0, description="Number of warmup steps")
    lr_decay_steps: int = Field(1000, ge=1, description="Total decay steps")
    lr_decay_style: Literal[
        "linear", "cosine", "inverse-square-root", "WSD", "constant"
    ] = Field("linear", description="Learning rate decay style")

    # Weight decay scheduling
    start_wd: float = Field(0.0, ge=0, description="Initial weight decay")
    end_wd: float = Field(0.0, ge=0, description="Final weight decay")
    wd_incr_steps: int = Field(0, ge=0, description="Weight decay increment steps")
    wd_incr_style: Literal["linear", "cosine", "constant"] = Field(
        "constant", description="Weight decay increment style"
    )

    # WSD-specific parameters (Warmup-Stable-Decay)
    wsd_decay_steps: Optional[int] = Field(
        None, ge=1, description="Steps for WSD decay phase"
    )
    lr_wsd_decay_style: Optional[
        Literal["linear", "cosine", "exponential", "minus_sqrt"]
    ] = Field(None, description="Decay style within WSD phase")

    @field_validator("max_lr")
    def validate_max_lr(cls, v, info):
        if "min_lr" in info.data and v < info.data["min_lr"]:
            raise ValueError("max_lr must be >= min_lr")
        return v

    @field_validator("init_lr")
    def validate_init_lr(cls, v, info):
        if "max_lr" in info.data and v > info.data["max_lr"]:
            raise ValueError("init_lr must be <= max_lr")
        return v

    @field_validator("lr_warmup_steps")
    def validate_warmup_steps(cls, v, info):
        if "lr_decay_steps" in info.data and v >= info.data["lr_decay_steps"]:
            raise ValueError("lr_warmup_steps must be < lr_decay_steps")
        return v

    @field_validator("end_wd")
    def validate_end_wd(cls, v, info):
        if "start_wd" in info.data and v < info.data["start_wd"]:
            raise ValueError("end_wd must be >= start_wd")
        return v

    @model_validator(mode="after")
    def validate_wsd_config(self):
        """Validate WSD-specific configuration."""
        if self.lr_decay_style == "WSD":
            if self.wsd_decay_steps is None:
                raise ValueError("wsd_decay_steps required for WSD decay style")
            if self.lr_wsd_decay_style is None:
                raise ValueError("lr_wsd_decay_style required for WSD decay style")
            if self.wsd_decay_steps > self.lr_decay_steps:
                raise ValueError("wsd_decay_steps must be <= lr_decay_steps")
        return self


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


def _default_scheduler() -> SchedulerConfig:
    """Create default scheduler configuration.

    Returns:
        SchedulerConfig: Default scheduler settings with linear decay
    """
    return SchedulerConfig()  # type: ignore[call-arg]


def _default_mixed_precision() -> MixedPrecisionConfig:
    """Create default mixed precision configuration.

    Returns:
        MixedPrecisionConfig: Default mixed precision settings with FP16 and
            dynamic scaling
    """
    return MixedPrecisionConfig()  # type: ignore[call-arg]


def _default_bucketing() -> BucketingConfig:
    """Create default bucketing configuration.

    Returns:
        BucketingConfig: Default bucketing settings with bucketing disabled
    """
    return BucketingConfig()  # type: ignore[call-arg]


def _default_position_embedding() -> PositionEmbeddingConfig:
    """Create default position embedding configuration.

    Returns:
        PositionEmbeddingConfig: Default position embedding settings with none type
    """
    return PositionEmbeddingConfig()  # type: ignore[call-arg]


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
    scheduler: SchedulerConfig = Field(default_factory=_default_scheduler)
    mixed_precision: MixedPrecisionConfig = Field(
        default_factory=_default_mixed_precision
    )
    bucketing: BucketingConfig = Field(default_factory=_default_bucketing)
    position_embedding: PositionEmbeddingConfig = Field(
        default_factory=_default_position_embedding
    )

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

    def to_dict(self) -> Dict[Any, Any]:
        """Convert to dictionary for backward compatibility."""
        config_dict = self.model_dump(mode="json")  # Serialize enums properly

        # Flatten nested configs for backward compatibility
        if "optimizer" in config_dict:
            config_dict["learning_rate"] = config_dict["optimizer"]["learning_rate"]
            config_dict["weight_decay"] = config_dict["optimizer"]["weight_decay"]

        if "gradient" in config_dict:
            config_dict["gradient_clip_type"] = config_dict["gradient"]["clip_type"]
            config_dict["gradient_clip_value"] = config_dict["gradient"]["clip_value"]

        return config_dict  # type: ignore[no-any-return]

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

        # Validate scheduler config
        if hasattr(validated, "scheduler"):
            validated.scheduler = SchedulerConfig(**validated.scheduler.model_dump())

        # Validate mixed precision config
        if hasattr(validated, "mixed_precision"):
            validated.mixed_precision = MixedPrecisionConfig(
                **validated.mixed_precision.model_dump()
            )

        # Validate bucketing config
        if hasattr(validated, "bucketing"):
            validated.bucketing = BucketingConfig(**validated.bucketing.model_dump())

        # Validate position embedding config
        if hasattr(validated, "position_embedding"):
            validated.position_embedding = PositionEmbeddingConfig(
                **validated.position_embedding.model_dump()
            )

    except Exception as e:
        raise ValueError(f"Sub-configuration validation failed: {e}")

    return validated
