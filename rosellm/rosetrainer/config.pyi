# Type stub file for better IDE support
from enum import Enum
from typing import Any, Dict, Optional, Union

class PrecisionType(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    MIXED = "mixed"

class GradientClipType(Enum):
    NORM = "norm"
    VALUE = "value"
    NONE = "none"

class OptimizerConfig:
    name: str
    learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float

class GradientConfig:
    clip_type: GradientClipType
    clip_value: Optional[float]
    accumulation_steps: int

class MemoryConfig:
    activation_checkpointing: bool
    cpu_offload: bool
    zero_optimization_stage: int
    gradient_checkpointing_ratio: float
    memory_efficient_attention: bool

class ParallelismConfig:
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: Optional[int]
    context_parallel_size: int
    expert_parallel_size: int

class TrainingConfig:
    # Properties
    batch_size: int
    num_epochs: int
    max_steps: Optional[int]
    warmup_steps: int
    seed: int
    precision: PrecisionType
    compile_model: bool
    use_cuda_graphs: bool
    optimizer: OptimizerConfig
    gradient: GradientConfig
    memory: MemoryConfig
    parallelism: ParallelismConfig
    checkpoint_interval: int
    checkpoint_dir: str
    resume_from_checkpoint: Optional[str]
    log_interval: int
    eval_interval: int
    tensorboard_dir: Optional[str]
    wandb_project: Optional[str]
    track_memory: bool
    track_throughput: bool
    profile_kernel: bool

    def __init__(
        self,
        batch_size: int = 32,
        num_epochs: int = 3,
        max_steps: Optional[int] = None,
        warmup_steps: int = 0,
        seed: int = 42,
        precision: PrecisionType = PrecisionType.FP32,
        compile_model: bool = False,
        use_cuda_graphs: bool = False,
        optimizer: Optional[Dict[str, Any]] = None,
        gradient: Optional[Dict[str, Any]] = None,
        memory: Optional[Dict[str, Any]] = None,
        parallelism: Optional[Dict[str, Any]] = None,
        checkpoint_interval: int = 1000,
        checkpoint_dir: str = "./checkpoints",
        resume_from_checkpoint: Optional[str] = None,
        log_interval: int = 10,
        eval_interval: int = 100,
        tensorboard_dir: Optional[str] = None,
        wandb_project: Optional[str] = None,
        track_memory: bool = True,
        track_throughput: bool = True,
        profile_kernel: bool = False,
    ) -> None: ...
    def to_dict(self) -> Dict[str, Any]: ...
    def model_dump(self, *, mode: str = "python", **kwargs) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig": ...

def validate_config(
    config: Union[Dict[str, Any], TrainingConfig],
) -> TrainingConfig: ...
