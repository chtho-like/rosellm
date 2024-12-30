from typing import Optional, Type
from config import LLMConfig
from executor import ExecutorBase

class EngineArgs:
    def __init__(
        model: str, 
        gpu_memory_utilization : float,
        tensor_parallel_size : int,
        enforce_eager : Optional[bool] = None,
    ) -> None:
        pass
    
    def create_config(
        self, 
    ) -> LLMConfig:
        pass


class LLMEngine:
    def __init__(
        llm_config: LLMConfig,
        executor_class: Type[ExecutorBase]
    ):
        pass
    
    @classmethod
    def from_engine_args(
        cls, 
        engine_args: EngineArgs,
    ) -> "LLMEngine":
        pass
