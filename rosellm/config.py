from typing import Literal 
from dataclasses import dataclass, field

TaskOption = Literal[
    "auto",
    "generate",
    "embedding",
    "embed",
    "classify",
    "score",
    "reward",
]

class ModelConfig:
    pass

@dataclass
class LLMConfig:
    model_config: ModelConfig = field(default=None, init=True)
