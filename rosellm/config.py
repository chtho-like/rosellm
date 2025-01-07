import enum
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

class LoadFormat(str, enum.Enum):
    """
    LoadFormat inherits from both str and enum.Enum.
    """
    # "auto" first tries to load weights in safetensors format.
    # Falls back to PyTorch format if safetensors is not available.
    AUTO = "auto"
    # "pt" is the standard PyTorch binary format (.bin files).
    # The traditional format for saving PyTorch models.
    # Uses Python's pickle serialization under the hood.
    PT = "pt"
    # "safetensors" is a safer alternative to PyTorch's native format.
    # Doesn't use Python's pickle, making it more secure.
    # Generally faster to load than PyTorch format.
    # Becoming the standard format for new model releases.
    SAFETENSORS = "safetensors"
    # "npcache" loads weights in PyTorch format but caches them as NumPy arrays.
    # Can speed up subsequent loads of the same model.
    # Trades disk space for faster loading times.
    NPCACHE = "npcache"

@dataclass 
class LoadConfig:
    """
    @dataclass is a decorator introduced in Python 3.7 that automatically 
    adds generated special methods such as __init__(), __repr__(), __eq__() etc. 
    to user-defined classes. Here's a breakdown of what it does:
    * Automatically generates an __init__() method with parameters for all class 
      variables
    * Automatically generates __repr__() for nice string representation
    * Automatically generates __eq__() for comparing instances
    * Makes the class more concise by reducing boilerplate code
    """
    load_format: LoadFormat = LoadFormat.AUTO

class ModelConfig:
    pass

@dataclass
class LLMConfig:
    model_config: ModelConfig = field(default=None, init=True)
