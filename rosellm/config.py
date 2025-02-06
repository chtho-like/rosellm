import enum
from typing import Literal, List
from dataclasses import dataclass, field

class DecodingParams:
    def __init__(
        self,
        # Number of tokens to sample.
        n: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        use_beam_search: bool = False,
        stop_token_ids: List[int] = [],
    ) -> None:
        assert n >= 1
        assert temperature >= 0.0
        assert 0.0 < top_p <= 1.0
        if use_beam_search:
            # When using beam search, we need to sample more than one token.
            assert n > 1
            # When using beam search, temperature must be greater than 0.0
            # to provide diversity to the beam paths.
            assert temperature > 0.0
            # When using beam search, top_p must be 1.0 to ensure that
            # all beam paths are considered.
            assert top_p == 1.0
        elif temperature == 0.0:
            # Zero temperature means greedy decoding.
            assert n == 1
            # When using greedy decoding, top_p must be 1.0 to ensure that
            # the most likely token will not be filtered out.
            assert top_p == 1.0
        
        self.n = n
        self.temperature = temperature
        self.top_p = top_p
        self.use_beam_search = use_beam_search
        self.stop_token_ids = stop_token_ids

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
