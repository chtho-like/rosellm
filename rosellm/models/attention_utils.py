from typing import Callable, Dict

from rosellm.models.flash_attention_utils import flash_attention_forward

ALL_ATTENTION_FUNCTIONS: Dict[str, Callable] = {}

ALL_ATTENTION_FUNCTIONS.update(
    {
        "flash_attention_2": flash_attention_forward,
    }
)

if __name__ == "__main__":
    print(ALL_ATTENTION_FUNCTIONS)
