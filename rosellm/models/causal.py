from typing import Optional

import torch
from torch import nn

from rosellm.config import ModelConfig


class CausalModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @classmethod
    def _load_config(cls, model_path: str):
        pass

    @classmethod
    def _get_model_class(cls, config):
        pass

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        args: Optional[ModelConfig] = None,
    ):
        config = cls._load_config(model_path)
        model_class = cls._get_model_class(config)

    def forward(self, input_ids: torch.LongTensor, positions: torch.LongTensor):
        pass
