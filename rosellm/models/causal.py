import os
import re
import sys
import uuid
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from torch import nn

from rosellm.config import ModelConfig
from rosellm.models.auto_config import AutoConfig
from rosellm.models.envs import CACHE_DIR, SESSION_ID
from rosellm.models.hub import extract_commit_hash, resolve_file
from rosellm.models.mappings import MODEL_FOR_CAUSAL_LM_MAPPING


class CausalModel(nn.Module):
    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @classmethod
    def _load_config(
        cls,
        model_path: str,
        args: Optional[ModelConfig] = None,
    ):
        resolved_config_file = resolve_file(
            model_path,
            "config.json",
        )
        if resolved_config_file is None:
            raise ValueError(f"Config file not found: {model_path}/config.json")
        commit_hash = extract_commit_hash(resolved_config_file)
        if commit_hash is None:
            raise ValueError(f"Invalid commit hash: {commit_hash}")
        return AutoConfig.from_pretrained(
            model_path,
            _commit_hash=commit_hash,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        args: Optional[ModelConfig] = None,
    ):
        config = cls._load_config(model_path, args)
        model_class = cls._model_mapping[type(config)]
        if model_class is None:
            raise ValueError(f"{type(config)} is not a valid model type.")
        return model_class._from_config(config, args)

    def forward(self, input_ids: torch.LongTensor, positions: torch.LongTensor):
        pass
