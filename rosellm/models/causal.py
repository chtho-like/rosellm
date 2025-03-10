import os
from typing import Optional

import torch
from torch import nn

from rosellm.config import ModelConfig

CACHE_DIR = os.getenv("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))


class CausalModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @classmethod
    def _try_to_load_from_cache(
        cls,
        repo_id: str,
        filename: str,
        cache_dir: str = CACHE_DIR,
        repo_type: str = "model",
        revision: str = "main",
    ):
        if cache_dir is None:
            cache_dir = CACHE_DIR
        if repo_type is None:
            repo_type = "model"
        object_id = repo_id.replace("/", "--")
        repo_cache = os.path.join(cache_dir, f"{repo_type}s--{object_id}")
        if not os.path.isdir(repo_cache):
            return None
        refs_dir = os.path.join(repo_cache, "refs")
        snapshots_dir = os.path.join(repo_cache, "snapshots")
        no_exist_dir = os.path.join(repo_cache, ".no_exist")
        if os.path.isdir(refs_dir):
            revision_file = os.path.join(refs_dir, revision)
            if os.path.isfile(revision_file):
                with open(revision_file) as f:
                    revision = f.read()
        if os.path.isfile(os.path.join(no_exist_dir, revision, filename)):
            return None
        if not os.path.exists(snapshots_dir):
            return None
        cached_shas = os.listdir(snapshots_dir)
        if revision not in cached_shas:
            return None
        cached_file = os.path.join(snapshots_dir, revision, filename)
        return cached_file if os.path.isfile(cached_file) else None

    @classmethod
    def _resolve_file(
        cls,
        model_path: str,
        filename: str,
        cache_dir: Optional[str] = CACHE_DIR,
        force_download: bool = False,
        _commit_hash: Optional[str] = None,
    ):
        if os.path.isdir(model_path):
            resolved_file = os.path.join(model_path, filename)
            if not os.path.isfile(resolved_file):
                raise FileNotFoundError(f"File {filename} not found in {model_path}")
            return resolved_file
        if cache_dir is None:
            cache_dir = CACHE_DIR
        if _commit_hash is not None and not force_download:
            resolved_file = cls._try_to_load_from_cache(
                model_path,
                filename,
                cache_dir=cache_dir,
                revision=_commit_hash,
            )
        return resolved_file

    @classmethod
    def _load_config(
        cls,
        model_path: str,
        args: Optional[ModelConfig] = None,
    ):
        resolved_config_file = cls._resolve_file(
            model_path,
            "config.json",
        )

    @classmethod
    def _get_model_class(cls, config):
        pass

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        args: Optional[ModelConfig] = None,
    ):
        config = cls._load_config(model_path, args)
        model_class = cls._get_model_class(config)

    def forward(self, input_ids: torch.LongTensor, positions: torch.LongTensor):
        pass
