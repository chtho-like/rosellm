import json
from typing import Optional

from rosellm.models.envs import CACHE_DIR
from rosellm.models.hub import extract_commit_hash, resolve_file


class PretrainedConfig:
    @classmethod
    def _dict_from_json_file(cls, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def get_config_dict(
        cls,
        pretrained_model_name_or_path: str,
        cache_dir: str = CACHE_DIR,
        force_download: bool = False,
        commit_hash: Optional[str] = None,
    ):
        resolved_config_file = resolve_file(
            pretrained_model_name_or_path,
            "config.json",
            cache_dir=cache_dir,
            force_download=force_download,
            _commit_hash=commit_hash,
        )
        if resolved_config_file is None:
            raise ValueError(
                f"Config file not found: {pretrained_model_name_or_path}/config.json"
            )
        if commit_hash is None:
            commit_hash = extract_commit_hash(resolved_config_file)
        config_dict = cls._dict_from_json_file(resolved_config_file)
        config_dict["_commit_hash"] = commit_hash
        return config_dict
