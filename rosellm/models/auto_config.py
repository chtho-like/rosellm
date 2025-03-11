from typing import Optional

from rosellm.models.mappings import CONFIG_MAPPING
from rosellm.models.pretrained_config import PretrainedConfig


class AutoConfig:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        _commit_hash: Optional[str] = None,
    ):
        config_dict = PretrainedConfig.get_config_dict(
            pretrained_model_name_or_path,
            commit_hash=_commit_hash,
        )
        if "model_type" in config_dict:
            try:
                config_class = CONFIG_MAPPING[config_dict["model_type"]]
            except KeyError:
                raise ValueError(
                    f"{config_dict['model_type']} is not a valid model type."
                )
        else:
            # Fallback: use pattern matching on the string.
            for pattern in sorted(CONFIG_MAPPING.keys(), key=len, reverse=True):
                if pattern in str(pretrained_model_name_or_path):
                    config_class = CONFIG_MAPPING[pattern]
        return config_class.from_dict(config_dict)
