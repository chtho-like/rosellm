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


if __name__ == "__main__":
    name = "Qwen/Qwen2.5-0.5B"
    config_dict = PretrainedConfig.get_config_dict(
        name,
        commit_hash="060db6499f32faf8b98477b0a26969ef7d8b9987",
    )
    print(config_dict)
    """
$ python rosellm/models/pretrained_config.py 
{
    "architectures": ["Qwen2ForCausalLM"],
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151643,
    "hidden_act": "silu",
    "hidden_size": 896,
    "initializer_range": 0.02,
    "intermediate_size": 4864,
    "max_position_embeddings": 32768,
    "max_window_layers": 24,
    "model_type": "qwen2",
    "num_attention_heads": 14,
    "num_hidden_layers": 24,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "sliding_window": 32768,
    "tie_word_embeddings": true,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.40.1",
    "use_cache": true,
    "use_mrope": false,
    "use_sliding_window": false,
    "vocab_size": 151936,
    "_commit_hash": "060db6499f32faf8b98477b0a26969ef7d8b9987"
}
    """
