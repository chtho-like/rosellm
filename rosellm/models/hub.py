import os
import re
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download

from rosellm.logging.logger import logger
from rosellm.models.envs import CACHE_DIR, SESSION_ID, _torch_version

REGEX_COMMIT_HASH = re.compile(r"^[0-9a-f]{40}$")


def extract_commit_hash(resolved_file: str):
    resolved_file = str(Path(resolved_file).as_posix())
    search = re.search(r"snapshots/([^/]+)/", resolved_file)
    if search is None:
        return None
    commit_hash = search.groups()[0]
    return commit_hash if REGEX_COMMIT_HASH.match(commit_hash) else None


def try_to_load_from_cache(
    repo_id: str,
    filename: str,
    cache_dir: str = CACHE_DIR,
    repo_type: str = "model",
    revision: str = "main",
):
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


def get_user_agent():
    ua = (
        f"transformers/4.48.2; "  # Hardcoded.
        + f"python/{sys.version.split()[0]}; "
        + f"session_id/{SESSION_ID}; "
        + f"torch/{_torch_version}"
    )
    return ua


def resolve_file(
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
        resolved_file = try_to_load_from_cache(
            model_path,
            filename,
            cache_dir=cache_dir,
            revision=_commit_hash,
        )
        if resolved_file is not None:
            return resolved_file
    user_agent = get_user_agent()
    try:
        logger.info(
            f"downloading {model_path} {filename} from hub, "
            + f"cache_dir: {cache_dir}, "
            + f"force_download: {force_download}, "
            + f"user_agent: {user_agent}"
        )
        return hf_hub_download(
            repo_id=model_path,
            filename=filename,
            cache_dir=cache_dir,
            user_agent=user_agent,
            force_download=force_download,
        )
    except Exception as e:
        raise e


if __name__ == "__main__":
    model_path = "Qwen/Qwen2.5-0.5B"
    filename = "config.json"
    try:
        resolved_file = resolve_file(model_path, filename)
        print(type(resolved_file), resolved_file)
        with open(resolved_file, "r") as f:
            print(f.read())
        commit_hash = extract_commit_hash(resolved_file)
        print("commit_hash:", commit_hash)
        resolved_file = resolve_file(model_path, filename, _commit_hash=commit_hash)
        print(type(resolved_file), resolved_file)
        """
<class 'str'> /home/wine/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987/config.json
{
  "architectures": [
    "Qwen2ForCausalLM"
  ],
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
  "vocab_size": 151936
}

commit_hash: 060db6499f32faf8b98477b0a26969ef7d8b9987
        """
    except Exception as e:
        print("error", e)
