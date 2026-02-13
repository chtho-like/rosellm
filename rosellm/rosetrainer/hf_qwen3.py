from __future__ import annotations

from typing import Any

import torch

from rosellm.rosetrainer.config import GPTConfig
from rosellm.rosetrainer.dataset import build_tokenizer
from rosellm.rosetrainer.qwen3 import Qwen3ForCausalLM


def gpt_config_from_hf_qwen3(hf_cfg: Any) -> GPTConfig:
    return GPTConfig(
        vocab_size=int(hf_cfg.vocab_size),
        max_position_embeddings=int(hf_cfg.max_position_embeddings),
        n_layers=int(hf_cfg.num_hidden_layers),
        # KV cache stores K/V heads for GQA.
        n_heads=int(hf_cfg.num_key_value_heads),
        d_model=int(hf_cfg.hidden_size),
        d_ff=int(hf_cfg.intermediate_size),
        dropout=0.0,
        activation=str(getattr(hf_cfg, "hidden_act", "silu")),
    )


def load_qwen3_from_hf_pretrained(
    model_id: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Qwen3ForCausalLM, GPTConfig, Any]:
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"torch_dtype": dtype}
    try:
        import accelerate  # noqa: F401
    except ImportError:
        pass
    else:
        kwargs["low_cpu_mem_usage"] = True
    hf = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    hf.eval()
    hf_cfg = hf.config
    if getattr(hf_cfg, "model_type", None) != "qwen3":
        raise ValueError(
            "only HF Qwen3 is supported, got model_type="
            f"{getattr(hf_cfg, 'model_type', None)}"
        )

    cfg = gpt_config_from_hf_qwen3(hf_cfg)
    model = Qwen3ForCausalLM(hf_cfg).to(device=device, dtype=dtype)
    model.load_state_dict(hf.state_dict(), strict=True)
    tokenizer = build_tokenizer(model_id)
    return model, cfg, tokenizer

