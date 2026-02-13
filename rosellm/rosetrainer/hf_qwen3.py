from __future__ import annotations

from typing import Any

import torch

from rosellm.rosetrainer.config import GPTConfig
from rosellm.rosetrainer.dataset import build_tokenizer
from rosellm.rosetrainer.qwen3 import Qwen3ForCausalLM


def gpt_config_from_hf_qwen3(hf_cfg: Any) -> GPTConfig:
    head_dim = int(getattr(hf_cfg, "head_dim", 0) or 0)
    if head_dim <= 0:
        head_dim = int(hf_cfg.hidden_size) // int(hf_cfg.num_attention_heads)
    kv_heads = int(hf_cfg.num_key_value_heads)
    # NOTE: InferenceEngine's KV cache sizing assumes:
    #   head_dim = config.d_model // config.n_heads
    # For GQA models, we store KV cache with Hkv heads, not Hq.
    # So we encode d_model as (Hkv * head_dim) for KV tensors.
    kv_d_model = int(kv_heads) * int(head_dim)
    return GPTConfig(
        vocab_size=int(hf_cfg.vocab_size),
        max_position_embeddings=int(hf_cfg.max_position_embeddings),
        n_layers=int(hf_cfg.num_hidden_layers),
        n_heads=kv_heads,
        d_model=kv_d_model,
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
