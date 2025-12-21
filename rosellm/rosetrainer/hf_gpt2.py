from __future__ import annotations

from typing import Any

import torch

from rosellm.rosetrainer.config import GPTConfig
from rosellm.rosetrainer.dataset import build_tokenizer
from rosellm.rosetrainer.model import GPTModel


def _t(w: torch.Tensor) -> torch.Tensor:
    return w.t().contiguous()


def gpt_config_from_hf_gpt2(hf_cfg: Any) -> GPTConfig:
    d_model = int(hf_cfg.n_embd)
    n_inner = getattr(hf_cfg, "n_inner", None)
    d_ff = int(n_inner) if n_inner is not None else 4 * d_model
    max_pos = int(
        getattr(hf_cfg, "n_positions", getattr(hf_cfg, "max_position_embeddings", 1024))
    )
    activation = str(getattr(hf_cfg, "activation_function", "gelu_new"))
    dropout = float(getattr(hf_cfg, "resid_pdrop", 0.0))
    return GPTConfig(
        vocab_size=int(hf_cfg.vocab_size),
        max_position_embeddings=max_pos,
        n_layers=int(hf_cfg.n_layer),
        n_heads=int(hf_cfg.n_head),
        d_model=d_model,
        d_ff=d_ff,
        dropout=dropout,
        activation=activation,
    )


def convert_hf_gpt2_state_dict(
    hf_sd: dict[str, torch.Tensor],
    *,
    n_layers: int,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}

    # embeddings
    out["token_embedding.weight"] = hf_sd["transformer.wte.weight"]
    out["position_embedding.weight"] = hf_sd["transformer.wpe.weight"]

    # final layernorm
    out["ln_f.weight"] = hf_sd["transformer.ln_f.weight"]
    out["ln_f.bias"] = hf_sd["transformer.ln_f.bias"]

    # tie lm_head
    out["lm_head.weight"] = out["token_embedding.weight"]

    # blocks
    for i in range(n_layers):
        h = f"transformer.h.{i}."
        b = f"blocks.{i}."

        # layer norms
        out[b + "ln1.weight"] = hf_sd[h + "ln_1.weight"]
        out[b + "ln1.bias"] = hf_sd[h + "ln_1.bias"]
        out[b + "ln2.weight"] = hf_sd[h + "ln_2.weight"]
        out[b + "ln2.bias"] = hf_sd[h + "ln_2.bias"]

        # attention: HF GPT2 uses Conv1D (weight shape [in, out]), ours is Linear (weight [out, in])
        out[b + "attn.qkv_proj.weight"] = _t(hf_sd[h + "attn.c_attn.weight"])
        out[b + "attn.qkv_proj.bias"] = hf_sd[h + "attn.c_attn.bias"]
        out[b + "attn.out_proj.weight"] = _t(hf_sd[h + "attn.c_proj.weight"])
        out[b + "attn.out_proj.bias"] = hf_sd[h + "attn.c_proj.bias"]

        # MLP: same Conv1D -> Linear transpose rule
        out[b + "mlp.fc1.weight"] = _t(hf_sd[h + "mlp.c_fc.weight"])
        out[b + "mlp.fc1.bias"] = hf_sd[h + "mlp.c_fc.bias"]
        out[b + "mlp.fc2.weight"] = _t(hf_sd[h + "mlp.c_proj.weight"])
        out[b + "mlp.fc2.bias"] = hf_sd[h + "mlp.c_proj.bias"]

    return out


def load_gpt2_from_hf_pretrained(
    model_id: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[GPTModel, GPTConfig, Any]:
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
    if getattr(hf_cfg, "model_type", None) != "gpt2":
        raise ValueError(
            f"only HF GPT2 is supported, got model_type={getattr(hf_cfg, 'model_type', None)}"
        )

    cfg = gpt_config_from_hf_gpt2(hf_cfg)
    model = GPTModel(cfg).to(device=device, dtype=dtype)
    mapped = convert_hf_gpt2_state_dict(hf.state_dict(), n_layers=cfg.n_layers)
    mapped = {k: v.to(device=device, dtype=dtype) for k, v in mapped.items()}
    model.load_state_dict(mapped, strict=True)
    tokenizer = build_tokenizer(model_id)
    return model, cfg, tokenizer
