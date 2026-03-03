from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rosellm.rosetrainer.config import GPTConfig
from rosellm.rosetrainer.dataset import build_tokenizer


@dataclass(frozen=True)
class DeepseekV2FP8LoadStats:
    loaded_tensors: int
    dequantized_tensors: int
    skipped_tensors: int


def gpt_config_from_hf_deepseek_v2(hf_cfg: Any) -> GPTConfig:
    # NOTE: DeepSeek-V2 uses q_head_dim != v_head_dim. When using flash-attn-2,
    # the implementation pads V to q_head_dim before caching, so K/V cache head
    # dim is q_head_dim. Roseinfer's KV cache expects a single head_dim, so we
    # encode KV d_model as (num_heads * q_head_dim).
    qk_nope = int(getattr(hf_cfg, "qk_nope_head_dim"))
    qk_rope = int(getattr(hf_cfg, "qk_rope_head_dim"))
    q_head_dim = int(qk_nope + qk_rope)
    num_heads = int(getattr(hf_cfg, "num_attention_heads"))
    kv_d_model = int(num_heads) * int(q_head_dim)
    max_pos = int(getattr(hf_cfg, "max_position_embeddings", 0) or 0)
    if max_pos <= 0:
        # Fallback to a sane default; roseinfer can still clamp via KV budget.
        max_pos = 8192
    return GPTConfig(
        vocab_size=int(getattr(hf_cfg, "vocab_size")),
        max_position_embeddings=max_pos,
        n_layers=int(getattr(hf_cfg, "num_hidden_layers")),
        n_heads=num_heads,
        d_model=kv_d_model,
        d_ff=int(getattr(hf_cfg, "intermediate_size", 0) or 0),
        dropout=0.0,
        activation=str(getattr(hf_cfg, "hidden_act", "silu")),
    )


def _dequantize_fp8_blockwise_weight_2d(
    w_fp8: torch.Tensor,
    *,
    scale_inv: torch.Tensor,
    block_m: int,
    block_n: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    if w_fp8.dim() != 2:
        raise ValueError("expected a 2D weight tensor")
    if scale_inv.dim() != 2:
        raise ValueError("expected a 2D scale_inv tensor")
    m, n = (int(w_fp8.size(0)), int(w_fp8.size(1)))
    m_blocks = int(scale_inv.size(0))
    n_blocks = int(scale_inv.size(1))
    if m_blocks <= 0 or n_blocks <= 0:
        raise ValueError("scale_inv must be non-empty")
    m_pad = int(m_blocks) * int(block_m)
    n_pad = int(n_blocks) * int(block_n)
    if m > m_pad or n > n_pad:
        raise ValueError(
            "scale_inv block grid is smaller than weight shape "
            f"(weight={tuple(w_fp8.shape)}, scale_inv={tuple(scale_inv.shape)}, "
            f"block={(block_m, block_n)})"
        )
    # Basic sanity: scale_inv should match ceil-div block grid.
    exp_m_blocks = (m + block_m - 1) // block_m
    exp_n_blocks = (n + block_n - 1) // block_n
    if m_blocks != exp_m_blocks or n_blocks != exp_n_blocks:
        raise ValueError(
            "scale_inv shape mismatch "
            f"(scale_inv={tuple(scale_inv.shape)}, expected={(exp_m_blocks, exp_n_blocks)})"
        )
    # Vectorized dequant:
    #   weight_fp8: [M, N]
    #   scale_inv:  [M/bs_m, N/bs_n]
    # Reshape into blocks and broadcast multiply per block.
    scale_inv_f = scale_inv.to(dtype=torch.float32, device=w_fp8.device)
    if m != m_pad or n != n_pad:
        w_padded = torch.zeros((m_pad, n_pad), device=w_fp8.device, dtype=w_fp8.dtype)
        w_padded[:m, :n].copy_(w_fp8)
        w_f = w_padded.to(dtype=torch.float32)
    else:
        w_f = w_fp8.to(dtype=torch.float32)
    w4 = w_f.view(m_blocks, block_m, n_blocks, block_n).permute(0, 2, 1, 3)
    w4 = w4 * scale_inv_f.unsqueeze(-1).unsqueeze(-1)
    w_out = (
        w4.permute(0, 2, 1, 3)
        .contiguous()
        .view(m_pad, n_pad)[:m, :n]
        .to(dtype=out_dtype)
    )
    return w_out


def load_deepseek_v2_fp8_from_hf_pretrained(
    model_id: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    ep_size: int = 1,
    attn_implementation: str = "flash_attention_2",
    trust_remote_code: bool = True,
    verbose: bool = True,
    init_on_meta: bool = True,
) -> tuple[torch.nn.Module, GPTConfig, Any, DeepseekV2FP8LoadStats]:
    """Load DeepSeek-V2 FP8 (block-quant) HF model into a bf16/fp16 module.

    This repository stores weights in float8 with per-128x128 `weight_scale_inv`
    tensors. Transformers' built-in fp8 quant loader support is version- and
    backend-dependent, so we do a deterministic dequantization on load.

    The returned model is a HF `DeepseekV2ForCausalLM` instance (remote code),
    moved to `device`/`dtype`.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    if ep_size <= 0:
        raise ValueError("ep_size must be >= 1")

    # Ensure shards are present in the local cache (rank0-only in distributed
    # launches to avoid redundant concurrent downloads).
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            for name in (
                "model.safetensors.index.json",
                "model-00001-of-000004.safetensors",
                "model-00002-of-000004.safetensors",
                "model-00003-of-000004.safetensors",
                "model-00004-of-000004.safetensors",
            ):
                hf_hub_download(model_id, name)
        torch.distributed.barrier()

    cfg = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=bool(trust_remote_code),
    )
    # Expert-parallelism in the remote MoE module assumes ep_size == world_size.
    if int(ep_size) > 1:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            raise RuntimeError("ep_size>1 requires torch.distributed to be initialized")
        world_size = int(torch.distributed.get_world_size())
        if int(ep_size) != world_size:
            raise ValueError(
                f"ep_size must match dist world_size (ep_size={ep_size}, world_size={world_size})"
            )
        setattr(cfg, "ep_size", int(ep_size))

    # Force flash-attn-2 path (also pads V to q_head_dim for cache compatibility).
    # We pass via `attn_implementation` so HF internal autoset behaves.
    tied_params_map: dict[int, dict[torch.device, torch.Tensor]] | None = None
    if init_on_meta:
        try:
            from accelerate import init_empty_weights
            from accelerate.utils import set_module_tensor_to_device
        except Exception:
            init_on_meta = False
        else:
            tied_params_map = {}
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(
                    cfg,
                    trust_remote_code=bool(trust_remote_code),
                    torch_dtype=dtype,
                    attn_implementation=str(attn_implementation),
                )
            model.eval()
    if not init_on_meta:
        model = AutoModelForCausalLM.from_config(
            cfg,
            trust_remote_code=bool(trust_remote_code),
            torch_dtype=dtype,
            attn_implementation=str(attn_implementation),
        )
        model.eval()
        model.to(device=device)

    gpt_cfg = gpt_config_from_hf_deepseek_v2(cfg)
    tokenizer = build_tokenizer(model_id)

    # Load weight index to map tensor -> shard.
    index_path = hf_hub_download(model_id, "model.safetensors.index.json")
    import json

    with open(index_path, "r", encoding="utf-8") as f:
        weight_map: dict[str, str] = json.load(f)["weight_map"]

    # Only load tensors that exist in the instantiated model (this naturally
    # skips non-local experts when ep_size>1).
    model_sd = model.state_dict()
    needed = [k for k in model_sd.keys() if k in weight_map]

    # Group by shard for fewer file opens.
    by_shard: dict[str, list[str]] = {}
    for k in needed:
        by_shard.setdefault(weight_map[k], []).append(k)

    loaded = 0
    deq = 0
    skipped = 0
    loaded_keys: set[str] = set()

    # Quantization config in HF repo uses 128x128 blocks.
    block_m, block_n = 128, 128

    for shard_name, keys in sorted(by_shard.items()):
        shard_path = hf_hub_download(model_id, shard_name)
        # Load directly on target device to speed up dequant.
        with safe_open(shard_path, framework="pt", device=str(device)) as f:
            for key in keys:
                try:
                    t = f.get_tensor(key)
                except Exception:
                    skipped += 1
                    continue

                if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    scale_key = key + "_scale_inv"
                    if scale_key not in weight_map:
                        raise KeyError(f"missing scale_inv tensor for {key}")
                    scale_inv = f.get_tensor(scale_key)
                    w = _dequantize_fp8_blockwise_weight_2d(
                        t,
                        scale_inv=scale_inv,
                        block_m=block_m,
                        block_n=block_n,
                        out_dtype=dtype,
                    )
                    if init_on_meta:
                        assert tied_params_map is not None
                        set_module_tensor_to_device(
                            model,
                            key,
                            device=device,
                            value=w,
                            dtype=dtype,
                            tied_params_map=tied_params_map,
                        )
                    else:
                        model_sd[key].copy_(w)
                    deq += 1
                    loaded += 1
                    loaded_keys.add(key)
                    continue

                # Non-fp8 tensors: copy with best-effort cast.
                if init_on_meta:
                    assert tied_params_map is not None
                    set_module_tensor_to_device(
                        model,
                        key,
                        device=device,
                        value=t.to(dtype=dtype) if t.dtype != dtype else t,
                        dtype=dtype,
                        tied_params_map=tied_params_map,
                    )
                else:
                    if t.dtype != model_sd[key].dtype:
                        t = t.to(dtype=model_sd[key].dtype)
                    model_sd[key].copy_(t)
                loaded += 1
                loaded_keys.add(key)

    # We copied directly into the tensors from `state_dict()`, which are views
    # of the underlying Parameters/Buffers, so no `load_state_dict` call is
    # needed (and would double-copy large weights).
    missing = [k for k in needed if k not in loaded_keys]
    unexpected: list[str] = []
    if verbose:
        if missing:
            print(f"[deepseek-fp8] missing tensors: {len(missing)} (showing up to 20)")
            for k in missing[:20]:
                print("  -", k)
        if unexpected:
            # Expect scale_inv tensors and non-local experts to be unexpected.
            print(f"[deepseek-fp8] unexpected tensors: {len(unexpected)} (showing up to 20)")
            for k in unexpected[:20]:
                print("  -", k)

    stats = DeepseekV2FP8LoadStats(
        loaded_tensors=int(loaded),
        dequantized_tensors=int(deq),
        skipped_tensors=int(skipped),
    )
    return model, gpt_cfg, tokenizer, stats
