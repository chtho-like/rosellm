import math

import torch


NAME = "torch_sdpa_flash_pytorch"


def flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    sm_scale: float | None = None,
) -> torch.Tensor:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/v must be rank-4: (B, H, S, D)")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q/k/v must have the same shape")
    if q.device.type != "cuda":
        raise ValueError("q must be on CUDA")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("q must be fp16 or bf16")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("q/k/v must be contiguous")

    head_dim = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    op = torch.ops.aten._scaled_dot_product_flash_attention.default
    out, *_ = op(
        q,
        k,
        v,
        0.0,  # dropout_p
        causal,
        False,  # return_debug_mask
        scale=sm_scale,
    )
    return out

