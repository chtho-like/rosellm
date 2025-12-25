from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

try:
    import flashinfer  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    flashinfer = None  # type: ignore[assignment]

try:
    from flash_attn import flash_attn_varlen_func  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    flash_attn_varlen_func = None  # type: ignore[assignment]


_DEFAULT_FLASHINFER_WORKSPACE_BYTES = 128 * 1024 * 1024


@dataclass
class _FlashInferPrefillCache:
    device: torch.device
    workspace: torch.Tensor
    wrapper: Any
    last_sig: tuple[Any, ...] | None = None


_FLASHINFER_PREFILL_CACHES: dict[int, _FlashInferPrefillCache] = {}


def _flashinfer_prefill_plan(
    *,
    device: torch.device,
    batch_size: int,
    lengths: torch.Tensor,  # [B] int32 cuda
    num_heads: int,
    head_dim: int,
    sm_scale: float,
    causal: bool,
    q_dtype: torch.dtype,
) -> Any:
    if flashinfer is None:
        raise RuntimeError(
            "flashinfer is not installed; install it to use attn_backend='flashinfer'"
        )
    if device.type != "cuda":
        raise RuntimeError("flashinfer attention backend requires CUDA")
    if q_dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            f"flashinfer attention backend requires fp16/bf16, got dtype={q_dtype}"
        )
    if lengths.dtype != torch.int32:
        raise ValueError("lengths must be int32")
    if lengths.dim() != 1 or int(lengths.numel()) != int(batch_size):
        raise ValueError("lengths shape mismatch")

    cache = _FLASHINFER_PREFILL_CACHES.get(device.index or 0)
    if cache is None or cache.device != device:
        workspace = torch.empty(
            (_DEFAULT_FLASHINFER_WORKSPACE_BYTES,),
            device=device,
            dtype=torch.uint8,
        )
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace,
            kv_layout="NHD",
        )
        cache = _FlashInferPrefillCache(
            device=device,
            workspace=workspace,
            wrapper=wrapper,
        )
        _FLASHINFER_PREFILL_CACHES[device.index or 0] = cache

    lengths_cpu = lengths.to(device="cpu", non_blocking=True)
    sig = (
        int(batch_size),
        tuple(int(x) for x in lengths_cpu.tolist()),
        int(num_heads),
        int(head_dim),
        bool(causal),
        float(sm_scale),
        str(q_dtype),
    )
    if cache.last_sig != sig:
        qo_indptr = torch.zeros(
            (batch_size + 1,),
            device=device,
            dtype=torch.int32,
        )
        qo_indptr[1:].copy_(
            torch.cumsum(lengths, dim=0, dtype=torch.int32),
            non_blocking=True,
        )
        # self-attention: kv has the same ragged layout.
        kv_indptr = qo_indptr
        cache.wrapper.plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            num_qo_heads=int(num_heads),
            num_kv_heads=int(num_heads),
            head_dim_qk=int(head_dim),
            causal=bool(causal),
            sm_scale=float(sm_scale),
            q_data_type=q_dtype,
            kv_data_type=q_dtype,
        )
        cache.last_sig = sig
    return cache.wrapper


def _ragged_token_indices(
    *,
    attention_mask: Optional[torch.Tensor],  # [B, T] or None
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if attention_mask is None:
        lengths = torch.full(
            (batch_size,),
            int(seq_len),
            device=device,
            dtype=torch.int32,
        )
        idx = torch.arange(
            int(batch_size * seq_len),
            device=device,
            dtype=torch.long,
        )
        return idx, lengths

    mask = attention_mask.to(device=device, dtype=torch.bool, non_blocking=True)
    if (
        mask.dim() != 2
        or int(mask.size(0)) != int(batch_size)
        or int(mask.size(1)) != int(seq_len)
    ):
        raise ValueError("attention_mask must have shape [B, T]")
    lengths = mask.sum(dim=1, dtype=torch.int32)
    idx = mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1).to(dtype=torch.long)
    if idx.numel() != int(lengths.sum().item()):
        raise RuntimeError("ragged packing mismatch (mask nonzero != sum(lengths))")
    return idx, lengths


def prefill_attention_flashinfer(
    *,
    q: torch.Tensor,  # [B, H, T, D]
    k: torch.Tensor,  # [B, H, T, D]
    v: torch.Tensor,  # [B, H, T, D]
    attention_mask: Optional[torch.Tensor],  # [B, T] or None
    sm_scale: float,
    causal: bool,
) -> torch.Tensor:
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q/k/v must be 4D [B, H, T, D]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q/k/v must have the same shape")
    bsz, n_heads, seq_len, head_dim = q.shape
    device = q.device
    if attention_mask is None:
        lengths = torch.full(
            (int(bsz),),
            int(seq_len),
            device=device,
            dtype=torch.int32,
        )
        idx = None
    else:
        idx, lengths = _ragged_token_indices(
            attention_mask=attention_mask,
            batch_size=int(bsz),
            seq_len=int(seq_len),
            device=device,
        )
        if idx.numel() == 0:
            return torch.zeros_like(q)

    wrapper = _flashinfer_prefill_plan(
        device=device,
        batch_size=int(bsz),
        lengths=lengths,
        num_heads=int(n_heads),
        head_dim=int(head_dim),
        sm_scale=float(sm_scale),
        causal=bool(causal),
        q_dtype=q.dtype,
    )

    # [B, T, H, D] -> [B*T, H, D]
    q_flat = (
        q.permute(0, 2, 1, 3)
        .contiguous()
        .view(int(bsz * seq_len), int(n_heads), int(head_dim))
    )
    k_flat = (
        k.permute(0, 2, 1, 3)
        .contiguous()
        .view(int(bsz * seq_len), int(n_heads), int(head_dim))
    )
    v_flat = (
        v.permute(0, 2, 1, 3)
        .contiguous()
        .view(int(bsz * seq_len), int(n_heads), int(head_dim))
    )

    if idx is None:
        o_flat = wrapper.run(q_flat, k_flat, v_flat)
    else:
        q_ragged = q_flat.index_select(0, idx)
        k_ragged = k_flat.index_select(0, idx)
        v_ragged = v_flat.index_select(0, idx)

        o_ragged = wrapper.run(q_ragged, k_ragged, v_ragged)
        o_flat = q_flat.new_zeros((int(bsz * seq_len), int(n_heads), int(head_dim)))
        o_flat.index_copy_(0, idx, o_ragged)
    # [B, T, H, D] -> [B, H, T, D]
    return (
        o_flat.view(int(bsz), int(seq_len), int(n_heads), int(head_dim))
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def prefill_attention_flashattn(
    *,
    q: torch.Tensor,  # [B, H, T, D]
    k: torch.Tensor,  # [B, H, T, D]
    v: torch.Tensor,  # [B, H, T, D]
    attention_mask: Optional[torch.Tensor],  # [B, T] or None
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    if flash_attn_varlen_func is None:
        raise RuntimeError(
            "flash-attn is not installed; install it to use attn_backend='flashattn'"
        )
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q/k/v must be 4D [B, H, T, D]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q/k/v must have the same shape")
    bsz, n_heads, seq_len, head_dim = q.shape
    device = q.device
    if device.type != "cuda":
        raise RuntimeError("flash-attn attention backend requires CUDA")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            f"flash-attn attention backend requires fp16/bf16, got dtype={q.dtype}"
        )
    if attention_mask is None:
        lengths = torch.full(
            (int(bsz),),
            int(seq_len),
            device=device,
            dtype=torch.int32,
        )
        idx = None
    else:
        idx, lengths = _ragged_token_indices(
            attention_mask=attention_mask,
            batch_size=int(bsz),
            seq_len=int(seq_len),
            device=device,
        )
        if idx.numel() == 0:
            return torch.zeros_like(q)
    cu_seqlens = torch.zeros(
        (int(bsz) + 1,),
        device=device,
        dtype=torch.int32,
    )
    cu_seqlens[1:].copy_(
        torch.cumsum(lengths, dim=0, dtype=torch.int32),
        non_blocking=True,
    )
    max_seqlen = int(lengths.max().item())

    q_flat = (
        q.permute(0, 2, 1, 3)
        .contiguous()
        .view(int(bsz * seq_len), int(n_heads), int(head_dim))
    )
    k_flat = (
        k.permute(0, 2, 1, 3)
        .contiguous()
        .view(int(bsz * seq_len), int(n_heads), int(head_dim))
    )
    v_flat = (
        v.permute(0, 2, 1, 3)
        .contiguous()
        .view(int(bsz * seq_len), int(n_heads), int(head_dim))
    )

    if idx is None:
        o_flat = flash_attn_varlen_func(
            q_flat,
            k_flat,
            v_flat,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=bool(causal),
            softmax_scale=float(softmax_scale),
        )
    else:
        q_ragged = q_flat.index_select(0, idx)
        k_ragged = k_flat.index_select(0, idx)
        v_ragged = v_flat.index_select(0, idx)
        o_ragged = flash_attn_varlen_func(
            q_ragged,
            k_ragged,
            v_ragged,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=bool(causal),
            softmax_scale=float(softmax_scale),
        )
        o_flat = q_flat.new_zeros((int(bsz * seq_len), int(n_heads), int(head_dim)))
        o_flat.index_copy_(0, idx, o_ragged)

    return (
        o_flat.view(int(bsz), int(seq_len), int(n_heads), int(head_dim))
        .permute(0, 2, 1, 3)
        .contiguous()
    )
