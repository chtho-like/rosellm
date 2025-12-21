from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]

TRITON_AVAILABLE = triton is not None


@dataclass(frozen=True)
class PagedKVCache:
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_tables: list[torch.Tensor]
    context_lens: torch.Tensor
    block_size: int


def paged_attention_decode_ref(
    q: torch.Tensor,  # [B, H, D]
    k_new: torch.Tensor,  # [B, H, D]
    v_new: torch.Tensor,  # [B, H, D]
    k_cache_layer: torch.Tensor,  # [N_BLOCKS, H, BS, D]
    v_cache_layer: torch.Tensor,  # [N_BLOCKS, H, BS, D], BS: block size
    block_table: torch.Tensor,  # [B, N_LOGICAL_BLOCKS]
    context_lens: torch.Tensor,  # [B]
    *,
    scale: float,
    block_size: int,
) -> torch.Tensor:  # [B, H, D]
    assert q.dim() == 3
    assert q.shape == k_new.shape == v_new.shape
    assert k_cache_layer.dim() == 4 and v_cache_layer.dim() == 4
    assert block_table.dim() == 2
    assert context_lens.dim() == 1
    assert k_cache_layer.size(2) == block_size
    device = q.device
    bsz, n_heads, head_dim = q.shape
    num_blocks = block_table.size(1)
    q_f = q.float()
    k_new_f = k_new.float()  # [B, H, D]
    v_new_f = v_new.float()  # [B, H, D]
    scores_cur = (q_f * k_new_f).sum(dim=-1) * scale  # [B, H]
    # m: max logits so far, [B, H]
    m = scores_cur
    # l: exp-sum, [B, H]
    l = torch.ones((bsz, n_heads), device=device, dtype=torch.float32)
    # o: weighted sum, [B, H, D]
    o = v_new_f
    pos = torch.arange(block_size, device=device).view(1, 1, block_size)
    for logical_block in range(num_blocks):
        block_ids = block_table[:, logical_block]  # [B]
        k_blk = k_cache_layer[block_ids].float()  # [B, H, BS, D]
        v_blk = v_cache_layer[block_ids].float()  # [B, H, BS, D]
        start = logical_block * block_size
        valid = (context_lens - start).clamp(min=0, max=block_size)  # [B]
        mask = pos < valid.view(bsz, 1, 1)  # [B, 1, BS] -> broadcast on head dim
        scores = torch.einsum("bhd,bhtd->bht", q_f, k_blk) * scale
        scores = scores.masked_fill(~mask, -float("inf"))  # [B, H, BS]
        m_ij = scores.max(dim=-1).values  # [B, H]
        m_new = torch.maximum(m, m_ij)  # [B, H]
        exp_scale_old = torch.exp(m - m_new)  # [B, H]
        exp_scores = torch.exp(scores - m_new.unsqueeze(-1))  # [B, H, BS]
        l = l * exp_scale_old + exp_scores.sum(dim=-1)  # [B, H]
        o = o * exp_scale_old.unsqueeze(-1) + torch.einsum(
            "bht,bhtd->bhd", exp_scores, v_blk
        )
        m = m_new
    out = (o / l.unsqueeze(-1)).to(dtype=q.dtype)  # [B, H, D]
    return out


if TRITON_AVAILABLE:

    @triton.jit
    def _paged_attn_decode_kernel(
        out_ptr,
        q_ptr,
        k_new_ptr,
        v_new_ptr,
        k_cache_ptr,
        v_cache_ptr,
        block_table_ptr,
        context_lens_ptr,
        stride_kcb: tl.constexpr,
        stride_kch: tl.constexpr,
        stride_kct: tl.constexpr,
        stride_kcd: tl.constexpr,
        stride_vcb: tl.constexpr,
        stride_vch: tl.constexpr,
        stride_vct: tl.constexpr,
        stride_vcd: tl.constexpr,
        scale: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        MAX_BLOCKS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H
        d = tl.arange(0, D)
        base = (b * H + h) * D + d
        q = tl.load(q_ptr + base, mask=d < D, other=0.0).to(tl.float32)
        k_new = tl.load(k_new_ptr + base, mask=d < D, other=0.0).to(tl.float32)
        v_new = tl.load(v_new_ptr + base, mask=d < D, other=0.0).to(tl.float32)
        score_cur = tl.sum(q * k_new, axis=0) * scale
        m = score_cur
        l = 1.0
        acc = v_new
        context_len = tl.load(context_lens_ptr + b).to(tl.int32)
        t = tl.arange(0, BLOCK_SIZE)
        for lb in tl.static_range(0, MAX_BLOCKS):
            start = lb * BLOCK_SIZE
            tok_pos = start + t
            tok_mask = tok_pos < context_len
            has_any = start < context_len
            block_id = tl.load(
                block_table_ptr + b * MAX_BLOCKS + lb,
                mask=has_any,
                other=0,
            ).to(tl.int32)
            k_ptrs = (
                k_cache_ptr
                + block_id * stride_kcb
                + h * stride_kch
                + t[:, None] * stride_kct
                + d[None, :] * stride_kcd
            )
            v_ptrs = (
                v_cache_ptr
                + block_id * stride_vcb
                + h * stride_vch
                + t[:, None] * stride_vct
                + d[None, :] * stride_vcd
            )
            k = tl.load(
                k_ptrs,
                mask=tok_mask[:, None] & (d[None, :] < D),
                other=0.0,
            ).to(tl.float32)
            v = tl.load(
                v_ptrs,
                mask=tok_mask[:, None] & (d[None, :] < D),
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(k * q[None, :], axis=1) * scale
            scores = tl.where(tok_mask, scores, -float("inf"))
            m_ij = tl.max(scores, axis=0)
            m_new = tl.maximum(m, m_ij)
            exp_scale_old = tl.exp(m - m_new)
            exp_scores = tl.exp(scores - m_new)
            l = l * exp_scale_old + tl.sum(exp_scores, axis=0)
            acc = acc * exp_scale_old + tl.sum(exp_scores[:, None] * v, axis=0)
            m = m_new
        out = acc / l
        tl.store(out_ptr + base, out, mask=d < D)


def paged_attention_decode_triton(
    q: torch.Tensor,  # [B, H, D]
    k_new: torch.Tensor,  # [B, H, D]
    v_new: torch.Tensor,  # [B, H, D]
    k_cache_layer: torch.Tensor,  # [N_BLOCKS, H, BS, D]
    v_cache_layer: torch.Tensor,  # [N_BLOCKS, H, BS, D]
    block_table: torch.Tensor,  # [B, MAX_BLOCKS] int32 cuda
    context_lens: torch.Tensor,  # [B] int32 cuda
    *,
    scale: float,
    block_size: int,
) -> torch.Tensor:  # [B, H, D]
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "paged_attention_decode_triton requires Triton; install it to use --paged-attn"
        )
    assert q.is_cuda
    q = q.contiguous()
    k_new = k_new.contiguous()
    v_new = v_new.contiguous()
    if block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)
    if context_lens.dtype != torch.int32:
        context_lens = context_lens.to(torch.int32)
    B, H, D = q.shape
    out = torch.empty_like(q)
    grid = (B * H,)
    _paged_attn_decode_kernel[grid](
        out,
        q,
        k_new,
        v_new,
        k_cache_layer,
        v_cache_layer,
        block_table,
        context_lens,
        stride_kcb=k_cache_layer.stride(0),
        stride_kch=k_cache_layer.stride(1),
        stride_kct=k_cache_layer.stride(2),
        stride_kcd=k_cache_layer.stride(3),
        stride_vcb=v_cache_layer.stride(0),
        stride_vch=v_cache_layer.stride(1),
        stride_vct=v_cache_layer.stride(2),
        stride_vcd=v_cache_layer.stride(3),
        scale=scale,
        H=H,
        D=D,
        BLOCK_SIZE=block_size,
        MAX_BLOCKS=block_table.size(1),
        num_warps=4,
    )
    return out
