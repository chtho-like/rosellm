from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]

TRITON_AVAILABLE = triton is not None
USE_TRITON_KV_APPEND = os.environ.get("ROSELLM_TRITON_KV_APPEND", "1") != "0"
TRITON_KV_APPEND_MIN_BATCH = int(
    os.environ.get("ROSELLM_TRITON_KV_APPEND_MIN_BATCH", "128")
)
TRITON_KV_APPEND_FULL_BATCH_MIN_BATCH = int(
    os.environ.get(
        "ROSELLM_TRITON_KV_APPEND_FULL_BATCH_MIN_BATCH", str(TRITON_KV_APPEND_MIN_BATCH)
    )
)


if TRITON_AVAILABLE:

    @triton.jit
    def _kv_append_kernel(
        k_cache_ptr,
        v_cache_ptr,
        key_ptr,
        value_ptr,
        batch_idx_ptr,
        block_idx_ptr,
        pos_ptr,
        stride_kb: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kt: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vt: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_sb: tl.constexpr,
        stride_sh: tl.constexpr,
        stride_sd: tl.constexpr,
        stride_tb: tl.constexpr,
        stride_th: tl.constexpr,
        stride_td: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ) -> None:
        b = tl.program_id(0)  # [0, B_fast)
        pid_d = tl.program_id(1)  # [0, ceil_div(D, BLOCK_D))
        d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = d < D

        src_b = tl.load(batch_idx_ptr + b).to(tl.int32)
        blk = tl.load(block_idx_ptr + b).to(tl.int32)
        pos = tl.load(pos_ptr + b).to(tl.int32)

        for h in tl.static_range(0, H):
            k_src_off = src_b * stride_sb + h * stride_sh + d * stride_sd
            v_src_off = src_b * stride_tb + h * stride_th + d * stride_td
            k = tl.load(key_ptr + k_src_off, mask=mask_d, other=0)
            v = tl.load(value_ptr + v_src_off, mask=mask_d, other=0)

            k_dst_off = (
                blk * stride_kb + h * stride_kh + pos * stride_kt + d * stride_kd
            )
            v_dst_off = (
                blk * stride_vb + h * stride_vh + pos * stride_vt + d * stride_vd
            )
            tl.store(k_cache_ptr + k_dst_off, k, mask=mask_d)
            tl.store(v_cache_ptr + v_dst_off, v, mask=mask_d)

    @triton.jit
    def _kv_append_full_batch_kernel(
        k_cache_ptr,
        v_cache_ptr,
        key_ptr,
        value_ptr,
        block_idx_ptr,
        pos,
        stride_kb: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kt: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vt: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_sb: tl.constexpr,
        stride_sh: tl.constexpr,
        stride_sd: tl.constexpr,
        stride_tb: tl.constexpr,
        stride_th: tl.constexpr,
        stride_td: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ) -> None:
        b = tl.program_id(0)  # [0, B)
        h = tl.program_id(1)  # [0, H)
        pid_d = tl.program_id(2)  # [0, ceil_div(D, BLOCK_D))
        d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = d < D

        blk = tl.load(block_idx_ptr + b).to(tl.int32)
        p = tl.full((), pos, tl.int32)

        k_src_off = b * stride_sb + h * stride_sh + d * stride_sd
        v_src_off = b * stride_tb + h * stride_th + d * stride_td
        k = tl.load(key_ptr + k_src_off, mask=mask_d, other=0)
        v = tl.load(value_ptr + v_src_off, mask=mask_d, other=0)

        k_dst_off = blk * stride_kb + h * stride_kh + p * stride_kt + d * stride_kd
        v_dst_off = blk * stride_vb + h * stride_vh + p * stride_vt + d * stride_vd
        tl.store(k_cache_ptr + k_dst_off, k, mask=mask_d)
        tl.store(v_cache_ptr + v_dst_off, v, mask=mask_d)


def kv_append_triton(
    *,
    k_cache_layer: torch.Tensor,  # [N_BLOCKS, H, BS, D]
    v_cache_layer: torch.Tensor,  # [N_BLOCKS, H, BS, D]
    key_new: torch.Tensor,  # [B, H, D]
    value_new: torch.Tensor,  # [B, H, D]
    batch_idx: torch.Tensor,  # [B_fast] int32 cuda
    block_idx: torch.Tensor,  # [B_fast] int32 cuda
    pos: torch.Tensor,  # [B_fast] int32 cuda
) -> None:
    if not (TRITON_AVAILABLE and USE_TRITON_KV_APPEND):
        raise RuntimeError("Triton KV append is disabled/unavailable")
    if not (
        k_cache_layer.is_cuda
        and v_cache_layer.is_cuda
        and key_new.is_cuda
        and value_new.is_cuda
    ):
        raise RuntimeError("kv_append_triton requires CUDA tensors")

    B_fast = int(batch_idx.numel())
    if B_fast == 0:
        return

    H = int(key_new.size(1))
    D = int(key_new.size(2))
    block_d = 128 if D >= 128 else 64
    grid = (B_fast, triton.cdiv(D, block_d))
    if D <= 64:
        num_warps = 2
    elif D <= 128:
        num_warps = 4
    else:
        num_warps = 8

    _kv_append_kernel[grid](
        k_cache_layer,
        v_cache_layer,
        key_new,
        value_new,
        batch_idx,
        block_idx,
        pos,
        stride_kb=k_cache_layer.stride(0),
        stride_kh=k_cache_layer.stride(1),
        stride_kt=k_cache_layer.stride(2),
        stride_kd=k_cache_layer.stride(3),
        stride_vb=v_cache_layer.stride(0),
        stride_vh=v_cache_layer.stride(1),
        stride_vt=v_cache_layer.stride(2),
        stride_vd=v_cache_layer.stride(3),
        stride_sb=key_new.stride(0),
        stride_sh=key_new.stride(1),
        stride_sd=key_new.stride(2),
        stride_tb=value_new.stride(0),
        stride_th=value_new.stride(1),
        stride_td=value_new.stride(2),
        H=H,
        D=D,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=1,
    )


def kv_append_triton_full_batch(
    *,
    k_cache_layer: torch.Tensor,  # [N_BLOCKS, H, BS, D]
    v_cache_layer: torch.Tensor,  # [N_BLOCKS, H, BS, D]
    key_new: torch.Tensor,  # [B, H, D]
    value_new: torch.Tensor,  # [B, H, D]
    block_idx: torch.Tensor,  # [B] int32 cuda
    pos: int,
) -> None:
    if not (TRITON_AVAILABLE and USE_TRITON_KV_APPEND):
        raise RuntimeError("Triton KV append is disabled/unavailable")
    if not (
        k_cache_layer.is_cuda
        and v_cache_layer.is_cuda
        and key_new.is_cuda
        and value_new.is_cuda
        and block_idx.is_cuda
    ):
        raise RuntimeError("kv_append_triton_full_batch requires CUDA tensors")

    B = int(key_new.size(0))
    if B == 0:
        return
    if int(block_idx.numel()) != B:
        raise ValueError("block_idx must have shape [B]")

    H = int(key_new.size(1))
    D = int(key_new.size(2))
    block_d = 128 if D >= 128 else 64
    grid = (B, H, triton.cdiv(D, block_d))
    if D <= 64:
        num_warps = 2
    elif D <= 128:
        num_warps = 4
    else:
        num_warps = 8

    _kv_append_full_batch_kernel[grid](
        k_cache_layer,
        v_cache_layer,
        key_new,
        value_new,
        block_idx,
        pos=int(pos),
        stride_kb=k_cache_layer.stride(0),
        stride_kh=k_cache_layer.stride(1),
        stride_kt=k_cache_layer.stride(2),
        stride_kd=k_cache_layer.stride(3),
        stride_vb=v_cache_layer.stride(0),
        stride_vh=v_cache_layer.stride(1),
        stride_vt=v_cache_layer.stride(2),
        stride_vd=v_cache_layer.stride(3),
        stride_sb=key_new.stride(0),
        stride_sh=key_new.stride(1),
        stride_sd=key_new.stride(2),
        stride_tb=value_new.stride(0),
        stride_th=value_new.stride(1),
        stride_td=value_new.stride(2),
        D=D,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=1,
    )
