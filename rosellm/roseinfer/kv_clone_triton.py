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
USE_TRITON_KV_CLONE = os.environ.get("ROSELLM_TRITON_KV_CLONE", "1") != "0"


if TRITON_AVAILABLE:

    @triton.jit
    def _kv_clone_blocks_kernel(
        k_cache_ptr,
        v_cache_ptr,
        src_block_idx_ptr,
        dst_block_idx_ptr,
        stride_kb: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kt: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vt: tl.constexpr,
        stride_vd: tl.constexpr,
        BS: tl.constexpr,
        D: tl.constexpr,
        BLOCK: tl.constexpr,
    ) -> None:
        n = tl.program_id(0)  # [0, N)
        h = tl.program_id(1)  # [0, H)
        pid = tl.program_id(2)  # [0, ceil_div(BS*D, BLOCK))

        td = pid * BLOCK + tl.arange(0, BLOCK)
        mask = td < (BS * D)
        t = td // D
        d = td - t * D

        src_blk = tl.load(src_block_idx_ptr + n).to(tl.int32)
        dst_blk = tl.load(dst_block_idx_ptr + n).to(tl.int32)

        k_src_off = src_blk * stride_kb + h * stride_kh + t * stride_kt + d * stride_kd
        v_src_off = src_blk * stride_vb + h * stride_vh + t * stride_vt + d * stride_vd
        k = tl.load(k_cache_ptr + k_src_off, mask=mask, other=0)
        v = tl.load(v_cache_ptr + v_src_off, mask=mask, other=0)

        k_dst_off = dst_blk * stride_kb + h * stride_kh + t * stride_kt + d * stride_kd
        v_dst_off = dst_blk * stride_vb + h * stride_vh + t * stride_vt + d * stride_vd
        tl.store(k_cache_ptr + k_dst_off, k, mask=mask)
        tl.store(v_cache_ptr + v_dst_off, v, mask=mask)


def kv_clone_blocks_triton(
    *,
    k_cache_layer: torch.Tensor,  # [N_BLOCKS, H, BS, D]
    v_cache_layer: torch.Tensor,  # [N_BLOCKS, H, BS, D]
    src_block_idx: torch.Tensor,  # [N] int32 cuda
    dst_block_idx: torch.Tensor,  # [N] int32 cuda
) -> None:
    if not (TRITON_AVAILABLE and USE_TRITON_KV_CLONE):
        raise RuntimeError("Triton KV clone is disabled/unavailable")
    if not (
        k_cache_layer.is_cuda
        and v_cache_layer.is_cuda
        and src_block_idx.is_cuda
        and dst_block_idx.is_cuda
    ):
        raise RuntimeError("kv_clone_blocks_triton requires CUDA tensors")

    n = int(src_block_idx.numel())
    if n == 0:
        return
    if int(dst_block_idx.numel()) != n:
        raise ValueError("src_block_idx and dst_block_idx must have the same shape")

    h = int(k_cache_layer.size(1))
    bs = int(k_cache_layer.size(2))
    d = int(k_cache_layer.size(3))
    block = 256

    grid = (n, h, triton.cdiv(bs * d, block))
    num_warps = 4 if block <= 256 else 8

    _kv_clone_blocks_kernel[grid](
        k_cache_layer,
        v_cache_layer,
        src_block_idx,
        dst_block_idx,
        stride_kb=k_cache_layer.stride(0),
        stride_kh=k_cache_layer.stride(1),
        stride_kt=k_cache_layer.stride(2),
        stride_kd=k_cache_layer.stride(3),
        stride_vb=v_cache_layer.stride(0),
        stride_vh=v_cache_layer.stride(1),
        stride_vt=v_cache_layer.stride(2),
        stride_vd=v_cache_layer.stride(3),
        BS=bs,
        D=d,
        BLOCK=block,
        num_warps=num_warps,
        num_stages=1,
    )
