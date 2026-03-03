import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


NAME = "triton_v8_dispatch_fixed_best"


@dataclass(frozen=True)
class KernelConfig:
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int


def _pick_config(seq_len: int, head_dim: int, causal: bool) -> KernelConfig:
    if not causal:
        raise ValueError("v8 dispatch only supports causal=True")

    if head_dim <= 64:
        return KernelConfig(
            block_m=64,
            block_n=32,
            num_warps=4,
            num_stages=3,
        )

    if head_dim <= 128:
        if seq_len <= 512:
            return KernelConfig(
                block_m=128,
                block_n=32,
                num_warps=8,
                num_stages=4,
            )
        return KernelConfig(
            block_m=128,
            block_n=64,
            num_warps=8,
            num_stages=3,
        )

    raise ValueError(f"unsupported head_dim={head_dim} for v8 dispatch")


@triton.jit
def _flashattn_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_km,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vm,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    nheads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    sm_scale,
    causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)

    b = pid_bh // nheads
    h = pid_bh - b * nheads

    even_m = (seq_len % BLOCK_M) == 0
    even_n = (seq_len % BLOCK_N) == 0
    even_d = (head_dim % 16) == 0

    q_ptrs = (
        q_ptr
        + b * stride_qb
        + h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    if even_m & even_d:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(
            q_ptrs,
            mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim),
            other=0.0,
        )

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, head_dim), tl.float32)

    end_n = seq_len
    if causal:
        end_n = tl.minimum(seq_len, start_m + BLOCK_M)

    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = (
            k_ptr
            + b * stride_kb
            + h * stride_kh
            + offs_n[:, None] * stride_km
            + offs_d[None, :] * stride_kd
        )
        if even_n & even_d:
            k = tl.load(k_ptrs)
        else:
            k = tl.load(
                k_ptrs,
                mask=(offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim),
                other=0.0,
            )

        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale

        if causal:
            q_pos = offs_m[:, None]
            k_pos = offs_n[None, :]
            qk = tl.where(k_pos > q_pos, -float("inf"), qk)

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        v_ptrs = (
            v_ptr
            + b * stride_vb
            + h * stride_vh
            + offs_n[:, None] * stride_vm
            + offs_d[None, :] * stride_vd
        )
        if even_n & even_d:
            v = tl.load(v_ptrs)
        else:
            v = tl.load(
                v_ptrs,
                mask=(offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim),
                other=0.0,
            )

        pv = tl.dot(p.to(v.dtype), v).to(tl.float32)
        acc = acc * alpha[:, None] + pv
        m_i = m_ij

    acc = acc / l_i[:, None]
    o_ptrs = (
        o_ptr
        + b * stride_ob
        + h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    if even_m & even_d:
        tl.store(o_ptrs, acc.to(q.dtype))
    else:
        tl.store(
            o_ptrs,
            acc.to(q.dtype),
            mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim),
        )


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

    bsz, nheads, seq_len, head_dim = q.shape
    if head_dim % 16 != 0:
        raise ValueError("head_dim must be a multiple of 16")
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    config = _pick_config(seq_len, head_dim, causal)

    o = torch.empty_like(q)
    grid = (
        triton.cdiv(seq_len, config.block_m),
        bsz * nheads,
    )
    _flashattn_fwd_kernel[grid](
        q,
        k,
        v,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        nheads=nheads,
        seq_len=seq_len,
        head_dim=head_dim,
        sm_scale=sm_scale,
        causal=causal,
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return o

