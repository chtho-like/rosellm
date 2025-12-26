from __future__ import annotations

import os
from typing import Final

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


TRITON_AVAILABLE: Final[bool] = triton is not None

USE_TRITON_FUSED_LAYERNORM: Final[bool] = (
    os.environ.get("ROSELLM_TRITON_FUSED_LAYERNORM", "1") != "0"
)
USE_TRITON_FUSED_ADD_LAYERNORM: Final[bool] = (
    os.environ.get("ROSELLM_TRITON_FUSED_ADD_LAYERNORM", "1") != "0"
)


def _can_use_triton_layernorm(x: torch.Tensor) -> bool:
    if not (TRITON_AVAILABLE and USE_TRITON_FUSED_LAYERNORM):
        return False
    if not x.is_cuda:
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if not x.is_contiguous():
        return False
    if x.dim() < 1:
        return False
    return True


def _can_use_triton_add_layernorm(x: torch.Tensor, residual: torch.Tensor) -> bool:
    if not (TRITON_AVAILABLE and USE_TRITON_FUSED_ADD_LAYERNORM):
        return False
    if not (x.is_cuda and residual.is_cuda):
        return False
    if x.dtype != residual.dtype:
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if x.shape != residual.shape:
        return False
    if not (x.is_contiguous() and residual.is_contiguous()):
        return False
    if x.dim() < 1:
        return False
    return True


def _pick_block_size(d: int) -> int:
    # Keep the inner tile reasonably sized. 1024 is a good default for typical
    # hidden sizes (768/1024/1280/1600) while supporting >1024 via a short loop.
    if d <= 128:
        return 128
    if d <= 256:
        return 256
    if d <= 512:
        return 512
    return 1024


def _pick_num_warps(d: int) -> int:
    if d <= 128:
        return 2
    if d <= 512:
        return 4
    return 8


if TRITON_AVAILABLE:

    @triton.jit
    def _layer_norm_fwd_kernel(
        x_ptr,
        y_ptr,
        w_ptr,
        b_ptr,
        n_rows,
        eps: tl.constexpr,
        d: tl.constexpr,
        block: tl.constexpr,
        n_blocks: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        if row >= n_rows:
            return
        row_start = row * d
        offs = tl.arange(0, block)

        mean = tl.zeros((), dtype=tl.float32)
        mean2 = tl.zeros((), dtype=tl.float32)
        for i in tl.static_range(0, n_blocks):
            cols = i * block + offs
            mask = cols < d
            x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
            mean += tl.sum(x, axis=0)
            mean2 += tl.sum(x * x, axis=0)

        inv_d = 1.0 / tl.full((), d, tl.float32)
        mu = mean * inv_d
        var = mean2 * inv_d - mu * mu
        rstd = tl.rsqrt(var + eps)

        for i in tl.static_range(0, n_blocks):
            cols = i * block + offs
            mask = cols < d
            x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            y = (x - mu) * rstd * w + b
            tl.store(y_ptr + row_start + cols, y, mask=mask)

    @triton.jit
    def _add_layer_norm_fwd_inplace_kernel(
        x_ptr,
        r_ptr,
        y_ptr,
        w_ptr,
        b_ptr,
        n_rows,
        eps: tl.constexpr,
        d: tl.constexpr,
        block: tl.constexpr,
        n_blocks: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        if row >= n_rows:
            return
        row_start = row * d
        offs = tl.arange(0, block)

        mean = tl.zeros((), dtype=tl.float32)
        mean2 = tl.zeros((), dtype=tl.float32)
        for i in tl.static_range(0, n_blocks):
            cols = i * block + offs
            mask = cols < d
            x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
            r = tl.load(r_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
            z = x + r
            mean += tl.sum(z, axis=0)
            mean2 += tl.sum(z * z, axis=0)

        inv_d = 1.0 / tl.full((), d, tl.float32)
        mu = mean * inv_d
        var = mean2 * inv_d - mu * mu
        rstd = tl.rsqrt(var + eps)

        for i in tl.static_range(0, n_blocks):
            cols = i * block + offs
            mask = cols < d
            x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
            r = tl.load(r_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
            z = x + r
            tl.store(x_ptr + row_start + cols, z, mask=mask)
            w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            y = (z - mu) * rstd * w + b
            tl.store(y_ptr + row_start + cols, y, mask=mask)


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    d = int(x.shape[-1])
    if d <= 0:
        raise ValueError("layer_norm expects a non-empty last dimension")
    if (
        _can_use_triton_layernorm(x)
        and weight.is_cuda
        and bias.is_cuda
        and weight.is_contiguous()
        and bias.is_contiguous()
        and weight.numel() == d
        and bias.numel() == d
    ):
        out = torch.empty_like(x)
        x2 = x.view(-1, d)
        y2 = out.view(-1, d)
        n_rows = int(x2.size(0))
        block = _pick_block_size(d)
        n_blocks = (d + block - 1) // block
        num_warps = _pick_num_warps(d)
        grid = (n_rows,)
        _layer_norm_fwd_kernel[grid](
            x2,
            y2,
            weight,
            bias,
            n_rows,
            eps=float(eps),
            d=d,
            block=block,
            n_blocks=n_blocks,
            num_warps=num_warps,
            num_stages=1,
        )
        return out
    return F.layer_norm(x, (d,), weight, bias, float(eps))


def add_layer_norm_(  # noqa: N802 (match torch in-place naming convention)
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    d = int(x.shape[-1])
    if d <= 0:
        raise ValueError("add_layer_norm_ expects a non-empty last dimension")
    if (
        _can_use_triton_add_layernorm(x, residual)
        and weight.is_cuda
        and bias.is_cuda
        and weight.is_contiguous()
        and bias.is_contiguous()
        and weight.numel() == d
        and bias.numel() == d
    ):
        out = torch.empty_like(x)
        x2 = x.view(-1, d)
        r2 = residual.view(-1, d)
        y2 = out.view(-1, d)
        n_rows = int(x2.size(0))
        block = _pick_block_size(d)
        n_blocks = (d + block - 1) // block
        num_warps = _pick_num_warps(d)
        grid = (n_rows,)
        _add_layer_norm_fwd_inplace_kernel[grid](
            x2,
            r2,
            y2,
            weight,
            bias,
            n_rows,
            eps=float(eps),
            d=d,
            block=block,
            n_blocks=n_blocks,
            num_warps=num_warps,
            num_stages=1,
        )
        return out
    x.add_(residual)
    return F.layer_norm(x, (d,), weight, bias, float(eps))
