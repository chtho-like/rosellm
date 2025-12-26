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

USE_TRITON_FUSED_MLP: Final[bool] = (
    os.environ.get("ROSELLM_TRITON_FUSED_MLP", "1") != "0"
)


def _can_use_triton(x: torch.Tensor, bias: torch.Tensor) -> bool:
    if not (TRITON_AVAILABLE and USE_TRITON_FUSED_MLP):
        return False
    if not (x.is_cuda and bias.is_cuda):
        return False
    if x.dtype != bias.dtype:
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if not (x.is_contiguous() and bias.is_contiguous()):
        return False
    if x.dim() != 2:
        return False
    if bias.dim() != 1:
        return False
    if x.size(1) != bias.numel():
        return False
    return True


def _pick_block_size(d: int) -> int:
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
    def _bias_gelu_new_inplace_kernel(
        x_ptr,
        bias_ptr,
        n_rows,
        d: tl.constexpr,
        block: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        blk = tl.program_id(1)
        if row >= n_rows:
            return
        cols = blk * block + tl.arange(0, block)
        mask = cols < d
        row_start = row * d
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        z = x + b
        c = 0.7978845608028654  # sqrt(2/pi)
        t = c * (z + 0.044715 * z * z * z)
        abs_t = tl.abs(t)
        e = tl.exp(-2.0 * abs_t)
        tanh_abs = (1.0 - e) / (1.0 + e)
        tanh_t = tl.where(t >= 0.0, tanh_abs, -tanh_abs)
        y = 0.5 * z * (1.0 + tanh_t)
        tl.store(x_ptr + row_start + cols, y, mask=mask)

    @triton.jit
    def _add_bias_residual_inplace_kernel(
        residual_ptr,
        y_ptr,
        bias_ptr,
        n_rows,
        d: tl.constexpr,
        block: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        blk = tl.program_id(1)
        if row >= n_rows:
            return
        cols = blk * block + tl.arange(0, block)
        mask = cols < d
        row_start = row * d
        r = tl.load(residual_ptr + row_start + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        y = tl.load(y_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = r + y + b
        tl.store(residual_ptr + row_start + cols, out, mask=mask)


def bias_gelu_new_(  # noqa: N802 (match torch in-place naming convention)
    x: torch.Tensor,  # [N, D]
    bias: torch.Tensor,  # [D]
) -> torch.Tensor:
    d = int(x.size(1))
    if d <= 0:
        raise ValueError("bias_gelu_new_ expects a non-empty last dimension")
    if _can_use_triton(x, bias):
        n_rows = int(x.size(0))
        block = _pick_block_size(d)
        n_blocks = (d + block - 1) // block
        num_warps = _pick_num_warps(d)
        _bias_gelu_new_inplace_kernel[(n_rows, n_blocks)](
            x,
            bias,
            n_rows,
            d=d,
            block=block,
            num_warps=num_warps,
            num_stages=1,
        )
        return x
    x.add_(bias)
    x.copy_(F.gelu(x, approximate="tanh"))
    return x


def add_bias_residual_(  # noqa: N802 (match torch in-place naming convention)
    residual: torch.Tensor,  # [N, D]
    y: torch.Tensor,  # [N, D]
    bias: torch.Tensor,  # [D]
) -> torch.Tensor:
    d = int(residual.size(1))
    if d <= 0:
        raise ValueError("add_bias_residual_ expects a non-empty last dimension")
    if not (residual.dim() == 2 and y.dim() == 2):
        raise ValueError("add_bias_residual_ expects 2D [N, D] inputs")
    if residual.shape != y.shape:
        raise ValueError("add_bias_residual_ expects residual and y with same shape")
    if (
        _can_use_triton(residual, bias)
        and y.is_cuda
        and y.dtype == residual.dtype
        and y.is_contiguous()
    ):
        n_rows = int(residual.size(0))
        block = _pick_block_size(d)
        n_blocks = (d + block - 1) // block
        num_warps = _pick_num_warps(d)
        _add_bias_residual_inplace_kernel[(n_rows, n_blocks)](
            residual,
            y,
            bias,
            n_rows,
            d=d,
            block=block,
            num_warps=num_warps,
            num_stages=1,
        )
        return residual
    residual.add_(y)
    residual.add_(bias)
    return residual
