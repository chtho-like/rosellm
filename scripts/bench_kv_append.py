#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BenchResult:
    batch_size: int
    baseline_us: float
    triton_us: float

    @property
    def speedup(self) -> float:
        if self.triton_us <= 0:
            return 0.0
        return self.baseline_us / self.triton_us


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Microbench KV append (torch vs triton)"
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="128,256,512,1024,2048",
        help='Comma-separated batch sizes (e.g. "128,256,512")',
    )
    parser.add_argument("--heads", type=int, default=12, help="Number of heads (H)")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dim (D)")
    parser.add_argument("--block-size", type=int, default=64, help="KV block size (BS)")
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=4096,
        help="Number of physical blocks in KV cache (N_BLOCKS)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
        help="KV/cache dtype on CUDA",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument(
        "--iters",
        type=int,
        default=200,
        help="Timed iterations (will auto-reduce for very large batch sizes)",
    )
    parser.add_argument(
        "--non-contig",
        action="store_true",
        help="Use a non-contiguous [B,H,D] view (select from [B,H,T,D])",
    )
    parser.add_argument(
        "--non-contig-t",
        type=int,
        default=128,
        help="T for non-contiguous source (only used with --non-contig)",
    )
    return parser.parse_args()


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def bench_one(
    *,
    batch_size: int,
    n_heads: int,
    head_dim: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    non_contig: bool,
    non_contig_t: int,
) -> BenchResult:
    device = torch.device("cuda")
    k_cache = torch.empty(
        (num_blocks, n_heads, block_size, head_dim), device=device, dtype=dtype
    )
    v_cache = torch.empty_like(k_cache)

    if non_contig:
        t = int(non_contig_t)
        k_big = torch.randn(
            (batch_size, n_heads, t, head_dim), device=device, dtype=dtype
        )
        v_big = torch.randn_like(k_big)
        key_new = k_big.select(2, t - 1)
        value_new = v_big.select(2, t - 1)
    else:
        key_new = torch.randn(
            (batch_size, n_heads, head_dim), device=device, dtype=dtype
        )
        value_new = torch.randn_like(key_new)

    batch_idx_i32 = torch.arange(batch_size, device=device, dtype=torch.int32)
    block_idx_i32 = torch.randint(
        0, num_blocks, (batch_size,), device=device, dtype=torch.int32
    )
    pos_i32 = torch.randint(
        0, block_size, (batch_size,), device=device, dtype=torch.int32
    )
    batch_idx = batch_idx_i32.to(torch.long)
    block_idx = block_idx_i32.to(torch.long)
    pos = pos_i32.to(torch.long)

    for _ in range(warmup):
        k_src = key_new.index_select(0, batch_idx)
        v_src = value_new.index_select(0, batch_idx)
        k_cache[block_idx, :, pos, :] = k_src
        v_cache[block_idx, :, pos, :] = v_src
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        k_src = key_new.index_select(0, batch_idx)
        v_src = value_new.index_select(0, batch_idx)
        k_cache[block_idx, :, pos, :] = k_src
        v_cache[block_idx, :, pos, :] = v_src
    end.record()
    torch.cuda.synchronize()
    baseline_ms = start.elapsed_time(end)

    from rosellm.roseinfer.kv_append_triton import TRITON_AVAILABLE, kv_append_triton

    if not TRITON_AVAILABLE:
        raise SystemExit(
            "Triton is not available; install it or use a CUDA env with Triton."
        )

    for _ in range(warmup):
        kv_append_triton(
            k_cache_layer=k_cache,
            v_cache_layer=v_cache,
            key_new=key_new,
            value_new=value_new,
            batch_idx=batch_idx_i32,
            block_idx=block_idx_i32,
            pos=pos_i32,
        )
    torch.cuda.synchronize()

    start.record()
    for _ in range(iters):
        kv_append_triton(
            k_cache_layer=k_cache,
            v_cache_layer=v_cache,
            key_new=key_new,
            value_new=value_new,
            batch_idx=batch_idx_i32,
            block_idx=block_idx_i32,
            pos=pos_i32,
        )
    end.record()
    torch.cuda.synchronize()
    triton_ms = start.elapsed_time(end)

    return BenchResult(
        batch_size=batch_size,
        baseline_us=baseline_ms / float(iters) * 1e3,
        triton_us=triton_ms / float(iters) * 1e3,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this microbench.")
    os.environ["ROSELLM_TRITON_KV_APPEND"] = "1"
    args = parse_args()
    batch_sizes = [
        int(x.strip()) for x in str(args.batch_sizes).split(",") if x.strip()
    ]
    if not batch_sizes or any(b <= 0 for b in batch_sizes):
        raise SystemExit("--batch-sizes must contain positive integers")

    dtype = _dtype_from_name(str(args.dtype))
    warmup = max(0, int(args.warmup))
    iters0 = max(1, int(args.iters))

    print("=== kv_append microbench ===")
    print(f"H={int(args.heads)} D={int(args.head_dim)} BS={int(args.block_size)}")
    print(
        f"N_BLOCKS={int(args.num_blocks)} dtype={args.dtype} non_contig={bool(args.non_contig)}"
    )
    print(f"warmup={warmup} iters={iters0}")
    print()

    results: list[BenchResult] = []
    for b in batch_sizes:
        iters = iters0
        if b >= 2048:
            iters = min(iters, 50)
        elif b >= 1024:
            iters = min(iters, 100)
        r = bench_one(
            batch_size=b,
            n_heads=int(args.heads),
            head_dim=int(args.head_dim),
            block_size=int(args.block_size),
            num_blocks=int(args.num_blocks),
            dtype=dtype,
            warmup=warmup,
            iters=iters,
            non_contig=bool(args.non_contig),
            non_contig_t=int(args.non_contig_t),
        )
        results.append(r)

    w_bs = max(4, max(len(str(r.batch_size)) for r in results))
    print(f"{'B':>{w_bs}}  {'baseline(us)':>12}  {'triton(us)':>10}  {'speedup':>8}")
    for r in results:
        print(
            f"{r.batch_size:>{w_bs}}  {r.baseline_us:>12.2f}  {r.triton_us:>10.2f}  {r.speedup:>8.3f}x"
        )

    speedups = [r.speedup for r in results if r.speedup > 0]
    if speedups:
        geo = math.exp(sum(math.log(s) for s in speedups) / float(len(speedups)))
        print()
        print(f"geomean speedup: {geo:.3f}x")


if __name__ == "__main__":
    main()
