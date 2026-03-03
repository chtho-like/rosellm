import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import triton

import triton_flashattn_8_dispatch_fixed_best as impl


@dataclass(frozen=True)
class KernelConfig:
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int


def _default_configs() -> list[KernelConfig]:
    configs: list[KernelConfig] = []
    for block_m in (32, 64, 128):
        for block_n in (16, 32, 64):
            for num_warps in (2, 4, 8):
                for num_stages in (2, 3, 4):
                    configs.append(
                        KernelConfig(
                            block_m=block_m,
                            block_n=block_n,
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                    )
    return configs


def _table_out_path() -> Path:
    major, minor = torch.cuda.get_device_capability()
    sm = f"sm{major}{minor}"
    return Path(__file__).resolve().with_name(f"triton_flashattn_v9_table_{sm}.json")


@torch.no_grad()
def _bench_cuda_events(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iters


def _tune_one(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    sm_scale: float,
    warmup: int,
    iters: int,
    configs: list[KernelConfig],
) -> tuple[KernelConfig, float]:
    bsz, nheads, seq_len, head_dim = q.shape
    o = torch.empty_like(q)
    kernel = impl._flashattn_fwd_kernel

    best_cfg = None
    best_ms = None
    for cfg in configs:
        grid = (
            triton.cdiv(seq_len, cfg.block_m),
            bsz * nheads,
        )

        def _fn():
            kernel[grid](
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
                BLOCK_M=cfg.block_m,
                BLOCK_N=cfg.block_n,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
            )

        try:
            ms = _bench_cuda_events(_fn, warmup=warmup, iters=iters)
        except Exception:
            continue

        if best_ms is None or ms < best_ms:
            best_ms = ms
            best_cfg = cfg

    if best_cfg is None or best_ms is None:
        raise RuntimeError("all configs failed")
    return best_cfg, best_ms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seq-lens", type=str, default="256,512,1024,2048,4096")
    parser.add_argument("--head-dims", type=str, default="64,128")
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--causal", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    torch.cuda.set_device(args.device)

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    causal = bool(args.causal)
    seq_lens = [int(x) for x in args.seq_lens.split(",") if x.strip()]
    head_dims = [int(x) for x in args.head_dims.split(",") if x.strip()]

    configs = _default_configs()

    table: dict[str, dict] = {}
    rows: list[dict] = []
    for d in head_dims:
        scale = 1.0 / math.sqrt(d)
        for s in seq_lens:
            q = torch.randn(
                (args.batch, args.heads, s, d),
                device="cuda",
                dtype=dtype,
            )
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            best_cfg, best_ms = _tune_one(
                q,
                k,
                v,
                causal=causal,
                sm_scale=scale,
                warmup=args.warmup,
                iters=args.iters,
                configs=configs,
            )

            key = f"c{int(causal)}_d{d}_s{s}"
            table[key] = asdict(best_cfg)
            rows.append(
                {
                    "key": key,
                    "avg_ms": best_ms,
                    **asdict(best_cfg),
                }
            )
            print(f"{key}: {best_cfg} avg_ms={best_ms:.6f}")

    out_path = Path(args.out) if args.out else _table_out_path()
    out = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": args.device,
            "gpu_name": torch.cuda.get_device_name(args.device),
            "sm": ".".join(map(str, torch.cuda.get_device_capability(args.device))),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": __import__("triton").__version__,
        },
        "sweep": {
            "batch": args.batch,
            "heads": args.heads,
            "seq_lens": seq_lens,
            "head_dims": head_dims,
            "dtype": args.dtype,
            "causal": causal,
            "warmup": args.warmup,
            "iters": args.iters,
            "num_configs": len(configs),
        },
        "table": table,
        "results": rows,
    }
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

