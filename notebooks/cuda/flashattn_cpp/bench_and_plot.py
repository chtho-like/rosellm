#!/usr/bin/env python3

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

import pandas as pd
import torch


@dataclasses.dataclass(frozen=True)
class RunConfig:
    device: int
    batch: int
    heads: int
    seq_lens_csv: str
    head_dims_csv: str
    causal: bool
    warmup: int
    iters: int
    warmup_cpp: int
    max_iters_cpp: int
    min_ms_cpp: float
    check: bool
    arch: str
    kernels: str


def run_checked(cmd: list[str], cwd: pathlib.Path) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd))


def ensure_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as exc:
        msg = (
            "matplotlib is not installed.\n"
            "Install it (system python is fine here):\n"
            "  python -m pip install matplotlib\n"
        )
        raise SystemExit(msg) from exc
    return plt


def list_kernels(makefile_dir: pathlib.Path) -> list[str]:
    out = subprocess.check_output(["make", "list"], cwd=str(makefile_dir))
    bins = [b for b in out.decode("utf-8").splitlines() if b.strip()]
    bins.sort(key=_kernel_sort_key)
    return bins


def _kernel_sort_key(name: str) -> tuple[int, str]:
    if name.startswith("flashattn_"):
        rest = name[len("flashattn_") :]
        num = ""
        for ch in rest:
            if not ch.isdigit():
                break
            num += ch
        if num:
            return (int(num), name)
    return (10**9, name)


@torch.no_grad()
def bench_cuda_events(fn, warmup: int, iters: int) -> float:
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


def attn_tflops(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    causal: bool,
    ms: float,
) -> float:
    pairs = seq_len * seq_len
    if causal:
        pairs = seq_len * (seq_len + 1) // 2
    flops = 4 * batch * heads * pairs * head_dim
    return flops / (ms * 1e-3) / 1e12


def torch_sdpa_auto(q, k, v, causal: bool, scale: float):
    from torch.nn.functional import scaled_dot_product_attention as sdpa

    return sdpa(q, k, v, is_causal=causal, scale=scale)


def torch_sdpa_flash_pytorch(q, k, v, causal: bool, scale: float):
    op = torch.ops.aten._scaled_dot_product_flash_attention.default
    out, *_ = op(q, k, v, 0.0, causal, False, scale=scale)
    return out


def torch_sdpa_efficient_cutlass(q, k, v, causal: bool, scale: float):
    op = torch.ops.aten._scaled_dot_product_efficient_attention.default
    out, *_ = op(q, k, v, None, False, 0.0, causal, scale=scale)
    return out


def torch_sdpa_cudnn(q, k, v, causal: bool, scale: float):
    op = torch.ops.aten._scaled_dot_product_cudnn_attention.default
    out, *_ = op(q, k, v, None, False, 0.0, causal, False, scale=scale)
    return out


def try_import_flash_attn():
    try:
        from flash_attn.flash_attn_interface import flash_attn_func

        return flash_attn_func
    except Exception:
        return None


def run_cpp_kernel(
    cwd: pathlib.Path,
    bin_name: str,
    cfg: RunConfig,
    out_dir: pathlib.Path,
) -> list[dict[str, Any]]:
    bin_path = cwd / bin_name
    json_path = out_dir / "cpp" / f"{bin_name}.json"
    log_path = out_dir / "cpp" / f"{bin_name}.log"
    json_path.parent.mkdir(parents=True, exist_ok=True)

    head_dims_csv = cfg.head_dims_csv
    if "_d64" in bin_name:
        head_dims_csv = "64"
    if "_d128" in bin_name:
        head_dims_csv = "128"

    cmd = [
        str(bin_path),
        "--device",
        str(cfg.device),
        "--batch",
        str(cfg.batch),
        "--heads",
        str(cfg.heads),
        "--seq-lens",
        cfg.seq_lens_csv,
        "--head-dims",
        head_dims_csv,
        "--causal",
        "1" if cfg.causal else "0",
        "--warmup",
        str(cfg.warmup_cpp),
        "--max-iters",
        str(cfg.max_iters_cpp),
        "--min-ms",
        str(cfg.min_ms_cpp),
        "--json-out",
        str(json_path),
    ]
    if cfg.check:
        cmd.append("--check")

    print("+", " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        return [
            {
                "backend": bin_name,
                "error": f"cpp run failed (rc={proc.returncode}), log={log_path}",
            }
        ]

    data = json.loads(json_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for pt in data.get("points", []):
        records.append(
            {
                "backend": str(data.get("kernel", bin_name)),
                "batch": int(data.get("batch", cfg.batch)),
                "heads": int(data.get("heads", cfg.heads)),
                "seq_len": int(pt["seq_len"]),
                "head_dim": int(pt["head_dim"]),
                "dtype": "fp16",
                "causal": bool(data.get("causal", cfg.causal)),
                "avg_ms": float(pt.get("avg_ms", 0.0)) or None,
                "tflops": float(pt.get("tflops", 0.0)) or None,
                "max_abs_err": float(pt.get("max_abs_err", 0.0)) or None,
                "config": None,
                "error": str(pt.get("error", "")) or None,
            }
        )
    return records


def plot(df: pd.DataFrame, out_dir: pathlib.Path) -> None:
    plt = ensure_matplotlib()
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    focus = [
        "flashattn_cpp_best",
        "flash_attn",
        "torch_sdpa_auto",
        "torch_sdpa_flash_pytorch",
        "torch_sdpa_efficient_cutlass",
        "torch_sdpa_cudnn",
        "flashattn_0_warp_naive",
        "flashattn_1_smem_tiled",
        "flashattn_2_causal_cutoff",
        "flashattn_3_mma_bm128_bn32_d64",
        "flashattn_4_mma_bm64_bn32_d128",
        "flashattn_5_mma_bm64_bn64_d128_optin",
        "flashattn_6_mma_bm64_bn32_d64",
        "flashattn_7_mma_bm128_bn64_d64_kvshare",
        "flashattn_8_mma_bm128_bn64_d64_exp2",
        "flashattn_9_mma_bm128_bn64_d64_8warps",
        "flashattn_10_mma_bm128_bn64_d64_8warps_exp2",
        "flashattn_11_mma_bm128_bn64_d64_preload_v_optin",
        "flashattn_12_mma_bm128_bn64_d64_kvshare_v_cp_async",
        "flashattn_13_mma_bm128_bn64_d64_kvshare_v_cp_async_ws1",
        "flashattn_14_dispatch_autotune_d64",
    ]

    head_dims = sorted({int(x) for x in df["head_dim"].dropna().unique()})
    for d in head_dims:
        sub_d = df[df["head_dim"] == d]

        for name, backends in (
            ("all", sorted(sub_d["backend"].unique())),
            ("focus", [b for b in focus if b in set(sub_d["backend"].unique())]),
        ):
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(1, 1, 1)
            for backend in backends:
                ssub = sub_d[
                    (sub_d["backend"] == backend) & sub_d["tflops"].notna()
                ].sort_values("seq_len")
                if ssub.empty:
                    continue
                ax.plot(
                    ssub["seq_len"],
                    ssub["tflops"],
                    marker="o",
                    label=backend,
                )
            ax.set_title(f"FlashAttention forward TFLOPs ({name}, D={d})")
            ax.set_xlabel("seq_len")
            ax.set_ylabel("TFLOPs (model)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(plots_dir / f"tflops_{name}_d{d}.png", dpi=160)
            plt.close(fig)

            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(1, 1, 1)
            for backend in backends:
                ssub = sub_d[
                    (sub_d["backend"] == backend) & sub_d["avg_ms"].notna()
                ].sort_values("seq_len")
                if ssub.empty:
                    continue
                ax.plot(
                    ssub["seq_len"],
                    ssub["avg_ms"],
                    marker="o",
                    label=backend,
                )
            ax.set_title(f"FlashAttention forward time ({name}, D={d})")
            ax.set_xlabel("seq_len")
            ax.set_ylabel("avg_ms")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(plots_dir / f"time_{name}_d{d}.png", dpi=160)
            plt.close(fig)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build flashattn_*.cu binaries, run them + PyTorch/flash-attn "
            "baselines, and plot on one GPU."
        )
    )
    p.add_argument("--device", type=int, default=1)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--seq-lens", type=str, default="256,512,1024,2048,4096")
    p.add_argument("--head-dims", type=str, default="64,128")
    p.add_argument("--causal", type=int, default=1)
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup-cpp", type=int, default=25)
    p.add_argument("--max-iters-cpp", type=int, default=200)
    p.add_argument("--min-ms-cpp", type=float, default=200.0)
    p.add_argument("--check", action="store_true")
    p.add_argument("--arch", type=str, default="sm_89")
    p.add_argument(
        "--kernels",
        type=str,
        default="all",
        help="Which flashattn_* binaries to run: 'all' or CSV list.",
    )
    p.add_argument("--out-dir", type=str, default="")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    cfg = RunConfig(
        device=args.device,
        batch=args.batch,
        heads=args.heads,
        seq_lens_csv=args.seq_lens,
        head_dims_csv=args.head_dims,
        causal=bool(args.causal),
        warmup=args.warmup,
        iters=args.iters,
        warmup_cpp=args.warmup_cpp,
        max_iters_cpp=args.max_iters_cpp,
        min_ms_cpp=args.min_ms_cpp,
        check=bool(args.check),
        arch=args.arch,
        kernels=args.kernels,
    )

    cwd = pathlib.Path(__file__).resolve().parent

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else (cwd / "out" / "latest")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cwd": str(cwd),
        "device": cfg.device,
        "batch": cfg.batch,
        "heads": cfg.heads,
        "seq_lens": cfg.seq_lens_csv,
        "head_dims": cfg.head_dims_csv,
        "causal": cfg.causal,
        "warmup": cfg.warmup,
        "iters": cfg.iters,
        "warmup_cpp": cfg.warmup_cpp,
        "max_iters_cpp": cfg.max_iters_cpp,
        "min_ms_cpp": cfg.min_ms_cpp,
        "check": cfg.check,
        "arch": cfg.arch,
        "kernels": cfg.kernels,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    run_checked(["make", f"ARCH={cfg.arch}", "-j"], cwd=cwd)

    all_bins = list_kernels(cwd)
    bins = all_bins
    if cfg.kernels != "all":
        want = [k.strip() for k in cfg.kernels.split(",") if k.strip()]
        missing = [k for k in want if k not in all_bins]
        if missing:
            raise SystemExit(f"unknown kernels: {', '.join(sorted(missing))}")
        want_set = set(want)
        bins = [b for b in all_bins if b in want_set]

    records: list[dict[str, Any]] = []

    # C++ kernels (standalone binaries)
    for b in bins:
        records.extend(run_cpp_kernel(cwd, b, cfg, out_dir))

    # Python baselines (torch/flash-attn)
    torch.cuda.set_device(cfg.device)
    flash_attn_func = try_import_flash_attn()

    seq_lens = [int(x) for x in cfg.seq_lens_csv.split(",") if x.strip()]
    head_dims = [int(x) for x in cfg.head_dims_csv.split(",") if x.strip()]

    for d in head_dims:
        scale = 1.0 / math.sqrt(d)
        for s in seq_lens:
            q = torch.randn((cfg.batch, cfg.heads, s, d), device="cuda", dtype=torch.float16)
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            check_this = cfg.check and (s == min(seq_lens)) and (d == min(head_dims))
            ref = None
            if check_this:
                ref = torch_sdpa_auto(q, k, v, cfg.causal, scale).float()

            def add_row(name: str, ms: float | None, out: torch.Tensor | None = None, err: str | None = None):
                max_abs_err = None
                if check_this and ref is not None and out is not None:
                    max_abs_err = float((out.float() - ref).abs().max().item())
                records.append(
                    {
                        "backend": name,
                        "batch": cfg.batch,
                        "heads": cfg.heads,
                        "seq_len": s,
                        "head_dim": d,
                        "dtype": "fp16",
                        "causal": cfg.causal,
                        "avg_ms": ms,
                        "tflops": None if ms is None else attn_tflops(cfg.batch, cfg.heads, s, d, cfg.causal, ms),
                        "max_abs_err": max_abs_err,
                        "config": None,
                        "error": err,
                    }
                )

            try:
                ms = bench_cuda_events(lambda: torch_sdpa_auto(q, k, v, cfg.causal, scale), cfg.warmup, cfg.iters)
                add_row("torch_sdpa_auto", ms)
            except Exception as exc:
                add_row("torch_sdpa_auto", None, err=str(exc))

            for name, fn in (
                ("torch_sdpa_flash_pytorch", torch_sdpa_flash_pytorch),
                ("torch_sdpa_efficient_cutlass", torch_sdpa_efficient_cutlass),
                ("torch_sdpa_cudnn", torch_sdpa_cudnn),
            ):
                try:
                    ms = bench_cuda_events(lambda: fn(q, k, v, cfg.causal, scale), cfg.warmup, cfg.iters)
                    out = fn(q, k, v, cfg.causal, scale) if check_this else None
                    add_row(name, ms, out=out)
                except Exception as exc:
                    add_row(name, None, err=str(exc))

            if flash_attn_func is not None:
                q_bshd = q.transpose(1, 2).contiguous()
                k_bshd = k.transpose(1, 2).contiguous()
                v_bshd = v.transpose(1, 2).contiguous()

                def fn_flash():
                    return flash_attn_func(
                        q_bshd,
                        k_bshd,
                        v_bshd,
                        dropout_p=0.0,
                        softmax_scale=scale,
                        causal=cfg.causal,
                    )

                try:
                    ms = bench_cuda_events(fn_flash, cfg.warmup, cfg.iters)
                    out = fn_flash().transpose(1, 2) if check_this else None
                    add_row("flash_attn", ms, out=out)
                except Exception as exc:
                    add_row("flash_attn", None, err=str(exc))

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(["head_dim", "seq_len", "backend"], ascending=[True, True, True])

    # "Autotune" aggregation: pick the best (fastest) C++ kernel per shape.
    group_cols = ["batch", "heads", "seq_len", "head_dim", "dtype", "causal"]
    cpp = df[df["backend"].astype(str).str.startswith("flashattn_")].copy()
    cpp = cpp[cpp["avg_ms"].notna()].copy()
    if not cpp.empty:
        idx = cpp.groupby(group_cols)["avg_ms"].idxmin()
        best = cpp.loc[idx].copy()
        best["config"] = best["backend"]
        best["backend"] = "flashattn_cpp_best"
        df = pd.concat([df, best], ignore_index=True)
        df = df.sort_values(
            ["head_dim", "seq_len", "backend"], ascending=[True, True, True]
        )

        table = []
        for _, row in best.sort_values(["head_dim", "seq_len"]).iterrows():
            table.append(
                {
                    "batch": int(row["batch"]),
                    "heads": int(row["heads"]),
                    "seq_len": int(row["seq_len"]),
                    "head_dim": int(row["head_dim"]),
                    "dtype": str(row["dtype"]),
                    "causal": bool(row["causal"]),
                    "best_kernel": str(row["config"]),
                    "best_avg_ms": float(row["avg_ms"]),
                    "best_tflops": float(row["tflops"]) if not pd.isna(row["tflops"]) else None,
                }
            )
        (out_dir / "autotune_table.json").write_text(
            json.dumps(table, indent=2) + "\n", encoding="utf-8"
        )
    df.to_csv(out_dir / "results.csv", index=False)
    (out_dir / "results.json").write_text(json.dumps({"meta": meta, "records": records}, indent=2) + "\n", encoding="utf-8")

    plot(df, out_dir)
    print("Wrote:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
