import argparse
import importlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import pandas as pd
import torch


def _try_import_flash_attn():
    try:
        import flash_attn  # noqa: F401
        from flash_attn.flash_attn_interface import flash_attn_func

        return flash_attn_func
    except Exception:
        return None


def _device_meta(device: int) -> dict:
    name = torch.cuda.get_device_name(device)
    major, minor = torch.cuda.get_device_capability(device)
    props = torch.cuda.get_device_properties(device)

    driver = None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        )
        driver = out.strip().splitlines()[0]
    except Exception:
        driver = None

    return {
        "gpu_name": name,
        "sm": f"{major}.{minor}",
        "total_mem_gb": float(props.total_memory) / (1024**3),
        "driver": driver,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def _attn_flops(b: int, h: int, s: int, d: int, causal: bool) -> int:
    pairs = s * s
    if causal:
        pairs = s * (s + 1) // 2
    return 4 * b * h * pairs * d


def _tflops(b: int, h: int, s: int, d: int, causal: bool, ms: float) -> float:
    flops = _attn_flops(b, h, s, d, causal)
    return flops / (ms * 1e-3) / 1e12


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


def _torch_sdpa(q, k, v, causal: bool):
    from torch.nn.functional import scaled_dot_product_attention as sdpa

    return sdpa(q, k, v, is_causal=causal)


def _torch_matmul_baseline(q, k, v, causal: bool, scale: float):
    att = torch.matmul(q, k.transpose(-1, -2)) * scale
    if causal:
        s = q.shape[-2]
        mask = torch.triu(
            torch.ones((s, s), device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        att = att.masked_fill(mask, float("-inf"))
    p = torch.softmax(att, dim=-1)
    return torch.matmul(p, v)


def _import_impl(module_name: str):
    return importlib.import_module(module_name)


def _backend_config(mod, seq_len: int, head_dim: int, causal: bool) -> str | None:
    kernel = getattr(mod, "_flashattn_fwd_kernel", None)
    if kernel is not None:
        best_cfg = getattr(kernel, "best_config", None)
        if best_cfg is not None:
            return str(best_cfg)
    if hasattr(mod, "BEST_CONFIG"):
        return str(getattr(mod, "BEST_CONFIG"))
    if hasattr(mod, "_pick_config"):
        try:
            cfg = mod._pick_config(seq_len, head_dim, causal)
            return str(cfg)
        except Exception:
            return None
    return None


def _default_out_dir() -> Path:
    root = Path(__file__).resolve().parent
    out_root = root / "out"
    out_root.mkdir(parents=True, exist_ok=True)
    latest = out_root / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    return latest


def _plot(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    focus = [
        "flash_attn",
        "torch_sdpa_auto",
        "torch_sdpa_flash_pytorch",
        "torch_sdpa_efficient_cutlass",
        "torch_sdpa_cudnn",
        "torch_matmul_softmax",
        "triton_v0_baseline",
        "triton_v4_causal_cutoff",
        "triton_v7_fixed_best_cutoff",
        "triton_v8_dispatch_fixed_best",
        "triton_v9_autotune_table",
    ]

    for causal in sorted(df["causal"].unique()):
        sub = df[df["causal"] == causal]
        for d in sorted(sub["head_dim"].unique()):
            sub_d = sub[sub["head_dim"] == d]
            for name, backends in (
                ("all", sorted(sub_d["backend"].unique())),
                ("focus", [b for b in focus if b in set(sub_d["backend"].unique())]),
            ):
                plt.figure(figsize=(8, 5))
                for backend in backends:
                    ssub = sub_d[
                        (sub_d["backend"] == backend) & sub_d["tflops"].notna()
                    ].sort_values("seq_len")
                    if ssub.empty:
                        continue
                    plt.plot(
                        ssub["seq_len"],
                        ssub["tflops"],
                        marker="o",
                        label=backend,
                    )
                plt.title(f"SDPA forward TFLOPs ({name}, causal={causal}, D={d})")
                plt.xlabel("seq_len")
                plt.ylabel("TFLOPs (model)")
                plt.grid(True, alpha=0.3)
                plt.legend()
                path = plots_dir / f"tflops_{name}_c{int(causal)}_d{d}.png"
                plt.tight_layout()
                plt.savefig(path)
                plt.close()

                plt.figure(figsize=(8, 5))
                for backend in backends:
                    ssub = sub_d[
                        (sub_d["backend"] == backend) & sub_d["avg_ms"].notna()
                    ].sort_values("seq_len")
                    if ssub.empty:
                        continue
                    plt.plot(
                        ssub["seq_len"],
                        ssub["avg_ms"],
                        marker="o",
                        label=backend,
                    )
                plt.title(f"SDPA forward time ({name}, causal={causal}, D={d})")
                plt.xlabel("seq_len")
                plt.ylabel("avg_ms")
                plt.grid(True, alpha=0.3)
                plt.legend()
                path = plots_dir / f"time_{name}_c{int(causal)}_d{d}.png"
                plt.tight_layout()
                plt.savefig(path)
                plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seq-lens", type=str, default="256,512,1024,2048")
    parser.add_argument("--head-dims", type=str, default="64")
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--causal", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--skip-flash-attn", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    torch.cuda.set_device(args.device)

    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    causal = bool(args.causal)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    seq_lens = [int(x) for x in args.seq_lens.split(",") if x.strip()]
    head_dims = [int(x) for x in args.head_dims.split(",") if x.strip()]

    impl_modules = [
        "triton_flashattn_0_baseline",
        "triton_flashattn_1_codegen_hints",
        "triton_flashattn_2_autotune",
        "triton_flashattn_3_fixed_best",
        "triton_flashattn_4_causal_cutoff",
        "triton_flashattn_5_even_fastpath",
        "triton_flashattn_6_autotune_cutoff",
        "triton_flashattn_7_fixed_best_cutoff",
        "triton_flashattn_8_dispatch_fixed_best",
        "triton_flashattn_9_autotune_table",
        "torch_sdpa_flash_pytorch",
        "torch_sdpa_efficient_cutlass",
        "torch_sdpa_cudnn",
    ]

    flash_attn_func = None
    if not args.skip_flash_attn:
        flash_attn_func = _try_import_flash_attn()

    meta = _device_meta(args.device)
    meta["triton"] = __import__("triton").__version__
    meta["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["cwd"] = os.getcwd()

    records: list[dict] = []
    for d in head_dims:
        if d % 16 != 0:
            raise ValueError("head_dim must be a multiple of 16")
        scale = 1.0 / math.sqrt(d)
        for s in seq_lens:
            q = torch.randn(
                (args.batch, args.heads, s, d),
                device="cuda",
                dtype=dtype,
            )
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            check_this = args.check and (s == min(seq_lens)) and (d == min(head_dims))
            ref = None
            if check_this:
                ref = _torch_sdpa(q, k, v, causal).float()

            for module_name in impl_modules:
                try:
                    mod = _import_impl(module_name)
                except Exception as exc:
                    records.append(
                        {
                            "backend": module_name,
                            "batch": args.batch,
                            "heads": args.heads,
                            "seq_len": s,
                            "head_dim": d,
                            "dtype": args.dtype,
                            "causal": causal,
                            "avg_ms": None,
                            "tflops": None,
                            "max_abs_err": None,
                            "config": None,
                            "error": f"import failed: {exc}",
                        }
                    )
                    continue
                backend_name = getattr(mod, "NAME", module_name)

                def _fn():
                    mod.flash_attn(q, k, v, causal=causal, sm_scale=scale)

                try:
                    ms = _bench_cuda_events(_fn, args.warmup, args.iters)
                    max_abs_err = None
                    if check_this and ref is not None:
                        out = mod.flash_attn(
                            q, k, v, causal=causal, sm_scale=scale
                        ).float()
                        max_abs_err = float((out - ref).abs().max().item())
                    records.append(
                        {
                            "backend": backend_name,
                            "batch": args.batch,
                            "heads": args.heads,
                            "seq_len": s,
                            "head_dim": d,
                            "dtype": args.dtype,
                            "causal": causal,
                            "avg_ms": ms,
                            "tflops": _tflops(
                                args.batch,
                                args.heads,
                                s,
                                d,
                                causal,
                                ms,
                            ),
                            "max_abs_err": max_abs_err,
                            "config": _backend_config(mod, s, d, causal),
                            "error": None,
                        }
                    )
                except Exception as exc:
                    records.append(
                        {
                            "backend": backend_name,
                            "batch": args.batch,
                            "heads": args.heads,
                            "seq_len": s,
                            "head_dim": d,
                            "dtype": args.dtype,
                            "causal": causal,
                            "avg_ms": None,
                            "tflops": None,
                            "max_abs_err": None,
                            "config": _backend_config(mod, s, d, causal),
                            "error": str(exc),
                        }
                    )
                    continue

            def _fn_sdpa():
                _torch_sdpa(q, k, v, causal)

            try:
                ms = _bench_cuda_events(_fn_sdpa, args.warmup, args.iters)
                records.append(
                    {
                        "backend": "torch_sdpa_auto",
                        "batch": args.batch,
                        "heads": args.heads,
                        "seq_len": s,
                        "head_dim": d,
                        "dtype": args.dtype,
                        "causal": causal,
                        "avg_ms": ms,
                        "tflops": _tflops(
                            args.batch,
                            args.heads,
                            s,
                            d,
                            causal,
                            ms,
                        ),
                        "max_abs_err": None,
                        "config": None,
                        "error": None,
                    }
                )
            except Exception as exc:
                records.append(
                    {
                        "backend": "torch_sdpa_auto",
                        "batch": args.batch,
                        "heads": args.heads,
                        "seq_len": s,
                        "head_dim": d,
                        "dtype": args.dtype,
                        "causal": causal,
                        "avg_ms": None,
                        "tflops": None,
                        "max_abs_err": None,
                        "config": None,
                        "error": str(exc),
                    }
                )

            if flash_attn_func is not None:
                q_bshd = q.transpose(1, 2).contiguous()
                k_bshd = k.transpose(1, 2).contiguous()
                v_bshd = v.transpose(1, 2).contiguous()

                def _fn_flash_attn():
                    flash_attn_func(
                        q_bshd,
                        k_bshd,
                        v_bshd,
                        dropout_p=0.0,
                        softmax_scale=scale,
                        causal=causal,
                    )

                try:
                    ms = _bench_cuda_events(
                        _fn_flash_attn, args.warmup, args.iters
                    )
                    max_abs_err = None
                    if check_this and ref is not None:
                        out = flash_attn_func(
                            q_bshd,
                            k_bshd,
                            v_bshd,
                            dropout_p=0.0,
                            softmax_scale=scale,
                            causal=causal,
                        ).transpose(1, 2)
                        max_abs_err = float((out.float() - ref).abs().max().item())
                    records.append(
                        {
                            "backend": "flash_attn",
                            "batch": args.batch,
                            "heads": args.heads,
                            "seq_len": s,
                            "head_dim": d,
                            "dtype": args.dtype,
                            "causal": causal,
                            "avg_ms": ms,
                            "tflops": _tflops(
                                args.batch,
                                args.heads,
                                s,
                                d,
                                causal,
                                ms,
                            ),
                            "max_abs_err": max_abs_err,
                            "config": None,
                            "error": None,
                        }
                    )
                except Exception as exc:
                    records.append(
                        {
                            "backend": "flash_attn",
                            "batch": args.batch,
                            "heads": args.heads,
                            "seq_len": s,
                            "head_dim": d,
                            "dtype": args.dtype,
                            "causal": causal,
                            "avg_ms": None,
                            "tflops": None,
                            "max_abs_err": None,
                            "config": None,
                            "error": str(exc),
                        }
                    )

            if s <= 1024:
                def _fn_mm():
                    _torch_matmul_baseline(q, k, v, causal, scale)

                try:
                    ms = _bench_cuda_events(_fn_mm, args.warmup, args.iters)
                    max_abs_err = None
                    if check_this and ref is not None:
                        out = _torch_matmul_baseline(
                            q, k, v, causal, scale
                        ).float()
                        max_abs_err = float((out - ref).abs().max().item())
                    records.append(
                        {
                            "backend": "torch_matmul_softmax",
                            "batch": args.batch,
                            "heads": args.heads,
                            "seq_len": s,
                            "head_dim": d,
                            "dtype": args.dtype,
                            "causal": causal,
                            "avg_ms": ms,
                            "tflops": _tflops(
                                args.batch,
                                args.heads,
                                s,
                                d,
                                causal,
                                ms,
                            ),
                            "max_abs_err": max_abs_err,
                            "config": None,
                            "error": None,
                        }
                    )
                except Exception as exc:
                    records.append(
                        {
                            "backend": "torch_matmul_softmax",
                            "batch": args.batch,
                            "heads": args.heads,
                            "seq_len": s,
                            "head_dim": d,
                            "dtype": args.dtype,
                            "causal": causal,
                            "avg_ms": None,
                            "tflops": None,
                            "max_abs_err": None,
                            "config": None,
                            "error": str(exc),
                        }
                    )

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(
        ["causal", "head_dim", "seq_len", "backend"],
        ascending=[True, True, True, True],
    )

    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results.csv", index=False)
    _plot(df, out_dir)

    out = {"meta": meta, "records": records}
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(meta, indent=2))
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
