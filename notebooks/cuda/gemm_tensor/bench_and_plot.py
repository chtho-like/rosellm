#!/usr/bin/env python3

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import pathlib
import subprocess
import sys
from typing import Any


@dataclasses.dataclass(frozen=True)
class RunConfig:
    device: int
    sizes_csv: str
    warmup: int
    max_iters: int
    min_ms: float
    check: bool
    arch: str
    accum: str
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
            "Create a venv and install plotting deps:\n"
            "  python3 -m venv .venv\n"
            "  . .venv/bin/activate\n"
            "  python3 -m pip install -U pip\n"
            "  python3 -m pip install matplotlib\n"
        )
        raise SystemExit(msg) from exc
    return plt


def list_kernels(makefile_dir: pathlib.Path) -> list[str]:
    out = subprocess.check_output(["make", "list"], cwd=str(makefile_dir))
    bins = [b for b in out.decode("utf-8").splitlines() if b.strip()]
    bins.sort(key=_kernel_sort_key)
    return bins


def _kernel_sort_key(name: str) -> tuple[int, str]:
    if name.startswith("hgemm_"):
        rest = name[len("hgemm_") :]
        num = ""
        for ch in rest:
            if not ch.isdigit():
                break
            num += ch
        if num:
            return (int(num), name)
    return (10**9, name)


def load_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sizes_from_json(data: dict[str, Any]) -> list[int]:
    return [int(p["m"]) for p in data["points"]]


def series_from_json(data: dict[str, Any], key: str) -> list[float]:
    return [float(p[key]) for p in data["points"]]


def plot_kernel(plt: Any, json_path: pathlib.Path, out_dir: pathlib.Path) -> None:
    data = load_json(json_path)
    kernel = str(data["kernel"])
    gpu = str(data["gpu"])
    accum = str(data.get("accum", "f16"))

    sizes = sizes_from_json(data)
    custom_tflops = series_from_json(data, "custom_tflops")
    cublas_tflops = series_from_json(data, "cublas_tflops")
    speedup = series_from_json(data, "speedup")

    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(sizes, cublas_tflops, marker="o", label=f"cuBLAS ({accum})")
    ax.plot(sizes, custom_tflops, marker="o", label=kernel)
    ax.set_title(f"HGEMM TFLOP/s vs size ({accum})\n{gpu}")
    ax.set_xlabel("Square size (M=N=K)")
    ax.set_ylabel("TFLOP/s (higher is better)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{kernel}_tflops.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(sizes, speedup, marker="o", label=f"{kernel} / cuBLAS")
    ax.axhline(1.0, linestyle="--", color="black", linewidth=1)
    ax.set_title(f"HGEMM speedup vs cuBLAS ({accum})\n{gpu}")
    ax.set_xlabel("Square size (M=N=K)")
    ax.set_ylabel("Speedup (1.0 == cuBLAS)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{kernel}_speedup.png", dpi=160)
    plt.close(fig)


def write_summary_md(out_dir: pathlib.Path, json_paths: list[pathlib.Path]) -> None:
    if not json_paths:
        return

    rows: list[str] = []
    header = (
        "| kernel | gpu | accum | sizes | best speedup | best size |"
        " best custom TF | best cuBLAS TF |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|\n"
    )
    for p in json_paths:
        data = load_json(p)
        kernel = str(data["kernel"])
        gpu = str(data["gpu"])
        accum = str(data.get("accum", "f16"))
        sizes = sizes_from_json(data)
        speedup = series_from_json(data, "speedup")
        custom_tflops = series_from_json(data, "custom_tflops")
        cublas_tflops = series_from_json(data, "cublas_tflops")
        best_i = max(range(len(speedup)), key=lambda i: speedup[i])
        rows.append(
            "| "
            + kernel
            + " | "
            + gpu
            + " | "
            + accum
            + " | "
            + str(len(sizes))
            + " | "
            + f"{speedup[best_i]:.3f}"
            + " | "
            + str(sizes[best_i])
            + " | "
            + f"{custom_tflops[best_i]:.3f}"
            + " | "
            + f"{cublas_tflops[best_i]:.3f}"
            + " |\n"
        )

    write_text(out_dir / "summary.md", header + "".join(rows))


def write_best_of_md(out_dir: pathlib.Path, json_paths: list[pathlib.Path]) -> None:
    if not json_paths:
        return

    by_kernel: dict[str, dict[int, dict[str, Any]]] = {}
    gpu = None
    accum = None
    for p in json_paths:
        data = load_json(p)
        gpu = gpu or str(data.get("gpu", ""))
        accum = accum or str(data.get("accum", "f16"))
        kernel = str(data.get("kernel", p.stem))
        pts: dict[int, dict[str, Any]] = {}
        for pt in data.get("points", []):
            pts[int(pt["m"])] = pt
        by_kernel[kernel] = pts

    sizes = sorted({s for pts in by_kernel.values() for s in pts})
    if not sizes:
        return

    header = (
        "| size (M=N=K) | best kernel | custom TFLOP/s | cuBLAS TFLOP/s"
        " | speedup |\n"
        "|---:|---|---:|---:|---:|\n"
    )
    rows: list[str] = []
    best_speedups: list[float] = []
    best_custom: list[float] = []
    best_cublas: list[float] = []
    for s in sizes:
        best_k = ""
        best_sp = float("-inf")
        best_ct = 0.0
        best_bt = 0.0
        for k, pts in by_kernel.items():
            pt = pts.get(s)
            if pt is None:
                continue
            sp = float(pt["speedup"])
            if sp > best_sp:
                best_sp = sp
                best_k = k
                best_ct = float(pt["custom_tflops"])
                best_bt = float(pt["cublas_tflops"])
        best_speedups.append(best_sp)
        best_custom.append(best_ct)
        best_cublas.append(best_bt)
        rows.append(
            "| "
            + str(s)
            + " | "
            + best_k
            + " | "
            + f"{best_ct:.3f}"
            + " | "
            + f"{best_bt:.3f}"
            + " | "
            + f"{best_sp:.3f}"
            + " |\n"
        )

    write_text(out_dir / "best_of.md", header + "".join(rows))

    plt = ensure_matplotlib()
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    title_gpu = f"\n{gpu}" if gpu else ""
    title_acc = f" ({accum})" if accum else ""

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(sizes, best_cublas, marker="o", label=f"cuBLAS{title_acc}")
    ax.plot(sizes, best_custom, marker="o", label="best custom")
    ax.set_title(f"Best HGEMM TFLOP/s vs size{title_acc}{title_gpu}")
    ax.set_xlabel("Square size (M=N=K)")
    ax.set_ylabel("TFLOP/s (higher is better)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "best_of_tflops.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(sizes, best_speedups, marker="o", label="best custom / cuBLAS")
    ax.axhline(1.0, linestyle="--", color="black", linewidth=1)
    ax.set_title(f"Best HGEMM speedup vs cuBLAS{title_acc}{title_gpu}")
    ax.set_xlabel("Square size (M=N=K)")
    ax.set_ylabel("Speedup (1.0 == cuBLAS)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "best_of_speedup.png", dpi=160)
    plt.close(fig)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build all hgemm_*.cu programs, run them, and plot vs cuBLAS."
    )
    p.add_argument("--device", type=int, default=0)
    p.add_argument(
        "--sizes",
        type=str,
        default="256,512,1024,2048,4096,6144,8192",
        help="CSV of square sizes (M=N=K).",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--max-iters", type=int, default=200)
    p.add_argument("--min-ms", type=float, default=200.0)
    p.add_argument(
        "--no-check",
        action="store_true",
        help="Skip correctness check on the first shape.",
    )
    p.add_argument("--arch", type=str, default="sm_89")
    p.add_argument(
        "--accum",
        type=str,
        default="f16",
        choices=["f16", "f32"],
        help="cuBLAS compute/accumulation type used for baseline.",
    )
    p.add_argument(
        "--kernels",
        type=str,
        default="all",
        choices=["all", "f16acc", "f32acc"],
        help=(
            "Which kernels to run. The split is inferred from the binary name: "
            "'f32acc' in name => f32acc, otherwise f16acc."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output dir for json + plots. Default is results/run_<ts>/.",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    cfg = RunConfig(
        device=args.device,
        sizes_csv=args.sizes,
        warmup=args.warmup,
        max_iters=args.max_iters,
        min_ms=args.min_ms,
        check=(not args.no_check),
        arch=args.arch,
        accum=args.accum,
        kernels=args.kernels,
    )

    cwd = pathlib.Path(__file__).resolve().parent
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else (
        cwd / "results" / f"run_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "cwd": str(cwd),
        "device": cfg.device,
        "sizes": cfg.sizes_csv,
        "warmup": cfg.warmup,
        "max_iters": cfg.max_iters,
        "min_ms": cfg.min_ms,
        "check": cfg.check,
        "arch": cfg.arch,
        "accum": cfg.accum,
        "kernels": cfg.kernels,
    }
    write_text(out_dir / "meta.json", json.dumps(meta, indent=2) + "\n")

    run_checked(["make", f"ARCH={cfg.arch}", "-j"], cwd=cwd)

    bins = list_kernels(cwd)
    if cfg.kernels == "f32acc":
        bins = [b for b in bins if "f32acc" in b]
    elif cfg.kernels == "f16acc":
        bins = [b for b in bins if "f32acc" not in b]
    json_paths: list[pathlib.Path] = []
    failures: list[dict[str, Any]] = []

    for b in bins:
        bin_path = cwd / b
        json_path = out_dir / f"{b}.json"
        log_path = out_dir / f"{b}.log"
        cmd = [
            str(bin_path),
            "--device",
            str(cfg.device),
            "--sizes",
            cfg.sizes_csv,
            "--warmup",
            str(cfg.warmup),
            "--max-iters",
            str(cfg.max_iters),
            "--min-ms",
            str(cfg.min_ms),
            "--accum",
            cfg.accum,
            "--json-out",
            str(json_path),
        ]
        if not cfg.check:
            cmd.append("--no-check")
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
            failures.append(
                {
                    "kernel": b,
                    "returncode": proc.returncode,
                    "log": str(log_path),
                }
            )
            continue
        json_paths.append(json_path)

    if failures:
        write_text(out_dir / "failures.json", json.dumps(failures, indent=2) + "\n")

    plt = ensure_matplotlib()
    plot_dir = out_dir / "plots"
    for p in json_paths:
        plot_kernel(plt, p, plot_dir)

    write_summary_md(out_dir, json_paths)
    write_best_of_md(out_dir, json_paths)
    print("Wrote:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
