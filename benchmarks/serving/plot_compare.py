from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BACKEND_LABELS = {
    "roseinfer": "roseinfer",
    "roseinfer+flashinfer": "roseinfer (flashinfer)",
    "roseinfer+flashattn": "roseinfer (flash-attn)",
    "roseinfer+chunked": "roseinfer (chunked prefill)",
    "roseinfer+inproc": "roseinfer (in-proc)",
    "vllm": "vLLM",
    "sglang": "SGLang",
    "trtllm": "TensorRT-LLM",
}

BACKEND_COLORS = {
    "roseinfer": "#1f77b4",
    "roseinfer+flashinfer": "#9467bd",
    "roseinfer+flashattn": "#d62728",
    "roseinfer+chunked": "#8c564b",
    "vllm": "#ff7f0e",
    "sglang": "#2ca02c",
    "trtllm": "#4d4d4d",
}

BACKEND_MARKERS = {
    "roseinfer": "o",
    "roseinfer+flashinfer": "D",
    "roseinfer+flashattn": "v",
    "roseinfer+chunked": "P",
    "roseinfer+inproc": "p",
    "vllm": "s",
    "sglang": "^",
    "trtllm": "H",
}


def _backend_label(key: str) -> str:
    if key in BACKEND_LABELS:
        return BACKEND_LABELS[key]
    if key.startswith("roseinfer"):
        parts = [p for p in key.split("+")[1:] if p]
        extras: list[str] = []
        for p in parts:
            if p == "flashinfer":
                extras.append("flashinfer")
            elif p == "flashattn":
                extras.append("flash-attn")
            elif p == "naive":
                extras.append("naive")
            elif p == "chunked":
                extras.append("chunked prefill")
            elif p == "inproc":
                extras.append("in-proc")
            elif p == "nofuse":
                extras.append("no fused ops")
            elif p == "nomlp":
                extras.append("no fused mlp")
            elif p == "nosampler":
                extras.append("no fused sampler")
            elif p == "nokv":
                extras.append("no fused kv append")
            elif p == "nooverlap":
                extras.append("no overlap schedule")
        if extras:
            return f"roseinfer ({', '.join(extras)})"
        return "roseinfer"
    return key


def _backend_color(key: str) -> str:
    if key in BACKEND_COLORS:
        return BACKEND_COLORS[key]
    if key.startswith("roseinfer"):
        if "nofuse" in key.split("+"):
            return "#7f7f7f"
        if "nooverlap" in key.split("+"):
            return "#aec7e8"
        if "nomlp" in key.split("+"):
            return "#e377c2"
        if "nosampler" in key.split("+"):
            return "#bcbd22"
        if "nokv" in key.split("+"):
            return "#17becf"
        return BACKEND_COLORS["roseinfer"]
    return "#333333"


def _backend_marker(key: str) -> str:
    if key in BACKEND_MARKERS:
        return BACKEND_MARKERS[key]
    if key.startswith("roseinfer"):
        parts = key.split("+")
        if "nofuse" in parts:
            return "X"
        if "nooverlap" in parts:
            return ">"
        if "nomlp" in parts:
            return "8"
        if "nosampler" in parts:
            return "*"
        if "nokv" in parts:
            return "h"
        if "inproc" in parts:
            return "p"
        if "chunked" in parts:
            return "P"
        if "flashinfer" in parts:
            return "D"
        if "flashattn" in parts:
            return "v"
        return "o"
    return "o"


def _semantic_legend_handles(*, band_alpha: float) -> tuple[list[object], list[str]]:
    return (
        [
            Line2D([0], [0], color="black", marker="o", linestyle="-"),
            Line2D(
                [0],
                [0],
                color="black",
                marker="o",
                linestyle="None",
                markerfacecolor="none",
                markeredgewidth=1.2,
            ),
            Patch(facecolor="black", alpha=band_alpha, edgecolor="none"),
        ],
        ["p90 (line)", "p99 (hollow)", "p50–p90 (band)"],
    )


def _paper_rcparams() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.linewidth": 1.2,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_online_summaries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in payload.get("summaries", []):
        row = {
            "backend": str(s["backend"]),
            "scale": float(s["scale"]),
        }
        for metric in ("ttft_ms", "tpot_ms", "itl_ms", "e2e_ms"):
            stats = s[metric]
            for k in ("p50", "p90", "p99", "mean", "max"):
                row[f"{metric}_{k}"] = float(stats[k])
        rows.append(row)
    return rows


def _write_online_summary_md(payload: dict[str, Any], out_dir: Path) -> Path:
    rows = _flatten_online_summaries(payload)
    if not rows:
        raise ValueError("no online summaries found")
    rows.sort(key=lambda r: (r["scale"], r["backend"]))

    meta = payload.get("meta", {})
    lines: list[str] = []
    lines.append("# Online Serving Benchmark Summary\n")
    lines.append(f"- model: `{meta.get('model')}`\n")
    if meta.get("dtype") is not None:
        lines.append(f"- dtype: `{meta.get('dtype')}`\n")
    lines.append(f"- n: `{meta.get('n')}`\n")
    lines.append(f"- scales: `{meta.get('scales')}`\n")
    lines.append(
        f"- sampling: temperature={meta.get('temperature')}, top_p={meta.get('top_p')}, top_k={meta.get('top_k')}\n"
    )
    if meta.get("wall_s") is not None:
        lines.append(f"- wall_s: `{meta.get('wall_s')}`\n")
    if meta.get("run_start_time") is not None:
        lines.append(f"- run_start_time: `{meta.get('run_start_time')}`\n")
    if meta.get("run_end_time") is not None:
        lines.append(f"- run_end_time: `{meta.get('run_end_time')}`\n")
    versions = meta.get("versions")
    if isinstance(versions, dict):
        keys = (
            "git_rev",
            "rosellm",
            "vllm",
            "sglang",
            "tensorrt_llm",
            "torch",
            "transformers",
            "python",
        )
        ver_str = ", ".join(
            f"{k}={versions[k]}"
            for k in keys
            if k in versions and versions[k] not in (None, "")
        )
        if ver_str:
            lines.append(f"- versions: `{ver_str}`\n")
    lines.append("\n")

    def fmt_ms(x: float) -> str:
        return f"{x:.2f}"

    header = (
        "| scale | backend | TTFT p50/p90/p99 (ms) | TPOT p50/p90/p99 (ms) |"
        " ITL p50/p90/p99 (ms) | E2E p50/p90/p99 (ms) |\n"
        "|---:|---|---:|---:|---:|---:|\n"
    )
    lines.append(header)
    for r in rows:
        backend = _backend_label(str(r["backend"]))
        lines.append(
            "| {scale:.2f} | {backend} | {ttft} | {tpot} | {itl} | {e2e} |\n".format(
                scale=r["scale"],
                backend=backend,
                ttft=f"{fmt_ms(r['ttft_ms_p50'])}/{fmt_ms(r['ttft_ms_p90'])}/{fmt_ms(r['ttft_ms_p99'])}",
                tpot=f"{fmt_ms(r['tpot_ms_p50'])}/{fmt_ms(r['tpot_ms_p90'])}/{fmt_ms(r['tpot_ms_p99'])}",
                itl=f"{fmt_ms(r['itl_ms_p50'])}/{fmt_ms(r['itl_ms_p90'])}/{fmt_ms(r['itl_ms_p99'])}",
                e2e=f"{fmt_ms(r['e2e_ms_p50'])}/{fmt_ms(r['e2e_ms_p90'])}/{fmt_ms(r['e2e_ms_p99'])}",
            )
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "online_summary.md"
    out_path.write_text("".join(lines), encoding="utf-8")
    return out_path


def _write_offline_summary_md(payload: dict[str, Any], out_dir: Path) -> Path:
    results = payload.get("results", [])
    if not results:
        raise ValueError("no offline results found")
    results = sorted(results, key=lambda r: r.get("backend", ""))

    meta = payload.get("meta", {})
    lines: list[str] = []
    lines.append("# Offline Throughput Benchmark Summary\n")
    lines.append(f"- model: `{meta.get('model')}`\n")
    if meta.get("dtype") is not None:
        lines.append(f"- dtype: `{meta.get('dtype')}`\n")
    lines.append(f"- num_prompts: `{meta.get('num_prompts')}`\n")
    lines.append(f"- input_len: `{meta.get('input_len')}`\n")
    lines.append(f"- output_len: `{meta.get('output_len')}`\n")
    lines.append(
        f"- sampling: temperature={meta.get('temperature')}, top_p={meta.get('top_p')}, top_k={meta.get('top_k')}\n"
    )
    if meta.get("wall_s") is not None:
        lines.append(f"- wall_s: `{meta.get('wall_s')}`\n")
    if meta.get("run_start_time") is not None:
        lines.append(f"- run_start_time: `{meta.get('run_start_time')}`\n")
    if meta.get("run_end_time") is not None:
        lines.append(f"- run_end_time: `{meta.get('run_end_time')}`\n")
    versions = meta.get("versions")
    if isinstance(versions, dict):
        keys = (
            "git_rev",
            "rosellm",
            "vllm",
            "sglang",
            "tensorrt_llm",
            "torch",
            "transformers",
            "python",
        )
        ver_str = ", ".join(
            f"{k}={versions[k]}"
            for k in keys
            if k in versions and versions[k] not in (None, "")
        )
        if ver_str:
            lines.append(f"- versions: `{ver_str}`\n")
    lines.append("\n")

    header = (
        "| backend | req/s | output tok/s | total tok/s | total latency (s) |\n"
        "|---|---:|---:|---:|---:|\n"
    )
    lines.append(header)
    for r in results:
        backend = _backend_label(str(r["backend"]))
        lines.append(
            "| {backend} | {rps:.2f} | {out_tps:.2f} | {tot_tps:.2f} | {lat:.3f} |\n".format(
                backend=backend,
                rps=float(r["request_throughput_rps"]),
                out_tps=float(r["output_throughput_tps"]),
                tot_tps=float(r["total_throughput_tps"]),
                lat=float(r["total_s"]),
            )
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "offline_summary.md"
    out_path.write_text("".join(lines), encoding="utf-8")
    return out_path


def _plot_online(payload: dict[str, Any], out_dir: Path) -> Path:
    rows = _flatten_online_summaries(payload)
    if not rows:
        raise ValueError("no online summaries found")
    backends = sorted({r["backend"] for r in rows})

    def series(backend: str, key: str) -> tuple[np.ndarray, np.ndarray]:
        pts = sorted((r["scale"], r[key]) for r in rows if r["backend"] == backend)
        x = np.array([p[0] for p in pts], dtype=np.float64)
        y = np.array([p[1] for p in pts], dtype=np.float64)
        return x, y

    _paper_rcparams()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=False)
    axes = axes.reshape(-1)
    specs = [
        ("ttft_ms", "TTFT", "TTFT (ms)"),
        ("tpot_ms", "TPOT", "TPOT (ms/token)"),
        ("itl_ms", "ITL", "ITL (ms/token)"),
        ("e2e_ms", "E2E", "E2E (ms)"),
    ]
    for idx, (ax, (metric, title, ylabel)) in enumerate(zip(axes, specs, strict=True)):
        for backend in backends:
            label = _backend_label(backend)
            color = _backend_color(backend)
            marker = _backend_marker(backend)
            x, y90 = series(backend, f"{metric}_p90")
            _, y50 = series(backend, f"{metric}_p50")
            _, y99 = series(backend, f"{metric}_p99")
            ax.plot(x, y90, marker=marker, color=color, label=label)
            ax.fill_between(x, y50, y90, color=color, alpha=0.12, linewidth=0)
            ax.scatter(
                x,
                y99,
                marker=marker,
                facecolors="none",
                edgecolors=color,
                linewidths=1.2,
                alpha=0.7,
                zorder=3,
            )

        if idx in (2, 3):
            ax.set_xlabel("Trace time scale (smaller = higher load)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    handles, labels = axes[0].get_legend_handles_labels()
    sem_handles, sem_labels = _semantic_legend_handles(band_alpha=0.12)
    fig.tight_layout(rect=(0.0, 0.0, 0.80, 1.0))
    fig.legend(
        handles + sem_handles,
        labels + sem_labels,
        loc="center left",
        bbox_to_anchor=(0.81, 0.5),
        ncol=1,
        handlelength=2.0,
        labelspacing=0.6,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "online_latency_compare.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_online_single(payload: dict[str, Any], out_dir: Path) -> list[Path]:
    rows = _flatten_online_summaries(payload)
    if not rows:
        raise ValueError("no online summaries found")
    backends = sorted({r["backend"] for r in rows})

    def series(backend: str, key: str) -> tuple[np.ndarray, np.ndarray]:
        pts = sorted((r["scale"], r[key]) for r in rows if r["backend"] == backend)
        x = np.array([p[0] for p in pts], dtype=np.float64)
        y = np.array([p[1] for p in pts], dtype=np.float64)
        return x, y

    specs = [
        ("ttft_ms", "TTFT (ms)", "online_ttft_ms.png"),
        ("tpot_ms", "TPOT (ms/token)", "online_tpot_ms.png"),
        ("itl_ms", "ITL (ms/token)", "online_itl_ms.png"),
        ("e2e_ms", "E2E (ms)", "online_e2e_ms.png"),
    ]
    out_paths: list[Path] = []
    _paper_rcparams()
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, filename in specs:
        fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.6), constrained_layout=True)
        for backend in backends:
            label = _backend_label(backend)
            color = _backend_color(backend)
            marker = _backend_marker(backend)
            x, y90 = series(backend, f"{metric}_p90")
            _, y50 = series(backend, f"{metric}_p50")
            _, y99 = series(backend, f"{metric}_p99")
            ax.plot(x, y90, marker=marker, color=color, label=label)
            ax.fill_between(x, y50, y90, color=color, alpha=0.12, linewidth=0)
            ax.scatter(
                x,
                y99,
                marker=marker,
                facecolors="none",
                edgecolors=color,
                linewidths=1.2,
                alpha=0.7,
                zorder=3,
            )

        ax.set_xlabel("Trace time scale (smaller = higher load)")
        ax.set_ylabel(ylabel)
        handles, labels = ax.get_legend_handles_labels()
        sem_handles, sem_labels = _semantic_legend_handles(band_alpha=0.12)
        ax.legend(
            handles + sem_handles,
            labels + sem_labels,
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 1.10),
            handlelength=2.0,
            columnspacing=1.4,
        )
        out_path = out_dir / filename
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


def _plot_offline(payload: dict[str, Any], out_dir: Path) -> Path:
    results = payload.get("results", [])
    if not results:
        raise ValueError("no offline results found")

    def tick_label(backend: str) -> str:
        label = _backend_label(backend)
        if not label.startswith("roseinfer (") or not label.endswith(")"):
            return label
        inner = label[len("roseinfer (") : -1]
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        short = {
            "no fused ops": "no ops",
            "no fused mlp": "no MLP",
            "no fused sampler": "no samp",
            "no fused kv append": "no KV",
            "no overlap schedule": "no overlap sched",
        }
        parts = [short.get(p, p) for p in parts]
        if not parts:
            return "roseinfer"
        return "\n".join(parts)

    _paper_rcparams()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True)

    backends = [r["backend"] for r in results]
    labels = [tick_label(str(b)) for b in backends]
    colors = [_backend_color(str(b)) for b in backends]

    metrics = [
        ("output_throughput_tps", "Output Throughput (tok/s)"),
        ("request_throughput_rps", "Request Throughput (req/s)"),
        ("total_throughput_tps", "Total Throughput (tok/s)"),
    ]
    for ax, (key, title) in zip(axes, metrics, strict=True):
        vals = [float(r[key]) for r in results]
        bar_width = 0.70
        x = np.arange(len(vals), dtype=np.float32) * 1.40
        ax.bar(
            x,
            vals,
            width=bar_width,
            color=colors,
            edgecolor="black",
            linewidth=0.8,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
        ax.tick_params(axis="x", pad=4)
        ax.set_xlim(x[0] - bar_width, x[-1] + bar_width)
        ax.set_title(title)
        ax.set_ylabel(title.split(" (", 1)[-1].rstrip(")"))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "offline_throughput_compare.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot online/offline benchmark results saved by benchmarks/serving/*_compare.py.",
    )
    parser.add_argument(
        "--online", type=str, default=None, help="Path to online_results.json."
    )
    parser.add_argument(
        "--offline", type=str, default=None, help="Path to offline_results.json."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/benchmarks/serving/figures",
        help="Directory to write PNG figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    if args.online:
        online_path = Path(args.online).expanduser().resolve()
        online_payload = _load_json(online_path)
        out_path = _plot_online(online_payload, out_dir)
        for p in _plot_online_single(online_payload, out_dir):
            print(f"Wrote: {p}")
        md_path = _write_online_summary_md(online_payload, out_dir)
        print(f"Wrote: {md_path}")
        print(f"Wrote: {out_path}")
    if args.offline:
        offline_path = Path(args.offline).expanduser().resolve()
        offline_payload = _load_json(offline_path)
        out_path = _plot_offline(offline_payload, out_dir)
        md_path = _write_offline_summary_md(offline_payload, out_dir)
        print(f"Wrote: {md_path}")
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
