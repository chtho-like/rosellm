from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_online(
    payloads: list[dict[str, Any]], *, sources: list[str]
) -> dict[str, Any]:
    if not payloads:
        raise ValueError("no payloads")
    base = dict(payloads[0])
    out: dict[str, Any] = {"meta": dict(base.get("meta", {}))}
    out["meta"]["merged_from"] = list(sources)

    summaries_by_key: dict[tuple[str, float], dict[str, Any]] = {}
    by_backend: dict[str, Any] = {}

    for payload in payloads:
        for item in payload.get("summaries", []) or []:
            backend = str(item.get("backend", ""))
            scale = float(item.get("scale", 0.0))
            summaries_by_key[(backend, scale)] = dict(item)
        for backend, entry in (payload.get("by_backend", {}) or {}).items():
            by_backend[str(backend)] = entry

    out["summaries"] = [
        summaries_by_key[k]
        for k in sorted(summaries_by_key.keys(), key=lambda x: (x[1], x[0]))
    ]
    out["by_backend"] = by_backend
    out["meta"]["backends"] = sorted(by_backend.keys())
    return out


def _merge_offline(
    payloads: list[dict[str, Any]], *, sources: list[str]
) -> dict[str, Any]:
    if not payloads:
        raise ValueError("no payloads")
    base = dict(payloads[0])
    out: dict[str, Any] = {"meta": dict(base.get("meta", {}))}
    out["meta"]["merged_from"] = list(sources)

    by_backend: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        for item in payload.get("results", []) or []:
            backend = str(item.get("backend", ""))
            by_backend[backend] = dict(item)

    out["results"] = [by_backend[k] for k in sorted(by_backend.keys())]
    out["meta"]["backends"] = sorted(by_backend.keys())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge *_results.json files from multiple runs."
    )
    parser.add_argument(
        "--kind",
        type=str,
        required=True,
        choices=["online", "offline"],
        help="Which result format to merge.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output JSON path.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input *_results.json paths (later files override duplicates).",
    )
    args = parser.parse_args()

    inputs = [Path(p).expanduser().resolve() for p in args.inputs]
    payloads = [_load(p) for p in inputs]
    sources = [str(p) for p in inputs]
    if args.kind == "online":
        merged = _merge_online(payloads, sources=sources)
    else:
        merged = _merge_offline(payloads, sources=sources)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
