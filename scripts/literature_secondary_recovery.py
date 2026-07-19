#!/usr/bin/env python3
"""Recover failed corpus PDFs through conservative arXiv title matching.

OpenAlex/DOI metadata does not always connect a journal article to its arXiv
preprint.  This second-pass tool searches arXiv by exact title, accepts only a
near-identical normalized title, and validates the resulting bytes with the
same PDF checks as the main corpus downloader.  It never edits inventories.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

from literature_corpus import (
    DEFAULT_ARCHIVE,
    DEFAULT_INVENTORY,
    REPO_ROOT,
    USER_AGENT,
    _artifact_path,
    _download_one,
    _normalize_title,
    _read_jsonl_if_present,
    _write_jsonl,
    load_records,
)


DEFAULT_OUTPUT = (
    REPO_ROOT / "research" / "literature" / "candidates" / "secondary-recovery.jsonl"
)
ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
ARXIV_ID_RE = re.compile(r"/(\d{4}\.\d{4,5})(?:v\d+)?$")


def _manifest_keys(archive: Path, statuses: set[str]) -> set[tuple[str, str]]:
    return {
        (str(row.get("org", "")).casefold(), str(row.get("id", "")))
        for row in _read_jsonl_if_present(archive / "manifest.jsonl")
        if row.get("status") in statuses
    }


def _arxiv_query_url(title: str | Iterable[str], max_results: int = 5) -> str:
    titles = [title] if isinstance(title, str) else list(title)
    search_query = " OR ".join(f'ti:"{value}"' for value in titles)
    query = urllib.parse.urlencode(
        {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
        }
    )
    return f"{ARXIV_API}?{query}"


def _parse_arxiv_feed(payload: bytes) -> list[dict[str, Any]]:
    root = ET.fromstring(payload)
    rows: list[dict[str, Any]] = []
    for entry in root.findall(f"{ATOM}entry"):
        identity = (entry.findtext(f"{ATOM}id") or "").strip()
        match = ARXIV_ID_RE.search(identity)
        if not match:
            continue
        arxiv_id = match.group(1)
        title = " ".join((entry.findtext(f"{ATOM}title") or "").split())
        doi = (entry.findtext(f"{ARXIV}doi") or "").strip().casefold() or None
        rows.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "doi": doi,
                "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            }
        )
    return rows


def _fetch_arxiv_many(
    titles: Iterable[str], timeout: int, retries: int
) -> list[dict[str, Any]]:
    values = list(titles)
    url = _arxiv_query_url(values, max_results=max(5, len(values) * 5))
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/atom+xml,application/xml;q=0.9",
                    "Accept-Encoding": "identity",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return _parse_arxiv_feed(response.read())
        except (OSError, ET.ParseError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"{type(last_error).__name__}: {last_error}")


def _fetch_arxiv(title: str, timeout: int, retries: int) -> list[dict[str, Any]]:
    return _fetch_arxiv_many([title], timeout, retries)


def _match_score(expected: str, candidate: str) -> float:
    left = _normalize_title(expected)
    right = _normalize_title(candidate)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def _best_match(
    record: dict[str, Any], candidates: Iterable[dict[str, Any]], threshold: float
) -> tuple[dict[str, Any] | None, float]:
    scored = [(_match_score(record["title"], row["title"]), row) for row in candidates]
    if not scored:
        return None, 0.0
    score, selected = max(scored, key=lambda item: item[0])
    record_doi = str(record.get("doi") or "").casefold()
    if record_doi and selected.get("doi") == record_doi:
        score = 1.0
    return (selected if score >= threshold else None), score


def _recover_from_candidates(
    record: dict[str, Any],
    candidates: list[dict[str, Any]],
    query_url: str,
    archive: Path,
    timeout: int,
    threshold: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "org": record["org"],
        "id": record["id"],
        "title": record["title"],
        "doi": record.get("doi"),
        "query_url": query_url,
        "candidates": [],
        "selected_url": None,
        "selected_arxiv_id": None,
        "match_score": None,
        "attempt": None,
        "status": "no_match",
        "error": None,
    }
    selected, score = _best_match(record, candidates, threshold)
    ranked = sorted(
        candidates,
        key=lambda row: _match_score(record["title"], row["title"]),
        reverse=True,
    )[:5]
    if selected and selected not in ranked:
        ranked.append(selected)
    result["candidates"] = ranked
    result["match_score"] = round(score, 6)
    if not selected:
        return result

    candidate = dict(record)
    candidate["pdf_url"] = selected["pdf_url"]
    candidate["arxiv_id"] = None
    attempt = _download_one(candidate, archive, timeout, 0, True)
    result["attempt"] = {
        key: attempt.get(key)
        for key in ("status", "url", "error", "sha256", "bytes", "path")
    }
    result["selected_url"] = selected["pdf_url"]
    result["selected_arxiv_id"] = selected["arxiv_id"]
    if attempt["status"] in {"downloaded", "existing"}:
        result["status"] = "recovered"
    else:
        result.update(status="candidate_failed", error=attempt.get("error"))
    return result


def _recover_one(
    record: dict[str, Any], archive: Path, timeout: int, retries: int, threshold: float
) -> dict[str, Any]:
    query_url = _arxiv_query_url(record["title"])
    try:
        candidates = _fetch_arxiv(record["title"], timeout, retries)
    except RuntimeError as exc:
        return {
            "org": record["org"],
            "id": record["id"],
            "title": record["title"],
            "doi": record.get("doi"),
            "query_url": query_url,
            "candidates": [],
            "selected_url": None,
            "selected_arxiv_id": None,
            "match_score": None,
            "attempt": None,
            "status": "arxiv_failed",
            "error": str(exc),
        }
    return _recover_from_candidates(
        record, candidates, query_url, archive, timeout, threshold
    )


def _merge_results(
    previous: Iterable[dict[str, Any]], current: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged = {
        (row.get("org"), row.get("id")): row
        for row in previous
        if row.get("org") and row.get("id")
    }
    for row in current:
        merged[(row.get("org"), row.get("id"))] = row
    return sorted(
        merged.values(),
        key=lambda row: (str(row.get("org", "")).casefold(), str(row.get("id", ""))),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--org", help="exact organization label, case-insensitive")
    parser.add_argument(
        "--id",
        dest="record_ids",
        action="append",
        help="limit to an exact inventory record ID; may be repeated",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="also search records whose manifest status is missing_pdf_url",
    )
    parser.add_argument(
        "--type",
        dest="types",
        action="append",
        help="limit to an inventory type; may be repeated",
    )
    parser.add_argument("--require-doi", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--threshold", type=float, default=0.96)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--refresh", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.delay < 0:
        raise SystemExit("--delay must not be negative")
    if args.batch_size < 1 or args.batch_size > 10:
        raise SystemExit("--batch-size must be between 1 and 10")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if not 0 < args.threshold <= 1:
        raise SystemExit("--threshold must be in (0, 1]")

    statuses = {"failed"}
    if args.include_missing:
        statuses.add("missing_pdf_url")
    eligible = _manifest_keys(args.archive, statuses)
    previous = _read_jsonl_if_present(args.output)
    completed = {
        (str(row.get("org", "")).casefold(), str(row.get("id", "")))
        for row in previous
        if row.get("status") not in {"arxiv_failed", "candidate_failed"}
    }
    records = [
        record
        for record in load_records(args.inventory)
        if (record["org"].casefold(), record["id"]) in eligible
        and (not args.org or record["org"].casefold() == args.org.casefold())
        and (not args.record_ids or record["id"] in args.record_ids)
        and (not args.types or record["type"] in args.types)
        and (not args.require_doi or bool(record.get("doi")))
        and (args.refresh or not _artifact_path(args.archive, record, "pdf").exists())
        and (
            args.refresh
            or (record["org"].casefold(), record["id"]) not in completed
        )
    ]

    results: list[dict[str, Any]] = []
    batches = [
        records[index : index + args.batch_size]
        for index in range(0, len(records), args.batch_size)
    ]
    processed = 0
    for batch_index, batch in enumerate(batches, 1):
        titles = [record["title"] for record in batch]
        query_url = _arxiv_query_url(titles, max_results=max(5, len(batch) * 5))
        try:
            candidates = _fetch_arxiv_many(titles, args.timeout, args.retries)
            batch_error = None
        except RuntimeError as exc:
            candidates = []
            batch_error = str(exc)
        if batch_error:
            batch_results = [
                {
                    "org": record["org"],
                    "id": record["id"],
                    "title": record["title"],
                    "doi": record.get("doi"),
                    "query_url": query_url,
                    "candidates": [],
                    "selected_url": None,
                    "selected_arxiv_id": None,
                    "match_score": None,
                    "attempt": None,
                    "status": "arxiv_failed",
                    "error": batch_error,
                }
                for record in batch
            ]
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(args.workers, len(batch))
            ) as pool:
                batch_results = list(
                    pool.map(
                        lambda record: _recover_from_candidates(
                            record,
                            candidates,
                            query_url,
                            args.archive,
                            args.timeout,
                            args.threshold,
                        ),
                        batch,
                    )
                )
        for result in batch_results:
            results.append(result)
            processed += 1
            print(
                f"[{processed}/{len(records)}] {result['org']}:{result['id']} "
                f"{result['status']} score={result['match_score']}",
                flush=True,
            )
        _write_jsonl(args.output, _merge_results(previous, results))
        if batch_index < len(batches) and args.delay:
            time.sleep(args.delay)

    combined = _merge_results(previous, results)
    _write_jsonl(args.output, combined)
    recovered = sum(row["status"] == "recovered" for row in results)
    print(f"secondary recovery summary: selected={len(records)} recovered={recovered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
