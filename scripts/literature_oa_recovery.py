#!/usr/bin/env python3
"""Recover failed corpus PDFs from alternate OpenAlex OA locations.

The script is deliberately conservative: it only considers records whose
normal corpus download failed, follows the record's OpenAlex work identity (or
DOI), and accepts an alternate URL only after the downloaded bytes pass the
same PDF validation as the main corpus driver.  It never edits an inventory.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from literature_corpus import (
    DEFAULT_ARCHIVE,
    DEFAULT_INVENTORY,
    REPO_ROOT,
    USER_AGENT,
    _artifact_path,
    _download_one,
    _read_jsonl_if_present,
    _resolved_pdf_url,
    _rewrite_pdf_url,
    _write_jsonl,
    load_records,
)


DEFAULT_OUTPUT = (
    REPO_ROOT / "research" / "literature" / "candidates" / "oa-recovery.jsonl"
)
OPENALEX_WORK_RE = re.compile(r"^https?://openalex\.org/(W\d+)/?$", re.I)
ARXIV_LANDING_RE = re.compile(
    r"^https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?$",
    re.I,
)


def _fetch_json(url: str, timeout: int, retries: int) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json",
                    "Accept-Encoding": "identity",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                decoded = json.load(response)
            if not isinstance(decoded, dict):
                raise ValueError("OpenAlex response was not an object")
            return decoded
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"{type(last_error).__name__}: {last_error}")


def _openalex_api_url(record: dict[str, Any]) -> str | None:
    for source in record.get("source_pages") or []:
        match = OPENALEX_WORK_RE.fullmatch(source)
        if match:
            return f"https://api.openalex.org/works/{match.group(1).upper()}"
    if record.get("doi"):
        doi_url = f"https://doi.org/{record['doi']}"
        return "https://api.openalex.org/works/" + urllib.parse.quote(
            doi_url, safe=":/"
        )
    return None


def _location_urls(work: dict[str, Any]) -> list[str]:
    """Return distinct plausible full-text URLs, preserving OA priority."""
    locations = []
    for key in ("best_oa_location", "primary_location"):
        value = work.get(key)
        if isinstance(value, dict):
            locations.append(value)
    locations.extend(
        value for value in (work.get("locations") or []) if isinstance(value, dict)
    )

    candidates = []
    for location in locations:
        pdf = location.get("pdf_url")
        if isinstance(pdf, str) and pdf.startswith(("http://", "https://")):
            candidates.append(_rewrite_pdf_url(pdf))
        landing = location.get("landing_page_url")
        if isinstance(landing, str):
            match = ARXIV_LANDING_RE.fullmatch(landing)
            if match:
                candidates.append(f"https://arxiv.org/pdf/{match.group(1)}")
    oa = work.get("open_access")
    if isinstance(oa, dict):
        url = oa.get("oa_url")
        if isinstance(url, str) and (
            urllib.parse.urlparse(url).path.casefold().endswith(".pdf")
            or ARXIV_LANDING_RE.fullmatch(url)
        ):
            match = ARXIV_LANDING_RE.fullmatch(url)
            candidates.append(
                f"https://arxiv.org/pdf/{match.group(1)}" if match else _rewrite_pdf_url(url)
            )
    return list(dict.fromkeys(candidates))


def _failed_keys(archive: Path) -> set[tuple[str, str]]:
    return {
        (str(row.get("org", "")).casefold(), str(row.get("id", "")))
        for row in _read_jsonl_if_present(archive / "manifest.jsonl")
        if row.get("status") == "failed"
    }


def _recover_one(
    record: dict[str, Any], archive: Path, timeout: int, retries: int, force: bool
) -> dict[str, Any]:
    current = _resolved_pdf_url(record)
    result: dict[str, Any] = {
        "org": record["org"],
        "id": record["id"],
        "title": record["title"],
        "doi": record.get("doi"),
        "arxiv_id": record.get("arxiv_id"),
        "current_url": current,
        "openalex_api_url": _openalex_api_url(record),
        "openalex_id": None,
        "candidate_urls": [],
        "attempts": [],
        "selected_url": None,
        "status": "no_openalex_identity",
        "error": None,
    }
    if not result["openalex_api_url"]:
        return result
    try:
        work = _fetch_json(result["openalex_api_url"], timeout, retries)
        result["openalex_id"] = work.get("id")
    except RuntimeError as exc:
        result.update(status="openalex_failed", error=str(exc))
        return result

    candidates = [url for url in _location_urls(work) if url != current]
    result["candidate_urls"] = candidates
    if not candidates:
        result["status"] = "no_alternate_location"
        return result

    for url in candidates:
        candidate = dict(record)
        candidate["pdf_url"] = url
        # Probe exactly this location rather than silently falling back to the
        # original inventory arXiv route.
        candidate["arxiv_id"] = None
        attempt = _download_one(candidate, archive, timeout, 0, force)
        result["attempts"].append(
            {
                "url": url,
                "status": attempt["status"],
                "error": attempt.get("error"),
                "sha256": attempt.get("sha256"),
                "bytes": attempt.get("bytes"),
            }
        )
        if attempt["status"] in {"downloaded", "existing"}:
            result.update(status="recovered", selected_url=url)
            return result
    result["status"] = "alternates_failed"
    return result


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
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    failed = _failed_keys(args.archive)
    records = [
        record
        for record in load_records(args.inventory)
        if (record["org"].casefold(), record["id"]) in failed
        and (not args.org or record["org"].casefold() == args.org.casefold())
        and not _artifact_path(args.archive, record, "pdf").exists()
    ]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _recover_one,
                record,
                args.archive,
                args.timeout,
                args.retries,
                args.force,
            ): record
            for record in records
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(
                f"[{index}/{len(futures)}] {result['org']}:{result['id']} "
                f"{result['status']}"
            )
    combined = _merge_results(_read_jsonl_if_present(args.output), results)
    _write_jsonl(args.output, combined)
    recovered = sum(row["status"] == "recovered" for row in results)
    print(f"recovery summary: selected={len(records)} recovered={recovered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
