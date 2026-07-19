#!/usr/bin/env python3
"""Snapshot authoritative publication indexes into a reviewable candidate queue.

This script does not decide that every indexed page is a paper.  It preserves
the official discovery universe so the curated inventory can prove what was
reviewed, included, excluded, or left unresolved.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT / "research" / "literature" / "candidates" / "official-pages.jsonl"
)
DEFAULT_OPENAI_RSS_OUTPUT = (
    REPO_ROOT / "research" / "literature" / "candidates" / "openai-rss.jsonl"
)
USER_AGENT = "RoseLLM-Literature-Discovery/1.0 (+https://github.com/chtho-like/rosellm)"

OPENAI_FEEDS = (
    "publication",
    "research",
    "safety",
    "milestone",
    "conclusion",
    "release",
    "engineering",
)


def _today() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


def fetch(url: str, timeout: int, retries: int) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.1",
                    "Accept-Encoding": "identity",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (TimeoutError, OSError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def parse_sitemap(payload: bytes, source: str) -> list[tuple[str, str | None]]:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise RuntimeError(f"invalid XML from {source}: {exc}") from exc
    rows: list[tuple[str, str | None]] = []
    for node in root:
        values = {child.tag.rsplit("}", 1)[-1]: child.text for child in node}
        url = values.get("loc")
        if url:
            rows.append((url.strip(), values.get("lastmod")))
    return rows


def record(
    org: str,
    url: str,
    source_url: str,
    source_kind: str,
    lastmod: str | None,
    discovered_at: str,
) -> dict[str, Any]:
    return {
        "org": org,
        "url": url,
        "source_url": source_url,
        "source_kind": source_kind,
        "lastmod": lastmod,
        "discovered_at": discovered_at,
        "review_status": "pending",
    }


def discover_openai(timeout: int, retries: int, date: str) -> list[dict[str, Any]]:
    rows = []
    for feed in OPENAI_FEEDS:
        source = f"https://openai.com/sitemap.xml/{feed}/"
        for url, lastmod in parse_sitemap(fetch(source, timeout, retries), source):
            rows.append(record("OpenAI", url, source, f"official_sitemap:{feed}", lastmod, date))
    return rows


def discover_openai_rss(timeout: int, retries: int, date: str) -> list[dict[str, Any]]:
    source = "https://openai.com/news/rss.xml"
    payload = fetch(source, timeout, retries)
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise RuntimeError(f"invalid RSS XML from {source}: {exc}") from exc
    rows = []
    for item in root.findall("./channel/item"):
        link = (item.findtext("link") or "").strip()
        title = (item.findtext("title") or "").strip()
        if not link or not title:
            continue
        published = (item.findtext("pubDate") or "").strip()
        try:
            published_at = parsedate_to_datetime(published).date().isoformat()
        except (TypeError, ValueError, OverflowError):
            published_at = None
        rows.append(
            {
                "org": "OpenAI",
                "url": link,
                "title": title,
                "description": (item.findtext("description") or "").strip() or None,
                "category": (item.findtext("category") or "").strip() or None,
                "published_at": published_at,
                "source_url": source,
                "discovered_at": date,
            }
        )
    return sorted(rows, key=lambda row: (row["published_at"] or "", row["url"]), reverse=True)


def discover_anthropic(timeout: int, retries: int, date: str) -> list[dict[str, Any]]:
    source = "https://www.anthropic.com/sitemap.xml"
    rows = []
    for url, lastmod in parse_sitemap(fetch(source, timeout, retries), source):
        path = urllib.parse.urlparse(url).path.rstrip("/")
        if path.startswith("/research/team/"):
            kind = "official_sitemap:research_team"
        elif path.startswith("/research/"):
            kind = "official_sitemap:research"
        elif path == "/system-cards":
            kind = "official_sitemap:system_cards_index"
        elif path.startswith("/engineering/"):
            kind = "official_sitemap:engineering"
        else:
            continue
        rows.append(record("Anthropic", url, source, kind, lastmod, date))
    return rows


def discover_deepmind(timeout: int, retries: int, date: str) -> list[dict[str, Any]]:
    source = "https://deepmind.google/sitemap.xml"
    rows = []
    publication_re = re.compile(r"^/research/publications/\d+/$")
    for url, lastmod in parse_sitemap(fetch(source, timeout, retries), source):
        path = urllib.parse.urlparse(url).path
        if publication_re.fullmatch(path):
            kind = "official_sitemap:publication"
        elif path.startswith("/models/model-cards/") and path.rstrip("/") != "/models/model-cards":
            kind = "official_sitemap:model_card"
        elif "gemini" in path.casefold():
            kind = "official_sitemap:gemini"
        else:
            continue
        rows.append(record("Google DeepMind", url, source, kind, lastmod, date))
    return rows


def _merge(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        row = dict(row)
        key = (row["org"], row["url"])
        if key not in merged:
            merged[key] = row
            merged[key]["source_kinds"] = [row.pop("source_kind")]
            merged[key]["source_urls"] = [row.pop("source_url")]
            continue
        current = merged[key]
        kind = row["source_kind"]
        source_url = row["source_url"]
        if kind not in current["source_kinds"]:
            current["source_kinds"].append(kind)
        if source_url not in current["source_urls"]:
            current["source_urls"].append(source_url)
        if row["lastmod"] and (not current["lastmod"] or row["lastmod"] > current["lastmod"]):
            current["lastmod"] = row["lastmod"]
    result = list(merged.values())
    for row in result:
        row["source_kinds"].sort()
        row["source_urls"].sort()
    return sorted(result, key=lambda row: (row["org"].casefold(), row["url"]))


def _read_jsonl_if_present(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{number}: expected a JSON object")
            rows.append(row)
    return rows


def _preserve_unselected(
    current: list[dict[str, Any]], existing: Iterable[dict[str, Any]], selected: set[str]
) -> list[dict[str, Any]]:
    """Keep other organizations when refreshing only one official source."""
    rows = list(current)
    rows.extend(row for row in existing if row.get("org") not in selected)
    return sorted(rows, key=lambda row: (str(row.get("org", "")).casefold(), row["url"]))


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--openai-rss-output", type=Path, default=DEFAULT_OPENAI_RSS_OUTPUT)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--source",
        choices=("all", "openai", "anthropic", "deepmind"),
        default="all",
    )
    args = parser.parse_args(argv)
    date = _today()
    rows: list[dict[str, Any]] = []
    openai_rss: list[dict[str, Any]] = []
    try:
        if args.source in {"all", "openai"}:
            rows.extend(discover_openai(args.timeout, args.retries, date))
            openai_rss = discover_openai_rss(args.timeout, args.retries, date)
        if args.source in {"all", "anthropic"}:
            rows.extend(discover_anthropic(args.timeout, args.retries, date))
        if args.source in {"all", "deepmind"}:
            rows.extend(discover_deepmind(args.timeout, args.retries, date))
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    merged = _merge(rows)
    if args.source != "all":
        selected = {
            "openai": {"OpenAI"},
            "anthropic": {"Anthropic"},
            "deepmind": {"Google DeepMind"},
        }[args.source]
        try:
            merged = _preserve_unselected(
                merged, _read_jsonl_if_present(args.output), selected
            )
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 2
    _write_jsonl(args.output, merged)
    if openai_rss:
        _write_jsonl(args.openai_rss_output, openai_rss)
    counts = Counter(row["org"] for row in merged)
    print(f"wrote {len(merged)} unique official candidate pages to {args.output}")
    for org, count in sorted(counts.items()):
        print(f"{org}: {count}")
    if openai_rss:
        rss_categories = Counter(row["category"] or "(none)" for row in openai_rss)
        print(f"OpenAI RSS: {len(openai_rss)} items")
        print(
            "OpenAI RSS categories: "
            + ", ".join(f"{key}={value}" for key, value in sorted(rss_categories.items()))
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
