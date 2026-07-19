#!/usr/bin/env python3
"""Archive official candidate pages and extract paper/report link evidence."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import html.parser
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = (
    REPO_ROOT / "research" / "literature" / "candidates" / "official-pages.jsonl"
)
DEFAULT_EVIDENCE = (
    REPO_ROOT / "research" / "literature" / "candidates" / "page-evidence.jsonl"
)
DEFAULT_OPENAI_RSS = (
    REPO_ROOT / "research" / "literature" / "candidates" / "openai-rss.jsonl"
)
DEFAULT_ARCHIVE = REPO_ROOT / "research" / "literature" / "archive" / "pages"
USER_AGENT = "RoseLLM-Literature-Scanner/1.0 (+https://github.com/chtho-like/rosellm)"
MAX_PAGE_BYTES = 20 * 1024 * 1024

PAPER_HOSTS = {
    "arxiv.org",
    "doi.org",
    "dx.doi.org",
    "openreview.net",
    "aclanthology.org",
    "proceedings.mlr.press",
    "dl.acm.org",
    "ieeexplore.ieee.org",
    "journals.plos.org",
    "link.springer.com",
    "nature.com",
    "www.nature.com",
    "science.org",
    "www.science.org",
    "papers.ssrn.com",
    "research.google",
}
PAPER_LABEL_RE = re.compile(
    r"\b(read|view|download)\s+(the\s+)?(paper|report|publication|system card|model card)\b|"
    r"\b(technical report|system card|model card|full paper|arxiv|doi)\b",
    re.I,
)


def _today() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


class PageParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.title_parts: list[str] = []
        self.in_title = False
        self.current_link: str | None = None
        self.current_link_text: list[str] = []
        self.links: list[tuple[str, str]] = []
        self.time_values: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.casefold(): value for key, value in attrs if value is not None}
        tag = tag.casefold()
        if tag == "title":
            self.in_title = True
        elif tag == "meta":
            key = values.get("property") or values.get("name") or values.get("itemprop")
            content = values.get("content")
            if key and content:
                self.meta.setdefault(key.casefold(), content.strip())
        elif tag == "a" and values.get("href"):
            self.current_link = values["href"]
            self.current_link_text = []
        elif tag == "time" and values.get("datetime"):
            self.time_values.append(values["datetime"])

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag == "title":
            self.in_title = False
        elif tag == "a" and self.current_link is not None:
            text = re.sub(r"\s+", " ", " ".join(self.current_link_text)).strip()
            self.links.append((self.current_link, text))
            self.current_link = None
            self.current_link_text = []

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)
        if self.current_link is not None and data.strip():
            self.current_link_text.append(data.strip())

    @property
    def title(self) -> str | None:
        for key in ("og:title", "twitter:title"):
            if self.meta.get(key):
                return self.meta[key]
        value = re.sub(r"\s+", " ", " ".join(self.title_parts)).strip()
        return value or None

    @property
    def published_at(self) -> str | None:
        for key in (
            "article:published_time",
            "datepublished",
            "publish-date",
            "parsely-pub-date",
        ):
            if self.meta.get(key):
                return self.meta[key]
        return self.time_values[0] if self.time_values else None


def _interesting_links(parser: PageParser, base_url: str) -> list[dict[str, str]]:
    selected: dict[str, str] = {}
    for href, label in parser.links:
        absolute = urllib.parse.urljoin(base_url, href)
        parsed = urllib.parse.urlparse(absolute)
        if parsed.scheme not in {"http", "https"}:
            continue
        path = parsed.path.casefold()
        host = parsed.netloc.casefold().split(":", 1)[0]
        is_pdf = path.endswith(".pdf") or ".pdf/" in path
        is_paper_host = host in PAPER_HOSTS
        is_paper_label = bool(PAPER_LABEL_RE.search(label))
        if is_pdf or is_paper_host or is_paper_label:
            clean = urllib.parse.urlunparse(parsed._replace(fragment=""))
            selected.setdefault(clean, label)
    return [{"url": url, "label": selected[url]} for url in sorted(selected)]


def _archive_path(root: Path, org: str, url: str) -> Path:
    org_slug = re.sub(r"[^a-z0-9]+", "-", org.casefold()).strip("-")
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    return root / org_slug / f"{digest}.html"


def _manifest_path(path: Path) -> str:
    """Use repository-relative artifact paths when possible."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fetch_page(url: str, timeout: int, retries: int) -> tuple[bytes, str, str, int]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
                    "Accept-Encoding": "identity",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                length = response.headers.get("Content-Length")
                if length and int(length) > MAX_PAGE_BYTES:
                    raise OSError(f"page exceeds {MAX_PAGE_BYTES} bytes")
                payload = response.read(MAX_PAGE_BYTES + 1)
                if len(payload) > MAX_PAGE_BYTES:
                    raise OSError(f"page exceeds {MAX_PAGE_BYTES} bytes")
                return (
                    payload,
                    response.geturl(),
                    response.headers.get_content_type(),
                    response.status,
                )
        except (TimeoutError, OSError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(str(last_error))


def scan_one(
    candidate: dict[str, Any], archive: Path, timeout: int, retries: int, force: bool
) -> dict[str, Any]:
    url = candidate["url"]
    destination = _archive_path(archive, candidate["org"], url)
    result: dict[str, Any] = {
        "org": candidate["org"],
        "url": url,
        "source_kinds": candidate["source_kinds"],
        "status": "failed",
        "http_status": None,
        "final_url": None,
        "content_type": None,
        "title": None,
        "description": None,
        "published_at": None,
        "paper_links": [],
        "archive_path": _manifest_path(destination),
        "sha256": None,
        "bytes": None,
        "scanned_at": _today(),
        "error": None,
    }
    try:
        if destination.exists() and not force:
            payload = destination.read_bytes()
            final_url, content_type, status = url, "text/html", 200
            result["status"] = "existing"
        else:
            payload, final_url, content_type, status = _fetch_page(url, timeout, retries)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".html.tmp")
            temporary.write_bytes(payload)
            os.replace(temporary, destination)
            result["status"] = "downloaded"
        parser = PageParser()
        parser.feed(payload.decode("utf-8", errors="replace"))
        result.update(
            http_status=status,
            final_url=final_url,
            content_type=content_type,
            title=parser.title,
            description=parser.meta.get("description") or parser.meta.get("og:description"),
            published_at=parser.published_at,
            paper_links=_interesting_links(parser, final_url),
            sha256=_sha256_bytes(payload),
            bytes=len(payload),
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    return rows


def _candidates_from_inventory(
    inventory_dir: Path, missing_pdf_only: bool
) -> list[dict[str, Any]]:
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    for source in sorted(inventory_dir.glob("*.jsonl")):
        for row in _read_jsonl(source):
            url = row.get("primary_url")
            org = row.get("org")
            if not isinstance(url, str) or not url or not isinstance(org, str) or not org:
                continue
            if missing_pdf_only and (row.get("pdf_url") or row.get("arxiv_id")):
                continue
            key = (org, url)
            candidates.setdefault(
                key,
                {
                    "org": org,
                    "url": url,
                    "source_kinds": ["inventory:primary_url"],
                },
            )
    return sorted(candidates.values(), key=lambda row: (row["org"].casefold(), row["url"]))


def _url_key(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return urllib.parse.urlunparse(parsed._replace(path=path, query="", fragment=""))


def rss_evidence(candidate: dict[str, Any], rss_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "org": candidate["org"],
        "url": candidate["url"],
        "source_kinds": candidate["source_kinds"],
        "status": "rss_metadata_only",
        "http_status": None,
        "final_url": rss_row["url"],
        "content_type": "application/rss+xml",
        "title": rss_row["title"],
        "description": rss_row["description"],
        "published_at": rss_row["published_at"],
        "paper_links": [],
        "archive_path": None,
        "sha256": None,
        "bytes": None,
        "scanned_at": _today(),
        "error": "Direct page fetch unavailable; metadata sourced from official OpenAI RSS.",
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument(
        "--inventory-dir",
        type=Path,
        help="derive candidate primary URLs from inventory JSONL files instead of the sitemap queue",
    )
    parser.add_argument(
        "--missing-pdf-only",
        action="store_true",
        help="with --inventory-dir, snapshot only records without a PDF URL or arXiv identifier",
    )
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--openai-rss", type=Path, default=DEFAULT_OPENAI_RSS)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--org", help="exact organization label, case-insensitive")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--rss-only",
        action="store_true",
        help="for OpenAI, join official RSS metadata without fetching Cloudflare-protected pages",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    if args.missing_pdf_only and not args.inventory_dir:
        parser.error("--missing-pdf-only requires --inventory-dir")
    try:
        candidates = (
            _candidates_from_inventory(args.inventory_dir, args.missing_pdf_only)
            if args.inventory_dir
            else _read_jsonl(args.candidates)
        )
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    if args.org:
        candidates = [
            row for row in candidates if row["org"].casefold() == args.org.casefold()
        ]

    rss_by_url: dict[str, dict[str, Any]] = {}
    if args.openai_rss.exists():
        rss_by_url = {_url_key(row["url"]): row for row in _read_jsonl(args.openai_rss)}

    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if args.evidence.exists() and not args.force:
        for row in _read_jsonl(args.evidence):
            existing[(row["org"], row["url"])] = row

    results: list[dict[str, Any]] = []
    pending = []
    for candidate in candidates:
        old = existing.get((candidate["org"], candidate["url"]))
        if old and old["status"] in {"downloaded", "existing", "rss_metadata_only"}:
            results.append(old)
        elif args.rss_only and candidate["org"] == "OpenAI" and _url_key(candidate["url"]) in rss_by_url:
            results.append(rss_evidence(candidate, rss_by_url[_url_key(candidate["url"])]))
        elif args.rss_only:
            results.append(
                {
                    "org": candidate["org"],
                    "url": candidate["url"],
                    "source_kinds": candidate["source_kinds"],
                    "status": "rss_unmatched",
                    "http_status": None,
                    "final_url": None,
                    "content_type": None,
                    "title": None,
                    "description": None,
                    "published_at": None,
                    "paper_links": [],
                    "archive_path": None,
                    "sha256": None,
                    "bytes": None,
                    "scanned_at": _today(),
                    "error": "Official candidate URL was not present in the OpenAI RSS snapshot.",
                }
            )
        else:
            pending.append(candidate)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                scan_one, candidate, args.archive, args.timeout, args.retries, args.force
            ): candidate
            for candidate in pending
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            results.append(result)
            if index % 25 == 0 or result["status"] == "failed":
                print(
                    f"[{index}/{len(futures)}] {result['org']} {result['status']} "
                    f"{result['url']}"
                )

    # Preserve evidence for organizations outside a filtered run.
    if args.org and args.evidence.exists():
        selected_keys = {(row["org"], row["url"]) for row in results}
        for row in _read_jsonl(args.evidence):
            key = (row["org"], row["url"])
            if key not in selected_keys:
                results.append(row)

    results.sort(key=lambda row: (row["org"].casefold(), row["url"]))
    _write_jsonl(args.evidence, results)
    failed = [row for row in results if row["status"] in {"failed", "rss_unmatched"}]
    linked = sum(bool(row["paper_links"]) for row in results)
    print(
        f"wrote {len(results)} evidence rows; paper/report links on {linked}; "
        f"failed {len(failed)}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
