#!/usr/bin/env python3
"""Validate, download, extract, and audit the frontier-lab paper corpus.

The tracked JSONL inventories are authoritative.  Large downloaded artifacts
are reconstructible outputs and stay under the Git-ignored archive directory.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = REPO_ROOT / "research" / "literature" / "inventory"
DEFAULT_ARCHIVE = REPO_ROOT / "research" / "literature" / "archive"

REQUIRED_FIELDS = (
    "id",
    "org",
    "title",
    "authors",
    "date",
    "type",
    "tier",
    "arxiv_id",
    "doi",
    "primary_url",
    "pdf_url",
    "source_pages",
    "affiliation_evidence",
    "topics",
    "notes",
    "retrieved_at",
)
TIERS = {"core", "direct", "affiliated"}
TYPES = {
    "technical_report",
    "research_paper",
    "system_card",
    "model_card",
    "dataset",
    "benchmark",
    "blog_with_report",
    "other",
}
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
ARXIV_RE = re.compile(r"^(?:[a-z-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})$", re.I)
DATE_RE = re.compile(r"^\d{4}(?:-\d{2}-\d{2})?$")
USER_AGENT = "RoseLLM-Literature-Corpus/1.0 (+https://github.com/chtho-like/rosellm)"
SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


class InventoryError(ValueError):
    """An inventory record failed a structural or consistency check."""


def _today() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


def _utf8_safe(text: str) -> str:
    """Replace invalid lone UTF-16 surrogates without hiding their position."""
    return SURROGATE_RE.sub("\ufffd", text)


def _prefer_alternate_text(primary: str, alternate: str) -> bool:
    """Accept a fallback extraction only when it is clearly less damaged."""
    alternate = alternate.strip()
    primary = primary.strip()
    if not alternate:
        return False
    if alternate.count("\ufffd") >= primary.count("\ufffd"):
        return False
    primary_alnum = sum(character.isalnum() for character in primary)
    alternate_alnum = sum(character.isalnum() for character in alternate)
    return alternate_alnum >= max(100, int(primary_alnum * 0.3))


def _poppler_text(source: Path) -> tuple[str | None, str | None]:
    binary = shutil.which("pdftotext")
    if not binary:
        return None, "pdftotext is not installed"
    try:
        completed = subprocess.run(
            [binary, "-layout", "-enc", "UTF-8", str(source), "-"],
            check=False,
            capture_output=True,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if completed.returncode != 0:
        error = completed.stderr.decode("utf-8", errors="replace").strip()
        return None, f"pdftotext exited {completed.returncode}: {error}"
    text = _utf8_safe(completed.stdout.decode("utf-8", errors="replace"))
    return text.strip() + "\n", None


def _normalize_title(title: str) -> str:
    normalized = unicodedata.normalize("NFKC", title).casefold()
    return re.sub(r"[^\w]+", "", normalized)


def _normalize_arxiv(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    value = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", value, flags=re.I)
    value = re.sub(r"\.pdf$", "", value, flags=re.I)
    return re.sub(r"v\d+$", "", value)


def _normalize_doi(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().casefold()
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value)
    return value or None


def _validate_url(value: Any, field: str, location: str, nullable: bool = True) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, str) or not value.strip():
        raise InventoryError(f"{location}: {field} must be a non-empty URL or null")
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise InventoryError(f"{location}: invalid {field}: {value!r}")


def validate_record(record: Any, location: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise InventoryError(f"{location}: expected a JSON object")
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise InventoryError(f"{location}: missing fields: {', '.join(missing)}")

    if not isinstance(record["id"], str) or not SLUG_RE.fullmatch(record["id"]):
        raise InventoryError(f"{location}: id must match {SLUG_RE.pattern!r}")
    for field in ("org", "title", "affiliation_evidence", "retrieved_at"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise InventoryError(f"{location}: {field} must be a non-empty string")
    if record["tier"] not in TIERS:
        raise InventoryError(f"{location}: unsupported tier {record['tier']!r}")
    if record["type"] not in TYPES:
        raise InventoryError(f"{location}: unsupported type {record['type']!r}")
    if record["date"] is not None and (
        not isinstance(record["date"], str) or not DATE_RE.fullmatch(record["date"])
    ):
        raise InventoryError(f"{location}: date must be YYYY, YYYY-MM-DD, or null")
    if not DATE_RE.fullmatch(record["retrieved_at"]):
        raise InventoryError(f"{location}: retrieved_at must be an ISO date")

    for field in ("authors", "source_pages", "topics"):
        if not isinstance(record[field], list) or not all(
            isinstance(item, str) and item.strip() for item in record[field]
        ):
            raise InventoryError(f"{location}: {field} must be an array of non-empty strings")
    if not record["authors"]:
        raise InventoryError(f"{location}: authors must not be empty")
    if not record["source_pages"]:
        raise InventoryError(f"{location}: source_pages must not be empty")
    for url in record["source_pages"]:
        _validate_url(url, "source_pages entry", location, nullable=False)
    _validate_url(record["primary_url"], "primary_url", location)
    _validate_url(record["pdf_url"], "pdf_url", location)

    if record["notes"] is not None and not isinstance(record["notes"], str):
        raise InventoryError(f"{location}: notes must be a string or null")
    for field in ("arxiv_id", "doi"):
        if record[field] is not None and not isinstance(record[field], str):
            raise InventoryError(f"{location}: {field} must be a string or null")

    normalized_arxiv = _normalize_arxiv(record["arxiv_id"])
    if normalized_arxiv and not ARXIV_RE.fullmatch(normalized_arxiv):
        raise InventoryError(f"{location}: invalid arXiv identifier {record['arxiv_id']!r}")
    record = dict(record)
    record["arxiv_id"] = normalized_arxiv
    record["doi"] = _normalize_doi(record["doi"])
    record["_location"] = location
    return record


def iter_records(inventory_dir: Path) -> Iterator[dict[str, Any]]:
    for source in sorted(inventory_dir.glob("*.jsonl")):
        with source.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                location = f"{source}:{line_number}"
                try:
                    decoded = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise InventoryError(f"{location}: invalid JSON: {exc}") from exc
                yield validate_record(decoded, location)


def _allowed_duplicate_title_pairs(
    inventory_dir: Path, equivalence_path: Path | None = None
) -> set[tuple[str, frozenset[str]]]:
    """Return same-organization record pairs explicitly linked as one work.

    Duplicate normalized titles remain an error unless the relationship ledger
    names both records.  This preserves typo/ingestion detection while allowing
    separately citable publication, preprint, and official-asset records to
    retain their source-accurate titles.
    """

    if equivalence_path is None:
        equivalence_path = (
            inventory_dir.parent / "candidates" / "document-equivalences.jsonl"
        )
    if not equivalence_path.exists():
        return set()

    pairs: set[tuple[str, frozenset[str]]] = set()
    with equivalence_path.open("r", encoding="utf-8") as handle:
        for number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            location = f"{equivalence_path}:{number}"
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise InventoryError(f"{location}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise InventoryError(f"{location}: expected a JSON object")
            org = row.get("org")
            first = row.get("publication_id")
            second = row.get("preprint_id")
            if not all(isinstance(value, str) and value for value in (org, first, second)):
                continue
            if first == second:
                raise InventoryError(f"{location}: equivalence pair repeats one record ID")
            pairs.add((org.casefold(), frozenset((first, second))))
    return pairs


def load_records(
    inventory_dir: Path, equivalence_path: Path | None = None
) -> list[dict[str, Any]]:
    records = list(iter_records(inventory_dir))
    allowed_title_pairs = _allowed_duplicate_title_pairs(
        inventory_dir, equivalence_path
    )
    seen_ids: dict[tuple[str, str], str] = {}
    seen_identifiers: dict[tuple[str, str, str], str] = {}
    seen_titles: dict[tuple[str, str], tuple[str, str]] = {}
    errors: list[str] = []

    for record in records:
        org_key = record["org"].casefold()
        id_key = (org_key, record["id"])
        if id_key in seen_ids:
            errors.append(
                f"{record['_location']}: duplicate org/id; first seen at {seen_ids[id_key]}"
            )
        else:
            seen_ids[id_key] = record["_location"]

        for kind in ("arxiv_id", "doi"):
            if not record[kind]:
                continue
            key = (org_key, kind, record[kind].casefold())
            if key in seen_identifiers:
                errors.append(
                    f"{record['_location']}: duplicate {kind} {record[kind]!r}; "
                    f"first seen at {seen_identifiers[key]}"
                )
            else:
                seen_identifiers[key] = record["_location"]

        title_key = (org_key, _normalize_title(record["title"]))
        if title_key in seen_titles:
            first_location, first_id = seen_titles[title_key]
            pair = (org_key, frozenset((first_id, record["id"])))
            if pair not in allowed_title_pairs:
                errors.append(
                    f"{record['_location']}: duplicate normalized title; "
                    f"first seen at {first_location}"
                )
        else:
            seen_titles[title_key] = (record["_location"], record["id"])

    if errors:
        raise InventoryError("\n".join(errors))
    return records


def _selected(
    records: Iterable[dict[str, Any]], org: str | None, tier: str | None
) -> list[dict[str, Any]]:
    result = []
    for record in records:
        if org and record["org"].casefold() != org.casefold():
            continue
        if tier and record["tier"] != tier:
            continue
        result.append(record)
    return result


def _rewrite_pdf_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.casefold()
    path = parsed.path
    if host in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"} and path.startswith(
        "/abs/"
    ):
        return urllib.parse.urlunparse(parsed._replace(path=path.replace("/abs/", "/pdf/", 1)))
    if host == "github.com" and "/blob/" in path:
        owner, repo, _blob, ref, *parts = path.strip("/").split("/")
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{'/'.join(parts)}"
    if host == "huggingface.co" and "/blob/" in path:
        return url.replace("/blob/", "/resolve/", 1)
    if host in {"psyarxiv.com", "www.psyarxiv.com"}:
        match = re.fullmatch(r"/([a-z0-9]+)/download/?", path, re.I)
        if match:
            # PsyArXiv's legacy endpoint now serves an HTML shell.  OSF owns
            # the underlying preprint object and keeps this stable redirect to
            # its current primary file.
            return f"https://osf.io/download/{match.group(1)}"
    if host == "proceedings.mlr.press" and parsed.scheme == "http":
        return urllib.parse.urlunparse(parsed._replace(scheme="https"))
    return url


def _looks_like_pdf_url(url: str | None) -> bool:
    if not url:
        return False
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.casefold().rstrip("/")
    if path.endswith(".pdf"):
        return True
    # OpenReview's stable document endpoint identifies the PDF in the query
    # string rather than the path suffix.
    return parsed.netloc.casefold() == "openreview.net" and path in {
        "/pdf",
        "/references/pdf",
    }


def _candidate_pdf_urls(record: dict[str, Any]) -> list[str]:
    urls = []
    if record["pdf_url"]:
        urls.append(_rewrite_pdf_url(record["pdf_url"]))
    if record["arxiv_id"]:
        urls.append(f"https://arxiv.org/pdf/{record['arxiv_id']}")
    for url in [record.get("primary_url"), *record.get("source_pages", [])]:
        if _looks_like_pdf_url(url):
            urls.append(_rewrite_pdf_url(url))
    return list(dict.fromkeys(urls))


def _resolved_pdf_url(record: dict[str, Any]) -> str | None:
    urls = _candidate_pdf_urls(record)
    return urls[0] if urls else None


def _artifact_path(archive: Path, record: dict[str, Any], suffix: str) -> Path:
    org_slug = re.sub(r"[^a-z0-9]+", "-", record["org"].casefold()).strip("-")
    return archive / suffix.lstrip(".") / org_slug / f"{record['id']}.{suffix.lstrip('.')}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pdf_is_valid(path: Path) -> tuple[bool, str | None]:
    try:
        with path.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                return False, "missing PDF magic header"
    except OSError as exc:
        return False, str(exc)
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo:
        try:
            checked = subprocess.run(
                [pdfinfo, str(path)],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"pdfinfo failed: {exc}"
        if checked.returncode != 0:
            detail = (checked.stderr or checked.stdout).strip().splitlines()
            return False, detail[-1] if detail else "pdfinfo rejected file"
    return True, None


def _download_one(
    record: dict[str, Any], archive: Path, timeout: int, retries: int, force: bool
) -> dict[str, Any]:
    urls = _candidate_pdf_urls(record)
    result: dict[str, Any] = {
        "org": record["org"],
        "id": record["id"],
        "title": record["title"],
        "tier": record["tier"],
        "url": urls[0] if urls else None,
        "attempted_urls": [],
        "path": None,
        "status": "missing_pdf_url",
        "sha256": None,
        "bytes": None,
        "content_type": None,
        "retrieved_at": _today(),
        "error": None,
        "inventory_location": record["_location"],
    }
    if not urls:
        return result

    destination = _artifact_path(archive, record, "pdf")
    result["path"] = str(destination.relative_to(REPO_ROOT))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        valid, error = _pdf_is_valid(destination)
        if valid:
            result.update(
                status="existing",
                sha256=_sha256(destination),
                bytes=destination.stat().st_size,
            )
            return result
        result["error"] = f"existing artifact invalid: {error}"

    part = destination.with_suffix(destination.suffix + ".part")
    last_error = None
    attempted_errors = []
    for url in urls:
        result["attempted_urls"].append(url)
        for attempt in range(retries + 1):
            try:
                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": USER_AGENT,
                        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.1",
                        "Accept-Encoding": "identity",
                    },
                )
                with urllib.request.urlopen(request, timeout=timeout) as response, part.open(
                    "wb"
                ) as handle:
                    result["content_type"] = response.headers.get_content_type()
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
                valid, error = _pdf_is_valid(part)
                if not valid:
                    raise OSError(error or "invalid PDF")
                os.replace(part, destination)
                result.update(
                    url=url,
                    status="downloaded",
                    sha256=_sha256(destination),
                    bytes=destination.stat().st_size,
                    error=None,
                )
                return result
            except Exception as exc:  # network/PDF stacks expose many exception types
                last_error = f"{type(exc).__name__}: {exc}"
                if part.exists():
                    part.unlink()
                if attempt < retries:
                    time.sleep(min(2**attempt, 8))
        attempted_errors.append(f"{url}: {last_error}")
    result.update(status="failed", error=" | ".join(attempted_errors))
    return result


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


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
                raise InventoryError(f"{path}:{number}: invalid manifest JSON: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _merge_manifest_rows(
    previous: Iterable[dict[str, Any]], current: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Replace matching org/id rows while preserving other resumable runs."""
    merged = {
        (row.get("org"), row.get("id")): row
        for row in previous
        if row.get("org") is not None and row.get("id") is not None
    }
    for row in current:
        merged[(row.get("org"), row.get("id"))] = row
    return sorted(
        merged.values(), key=lambda row: (str(row.get("org", "")).casefold(), str(row.get("id", "")))
    )


def _prune_manifest_rows(
    rows: Iterable[dict[str, Any]], records: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Discard stale manifest entries whose inventory record no longer exists."""
    valid = {(record["org"].casefold(), record["id"]) for record in records}
    return [
        row
        for row in rows
        if (str(row.get("org", "")).casefold(), str(row.get("id", ""))) in valid
    ]


def cmd_validate(args: argparse.Namespace) -> int:
    records = load_records(args.inventory)
    selected = _selected(records, args.org, args.tier)
    print(f"validated {len(records)} records; selected {len(selected)}")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    all_records = load_records(args.inventory)
    records = _selected(all_records, args.org, args.tier)
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _download_one,
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
    manifest_path = args.archive / "manifest.jsonl"
    combined = _prune_manifest_rows(
        _merge_manifest_rows(_read_jsonl_if_present(manifest_path), results),
        all_records,
    )
    _write_jsonl(manifest_path, combined)
    counts = Counter(result["status"] for result in results)
    print("download summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 1 if counts["failed"] else 0


def _extract_one(
    record: dict[str, Any],
    archive: Path,
    force: bool,
    repair_quality: bool = False,
) -> dict[str, Any]:
    source = _artifact_path(archive, record, "pdf")
    destination = _artifact_path(archive, record, "txt")
    result = {
        "org": record["org"],
        "id": record["id"],
        "source": str(source.relative_to(REPO_ROOT)),
        "text_path": str(destination.relative_to(REPO_ROOT)),
        "status": "missing_pdf",
        "pages": None,
        "characters": None,
        "engine": None,
        "replacement_characters": None,
        "fallback_attempted": False,
        "fallback_error": None,
        "error": None,
    }
    if not source.exists():
        return result
    previous_text = None
    if destination.exists() and not force:
        if repair_quality:
            previous_text = destination.read_text(encoding="utf-8", errors="replace")
            if "\ufffd" in previous_text:
                pass
            else:
                result.update(status="existing", characters=destination.stat().st_size)
                return result
        else:
            result.update(status="existing", characters=destination.stat().st_size)
            return result

    text = None
    pages_count = None
    engine = None
    primary_error = None
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(source), strict=False)
        pages = [(page.extract_text() or "") for page in reader.pages]
        text = _utf8_safe("\n\n".join(pages).strip() + "\n")
        pages_count = len(reader.pages)
        engine = "pypdf"
    except Exception as exc:  # malformed PDFs expose many library-specific errors
        primary_error = f"{type(exc).__name__}: {exc}"

    poppler_error = None
    if text is None or "\ufffd" in text:
        result["fallback_attempted"] = True
        alternate, poppler_error = _poppler_text(source)
        result["fallback_error"] = poppler_error
        if alternate is not None and (
            text is None or _prefer_alternate_text(text, alternate)
        ):
            text = alternate
            engine = "pdftotext"

    if text is None:
        errors = [value for value in (primary_error, poppler_error) if value]
        result.update(status="failed", error="; ".join(errors) or "no text extracted")
        temporary = destination.with_suffix(".txt.tmp")
        if temporary.exists():
            temporary.unlink()
        return result

    replacements = text.count("\ufffd")
    if previous_text is not None and text == previous_text:
        result.update(status="existing", characters=destination.stat().st_size)
        result.update(
            pages=pages_count,
            engine=engine,
            replacement_characters=replacements,
        )
        return result
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".txt.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, destination)
    result.update(
        status="repaired" if previous_text is not None else "extracted",
        pages=pages_count,
        characters=len(text),
        engine=engine,
        replacement_characters=replacements,
    )
    return result


def cmd_extract(args: argparse.Namespace) -> int:
    all_records = load_records(args.inventory)
    records = _selected(all_records, args.org, args.tier)
    results: list[dict[str, Any]] = []
    if args.workers == 1:
        for index, record in enumerate(records, 1):
            result = _extract_one(
                record, args.archive, args.force, args.repair_quality
            )
            results.append(result)
            print(
                f"[{index}/{len(records)}] {result['org']}:{result['id']} "
                f"{result['status']}"
            )
    else:
        # PDF text extraction is primarily CPU-bound.  Separate processes avoid
        # the Python GIL while each worker writes to a distinct record path;
        # only the parent process merges the manifest after all workers finish.
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers
        ) as pool:
            futures = {
                pool.submit(
                    _extract_one,
                    record,
                    args.archive,
                    args.force,
                    args.repair_quality,
                ): record
                for record in records
            }
            for index, future in enumerate(
                concurrent.futures.as_completed(futures), 1
            ):
                result = future.result()
                results.append(result)
                print(
                    f"[{index}/{len(futures)}] {result['org']}:{result['id']} "
                    f"{result['status']}"
                )
    manifest_path = args.archive / "extraction-manifest.jsonl"
    combined = _prune_manifest_rows(
        _merge_manifest_rows(_read_jsonl_if_present(manifest_path), results),
        all_records,
    )
    _write_jsonl(manifest_path, combined)
    counts = Counter(result["status"] for result in results)
    print("extraction summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 1 if counts["failed"] else 0


def _year(record: dict[str, Any]) -> str:
    return record["date"][:4] if record["date"] else "unknown"


def cmd_audit(args: argparse.Namespace) -> int:
    records = _selected(load_records(args.inventory), args.org, args.tier)
    by_org: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        by_org[record["org"]][f"tier:{record['tier']}"] += 1
        by_org[record["org"]][f"type:{record['type']}"] += 1
        by_org[record["org"]][f"year:{_year(record)}"] += 1
        if _resolved_pdf_url(record):
            by_org[record["org"]]["has_pdf_url"] += 1
        if _artifact_path(args.archive, record, "pdf").exists():
            by_org[record["org"]]["has_local_pdf"] += 1
        if _artifact_path(args.archive, record, "txt").exists():
            by_org[record["org"]]["has_text"] += 1

    print(f"total records: {len(records)}")
    for org in sorted(by_org, key=str.casefold):
        counts = by_org[org]
        print(f"\n{org}: {sum(v for k, v in counts.items() if k.startswith('tier:'))}")
        for tier in sorted(TIERS):
            print(f"  {tier}: {counts[f'tier:{tier}']}")
        print(f"  public PDF URL or arXiv: {counts['has_pdf_url']}")
        print(f"  local PDF: {counts['has_local_pdf']}")
        print(f"  extracted text: {counts['has_text']}")
        types = sorted((k[5:], v) for k, v in counts.items() if k.startswith("type:"))
        print("  types: " + ", ".join(f"{key}={value}" for key, value in types))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--org", help="exact organization label, case-insensitive")
    parser.add_argument("--tier", choices=sorted(TIERS))
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate JSONL and duplicates")
    validate.set_defaults(func=cmd_validate)

    audit = subparsers.add_parser("audit", help="print inventory and artifact coverage")
    audit.set_defaults(func=cmd_audit)

    download = subparsers.add_parser("download", help="download and hash public PDFs")
    download.add_argument("--workers", type=int, default=6)
    download.add_argument("--timeout", type=int, default=120)
    download.add_argument("--retries", type=int, default=3)
    download.add_argument("--force", action="store_true")
    download.set_defaults(func=cmd_download)

    extract = subparsers.add_parser("extract", help="extract searchable PDF text")
    extract.add_argument("--force", action="store_true")
    extract.add_argument(
        "--repair-quality",
        action="store_true",
        help="re-extract existing texts containing Unicode replacement characters",
    )
    extract.add_argument("--workers", type=int, default=4)
    extract.set_defaults(func=cmd_extract)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "workers", 1) < 1:
        raise SystemExit("--workers must be at least 1")
    try:
        return args.func(args)
    except InventoryError as exc:
        print(f"inventory error:\n{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
