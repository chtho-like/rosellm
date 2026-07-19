#!/usr/bin/env python3
"""Recover missing DOI research papers from the PMC Open Access subset.

This is a conservative third-pass recovery tool. It batches exact DOI queries
through Europe PMC, requires an exact DOI match plus a PMCID and ``hasPDF=Y``,
then asks the NCBI PMC OA Web Service for an official PDF file. Only PDF links
on NCBI's OA FTP host are considered. If such a legacy link fails, the tool
discovers the newest exact-PMCID version in NCBI's public PMC AWS dataset and
requires its canonical versioned PDF object. The downloaded bytes must pass the
main corpus driver's PDF magic and ``pdfinfo`` checks before they are atomically
linked into the canonical archive.

The inventory is read-only. A concurrently created canonical artifact always
wins: this tool records ``skipped_existing`` and never forces or overwrites it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from literature_corpus import (
    DEFAULT_ARCHIVE,
    DEFAULT_INVENTORY,
    REPO_ROOT,
    USER_AGENT,
    _artifact_path,
    _download_one,
    _normalize_doi,
    _read_jsonl_if_present,
    _write_jsonl,
    load_records,
)


DEFAULT_OUTPUT = (
    REPO_ROOT / "research" / "literature" / "candidates" / "pmc-recovery.jsonl"
)
DEFAULT_TMP = REPO_ROOT / "tmp" / "pdfs"
EUROPE_PMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
NCBI_OA_API = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
PMC_AWS_BUCKET = "pmc-oa-opendata"
PMC_AWS_BASE = f"https://{PMC_AWS_BUCKET}.s3.amazonaws.com"
S3_XML_NAMESPACE = "http://s3.amazonaws.com/doc/2006-03-01/"
PMCID_RE = re.compile(r"^PMC\d+$", re.I)
NCBI_PDF_HOST = "ftp.ncbi.nlm.nih.gov"
NCBI_PDF_PREFIX = "/pub/pmc/oa_pdf/"
RETRYABLE_STATUSES = {
    "candidate_failed",
    "europe_pmc_failed",
    "ncbi_oa_failed",
    "skipped_existing",
}


def _fetch_bytes(url: str, accept: str, timeout: int, retries: int) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": accept,
                    "Accept-Encoding": "identity",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"{type(last_error).__name__}: {last_error}")


def _escape_query_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _europe_pmc_query_url(dois: Iterable[str]) -> str:
    normalized = list(
        dict.fromkeys(
            value
            for value in (_normalize_doi(doi) for doi in dois)
            if value is not None
        )
    )
    if not normalized:
        raise ValueError("at least one DOI is required")
    query = " OR ".join(f'DOI:"{_escape_query_value(doi)}"' for doi in normalized)
    parameters = urllib.parse.urlencode(
        {
            "query": query,
            "format": "json",
            "resultType": "lite",
            "pageSize": 1000,
        }
    )
    return f"{EUROPE_PMC_SEARCH}?{parameters}"


def _yes(value: Any) -> bool:
    if value is True:
        return True
    return str(value or "").strip().casefold() in {"1", "true", "y", "yes"}


def _normalize_pmcid(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    return normalized if PMCID_RE.fullmatch(normalized) else None


def _europe_pmc_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": row.get("source"),
        "id": row.get("id"),
        "doi": _normalize_doi(row.get("doi")),
        "pmcid": _normalize_pmcid(row.get("pmcid")),
        "hasPDF": row.get("hasPDF"),
        "isOpenAccess": row.get("isOpenAccess"),
        "title": row.get("title"),
    }


def _fetch_europe_pmc_many(
    dois: Iterable[str], timeout: int, retries: int
) -> tuple[str, list[dict[str, Any]]]:
    url = _europe_pmc_query_url(dois)
    try:
        decoded = json.loads(_fetch_bytes(url, "application/json", timeout, retries))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"invalid Europe PMC JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("Europe PMC response was not an object")
    result_list = decoded.get("resultList") or {}
    raw_rows = result_list.get("result") or []
    if not isinstance(raw_rows, list) or not all(
        isinstance(row, dict) for row in raw_rows
    ):
        raise RuntimeError("Europe PMC result list was malformed")
    try:
        hit_count = int(decoded.get("hitCount") or 0)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Europe PMC hit count was malformed") from exc
    if hit_count > len(raw_rows):
        raise RuntimeError(
            f"Europe PMC response was truncated: {len(raw_rows)}/{hit_count}"
        )
    return url, [_europe_pmc_row(row) for row in raw_rows]


def _select_europe_pmc_match(
    doi: str, rows: Iterable[dict[str, Any]]
) -> tuple[str, dict[str, Any] | None, list[dict[str, Any]]]:
    expected = _normalize_doi(doi)
    exact = sorted(
        (row for row in rows if _normalize_doi(row.get("doi")) == expected),
        key=lambda row: (
            str(row.get("pmcid") or ""),
            str(row.get("source") or ""),
            str(row.get("id") or ""),
        ),
    )
    if not exact:
        return "no_exact_match", None, []
    confirmed = [
        row
        for row in exact
        if _normalize_pmcid(row.get("pmcid")) and _yes(row.get("hasPDF"))
    ]
    if not confirmed:
        return "no_confirmed_pdf", None, exact
    by_pmcid: dict[str, list[dict[str, Any]]] = {}
    for row in confirmed:
        pmcid = _normalize_pmcid(row.get("pmcid"))
        assert pmcid is not None
        by_pmcid.setdefault(pmcid, []).append(row)
    if len(by_pmcid) != 1:
        return "ambiguous_pmcid", None, exact
    pmcid = next(iter(by_pmcid))
    selected = sorted(
        by_pmcid[pmcid],
        key=lambda row: (
            not _yes(row.get("isOpenAccess")),
            str(row.get("source") or ""),
            str(row.get("id") or ""),
        ),
    )[0]
    return "matched", selected, exact


def _ncbi_oa_url(pmcid: str) -> str:
    return f"{NCBI_OA_API}?{urllib.parse.urlencode({'id': pmcid})}"


def _official_ncbi_pdf_url(href: Any) -> str | None:
    if not isinstance(href, str) or not href.strip():
        return None
    parsed = urllib.parse.urlparse(href.strip())
    if (
        parsed.scheme.casefold() not in {"ftp", "https"}
        or (parsed.hostname or "").casefold() != NCBI_PDF_HOST
        or not parsed.path.startswith(NCBI_PDF_PREFIX)
    ):
        return None
    return urllib.parse.urlunparse(parsed._replace(scheme="https"))


def _parse_ncbi_oa_response(payload: bytes, pmcid: str) -> dict[str, Any]:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise RuntimeError(f"invalid NCBI OA XML: {exc}") from exc
    expected = _normalize_pmcid(pmcid)
    records = [
        record
        for record in root.findall(".//record")
        if _normalize_pmcid(record.get("id")) == expected
    ]
    if len(records) != 1:
        return {
            "record_id": None,
            "citation": None,
            "license": None,
            "retracted": None,
            "pdf_urls": [],
        }
    record = records[0]
    urls = []
    for link in record.findall("./link"):
        if str(link.get("format") or "").casefold() != "pdf":
            continue
        url = _official_ncbi_pdf_url(link.get("href"))
        if url:
            urls.append(url)
    return {
        "record_id": expected,
        "citation": record.get("citation"),
        "license": record.get("license"),
        "retracted": record.get("retracted"),
        "pdf_urls": list(dict.fromkeys(urls)),
    }


def _fetch_ncbi_oa(
    pmcid: str, timeout: int, retries: int
) -> tuple[str, dict[str, Any]]:
    url = _ncbi_oa_url(pmcid)
    payload = _fetch_bytes(url, "application/xml,text/xml;q=0.9", timeout, retries)
    return url, _parse_ncbi_oa_response(payload, pmcid)


def _pmc_aws_list_url(prefix: str, delimiter: str | None = None) -> str:
    parameters = {"list-type": "2", "prefix": prefix}
    if delimiter is not None:
        parameters["delimiter"] = delimiter
    return f"{PMC_AWS_BASE}/?{urllib.parse.urlencode(parameters)}"


def _parse_pmc_aws_listing(payload: bytes) -> ET.Element:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise RuntimeError(f"invalid PMC AWS XML: {exc}") from exc
    namespace = f"{{{S3_XML_NAMESPACE}}}"
    if root.tag != f"{namespace}ListBucketResult":
        raise RuntimeError("PMC AWS response was not an S3 bucket listing")
    if root.findtext(f"{namespace}Name") != PMC_AWS_BUCKET:
        raise RuntimeError("PMC AWS response named an unexpected bucket")
    if str(root.findtext(f"{namespace}IsTruncated") or "").casefold() != "false":
        raise RuntimeError("PMC AWS response was truncated")
    return root


def _parse_pmc_aws_versions(payload: bytes, pmcid: str) -> list[str]:
    expected = _normalize_pmcid(pmcid)
    if expected is None:
        raise RuntimeError(f"invalid PMCID for PMC AWS lookup: {pmcid!r}")
    root = _parse_pmc_aws_listing(payload)
    namespace = f"{{{S3_XML_NAMESPACE}}}"
    version_re = re.compile(rf"{re.escape(expected)}\.([1-9]\d*)/")
    versions: dict[int, str] = {}
    for common_prefix in root.findall(f"{namespace}CommonPrefixes"):
        prefix = common_prefix.findtext(f"{namespace}Prefix")
        match = version_re.fullmatch(str(prefix or "").upper())
        if match:
            versions[int(match.group(1))] = f"{expected}.{int(match.group(1))}"
    return [versions[number] for number in sorted(versions)]


def _parse_pmc_aws_pdf_url(payload: bytes, pmcid: str, version: str) -> str | None:
    expected = _normalize_pmcid(pmcid)
    if expected is None or not re.fullmatch(
        rf"{re.escape(expected)}\.[1-9]\d*", version, re.I
    ):
        raise RuntimeError(f"invalid PMC AWS version for {pmcid!r}: {version!r}")
    canonical_version = version.upper()
    canonical_key = f"{canonical_version}/{canonical_version}.pdf"
    root = _parse_pmc_aws_listing(payload)
    namespace = f"{{{S3_XML_NAMESPACE}}}"
    keys = [
        content.findtext(f"{namespace}Key")
        for content in root.findall(f"{namespace}Contents")
    ]
    if keys.count(canonical_key) != 1:
        return None
    quoted = urllib.parse.quote(canonical_key, safe="/")
    return f"{PMC_AWS_BASE}/{quoted}"


def _fetch_pmc_aws_pdf(pmcid: str, timeout: int, retries: int) -> dict[str, Any]:
    expected = _normalize_pmcid(pmcid)
    if expected is None:
        raise RuntimeError(f"invalid PMCID for PMC AWS lookup: {pmcid!r}")
    versions_url = _pmc_aws_list_url(f"{expected}.", "/")
    versions_payload = _fetch_bytes(
        versions_url, "application/xml,text/xml;q=0.9", timeout, retries
    )
    versions = _parse_pmc_aws_versions(versions_payload, expected)
    result: dict[str, Any] = {
        "versions_api_url": versions_url,
        "versions": versions,
        "selected_version": None,
        "objects_api_url": None,
        "pdf_url": None,
    }
    if not versions:
        return result

    selected = versions[-1]
    objects_url = _pmc_aws_list_url(f"{selected}/")
    objects_payload = _fetch_bytes(
        objects_url, "application/xml,text/xml;q=0.9", timeout, retries
    )
    result.update(
        selected_version=selected,
        objects_api_url=objects_url,
        pdf_url=_parse_pmc_aws_pdf_url(objects_payload, expected, selected),
    )
    return result


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _download_candidate(
    record: dict[str, Any],
    url: str,
    archive: Path,
    staging_archive: Path,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    destination = _artifact_path(archive, record, "pdf")
    # This check is intentionally immediately before the network download.
    # The atomic link below closes the remaining race with other recoverers.
    if destination.exists():
        return {
            "status": "skipped_existing",
            "url": None,
            "error": None,
            "sha256": None,
            "bytes": None,
            "path": _display_path(destination),
            "content_type": None,
        }

    candidate = dict(record)
    candidate["pdf_url"] = url
    candidate["arxiv_id"] = None
    attempt = _download_one(candidate, staging_archive, timeout, retries, False)
    detail = {
        key: attempt.get(key)
        for key in (
            "status",
            "url",
            "error",
            "sha256",
            "bytes",
            "path",
            "content_type",
        )
    }
    if attempt.get("status") != "downloaded":
        return detail

    staged = _artifact_path(staging_archive, record, "pdf")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(staged, destination)
    except FileExistsError:
        detail.update(
            status="skipped_existing",
            url=None,
            error=None,
            sha256=None,
            bytes=None,
            path=_display_path(destination),
            content_type=None,
        )
        return detail
    except OSError as exc:
        detail.update(
            status="promotion_failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        return detail

    detail["path"] = _display_path(destination)
    return detail


def _base_result(record: dict[str, Any], query_url: str | None) -> dict[str, Any]:
    return {
        "org": record["org"],
        "id": record["id"],
        "title": record["title"],
        "doi": record.get("doi"),
        "europe_pmc_query_url": query_url,
        "europe_pmc_matches": [],
        "pmcid": None,
        "has_pdf": None,
        "ncbi_oa_api_url": None,
        "ncbi_oa": None,
        "pmc_aws": None,
        "candidate_urls": [],
        "selected_url": None,
        "attempts": [],
        "attempt": None,
        "status": "no_exact_match",
        "error": None,
    }


def _mark_skipped_existing(result: dict[str, Any], destination: Path) -> dict[str, Any]:
    result.update(
        status="skipped_existing",
        attempt={
            "status": "skipped_existing",
            "path": _display_path(destination),
        },
    )
    return result


def _recover_from_europe_pmc(
    record: dict[str, Any],
    rows: Iterable[dict[str, Any]],
    query_url: str,
    archive: Path,
    staging_archive: Path,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    result = _base_result(record, query_url)
    destination = _artifact_path(archive, record, "pdf")
    if destination.exists():
        return _mark_skipped_existing(result, destination)

    status, selected, exact = _select_europe_pmc_match(record["doi"], rows)
    result["europe_pmc_matches"] = exact
    if status != "matched" or selected is None:
        result["status"] = status
        return result

    pmcid = _normalize_pmcid(selected.get("pmcid"))
    assert pmcid is not None
    result.update(pmcid=pmcid, has_pdf=True)
    if destination.exists():
        return _mark_skipped_existing(result, destination)

    try:
        ncbi_url, ncbi = _fetch_ncbi_oa(pmcid, timeout, retries)
    except RuntimeError as exc:
        result.update(
            status="ncbi_oa_failed",
            ncbi_oa_api_url=_ncbi_oa_url(pmcid),
            error=str(exc),
        )
        return result
    result.update(ncbi_oa_api_url=ncbi_url, ncbi_oa=ncbi)
    urls = ncbi.get("pdf_urls") or []
    result["candidate_urls"] = urls
    if not urls:
        result["status"] = "no_ncbi_pdf"
        return result
    if len(urls) != 1:
        result["status"] = "ambiguous_ncbi_pdf"
        return result
    if destination.exists():
        return _mark_skipped_existing(result, destination)

    attempt = _download_candidate(
        record,
        urls[0],
        archive,
        staging_archive,
        timeout,
        retries,
    )
    result["attempts"].append(attempt)
    result["attempt"] = attempt
    if attempt.get("status") == "downloaded":
        result.update(status="recovered", selected_url=urls[0])
    elif attempt.get("status") == "skipped_existing":
        result["status"] = "skipped_existing"
    else:
        try:
            pmc_aws = _fetch_pmc_aws_pdf(pmcid, timeout, retries)
        except RuntimeError as exc:
            result.update(
                status="candidate_failed",
                error=f"{attempt.get('error')} | PMC AWS lookup: {exc}",
            )
            return result
        result["pmc_aws"] = pmc_aws
        aws_url = pmc_aws.get("pdf_url")
        if not isinstance(aws_url, str):
            result.update(
                status="candidate_failed",
                error=f"{attempt.get('error')} | PMC AWS: no canonical PDF",
            )
            return result

        result["candidate_urls"] = list(dict.fromkeys([*urls, aws_url]))
        if destination.exists():
            return _mark_skipped_existing(result, destination)
        fallback = _download_candidate(
            record,
            aws_url,
            archive,
            staging_archive,
            timeout,
            retries,
        )
        result["attempts"].append(fallback)
        result["attempt"] = fallback
        if fallback.get("status") == "downloaded":
            result.update(status="recovered", selected_url=aws_url, error=None)
        elif fallback.get("status") == "skipped_existing":
            result["status"] = "skipped_existing"
        else:
            result.update(
                status="candidate_failed",
                error=f"{attempt.get('error')} | {fallback.get('error')}",
            )
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
        key=lambda row: (
            str(row.get("org", "")).casefold(),
            str(row.get("id", "")),
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--org", help="exact organization label, case-insensitive")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--refresh", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1 or args.batch_size > 100:
        raise SystemExit("--batch-size must be between 1 and 100")
    if args.delay < 0:
        raise SystemExit("--delay must not be negative")
    if args.timeout < 1:
        raise SystemExit("--timeout must be at least 1")
    if args.retries < 0:
        raise SystemExit("--retries must not be negative")

    previous = _read_jsonl_if_present(args.output)
    completed = {
        (str(row.get("org", "")).casefold(), str(row.get("id", "")))
        for row in previous
        if row.get("status") not in RETRYABLE_STATUSES
    }
    records = [
        record
        for record in load_records(args.inventory)
        if record["type"] == "research_paper"
        and bool(record.get("doi"))
        and (not args.org or record["org"].casefold() == args.org.casefold())
        and not _artifact_path(args.archive, record, "pdf").exists()
        and (args.refresh or (record["org"].casefold(), record["id"]) not in completed)
    ]
    batches = [
        records[index : index + args.batch_size]
        for index in range(0, len(records), args.batch_size)
    ]
    results: list[dict[str, Any]] = []
    processed = 0
    DEFAULT_TMP.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="literature-pmc-", dir=DEFAULT_TMP
    ) as temporary:
        staging_archive = Path(temporary) / "archive"
        for batch_index, batch in enumerate(batches, 1):
            active = []
            for record in batch:
                if _artifact_path(args.archive, record, "pdf").exists():
                    result = _base_result(record, None)
                    destination = _artifact_path(args.archive, record, "pdf")
                    _mark_skipped_existing(result, destination)
                    results.append(result)
                    processed += 1
                    print(
                        f"[{processed}/{len(records)}] {record['org']}:{record['id']} "
                        "skipped_existing",
                        flush=True,
                    )
                else:
                    active.append(record)

            if active:
                try:
                    query_url, rows = _fetch_europe_pmc_many(
                        (record["doi"] for record in active),
                        args.timeout,
                        args.retries,
                    )
                    batch_error = None
                except RuntimeError as exc:
                    query_url = _europe_pmc_query_url(
                        record["doi"] for record in active
                    )
                    rows = []
                    batch_error = str(exc)

                for record in active:
                    if batch_error:
                        result = _base_result(record, query_url)
                        result.update(status="europe_pmc_failed", error=batch_error)
                    else:
                        result = _recover_from_europe_pmc(
                            record,
                            rows,
                            query_url,
                            args.archive,
                            staging_archive,
                            args.timeout,
                            args.retries,
                        )
                    results.append(result)
                    processed += 1
                    print(
                        f"[{processed}/{len(records)}] {result['org']}:{result['id']} "
                        f"{result['status']}",
                        flush=True,
                    )

            _write_jsonl(args.output, _merge_results(previous, results))
            if batch_index < len(batches) and args.delay:
                time.sleep(args.delay)

    combined = _merge_results(previous, results)
    _write_jsonl(args.output, combined)
    counts = Counter(row["status"] for row in results)
    summary = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
    print(
        f"PMC recovery summary: selected={len(records)}"
        + (f", {summary}" if summary else ""),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
