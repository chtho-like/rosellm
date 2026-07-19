#!/usr/bin/env python3
"""Read-only coverage audit for the frontier-lab literature documentation.

The audit treats the six JSONL inventories and ``coverage.json`` as the source
of truth. It verifies that every inventory record is represented by its exact
ID and primary URL in the mapped lab chapter, that the ID and URL occur together
on at least one Markdown line, and that the summary table in ``index.md`` agrees
with the coverage ledger.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_DIR = Path("research/literature/inventory")
COVERAGE_PATH = Path("research/literature/coverage.json")
DOCS_DIR = Path("docs/frontier-labs")
INDEX_PATH = DOCS_DIR / "index.md"

INVENTORY_NAMES = (
    "anthropic",
    "deepseek",
    "gemini",
    "glm",
    "kimi",
    "openai",
)

# The two organizations share one chapter; every other inventory has one.
DOCUMENT_SPECS = (
    ("anthropic.md", ("anthropic",), True),
    ("deepseek-kimi.md", ("deepseek", "kimi"), True),
    ("gemini.md", ("gemini",), True),
    ("glm.md", ("glm",), False),
    ("openai.md", ("openai",), False),
)

INDEX_ORGANIZATIONS = (
    ("Anthropic", "Anthropic"),
    ("DeepSeek", "DeepSeek"),
    ("Google DeepMind / Gemini", "Google DeepMind"),
    ("Moonshot AI / Kimi", "Moonshot AI / Kimi"),
    ("OpenAI", "OpenAI"),
    ("Zhipu AI / Z.ai", "Zhipu AI / Z.ai"),
)

INDEX_FIELDS = (
    ("records", "records"),
    ("public_pdf_or_arxiv", "public_pdf_or_arxiv"),
    ("local_pdf", "local_pdf"),
    ("extracted_text", "extracted_text"),
)

URL_RE = re.compile(r"https?://[^\s>]+")
BACKTICK_RE = re.compile(r"`([^`\n]+)`")
HTML_CODE_RE = re.compile(r"<code>([^<]+)</code>", re.IGNORECASE)
MARKDOWN_LABEL_RE = re.compile(r"\[([^\]\n]+)\]\(")


class AuditInputError(ValueError):
    """An input file is missing or structurally invalid."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AuditInputError(f"missing input: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AuditInputError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditInputError(f"{path}: expected a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise AuditInputError(f"missing input: {path}") from exc

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for number, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AuditInputError(f"{path}:{number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise AuditInputError(f"{path}:{number}: expected a JSON object")
        record_id = row.get("id")
        primary_url = row.get("primary_url")
        if not isinstance(record_id, str) or not record_id:
            raise AuditInputError(f"{path}:{number}: missing string id")
        if not isinstance(primary_url, str) or not primary_url:
            raise AuditInputError(f"{path}:{number}: missing string primary_url")
        if record_id in seen_ids:
            raise AuditInputError(f"{path}:{number}: duplicate id {record_id!r}")
        seen_ids.add(record_id)
        rows.append(row)
    return rows


def _record_prefixes(record_ids: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({value.split("-", 1)[0] for value in record_ids}))


def _candidate_pattern(prefixes: Iterable[str]) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(prefix) for prefix in prefixes)
    return re.compile(rf"(?<![A-Za-z0-9._-])(?:{alternatives})-[A-Za-z0-9._-]+")


def _code_and_label_tokens(text: str) -> set[str]:
    tokens = set(BACKTICK_RE.findall(text))
    tokens.update(html.unescape(value) for value in HTML_CODE_RE.findall(text))
    tokens.update(MARKDOWN_LABEL_RE.findall(text))
    return {value.strip() for value in tokens}


def _extract_candidate_ids(
    text: str, prefixes: Iterable[str], scan_plain_text: bool
) -> set[str]:
    """Extract record-like IDs without treating arbitrary links as records.

    Organization-scoped namespaces (``anthropic-*``, ``gdm-*`` and the
    DeepSeek/Kimi namespaces) are safe to scan in prose after URLs are removed.
    GLM and OpenAI use generic namespaces such as ``doi-*`` and ``web-*``;
    those are only accepted from inline-code, HTML-code, or link-label fields.
    """

    pattern = _candidate_pattern(prefixes)
    if scan_plain_text:
        without_urls = URL_RE.sub("", text)
        return {match.group(0).rstrip(".-") for match in pattern.finditer(without_urls)}
    return {token for token in _code_and_label_tokens(text) if pattern.fullmatch(token)}


def _audit_document(
    root: Path,
    filename: str,
    inventory_names: tuple[str, ...],
    scan_plain_text: bool,
    inventories: dict[str, list[dict[str, Any]]],
    all_primary_urls: dict[str, set[str]],
) -> dict[str, Any]:
    path = root / DOCS_DIR / filename
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise AuditInputError(f"missing input: {path}") from exc

    records = [record for name in inventory_names for record in inventories[name]]
    expected_ids = {str(record["id"]) for record in records}
    expected_urls = {str(record["primary_url"]) for record in records}
    prefixes = _record_prefixes(expected_ids)
    candidate_ids = _extract_candidate_ids(text, prefixes, scan_plain_text)

    missing_ids = sorted(expected_ids - candidate_ids)
    extra_ids = sorted(candidate_ids - expected_ids)
    missing_url_records = sorted(
        (
            {"id": str(record["id"]), "primary_url": str(record["primary_url"])}
            for record in records
            if str(record["primary_url"]) not in text
        ),
        key=lambda value: value["id"],
    )

    lines = text.splitlines()
    unpaired_records = sorted(
        (
            {"id": str(record["id"]), "primary_url": str(record["primary_url"])}
            for record in records
            if str(record["id"]) in text
            and str(record["primary_url"]) in text
            and not any(
                str(record["id"]) in line and str(record["primary_url"]) in line
                for line in lines
            )
        ),
        key=lambda value: value["id"],
    )

    # Cross-inventory citations are legitimate prose evidence, so report them
    # for visibility but do not classify them as coverage failures.
    cross_inventory_urls = sorted(
        url
        for url, owners in all_primary_urls.items()
        if url not in expected_urls
        and url in text
        and not owners.intersection(inventory_names)
    )

    return {
        "path": str(path.relative_to(root)),
        "inventories": list(inventory_names),
        "expected_records": len(records),
        "found_ids": len(expected_ids.intersection(candidate_ids)),
        "found_primary_urls": len(records) - len(missing_url_records),
        "missing_ids": missing_ids,
        "missing_primary_urls": missing_url_records,
        "unpaired_id_primary_url": unpaired_records,
        "extra_record_ids": extra_ids,
        "cross_inventory_primary_urls": cross_inventory_urls,
    }


def _parse_integer(value: str) -> int:
    cleaned = re.sub(r"[*_`]", "", value).replace(",", "").strip()
    if not cleaned.isdigit():
        raise ValueError(f"not an integer: {value!r}")
    return int(cleaned)


def _clean_label(value: str) -> str:
    return re.sub(r"[*_`]", "", value).strip()


def _parse_index_table(text: str) -> tuple[dict[str, tuple[int, ...]], list[str]]:
    match = re.search(
        r"^## 当前成果\s*$\n(?P<section>.*?)(?=^## |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise AuditInputError("index.md: missing '## 当前成果' section")

    rows: dict[str, tuple[int, ...]] = {}
    duplicates: list[str] = []
    for line in match.group("section").splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 5:
            continue
        label = _clean_label(cells[0])
        if label == "机构" or set(label) <= {"-", ":"}:
            continue
        try:
            values = tuple(_parse_integer(value) for value in cells[1:5])
        except ValueError:
            continue
        if label in rows:
            duplicates.append(label)
        rows[label] = values
    return rows, sorted(set(duplicates))


def _audit_index(root: Path, coverage: dict[str, Any]) -> dict[str, Any]:
    path = root / INDEX_PATH
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise AuditInputError(f"missing input: {path}") from exc
    rows, duplicate_rows = _parse_index_table(text)

    organizations = coverage.get("organizations")
    totals = coverage.get("totals")
    if not isinstance(organizations, dict) or not isinstance(totals, dict):
        raise AuditInputError("coverage.json: missing organizations or totals object")

    expected_labels = {label for label, _ in INDEX_ORGANIZATIONS}
    missing_rows = sorted(expected_labels - set(rows))
    extra_rows = sorted(set(rows) - expected_labels - {"总计"})
    mismatches: list[dict[str, Any]] = []

    for label, coverage_name in INDEX_ORGANIZATIONS:
        if label not in rows:
            continue
        organization = organizations.get(coverage_name)
        if not isinstance(organization, dict):
            mismatches.append(
                {
                    "organization": label,
                    "field": "coverage organization",
                    "expected": "present",
                    "actual": "missing",
                }
            )
            continue
        for index, (display_field, coverage_field) in enumerate(INDEX_FIELDS):
            expected = organization.get(coverage_field)
            actual = rows[label][index]
            if actual != expected:
                mismatches.append(
                    {
                        "organization": label,
                        "field": display_field,
                        "expected": expected,
                        "actual": actual,
                    }
                )

    missing_total_row = "总计" not in rows
    if not missing_total_row:
        for index, (display_field, coverage_field) in enumerate(INDEX_FIELDS):
            expected = totals.get(coverage_field)
            actual = rows["总计"][index]
            if actual != expected:
                mismatches.append(
                    {
                        "organization": "总计",
                        "field": display_field,
                        "expected": expected,
                        "actual": actual,
                    }
                )

    expected_coverage_organizations = {
        coverage_name for _, coverage_name in INDEX_ORGANIZATIONS
    }
    extra_coverage_organizations = sorted(
        set(organizations) - expected_coverage_organizations
    )
    missing_coverage_organizations = sorted(
        expected_coverage_organizations - set(organizations)
    )

    return {
        "path": str(path.relative_to(root)),
        "missing_rows": missing_rows,
        "extra_rows": extra_rows,
        "duplicate_rows": duplicate_rows,
        "missing_total_row": missing_total_row,
        "mismatches": mismatches,
        "missing_coverage_organizations": missing_coverage_organizations,
        "extra_coverage_organizations": extra_coverage_organizations,
    }


def audit_repository(root: Path) -> dict[str, Any]:
    """Return a deterministic audit report without modifying the repository."""

    root = root.resolve()
    inventories = {
        name: _read_jsonl(root / INVENTORY_DIR / f"{name}.jsonl")
        for name in INVENTORY_NAMES
    }
    coverage = _read_json(root / COVERAGE_PATH)

    all_primary_urls: dict[str, set[str]] = {}
    for name, records in inventories.items():
        for record in records:
            all_primary_urls.setdefault(str(record["primary_url"]), set()).add(name)

    documents = [
        _audit_document(
            root,
            filename,
            inventory_names,
            scan_plain_text,
            inventories,
            all_primary_urls,
        )
        for filename, inventory_names, scan_plain_text in DOCUMENT_SPECS
    ]
    return {
        "repository": str(root),
        "inventory_records": sum(len(rows) for rows in inventories.values()),
        "documents": documents,
        "index": _audit_index(root, coverage),
    }


def report_has_failures(report: dict[str, Any]) -> bool:
    for document in report["documents"]:
        if any(
            document[key]
            for key in (
                "missing_ids",
                "missing_primary_urls",
                "unpaired_id_primary_url",
                "extra_record_ids",
            )
        ):
            return True
    index = report["index"]
    return bool(
        index["missing_rows"]
        or index["extra_rows"]
        or index["duplicate_rows"]
        or index["missing_total_row"]
        or index["mismatches"]
        or index["missing_coverage_organizations"]
        or index["extra_coverage_organizations"]
    )


def _preview(values: list[Any], limit: int) -> str:
    rendered = [
        value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        for value in values[:limit]
    ]
    if len(values) > limit:
        rendered.append(f"... {len(values) - limit} more")
    return ", ".join(rendered)


def format_report(report: dict[str, Any], max_items: int = 20) -> str:
    lines = [
        "Literature documentation coverage audit",
        f"inventory records: {report['inventory_records']}",
    ]
    for document in report["documents"]:
        lines.append(
            "- {path}: records={expected_records}, ids={found_ids}/{expected_records}, "
            "primary_urls={found_primary_urls}/{expected_records}, extra_ids={extra}, "
            "cross_inventory_urls={cross}".format(
                **document,
                extra=len(document["extra_record_ids"]),
                cross=len(document["cross_inventory_primary_urls"]),
            )
        )
        for key, label in (
            ("missing_ids", "missing IDs"),
            ("missing_primary_urls", "missing primary URLs"),
            ("unpaired_id_primary_url", "unpaired ID/primary URL"),
            ("extra_record_ids", "extra IDs"),
        ):
            if document[key]:
                lines.append(f"  {label}: {_preview(document[key], max_items)}")

    index = report["index"]
    index_issue_count = sum(
        len(index[key])
        for key in (
            "missing_rows",
            "extra_rows",
            "duplicate_rows",
            "mismatches",
            "missing_coverage_organizations",
            "extra_coverage_organizations",
        )
    ) + int(index["missing_total_row"])
    lines.append(f"- {index['path']}: issues={index_issue_count}")
    for key, label in (
        ("missing_rows", "missing rows"),
        ("extra_rows", "extra rows"),
        ("duplicate_rows", "duplicate rows"),
        ("mismatches", "mismatches"),
        ("missing_coverage_organizations", "missing coverage organizations"),
        ("extra_coverage_organizations", "extra coverage organizations"),
    ):
        if index[key]:
            lines.append(f"  {label}: {_preview(index[key], max_items)}")
    if index["missing_total_row"]:
        lines.append("  missing total row: 总计")

    lines.append("FAILED" if report_has_failures(report) else "PASSED")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--json", action="store_true", help="emit the full JSON report")
    parser.add_argument(
        "--max-items",
        type=int,
        default=20,
        help="maximum failing items shown per category in text output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit_repository(args.repo_root)
    except AuditInputError as exc:
        print(
            f"Literature documentation coverage audit input error: {exc}",
            file=sys.stderr,
        )
        return 2
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(format_report(report, max_items=max(1, args.max_items)))
    return 1 if report_has_failures(report) else 0


if __name__ == "__main__":
    raise SystemExit(main())
