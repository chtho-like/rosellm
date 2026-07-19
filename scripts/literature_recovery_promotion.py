#!/usr/bin/env python3
"""Promote fully verified recovery results into literature inventories.

The recovery ledgers are evidence, not authority.  This tool only proposes a
promotion when a ``status=recovered`` row still matches the corresponding
inventory identity and its local artifact passes PDF magic, mandatory
``pdfinfo``, byte-count, and SHA-256 checks.  It is dry-run by default; only an
explicit ``--apply`` writes inventory files after every candidate and the full
proposed inventory have validated successfully.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any, Iterable

from literature_corpus import (
    ARXIV_RE,
    DEFAULT_ARCHIVE,
    DEFAULT_INVENTORY,
    InventoryError,
    REPO_ROOT,
    _artifact_path,
    _normalize_arxiv,
    _normalize_doi,
    _resolved_pdf_url,
    load_records,
)


DEFAULT_OA_RECOVERY = (
    REPO_ROOT / "research" / "literature" / "candidates" / "oa-recovery.jsonl"
)
DEFAULT_SECONDARY_RECOVERY = (
    REPO_ROOT
    / "research"
    / "literature"
    / "candidates"
    / "secondary-recovery.jsonl"
)
DEFAULT_MANUAL_RECOVERY = (
    REPO_ROOT
    / "research"
    / "literature"
    / "candidates"
    / "manual-recovery.jsonl"
)
DEFAULT_PMC_RECOVERY = (
    REPO_ROOT
    / "research"
    / "literature"
    / "candidates"
    / "pmc-recovery.jsonl"
)
ALLOWED_FIELDS = frozenset({"pdf_url", "arxiv_id", "source_pages"})
SUCCESS_STATUSES = frozenset({"downloaded", "existing"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.I)
PAGES_RE = re.compile(r"^Pages:\s+(\d+)\s*$", re.M)


class PromotionError(ValueError):
    """A recovery result cannot be safely promoted."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PromotionError(f"recovery ledger does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise PromotionError(f"{path}:{number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise PromotionError(f"{path}:{number}: expected a JSON object")
            row = dict(row)
            row["_promotion_location"] = f"{path}:{number}"
            rows.append(row)
    return rows


def _validate_url(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PromotionError(f"{location}: selected_url must be a non-empty URL")
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise PromotionError(f"{location}: invalid selected_url {value!r}")
    return value


def _arxiv_from_url(url: str) -> str | None:
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc.casefold() not in {
        "arxiv.org",
        "www.arxiv.org",
        "export.arxiv.org",
    }:
        return None
    match = re.fullmatch(r"/(?:abs|pdf)/(.+?)(?:\.pdf)?/?", parsed.path, re.I)
    if not match:
        return None
    value = _normalize_arxiv(match.group(1))
    return value if value and ARXIV_RE.fullmatch(value) else None


def _selected_attempt(row: dict[str, Any], kind: str) -> dict[str, Any]:
    location = row["_promotion_location"]
    selected_url = row["selected_url"]
    if kind == "oa":
        attempts = row.get("attempts")
        if not isinstance(attempts, list):
            raise PromotionError(f"{location}: attempts must be a list")
        matches = [
            attempt
            for attempt in attempts
            if isinstance(attempt, dict) and attempt.get("url") == selected_url
        ]
        if len(matches) != 1:
            raise PromotionError(
                f"{location}: selected_url must match exactly one OA attempt"
            )
        attempt = matches[0]
    elif kind in {"secondary", "pmc"}:
        attempt = row.get("attempt")
        if not isinstance(attempt, dict) or attempt.get("url") != selected_url:
            raise PromotionError(
                f"{location}: selected_url does not match the {kind} attempt"
            )
    else:
        validation = row.get("validation")
        if not isinstance(validation, dict) or not all(
            validation.get(field) is True for field in ("pdf_magic", "pdfinfo")
        ):
            raise PromotionError(
                f"{location}: manual recovery lacks explicit PDF validation"
            )
        if row.get("title_match") is not True:
            raise PromotionError(
                f"{location}: manual recovery lacks an explicit title match"
            )
        attempt = {
            "url": selected_url,
            "status": "downloaded",
            "sha256": row.get("sha256"),
            "bytes": row.get("bytes"),
            "path": row.get("path"),
        }

    if attempt.get("status") not in SUCCESS_STATUSES:
        raise PromotionError(f"{location}: selected attempt is not successful")
    sha256 = attempt.get("sha256")
    if not isinstance(sha256, str) or not SHA256_RE.fullmatch(sha256):
        raise PromotionError(f"{location}: selected attempt has no valid SHA-256")
    byte_count = attempt.get("bytes")
    if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 5:
        raise PromotionError(f"{location}: selected attempt has no valid byte count")
    return attempt


def _candidate(row: dict[str, Any], kind: str) -> dict[str, Any]:
    location = row["_promotion_location"]
    required_identity = ("org", "id") if kind == "manual" else ("org", "id", "title")
    for field in required_identity:
        if not isinstance(row.get(field), str) or not row[field]:
            raise PromotionError(f"{location}: missing recovery identity field {field}")
    selected_url = _validate_url(row.get("selected_url"), location)
    row = dict(row)
    row["selected_url"] = selected_url
    attempt = _selected_attempt(row, kind)

    url_arxiv = _arxiv_from_url(selected_url)
    raw_selected_arxiv = row.get("selected_arxiv_id")
    if raw_selected_arxiv is not None and not isinstance(raw_selected_arxiv, str):
        raise PromotionError(f"{location}: invalid selected_arxiv_id")
    selected_arxiv = _normalize_arxiv(raw_selected_arxiv)
    if selected_arxiv and not ARXIV_RE.fullmatch(selected_arxiv):
        raise PromotionError(f"{location}: invalid selected_arxiv_id")
    if selected_arxiv and selected_arxiv != url_arxiv:
        raise PromotionError(
            f"{location}: selected_arxiv_id does not match selected_url"
        )
    expected_pages = row.get("pages") if kind == "manual" else None
    if expected_pages is not None and (
        not isinstance(expected_pages, int)
        or isinstance(expected_pages, bool)
        or expected_pages < 1
    ):
        raise PromotionError(f"{location}: invalid manual recovery page count")
    return {
        "org": row["org"],
        "id": row["id"],
        "title": row.get("title"),
        "title_present": "title" in row,
        "doi": row.get("doi"),
        "doi_present": "doi" in row,
        "snapshot_arxiv_id": row.get("arxiv_id") if kind == "oa" else None,
        "snapshot_current_url": row.get("current_url") if kind in {"oa", "manual"} else None,
        "selected_url": selected_url,
        "selected_arxiv_id": selected_arxiv or url_arxiv,
        "sha256": attempt["sha256"].casefold(),
        "bytes": attempt["bytes"],
        "pages": expected_pages,
        "attempt_path": attempt.get("path"),
        "origins": [location],
    }


def _load_candidates(
    oa_recovery: Path,
    secondary_recovery: Path,
    manual_recovery: Path,
    pmc_recovery: Path,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    errors: list[str] = []
    for path, kind in (
        (oa_recovery, "oa"),
        (secondary_recovery, "secondary"),
        (manual_recovery, "manual"),
        (pmc_recovery, "pmc"),
    ):
        for row in _read_jsonl(path):
            if row.get("status") != "recovered":
                continue
            try:
                candidates.append(_candidate(row, kind))
            except PromotionError as exc:
                errors.append(str(exc))
    if errors:
        raise PromotionError("invalid recovered rows:\n- " + "\n- ".join(errors))

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        key = (candidate["org"].casefold(), candidate["id"])
        previous = merged.get(key)
        if previous is None:
            merged[key] = candidate
            continue
        comparable = ("selected_url", "selected_arxiv_id", "sha256", "bytes", "pages")
        if any(previous[field] != candidate[field] for field in comparable):
            raise PromotionError(
                f"conflicting recovered rows for {candidate['org']}:{candidate['id']}: "
                + ", ".join(previous["origins"] + candidate["origins"])
            )
        previous["origins"].extend(candidate["origins"])
    return sorted(
        merged.values(),
        key=lambda row: (row["org"].casefold(), row["id"]),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_fingerprints(paths: Iterable[Path]) -> dict[Path, str]:
    fingerprints: dict[Path, str] = {}
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise PromotionError(f"validated input does not exist: {path}")
        fingerprints[resolved] = _sha256(resolved)
    return fingerprints


def _inputs_unchanged(fingerprints: dict[Path, str]) -> None:
    changed = [
        str(path)
        for path, expected in fingerprints.items()
        if not path.is_file() or _sha256(path) != expected
    ]
    if changed:
        raise PromotionError(
            "validated inputs changed while the plan was being built: "
            + ", ".join(changed)
        )


def _validate_artifact(
    candidate: dict[str, Any], record: dict[str, Any], archive: Path, pdfinfo: str
) -> None:
    label = f"{candidate['org']}:{candidate['id']}"
    path = _artifact_path(archive, record, "pdf").resolve()
    attempt_path = candidate.get("attempt_path")
    if attempt_path:
        recorded = Path(attempt_path)
        if not recorded.is_absolute():
            recorded = REPO_ROOT / recorded
        if recorded.resolve() != path:
            raise PromotionError(
                f"{label}: recovery attempt path does not match canonical artifact path"
            )
    if not path.is_file():
        raise PromotionError(f"{label}: local artifact is missing: {path}")
    try:
        with path.open("rb") as handle:
            magic = handle.read(5)
    except OSError as exc:
        raise PromotionError(f"{label}: cannot read local artifact: {exc}") from exc
    if magic != b"%PDF-":
        raise PromotionError(f"{label}: local artifact is missing PDF magic")
    actual_bytes = path.stat().st_size
    if actual_bytes != candidate["bytes"]:
        raise PromotionError(
            f"{label}: local byte count {actual_bytes} != recovery {candidate['bytes']}"
        )
    actual_sha256 = _sha256(path)
    if actual_sha256 != candidate["sha256"]:
        raise PromotionError(
            f"{label}: local SHA-256 does not match the recovered attempt"
        )
    try:
        checked = subprocess.run(
            [pdfinfo, str(path)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PromotionError(f"{label}: pdfinfo failed: {exc}") from exc
    match = PAGES_RE.search(checked.stdout or "")
    if checked.returncode != 0 or not match or int(match.group(1)) < 1:
        detail = (checked.stderr or checked.stdout or "").strip().splitlines()
        reason = detail[-1] if detail else "pdfinfo did not report a positive page count"
        raise PromotionError(f"{label}: pdfinfo rejected artifact: {reason}")
    actual_pages = int(match.group(1))
    if candidate.get("pages") is not None and actual_pages != candidate["pages"]:
        raise PromotionError(
            f"{label}: pdfinfo page count {actual_pages} != recovery {candidate['pages']}"
        )


def _read_inventory_rows(
    inventory: Path,
) -> tuple[
    dict[Path, dict[str, Any]],
    dict[tuple[str, str], tuple[Path, int, dict[str, Any]]],
]:
    if not inventory.is_dir():
        raise PromotionError(f"inventory directory does not exist: {inventory}")
    files: dict[Path, dict[str, Any]] = {}
    indexed: dict[tuple[str, str], tuple[Path, int, dict[str, Any]]] = {}
    for path in sorted(inventory.glob("*.jsonl")):
        original = path.read_bytes()
        rows: list[dict[str, Any]] = []
        for number, raw in enumerate(original.decode("utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise PromotionError(f"{path}:{number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise PromotionError(f"{path}:{number}: expected a JSON object")
            rows.append(row)
            key = (str(row.get("org", "")).casefold(), str(row.get("id", "")))
            indexed[key] = (path, len(rows) - 1, row)
        files[path] = {"original": original, "rows": rows}
    return files, indexed


def _serialize(rows: Iterable[dict[str, Any]]) -> bytes:
    return (
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        )
    ).encode("utf-8")


def _validate_proposed_inventory(
    files: dict[Path, dict[str, Any]], equivalence_path: Path
) -> None:
    with tempfile.TemporaryDirectory(prefix="rosellm-promotion-") as temporary:
        root = Path(temporary)
        for source, state in files.items():
            (root / source.name).write_bytes(_serialize(state["rows"]))
        try:
            load_records(root, equivalence_path=equivalence_path)
        except InventoryError as exc:
            raise PromotionError(f"proposed inventory is invalid:\n{exc}") from exc


def _immutable_fields_unchanged(
    before: dict[str, Any], after: dict[str, Any], label: str
) -> None:
    fields = (set(before) | set(after)) - ALLOWED_FIELDS
    changed = sorted(field for field in fields if before.get(field) != after.get(field))
    if changed:
        raise PromotionError(
            f"{label}: attempted to change immutable fields: {', '.join(changed)}"
        )


def build_plan(
    inventory: Path,
    archive: Path,
    oa_recovery: Path,
    secondary_recovery: Path,
    manual_recovery: Path,
    pmc_recovery: Path,
) -> tuple[list[dict[str, Any]], dict[Path, bytes]]:
    try:
        records = load_records(inventory)
    except InventoryError as exc:
        raise PromotionError(f"current inventory is invalid:\n{exc}") from exc
    record_index = {
        (record["org"].casefold(), record["id"]): record for record in records
    }
    files, raw_index = _read_inventory_rows(inventory)
    candidates = _load_candidates(
        oa_recovery,
        secondary_recovery,
        manual_recovery,
        pmc_recovery,
    )
    pdfinfo = shutil.which("pdfinfo")
    if candidates and not pdfinfo:
        raise PromotionError("pdfinfo is required for recovery promotion")

    errors: list[str] = []
    arxiv_owners = {
        (record["org"].casefold(), record["arxiv_id"].casefold()): record["id"]
        for record in records
        if record.get("arxiv_id")
    }
    validated: list[tuple[dict[str, Any], dict[str, Any], str | None]] = []
    for candidate in candidates:
        key = (candidate["org"].casefold(), candidate["id"])
        record = record_index.get(key)
        if record is None:
            errors.append(
                f"{candidate['origins'][0]}: inventory record does not exist"
            )
            continue
        label = f"{candidate['org']}:{candidate['id']}"
        try:
            if candidate["org"] != record["org"]:
                raise PromotionError(f"{label}: organization label changed")
            if candidate["id"] != record["id"]:
                raise PromotionError(f"{label}: record ID changed")
            if candidate["title_present"] and candidate["title"] != record["title"]:
                raise PromotionError(f"{label}: title does not match inventory")
            if candidate["doi_present"] and _normalize_doi(
                candidate.get("doi")
            ) != record.get("doi"):
                raise PromotionError(f"{label}: DOI identity does not match inventory")
            snapshot_arxiv = _normalize_arxiv(candidate.get("snapshot_arxiv_id"))
            if snapshot_arxiv and record.get("arxiv_id") not in {
                snapshot_arxiv,
                candidate.get("selected_arxiv_id"),
            }:
                raise PromotionError(
                    f"{label}: arXiv identity no longer matches recovery evidence"
                )
            desired_arxiv = candidate.get("selected_arxiv_id")
            existing_arxiv_owner = (
                arxiv_owners.get((record["org"].casefold(), desired_arxiv.casefold()))
                if desired_arxiv
                else None
            )
            if existing_arxiv_owner == record["id"]:
                existing_arxiv_owner = None
            if (
                record.get("arxiv_id")
                and desired_arxiv
                and record["arxiv_id"] != desired_arxiv
                and existing_arxiv_owner is None
            ):
                raise PromotionError(
                    f"{label}: refusing to replace an existing arXiv identity"
                )
            snapshot_url = candidate.get("snapshot_current_url")
            current_url = _resolved_pdf_url(record)
            if snapshot_url is not None and current_url not in {
                snapshot_url,
                candidate["selected_url"],
            }:
                raise PromotionError(
                    f"{label}: current PDF route changed after recovery"
                )
            _validate_artifact(candidate, record, archive, pdfinfo or "")
            validated.append((candidate, record, existing_arxiv_owner))
        except PromotionError as exc:
            errors.append(str(exc))
    if errors:
        raise PromotionError("promotion validation failed:\n- " + "\n- ".join(errors))

    actions: list[dict[str, Any]] = []
    for candidate, record, existing_arxiv_owner in validated:
        key = (record["org"].casefold(), record["id"])
        path, index, raw = raw_index[key]
        proposed = copy.deepcopy(raw)
        proposed["pdf_url"] = candidate["selected_url"]
        if candidate.get("selected_arxiv_id") and existing_arxiv_owner is None:
            proposed["arxiv_id"] = candidate["selected_arxiv_id"]
        if candidate["selected_url"] not in proposed["source_pages"]:
            proposed["source_pages"].append(candidate["selected_url"])
        _immutable_fields_unchanged(raw, proposed, f"{record['org']}:{record['id']}")
        changed_fields = [
            field for field in sorted(ALLOWED_FIELDS) if raw.get(field) != proposed.get(field)
        ]
        files[path]["rows"][index] = proposed
        raw_index[key] = (path, index, proposed)
        actions.append(
            {
                "org": record["org"],
                "id": record["id"],
                "selected_url": candidate["selected_url"],
                "changed_fields": changed_fields,
                "existing_arxiv_owner": existing_arxiv_owner,
                "origins": candidate["origins"],
            }
        )

    _validate_proposed_inventory(
        files, manual_recovery.parent / "document-equivalences.jsonl"
    )
    changed_files = {
        path: payload
        for path, state in files.items()
        if (payload := _serialize(state["rows"])) != state["original"]
    }
    return actions, changed_files


def _write_files(changed_files: dict[Path, bytes]) -> None:
    staged: dict[Path, Path] = {}
    originals = {path: path.read_bytes() for path in changed_files}
    committed: list[Path] = []
    try:
        for path, payload in changed_files.items():
            descriptor, name = tempfile.mkstemp(
                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
            )
            temporary = Path(name)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            staged[path] = temporary
        for path in sorted(changed_files):
            os.replace(staged[path], path)
            committed.append(path)
    except OSError as exc:
        rollback_errors: list[str] = []
        for path in reversed(committed):
            try:
                descriptor, name = tempfile.mkstemp(
                    dir=path.parent, prefix=f".{path.name}.rollback.", suffix=".tmp"
                )
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(originals[path])
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(name, path)
            except OSError as rollback_exc:
                rollback_errors.append(f"{path}: {rollback_exc}")
        detail = f"inventory write failed: {exc}"
        if rollback_errors:
            detail += "; rollback failed: " + " | ".join(rollback_errors)
        raise PromotionError(detail) from exc
    finally:
        for temporary in staged.values():
            if temporary.exists():
                temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--oa-recovery", type=Path, default=DEFAULT_OA_RECOVERY)
    parser.add_argument(
        "--secondary-recovery", type=Path, default=DEFAULT_SECONDARY_RECOVERY
    )
    parser.add_argument("--manual-recovery", type=Path, default=DEFAULT_MANUAL_RECOVERY)
    parser.add_argument("--pmc-recovery", type=Path, default=DEFAULT_PMC_RECOVERY)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the fully validated plan; default is dry-run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        watched_inputs = [
            args.oa_recovery,
            args.secondary_recovery,
            args.manual_recovery,
            args.pmc_recovery,
            *sorted(args.inventory.glob("*.jsonl")),
        ]
        fingerprints = _input_fingerprints(watched_inputs)
        actions, changed_files = build_plan(
            args.inventory,
            args.archive,
            args.oa_recovery,
            args.secondary_recovery,
            args.manual_recovery,
            args.pmc_recovery,
        )
        changed = sum(bool(action["changed_fields"]) for action in actions)
        unchanged = len(actions) - changed
        mode = "APPLY" if args.apply else "DRY-RUN"
        print(
            f"recovery promotion {mode}: validated={len(actions)} "
            f"changed_records={changed} unchanged={unchanged}"
        )
        for action in actions:
            state = ",".join(action["changed_fields"]) or "unchanged"
            owner = (
                f" existing_arxiv_owner={action['existing_arxiv_owner']}"
                if action["existing_arxiv_owner"]
                else ""
            )
            print(
                f"{action['org']}:{action['id']} {state}{owner} "
                f"<- {action['selected_url']}"
            )
        if args.apply:
            _inputs_unchanged(fingerprints)
            _write_files(changed_files)
            print(f"applied {len(changed_files)} inventory file(s)")
        else:
            print("dry-run only; no inventory files were written (use --apply to commit)")
        return 0
    except (PromotionError, OSError) as exc:
        print(f"promotion error:\n{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
