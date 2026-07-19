#!/usr/bin/env python3
"""Audit recovery-linked and byte-identical document relationships.

Inventories intentionally retain separately citable publication, preprint, and
first-party announcement records.  This audit proves that every recovered
arXiv URL already owned by another record in the same organization is named in
``document-equivalences.jsonl`` instead of being silently double-counted.  It
also proves that every pair of same-organization inventory records sharing a
PDF SHA-256 in ``archive/manifest.jsonl`` has an explicit ledger relationship.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import urllib.parse
from collections import defaultdict
from pathlib import Path
from typing import Any

from literature_corpus import (
    DEFAULT_INVENTORY,
    REPO_ROOT,
    _normalize_arxiv,
    _read_jsonl_if_present,
    iter_records,
)


DEFAULT_CANDIDATES = REPO_ROOT / "research" / "literature" / "candidates"
ARXIV_PATH_RE = re.compile(r"/(?:abs|pdf)/(.+?)(?:\.pdf)?/?$", re.I)
LINK_RELATIONS = frozenset(
    {"published_version_of_preprint", "official_page_for_preprint"}
)


class EquivalenceAuditError(ValueError):
    """The equivalence ledger is incomplete or inconsistent."""


def _url_arxiv(url: Any) -> str | None:
    if not isinstance(url, str):
        return None
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc.casefold() not in {
        "arxiv.org",
        "www.arxiv.org",
        "export.arxiv.org",
    }:
        return None
    match = ARXIV_PATH_RE.fullmatch(parsed.path)
    return _normalize_arxiv(match.group(1)) if match else None


def _pair_key(org: str, left: str, right: str) -> tuple[str, str, str]:
    first, second = sorted((left, right))
    return org.casefold(), first, second


def audit(
    inventory: Path, candidates: Path, archive: Path | None = None
) -> dict[str, int]:
    # Equivalent publications may intentionally have the same normalized
    # title, so use schema-validating iteration rather than the corpus-wide
    # duplicate-title gate. Duplicate stable IDs and arXiv owners remain fatal.
    records = list(iter_records(inventory))
    by_id: dict[tuple[str, str], dict[str, Any]] = {}
    owners: dict[tuple[str, str], str] = {}
    for row in records:
        record_key = (row["org"].casefold(), row["id"])
        if record_key in by_id:
            raise EquivalenceAuditError(
                f"duplicate inventory org/id: {row['org']}:{row['id']}"
            )
        by_id[record_key] = row
        if row.get("arxiv_id"):
            owner_key = (row["org"].casefold(), row["arxiv_id"].casefold())
            previous = owners.get(owner_key)
            if previous and previous != row["id"]:
                raise EquivalenceAuditError(
                    f"duplicate inventory arXiv owner: {row['org']} "
                    f"{row['arxiv_id']} ({previous}, {row['id']})"
                )
            owners[owner_key] = row["id"]

    conflicts: set[tuple[str, str, str, str]] = set()
    for name in (
        "oa-recovery.jsonl",
        "secondary-recovery.jsonl",
        "manual-recovery.jsonl",
        "pmc-recovery.jsonl",
    ):
        for row in _read_jsonl_if_present(candidates / name):
            if row.get("status") != "recovered":
                continue
            arxiv_id = _normalize_arxiv(row.get("selected_arxiv_id")) or _url_arxiv(
                row.get("selected_url")
            )
            org = str(row.get("org", ""))
            publication_id = str(row.get("id", ""))
            if not arxiv_id or not org or not publication_id:
                continue
            owner = owners.get((org.casefold(), arxiv_id.casefold()))
            if owner and owner != publication_id:
                conflicts.add((org.casefold(), publication_id, owner, arxiv_id))

    ledger = _read_jsonl_if_present(candidates / "document-equivalences.jsonl")
    errors: list[str] = []
    linked: set[tuple[str, str, str, str]] = set()
    seen_pairs: set[tuple[str, str, str]] = set()
    ledger_pairs: set[tuple[str, str, str]] = set()
    for number, row in enumerate(ledger, 1):
        label = f"document-equivalences.jsonl:{number}"
        org = str(row.get("org", ""))
        publication_id = str(row.get("publication_id", ""))
        preprint_id = row.get("preprint_id")
        relation = str(row.get("relation", ""))
        arxiv_id = _normalize_arxiv(row.get("arxiv_id"))
        if not org or not publication_id or not relation:
            errors.append(f"{label}: missing org/publication_id/relation")
            continue
        if (org.casefold(), publication_id) not in by_id:
            errors.append(f"{label}: publication record does not exist")
        if preprint_id is None:
            pair = (org.casefold(), publication_id, "")
            related_record = None
        elif not isinstance(preprint_id, str) or not preprint_id:
            errors.append(f"{label}: invalid preprint_id")
            pair = (org.casefold(), publication_id, str(preprint_id))
            related_record = None
        else:
            pair = _pair_key(org, publication_id, preprint_id)
            related_record = by_id.get((org.casefold(), preprint_id))
            if related_record is None:
                errors.append(f"{label}: related record does not exist")
            else:
                ledger_pairs.add(pair)
        if pair in seen_pairs:
            errors.append(f"{label}: duplicate relationship")
        seen_pairs.add(pair)
        if relation in LINK_RELATIONS:
            if not isinstance(preprint_id, str) or not preprint_id:
                errors.append(f"{label}: linked relation has no preprint_id")
                continue
            if related_record is None:
                continue
            if not arxiv_id or related_record.get("arxiv_id") != arxiv_id:
                errors.append(f"{label}: arXiv ID does not match preprint owner")
                continue
            linked.add((org.casefold(), publication_id, preprint_id, arxiv_id))

    archive = archive or candidates.parent / "archive"
    identical_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in _read_jsonl_if_present(archive / "manifest.jsonl"):
        org = row.get("org")
        record_id = row.get("id")
        sha256 = row.get("sha256")
        if not all(
            isinstance(value, str) and value for value in (org, record_id, sha256)
        ):
            continue
        identical_groups[(org.casefold(), sha256.casefold())].add(record_id)

    byte_identical_pairs: set[tuple[str, str, str]] = set()
    for (org, _sha256), record_ids in identical_groups.items():
        for left, right in itertools.combinations(sorted(record_ids), 2):
            byte_identical_pairs.add(_pair_key(org, left, right))

    missing = sorted(conflicts - linked)
    extra = sorted(linked - conflicts)
    missing_identical = sorted(byte_identical_pairs - ledger_pairs)
    if missing:
        errors.append("untracked recovery conflicts: " + json.dumps(missing))
    if extra:
        errors.append("linked rows without a recovered owner conflict: " + json.dumps(extra))
    if missing_identical:
        errors.append(
            "untracked byte-identical manifest pairs: "
            + json.dumps(missing_identical)
        )
    if errors:
        raise EquivalenceAuditError("\n".join(errors))
    return {
        "recovery_owner_conflicts": len(conflicts),
        "linked_relationships": len(linked),
        "byte_identical_pairs": len(byte_identical_pairs),
        "ledger_rows": len(ledger),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args(argv)
    try:
        counts = audit(args.inventory, args.candidates, args.archive)
    except (EquivalenceAuditError, OSError, ValueError) as exc:
        print(f"equivalence audit failed:\n{exc}")
        return 1
    print(
        "equivalence audit passed: "
        f"owner_conflicts={counts['recovery_owner_conflicts']} "
        f"linked={counts['linked_relationships']} "
        f"byte_identical_pairs={counts['byte_identical_pairs']} "
        f"ledger_rows={counts['ledger_rows']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
