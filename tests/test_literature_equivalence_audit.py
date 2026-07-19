import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    SPEC = importlib.util.spec_from_file_location(
        "literature_equivalence_audit",
        SCRIPTS / "literature_equivalence_audit.py",
    )
    assert SPEC and SPEC.loader
    equivalence_audit = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(equivalence_audit)
finally:
    sys.path.remove(str(SCRIPTS))


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def inventory_record(record_id, arxiv_id=None, doi=None):
    return {
        "id": record_id,
        "org": "Lab",
        "title": record_id.replace("-", " "),
        "authors": ["Researcher"],
        "date": "2026-07-19",
        "type": "research_paper",
        "tier": "direct",
        "arxiv_id": arxiv_id,
        "doi": doi,
        "primary_url": f"https://example.org/{record_id}",
        "pdf_url": None,
        "source_pages": [f"https://example.org/{record_id}"],
        "affiliation_evidence": "Paper says Lab.",
        "topics": ["testing"],
        "notes": None,
        "retrieved_at": "2026-07-19",
    }


def fixture_tree(tmp_path):
    inventory = tmp_path / "inventory"
    candidates = tmp_path / "candidates"
    write_jsonl(
        inventory / "lab.jsonl",
        [
            inventory_record("publication", doi="10.1/example"),
            inventory_record("preprint", arxiv_id="2607.12345"),
        ],
    )
    write_jsonl(candidates / "oa-recovery.jsonl", [])
    write_jsonl(
        candidates / "secondary-recovery.jsonl",
        [
            {
                "org": "Lab",
                "id": "publication",
                "status": "recovered",
                "selected_arxiv_id": "2607.12345",
                "selected_url": "https://arxiv.org/pdf/2607.12345",
            }
        ],
    )
    write_jsonl(candidates / "manual-recovery.jsonl", [])
    write_jsonl(candidates / "pmc-recovery.jsonl", [])
    return inventory, candidates


def test_audit_accepts_explicit_owner_relationship(tmp_path):
    inventory, candidates = fixture_tree(tmp_path)
    write_jsonl(
        candidates / "document-equivalences.jsonl",
        [
            {
                "org": "Lab",
                "publication_id": "publication",
                "preprint_id": "preprint",
                "arxiv_id": "2607.12345",
                "relation": "published_version_of_preprint",
            }
        ],
    )
    assert equivalence_audit.audit(inventory, candidates) == {
        "recovery_owner_conflicts": 1,
        "linked_relationships": 1,
        "byte_identical_pairs": 0,
        "ledger_rows": 1,
    }


def test_audit_rejects_untracked_owner_conflict(tmp_path):
    inventory, candidates = fixture_tree(tmp_path)
    write_jsonl(candidates / "document-equivalences.jsonl", [])
    with pytest.raises(
        equivalence_audit.EquivalenceAuditError,
        match="untracked recovery conflicts",
    ):
        equivalence_audit.audit(inventory, candidates)


def test_audit_requires_and_accepts_identical_sha_relationship(tmp_path):
    inventory, candidates = fixture_tree(tmp_path)
    write_jsonl(
        inventory / "lab.jsonl",
        [
            inventory_record("publication", doi="10.1/example"),
            inventory_record("preprint", arxiv_id="2607.12345"),
            inventory_record("official-asset"),
            inventory_record("official-asset-alias"),
        ],
    )
    archive = candidates.parent / "archive"
    write_jsonl(
        archive / "manifest.jsonl",
        [
            {
                "org": "Lab",
                "id": "official-asset",
                "sha256": "a" * 64,
                "status": "existing",
            },
            {
                "org": "Lab",
                "id": "official-asset-alias",
                "sha256": "a" * 64,
                "status": "existing",
            },
        ],
    )
    owner_relationship = {
        "org": "Lab",
        "publication_id": "publication",
        "preprint_id": "preprint",
        "arxiv_id": "2607.12345",
        "relation": "published_version_of_preprint",
    }
    write_jsonl(
        candidates / "document-equivalences.jsonl", [owner_relationship]
    )
    with pytest.raises(
        equivalence_audit.EquivalenceAuditError,
        match="untracked byte-identical manifest pairs",
    ):
        equivalence_audit.audit(inventory, candidates)

    write_jsonl(
        candidates / "document-equivalences.jsonl",
        [
            owner_relationship,
            {
                "org": "Lab",
                "publication_id": "official-asset",
                "preprint_id": "official-asset-alias",
                "arxiv_id": None,
                "relation": "byte_identical_official_asset_alias",
            },
        ],
    )
    assert equivalence_audit.audit(inventory, candidates) == {
        "recovery_owner_conflicts": 1,
        "linked_relationships": 1,
        "byte_identical_pairs": 1,
        "ledger_rows": 2,
    }
