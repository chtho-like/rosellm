import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "literature_corpus.py"
SPEC = importlib.util.spec_from_file_location("literature_corpus", MODULE_PATH)
assert SPEC and SPEC.loader
literature_corpus = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(literature_corpus)


def record(**updates):
    value = {
        "id": "example-report",
        "org": "Example Lab",
        "title": "Example Technical Report",
        "authors": ["Example Lab"],
        "date": "2026-07-19",
        "type": "technical_report",
        "tier": "core",
        "arxiv_id": "2607.12345v2",
        "doi": "https://doi.org/10.1000/EXAMPLE",
        "primary_url": "https://example.com/report",
        "pdf_url": None,
        "source_pages": ["https://example.com/research"],
        "affiliation_evidence": "The paper names Example Lab.",
        "topics": ["language-models"],
        "notes": None,
        "retrieved_at": "2026-07-19",
    }
    value.update(updates)
    return value


def write_inventory(path: Path, rows):
    path.mkdir(parents=True, exist_ok=True)
    (path / "example.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_normalizes_identifiers_and_derives_arxiv_pdf(tmp_path):
    write_inventory(tmp_path, [record()])
    loaded = literature_corpus.load_records(tmp_path)
    assert loaded[0]["arxiv_id"] == "2607.12345"
    assert loaded[0]["doi"] == "10.1000/example"
    assert literature_corpus._resolved_pdf_url(loaded[0]) == (
        "https://arxiv.org/pdf/2607.12345"
    )


def test_rejects_duplicate_title_within_org(tmp_path):
    write_inventory(
        tmp_path,
        [
            record(arxiv_id="2607.12345"),
            record(
                id="renamed",
                arxiv_id="2607.54321",
                doi="10.1000/alternate",
                title="Example technical-report",
            ),
        ],
    )
    with pytest.raises(literature_corpus.InventoryError, match="duplicate normalized title"):
        literature_corpus.load_records(tmp_path)


def test_allows_duplicate_title_when_equivalence_ledger_links_records(tmp_path):
    inventory = tmp_path / "inventory"
    write_inventory(
        inventory,
        [
            record(arxiv_id="2607.12345"),
            record(
                id="renamed",
                arxiv_id="2607.54321",
                doi="10.1000/alternate",
                title="Example technical-report",
            ),
        ],
    )
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    (candidates / "document-equivalences.jsonl").write_text(
        json.dumps(
            {
                "org": "Example Lab",
                "publication_id": "example-report",
                "preprint_id": "renamed",
                "relation": "published_version_of_preprint",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = literature_corpus.load_records(inventory)

    assert [row["id"] for row in loaded] == ["example-report", "renamed"]


def test_rewrites_github_blob_pdf():
    validated = literature_corpus.validate_record(
        record(
            arxiv_id=None,
            pdf_url="https://github.com/example/lab/blob/main/report.pdf",
        ),
        "fixture:1",
    )
    assert literature_corpus._resolved_pdf_url(validated) == (
        "https://raw.githubusercontent.com/example/lab/main/report.pdf"
    )


def test_explicit_pdf_keeps_arxiv_as_a_distinct_fallback():
    validated = literature_corpus.validate_record(
        record(pdf_url="https://publisher.example/report.pdf"), "fixture:1"
    )
    assert literature_corpus._candidate_pdf_urls(validated) == [
        "https://publisher.example/report.pdf",
        "https://arxiv.org/pdf/2607.12345",
    ]


def test_rewrites_arxiv_abstract_to_pdf():
    validated = literature_corpus.validate_record(
        record(arxiv_id=None, pdf_url="https://arxiv.org/abs/2607.12345v2"),
        "fixture:1",
    )
    assert literature_corpus._resolved_pdf_url(validated) == (
        "https://arxiv.org/pdf/2607.12345v2"
    )


def test_rewrites_legacy_psyarxiv_download_to_osf_primary_file():
    validated = literature_corpus.validate_record(
        record(arxiv_id=None, pdf_url="https://psyarxiv.com/kv86m/download"),
        "fixture:1",
    )
    assert literature_corpus._resolved_pdf_url(validated) == (
        "https://osf.io/download/kv86m"
    )


def test_uses_explicit_pdf_primary_and_source_pages_as_fallbacks():
    validated = literature_corpus.validate_record(
        record(
            arxiv_id=None,
            primary_url="http://proceedings.mlr.press/v37/example.pdf",
            source_pages=["https://openreview.net/pdf?id=example"],
        ),
        "fixture:1",
    )
    assert literature_corpus._candidate_pdf_urls(validated) == [
        "https://proceedings.mlr.press/v37/example.pdf",
        "https://openreview.net/pdf?id=example",
    ]


@pytest.mark.parametrize("bad_date", ["26-07-19", "2026-07", "today", 2026])
def test_rejects_bad_dates(bad_date):
    with pytest.raises(literature_corpus.InventoryError, match="date must"):
        literature_corpus.validate_record(record(date=bad_date), "fixture:1")


def test_lone_surrogate_can_be_made_utf8_safe():
    malformed = "formula: \ud835"
    cleaned = literature_corpus._utf8_safe(malformed)
    assert cleaned == "formula: \ufffd"
    assert cleaned.encode("utf-8")


def test_alternate_text_must_reduce_damage_without_losing_most_content():
    primary = ("substantive text " * 100) + "\ufffd\ufffd"
    assert literature_corpus._prefer_alternate_text(primary, "clean text " * 60)
    assert not literature_corpus._prefer_alternate_text(primary, "tiny")
    assert not literature_corpus._prefer_alternate_text(primary, primary)


def test_manifest_merge_replaces_selected_rows_and_preserves_others():
    previous = [
        {"org": "A", "id": "one", "status": "failed"},
        {"org": "B", "id": "two", "status": "downloaded"},
    ]
    current = [{"org": "A", "id": "one", "status": "downloaded"}]
    assert literature_corpus._merge_manifest_rows(previous, current) == [
        {"org": "A", "id": "one", "status": "downloaded"},
        {"org": "B", "id": "two", "status": "downloaded"},
    ]


def test_manifest_pruning_removes_rows_no_longer_in_inventory():
    rows = [
        {"org": "Example Lab", "id": "example-report", "status": "existing"},
        {"org": "Example Lab", "id": "stale", "status": "existing"},
    ]
    assert literature_corpus._prune_manifest_rows(rows, [record()]) == [rows[0]]


def test_pdfinfo_timeout_is_a_validation_failure(tmp_path, monkeypatch):
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF-1.7\n")
    monkeypatch.setattr(literature_corpus.shutil, "which", lambda _: "/usr/bin/pdfinfo")

    def timeout(*args, **kwargs):
        raise literature_corpus.subprocess.TimeoutExpired(args[0], 60)

    monkeypatch.setattr(literature_corpus.subprocess, "run", timeout)
    valid, error = literature_corpus._pdf_is_valid(path)
    assert not valid
    assert error.startswith("pdfinfo failed:")
