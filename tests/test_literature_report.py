import importlib.util
import html
import json
import runpy
import sys
from pathlib import Path


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    SPEC = importlib.util.spec_from_file_location(
        "literature_report", SCRIPTS / "literature_report.py"
    )
    assert SPEC and SPEC.loader
    literature_report = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(literature_report)
finally:
    sys.path.remove(str(SCRIPTS))

MATH_CHECKER = runpy.run_path(str(SCRIPTS / "check_docs_math.py"))


def record(**updates):
    value = {
        "id": "example-report",
        "org": "Example Lab",
        "title": "Example Technical Report",
        "authors": ["Example Lab"],
        "date": "2026-07-19",
        "type": "technical_report",
        "tier": "core",
        "arxiv_id": "2607.12345",
        "doi": None,
        "primary_url": "https://example.com/report",
        "pdf_url": "https://arxiv.org/pdf/2607.12345",
        "source_pages": ["https://example.com/research"],
        "affiliation_evidence": "The report names Example Lab.",
        "topics": ["language-models"],
        "notes": None,
        "retrieved_at": "2026-07-19",
        "_location": "fixture:1",
    }
    value.update(updates)
    return value


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_ledger_distinguishes_public_pending_and_html_only(tmp_path):
    archive = tmp_path / "archive"
    candidates = tmp_path / "candidates"
    html_only = record(
        id="html-only",
        title="HTML Report",
        arxiv_id=None,
        pdf_url=None,
        primary_url="https://example.com/html",
    )
    write_jsonl(
        candidates / "inventory-page-evidence.jsonl",
        [
            {
                "org": "Example Lab",
                "url": "https://example.com/html",
                "status": "downloaded",
                "error": None,
            }
        ],
    )
    ledger = literature_report.build_ledger(
        [record(), html_only], archive, candidates, "2026-07-19"
    )
    assert ledger["totals"]["records"] == 2
    assert ledger["totals"]["records_without_public_pdf"] == 1
    assert ledger["totals"]["public_pdfs_not_local"] == 1
    assert ledger["organizations"]["Example Lab"][
        "archived_primary_html_for_no_pdf_records"
    ] == 1
    assert ledger["artifact_gaps"][0]["page_snapshot_status"] == "downloaded"
    assert ledger["totals"]["text_files_with_quality_flags"] == 0


def test_shared_identifier_requires_multiple_organizations():
    rows = [
        record(),
        record(id="second", title="Second", org="Other Lab"),
        record(id="third", title="Third", arxiv_id="2607.54321"),
    ]
    shared = literature_report._shared_identifiers(rows)
    assert len(shared) == 1
    assert shared[0]["identifier"] == "arxiv:2607.12345"
    assert shared[0]["organizations"] == ["Example Lab", "Other Lab"]


def test_source_snapshot_counts_recovery_and_version_relationships(tmp_path):
    candidates = tmp_path / "candidates"
    write_jsonl(
        candidates / "document-equivalences.jsonl",
        [
            {"relation": "published_version_of_preprint"},
            {"relation": "official_page_for_preprint"},
            {"relation": "withdrawn_preprint_without_public_pdf"},
        ],
    )
    write_jsonl(
        candidates / "secondary-recovery.jsonl",
        [{"status": "recovered"}, {"status": "no_match"}],
    )
    snapshot = literature_report._source_snapshot(candidates)
    assert snapshot["document_equivalences"] == {
        "rows": 3,
        "published_preprint_pairs": 1,
        "official_page_preprint_pairs": 1,
        "relations": {
            "official_page_for_preprint": 1,
            "published_version_of_preprint": 1,
            "withdrawn_preprint_without_public_pdf": 1,
        },
    }
    assert snapshot["recovery_ledgers"]["arxiv_title_match"] == {
        "rows": 2,
        "recovered": 1,
        "statuses": {"no_match": 1, "recovered": 1},
    }


def test_rendered_bibliography_keeps_tier_type_and_links():
    rendered = literature_report.render_bibliography([record()], "2026-07-19")
    assert "## Example Lab（1）" in rendered
    assert "[Example Technical Report](https://example.com/report)" in rendered
    assert "`core` / `technical_report`" in rendered
    assert "[PDF](https://arxiv.org/pdf/2607.12345)" in rendered


def test_external_metadata_is_inert_markdown_without_inventory_mutation(tmp_path):
    hostile = record(
        id="hostile-metadata",
        title=r"Q($$\lambda $$), policy (x_i = y^2)\n Appendix | [draft]",
        authors=[r"Rockt\"aschel, Tim", "A $B$"],
        topics=[r"Q($\lambda$)", "x_i"],
        notes=r"See \alpha and $5 (p=0.5) | raw.",
        arxiv_id=None,
        pdf_url=None,
    )
    original = json.loads(json.dumps(hostile))
    ledger = literature_report.build_ledger(
        [hostile], tmp_path / "archive", tmp_path / "candidates", "2026-07-19"
    )
    bibliography = literature_report.render_bibliography([hostile], "2026-07-19")
    coverage = literature_report.render_coverage(ledger)

    assert hostile == original
    assert ledger["artifact_gaps"][0]["title"] == original["title"]
    escaped_title = literature_report._escape(hostile["title"])
    assert html.unescape(escaped_title) == original["title"].replace(r"\n ", " ")
    assert "&#36;&#36;&#92;lambda" in bibliography
    assert r"\n Appendix" not in bibliography

    for name, rendered in (("bibliography", bibliography), ("coverage", coverage)):
        _, errors = MATH_CHECKER["scan_markdown"](
            Path(f"docs/frontier-labs/{name}.md"), rendered
        )
        assert errors == []
