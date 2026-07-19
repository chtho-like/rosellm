import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "literature_page_scan.py"
SPEC = importlib.util.spec_from_file_location("literature_page_scan", MODULE_PATH)
assert SPEC and SPEC.loader
literature_page_scan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(literature_page_scan)


def test_parser_extracts_primary_metadata_and_interesting_links():
    parser = literature_page_scan.PageParser()
    parser.feed(
        """<html><head><title>Fallback title</title>
        <meta property="og:title" content="Primary title">
        <meta property="article:published_time" content="2026-07-19T00:00:00Z">
        </head><body>
        <a href="/assets/report.pdf">Download report</a>
        <a href="https://arxiv.org/abs/2607.12345">Read the paper</a>
        <a href="/about">About us</a>
        </body></html>"""
    )
    assert parser.title == "Primary title"
    assert parser.published_at == "2026-07-19T00:00:00Z"
    links = literature_page_scan._interesting_links(parser, "https://example.com/post")
    assert links == [
        {"url": "https://arxiv.org/abs/2607.12345", "label": "Read the paper"},
        {"url": "https://example.com/assets/report.pdf", "label": "Download report"},
    ]


def test_archive_path_is_stable_and_organization_scoped(tmp_path):
    first = literature_page_scan._archive_path(tmp_path, "Google DeepMind", "https://x/a")
    second = literature_page_scan._archive_path(tmp_path, "Google DeepMind", "https://x/a")
    assert first == second
    assert first.parent.name == "google-deepmind"
    assert first.suffix == ".html"


def test_manifest_path_accepts_relative_and_external_archives(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).parents[1])
    relative = Path("research/literature/archive/pages/example.html")
    assert literature_page_scan._manifest_path(relative) == str(relative)
    external = tmp_path / "example.html"
    assert literature_page_scan._manifest_path(external) == str(external.resolve())


def test_rss_evidence_marks_metadata_only_access():
    candidate = {
        "org": "OpenAI",
        "url": "https://openai.com/index/example/",
        "source_kinds": ["official_sitemap:publication"],
    }
    rss = {
        "url": "https://openai.com/index/example",
        "title": "Example",
        "description": "Summary",
        "published_at": "2026-07-19",
    }
    result = literature_page_scan.rss_evidence(candidate, rss)
    assert result["status"] == "rss_metadata_only"
    assert result["title"] == "Example"
    assert result["archive_path"] is None
    assert literature_page_scan._url_key(candidate["url"]) == literature_page_scan._url_key(
        rss["url"]
    )


def test_inventory_candidates_can_select_only_missing_pdf_records(tmp_path):
    rows = [
        {
            "org": "Lab",
            "primary_url": "https://example.com/html-report",
            "pdf_url": None,
            "arxiv_id": None,
        },
        {
            "org": "Lab",
            "primary_url": "https://arxiv.org/abs/2607.12345",
            "pdf_url": None,
            "arxiv_id": "2607.12345",
        },
    ]
    (tmp_path / "lab.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    candidates = literature_page_scan._candidates_from_inventory(tmp_path, True)
    assert candidates == [
        {
            "org": "Lab",
            "url": "https://example.com/html-report",
            "source_kinds": ["inventory:primary_url"],
        }
    ]
