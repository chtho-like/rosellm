import importlib.util
import sys
from pathlib import Path


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    SPEC = importlib.util.spec_from_file_location(
        "literature_oa_recovery", SCRIPTS / "literature_oa_recovery.py"
    )
    assert SPEC and SPEC.loader
    recovery = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(recovery)
finally:
    sys.path.remove(str(SCRIPTS))


def test_openalex_identity_prefers_explicit_work_page_then_doi():
    explicit = {
        "source_pages": ["https://openalex.org/W123"],
        "doi": "10.1000/example",
    }
    assert recovery._openalex_api_url(explicit).endswith("/W123")
    assert recovery._openalex_api_url(
        {"source_pages": [], "doi": "10.1000/example"}
    ).endswith("https://doi.org/10.1000/example")


def test_location_urls_deduplicates_and_derives_arxiv_pdf():
    work = {
        "best_oa_location": {
            "pdf_url": "https://repo.example/paper.pdf",
            "landing_page_url": "https://arxiv.org/abs/2401.12345v2",
        },
        "primary_location": None,
        "locations": [
            {"pdf_url": "https://repo.example/paper.pdf"},
            {"landing_page_url": "https://arxiv.org/abs/2401.12345"},
        ],
        "open_access": {"oa_url": "https://arxiv.org/abs/2401.12345"},
    }
    assert recovery._location_urls(work) == [
        "https://repo.example/paper.pdf",
        "https://arxiv.org/pdf/2401.12345",
    ]


def test_merge_results_replaces_matching_rows():
    previous = [{"org": "Lab", "id": "a", "status": "old"}]
    current = [{"org": "Lab", "id": "a", "status": "recovered"}]
    assert recovery._merge_results(previous, current) == current
