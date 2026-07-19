import importlib.util
import sys
from pathlib import Path


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    SPEC = importlib.util.spec_from_file_location(
        "literature_secondary_recovery",
        SCRIPTS / "literature_secondary_recovery.py",
    )
    assert SPEC and SPEC.loader
    recovery = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(recovery)
finally:
    sys.path.remove(str(SCRIPTS))


def test_parse_arxiv_feed_extracts_identity_title_and_doi():
    payload = b"""<?xml version='1.0'?>
    <feed xmlns='http://www.w3.org/2005/Atom'
          xmlns:arxiv='http://arxiv.org/schemas/atom'>
      <entry>
        <id>http://arxiv.org/abs/1808.00177v5</id>
        <title>Learning\n Dexterous In-Hand Manipulation</title>
        <arxiv:doi>10.1177/0278364919887447</arxiv:doi>
      </entry>
    </feed>"""
    assert recovery._parse_arxiv_feed(payload) == [
        {
            "arxiv_id": "1808.00177",
            "title": "Learning Dexterous In-Hand Manipulation",
            "doi": "10.1177/0278364919887447",
            "abs_url": "https://arxiv.org/abs/1808.00177",
            "pdf_url": "https://arxiv.org/pdf/1808.00177",
        }
    ]


def test_best_match_requires_near_identical_title_or_matching_doi():
    record = {
        "title": "Learning dexterous in-hand manipulation",
        "doi": "10.1177/0278364919887447",
    }
    good = {
        "title": "Learning Dexterous In-Hand Manipulation",
        "doi": None,
    }
    selected, score = recovery._best_match(record, [good], 0.96)
    assert selected == good
    assert score == 1.0

    doi_match = {"title": "Short conference title", "doi": record["doi"]}
    selected, score = recovery._best_match(record, [doi_match], 0.96)
    assert selected == doi_match
    assert score == 1.0

    selected, score = recovery._best_match(
        record, [{"title": "Unrelated manipulation survey", "doi": None}], 0.96
    )
    assert selected is None
    assert score < 0.96


def test_merge_results_replaces_matching_rows():
    previous = [{"org": "Lab", "id": "a", "status": "old"}]
    current = [{"org": "Lab", "id": "a", "status": "recovered"}]
    assert recovery._merge_results(previous, current) == current


def test_batch_query_combines_exact_titles_with_or():
    url = recovery._arxiv_query_url(["First title", "Second title"], 10)
    assert "max_results=10" in url
    decoded = recovery.urllib.parse.unquote_plus(url)
    assert 'ti:"First title" OR ti:"Second title"' in decoded


def test_parser_accepts_exact_record_replay_for_interrupted_batches():
    args = recovery.build_parser().parse_args(
        ["--id", "record-a", "--id", "record-b", "--refresh"]
    )
    assert args.record_ids == ["record-a", "record-b"]
    assert args.refresh is True
