import importlib.util
import json
import sys
import urllib.parse
from pathlib import Path


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    SPEC = importlib.util.spec_from_file_location(
        "literature_pmc_recovery",
        SCRIPTS / "literature_pmc_recovery.py",
    )
    assert SPEC and SPEC.loader
    recovery = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(recovery)
finally:
    sys.path.remove(str(SCRIPTS))


def _record():
    return {
        "org": "Lab",
        "id": "doi-example",
        "title": "Example paper",
        "tier": "affiliated",
        "doi": "10.1000/Example",
        "arxiv_id": None,
        "pdf_url": None,
        "_location": "inventory.jsonl:1",
    }


def test_europe_pmc_query_batches_exact_dois():
    url = recovery._europe_pmc_query_url(
        ["10.1000/Example", "https://doi.org/10.2000/SECOND"]
    )
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    assert query["query"] == ['DOI:"10.1000/example" OR DOI:"10.2000/second"']
    assert query["format"] == ["json"]
    assert query["resultType"] == ["lite"]
    assert query["pageSize"] == ["1000"]


def test_select_europe_pmc_requires_exact_doi_pmcid_and_has_pdf():
    mismatched = {
        "doi": "10.9999/other",
        "pmcid": "PMC100",
        "hasPDF": "Y",
    }
    missing_pdf = {
        "doi": "10.1000/example",
        "pmcid": "PMC101",
        "hasPDF": "N",
    }
    status, selected, exact = recovery._select_europe_pmc_match(
        "10.1000/example", [mismatched, missing_pdf]
    )
    assert status == "no_confirmed_pdf"
    assert selected is None
    assert exact == [missing_pdf]

    confirmed = {
        "doi": "10.1000/EXAMPLE",
        "pmcid": "pmc101",
        "hasPDF": "Y",
        "isOpenAccess": "Y",
        "source": "MED",
        "id": "123",
    }
    status, selected, _ = recovery._select_europe_pmc_match(
        "10.1000/example", [mismatched, confirmed]
    )
    assert status == "matched"
    assert selected == confirmed

    second_pmcid = dict(confirmed, pmcid="PMC102", id="124")
    status, selected, _ = recovery._select_europe_pmc_match(
        "10.1000/example", [confirmed, second_pmcid]
    )
    assert status == "ambiguous_pmcid"
    assert selected is None


def test_fetch_europe_pmc_rejects_truncated_result(monkeypatch):
    payload = json.dumps(
        {"hitCount": 2, "resultList": {"result": [{"doi": "10.1/a"}]}}
    ).encode()
    monkeypatch.setattr(recovery, "_fetch_bytes", lambda *args: payload)
    try:
        recovery._fetch_europe_pmc_many(["10.1/a"], 1, 0)
    except RuntimeError as exc:
        assert "truncated" in str(exc)
    else:
        raise AssertionError("truncated Europe PMC response was accepted")


def test_parse_ncbi_oa_uses_exact_record_and_official_pdf_only():
    payload = b"""<OA><records>
      <record id="PMC999">
        <link format="pdf"
          href="ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/00/00/wrong.pdf" />
      </record>
      <record id="PMC101" citation="Example citation" license="CC BY"
              retracted="no">
        <link format="tgz"
          href="ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/aa/bb/a.tgz" />
        <link format="pdf"
          href="ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/aa/bb/paper.PMC101.pdf" />
        <link format="pdf"
          href="https://files.example.invalid/pub/pmc/oa_pdf/stolen.pdf" />
      </record>
    </records></OA>"""
    assert recovery._parse_ncbi_oa_response(payload, "pmc101") == {
        "record_id": "PMC101",
        "citation": "Example citation",
        "license": "CC BY",
        "retracted": "no",
        "pdf_urls": [
            "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/aa/bb/paper.PMC101.pdf"
        ],
    }


def test_parse_pmc_aws_versions_requires_exact_pmcid_and_numeric_versions():
    payload = b"""<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
      <Name>pmc-oa-opendata</Name><IsTruncated>false</IsTruncated>
      <CommonPrefixes><Prefix>PMC101.2/</Prefix></CommonPrefixes>
      <CommonPrefixes><Prefix>PMC101.1/</Prefix></CommonPrefixes>
      <CommonPrefixes><Prefix>PMC1010.3/</Prefix></CommonPrefixes>
      <CommonPrefixes><Prefix>PMC101.latest/</Prefix></CommonPrefixes>
    </ListBucketResult>"""
    assert recovery._parse_pmc_aws_versions(payload, "pmc101") == [
        "PMC101.1",
        "PMC101.2",
    ]


def test_parse_pmc_aws_pdf_requires_one_canonical_versioned_object():
    payload = b"""<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
      <Name>pmc-oa-opendata</Name><IsTruncated>false</IsTruncated>
      <Contents><Key>PMC101.2/PMC101.2.xml</Key></Contents>
      <Contents><Key>PMC101.2/PMC101.2.pdf</Key></Contents>
      <Contents><Key>PMC101.2/publisher.pdf</Key></Contents>
    </ListBucketResult>"""
    assert recovery._parse_pmc_aws_pdf_url(payload, "PMC101", "PMC101.2") == (
        "https://pmc-oa-opendata.s3.amazonaws.com/PMC101.2/PMC101.2.pdf"
    )
    assert recovery._parse_pmc_aws_pdf_url(payload, "PMC101", "PMC101.1") is None


def test_parse_pmc_aws_listing_rejects_truncated_or_wrong_bucket():
    truncated = b"""<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
      <Name>pmc-oa-opendata</Name><IsTruncated>true</IsTruncated>
    </ListBucketResult>"""
    wrong = b"""<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
      <Name>lookalike</Name><IsTruncated>false</IsTruncated>
    </ListBucketResult>"""
    for payload in (truncated, wrong):
        try:
            recovery._parse_pmc_aws_listing(payload)
        except RuntimeError:
            pass
        else:
            raise AssertionError("unsafe PMC AWS listing was accepted")


def test_download_candidate_skips_existing_without_network(tmp_path, monkeypatch):
    record = _record()
    archive = tmp_path / "archive"
    staging = tmp_path / "staging"
    destination = recovery._artifact_path(archive, record, "pdf")
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"existing")

    def fail_download(*args, **kwargs):
        raise AssertionError("download was attempted despite canonical artifact")

    monkeypatch.setattr(recovery, "_download_one", fail_download)
    attempt = recovery._download_candidate(
        record,
        "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/a/b/paper.pdf",
        archive,
        staging,
        1,
        0,
    )
    assert attempt["status"] == "skipped_existing"
    assert destination.read_bytes() == b"existing"


def test_download_candidate_stages_without_force_and_atomically_promotes(
    tmp_path, monkeypatch
):
    record = _record()
    archive = tmp_path / "archive"
    staging = tmp_path / "staging"
    calls = []

    def fake_download(candidate, stage, timeout, retries, force):
        calls.append((candidate["pdf_url"], stage, force))
        staged = recovery._artifact_path(stage, candidate, "pdf")
        staged.parent.mkdir(parents=True)
        staged.write_bytes(b"%PDF-staged")
        return {
            "status": "downloaded",
            "url": candidate["pdf_url"],
            "error": None,
            "sha256": "abc",
            "bytes": 11,
            "path": str(staged),
            "content_type": "application/pdf",
        }

    monkeypatch.setattr(recovery, "_download_one", fake_download)
    url = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/a/b/paper.pdf"
    attempt = recovery._download_candidate(record, url, archive, staging, 1, 0)
    destination = recovery._artifact_path(archive, record, "pdf")
    assert calls == [(url, staging, False)]
    assert attempt["status"] == "downloaded"
    assert destination.read_bytes() == b"%PDF-staged"


def test_recovery_falls_back_to_exact_pmc_aws_pdf(tmp_path, monkeypatch):
    record = _record()
    archive = tmp_path / "archive"
    staging = tmp_path / "staging"
    ftp_url = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/a/b/paper.pdf"
    aws_url = "https://pmc-oa-opendata.s3.amazonaws.com/PMC101.2/PMC101.2.pdf"
    monkeypatch.setattr(
        recovery,
        "_fetch_ncbi_oa",
        lambda *args: (
            "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC101",
            {
                "record_id": "PMC101",
                "citation": "Example",
                "license": "CC BY",
                "retracted": "no",
                "pdf_urls": [ftp_url],
            },
        ),
    )
    monkeypatch.setattr(
        recovery,
        "_fetch_pmc_aws_pdf",
        lambda *args: {
            "versions_api_url": "https://example.invalid/versions",
            "versions": ["PMC101.1", "PMC101.2"],
            "selected_version": "PMC101.2",
            "objects_api_url": "https://example.invalid/objects",
            "pdf_url": aws_url,
        },
    )

    def fake_download(record, url, archive, staging, timeout, retries):
        if url == ftp_url:
            return {"status": "failed", "url": url, "error": "HTTP Error 404"}
        assert url == aws_url
        return {
            "status": "downloaded",
            "url": url,
            "error": None,
            "sha256": "abc",
            "bytes": 123,
            "path": "archive/pdf/lab/doi-example.pdf",
            "content_type": "application/octet-stream",
        }

    monkeypatch.setattr(recovery, "_download_candidate", fake_download)
    rows = [
        {
            "doi": "10.1000/example",
            "pmcid": "PMC101",
            "hasPDF": "Y",
            "isOpenAccess": "Y",
            "source": "MED",
            "id": "123",
        }
    ]
    result = recovery._recover_from_europe_pmc(
        record, rows, "https://example.invalid/query", archive, staging, 1, 0
    )
    assert result["status"] == "recovered"
    assert result["selected_url"] == aws_url
    assert result["candidate_urls"] == [ftp_url, aws_url]
    assert [attempt["url"] for attempt in result["attempts"]] == [
        ftp_url,
        aws_url,
    ]
    assert result["attempt"]["url"] == aws_url


def test_download_candidate_does_not_overwrite_concurrent_artifact(
    tmp_path, monkeypatch
):
    record = _record()
    archive = tmp_path / "archive"
    staging = tmp_path / "staging"
    destination = recovery._artifact_path(archive, record, "pdf")

    def fake_download(candidate, stage, timeout, retries, force):
        staged = recovery._artifact_path(stage, candidate, "pdf")
        staged.parent.mkdir(parents=True)
        staged.write_bytes(b"%PDF-staged")
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"concurrent winner")
        return {
            "status": "downloaded",
            "url": candidate["pdf_url"],
            "error": None,
            "sha256": "abc",
            "bytes": 11,
            "path": str(staged),
            "content_type": "application/pdf",
        }

    monkeypatch.setattr(recovery, "_download_one", fake_download)
    url = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/a/b/paper.pdf"
    attempt = recovery._download_candidate(record, url, archive, staging, 1, 0)
    assert attempt["status"] == "skipped_existing"
    assert attempt["url"] is None
    assert destination.read_bytes() == b"concurrent winner"


def test_merge_results_replaces_matching_rows():
    previous = [{"org": "Lab", "id": "a", "status": "old"}]
    current = [{"org": "Lab", "id": "a", "status": "recovered"}]
    assert recovery._merge_results(previous, current) == current
