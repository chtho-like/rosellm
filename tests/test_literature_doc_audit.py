import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "literature_doc_audit.py"
SPEC = importlib.util.spec_from_file_location("literature_doc_audit", MODULE_PATH)
assert SPEC and SPEC.loader
literature_doc_audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(literature_doc_audit)


INVENTORY_FIXTURES = {
    "anthropic": ("Anthropic", "anthropic-web-a", "https://example.test/anthropic"),
    "deepseek": ("DeepSeek", "deepseek-2401.00001", "https://example.test/deepseek"),
    "gemini": ("Google DeepMind", "gdm-url-a", "https://example.test/gemini"),
    "glm": ("Zhipu AI / Z.ai", "arxiv-2601.00001", "https://example.test/glm"),
    "kimi": ("Moonshot AI / Kimi", "kimi-2401.00001", "https://example.test/kimi"),
    "openai": ("OpenAI", "web-a", "https://example.test/openai"),
}


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _record(organization: str, record_id: str, primary_url: str):
    return {"id": record_id, "org": organization, "primary_url": primary_url}


def _write_fixture(root: Path):
    docs = root / "docs" / "frontier-labs"
    docs.mkdir(parents=True)
    inventory = root / "research" / "literature" / "inventory"

    for name, (organization, record_id, primary_url) in INVENTORY_FIXTURES.items():
        _write_jsonl(
            inventory / f"{name}.jsonl",
            [_record(organization, record_id, primary_url)],
        )

    docs.joinpath("anthropic.md").write_text(
        "| record ID | source |\n"
        "|---|---|\n"
        "| anthropic-web-a | [source](https://example.test/anthropic) |\n",
        encoding="utf-8",
    )
    docs.joinpath("deepseek-kimi.md").write_text(
        "| `deepseek-2401.00001` | <https://example.test/deepseek> |\n"
        "| `kimi-2401.00001` | <https://example.test/kimi> |\n",
        encoding="utf-8",
    )
    docs.joinpath("gemini.md").write_text(
        "- [gdm-url-a](<https://example.test/gemini>)\n", encoding="utf-8"
    )
    docs.joinpath("glm.md").write_text(
        "| `arxiv-2601.00001` | <https://example.test/glm> |\n",
        encoding="utf-8",
    )
    docs.joinpath("openai.md").write_text(
        "| <code>web-a</code> | <https://example.test/openai> |\n",
        encoding="utf-8",
    )

    organizations = {
        organization: {
            "records": 1,
            "public_pdf_or_arxiv": 1,
            "local_pdf": 1,
            "extracted_text": 1,
        }
        for organization, _, _ in INVENTORY_FIXTURES.values()
    }
    coverage = {
        "organizations": organizations,
        "totals": {
            "records": 6,
            "public_pdf_or_arxiv": 6,
            "local_pdf": 6,
            "extracted_text": 6,
        },
    }
    coverage_path = root / "research" / "literature" / "coverage.json"
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_path.write_text(json.dumps(coverage), encoding="utf-8")

    docs.joinpath("index.md").write_text(
        "# Index\n\n"
        "## 当前成果\n\n"
        "| 机构 | 可审计记录 | 有公开 PDF / arXiv | 本地有效 PDF | 可检索全文 | 深度报告 |\n"
        "|---|---:|---:|---:|---:|---|\n"
        "| Anthropic | 1 | 1 | 1 | 1 | x |\n"
        "| DeepSeek | 1 | 1 | 1 | 1 | x |\n"
        "| Google DeepMind / Gemini | 1 | 1 | 1 | 1 | x |\n"
        "| Moonshot AI / Kimi | 1 | 1 | 1 | 1 | x |\n"
        "| OpenAI | 1 | 1 | 1 | 1 | x |\n"
        "| Zhipu AI / Z.ai | 1 | 1 | 1 | 1 | x |\n"
        "| **总计** | **6** | **6** | **6** | **6** | x |\n\n"
        "## Next\n",
        encoding="utf-8",
    )


def _document(report, filename):
    return next(
        value for value in report["documents"] if value["path"].endswith(filename)
    )


def test_complete_fixture_passes(tmp_path):
    _write_fixture(tmp_path)
    report = literature_doc_audit.audit_repository(tmp_path)

    assert report["inventory_records"] == 6
    assert not literature_doc_audit.report_has_failures(report)
    assert all(not item["missing_ids"] for item in report["documents"])
    assert all(not item["missing_primary_urls"] for item in report["documents"])
    assert report["index"]["mismatches"] == []


def test_document_audit_reports_missing_unpaired_and_extra_ids(tmp_path):
    _write_fixture(tmp_path)
    anthropic = tmp_path / "docs" / "frontier-labs" / "anthropic.md"
    anthropic.write_text(
        "anthropic-web-a is discussed here.\n"
        "The URL is on another line: https://example.test/anthropic\n"
        "| anthropic-web-stale | https://example.test/stale |\n",
        encoding="utf-8",
    )
    deepseek = tmp_path / "docs" / "frontier-labs" / "deepseek-kimi.md"
    deepseek.write_text(
        deepseek.read_text(encoding="utf-8").replace(
            "https://example.test/deepseek", "https://example.test/not-primary"
        ),
        encoding="utf-8",
    )

    report = literature_doc_audit.audit_repository(tmp_path)
    anthropic_result = _document(report, "anthropic.md")
    deepseek_result = _document(report, "deepseek-kimi.md")

    assert anthropic_result["extra_record_ids"] == ["anthropic-web-stale"]
    assert [value["id"] for value in anthropic_result["unpaired_id_primary_url"]] == [
        "anthropic-web-a"
    ]
    assert [value["id"] for value in deepseek_result["missing_primary_urls"]] == [
        "deepseek-2401.00001"
    ]
    assert literature_doc_audit.report_has_failures(report)


def test_index_audit_reports_missing_extra_and_mismatched_rows(tmp_path):
    _write_fixture(tmp_path)
    index = tmp_path / "docs" / "frontier-labs" / "index.md"
    text = index.read_text(encoding="utf-8")
    text = text.replace("| OpenAI | 1 | 1 | 1 | 1 | x |\n", "")
    text = text.replace(
        "| **总计** | **6** | **6** | **6** | **6** | x |",
        "| Extra Lab | 9 | 9 | 9 | 9 | x |\n"
        "| **总计** | **6** | **6** | **99** | **6** | x |",
    )
    index.write_text(text, encoding="utf-8")

    report = literature_doc_audit.audit_repository(tmp_path)
    result = report["index"]

    assert result["missing_rows"] == ["OpenAI"]
    assert result["extra_rows"] == ["Extra Lab"]
    assert result["mismatches"] == [
        {
            "organization": "总计",
            "field": "local_pdf",
            "expected": 6,
            "actual": 99,
        }
    ]
    assert literature_doc_audit.report_has_failures(report)


def test_main_emits_json_and_nonzero_on_failure(tmp_path, capsys):
    _write_fixture(tmp_path)
    openai = tmp_path / "docs" / "frontier-labs" / "openai.md"
    openai.write_text("<code>web-stale</code>\n", encoding="utf-8")

    exit_code = literature_doc_audit.main(["--repo-root", str(tmp_path), "--json"])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert output["documents"][-1]["extra_record_ids"] == ["web-stale"]
    assert output["documents"][-1]["missing_ids"] == ["web-a"]
