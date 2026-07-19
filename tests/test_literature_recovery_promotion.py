import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    SPEC = importlib.util.spec_from_file_location(
        "literature_recovery_promotion",
        SCRIPTS / "literature_recovery_promotion.py",
    )
    assert SPEC and SPEC.loader
    promotion = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(promotion)
finally:
    sys.path.remove(str(SCRIPTS))


def record(record_id, pdf_url="https://publisher.example/blocked.pdf"):
    return {
        "id": record_id,
        "org": "Example Lab",
        "title": f"Title for {record_id}",
        "authors": ["Example Author"],
        "date": "2026-07-19",
        "type": "research_paper",
        "tier": "affiliated",
        "arxiv_id": None,
        "doi": None,
        "primary_url": f"https://example.com/{record_id}",
        "pdf_url": pdf_url,
        "source_pages": [f"https://example.com/{record_id}"],
        "affiliation_evidence": "The paper names Example Lab.",
        "topics": ["testing"],
        "notes": None,
        "retrieved_at": "2026-07-19",
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def artifact(archive, item, marker):
    payload = b"%PDF-1.4\n" + marker.encode("ascii") + b"\n%%EOF\n"
    path = promotion._artifact_path(archive, item, "pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path, payload, hashlib.sha256(payload).hexdigest()


def fake_pdfinfo(monkeypatch, pages=1):
    monkeypatch.setattr(promotion.shutil, "which", lambda name: "/usr/bin/pdfinfo")
    monkeypatch.setattr(
        promotion.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=f"Pages: {pages}\n", stderr=""
        ),
    )


def arguments(inventory, archive, oa, secondary, manual, apply=False, pmc=None):
    if pmc is None:
        pmc = manual.with_name("pmc.jsonl")
        if not pmc.exists():
            write_jsonl(pmc, [])
    values = [
        "--inventory",
        str(inventory),
        "--archive",
        str(archive),
        "--oa-recovery",
        str(oa),
        "--secondary-recovery",
        str(secondary),
        "--manual-recovery",
        str(manual),
        "--pmc-recovery",
        str(pmc),
    ]
    if apply:
        values.append("--apply")
    return values


def test_dry_run_validates_all_four_recovery_schemas_without_writing(
    tmp_path, monkeypatch, capsys
):
    inventory = tmp_path / "inventory"
    archive = tmp_path / "archive"
    items = [
        record("oa-paper"),
        record("secondary-paper"),
        record("manual-paper"),
        record("pmc-paper"),
    ]
    inventory_path = inventory / "example.jsonl"
    write_jsonl(inventory_path, items)
    original = inventory_path.read_bytes()
    paths = [artifact(archive, item, item["id"]) for item in items]

    oa = tmp_path / "oa.jsonl"
    secondary = tmp_path / "secondary.jsonl"
    manual = tmp_path / "manual.jsonl"
    pmc = tmp_path / "pmc.jsonl"
    oa_url = "https://repository.example/oa-paper.pdf"
    write_jsonl(
        oa,
        [
            {
                "org": items[0]["org"],
                "id": items[0]["id"],
                "title": items[0]["title"],
                "doi": None,
                "arxiv_id": None,
                "current_url": items[0]["pdf_url"],
                "selected_url": oa_url,
                "status": "recovered",
                "attempts": [
                    {
                        "url": oa_url,
                        "status": "downloaded",
                        "bytes": len(paths[0][1]),
                        "sha256": paths[0][2],
                    }
                ],
            }
        ],
    )
    secondary_url = "https://arxiv.org/pdf/2607.12345"
    write_jsonl(
        secondary,
        [
            {
                "org": items[1]["org"],
                "id": items[1]["id"],
                "title": items[1]["title"],
                "doi": None,
                "selected_url": secondary_url,
                "selected_arxiv_id": "2607.12345",
                "status": "recovered",
                "attempt": {
                    "url": secondary_url,
                    "status": "downloaded",
                    "bytes": len(paths[1][1]),
                    "sha256": paths[1][2],
                    "path": str(paths[1][0]),
                },
            }
        ],
    )
    manual_url = "https://arxiv.org/pdf/2607.54321"
    write_jsonl(
        manual,
        [
            {
                "org": items[2]["org"],
                "id": items[2]["id"],
                "current_url": items[2]["pdf_url"],
                "selected_url": manual_url,
                "status": "recovered",
                "bytes": len(paths[2][1]),
                "sha256": paths[2][2],
                "pages": 1,
                "validation": {"pdf_magic": True, "pdfinfo": True},
                "title_match": True,
            }
        ],
    )
    pmc_url = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/example.pdf"
    write_jsonl(
        pmc,
        [
            {
                "org": items[3]["org"],
                "id": items[3]["id"],
                "title": items[3]["title"],
                "doi": None,
                "selected_url": pmc_url,
                "status": "recovered",
                "attempt": {
                    "url": pmc_url,
                    "status": "downloaded",
                    "bytes": len(paths[3][1]),
                    "sha256": paths[3][2],
                    "path": str(paths[3][0]),
                    "content_type": "application/pdf",
                },
            }
        ],
    )
    fake_pdfinfo(monkeypatch)

    assert (
        promotion.main(
            arguments(inventory, archive, oa, secondary, manual, pmc=pmc)
        )
        == 0
    )
    assert inventory_path.read_bytes() == original
    output = capsys.readouterr().out
    assert "DRY-RUN: validated=4 changed_records=4 unchanged=0" in output
    assert "dry-run only; no inventory files were written" in output


def test_apply_is_idempotent_and_only_changes_allowed_fields(
    tmp_path, monkeypatch, capsys
):
    inventory = tmp_path / "inventory"
    archive = tmp_path / "archive"
    item = record("manual-paper")
    inventory_path = inventory / "example.jsonl"
    write_jsonl(inventory_path, [item])
    path, payload, sha256 = artifact(archive, item, item["id"])
    oa = tmp_path / "oa.jsonl"
    secondary = tmp_path / "secondary.jsonl"
    manual = tmp_path / "manual.jsonl"
    write_jsonl(oa, [])
    write_jsonl(secondary, [])
    selected = "https://arxiv.org/pdf/2607.54321"
    write_jsonl(
        manual,
        [
            {
                "org": item["org"],
                "id": item["id"],
                "current_url": item["pdf_url"],
                "selected_url": selected,
                "status": "recovered",
                "bytes": len(payload),
                "sha256": sha256,
                "pages": 1,
                "validation": {"pdf_magic": True, "pdfinfo": True},
                "title_match": True,
            }
        ],
    )
    fake_pdfinfo(monkeypatch)
    apply_args = arguments(inventory, archive, oa, secondary, manual, apply=True)

    assert promotion.main(apply_args) == 0
    promoted = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert promoted["pdf_url"] == selected
    assert promoted["arxiv_id"] == "2607.54321"
    assert promoted["source_pages"] == item["source_pages"] + [selected]
    for field in set(item) - promotion.ALLOWED_FIELDS:
        assert promoted[field] == item[field]

    first_application = inventory_path.read_bytes()
    capsys.readouterr()
    assert promotion.main(apply_args) == 0
    assert inventory_path.read_bytes() == first_application
    assert "changed_records=0 unchanged=1" in capsys.readouterr().out
    assert path.is_file()


def test_any_validation_failure_prevents_all_inventory_writes(
    tmp_path, monkeypatch, capsys
):
    inventory = tmp_path / "inventory"
    archive = tmp_path / "archive"
    items = [record("good-paper"), record("bad-paper")]
    inventory_path = inventory / "example.jsonl"
    write_jsonl(inventory_path, items)
    original = inventory_path.read_bytes()
    artifacts = [artifact(archive, item, item["id"]) for item in items]
    oa = tmp_path / "oa.jsonl"
    secondary = tmp_path / "secondary.jsonl"
    manual = tmp_path / "manual.jsonl"
    write_jsonl(oa, [])
    write_jsonl(secondary, [])
    rows = []
    for index, item in enumerate(items):
        rows.append(
            {
                "org": item["org"],
                "id": item["id"],
                "current_url": item["pdf_url"],
                "selected_url": f"https://repository.example/{item['id']}.pdf",
                "status": "recovered",
                "bytes": len(artifacts[index][1]),
                "sha256": artifacts[index][2] if index == 0 else "0" * 64,
                "pages": 1,
                "validation": {"pdf_magic": True, "pdfinfo": True},
                "title_match": True,
            }
        )
    write_jsonl(manual, rows)
    fake_pdfinfo(monkeypatch)

    assert (
        promotion.main(
            arguments(inventory, archive, oa, secondary, manual, apply=True)
        )
        == 2
    )
    assert inventory_path.read_bytes() == original
    assert "local SHA-256 does not match" in capsys.readouterr().err


def test_pdfinfo_is_mandatory(tmp_path, monkeypatch, capsys):
    inventory = tmp_path / "inventory"
    archive = tmp_path / "archive"
    item = record("paper")
    write_jsonl(inventory / "example.jsonl", [item])
    _, payload, sha256 = artifact(archive, item, item["id"])
    oa = tmp_path / "oa.jsonl"
    secondary = tmp_path / "secondary.jsonl"
    manual = tmp_path / "manual.jsonl"
    write_jsonl(oa, [])
    write_jsonl(secondary, [])
    write_jsonl(
        manual,
        [
            {
                "org": item["org"],
                "id": item["id"],
                "current_url": item["pdf_url"],
                "selected_url": "https://repository.example/paper.pdf",
                "status": "recovered",
                "bytes": len(payload),
                "sha256": sha256,
                "pages": 1,
                "validation": {"pdf_magic": True, "pdfinfo": True},
                "title_match": True,
            }
        ],
    )
    monkeypatch.setattr(promotion.shutil, "which", lambda name: None)

    assert promotion.main(arguments(inventory, archive, oa, secondary, manual)) == 2
    assert "pdfinfo is required" in capsys.readouterr().err


def test_existing_same_org_arxiv_owner_suppresses_identifier_promotion(
    tmp_path, monkeypatch, capsys
):
    inventory = tmp_path / "inventory"
    archive = tmp_path / "archive"
    target = record("doi-version")
    owner = record("arxiv-owner", pdf_url="https://arxiv.org/pdf/2403.10519")
    owner["arxiv_id"] = "2403.10519"
    inventory_path = inventory / "example.jsonl"
    write_jsonl(inventory_path, [target, owner])
    path, payload, sha256 = artifact(archive, target, target["id"])
    oa = tmp_path / "oa.jsonl"
    secondary = tmp_path / "secondary.jsonl"
    manual = tmp_path / "manual.jsonl"
    pmc = tmp_path / "pmc.jsonl"
    write_jsonl(oa, [])
    write_jsonl(manual, [])
    write_jsonl(pmc, [])
    selected = "https://arxiv.org/pdf/2403.10519"
    write_jsonl(
        secondary,
        [
            {
                "org": target["org"],
                "id": target["id"],
                "title": target["title"],
                "doi": None,
                "selected_url": selected,
                "selected_arxiv_id": "2403.10519",
                "status": "recovered",
                "attempt": {
                    "url": selected,
                    "status": "downloaded",
                    "bytes": len(payload),
                    "sha256": sha256,
                    "path": str(path),
                },
            }
        ],
    )
    fake_pdfinfo(monkeypatch)
    apply_args = arguments(
        inventory,
        archive,
        oa,
        secondary,
        manual,
        apply=True,
        pmc=pmc,
    )

    assert promotion.main(apply_args) == 0
    rows = [json.loads(line) for line in inventory_path.read_text().splitlines()]
    promoted = next(row for row in rows if row["id"] == target["id"])
    preserved_owner = next(row for row in rows if row["id"] == owner["id"])
    assert promoted["pdf_url"] == selected
    assert promoted["arxiv_id"] is None
    assert promoted["source_pages"] == target["source_pages"] + [selected]
    assert preserved_owner == owner
    output = capsys.readouterr().out
    assert "pdf_url,source_pages existing_arxiv_owner=arxiv-owner" in output

    first_application = inventory_path.read_bytes()
    assert promotion.main(apply_args) == 0
    assert inventory_path.read_bytes() == first_application
    assert "changed_records=0 unchanged=1" in capsys.readouterr().out
