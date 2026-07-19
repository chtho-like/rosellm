import importlib.util
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    SPEC = importlib.util.spec_from_file_location(
        "literature_library", SCRIPTS / "literature_library.py"
    )
    assert SPEC and SPEC.loader
    literature_library = importlib.util.module_from_spec(SPEC)
    sys.modules[SPEC.name] = literature_library
    SPEC.loader.exec_module(literature_library)
finally:
    sys.path.remove(str(SCRIPTS))


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
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def create_canonical_pdf(archive: Path, row, body: bytes = b"%PDF-1.7\nfixture"):
    source = literature_library._artifact_path(archive, row, "pdf")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(body)
    return source


def test_filename_field_normalizes_markup_unicode_and_unsafe_punctuation():
    assert (
        literature_library.filename_field("  <scp>AlphaFold</scp>: β / structure?  ")
        == "AlphaFold-β-structure"
    )
    assert literature_library.filename_field("Model’s capabilities") == (
        "Models-capabilities"
    )
    assert literature_library.filename_field("CON.txt") == "_CON.txt"
    assert literature_library.filename_field("COM1.log") == "_COM1.log"


def test_readable_filename_uses_alias_title_tail_and_stable_fields():
    row = record(
        id="deepseek-2501.12948",
        title=(
            "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via "
            "Reinforcement Learning"
        ),
        date="2025-01-22",
    )
    assert literature_library.readable_filename(row, "DeepSeek-R1") == (
        "DeepSeek-R1--Incentivizing-Reasoning-Capability-in-LLMs-via-"
        "Reinforcement-Learning--2025--TR--deepseek-2501.12948.pdf"
    )


def test_alias_that_is_not_a_title_prefix_keeps_the_full_title():
    row = record(
        id="gdm-doi-10.1038-nature16961",
        title="Mastering the game of Go with deep neural networks and tree search",
        date="2016-01-26",
        type="research_paper",
    )
    name = literature_library.readable_filename(row, "AlphaGo")
    assert name.startswith("AlphaGo--Mastering-the-game-of-Go-")
    assert name.endswith("--2016--PAPER--gdm-doi-10.1038-nature16961.pdf")


def test_curated_report_name_does_not_repeat_a_terse_inventory_title():
    row = record(id="gpt-4", title="GPT-4", type="blog_with_report")
    assert literature_library.readable_filename(row, "GPT-4 Technical Report") == (
        "GPT-4-Technical-Report--2026--REPORT--gpt-4.pdf"
    )


def test_long_title_is_utf8_byte_bounded_and_preserves_identity_suffix():
    row = record(title=("超长标题 and a detailed explanation " * 20).strip())
    name = literature_library.readable_filename(row, "Model-β", 100)
    assert len(name.encode("utf-8")) <= 100
    assert "~--2026--TR--example-report.pdf" in name
    assert name.startswith("Model-β--")


def test_rejects_filename_limit_that_cannot_preserve_stable_fields():
    with pytest.raises(literature_library.LibraryError, match="at least 80"):
        literature_library.readable_filename(record(), None, 79)


def test_display_names_reject_duplicate_and_unknown_records(tmp_path):
    aliases = tmp_path / "aliases.jsonl"
    duplicate = {
        "org": "Example Lab",
        "id": "example-report",
        "short_name": "Example",
    }
    write_jsonl(aliases, [duplicate, duplicate])
    with pytest.raises(literature_library.LibraryError, match="duplicate"):
        literature_library.load_display_names(aliases, [record()])

    write_jsonl(
        aliases,
        [{"org": "Example Lab", "id": "missing", "short_name": "Missing"}],
    )
    with pytest.raises(literature_library.LibraryError, match="unknown inventory"):
        literature_library.load_display_names(aliases, [record()])

    write_jsonl(
        aliases,
        [
            {
                "org": "Example Lab",
                "id": "example-report",
                "short_name": "Example",
                "notez": "misspelled provenance field",
            }
        ],
    )
    with pytest.raises(literature_library.LibraryError, match="unknown fields"):
        literature_library.load_display_names(aliases, [record()])


def test_plan_and_build_create_hardlink_marker_and_index(tmp_path):
    archive = tmp_path / "archive"
    inventory = tmp_path / "inventory"
    inventory.mkdir()
    row = record()
    source = create_canonical_pdf(archive, row)
    write_jsonl(
        archive / "manifest.jsonl",
        [
            {
                "org": row["org"],
                "id": row["id"],
                "status": "downloaded",
                "sha256": "a" * 64,
            }
        ],
    )
    entries = literature_library.plan_library(
        [row], archive, {(row["org"].casefold(), row["id"]): "Example"}
    )
    library = tmp_path / "library"
    literature_library.build_library(
        entries, library, archive, inventory, link_mode="hardlink"
    )

    destination = library / entries[0].relative_path
    assert destination.is_file()
    assert os.path.samefile(source, destination)
    assert source.stat().st_ino == destination.stat().st_ino
    marker = json.loads(
        (library / literature_library.MARKER_NAME).read_text(encoding="utf-8")
    )
    assert marker["entries"] == 1
    rows = [
        json.loads(line)
        for line in (library / literature_library.INDEX_NAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows == [entries[0].index_row]
    assert rows[0]["sha256"] == "a" * 64
    assert rows[0]["artifact_kind"] == "canonical"


def test_preserved_version_gets_revision_suffix_and_hardlink(tmp_path):
    archive = tmp_path / "archive"
    inventory = tmp_path / "inventory"
    inventory.mkdir()
    row = record()
    create_canonical_pdf(archive, row)
    version_source = archive / "versions" / "example-lab" / row["id"] / "b.pdf"
    version_source.parent.mkdir(parents=True)
    version_source.write_bytes(b"%PDF-1.7\nolder revision")
    versions = tmp_path / "versions.jsonl"
    write_jsonl(
        versions,
        [
            {
                "org": row["org"],
                "canonical_id": row["id"],
                "relation": "alternate_official_revision",
                "sha256": "b" * 64,
                "archived_pdf_path": str(version_source),
                "source_url": "https://example.com/older.pdf",
            }
        ],
    )
    entries = literature_library.plan_library([row], archive, {}, versions=versions)
    assert len(entries) == 2
    version_entry = next(
        entry
        for entry in entries
        if entry.index_row["artifact_kind"] == "preserved_version"
    )
    assert version_entry.relative_path.name.endswith("--rev-bbbbbbbb.pdf")
    assert len(version_entry.relative_path.name.encode("utf-8")) <= 180

    library = tmp_path / "library"
    literature_library.build_library(
        entries, library, archive, inventory, link_mode="hardlink"
    )
    assert os.path.samefile(version_source, library / version_entry.relative_path)
    marker = json.loads(
        (library / literature_library.MARKER_NAME).read_text(encoding="utf-8")
    )
    assert marker["canonical_entries"] == 1
    assert marker["version_entries"] == 1


def test_rebuild_atomically_replaces_previous_generated_view(tmp_path):
    archive = tmp_path / "archive"
    inventory = tmp_path / "inventory"
    inventory.mkdir()
    first = record(id="first", title="First")
    second = record(id="second", title="Second")
    create_canonical_pdf(archive, first)
    create_canonical_pdf(archive, second)
    library = tmp_path / "library"

    first_entries = literature_library.plan_library([first], archive, {})
    literature_library.build_library(
        first_entries, library, archive, inventory, link_mode="hardlink"
    )
    assert (library / first_entries[0].relative_path).exists()

    second_entries = literature_library.plan_library([second], archive, {})
    literature_library.build_library(
        second_entries, library, archive, inventory, link_mode="hardlink"
    )
    assert not (library / first_entries[0].relative_path).exists()
    assert (library / second_entries[0].relative_path).exists()
    assert not list(tmp_path.glob(".library-backup-*"))
    assert not list(tmp_path.glob(".library-build-*"))


@pytest.mark.parametrize(
    "marker",
    [
        None,
        "not json\n",
        json.dumps({"schema_version": 1, "generator": "some-other-tool"}),
    ],
)
def test_refuses_to_replace_unmarked_or_unrecognized_directory(tmp_path, marker):
    archive = tmp_path / "archive"
    inventory = tmp_path / "inventory"
    inventory.mkdir()
    library = tmp_path / "library"
    library.mkdir()
    (library / "user-file.txt").write_text("preserve me", encoding="utf-8")
    if marker is not None:
        (library / literature_library.MARKER_NAME).write_text(marker, encoding="utf-8")

    with pytest.raises(
        literature_library.LibraryError, match="refusing|invalid|unrecognized"
    ):
        literature_library.build_library(
            [], library, archive, inventory, link_mode="hardlink"
        )
    assert (library / "user-file.txt").read_text(encoding="utf-8") == "preserve me"


def test_dry_run_creates_nothing(tmp_path):
    archive = tmp_path / "archive"
    inventory = tmp_path / "inventory"
    inventory.mkdir()
    row = record()
    create_canonical_pdf(archive, row)
    entries = literature_library.plan_library([row], archive, {})
    library = tmp_path / "library"
    literature_library.build_library(entries, library, archive, inventory, dry_run=True)
    assert not library.exists()


def test_library_must_not_contain_archive_or_inventory(tmp_path):
    library = tmp_path / "library"
    archive = library / "archive"
    inventory = tmp_path / "inventory"
    archive.mkdir(parents=True)
    inventory.mkdir()
    evidence = archive / "preserve.pdf"
    evidence.write_bytes(b"%PDF-1.7\npreserve")
    with pytest.raises(literature_library.LibraryError, match="must not overlap"):
        literature_library.build_library([], library, archive, inventory, dry_run=True)
    assert evidence.is_file()


def test_marker_symlink_and_unindexed_content_are_never_deleted(tmp_path):
    archive = tmp_path / "archive"
    inventory = tmp_path / "inventory"
    inventory.mkdir()
    library = tmp_path / "library"
    library.mkdir()
    marker_target = tmp_path / "marker.json"
    marker_target.write_text(
        json.dumps(
            {
                "schema_version": literature_library.SCHEMA_VERSION,
                "generator": "scripts/literature_library.py",
            }
        ),
        encoding="utf-8",
    )
    (library / literature_library.MARKER_NAME).symlink_to(marker_target)
    with pytest.raises(literature_library.LibraryError, match="unmarked"):
        literature_library.build_library([], library, archive, inventory)
    assert marker_target.is_file()

    (library / literature_library.MARKER_NAME).unlink()
    library.rmdir()
    literature_library.build_library(
        [], library, archive, inventory, link_mode="hardlink"
    )
    personal = library / "personal-notes.txt"
    personal.write_text("do not delete", encoding="utf-8")
    with pytest.raises(literature_library.LibraryError, match="unindexed content"):
        literature_library.build_library(
            [], library, archive, inventory, link_mode="hardlink"
        )
    assert personal.read_text(encoding="utf-8") == "do not delete"


def test_copied_marker_cannot_authorize_a_different_output_root(tmp_path):
    archive = tmp_path / "archive"
    inventory = tmp_path / "inventory"
    inventory.mkdir()
    original = tmp_path / "original-library"
    literature_library.build_library(
        [], original, archive, inventory, link_mode="hardlink"
    )

    copied = tmp_path / "copied-library"
    copied.mkdir()
    for name in (
        literature_library.MARKER_NAME,
        literature_library.INDEX_NAME,
    ):
        (copied / name).write_bytes((original / name).read_bytes())
    with pytest.raises(literature_library.LibraryError, match="different root"):
        literature_library.build_library(
            [], copied, archive, inventory, link_mode="hardlink"
        )
    assert (copied / literature_library.MARKER_NAME).is_file()


def test_sha_verification_rejects_stale_manifest(tmp_path):
    archive = tmp_path / "archive"
    row = record()
    source = create_canonical_pdf(archive, row)
    write_jsonl(
        archive / "manifest.jsonl",
        [
            {
                "org": row["org"],
                "id": row["id"],
                "status": "existing",
                "sha256": "a" * 64,
            }
        ],
    )
    with pytest.raises(literature_library.LibraryError, match="SHA-256 mismatch"):
        literature_library.plan_library([row], archive, {}, verify_sha=True)

    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    write_jsonl(
        archive / "manifest.jsonl",
        [
            {
                "org": row["org"],
                "id": row["id"],
                "status": "existing",
                "sha256": actual,
            }
        ],
    )
    entries = literature_library.plan_library([row], archive, {}, verify_sha=True)
    assert entries[0].index_row["sha256"] == actual


@pytest.mark.skipif(sys.platform != "darwin", reason="tests the macOS clonefile path")
def test_clone_view_has_independent_inode_and_cannot_write_through(tmp_path):
    archive = tmp_path / "archive"
    inventory = tmp_path / "inventory"
    inventory.mkdir()
    row = record()
    source = create_canonical_pdf(archive, row)
    original = source.read_bytes()
    entries = literature_library.plan_library([row], archive, {}, link_mode="clone")
    library = tmp_path / "library"
    literature_library.build_library(
        entries, library, archive, inventory, link_mode="clone"
    )
    destination = library / entries[0].relative_path
    assert not os.path.samefile(source, destination)
    assert source.stat().st_ino != destination.stat().st_ino
    destination.write_bytes(b"changed browsing copy")
    assert source.read_bytes() == original


def test_repository_aliases_and_all_derived_names_are_valid_and_unique():
    records = literature_library.load_records(literature_library.DEFAULT_INVENTORY)
    aliases = literature_library.load_display_names(
        literature_library.DEFAULT_ALIASES, records
    )
    assert len(aliases) >= 100
    seen = set()
    for row in records:
        key = (row["org"].casefold(), row["id"])
        name = literature_library.readable_filename(row, aliases.get(key))
        assert len(name.encode("utf-8")) <= 180
        assert not re.search(r'[\\/:*?"<>|]', name)
        year = (row.get("date") or "undated")[:4]
        path_key = (row["org"].casefold(), row["tier"], year, name.casefold())
        assert path_key not in seen
        seen.add(path_key)
