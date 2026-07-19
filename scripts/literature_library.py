#!/usr/bin/env python3
"""Build a human-readable copy-on-write view of the canonical PDF corpus.

The canonical archive remains keyed by immutable inventory IDs.  This tool
creates a disposable, Git-ignored browsing tree whose filenames are derived
from curated display names and inventory metadata. Copy-on-write clones have
independent inodes while initially sharing the canonical PDF data blocks.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from literature_corpus import (
    DEFAULT_ARCHIVE,
    DEFAULT_INVENTORY,
    REPO_ROOT,
    _artifact_path,
    _sha256,
    load_records,
)


DEFAULT_ALIASES = REPO_ROOT / "research" / "literature" / "display-names.jsonl"
DEFAULT_LIBRARY = REPO_ROOT / "research" / "literature" / "library"
DEFAULT_VERSIONS = (
    REPO_ROOT / "research" / "literature" / "candidates" / "document-versions.jsonl"
)
MARKER_NAME = ".rosellm-literature-library.json"
INDEX_NAME = "index.jsonl"
SCHEMA_VERSION = 2
DEFAULT_MAX_FILENAME_BYTES = 180
DEFAULT_LINK_MODE = "clone"

TYPE_CODES = {
    "technical_report": "TR",
    "research_paper": "PAPER",
    "system_card": "SC",
    "model_card": "MC",
    "dataset": "DATA",
    "benchmark": "BENCH",
    "blog_with_report": "REPORT",
    "other": "OTHER",
}
RESERVED_WINDOWS_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}
TAG_RE = re.compile(r"<[^>]*>")
SEPARATOR_RE = re.compile(r"-+")


class LibraryError(ValueError):
    """The readable library cannot be generated safely."""


@dataclass(frozen=True)
class LibraryEntry:
    """One preflighted canonical-PDF to readable-library mapping."""

    source: Path
    relative_path: Path
    index_row: dict[str, Any]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise LibraryError(f"required JSONL file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise LibraryError(f"{path}:{number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise LibraryError(f"{path}:{number}: expected a JSON object")
            rows.append(row)
    return rows


def _plain_text(value: str) -> str:
    """Normalize markup and Unicode before filename sanitization."""
    value = html.unescape(TAG_RE.sub("", value))
    value = unicodedata.normalize("NFKC", value)
    return " ".join(value.split())


def filename_field(value: str) -> str:
    """Return a readable filename field safe on common desktop filesystems."""
    value = _plain_text(value)
    pieces: list[str] = []
    pending_separator = False
    for character in value:
        if character.isalnum() or character in {".", "+"}:
            if pending_separator and pieces and pieces[-1] != "-":
                pieces.append("-")
            pieces.append(character)
            pending_separator = False
        elif character in {"'", "’", "`"}:
            # Apostrophes do not separate words and are unsafe/noisy in shells.
            continue
        else:
            pending_separator = True
    result = SEPARATOR_RE.sub("-", "".join(pieces)).strip("-. ")
    if not result:
        raise LibraryError(
            f"value becomes empty after filename normalization: {value!r}"
        )
    if result.split(".", 1)[0].casefold() in RESERVED_WINDOWS_NAMES:
        result = f"_{result}"
    return result


def _title_without_alias(title: str, short_name: str | None) -> str:
    title = _plain_text(title)
    if not short_name:
        return title
    alias = _plain_text(short_name)
    if title.casefold().startswith(alias.casefold()):
        remainder = title[len(alias) :].lstrip(" \t:;,.–—-_/")
        return remainder
    if alias.casefold().startswith(title.casefold()):
        remainder = alias[len(title) :]
        if not remainder or remainder[0] in " \t:;,.–—-_/([":
            return ""
    return title


def _truncate_utf8(value: str, maximum_bytes: int) -> str:
    if maximum_bytes <= 0:
        return ""
    encoded = value.encode("utf-8")
    if len(encoded) <= maximum_bytes:
        return value
    output: list[str] = []
    used = 0
    for character in value:
        size = len(character.encode("utf-8"))
        if used + size > maximum_bytes:
            break
        output.append(character)
        used += size
    return "".join(output).rstrip("-. ")


def readable_filename(
    record: dict[str, Any],
    short_name: str | None,
    maximum_bytes: int = DEFAULT_MAX_FILENAME_BYTES,
) -> str:
    """Derive a deterministic readable filename while preserving stable fields."""
    if maximum_bytes < 80:
        raise LibraryError("maximum filename size must be at least 80 UTF-8 bytes")
    year_match = re.match(r"^(\d{4})", str(record.get("date") or ""))
    year = year_match.group(1) if year_match else "undated"
    type_code = TYPE_CODES.get(str(record.get("type")))
    if not type_code:
        raise LibraryError(
            f"unsupported record type for {record.get('id')}: {record.get('type')}"
        )

    alias_field = filename_field(short_name) if short_name else None
    title_text = _title_without_alias(str(record["title"]), short_name)
    title_field = filename_field(title_text) if title_text else None
    stable_fields = [year, type_code, str(record["id"])]
    leading_fields = [field for field in (alias_field, title_field) if field]
    candidate = "--".join([*leading_fields, *stable_fields]) + ".pdf"
    if len(candidate.encode("utf-8")) <= maximum_bytes:
        return candidate

    # The alias and identity fields carry the navigation and collision contract.
    # Only the title may be shortened, with '~' making truncation explicit.
    if not title_field:
        raise LibraryError(
            f"stable filename fields exceed {maximum_bytes} bytes for {record['id']}"
        )
    fixed_fields = [field for field in (alias_field,) if field]
    fixed_prefix = "--".join(fixed_fields)
    fixed_suffix = "--".join(stable_fields) + ".pdf"
    separators = (2 if fixed_prefix else 0) + 2
    available = (
        maximum_bytes
        - len(fixed_prefix.encode("utf-8"))
        - len(fixed_suffix.encode("utf-8"))
        - separators
    )
    if available < 9:
        raise LibraryError(
            f"alias and stable fields leave no title space for {record['id']}"
        )
    shortened = _truncate_utf8(title_field, available - 1)
    if not shortened:
        raise LibraryError(f"could not shorten title safely for {record['id']}")
    shortened += "~"
    result = "--".join([*fixed_fields, shortened, *stable_fields]) + ".pdf"
    if len(result.encode("utf-8")) > maximum_bytes:
        raise LibraryError(f"internal filename length error for {record['id']}")
    return result


def load_display_names(
    path: Path, records: Iterable[dict[str, Any]]
) -> dict[tuple[str, str], str]:
    """Load and validate the curated, sparse short-name ledger."""
    record_index = {
        (str(record["org"]).casefold(), str(record["id"])): record for record in records
    }
    aliases: dict[tuple[str, str], str] = {}
    for number, row in enumerate(_read_jsonl(path), 1):
        allowed_fields = {"org", "id", "short_name", "notes"}
        extra = set(row) - allowed_fields
        if extra:
            raise LibraryError(f"{path}:{number}: unknown fields {sorted(extra)}")
        missing = {"org", "id", "short_name"} - set(row)
        if missing:
            raise LibraryError(f"{path}:{number}: missing fields {sorted(missing)}")
        if not all(
            isinstance(row[field], str) for field in ("org", "id", "short_name")
        ):
            raise LibraryError(
                f"{path}:{number}: org, id, and short_name must be strings"
            )
        key = (row["org"].casefold(), row["id"])
        if key in aliases:
            raise LibraryError(f"{path}:{number}: duplicate display-name key {key}")
        record = record_index.get(key)
        if not record:
            raise LibraryError(f"{path}:{number}: unknown inventory record {key}")
        if row["org"] != record["org"]:
            raise LibraryError(
                f"{path}:{number}: organization spelling must match inventory: {record['org']}"
            )
        normalized = filename_field(row["short_name"])
        if len(normalized.encode("utf-8")) > 64:
            raise LibraryError(f"{path}:{number}: short_name exceeds 64 UTF-8 bytes")
        if "notes" in row and not isinstance(row["notes"], str):
            raise LibraryError(f"{path}:{number}: notes must be a string when present")
        aliases[key] = row["short_name"]
    return aliases


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _manifest_sha_index(archive: Path) -> dict[tuple[str, str], str]:
    path = archive / "manifest.jsonl"
    if not path.exists():
        return {}
    rows: dict[tuple[str, str], str] = {}
    for number, row in enumerate(_read_jsonl(path), 1):
        if row.get("sha256") and row.get("org") and row.get("id"):
            key = (str(row["org"]).casefold(), str(row["id"]))
            if key in rows:
                raise LibraryError(f"{path}:{number}: duplicate manifest key {key}")
            sha256 = str(row["sha256"]).casefold()
            if not re.fullmatch(r"[0-9a-f]{64}", sha256):
                raise LibraryError(f"{path}:{number}: invalid SHA-256")
            if row.get("status") not in {"downloaded", "existing"}:
                raise LibraryError(
                    f"{path}:{number}: SHA-256 on non-success status {row.get('status')}"
                )
            rows[key] = sha256
    return rows


def plan_library(
    records: Iterable[dict[str, Any]],
    archive: Path,
    aliases: dict[tuple[str, str], str],
    maximum_filename_bytes: int = DEFAULT_MAX_FILENAME_BYTES,
    versions: Path | None = None,
    verify_sha: bool = False,
    link_mode: str = DEFAULT_LINK_MODE,
) -> list[LibraryEntry]:
    """Preflight local canonical and preserved-version PDF mappings."""
    sha_index = _manifest_sha_index(archive)
    entries: list[LibraryEntry] = []
    casefold_paths: dict[str, tuple[str, str]] = {}
    record_rows = sorted(records, key=lambda row: (row["org"].casefold(), row["id"]))
    record_index = {
        (record["org"].casefold(), record["id"]): record for record in record_rows
    }

    def append_entry(
        record: dict[str, Any],
        source: Path,
        relative: Path,
        index_row: dict[str, Any],
    ) -> None:
        folded = relative.as_posix().casefold()
        if folded in casefold_paths:
            first = casefold_paths[folded]
            raise LibraryError(
                f"case-insensitive readable path collision: {first} and "
                f"{(record['org'], record['id'])}"
            )
        casefold_paths[folded] = (record["org"], record["id"])
        entries.append(
            LibraryEntry(source=source, relative_path=relative, index_row=index_row)
        )

    def common_index_row(
        record: dict[str, Any],
        short_name: str | None,
        source: Path,
        canonical_source: Path,
        relative: Path,
        sha256: str | None,
        artifact_kind: str,
    ) -> dict[str, Any]:
        year_match = re.match(r"^(\d{4})", str(record.get("date") or ""))
        year = year_match.group(1) if year_match else "undated"
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": artifact_kind,
            "org": record["org"],
            "id": record["id"],
            "short_name": short_name,
            "title": record["title"],
            "date": record.get("date"),
            "year": year,
            "type": record["type"],
            "type_code": TYPE_CODES[record["type"]],
            "tier": record["tier"],
            "source_path": _display_path(source),
            "canonical_path": _display_path(canonical_source),
            "library_path": relative.as_posix(),
            "sha256": sha256,
            "link_mode": link_mode,
        }

    for record in record_rows:
        source = _artifact_path(archive, record, "pdf")
        if not source.is_file():
            continue
        key = (record["org"].casefold(), record["id"])
        expected_sha = sha_index.get(key)
        if verify_sha:
            if not expected_sha:
                raise LibraryError(f"manifest SHA-256 missing for local PDF {key}")
            actual_sha = _sha256(source)
            if actual_sha != expected_sha:
                raise LibraryError(
                    f"canonical PDF SHA-256 mismatch for {key}: "
                    f"manifest={expected_sha} actual={actual_sha}"
                )
        short_name = aliases.get(key)
        year_match = re.match(r"^(\d{4})", str(record.get("date") or ""))
        year = year_match.group(1) if year_match else "undated"
        org_slug = re.sub(r"[^a-z0-9]+", "-", record["org"].casefold()).strip("-")
        filename = readable_filename(record, short_name, maximum_filename_bytes)
        relative = Path(org_slug) / record["tier"] / year / filename
        append_entry(
            record,
            source,
            relative,
            common_index_row(
                record,
                short_name,
                source,
                source,
                relative,
                expected_sha,
                "canonical",
            ),
        )

    if versions is not None:
        version_root = (archive / "versions").resolve()
        for number, version in enumerate(_read_jsonl(versions), 1):
            missing = {
                "org",
                "canonical_id",
                "relation",
                "sha256",
                "archived_pdf_path",
            } - set(version)
            if missing:
                raise LibraryError(
                    f"{versions}:{number}: missing version fields {sorted(missing)}"
                )
            key = (str(version["org"]).casefold(), str(version["canonical_id"]))
            record = record_index.get(key)
            if not record or version["org"] != record["org"]:
                raise LibraryError(
                    f"{versions}:{number}: unknown canonical record {key}"
                )
            sha256 = str(version["sha256"])
            if not re.fullmatch(r"[0-9a-f]{64}", sha256):
                raise LibraryError(f"{versions}:{number}: invalid version SHA-256")
            source = Path(str(version["archived_pdf_path"]))
            if not source.is_absolute():
                source = REPO_ROOT / source
            if not _is_within(source, version_root):
                raise LibraryError(
                    f"{versions}:{number}: version PDF is outside archive/versions"
                )
            if not source.is_file():
                continue
            if verify_sha:
                actual_sha = _sha256(source)
                if actual_sha != sha256:
                    raise LibraryError(
                        f"preserved version SHA-256 mismatch for {key}: "
                        f"ledger={sha256} actual={actual_sha}"
                    )

            short_name = aliases.get(key)
            year_match = re.match(r"^(\d{4})", str(record.get("date") or ""))
            year = year_match.group(1) if year_match else "undated"
            org_slug = re.sub(r"[^a-z0-9]+", "-", record["org"].casefold()).strip("-")
            revision_suffix = f"--rev-{sha256[:8]}"
            base = readable_filename(
                record,
                short_name,
                maximum_filename_bytes - len(revision_suffix.encode("utf-8")),
            )
            filename = f"{base[:-4]}{revision_suffix}.pdf"
            relative = Path(org_slug) / record["tier"] / year / filename
            canonical_source = _artifact_path(archive, record, "pdf")
            index_row = common_index_row(
                record,
                short_name,
                source,
                canonical_source,
                relative,
                sha256,
                "preserved_version",
            )
            index_row["version_relation"] = version["relation"]
            index_row["version_source_url"] = version.get("source_url")
            append_entry(record, source, relative, index_row)
    return entries


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _validate_existing_generated_tree(library: Path, marker: dict[str, Any]) -> None:
    """Refuse to delete copied markers, symlinks, or unindexed user content."""
    if marker.get("library_root") != str(library.resolve()):
        raise LibraryError("generated-library marker belongs to a different root")
    if marker.get("entries") != marker.get("canonical_entries", 0) + marker.get(
        "version_entries", 0
    ):
        raise LibraryError("generated-library marker entry counts are inconsistent")

    index_path = library / INDEX_NAME
    if index_path.is_symlink() or not index_path.is_file():
        raise LibraryError(f"missing or unsafe generated-library index: {index_path}")
    index_rows = _read_jsonl(index_path)
    if len(index_rows) != marker.get("entries"):
        raise LibraryError("generated-library marker and index counts disagree")

    expected_files = {Path(MARKER_NAME), Path(INDEX_NAME)}
    expected_directories: set[Path] = set()
    for number, row in enumerate(index_rows, 1):
        value = row.get("library_path")
        if not isinstance(value, str):
            raise LibraryError(f"{index_path}:{number}: missing library_path")
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
            raise LibraryError(f"{index_path}:{number}: unsafe library_path {value!r}")
        expected_files.add(relative)
        parent = relative.parent
        while parent != Path("."):
            expected_directories.add(parent)
            parent = parent.parent

    actual_files: set[Path] = set()
    actual_directories: set[Path] = set()
    for path in library.rglob("*"):
        relative = path.relative_to(library)
        if path.is_symlink():
            raise LibraryError(f"refusing symlink in generated library: {path}")
        if path.is_file():
            if path.name == ".DS_Store" or path.name.startswith("._"):
                continue
            actual_files.add(relative)
        elif path.is_dir():
            actual_directories.add(relative)
        else:
            raise LibraryError(f"refusing special file in generated library: {path}")
    if actual_files != expected_files:
        extra = sorted(str(path) for path in actual_files - expected_files)
        missing = sorted(str(path) for path in expected_files - actual_files)
        raise LibraryError(
            f"generated library contains unindexed content: extra={extra[:5]} "
            f"missing={missing[:5]}"
        )
    if actual_directories != expected_directories:
        extra = sorted(str(path) for path in actual_directories - expected_directories)
        missing = sorted(
            str(path) for path in expected_directories - actual_directories
        )
        raise LibraryError(
            f"generated library directory set changed: extra={extra[:5]} "
            f"missing={missing[:5]}"
        )


def _validate_output_root(library: Path, archive: Path, inventory: Path) -> None:
    resolved = library.resolve()
    if resolved == Path(resolved.anchor) or resolved == REPO_ROOT.resolve():
        raise LibraryError(f"refusing unsafe library root: {library}")
    if _is_within(REPO_ROOT, resolved):
        raise LibraryError("library root must not contain the repository")
    if (
        _is_within(resolved, archive)
        or _is_within(archive, resolved)
        or _is_within(resolved, inventory)
        or _is_within(inventory, resolved)
    ):
        raise LibraryError("library must not overlap the archive or inventory")
    if library.is_symlink():
        raise LibraryError(f"refusing symlink library root: {library}")
    if library.exists():
        marker_path = library / MARKER_NAME
        if marker_path.is_symlink() or not marker_path.is_file():
            raise LibraryError(
                f"refusing to replace unmarked directory: {library}; expected {MARKER_NAME}"
            )
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LibraryError(
                f"invalid generated-library marker: {marker_path}"
            ) from exc
        if (
            not isinstance(marker, dict)
            or marker.get("generator") != ("scripts/literature_library.py")
            or marker.get("schema_version") != SCHEMA_VERSION
        ):
            raise LibraryError(f"unrecognized generated-library marker: {marker_path}")
        _validate_existing_generated_tree(library, marker)


def _write_index(path: Path, entries: Iterable[LibraryEntry]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in entries:
            handle.write(
                json.dumps(entry.index_row, ensure_ascii=False, sort_keys=True)
            )
            handle.write("\n")


def _clone_file(source: Path, destination: Path) -> None:
    """Create a copy-on-write clone without silently falling back to a copy."""
    if sys.platform == "darwin":
        command = ["/bin/cp", "-c", str(source), str(destination)]
    elif sys.platform.startswith("linux"):
        executable = shutil.which("cp")
        if not executable:
            raise LibraryError("copy-on-write clone requires the cp command")
        command = [executable, "--reflink=always", "--", str(source), str(destination)]
    else:
        raise LibraryError(
            f"copy-on-write clone is unsupported on {sys.platform}; "
            "use --link-mode hardlink only if write-through behavior is acceptable"
        )
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        if destination.exists():
            destination.unlink()
        raise LibraryError(f"copy-on-write clone timed out for {source}") from exc
    if completed.returncode != 0:
        if destination.exists():
            destination.unlink()
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise LibraryError(f"copy-on-write clone failed for {source}: {detail}")


def build_library(
    entries: list[LibraryEntry],
    library: Path,
    archive: Path,
    inventory: Path,
    maximum_filename_bytes: int = DEFAULT_MAX_FILENAME_BYTES,
    dry_run: bool = False,
    link_mode: str = DEFAULT_LINK_MODE,
) -> None:
    """Build and atomically install a marker-protected readable library."""
    _validate_output_root(library, archive, inventory)
    if link_mode not in {"clone", "hardlink"}:
        raise LibraryError(f"unsupported link mode: {link_mode}")
    if dry_run:
        return
    library.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{library.name}-build-", dir=str(library.parent))
    )
    backup = library.parent / f".{library.name}-backup-{os.getpid()}"
    moved_existing = False
    try:
        for entry in entries:
            destination = temporary / entry.relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            entry.index_row["link_mode"] = link_mode
            if link_mode == "clone":
                _clone_file(entry.source, destination)
            else:
                try:
                    os.link(entry.source, destination)
                except OSError as exc:
                    raise LibraryError(
                        f"hard-link creation failed for {entry.source} -> "
                        f"{destination}: {exc}"
                    ) from exc
            source_stat = entry.source.stat()
            destination_stat = destination.stat()
            same_inode = (
                source_stat.st_dev == destination_stat.st_dev
                and source_stat.st_ino == destination_stat.st_ino
            )
            if link_mode == "hardlink" and not same_inode:
                raise LibraryError(f"destination is not a hard link: {destination}")
            if link_mode == "clone" and same_inode:
                raise LibraryError(f"clone unexpectedly shares an inode: {destination}")
            if source_stat.st_size != destination_stat.st_size:
                raise LibraryError(f"destination size mismatch: {destination}")

        _write_index(temporary / INDEX_NAME, entries)
        marker = {
            "schema_version": SCHEMA_VERSION,
            "generator": "scripts/literature_library.py",
            "entries": len(entries),
            "canonical_entries": sum(
                entry.index_row["artifact_kind"] == "canonical" for entry in entries
            ),
            "version_entries": sum(
                entry.index_row["artifact_kind"] == "preserved_version"
                for entry in entries
            ),
            "link_mode": link_mode,
            "max_filename_bytes": maximum_filename_bytes,
            "library_root": str(library.resolve()),
        }
        (temporary / MARKER_NAME).write_text(
            json.dumps(marker, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if backup.exists():
            raise LibraryError(f"refusing to overwrite stale backup: {backup}")
        if library.exists():
            os.replace(library, backup)
            moved_existing = True
        try:
            os.replace(temporary, library)
        except BaseException:
            if moved_existing and not library.exists():
                os.replace(backup, library)
                moved_existing = False
            raise
        if moved_existing:
            shutil.rmtree(backup)
            moved_existing = False
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        if moved_existing and backup.exists() and not library.exists():
            os.replace(backup, library)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    build = subparsers.add_parser(
        "build", help="build or safely replace the readable library"
    )
    build.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    build.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    build.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    build.add_argument("--versions", type=Path, default=DEFAULT_VERSIONS)
    build.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    build.add_argument(
        "--max-filename-bytes", type=int, default=DEFAULT_MAX_FILENAME_BYTES
    )
    build.add_argument(
        "--link-mode", choices=("clone", "hardlink"), default=DEFAULT_LINK_MODE
    )
    build.add_argument(
        "--skip-sha-verification",
        action="store_true",
        help="skip the default manifest/ledger SHA-256 verification",
    )
    build.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "build":
        parser.print_help()
        return 2
    try:
        records = load_records(args.inventory)
        aliases = load_display_names(args.aliases, records)
        entries = plan_library(
            records,
            args.archive,
            aliases,
            args.max_filename_bytes,
            args.versions,
            not args.skip_sha_verification,
            args.link_mode,
        )
        build_library(
            entries,
            args.library,
            args.archive,
            args.inventory,
            args.max_filename_bytes,
            args.dry_run,
            args.link_mode,
        )
    except (LibraryError, OSError, ValueError) as exc:
        parser.error(str(exc))
    action = "planned" if args.dry_run else "built"
    canonical = sum(
        entry.index_row["artifact_kind"] == "canonical" for entry in entries
    )
    version_count = len(entries) - canonical
    print(
        f"{action} readable PDF library: entries={len(entries)} "
        f"canonical={canonical} versions={version_count} "
        f"curated_names={len(aliases)} root={args.library}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
