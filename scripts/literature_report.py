#!/usr/bin/env python3
"""Generate a machine coverage ledger and a browsable corpus bibliography."""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from literature_corpus import (
    DEFAULT_ARCHIVE,
    DEFAULT_INVENTORY,
    REPO_ROOT,
    _artifact_path,
    _normalize_title,
    _resolved_pdf_url,
    load_records,
)


DEFAULT_CANDIDATES = REPO_ROOT / "research" / "literature" / "candidates"
DEFAULT_LEDGER = REPO_ROOT / "research" / "literature" / "coverage.json"
DEFAULT_COVERAGE_DOC = REPO_ROOT / "docs" / "frontier-labs" / "coverage.md"
DEFAULT_BIBLIOGRAPHY_DOC = REPO_ROOT / "docs" / "frontier-labs" / "bibliography.md"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            rows.append(row)
    return rows


def _index(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row.get("org", "")).casefold(), str(row.get("id", ""))): row
        for row in rows
        if row.get("org") and row.get("id")
    }


def _source_snapshot(candidates: Path) -> dict[str, Any]:
    official = _read_jsonl(candidates / "official-pages.jsonl")
    evidence = _read_jsonl(candidates / "page-evidence.jsonl")
    inventory_evidence = _read_jsonl(candidates / "inventory-page-evidence.jsonl")
    rss = _read_jsonl(candidates / "openai-rss.jsonl")
    versions = _read_jsonl(candidates / "document-versions.jsonl")
    equivalences = _read_jsonl(candidates / "document-equivalences.jsonl")
    recovery_ledgers = {
        "open_access": _read_jsonl(candidates / "oa-recovery.jsonl"),
        "arxiv_title_match": _read_jsonl(candidates / "secondary-recovery.jsonl"),
        "manual_review": _read_jsonl(candidates / "manual-recovery.jsonl"),
        "pmc": _read_jsonl(candidates / "pmc-recovery.jsonl"),
    }

    official_by_org: dict[str, int] = Counter(str(row.get("org", "unknown")) for row in official)
    evidence_by_org: dict[str, Counter[str]] = defaultdict(Counter)
    for row in evidence:
        evidence_by_org[str(row.get("org", "unknown"))][str(row.get("status", "unknown"))] += 1
    inventory_by_org: dict[str, Counter[str]] = defaultdict(Counter)
    for row in inventory_evidence:
        inventory_by_org[str(row.get("org", "unknown"))][
            str(row.get("status", "unknown"))
        ] += 1
    rss_dates = sorted(str(row["published_at"]) for row in rss if row.get("published_at"))
    rss_categories = Counter(str(row.get("category", "unknown")) for row in rss)

    orgs = sorted(set(official_by_org) | set(evidence_by_org), key=str.casefold)
    return {
        "official_pages_total": len(official),
        "official_pages_by_org": dict(sorted(official_by_org.items())),
        "official_page_evidence_by_org": {
            org: dict(sorted(evidence_by_org[org].items())) for org in orgs
        },
        "inventory_page_evidence_total": len(inventory_evidence),
        "inventory_page_evidence_by_org": {
            org: dict(sorted(counts.items()))
            for org, counts in sorted(inventory_by_org.items())
        },
        "openai_rss": {
            "items": len(rss),
            "earliest": rss_dates[0] if rss_dates else None,
            "latest": rss_dates[-1] if rss_dates else None,
            "categories": dict(sorted(rss_categories.items())),
        },
        "document_versions": {
            "rows": len(versions),
            "canonical_records": len(
                {
                    (str(row.get("org", "")).casefold(), str(row.get("canonical_id", "")))
                    for row in versions
                    if row.get("org") and row.get("canonical_id")
                }
            ),
            "relations": dict(
                sorted(Counter(str(row.get("relation", "unknown")) for row in versions).items())
            ),
        },
        "document_equivalences": {
            "rows": len(equivalences),
            "published_preprint_pairs": sum(
                row.get("relation") == "published_version_of_preprint"
                for row in equivalences
            ),
            "official_page_preprint_pairs": sum(
                row.get("relation") == "official_page_for_preprint"
                for row in equivalences
            ),
            "relations": dict(
                sorted(
                    Counter(
                        str(row.get("relation", "unknown")) for row in equivalences
                    ).items()
                )
            ),
        },
        "recovery_ledgers": {
            name: {
                "rows": len(rows),
                "recovered": sum(row.get("status") == "recovered" for row in rows),
                "statuses": dict(
                    sorted(
                        Counter(
                            str(row.get("status", "unknown")) for row in rows
                        ).items()
                    )
                ),
            }
            for name, rows in recovery_ledgers.items()
        },
    }


def _shared_identifiers(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    identifiers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("doi"):
            identifiers[f"doi:{record['doi'].casefold()}"].append(record)
        if record.get("arxiv_id"):
            identifiers[f"arxiv:{record['arxiv_id'].casefold()}"].append(record)

    rows = []
    for identifier, matches in identifiers.items():
        orgs = sorted({record["org"] for record in matches}, key=str.casefold)
        if len(orgs) < 2:
            continue
        rows.append(
            {
                "identifier": identifier,
                "title": matches[0]["title"],
                "organizations": orgs,
                "records": [
                    {"org": record["org"], "id": record["id"]}
                    for record in sorted(matches, key=lambda item: (item["org"], item["id"]))
                ],
            }
        )
    return sorted(rows, key=lambda row: (row["title"].casefold(), row["identifier"]))


def build_ledger(
    records: list[dict[str, Any]], archive: Path, candidates: Path, as_of: str
) -> dict[str, Any]:
    download_rows = _read_jsonl(archive / "manifest.jsonl")
    extraction_rows = _read_jsonl(archive / "extraction-manifest.jsonl")
    download_index = _index(download_rows)
    extraction_index = _index(extraction_rows)
    page_rows = _read_jsonl(candidates / "inventory-page-evidence.jsonl")
    page_index = {
        (str(row.get("org", "")).casefold(), str(row.get("url", ""))): row
        for row in page_rows
    }

    by_org: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_org[record["org"]].append(record)

    organizations: dict[str, Any] = {}
    artifact_gaps: list[dict[str, Any]] = []
    pending_downloads: list[dict[str, Any]] = []
    failed_downloads: list[dict[str, Any]] = []
    text_quality_flags: list[dict[str, Any]] = []
    unicode_replacements = 0
    manifest_orphans = set(download_index)
    extraction_orphans = set(extraction_index)

    for org in sorted(by_org, key=str.casefold):
        org_records = by_org[org]
        tiers = Counter(record["tier"] for record in org_records)
        types = Counter(record["type"] for record in org_records)
        years = Counter(record["date"][:4] if record["date"] else "unknown" for record in org_records)
        public_pdf = local_pdf = extracted = html_backstop = 0

        for record in org_records:
            key = (org.casefold(), record["id"])
            manifest_orphans.discard(key)
            extraction_orphans.discard(key)
            resolved_pdf = _resolved_pdf_url(record)
            pdf_path = _artifact_path(archive, record, "pdf")
            text_path = _artifact_path(archive, record, "txt")
            download = download_index.get(key)
            extraction = extraction_index.get(key)
            if resolved_pdf:
                public_pdf += 1
            if pdf_path.exists():
                local_pdf += 1
            if text_path.exists():
                extracted += 1
                text = text_path.read_text(encoding="utf-8")
                replacement_count = text.count("\ufffd")
                unicode_replacements += replacement_count
                if len(text.strip()) < 500 or replacement_count:
                    text_quality_flags.append(
                        {
                            "org": org,
                            "id": record["id"],
                            "title": record["title"],
                            "characters": len(text),
                            "unicode_replacement_characters": replacement_count,
                            "reason": (
                                "very_short_extraction"
                                if len(text.strip()) < 500
                                else "embedded_font_decode_replacements"
                            ),
                            "engine": extraction.get("engine") if extraction else None,
                            "fallback_attempted": (
                                extraction.get("fallback_attempted") if extraction else None
                            ),
                            "fallback_error": (
                                extraction.get("fallback_error") if extraction else None
                            ),
                        }
                    )

            page = page_index.get((org.casefold(), str(record.get("primary_url") or "")))
            if (
                not resolved_pdf
                and page
                and page.get("status") in {"downloaded", "existing"}
            ):
                html_backstop += 1

            common = {
                "org": org,
                "id": record["id"],
                "title": record["title"],
                "tier": record["tier"],
                "type": record["type"],
                "primary_url": record.get("primary_url"),
                "pdf_url": resolved_pdf,
            }
            if not resolved_pdf:
                artifact_gaps.append(
                    {
                        **common,
                        "page_snapshot_status": page.get("status") if page else None,
                        "page_snapshot_error": page.get("error") if page else None,
                        "notes": record.get("notes"),
                    }
                )
            elif not pdf_path.exists():
                pending_downloads.append(
                    {
                        **common,
                        "manifest_status": download.get("status") if download else None,
                        "manifest_error": download.get("error") if download else None,
                    }
                )
            if download and download.get("status") == "failed":
                failed_downloads.append({**common, "error": download.get("error")})

        organizations[org] = {
            "records": len(org_records),
            "tiers": dict(sorted(tiers.items())),
            "types": dict(sorted(types.items())),
            "years": dict(sorted(years.items())),
            "public_pdf_or_arxiv": public_pdf,
            "local_pdf": local_pdf,
            "extracted_text": extracted,
            "archived_primary_html_for_no_pdf_records": html_backstop,
        }

    duplicate_titles: list[dict[str, Any]] = []
    titles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        titles[_normalize_title(record["title"])].append(record)
    for matches in titles.values():
        orgs = sorted({record["org"] for record in matches}, key=str.casefold)
        if len(orgs) > 1:
            doi_counts = Counter(
                record["doi"].casefold() for record in matches if record.get("doi")
            )
            arxiv_counts = Counter(
                record["arxiv_id"].casefold()
                for record in matches
                if record.get("arxiv_id")
            )
            duplicate_titles.append(
                {
                    "title": matches[0]["title"],
                    "organizations": orgs,
                    "has_shared_identifier": any(value > 1 for value in doi_counts.values())
                    or any(value > 1 for value in arxiv_counts.values()),
                }
            )

    return {
        "schema_version": 1,
        "as_of": as_of,
        "scope": (
            "Auditable public-source universe defined in research/literature/README.md; "
            "not a claim about unpublished, deleted, private, or affiliation-omitting work."
        ),
        "totals": {
            "records": len(records),
            "organizations": len(organizations),
            "public_pdf_or_arxiv": sum(
                row["public_pdf_or_arxiv"] for row in organizations.values()
            ),
            "local_pdf": sum(row["local_pdf"] for row in organizations.values()),
            "extracted_text": sum(row["extracted_text"] for row in organizations.values()),
            "records_without_public_pdf": len(artifact_gaps),
            "public_pdfs_not_local": len(pending_downloads),
            "failed_download_manifest_rows": len(failed_downloads),
            "unicode_replacement_characters_in_extracted_text": unicode_replacements,
            "text_files_with_quality_flags": len(text_quality_flags),
        },
        "organizations": organizations,
        "source_snapshot": _source_snapshot(candidates),
        "artifact_gaps": sorted(
            artifact_gaps, key=lambda row: (row["org"].casefold(), row["title"].casefold())
        ),
        "pending_downloads": sorted(
            pending_downloads, key=lambda row: (row["org"].casefold(), row["title"].casefold())
        ),
        "failed_downloads": sorted(
            failed_downloads, key=lambda row: (row["org"].casefold(), row["title"].casefold())
        ),
        "text_quality_flags": sorted(
            text_quality_flags,
            key=lambda row: (row["org"].casefold(), row["title"].casefold()),
        ),
        "cross_organization_shared_identifiers": _shared_identifiers(records),
        "cross_organization_normalized_title_matches": sorted(
            duplicate_titles, key=lambda row: row["title"].casefold()
        ),
        "orphan_download_manifest_rows": [
            {"org": org, "id": record_id} for org, record_id in sorted(manifest_orphans)
        ],
        "orphan_extraction_manifest_rows": [
            {"org": org, "id": record_id} for org, record_id in sorted(extraction_orphans)
        ],
    }


def _escape(text: Any) -> str:
    """Render external metadata as inert Markdown without changing its meaning.

    Titles, author names, topics, and notes come from heterogeneous upstream
    metadata.  Some contain literal ``\\n`` line-break artifacts, LaTeX, dollar
    delimiters, or Markdown punctuation.  Numeric character references keep the
    visible text intact while preventing those bytes from becoming site syntax.
    The inventory and machine-readable ledger retain their original strings.
    """

    value = re.sub(r"\\[nr](?=\s|$)\s*", " ", str(text))
    value = re.sub(r"[\r\n\t]+\s*", " ", value)
    value = html.escape(value, quote=False)
    for character, entity in (
        ("\\", "&#92;"),
        ("$", "&#36;"),
        ("|", "&#124;"),
        ("`", "&#96;"),
        ("*", "&#42;"),
        ("_", "&#95;"),
        ("^", "&#94;"),
        ("[", "&#91;"),
        ("]", "&#93;"),
        ("(", "&#40;"),
        (")", "&#41;"),
        ("{", "&#123;"),
        ("}", "&#125;"),
        ("~", "&#126;"),
    ):
        value = value.replace(character, entity)
    return value


def _link(label: str, url: str | None) -> str:
    return f"[{_escape(label)}]({url})" if url else _escape(label)


def render_coverage(ledger: dict[str, Any]) -> str:
    lines = [
        "# 前沿实验室公开文献覆盖率账本",
        "",
        f"> 审计日期：**{ledger['as_of']}**。这里的“全量”只指仓库方法定义的可审计公开来源并集；"
        "它不声称覆盖未公开、已删除、私有或论文未写明机构归属的工作。",
        "",
        "## 汇总",
        "",
        "| 机构 | 记录 | 核心 | 直接相关 | 机构署名 | 有公开 PDF/arXiv | 已落盘 PDF | 已抽取全文 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for org, row in ledger["organizations"].items():
        lines.append(
            f"| {_escape(org)} | {row['records']} | {row['tiers'].get('core', 0)} | "
            f"{row['tiers'].get('direct', 0)} | {row['tiers'].get('affiliated', 0)} | "
            f"{row['public_pdf_or_arxiv']} | {row['local_pdf']} | {row['extracted_text']} |"
        )

    totals = ledger["totals"]
    lines += [
        "",
        f"总计 **{totals['records']}** 条；{totals['public_pdf_or_arxiv']} 条有公开 PDF 或 arXiv，"
        f"{totals['local_pdf']} 份已落盘，{totals['extracted_text']} 份已抽取为可检索文本。",
        "",
        "## 官方来源快照",
        "",
        "| 机构 | 官方候选页 | 页面归档状态 |",
        "|---|---:|---|",
    ]
    snapshot = ledger["source_snapshot"]
    orgs = sorted(
        set(snapshot["official_pages_by_org"])
        | set(snapshot["official_page_evidence_by_org"]),
        key=str.casefold,
    )
    for org in orgs:
        statuses = snapshot["official_page_evidence_by_org"].get(org, {})
        status_text = ", ".join(f"{key}={value}" for key, value in sorted(statuses.items())) or "—"
        lines.append(
            f"| {_escape(org)} | {snapshot['official_pages_by_org'].get(org, 0)} | "
            f"{_escape(status_text)} |"
        )
    rss = snapshot["openai_rss"]
    recovery = snapshot["recovery_ledgers"]
    recovered_total = sum(row["recovered"] for row in recovery.values())
    recovery_rows = sum(row["rows"] for row in recovery.values())
    equivalences = snapshot["document_equivalences"]
    lines += [
        "",
        f"OpenAI 官方 RSS 快照含 **{rss['items']}** 条（{rss['earliest'] or '未知'} 至 "
        f"{rss['latest'] or '未知'}）；它用于补偿公开站点对自动化读取的访问限制，"
        "但 RSS 元数据不等同于正文归档。",
        "",
        f"四类恢复账本共记录 **{recovery_rows}** 次可复核身份检查，"
        f"其中 **{recovered_total}** 个经来源身份与 PDF 结构双重校验后恢复。"
        f"官方报告版本账本另存 **{snapshot['document_versions']['rows']}** 个 URL/字节版本，"
        f"归属于 **{snapshot['document_versions']['canonical_records']}** 个规范记录。",
        "",
        f"库存总数是可独立引用的**书目记录数**，不是去重后的概念论文数。"
        f"版本关系账本已明确连接 **{equivalences['published_preprint_pairs']}** 组"
        "同机构的出版版/预印本记录，以及 "
        f"**{equivalences['official_page_preprint_pairs']}** 组官方发布页/论文记录；"
        "另行保留撤稿但可确认身份的版本关系，"
        "且不会把没有公开 PDF 的撤稿版本伪装成已下载正文。",
        "",
        "## 未提供公开 PDF 的已知记录",
        "",
    ]
    gaps = ledger["artifact_gaps"]
    if not gaps:
        lines.append("当前清单没有此类记录。")
    else:
        lines += [
            "这些记录仍保留在清单中；若第一方 HTML 可访问，则另存网页快照。",
            "",
            "| 机构 | 层级 | 类型 | 记录 | HTML 快照 | 备注 |",
            "|---|---|---|---|---|---|",
        ]
        for row in gaps:
            lines.append(
                f"| {_escape(row['org'])} | {row['tier']} | {row['type']} | "
                f"{_link(row['title'], row.get('primary_url'))} | "
                f"{_escape(row.get('page_snapshot_status') or '未归档')} | "
                f"{_escape(row.get('notes') or row.get('page_snapshot_error') or '—')} |"
            )

    lines += ["", "## 有公开地址但尚未落盘", ""]
    pending = ledger["pending_downloads"]
    if not pending:
        lines.append("当前没有。")
    else:
        lines += [
            "| 机构 | 记录 | 下载状态 | 错误 |",
            "|---|---|---|---|",
        ]
        for row in pending:
            lines.append(
                f"| {_escape(row['org'])} | {_link(row['title'], row.get('pdf_url'))} | "
                f"{_escape(row.get('manifest_status') or '未尝试')} | "
                f"{_escape(row.get('manifest_error') or '—')} |"
            )

    shared = ledger["cross_organization_shared_identifiers"]
    quality = ledger["text_quality_flags"]
    lines += ["", "## 全文抽取质量", ""]
    if not quality:
        lines.append("所有已抽取文本均超过最低长度，且没有 Unicode 替换字符。")
    else:
        lines += [
            "PDF 是权威原文；以下可检索文本因嵌入字体映射或异常短输出需要人工留意。",
            "",
            "`--repair-quality` 已对这些文件比较 pypdf 与 Poppler；若两者在同一字体位置"
            "产生相同替换，则保留主抽取并以原 PDF 为最终依据。",
            "",
            "| 机构 | 记录 | 字符数 | Unicode 替换字符 | 引擎 | 后备已比较 | 标记 |",
            "|---|---|---:|---:|---|---|---|",
        ]
        for row in quality:
            lines.append(
                f"| {_escape(row['org'])} | {_escape(row['title'])} | {row['characters']} | "
                f"{row['unicode_replacement_characters']} | "
                f"{_escape(row.get('engine') or '未记录')} | "
                f"{'是' if row.get('fallback_attempted') else '否'} | `{row['reason']}` |"
            )

    lines += ["", "## 跨机构共同论文", ""]
    if not shared:
        lines.append("当前没有由同一 DOI/arXiv 标识符确认的跨机构重复记录。")
    else:
        lines += [
            "同一工作可因共同作者归属进入多个机构清单；这里明确列出，避免把它误算成不同论文。",
            "",
            "| 标识符 | 论文 | 机构 |",
            "|---|---|---|",
        ]
        for row in shared:
            lines.append(
                f"| `{row['identifier']}` | {_escape(row['title'])} | "
                f"{_escape(', '.join(row['organizations']))} |"
            )

    lines += [
        "",
        "## 复现",
        "",
        "```bash",
        "python3 scripts/literature_corpus.py validate",
        "python3 scripts/literature_corpus.py audit",
        "python3 scripts/literature_corpus.py extract --repair-quality --workers 6",
        "python3 scripts/literature_report.py",
        "```",
        "",
        "机器可读的完整缺口、失败原因、孤儿 manifest 行和来源快照统计位于 "
        "`research/literature/coverage.json`。",
        "",
    ]
    return "\n".join(lines)


def _short_authors(authors: list[str]) -> str:
    if len(authors) <= 6:
        return ", ".join(authors)
    return ", ".join(authors[:6]) + ", et al."


def render_bibliography(records: list[dict[str, Any]], as_of: str) -> str:
    by_org: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_org[record["org"]].append(record)
    lines = [
        "# 前沿实验室全量可审计书目",
        "",
        f"> 截止 **{as_of}**；共 {len(records)} 条。层级和纳入边界见"
        "[覆盖率账本](coverage.md)与 `research/literature/README.md`。",
        "",
        "符号：`core` 为模型/系统主报告，`direct` 为直接支撑或分析，`affiliated` 为论文署名机构研究。",
        "",
    ]
    for org in sorted(by_org, key=str.casefold):
        org_records = by_org[org]
        lines += [f"## {org}（{len(org_records)}）", ""]
        by_year: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in org_records:
            by_year[record["date"][:4] if record["date"] else "日期未知"].append(record)
        for year in sorted(by_year, reverse=True):
            lines += [f"### {year}", ""]
            for record in sorted(by_year[year], key=lambda item: item["title"].casefold()):
                primary = record.get("primary_url") or record.get("pdf_url")
                identifiers = []
                if record.get("arxiv_id"):
                    identifiers.append(f"arXiv:{record['arxiv_id']}")
                if record.get("doi"):
                    identifiers.append(f"DOI:{record['doi']}")
                suffix = f"; {'; '.join(identifiers)}" if identifiers else ""
                pdf = f" · [PDF]({record['pdf_url']})" if record.get("pdf_url") else ""
                topics = ", ".join(record.get("topics") or []) or "未标注"
                lines += [
                    f"- **{_link(record['title'], primary)}**{pdf}",
                    f"  - {_escape(_short_authors(record['authors']))}; "
                    f"`{record['tier']}` / `{record['type']}`{_escape(suffix)}",
                    f"  - 主题：{_escape(topics)}",
                ]
            lines.append("")
    return "\n".join(lines)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--coverage-doc", type=Path, default=DEFAULT_COVERAGE_DOC)
    parser.add_argument("--bibliography-doc", type=Path, default=DEFAULT_BIBLIOGRAPHY_DOC)
    parser.add_argument("--as-of", default=dt.datetime.now(dt.timezone.utc).date().isoformat())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.as_of):
        raise SystemExit("--as-of must be YYYY-MM-DD")
    records = load_records(args.inventory)
    ledger = build_ledger(records, args.archive, args.candidates, args.as_of)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    args.ledger.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_text(args.coverage_doc, render_coverage(ledger))
    _write_text(args.bibliography_doc, render_bibliography(records, args.as_of))
    print(
        f"wrote {args.ledger}, {args.coverage_doc}, and {args.bibliography_doc} "
        f"for {len(records)} records"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
