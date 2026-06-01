#!/usr/bin/env python3
"""Audit manuscript prose for internal-workflow contamination.

Top-journal review prose should synthesize the field's past, present, and
future. It should not expose internal review-engine artifacts such as
candidate-board counts, final-gate bookkeeping, RAG state, or citation-pool
health inside the article body. This guard flags those leaks before polish.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REF_HEADING_RE = re.compile(
    r"(?im)^(?P<prefix>\s{0,3}#{1,6}\s*)?"
    r"(?P<title>references|bibliography|works cited|参考文献|參考文獻|文献|引用文献)\s*$"
)
HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")

INTERNAL_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    ("internal_workflow_artifact", "Internal workflow artifacts should not appear in article body.", re.compile(r"\b(final[- ]?gate|published-only final gate|citation[- ]?gate|workflow gate|candidate board|candidate pool|candidate lane|screening packet|subagent packet|agent packet|verified union|official_citations|citation_sequence|TreeRAG|RAG index|literature pool|citation pool)\b", re.I)),
    ("internal_workflow_artifact_zh", "Internal workflow artifacts should not appear in article body.", re.compile(r"(文献池|引文池|候选池|候选文献池|候选队列|筛选队列|候选板|候选看板|工作流门|流程门|最终门|引文门|正式发表门|官方引文导出|引文序列|全文移交|人工补全文|智能体包|子智能体包|记忆卡片|文献卡片|证据包|检索包|RAG|TreeRAG)")),
    ("audit_count_leak", "Audit counts belong in methods/supplement, not narrative synthesis.", re.compile(r"\b(recalled|screened|verified|accepted|rejected|demoted|downgraded)\s+\d+\s+(records|references|papers|rows|items)\b", re.I)),
    ("audit_count_leak_zh", "Audit counts belong in methods/supplement, not narrative synthesis.", re.compile(r"(召回|筛选|核验|验证|接受|纳入|排除|降级|删除|剔除).{0,12}\d+\s*(条|篇|项|个|行)")),
    ("draft_process_leak", "Draft/source-audit process should not become a field claim.", re.compile(r"(草稿核验|文献核验|自动核验|核验显示|审计显示|质量门|源审计|引用审计|当前文献池|本轮流程|技能运行|本脚本|本工作流|检索日志)")),
    ("metadata_date_leak", "Internal retrieval dates rarely belong in abstract/body unless reporting systematic methods.", re.compile(r"(截至\s*\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日|as of\s+[A-Z][a-z]+\s+\d{1,2},\s+\d{4}|retrieved on\s+\d{4})", re.I)),
    ("placeholder_leak", "Placeholder text must be resolved before manuscript polish.", re.compile(r"(\[CITATION NEEDED|\[VERIFY REFERENCE|\[EVIDENCE GAP|\[TODO|\[待补|待核验|需要用户确认)")),
]

EVIDENCE_NEGATIVITY_RE = re.compile(r"(证据(仍然|依然|总体|普遍)?(不足|薄弱|有限|不成熟|不好)|evidence\s+(is\s+)?(weak|limited|immature|insufficient))", re.I)
PAST_RE = re.compile(r"(过去|早期|历史|源于|奠定|基础|传统|自.+以来|foundational|historical|early|origin|from .* to)", re.I)
PRESENT_RE = re.compile(r"(当前|目前|近年来|近年|现有|正在|已经|now|current|recent|emerging|has begun|is moving)", re.I)
FUTURE_RE = re.compile(r"(未来|下一步|展望|路线图|仍需|应当|需要建立|future|agenda|roadmap|next step|should|must)", re.I)
METHODS_HEADING_RE = re.compile(r"(methods?|search strategy|literature search|supplement|方法|检索策略|资料来源|附录|补充)", re.I)
BAD_ABSTRACT_RE = re.compile(r"(final[- ]?gate|文献池|召回|筛选|核验|降级|候选|RAG|TreeRAG|citation[- ]?gate|candidate board)", re.I)


@dataclass
class Finding:
    code: str
    severity: str
    line: int
    section: str
    snippet: str
    message: str
    action: str


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def clean_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def split_body_references(text: str) -> tuple[str, str]:
    matches = list(REF_HEADING_RE.finditer(text))
    if not matches:
        return text, ""
    match = matches[-1]
    return text[: match.start()].rstrip(), text[match.start() :].strip()


def section_for_line(lines: list[str], line_index: int) -> str:
    section = "front_matter"
    for idx in range(0, min(line_index, len(lines))):
        match = HEADING_RE.match(lines[idx])
        if match:
            section = clean_space(match.group(2))
    return section


def is_methods_section(section: str) -> bool:
    return bool(METHODS_HEADING_RE.search(section or ""))


def first_abstract_block(lines: list[str]) -> tuple[int, int, str]:
    start = -1
    for idx, line in enumerate(lines):
        text = clean_space(re.sub(r"^#+\s*", "", line))
        if text.lower() == "abstract" or text in {"摘要", "摘要："}:
            start = idx + 1
            break
    if start < 0:
        return 0, 0, ""
    end = len(lines)
    for idx in range(start, len(lines)):
        if HEADING_RE.match(lines[idx]) and clean_space(re.sub(r"^#+\s*", "", lines[idx])).lower() not in {"abstract", "摘要"}:
            end = idx
            break
    return start + 1, end, "\n".join(lines[start:end])


def audit_body(text: str) -> tuple[list[Finding], dict[str, Any]]:
    body, _references = split_body_references(text)
    lines = body.splitlines()
    findings: list[Finding] = []
    evidence_negative_count = 0
    past_hits = present_hits = future_hits = 0

    for idx, line in enumerate(lines, 1):
        stripped = clean_space(line)
        if not stripped or stripped.startswith("|"):
            continue
        section = section_for_line(lines, idx)
        if PAST_RE.search(stripped):
            past_hits += 1
        if PRESENT_RE.search(stripped):
            present_hits += 1
        if FUTURE_RE.search(stripped):
            future_hits += 1
        if EVIDENCE_NEGATIVITY_RE.search(stripped):
            evidence_negative_count += 1
        for code, message, pattern in INTERNAL_PATTERNS:
            if not pattern.search(stripped):
                continue
            severity = "warning" if is_methods_section(section) else "blocker"
            action = "Move this material to methods/supplement, or translate it into field-level synthesis without internal workflow terms."
            findings.append(
                Finding(
                    code=code,
                    severity=severity,
                    line=idx,
                    section=section,
                    snippet=stripped[:300],
                    message=message,
                    action=action,
                )
            )

    abstract_start, _abstract_end, abstract_text = first_abstract_block(lines)
    if abstract_text and BAD_ABSTRACT_RE.search(abstract_text):
        findings.append(
            Finding(
                code="abstract_workflow_log",
                severity="blocker",
                line=abstract_start,
                section="Abstract",
                snippet=clean_space(abstract_text)[:300],
                message="The abstract reads like a workflow/audit log rather than a reader-facing review abstract.",
                action="Rewrite the abstract around field problem, organizing lens, synthesis pattern, boundary correction, and future agenda.",
            )
        )

    if evidence_negative_count >= 5:
        findings.append(
            Finding(
                code="overweighted_evidence_negativity",
                severity="warning",
                line=0,
                section="whole_manuscript",
                snippet=f"evidence-negativity hits: {evidence_negative_count}",
                message="The manuscript repeatedly says evidence is weak/insufficient. Top-journal prose should convert this into a structured maturity framework and future agenda.",
                action="Replace repeated evidence-quality complaints with a taxonomy, evidence ladder, limitations paragraph, or roadmap.",
            )
        )

    if not past_hits or not present_hits or not future_hits:
        missing = []
        if not past_hits:
            missing.append("past/foundations")
        if not present_hits:
            missing.append("current state")
        if not future_hits:
            missing.append("future agenda")
        findings.append(
            Finding(
                code="past_present_future_arc_incomplete",
                severity="warning",
                line=0,
                section="whole_manuscript",
                snippet=", ".join(missing),
                message="The manuscript may not yet show a complete past-present-future review arc.",
                action="Add or strengthen a field genealogy, current evidence synthesis, and forward roadmap before polishing.",
            )
        )

    summary = {
        "body_lines": len(lines),
        "findings": len(findings),
        "blockers": sum(1 for item in findings if item.severity == "blocker"),
        "warnings": sum(1 for item in findings if item.severity == "warning"),
        "evidence_negativity_hits": evidence_negative_count,
        "past_hits": past_hits,
        "present_hits": present_hits,
        "future_hits": future_hits,
    }
    return findings, summary


def write_csv(path: Path, findings: list[Finding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["code", "severity", "line", "section", "snippet", "message", "action"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in findings:
            writer.writerow({field: getattr(item, field) for field in fields})


def write_report(path: Path, manuscript: Path, findings: list[Finding], summary: dict[str, Any]) -> None:
    lines = [
        "# Manuscript Narrative Guard",
        "",
        f"- Manuscript: `{manuscript}`",
        f"- Generated: {now_iso()}",
        f"- Status: {'blocked' if summary['blockers'] else 'pass_with_warnings' if summary['warnings'] else 'pass'}",
        f"- Blockers: {summary['blockers']}",
        f"- Warnings: {summary['warnings']}",
        f"- Past hits: {summary['past_hits']}",
        f"- Present hits: {summary['present_hits']}",
        f"- Future hits: {summary['future_hits']}",
        "",
        "## Blocking Findings",
        "",
    ]
    blockers = [item for item in findings if item.severity == "blocker"]
    if blockers:
        for item in blockers:
            lines.extend(
                [
                    f"- `{item.code}` at line {item.line}, section `{item.section}`",
                    f"  - Snippet: {item.snippet}",
                    f"  - Action: {item.action}",
                ]
            )
    else:
        lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    warnings = [item for item in findings if item.severity == "warning"]
    if warnings:
        for item in warnings:
            lines.extend(
                [
                    f"- `{item.code}` at line {item.line}, section `{item.section}`",
                    f"  - Snippet: {item.snippet}",
                    f"  - Action: {item.action}",
                ]
            )
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Rewrite Rule",
            "",
            "Internal review-engine state belongs in audit reports, methods notes, or supplements. Main text should translate evidence status into field-level synthesis: genealogy, current patterns, boundary conditions, controversies, and future agenda.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def command_audit(args: argparse.Namespace) -> int:
    manuscript = Path(args.manuscript)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = manuscript.read_text(encoding="utf-8-sig", errors="replace")
    findings, summary = audit_body(text)
    summary.update(
        {
            "schema_name": "manuscript_narrative_guard",
            "schema_version": "0.1",
            "generated_at": now_iso(),
            "manuscript": str(manuscript),
            "status": "blocked" if summary["blockers"] else "pass_with_warnings" if summary["warnings"] else "pass",
        }
    )
    write_csv(out_dir / "manuscript_narrative_findings.csv", findings)
    (out_dir / "manuscript_narrative_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(out_dir / "manuscript_narrative_guard.md", manuscript, findings, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if args.fail_on_blocker and summary["blockers"] else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="Audit manuscript narrative for workflow/process contamination.")
    audit.add_argument("--manuscript", required=True)
    audit.add_argument("--out-dir", required=True)
    audit.add_argument("--fail-on-blocker", action="store_true")
    audit.set_defaults(func=command_audit)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
