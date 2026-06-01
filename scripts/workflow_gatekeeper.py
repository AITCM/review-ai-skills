#!/usr/bin/env python3
"""Check interactive workflow gates before drafting a top-journal review.

This script is intentionally conservative. It does not decide scientific
content; it tells Codex whether it is about to draft before human checkpoints,
subagent screening, official citation export, full-text handoff, or RAG setup
have actually happened.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def count_rows(path: Path) -> int:
    return len(read_csv(path))


def glob_count(root: Path, pattern: str) -> int:
    return len(list(root.glob(pattern))) if root.exists() else 0


def first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def latest_glob(root: Path, pattern: str) -> Path | None:
    files = list(root.glob(pattern)) if root.exists() else []
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


def pool_paper_count(pool_json: Path) -> int:
    data = read_json(pool_json)
    if not isinstance(data, dict):
        return 0
    papers = data.get("papers")
    return len(papers) if isinstance(papers, dict) else 0


def candidate_board_screen_rows(board_dir: Path) -> int:
    rows = read_csv(board_dir / "candidate_lane_summary.csv")
    total = 0
    for row in rows:
        if row.get("next_action") == "screen_with_codex_subagent":
            try:
                total += int(float(row.get("rows") or 0))
            except ValueError:
                pass
    return total


def has_candidate_board_decisions(board_dir: Path) -> bool:
    if glob_count(board_dir, "agent_results/*.csv"):
        return True
    if glob_count(board_dir, "collected*/candidate_board_agent_decisions.csv"):
        return True
    if glob_count(board_dir, "collected*/candidate_board_candidates_for_verification.csv"):
        return True
    return False


def literature_discovery_packet_count(lit_dir: Path) -> int:
    return glob_count(lit_dir, "literature_discovery*/subagent_discovery_packets/*.md")


def literature_discovery_result_count(lit_dir: Path) -> int:
    patterns = [
        "literature_discovery*/agent_results/*.csv",
        "literature_discovery*/collected/*.csv",
        "literature_discovery*/collected*/*.csv",
    ]
    return sum(glob_count(lit_dir, pattern) for pattern in patterns)


def verified_count(audit_dir: Path) -> int:
    return sum(count_rows(path) for path in audit_dir.glob("**/verified_draft_papers.csv"))


def official_export_count(audit_dir: Path) -> int:
    return sum(count_rows(path) for path in audit_dir.glob("**/official_citations.csv"))


def missing_fulltext_count(pool_dir: Path) -> int:
    manifest = pool_dir / "contexts" / "missing_fulltext_manifest.csv"
    return count_rows(manifest)


def build_gate_report(project_dir: Path, stage: str) -> dict[str, Any]:
    review_data = project_dir / "review-data"
    literature_dir = review_data / "02_literature"
    framework_dir = review_data / "03_framework"
    audit_dir = review_data / "05_audit"
    agent_memory_dir = review_data / "06_agent_memory"
    output_dir = project_dir / "review-output"
    pool_dir = literature_dir / "pool"
    board_dir = literature_dir / "candidate_board"
    display_dir = framework_dir / "display_items"

    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    alignment_questions = first_existing(
        framework_dir / "logic_framework" / "user_alignment_questions.md",
        framework_dir / "user_alignment_questions.md",
    )
    alignment_decisions = first_existing(
        framework_dir / "logic_framework" / "user_alignment_decisions.md",
        framework_dir / "logic_framework" / "human_framework_decisions.md",
        agent_memory_dir / "shared" / "decisions.md",
    )
    if alignment_questions and not alignment_decisions:
        blockers.append(
            {
                "code": "missing_user_framework_confirmation",
                "message": "Framework questions exist but no user/framework decision file was found. Stop and discuss thesis, section spine, essential references, and display-item role before drafting.",
            }
        )

    packet_count = literature_discovery_packet_count(literature_dir)
    result_count = literature_discovery_result_count(literature_dir)
    if packet_count and not result_count:
        blockers.append(
            {
                "code": "subagent_packets_not_run",
                "message": f"{packet_count} literature-discovery packets exist, but no subagent/agent result CSVs were found. Do not describe the run as multi-agent; dispatch a small batch or explicitly record a local-only fallback.",
            }
        )

    screen_rows = candidate_board_screen_rows(board_dir)
    board_decisions = has_candidate_board_decisions(board_dir)
    if screen_rows and not board_decisions:
        blockers.append(
            {
                "code": "candidate_board_unscreened",
                "message": f"Candidate board still has {screen_rows} rows marked screen_with_codex_subagent, but no collected board decisions were found.",
            }
        )

    verified = verified_count(audit_dir)
    official_exports = official_export_count(audit_dir)
    if verified and not official_exports:
        blockers.append(
            {
                "code": "official_citation_export_missing",
                "message": f"{verified} verified paper rows exist, but no official_citations.csv was found. Export official citation-manager records before final references.",
            }
        )

    pool_json = pool_dir / "pool.json"
    pool_count = pool_paper_count(pool_json)
    if verified and not pool_json.exists():
        blockers.append(
            {
                "code": "verified_union_not_imported_to_pool",
                "message": "Verified papers exist but the governed literature pool has no pool.json. Run merge-verified/import before full text, cards, or RAG.",
            }
        )
    elif pool_json.exists() and not pool_count:
        warnings.append({"code": "empty_literature_pool", "message": "pool.json exists but contains no papers."})

    fulltext_audit = pool_dir / "contexts" / "fulltext_audit.md"
    if pool_count and not fulltext_audit.exists():
        blockers.append(
            {
                "code": "fulltext_handoff_missing",
                "message": "Literature pool exists, but no full-text audit/handoff was found. Run Europe PMC/OA fetch or create a user full-text handoff before detailed evidence drafting.",
            }
        )
    missing_fulltext = missing_fulltext_count(pool_dir)
    if missing_fulltext:
        blockers.append(
            {
                "code": "user_fulltext_needed",
                "message": f"{missing_fulltext} pool records still need user/full-text action. Present contexts/needs_user_fulltext.md to the user before detailed method/result/metric claims.",
            }
        )

    rag_db = pool_dir / "indexes" / "lit_rag.sqlite"
    if pool_count and not rag_db.exists():
        blockers.append(
            {
                "code": "rag_not_built",
                "message": "Literature pool exists but TreeRAG/structured RAG index is missing. Build RAG or explicitly limit drafting to abstract-level claims.",
            }
        )

    display_checkpoint = display_dir / "human_display_item_checkpoint.md"
    display_decisions = first_existing(
        display_dir / "human_display_item_decisions.md",
        display_dir / "approved_display_items.md",
        agent_memory_dir / "shared" / "display_item_decisions.md",
    )
    if display_checkpoint.exists() and not display_decisions:
        blockers.append(
            {
                "code": "display_items_not_approved",
                "message": "Display-item checkpoint exists but no approval/decision file was found. Stop before drafting around figures, tables, boxes, or GPT image generation.",
            }
        )

    manuscript_files = list(output_dir.glob("manuscript/*")) if output_dir.exists() else []
    narrative_guard_reports = list(audit_dir.glob("**/manuscript_narrative_guard.md"))
    narrative_guard_reports.extend(output_dir.glob("**/manuscript_narrative_guard.md") if output_dir.exists() else [])
    if manuscript_files and not narrative_guard_reports:
        item = {
            "code": "manuscript_narrative_guard_missing",
            "message": "Manuscript files exist, but no narrative-contamination guard report was found. Run manuscript_narrative_guard.py before final polish so workflow logs, evidence-count bookkeeping, and internal citation-pool state do not enter the article body.",
        }
        if stage in {"pre-final", "final", "submission"}:
            blockers.append(item)
        else:
            warnings.append(item)

    citation_sequence_reports = list(audit_dir.glob("**/citation_sequence_report.md"))
    citation_sequence_reports.extend(output_dir.glob("**/citation_sequence_report.md") if output_dir.exists() else [])
    if manuscript_files and not citation_sequence_reports:
        item = {
            "code": "citation_sequence_audit_missing",
            "message": "Manuscript files exist, but no citation-sequence audit was found. Run citation_sequence_manager.py before final DOCX/PDF export so numeric references follow first appearance and uncited/missing entries are removed.",
        }
        if stage in {"pre-final", "final", "submission"}:
            blockers.append(item)
        else:
            warnings.append(item)
    if manuscript_files and blockers:
        warnings.append(
            {
                "code": "draft_exists_despite_blockers",
                "message": f"{len(manuscript_files)} manuscript file(s) already exist while pre-draft blockers remain. Treat the manuscript as provisional and audit before reuse.",
            }
        )

    status = "blocked" if blockers else "pass"
    return {
        "schema_name": "top_journal_review_workflow_gate",
        "schema_version": "0.1",
        "generated_at": now_iso(),
        "project_dir": str(project_dir),
        "stage": stage,
        "status": status,
        "summary": {
            "literature_discovery_packets": packet_count,
            "literature_discovery_results": result_count,
            "candidate_board_screen_rows": screen_rows,
            "candidate_board_has_decisions": board_decisions,
            "verified_rows": verified,
            "official_citation_rows": official_exports,
            "pool_papers": pool_count,
            "missing_fulltext_rows": missing_fulltext,
            "rag_db_exists": rag_db.exists(),
            "display_checkpoint_exists": display_checkpoint.exists(),
            "display_decisions_exist": bool(display_decisions),
            "narrative_guard_reports": len(narrative_guard_reports),
            "citation_sequence_reports": len(citation_sequence_reports),
        },
        "blockers": blockers,
        "warnings": warnings,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Workflow Gate Report",
        "",
        f"- Project: `{report['project_dir']}`",
        f"- Stage: `{report['stage']}`",
        f"- Status: `{report['status']}`",
        f"- Generated at: {report['generated_at']}",
        "",
        "## Summary",
        "",
    ]
    for key, value in report.get("summary", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Blockers", ""])
    if report.get("blockers"):
        for item in report["blockers"]:
            lines.append(f"- `{item['code']}`: {item['message']}")
    else:
        lines.append("- [none]")
    lines.extend(["", "## Warnings", ""])
    if report.get("warnings"):
        for item in report["warnings"]:
            lines.append(f"- `{item['code']}`: {item['message']}")
    else:
        lines.append("- [none]")
    lines.extend(
        [
            "",
            "## Required Codex Behavior",
            "",
            "- If status is `blocked`, stop manuscript drafting and present this report to the user.",
            "- Ask the user to approve the framework, screen/repair literature lanes, provide missing full text, or approve display items as needed.",
            "- Resume drafting only after the blocking checkpoint is resolved or the user explicitly authorizes an abstract-only/provisional draft.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check human/subagent/fulltext/RAG gates before review drafting.")
    parser.add_argument("command", choices=["check"], nargs="?", default="check")
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--stage", default="pre-draft")
    parser.add_argument("--out", default="")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--fail-on-blocker", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_dir = Path(args.project_dir).resolve()
    report = build_gate_report(project_dir, args.stage)
    out = Path(args.out) if args.out else project_dir / "review-data" / "05_audit" / "workflow_gate" / f"{args.stage}_workflow_gate.md"
    write_markdown(out, report)
    if args.json_out:
        json_out = Path(args.json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "status": report["status"], "blockers": len(report["blockers"]), "warnings": len(report["warnings"])}, ensure_ascii=False))
    if args.fail_on_blocker and report["blockers"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
