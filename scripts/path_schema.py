#!/usr/bin/env python3
"""Canonical folder schema for governed review projects.

The schema keeps durable research assets separate from regenerable work
artifacts and final manuscript outputs. It is intentionally stdlib-only and
does not move existing files unless a future migration command is explicitly
implemented.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "0.5"


CANONICAL_PATHS: dict[str, dict[str, str]] = {
    "review_data": {"path": "review-data", "class": "durable", "description": "Governed persistent research data root."},
    "review_work": {"path": "review-work", "class": "working", "description": "Regenerable tool outputs, packets, logs, and scratch artifacts."},
    "review_output": {"path": "review-output", "class": "deliverable", "description": "Manuscripts, figures, tables, cover letters, and submission packages."},
    "review_archive": {"path": "review-archive", "class": "frozen", "description": "Frozen snapshots and dated releases."},
    "project_meta": {"path": "review-data/00_project", "class": "durable", "description": "Folder schema, path aliases, project ledger, and non-secret governance metadata."},
    "inputs_drafts_raw": {"path": "review-data/01_inputs/drafts_raw", "class": "raw", "description": "User-supplied GPT/Gemini/Deep Research drafts and source manuscripts."},
    "inputs_user_fulltext": {"path": "review-data/01_inputs/user_fulltext", "class": "raw", "description": "User-supplied PDFs/TXT/MD/DOCX before import into the pool."},
    "literature_draft_assets": {"path": "review-data/02_literature/draft_assets", "class": "governed", "description": "Draft-native citation assets: cleaned reference sections, in-text markers, claim-evidence clues, non-academic sources, and health reports."},
    "literature_pool": {"path": "review-data/02_literature/pool", "class": "governed", "description": "Primary literature pool: pool.json, evidence, cards, contexts, indexes, and logs."},
    "literature_supplemental_pool": {"path": "review-data/02_literature/supplemental_pool", "class": "governed", "description": "Strictly screened supplemental records added only after framework gaps are confirmed."},
    "framework_draft_memory": {"path": "review-data/03_framework/draft_memory", "class": "governed", "description": "Persistent memory cards and search index built from AI/user drafts."},
    "framework_logic": {"path": "review-data/03_framework/logic_framework", "class": "governed", "description": "Logic framework, blackboard, argument map, literature tasks, and material passport."},
    "framework_display_items": {"path": "review-data/03_framework/display_items", "class": "governed", "description": "Human-reviewed figure, table, and box blueprints linked to claims and evidence."},
    "rag_root": {"path": "review-data/04_rag", "class": "governed", "description": "RAG indexes, retrieval traces, and query results that support manuscript claims."},
    "audit_root": {"path": "review-data/05_audit", "class": "governed", "description": "Citation, claim, source, and data-lineage audit outputs."},
    "agent_memory": {"path": "review-data/06_agent_memory", "class": "governed", "description": "Persistent shared and per-agent memory for dynamic multi-agent orchestration."},
    "work_recall_runs": {"path": "review-work/recall_runs", "class": "working", "description": "Regenerable recall runs before import into the governed literature pool."},
    "work_agent_orchestration": {"path": "review-work/agent_orchestration", "class": "working", "description": "Agent packets and external-agent outputs."},
    "work_logs": {"path": "review-work/logs", "class": "working", "description": "Run logs and temporary diagnostics."},
    "output_manuscript": {"path": "review-output/manuscript", "class": "deliverable", "description": "Main manuscript drafts and clean versions."},
    "output_figures": {"path": "review-output/figures", "class": "deliverable", "description": "Figure source files and exported images."},
    "output_tables": {"path": "review-output/tables", "class": "deliverable", "description": "Tables and evidence matrices intended for manuscript/submission."},
    "output_cover_letter": {"path": "review-output/cover_letter", "class": "deliverable", "description": "Cover letter, referee suggestions, and submission notes."},
    "output_submission": {"path": "review-output/submission", "class": "deliverable", "description": "Final submission package."},
}


LEGACY_ALIASES: dict[str, str] = {
    "文献池": "review-data/02_literature/pool",
    "review-work/draft_memory": "review-data/03_framework/draft_memory",
    "review-work/draft_logic_framework": "review-data/03_framework/logic_framework",
    "review-work/literature_pool": "review-data/02_literature/pool",
    "review-work/cover_letter": "review-output/cover_letter",
}


def schema(project_dir: Path) -> dict[str, Any]:
    paths = {
        name: {
            **meta,
            "path": meta["path"],
            "absolute": str((project_dir / meta["path"]).resolve()),
            "exists": (project_dir / meta["path"]).exists(),
        }
        for name, meta in CANONICAL_PATHS.items()
    }
    aliases = {
        legacy: {
            "legacy_path": legacy,
            "legacy_absolute": str((project_dir / legacy).resolve()),
            "legacy_exists": (project_dir / legacy).exists(),
            "canonical_path": canonical,
            "canonical_absolute": str((project_dir / canonical).resolve()),
        }
        for legacy, canonical in LEGACY_ALIASES.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "project_dir": str(project_dir.resolve()),
        "paths": paths,
        "legacy_aliases": aliases,
        "policy": {
            "durable": "Stable assets that should survive across sessions and may be cited, audited, or reused.",
            "working": "Regenerable outputs; keep logs but do not treat as source-of-truth unless promoted.",
            "deliverable": "Manuscript-facing outputs.",
            "frozen": "Dated snapshots that should not be edited in place.",
        },
    }


def write_markdown(path: Path, data: dict[str, Any]) -> None:
    lines = [
        "# Review Project Folder Schema",
        "",
        f"- Schema version: {data['schema_version']}",
        f"- Project dir: `{data['project_dir']}`",
        "",
        "## Canonical Paths",
        "",
        "| Alias | Path | Class | Purpose |",
        "|---|---|---|---|",
    ]
    for name, item in data["paths"].items():
        lines.append(f"| `{name}` | `{item['path']}` | {item['class']} | {item['description']} |")
    lines.extend(["", "## Legacy Aliases", "", "| Legacy path | Canonical path | Exists |", "|---|---|---|"])
    for _name, item in data["legacy_aliases"].items():
        lines.append(f"| `{item['legacy_path']}` | `{item['canonical_path']}` | {item['legacy_exists']} |")
    lines.extend(
        [
            "",
            "## Rules",
            "",
            "- Put user/raw materials in `review-data/01_inputs` before processing.",
            "- Build draft-native citation assets in `review-data/02_literature/draft_assets` before any broad supplemental recall.",
            "- Use `review-data/02_literature/pool` as the default literature pool; `文献池` is a legacy/display alias.",
            "- Put strictly screened supplemental recall records in `review-data/02_literature/supplemental_pool` until they are promoted.",
            "- Put persistent framework memory in `review-data/03_framework`, not random run folders.",
            "- Put figure/table/box blueprints in `review-data/03_framework/display_items` before drafting around them.",
            "- Put shared and per-agent persistent memory in `review-data/06_agent_memory`; treat it as state, not verified evidence.",
            "- Use `review-work` for repeatable, disposable runs and agent packets.",
            "- Use `review-output` for manuscript-facing deliverables.",
            "- Never mix API keys or secrets into any governed data folder.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_show(args: argparse.Namespace) -> int:
    data = schema(Path(args.project_dir))
    if args.markdown:
        write_markdown(Path(args.markdown), data)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    data = schema(project_dir)
    actions = []
    for item in data["paths"].values():
        path = Path(item["absolute"])
        actions.append(("would mkdir: " if args.dry_run else "mkdir: ") + str(path))
        if not args.dry_run:
            path.mkdir(parents=True, exist_ok=True)
    config_path = project_dir / "review_path_aliases.json"
    schema_md = project_dir / "review-data" / "00_project" / "folder_schema.md"
    if args.dry_run:
        actions.append(f"would write: {config_path}")
        actions.append(f"would write: {schema_md}")
    else:
        config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        write_markdown(schema_md, data)
        actions.append(f"wrote: {config_path}")
        actions.append(f"wrote: {schema_md}")
    print(json.dumps({"project_dir": str(project_dir), "dry_run": args.dry_run, "actions": actions}, ensure_ascii=False, indent=2))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    data = schema(Path(args.project_dir))
    missing = [name for name, item in data["paths"].items() if not item["exists"]]
    legacy_present = [legacy for legacy, item in data["legacy_aliases"].items() if item["legacy_exists"]]
    result = {
        "schema_version": SCHEMA_VERSION,
        "project_dir": data["project_dir"],
        "missing_canonical_aliases": missing,
        "legacy_paths_present": legacy_present,
        "needs_init": bool(missing),
        "migration_note": "Legacy paths are not moved automatically. Use them as read-only inputs or migrate manually after backup.",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect or initialize canonical review project folders.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_show = sub.add_parser("show", help="Print folder schema JSON and optionally write Markdown.")
    p_show.add_argument("--project-dir", default=".")
    p_show.add_argument("--markdown", default="")
    p_show.set_defaults(func=cmd_show)
    p_init = sub.add_parser("init", help="Create canonical folders and path alias files.")
    p_init.add_argument("--project-dir", default=".")
    p_init.add_argument("--dry-run", action="store_true")
    p_init.set_defaults(func=cmd_init)
    p_check = sub.add_parser("check", help="Check missing canonical folders and legacy path presence.")
    p_check.add_argument("--project-dir", default=".")
    p_check.set_defaults(func=cmd_check)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
