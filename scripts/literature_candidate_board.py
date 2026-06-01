#!/usr/bin/env python3
"""Build a chief-editor board across all literature candidate lanes.

The board is read-only: it summarizes draft-native assets, user PubMed seed
sets, AI-mined candidates, argument-driven recall, supplemental recall, and
verification outputs without promoting or deleting anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from supplemental_recall_screening import (
    INCLUDE_DECISIONS,
    NO_VERIFIER_DECISIONS,
    SCREENING_FIELDS,
    VERIFY_FIELDS,
    clean_space,
    normalize_doi,
    normalize_title,
    now_iso,
    read_csv,
    safe_id,
    should_send_to_verifier,
    verification_lane,
    verifier_row,
    write_csv,
    write_json,
)


BOARD_FIELDS = [
    "board_id",
    "paper_key",
    "source_file",
    "source_channel",
    "stage",
    "candidate_id",
    "task_id",
    "lane",
    "source_route",
    "title",
    "authors",
    "year",
    "journal",
    "pmid",
    "doi",
    "url",
    "publication_type",
    "screening_priority",
    "screening_score",
    "screening_decision",
    "confidence",
    "claim_ids",
    "evidence_role",
    "argument_role",
    "evidence_level",
    "evidence_strength",
    "fulltext_need",
    "citation_role",
    "verification_status",
    "verification_confidence",
    "pool_status",
    "next_action",
    "notes",
]

SUMMARY_FIELDS = ["source_channel", "stage", "rows", "unique_papers", "next_action"]
PACKET_INDEX_FIELDS = ["packet_id", "agent", "packet_file", "worklist_file", "start_index", "end_index", "items", "purpose"]
BOARD_TRIAGE_FIELDS = [
    "board_id",
    "paper_key",
    "candidate_id",
    "source_channel",
    "current_stage",
    "recommended_agent",
    "decision",
    "priority",
    "rationale",
    "claim_or_gap",
    "next_action",
    "human_question",
]
VERIFICATION_TRIAGE_FIELDS = [
    "board_id",
    "paper_key",
    "candidate_id",
    "candidate_title",
    "pmid",
    "doi",
    "verification_route",
    "priority",
    "expected_status",
    "rationale",
    "human_question",
]
READY_VERIFICATION_ROUTES = {
    "pubmed_pmid",
    "pubmed_title",
    "doi",
    "crossref_openalex",
    "title_bibliographic",
    "publisher_page",
    "conference_page",
    "arxiv_openreview",
    "openreview",
    "official_page",
}
REPAIR_ROUTES = {"repair_needed", "repair_or_replace", "identity_repair"}
REJECT_ROUTES = {"reject", "delete", "not_a_paper", "irrelevant"}
FULLTEXT_ACTIONS = {"export_and_fetch_fulltext", "fetch_fulltext", "needs_fulltext"}
COLLECTED_DECISION_FIELDS = [
    "agent_result_file",
    "agent_type",
    "decision_family",
    *BOARD_FIELDS,
    "agent_decision",
    "agent_confidence",
    "agent_priority",
    "agent_route",
    "agent_expected_status",
    "agent_claim_fit",
    "agent_evidence_role",
    "agent_argument_role",
    "agent_evidence_level",
    "agent_evidence_strength",
    "agent_fulltext_need",
    "agent_citation_role",
    "agent_key_supported_claim",
    "agent_supported_claim_ids",
    "agent_rationale",
    "agent_human_question",
    "agent_required_next_action",
]
QUEUE_FIELDS = [
    "board_id",
    "paper_key",
    "candidate_id",
    "source_channel",
    "stage",
    "title",
    "pmid",
    "doi",
    "agent_type",
    "agent_decision",
    "agent_route",
    "priority",
    "reason",
    "next_action",
    "human_question",
]


def rel(path: Path, project_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_dir.resolve()))
    except ValueError:
        return str(path)


def first(row: dict[str, Any], *fields: str) -> str:
    for field in fields:
        value = clean_space(row.get(field))
        if value:
            return value
    return ""


def paper_key(row: dict[str, Any]) -> str:
    doi = normalize_doi(first(row, "doi", "draft_doi"))
    if doi:
        return "doi:" + doi
    pmid = first(row, "pmid", "draft_pmid", "paper_id")
    if pmid:
        return "pmid:" + pmid
    title = normalize_title(first(row, "title", "candidate_title", "draft_candidate_title"))
    year = first(row, "year")
    return f"title:{title}|year:{year}"


def guess_next_action(stage: str, row: dict[str, str]) -> str:
    decision = first(row, "screening_decision", "decision", "candidate_type")
    verification_status = first(row, "verification_status")
    if stage in {"screening_queue", "candidate", "seed_locked", "quarantine"}:
        return "screen_with_codex_subagent"
    if stage in {"screened_rejected", "rejected"}:
        return "keep_for_audit"
    if stage == "background_hold":
        return "keep_background_or_revisit_after_framework"
    if stage in {"verifier_ready", "screened_accepted"}:
        return "run_draft_reference_verifier"
    if stage == "verification":
        if verification_status in {"api_verified", "official_conference_paper", "publisher_page_verified"} or verification_status.startswith("verified_"):
            return "export_official_citation_and_fetch_fulltext"
        if verification_status:
            return "repair_or_replace_reference"
        return "inspect_verification_output"
    if decision == "needs_fulltext":
        return "fetch_fulltext_or_user_handoff"
    return "inspect_lane"


def normalize_row(row: dict[str, str], source_path: Path, project_dir: Path, source_channel: str, stage: str) -> dict[str, str]:
    title = first(row, "title", "candidate_title", "draft_candidate_title")
    doi = normalize_doi(first(row, "doi", "draft_doi"))
    pmid = first(row, "pmid", "draft_pmid", "paper_id")
    key = paper_key({**row, "title": title, "doi": doi, "pmid": pmid})
    candidate_id = first(row, "candidate_id", "ref_number") or safe_id(key, "BOARD")
    return {
        "board_id": safe_id("|".join([rel(source_path, project_dir), stage, candidate_id, key]), "BRD"),
        "paper_key": key,
        "source_file": rel(source_path, project_dir),
        "source_channel": source_channel,
        "stage": stage,
        "candidate_id": candidate_id,
        "task_id": first(row, "task_id"),
        "lane": first(row, "lane", "source_kind", "draft_source_kind"),
        "source_route": first(row, "source_route", "source", "source_kind_verified"),
        "title": title,
        "authors": first(row, "authors"),
        "year": first(row, "year"),
        "journal": first(row, "journal"),
        "pmid": pmid,
        "doi": doi,
        "url": first(row, "url", "urls", "draft_urls"),
        "publication_type": first(row, "publication_type"),
        "screening_priority": first(row, "screening_priority"),
        "screening_score": first(row, "screening_score"),
        "screening_decision": first(row, "screening_decision", "decision", "candidate_type"),
        "confidence": first(row, "screening_confidence", "confidence", "verification_confidence"),
        "claim_ids": first(row, "screening_supported_claim_ids", "supported_claim_ids", "claim_id", "linked_claim_id"),
        "evidence_role": first(row, "screening_evidence_role", "evidence_role", "evidence_need"),
        "argument_role": first(row, "screening_argument_role", "argument_role"),
        "evidence_level": first(row, "screening_evidence_level", "evidence_level"),
        "evidence_strength": first(row, "screening_evidence_strength", "evidence_strength"),
        "fulltext_need": first(row, "screening_fulltext_need", "fulltext_need"),
        "citation_role": first(row, "screening_citation_role", "citation_role"),
        "verification_status": first(row, "verification_status"),
        "verification_confidence": first(row, "verification_confidence"),
        "pool_status": first(row, "pool_status"),
        "next_action": guess_next_action(stage, row),
        "notes": first(row, "screening_rationale", "rationale", "why_relevant", "candidate_fit_reason", "notes")[:900],
    }


def add_existing(paths: list[tuple[Path, str, str]], seen: set[Path], path: Path, channel: str, stage: str) -> None:
    if path.exists() and path.is_file() and path.resolve() not in seen:
        seen.add(path.resolve())
        paths.append((path, channel, stage))


def should_include_source(path: Path, include_contains: list[str], exclude_contains: list[str]) -> bool:
    text = str(path).replace("\\", "/").lower()
    includes = [item.lower().replace("\\", "/") for item in include_contains if clean_space(item)]
    excludes = [item.lower().replace("\\", "/") for item in exclude_contains if clean_space(item)]
    if includes and not any(item in text for item in includes):
        return False
    return not any(item in text for item in excludes)


def filter_candidate_sources(
    paths: list[tuple[Path, str, str]],
    include_contains: list[str] | None = None,
    exclude_contains: list[str] | None = None,
) -> list[tuple[Path, str, str]]:
    include_contains = include_contains or []
    exclude_contains = exclude_contains or []
    return [(path, channel, stage) for path, channel, stage in paths if should_include_source(path, include_contains, exclude_contains)]


def discover_candidate_sources(project_dir: Path) -> list[tuple[Path, str, str]]:
    lit = project_dir / "review-data" / "02_literature"
    audit = project_dir / "review-data" / "05_audit"
    paths: list[tuple[Path, str, str]] = []
    seen: set[Path] = set()

    add_existing(paths, seen, lit / "draft_assets" / "candidate_paper_clues.csv", "draft_native", "candidate")
    add_existing(paths, seen, lit / "draft_assets" / "uncited_references.csv", "draft_native", "uncited_or_leftover")
    add_existing(paths, seen, lit / "draft_assets" / "non_academic_sources.csv", "draft_native", "non_academic")

    for path in lit.glob("literature_discovery/**/literature_discovery_candidates.csv"):
        add_existing(paths, seen, path, "literature_discovery", "candidate")
    for path in lit.glob("literature_discovery/**/candidate_paper_clues_for_verification.csv"):
        add_existing(paths, seen, path, "literature_discovery", "verifier_ready")

    for path in lit.glob("ai_candidate_mining*/**/subagent_hidden_candidate_clues.csv"):
        add_existing(paths, seen, path, "ai_candidate_mining", "candidate")
    for path in lit.glob("ai_candidate_mining*/**/candidate_paper_clues_for_verification.csv"):
        add_existing(paths, seen, path, "ai_candidate_mining", "verifier_ready")

    for path in lit.glob("user_pubmed_sets/*/pubmed_user_set_candidates.csv"):
        add_existing(paths, seen, path, "user_pubmed_seed", "candidate")
    for path in lit.glob("user_pubmed_sets/*/screening_queue.csv"):
        add_existing(paths, seen, path, "user_pubmed_seed", "screening_queue")
    for path in lit.glob("user_pubmed_sets/*/screened/accepted_supplemental_recall_candidates.csv"):
        add_existing(paths, seen, path, "user_pubmed_seed", "screened_accepted")
    for path in lit.glob("user_pubmed_sets/*/screened/rejected_supplemental_recall_candidates.csv"):
        add_existing(paths, seen, path, "user_pubmed_seed", "screened_rejected")
    for path in lit.glob("user_pubmed_sets/*/screened/background_hold_candidates.csv"):
        add_existing(paths, seen, path, "user_pubmed_seed", "background_hold")
    for path in lit.glob("user_pubmed_sets/*/screened/*candidates_for_verification.csv"):
        add_existing(paths, seen, path, "user_pubmed_seed", "verifier_ready")

    for path in lit.glob("argument_literature_expansion/**/pubmed_argument_candidates.csv"):
        add_existing(paths, seen, path, "argument_literature_expansion", "candidate")
    for path in lit.glob("argument_literature_expansion/**/*candidates_for_verification.csv"):
        add_existing(paths, seen, path, "argument_literature_expansion", "verifier_ready")

    for path in lit.glob("supplemental_recall_screening/**/supplemental_recall_candidates.csv"):
        add_existing(paths, seen, path, "supplemental_recall", "candidate")
    for path in lit.glob("supplemental_recall_screening/**/screening_queue.csv"):
        add_existing(paths, seen, path, "supplemental_recall", "screening_queue")
    for path in lit.glob("supplemental_recall_screening/**/seed_locked_candidates.csv"):
        add_existing(paths, seen, path, "supplemental_recall", "seed_locked")
    for path in lit.glob("supplemental_recall_screening/**/quarantined_candidates.csv"):
        add_existing(paths, seen, path, "supplemental_recall", "quarantine")
    for path in lit.glob("supplemental_recall_screening/**/accepted_supplemental_recall_candidates.csv"):
        add_existing(paths, seen, path, "supplemental_recall", "screened_accepted")
    for path in lit.glob("supplemental_recall_screening/**/rejected_supplemental_recall_candidates.csv"):
        add_existing(paths, seen, path, "supplemental_recall", "screened_rejected")
    for path in lit.glob("supplemental_recall_screening/**/*candidates_for_verification.csv"):
        add_existing(paths, seen, path, "supplemental_recall", "verifier_ready")

    for path in audit.glob("**/all_draft_reference_verification.csv"):
        add_existing(paths, seen, path, "verification", "verification")
    for path in audit.glob("**/verified_papers_for_pool.csv"):
        add_existing(paths, seen, path, "verification", "verification")

    return paths


def count_group(rows: list[dict[str, str]], *fields: str) -> list[dict[str, str]]:
    groups: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(field, "") or "(blank)" for field in fields)
        if key not in groups:
            groups[key] = {"rows": 0, "papers": set()}
        groups[key]["rows"] += 1
        groups[key]["papers"].add(row.get("paper_key", ""))
    out: list[dict[str, str]] = []
    for key, value in groups.items():
        item = {field: key[idx] for idx, field in enumerate(fields)}
        item["rows"] = str(value["rows"])
        item["unique_papers"] = str(len(value["papers"]))
        if "next_action" not in item:
            item["next_action"] = ""
        out.append(item)
    out.sort(key=lambda item: (-int(item["rows"]), item.get("source_channel", ""), item.get("stage", "")))
    return out


def dedupe_board_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        key = row.get("paper_key") or row.get("board_id")
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def markdown_table(headers: list[str], rows: list[dict[str, str]], limit: int = 30) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows[:limit]:
        values = [clean_space(row.get(header, "")).replace("|", "/") for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    if len(rows) > limit:
        values = ["..."] + [f"{len(rows) - limit} more rows omitted"] + [""] * max(0, len(headers) - 2)
        lines.append("| " + " | ".join(values[: len(headers)]) + " |")
    lines.append("")
    return lines


def write_board_markdown(path: Path, rows: list[dict[str, str]], source_paths: list[tuple[Path, str, str]], project_dir: Path) -> None:
    unique_keys = {row["paper_key"] for row in rows if row.get("paper_key")}
    by_lane = count_group(rows, "source_channel", "stage", "next_action")
    by_action = count_group(rows, "next_action")
    unscreened = dedupe_board_rows([row for row in rows if row.get("next_action") == "screen_with_codex_subagent"])
    verifier_ready = dedupe_board_rows([row for row in rows if row.get("next_action") == "run_draft_reference_verifier"])
    repair = [row for row in rows if row.get("next_action") == "repair_or_replace_reference"]
    lines = [
        "# Literature Candidate Board",
        "",
        f"- Generated at: {now_iso()}",
        f"- Project dir: `{project_dir}`",
        f"- Source files scanned: {len(source_paths)}",
        f"- Board rows: {len(rows)}",
        f"- Unique paper keys: {len(unique_keys)}",
        "",
        "Policy: this board is a chief-editor control surface. It does not promote candidates. Use it to decide what needs subagent screening, deterministic verification, official citation export, full text, or deletion/replacement.",
        "",
        "## Lane Summary",
        "",
    ]
    lines += markdown_table(["source_channel", "stage", "next_action", "rows", "unique_papers"], by_lane, limit=80)
    lines += ["## Action Summary", ""]
    lines += markdown_table(["next_action", "rows", "unique_papers"], by_action, limit=40)
    lines += ["## Next Screening Worklist", ""]
    lines += markdown_table(["source_channel", "stage", "screening_priority", "screening_score", "candidate_id", "title", "year", "next_action"], unscreened, limit=30)
    lines += ["## Verifier-Ready Worklist", ""]
    lines += markdown_table(["source_channel", "stage", "candidate_id", "title", "pmid", "doi", "claim_ids", "next_action"], verifier_ready, limit=30)
    if repair:
        lines += ["## Repair Or Replace", ""]
        lines += markdown_table(["source_channel", "verification_status", "candidate_id", "title", "pmid", "doi", "notes"], repair, limit=30)
    lines += ["## Source Files", ""]
    source_rows = [{"source_file": rel(path, project_dir), "source_channel": channel, "stage": stage} for path, channel, stage in source_paths]
    lines += markdown_table(["source_channel", "stage", "source_file"], source_rows, limit=120)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def chunk_rows(rows: list[dict[str, str]], size: int) -> list[list[dict[str, str]]]:
    if size <= 0:
        return [rows]
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def clip(value: Any, limit: int = 500) -> str:
    text = clean_space(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def resolve_dir(project_dir: Path, raw: str, default: Path) -> Path:
    if not raw:
        return default
    path = Path(raw)
    return path if path.is_absolute() else project_dir / path


def row_bullets(row: dict[str, str]) -> list[str]:
    return [
        f"- Board ID: `{row.get('board_id')}`",
        f"- Candidate ID: `{row.get('candidate_id')}`",
        f"- Source/stage/action: `{row.get('source_channel')}` / `{row.get('stage')}` / `{row.get('next_action')}`",
        f"- Title: {clip(row.get('title'), 220)}",
        f"- Year/journal: {row.get('year') or 'unknown'} / {row.get('journal') or 'unknown'}",
        f"- PMID/DOI: {row.get('pmid') or 'missing'} / {row.get('doi') or 'missing'}",
        f"- Evidence/argument role: {clip(row.get('evidence_role')) or 'unspecified'} / {clip(row.get('argument_role')) or 'unspecified'}",
        f"- Claims: {clip(row.get('claim_ids')) or 'unmapped'}",
        f"- Notes: {clip(row.get('notes'), 500) or '[none]'}",
    ]


def write_candidate_board_curator_packet(board_dir: Path, out_dir: Path, rows: list[dict[str, str]], summary_rows: list[dict[str, str]]) -> dict[str, str]:
    packet_path = out_dir / "candidate_board_curator_packet.md"
    screen_count = sum(1 for row in rows if row.get("next_action") == "screen_with_codex_subagent")
    verify_count = sum(1 for row in rows if row.get("next_action") == "run_draft_reference_verifier")
    repair_count = sum(1 for row in rows if row.get("next_action") == "repair_or_replace_reference")
    export_count = sum(1 for row in rows if row.get("next_action") == "export_official_citation_and_fetch_fulltext")
    screen_unique = len({row.get("paper_key") for row in rows if row.get("next_action") == "screen_with_codex_subagent" and row.get("paper_key")})
    verify_unique = len({row.get("paper_key") for row in rows if row.get("next_action") == "run_draft_reference_verifier" and row.get("paper_key")})
    lines = [
        "# Candidate Board Curator Packet",
        "",
        "Mission: read the candidate board as the chief-editor triage agent. Decide whether the next bottleneck is screening, deterministic verification, repair/delete, official citation export, full-text acquisition, or a genuinely justified new recall wave.",
        "",
        "Input files:",
        "",
        f"- `{board_dir / 'literature_candidate_board.md'}`",
        f"- `{board_dir / 'candidate_lane_summary.csv'}`",
        f"- `{board_dir / 'screening_worklist_deduped.csv'}`",
        f"- `{board_dir / 'verification_worklist_deduped.csv'}`",
        "",
        "Current counts:",
        "",
        f"- Screening rows: {screen_count} ({screen_unique} unique paper keys)",
        f"- Verifier-ready rows: {verify_count} ({verify_unique} unique paper keys)",
        f"- Repair/delete rows: {repair_count}",
        f"- Verified rows needing official citation/full text: {export_count}",
        "",
        "Return CSV with columns:",
        ",".join(BOARD_TRIAGE_FIELDS),
        "",
        "Decision vocabulary:",
        "",
        "- `screen_now`: send to literature_screener",
        "- `verify_now`: send to citation_verifier / deterministic verifier",
        "- `repair_or_replace`: fix identity, title, PMID/DOI, or delete from citation candidates",
        "- `export_and_fetch_fulltext`: official citation export and full-text retrieval",
        "- `defer_background`: keep for background only",
        "- `need_user_decision`: ask user before promotion or deletion",
        "- `justify_new_recall`: only when board evidence shows current lanes cannot support the framework",
        "",
        "Lane summary:",
        "",
    ]
    lines += markdown_table(["source_channel", "stage", "next_action", "rows", "unique_papers"], summary_rows, limit=100)
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    template_rows = [
        {
            **{field: "" for field in BOARD_TRIAGE_FIELDS},
            "board_id": row.get("board_id", ""),
            "paper_key": row.get("paper_key", ""),
            "candidate_id": row.get("candidate_id", ""),
            "source_channel": row.get("source_channel", ""),
            "current_stage": row.get("stage", ""),
        }
        for row in rows[:200]
    ]
    write_csv(out_dir / "candidate_board_triage_template.csv", template_rows, BOARD_TRIAGE_FIELDS)
    return {
        "packet_id": "candidate_board_curator",
        "agent": "candidate_board_curator",
        "packet_file": str(packet_path.relative_to(out_dir)),
        "worklist_file": "literature_candidate_board.csv",
        "start_index": "1",
        "end_index": str(len(rows)),
        "items": str(len(rows)),
        "purpose": "Triage candidate-board bottlenecks and assign next actions.",
    }


def write_literature_screener_packets(out_dir: Path, rows: list[dict[str, str]], packet_size: int, max_items: int) -> list[dict[str, str]]:
    selected = rows[:max_items] if max_items > 0 else rows
    packet_dir = out_dir / "literature_screener_packets"
    packet_rows: list[dict[str, str]] = []
    template_rows: list[dict[str, str]] = []
    for packet_index, chunk in enumerate(chunk_rows(selected, packet_size), start=1):
        packet_id = f"literature_screener_{packet_index:04d}"
        packet_path = packet_dir / f"{packet_id}.md"
        lines = [
            "# Literature Screener Packet",
            "",
            "Mission: screen each candidate against the confirmed review framework, evidence hierarchy, and published-only citation policy. Do not include a paper merely because it is interesting.",
            "",
            "Return CSV with columns:",
            ",".join(["board_id", "paper_key", *SCREENING_FIELDS]),
            "",
            "Required judgment: every inclusion must fill `argument_role`, `evidence_level`, `evidence_strength`, `fulltext_need`, `citation_role`, and `key_supported_claim`.",
            "",
            "Decision vocabulary: include_core, include_support, include_counterevidence, include_framework, include_historical_foundation, include_method_foundation, include_governance_background, include_background, reject, needs_fulltext, duplicate_already_verified.",
            "",
        ]
        for offset, row in enumerate(chunk, start=1):
            lines.extend([f"## {offset}. {clip(row.get('title'), 180)}", ""])
            lines.extend(row_bullets(row))
            lines.append("")
            template_rows.append({**{field: "" for field in ["board_id", "paper_key", *SCREENING_FIELDS]}, "board_id": row.get("board_id", ""), "paper_key": row.get("paper_key", ""), "candidate_id": row.get("candidate_id", ""), "task_id": row.get("task_id", "")})
        packet_path.parent.mkdir(parents=True, exist_ok=True)
        packet_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        packet_rows.append(
            {
                "packet_id": packet_id,
                "agent": "literature_screener",
                "packet_file": str(packet_path.relative_to(out_dir)),
                "worklist_file": "screening_worklist_deduped.csv",
                "start_index": str((packet_index - 1) * packet_size + 1),
                "end_index": str((packet_index - 1) * packet_size + len(chunk)),
                "items": str(len(chunk)),
                "purpose": "Screen candidate papers and bind inclusions to review claims.",
            }
        )
    write_csv(out_dir / "literature_screener_decision_template.csv", template_rows, ["board_id", "paper_key", *SCREENING_FIELDS])
    return packet_rows


def write_citation_verifier_packets(out_dir: Path, rows: list[dict[str, str]], packet_size: int, max_items: int) -> list[dict[str, str]]:
    selected = rows[:max_items] if max_items > 0 else rows
    packet_dir = out_dir / "citation_verifier_packets"
    packet_rows: list[dict[str, str]] = []
    template_rows: list[dict[str, str]] = []
    for packet_index, chunk in enumerate(chunk_rows(selected, packet_size), start=1):
        packet_id = f"citation_verifier_{packet_index:04d}"
        packet_path = packet_dir / f"{packet_id}.md"
        lines = [
            "# Citation Verifier Packet",
            "",
            "Mission: inspect verifier-ready candidates before deterministic API verification. Decide whether each row is ready for PubMed/DOI/title verification, needs identity repair, is a duplicate, or should be sent back to screening.",
            "",
            "Return CSV with columns:",
            ",".join(VERIFICATION_TRIAGE_FIELDS),
            "",
            "Verification routes: pubmed_pmid, doi, title_bibliographic, publisher_page, conference_page, repair_needed, duplicate, reject.",
            "",
        ]
        for offset, row in enumerate(chunk, start=1):
            lines.extend([f"## {offset}. {clip(row.get('title'), 180)}", ""])
            lines.extend(row_bullets(row))
            lines.append("")
            template_rows.append(
                {
                    **{field: "" for field in VERIFICATION_TRIAGE_FIELDS},
                    "board_id": row.get("board_id", ""),
                    "paper_key": row.get("paper_key", ""),
                    "candidate_id": row.get("candidate_id", ""),
                    "candidate_title": row.get("title", ""),
                    "pmid": row.get("pmid", ""),
                    "doi": row.get("doi", ""),
                }
            )
        packet_path.parent.mkdir(parents=True, exist_ok=True)
        packet_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        packet_rows.append(
            {
                "packet_id": packet_id,
                "agent": "citation_verifier",
                "packet_file": str(packet_path.relative_to(out_dir)),
                "worklist_file": "verification_worklist_deduped.csv",
                "start_index": str((packet_index - 1) * packet_size + 1),
                "end_index": str((packet_index - 1) * packet_size + len(chunk)),
                "items": str(len(chunk)),
                "purpose": "Preflight verifier-ready rows and choose verification routes.",
            }
        )
    write_csv(out_dir / "citation_verifier_triage_template.csv", template_rows, VERIFICATION_TRIAGE_FIELDS)
    return packet_rows


def write_packet_index_markdown(path: Path, packet_rows: list[dict[str, str]]) -> None:
    lines = [
        "# Candidate Board Agent Packet Index",
        "",
        "Assign these packets to Codex subtasks. The chief Codex conversation remains responsible for accepting, rejecting, and integrating agent outputs.",
        "",
        "| Packet | Agent | Items | Worklist | File |",
        "|---|---|---:|---|---|",
    ]
    for row in packet_rows:
        lines.append(f"| `{row['packet_id']}` | {row['agent']} | {row['items']} | `{row['worklist_file']}` | `{row['packet_file']}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def unique_fields(fields: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for field in fields:
        if field in seen:
            continue
        seen.add(field)
        out.append(field)
    return out


def iter_csv_inputs(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for item in inputs:
        if item.is_dir():
            candidates = sorted(item.rglob("*.csv"))
        elif item.exists():
            candidates = [item]
        else:
            candidates = []
        for path in candidates:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(path)
    return paths


def classify_agent_result(row: dict[str, str], path: Path) -> str:
    name = path.name.lower()
    if "verification_route" in row or "citation_verifier" in name:
        return "citation_verifier"
    if "recommended_agent" in row or "current_stage" in row or "candidate_board_triage" in name:
        return "candidate_board_curator"
    if "claim_fit" in row or "key_supported_claim" in row or "literature_screener" in name:
        return "literature_screener"
    return "unknown"


def has_agent_signal(agent_type: str, row: dict[str, str]) -> bool:
    if agent_type == "citation_verifier":
        return bool(clean_space(row.get("verification_route")))
    if agent_type == "candidate_board_curator":
        return bool(clean_space(row.get("decision")) or clean_space(row.get("next_action")))
    if agent_type == "literature_screener":
        return bool(clean_space(row.get("decision")))
    return any(clean_space(value) for value in row.values())


def board_indexes(board_rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    by_board_id = {row.get("board_id", ""): row for row in board_rows if row.get("board_id")}
    by_candidate_id: dict[str, dict[str, str]] = {}
    by_paper_key: dict[str, dict[str, str]] = {}
    for row in board_rows:
        by_candidate_id.setdefault(row.get("candidate_id", ""), row)
        by_paper_key.setdefault(row.get("paper_key", ""), row)
    return by_board_id, by_candidate_id, by_paper_key


def lookup_board_row(
    decision: dict[str, str],
    by_board_id: dict[str, dict[str, str]],
    by_candidate_id: dict[str, dict[str, str]],
    by_paper_key: dict[str, dict[str, str]],
) -> dict[str, str]:
    board_id = clean_space(decision.get("board_id"))
    if board_id and board_id in by_board_id:
        return by_board_id[board_id]
    candidate_id = clean_space(decision.get("candidate_id"))
    if candidate_id and candidate_id in by_candidate_id:
        return by_candidate_id[candidate_id]
    key = clean_space(decision.get("paper_key"))
    if key and key in by_paper_key:
        return by_paper_key[key]
    return {}


def join_agent_decision(board: dict[str, str], decision: dict[str, str], source_path: Path, agent_type: str, decision_family: str) -> dict[str, str]:
    joined = {field: board.get(field, "") for field in BOARD_FIELDS}
    if not board:
        joined.update(
            {
                "board_id": clean_space(decision.get("board_id")),
                "paper_key": clean_space(decision.get("paper_key")),
                "candidate_id": clean_space(decision.get("candidate_id")),
                "task_id": clean_space(decision.get("task_id")),
                "title": clean_space(decision.get("candidate_title")),
                "pmid": clean_space(decision.get("pmid")),
                "doi": normalize_doi(clean_space(decision.get("doi"))),
            }
        )
    joined.update(
        {
            "agent_result_file": str(source_path),
            "agent_type": agent_type,
            "decision_family": decision_family,
            "agent_decision": clean_space(decision.get("decision")),
            "agent_confidence": clean_space(decision.get("confidence")),
            "agent_priority": clean_space(decision.get("priority")),
            "agent_route": clean_space(decision.get("verification_route")),
            "agent_expected_status": clean_space(decision.get("expected_status")),
            "agent_claim_fit": clean_space(decision.get("claim_fit")),
            "agent_evidence_role": clean_space(decision.get("evidence_role")),
            "agent_argument_role": clean_space(decision.get("argument_role")),
            "agent_evidence_level": clean_space(decision.get("evidence_level")),
            "agent_evidence_strength": clean_space(decision.get("evidence_strength")),
            "agent_fulltext_need": clean_space(decision.get("fulltext_need")),
            "agent_citation_role": clean_space(decision.get("citation_role")),
            "agent_key_supported_claim": clean_space(decision.get("key_supported_claim") or decision.get("claim_or_gap")),
            "agent_supported_claim_ids": clean_space(decision.get("supported_claim_ids")),
            "agent_rationale": clean_space(decision.get("rationale")),
            "agent_human_question": clean_space(decision.get("human_question")),
            "agent_required_next_action": clean_space(decision.get("required_next_action") or decision.get("next_action")),
        }
    )
    return joined


def board_candidate_for_verifier(board: dict[str, str], decision: dict[str, str]) -> dict[str, str]:
    return {
        **board,
        "candidate_id": board.get("candidate_id") or clean_space(decision.get("candidate_id")),
        "title": board.get("title") or clean_space(decision.get("candidate_title")),
        "doi": board.get("doi") or normalize_doi(clean_space(decision.get("doi"))),
        "pmid": board.get("pmid") or clean_space(decision.get("pmid")),
        "url": board.get("url"),
        "linked_claim": board.get("claim_ids"),
        "linked_claim_id": board.get("claim_ids"),
        "candidate_fit_reason": board.get("notes"),
        "query": board.get("title") or clean_space(decision.get("candidate_title")),
    }


def screening_verifier_row(board: dict[str, str], decision: dict[str, str]) -> dict[str, str]:
    candidate = board_candidate_for_verifier(board, decision)
    row = verifier_row(candidate, decision)
    lane = verification_lane(candidate, decision)
    row["draft"] = "candidate_board"
    row["ref_number"] = board.get("board_id") or row.get("ref_number", "")
    row["source_kind"] = f"candidate_board_screened:{board.get('source_channel', '')}:{lane}"
    row["cited_in_body"] = clean_space("; ".join(part for part in [board.get("source_channel"), board.get("source_file")] if part))
    row["pubmed_query"] = board.get("title") or row.get("pubmed_query", "")
    row["why_relevant"] = clean_space(
        "; ".join(
            part
            for part in [
                f"board_id={board.get('board_id')}",
                row.get("why_relevant", ""),
                clean_space(decision.get("human_question")),
            ]
            if part
        )
    )[:700]
    return row


def direct_verifier_row(board: dict[str, str], decision: dict[str, str], route: str, agent_type: str) -> dict[str, str]:
    title = board.get("title") or clean_space(decision.get("candidate_title"))
    doi = board.get("doi") or normalize_doi(clean_space(decision.get("doi")))
    pmid = board.get("pmid") or clean_space(decision.get("pmid"))
    reason = clean_space(decision.get("rationale") or board.get("notes"))
    claim = clean_space(decision.get("key_supported_claim") or decision.get("claim_or_gap") or board.get("claim_ids"))
    return {
        "draft": "candidate_board",
        "ref_number": board.get("board_id") or board.get("candidate_id"),
        "candidate_id": board.get("candidate_id"),
        "claim_id": board.get("claim_ids"),
        "candidate_title": title,
        "raw_reference": f"{board.get('authors', '')} {board.get('year', '')}. {title}. {board.get('journal', '')}. {doi or pmid or board.get('url', '')}",
        "doi": doi,
        "pmid": pmid,
        "urls": board.get("url"),
        "source_kind": f"candidate_board_preflight:{agent_type}:{route}:{board.get('source_channel', '')}",
        "cited_in_body": clean_space("; ".join(part for part in [board.get("source_channel"), board.get("source_file")] if part)),
        "verification_priority": clean_space(decision.get("priority")) or "medium",
        "candidate_type": clean_space(decision.get("decision") or route),
        "pubmed_query": title,
        "raw_evidence_excerpt": claim[:900],
        "why_relevant": clean_space("; ".join(part for part in [f"board_id={board.get('board_id')}", reason, claim] if part))[:700],
    }


def queue_row(board: dict[str, str], decision: dict[str, str], agent_type: str, next_action: str) -> dict[str, str]:
    return {
        "board_id": board.get("board_id") or clean_space(decision.get("board_id")),
        "paper_key": board.get("paper_key") or clean_space(decision.get("paper_key")),
        "candidate_id": board.get("candidate_id") or clean_space(decision.get("candidate_id")),
        "source_channel": board.get("source_channel"),
        "stage": board.get("stage"),
        "title": board.get("title") or clean_space(decision.get("candidate_title")),
        "pmid": board.get("pmid") or clean_space(decision.get("pmid")),
        "doi": board.get("doi") or normalize_doi(clean_space(decision.get("doi"))),
        "agent_type": agent_type,
        "agent_decision": clean_space(decision.get("decision")),
        "agent_route": clean_space(decision.get("verification_route")),
        "priority": clean_space(decision.get("priority")),
        "reason": clean_space(decision.get("rationale") or board.get("notes"))[:900],
        "next_action": next_action,
        "human_question": clean_space(decision.get("human_question")),
    }


def curator_action(decision: dict[str, str]) -> str:
    dec = clean_space(decision.get("decision"))
    action = clean_space(decision.get("next_action"))
    text = f"{dec} {action}".lower()
    if dec == "verify_now" or "deterministic verifier" in text or "run citation_verifier" in text or "run citation verifier" in text:
        return "verify_now"
    if dec == "repair_or_replace" or "repair" in text or "replace" in text:
        return "repair_or_replace"
    if dec in FULLTEXT_ACTIONS or "full text" in text or "full-text" in text or "fetch" in text:
        return "export_and_fetch_fulltext"
    if dec in REJECT_ROUTES or "reject" in text or "delete" in text:
        return "reject"
    if dec == "screen_now" or "screen" in text:
        return "screen_now"
    if "human" in text or "manual" in text or clean_space(decision.get("human_question")):
        return "human"
    return action or dec


def dedupe_verify_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        key = normalize_doi(row.get("doi", "")) or clean_space(row.get("pmid")) or normalize_title(row.get("candidate_title", ""))
        if not key:
            key = clean_space(row.get("ref_number"))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def write_collection_markdown(path: Path, manifest: dict[str, Any], summary_rows: list[dict[str, str]]) -> None:
    lines = [
        "# Candidate Board Agent Decision Collection",
        "",
        f"- Generated at: {manifest['generated_at']}",
        f"- Board rows loaded: {manifest['board_rows']}",
        f"- Agent CSV files scanned: {manifest['agent_csv_files']}",
        f"- Agent decisions accepted: {manifest['agent_decisions']}",
        f"- Unresolved decisions: {manifest['unresolved_decisions']}",
        f"- Verifier rows: {manifest['verifier_rows']}",
        f"- Repair rows: {manifest['repair_rows']}",
        f"- Full-text rows: {manifest['fulltext_rows']}",
        f"- Manual questions: {manifest['manual_question_rows']}",
        "",
        "Policy: these outputs are routing queues. They do not promote candidates into the literature pool. Run deterministic verification, adjudication, official citation export, and final-gate checks before citing.",
        "",
        "## Outputs",
        "",
        "- `candidate_board_agent_decisions.csv`: all nonblank agent decisions joined to board metadata.",
        "- `candidate_board_candidates_for_verification.csv`: next deterministic verifier input.",
        "- `candidate_board_core_candidates_for_verification.csv`: verifier input for core system evidence.",
        "- `candidate_board_framework_candidates_for_verification.csv`: verifier input for history/method/governance/framework evidence.",
        "- `candidate_board_repair_queue.csv`: candidates that need identity repair, replacement, or deletion.",
        "- `candidate_board_fulltext_queue.csv`: candidates whose next useful step is official citation export/full text.",
        "- `candidate_board_manual_questions.csv`: rows requiring human adjudication.",
        "",
        "## Decision Summary",
        "",
    ]
    lines += markdown_table(["agent_type", "decision_family", "rows", "unique_papers", "next_action"], summary_rows, limit=80)
    lines += [
        "## Next Commands",
        "",
        "Run deterministic verification on `candidate_board_candidates_for_verification.csv`; do not import these rows into the main pool until verification and adjudication pass.",
        "",
    ]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def cmd_build(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else project_dir / "review-data" / "02_literature" / "candidate_board"
    source_paths = filter_candidate_sources(discover_candidate_sources(project_dir), args.include_path_contains, args.exclude_path_contains)
    rows: list[dict[str, str]] = []
    for path, channel, stage in source_paths:
        rows.extend(normalize_row(row, path, project_dir, channel, stage) for row in read_csv(path))
    rows.sort(
        key=lambda row: (
            row.get("next_action", ""),
            row.get("source_channel", ""),
            row.get("stage", ""),
            row.get("screening_priority", ""),
            -(int(row.get("screening_score") or 0) if str(row.get("screening_score") or "").isdigit() else 0),
            row.get("title", ""),
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "literature_candidate_board.csv", rows, BOARD_FIELDS)
    summary_rows = count_group(rows, "source_channel", "stage", "next_action")
    write_csv(out_dir / "candidate_lane_summary.csv", summary_rows, SUMMARY_FIELDS)
    screening_rows = [row for row in rows if row.get("next_action") == "screen_with_codex_subagent"]
    verification_rows = [row for row in rows if row.get("next_action") == "run_draft_reference_verifier"]
    write_csv(out_dir / "screening_worklist.csv", screening_rows, BOARD_FIELDS)
    write_csv(out_dir / "verification_worklist.csv", verification_rows, BOARD_FIELDS)
    write_csv(out_dir / "screening_worklist_deduped.csv", dedupe_board_rows(screening_rows), BOARD_FIELDS)
    write_csv(out_dir / "verification_worklist_deduped.csv", dedupe_board_rows(verification_rows), BOARD_FIELDS)
    write_board_markdown(out_dir / "literature_candidate_board.md", rows, source_paths, project_dir)
    write_json(
        out_dir / "literature_candidate_board.json",
        {
            "generated_at": now_iso(),
            "project_dir": str(project_dir),
            "out_dir": str(out_dir),
            "source_files": len(source_paths),
            "rows": len(rows),
            "unique_paper_keys": len({row["paper_key"] for row in rows if row.get("paper_key")}),
            "summary": summary_rows,
        },
    )
    print(json.dumps({"out_dir": str(out_dir), "source_files": len(source_paths), "rows": len(rows)}, ensure_ascii=False))
    return 0


def cmd_packets(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    board_dir = resolve_dir(project_dir, args.board_dir, project_dir / "review-data" / "02_literature" / "candidate_board")
    out_dir = resolve_dir(project_dir, args.out_dir, board_dir / "agent_packets")
    board_rows = read_csv(board_dir / "literature_candidate_board.csv")
    summary_rows = read_csv(board_dir / "candidate_lane_summary.csv")
    screening_rows = read_csv(board_dir / "screening_worklist_deduped.csv")
    verification_rows = read_csv(board_dir / "verification_worklist_deduped.csv")
    if not board_rows:
        raise SystemExit(f"No literature_candidate_board.csv found in {board_dir}. Run `build` first.")
    packet_rows: list[dict[str, str]] = []
    packet_rows.append(write_candidate_board_curator_packet(board_dir, out_dir, board_rows, summary_rows))
    packet_rows.extend(write_literature_screener_packets(out_dir, screening_rows, args.packet_size, args.max_screening_items))
    packet_rows.extend(write_citation_verifier_packets(out_dir, verification_rows, args.packet_size, args.max_verification_items))
    write_csv(out_dir / "agent_packet_index.csv", packet_rows, PACKET_INDEX_FIELDS)
    write_packet_index_markdown(out_dir / "agent_packet_index.md", packet_rows)
    write_json(
        out_dir / "agent_packet_manifest.json",
        {
            "generated_at": now_iso(),
            "board_dir": str(board_dir),
            "out_dir": str(out_dir),
            "board_rows": len(board_rows),
            "screening_rows": len(screening_rows),
            "verification_rows": len(verification_rows),
            "packets": packet_rows,
        },
    )
    print(json.dumps({"out_dir": str(out_dir), "packets": len(packet_rows), "screening_rows": len(screening_rows), "verification_rows": len(verification_rows)}, ensure_ascii=False))
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    board_dir = resolve_dir(project_dir, args.board_dir, project_dir / "review-data" / "02_literature" / "candidate_board")
    out_dir = resolve_dir(project_dir, args.out_dir, board_dir / "collected")
    board_rows = read_csv(board_dir / "literature_candidate_board.csv")
    if not board_rows:
        raise SystemExit(f"No literature_candidate_board.csv found in {board_dir}. Run `build` first.")

    input_items: list[Path] = []
    for raw in args.input_dir or []:
        input_items.append(resolve_dir(project_dir, raw, board_dir / "agent_results"))
    for raw in args.input_csv or []:
        path = Path(raw)
        input_items.append(path if path.is_absolute() else project_dir / path)
    if not input_items:
        input_items.append(board_dir / "agent_results")
    csv_paths = iter_csv_inputs(input_items)

    by_board_id, by_candidate_id, by_paper_key = board_indexes(board_rows)
    decisions: list[dict[str, str]] = []
    unresolved: list[dict[str, str]] = []
    screening_decisions: list[dict[str, str]] = []
    verifier_triage: list[dict[str, str]] = []
    curator_decisions: list[dict[str, str]] = []
    verifier_rows: list[dict[str, str]] = []
    core_verifier_rows: list[dict[str, str]] = []
    framework_verifier_rows: list[dict[str, str]] = []
    repair_queue: list[dict[str, str]] = []
    fulltext_queue: list[dict[str, str]] = []
    rejection_queue: list[dict[str, str]] = []
    duplicate_queue: list[dict[str, str]] = []
    manual_questions: list[dict[str, str]] = []

    def add_verifier(row: dict[str, str], board: dict[str, str]) -> None:
        verifier_rows.append(row)
        lane_hint = " ".join([row.get("source_kind", ""), board.get("argument_role", ""), board.get("evidence_role", "")]).lower()
        if "core_biomedical_system_evidence" in lane_hint or "core" in lane_hint:
            core_verifier_rows.append(row)
        else:
            framework_verifier_rows.append(row)

    for path in csv_paths:
        for raw_row in read_csv(path):
            agent_type = classify_agent_result(raw_row, path)
            if not has_agent_signal(agent_type, raw_row):
                continue
            board = lookup_board_row(raw_row, by_board_id, by_candidate_id, by_paper_key)
            decision_family = agent_type if board else "unresolved"
            joined = join_agent_decision(board, raw_row, path, agent_type, decision_family)
            decisions.append(joined)
            if not board:
                unresolved.append(joined)
                if clean_space(raw_row.get("human_question")):
                    manual_questions.append(queue_row({}, raw_row, agent_type, "resolve_unmatched_agent_decision"))
                continue

            if agent_type == "literature_screener":
                screening_decisions.append(joined)
                decision = clean_space(raw_row.get("decision"))
                candidate = board_candidate_for_verifier(board, raw_row)
                if decision in INCLUDE_DECISIONS and should_send_to_verifier(candidate, raw_row):
                    add_verifier(screening_verifier_row(board, raw_row), board)
                elif decision in NO_VERIFIER_DECISIONS:
                    fulltext_need = clean_space(raw_row.get("fulltext_need")).lower()
                    if "need" in fulltext_need or "required" in fulltext_need:
                        fulltext_queue.append(queue_row(board, raw_row, agent_type, "fulltext_or_background_handoff"))
                elif decision in REJECT_ROUTES or decision.startswith("reject"):
                    rejection_queue.append(queue_row(board, raw_row, agent_type, "keep_rejection_for_audit"))
                if decision == "needs_fulltext" or clean_space(raw_row.get("fulltext_need")).lower() in {"needed", "required", "yes"}:
                    fulltext_queue.append(queue_row(board, raw_row, agent_type, "fetch_fulltext_after_verification"))

            elif agent_type == "citation_verifier":
                verifier_triage.append(joined)
                route = clean_space(raw_row.get("verification_route"))
                if route in READY_VERIFICATION_ROUTES:
                    add_verifier(direct_verifier_row(board, raw_row, route, agent_type), board)
                elif route in REPAIR_ROUTES:
                    repair_queue.append(queue_row(board, raw_row, agent_type, "repair_identity_before_verification"))
                elif route == "duplicate":
                    duplicate_queue.append(queue_row(board, raw_row, agent_type, "dedupe_against_verified_union"))
                elif route in REJECT_ROUTES:
                    rejection_queue.append(queue_row(board, raw_row, agent_type, "keep_rejection_for_audit"))
                elif route in {"human", "manual", "needs_user_decision"}:
                    manual_questions.append(queue_row(board, raw_row, agent_type, "ask_human_before_verification"))

            elif agent_type == "candidate_board_curator":
                curator_decisions.append(joined)
                decision = clean_space(raw_row.get("decision"))
                action = curator_action(raw_row)
                if action == "verify_now":
                    add_verifier(direct_verifier_row(board, raw_row, "curator_verify_now", agent_type), board)
                elif action in REPAIR_ROUTES or decision == "repair_or_replace":
                    repair_queue.append(queue_row(board, raw_row, agent_type, "repair_identity_before_verification"))
                elif action in FULLTEXT_ACTIONS or decision in FULLTEXT_ACTIONS:
                    fulltext_queue.append(queue_row(board, raw_row, agent_type, "export_official_citation_and_fetch_fulltext"))
                elif action in REJECT_ROUTES or decision in REJECT_ROUTES:
                    rejection_queue.append(queue_row(board, raw_row, agent_type, "keep_rejection_for_audit"))
                elif action in {"human", "manual", "needs_user_decision"} or clean_space(raw_row.get("human_question")):
                    manual_questions.append(queue_row(board, raw_row, agent_type, "ask_human_before_next_action"))

            elif clean_space(raw_row.get("human_question")):
                manual_questions.append(queue_row(board, raw_row, agent_type, "ask_human_before_next_action"))

    verifier_rows = dedupe_verify_rows(verifier_rows)
    core_verifier_rows = dedupe_verify_rows(core_verifier_rows)
    framework_verifier_rows = dedupe_verify_rows(framework_verifier_rows)
    repair_queue = dedupe_board_rows(repair_queue)
    fulltext_queue = dedupe_board_rows(fulltext_queue)
    rejection_queue = dedupe_board_rows(rejection_queue)
    duplicate_queue = dedupe_board_rows(duplicate_queue)
    manual_questions = dedupe_board_rows(manual_questions)

    out_dir.mkdir(parents=True, exist_ok=True)
    decision_fields = unique_fields(COLLECTED_DECISION_FIELDS)
    write_csv(out_dir / "candidate_board_agent_decisions.csv", decisions, decision_fields)
    write_csv(out_dir / "candidate_board_screening_decisions.csv", screening_decisions, decision_fields)
    write_csv(out_dir / "candidate_board_verifier_triage.csv", verifier_triage, decision_fields)
    write_csv(out_dir / "candidate_board_curator_decisions.csv", curator_decisions, decision_fields)
    write_csv(out_dir / "candidate_board_unresolved_agent_decisions.csv", unresolved, decision_fields)
    write_csv(out_dir / "candidate_board_candidates_for_verification.csv", verifier_rows, VERIFY_FIELDS)
    write_csv(out_dir / "candidate_board_core_candidates_for_verification.csv", core_verifier_rows, VERIFY_FIELDS)
    write_csv(out_dir / "candidate_board_framework_candidates_for_verification.csv", framework_verifier_rows, VERIFY_FIELDS)
    write_csv(out_dir / "candidate_board_repair_queue.csv", repair_queue, QUEUE_FIELDS)
    write_csv(out_dir / "candidate_board_fulltext_queue.csv", fulltext_queue, QUEUE_FIELDS)
    write_csv(out_dir / "candidate_board_rejections.csv", rejection_queue, QUEUE_FIELDS)
    write_csv(out_dir / "candidate_board_duplicate_queue.csv", duplicate_queue, QUEUE_FIELDS)
    write_csv(out_dir / "candidate_board_manual_questions.csv", manual_questions, QUEUE_FIELDS)

    summary_rows = count_group(decisions, "agent_type", "decision_family", "next_action")
    write_csv(out_dir / "candidate_board_agent_decision_summary.csv", summary_rows, ["agent_type", "decision_family", "next_action", "rows", "unique_papers"])
    manifest = {
        "generated_at": now_iso(),
        "project_dir": str(project_dir),
        "board_dir": str(board_dir),
        "out_dir": str(out_dir),
        "board_rows": len(board_rows),
        "agent_csv_files": len(csv_paths),
        "agent_decisions": len(decisions),
        "unresolved_decisions": len(unresolved),
        "screening_decisions": len(screening_decisions),
        "verifier_triage_decisions": len(verifier_triage),
        "curator_decisions": len(curator_decisions),
        "verifier_rows": len(verifier_rows),
        "core_verifier_rows": len(core_verifier_rows),
        "framework_verifier_rows": len(framework_verifier_rows),
        "repair_rows": len(repair_queue),
        "fulltext_rows": len(fulltext_queue),
        "rejection_rows": len(rejection_queue),
        "duplicate_rows": len(duplicate_queue),
        "manual_question_rows": len(manual_questions),
        "input_files": [str(path) for path in csv_paths],
    }
    write_json(out_dir / "candidate_board_agent_decision_manifest.json", manifest)
    write_collection_markdown(out_dir / "candidate_board_next_actions.md", manifest, summary_rows)
    print(json.dumps({"out_dir": str(out_dir), "agent_decisions": len(decisions), "verifier_rows": len(verifier_rows), "repair_rows": len(repair_queue)}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a read-only board across literature candidate lanes.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_build = sub.add_parser("build", help="Build the literature candidate board.")
    p_build.add_argument("--project-dir", default=".")
    p_build.add_argument("--out-dir", default="")
    p_build.add_argument("--include-path-contains", action="append", default=[], help="Only include source files whose normalized path contains this substring; repeatable.")
    p_build.add_argument("--exclude-path-contains", action="append", default=[], help="Exclude source files whose normalized path contains this substring; repeatable.")
    p_build.set_defaults(func=cmd_build)

    p_packets = sub.add_parser("packets", help="Create Codex-subtask packets from candidate-board worklists.")
    p_packets.add_argument("--project-dir", default=".")
    p_packets.add_argument("--board-dir", default="./review-data/02_literature/candidate_board")
    p_packets.add_argument("--out-dir", default="")
    p_packets.add_argument("--packet-size", type=int, default=25)
    p_packets.add_argument("--max-screening-items", type=int, default=0)
    p_packets.add_argument("--max-verification-items", type=int, default=0)
    p_packets.set_defaults(func=cmd_packets)

    p_collect = sub.add_parser("collect", help="Collect candidate-board agent decisions into auditable next-step queues.")
    p_collect.add_argument("--project-dir", default=".")
    p_collect.add_argument("--board-dir", default="./review-data/02_literature/candidate_board")
    p_collect.add_argument("--input-dir", action="append", default=[])
    p_collect.add_argument("--input-csv", action="append", default=[])
    p_collect.add_argument("--out-dir", default="")
    p_collect.set_defaults(func=cmd_collect)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
