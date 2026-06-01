#!/usr/bin/env python3
"""Multi-agent orchestration for top-journal review writing.

Codex remains the chief editor/PI. This script prepares specialist agent
packets and can optionally call an OpenAI-compatible chat completion endpoint
such as DeepSeek. It is intentionally stdlib-only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import textwrap
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


DEFAULT_AGENTS: dict[str, dict[str, str]] = {
    "draft_frame_reader": {
        "title": "Multi-Draft Frame Reader",
        "mission": "Read all supplied GPT/Gemini/Deep Research drafts as high-value but untrusted scaffolds; extract reusable frames, thesis candidates, section logic, and claims without drafting new prose.",
        "outputs": "draft_consensus, unique_insights, weak_frames, thesis_candidates, merge_delete_heading_advice",
    },
    "draft_deep_reader": {
        "title": "Draft Deep-Reading Agent",
        "mission": "Study draft memory cards and raw draft packets carefully; recover what each draft is actually arguing, where its strongest insights live, and which ideas deserve user discussion before any manuscript prose is written.",
        "outputs": "draft_by_draft_memory, best_reusable_insights, misunderstood_or_underused_points, claims_requiring_literature_support, questions_for_user",
    },
    "draft_memory_curator": {
        "title": "Draft Memory Curator",
        "mission": "Maintain persistent draft memory across sessions; connect draft cards to the framework blackboard, mark which draft ideas are accepted/rejected/pending, and prevent Codex from losing the user's initial materials.",
        "outputs": "memory_updates, accepted_draft_elements, rejected_or_demoted_elements, pending_user_decisions, context_to_load_next",
    },
    "draft_literature_candidate_miner": {
        "title": "Draft Literature Candidate Miner",
        "mission": "After chief Codex has read all drafts and citation assets, read draft body prose for paper clues: reference-recovery cases, named systems, methods, author/year hints, venue clues, title fragments, hidden body candidates, and narrow PubMed deep-dive queries. Produce candidate clues only; do not certify or cite them.",
        "outputs": "ai_mined_candidate_clues, narrow_pubmed_queries, missing_landmark_hypotheses, claim_linked_evidence_needs, human_verification_questions",
    },
    "literature_strategist": {
        "title": "Literature Strategy Agent",
        "mission": "Read the draft-derived framework and argument map before searching. Convert each thesis, historical foundation, current-evidence claim, future-agenda claim, figure/table row, and disputed point into targeted evidence tasks. Review and narrow argument_literature_expander tasks before retrieval. Literature supports or challenges the draft logic; it must not replace the framework with broad topic recall.",
        "outputs": "argument_map_evidence_needs, past_present_future_evidence_plan, pubmed_abstract_triage_plan, expander_task_edits, methods_support_targets, inclusion_exclusion_rules, supplemental_pool_boundaries",
    },
    "literature_retriever": {
        "title": "Literature Retrieval Agent",
        "mission": "Find missing paper evidence for bounded argument-map tasks, replace web/blog sources with scholarly sources, and inspect PubMed abstract packets plus targeted Crossref/OpenAlex/arXiv/OpenReview results for later screening. Off-map discoveries go to supplemental_pool as background_or_defer.",
        "outputs": "claim_bound_search_queries, pubmed_abstract_candidates, abstract_only_limits, source_replacements, missing_evidence, background_or_defer_items",
    },
    "literature_screener": {
        "title": "Literature Inclusion/Exclusion Agent",
        "mission": "Screen candidate papers against review scope, evidence hierarchy, target journal standards, and published-only citation policy.",
        "outputs": "include_exclude_table, rationale, citation_pool_candidates, rejected_cases",
    },
    "candidate_board_curator": {
        "title": "Literature Candidate Board Curator",
        "mission": "Read the literature candidate board as a chief-editor triage surface. Identify screening backlog, verifier-ready rows, verified rows needing citation/full text, repair/delete rows, and whether more recall is justified.",
        "outputs": "candidate_lane_triage, screening_priorities, verification_priorities, repair_delete_tasks, fulltext_or_citation_export_tasks, recall_stop_or_continue_decision",
    },
    "literature_manager": {
        "title": "Literature Memory Manager",
        "mission": "Check the literature pool, citation pool, cards, evidence files, RAG index, and handoff list for completeness and progressive disclosure readiness.",
        "outputs": "pool_health, missing_cards, missing_fulltext, rag_readiness, next_commands",
    },
    "citation_verifier": {
        "title": "Final Citation Traceback Agent",
        "mission": "Audit every final citation and important claim against verified metadata, paper cards, abstracts/full text, and the published-only gate.",
        "outputs": "blocking_citation_issues, claim_support_table, citation_strength, required_rewrites",
    },
    "outline_architect": {
        "title": "Review Architecture Agent",
        "mission": "Use the framework blackboard, literature tasks, and verified evidence to propose one central thesis, 3-5 major section movements, transition logic, and a Nature-style contribution frame.",
        "outputs": "central_thesis, 3_to_5_section_spine, transition_logic, evidence_to_section_map, novelty_claim, scope_controls",
    },
    "argument_builder": {
        "title": "Argument Builder Agent",
        "mission": "Construct claim-evidence-reasoning chains for the chosen framework, identify counterarguments, score argument strength, and mark claims that need literature recall or demotion.",
        "outputs": "central_thesis, sub_arguments, CER_chains, counterarguments, evidence_strength_scores, claims_to_demote",
    },
    "framework_dialogue_moderator": {
        "title": "Framework Dialogue Moderator",
        "mission": "Turn draft memory, literature tasks, and argument-agent findings into a concise user-facing framework discussion: thesis choices, 3-5 possible section spines, tradeoffs, and explicit decisions needed.",
        "outputs": "discussion_brief, thesis_options, section_spine_options, tradeoffs, user_questions, recommended_next_agent",
    },
    "framework_devils_advocate": {
        "title": "Framework Devil's Advocate",
        "mission": "Attack the proposed review architecture for frame-lock, catalogue drift, weak novelty, unsupported claims, and too many parallel headings; concede only when rebuttal is evidence-backed.",
        "outputs": "blocking_architecture_issues, strongest_objections, suggested_reframes, headings_to_merge_or_delete, concession_score",
    },
    "blackboard_curator": {
        "title": "Framework Blackboard Curator",
        "mission": "Maintain the shared framework board and material passport: what is decided, what is pending, which claims have evidence, which searches must run, and what requires user approval.",
        "outputs": "passport_updates, blackboard_updates, pending_decisions, evidence_gaps, next_stage_readiness",
    },
    "figure_table_designer": {
        "title": "Figure and Table Design Agent",
        "mission": "Design evidence-grounded figures, tables, and boxes that make the review's argument reusable and citable. Focus on argument role, structure, evidence requirements, image-prompt readiness, and human decisions; do not invent citations, numerical results, or unsupported visual claims.",
        "outputs": "display_item_inventory, figure_blueprints, table_blueprints, evidence_needed, human_decisions_needed, image_prompts_ready_for_review",
    },
    "synthesis_writer": {
        "title": "Evidence Synthesis Agent",
        "mission": "Synthesize consensus, contradictions, mechanisms, evidence gaps, and future agenda from verified sources without paper-by-paper narration.",
        "outputs": "synthesis_claims, evidence_chains, uncertainty_notes, section_draft_plan",
    },
    "methods_reporting_editor": {
        "title": "Methods and Reporting Standards Agent",
        "mission": "Align the review with PRISMA-style transparency when appropriate and with TRIPOD+AI, DECIDE-AI, SPIRIT-AI, CONSORT-AI, and related medical AI reporting standards.",
        "outputs": "reporting_checklist, missing_methods_details, journal_compliance_risks",
    },
    "ethics_regulatory_agent": {
        "title": "Ethics and Regulatory Agent",
        "mission": "Audit clinical safety, medical AI regulation, human oversight, data governance, AI-use disclosure, and domain-specific risk boundaries.",
        "outputs": "risk_register, governance_requirements, unsafe_claims, disclosure_notes",
    },
    "reviewer_auditor": {
        "title": "Top-Journal Reviewer Agent",
        "mission": "Simulate editor and reviewers, prioritize blocking issues, major revisions, missing evidence, overclaims, and journal-fit risks.",
        "outputs": "editor_decision, major_comments, minor_comments, revision_roadmap",
    },
}


DEFAULT_INPUTS = [
    "初稿.docx",
    "review-work/draft_text.txt",
    "review-data/03_framework/draft_memory/draft_memory_index.md",
    "review-data/03_framework/draft_memory/chief_editor_reading_checkpoint.md",
    "review-data/03_framework/draft_memory/agent_reading_packet.md",
    "review-data/03_framework/draft_memory/claims_for_discussion.csv",
    "review-data/03_framework/draft_memory/draft_memory.json",
    "review-data/02_literature/ai_candidate_mining/ai_candidate_paper_clues.csv",
    "review-data/02_literature/ai_candidate_mining/ai_pubmed_deep_dive_tasks.csv",
    "review-data/02_literature/ai_candidate_mining/human_ai_candidate_checkpoint.md",
    "review-data/02_literature/ai_candidate_mining/pubmed_deep_dive/ai_pubmed_abstract_screening_packet.md",
    "review-data/03_framework/logic_framework/draft_logic_framework.md",
    "review-data/03_framework/logic_framework/framework_evidence_blackboard.md",
    "review-data/03_framework/logic_framework/argument_evidence_map.csv",
    "review-data/03_framework/logic_framework/literature_search_tasks.csv",
    "review-data/03_framework/logic_framework/framework_material_passport.json",
    "review-data/03_framework/logic_framework/user_alignment_questions.md",
    "review-data/02_literature/argument_literature_expansion/argument_literature_plan.md",
    "review-data/02_literature/argument_literature_expansion/argument_literature_tasks.csv",
    "review-data/02_literature/argument_literature_expansion/abstract_screening_packet.md",
    "review-data/02_literature/argument_literature_expansion/claim_candidate_links.csv",
    "review-data/02_literature/argument_literature_expansion/human_argument_literature_checkpoint.md",
    "review-data/02_literature/candidate_board/literature_candidate_board.md",
    "review-data/02_literature/candidate_board/candidate_lane_summary.csv",
    "review-data/02_literature/candidate_board/screening_worklist_deduped.csv",
    "review-data/02_literature/candidate_board/verification_worklist_deduped.csv",
    "review-data/02_literature/candidate_board/agent_packets/agent_packet_index.md",
    "review-data/03_framework/display_items/display_item_plan.md",
    "review-data/03_framework/display_items/display_items.json",
    "review-data/03_framework/display_items/figure_blueprints.md",
    "review-data/03_framework/display_items/figure_image_prompts.md",
    "review-data/03_framework/display_items/table_blueprints.csv",
    "review-data/03_framework/display_items/table_evidence_gaps.csv",
    "review-data/03_framework/display_items/human_display_item_checkpoint.md",
    "review-data/02_literature/pool/contexts/final_citation_gate.md",
    "review-data/02_literature/pool/contexts/fulltext_audit.md",
    "review-data/02_literature/pool/contexts/needs_user_fulltext.md",
    "review-data/02_literature/pool/contexts/missing_fulltext_manifest.csv",
    "review-data/02_literature/pool/contexts/candidate_packet.md",
    "review-data/02_literature/pool/contexts/citation_context.md",
    "review-data/02_literature/pool/contexts/pool_audit.md",
    # Legacy fallback paths for projects created before schema 0.2.
    "review-work/draft_memory/draft_memory_index.md",
    "review-work/draft_memory/chief_editor_reading_checkpoint.md",
    "review-work/draft_memory/agent_reading_packet.md",
    "review-work/draft_memory/claims_for_discussion.csv",
    "review-work/draft_memory/draft_memory.json",
    "review-work/draft_logic_framework/draft_logic_framework.md",
    "review-work/draft_logic_framework/framework_evidence_blackboard.md",
    "review-work/draft_logic_framework/argument_evidence_map.csv",
    "review-work/draft_logic_framework/literature_search_tasks.csv",
    "review-work/draft_logic_framework/framework_material_passport.json",
    "review-work/draft_logic_framework/user_alignment_questions.md",
    "review-work/draft_audit/source_audit.md",
    "review-work/draft_audit/verified_papers.csv",
    "review-work/draft_audit/web_sources.csv",
    "review-work/pubmed_recall/run_summary.md",
    "review-work/metadata_recall/run_summary.md",
    "review-work/preprint_recall/run_summary.md",
    "review-work/literature_pool/contexts/final_citation_gate.md",
    "review-work/literature_pool/contexts/fulltext_audit.md",
    "review-work/literature_pool/contexts/needs_user_fulltext.md",
    "review-work/literature_pool/contexts/missing_fulltext_manifest.csv",
    "review-work/literature_pool/contexts/candidate_packet.md",
    "review-work/literature_pool/contexts/citation_context.md",
    "review-work/literature_pool/contexts/pool_audit.md",
    "文献池/contexts/fulltext_audit.md",
    "文献池/contexts/needs_user_fulltext.md",
    "文献池/contexts/missing_fulltext_manifest.csv",
    "文献池/contexts/citation_context.md",
    "文献池/contexts/pool_audit.md",
]

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_AGENT_MEMORY_DIR = "review-data/06_agent_memory"
DEFAULT_LOCAL_ENV_FILES = (".env.local", ".env")


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def clean_space(text: str) -> str:
    return " ".join((text or "").split())


def read_docx(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            xml = zf.read("word/document.xml")
        root = ET.fromstring(xml)
    except Exception:
        return f"[Could not extract Word text from {path}]"
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    parts: list[str] = []
    for para in root.findall(".//w:p", ns):
        runs = [node.text or "" for node in para.findall(".//w:t", ns)]
        line = clean_space("".join(runs))
        if line:
            parts.append(line)
    return "\n".join(parts)


def read_text(path: Path, max_chars: int) -> str:
    if not path.exists() or path.is_dir():
        return ""
    try:
        if path.suffix.lower() == ".docx":
            return read_docx(path)[:max_chars]
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except Exception:
        return f"[Could not read text from {path}]"


def resolve_project_path(project_dir: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    parts = [part.lower() for part in path.parts if part not in {".", ""}]
    if parts and parts[0] == project_dir.name.lower():
        return path.resolve()
    candidate = (project_dir / path).resolve()
    cwd_candidate = path.resolve()
    if cwd_candidate.exists() and not candidate.exists():
        return cwd_candidate
    return candidate


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists() or path.is_dir():
        return values
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def is_placeholder_secret(value: str) -> bool:
    if not value:
        return True
    lowered = value.strip().lower()
    return any(token in lowered for token in ["your_", "example", "placeholder", "replace", "<", ">"])


def project_env_values(project_dir: Path, env_file: str = "") -> tuple[dict[str, str], list[str]]:
    paths: list[Path] = []
    if env_file:
        paths.append(resolve_project_path(project_dir, env_file))
    else:
        paths.extend(project_dir / name for name in DEFAULT_LOCAL_ENV_FILES)
    merged: dict[str, str] = {}
    loaded: list[str] = []
    for path in paths:
        parsed = parse_env_file(path)
        if not parsed:
            continue
        loaded.append(str(path))
        merged.update(parsed)
    return merged, loaded


def collect_inputs(project_dir: Path, extra_inputs: list[str], max_chars: int, no_default_inputs: bool = False) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for raw in [] if no_default_inputs else DEFAULT_INPUTS:
        path = (project_dir / raw).resolve()
        key = str(path)
        if key in seen or not path.exists() or path.is_dir():
            continue
        seen.add(key)
        out.append({"path": str(path), "content": read_text(path, max_chars)})
    for raw in extra_inputs:
        path = resolve_project_path(project_dir, raw)
        key = str(path)
        if key in seen or not path.exists() or path.is_dir():
            continue
        seen.add(key)
        out.append({"path": str(path), "content": read_text(path, max_chars)})
    return out


def read_tail(path: Path, max_chars: int) -> str:
    if not path.exists() or path.is_dir():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def collect_agent_memory(project_dir: Path, agent_name: str, memory_dir: str, max_chars: int) -> list[dict[str, str]]:
    root = Path(memory_dir)
    if not root.is_absolute():
        root = project_dir / root
    shared = root / "shared"
    agent_root = root / "agents" / agent_name
    candidates = [
        shared / "project_memory.md",
        shared / "decisions.md",
        shared / "open_questions.md",
        shared / "style_and_scope.md",
        shared / "handoff.md",
        agent_root / "profile.md",
        agent_root / "memory.md",
        agent_root / "handoff.md",
        agent_root / "retrieval_plan.md",
        agent_root / "events.jsonl",
    ]
    out: list[dict[str, str]] = []
    for path in candidates:
        if not path.exists() or path.is_dir():
            continue
        content = read_tail(path, max_chars) if path.name == "events.jsonl" else read_text(path, max_chars)
        if content.strip():
            out.append({"path": str(path), "content": content})
    return out


def memory_blocks(memory_items: list[dict[str, str]]) -> str:
    blocks = []
    for item in memory_items:
        blocks.append(f"### MEMORY: {item['path']}\n\n{item['content']}")
    return "\n\n".join(blocks)


def agent_prompt(agent_name: str, agent: dict[str, str], inputs: list[dict[str, str]], user_goal: str, memory_items: list[dict[str, str]] | None = None) -> str:
    context_blocks = []
    for item in inputs:
        if item["content"]:
            context_blocks.append(f"### FILE: {item['path']}\n\n{item['content']}")
        else:
            context_blocks.append(f"### FILE: {item['path']}\n\n[Binary or unreadable; inspect locally if needed.]")
    context = "\n\n".join(context_blocks)
    memory_context = memory_blocks(memory_items or []) or "[No persistent agent memory loaded.]"
    return textwrap.dedent(
        f"""
        You are the {agent['title']} in a top-journal review-writing workflow.

        Mission:
        {agent['mission']}

        User/project goal:
        {user_goal}

        Operating rules:
        - Do not fabricate citations, DOIs, authors, results, or paper claims.
        - Use final manuscript citations only from formally published papers or official guidelines/standards unless explicitly told otherwise.
        - Treat arXiv/OpenReview-only records as leads/background, not final references.
        - Separate evidence from interpretation.
        - If drafts are present, do not search from the broad topic alone. Tie every retrieval or screening recommendation to a draft-derived claim, historical bridge, current evidence need, future-agenda point, figure/table need, or explicit user question.
        - Keep new papers that are interesting but off-map in the supplemental pool as background_or_defer; do not let them rewrite the review spine before user discussion.
        - Stay inside your role. If downstream writing, source retrieval, or final audit is needed, recommend the next agent instead of doing that work yourself.
        - Use the framework blackboard/passport as the shared state when present; do not silently overwrite user decisions.
        - Use persistent agent memory as role-specific state. If memory conflicts with newer files, flag the conflict and prefer the newer verified source.
        - Return concrete memory updates that Codex should write back to your agent memory after this run.
        - Produce concise, actionable output grounded in the provided files.

        Required output fields:
        {agent['outputs']}

        Return Markdown with:
        1. Blocking issues
        2. Findings
        3. Recommended actions
        4. Commands or files to run/read next
        5. Evidence gaps
        6. Memory updates to write back

        Persistent agent memory:

        {memory_context}

        Context files:

        {context}
        """
    ).strip()


def write_packets(args: argparse.Namespace, agents: list[str]) -> dict[str, Any]:
    project_dir = Path(args.project_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    packet_dir = out_dir / "agent_packets"
    packet_dir.mkdir(parents=True, exist_ok=True)
    inputs = collect_inputs(project_dir, args.input, args.max_chars_per_file, args.no_default_inputs)
    manifest = {
        "created_at": now_iso(),
        "project_dir": str(project_dir),
        "out_dir": str(out_dir),
        "user_goal": args.goal,
        "agents": [],
        "inputs": [{"path": item["path"], "chars": len(item["content"])} for item in inputs],
    }
    for agent_name in agents:
        agent = DEFAULT_AGENTS[agent_name]
        agent_memory = [] if args.no_agent_memory else collect_agent_memory(project_dir, agent_name, args.agent_memory_dir, args.max_agent_memory_chars)
        prompt = agent_prompt(agent_name, agent, inputs, args.goal, agent_memory)
        packet_path = packet_dir / f"{agent_name}.md"
        packet_path.write_text(prompt, encoding="utf-8")
        manifest["agents"].append(
            {
                "name": agent_name,
                "title": agent["title"],
                "packet": str(packet_path),
                "mission": agent["mission"],
                "memory_files": [{"path": item["path"], "chars": len(item["content"])} for item in agent_memory],
            }
        )
    manifest_path = out_dir / "agent_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def api_config(args: argparse.Namespace) -> dict[str, Any]:
    project_dir = Path(getattr(args, "project_dir", ".")).resolve()
    env_values, loaded_env_files = project_env_values(project_dir, getattr(args, "env_file", ""))
    api_key = (
        args.api_key
        or os.environ.get("REVIEW_AGENT_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or env_values.get("REVIEW_AGENT_API_KEY")
        or env_values.get("DEEPSEEK_API_KEY")
        or ""
    )
    base_url = (
        args.base_url
        or os.environ.get("REVIEW_AGENT_BASE_URL")
        or os.environ.get("DEEPSEEK_BASE_URL")
        or env_values.get("REVIEW_AGENT_BASE_URL")
        or env_values.get("DEEPSEEK_BASE_URL")
        or DEFAULT_BASE_URL
    )
    model = (
        args.model
        or os.environ.get("REVIEW_AGENT_MODEL")
        or os.environ.get("DEEPSEEK_MODEL")
        or env_values.get("REVIEW_AGENT_MODEL")
        or env_values.get("DEEPSEEK_MODEL")
        or DEFAULT_MODEL
    )
    if is_placeholder_secret(api_key):
        api_key = ""
    return {"api_key": api_key, "base_url": base_url.rstrip("/"), "model": model, "loaded_env_files": loaded_env_files}


def chat_completion(prompt: str, args: argparse.Namespace) -> str:
    cfg = api_config(args)
    if not cfg["api_key"]:
        raise SystemExit("No external API key. Default to Codex-subtask packet mode with `plan`, or set REVIEW_AGENT_API_KEY/DEEPSEEK_API_KEY only after external calls are approved.")
    url = cfg["base_url"] + "/chat/completions"
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": "You are a rigorous specialist research agent assisting Codex."},
            {"role": "user", "content": prompt},
        ],
        "temperature": args.temperature,
    }
    if args.thinking != "omit":
        payload["thinking"] = {"type": args.thinking}
    if args.reasoning_effort:
        payload["reasoning_effort"] = args.reasoning_effort
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
    body = json.loads(raw)
    return body["choices"][0]["message"]["content"]


def selected_agents(raw: str) -> list[str]:
    if raw.strip().lower() in {"all", "*"}:
        return list(DEFAULT_AGENTS.keys())
    agents = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [agent for agent in agents if agent not in DEFAULT_AGENTS]
    if unknown:
        raise SystemExit(f"Unknown agent(s): {', '.join(unknown)}. Known: {', '.join(DEFAULT_AGENTS)}")
    return agents


def record_agent_result(project_dir: Path, memory_dir: str, agent_name: str, result_path: Path, output: str, max_chars: int) -> None:
    root = Path(memory_dir)
    if not root.is_absolute():
        root = project_dir / root
    folder = root / "agents" / agent_name
    folder.mkdir(parents=True, exist_ok=True)

    note = output[:max_chars]
    event = {
        "created_at": now_iso(),
        "agent": agent_name,
        "kind": "result",
        "source": str(result_path),
        "note": note,
    }
    with (folder / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    handoff_path = folder / "handoff.md"
    if not handoff_path.exists():
        handoff_path.write_text("# Agent Handoff\n\n", encoding="utf-8")
    with handoff_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n## result | {now_iso()}\n\nSource result: `{result_path}`\n\n{note.strip()}\n")

    state_path = folder / "state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
    else:
        state = {}
    state.setdefault("created_at", now_iso())
    state.update(
        {
            "schema_version": "0.1",
            "agent": agent_name,
            "updated_at": now_iso(),
            "status": "result_recorded",
            "last_result": str(result_path),
        }
    )
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cmd_plan(args: argparse.Namespace) -> int:
    agents = selected_agents(args.agents)
    manifest = write_packets(args, agents)
    print(json.dumps({"manifest": str(Path(args.out_dir).resolve() / "agent_manifest.json"), "agents": len(manifest["agents"])}, ensure_ascii=False))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    agents = selected_agents(args.agents)
    manifest = write_packets(args, agents)
    out_dir = Path(args.out_dir).resolve()
    result_dir = out_dir / "agent_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for agent in manifest["agents"]:
        prompt = Path(agent["packet"]).read_text(encoding="utf-8")
        if args.packet_only:
            output = "[Codex-subtask packet mode: no external LLM call]"
        else:
            output = chat_completion(prompt, args)
        result_path = result_dir / f"{agent['name']}.md"
        result_path.write_text(output, encoding="utf-8")
        if not args.no_agent_memory and args.record_agent_results and output and not args.packet_only:
            record_agent_result(Path(args.project_dir).resolve(), args.agent_memory_dir, agent["name"], result_path, output, args.max_agent_result_memory_chars)
        results.append({"agent": agent["name"], "result": str(result_path)})
    (out_dir / "agent_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"results": results, "packet_only": args.packet_only}, ensure_ascii=False))
    return 0


def cmd_verify_config(args: argparse.Namespace) -> int:
    cfg = api_config(args)
    handoff = ""
    if args.write_handoff and not cfg["api_key"]:
        handoff_path = Path(args.handoff_out)
        if not handoff_path.is_absolute():
            handoff_path = Path(args.project_dir).resolve() / handoff_path
        handoff_path.parent.mkdir(parents=True, exist_ok=True)
        handoff_path.write_text(
            "\n".join(
                [
                    "# External LLM Key Handoff",
                    "",
                    "No `DEEPSEEK_API_KEY` or `REVIEW_AGENT_API_KEY` was detected.",
                    "",
                    "No key is required for the default Codex-subtask workflow. Use `plan`, `packet`, or `build-cases` commands and let Codex subtasks read the generated packets.",
                    "",
                    "Configure an external key only if the user explicitly approves external LLM calls and data egress.",
                    "",
                    "Optional external variables:",
                    "",
                    "1. Set environment variables before running external agents.",
                    "2. Or create a local `.env.local` file that is not committed.",
                    "",
                    "Generic provider variables:",
                    "",
                    "```text",
                    "REVIEW_AGENT_API_KEY=<your-key>",
                    "REVIEW_AGENT_BASE_URL=<provider-base-url>",
                    "REVIEW_AGENT_MODEL=<provider-model>",
                    "```",
                    "",
                    "DeepSeek-compatible example:",
                    "",
                    "```text",
                    "DEEPSEEK_API_KEY=<your-key>",
                    "DEEPSEEK_BASE_URL=https://api.deepseek.com",
                    "DEEPSEEK_MODEL=deepseek-v4-pro",
                    "```",
                    "",
                    "Do not paste real keys into the chat transcript, manuscript files, logs, or Git commits.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        handoff = str(handoff_path)
    print(
        json.dumps(
            {
                "base_url": cfg["base_url"],
                "model": cfg["model"],
                "has_api_key": bool(cfg["api_key"]),
                "loaded_env_files": cfg["loaded_env_files"],
                "handoff": handoff,
            },
            ensure_ascii=False,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and optionally run specialist review-writing agents.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--project-dir", default=".")
        p.add_argument("--out-dir", default="./review-work/agent_orchestration")
        p.add_argument("--agents", default="all", help="Comma-separated agent names or all.")
        p.add_argument("--goal", default="Optimize a top-journal review with verified evidence and published-only final citations.")
        p.add_argument("--input", action="append", default=[], help="Extra input file path; repeatable.")
        p.add_argument("--no-default-inputs", action="store_true", help="Use only files passed with --input.")
        p.add_argument("--max-chars-per-file", type=int, default=12000)
        p.add_argument("--agent-memory-dir", default=DEFAULT_AGENT_MEMORY_DIR)
        p.add_argument("--no-agent-memory", action="store_true", help="Do not load persistent shared/per-agent memory into packets.")
        p.add_argument("--max-agent-memory-chars", type=int, default=6000)

    p_plan = sub.add_parser("plan", help="Create specialist agent packets without calling an external LLM.")
    common(p_plan)
    p_plan.set_defaults(func=cmd_plan)

    p_run = sub.add_parser("run", help="Create packets and optionally call an OpenAI-compatible LLM endpoint.")
    common(p_run)
    p_run.add_argument("--packet-only", action="store_true")
    p_run.add_argument("--base-url", default="", help=f"Optional external OpenAI-compatible base URL. Fallback when external mode is approved: {DEFAULT_BASE_URL}.")
    p_run.add_argument("--api-key", default="")
    p_run.add_argument("--model", default="", help=f"Optional external model name. Fallback when external mode is approved: {DEFAULT_MODEL}.")
    p_run.add_argument("--env-file", default="", help="Optional project-relative or absolute env file. Defaults to .env.local then .env.")
    p_run.add_argument("--temperature", type=float, default=0.2)
    p_run.add_argument("--thinking", default="enabled", choices=["enabled", "disabled", "omit"], help="Optional external-provider thinking mode. Defaults to enabled.")
    p_run.add_argument("--reasoning-effort", default="high", choices=["", "low", "medium", "high"], help="Reasoning effort for supported models. Defaults to high.")
    p_run.add_argument("--timeout", type=int, default=180)
    p_run.add_argument("--record-agent-results", action=argparse.BooleanOptionalAction, default=True)
    p_run.add_argument("--max-agent-result-memory-chars", type=int, default=8000)
    p_run.set_defaults(func=cmd_run)

    p_verify = sub.add_parser("verify-config", help="Show external LLM API config without making a request.")
    p_verify.add_argument("--project-dir", default=".")
    p_verify.add_argument("--base-url", default="", help=f"Optional external OpenAI-compatible base URL. Fallback when external mode is approved: {DEFAULT_BASE_URL}.")
    p_verify.add_argument("--api-key", default="")
    p_verify.add_argument("--model", default="", help=f"Optional external model name. Fallback when external mode is approved: {DEFAULT_MODEL}.")
    p_verify.add_argument("--env-file", default="", help="Optional project-relative or absolute env file. Defaults to .env.local then .env.")
    p_verify.add_argument("--write-handoff", action="store_true", help="Write a local instructions file when no API key is detected.")
    p_verify.add_argument("--handoff-out", default="./review-data/05_audit/review_agent_key_handoff.md")
    p_verify.set_defaults(func=cmd_verify_config)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
