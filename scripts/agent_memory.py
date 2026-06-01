#!/usr/bin/env python3
"""Persistent per-agent memory for complex review-writing workflows.

Each specialist agent gets a small, curated memory folder. Codex can update
these memories between runs and the orchestrator can dynamically load only the
relevant agent's memory into that agent's packet.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "0.1"
DEFAULT_MEMORY_DIR = "review-data/06_agent_memory"


AGENT_PROFILES: dict[str, dict[str, str]] = {
    "draft_deep_reader": {
        "title": "Draft Deep-Reading Agent",
        "focus": "Original draft arguments, neglected insights, section logic, and user-supplied framing.",
    },
    "draft_memory_curator": {
        "title": "Draft Memory Curator",
        "focus": "Accepted/rejected/pending draft ideas and links between draft cards and framework blackboard.",
    },
    "draft_frame_reader": {
        "title": "Multi-Draft Frame Reader",
        "focus": "Cross-draft consensus, unique frames, weak frames, thesis candidates, and heading compression.",
    },
    "literature_strategist": {
        "title": "Literature Strategy Agent",
        "focus": "Search plans, evidence sufficiency criteria, source hierarchy, and framework-driven recall gaps.",
    },
    "literature_retriever": {
        "title": "Literature Retrieval Agent",
        "focus": "Database-specific searches, source replacements, missing evidence, and high-value papers.",
    },
    "literature_screener": {
        "title": "Literature Inclusion/Exclusion Agent",
        "focus": "Screening decisions, inclusion/exclusion rationales, and citation-pool candidates.",
    },
    "literature_manager": {
        "title": "Literature Memory Manager",
        "focus": "Pool health, card coverage, full-text handoffs, RAG readiness, and next commands.",
    },
    "citation_verifier": {
        "title": "Final Citation Traceback Agent",
        "focus": "Citation metadata, claim-source fit, published-only gate, and required rewrites.",
    },
    "outline_architect": {
        "title": "Review Architecture Agent",
        "focus": "Central thesis, 3-5 section spine, transition logic, and scope controls.",
    },
    "argument_builder": {
        "title": "Argument Builder Agent",
        "focus": "Claim-evidence-reasoning chains, counterarguments, evidence strength, and claims to demote.",
    },
    "framework_dialogue_moderator": {
        "title": "Framework Dialogue Moderator",
        "focus": "User-facing framework choices, tradeoffs, decisions, and discussion handoff.",
    },
    "framework_devils_advocate": {
        "title": "Framework Devil's Advocate",
        "focus": "Frame-lock, weak novelty, unsupported sections, overbreadth, and architecture objections.",
    },
    "blackboard_curator": {
        "title": "Framework Blackboard Curator",
        "focus": "Shared blackboard state, material passport, pending decisions, and stage readiness.",
    },
    "figure_table_designer": {
        "title": "Figure and Table Design Agent",
        "focus": "Evidence-grounded display-item inventory, figure prompts for conceptual drafts, table blueprints, evidence gaps, human decisions, and citation-safe visual claims.",
    },
    "synthesis_writer": {
        "title": "Evidence Synthesis Agent",
        "focus": "Consensus, contradictions, mechanisms, uncertainty, and section synthesis plans.",
    },
    "methods_reporting_editor": {
        "title": "Methods and Reporting Standards Agent",
        "focus": "Reporting standards, review transparency, medical AI checklists, and compliance risks.",
    },
    "ethics_regulatory_agent": {
        "title": "Ethics and Regulatory Agent",
        "focus": "Clinical safety, regulation, human oversight, AI-use disclosure, and domain-specific risks.",
    },
    "reviewer_auditor": {
        "title": "Top-Journal Reviewer Agent",
        "focus": "Editor/reviewer simulation, blocking issues, major comments, and revision roadmap.",
    },
}


SHARED_FILES = {
    "project_memory.md": "# Project Memory\n\nPersistent project-level facts, scope decisions, and user preferences.\n",
    "decisions.md": "# Decisions\n\nRecord approved theses, section spines, citation policies, journal targets, and rejected options.\n",
    "open_questions.md": "# Open Questions\n\nQuestions requiring user input, literature retrieval, or agent follow-up.\n",
    "style_and_scope.md": "# Style And Scope\n\nTarget journal family, tone, audience, section constraints, and terminology preferences.\n",
    "handoff.md": "# Shared Handoff\n\nCurrent stage, next stage, blockers, and files to load first.\n",
}


SHARED_KINDS = [
    "observation",
    "decision",
    "handoff",
    "question",
    "style",
    "scope",
    "claim",
    "figure",
    "audit",
    "result",
]


AGENT_FILES = {
    "memory.md": "# Agent Memory\n\nDurable role-specific observations and decisions.\n",
    "handoff.md": "# Agent Handoff\n\nWhat the next run of this agent must know first.\n",
    "retrieval_plan.md": "# Retrieval Plan\n\nFiles, cards, RAG queries, and blackboard sections this agent should load dynamically.\n",
}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def safe_agent_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    if not value:
        raise SystemExit("Agent name cannot be empty.")
    return value


def selected_agents(raw: str) -> list[str]:
    if raw.strip().lower() in {"all", "*"}:
        return list(AGENT_PROFILES.keys())
    agents = [safe_agent_name(item) for item in raw.split(",") if item.strip()]
    return agents


def memory_root(project_dir: Path, memory_dir: str) -> Path:
    path = Path(memory_dir)
    return path if path.is_absolute() else project_dir / path


def agent_dir(root: Path, agent: str) -> Path:
    return root / "agents" / safe_agent_name(agent)


def write_if_missing(path: Path, content: str, force: bool = False) -> str:
    if path.exists() and not force:
        return f"exists: {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"wrote: {path}"


def profile_content(agent: str) -> str:
    profile = AGENT_PROFILES.get(agent, {"title": agent, "focus": "Custom specialist agent."})
    return (
        f"# {profile['title']}\n\n"
        f"- Agent id: `{agent}`\n"
        f"- Memory schema: {SCHEMA_VERSION}\n"
        f"- Role focus: {profile['focus']}\n\n"
        "## Dynamic Loading Rule\n\n"
        "Load only memory relevant to the current decision. If memory conflicts with current blackboard/passport/evidence, flag the conflict instead of silently overwriting it.\n"
    )


def initial_state(agent: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "agent": agent,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "status": "initialized",
        "last_result": "",
        "open_items": [],
    }


def cmd_init(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    root = memory_root(project_dir, args.memory_dir)
    agents = selected_agents(args.agents)
    actions = []
    if args.dry_run:
        actions.append(f"would mkdir: {root}")
    else:
        root.mkdir(parents=True, exist_ok=True)
    for name, content in SHARED_FILES.items():
        path = root / "shared" / name
        actions.append(f"would write: {path}" if args.dry_run else write_if_missing(path, content, args.force))
    schema_path = root / "agent_memory_schema.json"
    schema = {"schema_version": SCHEMA_VERSION, "agents": AGENT_PROFILES, "shared_files": list(SHARED_FILES)}
    if args.dry_run:
        actions.append(f"would write: {schema_path}")
    else:
        actions.append(write_if_missing(schema_path, json.dumps(schema, ensure_ascii=False, indent=2) + "\n", args.force))
    for agent in agents:
        folder = agent_dir(root, agent)
        if args.dry_run:
            actions.append(f"would mkdir: {folder}")
        else:
            folder.mkdir(parents=True, exist_ok=True)
        actions.append(f"would write: {folder / 'profile.md'}" if args.dry_run else write_if_missing(folder / "profile.md", profile_content(agent), args.force))
        for filename, content in AGENT_FILES.items():
            actions.append(f"would write: {folder / filename}" if args.dry_run else write_if_missing(folder / filename, content, args.force))
        state_path = folder / "state.json"
        events_path = folder / "events.jsonl"
        if args.dry_run:
            actions.append(f"would write: {state_path}")
            actions.append(f"would write: {events_path}")
        else:
            actions.append(write_if_missing(state_path, json.dumps(initial_state(agent), ensure_ascii=False, indent=2) + "\n", args.force))
            actions.append(write_if_missing(events_path, "", args.force))
    print(json.dumps({"memory_dir": str(root), "agents": agents, "actions": actions}, ensure_ascii=False, indent=2))
    return 0


def load_state(folder: Path, agent: str) -> dict[str, Any]:
    path = folder / "state.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return initial_state(agent)


def save_state(folder: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    (folder / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_event(folder: Path, event: dict[str, Any]) -> None:
    event = {"created_at": now_iso(), **event}
    with (folder / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def append_markdown(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n## {title}\n\n{body.strip()}\n")


def cmd_append(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    root = memory_root(project_dir, args.memory_dir)
    agent = safe_agent_name(args.agent)
    folder = agent_dir(root, agent)
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / "profile.md").exists():
        write_if_missing(folder / "profile.md", profile_content(agent))
    for filename, content in AGENT_FILES.items():
        write_if_missing(folder / filename, content)
    note = args.note.strip()
    if not note and args.from_file:
        note = Path(args.from_file).read_text(encoding="utf-8", errors="replace")[: args.max_chars]
    if not note:
        raise SystemExit("Provide --note or --from-file.")
    event = {"agent": agent, "kind": args.kind, "source": args.source, "note": note}
    append_event(folder, event)
    target = folder / ("handoff.md" if args.kind == "handoff" else "memory.md")
    append_markdown(target, f"{args.kind} | {now_iso()}", f"Source: {args.source or '[unspecified]'}\n\n{note}")
    state = load_state(folder, agent)
    state["status"] = "updated"
    if args.kind == "handoff":
        state["last_handoff"] = note[:500]
    save_state(folder, state)
    print(json.dumps({"agent": agent, "target": str(target), "event": str(folder / "events.jsonl")}, ensure_ascii=False))
    return 0


def cmd_append_shared(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    root = memory_root(project_dir, args.memory_dir)
    shared = root / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    for filename, content in SHARED_FILES.items():
        write_if_missing(shared / filename, content)
    note = args.note.strip()
    if not note and args.from_file:
        note = Path(args.from_file).read_text(encoding="utf-8", errors="replace")[: args.max_chars]
    if not note:
        raise SystemExit("Provide --note or --from-file.")
    target = shared / args.file
    event = {"scope": "shared", "kind": args.kind, "file": args.file, "source": args.source, "note": note}
    append_event(shared, event)
    append_markdown(target, f"{args.kind} | {now_iso()}", f"Source: {args.source or '[unspecified]'}\n\n{note}")
    print(json.dumps({"scope": "shared", "target": str(target), "event": str(shared / "events.jsonl")}, ensure_ascii=False))
    return 0


def read_limited(path: Path, max_chars: int) -> str:
    if not path.exists() or path.is_dir():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[:max_chars]


def recent_events(folder: Path, max_events: int, max_chars: int) -> str:
    path = folder / "events.jsonl"
    if not path.exists():
        return ""
    lines = [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    rows = []
    for line in lines[-max_events:]:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        note = str(event.get("note", ""))[:max_chars]
        rows.append(f"- {event.get('created_at', '')} [{event.get('kind', '')}] {event.get('source', '')}: {note}")
    return "\n".join(rows)


def packet_text(root: Path, agent: str, max_chars: int, max_events: int) -> str:
    folder = agent_dir(root, agent)
    shared = root / "shared"
    blocks = [
        ("Shared project memory", read_limited(shared / "project_memory.md", max_chars)),
        ("Shared decisions", read_limited(shared / "decisions.md", max_chars)),
        ("Shared open questions", read_limited(shared / "open_questions.md", max_chars)),
        ("Style and scope", read_limited(shared / "style_and_scope.md", max_chars)),
        ("Shared handoff", read_limited(shared / "handoff.md", max_chars)),
        ("Agent profile", read_limited(folder / "profile.md", max_chars)),
        ("Agent memory", read_limited(folder / "memory.md", max_chars)),
        ("Agent handoff", read_limited(folder / "handoff.md", max_chars)),
        ("Agent retrieval plan", read_limited(folder / "retrieval_plan.md", max_chars)),
        ("Recent agent events", recent_events(folder, max_events, max_chars // 2)),
    ]
    lines = [f"# Agent Memory Packet: {agent}", "", "Use this as persistent state, not as verified evidence. Prefer current blackboard/passport when conflicts arise.", ""]
    for title, body in blocks:
        if body.strip():
            lines.extend([f"## {title}", "", body.strip(), ""])
    return "\n".join(lines).strip() + "\n"


def cmd_packet(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    root = memory_root(project_dir, args.memory_dir)
    agent = safe_agent_name(args.agent)
    text = packet_text(root, agent, args.max_chars, args.max_events)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(json.dumps({"agent": agent, "out": str(out), "chars": len(text)}, ensure_ascii=False))
    else:
        print(text)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    root = memory_root(project_dir, args.memory_dir)
    rows = []
    for agent in selected_agents(args.agents):
        folder = agent_dir(root, agent)
        state = load_state(folder, agent) if folder.exists() else {"status": "missing"}
        rows.append(
            {
                "agent": agent,
                "title": AGENT_PROFILES.get(agent, {}).get("title", agent),
                "folder": str(folder),
                "exists": folder.exists(),
                "status": state.get("status", "missing"),
                "updated_at": state.get("updated_at", ""),
            }
        )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def cmd_ingest_result(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    root = memory_root(project_dir, args.memory_dir)
    result_path = Path(args.result)
    if not result_path.exists():
        raise SystemExit(f"Result file not found: {result_path}")
    note = result_path.read_text(encoding="utf-8", errors="replace")[: args.max_chars]
    folder = agent_dir(root, args.agent)
    folder.mkdir(parents=True, exist_ok=True)
    append_event(folder, {"agent": args.agent, "kind": "result", "source": str(result_path), "note": note})
    append_markdown(folder / "handoff.md", f"result | {now_iso()}", f"Source result: `{result_path}`\n\n{note}")
    state = load_state(folder, args.agent)
    state["last_result"] = str(result_path)
    state["status"] = "result_ingested"
    save_state(folder, state)
    print(json.dumps({"agent": args.agent, "ingested": str(result_path), "handoff": str(folder / "handoff.md")}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage persistent per-agent memory for review-writing workflows.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--project-dir", default=".")
        p.add_argument("--memory-dir", default=DEFAULT_MEMORY_DIR)

    p_init = sub.add_parser("init", help="Create shared and per-agent memory folders.")
    common(p_init)
    p_init.add_argument("--agents", default="all")
    p_init.add_argument("--force", action="store_true")
    p_init.add_argument("--dry-run", action="store_true")
    p_init.set_defaults(func=cmd_init)

    p_append = sub.add_parser("append", help="Append an observation, decision, handoff, or question to one agent's memory.")
    common(p_append)
    p_append.add_argument("--agent", required=True)
    p_append.add_argument("--kind", default="observation", choices=["observation", "decision", "handoff", "question", "claim", "figure", "audit", "result"])
    p_append.add_argument("--note", default="")
    p_append.add_argument("--from-file", default="")
    p_append.add_argument("--source", default="")
    p_append.add_argument("--max-chars", type=int, default=6000)
    p_append.set_defaults(func=cmd_append)

    p_append_shared = sub.add_parser("append-shared", help="Append a curated note to shared project memory.")
    common(p_append_shared)
    p_append_shared.add_argument("--file", required=True, choices=sorted(SHARED_FILES))
    p_append_shared.add_argument("--kind", default="observation", choices=SHARED_KINDS)
    p_append_shared.add_argument("--note", default="")
    p_append_shared.add_argument("--from-file", default="")
    p_append_shared.add_argument("--source", default="")
    p_append_shared.add_argument("--max-chars", type=int, default=6000)
    p_append_shared.set_defaults(func=cmd_append_shared)

    p_packet = sub.add_parser("packet", help="Build a dynamic memory packet for one agent.")
    common(p_packet)
    p_packet.add_argument("--agent", required=True)
    p_packet.add_argument("--out", default="")
    p_packet.add_argument("--max-chars", type=int, default=6000)
    p_packet.add_argument("--max-events", type=int, default=12)
    p_packet.set_defaults(func=cmd_packet)

    p_list = sub.add_parser("list", help="List agent memory folders and status.")
    common(p_list)
    p_list.add_argument("--agents", default="all")
    p_list.set_defaults(func=cmd_list)

    p_ingest = sub.add_parser("ingest-result", help="Append an agent result file to that agent's handoff memory.")
    common(p_ingest)
    p_ingest.add_argument("--agent", required=True)
    p_ingest.add_argument("--result", required=True)
    p_ingest.add_argument("--max-chars", type=int, default=8000)
    p_ingest.set_defaults(func=cmd_ingest_result)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
