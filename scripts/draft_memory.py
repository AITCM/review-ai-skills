#!/usr/bin/env python3
"""Build persistent memory cards from GPT/Gemini/Deep Research drafts.

This script is a companion to draft_logic_framework.py. The framework script
creates the argument map and architecture brief; this script creates reusable
memory so Codex and specialist agents can return to what each draft actually
said during framework discussions and revisions.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sqlite3
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


SUPPORTED_EXTENSIONS = {".md", ".txt", ".docx"}


@dataclass
class DraftMemory:
    memory_id: str
    path: str
    title: str
    chars: int
    top_headings: list[str]
    section_preview: list[dict[str, str]]
    thesis_claims: list[str]
    evidence_claims: list[str]
    limitation_claims: list[str]
    literature_leads: list[str]
    reusable_insights: list[str]
    risks: list[str]
    card_path: str = ""


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def clean_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def safe_filename(value: str, fallback: str = "draft") -> str:
    text = re.sub(r"[^A-Za-z0-9_.\-\u4e00-\u9fff]+", "-", value or fallback).strip("-")
    return text[:100] or fallback


def read_docx(path: Path) -> str:
    parts: list[str] = []
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    for para in root.findall(".//w:p", ns):
        runs = [node.text or "" for node in para.findall(".//w:t", ns)]
        line = clean_space("".join(runs))
        if line:
            parts.append(line)
    return "\n".join(parts)


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        return read_docx(path)
    return path.read_text(encoding="utf-8", errors="replace")


def discover_drafts(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for raw in args.draft:
        path = Path(raw).resolve()
        if path.exists() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            paths.append(path)
    for raw in args.draft_dir:
        root = Path(raw).resolve()
        if root.exists():
            paths.extend(p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def infer_title(lines: list[str], fallback: str) -> str:
    for line in lines[:40]:
        text = clean_space(re.sub(r"^#+\s*", "", line))
        if text and len(text) <= 180 and text.lower() not in {"abstract", "summary", "references"}:
            return text
    return fallback


def parse_heading(line: str) -> tuple[int, str] | None:
    raw = clean_space(line)
    if not raw:
        return None
    match = re.match(r"^(#{1,6})\s+(.+)$", raw)
    if match:
        return len(match.group(1)), clean_space(match.group(2))
    match = re.match(r"^([0-9]+(?:\.[0-9]+){0,3})\s+(.+)$", raw)
    if match and len(match.group(2)) <= 140:
        return 1 + match.group(1).count("."), clean_space(match.group(2))
    if len(raw) <= 80 and re.search(r"(框架|架构|证据|评估|治理|挑战|未来|agent|framework|evidence|evaluation|governance)", raw, re.I):
        return 2, raw
    return None


def split_sentences(text: str) -> list[str]:
    compact = re.sub(r"\n+", " ", text)
    parts = re.split(r"(?<=[。！？!?])\s+", compact)
    return [clean_space(part) for part in parts if 35 <= len(clean_space(part)) <= 420]


def classify_claim(sentence: str) -> str:
    if re.search(r"挑战|限制|风险|不足|鸿沟|gap|risk|limitation|challenge", sentence, re.I):
        return "limitation"
    if re.search(r"显示|表明|证明|发现|数据|结果|shows?|demonstrates?|evidence|found", sentence, re.I):
        return "evidence"
    if re.search(r"提出|认为|框架|架构|范式|核心|关键|argues?|proposes?|framework|architecture|paradigm", sentence, re.I):
        return "thesis"
    return "interpretive"


def claim_like(sentence: str) -> bool:
    return bool(
        re.search(
            r"提出|认为|显示|表明|证明|揭示|关键|核心|限制|挑战|风险|框架|架构|范式|证据|argues?|shows?|suggests?|demonstrates?|requires?|should|must|framework",
            sentence,
            re.I,
        )
    )


def extract_claims(text: str, limit: int) -> dict[str, list[str]]:
    buckets = {"thesis": [], "evidence": [], "limitation": [], "interpretive": []}
    for sentence in split_sentences(text):
        if not claim_like(sentence):
            continue
        bucket = classify_claim(sentence)
        if len(buckets[bucket]) < limit:
            buckets[bucket].append(sentence)
    return buckets


def extract_literature_leads(lines: list[str], limit: int) -> list[str]:
    leads: list[str] = []
    in_refs = False
    for line in lines:
        raw = clean_space(line)
        if not raw:
            continue
        if re.match(r"^(references|参考文献|bibliography)\b", raw, re.I):
            in_refs = True
            continue
        explicit = re.search(r"(doi\.org|10\.\d{4,9}/|arxiv|pubmed|pmid|openreview|bioRxiv|medRxiv)", raw, re.I)
        numbered = in_refs and re.match(r"^(\[[0-9]+\]|[0-9]+[.)])\s+.+", raw)
        if explicit or numbered:
            leads.append(raw[:600])
        if len(leads) >= limit:
            break
    return leads


def section_previews(lines: list[str], limit: int) -> list[dict[str, str]]:
    previews: list[dict[str, str]] = []
    current: dict[str, Any] | None = None
    buffer: list[str] = []
    for line in lines:
        heading = parse_heading(line)
        if heading:
            if current:
                current["preview"] = clean_space(" ".join(buffer))[:600]
                previews.append(current)
                if len(previews) >= limit:
                    return previews
            level, title = heading
            current = {"level": str(level), "title": title, "preview": ""}
            buffer = []
        elif current and clean_space(line):
            buffer.append(clean_space(line))
    if current and len(previews) < limit:
        current["preview"] = clean_space(" ".join(buffer))[:600]
        previews.append(current)
    return previews


def reusable_insights(claims: dict[str, list[str]], limit: int) -> list[str]:
    pool = claims["thesis"] + claims["evidence"] + claims["interpretive"]
    return pool[:limit]


def draft_risks(memory: DraftMemory) -> list[str]:
    risks: list[str] = []
    if len(memory.top_headings) > 8:
        risks.append("Too many top-level or major headings; likely needs compression into a 3-5 movement spine.")
    if len(memory.literature_leads) == 0:
        risks.append("No explicit literature leads detected; claims need source audit and recall.")
    if len(memory.evidence_claims) < 2:
        risks.append("Few evidence-like claims detected; may be mostly conceptual or speculative.")
    if any(re.search(r"革命|颠覆|重塑|revolutionary|paradigm-shifting", claim, re.I) for claim in memory.thesis_claims):
        risks.append("Contains grand framing language; require evidence-backed moderation.")
    return risks or ["No major heuristic risk detected; still requires citation and claim audit."]


def parse_draft(path: Path, claim_limit: int, lead_limit: int, section_limit: int) -> DraftMemory:
    text = read_text(path)
    lines = text.splitlines()
    headings = [h for h in (parse_heading(line) for line in lines) if h]
    top_headings = [title for level, title in headings if level <= 2][:30]
    claims = extract_claims(text, claim_limit)
    memory = DraftMemory(
        memory_id=safe_filename(path.stem),
        path=str(path),
        title=infer_title(lines, path.stem),
        chars=len(text),
        top_headings=top_headings,
        section_preview=section_previews(lines, section_limit),
        thesis_claims=claims["thesis"],
        evidence_claims=claims["evidence"],
        limitation_claims=claims["limitation"],
        literature_leads=extract_literature_leads(lines, lead_limit),
        reusable_insights=reusable_insights(claims, claim_limit),
        risks=[],
    )
    memory.risks = draft_risks(memory)
    return memory


def write_card(memory: DraftMemory, card_dir: Path) -> str:
    card_dir.mkdir(parents=True, exist_ok=True)
    path = card_dir / f"{safe_filename(memory.memory_id)}.md"
    lines = [
        f"# Draft Memory: {memory.title}",
        "",
        "## Passport",
        "",
        f"- Source: `{memory.path}`",
        f"- Characters: {memory.chars}",
        "- Verification: unverified AI/user draft; use as scaffold, not evidence.",
        "",
        "## Structure Observed",
        "",
    ]
    lines.extend(f"- {heading}" for heading in memory.top_headings[:20])
    lines.extend(["", "## Section Previews", ""])
    for item in memory.section_preview:
        lines.append(f"### L{item['level']} {item['title']}")
        lines.append(item["preview"] or "[no preview]")
        lines.append("")
    lines.extend(["## Candidate Thesis Claims", ""])
    lines.extend(f"- {claim}" for claim in memory.thesis_claims[:10])
    lines.extend(["", "## Candidate Evidence Claims", ""])
    lines.extend(f"- {claim}" for claim in memory.evidence_claims[:10])
    lines.extend(["", "## Limitation / Gap Claims", ""])
    lines.extend(f"- {claim}" for claim in memory.limitation_claims[:10])
    lines.extend(["", "## Literature Leads", ""])
    lines.extend(f"- {lead}" for lead in memory.literature_leads[:20])
    lines.extend(["", "## Reusable Insights", ""])
    lines.extend(f"- {insight}" for insight in memory.reusable_insights[:12])
    lines.extend(["", "## Risks To Challenge", ""])
    lines.extend(f"- {risk}" for risk in memory.risks)
    lines.extend(
        [
            "",
            "## Use Rule",
            "",
            "This card may guide framework discussion. Do not use its factual claims in manuscript prose until the cited paper/source has been verified and added to the literature pool.",
        ]
    )
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return str(path)


def write_claim_csv(memories: list[DraftMemory], out_dir: Path) -> str:
    path = out_dir / "claims_for_discussion.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["draft", "claim_role", "claim", "verification_status"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for memory in memories:
            for role, claims in [
                ("thesis", memory.thesis_claims),
                ("evidence", memory.evidence_claims),
                ("limitation", memory.limitation_claims),
            ]:
                for claim in claims:
                    writer.writerow(
                        {
                            "draft": Path(memory.path).name,
                            "claim_role": role,
                            "claim": claim,
                            "verification_status": "draft_memory_only_needs_source_audit",
                        }
                    )
    return str(path)


def write_index(memories: list[DraftMemory], out_dir: Path, topic: str) -> str:
    path = out_dir / "draft_memory_index.md"
    lines = [
        "# Draft Memory Index",
        "",
        f"- Topic: {topic or '[not specified]'}",
        f"- Generated at: {now_iso()}",
        "- Rule: use draft memory for framework discussion; use verified literature for manuscript claims.",
        "",
        "## Draft Cards",
        "",
    ]
    for memory in memories:
        lines.append(f"- `{Path(memory.path).name}` -> `{memory.card_path}`")
    lines.extend(["", "## Cross-Draft Signals", ""])
    all_thesis = [claim for memory in memories for claim in memory.thesis_claims]
    all_risks = [risk for memory in memories for risk in memory.risks]
    lines.append(f"- Candidate thesis claims captured: {len(all_thesis)}")
    lines.append(f"- Literature leads captured: {sum(len(memory.literature_leads) for memory in memories)}")
    lines.append(f"- Draft risks captured: {len(all_risks)}")
    lines.extend(["", "## Interaction Rule", ""])
    lines.append("Before drafting, Codex should load this index plus the relevant memory cards, then ask the user to choose the central thesis and 3-5 section spine.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def write_agent_packet(memories: list[DraftMemory], out_dir: Path) -> str:
    path = out_dir / "agent_reading_packet.md"
    lines = [
        "# Draft Reading Packet For Specialist Agents",
        "",
        "Use this packet for draft_deep_reader, draft_memory_curator, argument_builder, and outline_architect.",
        "",
        "## Non-Negotiables",
        "",
        "- Treat draft claims as memory, not verified evidence.",
        "- Quote source draft filenames when preserving or challenging an idea.",
        "- Convert unsupported claims into literature tasks.",
        "- Do not approve the framework until literature strategy and user alignment are recorded.",
        "",
        "## Condensed Memories",
        "",
    ]
    for memory in memories:
        lines.extend(
            [
                f"### {Path(memory.path).name}",
                f"- Title: {memory.title}",
                f"- Card: `{memory.card_path}`",
                f"- Top headings: {', '.join(memory.top_headings[:8]) or '[none]'}",
                f"- Key thesis claim: {memory.thesis_claims[0] if memory.thesis_claims else '[none detected]'}",
                f"- Key risk: {memory.risks[0] if memory.risks else '[none detected]'}",
                "",
            ]
        )
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return str(path)


def write_chief_editor_checkpoint(memories: list[DraftMemory], out_dir: Path, topic: str) -> str:
    path = out_dir / "chief_editor_reading_checkpoint.md"
    lines = [
        "# Chief Editor Reading Checkpoint",
        "",
        f"- Topic: {topic or '[not specified]'}",
        f"- Generated at: {now_iso()}",
        "- Owner: current Codex conversation as chief editor/PI",
        "- Status: pending chief-editor reading and user alignment",
        "",
        "## Rule",
        "",
        "The chief Codex instance in the current conversation must read this checkpoint, `draft_memory_index.md`, `agent_reading_packet.md`, and relevant draft cards before delegating framework work or drafting manuscript prose. Subagents can help, but they do not replace the chief editor's own reading judgment.",
        "",
        "## Drafts To Read",
        "",
    ]
    for memory in memories:
        lines.extend(
            [
                f"### {Path(memory.path).name}",
                f"- Card: `{memory.card_path}`",
                f"- Title: {memory.title}",
                f"- Structure signal: {', '.join(memory.top_headings[:6]) or '[none detected]'}",
                f"- Strongest reusable insight to inspect: {memory.reusable_insights[0] if memory.reusable_insights else '[none detected]'}",
                f"- Main risk to challenge: {memory.risks[0] if memory.risks else '[none detected]'}",
                "",
            ]
        )
    lines.extend(
        [
            "## Chief Codex Notes To Fill Before User Alignment",
            "",
            "- Which draft has the strongest central thesis, and why?",
            "- Which draft ideas should be preserved, merged, demoted, or deleted?",
            "- Which claims are attractive but unsupported?",
            "- Which 3-5 section spine looks most promising before literature adjudication?",
            "- What must be asked of the user before continuing?",
            "",
            "## Task Board",
            "",
            "- [ ] Chief Codex read draft memory index and packet.",
            "- [ ] Chief Codex opened the relevant per-draft cards.",
            "- [ ] Draft citation assets were built before supplemental recall.",
            "- [ ] Framework blackboard and material passport were checked.",
            "- [ ] User-facing alignment questions were prepared.",
            "- [ ] Subagent tasks, if any, were scoped after chief reading.",
        ]
    )
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return str(path)


def build_sqlite(memories: list[DraftMemory], out_dir: Path) -> str:
    db_path = out_dir / "draft_memory.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE IF EXISTS memories")
    conn.execute("DROP TABLE IF EXISTS memories_fts")
    conn.execute(
        "CREATE TABLE memories (id INTEGER PRIMARY KEY, draft TEXT, path TEXT, section TEXT, body TEXT)"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE memories_fts USING fts5(draft, section, body, content='memories', content_rowid='id')"
    )
    for memory in memories:
        blocks = [
            ("title", memory.title),
            ("top_headings", "\n".join(memory.top_headings)),
            ("thesis_claims", "\n".join(memory.thesis_claims)),
            ("evidence_claims", "\n".join(memory.evidence_claims)),
            ("limitation_claims", "\n".join(memory.limitation_claims)),
            ("literature_leads", "\n".join(memory.literature_leads)),
            ("risks", "\n".join(memory.risks)),
        ]
        for section, body in blocks:
            cursor = conn.execute(
                "INSERT INTO memories (draft, path, section, body) VALUES (?, ?, ?, ?)",
                (Path(memory.path).name, memory.path, section, body),
            )
            rowid = cursor.lastrowid
            conn.execute(
                "INSERT INTO memories_fts (rowid, draft, section, body) VALUES (?, ?, ?, ?)",
                (rowid, Path(memory.path).name, section, body),
            )
    conn.commit()
    conn.close()
    return str(db_path)


def fts_query(query: str) -> str:
    terms = re.findall(r"[\w\u4e00-\u9fff]+", query)
    return " OR ".join(terms) if terms else query


def cmd_build(args: argparse.Namespace) -> int:
    drafts = discover_drafts(args)
    if not drafts:
        raise SystemExit("No supported drafts found.")
    out_dir = Path(args.out_dir)
    card_dir = out_dir / "cards"
    out_dir.mkdir(parents=True, exist_ok=True)
    memories = [parse_draft(path, args.claim_limit, args.lead_limit, args.section_limit) for path in drafts]
    for memory in memories:
        memory.card_path = write_card(memory, card_dir)
    memory_json = out_dir / "draft_memory.json"
    memory_json.write_text(json.dumps([asdict(memory) for memory in memories], ensure_ascii=False, indent=2), encoding="utf-8")
    claim_csv = write_claim_csv(memories, out_dir)
    index_md = write_index(memories, out_dir, args.topic)
    packet = write_agent_packet(memories, out_dir)
    chief_checkpoint = write_chief_editor_checkpoint(memories, out_dir, args.topic)
    db = build_sqlite(memories, out_dir)
    summary = {
        "drafts": len(memories),
        "out_dir": str(out_dir.resolve()),
        "index": index_md,
        "packet": packet,
        "chief_editor_checkpoint": chief_checkpoint,
        "claims": claim_csv,
        "memory_json": str(memory_json),
        "db": db,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        """
        SELECT memories.draft, memories.path, memories.section,
               snippet(memories_fts, 2, '[', ']', ' ... ', 24) AS snippet,
               bm25(memories_fts) AS score
        FROM memories_fts
        JOIN memories ON memories.id = memories_fts.rowid
        WHERE memories_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (args.fts_expression or fts_query(args.query), args.top_k),
    ).fetchall()
    conn.close()
    result = [
        {"draft": row[0], "path": row[1], "section": row[2], "snippet": row[3], "score": row[4]}
        for row in rows
    ]
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build/search persistent memory for review drafts.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build")
    p_build.add_argument("--draft", action="append", default=[])
    p_build.add_argument("--draft-dir", action="append", default=[])
    p_build.add_argument("--topic", default="")
    p_build.add_argument("--out-dir", default="./review-data/03_framework/draft_memory")
    p_build.add_argument("--claim-limit", type=int, default=12)
    p_build.add_argument("--lead-limit", type=int, default=30)
    p_build.add_argument("--section-limit", type=int, default=20)
    p_build.set_defaults(func=cmd_build)

    p_search = sub.add_parser("search")
    p_search.add_argument("--db", required=True)
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--top-k", type=int, default=10)
    p_search.add_argument("--fts-expression", default="")
    p_search.add_argument("--out", default="")
    p_search.set_defaults(func=cmd_search)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
