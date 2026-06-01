#!/usr/bin/env python3
"""Plan evidence-grounded figures, tables, and boxes for a review.

The script connects the framework stage to literature evidence. It does not
invent data or final visual claims; it produces display-item blueprints,
evidence gaps, image prompts for conceptual drafts, and a human checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Any


DEFAULT_FRAMEWORK_DIR = "review-data/03_framework/logic_framework"
DEFAULT_POOL_DIR = "review-data/02_literature/pool"
DEFAULT_OUT_DIR = "review-data/03_framework/display_items"
SELECTED_STATUSES = {"citation_pool", "seminal", "method", "recent"}
PRESELECTION_STATUSES = {"candidate", "maybe", "background"}
EVIDENCE_STATUSES = {"ready", "needs_literature", "needs_fulltext", "needs_rag", "needs_user_decision"}
PREPRINT_DOI_PREFIXES = (
    "10.1101/",
    "10.21203/",
    "10.2139/ssrn",
    "10.48550/arxiv",
    "10.31219/osf.io",
    "10.20944/preprints",
)
PREPRINT_TEXT_MARKERS = ("arxiv", "openreview", "medrxiv", "biorxiv", "research square", "researchsquare", "ssrn")


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(clean(item) for item in value if item is not None)
    return re.sub(r"\s+", " ", str(value)).strip()


def slugify(text: str, limit: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", clean(text).lower()).strip("-")
    return (slug[:limit].strip("-") or "display-item")


def read_text(path: Path) -> str:
    if not path.exists() or path.is_dir():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
        return [dict(row) for row in csv.DictReader(handle, dialect=dialect)]


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def framework_paths(framework_dir: Path) -> dict[str, Path]:
    return {
        "chief_checkpoint": framework_dir / "chief_editor_framework_checkpoint.md",
        "blackboard": framework_dir / "framework_evidence_blackboard.md",
        "claim_map": framework_dir / "argument_evidence_map.csv",
        "tasks": framework_dir / "literature_search_tasks.csv",
        "passport": framework_dir / "framework_material_passport.json",
        "logic": framework_dir / "draft_logic_framework.md",
    }


def pool_paths(pool_dir: Path) -> dict[str, Path]:
    return {
        "pool": pool_dir / "pool.json",
        "cards": pool_dir / "cards",
        "contexts": pool_dir / "contexts",
        "indexes": pool_dir / "indexes",
    }


def noisy_claim(text: str) -> bool:
    value = clean(text)
    lower = value.lower()
    if not value:
        return True
    if value.startswith("#") or value.startswith("|") or value.count("|") >= 3:
        return True
    if re.match(r"^(abstract|summary|keywords|references|bibliography)\b", lower):
        return True
    if re.match(r"^(\u6458\u8981|\u5173\u952e\u8bcd|\u53c2\u8003\u6587\u732e)\b", value):
        return True
    if re.search(r"\b(final-gate|search log|screening count|records checked|candidate records|literature pool)\b", lower):
        return True
    if len(value) > 650:
        return True
    if len(re.findall(r"https?://|10\.\d{4,9}/", value)) >= 3:
        return True
    return False


def extract_claims(framework_dir: Path, max_claims: int = 12) -> list[str]:
    paths = framework_paths(framework_dir)
    rows = read_csv_rows(paths["claim_map"])
    claims: list[str] = []
    checkpoint = read_text(paths["chief_checkpoint"])
    if checkpoint:
        for line in checkpoint.splitlines():
            stripped = line.strip(" -\t")
            if len(stripped) < 25 or stripped.startswith("#") or noisy_claim(stripped):
                continue
            if re.search(r"thesis|argument|claim|evidence|framework|section|spine|gap|future", stripped, re.I):
                claims.append(stripped)
            if len(claims) >= max_claims:
                break
    claim_fields = ["claim", "sentence", "text", "argument", "insight", "evidence_claim"]
    for row in rows:
        for field in claim_fields:
            value = clean(row.get(field))
            if value and len(value) > 20 and not noisy_claim(value):
                claims.append(value)
                break
    if len(claims) < 4:
        text = "\n".join([read_text(paths["blackboard"]), read_text(paths["logic"])])
        for line in text.splitlines():
            stripped = line.strip(" -\t")
            if len(stripped) < 25 or stripped.startswith("#"):
                continue
            if noisy_claim(stripped):
                continue
            if re.search(r"claim|thesis|argu|evidence|framework|治理|证据|框架|论点|主张|显示|表明|需要", stripped, re.I):
                claims.append(stripped)
    seen: set[str] = set()
    out: list[str] = []
    for claim in claims:
        key = claim.lower()
        if key not in seen:
            seen.add(key)
            out.append(claim[:400])
        if len(out) >= max_claims:
            break
    return out


def extract_tasks(framework_dir: Path, max_tasks: int = 12) -> list[str]:
    rows = read_csv_rows(framework_paths(framework_dir)["tasks"])
    tasks: list[str] = []
    for row in rows:
        value = clean(row.get("query") or row.get("task") or row.get("claim") or next(iter(row.values()), ""))
        if value:
            tasks.append(value)
    if not tasks:
        text = read_text(framework_paths(framework_dir)["tasks"])
        tasks = [line.strip() for line in text.splitlines() if line.strip()]
    return tasks[:max_tasks]


def is_published(record: dict[str, Any]) -> bool:
    reason = clean(record.get("publication_gate")).lower()
    source = clean(record.get("source")).lower()
    pub_type = clean(record.get("publication_type")).lower()
    doi = clean(record.get("doi")).lower()
    if reason == "published":
        return True
    preprint_text = " ".join(clean(record.get(field)).lower() for field in ["source", "title", "url", "doi", "recall_queries", "notes"])
    if any(marker in preprint_text for marker in PREPRINT_TEXT_MARKERS):
        return False
    if doi.startswith(PREPRINT_DOI_PREFIXES) or "arxiv" in doi:
        return False
    if any(marker in pub_type for marker in ["preprint", "posted-content", "posted content"]):
        return False
    return bool(record.get("doi") or record.get("pmid") or record.get("journal"))


def load_sources(pool_dir: Path, limit: int = 40) -> list[dict[str, Any]]:
    pool = load_json(pool_paths(pool_dir)["pool"], {"papers": {}})
    records = list(pool.get("papers", {}).values())
    selected = [
        record
        for record in records
        if clean(record.get("pool_status")) in SELECTED_STATUSES or clean(record.get("status")) in SELECTED_STATUSES
    ]
    if not selected:
        selected = [
            record
            for record in records
            if clean(record.get("pool_status")) in PRESELECTION_STATUSES or clean(record.get("status")) in PRESELECTION_STATUSES
        ]
    if not selected:
        selected = [
            record
            for record in records
            if clean(record.get("pool_status")) not in {"rejected", "not_found", "preprint_only"}
        ]
    out = []
    for record in selected[:limit]:
        out.append(
            {
                "key": clean(record.get("key")),
                "title": clean(record.get("title")),
                "year": clean(record.get("year")),
                "doi": clean(record.get("doi")),
                "pmid": clean(record.get("pmid")),
                "source": clean(record.get("source")),
                "journal": clean(record.get("journal")),
                "pool_status": clean(record.get("pool_status") or record.get("status")),
                "claim_supported": clean(record.get("claim_supported")),
                "use_in_review": clean(record.get("use_in_review")),
                "publication_gate": clean(record.get("publication_gate")),
                "published": is_published(record),
                "card_path": clean(record.get("card_path")),
                "fulltext_path": clean(record.get("fulltext_text_path") or record.get("fulltext_path")),
                "abstract": clean(record.get("abstract")),
            }
        )
    return out


def has_existing_file(path_text: str) -> bool:
    return bool(path_text and Path(path_text).exists())


def rag_available(rag_db: str) -> bool:
    return bool(rag_db and Path(rag_db).exists())


def evidence_status(kind: str, claims: list[str], sources: list[dict[str, Any]], rag_db: str) -> str:
    published = [source for source in sources if source.get("published")]
    if not claims:
        return "needs_user_decision"
    if kind == "figure":
        return "needs_user_decision"
    if not published:
        return "needs_literature"
    if not any(has_existing_file(source.get("fulltext_path", "")) for source in published):
        return "needs_fulltext"
    if not rag_available(rag_db):
        return "needs_rag"
    return "ready"


def source_refs(sources: list[dict[str, Any]], limit: int = 8) -> list[dict[str, str]]:
    refs = []
    for source in sources[:limit]:
        refs.append(
            {
                "key": source.get("key", ""),
                "title": source.get("title", ""),
                "year": source.get("year", ""),
                "doi": source.get("doi", ""),
                "pmid": source.get("pmid", ""),
                "evidence_note": source.get("claim_supported") or source.get("use_in_review") or "[EVIDENCE GAP]",
            }
        )
    return refs


def prompt_for(title: str, role: str, labels: list[str], target_journal: str) -> str:
    label_text = "; ".join(labels[:8]) or "framework components; evidence flow; human oversight; governance"
    role_text = clean(role).rstrip(".")
    return (
        f"Create a high-resolution conceptual draft figure for a {target_journal} review. "
        f"Subject: {title}. Argument role: {role_text}. "
        "Use a restrained Nature Reviews-style scientific schematic: clean white background, modular panels, balanced spacing, clear directional arrows, subtle color coding, and editable-looking vector shapes. "
        f"Use only short placeholder labels for panel structure: {label_text}. "
        "Prefer simple icons and abstract workflows over decorative imagery. "
        "Do not include specific numerical results, author names, paper titles, citations, DOI strings, logos, unsupported performance claims, or dense body text. "
        "Leave generous white space for later manual label editing, citation audit, and professional redraw."
    )


def make_items(claims: list[str], tasks: list[str], sources: list[dict[str, Any]], target_journal: str, rag_db: str, max_items: int) -> list[dict[str, Any]]:
    core_claims = claims[:6]
    source_subset = [source for source in sources if source.get("published")] or sources
    specs = [
        (
            "F1",
            "figure",
            "Conceptual framework of the review",
            "Make the central thesis visible as a reusable framework rather than a topic list.",
            ["knowledge base", "multimodal evidence", "agent workflow", "human oversight", "clinical governance"],
        ),
        (
            "T1",
            "table",
            "Evidence table for core studies and systems",
            "Bind major claims to verified papers, methods, limitations, and citation status.",
            ["Citation key", "Study/system", "Evidence type", "Main support", "Limitations", "Citation status"],
        ),
        (
            "T2",
            "table",
            "Taxonomy of methods, tasks, and evidence readiness",
            "Compare model/agent types by task, data source, validation level, and governance need.",
            ["Category", "Representative papers", "Task", "Evidence readiness", "Open gap"],
        ),
        (
            "F2",
            "figure",
            "Evidence ladder from demonstration to trustworthy deployment",
            "Show why benchmark/demo performance is not enough for top-journal claims.",
            ["offline benchmark", "external validation", "workflow evaluation", "prospective study", "governed deployment"],
        ),
        (
            "F3",
            "figure",
            "Human-in-the-loop agent governance loop",
            "Show where retrieval, deliberation, clinical oversight, audit, and feedback enter the system.",
            ["retrieve", "reason", "act", "audit", "human decision", "feedback"],
        ),
        (
            "B1",
            "box",
            "Reporting and audit checklist",
            "Turn methods/reporting expectations into a reusable reader checklist.",
            ["data provenance", "citation traceability", "human oversight", "safety boundary", "model update record"],
        ),
        (
            "T3",
            "table",
            "Research agenda and unresolved evidence gaps",
            "Translate unsupported or weak claims into concrete literature and study-design tasks.",
            ["Gap", "Why it matters", "Evidence needed", "Likely sources", "Priority"],
        ),
    ]
    items = []
    for item_id, kind, title, role, columns_or_labels in specs[:max_items]:
        item_claims = core_claims[:3] if item_id in {"F1", "F2", "F3"} else core_claims[:5]
        if item_id == "T3" and tasks:
            item_claims = tasks[:5]
        status = evidence_status(kind, item_claims, source_subset, rag_db)
        item: dict[str, Any] = {
            "id": item_id,
            "kind": kind,
            "title": title,
            "argument_role": role,
            "linked_claims": item_claims,
            "candidate_sources": source_refs(source_subset, limit=10 if kind == "table" else 5),
            "evidence_status": status,
            "human_questions": [
                "Keep, merge, delete, or redesign this display item?",
                "Which central claim should this item support?",
                "Which citations are acceptable as evidence for this item?",
            ],
        }
        if kind == "figure":
            item["image_prompt"] = prompt_for(title, role, columns_or_labels, target_journal)
            item["panel_plan"] = columns_or_labels
            item["image_generation"] = {
                "tool": "Codex/GPT image generation",
                "status": "needs_human_prompt_approval",
                "aspect_ratio": "16:9 for review figure drafts; use 4:3 only when the target journal or user asks",
                "output_stub": f"{item_id}_{slugify(title)}_concept",
                "review_checklist": [
                    "central argument is visible without adding unsupported claims",
                    "labels are short placeholders and can be manually edited",
                    "no numbers, author names, paper titles, DOI strings, citation markers, or logos",
                    "layout is suitable for later vector redraw or BioRender/Figma refinement",
                    "caption skeleton and evidence status still match the figure",
                ],
            }
        else:
            item["table_columns"] = columns_or_labels
            item["row_unit"] = "verified paper or claim-evidence unit" if kind == "table" else "checklist item"
        items.append(item)
    return items


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def markdown_escape(text: str) -> str:
    return clean(text).replace("|", "\\|")


def write_plan(out_dir: Path, items: list[dict[str, Any]], framework_dir: Path, pool_dir: Path, rag_db: str) -> None:
    lines = [
        "# Display Item Plan",
        "",
        "This is a human-in-the-loop plan. Do not treat figure prompts or table gaps as final evidence.",
        "",
        f"- Framework dir: `{framework_dir}`",
        f"- Pool dir: `{pool_dir}`",
        f"- RAG db: `{rag_db or '[not provided]'}`",
        "",
        "## Inventory",
        "",
        "| ID | Kind | Title | Evidence status | Argument role |",
        "|---|---|---|---|---|",
    ]
    for item in items:
        lines.append(
            f"| `{item['id']}` | {item['kind']} | {markdown_escape(item['title'])} | `{item['evidence_status']}` | {markdown_escape(item['argument_role'])} |"
        )
    lines.extend(["", "## Human Checkpoint", "", "Before drawing or drafting around these display items, decide:"])
    for item in items:
        lines.append(f"- `{item['id']}` {item['title']}: keep / merge / delete / redesign; confirm linked claims and citations.")
    (out_dir / "display_item_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure_blueprints(out_dir: Path, items: list[dict[str, Any]]) -> None:
    lines = ["# Figure Blueprints", ""]
    for item in items:
        if item["kind"] != "figure":
            continue
        lines.extend(
            [
                f"## {item['id']}. {item['title']}",
                "",
                f"- Argument role: {item['argument_role']}",
                f"- Evidence status: `{item['evidence_status']}`",
                f"- Panel plan: {', '.join(item.get('panel_plan', []))}",
                "",
                "Linked claims:",
            ]
        )
        for claim in item.get("linked_claims", []) or ["[needs user decision]"]:
            lines.append(f"- {claim}")
        lines.extend(["", "Caption skeleton:", "", f"{item['title']}. This figure should explain {item['argument_role'].lower()} Evidence-specific labels must be finalized after citation audit.", ""])
    (out_dir / "figure_blueprints.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_prompt_pack(out_dir: Path, items: list[dict[str, Any]]) -> None:
    lines = [
        "# Figure Image Prompts",
        "",
        "Use these only for conceptual draft images after the human checkpoint approves the figure role and prompt. Do not submit generated images without human scientific and visual review.",
        "",
    ]
    for item in items:
        if item["kind"] == "figure":
            image_generation = item.get("image_generation", {})
            lines.extend(
                [
                    f"## {item['id']}. {item['title']}",
                    "",
                    f"- Generation status: `{image_generation.get('status', 'needs_human_prompt_approval')}`",
                    f"- Aspect ratio: {image_generation.get('aspect_ratio', '16:9')}",
                    f"- Suggested output stub: `{image_generation.get('output_stub', item['id'])}`",
                    "",
                    item.get("image_prompt", ""),
                    "",
                    "Review checklist:",
                ]
            )
            for check in image_generation.get("review_checklist", []):
                lines.append(f"- {check}")
            lines.append("")
    (out_dir / "figure_image_prompts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_image_generation_assets(out_dir: Path, items: list[dict[str, Any]]) -> None:
    figure_items = [item for item in items if item.get("kind") == "figure"]
    queue_path = out_dir / "gpt_image_generation_queue.csv"
    with queue_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "id",
            "title",
            "argument_role",
            "evidence_status",
            "generation_status",
            "human_checkpoint_required",
            "aspect_ratio",
            "output_stub",
            "image_prompt",
            "forbidden_items",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in figure_items:
            image_generation = item.get("image_generation", {})
            writer.writerow(
                {
                    "id": item.get("id", ""),
                    "title": item.get("title", ""),
                    "argument_role": item.get("argument_role", ""),
                    "evidence_status": item.get("evidence_status", ""),
                    "generation_status": image_generation.get("status", "needs_human_prompt_approval"),
                    "human_checkpoint_required": "yes",
                    "aspect_ratio": image_generation.get("aspect_ratio", "16:9"),
                    "output_stub": image_generation.get("output_stub", item.get("id", "")),
                    "image_prompt": item.get("image_prompt", ""),
                    "forbidden_items": "numbers; author names; paper titles; DOI strings; citation markers; logos; unsupported superiority claims; dense body text",
                }
            )

    lines = [
        "# GPT Image Generation Brief",
        "",
        "Use this file after `human_display_item_checkpoint.md` is approved. Codex may use GPT image generation for conceptual figure drafts, but the generated raster image is discussion material, not the journal-ready figure.",
        "",
        "Workflow:",
        "",
        "1. Confirm the figure is kept and the central claim is approved.",
        "2. Confirm linked evidence and caption skeleton are consistent with the current literature pool.",
        "3. Call GPT image generation with the reviewed prompt for one figure at a time.",
        "4. Save or reference the draft under `review-data/03_framework/display_items/generated_drafts/<output_stub>.png` when the runtime provides a file artifact.",
        "5. Run label, caption, and evidence audit before using the image in manuscript planning.",
        "6. Redraw or refine the final figure in BioRender, Figma, Illustrator, PowerPoint, or another professional tool if submission quality or editable/vector art is required.",
        "",
        "Do not use generated images to introduce evidence, numeric results, or claims not already present in the framework and literature audit.",
        "",
    ]
    for item in figure_items:
        image_generation = item.get("image_generation", {})
        lines.extend(
            [
                f"## {item.get('id')}. {item.get('title')}",
                "",
                f"- Status: `{image_generation.get('status', 'needs_human_prompt_approval')}`",
                f"- Output stub: `{image_generation.get('output_stub', item.get('id', 'figure'))}`",
                f"- Aspect ratio: {image_generation.get('aspect_ratio', '16:9')}",
                "",
                "Prompt:",
                "",
                item.get("image_prompt", ""),
                "",
                "Post-generation audit:",
            ]
        )
        for check in image_generation.get("review_checklist", []):
            lines.append(f"- {check}")
        lines.append("")
    (out_dir / "gpt_image_generation_brief.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table_files(out_dir: Path, items: list[dict[str, Any]]) -> None:
    with (out_dir / "table_blueprints.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "kind", "title", "argument_role", "row_unit", "table_columns", "evidence_status"])
        writer.writeheader()
        for item in items:
            if item["kind"] in {"table", "box"}:
                writer.writerow(
                    {
                        "id": item["id"],
                        "kind": item["kind"],
                        "title": item["title"],
                        "argument_role": item["argument_role"],
                        "row_unit": item.get("row_unit", ""),
                        "table_columns": "; ".join(item.get("table_columns", [])),
                        "evidence_status": item["evidence_status"],
                    }
                )
    write_evidence_gaps(out_dir / "table_evidence_gaps.csv", items)


def write_evidence_gaps(path: Path, items: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["item_id", "kind", "claim_or_row", "candidate_sources", "gap", "rag_query", "human_action"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in items:
            if item["kind"] not in {"table", "box"}:
                continue
            sources = item.get("candidate_sources", [])
            source_text = "; ".join(src.get("key") or src.get("title", "") for src in sources if src)
            for claim in item.get("linked_claims", []) or ["[EVIDENCE GAP]"]:
                gap = item["evidence_status"]
                if not sources:
                    gap = "needs_literature"
                writer.writerow(
                    {
                        "item_id": item["id"],
                        "kind": item["kind"],
                        "claim_or_row": claim if sources else "[EVIDENCE GAP] " + claim,
                        "candidate_sources": source_text or "[EVIDENCE GAP]",
                        "gap": gap,
                        "rag_query": claim,
                        "human_action": "confirm evidence, request full text, run RAG, or remove row",
                    }
                )


def write_checkpoint(out_dir: Path, items: list[dict[str, Any]]) -> None:
    lines = [
        "# Human Display Item Checkpoint",
        "",
        "Discuss this before drafting around figures/tables or generating images.",
        "",
    ]
    for item in items:
        lines.extend(
            [
                f"## {item['id']}. {item['title']}",
                "",
                f"- Kind: {item['kind']}",
                f"- Argument role: {item['argument_role']}",
                f"- Evidence status: `{item['evidence_status']}`",
                "- Decision: keep / merge / delete / redesign",
                "- Confirmed claims:",
                "- Confirmed citations:",
                "- Changes requested:",
                "",
            ]
        )
    (out_dir / "human_display_item_checkpoint.md").write_text("\n".join(lines), encoding="utf-8")


def query_rag(db_path: str, query: str, top_k: int) -> list[dict[str, Any]]:
    if not rag_available(db_path):
        return []
    conn = sqlite3.connect(db_path)
    terms = re.findall(r"[\w\u4e00-\u9fff]+", query)
    expression = " OR ".join(terms) if terms else query
    try:
        rows = conn.execute(
            """
            SELECT nodes.doc_path, nodes.node_path, nodes.title,
                   snippet(nodes_fts, 1, '[', ']', ' ... ', 24) AS snippet,
                   bm25(nodes_fts) AS score
            FROM nodes_fts
            JOIN nodes ON nodes.id = nodes_fts.rowid
            WHERE nodes_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (expression, top_k),
        ).fetchall()
    except sqlite3.Error:
        rows = []
    finally:
        conn.close()
    return [{"doc_path": r[0], "node_path": r[1], "title": r[2], "snippet": r[3], "score": r[4]} for r in rows]


def cmd_plan(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    framework_dir = resolve_project_path(project_dir, args.framework_dir)
    pool_dir = resolve_project_path(project_dir, args.pool_dir)
    out_dir = resolve_project_path(project_dir, args.out_dir)
    rag_db = args.rag_db
    if rag_db and not Path(rag_db).is_absolute():
        rag_db = str(resolve_project_path(project_dir, rag_db))
    out_dir.mkdir(parents=True, exist_ok=True)
    claims = extract_claims(framework_dir, max_claims=16)
    tasks = extract_tasks(framework_dir, max_tasks=12)
    sources = load_sources(pool_dir, limit=args.max_sources)
    items = make_items(claims, tasks, sources, args.target_journal, rag_db, args.max_items)
    data = {
        "schema_name": "top_journal_review_display_items",
        "schema_version": "0.1",
        "target_journal": args.target_journal,
        "framework_dir": str(framework_dir),
        "pool_dir": str(pool_dir),
        "rag_db": rag_db,
        "items": items,
    }
    write_json(out_dir / "display_items.json", data)
    write_plan(out_dir, items, framework_dir, pool_dir, rag_db)
    write_figure_blueprints(out_dir, items)
    write_prompt_pack(out_dir, items)
    write_image_generation_assets(out_dir, items)
    write_table_files(out_dir, items)
    write_checkpoint(out_dir, items)
    print(json.dumps({"out_dir": str(out_dir), "items": len(items), "rag_available": rag_available(rag_db)}, ensure_ascii=False))
    return 0


def load_display_items(path: Path) -> list[dict[str, Any]]:
    data = load_json(path, {})
    if isinstance(data, dict):
        return list(data.get("items", []))
    if isinstance(data, list):
        return data
    return []


def selected_items(items: list[dict[str, Any]], item_id: str) -> list[dict[str, Any]]:
    if not item_id or item_id.lower() in {"all", "*"}:
        return items
    wanted = {part.strip() for part in item_id.split(",") if part.strip()}
    return [item for item in items if item.get("id") in wanted]


def cmd_evidence_pack(args: argparse.Namespace) -> int:
    items = selected_items(load_display_items(Path(args.display_items)), args.item)
    rows = []
    for item in items:
        for claim in item.get("linked_claims", []) or ["[EVIDENCE GAP]"]:
            rag_hits = query_rag(args.rag_db, claim, args.top_k) if args.rag_db else []
            rows.append(
                {
                    "item_id": item.get("id", ""),
                    "kind": item.get("kind", ""),
                    "claim_or_row": claim,
                    "evidence_status": item.get("evidence_status", ""),
                    "candidate_sources": item.get("candidate_sources", []),
                    "rag_hits": rag_hits,
                    "human_action": "confirm, demote, request full text, or remove before drafting",
                }
            )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".json":
        write_json(out, rows)
    else:
        lines = ["# Display Item Evidence Pack", ""]
        for row in rows:
            lines.extend([f"## {row['item_id']} | {row['claim_or_row']}", "", f"- Status: `{row['evidence_status']}`", "- Candidate sources:"])
            if row["candidate_sources"]:
                for source in row["candidate_sources"]:
                    lines.append(f"  - `{source.get('key')}` {source.get('title')} {source.get('doi') or source.get('pmid') or ''}")
            else:
                lines.append("  - [EVIDENCE GAP]")
            lines.append("- RAG hits:")
            if row["rag_hits"]:
                for hit in row["rag_hits"]:
                    lines.append(f"  - {hit['title']} (`{hit['node_path']}`): {hit['snippet']}")
            else:
                lines.append("  - [needs RAG or no hits]")
            lines.extend(["", f"- Human action: {row['human_action']}", ""])
        out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(out), "rows": len(rows)}, ensure_ascii=False))
    return 0


def cmd_prompt_pack(args: argparse.Namespace) -> int:
    items = selected_items(load_display_items(Path(args.display_items)), args.item)
    figure_items = [item for item in items if item.get("kind") == "figure"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Reviewed Figure Prompt Pack",
        "",
        "Use only after human checkpoint review. These prompts are for Codex/GPT image conceptual drafts, not final submission figures.",
        "",
        "Run one figure at a time, then audit the generated draft against the checklist before generating another variant.",
        "",
    ]
    for item in figure_items:
        image_generation = item.get("image_generation", {})
        lines.extend(
            [
                f"## {item.get('id')}. {item.get('title')}",
                "",
                f"- Status: `{image_generation.get('status', 'needs_human_prompt_approval')}`",
                f"- Aspect ratio: {image_generation.get('aspect_ratio', '16:9')}",
                f"- Output stub: `{image_generation.get('output_stub', item.get('id', 'figure'))}`",
                "",
                item.get("image_prompt", ""),
                "",
                "Caption skeleton:",
                "",
                f"{item.get('title')}. {item.get('argument_role')} Final labels and citations require separate audit.",
                "",
                "Forbidden in generated image: numerical results, author names, paper titles, DOI strings, citation markers, logos, dense body text, unsupported superiority claims.",
                "",
                "Post-generation audit:",
            ]
        )
        for check in image_generation.get("review_checklist", []):
            lines.append(f"- {check}")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(out), "figures": len(figure_items)}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan top-journal review figures, tables, and boxes from framework and evidence state.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="Generate display-item inventory, blueprints, evidence gaps, prompts, and human checkpoint.")
    p_plan.add_argument("--project-dir", default=".")
    p_plan.add_argument("--framework-dir", default=DEFAULT_FRAMEWORK_DIR)
    p_plan.add_argument("--pool-dir", default=DEFAULT_POOL_DIR)
    p_plan.add_argument("--rag-db", default="")
    p_plan.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p_plan.add_argument("--target-journal", default="Nature Reviews-style journal")
    p_plan.add_argument("--max-items", type=int, default=7)
    p_plan.add_argument("--max-sources", type=int, default=40)
    p_plan.set_defaults(func=cmd_plan)

    p_evidence = sub.add_parser("evidence-pack", help="Build claim/row-level evidence and RAG packet for display items.")
    p_evidence.add_argument("--display-items", required=True)
    p_evidence.add_argument("--item", default="all")
    p_evidence.add_argument("--rag-db", default="")
    p_evidence.add_argument("--top-k", type=int, default=5)
    p_evidence.add_argument("--out", required=True)
    p_evidence.set_defaults(func=cmd_evidence_pack)

    p_prompt = sub.add_parser("prompt-pack", help="Export reviewed conceptual image prompts for figure items.")
    p_prompt.add_argument("--display-items", required=True)
    p_prompt.add_argument("--item", default="all")
    p_prompt.add_argument("--out", required=True)
    p_prompt.set_defaults(func=cmd_prompt_pack)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
