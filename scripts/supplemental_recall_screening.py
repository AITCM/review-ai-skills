#!/usr/bin/env python3
"""Broad-but-bounded supplemental recall with agent screening gates.

This script starts after chief Codex has read the drafts and the user-facing
framework is at least provisional. It expands the candidate pool from claims,
body-mentioned systems/methods, and literature-discovery gaps, then keeps all
new records in a screening lane until Codex subagents or humans decide what can
enter deterministic verification.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import hashlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import pubmed_recall


TASK_FIELDS = [
    "task_id",
    "lane",
    "source_route",
    "query",
    "linked_claim_id",
    "linked_claim",
    "evidence_need",
    "argument_role",
    "time_role",
    "priority",
    "max_results",
    "screening_prompt",
    "status",
    "origin_file",
]

CANDIDATE_FIELDS = [
    "candidate_id",
    "task_id",
    "lane",
    "source_route",
    "query",
    "title",
    "authors",
    "year",
    "journal",
    "source",
    "paper_id",
    "pmid",
    "doi",
    "url",
    "pdf_url",
    "abstract",
    "publication_type",
    "linked_claim_id",
    "linked_claim",
    "evidence_need",
    "argument_role",
    "time_role",
    "recall_queries",
    "candidate_fit_status",
    "candidate_fit_reason",
    "screening_score",
    "screening_priority",
    "screening_reasons",
    "seed_lock",
    "seed_origin",
    "pool_status",
    "human_decision_needed",
    "notes",
]

SCREENING_FIELDS = [
    "candidate_id",
    "task_id",
    "decision",
    "confidence",
    "claim_fit",
    "evidence_role",
    "argument_role",
    "evidence_level",
    "evidence_strength",
    "fulltext_need",
    "citation_role",
    "key_supported_claim",
    "rationale",
    "supported_claim_ids",
    "human_question",
    "required_next_action",
]

SUPERVISION_FIELDS = [
    "supervision_id",
    "source_candidate_id",
    "supervision_type",
    "decision",
    "evidence_lane",
    "preferred_route",
    "fallback_route",
    "priority",
    "source_title",
    "proposed_title",
    "proposed_doi",
    "proposed_pubmed_query",
    "proposed_crossref_openalex_query",
    "rationale",
    "human_question",
    "next_action",
]

VERIFY_FIELDS = [
    "draft",
    "ref_number",
    "candidate_id",
    "claim_id",
    "candidate_title",
    "raw_reference",
    "doi",
    "pmid",
    "urls",
    "source_kind",
    "cited_in_body",
    "verification_priority",
    "candidate_type",
    "pubmed_query",
    "raw_evidence_excerpt",
    "why_relevant",
]

ROUTES = {"pubmed", "crossref_openalex", "arxiv_openreview", "paper_search", "publisher", "human"}
CORE_VERIFY_DECISIONS = {"include_core", "include_support", "include_counterevidence", "needs_fulltext"}
FRAMEWORK_VERIFY_DECISIONS = {
    "include_framework",
    "include_historical_foundation",
    "include_method_foundation",
    "include_governance_background",
}
INCLUDE_DECISIONS = CORE_VERIFY_DECISIONS | FRAMEWORK_VERIFY_DECISIONS | {"include_background", "duplicate_already_verified"}
NO_VERIFIER_DECISIONS = {"include_background", "duplicate_already_verified"}
FRAMEWORK_BACKGROUND_HINTS = re.compile(
    r"\b(history|historical|framework|method|methodology|mechanism|taxonomy|genealogy|lineage|"
    r"foundation|landscape|survey|review|governance|safety|regulation|background|contrast|"
    r"definition|robot scientist|react|reflexion|nobel turing)\b",
    flags=re.IGNORECASE,
)
AGENT_RECALL_SIGNAL = re.compile(
    r"\b(agent|agentic|multi[- ]agent|autonomous|co[- ]scientist|robot scientist|scientific discovery|"
    r"self[- ]evolv|self[- ]improv|reflection|reflexion|react|tool[- ]use|llm|large language model)\b",
    flags=re.IGNORECASE,
)
BIOMED_RECALL_SIGNAL = re.compile(
    r"\b(biomed\w*|medic\w*|clinical|healthcare|patient|diagnos\w*|therap\w*|drug|pharma\w*|"
    r"protein|genom\w*|cell|single[- ]cell|omics|biology|biological|molecular|crispr|chemistry)\b",
    flags=re.IGNORECASE,
)
QUERY_STOPWORDS = {
    "about",
    "across",
    "adaptive",
    "advanced",
    "analysis",
    "approach",
    "based",
    "benchmark",
    "bridging",
    "challenge",
    "challenges",
    "comprehensive",
    "data",
    "design",
    "development",
    "dynamic",
    "enabled",
    "enabling",
    "enhanced",
    "evaluation",
    "framework",
    "from",
    "large",
    "learning",
    "model",
    "models",
    "multi",
    "new",
    "novel",
    "platform",
    "research",
    "review",
    "science",
    "scientific",
    "system",
    "systems",
    "through",
    "toward",
    "towards",
    "using",
    "with",
}
AMBIGUOUS_SYSTEM_HINTS = {
    "agent",
    "agents",
    "ai",
    "bioagent",
    "bioagents",
    "generic",
    "huntingtin",
    "integrating",
    "modeling",
    "multimodal",
    "model",
    "models",
    "platform",
    "system",
    "systems",
    "viral",
}
WEAK_TITLE_FALLBACK_TERMS = AMBIGUOUS_SYSTEM_HINTS | {
    "adaptive",
    "analysis",
    "autonomous",
    "benchmarking",
    "bridging",
    "dynamic",
    "evidence",
    "framework",
    "machine",
    "methodological",
    "performing",
    "research",
    "survey",
}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def clean_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_title(value: Any) -> str:
    text = clean_space(value).lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_doi(value: Any) -> str:
    text = clean_space(value).lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi\s*:\s*", "", text)
    return text.rstrip(".,;)")


def safe_id(value: str, prefix: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False)
    def write_once() -> None:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")

    for attempt in range(5):
        try:
            if lock:
                with lock:
                    write_once()
            else:
                write_once()
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.1 * (attempt + 1))


def add_task(tasks: list[dict[str, str]], **kwargs: str) -> None:
    route = clean_space(kwargs.get("source_route")) or "pubmed"
    if route not in ROUTES:
        route = "pubmed"
    query = clean_space(kwargs.get("query"))
    if not query:
        return
    lane = clean_space(kwargs.get("lane")) or "framework_gap"
    task_id = safe_id("|".join([lane, route, query, clean_space(kwargs.get("linked_claim_id"))]), "SRT")
    tasks.append(
        {
            "task_id": task_id,
            "lane": lane,
            "source_route": route,
            "query": query,
            "linked_claim_id": clean_space(kwargs.get("linked_claim_id")),
            "linked_claim": clean_space(kwargs.get("linked_claim"))[:900],
            "evidence_need": clean_space(kwargs.get("evidence_need")) or "claim_support_or_boundary",
            "argument_role": clean_space(kwargs.get("argument_role")) or "evidence_support",
            "time_role": clean_space(kwargs.get("time_role")) or "current_or_recent",
            "priority": clean_space(kwargs.get("priority")) or "P2",
            "max_results": clean_space(kwargs.get("max_results")) or "20",
            "screening_prompt": clean_space(kwargs.get("screening_prompt")) or "Does this paper directly support, challenge, or contextualize the linked review claim?",
            "status": clean_space(kwargs.get("status")) or "ready_for_recall",
            "origin_file": clean_space(kwargs.get("origin_file")),
        }
    )


def topic_seed_terms(topic: str) -> list[str]:
    topic = clean_space(topic)
    terms = []
    if topic:
        terms.append(topic)
    lower = topic.lower()
    if re.search(r"agent|智能体", lower):
        terms.extend(['"AI agent"', '"autonomous agent"', '"language agent"', '"multi-agent"'])
    if re.search(r"biomed|medical|medicine|clinical|生物|医学|医疗", lower):
        terms.extend(["biomedical", "medical", "clinical", "drug discovery", "biology"])
    if re.search(r"evol|self|自进化|自我", lower):
        terms.extend(['"self-improving"', '"self-reflection"', '"closed-loop"', '"autonomous scientific discovery"'])
    if re.search(r"large language model|llm|大模型", lower):
        terms.extend(['"large language model"', "LLM", '"foundation model"'])
    return list(dict.fromkeys(term for term in terms if term))


def system_queries(name: str, topic: str) -> list[tuple[str, str]]:
    base = clean_space(name)
    if not base:
        return []
    topic_terms = " ".join(topic_seed_terms(topic)[:4])
    return [
        ("crossref_openalex", f'"{base}"'),
        ("paper_search", f'"{base}" paper'),
        ("pubmed", f'"{base}" AND (biomedical OR biology OR medicine OR clinical OR drug)'),
        ("arxiv_openreview", f'"{base}"'),
        ("crossref_openalex", f'"{base}" {topic_terms}'.strip()),
    ]


def claim_query(claim: str, topic: str, route: str) -> str:
    cleaned = clean_space(re.sub(r"\[[0-9,\-\s;]+\]", " ", claim))
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}|[\u4e00-\u9fff]{2,}", cleaned) if len(w) > 3]
    compact = " ".join(words[:10]) or cleaned[:160]
    if route == "pubmed":
        return f"({compact}) AND (biomedical OR medical OR clinical OR biology OR drug)"
    if route == "arxiv_openreview":
        return compact
    return f"{compact} {topic}".strip()


def framework_queries(topic: str, current_year: int) -> list[tuple[str, str, str]]:
    seeds = topic_seed_terms(topic)
    broad = " AND ".join(seeds[:4]) if seeds else topic
    start = max(2000, current_year - 8)
    return [
        ("historical_foundation", "crossref_openalex", '"autonomous scientific discovery" OR "robot scientist"'),
        ("historical_foundation", "pubmed", '("autonomous" OR "closed-loop") AND ("scientific discovery" OR "biomedical discovery")'),
        ("current_landmark", "pubmed", f'({broad}) AND ({start}:{current_year}[dp])'),
        ("current_landmark", "crossref_openalex", f"{broad} {start} {current_year}"),
        ("methodology", "arxiv_openreview", f'{broad} benchmark evaluation agent'),
        ("governance", "pubmed", f'({broad}) AND (safety OR evaluation OR governance OR regulation)'),
        ("counterevidence", "pubmed", f'({broad}) AND (limitation OR failure OR risk OR hallucination OR validation)'),
    ]


def dedupe_tasks(tasks: list[dict[str, str]], max_tasks: int) -> list[dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    order: list[str] = []
    rank = {"P0": 5, "P1": 4, "P2": 3, "P3": 2, "P4": 1}
    for row in tasks:
        key = "|".join([row.get("source_route", ""), normalize_title(row.get("query", ""))])
        if key not in best:
            best[key] = row
            order.append(key)
        elif rank.get(row.get("priority"), 0) > rank.get(best[key].get("priority"), 0):
            best[key] = row
    out = [best[key] for key in order]
    priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4}
    out.sort(key=lambda row: (priority_order.get(row.get("priority"), 9), row.get("lane", ""), row.get("source_route", ""), row.get("query", "")))
    return out[:max_tasks] if max_tasks > 0 else out


def cmd_plan(args: argparse.Namespace) -> int:
    tasks: list[dict[str, str]] = []
    current_year = dt.datetime.now().year
    for need, route, query in framework_queries(args.topic, current_year):
        add_task(
            tasks,
            lane="broad_framework_landscape",
            source_route=route,
            query=query,
            evidence_need=need,
            argument_role="framework_or_landscape",
            time_role="past_present_future",
            priority="P1",
            max_results=str(args.max_results_per_task),
            screening_prompt="Is this a seminal/current/counterevidence paper that should shape the review framework?",
        )

    draft_assets_dir = Path(args.draft_assets_dir) if args.draft_assets_dir else Path()
    systems_path = draft_assets_dir / "body_mentioned_systems.csv"
    for row in read_csv(systems_path)[: args.max_systems]:
        system = clean_space(row.get("system_or_method"))
        for route, query in system_queries(system, args.topic):
            add_task(
                tasks,
                lane="body_system_landmark_recall",
                source_route=route,
                query=query,
                linked_claim=row.get("context", ""),
                evidence_need="official_paper_identity_or_landmark_context",
                argument_role="system_or_method_evidence",
                time_role="current_or_recent",
                priority="P1",
                max_results=str(args.max_results_per_task),
                screening_prompt=f"Does this result identify the official paper, published version, or key evaluation of `{system}`?",
                origin_file=str(systems_path),
            )

    unmapped_path = draft_assets_dir / "unmapped_claims.csv"
    for row in read_csv(unmapped_path)[: args.max_claims]:
        claim = clean_space(row.get("claim"))
        for route in ["pubmed", "crossref_openalex"]:
            add_task(
                tasks,
                lane="unmapped_claim_gap_recall",
                source_route=route,
                query=claim_query(claim, args.topic, route),
                linked_claim_id=row.get("claim_id", ""),
                linked_claim=claim,
                evidence_need="claim_gap_support_or_boundary",
                argument_role="claim_gap",
                time_role="current_or_recent",
                priority="P2",
                max_results=str(args.max_results_per_task),
                screening_prompt="Does this paper directly support or challenge the linked unmapped draft claim?",
                origin_file=str(unmapped_path),
            )

    discovery_dir = Path(args.literature_discovery_dir) if args.literature_discovery_dir else Path()
    for row in read_csv(discovery_dir / "api_search_tasks.csv"):
        route = clean_space(row.get("api_route"))
        if route == "human":
            continue
        add_task(
            tasks,
            lane="literature_discovery_api_task",
            source_route=route,
            query=row.get("query", ""),
            linked_claim=row.get("claim_or_argument", ""),
            evidence_need=row.get("evidence_need", ""),
            argument_role=row.get("candidate_lane", "literature_discovery"),
            time_role="as_stated_by_subagent",
            priority=row.get("priority", "P2"),
            max_results=str(args.max_results_per_task),
            screening_prompt="Does this API hit satisfy the subagent's evidence need without drifting away from the confirmed framework?",
            origin_file=str(discovery_dir / "api_search_tasks.csv"),
        )

    tasks = dedupe_tasks(tasks, args.max_tasks)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "supplemental_recall_tasks.csv", tasks, TASK_FIELDS)
    query_dir = out_dir / "queries"
    query_dir.mkdir(exist_ok=True)
    for route in sorted(ROUTES):
        lines = [row["query"] for row in tasks if row.get("source_route") == route]
        (query_dir / f"{route}_queries.txt").write_text("\n".join(dict.fromkeys(lines)).rstrip() + ("\n" if lines else ""), encoding="utf-8")
    lines = [
        "# Supplemental Recall Plan",
        "",
        f"- Generated at: {now_iso()}",
        f"- Topic: {args.topic or '[not specified]'}",
        f"- Tasks: {len(tasks)}",
        "",
        "This plan expands the candidate pool after draft-derived logic has been read. It is not permission to promote papers directly.",
        "",
        "## Route Counts",
        "",
    ]
    counts: dict[str, int] = {}
    for row in tasks:
        counts[row["source_route"]] = counts.get(row["source_route"], 0) + 1
    for route, count in sorted(counts.items()):
        lines.append(f"- `{route}`: {count}")
    lines.extend(["", "## Gates", "", "- Run route-specific API recall.", "- Merge candidates and create agent screening packets.", "- Only screened inclusions go to deterministic verification.", "- Verified/adjudicated inclusions join the verified union; all others stay in supplemental/background lanes."])
    (out_dir / "supplemental_recall_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out_dir / "supplemental_recall_manifest.json", {"generated_at": now_iso(), "topic": args.topic, "tasks": len(tasks), "counts": counts})
    print(json.dumps({"out_dir": str(out_dir), "tasks": len(tasks), "route_counts": counts}, ensure_ascii=False))
    return 0


def load_tasks(path: Path, route: str = "") -> list[dict[str, str]]:
    rows = read_csv(path)
    if route:
        rows = [row for row in rows if row.get("source_route") == route]
    if not rows:
        raise SystemExit(f"No tasks found: {path} route={route or '*'}")
    return rows


def pubmed_task(task: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    calls: list[dict[str, Any]] = []
    query = clean_space(task.get("query"))
    papers: list[dict[str, str]] = []
    ids: list[str] = []
    search_url = ""
    fetch_url = ""
    error = ""
    retries = max(int(getattr(args, "retries", 0) or 0), 0)
    for attempt in range(retries + 1):
        try:
            ids, search_url = pubmed_recall.esearch(query, args)
            time.sleep(args.sleep)
            papers, fetch_url = pubmed_recall.efetch(ids, args)
            error = ""
            break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(max(args.sleep, 0.1) * (attempt + 2))
            else:
                calls.append({"task_id": task.get("task_id"), "query": query, "error": error, "attempts": retries + 1})
                return {"task_id": task.get("task_id"), "rows": rows, "calls": calls, "error": error}
    calls.append({"task_id": task.get("task_id"), "query": query, "search_url": search_url, "fetch_url": fetch_url, "pmids": ids})
    for paper in papers:
        pmid = paper.get("pmid") or paper.get("paper_id", "")
        title = clean_space(paper.get("title"))
        cid = safe_id("|".join([task.get("task_id", ""), pmid, title]), "SRC")
        rows.append(
            {
                "candidate_id": cid,
                "task_id": task.get("task_id", ""),
                "lane": task.get("lane", ""),
                "source_route": "pubmed",
                "query": query,
                "title": title,
                "authors": paper.get("authors", ""),
                "year": paper.get("year", ""),
                "journal": paper.get("journal", ""),
                "source": "pubmed",
                "paper_id": paper.get("paper_id", ""),
                "pmid": pmid,
                "doi": normalize_doi(paper.get("doi")),
                "url": paper.get("url", ""),
                "pdf_url": paper.get("pdf_url", ""),
                "abstract": paper.get("abstract", ""),
                "publication_type": paper.get("publication_type", ""),
                "linked_claim_id": task.get("linked_claim_id", ""),
                "linked_claim": task.get("linked_claim", ""),
                "evidence_need": task.get("evidence_need", ""),
                "argument_role": task.get("argument_role", ""),
                "time_role": task.get("time_role", ""),
                "recall_queries": query,
                "candidate_fit_status": "needs_agent_screening",
                "candidate_fit_reason": "Broad supplemental PubMed recall; do not promote without agent/human claim-fit screening.",
                "pool_status": "supplemental_unscreened",
                "human_decision_needed": "include_core | include_support | include_counterevidence | include_background | reject | needs_fulltext",
                "notes": task.get("screening_prompt", ""),
            }
        )
    return {"task_id": task.get("task_id"), "rows": rows, "calls": calls}


def cmd_pubmed(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(Path(args.tasks_csv), route="pubmed")
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
    all_rows: list[dict[str, str]] = []
    run_log: list[dict[str, Any]] = []
    progress = out_dir / "pubmed_supplemental_progress.jsonl"
    lock = threading.Lock()
    if not args.resume and progress.exists():
        progress.unlink()
    completed = set()
    if args.resume and progress.exists():
        for line in progress.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                completed.add(item.get("task_id"))
                all_rows.extend(item.get("rows") or [])
                run_log.extend(item.get("calls") or [])
    todo = [task for task in tasks if task.get("task_id") not in completed]
    if args.workers > 1 and len(todo) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(pubmed_task, task, args): task for task in todo}
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    task = futures[future]
                    result = {
                        "task_id": task.get("task_id"),
                        "rows": [],
                        "calls": [{"task_id": task.get("task_id"), "query": task.get("query"), "error": f"{type(exc).__name__}: {exc}"}],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                all_rows.extend(result["rows"])
                run_log.extend(result["calls"])
                append_jsonl(progress, result, lock)
    else:
        for task in todo:
            result = pubmed_task(task, args)
            all_rows.extend(result["rows"])
            run_log.extend(result["calls"])
            append_jsonl(progress, result, lock)
    rows = dedupe_candidates(all_rows)
    error_count = sum(1 for call in run_log if call.get("error"))
    write_csv(out_dir / "pubmed_supplemental_candidates.csv", rows, CANDIDATE_FIELDS)
    write_json(out_dir / "pubmed_supplemental_run_log.json", {"generated_at": now_iso(), "tasks": len(tasks), "rows": len(rows), "errors": error_count, "calls": run_log})
    write_screening_packet(out_dir / "agent_screening_packet.md", rows, args.max_abstract_chars)
    write_csv(out_dir / "agent_screening_template.csv", [{field: "" for field in SCREENING_FIELDS} for _ in rows], SCREENING_FIELDS)
    print(json.dumps({"out_dir": str(out_dir), "tasks": len(tasks), "rows": len(rows), "errors": error_count}, ensure_ascii=False))
    return 0


def candidate_key(row: dict[str, str]) -> str:
    doi = normalize_doi(row.get("doi"))
    if doi:
        return "doi:" + doi
    pmid = clean_space(row.get("pmid") or row.get("paper_id"))
    if pmid and row.get("source") == "pubmed":
        return "pmid:" + pmid
    title = normalize_title(row.get("title"))
    year = clean_space(row.get("year"))
    return f"title:{title}|year:{year}"


def dedupe_candidates(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for row in rows:
        key = candidate_key(row)
        if not key.strip(":|"):
            continue
        if key not in best:
            best[key] = row
            order.append(key)
            continue
        existing = best[key]
        for field in ["task_id", "lane", "source_route", "query", "recall_queries", "linked_claim_id", "linked_claim", "evidence_need", "argument_role", "time_role", "notes"]:
            parts = {part.strip() for part in (existing.get(field, "") + ";" + row.get(field, "")).split(";") if part.strip()}
            existing[field] = "; ".join(sorted(parts))
        for field, value in row.items():
            if not existing.get(field) and value:
                existing[field] = value
    out = [best[key] for key in order]
    seen_ids: set[str] = set()
    for row in out:
        cid = clean_space(row.get("candidate_id")) or safe_id(candidate_key(row), "SRC")
        if cid in seen_ids:
            suffix = safe_id(candidate_key(row), "DUP").split("-", 1)[1][:8]
            cid = f"{cid}-{suffix}"
        row["candidate_id"] = cid
        seen_ids.add(cid)
    return out


def normalize_candidate_row(row: dict[str, str], source_path: Path) -> dict[str, str]:
    title = clean_space(row.get("title") or row.get("candidate_title") or row.get("draft_candidate_title"))
    doi = normalize_doi(row.get("doi"))
    pmid = clean_space(row.get("pmid") or (row.get("paper_id") if clean_space(row.get("source")) == "pubmed" else ""))
    cid = clean_space(row.get("candidate_id")) or safe_id("|".join([title, doi, pmid, str(source_path)]), "SRC")
    query = clean_space(row.get("query") or row.get("recall_queries") or row.get("_recall_query"))
    return {
        "candidate_id": cid,
        "task_id": clean_space(row.get("task_id")),
        "lane": clean_space(row.get("lane") or row.get("retrieval_lane") or "external_recall"),
        "source_route": clean_space(row.get("source_route") or row.get("source") or source_path.parent.name),
        "query": query,
        "title": title,
        "authors": clean_space(row.get("authors")),
        "year": clean_space(row.get("year") or row.get("publication_year")),
        "journal": clean_space(row.get("journal")),
        "source": clean_space(row.get("source")),
        "paper_id": clean_space(row.get("paper_id") or row.get("id")),
        "pmid": pmid,
        "doi": doi,
        "url": clean_space(row.get("url") or row.get("paper_url") or row.get("landing_page_url")),
        "pdf_url": clean_space(row.get("pdf_url") or row.get("open_access_pdf")),
        "abstract": clean_space(row.get("abstract") or row.get("summary")),
        "publication_type": clean_space(row.get("publication_type")),
        "linked_claim_id": clean_space(row.get("linked_claim_id") or row.get("claim_id")),
        "linked_claim": clean_space(row.get("linked_claim") or row.get("claim_supported")),
        "evidence_need": clean_space(row.get("evidence_need")),
        "argument_role": clean_space(row.get("argument_role") or row.get("claim_role")),
        "time_role": clean_space(row.get("time_role") or row.get("argument_time_role")),
        "recall_queries": query,
        "candidate_fit_status": clean_space(row.get("candidate_fit_status") or "needs_agent_screening"),
        "candidate_fit_reason": clean_space(row.get("candidate_fit_reason") or row.get("screening_reason") or "Needs agent/human screening before verification."),
        "screening_score": clean_space(row.get("screening_score")),
        "screening_priority": clean_space(row.get("screening_priority")),
        "screening_reasons": clean_space(row.get("screening_reasons")),
        "seed_lock": clean_space(row.get("seed_lock")),
        "seed_origin": clean_space(row.get("seed_origin")),
        "pool_status": clean_space(row.get("pool_status") or row.get("screening_status") or "supplemental_unscreened"),
        "human_decision_needed": clean_space(row.get("human_decision_needed") or "include_core | include_support | include_counterevidence | include_background | reject | needs_fulltext"),
        "notes": clean_space(row.get("notes")),
    }


def discover_candidate_csvs(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for raw in args.candidate_csv:
        path = Path(raw)
        if path.exists():
            paths.append(path)
    for raw in args.recall_dir:
        root = Path(raw)
        for name in ["pubmed_supplemental_candidates.csv", "pubmed_argument_candidates.csv", "papers.csv", "screening.csv"]:
            path = root / name
            if path.exists():
                paths.append(path)
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path.resolve()).lower()
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def load_seed_rows(paths: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        for row in read_csv(path):
            seed = normalize_candidate_row(row, path)
            seed["lane"] = "seed_lock"
            seed["source_route"] = clean_space(seed.get("source_route") or row.get("source") or "seed_csv")
            seed["linked_claim_id"] = clean_space(seed.get("linked_claim_id") or row.get("claim_supported"))
            seed["linked_claim"] = clean_space(seed.get("linked_claim") or row.get("use_in_review") or row.get("claim_supported"))
            seed["evidence_need"] = clean_space(seed.get("evidence_need") or row.get("use_in_review") or "core_seed_or_verified_draft_asset")
            seed["candidate_fit_status"] = "seed_locked_needs_agent_role_assignment"
            seed["candidate_fit_reason"] = clean_space(row.get("decision_rationale") or row.get("notes") or "Already verified or curated before supplemental recall; do not drop during broad screening.")
            seed["pool_status"] = clean_space(row.get("pool_status") or "seed_locked")
            seed["seed_lock"] = "yes"
            seed["seed_origin"] = str(path)
            seed["human_decision_needed"] = "include_core | include_support | include_background | needs_fulltext | duplicate_already_verified"
            rows.append(seed)
    return rows


def parse_year(value: Any) -> int:
    match = re.search(r"(19|20)\d{2}", clean_space(value))
    return int(match.group(0)) if match else 0


def score_candidate(row: dict[str, str], topic: str = "") -> tuple[int, list[str]]:
    """Rank candidates for agent attention without making inclusion decisions."""
    title = clean_space(row.get("title"))
    abstract = clean_space(row.get("abstract"))
    journal = clean_space(row.get("journal"))
    source = clean_space(row.get("source")).lower()
    query = clean_space(row.get("query") or row.get("recall_queries"))
    evidence_need = clean_space(row.get("evidence_need")).lower()
    argument_role = clean_space(row.get("argument_role") or row.get("lane")).lower()
    time_role = clean_space(row.get("time_role")).lower()
    doi = normalize_doi(row.get("doi"))
    topic_text = clean_space(topic).lower()
    title_text = title.lower()
    text = " ".join([title, abstract, journal]).lower()
    query_text = query.lower()
    score = 0
    reasons: list[str] = []
    title_core_signal = False
    is_seed_locked = clean_space(row.get("seed_lock")).lower() in {"yes", "true", "1", "seed_locked"}
    if is_seed_locked:
        score += 20
        reasons.append("seed_locked_core_asset")
        title_core_signal = True

    positive_patterns = [
        (r"\b(ai co[- ]scientist|co[- ]scientist|coscientist)\b", 6, "ai_co_scientist"),
        (r"\b(robot scientist|scientific discovery|autonomous scientific discovery)\b", 5, "scientific_discovery_agent"),
        (r"\b(self[- ]?(evolving|improving|refinement|reflection)|reflexion)\b", 4, "self_evolving_or_reflective"),
        (r"\b(closed[- ]loop|self[- ]driving laborator|autonomous laborator|autonomous experiment)\b", 4, "closed_loop_or_self_driving_lab"),
        (r"\b(large language model|llm|language model agent|agentic|multi[- ]agent|autogen|hugginggpt)\b", 4, "llm_agent"),
        (r"\b(autonomous agent|ai agent|foundation model agent)\b", 4, "ai_agent"),
    ]
    biomedical_patterns = [
        (r"\b(biomed\w*|medic\w*|clinical|healthcare|patient|diagnos\w*|therap\w*|drug discovery|pharmaceutical)\b", 3, "medical_or_clinical"),
        (r"\b(protein|proteom\w*|genom\w*|cell|cellular|omics|biology|biological|chemistry|chemical|molecular|crispr|single[- ]cell)\b", 2, "biomedical_science_domain"),
    ]
    venue_patterns = [
        (r"\b(nature|science|cell|lancet|nejm|jama|pnas)\b", 3, "high_visibility_venue"),
        (r"\b(nature biotechnology|nature methods|nature medicine|nature machine intelligence|nature biomedical engineering)\b", 4, "nature_family_specialty"),
    ]
    for pattern, weight, reason in positive_patterns:
        if re.search(pattern, title_text, flags=re.IGNORECASE):
            score += weight + 2
            reasons.append("title_" + reason)
            title_core_signal = True
        elif re.search(pattern, text, flags=re.IGNORECASE):
            score += weight
            reasons.append(reason)
    for pattern, weight, reason in biomedical_patterns + venue_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            score += weight
            reasons.append(reason)
    has_biomedical_signal = any(reason in reasons for reason in ["medical_or_clinical", "biomedical_science_domain"])

    if re.search(r"\b(agent|autonomous|scientific discovery|large language model|llm|co[- ]scientist)\b", query_text, flags=re.IGNORECASE):
        score += 1
        reasons.append("query_is_topic_relevant")
    if doi:
        score += 1
        reasons.append("has_doi")
    if row.get("pmid"):
        score += 2
        reasons.append("has_pmid")
    if abstract:
        score += 1
        reasons.append("has_abstract")
    if "pubmed" in source:
        score += 2
        reasons.append("pubmed_record")
    elif "openalex" in source and "crossref" in source:
        score += 2
        reasons.append("cross_source_match")
    elif "openalex" in source or "crossref" in source:
        score += 1
        reasons.append("bibliographic_record")

    year = parse_year(row.get("year"))
    if year >= 2020:
        score += 1
        reasons.append("recent")
    elif year and year < 2000:
        score -= 1
        reasons.append("older_background_candidate")

    if re.search(r"(\.s\d+$|/mm\d+$|supplement|supporting information|_supp|\.pdf$)", doi + " " + title, flags=re.IGNORECASE):
        score -= 7
        reasons.append("supplement_or_media_artifact")
    pub_type = clean_space(row.get("publication_type")).lower()
    if pub_type in {"component", "dataset", "peer-review", "book-chapter", "dissertation", "grant", "reference-entry", "posted-content", "report-component"}:
        score -= 7
        reasons.append("non_article_metadata_type")
    if "preprint" in pub_type or "preprint server" in journal.lower():
        score -= 2
        reasons.append("preprint_record_needs_formal_version")
    if re.search(r"\b(accounting|education|network engineering|autonomous driving|lidar|tuberculosis model)\b", text, flags=re.IGNORECASE):
        score -= 3
        reasons.append("likely_topic_drift")
    if re.search(r"\b(biomed\w*|medic\w*|clinical|healthcare|patient|drug|biology|biological|生物|医学|医疗)\b", topic_text, flags=re.IGNORECASE) and not has_biomedical_signal:
        score -= 5
        reasons.append("topic_domain_mismatch")
    if not abstract and not re.search(r"\b(agent|autonomous|scientific discovery|large language model|llm|co[- ]scientist|biomed\w*|medic\w*|clinical|drug|protein|genom\w*|cell|biology|chemistry)\b", text, flags=re.IGNORECASE):
        score -= 3
        reasons.append("no_abstract_and_weak_title_signal")
    if not title_core_signal:
        score -= 3
        reasons.append("no_title_core_signal")
    if (
        "pubmed_deep_dive" in evidence_need
        and "core_biomedical_system_evidence" in argument_role
        and "agent_supervised_strategic_pubmed" in time_role
        and not title_core_signal
    ):
        score -= 13
        reasons.append("strategic_pubmed_broad_hit_without_agent_title_signal")
    elif "pubmed_deep_dive" in evidence_need and "core_biomedical_system_evidence" in argument_role and not title_core_signal:
        score -= 6
        reasons.append("core_deep_dive_without_agent_title_signal")
    if not year:
        score -= 1
        reasons.append("missing_year")

    return score, list(dict.fromkeys(reasons))


def screening_priority(score: int, reasons: list[str]) -> str:
    lacks_title_core = "no_title_core_signal" in reasons
    has_artifact = "supplement_or_media_artifact" in reasons
    is_seed_locked = "seed_locked_core_asset" in reasons
    has_domain_mismatch = "topic_domain_mismatch" in reasons
    if is_seed_locked:
        return "P0_screen_first"
    if has_artifact and score < 8:
        return "P4_likely_noise"
    if "non_article_metadata_type" in reasons and score < 8:
        return "P4_likely_noise"
    if "strategic_pubmed_broad_hit_without_agent_title_signal" in reasons:
        return "P3_low" if score >= 4 else "P4_likely_noise"
    if "core_deep_dive_without_agent_title_signal" in reasons and has_domain_mismatch:
        return "P3_low" if score >= 4 else "P4_likely_noise"
    if has_domain_mismatch and score >= 7:
        return "P1_high"
    if has_domain_mismatch:
        return "P2_medium" if score >= 4 else ("P3_low" if score >= 1 else "P4_likely_noise")
    if lacks_title_core and score >= 4:
        return "P2_medium"
    if score >= 10:
        return "P0_screen_first"
    if score >= 7:
        return "P1_high"
    if score >= 4:
        return "P2_medium"
    if score >= 1:
        return "P3_low"
    return "P4_likely_noise"


def rank_candidates(rows: list[dict[str, str]], topic: str = "") -> list[dict[str, str]]:
    for row in rows:
        score, reasons = score_candidate(row, topic)
        row["screening_score"] = str(score)
        row["screening_priority"] = screening_priority(score, reasons)
        row["screening_reasons"] = "; ".join(reasons)
    priority_order = {"P0_screen_first": 0, "P1_high": 1, "P2_medium": 2, "P3_low": 3, "P4_likely_noise": 4}
    rows.sort(
        key=lambda row: (
            priority_order.get(row.get("screening_priority", ""), 9),
            -int(row.get("screening_score") or 0),
            row.get("year", ""),
            row.get("title", ""),
        )
    )
    return rows


def select_screening_queue(rows: list[dict[str, str]], min_score: int, max_candidates: int, include_quarantine: bool) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if int(row.get("screening_score") or 0) >= min_score
        and (include_quarantine or row.get("screening_priority") != "P4_likely_noise")
    ]
    return selected[:max_candidates] if max_candidates > 0 else selected


def write_screening_packet(
    path: Path,
    rows: list[dict[str, str]],
    max_abstract_chars: int = 1200,
    start_index: int = 1,
    total_candidates: int | None = None,
    packet_label: str = "",
) -> None:
    lines = [
        "# Supplemental Recall Agent Screening Packet",
        "",
        "Screen these broad-recall candidates against the confirmed review framework. Do not promote a paper merely because it is interesting.",
        "",
        "Allowed decisions: include_core, include_support, include_counterevidence, include_framework, include_historical_foundation, include_method_foundation, include_governance_background, include_background, reject, needs_fulltext, duplicate_already_verified.",
        "",
        "Decision contract: bind each inclusion to a concrete review claim. Fill `argument_role`, `evidence_level`, `evidence_strength`, `fulltext_need`, `citation_role`, and `key_supported_claim`; use `needs_fulltext` whenever the abstract is insufficient for method/result/metric claims.",
        "",
        "The deterministic score only controls review order; it is not an inclusion decision.",
        "",
        "Use include_framework/include_historical_foundation/include_method_foundation for papers that shape the review's genealogy or conceptual architecture but are not core biomedical-system evidence. These still go through formal source verification.",
        "",
        "Return CSV with columns:",
        ",".join(SCREENING_FIELDS),
        "",
    ]
    if packet_label:
        lines.extend([f"- Packet: {packet_label}", ""])
    if total_candidates is not None:
        end_index = start_index + len(rows) - 1
        lines.extend([f"- Candidate range: {start_index}-{end_index} of {total_candidates}", ""])
    for offset, row in enumerate(rows):
        idx = start_index + offset
        lines.extend(
            [
                f"## {idx}. {row.get('title') or row.get('candidate_id')}",
                "",
                f"- Candidate: `{row.get('candidate_id')}`",
                f"- Task/lane: `{row.get('task_id')}` / `{row.get('lane')}`",
                f"- Screening order: `{row.get('screening_priority') or 'unranked'}`; score `{row.get('screening_score') or '0'}`; reasons `{row.get('screening_reasons') or 'none'}`",
                f"- Source/year: {row.get('source') or 'unknown'}; {row.get('journal') or 'unknown'} ({row.get('year') or 'unknown'})",
                f"- PMID/DOI: {row.get('pmid') or 'missing'} / {row.get('doi') or 'missing'}",
                f"- Linked claim: {row.get('linked_claim')[:500] or '[none]'}",
                f"- Evidence need: `{row.get('evidence_need') or 'unspecified'}`",
                f"- Query: `{row.get('query') or row.get('recall_queries')}`",
                "",
                (row.get("abstract") or "[No abstract available]")[:max_abstract_chars],
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_screening_packets(out_dir: Path, rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    packet_size = max(0, int(args.packet_size))
    if packet_size == 0 or len(rows) <= packet_size:
        write_screening_packet(out_dir / "agent_screening_packet.md", rows, args.max_abstract_chars, 1, len(rows))
        return [
            {
                "packet_id": "packet_0001",
                "packet_file": "agent_screening_packet.md",
                "start_index": "1",
                "end_index": str(len(rows)),
                "candidates": str(len(rows)),
                "priorities": "; ".join(sorted({row.get("screening_priority", "") for row in rows if row.get("screening_priority")})),
            }
        ]

    packet_dir = out_dir / "screening_packets"
    packet_rows: list[dict[str, str]] = []
    for chunk_idx, start in enumerate(range(0, len(rows), packet_size), 1):
        chunk = rows[start : start + packet_size]
        packet_id = f"packet_{chunk_idx:04d}"
        packet_path = packet_dir / f"agent_screening_{chunk_idx:04d}.md"
        write_screening_packet(
            packet_path,
            chunk,
            args.max_abstract_chars,
            start + 1,
            len(rows),
            packet_id,
        )
        rel_path = packet_path.relative_to(out_dir)
        packet_rows.append(
            {
                "packet_id": packet_id,
                "packet_file": str(rel_path),
                "start_index": str(start + 1),
                "end_index": str(start + len(chunk)),
                "candidates": str(len(chunk)),
                "priorities": "; ".join(sorted({row.get("screening_priority", "") for row in chunk if row.get("screening_priority")})),
            }
        )

    lines = [
        "# Supplemental Recall Agent Screening Packet Index",
        "",
        f"- Screening queue candidates: {len(rows)}",
        f"- Packet size: {packet_size}",
        "",
        "Assign packets to Codex subagents. Each packet returns CSV rows with the shared screening schema.",
        "",
        "| Packet | Range | Candidates | Priorities | File |",
        "|---|---:|---:|---|---|",
    ]
    for row in packet_rows:
        lines.append(
            f"| {row['packet_id']} | {row['start_index']}-{row['end_index']} | {row['candidates']} | {row['priorities']} | `{row['packet_file']}` |"
        )
    (out_dir / "agent_screening_packet.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_csv(out_dir / "screening_packet_index.csv", packet_rows, ["packet_id", "packet_file", "start_index", "end_index", "candidates", "priorities"])
    return packet_rows


def cmd_collect_candidates(args: argparse.Namespace) -> int:
    paths = discover_candidate_csvs(args)
    seed_rows = load_seed_rows(args.seed_csv)
    if not paths and not seed_rows:
        raise SystemExit("No candidate CSVs found.")
    rows: list[dict[str, str]] = seed_rows[:]
    for path in paths:
        rows.extend(normalize_candidate_row(row, path) for row in read_csv(path))
    rows = rank_candidates(dedupe_candidates(rows), args.topic)
    screening_queue = select_screening_queue(rows, args.min_screening_score, args.max_screening_candidates, args.include_quarantine)
    seed_locked = [row for row in rows if clean_space(row.get("seed_lock")).lower() in {"yes", "true", "1", "seed_locked"}]
    quarantined = [row for row in rows if row.get("screening_priority") == "P4_likely_noise"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "supplemental_recall_candidates.csv", rows, CANDIDATE_FIELDS)
    write_csv(out_dir / "screening_queue.csv", screening_queue, CANDIDATE_FIELDS)
    write_csv(out_dir / "seed_locked_candidates.csv", seed_locked, CANDIDATE_FIELDS)
    write_csv(out_dir / "quarantined_candidates.csv", quarantined, CANDIDATE_FIELDS)
    packet_rows = write_screening_packets(out_dir, screening_queue, args)
    write_csv(out_dir / "agent_screening_template.csv", [{**{field: "" for field in SCREENING_FIELDS}, "candidate_id": row["candidate_id"], "task_id": row["task_id"]} for row in screening_queue], SCREENING_FIELDS)
    write_json(
        out_dir / "candidate_collection_manifest.json",
        {
            "generated_at": now_iso(),
            "input_files": [str(path) for path in paths],
            "seed_files": args.seed_csv,
            "candidates": len(rows),
            "screening_queue": len(screening_queue),
            "seed_locked": len(seed_locked),
            "quarantined": len(quarantined),
            "min_screening_score": args.min_screening_score,
            "max_screening_candidates": args.max_screening_candidates,
            "packet_size": args.packet_size,
            "topic": args.topic,
            "packets": packet_rows,
        },
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "input_files": len(paths),
                "seed_files": len(args.seed_csv),
                "candidates": len(rows),
                "screening_queue": len(screening_queue),
                "seed_locked": len(seed_locked),
                "quarantined": len(quarantined),
                "packets": len(packet_rows),
            },
            ensure_ascii=False,
        )
    )
    return 0


def verification_lane(row: dict[str, str], decision: dict[str, str]) -> str:
    dec = clean_space(decision.get("decision"))
    if dec in {"include_core", "include_support", "include_counterevidence", "needs_fulltext"}:
        return "core_biomedical_system_evidence"
    if dec == "include_historical_foundation":
        return "historical_foundation"
    if dec == "include_method_foundation":
        return "method_foundation"
    if dec == "include_governance_background":
        return "governance_background"
    if dec == "include_framework":
        return "framework_verified"
    evidence_text = " ".join(
        clean_space(decision.get(field))
        for field in ["evidence_role", "required_next_action", "rationale", "claim_fit"]
    )
    evidence_text = f"{evidence_text} {row.get('evidence_need', '')} {row.get('argument_role', '')} {row.get('lane', '')}"
    if FRAMEWORK_BACKGROUND_HINTS.search(evidence_text):
        return "framework_or_historical_background"
    return "background_hold"


def should_send_to_verifier(row: dict[str, str], decision: dict[str, str]) -> bool:
    dec = clean_space(decision.get("decision"))
    if dec in CORE_VERIFY_DECISIONS or dec in FRAMEWORK_VERIFY_DECISIONS:
        return True
    if dec == "include_background":
        return verification_lane(row, decision) != "background_hold"
    return False


def verifier_row(row: dict[str, str], decision: dict[str, str]) -> dict[str, str]:
    reason = clean_space(decision.get("rationale") or row.get("candidate_fit_reason") or row.get("notes"))
    linked = clean_space(row.get("linked_claim"))
    claim_text = clean_space(decision.get("key_supported_claim") or linked)
    evidence_note = "; ".join(
        part
        for part in [
            clean_space(decision.get("argument_role")),
            clean_space(decision.get("evidence_level")),
            clean_space(decision.get("evidence_strength")),
            clean_space(decision.get("citation_role")),
            clean_space(decision.get("fulltext_need")),
        ]
        if part
    )
    lane = verification_lane(row, decision)
    priority = "high" if lane == "core_biomedical_system_evidence" else "medium"
    return {
        "draft": "supplemental_recall",
        "ref_number": row.get("candidate_id", ""),
        "candidate_id": row.get("candidate_id", ""),
        "claim_id": clean_space(decision.get("supported_claim_ids") or row.get("linked_claim_id")),
        "candidate_title": row.get("title", ""),
        "raw_reference": f"{row.get('authors', '')} {row.get('year', '')}. {row.get('title', '')}. {row.get('journal', '')}. {row.get('doi') or row.get('url')}",
        "doi": row.get("doi", ""),
        "pmid": row.get("pmid", ""),
        "urls": row.get("url", ""),
        "source_kind": f"screened_supplemental_recall_candidate:{lane}",
        "cited_in_body": "supplemental_recall",
        "verification_priority": priority,
        "candidate_type": decision.get("decision", ""),
        "pubmed_query": row.get("query", ""),
        "raw_evidence_excerpt": (row.get("abstract") or claim_text or linked)[:900],
        "why_relevant": clean_space("; ".join(part for part in [reason, claim_text, evidence_note] if part))[:700],
    }


def cmd_collect_screening(args: argparse.Namespace) -> int:
    candidates = {row.get("candidate_id", ""): row for row in read_csv(Path(args.candidates_csv))}
    decisions: list[dict[str, str]] = []
    for path in args.screening_csv:
        decisions.extend(read_csv(Path(path)))
    if not candidates:
        raise SystemExit(f"No candidates found: {args.candidates_csv}")
    if not decisions:
        raise SystemExit(f"No screening decisions found: {args.screening_csv}")
    accepted: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    verifier: list[dict[str, str]] = []
    core_verifier: list[dict[str, str]] = []
    framework_verifier: list[dict[str, str]] = []
    background_hold: list[dict[str, str]] = []
    for decision in decisions:
        cid = clean_space(decision.get("candidate_id"))
        row = candidates.get(cid)
        if not row:
            continue
        merged = {**row, **{f"screening_{k}": v for k, v in decision.items()}}
        dec = clean_space(decision.get("decision"))
        if dec in INCLUDE_DECISIONS:
            accepted.append(merged)
            if should_send_to_verifier(row, decision):
                verify_row = verifier_row(row, decision)
                verifier.append(verify_row)
                if verify_row.get("source_kind", "").endswith(":core_biomedical_system_evidence"):
                    core_verifier.append(verify_row)
                else:
                    framework_verifier.append(verify_row)
            elif dec in NO_VERIFIER_DECISIONS:
                background_hold.append(merged)
        else:
            rejected.append(merged)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    accept_fields = CANDIDATE_FIELDS + [f"screening_{field}" for field in SCREENING_FIELDS]
    write_csv(out_dir / "accepted_supplemental_recall_candidates.csv", accepted, accept_fields)
    write_csv(out_dir / "rejected_supplemental_recall_candidates.csv", rejected, accept_fields)
    write_csv(out_dir / "background_hold_candidates.csv", background_hold, accept_fields)
    write_csv(out_dir / "supplemental_candidates_for_verification.csv", verifier, VERIFY_FIELDS)
    write_csv(out_dir / "core_candidates_for_verification.csv", core_verifier, VERIFY_FIELDS)
    write_csv(out_dir / "framework_candidates_for_verification.csv", framework_verifier, VERIFY_FIELDS)
    lines = [
        "# Supplemental Screening Collection",
        "",
        f"- Candidates: {len(candidates)}",
        f"- Decisions: {len(decisions)}",
        f"- Accepted/background/fulltext: {len(accepted)}",
        f"- Verifier-ready inclusions: {len(verifier)}",
        f"- Core biomedical verifier-ready: {len(core_verifier)}",
        f"- Framework/historical/background verifier-ready: {len(framework_verifier)}",
        f"- Background hold, not yet citable: {len(background_hold)}",
        f"- Rejected/other: {len(rejected)}",
        "",
        "Run deterministic verification on `core_candidates_for_verification.csv` and `framework_candidates_for_verification.csv` as separate lanes, then source adjudication, then merge into the verified union with lane labels preserved.",
    ]
    (out_dir / "supplemental_screening_collection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "accepted": len(accepted),
                "verifier_ready": len(verifier),
                "core_verifier_ready": len(core_verifier),
                "framework_verifier_ready": len(framework_verifier),
                "background_hold": len(background_hold),
                "rejected": len(rejected),
            },
            ensure_ascii=False,
        )
    )
    return 0


def count_values(rows: list[dict[str, str]], field: str) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for row in rows:
        value = clean_space(row.get(field)) or "(blank)"
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def markdown_count_table(title: str, rows: list[tuple[str, int]], limit: int = 20) -> list[str]:
    lines = [f"## {title}", "", "| Value | Count |", "|---|---:|"]
    for value, count in rows[:limit]:
        lines.append(f"| {value.replace('|', '/')} | {count} |")
    lines.append("")
    return lines


def cmd_audit_pool(args: argparse.Namespace) -> int:
    collection_dir = Path(args.collection_dir)
    out_dir = Path(args.out_dir) if args.out_dir else collection_dir
    all_rows = read_csv(collection_dir / "supplemental_recall_candidates.csv")
    queue_rows = read_csv(collection_dir / "screening_queue.csv")
    quarantine_rows = read_csv(collection_dir / "quarantined_candidates.csv")
    seed_rows = read_csv(collection_dir / "seed_locked_candidates.csv")
    packet_rows = read_csv(collection_dir / "screening_packet_index.csv")
    if not all_rows:
        raise SystemExit(f"No supplemental_recall_candidates.csv found in {collection_dir}")

    screened_ids: set[str] = set()
    accepted_rows: list[dict[str, str]] = []
    rejected_rows: list[dict[str, str]] = []
    verifier_rows: list[dict[str, str]] = []
    if args.screened_dir:
        screened_dir = Path(args.screened_dir)
        accepted_rows = read_csv(screened_dir / "accepted_supplemental_recall_candidates.csv")
        rejected_rows = read_csv(screened_dir / "rejected_supplemental_recall_candidates.csv")
        verifier_rows = read_csv(screened_dir / "supplemental_candidates_for_verification.csv")
        for row in accepted_rows + rejected_rows:
            cid = clean_space(row.get("candidate_id"))
            if cid:
                screened_ids.add(cid)

    verification_rows: list[dict[str, str]] = []
    if args.verification_dir:
        verification_rows = read_csv(Path(args.verification_dir) / "all_draft_reference_verification.csv")

    queue_ids = {clean_space(row.get("candidate_id")) for row in queue_rows if clean_space(row.get("candidate_id"))}
    quarantine_ids = {clean_space(row.get("candidate_id")) for row in quarantine_rows if clean_space(row.get("candidate_id"))}
    backlog = [row for row in queue_rows if clean_space(row.get("candidate_id")) not in screened_ids]
    deferred = [
        row
        for row in all_rows
        if clean_space(row.get("candidate_id")) not in queue_ids and clean_space(row.get("candidate_id")) not in quarantine_ids
    ]
    rescue = quarantine_rows[: args.top_n]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "screening_backlog.csv", backlog, CANDIDATE_FIELDS)
    write_csv(out_dir / "deferred_not_queued_candidates.csv", deferred, CANDIDATE_FIELDS)
    write_csv(out_dir / "quarantine_rescue_sample.csv", rescue, CANDIDATE_FIELDS)

    lines = [
        "# Supplemental Recall Pool Audit",
        "",
        f"- Generated at: {now_iso()}",
        f"- Collection dir: `{collection_dir}`",
        f"- All deduplicated candidates: {len(all_rows)}",
        f"- Screening queue: {len(queue_rows)}",
        f"- Screened decisions observed: {len(screened_ids)}",
        f"- Unscreened queue backlog: {len(backlog)}",
        f"- Deferred, not queued because of caps: {len(deferred)}",
        f"- Seed-locked records: {len(seed_rows)}",
        f"- Quarantined records: {len(quarantine_rows)}",
        f"- Screening packets: {len(packet_rows)}",
        "",
        "## Interpretation",
        "",
        "- `supplemental_recall_candidates.csv` is the broad recall pool and should not be read as accepted literature.",
        "- `screening_queue.csv` is the ranked worklist for Codex subagents or human screeners.",
        "- `quarantined_candidates.csv` is not deleted; it is a lower-priority rescue lane for sampling or targeted recovery.",
        "- `seed_locked_candidates.csv` preserves draft-derived verified assets so broad recall cannot drown them out.",
        "- Framework, historical, method, and governance papers should be source-verified in their own lane; they should not be mixed into the core biomedical-system evidence table.",
        "",
    ]
    lines += markdown_count_table("All Candidates By Source Route", count_values(all_rows, "source_route"))
    lines += markdown_count_table("All Candidates By Lane", count_values(all_rows, "lane"))
    lines += markdown_count_table("Screening Queue By Priority", count_values(queue_rows, "screening_priority"))
    lines += markdown_count_table("Screening Queue By Lane", count_values(queue_rows, "lane"))
    if deferred:
        lines += markdown_count_table("Deferred Not Queued By Priority", count_values(deferred, "screening_priority"))
        lines += markdown_count_table("Deferred Not Queued By Lane", count_values(deferred, "lane"))
    if accepted_rows or rejected_rows:
        lines += markdown_count_table("Screened Accepted Decisions", count_values(accepted_rows, "screening_decision"))
        lines += markdown_count_table("Screened Rejected Decisions", count_values(rejected_rows, "screening_decision"))
    if verifier_rows:
        lines += markdown_count_table("Verifier-Ready Candidate Types", count_values(verifier_rows, "candidate_type"))
        lines += markdown_count_table("Verifier-Ready Source Lanes", count_values(verifier_rows, "source_kind"))
    if verification_rows:
        lines += markdown_count_table("Verification Outcomes", count_values(verification_rows, "verification_status"))

    lines.extend(["## Next Unscreened Queue Sample", "", "| Rank | Candidate ID | Priority | Score | Title | Source | Year |", "|---:|---|---|---:|---|---|---:|"])
    for idx, row in enumerate(backlog[: args.top_n], start=1):
        title = clean_space(row.get("title")).replace("|", "/")
        lines.append(
            f"| {idx} | `{row.get('candidate_id', '')}` | {row.get('screening_priority', '')} | {row.get('screening_score', '')} | {title[:120]} | {row.get('source_route', '')} | {row.get('year', '')} |"
        )
    lines.extend(["", "## Deferred Not Queued Sample", "", "| Rank | Candidate ID | Priority | Score | Title | Source | Year |", "|---:|---|---|---:|---|---|---:|"])
    for idx, row in enumerate(deferred[: args.top_n], start=1):
        title = clean_space(row.get("title")).replace("|", "/")
        lines.append(
            f"| {idx} | `{row.get('candidate_id', '')}` | {row.get('screening_priority', '')} | {row.get('screening_score', '')} | {title[:120]} | {row.get('source_route', '')} | {row.get('year', '')} |"
        )
    lines.extend(["", "## Quarantine Rescue Sample", "", "| Rank | Candidate ID | Score | Title | Reasons |", "|---:|---|---:|---|---|"])
    for idx, row in enumerate(rescue, start=1):
        title = clean_space(row.get("title")).replace("|", "/")
        reasons = clean_space(row.get("screening_reasons")).replace("|", "/")
        lines.append(f"| {idx} | `{row.get('candidate_id', '')}` | {row.get('screening_score', '')} | {title[:100]} | {reasons[:140]} |")
    lines.append("")

    report_path = out_dir / "recall_pool_audit.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "all_candidates": len(all_rows),
                "screening_queue": len(queue_rows),
                "screened": len(screened_ids),
                "backlog": len(backlog),
                "deferred_not_queued": len(deferred),
                "quarantined": len(quarantine_rows),
                "report": str(report_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


def recall_row_text(row: dict[str, str]) -> str:
    return " ".join(
        clean_space(row.get(field))
        for field in ["title", "abstract", "journal", "query", "screening_reasons", "evidence_need", "argument_role"]
    )


def candidate_intrinsic_text(row: dict[str, str]) -> str:
    return " ".join(clean_space(row.get(field)) for field in ["title", "abstract", "journal", "publication_type", "source"])


def is_preprintish_candidate(row: dict[str, str]) -> bool:
    text = candidate_intrinsic_text(row).lower() + " " + normalize_doi(row.get("doi"))
    pub_type = clean_space(row.get("publication_type")).lower()
    return any(marker in text for marker in ["arxiv", "biorxiv", "medrxiv", "10.1101/", "10.48550/arxiv"]) or pub_type in {
        "posted-content",
        "preprint",
    }


def is_framework_lineage_candidate(row: dict[str, str]) -> bool:
    text = " ".join(clean_space(row.get(field)) for field in ["title", "abstract", "journal", "evidence_need", "argument_role"])
    return bool(FRAMEWORK_BACKGROUND_HINTS.search(text)) or bool(
        re.search(r"\b(robot scientist|nobel turing|react|reflexion|scientific discovery|closed-loop|self-driving laborator)\b", text, flags=re.I)
    )


def is_pubmed_route(row: dict[str, str]) -> bool:
    return "pubmed" in clean_space(row.get("source_route") or row.get("source")).lower()


def exact_pubmed_title_query(title: str) -> str:
    title = clean_space(title).replace('"', "")
    if not title:
        return ""
    if len(title) > 220:
        title = title[:220].rsplit(" ", 1)[0]
    return f'"{title}"[Title]'


def pubmed_title_phrases(title: str, limit: int = 2) -> list[str]:
    title = clean_space(title).replace('"', "")
    phrases: list[str] = []
    parts = [part.strip(" .;:,-") for part in re.split(r"[:.]", title) if part.strip(" .;:,-")]
    for part in parts:
        words = [word for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", part) if word.lower() not in QUERY_STOPWORDS]
        if len(words) >= 4:
            phrase = " ".join(words[:8])
            if phrase and phrase.lower() != title.lower() and phrase not in phrases:
                phrases.append(phrase)
        if len(phrases) >= limit:
            break
    if len(phrases) < limit:
        words = [word for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", title) if word.lower() not in QUERY_STOPWORDS]
        if len(words) >= 4:
            phrase = " ".join(words[:8])
            if phrase and phrase not in phrases:
                phrases.append(phrase)
    return phrases[:limit]


def robust_pubmed_identity_query(title: str, doi: str = "", allow_title_fragments: bool = True) -> str:
    clauses: list[str] = []
    exact = exact_pubmed_title_query(title)
    if exact:
        clauses.append(exact)
    if allow_title_fragments:
        for phrase in pubmed_title_phrases(title):
            first = clean_space(phrase).split(" ", 1)[0].lower()
            if first and first not in WEAK_TITLE_FALLBACK_TERMS:
                clauses.append(f'"{phrase}"[Title]')
    doi = normalize_doi(doi)
    if doi:
        clauses.append(f'"{doi}"[AID]')
    clauses = list(dict.fromkeys(clauses))
    if not clauses:
        return ""
    return clauses[0] if len(clauses) == 1 else "(" + " OR ".join(clauses) + ")"


def title_key_terms(title: str, limit: int = 4) -> list[str]:
    terms: list[str] = []
    for match in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", title):
        token = match.strip("-").lower()
        if len(token) < 4 or token in QUERY_STOPWORDS or token in WEAK_TITLE_FALLBACK_TERMS:
            continue
        if token not in terms:
            terms.append(token)
    return terms[:limit]


def system_name_hint(title: str) -> str:
    head = clean_space(title).split(":", 1)[0]
    if head.lower() in AMBIGUOUS_SYSTEM_HINTS:
        return ""
    if 2 <= len(head) <= 48 and re.search(r"[A-Z0-9]", head) and len(head.split()) <= 5:
        return head.replace('"', "")
    return ""


def has_weak_single_term_query(query: str) -> bool:
    query_l = query.lower()
    for term in WEAK_TITLE_FALLBACK_TERMS:
        if re.search(rf'["( ]{re.escape(term)}"?\[title/abstract\]\s+or\b', query_l):
            return True
    if re.search(r'\b(robot\[title/abstract\]\s+or\s+scientist\[title/abstract\]\s+or\s+adam\[title/abstract\])', query_l):
        return True
    return False


def pubmed_supervision_query(row: dict[str, str], supervision_type: str) -> str:
    title = clean_space(row.get("title")).replace('"', "")
    if not title:
        doi = normalize_doi(row.get("doi"))
        return f'"{doi}"[AID]' if doi else ""
    identity = robust_pubmed_identity_query(title, row.get("doi"), allow_title_fragments=supervision_type != "formal_version_search")
    exact = exact_pubmed_title_query(title)
    if supervision_type == "formal_version_search":
        return identity
    if supervision_type in {
        "pubmed_deep_dive",
        "framework_lineage_pubmed",
        "method_foundation_pubmed",
        "governance_pubmed",
        "quarantine_rescue_review",
        "pool_triage_review",
    } and identity:
        return identity

    text = candidate_intrinsic_text(row).lower()
    system = system_name_hint(title)
    if ":" in title and not system and supervision_type == "pubmed_deep_dive":
        return identity
    agent_terms = []
    if re.search(r"\bagentic\b", text):
        agent_terms.append('"agentic AI"[Title/Abstract]')
    if re.search(r"\bmulti[- ]agent\b", text):
        agent_terms.append('"multi-agent"[Title/Abstract]')
    if re.search(r"\bllm|large language model\b", text):
        agent_terms.extend(['"large language model"[Title/Abstract]', "LLM[Title/Abstract]"])
    if re.search(r"\bautonomous\b", text):
        agent_terms.append("autonomous[Title/Abstract]")
    if re.search(r"\bco[- ]scientist\b", text):
        agent_terms.append('"co-scientist"[Title/Abstract]')
    if re.search(r"\bagent\b", text):
        agent_terms.append("agent[Title/Abstract]")
    agent_terms = list(dict.fromkeys(agent_terms))[:4]

    domain_terms = []
    for pattern, term in [
        (r"bioinformatics", "bioinformatics[Title/Abstract]"),
        (r"healthcare|clinical|patient", "(healthcare[Title/Abstract] OR clinical[Title/Abstract])"),
        (r"drug|therapeutic|pharma|antibiotic", "(drug[Title/Abstract] OR therapeutic[Title/Abstract] OR antibiotic[Title/Abstract])"),
        (r"single[- ]cell|cell", '("single-cell"[Title/Abstract] OR cell[Title/Abstract])'),
        (r"genom|crispr|gene", "(genomic[Title/Abstract] OR gene[Title/Abstract] OR CRISPR[Title/Abstract])"),
        (r"protein", "protein[Title/Abstract]"),
        (r"biomedical|biology|biological", "(biomedical[Title/Abstract] OR biology[Title/Abstract])"),
    ]:
        if re.search(pattern, text):
            domain_terms.append(term)
    domain_terms = list(dict.fromkeys(domain_terms))[:3]

    if system and agent_terms:
        return f'("{system}"[Title/Abstract] OR {exact})'
    if agent_terms and domain_terms:
        return f"({' OR '.join(agent_terms)}) AND ({' OR '.join(domain_terms)})"
    key_terms = title_key_terms(title, limit=3)
    if key_terms and (agent_terms or domain_terms):
        key_query = " OR ".join(f"{term}[Title/Abstract]" for term in key_terms)
        support_terms = " OR ".join(agent_terms or domain_terms)
        return f"({key_query}) AND ({support_terms})"
    return identity


def supervision_lane(row: dict[str, str], supervision_type: str) -> str:
    text = candidate_intrinsic_text(row)
    if supervision_type in {"framework_lineage_pubmed", "method_foundation_pubmed"}:
        if re.search(r"\b(robot scientist|nobel turing|historical|history|closed-loop|self-driving)\b", text, flags=re.I):
            return "historical_foundation"
        return "method_foundation"
    if supervision_type == "governance_pubmed":
        return "governance_background"
    if BIOMED_RECALL_SIGNAL.search(text) and AGENT_RECALL_SIGNAL.search(text):
        return "core_biomedical_system_evidence"
    if is_framework_lineage_candidate(row):
        return "framework_verified"
    return "background_hold"


def supervision_type_for_candidate(row: dict[str, str], bucket: str) -> str:
    text = candidate_intrinsic_text(row)
    if is_preprintish_candidate(row):
        return "formal_version_search"
    if is_framework_lineage_candidate(row):
        return "framework_lineage_pubmed"
    if re.search(r"\b(governance|safety|evaluation|regulation|clinical evidence|risk)\b", text, flags=re.I):
        return "governance_pubmed"
    if bucket == "quarantine":
        return "quarantine_rescue_review"
    if BIOMED_RECALL_SIGNAL.search(text) and AGENT_RECALL_SIGNAL.search(text):
        return "pubmed_deep_dive"
    return "pool_triage_review"


def supervision_priority(row: dict[str, str], bucket: str, supervision_type: str) -> str:
    if supervision_type in {"pubmed_deep_dive", "formal_version_search"}:
        return "high"
    if bucket == "backlog" and row.get("screening_priority") in {"P0_screen_first", "P1_high"}:
        return "high"
    if supervision_type in {"framework_lineage_pubmed", "governance_pubmed"}:
        return "medium"
    return "low" if bucket == "quarantine" else "medium"


def supervision_routes(row: dict[str, str], supervision_type: str, lane: str) -> tuple[str, str]:
    if supervision_type == "pubmed_deep_dive":
        return "pubmed", "crossref_openalex;publisher;human"
    if supervision_type == "formal_version_search":
        if normalize_doi(row.get("doi")):
            return "crossref_openalex;publisher", "pubmed"
        return "crossref_openalex;pubmed", "publisher;human"
    if supervision_type in {"framework_lineage_pubmed", "method_foundation_pubmed"}:
        if lane in {"historical_foundation", "method_foundation"} and not BIOMED_RECALL_SIGNAL.search(candidate_intrinsic_text(row)):
            return "crossref_openalex", "pubmed;arxiv_openreview"
        return "pubmed;crossref_openalex", "publisher;human"
    if supervision_type == "governance_pubmed":
        return "pubmed", "crossref_openalex;official_guideline"
    if supervision_type == "quarantine_rescue_review":
        return "pubmed;crossref_openalex", "human"
    return "human", "pubmed;crossref_openalex"


def supervision_seed_row(row: dict[str, str], bucket: str) -> dict[str, str]:
    supervision_type = supervision_type_for_candidate(row, bucket)
    title = clean_space(row.get("title"))
    query = pubmed_supervision_query(row, supervision_type)
    if normalize_doi(row.get("doi")) and not query:
        query = f'"{normalize_doi(row.get("doi"))}"[AID]'
    lane = supervision_lane(row, supervision_type)
    decision = "add_pubmed_query" if query and not is_pubmed_route(row) else "inspect_pool_candidate"
    if bucket == "quarantine":
        intrinsic = candidate_intrinsic_text(row)
        decision = "rescue_to_screening" if query and (AGENT_RECALL_SIGNAL.search(intrinsic) or BIOMED_RECALL_SIGNAL.search(intrinsic)) else "inspect_pool_candidate"
    if supervision_type == "formal_version_search":
        decision = "add_pubmed_query"
    preferred_route, fallback_route = supervision_routes(row, supervision_type, lane)
    return {
        "supervision_id": safe_id("|".join([bucket, row.get("candidate_id", ""), title, supervision_type]), "SUP"),
        "source_candidate_id": row.get("candidate_id", ""),
        "supervision_type": supervision_type,
        "decision": decision,
        "evidence_lane": lane,
        "preferred_route": preferred_route,
        "fallback_route": fallback_route,
        "priority": supervision_priority(row, bucket, supervision_type),
        "source_title": title,
        "proposed_title": title,
        "proposed_doi": normalize_doi(row.get("doi")),
        "proposed_pubmed_query": query,
        "proposed_crossref_openalex_query": f'"{title}"' if title else "",
        "rationale": clean_space(
            f"{bucket}; {row.get('screening_priority', '')}; score={row.get('screening_score', '')}; {row.get('screening_reasons', '')}"
        )[:900],
        "human_question": "",
        "next_action": "Run PubMed deep-dive, then deterministic verification if formal metadata is found.",
    }


def supervision_to_pubmed_task(row: dict[str, str], topic: str) -> dict[str, str] | None:
    query = clean_space(row.get("proposed_pubmed_query"))
    title = clean_space(row.get("proposed_title") or row.get("source_title"))
    supervision_type = clean_space(row.get("supervision_type") or row.get("evidence_need"))
    if title and query == exact_pubmed_title_query(title):
        allow_fragments = supervision_type != "formal_version_search"
        query = robust_pubmed_identity_query(title, row.get("proposed_doi"), allow_title_fragments=allow_fragments) or query
    elif title and has_weak_single_term_query(query):
        query = pubmed_supervision_query({"title": title, "doi": row.get("proposed_doi", "")}, supervision_type) or query
    if not query:
        return None
    lane = clean_space(row.get("evidence_lane")) or "agent_supervised_pubmed_deep_dive"
    decision = clean_space(row.get("decision"))
    if lane == "background_hold" and decision not in {"verify_as_core", "verify_as_framework", "verify_as_historical", "rescue_to_screening"}:
        return None
    tasks: list[dict[str, str]] = []
    add_task(
        tasks,
        lane=f"agent_supervised_{lane}",
        source_route="pubmed",
        query=query,
        linked_claim=clean_space(row.get("rationale")),
        evidence_need=clean_space(row.get("supervision_type")) or "agent_supervised_pubmed_deep_dive",
        argument_role=lane,
        time_role="agent_supervised_recall",
        priority="P0" if row.get("priority") == "high" else "P1",
        max_results="10",
        screening_prompt=f"Use PubMed to verify whether this supervised candidate has formal biomedical literature support for {topic}.",
        origin_file=clean_space(row.get("source_candidate_id")),
    )
    return tasks[0] if tasks else None


def add_strategic_pubmed_tasks(tasks: list[dict[str, str]], topic: str, max_tasks: int) -> None:
    if max_tasks <= 0:
        return
    topic_text = clean_space(topic) or "self-evolving biomedical agents"
    strategies = [
        {
            "lane": "agent_supervised_core_biomedical_system_evidence",
            "query": '("AI agent"[Title/Abstract] OR "multi-agent"[Title/Abstract] OR agentic[Title/Abstract] OR "large language model"[Title/Abstract]) AND (bioinformatics[Title/Abstract] OR biomedical[Title/Abstract] OR biology[Title/Abstract])',
            "evidence_need": "pubmed_deep_dive",
            "argument_role": "core_biomedical_system_evidence",
            "priority": "P0",
            "linked_claim": "Strategic PubMed deep dive: biomedical/bioinformatics agent systems possibly under-represented by Crossref/OpenAlex ranking.",
        },
        {
            "lane": "agent_supervised_core_biomedical_system_evidence",
            "query": '("AI agent"[Title/Abstract] OR "multi-agent"[Title/Abstract] OR LLM[Title/Abstract]) AND ("single-cell"[Title/Abstract] OR transcriptomics[Title/Abstract] OR "cell annotation"[Title/Abstract])',
            "evidence_need": "pubmed_deep_dive",
            "argument_role": "core_biomedical_system_evidence",
            "priority": "P0",
            "linked_claim": "Strategic PubMed deep dive: single-cell/transcriptomics agents and biomedical workflow automation.",
        },
        {
            "lane": "agent_supervised_core_biomedical_system_evidence",
            "query": '(agentic[Title/Abstract] OR "AI agent"[Title/Abstract] OR autonomous[Title/Abstract] OR "multi-agent"[Title/Abstract]) AND ("drug discovery"[Title/Abstract] OR therapeutic[Title/Abstract] OR pharmacology[Title/Abstract])',
            "evidence_need": "pubmed_deep_dive",
            "argument_role": "core_biomedical_system_evidence",
            "priority": "P0",
            "linked_claim": "Strategic PubMed deep dive: drug-discovery and therapeutic-discovery agent systems.",
        },
        {
            "lane": "agent_supervised_core_biomedical_system_evidence",
            "query": '("AI agent"[Title/Abstract] OR agentic[Title/Abstract] OR autonomous[Title/Abstract] OR LLM[Title/Abstract]) AND (CRISPR[Title/Abstract] OR "gene editing"[Title/Abstract] OR genomic[Title/Abstract])',
            "evidence_need": "pubmed_deep_dive",
            "argument_role": "core_biomedical_system_evidence",
            "priority": "P0",
            "linked_claim": "Strategic PubMed deep dive: genome-editing and CRISPR agent systems.",
        },
        {
            "lane": "agent_supervised_core_biomedical_system_evidence",
            "query": '("AI agent"[Title/Abstract] OR "large language model"[Title/Abstract] OR LLM[Title/Abstract]) AND ("protein design"[Title/Abstract] OR "protein engineering"[Title/Abstract] OR "protein sequences"[Title/Abstract])',
            "evidence_need": "pubmed_deep_dive",
            "argument_role": "core_biomedical_system_evidence",
            "priority": "P0",
            "linked_claim": "Strategic PubMed deep dive: protein-design or protein-engineering agent/foundation-model systems.",
        },
        {
            "lane": "agent_supervised_historical_foundation",
            "query": '("robot scientist"[Title/Abstract] OR "autonomous scientific discovery"[Title/Abstract] OR "closed-loop"[Title/Abstract] OR "self-driving laboratory"[Title/Abstract]) AND (biology[Title/Abstract] OR biomedical[Title/Abstract] OR scientific[Title/Abstract])',
            "evidence_need": "framework_lineage_pubmed",
            "argument_role": "historical_foundation",
            "priority": "P1",
            "linked_claim": "Strategic PubMed lineage query: robot scientist, autonomous scientific discovery, and closed-loop laboratory foundations.",
        },
        {
            "lane": "agent_supervised_governance_background",
            "query": '(agentic[Title/Abstract] OR "AI agent"[Title/Abstract] OR "large language model"[Title/Abstract]) AND (validation[Title/Abstract] OR evaluation[Title/Abstract] OR "clinical evidence"[Title/Abstract] OR limitation[Title/Abstract])',
            "evidence_need": "governance_pubmed",
            "argument_role": "governance_background",
            "priority": "P1",
            "linked_claim": "Strategic PubMed governance query: validation, evaluation, and limitations for clinical/biomedical agentic AI.",
        },
        {
            "lane": "agent_supervised_framework_verified",
            "query": '("tool use"[Title/Abstract] OR "retrieval augmented"[Title/Abstract] OR "knowledge graph"[Title/Abstract]) AND ("large language model"[Title/Abstract] OR LLM[Title/Abstract]) AND (biomedical[Title/Abstract] OR clinical[Title/Abstract])',
            "evidence_need": "framework_lineage_pubmed",
            "argument_role": "framework_verified",
            "priority": "P1",
            "linked_claim": "Strategic PubMed framework query: tool use, RAG, and knowledge-graph foundations in biomedical LLM systems.",
        },
    ]
    existing = {normalize_title(task.get("query")) for task in tasks}
    added = 0
    for strategy in strategies:
        if added >= max_tasks:
            break
        if normalize_title(strategy["query"]) in existing:
            continue
        add_task(
            tasks,
            lane=strategy["lane"],
            source_route="pubmed",
            query=strategy["query"],
            linked_claim=strategy["linked_claim"],
            evidence_need=strategy["evidence_need"],
            argument_role=strategy["argument_role"],
            time_role="agent_supervised_strategic_pubmed",
            priority=strategy["priority"],
            max_results="20",
            screening_prompt=f"Strategic PubMed query for {topic_text}: screen returned abstracts for direct relevance before verification.",
            origin_file="agent_supervised_strategic_pubmed",
        )
        added += 1


def dedupe_tasks_preserve_order(tasks: list[dict[str, str]]) -> list[dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    order: list[str] = []
    rank = {"P0": 5, "P1": 4, "P2": 3, "P3": 2, "P4": 1}
    for row in tasks:
        key = "|".join([row.get("source_route", ""), normalize_title(row.get("query", ""))])
        if key not in best:
            best[key] = row
            order.append(key)
        elif rank.get(row.get("priority"), 0) > rank.get(best[key].get("priority"), 0):
            best[key] = row
    return [best[key] for key in order]


def supervised_task_group(row: dict[str, str]) -> str:
    evidence_need = clean_space(row.get("evidence_need"))
    lane = clean_space(row.get("argument_role") or row.get("lane"))
    if evidence_need == "pubmed_deep_dive":
        return "pubmed_deep_dive"
    if evidence_need == "formal_version_search":
        return "formal_version_search"
    if evidence_need in {"framework_lineage_pubmed", "method_foundation_pubmed"} or lane in {
        "framework_verified",
        "historical_foundation",
        "method_foundation",
    }:
        return "framework_foundation"
    if evidence_need == "governance_pubmed" or lane == "governance_background":
        return "governance"
    if evidence_need == "quarantine_rescue_review" or "background_hold" in clean_space(row.get("lane")):
        return "quarantine_rescue"
    return "other"


def default_supervised_task_quotas(max_tasks: int) -> dict[str, int]:
    if max_tasks <= 0:
        return {}
    return {
        "pubmed_deep_dive": max(1, round(max_tasks * 0.40)),
        "formal_version_search": max(1, round(max_tasks * 0.30)),
        "framework_foundation": max(1, round(max_tasks * 0.17)),
        "governance": max(0, round(max_tasks * 0.06)),
        "quarantine_rescue": max(0, round(max_tasks * 0.07)),
    }


def supervised_task_quotas(args: argparse.Namespace) -> dict[str, int]:
    quotas = default_supervised_task_quotas(int(getattr(args, "max_pubmed_tasks", 0) or 0))
    overrides = {
        "pubmed_deep_dive": int(getattr(args, "deep_dive_task_quota", 0) or 0),
        "formal_version_search": int(getattr(args, "formal_version_task_quota", 0) or 0),
        "framework_foundation": int(getattr(args, "framework_task_quota", 0) or 0),
        "governance": int(getattr(args, "governance_task_quota", 0) or 0),
        "quarantine_rescue": int(getattr(args, "rescue_task_quota", 0) or 0),
    }
    for group, value in overrides.items():
        if value > 0:
            quotas[group] = value
    return quotas


def select_supervised_pubmed_tasks(tasks: list[dict[str, str]], max_tasks: int, quotas: dict[str, int]) -> list[dict[str, str]]:
    tasks = dedupe_tasks_preserve_order(tasks)
    if max_tasks <= 0:
        return tasks
    groups = {
        "pubmed_deep_dive": [],
        "formal_version_search": [],
        "framework_foundation": [],
        "governance": [],
        "quarantine_rescue": [],
        "other": [],
    }
    for task in tasks:
        groups.setdefault(supervised_task_group(task), []).append(task)

    selected: list[dict[str, str]] = []
    selected_ids: set[str] = set()

    def add_from(group: str, limit: int) -> None:
        if limit <= 0:
            return
        for task in groups.get(group, [])[:limit]:
            tid = task.get("task_id", "")
            if tid and tid not in selected_ids:
                selected.append(task)
                selected_ids.add(tid)
            if len([item for item in selected if supervised_task_group(item) == group]) >= limit:
                break

    for group in ["pubmed_deep_dive", "formal_version_search", "framework_foundation", "governance", "quarantine_rescue"]:
        add_from(group, quotas.get(group, 0))

    for task in tasks:
        if len(selected) >= max_tasks:
            break
        tid = task.get("task_id", "")
        if tid and tid not in selected_ids:
            selected.append(task)
            selected_ids.add(tid)
    return selected[:max_tasks]


def task_group_counts(tasks: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        group = supervised_task_group(task)
        counts[group] = counts.get(group, 0) + 1
    return counts


def supervision_coverage_warnings(tasks: list[dict[str, str]], max_tasks: int) -> list[str]:
    if not tasks or max_tasks <= 0:
        return []
    counts = task_group_counts(tasks)
    total = len(tasks)
    warnings: list[str] = []
    deep_share = counts.get("pubmed_deep_dive", 0) / total
    foundation_share = counts.get("framework_foundation", 0) / total
    if total >= min(max_tasks, 10) and deep_share < 0.30:
        warnings.append("PubMed deep-dive tasks are below 30%; the queue may still be dominated by formal-version checks.")
    if total >= min(max_tasks, 10) and foundation_share < 0.15:
        warnings.append("Framework/history/method foundation tasks are below 15%; historical lineage may be under-supervised.")
    if counts.get("formal_version_search", 0) > max(6, total * 0.45):
        warnings.append("Formal-version searches exceed 45%; route these mostly to Crossref/OpenAlex/publisher before consuming PubMed budget.")
    return warnings


def supervision_sort_key(row: dict[str, str]) -> tuple[int, int, int, str]:
    priority_order = {"high": 0, "medium": 1, "low": 2}
    lane_order = {
        "core_biomedical_system_evidence": 0,
        "historical_foundation": 1,
        "framework_verified": 2,
        "method_foundation": 3,
        "governance_background": 4,
        "background_hold": 9,
    }
    type_order = {
        "pubmed_deep_dive": 0,
        "formal_version_search": 1,
        "framework_lineage_pubmed": 2,
        "method_foundation_pubmed": 3,
        "governance_pubmed": 4,
        "quarantine_rescue_review": 5,
        "pool_triage_review": 6,
    }
    return (
        priority_order.get(clean_space(row.get("priority")), 9),
        lane_order.get(clean_space(row.get("evidence_lane")), 8),
        type_order.get(clean_space(row.get("supervision_type")), 9),
        clean_space(row.get("source_title")).lower(),
    )


def write_recall_supervision_packet(
    path: Path,
    seed_rows: list[dict[str, str]],
    counts: dict[str, int],
    topic: str,
    max_rows: int,
) -> None:
    lines = [
        "# Agent-Supervised Recall Packet",
        "",
        "You are supervising the literature recall loop, not merely screening already-ranked hits.",
        "",
        f"- Topic: {topic or '(not specified)'}",
        f"- All candidates: {counts.get('all', 0)}",
        f"- Screening queue: {counts.get('queue', 0)}",
        f"- Already screened: {counts.get('screened', 0)}",
        f"- Backlog in queue: {counts.get('backlog', 0)}",
        f"- Deferred but not queued: {counts.get('deferred', 0)}",
        f"- Quarantined: {counts.get('quarantine', 0)}",
        "",
        "Core responsibilities:",
        "",
        "1. Identify valuable candidates that mechanical ranking underweighted.",
        "2. Convert informal system names, preprint titles, or broad records into narrow PubMed queries.",
        "3. Promote framework/historical/method/governance papers into their own verification lane without mixing them into the core biomedical evidence table.",
        "4. Reject obvious drift, duplicates, supplements, comments, or non-paper records.",
        "5. Add a human question only when the pool cannot resolve identity or scope.",
        "",
        "Return CSV with columns:",
        ",".join(SUPERVISION_FIELDS),
        "",
        "Allowed decisions: add_pubmed_query, rescue_to_screening, verify_as_core, verify_as_framework, verify_as_historical, reject_noise, merge_duplicate, ask_user.",
        "",
        "Seed supervision rows follow. You may edit the proposed queries, change lanes, add missing PubMed queries, or reject rows.",
        "",
        "| ID | Type | Decision | Lane | Priority | Candidate | PubMed query | Rationale |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in seed_rows[:max_rows]:
        lines.append(
            "| {id} | {typ} | {dec} | {lane} | {prio} | {title} | `{query}` | {rat} |".format(
                id=row.get("supervision_id", ""),
                typ=row.get("supervision_type", ""),
                dec=row.get("decision", ""),
                lane=row.get("evidence_lane", ""),
                prio=row.get("priority", ""),
                title=clean_space(row.get("source_title")).replace("|", "/")[:110],
                query=clean_space(row.get("proposed_pubmed_query")).replace("|", "/")[:160],
                rat=clean_space(row.get("rationale")).replace("|", "/")[:140],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_supervise_pool(args: argparse.Namespace) -> int:
    collection_dir = Path(args.collection_dir)
    out_dir = Path(args.out_dir) if args.out_dir else collection_dir / "recall_supervision"
    all_rows = read_csv(collection_dir / "supplemental_recall_candidates.csv")
    queue_rows = read_csv(collection_dir / "screening_queue.csv")
    quarantine_rows = read_csv(collection_dir / "quarantined_candidates.csv")
    if not all_rows:
        raise SystemExit(f"No supplemental_recall_candidates.csv found in {collection_dir}")

    screened_ids: set[str] = set()
    if args.screened_dir:
        screened_dir = Path(args.screened_dir)
        for row in read_csv(screened_dir / "accepted_supplemental_recall_candidates.csv") + read_csv(screened_dir / "rejected_supplemental_recall_candidates.csv"):
            cid = clean_space(row.get("candidate_id"))
            if cid:
                screened_ids.add(cid)
    queue_ids = {clean_space(row.get("candidate_id")) for row in queue_rows if clean_space(row.get("candidate_id"))}
    quarantine_ids = {clean_space(row.get("candidate_id")) for row in quarantine_rows if clean_space(row.get("candidate_id"))}
    backlog = [row for row in queue_rows if clean_space(row.get("candidate_id")) not in screened_ids]
    deferred = [
        row
        for row in all_rows
        if clean_space(row.get("candidate_id")) not in queue_ids and clean_space(row.get("candidate_id")) not in quarantine_ids
    ]

    selected: list[tuple[str, dict[str, str]]] = []
    selected.extend(("backlog", row) for row in backlog[: args.backlog_n])
    selected.extend(("deferred", row) for row in deferred[: args.deferred_n])
    selected.extend(("quarantine", row) for row in quarantine_rows[: args.quarantine_n])

    seen: set[str] = set()
    supervision_rows: list[dict[str, str]] = []
    for bucket, row in selected:
        key = normalize_doi(row.get("doi")) or normalize_title(row.get("title"))
        if not key or key in seen:
            continue
        seen.add(key)
        text = candidate_intrinsic_text(row)
        if bucket != "quarantine" or AGENT_RECALL_SIGNAL.search(text) or BIOMED_RECALL_SIGNAL.search(text) or is_framework_lineage_candidate(row):
            supervision_rows.append(supervision_seed_row(row, bucket))
    supervision_rows.sort(key=supervision_sort_key)

    tasks: list[dict[str, str]] = []
    for row in supervision_rows:
        task = supervision_to_pubmed_task(row, args.topic)
        if task:
            tasks.append(task)
    add_strategic_pubmed_tasks(tasks, args.topic, args.strategic_pubmed_tasks)
    candidate_task_count = len(dedupe_tasks_preserve_order(tasks))
    quotas = supervised_task_quotas(args)
    tasks = select_supervised_pubmed_tasks(tasks, args.max_pubmed_tasks, quotas)
    group_counts = task_group_counts(tasks)
    coverage_warnings = supervision_coverage_warnings(tasks, args.max_pubmed_tasks)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "recall_supervision_seed_decisions.csv", supervision_rows, SUPERVISION_FIELDS)
    write_csv(out_dir / "recall_supervision_template.csv", [{field: "" for field in SUPERVISION_FIELDS} for _ in supervision_rows], SUPERVISION_FIELDS)
    write_csv(out_dir / "agent_supervised_pubmed_tasks.csv", tasks, TASK_FIELDS)
    write_recall_supervision_packet(
        out_dir / "recall_supervision_packet.md",
        supervision_rows,
        {
            "all": len(all_rows),
            "queue": len(queue_rows),
            "screened": len(screened_ids),
            "backlog": len(backlog),
            "deferred": len(deferred),
            "quarantine": len(quarantine_rows),
        },
        args.topic,
        args.max_packet_rows,
    )
    write_json(
        out_dir / "recall_supervision_manifest.json",
        {
            "generated_at": now_iso(),
            "collection_dir": str(collection_dir),
            "screened_dir": args.screened_dir,
            "seed_decisions": len(supervision_rows),
            "pubmed_tasks": len(tasks),
            "candidate_pubmed_tasks_before_quota": candidate_task_count,
            "max_pubmed_tasks": args.max_pubmed_tasks,
            "task_quotas": quotas,
            "task_group_counts": group_counts,
            "coverage_warnings": coverage_warnings,
            "backlog_sampled": min(args.backlog_n, len(backlog)),
            "deferred_sampled": min(args.deferred_n, len(deferred)),
            "quarantine_sampled": min(args.quarantine_n, len(quarantine_rows)),
        },
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "seed_decisions": len(supervision_rows),
                "pubmed_tasks": len(tasks),
                "task_group_counts": group_counts,
                "coverage_warnings": coverage_warnings,
            },
            ensure_ascii=False,
        )
    )
    return 0


def cmd_collect_supervision(args: argparse.Namespace) -> int:
    decisions: list[dict[str, str]] = []
    for path in args.supervision_csv:
        decisions.extend(read_csv(Path(path)))
    if not decisions:
        raise SystemExit("No supervision decisions found.")
    candidates = {row.get("candidate_id", ""): row for row in read_csv(Path(args.candidates_csv))} if args.candidates_csv else {}

    tasks: list[dict[str, str]] = []
    rescue_rows: list[dict[str, str]] = []
    for row in decisions:
        decision = clean_space(row.get("decision"))
        if decision in {"add_pubmed_query", "verify_as_core", "verify_as_framework", "verify_as_historical", "rescue_to_screening"}:
            task = supervision_to_pubmed_task(row, args.topic)
            if task:
                tasks.append(task)
        cid = clean_space(row.get("source_candidate_id"))
        if cid and cid in candidates and decision in {"rescue_to_screening", "verify_as_core", "verify_as_framework", "verify_as_historical"}:
            rescued = dict(candidates[cid])
            rescued["pool_status"] = "agent_supervised_rescue"
            rescued["candidate_fit_reason"] = clean_space(row.get("rationale")) or rescued.get("candidate_fit_reason", "")
            rescued["human_decision_needed"] = clean_space(row.get("human_question"))
            rescue_rows.append(rescued)
    add_strategic_pubmed_tasks(tasks, args.topic, args.strategic_pubmed_tasks)
    candidate_task_count = len(dedupe_tasks_preserve_order(tasks))
    quotas = supervised_task_quotas(args)
    tasks = select_supervised_pubmed_tasks(tasks, args.max_pubmed_tasks, quotas)
    group_counts = task_group_counts(tasks)
    coverage_warnings = supervision_coverage_warnings(tasks, args.max_pubmed_tasks)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "agent_supervised_pubmed_tasks.csv", tasks, TASK_FIELDS)
    write_csv(out_dir / "agent_supervised_rescue_candidates.csv", rescue_rows, CANDIDATE_FIELDS)
    lines = [
        "# Recall Supervision Collection",
        "",
        f"- Decisions: {len(decisions)}",
        f"- PubMed tasks: {len(tasks)}",
        f"- Candidate PubMed tasks before quota: {candidate_task_count}",
        f"- Max PubMed tasks: {args.max_pubmed_tasks}",
        f"- Task group counts: {json.dumps(group_counts, ensure_ascii=False)}",
        f"- Task quotas: {json.dumps(quotas, ensure_ascii=False)}",
        f"- Rescue candidates: {len(rescue_rows)}",
        "",
        "The task list is selected by lane/type quota so PubMed deep dives, formal-version checks, framework/history papers, governance, and rescue candidates remain visible to Codex/subagents. A zero-hit PubMed query is a routing signal, not a final exclusion.",
        "",
        "## Coverage Warnings",
        "",
        *(f"- {warning}" for warning in coverage_warnings),
        "" if coverage_warnings else "- None.",
        "",
        "Run the PubMed tasks, merge returned candidates, and re-run `collect-candidates` or targeted verification with lane labels preserved.",
    ]
    (out_dir / "recall_supervision_collection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "pubmed_tasks": len(tasks),
                "rescue_candidates": len(rescue_rows),
                "task_group_counts": group_counts,
                "coverage_warnings": coverage_warnings,
            },
            ensure_ascii=False,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan broad supplemental recall and screen candidates with agents before verification.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="Create broad-but-bounded supplemental recall tasks and query files.")
    p_plan.add_argument("--topic", default="")
    p_plan.add_argument("--draft-assets-dir", default="./review-data/02_literature/draft_assets")
    p_plan.add_argument("--literature-discovery-dir", default="./review-data/02_literature/literature_discovery/collected")
    p_plan.add_argument("--out-dir", default="./review-data/02_literature/supplemental_recall_screening")
    p_plan.add_argument("--max-systems", type=int, default=80)
    p_plan.add_argument("--max-claims", type=int, default=80)
    p_plan.add_argument("--max-tasks", type=int, default=160)
    p_plan.add_argument("--max-results-per-task", type=int, default=20)
    p_plan.set_defaults(func=cmd_plan)

    p_pubmed = sub.add_parser("pubmed", help="Run PubMed recall for supplemental recall tasks.")
    p_pubmed.add_argument("--tasks-csv", default="./review-data/02_literature/supplemental_recall_screening/supplemental_recall_tasks.csv")
    p_pubmed.add_argument("--out-dir", default="./review-data/02_literature/supplemental_recall_screening/pubmed")
    p_pubmed.add_argument("--max-results", type=int, default=20)
    p_pubmed.add_argument("--max-tasks", type=int, default=0)
    p_pubmed.add_argument("--workers", type=int, default=2)
    p_pubmed.add_argument("--resume", action="store_true")
    p_pubmed.add_argument("--mindate", default="")
    p_pubmed.add_argument("--maxdate", default="")
    p_pubmed.add_argument("--sort", default="relevance", choices=["relevance", "pub date", "first author", "journal"])
    p_pubmed.add_argument("--email", default="")
    p_pubmed.add_argument("--api-key", default="")
    p_pubmed.add_argument("--tool", default="top-journal-review-writer")
    p_pubmed.add_argument("--timeout", type=int, default=40)
    p_pubmed.add_argument("--sleep", type=float, default=0.34)
    p_pubmed.add_argument("--retries", type=int, default=2)
    p_pubmed.add_argument("--user-agent", default="top-journal-review-writer/supplemental-recall-screening")
    p_pubmed.add_argument("--dry-run", action="store_true")
    p_pubmed.add_argument("--max-abstract-chars", type=int, default=1200)
    p_pubmed.set_defaults(func=cmd_pubmed)

    p_collect = sub.add_parser("collect-candidates", help="Merge candidate CSVs from PubMed/paper-search/Crossref/OpenAlex/arXiv recall and write screening packets.")
    p_collect.add_argument("--candidate-csv", action="append", default=[])
    p_collect.add_argument("--seed-csv", action="append", default=[])
    p_collect.add_argument("--recall-dir", action="append", default=[])
    p_collect.add_argument("--out-dir", default="./review-data/02_literature/supplemental_recall_screening/collected")
    p_collect.add_argument("--topic", default="")
    p_collect.add_argument("--max-abstract-chars", type=int, default=1200)
    p_collect.add_argument("--packet-size", type=int, default=50)
    p_collect.add_argument("--min-screening-score", type=int, default=-999)
    p_collect.add_argument("--max-screening-candidates", type=int, default=0)
    p_collect.add_argument("--include-quarantine", action="store_true")
    p_collect.set_defaults(func=cmd_collect_candidates)

    p_screen = sub.add_parser("collect-screening", help="Collect agent/human screening decisions and emit verifier-ready supplemental candidates.")
    p_screen.add_argument("--candidates-csv", default="./review-data/02_literature/supplemental_recall_screening/collected/supplemental_recall_candidates.csv")
    p_screen.add_argument("--screening-csv", action="append", required=True)
    p_screen.add_argument("--out-dir", default="./review-data/02_literature/supplemental_recall_screening/screened")
    p_screen.set_defaults(func=cmd_collect_screening)

    p_audit = sub.add_parser("audit-pool", help="Explain broad recall pool formation, screened coverage, backlog, and quarantine rescue candidates.")
    p_audit.add_argument("--collection-dir", default="./review-data/02_literature/supplemental_recall_screening/collected")
    p_audit.add_argument("--screened-dir", default="")
    p_audit.add_argument("--verification-dir", default="")
    p_audit.add_argument("--out-dir", default="")
    p_audit.add_argument("--top-n", type=int, default=40)
    p_audit.set_defaults(func=cmd_audit_pool)

    p_supervise = sub.add_parser("supervise-pool", help="Create Codex/subagent recall-supervision packets and PubMed deep-dive tasks from the current recall pool.")
    p_supervise.add_argument("--collection-dir", default="./review-data/02_literature/supplemental_recall_screening/collected")
    p_supervise.add_argument("--screened-dir", default="")
    p_supervise.add_argument("--out-dir", default="")
    p_supervise.add_argument("--topic", default="")
    p_supervise.add_argument("--backlog-n", type=int, default=80)
    p_supervise.add_argument("--deferred-n", type=int, default=80)
    p_supervise.add_argument("--quarantine-n", type=int, default=40)
    p_supervise.add_argument("--max-packet-rows", type=int, default=120)
    p_supervise.add_argument("--max-pubmed-tasks", type=int, default=60)
    p_supervise.add_argument("--deep-dive-task-quota", type=int, default=0)
    p_supervise.add_argument("--formal-version-task-quota", type=int, default=0)
    p_supervise.add_argument("--framework-task-quota", type=int, default=0)
    p_supervise.add_argument("--governance-task-quota", type=int, default=0)
    p_supervise.add_argument("--rescue-task-quota", type=int, default=0)
    p_supervise.add_argument("--strategic-pubmed-tasks", type=int, default=12)
    p_supervise.set_defaults(func=cmd_supervise_pool)

    p_collect_supervision = sub.add_parser("collect-supervision", help="Collect Codex/subagent recall-supervision decisions into PubMed tasks and rescue candidates.")
    p_collect_supervision.add_argument("--supervision-csv", action="append", required=True)
    p_collect_supervision.add_argument("--candidates-csv", default="")
    p_collect_supervision.add_argument("--out-dir", default="./review-data/02_literature/supplemental_recall_screening/recall_supervision/collected")
    p_collect_supervision.add_argument("--topic", default="")
    p_collect_supervision.add_argument("--max-pubmed-tasks", type=int, default=80)
    p_collect_supervision.add_argument("--deep-dive-task-quota", type=int, default=0)
    p_collect_supervision.add_argument("--formal-version-task-quota", type=int, default=0)
    p_collect_supervision.add_argument("--framework-task-quota", type=int, default=0)
    p_collect_supervision.add_argument("--governance-task-quota", type=int, default=0)
    p_collect_supervision.add_argument("--rescue-task-quota", type=int, default=0)
    p_collect_supervision.add_argument("--strategic-pubmed-tasks", type=int, default=12)
    p_collect_supervision.set_defaults(func=cmd_collect_supervision)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
