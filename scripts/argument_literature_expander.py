#!/usr/bin/env python3
"""Argument-driven supplemental literature expansion.

The script reads a draft-derived framework and turns specific claims into
bounded retrieval tasks. It then fetches PubMed metadata/abstract candidates
for those tasks and writes screening packets. New candidates are meant for
`supplemental_pool` first; they are not main-pool citations until screened.
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
    "origin",
    "draft",
    "claim_id",
    "claim_role",
    "argument_time_role",
    "evidence_need",
    "priority",
    "claim",
    "query",
    "allowed_destination",
    "promotion_gate",
    "screening_question",
]

CANDIDATE_FIELDS = [
    "task_id",
    "claim_id",
    "linked_claim",
    "claim_role",
    "argument_time_role",
    "evidence_need",
    "query",
    "key",
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
    "recall_queries",
    "pool_status",
    "candidate_fit_score",
    "candidate_fit_status",
    "candidate_fit_reason",
    "retrieval_lane",
    "fulltext_need",
    "human_decision_needed",
    "notes",
]

AI_TERM_MAP = [
    (r"self[-\s]?evol|self[-\s]?improv|self[-\s]?reflect", '"self-evolving" OR "self-improving" OR "self-reflection"'),
    (r"\bagents?\b|autonomous|language agent", '"autonomous agent" OR "AI agent" OR "language agent"'),
    (r"large language model|LLM|foundation model", '"large language model" OR LLM'),
    (r"biomed|medical|clinical|medicine|health", "biomedical OR medical OR clinical OR health"),
    (r"tool|tool[-\s]?use|function call", '"tool use" OR "tool-using"'),
    (r"retrieval|RAG", '"retrieval augmented generation" OR RAG'),
    (r"knowledge graph", '"knowledge graph"'),
    (r"multi[-\s]?modal", "multimodal"),
    (r"evaluat|benchmark", "evaluation OR benchmark"),
    (r"governance|regulat|safety", "governance OR safety OR regulation"),
    (r"memory", "memory"),
    (r"planning|plan", "planning"),
    (r"reinforcement|RL", '"reinforcement learning"'),
    (r"clinical trial|prospective", '"clinical trial" OR prospective'),
]

STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "that",
    "this",
    "these",
    "those",
    "review",
    "paper",
    "article",
    "study",
    "studies",
    "research",
    "based",
    "using",
    "toward",
    "towards",
    "model",
    "models",
    "large",
    "language",
}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(item) for item in value if item is not None)
    return str(value).strip()


def clean_space(value: Any) -> str:
    return re.sub(r"\s+", " ", normalize_text(value)).strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False)
    if lock:
        with lock:
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
    else:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")


def stable_id(value: str, prefix: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{digest}"


def markdown_escape(text: Any) -> str:
    return clean_space(text).replace("|", "\\|").replace("\n", " ")


def truncate(text: Any, max_chars: int) -> str:
    value = clean_space(text)
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 18].rstrip() + " [truncated]"


def clip(text: Any, max_chars: int) -> str:
    value = clean_space(text)
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip()


def compact_claim(text: Any, max_chars: int = 700) -> str:
    value = re.sub(r"[*_`#]+", " ", clean_space(text))
    value = re.sub(r"\s+-{3,}\s+", " ", value)
    abstract_marker = "\u6458\u8981"
    if abstract_marker in value and len(value) > max_chars:
        value = value.split(abstract_marker, 1)[-1]
    breaks = ".!?;" + "".join(chr(code) for code in [0x3002, 0xFF01, 0xFF1F, 0xFF1B])
    parts = [clean_space(part) for part in re.split("[" + re.escape(breaks) + "]", value) if clean_space(part)]
    candidates = [part for part in parts if 30 <= len(part) <= max_chars]
    if candidates:
        return candidates[0]
    return clip(value, max_chars)


def noisy_claim(text: Any) -> bool:
    value = clean_space(text)
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
    if re.search(r"\b(accepted|rejected|demoted)\s+\d+\s+(records|references|papers)\b", lower):
        return True
    if len(value) > 650:
        return True
    if len(re.findall(r"https?://|10\.\d{4,9}/", value)) >= 3:
        return True
    return False


def strip_citations(text: str) -> str:
    text = re.sub(r"\[[0-9,\-\s;]+\]", " ", text)
    text = re.sub(r"\((?:[A-Z][A-Za-z\-]+(?:\s+et al\.)?,?\s*)?[12][0-9]{3}[a-z]?\)", " ", text)
    text = re.sub(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", " ", text)
    return clean_space(text)


def evidence_need(row: dict[str, str]) -> str:
    time_role = normalize_text(row.get("argument_time_role"))
    claim_role = normalize_text(row.get("claim_role"))
    if time_role == "background_or_history":
        return "historical_foundation"
    if time_role == "future_agenda":
        return "future_agenda_support"
    if claim_role == "limitation_or_gap":
        return "counterevidence_or_boundary"
    if claim_role == "thesis_or_framework":
        return "conceptual_framework_support"
    if claim_role == "evidence_claim" or time_role == "current_evidence":
        return "current_evidence_support"
    return "interpretive_bridge_support"


def priority_for(row: dict[str, str]) -> str:
    status = normalize_text(row.get("evidence_status")).lower()
    role = normalize_text(row.get("claim_role"))
    time_role = normalize_text(row.get("argument_time_role"))
    if "needs literature recall" in status:
        return "P1"
    if role in {"thesis_or_framework", "limitation_or_gap"}:
        return "P1"
    if time_role in {"background_or_history", "future_agenda"}:
        return "P2"
    return "P3"


def mapped_terms(text: str, topic: str) -> list[str]:
    haystack = f"{text} {topic}"
    terms: list[str] = []
    for pattern, term in AI_TERM_MAP:
        if re.search(pattern, haystack, re.I):
            terms.append(term)
    if not terms:
        terms.extend(['"large language model" OR LLM', "biomedical OR medical"])
    seen: set[str] = set()
    out: list[str] = []
    for term in terms:
        key = term.lower()
        if key not in seen:
            seen.add(key)
            out.append(term)
    return out[:5]


def english_keywords(text: str, limit: int = 6) -> list[str]:
    tokens: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text):
        key = token.lower().strip("-")
        if key in STOP_WORDS or len(key) <= 2:
            continue
        if key not in tokens:
            tokens.append(key)
        if len(tokens) >= limit:
            break
    return tokens


def build_query(row: dict[str, str], topic: str, max_query_chars: int) -> str:
    claim = strip_citations(row.get("claim") or row.get("query") or "")
    need = evidence_need(row)
    terms = mapped_terms(claim, topic)
    extra = english_keywords(claim, limit=4)
    if need == "historical_foundation":
        extra.extend(["foundation", "history"])
    elif need == "counterevidence_or_boundary":
        extra.extend(["limitation", "evaluation", "safety"])
    elif need == "future_agenda_support":
        extra.extend(["prospective", "evaluation", "governance"])
    elif need == "conceptual_framework_support":
        extra.extend(["framework", "architecture"])
    clauses = [f"({term})" if " OR " in term else term for term in terms]
    for term in extra:
        if term.lower() not in " ".join(clauses).lower():
            clauses.append(term)
        if len(" AND ".join(clauses)) >= max_query_chars:
            break
    query = " AND ".join(clauses[:7])
    if len(query) > max_query_chars:
        shortened = query[:max_query_chars].rsplit(" AND ", 1)[0]
        query = shortened or query[:max_query_chars]
    return query.strip()


def screening_question(need: str) -> str:
    questions = {
        "historical_foundation": "Does this paper establish the past or foundation needed for the draft argument?",
        "future_agenda_support": "Does this paper justify or constrain the proposed future agenda?",
        "counterevidence_or_boundary": "Does this paper document a limitation, failure mode, safety issue, or boundary condition?",
        "conceptual_framework_support": "Does this paper support the framework rather than merely share keywords?",
        "current_evidence_support": "Does this paper provide current empirical or method evidence for the claim?",
        "interpretive_bridge_support": "Does this paper help connect two sections of the argument?",
    }
    return questions.get(need, "Does this paper directly support or challenge the linked claim?")


def task_from_claim(row: dict[str, str], topic: str, max_query_chars: int) -> dict[str, str]:
    claim = compact_claim(row.get("claim"))
    key = "|".join([row.get("draft", ""), row.get("claim_id", ""), claim])
    need = evidence_need(row)
    return {
        "task_id": stable_id(key, "ALT"),
        "origin": "argument_evidence_map",
        "draft": normalize_text(row.get("draft")),
        "claim_id": normalize_text(row.get("claim_id")),
        "claim_role": normalize_text(row.get("claim_role")),
        "argument_time_role": normalize_text(row.get("argument_time_role")),
        "evidence_need": need,
        "priority": priority_for(row),
        "claim": claim,
        "query": build_query(row, topic, max_query_chars),
        "allowed_destination": "supplemental_pool",
        "promotion_gate": "abstract_screening_then_metadata_gate_then_llm_or_human_claim_fit",
        "screening_question": screening_question(need),
    }


def task_from_literature_task(row: dict[str, str], topic: str, max_query_chars: int) -> dict[str, str]:
    claim = compact_claim(row.get("query"))
    synthetic = {
        "claim": claim,
        "claim_role": normalize_text(row.get("claim_role")) or "interpretive_claim",
        "argument_time_role": normalize_text(row.get("argument_time_role")) or "interpretive_bridge",
        "evidence_status": "needs literature recall",
    }
    key = "|".join([row.get("origin", ""), row.get("task_type", ""), claim])
    need = evidence_need(synthetic)
    return {
        "task_id": stable_id(key, "ALT"),
        "origin": normalize_text(row.get("task_type")) or "literature_search_tasks",
        "draft": normalize_text(row.get("origin")),
        "claim_id": "",
        "claim_role": synthetic["claim_role"],
        "argument_time_role": synthetic["argument_time_role"],
        "evidence_need": need,
        "priority": "P1" if row.get("task_type") == "support_claim_or_demote" else "P2",
        "claim": claim,
        "query": build_query(synthetic, topic, max_query_chars),
        "allowed_destination": "supplemental_pool",
        "promotion_gate": "abstract_screening_then_metadata_gate_then_llm_or_human_claim_fit",
        "screening_question": screening_question(need),
    }


def build_tasks(args: argparse.Namespace) -> list[dict[str, str]]:
    framework_dir = Path(args.framework_dir)
    claim_rows = read_csv(framework_dir / "argument_evidence_map.csv")
    task_rows = read_csv(framework_dir / "literature_search_tasks.csv")
    tasks: list[dict[str, str]] = []
    seen: set[str] = set()

    for row in claim_rows:
        claim = clean_space(row.get("claim"))
        if not claim or noisy_claim(claim):
            continue
        status = normalize_text(row.get("evidence_status")).lower()
        role = normalize_text(row.get("claim_role"))
        time_role = normalize_text(row.get("argument_time_role"))
        needed = (
            "needs literature recall" in status
            or role in {"thesis_or_framework", "limitation_or_gap"}
            or time_role in {"background_or_history", "future_agenda"}
        )
        if not needed and not args.include_cited_claims:
            continue
        task = task_from_claim(row, args.topic, args.max_query_chars)
        key = clean_space(task["query"]).lower()
        if key and key not in seen:
            seen.add(key)
            tasks.append(task)
        if len(tasks) >= args.max_tasks:
            return tasks

    for row in task_rows:
        if len(tasks) >= args.max_tasks:
            break
        if noisy_claim(row.get("query") or row.get("task") or row.get("claim")):
            continue
        task = task_from_literature_task(row, args.topic, args.max_query_chars)
        key = clean_space(task["query"]).lower()
        if key and key not in seen:
            seen.add(key)
            tasks.append(task)
    return tasks


def write_plan_markdown(path: Path, args: argparse.Namespace, tasks: list[dict[str, str]]) -> None:
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task["evidence_need"]] = counts.get(task["evidence_need"], 0) + 1
    lines = [
        "# Argument-Driven Literature Expansion Plan",
        "",
        "This plan is derived from the draft logic framework. It is not broad topic recall.",
        "",
        f"- Topic: {args.topic or '[not specified]'}",
        f"- Framework directory: `{args.framework_dir}`",
        f"- Tasks: {len(tasks)}",
        "- Allowed destination before screening: `supplemental_pool`",
        "",
        "## Evidence Need Counts",
        "",
    ]
    for need, count in sorted(counts.items()):
        lines.append(f"- {need}: {count}")
    lines.extend(["", "## Tasks", ""])
    for task in tasks:
        lines.extend(
            [
                f"### {task['task_id']} ({task['priority']})",
                "",
                f"- Evidence need: `{task['evidence_need']}`",
                f"- Claim role: `{task['claim_role']}` / `{task['argument_time_role']}`",
                f"- Draft/origin: `{task['draft'] or task['origin']}`",
                f"- Claim: {truncate(task['claim'], 420)}",
                f"- PubMed query: `{task['query']}`",
                f"- Screening question: {task['screening_question']}",
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_human_checkpoint(path: Path, tasks: list[dict[str, str]]) -> None:
    lines = [
        "# Human Checkpoint: Argument-Driven Supplemental Literature",
        "",
        "Confirm which tasks should run before Codex subtasks or human reviewers screen new papers. External LLM screening is optional only after approval.",
        "",
        "For each task, decide: keep, merge, delete, narrow query, or mark as user-supplied seed.",
        "",
        "| Task | Priority | Evidence need | Claim | Query | Decision |",
        "|---|---|---|---|---|---|",
    ]
    for task in tasks:
        cells = [
            f"`{task['task_id']}`",
            markdown_escape(task["priority"]),
            markdown_escape(task["evidence_need"]),
            markdown_escape(truncate(task["claim"], 180)),
            f"`{markdown_escape(task['query'])}`",
            "pending",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plan_outputs(args: argparse.Namespace, tasks: list[dict[str, str]]) -> dict[str, str]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "argument_literature_tasks.csv", tasks, TASK_FIELDS)
    write_json(out_dir / "argument_literature_tasks.json", tasks)
    (out_dir / "pubmed_queries.txt").write_text(
        "\n".join(task["query"] for task in tasks if task.get("query")) + ("\n" if tasks else ""),
        encoding="utf-8",
    )
    write_plan_markdown(out_dir / "argument_literature_plan.md", args, tasks)
    write_human_checkpoint(out_dir / "human_argument_literature_checkpoint.md", tasks)
    return {
        "tasks_csv": str(out_dir / "argument_literature_tasks.csv"),
        "queries": str(out_dir / "pubmed_queries.txt"),
        "checkpoint": str(out_dir / "human_argument_literature_checkpoint.md"),
    }


def normalize_for_match(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", normalize_text(text).lower())
    return re.sub(r"\s+", " ", text)


def overlap_score(query: str, claim: str, title: str, abstract: str) -> tuple[int, list[str]]:
    target = normalize_for_match(f"{title} {abstract}")
    source_terms = set(english_keywords(f"{query} {claim}", limit=20))
    for _pattern, term in AI_TERM_MAP:
        source_terms.update(english_keywords(term, limit=10))
    hits = sorted(term for term in source_terms if term and term in target)
    return min(len(hits), 10), hits[:12]


def candidate_status(score: int, abstract: str) -> tuple[str, str, str]:
    if not abstract:
        return "maybe", "needs_llm_or_human_screening", "No abstract returned; keep only as metadata lead."
    if score >= 4:
        return "candidate", "abstract_fit_needs_confirmation", "Abstract/title has several overlaps with the linked claim."
    if score >= 2:
        return "maybe", "weak_abstract_fit_needs_confirmation", "Some overlap, but claim fit is uncertain."
    return "background", "off_map_or_weak_match", "Low overlap; keep out of citation candidates unless a reviewer promotes it."


def fallback_queries(task: dict[str, str]) -> list[str]:
    primary = clean_space(task.get("query"))
    clauses = [clean_space(part) for part in primary.split(" AND ") if clean_space(part)]
    out: list[str] = []
    if len(clauses) > 3:
        out.append(" AND ".join(clauses[:3]))
    if len(clauses) > 2:
        out.append(" AND ".join(clauses[:2]))
    need = task.get("evidence_need", "")
    if need in {"counterevidence_or_boundary", "current_evidence_support"}:
        out.append('("large language model" OR LLM) AND (biomedical OR medical OR clinical) AND (evaluation OR benchmark OR safety)')
    elif need == "historical_foundation":
        out.append('("AI agent" OR "autonomous agent") AND biomedical AND history')
    else:
        out.append('("large language model" OR LLM) AND (biomedical OR medical OR clinical) AND agent')
    seen: set[str] = {primary.lower()}
    unique: list[str] = []
    for query in out:
        key = query.lower()
        if query and key not in seen:
            seen.add(key)
            unique.append(query)
    return unique[:3]


def fetch_task(task: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    search_calls: list[dict[str, Any]] = []
    papers: list[dict[str, str]] = []
    seen_pmids: set[str] = set()
    for query in [task["query"], *fallback_queries(task)]:
        ids, search_url = pubmed_recall.esearch(query, args)
        search_calls.append({"query": query, "search_url": search_url, "pmids": ids})
        time.sleep(args.sleep)
        fetched, fetch_url = pubmed_recall.efetch(ids, args)
        search_calls[-1]["fetch_url"] = fetch_url
        search_calls[-1]["fetched"] = len(fetched)
        for paper in fetched:
            pmid = paper.get("pmid") or paper.get("paper_id") or paper.get("key", "")
            if pmid and pmid in seen_pmids:
                continue
            seen_pmids.add(pmid)
            paper["_recall_query_used"] = query
            papers.append(paper)
        if papers or args.dry_run:
            break
    rows: list[dict[str, str]] = []
    for paper in papers:
        score, hits = overlap_score(task["query"], task["claim"], paper.get("title", ""), paper.get("abstract", ""))
        pool_status, fit_status, reason = candidate_status(score, paper.get("abstract", ""))
        fit_reason = reason + (f" Matched terms: {', '.join(hits)}." if hits else "")
        rows.append(
            {
                "task_id": task["task_id"],
                "claim_id": task.get("claim_id", ""),
                "linked_claim": task.get("claim", ""),
                "claim_role": task.get("claim_role", ""),
                "argument_time_role": task.get("argument_time_role", ""),
                "evidence_need": task.get("evidence_need", ""),
                "query": task.get("query", ""),
                "key": paper.get("key", ""),
                "title": paper.get("title", ""),
                "authors": paper.get("authors", ""),
                "year": paper.get("year", ""),
                "journal": paper.get("journal", ""),
                "source": "pubmed",
                "paper_id": paper.get("paper_id", ""),
                "pmid": paper.get("pmid", ""),
                "doi": paper.get("doi", ""),
                "url": paper.get("url", ""),
                "pdf_url": paper.get("pdf_url", ""),
                "abstract": paper.get("abstract", ""),
                "publication_type": paper.get("publication_type", ""),
                "recall_queries": paper.get("_recall_query_used", task.get("query", "")),
                "pool_status": pool_status,
                "candidate_fit_score": str(score),
                "candidate_fit_status": fit_status,
                "candidate_fit_reason": fit_reason,
                "retrieval_lane": "argument_driven_supplemental_pubmed",
                "fulltext_need": "fetch_fulltext_if_promoted",
                "human_decision_needed": "accept_as_support | accept_as_counterevidence | background_only | reject",
                "notes": f"Linked to {task['task_id']}; not promoted until claim-fit screening. Primary query: {task.get('query', '')}",
            }
        )
    return {
        "task_id": task["task_id"],
        "query": task["query"],
        "pmids": sorted(seen_pmids),
        "calls": search_calls,
        "rows": rows,
        "status": "ok",
    }


def load_tasks(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    if not rows:
        raise SystemExit(f"No tasks found: {path}")
    return rows


def write_abstract_packet(path: Path, rows: list[dict[str, str]], max_abstract_chars: int) -> None:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["task_id"], []).append(row)
    lines = [
        "# Abstract Screening Packet",
        "",
        "Screen these candidates against linked draft-derived claims before promotion. Abstracts support triage; detailed method/result claims require full text.",
        "",
    ]
    for task_id, task_rows in grouped.items():
        first = task_rows[0]
        lines.extend(
            [
                f"## {task_id}",
                "",
                f"- Evidence need: `{first.get('evidence_need')}`",
                f"- Linked claim: {truncate(first.get('linked_claim', ''), 420)}",
                f"- Query: `{first.get('query')}`",
                "",
            ]
        )
        for index, row in enumerate(task_rows[:8], 1):
            lines.extend(
                [
                    f"### Candidate {index}: {row.get('title') or row.get('key')}",
                    "",
                    f"- Key: `{row.get('key')}`",
                    f"- PMID/DOI: {row.get('pmid') or 'missing'} / {row.get('doi') or 'missing'}",
                    f"- Journal/year: {row.get('journal') or 'unknown'} ({row.get('year') or 'unknown'})",
                    f"- Fit: `{row.get('candidate_fit_status')}` score {row.get('candidate_fit_score')}",
                    "- Triage decision: accept_as_support / accept_as_counterevidence / background_only / reject",
                    "",
                    truncate(row.get("abstract", "") or "[No abstract returned]", max_abstract_chars),
                    "",
                ]
            )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_import_and_fulltext_next_steps(path: Path, args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parent.as_posix()
    out_dir = Path(args.out_dir).as_posix()
    lines = [
        "# Supplemental Import And Full-Text Next Steps",
        "",
        "Do not import these candidates into the main literature pool directly.",
        "",
        "1. Screen `abstract_screening_packet.md` with Codex/subagents/DeepSeek and the user if needed.",
        "2. Import screened candidates into the supplemental pool, not the main pool:",
        "",
        "```powershell",
        f"python \"{script_dir}/literature_pool.py\" import-csv --pool-dir ./review-data/02_literature/supplemental_pool --csv {out_dir}/pubmed_argument_candidates.csv --origin argument-driven-supplemental-pubmed",
        "```",
        "",
        "3. Promote only accepted rows to the main pool after claim-fit screening, topic filtering, and final-gate checks.",
        "4. For promoted/selected rows, fetch full text and create the user handoff:",
        "",
        "```powershell",
        f"python \"{script_dir}/fulltext_manager.py\" fetch-europepmc --pool-dir ./review-data/02_literature/pool --status citation_pool,seminal,method,recent",
        f"python \"{script_dir}/fulltext_manager.py\" audit --pool-dir ./review-data/02_literature/pool --status citation_pool,seminal,method,recent",
        "```",
        "",
        "Rows with only abstracts may inform screening and framing, but not detailed result, metric, or clinical-effect claims.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_candidate_outputs(out_dir: Path, rows: list[dict[str, str]], run_log: list[dict[str, Any]], args: argparse.Namespace) -> None:
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            r.get("pool_status") == "candidate",
            int(r.get("candidate_fit_score") or 0),
            r.get("year", ""),
        ),
        reverse=True,
    )
    write_csv(out_dir / "pubmed_argument_candidates.csv", rows_sorted, CANDIDATE_FIELDS)
    with (out_dir / "pubmed_argument_candidates.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows_sorted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(
        out_dir / "claim_candidate_links.csv",
        rows_sorted,
        [
            "task_id",
            "claim_id",
            "linked_claim",
            "evidence_need",
            "key",
            "title",
            "pmid",
            "doi",
            "candidate_fit_score",
            "candidate_fit_status",
            "pool_status",
        ],
    )
    write_json(
        out_dir / "pubmed_run_log.json",
        {
            "schema_version": 1,
            "date_run": now_iso(),
            "tasks_path": args.tasks_csv,
            "tasks": len(run_log),
            "candidates": len(rows_sorted),
            "max_results": args.max_results,
            "workers": args.workers,
            "calls": run_log,
        },
    )
    write_abstract_packet(out_dir / "abstract_screening_packet.md", rows_sorted, args.max_abstract_chars)
    write_import_and_fulltext_next_steps(out_dir / "supplemental_import_and_fulltext_next_steps.md", args)


def cmd_plan(args: argparse.Namespace) -> int:
    tasks = build_tasks(args)
    outputs = write_plan_outputs(args, tasks)
    print(json.dumps({"tasks": len(tasks), **outputs}, ensure_ascii=False))
    return 0


def cmd_pubmed(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(Path(args.tasks_csv))
    if args.limit:
        tasks = tasks[: args.limit]

    progress_path = out_dir / "pubmed_task_progress.jsonl"
    candidate_progress = out_dir / "pubmed_candidate_progress.jsonl"
    if not args.resume:
        for path in [progress_path, candidate_progress]:
            if path.exists():
                path.unlink()

    completed: set[str] = set()
    if args.resume and progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("status") == "ok":
                completed.add(normalize_text(item.get("task_id")))

    pending = [task for task in tasks if task.get("task_id") not in completed]
    existing_csv = out_dir / "pubmed_argument_candidates.csv"
    all_rows: list[dict[str, str]] = read_csv(existing_csv) if args.resume and existing_csv.exists() else []
    run_log: list[dict[str, Any]] = []
    run_log_path = out_dir / "pubmed_run_log.json"
    if args.resume and run_log_path.exists():
        try:
            previous = json.loads(run_log_path.read_text(encoding="utf-8"))
            if isinstance(previous.get("calls"), list):
                run_log.extend(previous["calls"])
        except json.JSONDecodeError:
            pass

    lock = threading.Lock()

    def record_result(result: dict[str, Any]) -> None:
        safe_log = {key: value for key, value in result.items() if key != "rows"}
        run_log.append(safe_log)
        all_rows.extend(result.get("rows", []))
        append_jsonl(progress_path, safe_log, lock=lock)
        append_jsonl(candidate_progress, {"task_id": result.get("task_id"), "rows": result.get("rows", [])}, lock=lock)
        if len(run_log) % max(1, args.flush_every) == 0:
            write_candidate_outputs(out_dir, all_rows, run_log, args)
        print(json.dumps({"task_id": result.get("task_id"), "status": result.get("status"), "candidates": len(result.get("rows", []))}, ensure_ascii=False))

    if args.workers <= 1:
        for task in pending:
            try:
                record_result(fetch_task(task, args))
            except Exception as exc:
                result = {"task_id": task.get("task_id"), "query": task.get("query"), "status": "failed", "error": repr(exc)}
                run_log.append(result)
                append_jsonl(progress_path, result, lock=lock)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(fetch_task, task, args): task for task in pending}
            for future in concurrent.futures.as_completed(future_map):
                task = future_map[future]
                try:
                    record_result(future.result())
                except Exception as exc:
                    result = {"task_id": task.get("task_id"), "query": task.get("query"), "status": "failed", "error": repr(exc)}
                    run_log.append(result)
                    append_jsonl(progress_path, result, lock=lock)
                    print(json.dumps(result, ensure_ascii=False))

    write_candidate_outputs(out_dir, all_rows, run_log, args)
    print(json.dumps({"out_dir": str(out_dir), "tasks_completed": len(run_log), "candidates": len(all_rows)}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan and run argument-driven supplemental literature expansion.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="Build retrieval tasks from the draft-derived argument map.")
    p_plan.add_argument("--framework-dir", default="./review-data/03_framework/logic_framework")
    p_plan.add_argument("--topic", default="")
    p_plan.add_argument("--out-dir", default="./review-data/02_literature/argument_literature_expansion")
    p_plan.add_argument("--max-tasks", type=int, default=40)
    p_plan.add_argument("--max-query-chars", type=int, default=220)
    p_plan.add_argument("--include-cited-claims", action="store_true")
    p_plan.set_defaults(func=cmd_plan)

    p_pubmed = sub.add_parser("pubmed", help="Fetch PubMed metadata/abstract candidates for argument tasks.")
    p_pubmed.add_argument("--tasks-csv", default="./review-data/02_literature/argument_literature_expansion/argument_literature_tasks.csv")
    p_pubmed.add_argument("--out-dir", default="./review-data/02_literature/argument_literature_expansion")
    p_pubmed.add_argument("--max-results", type=int, default=8)
    p_pubmed.add_argument("--limit", type=int, default=0)
    p_pubmed.add_argument("--workers", type=int, default=2)
    p_pubmed.add_argument("--flush-every", type=int, default=2)
    p_pubmed.add_argument("--resume", action="store_true")
    p_pubmed.add_argument("--mindate", help="Publication date start year/date.")
    p_pubmed.add_argument("--maxdate", help="Publication date end year/date.")
    p_pubmed.add_argument("--sort", default="relevance", choices=["relevance", "pub date", "first author", "journal"])
    p_pubmed.add_argument("--email", default="", help="NCBI contact email, recommended.")
    p_pubmed.add_argument("--api-key", default="", help="NCBI API key, optional.")
    p_pubmed.add_argument("--tool", default="top-journal-review-writer")
    p_pubmed.add_argument("--timeout", type=int, default=40)
    p_pubmed.add_argument("--sleep", type=float, default=0.34)
    p_pubmed.add_argument("--user-agent", default="top-journal-review-writer/argument-literature-expander")
    p_pubmed.add_argument("--dry-run", action="store_true")
    p_pubmed.add_argument("--max-abstract-chars", type=int, default=1400)
    p_pubmed.set_defaults(func=cmd_pubmed)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
