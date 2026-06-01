#!/usr/bin/env python3
"""Manage a review literature pool, citation pool, and literature cards.

The pool is the persistent memory layer for a review project:

- literature pool: all recalled or draft-derived candidate papers
- citation pool: high-relevance papers selected for manuscript citation
- literature cards: compact, progressively loadable memory files
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


DEFAULT_POOL_DIR = "review-data/02_literature/pool"
SELECTED_STATUSES = {"citation_pool", "seminal", "method", "recent"}
DEFAULT_TOPIC_KEYWORDS = [
    "review",
    "evidence",
    "framework",
    "method",
    "model",
    "agent",
    "agents",
    "large language model",
    "llm",
    "artificial intelligence",
    "biomedical",
    "biomedicine",
    "health",
    "medical ai",
    "clinical",
    "autonomous",
    "closed-loop",
    "closed loop",
    "rag",
    "knowledge graph",
    "governance",
    "evaluation",
    "benchmark",
    "综述",
    "证据",
    "框架",
    "方法",
    "模型",
    "智能体",
    "大模型",
    "人工智能",
    "医学",
    "生物医学",
    "临床",
    "治理",
    "评估",
    "基准",
]
TOPIC_STOP_TERMS = {
    "and",
    "or",
    "the",
    "for",
    "with",
    "from",
    "into",
    "toward",
    "towards",
    "large",
    "language",
    "model",
    "models",
    "discovery",
    "review",
    "survey",
    "research",
    "paper",
    "study",
}
PREPRINT_DOI_PREFIXES = (
    "10.1101/",
    "10.21203/",
    "10.2139/ssrn",
    "10.48550/arxiv",
    "10.31219/osf.io",
    "10.20944/preprints",
)
PREPRINT_TEXT_MARKERS = ("arxiv", "openreview", "medrxiv", "biorxiv", "research square", "researchsquare", "ssrn")
OFFICIAL_CONFERENCE_SOURCE_MARKERS = (
    "openreview.net",
    "openreview",
    "papers.nips.cc",
    "proceedings.neurips.cc",
    "proceedings.mlr.press",
)
OFFICIAL_CONFERENCE_VENUE_MARKERS = (
    "iclr",
    "international conference on learning representations",
    "neurips",
    "nips",
    "advances in neural information processing systems",
    "icml",
    "international conference on machine learning",
)
HARD_EXCLUSION_PATTERNS = [
    (r"\bdecision letter\b", "decision_letter"),
    (r"\breview for\s+[\"']", "peer_review_record"),
    (r"\bresponse to reviewers?\b", "reviewer_response"),
    (r"\bsupplement(?:ary|al)?\b|\bsupp\d*\b|[_\-.]s00\d\b", "supplementary_material"),
    (r"\.(mp4|mov|avi|zip|xlsx?|pptx?)\b", "non_article_file"),
    (r"(^|\s)(figure|fig\.|table)\s*\d+|^图\s*\d+|^表\s*\d+|注[:：]", "figure_or_table_caption"),
    (r"\b(database|dataset|data set)\b|数据库|数据集", "database_or_dataset_record"),
    (r"\bindex\b|author index|cumulative author", "index_record"),
]
ALL_STATUSES = {
    "pool",
    "candidate",
    "citation_pool",
    "maybe",
    "background",
    "rejected",
    "not_found",
    "preprint_only",
    "seminal",
    "method",
    "recent",
}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def pool_paths(pool_dir: Path) -> dict[str, Path]:
    return {
        "root": pool_dir,
        "pool": pool_dir / "pool.json",
        "queries": pool_dir / "queries",
        "recall_runs": pool_dir / "recall_runs",
        "papers": pool_dir / "papers",
        "cards": pool_dir / "cards",
        "contexts": pool_dir / "contexts",
        "decisions": pool_dir / "decisions",
        "logs": pool_dir / "logs",
    }


def ensure_pool_dirs(pool_dir: Path) -> None:
    paths = pool_paths(pool_dir)
    for key, path in paths.items():
        if key != "pool":
            path.mkdir(parents=True, exist_ok=True)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(item) for item in value if item is not None)
    return str(value).strip()


def normalize_doi(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    return text.strip().rstrip(".")


def normalize_title(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", "", text)
    return text.strip()


def dedupe_key(record: dict[str, Any]) -> str:
    doi = normalize_doi(record.get("doi"))
    if doi:
        return "doi:" + doi
    paper_id = normalize_text(record.get("paper_id") or record.get("id")).lower()
    source = normalize_text(record.get("source")).lower()
    if source and paper_id:
        return f"id:{source}:{paper_id}"
    title = normalize_title(record.get("title"))
    authors = normalize_title(record.get("authors"))
    year = normalize_text(record.get("year"))
    return f"title:{title}|authors:{authors}|year:{year}"


def stable_key(record: dict[str, Any]) -> str:
    doi = normalize_doi(record.get("doi"))
    if doi:
        return "doi-" + re.sub(r"[^a-z0-9]+", "-", doi).strip("-")[:90]
    raw = dedupe_key(record)
    return "paper-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def slug(text: str, max_len: int = 80) -> str:
    text = normalize_title(text)
    text = re.sub(r"\s+", "-", text).strip("-")
    if not text:
        return "untitled"
    return text[:max_len].strip("-") or "untitled"


def empty_pool() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "papers": {},
        "dedupe_index": {},
        "decisions": [],
    }


def load_pool(pool_dir: Path) -> dict[str, Any]:
    ensure_pool_dirs(pool_dir)
    pool_path = pool_paths(pool_dir)["pool"]
    if not pool_path.exists():
        return empty_pool()
    data = json.loads(pool_path.read_text(encoding="utf-8"))
    data.setdefault("papers", {})
    data.setdefault("dedupe_index", {})
    data.setdefault("decisions", [])
    return data


def save_pool(pool_dir: Path, pool: dict[str, Any]) -> None:
    ensure_pool_dirs(pool_dir)
    pool["updated_at"] = now_iso()
    pool_paths(pool_dir)["pool"].write_text(
        json.dumps(pool, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def canonical_record(row: dict[str, Any], origin: str = "") -> dict[str, Any]:
    status = normalize_text(row.get("pool_status") or row.get("screening_status")) or "pool"
    if status == "include":
        status = "citation_pool"
    record = {
        "key": normalize_text(row.get("key")),
        "title": normalize_text(row.get("title")),
        "authors": normalize_text(row.get("authors")),
        "year": normalize_text(row.get("year") or row.get("publication_year") or row.get("published")),
        "source": normalize_text(row.get("source")),
        "paper_id": normalize_text(row.get("paper_id") or row.get("id")),
        "pmid": normalize_text(row.get("pmid")),
        "journal": normalize_text(row.get("journal")),
        "publication_type": normalize_text(row.get("publication_type")),
        "doi": normalize_doi(row.get("doi")),
        "url": normalize_text(row.get("url") or row.get("paper_url") or row.get("landing_page_url")),
        "pdf_url": normalize_text(row.get("pdf_url") or row.get("open_access_pdf")),
        "abstract": normalize_text(row.get("abstract") or row.get("summary")),
        "recall_queries": normalize_text(row.get("recall_queries") or row.get("_recall_query")),
        "pool_status": status if status in ALL_STATUSES else "pool",
        "relevance_score": normalize_text(row.get("relevance_score")),
        "decision_rationale": normalize_text(row.get("decision_rationale") or row.get("screening_reason")),
        "claim_supported": normalize_text(row.get("claim_supported")),
        "limitations": normalize_text(row.get("limitations")),
        "use_in_review": normalize_text(row.get("use_in_review")),
        "card_path": normalize_text(row.get("card_path")),
        "local_pdf": normalize_text(row.get("local_pdf")),
        "notes": normalize_text(row.get("notes")),
        "origin": origin or normalize_text(row.get("origin")),
        "imported_at": normalize_text(row.get("imported_at")) or now_iso(),
    }
    for field in [
        "retrieval_lane",
        "retrieval_task_id",
        "task_id",
        "linked_claim_id",
        "claim_id",
        "linked_claim",
        "claim_role",
        "argument_time_role",
        "evidence_need",
        "candidate_fit_score",
        "candidate_fit_status",
        "candidate_fit_reason",
        "fulltext_need",
        "human_decision_needed",
    ]:
        value = normalize_text(row.get(field))
        if value:
            record[field] = value
    return record


def merge_record(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for field, value in incoming.items():
        value_text = normalize_text(value)
        if field in {"source", "recall_queries", "origin"} and value_text:
            parts = {
                item.strip()
                for item in (normalize_text(existing.get(field)) + ";" + value_text).split(";")
                if item.strip()
            }
            merged[field] = "; ".join(sorted(parts))
        elif not normalize_text(merged.get(field)) and value_text:
            merged[field] = value
    if existing.get("pool_status") in SELECTED_STATUSES:
        merged["pool_status"] = existing["pool_status"]
    return merged


def import_records(pool: dict[str, Any], rows: list[dict[str, Any]], origin: str) -> tuple[int, int]:
    added = 0
    updated = 0
    for row in rows:
        record = canonical_record(row, origin=origin)
        dkey = dedupe_key(record)
        key = pool["dedupe_index"].get(dkey) or record.get("key") or stable_key(record)
        record["key"] = key
        if key in pool["papers"]:
            pool["papers"][key] = merge_record(pool["papers"][key], record)
            updated += 1
        else:
            pool["papers"][key] = record
            added += 1
        pool["dedupe_index"][dkey] = key
    return added, updated


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def markdown_escape(text: Any) -> str:
    return normalize_text(text).replace("|", "\\|").replace("\n", " ")


def read_docx_text(path: Path) -> str:
    pieces: list[str] = []
    with zipfile.ZipFile(path) as archive:
        with archive.open("word/document.xml") as handle:
            root = ElementTree.fromstring(handle.read())
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    for paragraph in root.findall(".//w:p", ns):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", ns)]
        if texts:
            pieces.append("".join(texts))
    return "\n".join(pieces)


def read_any_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return read_docx_text(path)
    return path.read_text(encoding="utf-8", errors="replace")


def split_reference_region(text: str) -> str:
    patterns = [
        r"(参考文献|參考文獻|references|bibliography)\s*[:：]?",
        r"(参考资料|參考資料)\s*[:：]?",
    ]
    lower_text = text.lower()
    for pattern in patterns:
        match = re.search(pattern, lower_text, flags=re.IGNORECASE)
        if match:
            return text[match.end() :]
    return text


def extract_reference_queries(text: str, max_queries: int) -> list[str]:
    region = split_reference_region(text)
    raw_lines = [line.strip() for line in region.splitlines() if line.strip()]
    candidates: list[str] = []
    for line in raw_lines:
        line = re.sub(r"^\s*(\[\d+\]|\d+[\.\)]|[（(]\d+[）)])\s*", "", line).strip()
        if len(line) < 12:
            continue
        quoted = re.findall(r"[\"“](.*?)[\"”]", line)
        if quoted:
            candidates.extend(q.strip() for q in quoted if len(q.strip()) >= 8)
            continue
        # Prefer the title-like middle of common reference lines.
        year_match = re.search(r"(19|20)\d{2}", line)
        if year_match:
            after_year = line[year_match.end() :].strip(" .,:;，。；：")
            if len(after_year) >= 12:
                line = after_year
        line = re.sub(r"\s+", " ", line)
        # Stop before obvious journal metadata when present.
        line = re.split(r"\.\s+(doi|DOI|https?://|[A-Z][A-Za-z &]+,?\s+\d{4})", line)[0].strip()
        if len(line) >= 12:
            candidates.append(line)
    seen: set[str] = set()
    queries: list[str] = []
    for candidate in candidates:
        key = normalize_title(candidate)
        if key and key not in seen:
            seen.add(key)
            queries.append(candidate)
        if len(queries) >= max_queries:
            break
    return queries


def write_query_files(pool_dir: Path, queries: list[str], prefix: str) -> tuple[Path, Path]:
    paths = pool_paths(pool_dir)
    txt_path = paths["queries"] / f"{prefix}_queries.txt"
    csv_path = paths["queries"] / f"{prefix}_queries.csv"
    txt_path.write_text("\n".join(queries) + ("\n" if queries else ""), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query", "origin", "status"])
        writer.writeheader()
        for query in queries:
            writer.writerow({"query": query, "origin": prefix, "status": "ready"})
    return txt_path, csv_path


def filtered_records(pool: dict[str, Any], statuses: set[str] | None = None) -> list[dict[str, Any]]:
    records = list(pool["papers"].values())
    if statuses:
        records = [r for r in records if normalize_text(r.get("pool_status")) in statuses]
    records.sort(
        key=lambda r: (
            float(r.get("relevance_score") or 0) if re.match(r"^\d+(\.\d+)?$", normalize_text(r.get("relevance_score"))) else 0,
            normalize_text(r.get("year")),
            normalize_text(r.get("title")),
        ),
        reverse=True,
    )
    return records


def topic_terms(topic: str, include_keywords: str = "") -> list[str]:
    raw_terms = list(DEFAULT_TOPIC_KEYWORDS)
    raw_terms.extend(term.strip() for term in include_keywords.split(",") if term.strip())
    raw_terms.extend(re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}|\b[A-Z]{2,}\b|[\u4e00-\u9fff]{2,}", topic or ""))
    seen: set[str] = set()
    terms: list[str] = []
    for term in raw_terms:
        normalized = normalize_text(term).lower()
        if normalized and normalized not in TOPIC_STOP_TERMS and normalized not in seen:
            seen.add(normalized)
            terms.append(normalized)
    return terms


def hard_exclusion_reason(record: dict[str, Any]) -> str:
    text = " ".join(
        normalize_text(record.get(field))
        for field in ["title", "doi", "url", "recall_queries", "notes"]
    ).lower()
    for pattern, reason in HARD_EXCLUSION_PATTERNS:
        if re.search(pattern, text, re.I):
            return reason
    return ""


def topic_relevance(record: dict[str, Any], terms: list[str]) -> tuple[int, list[str]]:
    text = " ".join(
        normalize_text(record.get(field))
        for field in ["title", "abstract", "notes", "claim_supported", "use_in_review"]
    ).lower()
    score = 0
    matched: list[str] = []
    for term in terms:
        if not term:
            continue
        if term in text:
            matched.append(term)
            score += 2 if " " in term or "-" in term else 1
    if matched and normalize_text(record.get("doi")):
        score += 1
    if matched and normalize_text(record.get("abstract")):
        score += 1
    return score, matched[:20]


def cmd_topic_filter(args: argparse.Namespace) -> int:
    pool_dir = Path(args.pool_dir)
    pool = load_pool(pool_dir)
    statuses = set(args.status.split(",")) if args.status else None
    records = filtered_records(pool, statuses=statuses)
    terms = topic_terms(args.topic, args.include_keywords)
    rows: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    for record in records:
        old_status = normalize_text(record.get("pool_status")) or "pool"
        exclusion = hard_exclusion_reason(record)
        score, matched = topic_relevance(record, terms)
        if exclusion:
            new_status = "rejected"
            rationale = f"hard_exclusion:{exclusion}"
        elif score >= args.min_score and len(matched) >= args.min_matched_terms:
            new_status = "candidate"
            rationale = "topic_match"
        elif score > 0:
            new_status = "background"
            rationale = "weak_topic_match"
        else:
            new_status = "rejected"
            rationale = "no_topic_match"
        counts[new_status] = counts.get(new_status, 0) + 1
        rows.append(
            {
                "key": normalize_text(record.get("key")),
                "title": normalize_text(record.get("title")),
                "old_status": old_status,
                "new_status": new_status,
                "score": str(score),
                "hard_exclusion": exclusion,
                "matched_terms": "; ".join(matched),
                "rationale": rationale,
            }
        )
        if args.apply:
            record["pool_status"] = new_status
            record["relevance_score"] = str(score)
            record["decision_rationale"] = (
                normalize_text(record.get("decision_rationale"))
                or f"Topic filter: {rationale}; matched terms: {', '.join(matched) or '[none]'}."
            )
    out = Path(args.out) if args.out else pool_paths(pool_dir)["logs"] / "topic_filter_report.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["key", "title", "old_status", "new_status", "score", "hard_exclusion", "matched_terms", "rationale"],
        )
        writer.writeheader()
        writer.writerows(rows)
    if args.apply:
        pool.setdefault("events", []).append(
            {
                "time": now_iso(),
                "event": "topic_filter",
                "topic": args.topic,
                "status_filter": args.status,
                "min_score": args.min_score,
                "counts": counts,
                "report": str(out),
            }
        )
        save_pool(pool_dir, pool)
    print(json.dumps({"out": str(out), "checked": len(rows), "counts": counts, "applied": args.apply}, ensure_ascii=False))
    return 0


def card_text(record: dict[str, Any]) -> str:
    doi_or_url = record.get("doi") or record.get("url") or record.get("pdf_url")
    lines = [
        f"# {record.get('title') or record.get('key')}",
        "",
        "## Metadata",
        "",
        f"- Key: `{record.get('key')}`",
        f"- Authors/year: {record.get('authors') or 'unknown'} ({record.get('year') or 'unknown'})",
        f"- Source: {record.get('source') or 'unknown'}",
        f"- Journal/type: {record.get('journal') or 'unknown'}; {record.get('publication_type') or 'unknown'}",
        f"- PMID: {record.get('pmid') or record.get('paper_id') if record.get('source') == 'pubmed' else record.get('pmid') or 'missing'}",
        f"- DOI/URL: {doi_or_url or 'missing'}",
        f"- Local PDF: {record.get('local_pdf') or 'not downloaded'}",
        "",
        "## Argument Link",
        "",
        f"- Evidence need: {record.get('evidence_need') or '[not assigned]'}",
        f"- Claim role: {record.get('claim_role') or '[not assigned]'} / {record.get('argument_time_role') or '[not assigned]'}",
        f"- Linked claim: {record.get('linked_claim') or record.get('claim_supported') or '[not linked]'}",
        f"- Candidate fit: {record.get('candidate_fit_status') or '[not screened]'} {record.get('candidate_fit_score') or ''}",
        f"- Fit reason: {record.get('candidate_fit_reason') or '[not screened]'}",
        "",
        "## Why In Citation Pool",
        "",
        record.get("decision_rationale") or "[Fill after relevance reasoning]",
        "",
        "## Claims Supported",
        "",
        record.get("claim_supported") or "[Add claim-level support before citing]",
        "",
        "## Limits And Cautions",
        "",
        record.get("limitations") or "[Add limitations before citing]",
        "",
        "## Use In Review",
        "",
        record.get("use_in_review") or "[Assign section, figure, table, or argument]",
        "",
        "## Abstract Or Notes",
        "",
        record.get("abstract") or record.get("notes") or "[No abstract/notes available]",
        "",
    ]
    return "\n".join(lines)


def build_base_command(args: argparse.Namespace) -> list[str]:
    if getattr(args, "paper_search_repo", ""):
        return ["uv", "run", "--directory", args.paper_search_repo, "paper-search"]
    if getattr(args, "paper_search_cmd", ""):
        return shlex.split(args.paper_search_cmd)
    return ["paper-search"]


def first_source_and_id(record: dict[str, Any]) -> tuple[str, str]:
    source = normalize_text(record.get("source")).split(";")[0].strip()
    paper_id = normalize_text(record.get("paper_id")).split(";")[0].strip()
    return source, paper_id


def cmd_init(args: argparse.Namespace) -> int:
    pool_dir = Path(args.pool_dir)
    ensure_pool_dirs(pool_dir)
    pool_path = pool_paths(pool_dir)["pool"]
    if pool_path.exists() and not args.force:
        raise SystemExit(f"Pool already exists: {pool_path}. Use --force to overwrite.")
    save_pool(pool_dir, empty_pool())
    print(json.dumps({"pool_dir": str(pool_dir), "status": "created"}, ensure_ascii=False))
    return 0


def cmd_draft_queries(args: argparse.Namespace) -> int:
    pool_dir = Path(args.pool_dir)
    pool = load_pool(pool_dir)
    draft_path = Path(args.draft)
    text = read_any_text(draft_path)
    queries = extract_reference_queries(text, max_queries=args.max_queries)
    txt_path, csv_path = write_query_files(pool_dir, queries, "draft_reference")
    pool["draft_reference_query_file"] = str(txt_path)
    pool.setdefault("events", []).append(
        {"time": now_iso(), "event": "draft_queries", "draft": str(draft_path), "count": len(queries)}
    )
    save_pool(pool_dir, pool)
    print(json.dumps({"queries": len(queries), "query_file": str(txt_path), "csv": str(csv_path)}, ensure_ascii=False))
    return 0


def cmd_import_recall(args: argparse.Namespace) -> int:
    pool_dir = Path(args.pool_dir)
    pool = load_pool(pool_dir)
    recall_dir = Path(args.recall_dir)
    candidates = [recall_dir / "papers.jsonl", recall_dir / "screening.csv", recall_dir / "papers.csv"]
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.exists():
            rows = read_jsonl(candidate) if candidate.suffix == ".jsonl" else read_csv(candidate)
            break
    if not rows:
        raise SystemExit(f"No recall output found in {recall_dir}")
    added, updated = import_records(pool, rows, origin=args.origin or str(recall_dir))
    save_pool(pool_dir, pool)
    print(json.dumps({"pool_dir": str(pool_dir), "added": added, "updated": updated, "total": len(pool["papers"])}, ensure_ascii=False))
    return 0


def cmd_import_csv(args: argparse.Namespace) -> int:
    pool_dir = Path(args.pool_dir)
    pool = load_pool(pool_dir)
    rows = read_csv(Path(args.csv))
    added, updated = import_records(pool, rows, origin=args.origin or str(args.csv))
    save_pool(pool_dir, pool)
    print(json.dumps({"pool_dir": str(pool_dir), "added": added, "updated": updated, "total": len(pool["papers"])}, ensure_ascii=False))
    return 0


def verified_union_key(row: dict[str, Any]) -> str:
    doi = normalize_doi(row.get("doi") or row.get("draft_doi"))
    if doi:
        return "doi:" + doi
    pmid = normalize_text(row.get("pmid") or row.get("paper_id"))
    if re.fullmatch(r"\d+", pmid):
        return "pmid:" + pmid
    title = normalize_title(row.get("title") or row.get("draft_candidate_title") or row.get("candidate_title"))
    if title:
        return "title:" + title
    return "row:" + hashlib.sha1(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def verified_union_score(row: dict[str, Any]) -> int:
    score = 0
    if normalize_text(row.get("verification_status")).startswith("verified_"):
        score += 100
    if normalize_text(row.get("llm_decision")) == "accept_verified":
        score += 40
    if normalize_text(row.get("publication_status")) == "published_paper":
        score += 30
    if normalize_text(row.get("pmid") or row.get("paper_id")):
        score += 20
    if normalize_doi(row.get("doi") or row.get("draft_doi")):
        score += 15
    if normalize_text(row.get("abstract")):
        score += 5
    return score


def merge_union_row(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    if verified_union_score(incoming) > verified_union_score(existing):
        base = dict(incoming)
        other = existing
    else:
        base = dict(existing)
        other = incoming
    multi_fields = {
        "origin",
        "source_lane",
        "source_file",
        "claim_supported",
        "supported_claim_ids",
        "decision_rationale",
        "notes",
        "recall_queries",
        "raw_reference",
        "draft",
        "ref_number",
        "candidate_id",
    }
    for field, value in other.items():
        value_text = normalize_text(value)
        if not value_text:
            continue
        if field in multi_fields:
            parts = {
                item.strip()
                for item in (normalize_text(base.get(field)) + ";" + value_text).split(";")
                if item.strip()
            }
            base[field] = "; ".join(sorted(parts))
        elif not normalize_text(base.get(field)):
            base[field] = value
    return base


def cmd_merge_verified(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    for csv_path in args.csv:
        path = Path(csv_path)
        if not path.exists():
            if args.skip_missing:
                continue
            raise SystemExit(f"Missing verified CSV: {path}")
        for row in read_csv(path):
            row = dict(row)
            row["source_file"] = str(path)
            row["source_lane"] = normalize_text(row.get("origin")) or path.parent.name or path.stem
            if args.origin and not normalize_text(row.get("origin")):
                row["origin"] = args.origin
            rows.append(row)
    chosen: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    duplicates: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, 1):
        key = verified_union_key(row)
        row["union_key"] = key
        if key not in chosen:
            chosen[key] = row
            order.append(key)
            continue
        previous = chosen[key]
        duplicates.append(
            {
                "union_key": key,
                "kept_title": normalize_text(previous.get("title") or previous.get("draft_candidate_title")),
                "duplicate_title": normalize_text(row.get("title") or row.get("draft_candidate_title")),
                "kept_source_file": normalize_text(previous.get("source_file")),
                "duplicate_source_file": normalize_text(row.get("source_file")),
                "kept_score_before_merge": str(verified_union_score(previous)),
                "duplicate_score": str(verified_union_score(row)),
            }
        )
        chosen[key] = merge_union_row(previous, row)
    union_rows = [chosen[key] for key in order]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preferred = [
        "union_key",
        "key",
        "title",
        "draft_candidate_title",
        "authors",
        "year",
        "journal",
        "source",
        "paper_id",
        "pmid",
        "doi",
        "url",
        "verification_status",
        "llm_decision",
        "publication_status",
        "claim_fit",
        "claim_supported",
        "origin",
        "source_lane",
        "source_file",
        "pool_status",
        "decision_rationale",
        "notes",
        "abstract",
    ]
    all_fields = preferred + sorted({field for row in union_rows for field in row.keys()} - set(preferred))
    write_csv(out_dir / "verified_papers_union.csv", union_rows, all_fields)
    write_csv(out_dir / "verified_papers_union_duplicates.csv", duplicates, ["union_key", "kept_title", "duplicate_title", "kept_source_file", "duplicate_source_file", "kept_score_before_merge", "duplicate_score"])
    lane_counts: dict[str, int] = {}
    for row in union_rows:
        for lane in normalize_text(row.get("source_lane") or row.get("origin")).split(";"):
            lane = lane.strip() or "unknown"
            lane_counts[lane] = lane_counts.get(lane, 0) + 1
    lines = [
        "# Verified Papers Union",
        "",
        f"- Input CSV files: {len(args.csv)}",
        f"- Input rows: {len(rows)}",
        f"- Unique papers: {len(union_rows)}",
        f"- Duplicate rows collapsed: {len(duplicates)}",
        "",
        "## Lane Counts",
        "",
    ]
    for lane, count in sorted(lane_counts.items()):
        lines.append(f"- `{lane}`: {count}")
    lines.extend(["", "## Next Gates", "", "- Run official citation export on `verified_papers_union.csv`.", "- Import the union, not separate branches, into the main pool when the project is ready."])
    (out_dir / "verified_papers_union_report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    imported: dict[str, Any] = {}
    if args.import_to_pool:
        pool_dir = Path(args.pool_dir)
        pool = load_pool(pool_dir)
        added, updated = import_records(pool, union_rows, origin=args.origin or "verified-union")
        save_pool(pool_dir, pool)
        imported = {"pool_dir": str(pool_dir), "added": added, "updated": updated, "total": len(pool["papers"])}
    print(json.dumps({"out_dir": str(out_dir), "input_rows": len(rows), "unique_papers": len(union_rows), "duplicates": len(duplicates), "imported": imported}, ensure_ascii=False))
    return 0


def cmd_export_candidates(args: argparse.Namespace) -> int:
    pool = load_pool(Path(args.pool_dir))
    statuses = set(args.status.split(",")) if args.status else None
    records = filtered_records(pool, statuses=statuses)[: args.limit]
    lines = [
        "# Citation Pool Selection Packet",
        "",
        "Use this packet for model-assisted relevance reasoning. Select only papers that directly support the review thesis, framework, evidence table, or major claims.",
        "",
        "For each selected paper, record: relevance score 1-5, rationale, claim supported, limitation, and intended section.",
        "",
    ]
    for index, record in enumerate(records, 1):
        lines.extend(
            [
                f"## {index}. {record.get('title') or record.get('key')}",
                "",
                f"- Key: `{record.get('key')}`",
                f"- Authors/year: {record.get('authors') or 'unknown'} ({record.get('year') or 'unknown'})",
                f"- Status: {record.get('pool_status')}",
                f"- Source: {record.get('source')}",
                f"- DOI/URL: {record.get('doi') or record.get('url') or record.get('pdf_url') or 'missing'}",
                f"- Recall queries: {record.get('recall_queries') or 'unknown'}",
                "",
                "Abstract/notes:",
                "",
                (record.get("abstract") or record.get("notes") or "[No abstract available]")[: args.max_abstract_chars],
                "",
            ]
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(out), "records": len(records)}, ensure_ascii=False))
    return 0


def cmd_decide(args: argparse.Namespace) -> int:
    status = args.status.strip()
    if status not in ALL_STATUSES:
        raise SystemExit(f"Unknown status: {status}. Allowed: {', '.join(sorted(ALL_STATUSES))}")
    pool_dir = Path(args.pool_dir)
    pool = load_pool(pool_dir)
    record = pool["papers"].get(args.key)
    if not record:
        raise SystemExit(f"Paper key not found: {args.key}")
    record["pool_status"] = status
    if args.score:
        record["relevance_score"] = str(args.score)
    if args.rationale:
        record["decision_rationale"] = args.rationale
    if args.claim:
        record["claim_supported"] = args.claim
    if args.limitation:
        record["limitations"] = args.limitation
    if args.use:
        record["use_in_review"] = args.use
    decision = {
        "time": now_iso(),
        "key": args.key,
        "status": status,
        "score": args.score,
        "rationale": args.rationale,
        "claim": args.claim,
        "limitation": args.limitation,
        "use": args.use,
    }
    pool["decisions"].append(decision)
    save_pool(pool_dir, pool)
    print(json.dumps({"key": args.key, "status": status}, ensure_ascii=False))
    return 0


def cmd_build_cards(args: argparse.Namespace) -> int:
    pool_dir = Path(args.pool_dir)
    pool = load_pool(pool_dir)
    statuses = set(args.status.split(",")) if args.status else SELECTED_STATUSES
    cards_dir = pool_paths(pool_dir)["cards"]
    records = filtered_records(pool, statuses=statuses)
    index_lines = ["# Literature Card Index", ""]
    for record in records:
        key = record.get("key") or stable_key(record)
        filename = f"{key[:50]}-{slug(record.get('title', ''), max_len=50)}.md"
        path = cards_dir / filename
        path.write_text(card_text(record), encoding="utf-8")
        record["card_path"] = str(path)
        index_lines.append(f"- [{markdown_escape(record.get('title'))}]({filename}) - `{key}`")
    (cards_dir / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    save_pool(pool_dir, pool)
    print(json.dumps({"cards_dir": str(cards_dir), "cards": len(records)}, ensure_ascii=False))
    return 0


def cmd_export_context(args: argparse.Namespace) -> int:
    pool_dir = Path(args.pool_dir)
    pool = load_pool(pool_dir)
    statuses = set(args.status.split(",")) if args.status else SELECTED_STATUSES
    records = filtered_records(pool, statuses=statuses)[: args.max_cards]
    lines = [
        "# Citation Context",
        "",
        "Load this compact context when drafting citation-heavy sections. For detailed notes, open the individual card files listed below.",
        "",
    ]
    for record in records:
        card = card_text(record)
        if len(card) > args.max_chars_per_card:
            card = card[: args.max_chars_per_card].rstrip() + "\n[Card truncated; open card file for details.]"
        lines.extend([card, ""])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(out), "cards": len(records)}, ensure_ascii=False))
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    pool_dir = Path(args.pool_dir)
    pool = load_pool(pool_dir)
    base = build_base_command(args)
    papers_dir = pool_paths(pool_dir)["papers"]
    statuses = set(args.status.split(",")) if args.status else SELECTED_STATUSES
    records = filtered_records(pool, statuses=statuses)[: args.limit]
    log: list[dict[str, Any]] = []
    for record in records:
        source, paper_id = first_source_and_id(record)
        if not source or not paper_id:
            log.append({"key": record.get("key"), "status": "skipped", "reason": "missing source or paper_id"})
            continue
        paper_out = papers_dir / (record.get("key") or stable_key(record))
        paper_out.mkdir(parents=True, exist_ok=True)
        command = [*base, "download", source, paper_id, "-o", str(paper_out)]
        if args.dry_run:
            log.append({"key": record.get("key"), "status": "dry_run", "command": command})
            continue
        completed = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=args.timeout,
        )
        files = [str(path) for path in paper_out.glob("*") if path.is_file()]
        if files:
            record["local_pdf"] = files[0]
        log.append(
            {
                "key": record.get("key"),
                "status": "ok" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "command": command,
                "stdout": completed.stdout[-1000:],
                "stderr": completed.stderr[-1000:],
                "files": files,
            }
        )
    log_path = pool_paths(pool_dir)["logs"] / f"download_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    save_pool(pool_dir, pool)
    print(json.dumps({"download_log": str(log_path), "attempted": len(records)}, ensure_ascii=False))
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    pool_dir = Path(args.pool_dir)
    pool = load_pool(pool_dir)
    records = list(pool["papers"].values())
    selected = filtered_records(pool, statuses=SELECTED_STATUSES)
    missing_cards = [r for r in selected if not r.get("card_path")]
    missing_claims = [r for r in selected if not r.get("claim_supported")]
    missing_identifiers = [r for r in records if not (r.get("doi") or r.get("url") or r.get("pdf_url"))]
    counts: dict[str, int] = {}
    for record in records:
        status = normalize_text(record.get("pool_status")) or "pool"
        counts[status] = counts.get(status, 0) + 1
    lines = [
        "# Literature Pool Audit",
        "",
        f"- Total pool records: {len(records)}",
        f"- Citation pool records: {len(selected)}",
        f"- Records missing DOI/URL/PDF URL: {len(missing_identifiers)}",
        f"- Citation pool records missing cards: {len(missing_cards)}",
        f"- Citation pool records missing supported claims: {len(missing_claims)}",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"- {status}: {count}")
    if missing_claims:
        lines.extend(["", "## Selected Records Missing Claims", ""])
        for record in missing_claims[:50]:
            lines.append(f"- `{record.get('key')}` {record.get('title')}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "records": len(records), "selected": len(selected)}, ensure_ascii=False))
    return 0


def is_formally_published(record: dict[str, Any]) -> bool:
    if is_official_conference_paper(record):
        return True
    source = normalize_text(record.get("source")).lower()
    pub_type = normalize_text(record.get("publication_type")).lower()
    journal = normalize_text(record.get("journal"))
    doi = normalize_doi(record.get("doi"))
    pmid = normalize_text(record.get("pmid") or record.get("paper_id") if "pubmed" in source else record.get("pmid"))
    preprint_text = " ".join(
        normalize_text(record.get(field)).lower()
        for field in ["source", "title", "url", "doi", "recall_queries", "notes"]
    )
    if any(marker in preprint_text for marker in PREPRINT_TEXT_MARKERS):
        return False
    if doi.startswith(PREPRINT_DOI_PREFIXES) or "arxiv" in doi:
        return False
    if any(marker in pub_type for marker in ["preprint", "posted-content", "posted content"]):
        return False
    if source in {"pubmed", "pmc", "crossref", "openalex"} and (doi or pmid or journal):
        return True
    if doi:
        return True
    if doi and journal and not any(marker in journal.lower() for marker in ["arxiv", "openreview", "preprint"]):
        return True
    return False


def is_official_conference_paper(record: dict[str, Any]) -> bool:
    source_kind = normalize_text(record.get("source_kind_verified")).lower()
    pub_type = normalize_text(record.get("publication_type")).lower()
    text = " ".join(
        normalize_text(record.get(field)).lower()
        for field in ["source", "title", "url", "journal", "publication_type", "recall_queries", "notes"]
    )
    has_source = any(marker in text for marker in OFFICIAL_CONFERENCE_SOURCE_MARKERS)
    has_venue = any(marker in text for marker in OFFICIAL_CONFERENCE_VENUE_MARKERS)
    if "official_conference" in source_kind:
        return True
    if "conference" in pub_type and (has_source or has_venue):
        return True
    return has_source and has_venue


def publication_gate_reason(record: dict[str, Any]) -> str:
    if is_official_conference_paper(record):
        return "official_conference_paper"
    if is_formally_published(record):
        return "published"
    source = normalize_text(record.get("source")).lower()
    pub_type = normalize_text(record.get("publication_type")).lower()
    doi = normalize_doi(record.get("doi"))
    preprint_text = " ".join(
        normalize_text(record.get(field)).lower()
        for field in ["source", "title", "url", "doi", "recall_queries", "notes"]
    )
    if doi.startswith(PREPRINT_DOI_PREFIXES) or "arxiv" in doi or any(marker in preprint_text for marker in PREPRINT_TEXT_MARKERS) or any(marker in pub_type for marker in ["preprint", "posted-content", "posted content"]):
        return "preprint_or_review_platform_only"
    if not (record.get("doi") or record.get("pmid") or record.get("url")):
        return "no_verifiable_identifier"
    return "not_confirmed_as_published"


def cmd_final_gate(args: argparse.Namespace) -> int:
    pool_dir = Path(args.pool_dir)
    pool = load_pool(pool_dir)
    statuses = set(args.status.split(",")) if args.status else SELECTED_STATUSES
    records = filtered_records(pool, statuses=statuses)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record in records:
        reason = publication_gate_reason(record)
        record["publication_gate"] = reason
        if reason in {"published", "official_conference_paper"}:
            accepted.append(record)
        else:
            rejected.append(record)
            if args.apply:
                if reason == "preprint_or_review_platform_only":
                    record["pool_status"] = "preprint_only"
                elif reason == "no_verifiable_identifier":
                    record["pool_status"] = "not_found"
                else:
                    record["pool_status"] = "rejected"
                record["decision_rationale"] = (
                    normalize_text(record.get("decision_rationale"))
                    + ("; " if record.get("decision_rationale") else "")
                    + f"Final citation gate: {reason}"
                )
    lines = [
        "# Final Citation Gate",
        "",
        "Policy: final manuscript citations must be formally published papers, official conference papers/proceedings, or official guidelines/standards. Preprints and OpenReview-only records without official venue evidence can remain as background leads but should not enter the final reference list.",
        "",
        f"- Checked records: {len(records)}",
        f"- Accepted final citations: {len(accepted)}",
        f"- Rejected/demoted: {len(rejected)}",
        f"- Applied to pool: {args.apply}",
        "",
        "## Accepted",
        "",
        "| Key | Title | Year | Source | DOI/URL |",
        "|---|---|---:|---|---|",
    ]
    for record in accepted:
        lines.append(
            "| "
            + " | ".join(
                markdown_escape(value)
                for value in [
                    record.get("key", ""),
                    record.get("title", ""),
                    record.get("year", ""),
                    record.get("source", ""),
                    record.get("doi") or record.get("url") or record.get("pmid") or "",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Rejected Or Demoted", "", "| Key | Title | Source | Reason | Action |", "|---|---|---|---|---|"])
    for record in rejected:
        reason = record.get("publication_gate") or publication_gate_reason(record)
        action = "demote to background/preprint-only; replace with published paper" if reason == "preprint_or_review_platform_only" else "delete from citation candidates unless verified elsewhere"
        lines.append(
            "| "
            + " | ".join(
                markdown_escape(value)
                for value in [
                    record.get("key", ""),
                    record.get("title", ""),
                    record.get("source", ""),
                    reason,
                    action,
                ]
            )
            + " |"
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.apply:
        save_pool(pool_dir, pool)
    print(json.dumps({"out": str(out), "checked": len(records), "accepted": len(accepted), "rejected": len(rejected), "applied": args.apply}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage a literature pool, citation pool, and literature cards.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Create literature pool directories and pool.json.")
    p_init.add_argument("--pool-dir", default=DEFAULT_POOL_DIR)
    p_init.add_argument("--force", action="store_true")
    p_init.set_defaults(func=cmd_init)

    p_draft = sub.add_parser("draft-queries", help="Extract title-like recall queries from a draft document.")
    p_draft.add_argument("--pool-dir", default=DEFAULT_POOL_DIR)
    p_draft.add_argument("--draft", required=True)
    p_draft.add_argument("--max-queries", type=int, default=80)
    p_draft.set_defaults(func=cmd_draft_queries)

    p_import = sub.add_parser("import-recall", help="Import paper_recall.py outputs into the literature pool.")
    p_import.add_argument("--pool-dir", default=DEFAULT_POOL_DIR)
    p_import.add_argument("--recall-dir", required=True)
    p_import.add_argument("--origin", default="")
    p_import.set_defaults(func=cmd_import_recall)

    p_import_csv = sub.add_parser("import-csv", help="Import a CSV of verified papers or user corpus rows.")
    p_import_csv.add_argument("--pool-dir", default=DEFAULT_POOL_DIR)
    p_import_csv.add_argument("--csv", required=True)
    p_import_csv.add_argument("--origin", default="")
    p_import_csv.set_defaults(func=cmd_import_csv)

    p_merge_verified = sub.add_parser("merge-verified", help="Merge accepted verified papers from explicit, discovery, hidden, and PubMed-deep-dive lanes into one deduped union.")
    p_merge_verified.add_argument("--csv", action="append", required=True, help="Accepted verified CSV; repeat for each lane.")
    p_merge_verified.add_argument("--out-dir", default="review-data/05_audit/verified_papers_union")
    p_merge_verified.add_argument("--origin", default="")
    p_merge_verified.add_argument("--pool-dir", default=DEFAULT_POOL_DIR)
    p_merge_verified.add_argument("--import-to-pool", action="store_true", help="Import the deduped union into the governed pool after writing reports.")
    p_merge_verified.add_argument("--skip-missing", action="store_true", help="Ignore optional lane CSVs that were not produced in this project.")
    p_merge_verified.set_defaults(func=cmd_merge_verified)

    p_topic = sub.add_parser("topic-filter", help="Screen pool records for topic relevance and obvious non-article pollution.")
    p_topic.add_argument("--pool-dir", default=DEFAULT_POOL_DIR)
    p_topic.add_argument("--topic", required=True)
    p_topic.add_argument("--include-keywords", default="", help="Comma-separated extra domain keywords.")
    p_topic.add_argument("--status", default="pool,candidate,maybe,background")
    p_topic.add_argument("--min-score", type=int, default=3)
    p_topic.add_argument("--min-matched-terms", type=int, default=2)
    p_topic.add_argument("--out", default="")
    p_topic.add_argument("--apply", action="store_true")
    p_topic.set_defaults(func=cmd_topic_filter)

    p_candidates = sub.add_parser("export-candidates", help="Export a model-readable selection packet.")
    p_candidates.add_argument("--pool-dir", default=DEFAULT_POOL_DIR)
    p_candidates.add_argument("--out", required=True)
    p_candidates.add_argument("--status", default="pool,candidate,maybe,background")
    p_candidates.add_argument("--limit", type=int, default=120)
    p_candidates.add_argument("--max-abstract-chars", type=int, default=1200)
    p_candidates.set_defaults(func=cmd_export_candidates)

    p_decide = sub.add_parser("decide", help="Record model/human decision for a paper.")
    p_decide.add_argument("--pool-dir", default=DEFAULT_POOL_DIR)
    p_decide.add_argument("--key", required=True)
    p_decide.add_argument("--status", required=True)
    p_decide.add_argument("--score", default="")
    p_decide.add_argument("--rationale", default="")
    p_decide.add_argument("--claim", default="")
    p_decide.add_argument("--limitation", default="")
    p_decide.add_argument("--use", default="")
    p_decide.set_defaults(func=cmd_decide)

    p_cards = sub.add_parser("build-cards", help="Build literature cards for selected papers.")
    p_cards.add_argument("--pool-dir", default=DEFAULT_POOL_DIR)
    p_cards.add_argument("--status", default="citation_pool,seminal,method,recent")
    p_cards.set_defaults(func=cmd_build_cards)

    p_context = sub.add_parser("export-context", help="Export compact citation context from selected cards.")
    p_context.add_argument("--pool-dir", default=DEFAULT_POOL_DIR)
    p_context.add_argument("--out", required=True)
    p_context.add_argument("--status", default="citation_pool,seminal,method,recent")
    p_context.add_argument("--max-cards", type=int, default=40)
    p_context.add_argument("--max-chars-per-card", type=int, default=1600)
    p_context.set_defaults(func=cmd_export_context)

    p_download = sub.add_parser("download", help="Download selected papers through paper-search CLI.")
    p_download.add_argument("--pool-dir", default=DEFAULT_POOL_DIR)
    p_download.add_argument("--status", default="citation_pool,seminal,method,recent")
    p_download.add_argument("--limit", type=int, default=50)
    p_download.add_argument("--paper-search-cmd", default="")
    p_download.add_argument("--paper-search-repo", default="")
    p_download.add_argument("--timeout", type=int, default=120)
    p_download.add_argument("--dry-run", action="store_true")
    p_download.set_defaults(func=cmd_download)

    p_audit = sub.add_parser("audit", help="Audit the pool, citation pool, cards, and claim fields.")
    p_audit.add_argument("--pool-dir", default=DEFAULT_POOL_DIR)
    p_audit.add_argument("--out", required=True)
    p_audit.set_defaults(func=cmd_audit)

    p_gate = sub.add_parser("final-gate", help="Enforce published-only final citation policy.")
    p_gate.add_argument("--pool-dir", default=DEFAULT_POOL_DIR)
    p_gate.add_argument("--status", default="citation_pool,seminal,method,recent")
    p_gate.add_argument("--out", required=True)
    p_gate.add_argument("--apply", action="store_true", help="Demote/reject records that fail the gate in pool.json.")
    p_gate.set_defaults(func=cmd_final_gate)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
