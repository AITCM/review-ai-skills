#!/usr/bin/env python3
"""Codex-subtask literature candidate mining from draft full text.

This script runs before broad supplemental recall. It creates Codex packets by
default so Codex subtasks can read draft passages and extract paper clues that
are visible in the prose but missing or incomplete in the reference section:
system names, author/year hints, venue clues, title fragments, DOI/URL
fragments, and claim-linked PubMed deep-dive queries. Optional external LLM
calls require explicit approval for data egress.

The output is only a candidate layer. Every row must still pass deterministic
verification and claim-fit screening before entering the governed pool.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import textwrap
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pubmed_recall


SUPPORTED_EXTENSIONS = {".md", ".txt", ".docx"}
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_ENV_FILES = (".env.local", ".env")

CANDIDATE_FIELDS = [
    "candidate_id",
    "draft",
    "chunk_id",
    "claim_id",
    "candidate_title",
    "authors_or_group",
    "year_hint",
    "venue_hint",
    "system_or_method",
    "doi",
    "pmid",
    "urls",
    "raw_evidence_excerpt",
    "why_relevant",
    "candidate_type",
    "confidence",
    "source_kind",
    "cited_in_body",
    "verification_priority",
    "pubmed_query",
    "notes",
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

SUBAGENT_INPUT_FIELDS = [
    "packet",
    "file_or_context",
    "candidate_title_or_system",
    "authors_if_any",
    "venue_year_if_any",
    "why_it_is_likely_a_paper",
    "search_query",
    "claim_context",
    "confidence",
    "notes",
]

NORMALIZED_IDENTITY_FIELDS = [
    "candidate_id",
    "draft",
    "chunk_id",
    "candidate_title",
    "system_or_method",
    "official_title",
    "official_authors",
    "official_year",
    "official_venue",
    "doi",
    "pmid",
    "official_url",
    "normalization_status",
    "normalization_confidence",
    "normalization_rationale",
    "evidence_source",
    "raw_evidence_excerpt",
    "claim_context",
    "notes",
]

QUERY_FIELDS = [
    "task_id",
    "candidate_id",
    "draft",
    "claim_id",
    "candidate_title",
    "system_or_method",
    "query",
    "reason",
    "priority",
    "status",
]

PUBMED_FIELDS = [
    "task_id",
    "candidate_id",
    "draft",
    "claim_id",
    "candidate_title",
    "query",
    "reason",
    "priority",
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
    "abstract",
    "publication_type",
    "candidate_fit_status",
    "candidate_fit_reason",
    "notes",
]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def clean_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def safe_id(value: str, fallback: str = "item") -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", clean_space(value) or fallback).strip("-")
    return text[:90] or fallback


def normalize_doi(value: Any) -> str:
    text = clean_space(value).lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi\s*:\s*", "", text)
    return text.rstrip(".,;)")


def normalize_title(value: Any) -> str:
    text = clean_space(value).lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def read_docx(path: Path) -> str:
    pieces: list[str] = []
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    for para in root.findall(".//w:p", ns):
        runs = [node.text or "" for node in para.findall(".//w:t", ns)]
        line = clean_space("".join(runs))
        if line:
            pieces.append(line)
    return "\n".join(pieces)


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
    out: list[Path] = []
    for path in paths:
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists() or path.is_dir():
        return values
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def is_placeholder_secret(value: str) -> bool:
    if not value:
        return True
    lowered = value.lower()
    return any(token in lowered for token in ["your_", "placeholder", "example", "replace", "<", ">"])


def project_env(project_dir: Path, env_file: str = "") -> tuple[dict[str, str], list[str]]:
    paths = [Path(env_file)] if env_file else [project_dir / name for name in DEFAULT_ENV_FILES]
    merged: dict[str, str] = {}
    loaded: list[str] = []
    for path in paths:
        if not path.is_absolute():
            path = project_dir / path
        if path.exists():
            merged.update(parse_env_file(path))
            loaded.append(str(path))
    return merged, loaded


def api_config(args: argparse.Namespace) -> dict[str, Any]:
    project_dir = Path(args.project_dir).resolve()
    env_values, loaded = project_env(project_dir, args.env_file)
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
    return {"api_key": api_key, "base_url": base_url.rstrip("/"), "model": model, "loaded_env_files": loaded}


def chat_once(prompt: str, args: argparse.Namespace, user_id: str = "") -> str:
    cfg = api_config(args)
    if not cfg["api_key"]:
        raise SystemExit("No API key. Set DEEPSEEK_API_KEY/REVIEW_AGENT_API_KEY or use packet mode.")
    payload: dict[str, Any] = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": "You mine paper candidates from draft text for a rigorous review. Return strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": args.temperature,
        "response_format": {"type": "json_object"},
    }
    if user_id:
        payload["user_id"] = user_id
    if args.thinking != "omit":
        payload["thinking"] = {"type": args.thinking}
    if args.reasoning_effort:
        payload["reasoning_effort"] = args.reasoning_effort
    request = urllib.request.Request(
        cfg["base_url"] + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        body = json.loads(response.read().decode("utf-8", errors="replace"))
    return body["choices"][0]["message"]["content"]


def chat(prompt: str, args: argparse.Namespace, user_id: str = "") -> str:
    last: Exception | None = None
    for attempt in range(max(1, args.max_retries + 1)):
        try:
            return chat_once(prompt, args, user_id=user_id)
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt >= args.max_retries:
                raise
            time.sleep(args.retry_backoff * (2**attempt))
        except urllib.error.URLError as exc:
            last = exc
            if attempt >= args.max_retries:
                raise
            time.sleep(args.retry_backoff * (2**attempt))
    raise RuntimeError(f"LLM call failed after retries: {last}")


def coerce_json_results(text: str) -> list[dict[str, Any]]:
    raw = text.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", raw, flags=re.S)
        if not match:
            raise
        data = json.loads(match.group(1))
    if isinstance(data, dict):
        for key in ["candidates", "results", "items", "papers"]:
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    if isinstance(data, list):
        return data
    return []


def split_reference_section(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        cleaned = clean_space(re.sub(r"^#{1,6}\s*", "", line)).strip("*_` ")
        if re.match(r"^\s*(references|bibliography|works cited|reference list|参考文献|參考文獻|引用文献|引用文獻|参考资料|參考資料|文献|文獻)\s*[:：]?\s*$", cleaned, re.I):
            return "\n".join(lines[:idx]).strip(), "\n".join(lines[idx + 1 :]).strip()
    return text.strip(), ""


def interesting_line(line: str) -> bool:
    if len(line) < 35:
        return False
    patterns = [
        r"\bNature\b|\bScience\b|\bCell\b|\bLancet\b|\bNEJM\b|\bJAMA\b",
        r"\bPubMed\b|\bPMID\b|\bDOI\b|10\.\d{4,9}/",
        r"\barXiv\b|\bOpenReview\b|\bNeurIPS\b|\bICLR\b|\bICML\b",
        r"\bagent\b|\bagents\b|\bLLM\b|\bfoundation model\b|\bself[-\s]?evol",
        r"\b[A-Z][A-Za-z0-9-]*(?:GPT|Agent|Voyager|Gene|Cell|Scientist|Bench|Eval)\b",
        r"\bbenchmark\b|\bevaluation\b|\bclinical\b|\bbiomedical\b|\bmedical\b",
    ]
    return any(re.search(pattern, line, re.I) for pattern in patterns)


def chunks_for_draft(path: Path, max_chars: int, overlap_lines: int) -> list[dict[str, str]]:
    text = read_text(path)
    body, refs = split_reference_section(text)
    lines = [clean_space(line) for line in body.splitlines() if clean_space(line)]
    if not lines:
        lines = [clean_space(part) for part in re.split(r"(?<=[.!?;])\s+", body) if clean_space(part)]
    chunks: list[dict[str, str]] = []
    buffer: list[str] = []
    chunk_no = 1
    for line in lines:
        if not buffer:
            buffer.append(line)
            continue
        if sum(len(item) + 1 for item in buffer) + len(line) <= max_chars:
            buffer.append(line)
            continue
        text_part = "\n".join(buffer)
        if any(interesting_line(item) for item in buffer):
            chunks.append({"draft": path.name, "path": str(path), "chunk_id": f"{safe_id(path.stem)}-chunk-{chunk_no:03d}", "text": text_part})
            chunk_no += 1
        buffer = buffer[-overlap_lines:] + [line] if overlap_lines > 0 else [line]
    if buffer and any(interesting_line(item) for item in buffer):
        chunks.append({"draft": path.name, "path": str(path), "chunk_id": f"{safe_id(path.stem)}-chunk-{chunk_no:03d}", "text": "\n".join(buffer)})
    if refs.strip():
        chunks.append({"draft": path.name, "path": str(path), "chunk_id": f"{safe_id(path.stem)}-references", "text": refs[:max_chars]})
    return chunks


def mining_prompt(chunk: dict[str, str], topic: str, max_candidates: int) -> str:
    schema = {
        "candidate_title": "best paper title or title fragment; blank only if impossible",
        "authors_or_group": "authors, lab, system authors, or group if visible",
        "year_hint": "year if visible",
        "venue_hint": "journal/conference clue such as Nature, Nature Biomedical Engineering, Nature Methods, ICLR, NeurIPS",
        "system_or_method": "named system, method, benchmark, dataset, agent, or model",
        "doi": "DOI if visible; do not invent",
        "pmid": "PMID if visible; do not invent",
        "urls": "semicolon-separated URLs if visible",
        "raw_evidence_excerpt": "short excerpt from the draft proving why you extracted it",
        "why_relevant": "which review argument this paper likely supports or challenges",
        "candidate_type": "explicit_reference | body_mention | named_system | method_family | missing_foundational_paper | possible_recent_landmark",
        "confidence": "high | medium | low",
        "pubmed_query": "precise PubMed query to find/verify this paper; include title/system/venue terms, not a broad topic query",
        "notes": "uncertainties or human questions",
    }
    return textwrap.dedent(
        f"""
        You are the Literature Candidate Mining Agent for a top-journal review.

        Review topic:
        {topic or "[not specified]"}

        Read the draft passage below. Mine candidate papers that the passage
        explicitly or implicitly depends on. Focus on:
        - body mentions of named systems/methods, not only reference-list entries;
        - author/year/venue/title fragments;
        - recent landmark papers in Nature-family, Science-family, Cell-family,
          PubMed-indexed journals, and official AI conference proceedings;
        - PubMed-verifiable biomedical/medical papers;
        - claims where the draft clearly needs a paper even if the reference is incomplete.

        Do NOT invent identifiers, titles, authors, or publication facts. If the
        passage only gives a system name or partial title, keep it as a candidate
        with a precise PubMed query. Prefer narrow queries like:
        ("system name"[Title/Abstract] OR "title fragment"[Title]) AND Nature[Journal]

        Return strict JSON only:
        {{"candidates": [ ... ]}}

        Each candidate must follow this schema:
        {json.dumps(schema, ensure_ascii=False, indent=2)}

        Return at most {max_candidates} candidates. If no paper clue exists, return
        {{"candidates": []}}.

        Draft: {chunk["draft"]}
        Chunk: {chunk["chunk_id"]}

        Passage:
        {chunk["text"]}
        """
    ).strip()


def call_chunk(item: tuple[int, dict[str, str], str], args: argparse.Namespace) -> tuple[int, str]:
    index, chunk, prompt = item
    prefix = re.sub(r"[^A-Za-z0-9_-]+", "-", args.user_id_prefix).strip("-")[:120]
    user_id = f"{prefix}-mine-{index:04d}" if prefix else ""
    return index, chat(prompt, args, user_id=user_id)


def normalize_candidate(raw: dict[str, Any], chunk: dict[str, str], seq: int) -> dict[str, str]:
    title = clean_space(raw.get("candidate_title"))
    system = clean_space(raw.get("system_or_method"))
    query = clean_space(raw.get("pubmed_query"))
    doi = normalize_doi(raw.get("doi"))
    urls = clean_space(raw.get("urls"))
    excerpt = clean_space(raw.get("raw_evidence_excerpt"))[:700]
    cid_base = "|".join([chunk["draft"], chunk["chunk_id"], title, system, doi, query, str(seq)])
    candidate_id = "AIC-" + hashlib.sha1(cid_base.encode("utf-8")).hexdigest()[:12]
    source_kind = "paper_identifier" if doi or raw.get("pmid") else "ai_mined_body_clue"
    priority = "high" if doi or title or raw.get("venue_hint") else "medium"
    cited = "yes" if re.search(r"\[[0-9,\-\s;]+\]|doi|pmid|https?://", excerpt, re.I) else "body_mention"
    return {
        "candidate_id": candidate_id,
        "draft": chunk["draft"],
        "chunk_id": chunk["chunk_id"],
        "claim_id": chunk["chunk_id"],
        "candidate_title": title or system or query[:180],
        "authors_or_group": clean_space(raw.get("authors_or_group")),
        "year_hint": clean_space(raw.get("year_hint")),
        "venue_hint": clean_space(raw.get("venue_hint")),
        "system_or_method": system,
        "doi": doi,
        "pmid": clean_space(raw.get("pmid")),
        "urls": urls,
        "raw_evidence_excerpt": excerpt,
        "why_relevant": clean_space(raw.get("why_relevant"))[:500],
        "candidate_type": clean_space(raw.get("candidate_type")) or "body_mention",
        "confidence": clean_space(raw.get("confidence")) or "medium",
        "source_kind": source_kind,
        "cited_in_body": cited,
        "verification_priority": priority,
        "pubmed_query": query or build_query(title, system, raw.get("venue_hint"), args_topic=""),
        "notes": clean_space(raw.get("notes")),
    }


def build_query(title: str, system: str, venue: Any, args_topic: str) -> str:
    parts: list[str] = []
    if title:
        parts.append(f'"{title[:120]}"[Title]')
    if system and system.lower() not in (title or "").lower():
        parts.append(f'"{system[:80]}"[Title/Abstract]')
    venue_text = clean_space(venue)
    if venue_text:
        parts.append(f'"{venue_text[:80]}"[Journal]')
    if not parts and args_topic:
        parts.append(f'"{args_topic[:120]}"[Title/Abstract]')
    return " AND ".join(parts[:3])


def candidate_key(row: dict[str, str]) -> str:
    doi = normalize_doi(row.get("doi"))
    if doi:
        return "doi:" + doi
    title = clean_space(row.get("candidate_title")).lower()
    system = clean_space(row.get("system_or_method")).lower()
    query = clean_space(row.get("pubmed_query")).lower()
    return "|".join([title, system, query])[:300]


def dedupe_candidates(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    order: list[str] = []
    rank = {"high": 3, "medium": 2, "low": 1}
    for row in rows:
        key = candidate_key(row)
        if not key:
            continue
        if key not in best:
            best[key] = row
            order.append(key)
            continue
        old = best[key]
        if rank.get(row.get("confidence", "").lower(), 0) > rank.get(old.get("confidence", "").lower(), 0):
            best[key] = row
    return [best[key] for key in order]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def coalesce(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = clean_space(row.get(key))
        if value:
            return value
    return ""


def parse_csv_text(text: str) -> list[dict[str, str]]:
    handle = io.StringIO(text.strip())
    try:
        rows = list(csv.DictReader(handle))
    except csv.Error:
        return []
    if not rows or not rows[0]:
        return []
    return [{str(k or "").strip(): clean_space(v) for k, v in row.items()} for row in rows if any(clean_space(v) for v in row.values())]


def parse_subagent_markdown(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, str]] = []
    for match in re.finditer(r"```csv\s*(.*?)```", text, flags=re.I | re.S):
        rows.extend(parse_csv_text(match.group(1)))
    return rows


def discover_subagent_inputs(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for raw in args.input:
        path = Path(raw).resolve()
        if path.exists() and path.is_file():
            paths.append(path)
    for raw in args.input_dir:
        root = Path(raw).resolve()
        if root.exists():
            paths.extend(p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in {".csv", ".md", ".txt"})
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def subagent_row_to_candidate(row: dict[str, Any], source_path: Path, seq: int) -> dict[str, str]:
    title_or_system = coalesce(row, "candidate_title_or_system", "candidate_title", "title", "system_or_method", "system")
    system = coalesce(row, "system_or_method", "system", "candidate_title_or_system")
    query = coalesce(row, "search_query", "pubmed_query", "query")
    context = coalesce(row, "claim_context", "raw_evidence_excerpt", "context", "why_it_is_likely_a_paper")
    packet = coalesce(row, "packet", "chunk_id") or safe_id(source_path.stem)
    draft = coalesce(row, "file_or_context", "draft", "source_file") or source_path.name
    venue_year = coalesce(row, "venue_year_if_any", "venue_hint", "year_hint")
    year_match = re.search(r"\b(19|20)\d{2}\b", venue_year)
    doi = normalize_doi(coalesce(row, "doi", "DOI"))
    pmid = coalesce(row, "pmid", "PMID")
    cid_base = "|".join([source_path.name, str(seq), title_or_system, system, query, context, doi, pmid])
    candidate_id = coalesce(row, "candidate_id") or "AIC-" + hashlib.sha1(cid_base.encode("utf-8")).hexdigest()[:12]
    confidence = coalesce(row, "confidence") or "medium"
    priority = coalesce(row, "verification_priority") or ("high" if confidence.lower() == "high" else "medium")
    return {
        "candidate_id": candidate_id,
        "draft": draft,
        "chunk_id": packet,
        "claim_id": packet,
        "candidate_title": title_or_system,
        "authors_or_group": coalesce(row, "authors_if_any", "authors_or_group", "authors"),
        "year_hint": year_match.group(0) if year_match else coalesce(row, "year_hint"),
        "venue_hint": venue_year,
        "system_or_method": system if system != title_or_system else "",
        "doi": doi,
        "pmid": pmid,
        "urls": coalesce(row, "urls", "url", "official_url"),
        "raw_evidence_excerpt": context[:700],
        "why_relevant": coalesce(row, "why_it_is_likely_a_paper", "why_relevant", "reason")[:500],
        "candidate_type": coalesce(row, "candidate_type") or "subagent_hidden_body_clue",
        "confidence": confidence,
        "source_kind": "codex_subagent_hidden_clue",
        "cited_in_body": coalesce(row, "cited_in_body") or "body_mention",
        "verification_priority": priority,
        "pubmed_query": query or build_query(title_or_system, system, venue_year, args_topic=""),
        "notes": coalesce(row, "notes"),
    }


def reference_identity_keys(reference_csv: str) -> set[tuple[str, str]]:
    if not reference_csv:
        return set()
    keys: set[tuple[str, str]] = set()
    for row in read_csv(Path(reference_csv)):
        title = clean_space(row.get("candidate_title") or row.get("title"))
        if title:
            keys.add(("title", normalize_title(title)))
        raw = clean_space(row.get("raw_reference"))
        if raw:
            for doi in re.findall(r"10\.\d{4,9}/[^\s,;)\]]+", raw, flags=re.I):
                keys.add(("doi", normalize_doi(doi)))
        for url in re.split(r";|\|", clean_space(row.get("urls") or row.get("url"))):
            url = url.strip().rstrip(".,)")
            if url:
                keys.add(("url", url.lower()))
        doi = normalize_doi(row.get("doi"))
        if doi:
            keys.add(("doi", doi))
        pmid = clean_space(row.get("pmid"))
        if pmid:
            keys.add(("pmid", pmid))
    return {item for item in keys if item[1]}


def mark_reference_overlap(row: dict[str, str], ref_keys: set[tuple[str, str]]) -> dict[str, str]:
    if not ref_keys:
        return row
    title_key = ("title", normalize_title(row.get("candidate_title", "")))
    doi_key = ("doi", normalize_doi(row.get("doi", "")))
    pmid_key = ("pmid", clean_space(row.get("pmid", "")))
    url_keys = {
        ("url", url.strip().rstrip(".,)").lower())
        for url in re.split(r";|\|", clean_space(row.get("urls")))
        if url.strip()
    }
    overlaps = [key for key in [title_key, doi_key, pmid_key] if key[1] and key in ref_keys]
    overlaps.extend(key for key in url_keys if key[1] and key in ref_keys)
    if not overlaps:
        return row
    row = dict(row)
    row["candidate_type"] = "explicit_reference_recovered_by_subagent"
    row["source_kind"] = "codex_subagent_reference_recovery"
    note = "Also found in draft-native reference assets; treat as reference recovery/identity normalization, not hidden-only."
    row["notes"] = "; ".join(part for part in [row.get("notes", ""), note] if part)
    return row


def verifier_rows_from_candidates(rows: list[dict[str, str]], source_kind: str = "codex_subagent_hidden_clue") -> list[dict[str, str]]:
    verify_rows: list[dict[str, str]] = []
    for row in rows:
        verify_rows.append(
            {
                "draft": row.get("draft", ""),
                "ref_number": row.get("candidate_id") or row.get("ref_number", ""),
                "candidate_id": row.get("candidate_id") or row.get("ref_number", ""),
                "claim_id": row.get("claim_id", ""),
                "candidate_title": row.get("candidate_title", ""),
                "raw_reference": row.get("raw_evidence_excerpt") or row.get("why_relevant") or row.get("notes", ""),
                "doi": row.get("doi", ""),
                "pmid": row.get("pmid", ""),
                "urls": row.get("urls", ""),
                "source_kind": row.get("source_kind") or source_kind,
                "cited_in_body": row.get("cited_in_body", "body_mention"),
                "verification_priority": row.get("verification_priority", "medium"),
                "candidate_type": row.get("candidate_type", ""),
                "pubmed_query": row.get("pubmed_query", ""),
                "raw_evidence_excerpt": row.get("raw_evidence_excerpt", ""),
                "why_relevant": row.get("why_relevant", ""),
            }
        )
    return verify_rows


def normalization_prompt(rows: list[dict[str, str]], topic: str, packet_no: int) -> str:
    compact = []
    for row in rows:
        compact.append(
            {
                "candidate_id": row.get("candidate_id", ""),
                "candidate_title": row.get("candidate_title", ""),
                "system_or_method": row.get("system_or_method", ""),
                "authors_or_group": row.get("authors_or_group", ""),
                "year_hint": row.get("year_hint", ""),
                "venue_hint": row.get("venue_hint", ""),
                "search_query": row.get("pubmed_query", ""),
                "claim_context": row.get("raw_evidence_excerpt", ""),
                "notes": row.get("notes", ""),
            }
        )
    schema = {
        "candidate_id": "copy from input",
        "official_title": "official paper title if confidently resolved; blank if unresolved",
        "official_authors": "authors if visible from input or verified search; do not invent",
        "official_year": "year if visible or verified; do not invent",
        "official_venue": "journal/conference if visible or verified; do not invent",
        "doi": "DOI if verified; do not invent",
        "pmid": "PMID if verified; do not invent",
        "official_url": "official publisher/conference/PubMed URL if verified",
        "normalization_status": "resolved | needs_web_search | ambiguous | preprint_only | non_paper | delete_or_replace",
        "normalization_confidence": "high | medium | low",
        "normalization_rationale": "short explanation of how system/informal title maps to official paper identity",
        "evidence_source": "input_text | PubMed | Crossref | OpenAlex | publisher_page | official_conference_page | web_search | human_needed",
        "notes": "remaining uncertainty or human question",
    }
    return textwrap.dedent(
        f"""
        You are the Paper Identity Normalization Agent for a top-journal review.

        Review topic:
        {topic or "[not specified]"}

        These candidates came from draft prose and may use informal system names
        rather than official paper titles. Normalize each candidate to a real paper
        identity before deterministic verification. Do not invent DOI, PMID,
        authors, venues, years, or URLs. If the input is ambiguous, mark it as
        `needs_web_search` or `ambiguous` and provide a precise query.

        Return strict CSV with exactly these columns:
        {",".join(NORMALIZED_IDENTITY_FIELDS)}

        Field schema:
        {json.dumps(schema, ensure_ascii=False, indent=2)}

        Packet: {packet_no}

        Candidates:
        {json.dumps(compact, ensure_ascii=False, indent=2)}
        """
    ).strip()


def cmd_collect(args: argparse.Namespace) -> int:
    inputs = discover_subagent_inputs(args)
    if not inputs:
        raise SystemExit("No subagent output files found.")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    raw_count = 0
    ref_keys = reference_identity_keys(args.reference_candidates_csv)
    for path in inputs:
        if path.suffix.lower() == ".csv":
            parsed = read_csv(path)
        else:
            parsed = parse_subagent_markdown(path)
        raw_count += len(parsed)
        for seq, item in enumerate(parsed, 1):
            rows.append(mark_reference_overlap(subagent_row_to_candidate(item, path, seq), ref_keys))
    rows = dedupe_candidates(rows)
    write_csv(out_dir / "subagent_hidden_candidate_clues.csv", rows, CANDIDATE_FIELDS)
    write_csv(out_dir / "candidate_paper_clues_for_verification.csv", verifier_rows_from_candidates(rows), VERIFY_FIELDS)
    tasks = [
        {
            "task_id": "AIN-" + safe_id(row.get("candidate_id", "")),
            "candidate_id": row.get("candidate_id", ""),
            "draft": row.get("draft", ""),
            "claim_id": row.get("claim_id", ""),
            "candidate_title": row.get("candidate_title", ""),
            "system_or_method": row.get("system_or_method", ""),
            "query": row.get("pubmed_query", ""),
            "reason": row.get("why_relevant", ""),
            "priority": row.get("verification_priority", ""),
            "status": "needs_identity_normalization",
        }
        for row in rows
    ]
    write_csv(out_dir / "identity_normalization_tasks.csv", tasks, QUERY_FIELDS)
    lines = [
        "# Human Checkpoint: Codex-Subtask Hidden Literature Candidates",
        "",
        f"- Inputs: {len(inputs)}",
        f"- Raw rows parsed: {raw_count}",
        f"- Deduplicated candidates: {len(rows)}",
        f"- Reference-asset overlap enabled: {bool(ref_keys)}",
        "",
        "Next step: run `normalize-packet`, let Codex subtasks normalize informal system names to official paper identities, then run `merge-normalized` and deterministic verification.",
        "",
        "| Candidate | Confidence | Title/System | Venue hint | Query | Decision |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows[:150]:
        lines.append(
            "| "
            + " | ".join(
                clean_space(value).replace("|", "\\|")
                for value in [
                    row.get("candidate_id", ""),
                    row.get("confidence", ""),
                    row.get("candidate_title") or row.get("system_or_method", ""),
                    row.get("venue_hint", ""),
                    row.get("pubmed_query", ""),
                    "normalize/verify/delete/ask-user",
                ]
            )
            + " |"
        )
    (out_dir / "human_hidden_candidate_checkpoint.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "inputs": len(inputs), "raw_rows": raw_count, "candidates": len(rows)}, ensure_ascii=False))
    return 0


def cmd_normalize_packet(args: argparse.Namespace) -> int:
    rows = read_csv(Path(args.candidate_csv))
    if not rows:
        raise SystemExit(f"No candidates found: {args.candidate_csv}")
    rows = [{field: clean_space(row.get(field)) for field in CANDIDATE_FIELDS} for row in rows]
    rows = rows[: args.max_candidates if args.max_candidates > 0 else None]
    out_dir = Path(args.out_dir).resolve()
    packet_dir = out_dir / "identity_normalization_packets"
    packet_dir.mkdir(parents=True, exist_ok=True)
    batch = max(1, args.batch_size)
    packet_count = 0
    for start in range(0, len(rows), batch):
        packet_count += 1
        packet = normalization_prompt(rows[start : start + batch], args.topic, packet_count)
        (packet_dir / f"identity_normalization_packet_{packet_count:04d}.md").write_text(packet + "\n", encoding="utf-8")
    lines = [
        "# Paper Identity Normalization Packet Index",
        "",
        f"- Generated at: {now_iso()}",
        f"- Candidates: {len(rows)}",
        f"- Packets: {packet_count}",
        "",
        "Codex subtasks should fill a normalized CSV with official titles/DOIs/URLs when confident, and mark unresolved rows as `needs_web_search`, `ambiguous`, `preprint_only`, or `delete_or_replace`.",
        "",
    ]
    for idx in range(1, packet_count + 1):
        lines.append(f"- `identity_normalization_packets/identity_normalization_packet_{idx:04d}.md`")
    (out_dir / "identity_normalization_packet_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "candidates": len(rows), "packets": packet_count}, ensure_ascii=False))
    return 0


def cmd_merge_normalized(args: argparse.Namespace) -> int:
    candidates = {row.get("candidate_id", ""): row for row in read_csv(Path(args.candidate_csv))}
    normalized_rows = read_csv(Path(args.normalized_csv))
    if not normalized_rows:
        raise SystemExit(f"No normalized rows found: {args.normalized_csv}")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged: list[dict[str, str]] = []
    unresolved: list[dict[str, str]] = []
    for raw in normalized_rows:
        candidate_id = coalesce(raw, "candidate_id", "ref_number", "id")
        source = candidates.get(candidate_id, {})
        official_title = coalesce(raw, "official_title", "title")
        if not official_title and normalize_doi(raw.get("doi")):
            official_title = coalesce(raw, "candidate_title", "candidate_title_or_system")
        status = coalesce(raw, "normalization_status", "status") or ("resolved" if official_title or raw.get("doi") or raw.get("pmid") else "needs_web_search")
        row = {
            "candidate_id": candidate_id or source.get("candidate_id", ""),
            "draft": source.get("draft") or raw.get("draft", ""),
            "chunk_id": source.get("chunk_id") or raw.get("chunk_id", ""),
            "claim_id": source.get("claim_id") or raw.get("chunk_id", ""),
            "candidate_title": official_title or source.get("candidate_title") or coalesce(raw, "candidate_title", "candidate_title_or_system"),
            "authors_or_group": coalesce(raw, "official_authors", "authors") or source.get("authors_or_group", ""),
            "year_hint": coalesce(raw, "official_year", "year") or source.get("year_hint", ""),
            "venue_hint": coalesce(raw, "official_venue", "venue", "journal") or source.get("venue_hint", ""),
            "system_or_method": source.get("system_or_method") or raw.get("system_or_method", ""),
            "doi": normalize_doi(raw.get("doi")),
            "pmid": clean_space(raw.get("pmid")),
            "urls": coalesce(raw, "official_url", "url", "urls") or source.get("urls", ""),
            "raw_evidence_excerpt": source.get("raw_evidence_excerpt") or coalesce(raw, "raw_evidence_excerpt", "claim_context"),
            "why_relevant": source.get("why_relevant") or coalesce(raw, "normalization_rationale", "claim_context"),
            "candidate_type": source.get("candidate_type") or "identity_normalized_hidden_clue",
            "confidence": coalesce(raw, "normalization_confidence", "confidence") or source.get("confidence", "medium"),
            "source_kind": "identity_normalized_hidden_clue",
            "cited_in_body": source.get("cited_in_body", "body_mention"),
            "verification_priority": source.get("verification_priority", "high"),
            "pubmed_query": source.get("pubmed_query") or build_query(official_title, source.get("system_or_method", ""), raw.get("official_venue"), args_topic=args.topic),
            "notes": "; ".join(part for part in [source.get("notes", ""), coalesce(raw, "normalization_rationale", "notes")] if part),
        }
        if status in {"resolved", "preprint_only"} and (row["candidate_title"] or row["doi"] or row["pmid"] or row["urls"]):
            merged.append(row)
        else:
            row["notes"] = (row.get("notes", "") + f"; normalization_status={status}").strip("; ")
            unresolved.append(row)
    merged = dedupe_candidates(merged)
    write_csv(out_dir / "normalized_identity_candidates.csv", merged, CANDIDATE_FIELDS)
    write_csv(out_dir / "candidate_paper_clues_for_verification.csv", verifier_rows_from_candidates(merged, source_kind="identity_normalized_hidden_clue"), VERIFY_FIELDS)
    write_csv(out_dir / "unresolved_identity_candidates.csv", unresolved, CANDIDATE_FIELDS)
    report = [
        "# Paper Identity Normalization Merge",
        "",
        f"- Generated at: {now_iso()}",
        f"- Normalized input rows: {len(normalized_rows)}",
        f"- Verification-ready rows: {len(merged)}",
        f"- Unresolved rows: {len(unresolved)}",
        "",
        "Run deterministic verification next on `candidate_paper_clues_for_verification.csv`.",
        "",
    ]
    (out_dir / "identity_normalization_merge_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "verification_ready": len(merged), "unresolved": len(unresolved)}, ensure_ascii=False))
    return 0


def write_packets(out_dir: Path, chunks: list[dict[str, str]], prompts: list[str]) -> None:
    packet_dir = out_dir / "packets"
    packet_dir.mkdir(parents=True, exist_ok=True)
    for idx, prompt in enumerate(prompts, 1):
        (packet_dir / f"candidate_mining_packet_{idx:04d}.md").write_text(prompt + "\n", encoding="utf-8")
    with (out_dir / "candidate_mining_chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps({k: v for k, v in chunk.items() if k != "text"}, ensure_ascii=False) + "\n")
    lines = [
        "# AI Literature Candidate Mining Packets",
        "",
        f"- Generated at: {now_iso()}",
        f"- Chunks: {len(chunks)}",
        f"- Packet files: {len(prompts)}",
        "",
        "Use these packets with Codex subtasks before deterministic verification. The miner extracts candidate clues; it does not certify papers. External LLM review is optional only after approval.",
        "",
    ]
    for idx in range(1, len(prompts) + 1):
        lines.append(f"- `packets/candidate_mining_packet_{idx:04d}.md`")
    (out_dir / "candidate_mining_packet_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(out_dir: Path, rows: list[dict[str, str]], raw_outputs: list[str], args: argparse.Namespace) -> None:
    rows = dedupe_candidates(rows)
    write_csv(out_dir / "ai_candidate_paper_clues.csv", rows, CANDIDATE_FIELDS)
    verify_rows: list[dict[str, str]] = []
    tasks: list[dict[str, str]] = []
    seen_queries: set[str] = set()
    for row in rows:
        verify_rows.append(
            {
                "draft": row["draft"],
                "ref_number": row["candidate_id"],
                "candidate_id": row["candidate_id"],
                "claim_id": row.get("claim_id", ""),
                "candidate_title": row["candidate_title"],
                "raw_reference": row["raw_evidence_excerpt"] or row["why_relevant"],
                "doi": row["doi"],
                "pmid": row["pmid"],
                "urls": row["urls"],
                "source_kind": row["source_kind"],
                "cited_in_body": row["cited_in_body"],
                "verification_priority": row["verification_priority"],
                "candidate_type": row.get("candidate_type", ""),
                "pubmed_query": row.get("pubmed_query", ""),
                "raw_evidence_excerpt": row.get("raw_evidence_excerpt", ""),
                "why_relevant": row.get("why_relevant", ""),
            }
        )
        query = clean_space(row.get("pubmed_query"))
        if query and query.lower() not in seen_queries:
            seen_queries.add(query.lower())
            tasks.append(
                {
                    "task_id": "AIP-" + safe_id(row["candidate_id"]),
                    "candidate_id": row["candidate_id"],
                    "draft": row["draft"],
                    "claim_id": row["claim_id"],
                    "candidate_title": row["candidate_title"],
                    "system_or_method": row["system_or_method"],
                    "query": query,
                    "reason": row["why_relevant"],
                    "priority": row["verification_priority"],
                    "status": "ready_for_pubmed_deep_dive",
                }
            )
    write_csv(out_dir / "candidate_paper_clues_for_verification.csv", verify_rows, VERIFY_FIELDS)
    write_csv(out_dir / "ai_pubmed_deep_dive_tasks.csv", tasks, QUERY_FIELDS)
    with (out_dir / "raw_ai_candidate_mining_outputs.jsonl").open("w", encoding="utf-8") as handle:
        for raw in raw_outputs:
            handle.write(json.dumps({"raw": raw}, ensure_ascii=False) + "\n")
    lines = [
        "# Human Checkpoint: AI-Mined Literature Candidates",
        "",
        "These rows came from AI reading of draft prose. Confirm which should be verified, searched in PubMed, merged, or deleted.",
        "",
        f"- Candidate clues: {len(rows)}",
        f"- PubMed deep-dive tasks: {len(tasks)}",
        f"- Deterministic verifier input: `candidate_paper_clues_for_verification.csv`",
        "",
        "| Candidate | Confidence | Title/System | Venue hint | PubMed query | Decision |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows[:120]:
        title = row.get("candidate_title") or row.get("system_or_method")
        query = row.get("pubmed_query")
        lines.append(
            "| "
            + " | ".join(
                clean_space(value).replace("|", "\\|")
                for value in [row["candidate_id"], row["confidence"], title, row["venue_hint"], query, "verify/search/delete/merge"]
            )
            + " |"
        )
    (out_dir / "human_ai_candidate_checkpoint.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "generated_at": now_iso(),
        "mode": args.command,
        "candidates": len(rows),
        "pubmed_tasks": len(tasks),
        "out_dir": str(out_dir),
    }
    (out_dir / "ai_candidate_mining_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cmd_packet(args: argparse.Namespace) -> int:
    drafts = discover_drafts(args)
    if not drafts:
        raise SystemExit("No supported drafts found.")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[dict[str, str]] = []
    for path in drafts:
        chunks.extend(chunks_for_draft(path, args.chunk_chars, args.overlap_lines))
    chunks = chunks[: args.max_chunks] if args.max_chunks > 0 else chunks
    prompts = [mining_prompt(chunk, args.topic, args.max_candidates_per_chunk) for chunk in chunks]
    write_packets(out_dir, chunks, prompts)
    print(json.dumps({"out_dir": str(out_dir), "chunks": len(chunks), "packets": len(prompts)}, ensure_ascii=False))
    return 0


def cmd_mine(args: argparse.Namespace) -> int:
    drafts = discover_drafts(args)
    if not drafts:
        raise SystemExit("No supported drafts found.")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[dict[str, str]] = []
    for path in drafts:
        chunks.extend(chunks_for_draft(path, args.chunk_chars, args.overlap_lines))
    chunks = chunks[: args.max_chunks] if args.max_chunks > 0 else chunks
    prompts = [mining_prompt(chunk, args.topic, args.max_candidates_per_chunk) for chunk in chunks]
    write_packets(out_dir, chunks, prompts)
    raw_by_index: dict[int, str] = {}
    items = list(enumerate(zip(chunks, prompts), 1))
    jobs = [(idx, chunk, prompt) for idx, (chunk, prompt) in items]
    if args.max_workers > 1 and len(jobs) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_map = {executor.submit(call_chunk, job, args): job[0] for job in jobs}
            for future in concurrent.futures.as_completed(future_map):
                idx, raw = future.result()
                raw_by_index[idx] = raw
    else:
        for job in jobs:
            idx, raw = call_chunk(job, args)
            raw_by_index[idx] = raw
    rows: list[dict[str, str]] = []
    raw_outputs: list[str] = []
    for idx in sorted(raw_by_index):
        raw = raw_by_index[idx]
        raw_outputs.append(raw)
        chunk = chunks[idx - 1]
        for seq, item in enumerate(coerce_json_results(raw), 1):
            rows.append(normalize_candidate(item, chunk, seq))
    write_outputs(out_dir, rows, raw_outputs, args)
    print(json.dumps({"out_dir": str(out_dir), "chunks": len(chunks), "candidates": len(rows)}, ensure_ascii=False))
    return 0


def cmd_pubmed(args: argparse.Namespace) -> int:
    tasks = read_csv(Path(args.tasks_csv))
    if not tasks:
        raise SystemExit(f"No tasks found: {args.tasks_csv}")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    run_log: list[dict[str, Any]] = []
    seen_pmids: set[str] = set()
    for task in tasks[: args.max_tasks if args.max_tasks > 0 else None]:
        query = clean_space(task.get("query"))
        if not query:
            continue
        try:
            ids, search_url = pubmed_recall.esearch(query, args)
            time.sleep(args.sleep)
            papers, fetch_url = pubmed_recall.efetch(ids, args)
            run_log.append({"task_id": task.get("task_id"), "query": query, "pmids": ids, "search_url": search_url, "fetch_url": fetch_url})
            for paper in papers:
                pmid = paper.get("pmid") or paper.get("paper_id")
                if pmid and pmid in seen_pmids:
                    continue
                if pmid:
                    seen_pmids.add(pmid)
                rows.append(
                    {
                        "task_id": task.get("task_id", ""),
                        "candidate_id": task.get("candidate_id", ""),
                        "draft": task.get("draft", ""),
                        "claim_id": task.get("claim_id", ""),
                        "candidate_title": task.get("candidate_title", ""),
                        "query": query,
                        "reason": task.get("reason", ""),
                        "priority": task.get("priority", ""),
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
                        "abstract": paper.get("abstract", ""),
                        "publication_type": paper.get("publication_type", ""),
                        "candidate_fit_status": "needs_llm_or_human_screening",
                        "candidate_fit_reason": "PubMed abstract returned from AI-mined draft clue; verify title and claim fit before promotion.",
                        "notes": "Do not cite from abstract alone for detailed methods/results claims.",
                    }
                )
            time.sleep(args.sleep)
        except Exception as exc:
            run_log.append({"task_id": task.get("task_id"), "query": query, "error": repr(exc)})
    write_csv(out_dir / "ai_pubmed_candidate_abstracts.csv", rows, PUBMED_FIELDS)
    with (out_dir / "ai_pubmed_deep_dive_log.json").open("w", encoding="utf-8") as handle:
        json.dump({"generated_at": now_iso(), "tasks": len(tasks), "rows": len(rows), "calls": run_log}, handle, ensure_ascii=False, indent=2)
    lines = [
        "# AI-Mined PubMed Abstract Packet",
        "",
        "Screen these PubMed hits against the AI-mined draft clue before verification/promotion.",
        "",
    ]
    for row in rows[:120]:
        lines.extend(
            [
                f"## {row.get('title') or row.get('pmid')}",
                "",
                f"- Task/candidate: `{row.get('task_id')}` / `{row.get('candidate_id')}`",
                f"- PMID/DOI: {row.get('pmid') or 'missing'} / {row.get('doi') or 'missing'}",
                f"- Journal/year: {row.get('journal') or 'unknown'} ({row.get('year') or 'unknown'})",
                f"- Query: `{row.get('query')}`",
                "",
                clean_space(row.get("abstract") or "[No abstract returned]")[:1800],
                "",
            ]
        )
    (out_dir / "ai_pubmed_abstract_screening_packet.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "rows": len(rows)}, ensure_ascii=False))
    return 0


def cmd_pubmed_to_verifier(args: argparse.Namespace) -> int:
    abstract_rows = read_csv(Path(args.abstracts_csv))
    if not abstract_rows:
        raise SystemExit(f"No PubMed abstract rows found: {args.abstracts_csv}")
    task_rows = read_csv(Path(args.tasks_csv)) if args.tasks_csv else []
    tasks_by_id = {clean_space(row.get("task_id")): row for row in task_rows if clean_space(row.get("task_id"))}
    tasks_by_candidate = {clean_space(row.get("candidate_id")): row for row in task_rows if clean_space(row.get("candidate_id"))}
    allowed = {item.strip().lower() for item in args.include_status.split(",") if item.strip()}
    rejected_markers = {"reject", "rejected", "exclude", "excluded", "delete", "delete_or_replace", "not_relevant"}
    verifier: list[dict[str, str]] = []
    evidence_rows: list[dict[str, str]] = []
    for idx, row in enumerate(abstract_rows, 1):
        fit_status = clean_space(row.get("candidate_fit_status")).lower()
        if allowed and fit_status and fit_status not in allowed:
            continue
        if not allowed and any(marker in fit_status for marker in rejected_markers):
            continue
        task = tasks_by_id.get(clean_space(row.get("task_id")), {})
        if not task:
            task = tasks_by_candidate.get(clean_space(row.get("candidate_id")), {})
        pmid = clean_space(row.get("pmid") or row.get("paper_id"))
        title = coalesce(row, "title", "candidate_title") or coalesce(task, "candidate_title")
        if not (title or pmid or normalize_doi(row.get("doi"))):
            continue
        candidate_id = clean_space(row.get("candidate_id")) or clean_space(task.get("candidate_id"))
        if pmid:
            candidate_id = f"{candidate_id or 'PMD'}-pmid-{pmid}"
        else:
            candidate_id = candidate_id or "PMD-" + hashlib.sha1("|".join([title, clean_space(row.get("query")), str(idx)]).encode("utf-8")).hexdigest()[:12]
        abstract = clean_space(row.get("abstract"))
        reason = clean_space(row.get("reason") or task.get("reason") or row.get("candidate_fit_reason"))
        query = clean_space(row.get("query") or task.get("query"))
        verifier_row = {
            "draft": clean_space(row.get("draft") or task.get("draft") or "pubmed_deep_dive"),
            "ref_number": candidate_id,
            "candidate_id": candidate_id,
            "claim_id": clean_space(row.get("claim_id") or task.get("claim_id")),
            "candidate_title": title,
            "raw_reference": f"PubMed abstract candidate from query: {query}. PMID: {pmid}. {abstract[:700]}",
            "doi": normalize_doi(row.get("doi")),
            "pmid": pmid,
            "urls": clean_space(row.get("url")),
            "source_kind": "pubmed_abstract_candidate",
            "cited_in_body": "api_deep_dive",
            "verification_priority": clean_space(row.get("priority") or task.get("priority") or "medium"),
            "candidate_type": "pubmed_deep_dive_candidate",
            "pubmed_query": query,
            "raw_evidence_excerpt": abstract[:900],
            "why_relevant": reason[:600],
        }
        verifier.append(verifier_row)
        evidence_rows.append(
            {
                **verifier_row,
                "authors": clean_space(row.get("authors")),
                "year": clean_space(row.get("year")),
                "journal": clean_space(row.get("journal")),
                "publication_type": clean_space(row.get("publication_type")),
                "candidate_fit_status": clean_space(row.get("candidate_fit_status")),
                "candidate_fit_reason": clean_space(row.get("candidate_fit_reason")),
            }
        )
    verifier = verifier_rows_from_candidates(
        [
            {
                "candidate_id": row["candidate_id"],
                "draft": row["draft"],
                "claim_id": row["claim_id"],
                "candidate_title": row["candidate_title"],
                "raw_evidence_excerpt": row["raw_evidence_excerpt"],
                "why_relevant": row["why_relevant"],
                "doi": row["doi"],
                "pmid": row["pmid"],
                "urls": row["urls"],
                "source_kind": row["source_kind"],
                "cited_in_body": row["cited_in_body"],
                "verification_priority": row["verification_priority"],
                "candidate_type": row["candidate_type"],
                "pubmed_query": row["pubmed_query"],
            }
            for row in verifier
        ],
        source_kind="pubmed_abstract_candidate",
    )
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "pubmed_abstract_candidates_for_verification.csv", verifier, VERIFY_FIELDS)
    write_csv(
        out_dir / "pubmed_abstract_candidates_with_context.csv",
        evidence_rows,
        VERIFY_FIELDS + ["authors", "year", "journal", "publication_type", "candidate_fit_status", "candidate_fit_reason"],
    )
    lines = [
        "# PubMed Abstract Candidates Converted To Verifier Input",
        "",
        f"- Generated at: {now_iso()}",
        f"- Abstract rows read: {len(abstract_rows)}",
        f"- Verifier rows written: {len(verifier)}",
        "",
        "These rows are still candidates. Run deterministic verification and claim-fit adjudication before import.",
        "",
    ]
    for row in verifier[:120]:
        lines.append(f"- `{row.get('candidate_id')}` {row.get('candidate_title')} | PMID {row.get('pmid') or 'missing'} | query `{row.get('pubmed_query')}`")
    (out_dir / "pubmed_to_verifier_report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "abstract_rows": len(abstract_rows), "verifier_rows": len(verifier)}, ensure_ascii=False))
    return 0


def add_draft_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--draft", action="append", default=[], help="Draft file path; repeatable.")
    parser.add_argument("--draft-dir", action="append", default=[], help="Directory containing .md/.txt/.docx drafts; repeatable.")
    parser.add_argument("--topic", default="")
    parser.add_argument("--out-dir", default="./review-data/02_literature/ai_candidate_mining")
    parser.add_argument("--chunk-chars", type=int, default=5500)
    parser.add_argument("--overlap-lines", type=int, default=3)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--max-candidates-per-chunk", type=int, default=8)


def add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--env-file", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking", default="enabled", choices=["enabled", "disabled", "omit"])
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--user-id-prefix", default="topjournal-review")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine draft prose for AI-assisted literature candidates and PubMed deep-dive queries.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_packet = sub.add_parser("packet", help="Create Codex/DeepSeek mining packets without calling an external model.")
    add_draft_args(p_packet)
    p_packet.set_defaults(func=cmd_packet)

    p_mine = sub.add_parser("mine", help="Call DeepSeek/OpenAI-compatible model to mine draft literature candidates.")
    add_draft_args(p_mine)
    add_llm_args(p_mine)
    p_mine.set_defaults(func=cmd_mine)

    p_pubmed = sub.add_parser("pubmed", help="Run PubMed deep-dive searches from AI-mined candidate queries.")
    p_pubmed.add_argument("--tasks-csv", default="./review-data/02_literature/ai_candidate_mining/ai_pubmed_deep_dive_tasks.csv")
    p_pubmed.add_argument("--out-dir", default="./review-data/02_literature/ai_candidate_mining/pubmed_deep_dive")
    p_pubmed.add_argument("--max-results", type=int, default=5)
    p_pubmed.add_argument("--max-tasks", type=int, default=0)
    p_pubmed.add_argument("--email", default="")
    p_pubmed.add_argument("--api-key", default="")
    p_pubmed.add_argument("--tool", default="top-journal-review-writer")
    p_pubmed.add_argument("--timeout", type=int, default=30)
    p_pubmed.add_argument("--sleep", type=float, default=0.34)
    p_pubmed.add_argument("--user-agent", default="top-journal-review-writer/1.0")
    p_pubmed.add_argument("--mindate", default="")
    p_pubmed.add_argument("--maxdate", default="")
    p_pubmed.add_argument("--sort", default="relevance", choices=["relevance", "pub date", "first author", "journal"])
    p_pubmed.add_argument("--dry-run", action="store_true")
    p_pubmed.set_defaults(func=cmd_pubmed)

    p_pubmed_verifier = sub.add_parser("pubmed-to-verifier", help="Convert PubMed deep-dive abstract hits into verifier-ready candidate rows after screening.")
    p_pubmed_verifier.add_argument("--abstracts-csv", default="./review-data/02_literature/ai_candidate_mining/pubmed_deep_dive/ai_pubmed_candidate_abstracts.csv")
    p_pubmed_verifier.add_argument("--tasks-csv", default="./review-data/02_literature/ai_candidate_mining/ai_pubmed_deep_dive_tasks.csv")
    p_pubmed_verifier.add_argument("--out-dir", default="./review-data/02_literature/ai_candidate_mining/pubmed_deep_dive/verifier_input")
    p_pubmed_verifier.add_argument("--include-status", default="", help="Comma-separated candidate_fit_status values to include. Blank includes all non-rejected rows.")
    p_pubmed_verifier.set_defaults(func=cmd_pubmed_to_verifier)

    p_collect = sub.add_parser("collect", help="Collect Codex-subtask hidden candidate CSV/markdown outputs into governed candidate files.")
    p_collect.add_argument("--input", action="append", default=[], help="Subagent output .csv/.md/.txt file; repeatable.")
    p_collect.add_argument("--input-dir", action="append", default=[], help="Directory containing subagent outputs; repeatable.")
    p_collect.add_argument("--out-dir", default="./review-data/02_literature/ai_candidate_mining/subagent_collected")
    p_collect.add_argument("--reference-candidates-csv", default="", help="Optional draft-native candidate_paper_clues.csv for provenance overlap marking.")
    p_collect.set_defaults(func=cmd_collect)

    p_norm = sub.add_parser("normalize-packet", help="Create Codex-subtask packets to normalize informal system names into official paper identities.")
    p_norm.add_argument("--candidate-csv", default="./review-data/02_literature/ai_candidate_mining/subagent_collected/subagent_hidden_candidate_clues.csv")
    p_norm.add_argument("--out-dir", default="./review-data/02_literature/ai_candidate_mining/identity_normalization")
    p_norm.add_argument("--topic", default="")
    p_norm.add_argument("--batch-size", type=int, default=12)
    p_norm.add_argument("--max-candidates", type=int, default=0)
    p_norm.set_defaults(func=cmd_normalize_packet)

    p_merge = sub.add_parser("merge-normalized", help="Merge Codex-normalized paper identities into verifier-ready candidate rows.")
    p_merge.add_argument("--candidate-csv", default="./review-data/02_literature/ai_candidate_mining/subagent_collected/subagent_hidden_candidate_clues.csv")
    p_merge.add_argument("--normalized-csv", required=True)
    p_merge.add_argument("--out-dir", default="./review-data/02_literature/ai_candidate_mining/identity_normalization/merged")
    p_merge.add_argument("--topic", default="")
    p_merge.set_defaults(func=cmd_merge_normalized)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
