#!/usr/bin/env python3
"""Create and collect Codex-subagent literature discovery work.

This script is the reasoning-first layer before deterministic verification. It
does not certify papers. It forces the workflow to start from full-draft reading,
argument roles, and bounded API search tasks instead of only parsing reference
lists.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import re
import textwrap
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


SUPPORTED_EXTENSIONS = {".md", ".txt", ".docx"}

DISCOVERY_FIELDS = [
    "packet",
    "draft",
    "claim_or_argument",
    "claim_role",
    "time_role",
    "candidate_lane",
    "candidate_title_or_system",
    "authors_if_any",
    "venue_year_if_any",
    "known_identifier",
    "url",
    "why_it_is_likely_a_paper",
    "evidence_need",
    "api_route",
    "pubmed_query",
    "crossref_query",
    "arxiv_query",
    "openreview_query",
    "confidence",
    "notes",
]

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

API_TASK_FIELDS = [
    "task_id",
    "candidate_id",
    "packet",
    "draft",
    "claim_or_argument",
    "candidate_title_or_system",
    "api_route",
    "query",
    "evidence_need",
    "candidate_lane",
    "priority",
    "status",
    "notes",
]


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


def safe_id(value: str, fallback: str = "item") -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", clean_space(value) or fallback).strip("-")
    return text[:90] or fallback


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


def split_reference_section(text: str) -> tuple[str, str]:
    patterns = [
        r"(?im)^\s*#{0,3}\s*(references|bibliography|参考文献|參考文獻)\s*$",
        r"(?im)^\s*\*\*(references|bibliography|参考文献|參考文獻)\*\*\s*$",
    ]
    starts = [m.start() for pattern in patterns for m in re.finditer(pattern, text)]
    if not starts:
        return text, ""
    start = min(starts)
    return text[:start].strip(), text[start:].strip()


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


def chunk_body(path: Path, max_chars: int, overlap_lines: int) -> list[dict[str, str]]:
    text = read_text(path)
    body, refs = split_reference_section(text)
    lines = [clean_space(line) for line in body.splitlines() if clean_space(line)]
    if not lines:
        lines = [clean_space(part) for part in re.split(r"(?<=[.!?。！？；;])\s+", body) if clean_space(part)]
    chunks: list[dict[str, str]] = []
    buffer: list[str] = []
    chunk_no = 1
    for line in lines:
        if sum(len(item) + 1 for item in buffer) + len(line) <= max_chars:
            buffer.append(line)
            continue
        if buffer:
            chunks.append({"draft": path.name, "path": str(path), "packet": f"{safe_id(path.stem)}-body-{chunk_no:03d}", "text": "\n".join(buffer)})
            chunk_no += 1
        buffer = buffer[-overlap_lines:] + [line] if overlap_lines > 0 else [line]
    if buffer:
        chunks.append({"draft": path.name, "path": str(path), "packet": f"{safe_id(path.stem)}-body-{chunk_no:03d}", "text": "\n".join(buffer)})
    if refs:
        chunks.append({"draft": path.name, "path": str(path), "packet": f"{safe_id(path.stem)}-references", "text": refs[:max_chars]})
    return chunks


def load_asset_summary(draft_assets_dir: str) -> dict[str, Any]:
    root = Path(draft_assets_dir).resolve() if draft_assets_dir else Path()
    if not root.exists():
        return {"root": "", "candidate_titles": [], "claims": [], "health": ""}
    candidates = read_csv(root / "candidate_paper_clues.csv")
    claims = read_csv(root / "claim_evidence_map.csv")
    health_path = root / "draft_citation_health.md"
    health = health_path.read_text(encoding="utf-8", errors="replace")[:4000] if health_path.exists() else ""
    return {
        "root": str(root),
        "candidate_titles": [clean_space(row.get("candidate_title")) for row in candidates if clean_space(row.get("candidate_title"))][:120],
        "claims": [clean_space(row.get("claim") or row.get("claim_text") or row.get("original_text")) for row in claims if clean_space(row.get("claim") or row.get("claim_text") or row.get("original_text"))][:80],
        "health": health,
    }


def chief_packet(drafts: list[Path], asset_summary: dict[str, Any], topic: str) -> str:
    draft_list = "\n".join(f"- {path.name} ({path.stat().st_size} bytes)" for path in drafts)
    candidates = "\n".join(f"- {title}" for title in asset_summary.get("candidate_titles", [])[:80])
    claims = "\n".join(f"- {claim[:260]}" for claim in asset_summary.get("claims", [])[:50])
    return textwrap.dedent(
        f"""
        # Chief Codex Literature And Argument Reading Packet

        Topic: {topic or "[not specified]"}

        ## Mandatory Chief-Editor Work

        Read all draft memory cards/raw drafts before accepting subagent outputs. Produce a
        `chief_literature_brief.md` in the project that contains:

        1. the review's provisional central thesis;
        2. 3-5 section-spine candidates, not 8-9 fragmented headings;
        3. the past-present-future argument arc;
        4. claim clusters that need evidence;
        5. must-have seminal papers, current landmark papers, and future/governance papers;
        6. which existing draft references are explicit, polluted, missing, or identity-recovery cases;
        7. bounded API search tasks for literature agents.

        Literature agents may suggest papers and API queries, but they do not decide the
        manuscript's center of gravity. The chief Codex conversation owns that decision.

        ## Drafts To Read

        {draft_list or "[none]"}

        ## Existing Draft-Native Candidate Titles

        {candidates or "[candidate_paper_clues.csv not available]"}

        ## Claim Snippets From Draft Assets

        {claims or "[claim_evidence_map.csv not available]"}

        ## Citation Health Excerpt

        {asset_summary.get("health") or "[draft_citation_health.md not available]"}
        """
    ).strip() + "\n"


def subagent_prompt(chunk: dict[str, str], topic: str, asset_summary: dict[str, Any]) -> str:
    known = "; ".join(asset_summary.get("candidate_titles", [])[:40])
    return textwrap.dedent(
        f"""
        # Literature Discovery Subagent Packet

        Topic: {topic or "[not specified]"}
        Draft: {chunk["draft"]}
        Packet: {chunk["packet"]}

        You are not verifying papers. You are reading draft prose to recover
        literature needs for a top-journal review.

        Tasks:
        - Extract candidate papers, named systems, benchmarks, methods, datasets, and author/year/venue clues.
        - Identify missing seminal/current papers implied by the argument.
        - Mark whether a candidate is already in draft-native assets, a reference-recovery case, a hidden body candidate, a claim-gap candidate, a seminal background need, or counterevidence.
        - Provide exact API queries. Use PubMed for biomedical/medical papers, Crossref/OpenAlex for formal DOI/title metadata, arXiv/OpenReview for preprint or conference leads, and publisher/human when needed.
        - Do not invent DOI, PMID, titles, authors, or venues. If only a system name is visible, keep it as a system/title fragment and give a precise query.

        Known draft-native candidate titles for overlap awareness:
        {known or "[none supplied]"}

        Return one CSV fenced block with exactly these columns:
        {",".join(DISCOVERY_FIELDS)}

        Allowed `candidate_lane` values:
        explicit_reference | reference_recovery | hidden_body_candidate | claim_gap_candidate | seminal_background | current_landmark | counterevidence | governance_standard | delete_or_replace

        Allowed `api_route` values:
        pubmed | crossref_openalex | arxiv_openreview | publisher | human

        Draft passage:
        {chunk["text"]}
        """
    ).strip() + "\n"


def cmd_plan(args: argparse.Namespace) -> int:
    drafts = discover_drafts(args)
    if not drafts:
        raise SystemExit("No supported drafts found.")
    out_dir = Path(args.out_dir).resolve()
    packets_dir = out_dir / "subagent_discovery_packets"
    packets_dir.mkdir(parents=True, exist_ok=True)
    asset_summary = load_asset_summary(args.draft_assets_dir)
    chunks: list[dict[str, str]] = []
    for path in drafts:
        chunks.extend(chunk_body(path, args.packet_chars, args.overlap_lines))
    if args.max_packets > 0:
        chunks = chunks[: args.max_packets]
    (out_dir / "chief_literature_brief_packet.md").write_text(chief_packet(drafts, asset_summary, args.topic), encoding="utf-8")
    index_rows: list[dict[str, str]] = []
    for index, chunk in enumerate(chunks, 1):
        packet_name = f"packet_{index:03d}_{chunk['packet']}.md"
        packet_path = packets_dir / packet_name
        packet_path.write_text(subagent_prompt(chunk, args.topic, asset_summary), encoding="utf-8")
        index_rows.append({"packet": chunk["packet"], "draft": chunk["draft"], "path": str(packet_path), "chars": str(len(chunk["text"]))})
    write_csv(out_dir / "subagent_packet_index.csv", index_rows, ["packet", "draft", "path", "chars"])
    write_csv(out_dir / "subagent_output_template.csv", [], DISCOVERY_FIELDS)
    write_csv(out_dir / "api_search_task_template.csv", [], API_TASK_FIELDS)
    checkpoint = [
        "# Human Literature Discovery Checkpoint",
        "",
        "Before broad recall, confirm:",
        "",
        "- [ ] Chief Codex has read all drafts and written/updated the chief literature brief.",
        "- [ ] Subagents have read the discovery packets and returned CSV outputs.",
        "- [ ] Candidates are separated into reference recovery, hidden body candidates, claim gaps, and supplemental/background.",
        "- [ ] API tasks are bounded by argument roles, not broad topic recall.",
        "- [ ] No row is promoted to the main pool before deterministic verification and claim-fit adjudication.",
        "",
    ]
    (out_dir / "human_literature_discovery_checkpoint.md").write_text("\n".join(checkpoint), encoding="utf-8")
    write_json(
        out_dir / "literature_discovery_manifest.json",
        {
            "generated_at": now_iso(),
            "topic": args.topic,
            "drafts": [str(path) for path in drafts],
            "draft_count": len(drafts),
            "packet_count": len(chunks),
            "draft_assets_dir": asset_summary.get("root", ""),
            "outputs": {
                "chief_packet": str(out_dir / "chief_literature_brief_packet.md"),
                "packet_index": str(out_dir / "subagent_packet_index.csv"),
                "packets_dir": str(packets_dir),
            },
        },
    )
    print(json.dumps({"out_dir": str(out_dir), "drafts": len(drafts), "packets": len(chunks)}, ensure_ascii=False))
    return 0


def parse_csv_text(text: str) -> list[dict[str, str]]:
    handle = io.StringIO(text.strip())
    try:
        rows = list(csv.DictReader(handle))
    except csv.Error:
        return []
    return [{str(k or "").strip(): clean_space(v) for k, v in row.items()} for row in rows if any(clean_space(v) for v in row.values())]


def parse_input_file(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".csv":
        return read_csv(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, str]] = []
    for match in re.finditer(r"```csv\s*(.*?)```", text, flags=re.I | re.S):
        rows.extend(parse_csv_text(match.group(1)))
    return rows


def discover_inputs(args: argparse.Namespace) -> list[Path]:
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


def reference_keys(reference_csv: str) -> set[str]:
    keys: set[str] = set()
    if not reference_csv:
        return keys
    for row in read_csv(Path(reference_csv)):
        title = clean_space(row.get("candidate_title") or row.get("title"))
        doi = normalize_doi(row.get("doi"))
        pmid = clean_space(row.get("pmid"))
        if title:
            keys.add("title:" + normalize_title(title))
        if doi:
            keys.add("doi:" + doi)
        if pmid:
            keys.add("pmid:" + pmid)
    return keys


def coalesce(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = clean_space(row.get(key))
        if value:
            return value
    return ""


def parse_identifier(value: str) -> tuple[str, str]:
    text = clean_space(value)
    doi_match = re.search(r"10\.\d{4,9}/[^\s,;)\]]+", text, flags=re.I)
    pmid_match = re.search(r"\bPMID\s*:?\s*(\d+)\b", text, flags=re.I)
    doi = normalize_doi(doi_match.group(0)) if doi_match else ""
    pmid = pmid_match.group(1) if pmid_match else ""
    return doi, pmid


def row_to_candidate(row: dict[str, str], ref_keys: set[str]) -> dict[str, str]:
    title = coalesce(row, "candidate_title_or_system", "candidate_title", "title", "system_or_method")
    doi, pmid = parse_identifier(coalesce(row, "known_identifier", "doi", "pmid"))
    lane = coalesce(row, "candidate_lane") or "hidden_body_candidate"
    keys = {"title:" + normalize_title(title)}
    if doi:
        keys.add("doi:" + doi)
    if pmid:
        keys.add("pmid:" + pmid)
    source_kind = "codex_subagent_literature_discovery"
    if ref_keys and keys & ref_keys and lane not in {"explicit_reference", "reference_recovery"}:
        lane = "reference_recovery"
        source_kind = "codex_subagent_reference_recovery"
    cid_base = "|".join([coalesce(row, "packet"), coalesce(row, "draft"), title, doi, pmid, coalesce(row, "pubmed_query", "crossref_query", "arxiv_query")])
    candidate_id = "LD-" + hashlib.sha1(cid_base.encode("utf-8")).hexdigest()[:12]
    venue_year = coalesce(row, "venue_year_if_any", "venue_hint", "year_hint")
    year_match = re.search(r"\b(19|20)\d{2}\b", venue_year)
    priority = "high" if coalesce(row, "confidence").lower() == "high" or lane in {"current_landmark", "seminal_background"} else "medium"
    return {
        "candidate_id": candidate_id,
        "draft": coalesce(row, "draft"),
        "chunk_id": coalesce(row, "packet"),
        "claim_id": coalesce(row, "packet"),
        "candidate_title": title,
        "authors_or_group": coalesce(row, "authors_if_any", "authors_or_group"),
        "year_hint": year_match.group(0) if year_match else "",
        "venue_hint": venue_year,
        "system_or_method": title if not re.search(r"\s", title) else "",
        "doi": doi,
        "pmid": pmid,
        "urls": coalesce(row, "url", "urls"),
        "raw_evidence_excerpt": coalesce(row, "claim_or_argument", "why_it_is_likely_a_paper")[:700],
        "why_relevant": coalesce(row, "evidence_need", "why_it_is_likely_a_paper")[:500],
        "candidate_type": lane,
        "confidence": coalesce(row, "confidence") or "medium",
        "source_kind": source_kind,
        "cited_in_body": "body_mention",
        "verification_priority": priority,
        "pubmed_query": coalesce(row, "pubmed_query"),
        "notes": coalesce(row, "notes"),
    }


def candidate_key(row: dict[str, str]) -> str:
    if row.get("doi"):
        return "doi:" + normalize_doi(row.get("doi"))
    if row.get("pmid"):
        return "pmid:" + clean_space(row.get("pmid"))
    return "title:" + normalize_title(row.get("candidate_title"))


def dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    order: list[str] = []
    rank = {"high": 3, "medium": 2, "low": 1}
    for row in rows:
        key = candidate_key(row)
        if key not in best:
            best[key] = row
            order.append(key)
            continue
        if rank.get(row.get("confidence", "").lower(), 0) > rank.get(best[key].get("confidence", "").lower(), 0):
            best[key] = row
    return [best[key] for key in order]


def verifier_rows(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "draft": row.get("draft", ""),
            "ref_number": row.get("candidate_id", ""),
            "candidate_id": row.get("candidate_id", ""),
            "claim_id": row.get("claim_id", ""),
            "candidate_title": row.get("candidate_title", ""),
            "raw_reference": row.get("raw_evidence_excerpt") or row.get("why_relevant") or row.get("notes", ""),
            "doi": row.get("doi", ""),
            "pmid": row.get("pmid", ""),
            "urls": row.get("urls", ""),
            "source_kind": row.get("source_kind", ""),
            "cited_in_body": row.get("cited_in_body", ""),
            "verification_priority": row.get("verification_priority", "medium"),
            "candidate_type": row.get("candidate_type", ""),
            "pubmed_query": row.get("pubmed_query", ""),
            "raw_evidence_excerpt": row.get("raw_evidence_excerpt", ""),
            "why_relevant": row.get("why_relevant", ""),
        }
        for row in candidates
    ]


def api_tasks(rows: list[dict[str, str]], candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    id_by_key = {candidate_key(row): row.get("candidate_id", "") for row in candidates}
    tasks: list[dict[str, str]] = []
    for row in rows:
        title = coalesce(row, "candidate_title_or_system", "candidate_title", "title", "system_or_method")
        lane = coalesce(row, "candidate_lane") or "hidden_body_candidate"
        priority = "high" if coalesce(row, "confidence").lower() == "high" or lane in {"current_landmark", "seminal_background"} else "medium"
        query_map = {
            "pubmed": coalesce(row, "pubmed_query"),
            "crossref_openalex": coalesce(row, "crossref_query") or title,
            "arxiv_openreview": coalesce(row, "arxiv_query", "openreview_query") or title,
            "publisher": coalesce(row, "url") or title,
            "human": title,
        }
        route = coalesce(row, "api_route") or ("pubmed" if coalesce(row, "pubmed_query") else "crossref_openalex")
        query = query_map.get(route, "") or title
        if not query:
            continue
        candidate_stub = row_to_candidate(row, set())
        task_id = "LDT-" + hashlib.sha1("|".join([route, query, title]).encode("utf-8")).hexdigest()[:12]
        tasks.append(
            {
                "task_id": task_id,
                "candidate_id": id_by_key.get(candidate_key(candidate_stub), ""),
                "packet": coalesce(row, "packet"),
                "draft": coalesce(row, "draft"),
                "claim_or_argument": coalesce(row, "claim_or_argument")[:500],
                "candidate_title_or_system": title,
                "api_route": route,
                "query": query,
                "evidence_need": coalesce(row, "evidence_need"),
                "candidate_lane": lane,
                "priority": priority,
                "status": "needs_api_search",
                "notes": coalesce(row, "notes"),
            }
        )
    return dedupe_tasks(tasks)


def dedupe_tasks(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        key = "|".join([row.get("api_route", ""), normalize_title(row.get("query", ""))])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def cmd_collect(args: argparse.Namespace) -> int:
    inputs = discover_inputs(args)
    if not inputs:
        raise SystemExit("No subagent output files found.")
    raw_rows: list[dict[str, str]] = []
    for path in inputs:
        raw_rows.extend(parse_input_file(path))
    ref_keys = reference_keys(args.reference_candidates_csv)
    candidates = dedupe([row_to_candidate(row, ref_keys) for row in raw_rows if coalesce(row, "candidate_title_or_system", "candidate_title", "title", "system_or_method")])
    tasks = api_tasks(raw_rows, candidates)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "subagent_literature_discovery_rows.csv", raw_rows, DISCOVERY_FIELDS)
    write_csv(out_dir / "literature_discovery_candidates.csv", candidates, CANDIDATE_FIELDS)
    write_csv(out_dir / "candidate_paper_clues_for_verification.csv", verifier_rows(candidates), VERIFY_FIELDS)
    write_csv(out_dir / "api_search_tasks.csv", tasks, API_TASK_FIELDS)
    query_dir = out_dir / "queries"
    query_dir.mkdir(exist_ok=True)
    for route in ["pubmed", "crossref_openalex", "arxiv_openreview", "publisher", "human"]:
        lines = [row["query"] for row in tasks if row.get("api_route") == route and row.get("query")]
        (query_dir / f"{route}_queries.txt").write_text("\n".join(dict.fromkeys(lines)).rstrip() + ("\n" if lines else ""), encoding="utf-8")
    lines = [
        "# Literature Discovery Collection Checkpoint",
        "",
        f"- Raw subagent rows: {len(raw_rows)}",
        f"- Deduped candidates: {len(candidates)}",
        f"- API search tasks: {len(tasks)}",
        "",
        "Next gates:",
        "",
        "- Run deterministic verifier on `candidate_paper_clues_for_verification.csv`.",
        "- Run route-specific API recall for `queries/*.txt` when the candidate title is incomplete.",
        "- Keep all results out of the main pool until verification and claim-fit adjudication pass.",
    ]
    (out_dir / "human_literature_discovery_checkpoint.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "inputs": len(inputs), "rows": len(raw_rows), "candidates": len(candidates), "api_tasks": len(tasks)}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan and collect reasoning-first draft literature discovery work.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="Create chief Codex and subagent literature discovery packets.")
    p_plan.add_argument("--draft", action="append", default=[])
    p_plan.add_argument("--draft-dir", action="append", default=[])
    p_plan.add_argument("--draft-assets-dir", default="./review-data/02_literature/draft_assets")
    p_plan.add_argument("--topic", default="")
    p_plan.add_argument("--out-dir", default="./review-data/02_literature/literature_discovery")
    p_plan.add_argument("--packet-chars", type=int, default=9000)
    p_plan.add_argument("--overlap-lines", type=int, default=4)
    p_plan.add_argument("--max-packets", type=int, default=0)
    p_plan.set_defaults(func=cmd_plan)

    p_collect = sub.add_parser("collect", help="Collect subagent literature discovery CSV/markdown outputs.")
    p_collect.add_argument("--input", action="append", default=[])
    p_collect.add_argument("--input-dir", action="append", default=[])
    p_collect.add_argument("--reference-candidates-csv", default="")
    p_collect.add_argument("--out-dir", default="./review-data/02_literature/literature_discovery/collected")
    p_collect.set_defaults(func=cmd_collect)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
