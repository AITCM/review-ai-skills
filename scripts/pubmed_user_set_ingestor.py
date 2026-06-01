#!/usr/bin/env python3
"""Ingest user-supplied PubMed summary/abstract text exports.

PubMed sets supplied by the user are high-value seed material, but they are not
automatically accepted citations. This script parses PubMed-generated text into
the same candidate shape used by supplemental recall, then writes an agent
screening packet so Codex/subagents can decide what supports the review frame.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from supplemental_recall_screening import (
    CANDIDATE_FIELDS,
    SCREENING_FIELDS,
    VERIFY_FIELDS,
    clean_space,
    cmd_collect_screening,
    dedupe_candidates,
    normalize_doi,
    normalize_title,
    now_iso,
    rank_candidates,
    read_csv,
    safe_id,
    select_screening_queue,
    write_csv,
    write_json,
    write_screening_packets,
)


RECORD_START = re.compile(r"(?m)^(?P<num>\d+)(?P<sep>[.:])\s+")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
TAG_RE = re.compile(r"<[^>]+>")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")


def split_records(text: str) -> list[tuple[str, str, str]]:
    matches = list(RECORD_START.finditer(text))
    records: list[tuple[str, str, str]] = []
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[match.start() : end].strip()
        record_type = "summary" if match.group("sep") == ":" else "abstract"
        records.append((match.group("num"), record_type, block))
    return records


def paragraphs(block: str) -> list[str]:
    return [clean_space(part) for part in re.split(r"\n\s*\n", block) if clean_space(part)]


def first_match(pattern: str, text: str, flags: int = re.I) -> str:
    match = re.search(pattern, text, flags=flags)
    return clean_space(match.group(1)) if match else ""


def strip_markup(text: str) -> str:
    return clean_space(TAG_RE.sub("", text or ""))


def parse_year(text: str) -> str:
    match = YEAR_RE.search(text)
    return match.group(0) if match else ""


def parse_summary_record(index: str, block: str, source_file: Path) -> dict[str, str]:
    body = clean_space(re.sub(r"^\d+:\s*", "", block))
    pmid = first_match(r"\bPMID:\s*(\d+)", body)
    pmcid = first_match(r"\bPMCID:\s*([A-Z0-9]+)", body)
    doi = normalize_doi(first_match(r"\bdoi:\s*([^;\s]+)", body))
    authors = ""
    title = ""
    journal = ""
    year = parse_year(body)

    first_split = re.match(r"(?P<authors>.+?)\.\s+(?P<tail>.+)", body)
    if first_split:
        authors = clean_space(first_split.group("authors"))
        tail = clean_space(first_split.group("tail"))
        title_match = re.match(r"(?P<title>.+?)\.\s+(?P<journal>[A-Za-z0-9][A-Za-z0-9 .&()/-]+?)\.\s+(?P<year>(?:19|20)\d{2})\b", tail)
        if title_match:
            title = strip_markup(title_match.group("title"))
            journal = strip_markup(title_match.group("journal"))
            year = clean_space(title_match.group("year")) or year
        else:
            title = strip_markup(tail.split(". ", 1)[0].strip())

    return {
        "record_index": index,
        "record_format": "summary",
        "source_file": str(source_file),
        "raw_citation": body,
        "title": strip_markup(title),
        "authors": strip_markup(authors),
        "year": year,
        "journal": journal,
        "pmid": pmid,
        "pmcid": pmcid,
        "doi": doi,
        "abstract": "",
        "publication_type": "preprint" if re.search(r"\b(preprint|biorxiv|medrxiv)\b", body, flags=re.I) else "pubmed-summary",
    }


def parse_abstract_record(index: str, block: str, source_file: Path) -> dict[str, str]:
    parts = paragraphs(block)
    body = clean_space(re.sub(r"^\d+\.\s*", "", block))
    citation = clean_space(re.sub(r"^\d+\.\s*", "", parts[0])) if parts else ""
    title = strip_markup(parts[1]) if len(parts) > 1 else ""
    authors = strip_markup(parts[2]) if len(parts) > 2 else ""
    pmid = first_match(r"\bPMID:\s*(\d+)", body)
    pmcid = first_match(r"\bPMCID:\s*([A-Z0-9]+)", body)
    doi = normalize_doi(first_match(r"(?:\bdoi:\s*|\bDOI:\s*)([^;\s]+)", body))
    year = parse_year(citation)
    journal = citation.split(".", 1)[0] if citation else ""

    abstract_parts: list[str] = []
    after_author_info = False
    for part in parts[3:]:
        low = part.lower()
        if low.startswith("author information"):
            after_author_info = True
            continue
        if re.match(r"^(doi:|pmcid:|pmid:|conflict of interest|copyright|publication types?:)", low):
            break
        if low.startswith("copyright") or "the author(s)" in low or "published by" in low:
            break
        if after_author_info or not re.search(r"^\(?\d+\)", part):
            abstract_parts.append(part)
            after_author_info = True
    abstract = clean_space(" ".join(abstract_parts))

    return {
        "record_index": index,
        "record_format": "abstract",
        "source_file": str(source_file),
        "raw_citation": citation,
        "title": title,
        "authors": authors,
        "year": year,
        "journal": strip_markup(journal),
        "pmid": pmid,
        "pmcid": pmcid,
        "doi": doi,
        "abstract": abstract,
        "publication_type": "preprint" if re.search(r"\b(preprint|biorxiv|medrxiv)\b", body, flags=re.I) else "pubmed-abstract",
    }


def parse_file(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    text = read_text(path)
    for index, record_type, block in split_records(text):
        row = parse_summary_record(index, block, path) if record_type == "summary" else parse_abstract_record(index, block, path)
        if row.get("pmid") or row.get("doi") or row.get("title"):
            rows.append(row)
    return rows


def record_key(row: dict[str, str]) -> str:
    if row.get("pmid"):
        return "pmid:" + row["pmid"]
    if normalize_doi(row.get("doi")):
        return "doi:" + normalize_doi(row.get("doi"))
    return "title:" + normalize_title(row.get("title"))


def merge_records(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for row in rows:
        key = record_key(row)
        if not key or key == "title:":
            continue
        if key not in merged:
            merged[key] = dict(row)
            order.append(key)
            continue
        target = merged[key]
        for field in ["title", "authors", "year", "journal", "pmid", "pmcid", "doi", "publication_type", "raw_citation"]:
            if not target.get(field) and row.get(field):
                target[field] = row[field]
        if len(row.get("abstract", "")) > len(target.get("abstract", "")):
            target["abstract"] = row["abstract"]
        files = set(filter(None, target.get("source_file", "").split("; ")))
        files.add(row.get("source_file", ""))
        target["source_file"] = "; ".join(sorted(files))
        formats = set(filter(None, target.get("record_format", "").split("; ")))
        formats.add(row.get("record_format", ""))
        target["record_format"] = "; ".join(sorted(formats))
    return [merged[key] for key in order]


def to_candidate(row: dict[str, str], topic: str, set_name: str) -> dict[str, str]:
    title = clean_space(row.get("title"))
    pmid = clean_space(row.get("pmid"))
    doi = normalize_doi(row.get("doi"))
    candidate_id = safe_id("|".join([set_name, pmid, doi, title]), "UPS")
    return {
        "candidate_id": candidate_id,
        "task_id": safe_id("|".join(["user_pubmed_seed_set", set_name, topic]), "UPT"),
        "lane": "user_pubmed_seed_set",
        "source_route": "user_pubmed_export",
        "query": topic or set_name,
        "title": title,
        "authors": clean_space(row.get("authors")),
        "year": clean_space(row.get("year")),
        "journal": clean_space(row.get("journal")),
        "source": "pubmed_user_export",
        "paper_id": pmid,
        "pmid": pmid,
        "doi": doi,
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else (f"https://doi.org/{doi}" if doi else ""),
        "pdf_url": "",
        "abstract": clean_space(row.get("abstract")),
        "publication_type": clean_space(row.get("publication_type")),
        "linked_claim_id": "",
        "linked_claim": f"User-supplied PubMed set `{set_name}`; screen against the confirmed review framework before verification.",
        "evidence_need": "user_pubmed_seed_screening",
        "argument_role": "candidate_evidence_or_background",
        "time_role": "user_preparation_seed",
        "recall_queries": topic or set_name,
        "candidate_fit_status": "needs_agent_screening",
        "candidate_fit_reason": "User-supplied PubMed export; high-priority seed material but not automatically included.",
        "screening_score": "",
        "screening_priority": "",
        "screening_reasons": "",
        "seed_lock": "",
        "seed_origin": clean_space(row.get("source_file")),
        "pool_status": "user_pubmed_seed_unscreened",
        "human_decision_needed": "include_core | include_support | include_framework | include_background | reject | needs_fulltext",
        "notes": clean_space(row.get("raw_citation"))[:900],
    }


def write_summary(path: Path, raw_count: int, rows: list[dict[str, str]], queue: list[dict[str, str]], args: argparse.Namespace) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        priority = row.get("screening_priority") or "unranked"
        counts[priority] = counts.get(priority, 0) + 1
    lines = [
        "# User PubMed Set Ingestion",
        "",
        f"- Generated at: {now_iso()}",
        f"- Set name: {args.set_name}",
        f"- Topic: {args.topic or '[not specified]'}",
        f"- Input files: {len(args.input)}",
        f"- Raw parsed records: {raw_count}",
        f"- Deduped candidates: {len(rows)}",
        f"- Screening queue: {len(queue)}",
        f"- Priority counts: {json.dumps(counts, ensure_ascii=False)}",
        "",
        "Policy: this is a user-supplied PubMed seed set. It is valuable for citation discovery, but rows must still pass agent/human screening, deterministic metadata verification, claim-fit adjudication, and the published-only final gate before entering final references.",
        "",
        "Next steps:",
        "",
        "1. Assign `agent_screening_packet.md` or `screening_packets/` to Codex subagents.",
        "2. Save decisions to `agent_screening_decisions.csv` using the provided template.",
        "3. Run `supplemental_recall_screening.py collect-screening` with this candidate CSV and the decisions CSV to create verifier-ready rows.",
        "4. Run `draft_reference_verifier.py verify` and citation adjudication before pool import.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_user_pubmed_screening_guide(path: Path, rows: list[dict[str, str]], queue: list[dict[str, str]], packet_rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    lines = [
        "# User PubMed Seed Screening Guide",
        "",
        f"- Set name: {args.set_name}",
        f"- Topic: {args.topic or '[not specified]'}",
        f"- Deduped candidates: {len(rows)}",
        f"- Candidates queued for screening: {len(queue)}",
        "",
        "This set comes from a user-prepared PubMed export. Treat it as a high-value seed lane, not as an accepted reference list.",
        "",
        "Screening rules:",
        "",
        "- Include only papers that support, challenge, historicize, or operationalize the confirmed review framework.",
        "- Mark papers as `include_core`, `include_support`, `include_framework`, `include_background`, `reject`, or `needs_fulltext`.",
        "- Fill `argument_role`, `evidence_level`, `evidence_strength`, `fulltext_need`, `citation_role`, and `key_supported_claim`; do not leave the claim-evidence link implicit.",
        "- Preprints or ahead-of-print records can remain candidates, but final references still need the published-only gate.",
        "- If a paper is good but off-frame, mark `include_background` or `reject` with a reason rather than letting it reshape the article silently.",
        "- Use abstracts for triage only; detailed method/result/metric claims require full text or user-supplied PDFs.",
        "",
        "Packet files:",
        "",
        "| Packet | Range | Candidates | Priorities | File |",
        "|---|---:|---:|---|---|",
    ]
    for row in packet_rows:
        lines.append(f"| {row['packet_id']} | {row['start_index']}-{row['end_index']} | {row['candidates']} | {row['priorities']} | `{row['packet_file']}` |")
    lines.extend([
        "",
        "After screening, collect decisions with `supplemental_recall_screening.py collect-screening`, then run deterministic verification and citation adjudication before promotion.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_parse(args: argparse.Namespace) -> int:
    raw_rows: list[dict[str, str]] = []
    for item in args.input:
        raw_rows.extend(parse_file(Path(item)))
    records = merge_records(raw_rows)
    candidates = [to_candidate(row, args.topic, args.set_name) for row in records]
    candidates = rank_candidates(dedupe_candidates(candidates), args.topic)
    queue = select_screening_queue(candidates, args.min_screening_score, args.max_screening_candidates, args.include_quarantine)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "pubmed_user_set_records.csv", records, [
        "record_index",
        "record_format",
        "source_file",
        "raw_citation",
        "title",
        "authors",
        "year",
        "journal",
        "pmid",
        "pmcid",
        "doi",
        "abstract",
        "publication_type",
    ])
    write_csv(out_dir / "pubmed_user_set_candidates.csv", candidates, CANDIDATE_FIELDS)
    write_csv(out_dir / "screening_queue.csv", queue, CANDIDATE_FIELDS)
    write_csv(out_dir / "agent_screening_template.csv", [{**{field: "" for field in SCREENING_FIELDS}, "candidate_id": row["candidate_id"], "task_id": row["task_id"]} for row in queue], SCREENING_FIELDS)
    packet_rows = write_screening_packets(out_dir, queue, args)
    write_user_pubmed_screening_guide(out_dir / "user_pubmed_screening_guide.md", candidates, queue, packet_rows, args)
    write_summary(out_dir / "pubmed_user_set_ingestion_summary.md", len(raw_rows), candidates, queue, args)
    write_json(
        out_dir / "pubmed_user_set_manifest.json",
        {
            "generated_at": now_iso(),
            "set_name": args.set_name,
            "topic": args.topic,
            "input_files": args.input,
            "raw_records": len(raw_rows),
            "deduped_candidates": len(candidates),
            "screening_queue": len(queue),
            "packets": packet_rows,
            "screening_guide": "user_pubmed_screening_guide.md",
        },
    )
    print(json.dumps({"out_dir": str(out_dir), "raw_records": len(raw_rows), "candidates": len(candidates), "screening_queue": len(queue)}, ensure_ascii=False))
    return 0


def resolve_collect_paths(args: argparse.Namespace) -> tuple[Path, list[Path], Path]:
    set_dir = Path(args.set_dir) if args.set_dir else None
    candidates_csv = Path(args.candidates_csv) if args.candidates_csv else (set_dir / "pubmed_user_set_candidates.csv" if set_dir else None)
    screening_csvs = [Path(path) for path in args.screening_csv]
    if not screening_csvs and set_dir:
        screening_csvs = [set_dir / "agent_screening_decisions.csv"]
    out_dir = Path(args.out_dir) if args.out_dir else (set_dir / "screened" if set_dir else None)
    if candidates_csv is None:
        raise SystemExit("Provide --set-dir or --candidates-csv.")
    if not screening_csvs:
        raise SystemExit("Provide --screening-csv or put agent_screening_decisions.csv under --set-dir.")
    if out_dir is None:
        raise SystemExit("Provide --out-dir or --set-dir.")
    return candidates_csv, screening_csvs, out_dir


def write_verification_handoff(out_dir: Path, candidates_csv: Path, screening_csvs: list[Path], verifier_rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    lines = [
        "# User PubMed Verification Handoff",
        "",
        f"- Generated at: {now_iso()}",
        f"- Candidates CSV: `{candidates_csv}`",
        f"- Screening CSVs: {', '.join(f'`{path}`' for path in screening_csvs)}",
        f"- Verifier-ready rows: {len(verifier_rows)}",
        "",
        "Next command:",
        "",
        "```bash",
        f"python $SKILL_DIR/scripts/draft_reference_verifier.py verify \\",
        f"  --candidate-csv {out_dir / 'user_pubmed_candidates_for_verification.csv'} \\",
        f"  --out-dir {args.verification_out_dir or './review-data/05_audit/user_pubmed_seed_verification'} \\",
        "  --email <email> \\",
        "  --workers 4 \\",
        "  --resume",
        "```",
        "",
        "After verification, run source adjudication and merge only accepted verified rows into the governed literature pool. Keep rejected/background rows in the user PubMed seed folder for audit.",
    ]
    (out_dir / "user_pubmed_verification_handoff.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_collect_user_pubmed_screening(args: argparse.Namespace) -> int:
    candidates_csv, screening_csvs, out_dir = resolve_collect_paths(args)
    shared_args = argparse.Namespace(candidates_csv=str(candidates_csv), screening_csv=[str(path) for path in screening_csvs], out_dir=str(out_dir))
    rc = cmd_collect_screening(shared_args)
    verifier_rows = read_csv(out_dir / "supplemental_candidates_for_verification.csv")
    core_rows = read_csv(out_dir / "core_candidates_for_verification.csv")
    framework_rows = read_csv(out_dir / "framework_candidates_for_verification.csv")
    write_csv(out_dir / "user_pubmed_candidates_for_verification.csv", verifier_rows, VERIFY_FIELDS)
    write_csv(out_dir / "user_pubmed_core_candidates_for_verification.csv", core_rows, VERIFY_FIELDS)
    write_csv(out_dir / "user_pubmed_framework_candidates_for_verification.csv", framework_rows, VERIFY_FIELDS)
    write_verification_handoff(out_dir, candidates_csv, screening_csvs, verifier_rows, args)
    write_json(
        out_dir / "user_pubmed_screening_manifest.json",
        {
            "generated_at": now_iso(),
            "candidates_csv": str(candidates_csv),
            "screening_csvs": [str(path) for path in screening_csvs],
            "out_dir": str(out_dir),
            "verifier_ready": len(verifier_rows),
            "core_verifier_ready": len(core_rows),
            "framework_verifier_ready": len(framework_rows),
            "verification_handoff": "user_pubmed_verification_handoff.md",
        },
    )
    print(json.dumps({"out_dir": str(out_dir), "verifier_ready": len(verifier_rows), "core_verifier_ready": len(core_rows), "framework_verifier_ready": len(framework_rows)}, ensure_ascii=False))
    return rc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse user-supplied PubMed summary/abstract text exports into screened candidate lanes.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_parse = sub.add_parser("parse", help="Parse PubMed text exports and create agent screening packets.")
    p_parse.add_argument("--input", action="append", required=True, help="PubMed Summary or Abstract text export. Repeat for multiple files.")
    p_parse.add_argument("--topic", default="")
    p_parse.add_argument("--set-name", default="user_pubmed_set")
    p_parse.add_argument("--out-dir", default="./review-data/02_literature/user_pubmed_sets/user_pubmed_set")
    p_parse.add_argument("--packet-size", type=int, default=40)
    p_parse.add_argument("--min-screening-score", type=int, default=3)
    p_parse.add_argument("--max-screening-candidates", type=int, default=0)
    p_parse.add_argument("--include-quarantine", action="store_true")
    p_parse.add_argument("--max-abstract-chars", type=int, default=1200)
    p_parse.set_defaults(func=cmd_parse)

    p_collect = sub.add_parser("collect-screening", help="Collect user PubMed seed screening decisions and create verifier-ready aliases.")
    p_collect.add_argument("--set-dir", default="", help="Directory created by the parse command.")
    p_collect.add_argument("--candidates-csv", default="", help="Override candidate CSV; defaults to <set-dir>/pubmed_user_set_candidates.csv.")
    p_collect.add_argument("--screening-csv", action="append", default=[], help="Agent/human screening decision CSV. Repeat for multiple files.")
    p_collect.add_argument("--out-dir", default="", help="Defaults to <set-dir>/screened.")
    p_collect.add_argument("--verification-out-dir", default="", help="Suggested downstream verification output directory for the handoff note.")
    p_collect.set_defaults(func=cmd_collect_user_pubmed_screening)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
