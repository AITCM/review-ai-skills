#!/usr/bin/env python3
"""Renumber manuscript citations by first appearance and rebuild references.

The writing model may draft useful prose, but final numeric citation order must
be deterministic. This script scans a Markdown manuscript body, maps numeric
citation markers to first appearance order, rebuilds the References section,
and writes audit tables for missing, uncited, duplicate, or unofficial entries.
It never invents references.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REF_HEADING_RE = re.compile(
    r"(?im)^(?P<prefix>\s{0,3}#{1,6}\s*)?"
    r"(?P<title>references|bibliography|works cited|参考文献|參考文獻|文献|引用文献)\s*$"
)
CITATION_RE = re.compile(r"[\[［]([0-9][0-9,\-–—\s;；，、]*)[\]］]")
REF_ENTRY_RE = re.compile(r"^\s*(?:\[(?P<bracket>\d+)\]|(?P<plain>\d+)[\.\)]|(?P<full>［\d+］))\s*(?P<text>.*)")
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\]\)；;，,]+", flags=re.I)
PMID_RE = re.compile(r"\bPMID\s*:?\s*(\d+)\b|\bPubMed\s*:?\s*(\d+)\b", flags=re.I)
TITLE_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "using",
    "from",
    "into",
    "toward",
    "towards",
    "large",
    "language",
    "model",
    "models",
    "review",
}


@dataclass
class ReferenceEntry:
    old_number: int
    text: str


@dataclass
class CitationOccurrence:
    marker: str
    raw: str
    numbers: list[int]
    start: int
    line: int
    order: int


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def clean_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_doi(value: Any) -> str:
    text = clean_space(value).lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi\s*:\s*", "", text)
    return text.rstrip(".,;)）]")


def normalize_pmid(value: Any) -> str:
    text = clean_space(value)
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text if re.fullmatch(r"\d+", text) else ""


def normalize_title(value: Any) -> str:
    text = clean_space(value).lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_tokens(value: Any) -> set[str]:
    return {token for token in normalize_title(value).split() if len(token) > 2 and token not in TITLE_STOPWORDS}


def title_similarity(a: Any, b: Any) -> float:
    left = title_tokens(a)
    right = title_tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left), len(right))


def line_number_at(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def split_body_references(text: str) -> tuple[str, str, str]:
    matches = list(REF_HEADING_RE.finditer(text))
    if not matches:
        return text, "", ""
    match = matches[-1]
    body = text[: match.start()].rstrip()
    heading = text[match.start() : match.end()].strip()
    references = text[match.end() :].strip()
    return body, heading, references


def parse_ref_number(token: str) -> int | None:
    digits = re.sub(r"\D+", "", token or "")
    return int(digits) if digits else None


def parse_reference_entries(ref_text: str) -> dict[int, ReferenceEntry]:
    entries: dict[int, ReferenceEntry] = {}
    current_number: int | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_number, current_lines
        if current_number is None:
            return
        text = clean_space(" ".join(current_lines))
        entries[current_number] = ReferenceEntry(current_number, text)
        current_number = None
        current_lines = []

    for raw_line in ref_text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        match = REF_ENTRY_RE.match(line)
        if match:
            flush()
            number = parse_ref_number(match.group("bracket") or match.group("plain") or match.group("full") or "")
            current_number = number
            current_lines = [match.group("text").strip()]
        elif current_number is not None:
            current_lines.append(line.strip())
    flush()
    return entries


def parse_citation_numbers(raw: str) -> list[int]:
    numbers: list[int] = []
    parts = re.split(r"[,;；，、]\s*", raw.strip())
    for part in parts:
        part = part.strip()
        if not part:
            continue
        range_match = re.fullmatch(r"(\d+)\s*[-–—]\s*(\d+)", part)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            if start <= end and end - start <= 100:
                numbers.extend(range(start, end + 1))
            else:
                numbers.extend([start, end])
            continue
        for value in re.findall(r"\d+", part):
            numbers.append(int(value))
    result: list[int] = []
    seen: set[int] = set()
    for number in numbers:
        if number not in seen:
            result.append(number)
            seen.add(number)
    return result


def find_citations(body: str) -> list[CitationOccurrence]:
    occurrences: list[CitationOccurrence] = []
    for order, match in enumerate(CITATION_RE.finditer(body), 1):
        raw = match.group(1)
        numbers = parse_citation_numbers(raw)
        if not numbers:
            continue
        occurrences.append(
            CitationOccurrence(
                marker=match.group(0),
                raw=raw,
                numbers=numbers,
                start=match.start(),
                line=line_number_at(body, match.start()),
                order=order,
            )
        )
    return occurrences


def first_appearance_mapping(occurrences: list[CitationOccurrence]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for occurrence in occurrences:
        for old_number in occurrence.numbers:
            if old_number not in mapping:
                mapping[old_number] = len(mapping) + 1
    return mapping


def extract_doi(text: str) -> str:
    match = DOI_RE.search(text or "")
    return normalize_doi(match.group(0)) if match else ""


def extract_pmid(text: str) -> str:
    match = PMID_RE.search(text or "")
    if not match:
        return ""
    return normalize_pmid(match.group(1) or match.group(2) or "")


def load_official_rows(path: Path | None) -> list[dict[str, str]]:
    if not path or not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def build_official_indexes(rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    by_doi: dict[str, dict[str, str]] = {}
    by_pmid: dict[str, dict[str, str]] = {}
    for row in rows:
        doi = normalize_doi(row.get("doi") or row.get("draft_doi"))
        pmid = normalize_pmid(row.get("pmid") or row.get("paper_id"))
        if doi and doi not in by_doi:
            by_doi[doi] = row
        if pmid and pmid not in by_pmid:
            by_pmid[pmid] = row
    return by_doi, by_pmid


def match_official_row(
    reference: ReferenceEntry,
    official_rows: list[dict[str, str]],
    by_doi: dict[str, dict[str, str]],
    by_pmid: dict[str, dict[str, str]],
) -> tuple[dict[str, str] | None, str, float]:
    ref_text = reference.text
    doi = extract_doi(ref_text)
    if doi and doi in by_doi:
        return by_doi[doi], "doi", 1.0
    pmid = extract_pmid(ref_text)
    if pmid and pmid in by_pmid:
        return by_pmid[pmid], "pmid", 1.0

    best_row: dict[str, str] | None = None
    best_score = 0.0
    for row in official_rows:
        title = row.get("title") or row.get("draft_candidate_title") or row.get("candidate_title") or ""
        if not title:
            continue
        norm_title = normalize_title(title)
        norm_ref = normalize_title(ref_text)
        if norm_title and (norm_title in norm_ref or norm_ref in norm_title):
            score = 0.95
        else:
            score = title_similarity(title, ref_text)
        if score > best_score:
            best_row = row
            best_score = score
    if best_row and best_score >= 0.55:
        return best_row, "title", best_score
    return None, "", 0.0


def official_citation_text(row: dict[str, str] | None, old_text: str, args: argparse.Namespace) -> tuple[str, str]:
    if not row:
        return old_text, "no_official_match"
    citation = clean_space(row.get("citation_text"))
    export_status = clean_space(row.get("export_status"))
    if citation and (args.allow_derived or export_status not in {"derived_fallback", "offline_derived"}):
        return citation, export_status or "official_row"
    if citation:
        return old_text, "derived_not_allowed"
    return old_text, export_status or "official_row_without_citation_text"


def format_marker(numbers: list[int], compress_ranges: bool) -> str:
    unique = []
    seen: set[int] = set()
    for number in numbers:
        if number not in seen:
            unique.append(number)
            seen.add(number)
    if not compress_ranges:
        return "[" + ",".join(str(number) for number in unique) + "]"
    parts: list[str] = []
    i = 0
    while i < len(unique):
        start = unique[i]
        end = start
        j = i + 1
        while j < len(unique) and unique[j] == end + 1:
            end = unique[j]
            j += 1
        parts.append(str(start) if start == end else f"{start}-{end}")
        i = j
    return "[" + ",".join(parts) + "]"


def rewrite_body(
    body: str,
    mapping: dict[int, int],
    occurrences: list[CitationOccurrence],
    args: argparse.Namespace,
) -> tuple[str, list[dict[str, Any]]]:
    unresolved: list[dict[str, Any]] = []

    def replace(match: re.Match[str]) -> str:
        raw = match.group(1)
        old_numbers = parse_citation_numbers(raw)
        missing = [number for number in old_numbers if number not in mapping]
        if missing:
            unresolved.append(
                {
                    "line": line_number_at(body, match.start()),
                    "marker": match.group(0),
                    "old_numbers": ",".join(str(number) for number in old_numbers),
                    "missing_numbers": ",".join(str(number) for number in missing),
                    "reason": "number_seen_in_body_but_not_mapped",
                }
            )
            return match.group(0)
        new_numbers = [mapping[number] for number in old_numbers]
        if not args.preserve_marker_order:
            new_numbers = sorted(new_numbers)
        return format_marker(new_numbers, args.compress_ranges)

    return CITATION_RE.sub(replace, body), unresolved


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def build_reference_section(
    mapping: dict[int, int],
    reference_entries: dict[int, ReferenceEntry],
    official_rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_doi, by_pmid = build_official_indexes(official_rows)
    map_rows: list[dict[str, Any]] = []
    missing_refs: list[dict[str, Any]] = []
    missing_official: list[dict[str, Any]] = []
    final_entries: list[tuple[int, str]] = []

    for old_number, new_number in sorted(mapping.items(), key=lambda item: item[1]):
        reference = reference_entries.get(old_number)
        if not reference:
            missing_refs.append(
                {
                    "old_number": old_number,
                    "new_number": new_number,
                    "reason": "cited_number_missing_from_reference_section",
                }
            )
            map_rows.append(
                {
                    "old_number": old_number,
                    "new_number": new_number,
                    "old_reference": "",
                    "final_reference": "",
                    "match_method": "",
                    "match_score": "",
                    "doi": "",
                    "pmid": "",
                    "official_key": "",
                    "official_export_status": "",
                    "notes": "missing_reference_entry",
                }
            )
            continue

        if official_rows:
            official_row, match_method, match_score = match_official_row(reference, official_rows, by_doi, by_pmid)
            final_text, export_status = official_citation_text(official_row, reference.text, args)
        else:
            official_row, match_method, match_score = None, "", 0.0
            final_text, export_status = reference.text, "official_csv_not_supplied"
        if official_rows and (not official_row or export_status in {"no_official_match", "derived_not_allowed", "official_row_without_citation_text"}):
            missing_official.append(
                {
                    "old_number": old_number,
                    "new_number": new_number,
                    "old_reference": reference.text,
                    "match_method": match_method,
                    "match_score": f"{match_score:.3f}" if match_score else "",
                    "reason": export_status,
                }
            )
        final_entries.append((new_number, final_text))
        map_rows.append(
            {
                "old_number": old_number,
                "new_number": new_number,
                "old_reference": reference.text,
                "final_reference": final_text,
                "match_method": match_method,
                "match_score": f"{match_score:.3f}" if match_score else "",
                "doi": normalize_doi((official_row or {}).get("doi")) or extract_doi(reference.text),
                "pmid": normalize_pmid((official_row or {}).get("pmid")) or extract_pmid(reference.text),
                "official_key": (official_row or {}).get("key", ""),
                "official_export_status": export_status,
                "notes": "",
            }
        )

    section_lines = [f"{new_number}. {text}".rstrip() for new_number, text in final_entries]
    return "\n".join(section_lines), map_rows, missing_refs, missing_official


def uncited_reference_rows(reference_entries: dict[int, ReferenceEntry], mapping: dict[int, int]) -> list[dict[str, Any]]:
    cited = set(mapping)
    rows = []
    for old_number, entry in sorted(reference_entries.items()):
        if old_number not in cited:
            rows.append({"old_number": old_number, "old_reference": entry.text, "reason": "reference_not_cited_in_body"})
    return rows


def write_report(
    path: Path,
    args: argparse.Namespace,
    occurrences: list[CitationOccurrence],
    reference_entries: dict[int, ReferenceEntry],
    mapping: dict[int, int],
    missing_refs: list[dict[str, Any]],
    missing_official: list[dict[str, Any]],
    uncited_refs: list[dict[str, Any]],
    unresolved_markers: list[dict[str, Any]],
    official_rows: list[dict[str, str]],
) -> None:
    blockers = []
    warnings = []
    if missing_refs:
        blockers.append(f"{len(missing_refs)} cited numeric marker(s) have no matching reference entry.")
    if unresolved_markers:
        blockers.append(f"{len(unresolved_markers)} citation marker(s) could not be safely rewritten.")
    if args.require_official and not official_rows:
        blockers.append("Official citation export is required, but no official_citations.csv rows were supplied.")
    if missing_official:
        message = f"{len(missing_official)} cited reference(s) did not match an official exported citation text."
        if args.require_official:
            blockers.append(message)
        else:
            warnings.append(message)
    if uncited_refs:
        warnings.append(f"{len(uncited_refs)} reference-list entrie(s) are not cited in the body and were removed from the rebuilt bibliography.")

    lines = [
        "# Citation Sequence Report",
        "",
        f"- Generated: {now_iso()}",
        f"- Manuscript: `{args.manuscript}`",
        f"- Official citations CSV: `{args.official_citations_csv}`" if args.official_citations_csv else "- Official citations CSV: not supplied",
        f"- Citation markers found: {len(occurrences)}",
        f"- Unique cited reference numbers: {len(mapping)}",
        f"- Reference-list entries parsed: {len(reference_entries)}",
        f"- Official citation rows loaded: {len(official_rows)}",
        f"- Status: {'blocked' if blockers else 'pass_with_warnings' if warnings else 'pass'}",
        "",
        "## Blocking Issues",
        "",
    ]
    lines.extend([f"- {item}" for item in blockers] or ["- None"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {item}" for item in warnings] or ["- None"])
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `manuscript_renumbered.md`: body and reference list rebuilt by first appearance.",
            "- `citation_sequence_map.csv`: old-to-new citation mapping and official citation match status.",
            "- `missing_reference_entries.csv`: cited numbers with no parsed reference entry.",
            "- `missing_official_citations.csv`: cited references that need official citation-manager export/manual handoff.",
            "- `uncited_references.csv`: original reference entries removed because they were not cited.",
            "",
            "## Policy",
            "",
            "Use this script after official citation export and before final DOCX/PDF generation. Do not let the writing model assign final numeric reference order.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def command_renumber(args: argparse.Namespace) -> int:
    manuscript = Path(args.manuscript)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = manuscript.read_text(encoding="utf-8-sig", errors="replace")
    body, heading, ref_text = split_body_references(text)
    reference_entries = parse_reference_entries(ref_text)
    occurrences = find_citations(body)
    mapping = first_appearance_mapping(occurrences)
    official_rows = load_official_rows(Path(args.official_citations_csv) if args.official_citations_csv else None)
    new_body, unresolved_markers = rewrite_body(body, mapping, occurrences, args)
    new_references, map_rows, missing_refs, missing_official = build_reference_section(
        mapping, reference_entries, official_rows, args
    )
    uncited_refs = uncited_reference_rows(reference_entries, mapping)

    final_heading = heading or "## References"
    rebuilt = new_body.rstrip() + "\n\n" + final_heading + "\n\n" + new_references.rstrip() + "\n"
    (out_dir / "manuscript_renumbered.md").write_text(rebuilt, encoding="utf-8")

    write_csv(
        out_dir / "citation_sequence_map.csv",
        map_rows,
        [
            "old_number",
            "new_number",
            "old_reference",
            "final_reference",
            "match_method",
            "match_score",
            "doi",
            "pmid",
            "official_key",
            "official_export_status",
            "notes",
        ],
    )
    write_csv(out_dir / "missing_reference_entries.csv", missing_refs, ["old_number", "new_number", "reason"])
    write_csv(
        out_dir / "missing_official_citations.csv",
        missing_official,
        ["old_number", "new_number", "old_reference", "match_method", "match_score", "reason"],
    )
    write_csv(out_dir / "uncited_references.csv", uncited_refs, ["old_number", "old_reference", "reason"])
    write_csv(
        out_dir / "unresolved_citation_markers.csv",
        unresolved_markers,
        ["line", "marker", "old_numbers", "missing_numbers", "reason"],
    )
    summary = {
        "schema_name": "citation_sequence_manager",
        "schema_version": "0.1",
        "generated_at": now_iso(),
        "manuscript": str(manuscript),
        "markers": len(occurrences),
        "unique_cited_numbers": len(mapping),
        "reference_entries": len(reference_entries),
        "official_rows": len(official_rows),
        "missing_reference_entries": len(missing_refs),
        "missing_official_citations": len(missing_official),
        "uncited_references": len(uncited_refs),
        "unresolved_citation_markers": len(unresolved_markers),
        "status": "blocked"
        if missing_refs or unresolved_markers or (args.require_official and (missing_official or not official_rows))
        else "pass",
    }
    (out_dir / "citation_sequence_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(
        out_dir / "citation_sequence_report.md",
        args,
        occurrences,
        reference_entries,
        mapping,
        missing_refs,
        missing_official,
        uncited_refs,
        unresolved_markers,
        official_rows,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if args.fail_on_blocker and summary["status"] == "blocked" else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    renumber = subparsers.add_parser("renumber", help="Renumber numeric citations by first appearance.")
    renumber.add_argument("--manuscript", required=True, help="Markdown manuscript to audit and renumber.")
    renumber.add_argument("--official-citations-csv", default="", help="Optional official_citations.csv from official_citation_exporter.py.")
    renumber.add_argument("--out-dir", required=True, help="Output directory for the renumbered manuscript and audits.")
    renumber.add_argument("--compress-ranges", action="store_true", help="Write consecutive citations as ranges, e.g. [1-3].")
    renumber.add_argument("--preserve-marker-order", action="store_true", help="Preserve old order inside each citation marker instead of sorting final numbers ascending.")
    renumber.add_argument("--allow-derived", action="store_true", help="Allow derived citation_text rows when official export text is unavailable.")
    renumber.add_argument("--require-official", action="store_true", help="Block finalization when official citation-manager exports are missing or unmatched.")
    renumber.add_argument("--fail-on-blocker", action="store_true", help="Exit 2 if cited markers cannot be mapped to reference entries.")
    renumber.set_defaults(func=command_renumber)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
