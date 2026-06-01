#!/usr/bin/env python3
"""Export official citation-manager records for verified papers.

This script runs after paper identity verification and citation adjudication.
It does not treat draft DOI/PMID strings as authoritative. Input rows should
come from verifier/adjudicator outputs that already resolved the paper identity.

Preferred outputs are official/authority-provided exports:
- PubMed EFetch NBIB/MEDLINE for PMID rows.
- Crossref DOI transform or DOI content negotiation for DOI rows.
- arXiv BibTeX for arXiv identifiers.

When no official export can be fetched, the row is written to a manual handoff
list instead of silently fabricating a final reference.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
DEFAULT_UA = "review-ai-skills/1.0"
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
}
VERIFIED_STATUSES = {
    "verified_pubmed",
    "verified_crossref",
    "verified_openalex",
    "verified_openreview",
    "verified_conference_page",
    "verified_publisher_url",
}
EXPORT_FIELDS = [
    "key",
    "title",
    "authors",
    "year",
    "journal",
    "doi",
    "pmid",
    "url",
    "source",
    "verification_status",
    "preferred_export_source",
    "citation_text",
    "bibtex",
    "ris",
    "nbib_or_medline",
    "export_status",
    "exported_at",
    "notes",
]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def clean_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_doi(value: Any) -> str:
    text = clean_space(value).lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi\s*:\s*", "", text)
    return text.rstrip(".,;)")


def normalize_pmid(value: Any) -> str:
    text = clean_space(value)
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text if re.fullmatch(r"\d+", text) else ""


def normalize_title(value: Any) -> str:
    text = clean_space(value).lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_similarity(a: Any, b: Any) -> float:
    left = {token for token in normalize_title(a).split() if len(token) > 2 and token not in TITLE_STOPWORDS}
    right = {token for token in normalize_title(b).split() if len(token) > 2 and token not in TITLE_STOPWORDS}
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left), len(right))


def title_matches(expected: Any, exported: Any, min_similarity: float = 0.55) -> bool:
    expected_norm = normalize_title(expected)
    exported_norm = normalize_title(exported)
    if not expected_norm or not exported_norm:
        return True
    if expected_norm in exported_norm or exported_norm in expected_norm:
        return True
    return title_similarity(expected, exported) >= min_similarity


def first_list_value(value: Any) -> str:
    for part in re.split(r";|\|", clean_space(value)):
        part = part.strip()
        if part:
            return part
    return ""


def citation_key(row: dict[str, Any]) -> str:
    doi = normalize_doi(row.get("doi"))
    pmid = normalize_pmid(row.get("pmid") or row.get("paper_id"))
    if doi:
        return "doi-" + re.sub(r"[^a-z0-9]+", "-", doi).strip("-")
    if pmid:
        return "pmid-" + re.sub(r"[^0-9A-Za-z]+", "-", pmid).strip("-")
    title = clean_space(row.get("title") or row.get("draft_candidate_title"))
    return "title-" + hashlib.sha1(title.encode("utf-8", errors="ignore")).hexdigest()[:12]


def input_dedupe_key(row: dict[str, Any], mode: str) -> str:
    if mode == "none":
        return "row-" + hashlib.sha1(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    doi = normalize_doi(row.get("doi") or row.get("draft_doi"))
    if doi and "doi" in mode:
        return "doi:" + doi
    pmid = normalize_pmid(first_list_value(row.get("pmid") or row.get("paper_id")))
    if pmid and "pmid" in mode:
        return "pmid:" + pmid
    title = normalize_title(row.get("title") or row.get("draft_candidate_title") or row.get("candidate_title"))
    if title and "title" in mode:
        return "title:" + title
    return "row-" + hashlib.sha1(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def row_quality_score(row: dict[str, Any]) -> int:
    score = 0
    if clean_space(row.get("verification_status")) in VERIFIED_STATUSES:
        score += 100
    if normalize_pmid(first_list_value(row.get("pmid") or row.get("paper_id"))):
        score += 30
    if normalize_doi(row.get("doi") or row.get("draft_doi")):
        score += 25
    if clean_space(row.get("abstract")):
        score += 10
    if clean_space(row.get("source")).lower() == "pubmed":
        score += 5
    return score


def dedupe_input_rows(rows: list[dict[str, Any]], mode: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if mode == "none":
        return rows, []
    chosen: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    duplicates: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, 1):
        key = input_dedupe_key(row, mode)
        row_with_index = dict(row)
        row_with_index["_input_row"] = str(idx)
        if key not in chosen:
            chosen[key] = row_with_index
            order.append(key)
            continue
        previous = chosen[key]
        duplicate = {
            "dedupe_key": key,
            "kept_input_row": previous.get("_input_row", ""),
            "duplicate_input_row": str(idx),
            "kept_title": clean_space(previous.get("title") or previous.get("draft_candidate_title")),
            "duplicate_title": clean_space(row.get("title") or row.get("draft_candidate_title")),
            "kept_doi": normalize_doi(previous.get("doi") or previous.get("draft_doi")),
            "duplicate_doi": normalize_doi(row.get("doi") or row.get("draft_doi")),
            "kept_pmid": normalize_pmid(first_list_value(previous.get("pmid") or previous.get("paper_id"))),
            "duplicate_pmid": normalize_pmid(first_list_value(row.get("pmid") or row.get("paper_id"))),
        }
        if row_quality_score(row_with_index) > row_quality_score(previous):
            duplicate["kept_input_row"] = str(idx)
            duplicate["duplicate_input_row"] = previous.get("_input_row", "")
            chosen[key] = row_with_index
        duplicates.append(duplicate)
    unique = []
    for key in order:
        row = dict(chosen[key])
        row.pop("_input_row", None)
        unique.append(row)
    return unique, duplicates


def http_text(url: str, args: argparse.Namespace, accept: str = "text/plain") -> str:
    headers = {"User-Agent": args.user_agent}
    if accept:
        headers["Accept"] = accept
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        return response.read().decode("utf-8", errors="replace").strip()


def try_http_text(url: str, args: argparse.Namespace, accept: str = "text/plain") -> str:
    try:
        text = http_text(url, args, accept=accept)
    except Exception:
        return ""
    if re.search(r"^\s*(resource not found|not found|<html\b)", text, flags=re.I):
        return ""
    return text


def fetch_pubmed_export(pmid: str, args: argparse.Namespace) -> str:
    if not pmid or args.offline:
        return ""
    params_base = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "text",
        "tool": "review-ai-skills",
    }
    if args.email:
        params_base["email"] = args.email
    if args.api_key:
        params_base["api_key"] = args.api_key
    for rettype in ["nbib", "medline"]:
        params = {**params_base, "rettype": rettype}
        url = PUBMED_BASE + "?" + urllib.parse.urlencode(params)
        text = try_http_text(url, args)
        if text and ("PMID-" in text or "TY  -" in text or "TI  -" in text):
            return text
    return ""


def fetch_crossref_transform(doi: str, args: argparse.Namespace, mime: str) -> str:
    if not doi or args.offline:
        return ""
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="") + "/transform/" + mime
    return try_http_text(url, args, accept=mime)


def fetch_doi_content_negotiation(doi: str, args: argparse.Namespace, accept: str) -> str:
    if not doi or args.offline:
        return ""
    url = "https://doi.org/" + urllib.parse.quote(doi, safe="/")
    return try_http_text(url, args, accept=accept)


def extract_arxiv_id(row: dict[str, Any]) -> str:
    text = " ".join(clean_space(row.get(field)) for field in ["doi", "url", "paper_id", "raw_reference", "title"])
    patterns = [
        r"10\.48550/arxiv\.([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)",
        r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)",
        r"\barxiv[:\s]+([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return match.group(1)
    return ""


def fetch_arxiv_bibtex(arxiv_id: str, args: argparse.Namespace) -> str:
    if not arxiv_id or args.offline:
        return ""
    return try_http_text("https://arxiv.org/bibtex/" + urllib.parse.quote(arxiv_id), args, accept="application/x-bibtex,text/plain")


def extract_structured_export_titles(*texts: str) -> list[str]:
    titles: list[str] = []
    for text in texts:
        if not text:
            continue
        matches = list(re.finditer(r"\btitle\s*=\s*[{\"'](.+?)[}\"']\s*,", text, flags=re.I | re.S))
        matches.extend(re.finditer(r"^(?:TI|T1)\s+-\s+(.+)$", text, flags=re.I | re.M))
        for match in matches:
            title = clean_space(re.sub(r"\s+", " ", match.group(1)))
            title = re.sub(r"\s+[A-Z0-9]{2}\s+-\s+.*$", "", title)
            if title and normalize_title(title) not in {normalize_title(item) for item in titles}:
                titles.append(title)
    return titles[:5]


def derived_citation(row: dict[str, Any]) -> str:
    authors = clean_space(row.get("authors"))
    year = clean_space(row.get("year"))
    title = clean_space(row.get("title") or row.get("draft_candidate_title"))
    journal = clean_space(row.get("journal"))
    doi = normalize_doi(row.get("doi"))
    pieces = []
    if authors:
        pieces.append(authors)
    if year:
        pieces.append(f"({year})")
    if title:
        pieces.append(title + ".")
    if journal:
        pieces.append(journal + ".")
    if doi:
        pieces.append("https://doi.org/" + doi)
    return " ".join(pieces)


def export_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, str]:
    doi = normalize_doi(row.get("doi") or row.get("draft_doi"))
    pmid = normalize_pmid(first_list_value(row.get("pmid")))
    if not pmid and clean_space(row.get("source")).lower() == "pubmed":
        pmid = normalize_pmid(row.get("paper_id"))
    status = clean_space(row.get("verification_status"))
    source = clean_space(row.get("source"))
    if not args.include_unverified and status and status not in VERIFIED_STATUSES:
        return {
            "key": citation_key(row),
            "title": clean_space(row.get("title") or row.get("draft_candidate_title")),
            "authors": clean_space(row.get("authors")),
            "year": clean_space(row.get("year")),
            "journal": clean_space(row.get("journal")),
            "doi": doi,
            "pmid": pmid,
            "url": clean_space(row.get("url")),
            "source": source,
            "verification_status": status,
            "export_status": "skipped_unverified",
            "exported_at": now_iso(),
            "notes": "Run identity verification and citation adjudication before official citation export.",
        }

    nbib = fetch_pubmed_export(pmid, args)
    bibtex = fetch_crossref_transform(doi, args, "application/x-bibtex")
    ris = fetch_crossref_transform(doi, args, "application/x-research-info-systems")
    citation_text = fetch_doi_content_negotiation(doi, args, f"text/x-bibliography; style={args.style}; locale=en-US")
    arxiv_id = extract_arxiv_id(row)
    arxiv_bibtex = fetch_arxiv_bibtex(arxiv_id, args)
    if not bibtex and arxiv_bibtex:
        bibtex = arxiv_bibtex
    preferred = []
    if nbib:
        preferred.append("pubmed_efetch")
    if bibtex or ris or citation_text:
        preferred.append("doi_crossref_or_content_negotiation")
    if arxiv_bibtex:
        preferred.append("arxiv_bibtex")

    expected_title = clean_space(row.get("title") or row.get("draft_candidate_title"))
    exported_titles = extract_structured_export_titles(nbib, bibtex, ris)
    title_mismatch_note = ""
    if exported_titles and expected_title and not any(title_matches(expected_title, title) for title in exported_titles):
        title_mismatch_note = (
            "Official export title did not match verified title; cleared automatic export and routed to manual check. "
            f"Export title(s): {' | '.join(exported_titles)}"
        )
        nbib = ""
        bibtex = ""
        ris = ""
        citation_text = ""
        preferred = []

    export_status = "exported_official" if preferred else "needs_manual_official_export"
    notes = ""
    if export_status != "exported_official" and args.allow_derived:
        citation_text = derived_citation(row)
        export_status = "derived_from_verified_metadata"
        notes = "No official export fetched; citation text was derived from verified metadata and needs manual confirmation."
    elif export_status != "exported_official":
        notes = title_mismatch_note or "No official citation-manager export fetched. Use the DOI/PubMed/publisher page to export manually."

    return {
        "key": citation_key(row),
        "title": clean_space(row.get("title") or row.get("draft_candidate_title")),
        "authors": clean_space(row.get("authors")),
        "year": clean_space(row.get("year")),
        "journal": clean_space(row.get("journal")),
        "doi": doi,
        "pmid": pmid,
        "url": clean_space(row.get("url")),
        "source": source,
        "verification_status": status,
        "preferred_export_source": ";".join(preferred),
        "citation_text": citation_text,
        "bibtex": bibtex,
        "ris": ris,
        "nbib_or_medline": nbib,
        "export_status": export_status,
        "exported_at": now_iso(),
        "notes": notes,
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_text_bundle(path: Path, rows: list[dict[str, str]], field: str) -> None:
    chunks = [clean_space(row.get(field)) for row in rows if clean_space(row.get(field))]
    path.write_text(("\n\n".join(chunks).rstrip() + "\n") if chunks else "", encoding="utf-8")


def write_report(path: Path, rows: list[dict[str, str]], args: argparse.Namespace, input_rows: int, duplicates: int) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        status = clean_space(row.get("export_status"))
        counts[status] = counts.get(status, 0) + 1
    lines = [
        "# Official Citation Export Report",
        "",
        f"- Input rows: {input_rows}",
        f"- Unique export rows: {len(rows)}",
        f"- Duplicate input rows collapsed: {duplicates}",
        f"- Dedupe mode: `{args.dedupe_by}`",
        f"- Style requested: `{args.style}`",
        f"- Offline mode: `{args.offline}`",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"- `{status}`: {count}")
    manual = [row for row in rows if row.get("export_status") == "needs_manual_official_export"]
    if manual:
        lines.extend(["", "## Manual Official Export Needed", ""])
        for row in manual[:200]:
            doi = row.get("doi")
            pmid = row.get("pmid")
            links = []
            if doi:
                links.append(f"https://doi.org/{doi}")
            if pmid:
                links.append(f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
            if row.get("url"):
                links.append(row["url"])
            lines.append(f"- {row.get('title') or row.get('key')}: {'; '.join(links) or 'no official link in row'}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def cmd_export(args: argparse.Namespace) -> int:
    rows = read_csv(Path(args.verified_csv))
    if args.max_records > 0:
        rows = rows[: args.max_records]
    input_rows = len(rows)
    rows, duplicates = dedupe_input_rows(rows, args.dedupe_by)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    exported = [export_row(row, args) for row in rows]
    write_csv(out_dir / "official_citations.csv", exported, EXPORT_FIELDS)
    write_text_bundle(out_dir / "official_citations.bib", exported, "bibtex")
    write_text_bundle(out_dir / "official_citations.ris", exported, "ris")
    write_text_bundle(out_dir / "official_citations.nbib", exported, "nbib_or_medline")
    write_text_bundle(out_dir / f"official_citations_{args.style}.txt", exported, "citation_text")
    with (out_dir / "official_citations.jsonl").open("w", encoding="utf-8") as handle:
        for row in exported:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_report(out_dir / "official_citation_export_report.md", exported, args, input_rows, len(duplicates))
    write_csv(out_dir / "official_citation_duplicate_report.csv", duplicates, ["dedupe_key", "kept_input_row", "duplicate_input_row", "kept_title", "duplicate_title", "kept_doi", "duplicate_doi", "kept_pmid", "duplicate_pmid"])
    summary = {
        "out_dir": str(out_dir),
        "input_rows": input_rows,
        "rows": len(exported),
        "unique_rows": len(exported),
        "duplicates_collapsed": len(duplicates),
        "dedupe_by": args.dedupe_by,
        "exported_official": sum(1 for row in exported if row.get("export_status") == "exported_official"),
        "manual_needed": sum(1 for row in exported if row.get("export_status") == "needs_manual_official_export"),
        "derived": sum(1 for row in exported if row.get("export_status") == "derived_from_verified_metadata"),
    }
    (out_dir / "official_citation_export_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export official citation-manager records for verified paper rows.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_export = sub.add_parser("export", help="Fetch official BibTeX/RIS/NBIB/citation text where available.")
    p_export.add_argument("--verified-csv", required=True, help="Verified/adjudicated paper CSV.")
    p_export.add_argument("--out-dir", default="./review-data/05_audit/official_citations")
    p_export.add_argument("--style", default="vancouver", help="CSL style for DOI content-negotiated bibliography text.")
    p_export.add_argument("--email", default="")
    p_export.add_argument("--api-key", default="", help="NCBI API key for PubMed EFetch, if available.")
    p_export.add_argument("--timeout", type=int, default=30)
    p_export.add_argument("--user-agent", default=DEFAULT_UA)
    p_export.add_argument("--max-records", type=int, default=0)
    p_export.add_argument("--offline", action="store_true", help="Do not call network APIs; produce manual/derived export status only.")
    p_export.add_argument("--allow-derived", action="store_true", help="When official export is unavailable, create derived citation text from verified metadata.")
    p_export.add_argument("--include-unverified", action="store_true", help="Allow export attempts for rows not marked verified.")
    p_export.add_argument("--dedupe-by", default="doi,pmid,title", choices=["none", "doi,title", "doi,pmid,title"], help="Collapse duplicate input rows before official export so row counts are not confused with unique paper counts.")
    p_export.set_defaults(func=cmd_export)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
