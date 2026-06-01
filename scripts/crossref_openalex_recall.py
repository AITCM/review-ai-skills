#!/usr/bin/env python3
"""Crossref/OpenAlex fallback recall and metadata verification.

Use this when `paper-search` is unavailable or when draft references need DOI,
publication metadata, and normalized citations. Outputs follow the same shape
as paper_recall.py/pubmed_recall.py for literature_pool.py import-recall.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


CROSSREF = "https://api.crossref.org"
OPENALEX = "https://api.openalex.org"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(v) for v in value if v is not None)
    return str(value).strip()


def normalize_doi(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    return text.strip().rstrip(".,;)")


def normalize_title(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", "", text)
    return text.strip()


def paper_key(row: dict[str, str]) -> str:
    doi = normalize_doi(row.get("doi"))
    if doi:
        return "doi-" + re.sub(r"[^a-z0-9]+", "-", doi).strip("-")[:90]
    source = row.get("source", "")
    paper_id = row.get("paper_id", "")
    if paper_id:
        return re.sub(r"[^a-z0-9]+", "-", f"{source}-{paper_id}".lower()).strip("-")[:100]
    digest = hashlib.sha1((row.get("title", "") + row.get("year", "")).encode("utf-8")).hexdigest()[:12]
    return f"work-{digest}"


def dedupe_key(row: dict[str, str]) -> str:
    doi = normalize_doi(row.get("doi"))
    if doi:
        return "doi:" + doi
    title = normalize_title(row.get("title"))
    year = normalize_text(row.get("year"))
    return f"title:{title}|year:{year}"


def http_json(url: str, timeout: int, user_agent: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def load_lines(path: str | None) -> list[str]:
    if not path:
        return []
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def load_inputs(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    queries = [q.strip() for q in args.query if q.strip()] + load_lines(args.query_file)
    dois = [normalize_doi(d) for d in args.doi if normalize_doi(d)] + [normalize_doi(d) for d in load_lines(args.doi_file)]
    return list(dict.fromkeys(queries)), list(dict.fromkeys([d for d in dois if d]))


def date_parts_year(value: Any) -> str:
    parts = ((value or {}).get("date-parts") or [])
    if parts and parts[0]:
        return str(parts[0][0])
    return ""


def crossref_authors(item: dict[str, Any]) -> str:
    authors = []
    for author in item.get("author") or []:
        name = " ".join(part for part in [author.get("given"), author.get("family")] if part)
        if name:
            authors.append(name)
    return "; ".join(authors)


def crossref_year(item: dict[str, Any]) -> str:
    for key in ["published-print", "published-online", "published", "issued", "created"]:
        year = date_parts_year(item.get(key))
        if year:
            return year
    return ""


def crossref_to_row(item: dict[str, Any], recall_query: str) -> dict[str, str]:
    title = normalize_text((item.get("title") or [""])[0] if isinstance(item.get("title"), list) else item.get("title"))
    doi = normalize_doi(item.get("DOI"))
    venue = normalize_text((item.get("container-title") or [""])[0] if isinstance(item.get("container-title"), list) else item.get("container-title"))
    row = {
        "title": title,
        "authors": crossref_authors(item),
        "year": crossref_year(item),
        "journal": venue,
        "source": "crossref",
        "paper_id": doi or normalize_text(item.get("URL")),
        "doi": doi,
        "url": normalize_text(item.get("URL")) or (f"https://doi.org/{doi}" if doi else ""),
        "pdf_url": "",
        "abstract": re.sub(r"<[^>]+>", "", normalize_text(item.get("abstract"))),
        "publication_type": normalize_text(item.get("type")),
        "recall_queries": recall_query,
        "screening_status": "unscreened",
        "screening_reason": "",
        "claim_supported": "",
        "limitations": "",
        "use_in_review": "",
        "notes": "",
    }
    row["key"] = paper_key(row)
    return row


def openalex_authors(item: dict[str, Any]) -> str:
    names = []
    for authorship in item.get("authorships") or []:
        author = authorship.get("author") or {}
        if author.get("display_name"):
            names.append(author["display_name"])
    return "; ".join(names)


def openalex_abstract(item: dict[str, Any]) -> str:
    inverted = item.get("abstract_inverted_index") or {}
    if not inverted:
        return ""
    pairs: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        for pos in positions:
            pairs.append((int(pos), word))
    return " ".join(word for _, word in sorted(pairs))


def openalex_to_row(item: dict[str, Any], recall_query: str) -> dict[str, str]:
    location = item.get("primary_location") or {}
    source = location.get("source") or {}
    pdf_url = normalize_text((location.get("pdf_url") or "") or ((item.get("open_access") or {}).get("oa_url") or ""))
    doi = normalize_doi(item.get("doi"))
    row = {
        "title": normalize_text(item.get("title") or item.get("display_name")),
        "authors": openalex_authors(item),
        "year": normalize_text(item.get("publication_year")),
        "journal": normalize_text(source.get("display_name")),
        "source": "openalex",
        "paper_id": normalize_text(item.get("id")),
        "doi": doi,
        "url": normalize_text(item.get("id")) or (f"https://doi.org/{doi}" if doi else ""),
        "pdf_url": pdf_url,
        "abstract": openalex_abstract(item),
        "publication_type": normalize_text(item.get("type")),
        "recall_queries": recall_query,
        "screening_status": "unscreened",
        "screening_reason": "",
        "claim_supported": "",
        "limitations": "",
        "use_in_review": "",
        "notes": "",
    }
    row["key"] = paper_key(row)
    return row


def crossref_search(query: str, args: argparse.Namespace) -> tuple[list[dict[str, str]], str]:
    params = {
        "query.bibliographic": query,
        "rows": str(args.max_results),
        "select": "DOI,title,author,issued,published,published-print,published-online,container-title,type,URL,abstract",
    }
    if args.from_year:
        params["filter"] = f"from-pub-date:{args.from_year}"
        if args.to_year:
            params["filter"] += f",until-pub-date:{args.to_year}"
    elif args.to_year:
        params["filter"] = f"until-pub-date:{args.to_year}"
    if args.email:
        params["mailto"] = args.email
    url = f"{CROSSREF}/works?{urllib.parse.urlencode(params)}"
    if args.dry_run:
        return [], url
    data = http_json(url, args.timeout, args.user_agent)
    items = ((data.get("message") or {}).get("items") or [])
    return [crossref_to_row(item, query) for item in items], url


def crossref_doi(doi: str, args: argparse.Namespace) -> tuple[list[dict[str, str]], str]:
    url = f"{CROSSREF}/works/{urllib.parse.quote(doi)}"
    if args.email:
        url += "?" + urllib.parse.urlencode({"mailto": args.email})
    if args.dry_run:
        return [], url
    data = http_json(url, args.timeout, args.user_agent)
    item = data.get("message")
    return [crossref_to_row(item, doi)] if item else [], url


def openalex_search(query: str, args: argparse.Namespace) -> tuple[list[dict[str, str]], str]:
    params = {
        "search": query,
        "per-page": str(args.max_results),
    }
    filters = []
    if args.from_year:
        filters.append(f"from_publication_date:{args.from_year}-01-01")
    if args.to_year:
        filters.append(f"to_publication_date:{args.to_year}-12-31")
    if filters:
        params["filter"] = ",".join(filters)
    if args.email:
        params["mailto"] = args.email
    url = f"{OPENALEX}/works?{urllib.parse.urlencode(params)}"
    if args.dry_run:
        return [], url
    data = http_json(url, args.timeout, args.user_agent)
    return [openalex_to_row(item, query) for item in data.get("results") or []], url


def openalex_doi(doi: str, args: argparse.Namespace) -> tuple[list[dict[str, str]], str]:
    # OpenAlex supports DOI lookup as /works/https://doi.org/{doi}.
    url = f"{OPENALEX}/works/{urllib.parse.quote('https://doi.org/' + doi, safe='')}"
    if args.email:
        url += "?" + urllib.parse.urlencode({"mailto": args.email})
    if args.dry_run:
        return [], url
    data = http_json(url, args.timeout, args.user_agent)
    return [openalex_to_row(data, doi)] if data else [], url


def merge_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: dict[str, dict[str, str]] = {}
    for row in rows:
        key = dedupe_key(row)
        if key in deduped:
            existing = deduped[key]
            for field in ["source", "recall_queries"]:
                parts = {
                    p.strip()
                    for p in (existing.get(field, "") + ";" + row.get(field, "")).split(";")
                    if p.strip()
                }
                existing[field] = "; ".join(sorted(parts))
            for field, value in row.items():
                if not existing.get(field) and value:
                    existing[field] = value
        else:
            deduped[key] = row
    return sorted(deduped.values(), key=lambda r: (r.get("year", ""), r.get("title", "")), reverse=True)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def md_escape(text: str) -> str:
    return normalize_text(text).replace("|", "\\|").replace("\n", " ")


def write_matrix(path: Path, rows: list[dict[str, str]]) -> None:
    lines = ["| Key | Title | Year | Source | DOI/URL | Recall |", "|---|---|---:|---|---|---|"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                md_escape(value)
                for value in [
                    row.get("key", ""),
                    row.get("title", ""),
                    row.get("year", ""),
                    row.get("source", ""),
                    row.get("doi") or row.get("url", ""),
                    row.get("recall_queries", ""),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, log: dict[str, Any], rows: list[dict[str, str]]) -> None:
    lines = [
        "# Crossref/OpenAlex Recall Summary",
        "",
        f"- Date run: {log['date_run']}",
        f"- Queries: {len(log['queries'])}",
        f"- DOI lookups: {len(log['dois'])}",
        f"- Raw rows: {log['raw_total']}",
        f"- Deduplicated total: {len(rows)}",
        "",
    ]
    if log["errors"]:
        lines.extend(["## Errors", ""])
        lines.extend(f"- `{e['input']}` via `{e['source']}`: {e['error']}" for e in log["errors"])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Recall and verify scholarly metadata through Crossref/OpenAlex.")
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--query-file")
    parser.add_argument("--doi", action="append", default=[])
    parser.add_argument("--doi-file")
    parser.add_argument("--sources", default="crossref,openalex")
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--from-year")
    parser.add_argument("--to-year")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--email", default="")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep", type=float, default=0.34)
    parser.add_argument("--user-agent", default="top-journal-review-writer/1.0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    queries, dois = load_inputs(args)
    if not queries and not dois:
        parser.error("Provide at least one --query/--query-file or --doi/--doi-file.")
    sources = {s.strip().lower() for s in args.sources.split(",") if s.strip()}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = {
        "schema_version": 1,
        "date_run": now_iso(),
        "queries": queries,
        "dois": dois,
        "sources": sorted(sources),
        "raw_total": 0,
        "calls": [],
        "errors": [],
        "dry_run": args.dry_run,
    }
    all_rows: list[dict[str, str]] = []

    for query in queries:
        for source in sorted(sources):
            try:
                if source == "crossref":
                    rows, url = crossref_search(query, args)
                elif source == "openalex":
                    rows, url = openalex_search(query, args)
                else:
                    continue
                log["calls"].append({"source": source, "input": query, "url": url, "count": len(rows)})
                all_rows.extend(rows)
                log["raw_total"] += len(rows)
                time.sleep(args.sleep)
            except Exception as exc:
                log["errors"].append({"source": source, "input": query, "error": repr(exc)})
    for doi in dois:
        for source in sorted(sources):
            try:
                if source == "crossref":
                    rows, url = crossref_doi(doi, args)
                elif source == "openalex":
                    rows, url = openalex_doi(doi, args)
                else:
                    continue
                log["calls"].append({"source": source, "input": doi, "url": url, "count": len(rows)})
                all_rows.extend(rows)
                log["raw_total"] += len(rows)
                time.sleep(args.sleep)
            except Exception as exc:
                log["errors"].append({"source": source, "input": doi, "error": repr(exc)})

    rows = merge_rows(all_rows)
    fields = [
        "key", "title", "authors", "year", "journal", "source", "paper_id", "doi", "url", "pdf_url",
        "abstract", "publication_type", "recall_queries", "screening_status", "screening_reason",
        "claim_supported", "limitations", "use_in_review", "notes",
    ]
    write_json(out_dir / "search_log.json", log)
    write_jsonl(out_dir / "papers.jsonl", rows)
    write_csv(out_dir / "papers.csv", rows, fields)
    write_csv(out_dir / "screening.csv", rows, fields)
    write_matrix(out_dir / "evidence_matrix.md", rows)
    write_summary(out_dir / "run_summary.md", log, rows)
    print(json.dumps({"out_dir": str(out_dir), "raw_total": log["raw_total"], "deduped_total": len(rows), "dry_run": args.dry_run}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
