#!/usr/bin/env python3
"""arXiv/OpenReview fallback recall for preprints and review submissions."""

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
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ARXIV = "https://export.arxiv.org/api/query"
OPENREVIEW = "https://api2.openreview.net"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(v) for v in value if v is not None)
    if isinstance(value, dict) and "value" in value:
        return normalize_text(value["value"])
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
    paper_id = row.get("paper_id", "")
    if paper_id:
        return re.sub(r"[^a-z0-9]+", "-", f"{row.get('source')}-{paper_id}".lower()).strip("-")[:100]
    digest = hashlib.sha1((row.get("title", "") + row.get("year", "")).encode("utf-8")).hexdigest()[:12]
    return f"preprint-{digest}"


def dedupe_key(row: dict[str, str]) -> str:
    doi = normalize_doi(row.get("doi"))
    if doi:
        return "doi:" + doi
    if row.get("source") and row.get("paper_id"):
        return f"id:{row['source']}:{row['paper_id']}"
    return f"title:{normalize_title(row.get('title'))}|year:{row.get('year', '')}"


def http_bytes(url: str, timeout: int, user_agent: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def http_json(url: str, timeout: int, user_agent: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def load_lines(path: str | None) -> list[str]:
    if not path:
        return []
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def load_inputs(args: argparse.Namespace) -> tuple[list[str], list[str], list[str]]:
    queries = list(dict.fromkeys([q.strip() for q in args.query if q.strip()] + load_lines(args.query_file)))
    arxiv_ids = list(dict.fromkeys([q.strip() for q in args.arxiv_id if q.strip()] + load_lines(args.arxiv_id_file)))
    openreview_ids = list(dict.fromkeys([q.strip() for q in args.openreview_id if q.strip()] + load_lines(args.openreview_id_file)))
    return queries, arxiv_ids, openreview_ids


def arxiv_year(value: str) -> str:
    match = re.search(r"(19|20)\d{2}", value)
    return match.group(0) if match else ""


def arxiv_row(entry: ET.Element, recall_query: str) -> dict[str, str]:
    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    entry_id = normalize_text(entry.findtext("a:id", default="", namespaces=ns))
    arxiv_id = entry_id.rstrip("/").split("/")[-1]
    pdf_url = ""
    for link in entry.findall("a:link", ns):
        if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
            pdf_url = link.attrib.get("href", "")
            break
    doi = normalize_doi(entry.findtext("arxiv:doi", default="", namespaces=ns))
    authors = "; ".join(normalize_text(author.findtext("a:name", default="", namespaces=ns)) for author in entry.findall("a:author", ns))
    row = {
        "title": " ".join(normalize_text(entry.findtext("a:title", default="", namespaces=ns)).split()),
        "authors": authors,
        "year": arxiv_year(entry.findtext("a:published", default="", namespaces=ns)),
        "journal": "arXiv",
        "source": "arxiv",
        "paper_id": arxiv_id,
        "doi": doi or (f"10.48550/arXiv.{arxiv_id}" if arxiv_id else ""),
        "url": entry_id,
        "pdf_url": pdf_url,
        "abstract": " ".join(normalize_text(entry.findtext("a:summary", default="", namespaces=ns)).split()),
        "publication_type": "preprint",
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


def arxiv_search(query: str, args: argparse.Namespace) -> tuple[list[dict[str, str]], str]:
    params = {
        "search_query": f"all:{query}",
        "start": "0",
        "max_results": str(args.max_results),
        "sortBy": args.arxiv_sort_by,
        "sortOrder": args.arxiv_sort_order,
    }
    url = ARXIV + "?" + urllib.parse.urlencode(params)
    if args.dry_run:
        return [], url
    data = http_bytes(url, args.timeout, args.user_agent)
    root = ET.fromstring(data)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    return [arxiv_row(entry, query) for entry in root.findall("a:entry", ns)], url


def arxiv_lookup(arxiv_id: str, args: argparse.Namespace) -> tuple[list[dict[str, str]], str]:
    params = {"id_list": arxiv_id, "max_results": "1"}
    url = ARXIV + "?" + urllib.parse.urlencode(params)
    if args.dry_run:
        return [], url
    data = http_bytes(url, args.timeout, args.user_agent)
    root = ET.fromstring(data)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    return [arxiv_row(entry, arxiv_id) for entry in root.findall("a:entry", ns)], url


def content_value(content: dict[str, Any], field: str) -> str:
    return normalize_text(content.get(field))


def openreview_year(note: dict[str, Any]) -> str:
    for key in ["pdate", "odate", "cdate", "tmdate", "mdate"]:
        value = note.get(key)
        if isinstance(value, int) and value > 0:
            return str(dt.datetime.fromtimestamp(value / 1000, dt.timezone.utc).year)
    return ""


def openreview_row(note: dict[str, Any], recall_query: str) -> dict[str, str]:
    content = note.get("content") or {}
    note_id = note.get("id", "")
    venue = content_value(content, "venue") or content_value(content, "venueid") or normalize_text(note.get("invitation"))
    authors_raw = content.get("authors")
    if isinstance(authors_raw, dict) and "value" in authors_raw:
        authors_raw = authors_raw["value"]
    authors = normalize_text(authors_raw)
    row = {
        "title": content_value(content, "title"),
        "authors": authors,
        "year": openreview_year(note),
        "journal": venue,
        "source": "openreview",
        "paper_id": note_id,
        "doi": "",
        "url": f"https://openreview.net/forum?id={note_id}" if note_id else "",
        "pdf_url": f"https://openreview.net/pdf?id={note_id}" if note_id else "",
        "abstract": content_value(content, "abstract"),
        "publication_type": "review_platform_submission",
        "recall_queries": recall_query,
        "screening_status": "unscreened",
        "screening_reason": "",
        "claim_supported": "",
        "limitations": "",
        "use_in_review": "",
        "notes": content_value(content, "decision") or content_value(content, "venue"),
    }
    row["key"] = paper_key(row)
    return row


def openreview_search(query: str, args: argparse.Namespace) -> tuple[list[dict[str, str]], str]:
    params = {
        "term": query,
        "content": args.openreview_content,
        "source": "forum",
        "limit": str(args.max_results),
        "sort": "tmdate:desc",
    }
    url = f"{OPENREVIEW}/notes/search?{urllib.parse.urlencode(params)}"
    if args.dry_run:
        return [], url
    data = http_json(url, args.timeout, args.user_agent)
    notes = data.get("notes") or []
    return [openreview_row(note, query) for note in notes], url


def openreview_lookup(note_id: str, args: argparse.Namespace) -> tuple[list[dict[str, str]], str]:
    url = f"{OPENREVIEW}/notes?{urllib.parse.urlencode({'id': note_id})}"
    if args.dry_run:
        return [], url
    data = http_json(url, args.timeout, args.user_agent)
    notes = data.get("notes") or []
    return [openreview_row(note, note_id) for note in notes], url


def merge_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: dict[str, dict[str, str]] = {}
    for row in rows:
        key = dedupe_key(row)
        if key in deduped:
            existing = deduped[key]
            for field in ["source", "recall_queries"]:
                parts = {p.strip() for p in (existing.get(field, "") + ";" + row.get(field, "")).split(";") if p.strip()}
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
    lines = ["| Key | Title | Year | Source | URL | Recall |", "|---|---|---:|---|---|---|"]
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
                    row.get("url", ""),
                    row.get("recall_queries", ""),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, log: dict[str, Any], rows: list[dict[str, str]]) -> None:
    lines = [
        "# arXiv/OpenReview Recall Summary",
        "",
        f"- Date run: {log['date_run']}",
        f"- Queries: {len(log['queries'])}",
        f"- arXiv IDs: {len(log['arxiv_ids'])}",
        f"- OpenReview IDs: {len(log['openreview_ids'])}",
        f"- Raw rows: {log['raw_total']}",
        f"- Deduplicated total: {len(rows)}",
        "",
    ]
    if log["errors"]:
        lines.extend(["## Errors", ""])
        lines.extend(f"- `{e['input']}` via `{e['source']}`: {e['error']}" for e in log["errors"])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Recall arXiv and OpenReview preprints/submissions.")
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--query-file")
    parser.add_argument("--arxiv-id", action="append", default=[])
    parser.add_argument("--arxiv-id-file")
    parser.add_argument("--openreview-id", action="append", default=[])
    parser.add_argument("--openreview-id-file")
    parser.add_argument("--sources", default="arxiv,openreview")
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--arxiv-sort-by", default="relevance", choices=["relevance", "lastUpdatedDate", "submittedDate"])
    parser.add_argument("--arxiv-sort-order", default="descending", choices=["ascending", "descending"])
    parser.add_argument("--openreview-content", default="title", choices=["title", "abstract", "authors", "keywords", "all"])
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--user-agent", default="review-ai-skills/1.0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    queries, arxiv_ids, openreview_ids = load_inputs(args)
    if not queries and not arxiv_ids and not openreview_ids:
        parser.error("Provide at least one query or ID.")
    sources = {s.strip().lower() for s in args.sources.split(",") if s.strip()}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = {
        "schema_version": 1,
        "date_run": now_iso(),
        "queries": queries,
        "arxiv_ids": arxiv_ids,
        "openreview_ids": openreview_ids,
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
                if source == "arxiv":
                    rows, url = arxiv_search(query, args)
                elif source == "openreview":
                    rows, url = openreview_search(query, args)
                else:
                    continue
                log["calls"].append({"source": source, "input": query, "url": url, "count": len(rows)})
                all_rows.extend(rows)
                log["raw_total"] += len(rows)
                time.sleep(args.sleep)
            except Exception as exc:
                log["errors"].append({"source": source, "input": query, "error": repr(exc)})
    for arxiv_id in arxiv_ids:
        try:
            rows, url = arxiv_lookup(arxiv_id, args)
            log["calls"].append({"source": "arxiv", "input": arxiv_id, "url": url, "count": len(rows)})
            all_rows.extend(rows)
            log["raw_total"] += len(rows)
            time.sleep(args.sleep)
        except Exception as exc:
            log["errors"].append({"source": "arxiv", "input": arxiv_id, "error": repr(exc)})
    for openreview_id in openreview_ids:
        try:
            rows, url = openreview_lookup(openreview_id, args)
            log["calls"].append({"source": "openreview", "input": openreview_id, "url": url, "count": len(rows)})
            all_rows.extend(rows)
            log["raw_total"] += len(rows)
            time.sleep(args.sleep)
        except Exception as exc:
            log["errors"].append({"source": "openreview", "input": openreview_id, "error": repr(exc)})

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
