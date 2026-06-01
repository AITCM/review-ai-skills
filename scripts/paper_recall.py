#!/usr/bin/env python3
"""Run repeatable literature recall through the paper-search CLI.

The script is intentionally stdlib-only. It calls the `paper-search` CLI, which
wraps the same library exposed by openags/paper-search-mcp's MCP server.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_SOURCES = "openalex,crossref,semantic,pubmed"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


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


def paper_key(paper: dict[str, Any]) -> str:
    doi = normalize_doi(paper.get("doi"))
    if doi:
        return "doi-" + re.sub(r"[^a-z0-9]+", "-", doi).strip("-")[:80]
    paper_id = normalize_text(paper.get("paper_id") or paper.get("id"))
    source = normalize_text(paper.get("source"))
    if source and paper_id:
        raw = f"{source}:{paper_id}".lower()
        return "id-" + re.sub(r"[^a-z0-9]+", "-", raw).strip("-")[:80]
    title = normalize_title(paper.get("title"))
    year = normalize_text(paper.get("year") or paper.get("published") or paper.get("publication_year"))
    digest = hashlib.sha1(f"{title}|{year}".encode("utf-8")).hexdigest()[:12]
    return f"title-{digest}"


def dedupe_key(paper: dict[str, Any]) -> str:
    doi = normalize_doi(paper.get("doi"))
    if doi:
        return "doi:" + doi
    paper_id = normalize_text(paper.get("paper_id") or paper.get("id")).lower()
    source = normalize_text(paper.get("source")).lower()
    if source and paper_id:
        return f"id:{source}:{paper_id}"
    title = normalize_title(paper.get("title"))
    authors = normalize_title(paper.get("authors"))
    year = normalize_text(paper.get("year") or paper.get("publication_year"))
    return f"title:{title}|authors:{authors}|year:{year}"


def as_row(paper: dict[str, Any]) -> dict[str, str]:
    return {
        "key": paper_key(paper),
        "title": normalize_text(paper.get("title")),
        "authors": normalize_text(paper.get("authors")),
        "year": normalize_text(paper.get("year") or paper.get("publication_year") or paper.get("published")),
        "source": normalize_text(paper.get("source")),
        "paper_id": normalize_text(paper.get("paper_id") or paper.get("id")),
        "doi": normalize_doi(paper.get("doi")),
        "url": normalize_text(paper.get("url") or paper.get("paper_url") or paper.get("landing_page_url")),
        "pdf_url": normalize_text(paper.get("pdf_url") or paper.get("open_access_pdf")),
        "abstract": normalize_text(paper.get("abstract") or paper.get("summary")),
    }


def markdown_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def build_base_command(args: argparse.Namespace) -> list[str]:
    if args.paper_search_repo:
        return ["uv", "run", "--directory", args.paper_search_repo, "paper-search"]
    if args.paper_search_cmd:
        return shlex.split(args.paper_search_cmd)
    if shutil.which("paper-search"):
        return ["paper-search"]
    return ["paper-search"]


def load_queries(args: argparse.Namespace) -> list[str]:
    queries = [q.strip() for q in args.query if q.strip()]
    if args.query_file:
        path = Path(args.query_file)
        queries.extend(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    seen: set[str] = set()
    out: list[str] = []
    for query in queries:
        if query not in seen:
            seen.add(query)
            out.append(query)
    return out


def run_search(base: list[str], query: str, args: argparse.Namespace) -> dict[str, Any]:
    command = [
        *base,
        "search",
        query,
        "-n",
        str(args.max_results),
        "-s",
        args.sources,
    ]
    if args.year:
        command.extend(["-y", args.year])
    if args.dry_run:
        return {
            "query": query,
            "command": command,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "json": {"papers": [], "total": 0, "source_results": {}},
        }
    completed = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=args.timeout,
    )
    parsed: dict[str, Any]
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError:
        parsed = {"parse_error": "stdout was not valid JSON", "stdout": completed.stdout}
    return {
        "query": query,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "json": parsed,
    }


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


def write_matrix(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["key", "title", "year", "source", "doi", "url", "paper_id"]
    lines = [
        "| Key | Title | Year | Source | DOI | URL | Paper ID |",
        "|---|---|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(markdown_escape(row.get(field, "")) for field in fields)
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, log: dict[str, Any], rows: list[dict[str, str]]) -> None:
    source_counts: dict[str, int] = {}
    for row in rows:
        source_counts[row.get("source", "")] = source_counts.get(row.get("source", ""), 0) + 1
    lines = [
        "# Literature Recall Summary",
        "",
        f"- Date run: {log['date_run']}",
        f"- Sources requested: `{log['sources']}`",
        f"- Max results per source: {log['max_results']}",
        f"- Year filter: {log.get('year') or 'none'}",
        f"- Query variants: {len(log['queries'])}",
        f"- Raw total: {log['raw_total']}",
        f"- Deduplicated total: {len(rows)}",
        "",
        "## Source Counts",
        "",
    ]
    for source, count in sorted(source_counts.items()):
        lines.append(f"- {source or 'unknown'}: {count}")
    if log["errors"]:
        lines.extend(["", "## Errors", ""])
        for item in log["errors"]:
            lines.append(f"- `{item['query']}` / `{item['source']}`: {item['error']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repeatable paper-search recall.")
    parser.add_argument("--query", action="append", default=[], help="Search query; repeat for variants.")
    parser.add_argument("--query-file", help="UTF-8 text file with one query per line.")
    parser.add_argument("--sources", default=DEFAULT_SOURCES, help="Comma-separated paper-search sources.")
    parser.add_argument("--max-results", type=int, default=10, help="Max results per source per query.")
    parser.add_argument("--year", help="Semantic Scholar year filter, e.g. <start-year>-<end-year>.")
    parser.add_argument("--out-dir", required=True, help="Directory for recall outputs.")
    parser.add_argument("--paper-search-cmd", help="Command prefix, e.g. 'python -m paper_search_mcp.cli'.")
    parser.add_argument("--paper-search-repo", help="Local openags/paper-search-mcp clone for uv run.")
    parser.add_argument("--timeout", type=int, default=120, help="Seconds per query.")
    parser.add_argument("--dry-run", action="store_true", help="Write planned commands without calling paper-search.")
    args = parser.parse_args()

    queries = load_queries(args)
    if not queries:
        parser.error("Provide at least one --query or --query-file entry.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = build_base_command(args)

    runs: list[dict[str, Any]] = []
    raw_papers: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for query in queries:
        run = run_search(base, query, args)
        runs.append({k: v for k, v in run.items() if k != "stdout"})
        payload = run.get("json") or {}
        if run["returncode"] != 0:
            errors.append({"query": query, "source": "command", "error": run.get("stderr", "")})
        if "errors" in payload and isinstance(payload["errors"], dict):
            for source, message in payload["errors"].items():
                errors.append({"query": query, "source": str(source), "error": str(message)})
        papers = payload.get("papers") if isinstance(payload, dict) else []
        if isinstance(papers, list):
            for paper in papers:
                if isinstance(paper, dict):
                    paper["_recall_query"] = query
                    raw_papers.append(paper)

    deduped: dict[str, dict[str, str]] = {}
    for paper in raw_papers:
        row = as_row(paper)
        row["recall_queries"] = normalize_text(paper.get("_recall_query"))
        key = dedupe_key(paper)
        if key in deduped:
            existing = deduped[key]
            queries_seen = {
                q.strip()
                for q in (existing.get("recall_queries", "") + ";" + row["recall_queries"]).split(";")
                if q.strip()
            }
            existing["recall_queries"] = "; ".join(sorted(queries_seen))
            sources_seen = {
                s.strip()
                for s in (existing.get("source", "") + ";" + row["source"]).split(";")
                if s.strip()
            }
            existing["source"] = "; ".join(sorted(sources_seen))
            continue
        deduped[key] = row

    rows = sorted(deduped.values(), key=lambda r: (r.get("year", ""), r.get("title", "")), reverse=True)
    log = {
        "schema_version": 1,
        "date_run": now_iso(),
        "base_command": base,
        "queries": queries,
        "sources": args.sources,
        "max_results": args.max_results,
        "year": args.year,
        "dry_run": args.dry_run,
        "runs": runs,
        "raw_total": len(raw_papers),
        "deduped_total": len(rows),
        "errors": errors,
    }

    write_json(out_dir / "search_log.json", log)
    write_jsonl(out_dir / "papers.jsonl", rows)
    write_csv(out_dir / "papers.csv", rows, list(rows[0].keys()) if rows else ["key", "title", "year", "source"])
    screening_rows = [
        {
            **row,
            "screening_status": "unscreened",
            "screening_reason": "",
            "claim_supported": "",
            "limitations": "",
            "use_in_review": "",
            "notes": "",
        }
        for row in rows
    ]
    write_csv(
        out_dir / "screening.csv",
        screening_rows,
        [
            "key",
            "screening_status",
            "screening_reason",
            "title",
            "authors",
            "year",
            "source",
            "doi",
            "url",
            "paper_id",
            "claim_supported",
            "limitations",
            "use_in_review",
            "notes",
        ],
    )
    write_matrix(out_dir / "evidence_matrix.md", rows)
    write_summary(out_dir / "run_summary.md", log, rows)

    print(json.dumps({"out_dir": str(out_dir), "raw_total": len(raw_papers), "deduped_total": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
