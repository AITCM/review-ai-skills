#!/usr/bin/env python3
"""PubMed E-utilities fallback recall for top-journal review projects.

Use this when `paper-search` is unavailable. It writes the same core output
shape as `paper_recall.py` so results can be imported into `literature_pool.py`.
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
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def load_queries(args: argparse.Namespace) -> list[str]:
    queries = [q.strip() for q in args.query if q.strip()]
    if args.query_file:
        path = Path(args.query_file)
        queries.extend(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    out: list[str] = []
    seen: set[str] = set()
    for query in queries:
        if query not in seen:
            seen.add(query)
            out.append(query)
    return out


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_doi(value: str) -> str:
    value = normalize_text(value).lower()
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value)
    value = re.sub(r"^doi:\s*", "", value)
    return value.rstrip(".,;")


def paper_key(paper: dict[str, str]) -> str:
    doi = normalize_doi(paper.get("doi", ""))
    if doi:
        return "doi-" + re.sub(r"[^a-z0-9]+", "-", doi).strip("-")[:90]
    pmid = paper.get("paper_id") or paper.get("pmid")
    if pmid:
        return "pmid-" + re.sub(r"[^0-9A-Za-z]+", "-", pmid).strip("-")
    digest = hashlib.sha1((paper.get("title", "") + paper.get("year", "")).encode("utf-8")).hexdigest()[:12]
    return f"pubmed-{digest}"


def http_get(url: str, timeout: int, user_agent: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def build_params(args: argparse.Namespace, extra: dict[str, str]) -> str:
    params = {
        "tool": args.tool,
        "retmode": "xml",
        **extra,
    }
    if args.email:
        params["email"] = args.email
    if args.api_key:
        params["api_key"] = args.api_key
    return urllib.parse.urlencode(params)


def esearch(query: str, args: argparse.Namespace) -> tuple[list[str], str]:
    term = query
    if args.mindate or args.maxdate:
        mindate = args.mindate or "1900"
        maxdate = args.maxdate or "3000"
        term = f"({term}) AND ({mindate}:{maxdate}[dp])"
    params = build_params(
        args,
        {
            "db": "pubmed",
            "term": term,
            "retmax": str(args.max_results),
            "sort": args.sort,
            "retstart": "0",
        },
    )
    url = f"{BASE}/esearch.fcgi?{params}"
    if args.dry_run:
        return [], url
    data = http_get(url, timeout=args.timeout, user_agent=args.user_agent)
    root = ET.fromstring(data)
    ids = [node.text or "" for node in root.findall(".//IdList/Id") if node.text]
    return ids, url


def efetch(pmids: list[str], args: argparse.Namespace) -> tuple[list[dict[str, str]], str]:
    if not pmids:
        return [], ""
    params = build_params(
        args,
        {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
        },
    )
    url = f"{BASE}/efetch.fcgi?{params}"
    if args.dry_run:
        return [], url
    data = http_get(url, timeout=args.timeout, user_agent=args.user_agent)
    root = ET.fromstring(data)
    papers = [parse_article(article) for article in root.findall(".//PubmedArticle")]
    return papers, url


def text_at(node: ET.Element | None, path: str) -> str:
    if node is None:
        return ""
    found = node.find(path)
    return "".join(found.itertext()).strip() if found is not None else ""


def article_year(article: ET.Element) -> str:
    for path in [
        ".//Article/Journal/JournalIssue/PubDate/Year",
        ".//Article/ArticleDate/Year",
        ".//PubMedPubDate[@PubStatus='pubmed']/Year",
        ".//PubMedPubDate[@PubStatus='entrez']/Year",
    ]:
        value = text_at(article, path)
        if value:
            return value
    medline = text_at(article, ".//Article/Journal/JournalIssue/PubDate/MedlineDate")
    match = re.search(r"(19|20)\d{2}", medline)
    return match.group(0) if match else ""


def article_authors(article: ET.Element) -> str:
    authors = []
    for author in article.findall(".//Article/AuthorList/Author"):
        collective = text_at(author, "CollectiveName")
        if collective:
            authors.append(collective)
            continue
        last = text_at(author, "LastName")
        fore = text_at(author, "ForeName")
        initials = text_at(author, "Initials")
        name = " ".join(part for part in [fore or initials, last] if part)
        if name:
            authors.append(name)
    return "; ".join(authors)


def article_doi(article: ET.Element) -> str:
    for node in article.findall(".//ArticleIdList/ArticleId"):
        if (node.attrib.get("IdType") or "").lower() == "doi" and node.text:
            return normalize_doi(node.text)
    for node in article.findall(".//ELocationID"):
        if (node.attrib.get("EIdType") or "").lower() == "doi" and node.text:
            return normalize_doi(node.text)
    return ""


def abstract_text(article: ET.Element) -> str:
    parts = []
    for node in article.findall(".//Article/Abstract/AbstractText"):
        label = node.attrib.get("Label") or node.attrib.get("NlmCategory") or ""
        text = " ".join("".join(node.itertext()).split())
        if text:
            parts.append(f"{label}: {text}" if label else text)
    return "\n".join(parts)


def parse_article(article: ET.Element) -> dict[str, str]:
    pmid = text_at(article, ".//MedlineCitation/PMID")
    title = " ".join(text_at(article, ".//Article/ArticleTitle").split())
    journal = text_at(article, ".//Article/Journal/Title") or text_at(article, ".//Article/Journal/ISOAbbreviation")
    doi = article_doi(article)
    paper = {
        "title": title,
        "authors": article_authors(article),
        "year": article_year(article),
        "source": "pubmed",
        "paper_id": pmid,
        "pmid": pmid,
        "doi": doi,
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
        "pdf_url": "",
        "abstract": abstract_text(article),
        "journal": journal,
        "publication_type": "; ".join(text_at(pt, ".") for pt in article.findall(".//PublicationTypeList/PublicationType") if text_at(pt, ".")),
    }
    paper["key"] = paper_key(paper)
    return paper


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


def markdown_escape(text: str) -> str:
    return normalize_text(text).replace("|", "\\|").replace("\n", " ")


def write_matrix(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "| Key | Title | Year | Journal | PMID | DOI | Relevance Note |",
        "|---|---|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                markdown_escape(row.get(field, ""))
                for field in ["key", "title", "year", "journal", "pmid", "doi", "recall_queries"]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, log: dict[str, Any], rows: list[dict[str, str]]) -> None:
    lines = [
        "# PubMed Recall Summary",
        "",
        f"- Date run: {log['date_run']}",
        f"- Query variants: {len(log['queries'])}",
        f"- Raw PMID hits fetched: {log['raw_total']}",
        f"- Deduplicated total: {len(rows)}",
        f"- Date filter: {log.get('mindate') or 'none'} to {log.get('maxdate') or 'none'}",
        "",
        "## Queries",
        "",
    ]
    lines.extend(f"- `{query}`" for query in log["queries"])
    if log.get("errors"):
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- `{item['query']}`: {item['error']}" for item in log["errors"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Recall PubMed papers through NCBI E-utilities.")
    parser.add_argument("--query", action="append", default=[], help="Search query; repeat for variants.")
    parser.add_argument("--query-file", help="UTF-8 text file with one query per line.")
    parser.add_argument("--max-results", type=int, default=20, help="Max PubMed IDs per query.")
    parser.add_argument("--mindate", help="Publication date start year/date, e.g. 2020 or 2020/01/01.")
    parser.add_argument("--maxdate", help="Publication date end year/date.")
    parser.add_argument("--sort", default="relevance", choices=["relevance", "pub date", "first author", "journal"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--email", default="", help="NCBI contact email, recommended.")
    parser.add_argument("--api-key", default="", help="NCBI API key, optional.")
    parser.add_argument("--tool", default="review-ai-skills")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep", type=float, default=0.34, help="Delay between API calls.")
    parser.add_argument("--user-agent", default="review-ai-skills/1.0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    queries = load_queries(args)
    if not queries:
        parser.error("Provide at least one --query or --query-file entry.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log: dict[str, Any] = {
        "schema_version": 1,
        "date_run": now_iso(),
        "queries": queries,
        "mindate": args.mindate,
        "maxdate": args.maxdate,
        "max_results": args.max_results,
        "sort": args.sort,
        "dry_run": args.dry_run,
        "calls": [],
        "errors": [],
        "raw_total": 0,
    }
    deduped: dict[str, dict[str, str]] = {}
    for query in queries:
        try:
            ids, search_url = esearch(query, args)
            log["calls"].append({"query": query, "type": "esearch", "url": search_url, "pmids": ids})
            if args.dry_run:
                continue
            time.sleep(args.sleep)
            papers, fetch_url = efetch(ids, args)
            log["calls"].append({"query": query, "type": "efetch", "url": fetch_url, "count": len(papers)})
            log["raw_total"] += len(papers)
            for paper in papers:
                key = paper.get("doi") or paper.get("pmid") or paper["key"]
                existing = deduped.get(key)
                if existing:
                    recall_queries = {
                        q.strip()
                        for q in (existing.get("recall_queries", "") + ";" + query).split(";")
                        if q.strip()
                    }
                    existing["recall_queries"] = "; ".join(sorted(recall_queries))
                else:
                    paper["recall_queries"] = query
                    paper["screening_status"] = "unscreened"
                    paper["screening_reason"] = ""
                    paper["claim_supported"] = ""
                    paper["limitations"] = ""
                    paper["use_in_review"] = ""
                    paper["notes"] = ""
                    deduped[key] = paper
            time.sleep(args.sleep)
        except Exception as exc:
            log["errors"].append({"query": query, "error": repr(exc)})

    rows = sorted(deduped.values(), key=lambda r: (r.get("year", ""), r.get("title", "")), reverse=True)
    fields = [
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
        "screening_status",
        "screening_reason",
        "claim_supported",
        "limitations",
        "use_in_review",
        "notes",
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
