#!/usr/bin/env python3
"""Audit Deep Research drafts for paper vs non-paper sources and verify papers.

This script treats GPT/Gemini/other Deep Research drafts as useful but untrusted
inputs. It extracts likely references, URLs, DOIs, headings, and claim-like
sentences, then verifies paper references through public metadata APIs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


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


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        return read_docx_text(path)
    return path.read_text(encoding="utf-8", errors="replace")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_doi(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    return text.rstrip(".,;)")


def extract_dois(text: str) -> list[str]:
    pattern = r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b"
    seen: set[str] = set()
    out: list[str] = []
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        doi = normalize_doi(match.group(0))
        if doi and doi not in seen:
            seen.add(doi)
            out.append(doi)
    return out


def extract_urls(text: str) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for match in re.finditer(r"https?://[^\s<>\]\)\"']+", text):
        url = match.group(0).rstrip(".,;")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def source_kind(url: str) -> str:
    host = urllib.parse.urlparse(url).netloc.lower()
    if "doi.org" in host:
        return "doi"
    if any(x in host for x in ["pubmed", "ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov"]):
        return "biomedical_index"
    if "arxiv.org" in host:
        return "preprint"
    if any(x in host for x in ["nature.com", "science.org", "cell.com", "springer", "wiley", "elsevier", "sciencedirect", "tandfonline", "bmj.com", "thelancet.com"]):
        return "publisher"
    if any(x in host for x in ["github.com", "huggingface.co", "openai.com", "googleblog.com", "medium.com", "substack.com", "wordpress", "blog"]):
        return "web_or_blog"
    return "web"


def reference_region(text: str) -> str:
    patterns = [r"参考文献", r"參考文獻", r"references", r"bibliography"]
    lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, lower, flags=re.IGNORECASE)
        if match:
            return text[match.end() :]
    return text


def split_reference_lines(text: str) -> list[str]:
    region = reference_region(text)
    lines = [line.strip() for line in region.splitlines() if line.strip()]
    refs: list[str] = []
    buffer = ""
    for line in lines:
        if re.match(r"^(\[\d+\]|\d+[\.\)]|[（(]\d+[）)])\s+", line):
            if buffer:
                refs.append(buffer.strip())
            buffer = line
        elif buffer and len(line) > 20:
            buffer += " " + line
        elif len(line) > 35 and re.search(r"(19|20)\d{2}", line):
            if buffer:
                refs.append(buffer.strip())
            buffer = line
    if buffer:
        refs.append(buffer.strip())
    if not refs:
        refs = [line for line in lines if len(line) > 35 and re.search(r"(19|20)\d{2}", line)]
    return [re.sub(r"^\s*(\[\d+\]|\d+[\.\)]|[（(]\d+[）)])\s*", "", r).strip() for r in refs]


def title_guess(reference: str) -> str:
    quoted = re.findall(r"[\"“](.*?)[\"”]", reference)
    if quoted:
        return normalize_space(max(quoted, key=len))
    cleaned = re.sub(r"\bdoi\s*:\s*\S+", "", reference, flags=re.IGNORECASE)
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    parts = [p.strip() for p in re.split(r"\.\s+", cleaned) if p.strip()]
    candidates = [p for p in parts if len(p) > 12 and not re.match(r"^[A-Z][a-z]+,?\s+[A-Z]", p)]
    if candidates:
        return normalize_space(max(candidates[:3], key=len))
    return normalize_space(cleaned[:180])


def extract_headings(text: str) -> list[str]:
    headings: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^#{1,6}\s+", stripped):
            headings.append(re.sub(r"^#{1,6}\s+", "", stripped))
        elif re.match(r"^(\d+(\.\d+)*|[一二三四五六七八九十]+)[、.．]\s*\S+", stripped) and len(stripped) < 120:
            headings.append(stripped)
    return headings


def extract_claims(text: str, max_claims: int) -> list[str]:
    sentence_candidates = re.split(r"(?<=[。！？.!?])\s+", normalize_space(text))
    markers = ["suggest", "indicate", "show", "demonstrate", "argue", "therefore", "however", "表明", "说明", "提示", "因此", "然而", "挑战", "机会", "框架"]
    claims: list[str] = []
    for sent in sentence_candidates:
        if 40 <= len(sent) <= 260 and any(marker.lower() in sent.lower() for marker in markers):
            claims.append(sent)
        if len(claims) >= max_claims:
            break
    return claims


def http_json(url: str, timeout: int, user_agent: str) -> dict[str, Any] | None:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception:
        return None


def verify_crossref_by_doi(doi: str, timeout: int, mailto: str) -> dict[str, Any] | None:
    ua = f"top-journal-review-writer/1.0 (mailto:{mailto})" if mailto else "top-journal-review-writer/1.0"
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    data = http_json(url, timeout=timeout, user_agent=ua)
    if data and data.get("status") == "ok":
        return data.get("message")
    return None


def verify_crossref_by_title(title: str, timeout: int, mailto: str) -> dict[str, Any] | None:
    if not title:
        return None
    ua = f"top-journal-review-writer/1.0 (mailto:{mailto})" if mailto else "top-journal-review-writer/1.0"
    params = urllib.parse.urlencode({"query.bibliographic": title, "rows": 1})
    url = f"https://api.crossref.org/works?{params}"
    data = http_json(url, timeout=timeout, user_agent=ua)
    items = (((data or {}).get("message") or {}).get("items") or [])
    return items[0] if items else None


def verify_openalex_by_title(title: str, timeout: int, mailto: str) -> dict[str, Any] | None:
    if not title:
        return None
    params = {"search": title, "per-page": "1"}
    if mailto:
        params["mailto"] = mailto
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    data = http_json(url, timeout=timeout, user_agent="top-journal-review-writer/1.0")
    results = (data or {}).get("results") or []
    return results[0] if results else None


def authors_crossref(item: dict[str, Any]) -> str:
    authors = []
    for author in item.get("author") or []:
        name = " ".join(part for part in [author.get("given"), author.get("family")] if part)
        if name:
            authors.append(name)
    return "; ".join(authors)


def year_crossref(item: dict[str, Any]) -> str:
    for field in ["published-print", "published-online", "published", "issued"]:
        parts = ((item.get(field) or {}).get("date-parts") or [])
        if parts and parts[0]:
            return str(parts[0][0])
    return ""


def apa_citation(title: str, authors: str, year: str, venue: str, doi: str) -> str:
    author_text = authors or "Unknown author"
    year_text = year or "n.d."
    venue_text = f" {venue}." if venue else ""
    doi_text = f" https://doi.org/{doi}" if doi else ""
    return f"{author_text} ({year_text}). {title}.{venue_text}{doi_text}".strip()


def bibtex_key(authors: str, year: str, title: str) -> str:
    first = "unknown"
    if authors:
        first = re.split(r"[;,\s]+", authors.strip())[0].lower()
    title_word = "paper"
    words = re.findall(r"[A-Za-z0-9]+", title)
    if words:
        title_word = words[0].lower()
    return re.sub(r"[^a-z0-9]+", "", f"{first}{year or 'nd'}{title_word}")


def row_from_crossref(reference: str, title: str, doi: str, item: dict[str, Any] | None) -> dict[str, str]:
    if not item:
        return {
            "reference": reference,
            "kind": "paper_unverified",
            "verified": "no",
            "title": title,
            "authors": "",
            "year": "",
            "venue": "",
            "doi": doi,
            "url": "",
            "citation_apa": "",
            "bibtex_key": "",
            "verification_source": "",
            "notes": "No Crossref/OpenAlex match",
        }
    normalized_title = (item.get("title") or [title])[0] if isinstance(item.get("title"), list) else item.get("title", title)
    normalized_doi = normalize_doi(item.get("DOI") or doi)
    authors = authors_crossref(item)
    year = year_crossref(item)
    venue = ""
    for field in ["container-title", "publisher"]:
        value = item.get(field)
        if isinstance(value, list) and value:
            venue = value[0]
            break
        if isinstance(value, str):
            venue = value
            break
    return {
        "reference": reference,
        "kind": "paper",
        "verified": "yes",
        "title": normalized_title,
        "authors": authors,
        "year": year,
        "venue": venue,
        "doi": normalized_doi,
        "url": item.get("URL", ""),
        "citation_apa": apa_citation(normalized_title, authors, year, venue, normalized_doi),
        "bibtex_key": bibtex_key(authors, year, normalized_title),
        "verification_source": "crossref",
        "notes": "",
    }


def row_from_openalex(reference: str, title: str, item: dict[str, Any] | None) -> dict[str, str]:
    if not item:
        return {
            "reference": reference,
            "kind": "paper_unverified",
            "verified": "no",
            "title": title,
            "authors": "",
            "year": "",
            "venue": "",
            "doi": "",
            "url": "",
            "citation_apa": "",
            "bibtex_key": "",
            "verification_source": "",
            "notes": "No Crossref/OpenAlex match",
        }
    authors = []
    for authorship in item.get("authorships") or []:
        author = authorship.get("author") or {}
        if author.get("display_name"):
            authors.append(author["display_name"])
    normalized_title = item.get("title") or title
    year = str(item.get("publication_year") or "")
    venue = ((item.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
    doi = normalize_doi(item.get("doi") or "")
    return {
        "reference": reference,
        "kind": "paper",
        "verified": "yes",
        "title": normalized_title,
        "authors": "; ".join(authors),
        "year": year,
        "venue": venue,
        "doi": doi,
        "url": item.get("id") or "",
        "citation_apa": apa_citation(normalized_title, "; ".join(authors), year, venue, doi),
        "bibtex_key": bibtex_key("; ".join(authors), year, normalized_title),
        "verification_source": "openalex",
        "notes": "",
    }


def verify_reference(reference: str, timeout: int, mailto: str, offline: bool) -> dict[str, str]:
    dois = extract_dois(reference)
    title = title_guess(reference)
    urls = extract_urls(reference)
    url_kinds = {source_kind(url) for url in urls}
    if urls and not dois and url_kinds and url_kinds <= {"web", "web_or_blog"}:
        return {
            "reference": reference,
            "kind": "nonpaper_web",
            "verified": "no",
            "title": title,
            "authors": "",
            "year": "",
            "venue": "",
            "doi": "",
            "url": urls[0],
            "citation_apa": "",
            "bibtex_key": "",
            "verification_source": "",
            "notes": "Non-paper web/blog source; keep as background only",
        }
    if offline:
        return {
            "reference": reference,
            "kind": "paper_candidate",
            "verified": "not_checked",
            "title": title,
            "authors": "",
            "year": "",
            "venue": "",
            "doi": dois[0] if dois else "",
            "url": "",
            "citation_apa": "",
            "bibtex_key": "",
            "verification_source": "",
            "notes": "Offline mode; verify with API before citing",
        }
    if dois:
        item = verify_crossref_by_doi(dois[0], timeout=timeout, mailto=mailto)
        return row_from_crossref(reference, title, dois[0], item)
    item = verify_crossref_by_title(title, timeout=timeout, mailto=mailto)
    if item:
        return row_from_crossref(reference, title, "", item)
    openalex_item = verify_openalex_by_title(title, timeout=timeout, mailto=mailto)
    return row_from_openalex(reference, title, openalex_item)


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_report(path: Path, drafts: list[dict[str, Any]], refs: list[dict[str, str]], urls: list[dict[str, str]]) -> None:
    verified = [r for r in refs if r.get("verified") == "yes"]
    unverified = [r for r in refs if r.get("verified") != "yes"]
    web_sources = [u for u in urls if u["kind"] not in {"doi", "publisher", "biomedical_index", "preprint"}]
    lines = [
        "# Deep Research Draft Source Audit",
        "",
        f"- Date run: {now_iso()}",
        f"- Drafts: {len(drafts)}",
        f"- Reference-like entries: {len(refs)}",
        f"- Verified papers: {len(verified)}",
        f"- Unverified paper candidates: {len(unverified)}",
        f"- Non-paper/web sources: {len(web_sources)}",
        "",
        "## Draft Frames",
        "",
    ]
    for draft in drafts:
        lines.extend([f"### {draft['path']}", ""])
        if draft["headings"]:
            lines.append("Headings/framework:")
            lines.extend(f"- {heading}" for heading in draft["headings"][:30])
        if draft["claims"]:
            lines.extend(["", "Claim-like sentences:"])
            lines.extend(f"- {claim}" for claim in draft["claims"][:20])
        lines.append("")
    if web_sources:
        lines.extend(["## Non-Paper Sources To Replace Or Demote", ""])
        for item in web_sources[:100]:
            lines.append(f"- `{item['kind']}` {item['url']}")
    if unverified:
        lines.extend(["", "## Unverified Paper Candidates", ""])
        for item in unverified[:100]:
            lines.append(f"- {item.get('title') or item.get('reference')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Deep Research drafts and verify cited papers.")
    parser.add_argument("--draft", action="append", required=True, help="Draft file path; repeat for GPT/Gemini drafts.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--offline", action="store_true", help="Skip API verification; only extract and classify.")
    parser.add_argument("--mailto", default="", help="Email for polite Crossref/OpenAlex API use.")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--max-claims", type=int, default=40)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    drafts: list[dict[str, Any]] = []
    all_refs: list[str] = []
    url_rows: list[dict[str, str]] = []
    doi_rows: list[dict[str, str]] = []

    for draft in args.draft:
        path = Path(draft)
        text = read_text(path)
        refs = split_reference_lines(text)
        urls = extract_urls(text)
        dois = extract_dois(text)
        all_refs.extend(refs)
        for doi in dois:
            doi_rows.append({"draft": str(path), "doi": doi})
        for url in urls:
            url_rows.append({"draft": str(path), "url": url, "kind": source_kind(url)})
        drafts.append(
            {
                "path": str(path),
                "headings": extract_headings(text),
                "claims": extract_claims(text, max_claims=args.max_claims),
                "reference_count": len(refs),
                "url_count": len(urls),
                "doi_count": len(dois),
            }
        )

    seen: set[str] = set()
    unique_refs = []
    for ref in all_refs:
        key = normalize_space(ref).lower()
        if key not in seen:
            seen.add(key)
            unique_refs.append(ref)

    audited_reference_rows = [verify_reference(ref, timeout=args.timeout, mailto=args.mailto, offline=args.offline) for ref in unique_refs]
    verified_rows = [
        row
        for row in audited_reference_rows
        if row.get("kind") not in {"nonpaper_web", "web_or_blog", "web"}
    ]
    fields = [
        "reference",
        "kind",
        "verified",
        "title",
        "authors",
        "year",
        "venue",
        "doi",
        "url",
        "citation_apa",
        "bibtex_key",
        "verification_source",
        "notes",
    ]
    write_csv(out_dir / "verified_papers.csv", verified_rows, fields)
    write_csv(out_dir / "all_reference_audit.csv", audited_reference_rows, fields)
    write_csv(out_dir / "web_sources.csv", url_rows, ["draft", "url", "kind"])
    write_csv(out_dir / "dois.csv", doi_rows, ["draft", "doi"])
    paper_queries = [row.get("title", "") for row in verified_rows if row.get("title")]
    (out_dir / "paper_title_queries.txt").write_text("\n".join(paper_queries) + ("\n" if paper_queries else ""), encoding="utf-8")
    write_report(out_dir / "source_audit.md", drafts, audited_reference_rows, url_rows)
    (out_dir / "draft_frames.json").write_text(json.dumps(drafts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "drafts": len(drafts),
                "references": len(unique_refs),
                "paper_rows": len(verified_rows),
                "verified": sum(1 for row in verified_rows if row.get("verified") == "yes"),
                "web_sources": len(url_rows),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
