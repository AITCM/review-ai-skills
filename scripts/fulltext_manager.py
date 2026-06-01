#!/usr/bin/env python3
"""Audit, import, and prepare full text for review literature pools.

The literature pool can contain verified metadata and abstracts, but detailed
review writing needs full text whenever a claim depends on methods, results,
limitations, or comparisons. This script makes that gate explicit:

- audit selected papers for usable full text
- generate a user handoff list for missing PDFs/text files
- import user-supplied PDFs/TXT/MD/DOCX into the pool
- extract text from PDFs when optional local libraries are available

It is intentionally stdlib-first. PDF extraction is best-effort with optional
`pypdf`, `PyPDF2`, or `fitz` (PyMuPDF) if installed.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import shutil
import sqlite3
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


SELECTED_STATUSES = {"citation_pool", "seminal", "method", "recent"}
SUPPORTED_FULLTEXT = {".pdf", ".txt", ".md", ".docx"}
EUROPE_PMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
MINERU_AGENT_BASE = "https://mineru.net/api/v1/agent"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(item) for item in value if item is not None)
    return str(value).strip()


def clean_space(value: Any) -> str:
    return re.sub(r"\s+", " ", normalize_text(value)).strip()


def normalize_doi(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    return text.strip().rstrip(".")


def normalize_title(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_set(value: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9\u4e00-\u9fff]{2,}", normalize_title(value)) if len(t) >= 2}


def safe_filename(value: Any, fallback: str = "paper") -> str:
    text = normalize_text(value) or fallback
    text = re.sub(r"[^A-Za-z0-9_.\-\u4e00-\u9fff]+", "-", text).strip("-")
    return text[:120] or fallback


def pool_paths(pool_dir: Path) -> dict[str, Path]:
    return {
        "pool": pool_dir / "pool.json",
        "evidence": pool_dir / "evidence",
        "contexts": pool_dir / "contexts",
        "logs": pool_dir / "logs",
        "user_fulltext": pool_dir / "user_fulltext",
        "indexes": pool_dir / "indexes",
    }


def ensure_dirs(pool_dir: Path) -> None:
    for key, path in pool_paths(pool_dir).items():
        if key != "pool":
            path.mkdir(parents=True, exist_ok=True)


def load_pool(pool_dir: Path) -> dict[str, Any]:
    ensure_dirs(pool_dir)
    path = pool_paths(pool_dir)["pool"]
    if not path.exists():
        raise SystemExit(f"Pool not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("papers", {})
    return data


def save_pool(pool_dir: Path, pool: dict[str, Any]) -> None:
    pool["updated_at"] = now_iso()
    pool_paths(pool_dir)["pool"].write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")


def selected_records(pool: dict[str, Any], statuses: set[str]) -> list[dict[str, Any]]:
    records = []
    for record in pool.get("papers", {}).values():
        if normalize_text(record.get("pool_status")) in statuses:
            records.append(record)
    records.sort(key=lambda r: (normalize_text(r.get("relevance_score")), normalize_text(r.get("year"))), reverse=True)
    return records


def record_dir(pool_dir: Path, record: dict[str, Any]) -> Path:
    key = normalize_text(record.get("key")) or safe_filename(record.get("title"))
    return pool_paths(pool_dir)["evidence"] / safe_filename(key)


def path_exists(value: Any) -> bool:
    text = normalize_text(value)
    return bool(text) and Path(text).exists()


def fulltext_text_path(record: dict[str, Any]) -> str:
    for field in ("fulltext_text_path", "fulltext_path", "text_path"):
        value = normalize_text(record.get(field))
        if value and Path(value).exists() and Path(value).suffix.lower() in {".txt", ".md"}:
            return value
    return ""


def pdf_path(record: dict[str, Any]) -> str:
    for field in ("fulltext_pdf_path", "local_pdf", "pdf_path", "fulltext_path"):
        value = normalize_text(record.get(field))
        if value and Path(value).exists() and Path(value).suffix.lower() == ".pdf":
            return value
    return ""


def fulltext_status(record: dict[str, Any]) -> str:
    if fulltext_text_path(record):
        return "fulltext_text_available"
    if pdf_path(record):
        return "pdf_available_text_missing"
    if normalize_text(record.get("abstract")) or path_exists(record.get("abstract_path")):
        return "abstract_only"
    return "missing_fulltext"


def suggested_filename(record: dict[str, Any]) -> str:
    doi = normalize_doi(record.get("doi"))
    if doi:
        return safe_filename("doi-" + doi.replace("/", "-")) + ".pdf"
    key = normalize_text(record.get("key"))
    title = normalize_text(record.get("title"))
    return safe_filename(key or title) + ".pdf"


def audit_rows(pool_dir: Path, records: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in records:
        status = fulltext_status(record)
        rows.append(
            {
                "key": normalize_text(record.get("key")),
                "title": normalize_text(record.get("title")),
                "doi": normalize_doi(record.get("doi")),
                "url": normalize_text(record.get("url") or record.get("pdf_url")),
                "pool_status": normalize_text(record.get("pool_status")),
                "fulltext_status": status,
                "fulltext_text_path": fulltext_text_path(record),
                "pdf_path": pdf_path(record),
                "suggested_filename": suggested_filename(record),
                "needed_action": needed_action(status, pool_dir),
            }
        )
    return rows


def needed_action(status: str, pool_dir: Path) -> str:
    if status == "fulltext_text_available":
        return "Ready for TreeRAG and detailed claim support."
    if status == "pdf_available_text_missing":
        return "Run fulltext_manager.py extract, or install pypdf/PyPDF2/PyMuPDF and rerun import."
    return f"Place PDF/TXT/MD/DOCX in {pool_paths(pool_dir)['user_fulltext']} and rerun import."


def write_audit_outputs(pool_dir: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    paths = pool_paths(pool_dir)
    contexts = paths["contexts"]
    logs = paths["logs"]
    missing = [row for row in rows if row["fulltext_status"] != "fulltext_text_available"]
    ready = [row for row in rows if row["fulltext_status"] == "fulltext_text_available"]
    script_path = Path(__file__).resolve().as_posix()

    handoff_path = contexts / "needs_user_fulltext.md"
    lines = [
        "# Full-Text Handoff For User",
        "",
        "These selected papers still need usable full text before detailed claim comparison, method/result extraction, or TreeRAG-based writing.",
        "",
        "User action:",
        "",
        f"1. Put PDFs, TXT, MD, or DOCX files into `{paths['user_fulltext']}`.",
        "2. Keep the DOI, citation key, or exact title in the filename when possible.",
        f"3. Run `python {script_path} import --pool-dir <POOL_DIR>`.",
        f"4. Re-run `python {script_path} audit --pool-dir <POOL_DIR>` before detailed drafting.",
        "",
        "| Key | Title | DOI/URL | Status | Needed action | Suggested filename |",
        "|---|---|---|---|---|---|",
    ]
    if not missing:
        lines.append("| - | All selected records have extracted full text. | - | ready | - | - |")
    for row in missing:
        doi_or_url = row["doi"] or row["url"]
        title = row["title"].replace("|", "\\|")
        action = row["needed_action"].replace("|", "\\|")
        lines.append(
            f"| `{row['key']}` | {title} | {doi_or_url} | {row['fulltext_status']} | {action} | `{row['suggested_filename']}` |"
        )
    handoff_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    csv_path = contexts / "missing_fulltext_manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "key",
            "title",
            "doi",
            "url",
            "pool_status",
            "fulltext_status",
            "fulltext_text_path",
            "pdf_path",
            "suggested_filename",
            "needed_action",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(missing)

    json_path = contexts / "fulltext_audit.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = contexts / "fulltext_audit.md"
    summary = [
        "# Full-Text Audit",
        "",
        f"- Selected records: {len(rows)}",
        f"- Ready with extracted full text: {len(ready)}",
        f"- Need user/full-text action: {len(missing)}",
        f"- Handoff: `{handoff_path}`",
        f"- Missing manifest: `{csv_path}`",
        "",
        "Do not use TreeRAG for detailed method/result/metric claims until the relevant paper is ready with extracted full text, or the user explicitly approves abstract-only drafting for that claim.",
    ]
    md_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

    log_path = logs / f"fulltext_audit_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path.write_text(json.dumps({"rows": rows, "missing": missing}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "selected": len(rows),
        "ready": len(ready),
        "missing": len(missing),
        "handoff_path": str(handoff_path),
        "manifest_csv": str(csv_path),
        "audit_md": str(md_path),
        "log": str(log_path),
    }


def read_docx(path: Path) -> str:
    parts: list[str] = []
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    for para in root.findall(".//w:p", ns):
        runs = [node.text or "" for node in para.findall(".//w:t", ns)]
        line = " ".join("".join(runs).split())
        if line:
            parts.append(line)
    return "\n".join(parts)


def extract_pdf_with_pypdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except Exception:
            return ""
    try:
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n\n".join(pages).strip()
    except Exception:
        return ""


def extract_pdf_with_fitz(path: Path) -> str:
    try:
        import fitz  # type: ignore
    except Exception:
        return ""
    try:
        doc = fitz.open(str(path))
        return "\n\n".join(page.get_text("text") for page in doc).strip()
    except Exception:
        return ""


def extract_text(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace"), "plain_text"
    if suffix == ".docx":
        try:
            return read_docx(path), "docx"
        except Exception as exc:
            return "", f"docx_extract_failed: {exc}"
    if suffix == ".pdf":
        text = extract_pdf_with_pypdf(path)
        if text:
            return text, "pdf:pypdf"
        text = extract_pdf_with_fitz(path)
        if text:
            return text, "pdf:pymupdf"
        return "", "pdf_text_extractor_unavailable_or_failed"
    return "", "unsupported"


def discover_user_files(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        return []
    return sorted(p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_FULLTEXT)


def match_score(record: dict[str, Any], path: Path) -> int:
    name = normalize_title(path.stem)
    score = 0
    key = normalize_title(record.get("key"))
    doi = normalize_doi(record.get("doi"))
    title = normalize_title(record.get("title"))
    if key and key in name:
        score += 5
    if doi:
        doi_safe = normalize_title(doi.replace("/", " "))
        if doi_safe and doi_safe in name:
            score += 8
    if title:
        if title[:60] and title[:60] in name:
            score += 8
        title_tokens = token_set(title)
        name_tokens = token_set(name)
        overlap = len(title_tokens & name_tokens)
        if overlap:
            score += min(overlap, 6)
    return score


def find_best_file(record: dict[str, Any], files: list[Path], used: set[Path], min_score: int) -> tuple[Path | None, int]:
    best: tuple[Path | None, int] = (None, 0)
    for path in files:
        if path in used:
            continue
        score = match_score(record, path)
        if score > best[1]:
            best = (path, score)
    if best[0] is not None and best[1] >= min_score:
        return best
    return None, best[1]


def import_file(pool_dir: Path, record: dict[str, Any], source: Path, copy: bool) -> dict[str, Any]:
    out_dir = record_dir(pool_dir, record)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / safe_filename(source.name, fallback="fulltext" + source.suffix.lower())
    if copy:
        shutil.copy2(source, dest)
    else:
        dest = source

    text, method = extract_text(dest)
    record["user_fulltext_file"] = str(source)
    record["fulltext_imported_at"] = now_iso()
    record["fulltext_extract_method"] = method
    if dest.suffix.lower() == ".pdf":
        record["local_pdf"] = str(dest)
        record["fulltext_pdf_path"] = str(dest)
    if text and len(text.strip()) >= 500:
        text_path = out_dir / "fulltext.txt"
        text_path.write_text(text, encoding="utf-8")
        record["fulltext_text_path"] = str(text_path)
        record["fulltext_path"] = str(text_path)
        record["fulltext_status"] = "fulltext_text_available"
    else:
        record["fulltext_status"] = "pdf_available_text_missing" if dest.suffix.lower() == ".pdf" else "text_too_short_or_unreadable"
    return {"key": record.get("key"), "source": str(source), "dest": str(dest), "method": method, "chars": len(text)}


def build_memory_index(pool_dir: Path, rows: list[dict[str, str]]) -> str:
    index_dir = pool_paths(pool_dir)["indexes"]
    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = index_dir / "fulltext_status.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE IF EXISTS fulltext_status")
    conn.execute(
        "CREATE TABLE fulltext_status (key TEXT, title TEXT, doi TEXT, status TEXT, text_path TEXT, pdf_path TEXT, needed_action TEXT)"
    )
    conn.executemany(
        "INSERT INTO fulltext_status VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                row["key"],
                row["title"],
                row["doi"],
                row["fulltext_status"],
                row["fulltext_text_path"],
                row["pdf_path"],
                row["needed_action"],
            )
            for row in rows
        ],
    )
    conn.commit()
    conn.close()
    return str(db_path)


def http_get_bytes(url: str, timeout: int) -> tuple[bytes, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "review-ai-skills/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("content-type", "")
        return response.read(), content_type


def http_get_json(url: str, timeout: int) -> dict[str, Any]:
    data, _content_type = http_get_bytes(url, timeout)
    return json.loads(data.decode("utf-8", errors="replace"))


def http_post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "review-ai-skills/0.1"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def put_file(url: str, path: Path, timeout: int) -> int:
    request = urllib.request.Request(url, data=path.read_bytes(), method="PUT")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status


def find_pmcid(record: dict[str, Any]) -> str:
    for field in ("pmcid", "pmc_id", "pmc"):
        value = normalize_text(record.get(field))
        match = re.search(r"PMC\d+", value, re.I)
        if match:
            return match.group(0).upper()
    for field in ("paper_id", "url", "pdf_url"):
        value = normalize_text(record.get(field))
        match = re.search(r"PMC\d+", value, re.I)
        if match:
            return match.group(0).upper()
    return ""


def europepmc_search_queries(record: dict[str, Any]) -> list[str]:
    queries: list[str] = []
    doi = normalize_doi(record.get("doi"))
    if doi:
        queries.append(f'DOI:"{doi}"')
    pmid = normalize_text(record.get("pmid"))
    if not pmid and normalize_text(record.get("source")).lower() in {"pubmed", "med", "pmid"}:
        pmid = normalize_text(record.get("paper_id"))
    if pmid and re.fullmatch(r"\d+", pmid):
        queries.append(f"EXT_ID:{pmid} AND SRC:MED")
    title = normalize_text(record.get("title"))
    if title:
        queries.append(f'TITLE:"{title[:220]}"')
    return queries


def search_europepmc_pmcid(record: dict[str, Any], timeout: int) -> tuple[str, dict[str, Any]]:
    direct = find_pmcid(record)
    if direct:
        return direct, {"match": "record_pmcid"}
    for query in europepmc_search_queries(record):
        params = urllib.parse.urlencode({"query": query, "format": "json", "resultType": "core", "pageSize": "3"})
        url = f"{EUROPE_PMC_BASE}/search?{params}"
        body = http_get_json(url, timeout)
        results = body.get("resultList", {}).get("result", [])
        for result in results:
            pmcid = normalize_text(result.get("pmcid"))
            is_oa = normalize_text(result.get("isOpenAccess")).upper()
            if pmcid and (is_oa in {"Y", "YES", "TRUE", "1"} or result.get("hasFullText") == "Y"):
                return pmcid.upper(), {"match": "europepmc_search", "query": query, "result": result}
        for result in results:
            pmcid = normalize_text(result.get("pmcid"))
            if pmcid:
                return pmcid.upper(), {"match": "europepmc_search_non_oa_flag", "query": query, "result": result}
    return "", {"match": "none"}


def local_name(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def element_text(element: ET.Element) -> str:
    return clean_space(" ".join(text for text in element.itertext() if text))


def first_child_text(root: ET.Element, name: str) -> str:
    for element in root.iter():
        if local_name(element.tag) == name:
            value = element_text(element)
            if value:
                return value
    return ""


def direct_child(element: ET.Element, name: str) -> ET.Element | None:
    for child in list(element):
        if local_name(child.tag) == name:
            return child
    return None


def jats_xml_to_markdown(xml_text: str, fallback_title: str) -> str:
    root = ET.fromstring(xml_text.encode("utf-8"))
    title = first_child_text(root, "article-title") or fallback_title or "Europe PMC full text"
    lines = [f"# {title}", "", "## Abstract", ""]
    abstract_parts = []
    for element in root.iter():
        if local_name(element.tag) == "abstract":
            abstract_parts.append(element_text(element))
    lines.append("\n\n".join(part for part in abstract_parts if part) or "[No abstract extracted]")
    lines.extend(["", "## Body", ""])

    def walk_sec(sec: ET.Element, level: int) -> None:
        title_node = direct_child(sec, "title")
        title_text = element_text(title_node) if title_node is not None else "Section"
        lines.append("#" * min(level, 6) + " " + title_text)
        lines.append("")
        for child in list(sec):
            child_name = local_name(child.tag)
            if child_name == "p":
                text = element_text(child)
                if text:
                    lines.append(text)
                    lines.append("")
            elif child_name == "sec":
                walk_sec(child, level + 1)

    body = None
    for element in root.iter():
        if local_name(element.tag) == "body":
            body = element
            break
    if body is not None:
        for child in list(body):
            if local_name(child.tag) == "sec":
                walk_sec(child, 3)
            elif local_name(child.tag) == "p":
                text = element_text(child)
                if text:
                    lines.append(text)
                    lines.append("")
    return "\n".join(lines).strip() + "\n"


def fetch_europepmc_record(pool_dir: Path, record: dict[str, Any], timeout: int) -> dict[str, Any]:
    pmcid, match_info = search_europepmc_pmcid(record, timeout)
    if not pmcid:
        return {"key": record.get("key"), "status": "not_found", "match": match_info}
    url = f"{EUROPE_PMC_BASE}/{pmcid}/fullTextXML"
    raw, content_type = http_get_bytes(url, timeout)
    xml_text = raw.decode("utf-8", errors="replace")
    if "<" not in xml_text[:200]:
        return {"key": record.get("key"), "status": "no_xml", "pmcid": pmcid, "content_type": content_type}
    out_dir = record_dir(pool_dir, record)
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = out_dir / "europepmc_fulltext.xml"
    md_path = out_dir / "europepmc_fulltext.md"
    xml_path.write_text(xml_text, encoding="utf-8")
    md_path.write_text(jats_xml_to_markdown(xml_text, normalize_text(record.get("title"))), encoding="utf-8")
    record["pmcid"] = pmcid
    record["fulltext_xml_path"] = str(xml_path)
    record["fulltext_text_path"] = str(md_path)
    record["fulltext_path"] = str(md_path)
    record["fulltext_source"] = "Europe PMC fullTextXML"
    record["fulltext_status"] = "fulltext_text_available"
    record["fulltext_imported_at"] = now_iso()
    return {"key": record.get("key"), "status": "ok", "pmcid": pmcid, "xml": str(xml_path), "markdown": str(md_path), "match": match_info}


def mineru_poll(task_id: str, timeout: int, interval: int) -> dict[str, Any]:
    deadline = time.time() + timeout
    last: dict[str, Any] = {}
    while time.time() < deadline:
        body = http_get_json(f"{MINERU_AGENT_BASE}/parse/{task_id}", timeout=30)
        last = body
        data = body.get("data", {})
        state = data.get("state")
        if state == "done":
            markdown_url = data.get("markdown_url", "")
            if not markdown_url:
                return {"status": "failed", "error": "done_without_markdown_url", "body": body}
            md_bytes, _ = http_get_bytes(markdown_url, timeout=60)
            return {"status": "ok", "task_id": task_id, "markdown_url": markdown_url, "markdown": md_bytes.decode("utf-8", errors="replace"), "body": body}
        if state == "failed":
            return {"status": "failed", "task_id": task_id, "body": body, "error": data.get("err_msg", "")}
        time.sleep(interval)
    return {"status": "timeout", "task_id": task_id, "body": last}


def mineru_parse_file(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "file_name": path.name,
        "language": args.language,
        "enable_table": args.enable_table,
        "is_ocr": args.is_ocr,
        "enable_formula": args.enable_formula,
    }
    if args.page_range:
        payload["page_range"] = args.page_range
    created = http_post_json(f"{MINERU_AGENT_BASE}/parse/file", payload, timeout=args.timeout)
    data = created.get("data", {})
    task_id = data.get("task_id")
    file_url = data.get("file_url")
    if not task_id or not file_url:
        return {"status": "failed", "error": "missing_task_or_upload_url", "body": created}
    put_status = put_file(file_url, path, timeout=args.timeout)
    if put_status not in {200, 201}:
        return {"status": "failed", "task_id": task_id, "error": f"upload_failed_http_{put_status}"}
    return mineru_poll(task_id, args.timeout, args.interval)


def mineru_parse_url(url: str, args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "url": url,
        "language": args.language,
        "enable_table": args.enable_table,
        "is_ocr": args.is_ocr,
        "enable_formula": args.enable_formula,
    }
    if args.page_range:
        payload["page_range"] = args.page_range
    created = http_post_json(f"{MINERU_AGENT_BASE}/parse/url", payload, timeout=args.timeout)
    task_id = created.get("data", {}).get("task_id")
    if not task_id:
        return {"status": "failed", "error": "missing_task_id", "body": created}
    return mineru_poll(task_id, args.timeout, args.interval)


def save_mineru_markdown(pool_dir: Path, record: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    markdown = normalize_text(result.get("markdown"))
    if not markdown:
        return {"key": record.get("key"), "status": result.get("status"), "error": result.get("error", "no_markdown")}
    out_dir = record_dir(pool_dir, record)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "mineru_fulltext.md"
    md_path.write_text(markdown, encoding="utf-8")
    record["fulltext_text_path"] = str(md_path)
    record["fulltext_path"] = str(md_path)
    record["fulltext_source"] = "MinerU Agent API"
    record["fulltext_status"] = "fulltext_text_available"
    record["mineru_task_id"] = normalize_text(result.get("task_id"))
    record["mineru_markdown_url"] = normalize_text(result.get("markdown_url"))
    record["fulltext_imported_at"] = now_iso()
    return {"key": record.get("key"), "status": "ok", "markdown": str(md_path), "task_id": result.get("task_id")}


def cmd_audit(args: argparse.Namespace) -> int:
    pool_dir = Path(args.pool_dir)
    pool = load_pool(pool_dir)
    statuses = {item.strip() for item in args.status.split(",") if item.strip()}
    records = selected_records(pool, statuses)
    rows = audit_rows(pool_dir, records)
    summary = write_audit_outputs(pool_dir, rows)
    summary["status_db"] = build_memory_index(pool_dir, rows)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def cmd_fetch_europepmc(args: argparse.Namespace) -> int:
    pool_dir = Path(args.pool_dir)
    pool = load_pool(pool_dir)
    statuses = {item.strip() for item in args.status.split(",") if item.strip()}
    records = selected_records(pool, statuses)
    if args.limit:
        records = records[: args.limit]
    results = []
    for record in records:
        if fulltext_text_path(record) and not args.force:
            results.append({"key": record.get("key"), "status": "exists"})
            continue
        if args.dry_run:
            results.append({"key": record.get("key"), "status": "dry_run", "pmcid": find_pmcid(record), "queries": europepmc_search_queries(record)})
            continue
        try:
            results.append(fetch_europepmc_record(pool_dir, record, args.timeout))
        except Exception as exc:
            results.append({"key": record.get("key"), "status": "failed", "error": str(exc)})
    save_pool(pool_dir, pool)
    rows = audit_rows(pool_dir, selected_records(pool, statuses))
    summary = write_audit_outputs(pool_dir, rows)
    log_path = pool_paths(pool_dir)["logs"] / f"europepmc_fulltext_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for item in results if item.get("status") == "ok")
    print(json.dumps({**summary, "attempted": len(results), "downloaded": ok, "europepmc_log": str(log_path)}, ensure_ascii=False))
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    pool_dir = Path(args.pool_dir)
    pool = load_pool(pool_dir)
    statuses = {item.strip() for item in args.status.split(",") if item.strip()}
    records = selected_records(pool, statuses)
    source_dir = Path(args.source_dir) if args.source_dir else pool_paths(pool_dir)["user_fulltext"]
    source_dir.mkdir(parents=True, exist_ok=True)
    files = discover_user_files(source_dir)
    used: set[Path] = set()
    imported = []
    unmatched_records = []
    for record in records:
        if fulltext_status(record) == "fulltext_text_available" and not args.force:
            continue
        match, score = find_best_file(record, files, used, args.min_score)
        if match is None:
            unmatched_records.append({"key": record.get("key"), "title": record.get("title"), "best_score": score})
            continue
        used.add(match)
        imported.append(import_file(pool_dir, record, match, copy=not args.link_only))
    save_pool(pool_dir, pool)
    rows = audit_rows(pool_dir, records)
    summary = write_audit_outputs(pool_dir, rows)
    log = {
        "imported": imported,
        "unmatched_records": unmatched_records,
        "unused_files": [str(path) for path in files if path not in used],
        "audit": summary,
    }
    log_path = pool_paths(pool_dir)["logs"] / f"fulltext_import_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "imported": len(imported), "unmatched_records": len(unmatched_records), "unused_files": len(log["unused_files"]), "import_log": str(log_path)}, ensure_ascii=False))
    return 0


def cmd_mineru(args: argparse.Namespace) -> int:
    pool_dir = Path(args.pool_dir)
    pool = load_pool(pool_dir)
    statuses = {item.strip() for item in args.status.split(",") if item.strip()}
    records = selected_records(pool, statuses)
    if args.limit:
        records = records[: args.limit]
    results = []
    for record in records:
        if fulltext_text_path(record) and not args.force:
            results.append({"key": record.get("key"), "status": "exists"})
            continue
        local_pdf = pdf_path(record)
        remote_url = normalize_text(record.get("pdf_url") or record.get("url"))
        if args.dry_run:
            results.append({"key": record.get("key"), "status": "dry_run", "local_pdf": local_pdf, "url": remote_url})
            continue
        try:
            if local_pdf and not args.prefer_url:
                parsed = mineru_parse_file(Path(local_pdf), args)
            elif remote_url:
                parsed = mineru_parse_url(remote_url, args)
            elif local_pdf:
                parsed = mineru_parse_file(Path(local_pdf), args)
            else:
                results.append({"key": record.get("key"), "status": "skipped", "reason": "no local_pdf or url"})
                continue
            if parsed.get("status") == "ok":
                results.append(save_mineru_markdown(pool_dir, record, parsed))
            else:
                results.append({"key": record.get("key"), **{k: v for k, v in parsed.items() if k != "markdown"}})
        except Exception as exc:
            results.append({"key": record.get("key"), "status": "failed", "error": str(exc)})
    save_pool(pool_dir, pool)
    rows = audit_rows(pool_dir, selected_records(pool, statuses))
    summary = write_audit_outputs(pool_dir, rows)
    log_path = pool_paths(pool_dir)["logs"] / f"mineru_parse_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for item in results if item.get("status") == "ok")
    print(json.dumps({**summary, "attempted": len(results), "parsed": ok, "mineru_log": str(log_path)}, ensure_ascii=False))
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    pool_dir = Path(args.pool_dir)
    pool = load_pool(pool_dir)
    statuses = {item.strip() for item in args.status.split(",") if item.strip()}
    records = selected_records(pool, statuses)
    extracted = []
    skipped = []
    for record in records:
        if fulltext_text_path(record) and not args.force:
            skipped.append({"key": record.get("key"), "reason": "text already exists"})
            continue
        pdf = pdf_path(record)
        if not pdf:
            skipped.append({"key": record.get("key"), "reason": "no local pdf"})
            continue
        extracted.append(import_file(pool_dir, record, Path(pdf), copy=False))
    save_pool(pool_dir, pool)
    rows = audit_rows(pool_dir, records)
    summary = write_audit_outputs(pool_dir, rows)
    print(json.dumps({"extracted": len(extracted), "skipped": len(skipped), **summary}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit/import selected paper full text for a review literature pool.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_audit = sub.add_parser("audit", help="Write full-text status and user handoff files.")
    p_audit.add_argument("--pool-dir", required=True)
    p_audit.add_argument("--status", default="citation_pool,seminal,method,recent")
    p_audit.set_defaults(func=cmd_audit)

    p_epmc = sub.add_parser("fetch-europepmc", help="Fetch Europe PMC Open Access fullTextXML and convert it to Markdown.")
    p_epmc.add_argument("--pool-dir", required=True)
    p_epmc.add_argument("--status", default="citation_pool,seminal,method,recent")
    p_epmc.add_argument("--limit", type=int, default=0)
    p_epmc.add_argument("--timeout", type=int, default=60)
    p_epmc.add_argument("--force", action="store_true")
    p_epmc.add_argument("--dry-run", action="store_true")
    p_epmc.set_defaults(func=cmd_fetch_europepmc)

    p_import = sub.add_parser("import", help="Import user-supplied full text files into the evidence folder.")
    p_import.add_argument("--pool-dir", required=True)
    p_import.add_argument("--source-dir", default="")
    p_import.add_argument("--status", default="citation_pool,seminal,method,recent")
    p_import.add_argument("--min-score", type=int, default=3)
    p_import.add_argument("--force", action="store_true")
    p_import.add_argument("--link-only", action="store_true", help="Do not copy files; reference them in place.")
    p_import.set_defaults(func=cmd_import)

    p_extract = sub.add_parser("extract", help="Extract text from already linked/downloaded local PDFs.")
    p_extract.add_argument("--pool-dir", required=True)
    p_extract.add_argument("--status", default="citation_pool,seminal,method,recent")
    p_extract.add_argument("--force", action="store_true")
    p_extract.set_defaults(func=cmd_extract)

    p_mineru = sub.add_parser("mineru-agent", help="Parse local PDFs or URLs with the MinerU Agent lightweight API into Markdown.")
    p_mineru.add_argument("--pool-dir", required=True)
    p_mineru.add_argument("--status", default="citation_pool,seminal,method,recent")
    p_mineru.add_argument("--limit", type=int, default=0)
    p_mineru.add_argument("--timeout", type=int, default=300)
    p_mineru.add_argument("--interval", type=int, default=3)
    p_mineru.add_argument("--language", default="ch")
    p_mineru.add_argument("--page-range", default="")
    p_mineru.add_argument("--enable-table", action=argparse.BooleanOptionalAction, default=True)
    p_mineru.add_argument("--is-ocr", action=argparse.BooleanOptionalAction, default=True)
    p_mineru.add_argument("--enable-formula", action=argparse.BooleanOptionalAction, default=True)
    p_mineru.add_argument("--prefer-url", action="store_true")
    p_mineru.add_argument("--force", action="store_true")
    p_mineru.add_argument("--dry-run", action="store_true")
    p_mineru.set_defaults(func=cmd_mineru)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
