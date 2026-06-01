#!/usr/bin/env python3
"""Inspect draft-native citation assets before any supplemental recall.

This script is deliberately conservative. It reads GPT/Gemini/Deep Research
drafts as untrusted but valuable source material, separates the body from the
reference section, maps in-text citation markers to reference entries, extracts
claim snippets, and writes audit tables that should be reviewed before broad
literature recall is allowed.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import urllib.parse
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


SUPPORTED_EXTENSIONS = {".md", ".txt", ".docx"}
REFERENCE_HEADING_RE = re.compile(
    r"^\s*(#{1,6}\s*)?(references|bibliography|works cited|参考文献|參考文獻|引用文献|参考资料)\s*[:：]?\s*$",
    re.I,
)
REFERENCE_HEADING_RE = re.compile(
    r"^\s*(references|bibliography|works cited|reference list|参考文献|參考文獻|引用文献|引用文獻|参考资料|參考資料|文献|文獻)\s*[:：]?\s*$",
    re.I,
)
ACADEMIC_HOST_MARKERS = [
    "doi.org",
    "pubmed",
    "ncbi.nlm.nih.gov",
    "pmc.ncbi.nlm.nih.gov",
    "crossref.org",
    "openalex.org",
    "arxiv.org",
    "openreview.net",
    "biorxiv.org",
    "medrxiv.org",
    "nature.com",
    "science.org",
    "cell.com",
    "springer",
    "wiley",
    "elsevier",
    "sciencedirect",
    "tandfonline",
    "bmj.com",
    "thelancet.com",
    "acm.org",
    "ieee.org",
]
NON_ACADEMIC_HOST_MARKERS = [
    "github.com",
    "huggingface.co",
    "openai.com",
    "googleblog.com",
    "medium.com",
    "substack.com",
    "wordpress",
    "blog",
    "ibm.com",
    "microsoft.com",
    "nvidia.com",
    "recursion.com",
]
CLAIM_MARKERS = re.compile(
    r"提出|认为|显示|表明|证明|揭示|指出|需要|应该|可以|能够|关键|核心|限制|挑战|机会|趋势|转向|框架|架构|范式|证据|治理|"
    r"\b(argues?|shows?|suggests?|demonstrates?|requires?|should|can|could|must|enables?|limits?|reveals?|proposes?)\b",
    re.I,
)


@dataclass
class DraftParts:
    path: str
    title: str
    body: str
    references_text: str


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def clean_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def safe_filename(value: str, fallback: str = "draft") -> str:
    text = re.sub(r"[^A-Za-z0-9_.\-\u4e00-\u9fff]+", "-", value or fallback).strip("-")
    return text[:100] or fallback


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
    unique: list[Path] = []
    for path in paths:
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def infer_title(text: str, fallback: str) -> str:
    for line in text.splitlines()[:40]:
        candidate = clean_space(re.sub(r"^#+\s*", "", line))
        if candidate and len(candidate) <= 180 and candidate.lower() not in {"abstract", "summary", "references"}:
            return candidate
    return fallback


def split_reference_section(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if is_reference_heading(line):
            return "\n".join(lines[:idx]).strip(), "\n".join(lines[idx + 1 :]).strip()
    return text.strip(), ""


def is_reference_heading(line: str) -> bool:
    text = clean_space(line)
    text = re.sub(r"^#{1,6}\s*", "", text)
    text = re.sub(r"^[>*\s_`]+|[>*\s_`]+$", "", text)
    return bool(REFERENCE_HEADING_RE.match(text))


def parse_draft(path: Path) -> DraftParts:
    text = read_text(path)
    body, refs = split_reference_section(text)
    return DraftParts(path=str(path), title=infer_title(text, path.name), body=body, references_text=refs)


def extract_dois(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", text, re.I):
        doi = match.group(0).rstrip(".,;)").lower()
        if doi not in seen:
            seen.add(doi)
            out.append(doi)
    return out


def extract_pmids(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    patterns = [
        r"\bPMID\s*[:：]?\s*([1-9][0-9]{5,9})\b",
        r"pubmed\.ncbi\.nlm\.nih\.gov/([1-9][0-9]{5,9})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            pmid = match.group(1)
            if pmid not in seen:
                seen.add(pmid)
                out.append(pmid)
    return out


def write_query_file(path: Path, tasks: list[dict[str, str]]) -> None:
    blocked_markers = ["摘要", "关键词", "abstract", "keywords", "本综述", "本文", "我们提出", "旨在"]
    seen: set[str] = set()
    lines: list[str] = []
    for row in tasks:
        query = clean_space(row.get("query_seed", ""))
        lower = query.lower()
        if not query or query in seen:
            continue
        if len(query) < 28 or len(query) > 180:
            continue
        if any(marker in query or marker in lower for marker in blocked_markers):
            continue
        seen.add(query)
        lines.append(query)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def extract_urls(text: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"https?://[^\s<>\]\)\"']+", text):
        url = match.group(0).rstrip(".,;")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def source_kind(urls: list[str], raw: str) -> str:
    if extract_dois(raw) or extract_pmids(raw):
        return "paper_identifier"
    if not urls:
        return "text_only_reference"
    hosts = [urllib.parse.urlparse(url).netloc.lower() for url in urls]
    if any(any(marker in host for marker in ACADEMIC_HOST_MARKERS) for host in hosts):
        return "academic_or_publisher_url"
    if any(any(marker in host for marker in NON_ACADEMIC_HOST_MARKERS) for host in hosts):
        return "web_or_product_source"
    return "web_unknown"


def split_reference_entries(references_text: str) -> list[dict[str, str]]:
    if not references_text:
        return []
    entries: list[str] = []
    buffer = ""
    for raw_line in references_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        numbered = re.match(r"^(\[\d+\]|\d+[\.)]|[（(]\d+[）)])\s*(.+)$", line)
        if numbered:
            if buffer:
                entries.append(buffer.strip())
            buffer = line
        elif buffer and (len(line) > 18 or extract_urls(line) or extract_dois(line)):
            buffer += " " + line
        elif len(line) > 30 and (re.search(r"\b(19|20)\d{2}\b", line) or extract_urls(line) or extract_dois(line)):
            if buffer:
                entries.append(buffer.strip())
            buffer = line
    if buffer:
        entries.append(buffer.strip())
    if not entries:
        entries = [line.strip() for line in references_text.splitlines() if len(line.strip()) > 30]

    rows: list[dict[str, str]] = []
    for idx, entry in enumerate(entries, 1):
        marker_match = re.match(r"^(\[(\d+)\]|(\d+)[\.)]|[（(](\d+)[）)])\s*(.+)$", entry)
        marker = ""
        number = ""
        body = entry
        if marker_match:
            marker = marker_match.group(1)
            number = next((g for g in marker_match.groups()[1:4] if g), "")
            body = marker_match.group(5)
        urls = extract_urls(body)
        rows.append(
            {
                "ref_index": str(idx),
                "ref_number": number,
                "ref_marker": marker,
                "raw_reference": body.strip(),
                "doi": "; ".join(extract_dois(body)),
                "pmid": "; ".join(extract_pmids(body)),
                "urls": "; ".join(urls),
                "source_kind": source_kind(urls, body),
                "candidate_title": guess_title(body),
            }
        )
    return rows


def guess_title(reference: str) -> str:
    journal_hint = r"(?:Nature(?:\s+[A-Za-z]+)?|Science|Cell|Lancet|NEJM|JAMA|BMJ|Bioinformatics|Journal of [A-Za-z ]+|Proceedings|NeurIPS|ICLR|ICML|arXiv|bioRxiv|medRxiv)"
    title_match = re.search(r"\bet al\.\s+(.+?)\.\s+\*?" + journal_hint + r"\*?[,.\s]", reference, re.I)
    if title_match:
        return clean_space(title_match.group(1))[:260]
    first_author_match = re.search(r"^[A-Z][A-Za-z'\-]+,\s*(?:[A-Z]\.\s*)+(?:et al\.)?\s+(.+?)\.\s+\*?" + journal_hint + r"\*?[,.\s]", reference, re.I)
    if first_author_match:
        return clean_space(first_author_match.group(1))[:260]
    quoted = re.findall(r"[\"“](.*?)[\"”]", reference)
    if quoted:
        return clean_space(max(quoted, key=len))[:260]
    text = re.sub(r"https?://\S+", "", reference)
    text = re.sub(r"\bdoi\s*[:：]?\s*10\.\S+", "", text, flags=re.I)
    parts = [clean_space(part) for part in re.split(r"\.\s+", text) if clean_space(part)]
    candidates = [part for part in parts if 8 <= len(part) <= 260 and not re.match(r"^[A-Z][A-Za-z\-]+,?\s+[A-Z]", part)]
    if candidates:
        return max(candidates[:4], key=len)
    return clean_space(text)[:260]


def intext_citation_markers(body: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for match in re.finditer(r"\[([0-9,\-\s;]+)\]", body):
        raw = match.group(0)
        nums: list[str] = []
        for part in re.split(r"[,;\s]+", match.group(1)):
            if not part:
                continue
            if "-" in part:
                a, _, b = part.partition("-")
                if a.isdigit() and b.isdigit() and int(a) <= int(b) <= int(a) + 50:
                    nums.extend(str(i) for i in range(int(a), int(b) + 1))
            elif part.isdigit():
                nums.append(part)
        context = clean_space(body[max(0, match.start() - 180) : match.end() + 180])
        for number in nums:
            key = (raw, number)
            if key not in seen:
                seen.add(key)
                rows.append({"marker": raw, "ref_number": number, "context": context})
    return rows


def split_sentences(text: str) -> list[str]:
    compact = re.sub(r"\n+", " ", text)
    parts = re.split(r"(?<=[。！？!?])\s+|(?<=[。！？!?])", compact)
    return [clean_space(part) for part in parts if 40 <= len(clean_space(part)) <= 520]


def markers_in_sentence(sentence: str) -> tuple[str, str]:
    markers = re.findall(r"\[[0-9,\-\s;]+\]", sentence)
    numbers: list[str] = []
    for marker in markers:
        for row in intext_citation_markers(marker):
            if row["ref_number"]:
                numbers.append(row["ref_number"])
    return "; ".join(dict.fromkeys(markers)), "; ".join(dict.fromkeys(numbers))


def claim_rows(draft: DraftParts, max_claims: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for idx, sentence in enumerate(split_sentences(draft.body), 1):
        has_marker = bool(re.search(r"\[[0-9,\-\s;]+\]", sentence))
        if not has_marker and not CLAIM_MARKERS.search(sentence):
            continue
        markers, refs = markers_in_sentence(sentence)
        rows.append(
            {
                "claim_id": f"{safe_filename(Path(draft.path).stem)}-C{idx:04d}",
                "draft": Path(draft.path).name,
                "claim": sentence,
                "original_excerpt": sentence[:520],
                "citation_markers": markers,
                "mapped_ref_numbers": refs,
                "evidence_scope": "draft_cited_reference" if refs else "needs_evidence_or_user_decision",
            }
        )
        if len(rows) >= max_claims:
            break
    return rows


SYSTEM_NAME_RE = re.compile(
    r"\b("
    r"AI\s+co-?scientist|Coscientist|HuggingGPT|Reflexion|AutoGen|GeneGPT|scGPT|ChemCrow|BioPlanner|"
    r"CRISPR-?GPT|DrugGPT|BioMedAgent|CellAgent|CellVoyager|AlphaFold\s*3|"
    r"[A-Z][A-Za-z0-9-]{1,40}(?:GPT|Agent|Planner|Voyager|Crow|Fold|Former|BERT|T5|RAG|KG)"
    r")\b"
)


def body_mentioned_system_rows(draft: DraftParts, max_rows: int = 160) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    blocked = {"multi-agent", "single-agent", "agent", "multiagent"}
    body = draft.body
    for match in SYSTEM_NAME_RE.finditer(body):
        name = clean_space(match.group(1))
        key = name.lower()
        if key in blocked:
            continue
        if key in seen:
            continue
        seen.add(key)
        context = clean_space(body[max(0, match.start() - 260) : match.end() + 260])
        rows.append(
            {
                "draft": Path(draft.path).name,
                "system_or_method": name,
                "context": context,
                "candidate_query": f'"{name}" paper',
                "source_kind": "body_mentioned_system_or_method",
                "evidence_scope": "needs_codex_subagent_identity_normalization",
            }
        )
        if len(rows) >= max_rows:
            break
    return rows


def query_seed_from_claim(claim: str, max_chars: int = 180) -> str:
    text = clean_space(claim)
    text = re.sub(r"\[[0-9,\-\s;]+\]", " ", text)
    text = re.sub(r"[*_`#>]+", " ", text)
    text = re.sub(r"^\s*(\d+(\.\d+)*|[一二三四五六七八九十]+)[、.．\s-]+", "", text)
    parts = [clean_space(part) for part in re.split(r"(?<=[。！？.!?])\s*|[；;]\s*", text) if clean_space(part)]
    preferred = [
        part
        for part in parts
        if 28 <= len(part) <= max_chars
        and not re.match(r"^(摘要|关键词|abstract|keywords|introduction|引言)\b", part, flags=re.I)
    ]
    if preferred:
        return preferred[0]
    for part in parts:
        if len(part) >= 20:
            return part[:max_chars].rstrip(" ，,。.;；")
    return text[:max_chars].rstrip(" ，,。.;；")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_health_report(out_dir: Path, manifest: dict[str, Any], refs: list[dict[str, str]], claims: list[dict[str, str]]) -> None:
    by_kind: dict[str, int] = {}
    for row in refs:
        by_kind[row["source_kind"]] = by_kind.get(row["source_kind"], 0) + 1
    uncited = [row for row in refs if row.get("cited_in_body") == "no"]
    unresolved = manifest["summary"]["unresolved_intext_markers"]
    lines = [
        "# Draft Citation Asset Health Check",
        "",
        f"- Generated at: {manifest['generated_at']}",
        f"- Drafts: {manifest['summary']['drafts']}",
        f"- Reference entries: {len(refs)}",
        f"- Claim snippets: {len(claims)}",
        f"- Body-mentioned systems/methods: {manifest['summary'].get('body_mentioned_systems', 0)}",
        f"- Uncited reference entries: {len(uncited)}",
        f"- Unresolved in-text markers: {unresolved}",
        "",
        "## Source Kinds",
        "",
    ]
    for kind, count in sorted(by_kind.items()):
        lines.append(f"- {kind}: {count}")
    lines.extend(
        [
            "",
            "## Blocking Checks",
            "",
            "- `candidate_paper_clues.csv` is extracted from draft reference sections and mapped citation assets only; it is not the full literature asset count for the drafts.",
            "- Run chief-Codex literature discovery and subagent packet review before concluding that the draft corpus has too few papers.",
            "- Use `body_mentioned_systems.csv` as a subagent normalization prompt source, not as verified references.",
            "- Do not run broad supplemental recall until this asset report has been reviewed.",
            "- Treat `web_or_product_source` and `web_unknown` as background leads, not manuscript evidence.",
            "- Resolve unmatched citation markers and uncited reference entries before preserving numbering.",
            "- Use multi-signal PubMed/PMID/DOI/title/source verification on `candidate_paper_clues.csv` before import into the governed literature pool.",
            "",
            "## Output Files",
            "",
        ]
    )
    for key, value in manifest["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    (out_dir / "draft_citation_health.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_build(args: argparse.Namespace) -> int:
    drafts = discover_drafts(args)
    if not drafts:
        raise SystemExit("No supported drafts found. Use --draft or --draft-dir with .md, .txt, or .docx files.")
    out_dir = Path(args.out_dir).resolve()
    cleaned_dir = out_dir / "cleaned_drafts"
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    all_refs: list[dict[str, str]] = []
    all_markers: list[dict[str, str]] = []
    all_claims: list[dict[str, str]] = []
    all_systems: list[dict[str, str]] = []
    for path in drafts:
        draft = parse_draft(path)
        cleaned_path = cleaned_dir / f"{safe_filename(path.stem)}.body.md"
        cleaned_path.write_text(draft.body.strip() + "\n", encoding="utf-8")

        refs = split_reference_entries(draft.references_text)
        markers = intext_citation_markers(draft.body)
        cited_numbers = {row["ref_number"] for row in markers if row["ref_number"]}
        known_numbers = {row["ref_number"] for row in refs if row["ref_number"]}
        for row in refs:
            row.update(
                {
                    "draft": path.name,
                    "draft_path": str(path),
                    "cleaned_body_path": str(cleaned_path),
                    "cited_in_body": "yes" if row["ref_number"] and row["ref_number"] in cited_numbers else "no",
                    "verification_priority": "high" if row.get("doi") or row.get("pmid") else "medium",
                }
            )
        for row in markers:
            row.update(
                {
                    "draft": path.name,
                    "resolved": "yes" if row["ref_number"] in known_numbers else "no",
                }
            )
        all_refs.extend(refs)
        all_markers.extend(markers)
        all_claims.extend(claim_rows(draft, args.max_claims_per_draft))
        all_systems.extend(body_mentioned_system_rows(draft))

    for claim in all_claims:
        ref_numbers = {num for num in claim.get("mapped_ref_numbers", "").split("; ") if num}
        matched_refs = [
            row
            for row in all_refs
            if row["draft"] == claim["draft"] and row["ref_number"] and row["ref_number"] in ref_numbers
        ]
        claim["candidate_ref_titles"] = "; ".join(row["candidate_title"] for row in matched_refs[:5])
        claim["candidate_ref_dois"] = "; ".join(row["doi"] for row in matched_refs if row["doi"])
        claim["candidate_ref_pmids"] = "; ".join(row["pmid"] for row in matched_refs if row["pmid"])

    candidate_clues = [
        row
        for row in all_refs
        if row["source_kind"] in {"paper_identifier", "academic_or_publisher_url", "text_only_reference"}
    ]
    non_academic = [row for row in all_refs if row["source_kind"] in {"web_or_product_source", "web_unknown"}]
    uncited = [row for row in all_refs if row["cited_in_body"] == "no"]
    unresolved = [row for row in all_markers if row["resolved"] == "no"]
    supplemental_tasks = [
        {
            "draft": row["draft"],
            "claim_id": row["claim_id"],
            "gap_type": "claim_without_mapped_reference",
            "query_seed": query_seed_from_claim(row["claim"]),
            "status": "needs_user_or_llm_triage_before_search",
        }
        for row in all_claims
        if not row.get("mapped_ref_numbers")
    ]
    unmapped_claims = [row for row in all_claims if not row.get("mapped_ref_numbers")]

    outputs = {
        "reference_inventory": str(out_dir / "draft_reference_inventory.csv"),
        "intext_citations": str(out_dir / "draft_intext_citations.csv"),
        "claim_evidence_map": str(out_dir / "claim_evidence_map.csv"),
        "candidate_paper_clues": str(out_dir / "candidate_paper_clues.csv"),
        "non_academic_sources": str(out_dir / "non_academic_sources.csv"),
        "uncited_references": str(out_dir / "uncited_references.csv"),
        "unresolved_intext_markers": str(out_dir / "unresolved_intext_markers.csv"),
        "body_mentioned_systems": str(out_dir / "body_mentioned_systems.csv"),
        "unmapped_claims": str(out_dir / "unmapped_claims.csv"),
        "supplemental_recall_tasks": str(out_dir / "supplemental_recall_tasks.csv"),
        "supplemental_recall_queries": str(out_dir / "supplemental_recall_queries.txt"),
        "cleaned_drafts": str(cleaned_dir),
        "health_report": str(out_dir / "draft_citation_health.md"),
        "manifest": str(out_dir / "asset_manifest.json"),
    }

    write_csv(
        out_dir / "draft_reference_inventory.csv",
        all_refs,
        [
            "draft",
            "ref_index",
            "ref_number",
            "ref_marker",
            "candidate_title",
            "raw_reference",
            "doi",
            "pmid",
            "urls",
            "source_kind",
            "cited_in_body",
            "verification_priority",
            "draft_path",
            "cleaned_body_path",
        ],
    )
    write_csv(out_dir / "draft_intext_citations.csv", all_markers, ["draft", "marker", "ref_number", "resolved", "context"])
    write_csv(
        out_dir / "claim_evidence_map.csv",
        all_claims,
        [
            "claim_id",
            "draft",
            "claim",
            "original_excerpt",
            "citation_markers",
            "mapped_ref_numbers",
            "candidate_ref_titles",
            "candidate_ref_dois",
            "candidate_ref_pmids",
            "evidence_scope",
        ],
    )
    write_csv(
        out_dir / "candidate_paper_clues.csv",
        candidate_clues,
        ["draft", "ref_number", "candidate_title", "raw_reference", "doi", "pmid", "urls", "source_kind", "cited_in_body", "verification_priority"],
    )
    write_csv(out_dir / "non_academic_sources.csv", non_academic, ["draft", "ref_number", "candidate_title", "raw_reference", "urls", "source_kind", "cited_in_body"])
    write_csv(out_dir / "uncited_references.csv", uncited, ["draft", "ref_number", "candidate_title", "raw_reference", "doi", "pmid", "urls", "source_kind"])
    write_csv(out_dir / "unresolved_intext_markers.csv", unresolved, ["draft", "marker", "ref_number", "context"])
    write_csv(out_dir / "body_mentioned_systems.csv", all_systems, ["draft", "system_or_method", "context", "candidate_query", "source_kind", "evidence_scope"])
    write_csv(
        out_dir / "unmapped_claims.csv",
        unmapped_claims,
        ["claim_id", "draft", "claim", "original_excerpt", "citation_markers", "mapped_ref_numbers", "candidate_ref_titles", "candidate_ref_dois", "candidate_ref_pmids", "evidence_scope"],
    )
    write_csv(out_dir / "supplemental_recall_tasks.csv", supplemental_tasks, ["draft", "claim_id", "gap_type", "query_seed", "status"])
    write_query_file(out_dir / "supplemental_recall_queries.txt", supplemental_tasks)

    manifest = {
        "schema_name": "draft_citation_assets",
        "schema_version": "0.1",
        "generated_at": now_iso(),
        "topic": args.topic,
        "summary": {
            "drafts": len(drafts),
            "reference_entries": len(all_refs),
            "candidate_paper_clues": len(candidate_clues),
            "non_academic_sources": len(non_academic),
            "uncited_references": len(uncited),
            "intext_markers": len(all_markers),
            "unresolved_intext_markers": len(unresolved),
            "body_mentioned_systems": len(all_systems),
            "unmapped_claims": len(unmapped_claims),
            "claim_rows": len(all_claims),
            "supplemental_recall_tasks": len(supplemental_tasks),
        },
        "outputs": outputs,
    }
    (out_dir / "asset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_health_report(out_dir, manifest, all_refs, all_claims)
    print(json.dumps({"out_dir": str(out_dir), **manifest["summary"]}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a draft-native citation asset health report before supplemental recall.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_build = sub.add_parser("build", help="Inspect draft citation assets and write audit tables.")
    p_build.add_argument("--draft", action="append", default=[], help="Draft file path; repeatable.")
    p_build.add_argument("--draft-dir", action="append", default=[], help="Directory containing .md/.txt/.docx drafts; repeatable.")
    p_build.add_argument("--topic", default="")
    p_build.add_argument("--out-dir", default="./review-data/02_literature/draft_assets")
    p_build.add_argument("--max-claims-per-draft", type=int, default=80)
    p_build.set_defaults(func=cmd_build)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
