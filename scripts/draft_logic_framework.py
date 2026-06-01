#!/usr/bin/env python3
"""Extract a review logic framework from one or more AI-generated drafts.

This script is a pre-drafting gate. It reads GPT/Gemini/Deep Research drafts,
summarizes recurring frames, extracts claim-like statements and citation
markers, and creates a user-alignment brief before Codex drafts the manuscript.
It is intentionally heuristic and stdlib-only; Codex should review and refine
the outputs with the user.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import textwrap
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from xml.etree import ElementTree as ET


SUPPORTED_EXTENSIONS = {".md", ".txt", ".docx"}
STOP_HEADING_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "of",
    "in",
    "to",
    "as",
    "by",
    "on",
    "via",
    "towards",
    "review",
    "article",
    "introduction",
    "conclusion",
    "summary",
    "http",
    "https",
    "www",
    "com",
    "org",
    "net",
    "accessed",
    "retrieved",
    "pubmed",
    "pmc",
    "doi",
    "arxiv",
    "openreview",
    "参考文献",
    "摘要",
    "引言",
    "结论",
    "展望",
    "综述",
    "访问时间为",
    "一月",
    "二月",
    "三月",
    "四月",
    "五月",
    "六月",
    "七月",
    "八月",
    "九月",
    "十月",
    "十一月",
    "十二月",
}


@dataclass
class DraftDoc:
    path: str
    title: str
    chars: int
    headings: list[dict[str, str]]
    claims: list[dict[str, str]]
    references: list[str]
    abstract: str


def clean_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def read_docx(path: Path) -> str:
    parts: list[str] = []
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    for para in root.findall(".//w:p", ns):
        runs = [node.text or "" for node in para.findall(".//w:t", ns)]
        line = clean_space("".join(runs))
        if line:
            parts.append(line)
    return "\n".join(parts)


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
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    paths.append(path)
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def infer_title(lines: list[str]) -> str:
    for line in lines[:30]:
        text = clean_space(re.sub(r"^#+\s*", "", line))
        if not text:
            continue
        if text.lower() in {"abstract", "summary", "references", "introduction"}:
            continue
        if len(text) <= 180:
            return text
    return "[untitled draft]"


def heading_level_from_prefix(raw: str) -> tuple[int, str] | None:
    line = clean_space(raw)
    if not line:
        return None
    m = re.match(r"^(#{1,6})\s+(.+)$", line)
    if m:
        return len(m.group(1)), clean_space(m.group(2))
    m = re.match(r"^第?([一二三四五六七八九十百\d]+)[章节部分][：:\s]*(.+)$", line)
    if m and len(m.group(2)) <= 120:
        return 1, clean_space(m.group(2))
    m = re.match(r"^([一二三四五六七八九十\d]+)[、.．]\s*(.+)$", line)
    if m and len(m.group(2)) <= 120:
        level = 1 if len(m.group(1)) <= 2 else 2
        return level, clean_space(m.group(2))
    m = re.match(r"^([0-9]+(?:\.[0-9]+){1,3})\s+(.+)$", line)
    if m and len(m.group(2)) <= 120:
        return min(1 + m.group(1).count("."), 4), clean_space(m.group(2))
    if len(line) <= 80 and not re.search(r"[。！？.!?；;]$", line):
        if re.search(r"(框架|架构|范式|挑战|未来|治理|评估|基准|智能体|多模态|路线图|展望|局限|方法|evolution|agent|framework|architecture|evaluation|governance|outlook)", line, re.I):
            return 2, line
    return None


def extract_headings(lines: list[str]) -> list[dict[str, str]]:
    headings = []
    for idx, line in enumerate(lines, 1):
        parsed = heading_level_from_prefix(line)
        if not parsed:
            continue
        level, title = parsed
        title = re.sub(r"[*_`]+", "", title).strip()
        if title and len(title) <= 160:
            headings.append({"line": str(idx), "level": str(level), "title": title})
    return headings[:300]


def split_sentences(text: str) -> list[str]:
    compact = re.sub(r"\n+", " ", text)
    parts = re.split(r"(?<=[。！？.!?])\s+", compact)
    out = []
    for part in parts:
        sent = clean_space(part)
        if 30 <= len(sent) <= 420 and not noisy_claim_sentence(sent):
            out.append(sent)
    return out


def noisy_claim_sentence(sentence: str) -> bool:
    text = clean_space(sentence)
    lower = text.lower()
    if not text:
        return True
    if text.startswith("#") or text.startswith("|") or text.count("|") >= 3:
        return True
    if re.match(r"^(abstract|summary|keywords|references|bibliography)\b", lower):
        return True
    if re.match(r"^(\u6458\u8981|\u5173\u952e\u8bcd|\u53c2\u8003\u6587\u732e)\b", text):
        return True
    if re.search(r"\b(final-gate|search log|screening count|records checked|candidate records|literature pool)\b", lower):
        return True
    if re.search(r"\b(accepted|rejected|demoted)\s+\d+\s+(records|references|papers)\b", lower):
        return True
    if len(re.findall(r"\[[0-9,\-\s;]+\]", text)) >= 5:
        return True
    if len(re.findall(r"https?://|10\.\d{4,9}/", text)) >= 3:
        return True
    if len(text) > 300 and re.search(r"\b(table|figure|fig\.|supplementary|appendix)\b", lower):
        return True
    return False


def citation_markers(sentence: str) -> str:
    markers = []
    markers.extend(re.findall(r"\[[0-9,\-\s;]+\]", sentence))
    markers.extend(re.findall(r"\((?:[A-Z][A-Za-z\-]+(?:\s+et al\.)?,?\s*)?[12][0-9]{3}[a-z]?\)", sentence))
    markers.extend(re.findall(r"(?:doi:|DOI:)\s*10\.\S+", sentence))
    markers.extend(re.findall(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", sentence))
    return "; ".join(dict.fromkeys(markers))


def claim_role(sentence: str) -> str:
    if re.search(r"问题|挑战|限制|风险|failure|challenge|limitation|risk|gap", sentence, re.I):
        return "limitation_or_gap"
    if re.search(r"未来|展望|路线图|应该|需要|agenda|future|roadmap|should|must", sentence, re.I):
        return "future_agenda_or_normative"
    if re.search(r"提出|框架|架构|范式|认为|argues?|proposes?|framework|architecture|paradigm", sentence, re.I):
        return "thesis_or_framework"
    if re.search(r"显示|表明|证明|发现|shows?|demonstrates?|found|evidence", sentence, re.I):
        return "evidence_claim"
    return "interpretive_claim"


def argument_time_role(sentence: str) -> str:
    if re.search(r"history|historical|origin|early|foundation|background|from .* to |过去|历史|起源|早期|基础|从.+到", sentence, re.I):
        return "background_or_history"
    if re.search(r"future|agenda|roadmap|outlook|next|should|must|未来|展望|路线图|下一步|应该|需要", sentence, re.I):
        return "future_agenda"
    if re.search(r"current|now|recent|emerging|today|evidence|shows?|demonstrates?|目前|当前|现在|近年|证据|显示|表明", sentence, re.I):
        return "current_evidence"
    return "interpretive_bridge"


def is_claim_like(sentence: str) -> bool:
    if noisy_claim_sentence(sentence):
        return False
    patterns = [
        r"提出|认为|显示|表明|揭示|证明|指出|需要|应该|可以|能够|关键|核心|限制|挑战|机会|趋势|转向|重构|框架|架构|范式|证据|治理",
        r"\b(argues?|shows?|suggests?|demonstrates?|requires?|should|can|could|must|enables?|limits?|reveals?|proposes?)\b",
    ]
    return any(re.search(pattern, sentence, re.I) for pattern in patterns)


def extract_claims(text: str, limit: int) -> list[dict[str, str]]:
    claims = []
    for idx, sentence in enumerate(split_sentences(text), 1):
        if not is_claim_like(sentence):
            continue
        claims.append(
            {
                "claim_id": f"C{idx:04d}",
                "claim": sentence,
                "claim_role": claim_role(sentence),
                "argument_time_role": argument_time_role(sentence),
                "citation_markers": citation_markers(sentence),
                "evidence_status": "has citation marker; still needs source audit" if citation_markers(sentence) else "needs literature recall",
                "verification_status": "needs source audit",
            }
        )
        if len(claims) >= limit:
            break
    return claims


def extract_references(lines: list[str], limit: int = 200) -> list[str]:
    refs = []
    in_refs = False
    for line in lines:
        raw = clean_space(line)
        if not raw:
            continue
        if re.match(r"^(references|参考文献|bibliography|works cited)\b", raw, re.I):
            in_refs = True
            continue
        explicit_id = bool(re.search(r"(doi\.org|10\.\d{4,9}/|arxiv:\s*\d|arxiv\.org|pubmed\.ncbi\.nlm\.nih\.gov|pmid\s*:|openreview\.net)", raw, re.I))
        numbered_ref = bool(in_refs and re.match(r"^(\[[0-9]+\]|[0-9]+[.)])\s+.+", raw))
        bibliographic_shape = bool(
            in_refs
            and re.search(r"\b(19|20)\d{2}\b", raw)
            and re.search(r"\b(et al\.|Nature|Science|Cell|Lancet|NEJM|IEEE|ACM|Springer|Elsevier|bioRxiv|medRxiv|NeurIPS|ICLR|ICML|AAAI|Journal|Proceedings)\b", raw, re.I)
        )
        if len(raw) > 320 and not re.search(r"(doi\.org|10\.\d{4,9}/|arxiv|bioRxiv|medRxiv|openreview)", raw, re.I):
            continue
        if explicit_id or numbered_ref or bibliographic_shape:
            if len(raw) >= 20:
                refs.append(raw[:1000])
        if len(refs) >= limit:
            break
    return refs


def extract_abstract(lines: list[str]) -> str:
    for idx, line in enumerate(lines[:100]):
        if re.match(r"^(#+\s*)?(abstract|摘要|summary)\b[:：]?$", clean_space(line), re.I):
            parts = []
            for nxt in lines[idx + 1 : idx + 8]:
                if heading_level_from_prefix(nxt):
                    break
                value = clean_space(nxt)
                if value:
                    parts.append(value)
            return clean_space(" ".join(parts))[:1500]
        if re.match(r"^(摘要|abstract|summary)[:：]", clean_space(line), re.I):
            return clean_space(re.sub(r"^(摘要|abstract|summary)[:：]\s*", "", line, flags=re.I))[:1500]
    return ""


def parse_draft(path: Path, claim_limit: int) -> DraftDoc:
    text = read_text(path)
    lines = text.splitlines()
    return DraftDoc(
        path=str(path),
        title=infer_title(lines),
        chars=len(text),
        headings=extract_headings(lines),
        claims=extract_claims(text, claim_limit),
        references=extract_references(lines),
        abstract=extract_abstract(lines),
    )


def tokenize_heading(text: str) -> list[str]:
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text.lower())
    rough = text.split()
    tokens = []
    for token in rough:
        if token in STOP_HEADING_WORDS or len(token) <= 1:
            continue
        if token.isdigit() or re.fullmatch(r"(19|20)\d{2}", token):
            continue
        if re.fullmatch(r"[a-f0-9]{8,}", token):
            continue
        tokens.append(token)
    chinese_keywords = re.findall(r"(框架|架构|范式|智能体|自进化|大模型|多模态|评估|基准|治理|临床|证据|安全|路线图|知识图谱|RAG|强化学习|记忆|规划|工具调用|反思|迭代)", text, re.I)
    tokens.extend([kw.lower() for kw in chinese_keywords])
    return tokens


def heading_statistics(docs: list[DraftDoc]) -> tuple[Counter[str], Counter[str], dict[str, list[str]]]:
    title_counts: Counter[str] = Counter()
    token_counts: Counter[str] = Counter()
    source_map: dict[str, list[str]] = defaultdict(list)
    for doc in docs:
        for heading in doc.headings:
            title = heading["title"]
            title_counts[title] += 1
            source_map[title].append(Path(doc.path).name)
            token_counts.update(tokenize_heading(title))
    return title_counts, token_counts, source_map


def classify_heading(title: str) -> str:
    rules = [
        ("problem-and-timeliness", r"背景|引言|问题|挑战|why|timely|problem|motivation"),
        ("conceptual-framework", r"框架|架构|范式|taxonomy|framework|architecture|paradigm"),
        ("mechanism-or-technical-stack", r"机制|技术|RAG|知识图谱|多模态|工具|规划|记忆|强化学习|agent|tool|memory|planning|multimodal"),
        ("evidence-and-evaluation", r"证据|评估|基准|实验|验证|benchmark|evaluation|evidence|validation"),
        ("translation-and-governance", r"临床|治理|监管|安全|伦理|风险|转化|governance|clinical|safety|regulation|translation"),
        ("future-agenda", r"未来|展望|路线图|前景|agenda|future|outlook|roadmap"),
    ]
    for label, pattern in rules:
        if re.search(pattern, title, re.I):
            return label
    return "other"


def architecture_options(token_counts: Counter[str], topic: str, target_journal: str) -> list[dict[str, object]]:
    topic_text = topic or "the review topic"
    return [
        {
            "name": "Problem-to-framework spine",
            "best_for": "Nature Reviews-style narrative where the contribution is a new organizing lens.",
            "major_sections": [
                f"Why {topic_text} needs a new synthesis now",
                "A compact conceptual framework and terminology map",
                "Evidence layers: capabilities, evaluation, and failure modes",
                "Translation, governance, and boundary conditions",
                "A research agenda with testable next steps",
            ],
            "risk": "Can become too abstract if not anchored in verified evidence and display items.",
        },
        {
            "name": "Evidence-ladder spine",
            "best_for": "Medical AI or biomedical reviews where claims must be separated by evidence maturity.",
            "major_sections": [
                "From demonstrations to evidence: the field's maturity problem",
                "What is established, what is plausible, and what remains unvalidated",
                "Methods that change the evidence level rather than only performance",
                "Clinical or real-world workflow constraints",
                "Standards, reporting, and prospective agenda",
            ],
            "risk": "Can sound bureaucratic if the abstract and introduction overemphasize audit counts.",
        },
        {
            "name": "System-architecture spine",
            "best_for": "Technology reviews where readers need a reusable stack or closed-loop model.",
            "major_sections": [
                "The system-level problem and central thesis",
                "Core components and interactions",
                "Learning, feedback, and evaluation loops",
                "Failure modes, safeguards, and governance",
                "Design principles for the next generation",
            ],
            "risk": "Can turn into a component catalogue if every technology receives equal heading weight.",
        },
    ]


def abstract_audit(docs: list[DraftDoc]) -> str:
    rows = []
    red_flags = [
        ("search bookkeeping", r"召回|筛选|final-gate|候选|接受\s*\d+|降级\s*\d+|文献池|截至"),
        ("citation/legalistic detail", r"正式发表|预印本|撤回|引用|参考文献|published-only"),
        ("overlong or list-like", r"、.*、.*、.*、.*、"),
        ("unsupported grand claim", r"革命|颠覆|重塑|彻底|paradigm-shifting|revolutionary"),
    ]
    for doc in docs:
        abstract = doc.abstract
        if not abstract:
            continue
        flags = [label for label, pattern in red_flags if re.search(pattern, abstract, re.I)]
        rows.append((Path(doc.path).name, len(abstract), flags, abstract[:350]))
    if not rows:
        return "No explicit abstracts were detected in the supplied drafts.\n"
    lines = ["# Abstract Diagnosis", ""]
    for name, length, flags, preview in rows:
        lines.append(f"## {name}")
        lines.append(f"- Characters: {length}")
        lines.append(f"- Red flags: {', '.join(flags) if flags else 'none detected by heuristic'}")
        lines.append(f"- Preview: {preview}")
        lines.append("")
    lines.append("Nature-style abstract target: unstructured, broad, concise, thesis-forward, minimal specialist details, no references, no figure citations, and no search bookkeeping.")
    return "\n".join(lines).strip() + "\n"


def write_inventory(docs: list[DraftDoc], out_dir: Path) -> None:
    path = out_dir / "draft_inventory.json"
    path.write_text(json.dumps([asdict(doc) for doc in docs], ensure_ascii=False, indent=2), encoding="utf-8")


def write_claim_map(docs: list[DraftDoc], out_dir: Path) -> None:
    path = out_dir / "argument_evidence_map.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        fields = ["draft", "claim_id", "claim_role", "argument_time_role", "claim", "citation_markers", "evidence_status", "verification_status"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for doc in docs:
            for claim in doc.claims:
                row = {"draft": Path(doc.path).name, **claim}
                writer.writerow(row)


def short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def top_claims_by_role(docs: list[DraftDoc], role: str, limit: int = 8) -> list[tuple[str, str]]:
    out = []
    for doc in docs:
        for claim in doc.claims:
            if claim.get("claim_role") == role:
                out.append((Path(doc.path).name, claim["claim"]))
                if len(out) >= limit:
                    return out
    return out


def top_claims_by_time_role(docs: list[DraftDoc], role: str, limit: int = 6) -> list[tuple[str, str]]:
    out = []
    for doc in docs:
        for claim in doc.claims:
            if claim.get("argument_time_role") == role:
                out.append((Path(doc.path).name, claim["claim"]))
                if len(out) >= limit:
                    return out
    return out


def clean_search_query(raw: str) -> str:
    query = re.sub(r"^\s*(\[[0-9]+\]|[0-9]+[.)])\s*", "", raw)
    query = query.replace("\\-", "-")
    query = query.replace("\\[", "[").replace("\\]", "]")
    query = re.sub(r"^\s*\[?\d{4}\.\d{4,5}(?:v\d+)?\]?\s*", "", query)
    query = re.sub(r"https?://\S+", "", query)
    query = re.sub(r"\s*[-–—]\s*(arXiv|PubMed|PMC|OpenReview|bioRxiv|medRxiv)\s*,?\s*$", "", query, flags=re.I)
    query = re.sub(
        r"\s*\|\s*.*?(PubMed|PMC|arXiv|OpenReview|bioRxiv|medRxiv|ACS Publications|SpringerLink|ScienceDirect).*$",
        "",
        query,
        flags=re.I,
    )
    query = re.sub(r"\s*[-–—]\s*(arXiv|PubMed|PMC|OpenReview|bioRxiv|medRxiv)\s*,?\s*$", "", query, flags=re.I)
    query = re.sub(r"(访问时间为|accessed|retrieved)\s*[:：]?.*$", "", query, flags=re.I)
    query = re.sub(r"\s*\[[^\]]*$", "", query)
    query = re.sub(r"\s*\([^)]*(访问时间为|accessed|retrieved)[^)]*\)", "", query, flags=re.I)
    query = clean_space(query).strip(" -–—,，;；[]()")
    query = re.sub(r"\s*[-–—]\s*(arXiv|PubMed|PMC|OpenReview|bioRxiv|medRxiv)\s*,?\s*$", "", query, flags=re.I)
    return clean_space(query).strip(" -–—,，;；[]()")


def derive_search_tasks(docs: list[DraftDoc], token_counts: Counter[str], limit: int = 40) -> list[dict[str, str]]:
    tasks: list[dict[str, str]] = []
    seen: set[str] = set()
    for doc in docs:
        for ref in doc.references[:20]:
            query = clean_search_query(ref)[:220]
            key = query.lower()
            if len(query) >= 20 and key not in seen:
                seen.add(key)
                tasks.append(
                    {
                        "task_type": "verify_reference_lead",
                        "query": query,
                        "origin": Path(doc.path).name,
                        "recommended_sources": "PubMed,Crossref,OpenAlex,arXiv,OpenReview",
                    }
                )
            if len(tasks) >= limit:
                return tasks
    for term, _count in token_counts.most_common(20):
        if re.search(r"^[a-z0-9_\-\u4e00-\u9fff]{2,}$", term, re.I):
            query = clean_search_query(term)
            if not query:
                continue
            key = query.lower()
            if key not in seen:
                seen.add(key)
                tasks.append(
                    {
                        "task_type": "fill_framework_gap",
                        "query": query,
                        "origin": "heading_theme",
                        "recommended_sources": "OpenAlex,Crossref,Semantic Scholar,PubMed where biomedical",
                    }
                )
        if len(tasks) >= limit:
            return tasks
    for doc in docs:
        for claim in doc.claims:
            if claim.get("evidence_status") != "needs literature recall":
                continue
            query = clean_search_query(claim["claim"])[:180]
            key = query.lower()
            if key not in seen:
                seen.add(key)
                tasks.append(
                    {
                        "task_type": "support_claim_or_demote",
                        "query": query,
                        "origin": Path(doc.path).name,
                        "claim_role": claim.get("claim_role", ""),
                        "argument_time_role": claim.get("argument_time_role", ""),
                        "recommended_sources": "Crossref,OpenAlex,PubMed/arXiv/OpenReview by field",
                    }
                )
            if len(tasks) >= limit:
                return tasks
    return tasks


def write_literature_tasks(docs: list[DraftDoc], token_counts: Counter[str], out_dir: Path) -> list[dict[str, str]]:
    tasks = derive_search_tasks(docs, token_counts)
    txt = out_dir / "literature_search_tasks.txt"
    csv_path = out_dir / "literature_search_tasks.csv"
    txt.write_text("\n".join(task["query"] for task in tasks) + ("\n" if tasks else ""), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["task_type", "query", "origin", "claim_role", "argument_time_role", "recommended_sources"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(tasks)
    return tasks


def write_framework_blackboard(args: argparse.Namespace, docs: list[DraftDoc], out_dir: Path, tasks: list[dict[str, str]]) -> None:
    thesis_claims = top_claims_by_role(docs, "thesis_or_framework", 10)
    evidence_claims = top_claims_by_role(docs, "evidence_claim", 10)
    limitation_claims = top_claims_by_role(docs, "limitation_or_gap", 10)
    history_claims = top_claims_by_time_role(docs, "background_or_history", 6)
    current_claims = top_claims_by_time_role(docs, "current_evidence", 6)
    future_claims = top_claims_by_time_role(docs, "future_agenda", 6)
    lines = [
        "# Framework Evidence Blackboard",
        "",
        "This board is the shared working surface for draft readers, literature agents, argument builders, and the user. Update it before drafting.",
        "",
        "## Project",
        "",
        f"- Topic: {args.topic or '[not specified]'}",
        f"- Target journal family: {args.target_journal}",
        "- Framework status: awaiting user alignment",
        "- Evidence status: preliminary draft-derived leads; literature recall still required",
        "",
        "## Candidate Thesis Claims",
        "",
    ]
    if thesis_claims:
        for source, claim in thesis_claims:
            lines.append(f"- `{source}`: {claim}")
    else:
        lines.append("- [No thesis-like claim detected; user/Codex must formulate one.]")
    lines.extend(["", "## Candidate Evidence Claims", ""])
    if evidence_claims:
        for source, claim in evidence_claims:
            lines.append(f"- `{source}`: {claim}")
    else:
        lines.append("- [No evidence-like claim detected.]")
    lines.extend(["", "## Limitation / Gap Claims", ""])
    if limitation_claims:
        for source, claim in limitation_claims:
            lines.append(f"- `{source}`: {claim}")
    else:
        lines.append("- [No gap-like claim detected.]")
    lines.extend(["", "## Past-Present-Future Draft Arc", ""])
    for heading, group in [
        ("Historical foundations / past", history_claims),
        ("Current evidence / present", current_claims),
        ("Future agenda", future_claims),
    ]:
        lines.append(f"### {heading}")
        if group:
            for source, claim in group:
                lines.append(f"- `{source}`: {claim}")
        else:
            lines.append("- [No draft-derived claim detected.]")
        lines.append("")
    lines.extend(["", "## Literature Tasks That Must Run Before Framework Approval", ""])
    if tasks:
        for task in tasks[:20]:
            lines.append(f"- [{task['task_type']}] {task['query']} ({task['recommended_sources']})")
    else:
        lines.append("- [No search tasks derived.]")
    lines.extend(
        [
            "",
            "## Working Rules",
            "",
            "- Framework claims without evidence become search tasks, not manuscript prose.",
            "- A first-level section cannot be approved if it has no candidate evidence stream.",
            "- Supplemental search must serve the draft-derived argument map; interesting off-map papers stay in the supplemental pool.",
            "- Search results can demote, qualify, or challenge the framework, but they should not replace the draft-derived logic before user discussion.",
            "- User approval is required before moving from framework board to full drafting.",
            "",
            "## User Decisions Needed",
            "",
            "- Choose one central thesis.",
            "- Choose one architecture spine.",
            "- Approve 3-5 first-level sections.",
            "- Identify must-keep and must-delete draft elements.",
            "- Decide whether preprints can appear as background only or be excluded entirely.",
        ]
    )
    (out_dir / "framework_evidence_blackboard.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_material_passport(args: argparse.Namespace, docs: list[DraftDoc], out_dir: Path) -> None:
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    materials = []
    for doc in docs:
        materials.append(
            {
                "material_id": "draft-" + short_hash(doc.path),
                "kind": "ai_generated_review_draft",
                "path": doc.path,
                "version_label": "draft_input_v1",
                "data_access_level": "raw",
                "verification_status": "unverified",
                "produced_by": "external_ai_or_user",
                "consumed_by": ["draft_logic_framework", "draft_source_audit"],
                "notes": "Useful as scaffold; claims and references require verification.",
            }
        )
    output_files = [
        "draft_logic_framework.md",
        "argument_evidence_map.csv",
        "abstract_diagnosis.md",
        "user_alignment_questions.md",
        "framework_evidence_blackboard.md",
        "literature_search_tasks.csv",
        "draft_inventory.json",
    ]
    for filename in output_files:
        materials.append(
            {
                "material_id": "framework-" + short_hash(str(out_dir / filename)),
                "kind": "framework_artifact",
                "path": str(out_dir / filename),
                "version_label": "framework_v1",
                "data_access_level": "redacted",
                "verification_status": "needs_user_approval",
                "produced_by": "draft_logic_framework",
                "consumed_by": ["literature_retriever", "argument_builder", "outline_architect", "reviewer_auditor"],
                "notes": "Do not draft manuscript until framework and evidence gaps are reviewed.",
            }
        )
    passport = {
        "schema_name": "top_journal_review_material_passport",
        "schema_version": "0.1",
        "generated_at": now,
        "topic": args.topic,
        "target_journal": args.target_journal,
        "current_stage": "framework_alignment",
        "checkpoint": {
            "required": True,
            "question": "Approve one thesis and 3-5 first-level sections before drafting?",
            "status": "pending_user_decision",
        },
        "materials": materials,
        "reset_boundary": [
            {
                "kind": "boundary",
                "stage": "framework_alignment",
                "hash": short_hash(now + str(out_dir)),
                "next": "literature_recall_and_argument_binding",
                "verification_status": "UNVERIFIED",
                "pending_decision": {
                    "question": "Which architecture spine should govern the review?",
                    "options": [
                        {"value": "problem_to_framework", "next_stage": "literature_recall_and_argument_binding"},
                        {"value": "evidence_ladder", "next_stage": "literature_recall_and_argument_binding"},
                        {"value": "system_architecture", "next_stage": "literature_recall_and_argument_binding"},
                        {"value": "redo_framework", "next_stage": "framework_alignment"},
                    ],
                },
            }
        ],
    }
    (out_dir / "framework_material_passport.json").write_text(json.dumps(passport, ensure_ascii=False, indent=2), encoding="utf-8")


def write_framework(args: argparse.Namespace, docs: list[DraftDoc], out_dir: Path) -> None:
    title_counts, token_counts, source_map = heading_statistics(docs)
    class_counts = Counter()
    for title, count in title_counts.items():
        class_counts[classify_heading(title)] += count

    repeated = [(title, count, source_map[title]) for title, count in title_counts.most_common(30) if count > 1]
    unique = [(title, source_map[title]) for title, count in title_counts.most_common(80) if count == 1]
    top_level_counts = []
    for doc in docs:
        top_n = sum(1 for item in doc.headings if item["level"] == "1")
        top_level_counts.append((Path(doc.path).name, top_n))

    options = architecture_options(token_counts, args.topic, args.target_journal)
    lines = [
        "# Draft Logic Framework",
        "",
        "## Purpose",
        "",
        "This is a pre-drafting alignment brief. Use it to discuss the review architecture with the user before writing prose.",
        "",
        "## Draft Set",
        "",
    ]
    for doc in docs:
        lines.append(f"- `{Path(doc.path).name}`: {doc.chars} chars, {len(doc.headings)} headings, {len(doc.claims)} claim-like sentences, {len(doc.references)} reference-like lines")
    lines.extend(
        [
            "",
            "## Architecture Warnings",
            "",
        ]
    )
    for name, count in top_level_counts:
        status = "OK" if count <= 6 else "TOO MANY TOP-LEVEL SECTIONS"
        lines.append(f"- `{name}`: {count} top-level headings -> {status}")
    lines.extend(
        [
            "",
            "Top-journal heuristic: aim for one central thesis, 3-5 major body movements, and 5-7 display items. Avoid 8-9 parallel top-level sections unless the journal specifically asks for a handbook-like format.",
            "",
            "## Recurring Frames Across Drafts",
            "",
        ]
    )
    if repeated:
        for title, count, sources in repeated[:20]:
            lines.append(f"- {title} ({count} drafts/occurrences; sources: {', '.join(sorted(set(sources))[:5])})")
    else:
        lines.append("- No exact repeated headings detected; use thematic clusters below.")
    lines.extend(
        [
            "",
            "## Thematic Weight From Headings",
            "",
        ]
    )
    for label, count in class_counts.most_common():
        lines.append(f"- {label}: {count}")
    lines.extend(
        [
            "",
            "## Frequent Heading Terms",
            "",
            ", ".join(f"{term} ({count})" for term, count in token_counts.most_common(25)) or "[none]",
            "",
            "## Candidate Architecture Options",
            "",
        ]
    )
    for idx, option in enumerate(options, 1):
        lines.append(f"### Option {idx}: {option['name']}")
        lines.append(f"- Best for: {option['best_for']}")
        lines.append(f"- Risk: {option['risk']}")
        lines.append("- Major sections:")
        for section in option["major_sections"]:
            lines.append(f"  - {section}")
        lines.append("")
    lines.extend(
        [
            "## Accepted Draft Elements To Preserve",
            "",
            "- Extract the strongest thesis sentences from `argument_evidence_map.csv` and keep only those that support the chosen spine.",
            "- Preserve unique draft insights that clarify mechanisms, controversies, or governance boundaries.",
            "- Convert useful but unsupported ideas into search tasks rather than prose.",
            "- Treat every claim without a citation marker as a literature-recall task before it becomes manuscript prose.",
            "",
            "## Elements To Demote Or Remove",
            "",
            "- Search logs, final-gate counts, and citation-policy bookkeeping should move to methods/supplement or internal audit notes, not the abstract.",
            "- Parallel technology catalogues should be collapsed into a framework, evidence ladder, or system architecture.",
            "- Web/blog claims should not become scientific claims until replaced by verified paper evidence.",
            "",
        ]
    )
    if unique:
        lines.extend(["## Unique Heading Leads", ""])
        for title, sources in unique[:25]:
            lines.append(f"- {title} (source: {', '.join(sorted(set(sources))[:3])})")
    (out_dir / "draft_logic_framework.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_alignment_questions(args: argparse.Namespace, out_dir: Path) -> None:
    questions = textwrap.dedent(
        f"""
        # User Alignment Questions

        Discuss these before drafting:

        1. What is the one-sentence thesis of the review?
        2. Which architecture should govern the paper: problem-to-framework, evidence ladder, or system architecture?
        3. Which 3-5 major body sections should survive? Which draft headings should be merged or deleted?
        4. Which claims are non-negotiable, and which are only hypotheses or future agenda?
        5. Which references are essential evidence rather than decorative background?
        6. What should Figure 1 make instantly clear to a non-specialist reader?
        7. What should be left out of the abstract even if it is important for the audit trail?

        Default recommendation for {args.target_journal or "a Nature-style review"}:

        - Keep the abstract thesis-forward and reader-facing.
        - Keep top-level body sections to about 3-5 major movements.
        - Move literature-pool counts, final-gate results, and verification logs to methods/reporting notes.
        - Make display items carry the framework, evidence ladder, and roadmap.
        """
    ).strip()
    (out_dir / "user_alignment_questions.md").write_text(questions + "\n", encoding="utf-8")


def write_summary(docs: list[DraftDoc], out_dir: Path) -> None:
    title_counts, token_counts, source_map = heading_statistics(docs)
    summary = {
        "drafts": len(docs),
        "total_claims": sum(len(doc.claims) for doc in docs),
        "total_references_like_lines": sum(len(doc.references) for doc in docs),
        "top_heading_terms": token_counts.most_common(20),
        "repeated_headings": [
            {"heading": title, "count": count, "sources": sorted(set(source_map[title]))}
            for title, count in title_counts.most_common(30)
            if count > 1
        ],
        "outputs": [
            "draft_logic_framework.md",
            "argument_evidence_map.csv",
            "abstract_diagnosis.md",
            "user_alignment_questions.md",
            "framework_evidence_blackboard.md",
            "literature_search_tasks.csv",
            "framework_material_passport.json",
            "draft_inventory.json",
        ],
    }
    (out_dir / "logic_framework_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a discussion-ready logic framework from review drafts.")
    parser.add_argument("--draft", action="append", default=[], help="Draft file path (.md/.txt/.docx); repeatable.")
    parser.add_argument("--draft-dir", action="append", default=[], help="Directory containing draft files; repeatable.")
    parser.add_argument("--out-dir", default="./review-data/03_framework/logic_framework")
    parser.add_argument("--topic", default="")
    parser.add_argument("--target-journal", default="Nature Reviews-style journal")
    parser.add_argument("--claim-limit-per-draft", type=int, default=80)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    drafts = discover_drafts(args)
    if not drafts:
        raise SystemExit("No supported drafts found. Use --draft or --draft-dir with .md, .txt, or .docx files.")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = [parse_draft(path, args.claim_limit_per_draft) for path in drafts]
    _title_counts, token_counts, _source_map = heading_statistics(docs)
    write_inventory(docs, out_dir)
    write_claim_map(docs, out_dir)
    tasks = write_literature_tasks(docs, token_counts, out_dir)
    write_framework(args, docs, out_dir)
    (out_dir / "abstract_diagnosis.md").write_text(abstract_audit(docs), encoding="utf-8")
    write_alignment_questions(args, out_dir)
    write_framework_blackboard(args, docs, out_dir, tasks)
    write_material_passport(args, docs, out_dir)
    write_summary(docs, out_dir)
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "drafts": len(docs),
                "framework": str(out_dir / "draft_logic_framework.md"),
                "claim_map": str(out_dir / "argument_evidence_map.csv"),
                "blackboard": str(out_dir / "framework_evidence_blackboard.md"),
                "passport": str(out_dir / "framework_material_passport.json"),
                "literature_tasks": str(out_dir / "literature_search_tasks.csv"),
                "alignment_questions": str(out_dir / "user_alignment_questions.md"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
