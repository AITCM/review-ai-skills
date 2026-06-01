#!/usr/bin/env python3
"""Verify draft-native reference clues before they enter the main pool.

The intended input is `draft_assets/candidate_paper_clues.csv` from
`draft_citation_assets.py`. Verification is PubMed-first for biomedical
projects, then DOI/title metadata fallback through Crossref and OpenAlex.
PMID/DOI values from drafts are treated as strong clues, not truth: when an
identifier resolves to a title that conflicts with the draft title, the verifier
falls back to title/bibliographic search instead of accepting the identifier.
Preprint-only records are kept as leads, not final citation candidates.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_UA = "review-ai-skills/1.0"
PREPRINT_MARKERS = [
    "arxiv",
    "openreview",
    "biorxiv",
    "medrxiv",
    "arxiv.org",
    "openreview.net",
    "biorxiv.org",
    "medrxiv.org",
    "researchsquare.com",
    "10.48550/arxiv",
    "10.1101/",
]
CORRECTION_TITLE_MARKERS = (
    "publisher correction",
    "correction:",
    "erratum",
    "corrigendum",
    "retraction",
    "withdrawn",
)
TITLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "the",
    "to",
    "toward",
    "towards",
    "using",
    "with",
}
ACCEPTED_PUBLICATION_TYPES = {
    "journal-article",
    "proceedings-article",
    "article",
    "conference-paper",
    "book-chapter",
}
VERIFIED_STATUSES = {
    "verified_pubmed",
    "verified_crossref",
    "verified_openalex",
    "verified_openreview",
    "verified_conference_page",
    "verified_publisher_url",
}
VERSION_DUPLICATE_STATUS = "duplicate_formal_version_available"
OFFICIAL_CONFERENCE_SOURCES = (
    "openreview",
    "openreview.net",
    "papers.nips.cc",
    "proceedings.neurips.cc",
    "proceedings.mlr.press",
)
OFFICIAL_CONFERENCE_VENUES = (
    "iclr",
    "international conference on learning representations",
    "neurips",
    "nips",
    "advances in neural information processing systems",
    "icml",
    "international conference on machine learning",
)
PUBLISHER_HOST_MARKERS = (
    "nature.com",
    "science.org",
    "cell.com",
    "thelancet.com",
    "nejm.org",
    "bmj.com",
    "jamanetwork.com",
    "springer.com",
    "link.springer.com",
    "sciencedirect.com",
    "wiley.com",
    "onlinelibrary.wiley.com",
    "tandfonline.com",
    "oup.com",
    "academic.oup.com",
    "cambridge.org",
    "frontiersin.org",
    "mdpi.com",
    "plos.org",
    "ieee.org",
    "dl.acm.org",
)
STATUS_RANK = {
    "verified_pubmed": 90,
    "verified_crossref": 82,
    "verified_openalex": 78,
    "verified_publisher_url": 74,
    "verified_conference_page": 72,
    "verified_openreview": 70,
    "preprint_lead": 30,
    "needs_api_verification": 10,
    "unverified_delete_or_replace": 0,
}
OUTPUT_FIELDS = [
    "draft",
    "ref_number",
    "draft_candidate_title",
    "title",
    "authors",
    "year",
    "journal",
    "source",
    "source_kind_verified",
    "paper_id",
    "pmid",
    "doi",
    "url",
    "abstract",
    "publication_type",
    "title_similarity",
    "verification_status",
    "verification_confidence",
    "pool_status",
    "recall_queries",
    "claim_supported",
    "limitations",
    "use_in_review",
    "notes",
    "raw_reference",
    "draft_doi",
    "draft_pmid",
    "draft_urls",
    "draft_source_kind",
    "cited_in_body",
    "candidate_id",
    "claim_id",
    "candidate_type",
    "pubmed_query",
    "raw_evidence_excerpt",
    "why_relevant",
    "verified_at",
]
SEARCH_TRACE_FIELDS = [
    "draft",
    "ref_number",
    "candidate_id",
    "draft_candidate_title",
    "route",
    "mode",
    "query",
    "attempt_note",
    "hit_count",
    "returned_ids",
    "candidate_pmid",
    "candidate_doi",
    "candidate_title",
    "candidate_year",
    "candidate_journal",
    "title_similarity",
    "decision",
    "decision_note",
]
WEAK_TITLE_MARKERS = [
    "official page",
    "developer documentation",
    "documentation",
    "recommendations for a predetermined change control plan",
    "health and food safety directorate",
    "国家药品监督管理局",
    "科技伦理审查办法",
]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def clean_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_doi(text: Any) -> str:
    value = clean_space(text).lower()
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value)
    value = re.sub(r"^doi\s*:\s*", "", value)
    return value.strip().rstrip(".,;)")


def doi_from_text(text: Any) -> str:
    """Extract a DOI from free text or publisher URLs when possible."""
    value = urllib.parse.unquote(clean_space(text))
    if not value:
        return ""
    match = re.search(r"\b(10\.\d{4,9}/[^\s\"<>]+)", value, flags=re.I)
    if match:
        return normalize_doi(match.group(1))
    match = re.search(r"nature\.com/articles/(s\d{5}-\d{3}-\d{5}-[0-9a-z]+)", value, flags=re.I)
    if match:
        return normalize_doi("10.1038/" + match.group(1))
    return ""


def first_list_value(text: Any) -> str:
    for part in re.split(r";|\|", clean_space(text)):
        part = part.strip()
        if part:
            return part
    return ""


def normalize_title(text: str) -> str:
    text = clean_space(text).lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_similarity(a: str, b: str) -> float:
    left = {w for w in normalize_title(a).split() if len(w) > 2 and w not in TITLE_STOPWORDS}
    right = {w for w in normalize_title(b).split() if len(w) > 2 and w not in TITLE_STOPWORDS}
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left), len(right))


def title_matches(candidate_title: str, found_title: str, min_similarity: float) -> bool:
    if not candidate_title or not found_title:
        return True
    candidate_norm = normalize_title(candidate_title)
    found_norm = normalize_title(found_title)
    candidate_tokens = [token for token in candidate_norm.split() if token not in TITLE_STOPWORDS]
    if candidate_tokens and len(candidate_tokens) <= 4 and (len(candidate_tokens) > 1 or len(candidate_tokens[0]) >= 5):
        if candidate_norm and candidate_norm in found_norm:
            return True
    return title_similarity(candidate_title, found_title) >= min_similarity


def is_web_challenge_title(title: Any) -> bool:
    normalized = normalize_title(clean_space(title))
    return normalized in {
        "client challenge",
        "access denied",
        "just a moment",
        "forbidden",
        "request blocked",
        "security check",
    }


def parse_year(value: Any) -> int:
    match = re.search(r"\b(18|19|20|21)\d{2}\b", clean_space(value))
    return int(match.group(0)) if match else 0


def attach_identity_note(record: dict[str, str], note: str) -> dict[str, str]:
    if note:
        record = dict(record)
        record["identity_resolution_notes"] = note
    return record


def normalize_ref_number(value: Any) -> str:
    text = clean_space(value)
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def host_path(url: str) -> str:
    parsed = urllib.parse.urlparse(clean_space(url))
    return f"{parsed.netloc.lower()}{parsed.path.lower()}"


def has_official_conference_context(record: dict[str, Any]) -> bool:
    text = " ".join(
        clean_space(record.get(field)).lower()
        for field in [
            "source",
            "url",
            "journal",
            "publication_type",
            "title",
            "raw_reference",
            "draft_source_kind",
            "notes",
        ]
    )
    has_source = any(marker in text for marker in OFFICIAL_CONFERENCE_SOURCES)
    has_venue = any(marker in text for marker in OFFICIAL_CONFERENCE_VENUES)
    pub_type = clean_space(record.get("publication_type")).lower()
    source_kind = clean_space(record.get("source_kind_verified")).lower()
    return (
        "official_conference" in source_kind
        or ("conference" in pub_type and (has_source or has_venue))
        or (has_source and has_venue)
    )


def is_preprint_lead(row: dict[str, Any]) -> bool:
    text = " ".join(clean_space(row.get(field)).lower() for field in ["raw_reference", "doi", "urls", "candidate_title"])
    return any(marker in text for marker in PREPRINT_MARKERS)


def is_preprint_record(record: dict[str, Any]) -> bool:
    if has_official_conference_context(record):
        return False
    text = " ".join(
        clean_space(record.get(field)).lower()
        for field in ["doi", "url", "journal", "source", "title", "publication_type"]
    )
    return any(marker in text for marker in PREPRINT_MARKERS)


def is_correction_like_title(title: Any) -> bool:
    text = normalize_title(clean_space(title))
    if not text:
        return False
    return any(text.startswith(normalize_title(marker)) for marker in CORRECTION_TITLE_MARKERS)


def is_correction_record(record: dict[str, Any]) -> bool:
    title = clean_space(record.get("title"))
    pub_type = clean_space(record.get("publication_type")).lower()
    return is_correction_like_title(title) or any(marker in pub_type for marker in ["erratum", "correction", "retraction"])


def is_weak_title_clue(row: dict[str, Any]) -> bool:
    title = clean_space(row.get("candidate_title"))
    raw = clean_space(row.get("raw_reference"))
    text = f"{title} {raw}".lower()
    if any(marker in text for marker in WEAK_TITLE_MARKERS):
        return True
    normalized = normalize_title(title)
    tokens = [token for token in normalized.split() if token not in TITLE_STOPWORDS]
    if len(tokens) <= 3 and re.search(r"\b(19|20)\d{2}\b", title):
        return True
    if re.fullmatch(r"\*?[\w &-]+\*?,?\s*(19|20)\d{2}\.?", title, flags=re.I):
        return True
    return False


def row_key(row: dict[str, Any]) -> str:
    draft = clean_space(row.get("draft"))
    ref_number = normalize_ref_number(row.get("ref_number"))
    title = clean_space(row.get("draft_candidate_title") or row.get("candidate_title") or row.get("title"))
    if "draft_candidate_title" in row or "verification_status" in row:
        doi = normalize_doi(row.get("draft_doi"))
    else:
        doi = normalize_doi(row.get("draft_doi") or first_list_value(row.get("doi")))
    return "||".join([draft, ref_number, normalize_title(title), doi])


def is_accepted_publication_type(record: dict[str, Any]) -> bool:
    if has_official_conference_context(record):
        return True
    source = clean_space(record.get("source")).lower()
    if source in {"pubmed", "publisher_url"}:
        return True
    pub_type = clean_space(record.get("publication_type")).lower()
    if not pub_type:
        return source in {"openalex"}
    return pub_type in ACCEPTED_PUBLICATION_TYPES


def preprint_result(base: dict[str, str], row: dict[str, Any], found: dict[str, str] | None = None) -> dict[str, str]:
    found = found or {}
    return {
        **base,
        "verification_status": "preprint_lead",
        "verification_confidence": "lead_only",
        "title": found.get("title") or base["draft_candidate_title"],
        "authors": found.get("authors", ""),
        "year": found.get("year", ""),
        "source": found.get("source") or "preprint_or_review_platform",
        "source_kind_verified": found.get("source_kind_verified", "preprint_or_review_platform"),
        "paper_id": found.get("paper_id", ""),
        "pmid": found.get("pmid", ""),
        "doi": found.get("doi") or base["draft_doi"],
        "url": found.get("url") or first_list_value(row.get("urls")),
        "journal": found.get("journal", ""),
        "abstract": found.get("abstract", ""),
        "publication_type": found.get("publication_type", ""),
        "title_similarity": found.get("title_similarity", ""),
        "pool_status": "preprint_only",
        "notes": "Keep as a search lead or background only unless user explicitly allows preprints.",
    }


def rejected_result(base: dict[str, str], row: dict[str, Any], note: str) -> dict[str, str]:
    return {
        **base,
        "verification_status": "unverified_delete_or_replace",
        "verification_confidence": "none",
        "title": base["draft_candidate_title"],
        "authors": "",
        "year": "",
        "source": "",
        "source_kind_verified": "",
        "paper_id": "",
        "pmid": "",
        "doi": base["draft_doi"],
        "url": first_list_value(row.get("urls")),
        "journal": "",
        "abstract": "",
        "publication_type": "",
        "title_similarity": "",
        "pool_status": "rejected",
        "notes": note,
    }


def http_text(url: str, timeout: int, user_agent: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def http_json(url: str, timeout: int, user_agent: str) -> dict[str, Any] | None:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception:
        return None


def html_attr(tag: str, name: str) -> str:
    match = re.search(rf"\b{name}\s*=\s*(['\"])(.*?)\1", tag, flags=re.I | re.S)
    return html.unescape(clean_space(match.group(2))) if match else ""


def html_metadata(markup: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    for tag in re.findall(r"<meta\b[^>]*>", markup, flags=re.I | re.S):
        key = (html_attr(tag, "name") or html_attr(tag, "property")).lower()
        content = html_attr(tag, "content")
        if key and content and key not in meta:
            meta[key] = content
    title = meta.get("citation_title") or meta.get("dc.title") or meta.get("og:title") or ""
    if not title:
        match = re.search(r"<title[^>]*>(.*?)</title>", markup, flags=re.I | re.S)
        if match:
            title = html.unescape(clean_space(re.sub(r"<[^>]+>", " ", match.group(1))))
    return {
        "title": re.sub(r"\s+\|\s+(OpenReview|Nature|ScienceDirect|SpringerLink).*$", "", title, flags=re.I),
        "doi": normalize_doi(meta.get("citation_doi") or meta.get("dc.identifier") or ""),
        "journal": clean_space(meta.get("citation_journal_title") or meta.get("citation_conference_title") or meta.get("citation_publisher") or ""),
        "year": clean_space((meta.get("citation_publication_date") or meta.get("citation_online_date") or "")[:4]),
        "authors": clean_space("; ".join(value for key, value in meta.items() if key == "citation_author")),
        "abstract": clean_space(meta.get("description") or meta.get("og:description") or ""),
    }


def venue_from_text(text: str) -> str:
    lowered = text.lower()
    if "iclr" in lowered or "learning representations" in lowered:
        return "International Conference on Learning Representations"
    if "neurips" in lowered or "nips" in lowered or "neural information processing systems" in lowered:
        return "Advances in Neural Information Processing Systems"
    if "icml" in lowered or "machine learning" in lowered:
        return "International Conference on Machine Learning"
    return ""


def first_url(row: dict[str, Any]) -> str:
    return first_list_value(row.get("urls") or row.get("url"))


def candidate_doi(row: dict[str, Any]) -> str:
    return (
        normalize_doi(first_list_value(row.get("doi")))
        or doi_from_text(first_url(row))
        or doi_from_text(row.get("raw_reference"))
    )


def pubmed_params(args: argparse.Namespace) -> dict[str, str]:
    params = {"tool": "review-ai-skills"}
    if args.email:
        params["email"] = args.email
    if args.api_key:
        params["api_key"] = args.api_key
    return params


def pubmed_esearch(term: str, args: argparse.Namespace) -> list[str]:
    retmax = max(int(getattr(args, "pubmed_retmax", 10) or 10), 1)
    params = {
        "db": "pubmed",
        "retmode": "json",
        "retmax": str(retmax),
        "term": term,
        **pubmed_params(args),
    }
    url = f"{PUBMED_BASE}/esearch.fcgi?" + urllib.parse.urlencode(params)
    data = http_json(url, timeout=args.timeout, user_agent=args.user_agent)
    return (((data or {}).get("esearchresult") or {}).get("idlist") or [])[:retmax]


def pubmed_efetch(pmids: list[str], args: argparse.Namespace) -> ET.Element | None:
    if not pmids:
        return None
    params = {
        "db": "pubmed",
        "retmode": "xml",
        "id": ",".join(pmids),
        **pubmed_params(args),
    }
    url = f"{PUBMED_BASE}/efetch.fcgi?" + urllib.parse.urlencode(params)
    try:
        return ET.fromstring(http_text(url, timeout=args.timeout, user_agent=args.user_agent))
    except Exception:
        return None


def pubmed_quote(text: Any) -> str:
    value = clean_space(text).replace('"', " ")
    value = re.sub(r"\s+", " ", value).strip()
    return f'"{value}"' if value else ""


def extract_system_phrases(row: dict[str, Any], title: str) -> list[str]:
    context = " ".join(
        clean_space(row.get(field))
        for field in ["candidate_title", "raw_reference", "pubmed_query", "raw_evidence_excerpt", "why_relevant"]
    )
    phrases: list[str] = []
    patterns = [
        r"\bAI\s+co-scientist\b",
        r"\b[A-Za-z0-9]+(?:[- ]?(?:GPT|Agent|Planner|Scientist|scientist))\b",
        r"\b(?:Coscientist|co-scientist|scGPT|GeneGPT|DrugGPT|CRISPR-GPT|BioPlanner|CellAgent)\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, context, flags=re.IGNORECASE):
            phrase = clean_space(match.group(0))
            if phrase and normalize_title(phrase) not in {normalize_title(item) for item in phrases}:
                phrases.append(phrase)
    if ":" in title:
        prefix = clean_space(title.split(":", 1)[0])
        if len(prefix.split()) >= 2 and normalize_title(prefix) not in {normalize_title(item) for item in phrases}:
            phrases.append(prefix)
    return phrases[:5]


def compact_title_terms(title: str, limit: int = 6) -> list[str]:
    tokens = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]+", title):
        normalized = normalize_title(token)
        if len(normalized) <= 2 or normalized in TITLE_STOPWORDS:
            continue
        if normalized not in tokens:
            tokens.append(normalized)
    return tokens[:limit]


def pubmed_title_query_variants(row: dict[str, Any], title: str) -> list[tuple[str, str]]:
    variants: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(term: str, note: str) -> None:
        term = clean_space(term)
        key = term.lower()
        if term and key not in seen:
            seen.add(key)
            variants.append((term, note))

    quoted_title = pubmed_quote(title)
    if quoted_title:
        add(f"{quoted_title}[Title]", "Exact title PubMed search")

    for phrase in extract_system_phrases(row, title):
        quoted = pubmed_quote(phrase)
        if quoted:
            add(f"{quoted}[Title/Abstract]", f"System-name PubMed deep dive for {phrase}")

    if title and ":" in title:
        prefix = clean_space(title.split(":", 1)[0])
        if prefix:
            add(f"{pubmed_quote(prefix)}[Title]", "Main-title PubMed fallback")

    terms = compact_title_terms(title)
    if len(terms) >= 3:
        add(" AND ".join(f"{term}[Title/Abstract]" for term in terms[:4]), "Compact title-token PubMed fallback")
    if title:
        add(title, "Broad title/bibliographic PubMed search")
    return variants[:7]


def trace_context(row: dict[str, Any]) -> dict[str, str]:
    return {
        "draft": clean_space(row.get("draft")),
        "ref_number": clean_space(row.get("ref_number")),
        "candidate_id": clean_space(row.get("candidate_id") or row.get("ref_number")),
        "draft_candidate_title": clean_space(row.get("candidate_title")),
    }


def pubmed_trace_row(
    row: dict[str, Any],
    mode: str,
    query: str,
    note: str,
    ids: list[str],
    decision: str,
    decision_note: str = "",
    record: dict[str, str] | None = None,
    similarity: str = "",
) -> dict[str, str]:
    record = record or {}
    return {
        **trace_context(row),
        "route": "pubmed",
        "mode": mode,
        "query": query,
        "attempt_note": note,
        "hit_count": str(len(ids)),
        "returned_ids": ";".join(ids[:20]),
        "candidate_pmid": clean_space(record.get("pmid")),
        "candidate_doi": normalize_doi(record.get("doi")),
        "candidate_title": clean_space(record.get("title")),
        "candidate_year": clean_space(record.get("year")),
        "candidate_journal": clean_space(record.get("journal")),
        "title_similarity": similarity,
        "decision": decision,
        "decision_note": decision_note,
    }


def xml_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return clean_space("".join(node.itertext()))


def pubmed_record_from_article(article: ET.Element) -> dict[str, str]:
    pmid = xml_text(article.find(".//PMID"))
    title = xml_text(article.find(".//ArticleTitle"))
    journal = xml_text(article.find(".//Journal/Title"))
    year = ""
    for path in [".//PubDate/Year", ".//ArticleDate/Year"]:
        year = xml_text(article.find(path))
        if year:
            break
    doi = ""
    for node in article.findall(".//ArticleId"):
        if (node.attrib.get("IdType") or "").lower() == "doi":
            doi = normalize_doi(xml_text(node))
            break
    authors: list[str] = []
    for author in article.findall(".//Author")[:12]:
        name = clean_space(" ".join([xml_text(author.find("ForeName")), xml_text(author.find("LastName"))]))
        if name:
            authors.append(name)
    abstract = xml_text(article.find(".//Abstract"))
    pub_types = [xml_text(node) for node in article.findall(".//PublicationTypeList/PublicationType")]
    publication_type = "; ".join([value for value in pub_types if value]) or "journal-article"
    return {
        "title": title,
        "authors": "; ".join(authors),
        "year": year,
        "source": "pubmed",
        "paper_id": pmid,
        "pmid": pmid,
        "doi": doi,
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
        "journal": journal,
        "abstract": abstract,
        "publication_type": publication_type,
    }


def pubmed_candidate_rank(
    record: dict[str, str],
    mode: str,
    title: str,
    doi: str,
    pmid: str,
    args: argparse.Namespace,
) -> tuple[int, int, int, int, int, int, int, int, int]:
    similarity = title_similarity(title, record.get("title", "")) if title else 1.0
    record_doi = normalize_doi(record.get("doi"))
    record_pmid = clean_space(record.get("pmid"))
    is_preprint = is_preprint_record(record)
    is_correction = is_correction_record(record)
    doi_match = bool(doi and record_doi and doi == record_doi)
    pmid_match = bool(pmid and record_pmid and pmid == record_pmid)
    formal_bonus = 1 if not is_preprint and not is_correction else 0
    correction_bonus = 1 if not is_correction else 0
    preprint_bonus = 1 if not is_preprint else 0
    identity_bonus = 2 if pmid_match else 0
    if doi_match and not is_preprint:
        identity_bonus += 2
    elif doi_match:
        identity_bonus += 1
    mode_bonus = {"pmid": 3, "doi": 2, "title": 1}.get(mode, 0)
    return (
        formal_bonus,
        correction_bonus,
        preprint_bonus,
        int(similarity * 1000),
        identity_bonus,
        1 if record_doi else 0,
        1 if clean_space(record.get("abstract")) else 0,
        parse_year(record.get("year")),
        mode_bonus,
    )


def verify_pubmed(row: dict[str, Any], args: argparse.Namespace) -> dict[str, str] | None:
    pmid = first_list_value(row.get("pmid"))
    doi = candidate_doi(row)
    title = clean_space(row.get("candidate_title"))
    attempts: list[tuple[str, list[str], str, str]] = []
    trace: list[dict[str, str]] = []
    if pmid:
        attempts.append(("pmid", [pmid], f"PMID:{pmid}", "PMID resolved from draft clue"))
    if doi:
        query = f'"{doi}"[AID]'
        attempts.append(("doi", pubmed_esearch(query, args), query, "DOI resolved through PubMed ArticleId"))
    if title:
        for query, note in pubmed_title_query_variants(row, title):
            attempts.append(("title", pubmed_esearch(query, args), query, note))

    mismatch_notes: list[str] = []
    candidates: list[tuple[tuple[int, int, int, int, int, int, int, int, int], str, str, dict[str, str]]] = []
    seen_pmids: set[str] = set()
    for mode, ids, query, note in attempts:
        if not ids:
            trace.append(pubmed_trace_row(row, mode, query, note, ids, "no_hits"))
            continue
        root = pubmed_efetch(ids, args)
        articles = root.findall(".//PubmedArticle") if root is not None else []
        if not articles:
            trace.append(pubmed_trace_row(row, mode, query, note, ids, "efetch_empty", "IDs returned by esearch but efetch returned no articles."))
            continue
        for article in articles:
            record = pubmed_record_from_article(article)
            record_pmid = clean_space(record.get("pmid"))
            if record_pmid and record_pmid in seen_pmids:
                continue
            if record_pmid:
                seen_pmids.add(record_pmid)
            similarity = f"{title_similarity(title, record.get('title', '')):.3f}" if title and record.get("title") else ""
            if title and not title_matches(title, record.get("title", ""), args.min_title_similarity):
                if len(mismatch_notes) < 4:
                    mismatch_notes.append(
                        f"{mode} resolved to title `{record.get('title', '')}`; ignored because it did not match draft title."
                    )
                trace.append(pubmed_trace_row(row, mode, query, note, ids, "title_mismatch_ignored", "Returned title did not match the candidate clue.", record, similarity))
                continue
            trace.append(pubmed_trace_row(row, mode, query, note, ids, "candidate_ranked", "Title-compatible candidate kept for ranking.", record, similarity))
            candidates.append((pubmed_candidate_rank(record, mode, title, doi, pmid, args), mode, note, record))
    if not candidates:
        row["_search_trace"] = trace
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    _score, mode, note, record = candidates[0]
    if is_correction_record(record) and not is_correction_like_title(title):
        note = f"{note} PubMed returned only correction/erratum-like top matches after multi-hit ranking."
    record["verification_status"] = "verified_pubmed"
    record["verification_confidence"] = "high" if mode in {"pmid", "doi"} and not mismatch_notes else "medium"
    if mode == "title" and title and title_similarity(title, record.get("title", "")) >= 0.9:
        record["verification_confidence"] = "high"
    record["title_similarity"] = f"{title_similarity(title, record.get('title', '')):.3f}" if title and record.get("title") else ""
    ranked_note = (
        f"{note} PubMed multi-hit ranking inspected {len(candidates)} title-compatible candidate(s); "
        "formal non-correction records are preferred over preprints and correction notices."
    )
    trace.append(
        pubmed_trace_row(
            row,
            mode,
            "selected_from_ranked_candidates",
            note,
            [clean_space(record.get("pmid"))],
            "accepted_top_ranked",
            ranked_note,
            record,
            record.get("title_similarity", ""),
        )
    )
    record["_search_trace"] = trace
    return attach_identity_note(record, " ".join(mismatch_notes + [ranked_note]))


def crossref_record(item: dict[str, Any]) -> dict[str, str]:
    title = clean_space((item.get("title") or [""])[0])
    authors = []
    for author in item.get("author") or []:
        name = clean_space(" ".join([author.get("given") or "", author.get("family") or ""]))
        if name:
            authors.append(name)
    year = ""
    for field in ["published-print", "published-online", "published", "issued"]:
        parts = ((item.get(field) or {}).get("date-parts") or [])
        if parts and parts[0]:
            year = str(parts[0][0])
            break
    return {
        "title": title,
        "authors": "; ".join(authors[:12]),
        "year": year,
        "source": "crossref",
        "paper_id": clean_space(item.get("DOI")),
        "doi": normalize_doi(item.get("DOI")),
        "url": clean_space(item.get("URL")),
        "journal": clean_space((item.get("container-title") or [""])[0]),
        "abstract": clean_space(item.get("abstract")),
        "publication_type": clean_space(item.get("type")),
    }


def verify_crossref(row: dict[str, Any], args: argparse.Namespace) -> dict[str, str] | None:
    doi = candidate_doi(row)
    title = clean_space(row.get("candidate_title"))
    ua = f"{args.user_agent} (mailto:{args.email})" if args.email else args.user_agent
    mismatch_notes: list[str] = []
    if doi:
        url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
        data = http_json(url, timeout=args.timeout, user_agent=ua)
        item = (data or {}).get("message") if (data or {}).get("status") == "ok" else None
        if item:
            record = crossref_record(item)
            if title and not title_matches(title, record.get("title", ""), args.min_title_similarity):
                mismatch_notes.append(f"Draft DOI `{doi}` resolved to Crossref title `{record.get('title', '')}`; ignored because it did not match draft title.")
            else:
                record["verification_status"] = "verified_crossref"
                record["verification_confidence"] = "high"
                return attach_identity_note(record, " ".join(mismatch_notes) or "DOI resolved through Crossref.")
    if title:
        params = urllib.parse.urlencode({"query.bibliographic": title, "rows": 1})
        data = http_json(f"https://api.crossref.org/works?{params}", timeout=args.timeout, user_agent=ua)
        item = ((((data or {}).get("message") or {}).get("items") or [])[:1] or [None])[0]
    else:
        item = None
    if not item:
        return None
    record = crossref_record(item)
    if title and not title_matches(title, record.get("title", ""), args.min_title_similarity):
        return None
    record["verification_status"] = "verified_crossref"
    record["verification_confidence"] = "medium"
    note = "Title/bibliographic Crossref search."
    return attach_identity_note(record, " ".join(mismatch_notes + [note]))


def openalex_record(item: dict[str, Any]) -> dict[str, str]:
    title = clean_space(item.get("title") or item.get("display_name"))
    authors = []
    for authorship in item.get("authorships") or []:
        name = clean_space(((authorship.get("author") or {}).get("display_name")))
        if name:
            authors.append(name)
    source = ((item.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
    doi = normalize_doi(item.get("doi"))
    return {
        "title": title,
        "authors": "; ".join(authors[:12]),
        "year": clean_space(item.get("publication_year")),
        "source": "openalex",
        "paper_id": clean_space(item.get("id")),
        "doi": doi,
        "url": clean_space(item.get("doi") or item.get("id")),
        "journal": clean_space(source),
        "abstract": "",
        "publication_type": clean_space(item.get("type")),
    }


def verify_openalex(row: dict[str, Any], args: argparse.Namespace) -> dict[str, str] | None:
    doi = candidate_doi(row)
    title = clean_space(row.get("candidate_title"))
    params = {"per-page": "1"}
    if args.email:
        params["mailto"] = args.email
    mismatch_notes: list[str] = []
    if doi:
        url = "https://api.openalex.org/works/" + urllib.parse.quote("https://doi.org/" + doi, safe="")
        data = http_json(url, timeout=args.timeout, user_agent=args.user_agent)
        item = data if data and data.get("id") else None
        if item:
            record = openalex_record(item)
            if title and not title_matches(title, record.get("title", ""), args.min_title_similarity):
                mismatch_notes.append(f"Draft DOI `{doi}` resolved to OpenAlex title `{record.get('title', '')}`; ignored because it did not match draft title.")
            else:
                record["verification_status"] = "verified_openalex"
                record["verification_confidence"] = "high"
                return attach_identity_note(record, " ".join(mismatch_notes) or "DOI resolved through OpenAlex.")
    if title:
        params["search"] = title
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    else:
        return None
    data = http_json(url, timeout=args.timeout, user_agent=args.user_agent)
    item = (((data or {}).get("results") or [])[:1] or [None])[0]
    if not item:
        return None
    record = openalex_record(item)
    if title and not title_matches(title, record.get("title", ""), args.min_title_similarity):
        return None
    record["verification_status"] = "verified_openalex"
    record["verification_confidence"] = "medium"
    return attach_identity_note(record, " ".join(mismatch_notes + ["Title/bibliographic OpenAlex search."]))


def doi_metadata_record(doi: str, args: argparse.Namespace) -> dict[str, str] | None:
    """Resolve DOI metadata without promoting it; used to guard publisher URL fallbacks."""
    doi = normalize_doi(doi)
    if not doi:
        return None
    ua = f"{args.user_agent} (mailto:{args.email})" if args.email else args.user_agent
    data = http_json(
        "https://api.crossref.org/works/" + urllib.parse.quote(doi),
        timeout=args.timeout,
        user_agent=ua,
    )
    item = (data or {}).get("message") if (data or {}).get("status") == "ok" else None
    if item:
        record = crossref_record(item)
        record["source"] = "crossref_doi_guard"
        return record
    params = {}
    if args.email:
        params["mailto"] = args.email
    url = "https://api.openalex.org/works/" + urllib.parse.quote("https://doi.org/" + doi, safe="")
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = http_json(url, timeout=args.timeout, user_agent=args.user_agent)
    if data and data.get("id"):
        record = openalex_record(data)
        record["source"] = "openalex_doi_guard"
        return record
    return None


def verify_official_url(row: dict[str, Any], args: argparse.Namespace) -> dict[str, str] | None:
    """Verify official publisher/conference URLs when metadata APIs miss them."""
    url = first_url(row)
    if not url or not re.match(r"https?://", url, re.I):
        return None
    hp = host_path(url)
    raw_context = " ".join(clean_space(row.get(field)) for field in ["raw_reference", "candidate_title", "source_kind", "urls"])
    is_conference_url = any(marker in hp for marker in OFFICIAL_CONFERENCE_SOURCES)
    is_publisher_url = any(marker in hp for marker in PUBLISHER_HOST_MARKERS)
    if not is_conference_url and not is_publisher_url:
        return None
    try:
        metadata = html_metadata(http_text(url, timeout=args.timeout, user_agent=args.user_agent))
    except Exception:
        metadata = {}
    title = metadata.get("title", "")
    if is_web_challenge_title(title):
        title = ""
    candidate_title = clean_space(row.get("candidate_title"))
    score = title_similarity(candidate_title, title) if title and candidate_title else 0.0
    doi = normalize_doi(metadata.get("doi")) or candidate_doi(row)
    doi_record = doi_metadata_record(doi, args) if doi else None
    if doi_record and candidate_title and not title_matches(candidate_title, doi_record.get("title", ""), args.min_title_similarity):
        return None
    if doi_record and (not title or (candidate_title and score < args.min_title_similarity)):
        title = doi_record.get("title", "") or title
        score = title_similarity(candidate_title, title) if title and candidate_title else score
        metadata = {
            **metadata,
            "authors": metadata.get("authors") or doi_record.get("authors", ""),
            "year": metadata.get("year") or doi_record.get("year", ""),
            "journal": metadata.get("journal") or doi_record.get("journal", ""),
            "abstract": metadata.get("abstract") or doi_record.get("abstract", ""),
        }
    venue = metadata.get("journal") or venue_from_text(f"{raw_context} {title} {url}")
    if is_conference_url:
        has_venue = bool(venue_from_text(f"{raw_context} {title} {url}"))
        if title and candidate_title and score < args.min_title_similarity and not has_venue:
            return None
        if not (title or has_venue):
            return None
        source = "openreview" if "openreview" in hp else "official_conference_page"
        return {
            "title": title or candidate_title,
            "authors": metadata.get("authors", ""),
            "year": metadata.get("year", ""),
            "source": source,
            "source_kind_verified": "official_conference_paper",
            "paper_id": url,
            "pmid": "",
            "doi": doi,
            "url": url,
            "journal": venue or "Official conference proceedings",
            "abstract": metadata.get("abstract", ""),
            "publication_type": "conference-paper",
            "title_similarity": f"{score:.3f}" if title and candidate_title else "",
            "verification_status": "verified_openreview" if source == "openreview" else "verified_conference_page",
            "verification_confidence": "high" if score >= args.min_title_similarity or doi else "medium",
        }
    if is_publisher_url:
        metadata_title_mismatch = bool(title and candidate_title and score < args.min_title_similarity)
        if metadata_title_mismatch:
            if not doi:
                return None
            title = ""
            score = 0.0
        if not (title or doi):
            return None
        note = ""
        if doi and not title and candidate_title:
            note = "DOI inferred from official publisher URL; page metadata title was unavailable or unusable, so keep candidate title pending official citation export."
        return {
            "title": title or candidate_title,
            "authors": metadata.get("authors", ""),
            "year": metadata.get("year", ""),
            "source": "publisher_url",
            "source_kind_verified": "publisher_article_page",
            "paper_id": url,
            "pmid": "",
            "doi": doi,
            "url": url,
            "journal": metadata.get("journal", ""),
            "abstract": metadata.get("abstract", ""),
            "publication_type": "journal-article" if doi or metadata.get("journal") else "article",
            "title_similarity": f"{score:.3f}" if title and candidate_title else "",
            "verification_status": "verified_publisher_url",
            "verification_confidence": "high" if doi else "medium",
            "identity_resolution_notes": note,
        }
    return None


def verify_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, str]:
    base = {
        "draft": clean_space(row.get("draft")),
        "ref_number": clean_space(row.get("ref_number")),
        "draft_candidate_title": clean_space(row.get("candidate_title")),
        "raw_reference": clean_space(row.get("raw_reference")),
        "draft_doi": candidate_doi(row),
        "draft_pmid": first_list_value(row.get("pmid")),
        "draft_urls": clean_space(row.get("urls")),
        "draft_source_kind": clean_space(row.get("source_kind")),
        "cited_in_body": clean_space(row.get("cited_in_body")),
        "candidate_id": clean_space(row.get("candidate_id") or row.get("ref_number")),
        "claim_id": clean_space(row.get("claim_id")),
        "candidate_type": clean_space(row.get("candidate_type")),
        "pubmed_query": clean_space(row.get("pubmed_query")),
        "raw_evidence_excerpt": clean_space(row.get("raw_evidence_excerpt")),
        "why_relevant": clean_space(row.get("why_relevant")),
    }
    if is_weak_title_clue(row) and not base["draft_doi"] and not base["draft_pmid"]:
        return rejected_result(base, row, "Weak or non-paper reference clue. Do not title-search this into a paper candidate; delete, replace, or ask the user.")
    raw_preprint_lead = is_preprint_lead(row)
    if args.offline:
        return {
            **base,
            "verification_status": "needs_api_verification",
            "verification_confidence": "none",
            "title": base["draft_candidate_title"],
            "authors": "",
            "year": "",
            "source": "",
            "paper_id": "",
            "doi": base["draft_doi"],
            "url": first_list_value(row.get("urls")),
            "journal": "",
            "abstract": "",
            "publication_type": "",
            "source_kind_verified": "",
            "title_similarity": "",
            "pool_status": "maybe",
            "notes": "Offline extraction only; verify through PubMed/Crossref/OpenAlex before citing.",
        }
    for verifier in [verify_pubmed, verify_crossref, verify_openalex, verify_official_url]:
        try:
            found = verifier(row, args)
        except Exception as exc:
            found = None
            base["notes"] = f"{verifier.__name__} error: {exc}"
        if found:
            if is_correction_record(found) and not is_correction_like_title(base["draft_candidate_title"]):
                base["notes"] = clean_space(
                    f"{base.get('notes', '')} {verifier.__name__} matched a correction/erratum record, not the target paper; continuing search."
                )
                continue
            if is_preprint_record(found):
                result = preprint_result(base, row, found)
                result["_search_trace"] = found.get("_search_trace") or row.get("_search_trace", [])
                return result
            if not is_accepted_publication_type(found):
                result = rejected_result(
                    base,
                    row,
                    f"Metadata matched non-paper publication type `{found.get('publication_type', '')}`. Keep outside final paper references unless user overrides.",
                )
                result["_search_trace"] = found.get("_search_trace") or row.get("_search_trace", [])
                return result
            return {
                **base,
                **found,
                "source_kind_verified": found.get("source_kind_verified") or found.get("source") or "",
                "title_similarity": found.get("title_similarity")
                or (f"{title_similarity(base['draft_candidate_title'], found.get('title', '')):.3f}" if base["draft_candidate_title"] and found.get("title") else ""),
                "pool_status": "candidate",
                "recall_queries": base["draft_candidate_title"],
                "claim_supported": "",
                "limitations": "",
                "use_in_review": "",
                "notes": clean_space(
                    f"Verified from draft-native citation asset using {found['verification_status']}. {found.get('identity_resolution_notes', '')}"
                ),
                "_search_trace": found.get("_search_trace") or row.get("_search_trace", []),
            }
        if args.sleep:
            time.sleep(args.sleep)
    if raw_preprint_lead:
        result = preprint_result(base, row)
        result["_search_trace"] = row.get("_search_trace", [])
        return result
    result = rejected_result(
        base,
        row,
        "No PubMed/Crossref/OpenAlex verification found. Check arXiv/OpenReview if the project allows lead discovery; otherwise delete from citation candidates or ask user to supply evidence.",
    )
    result["_search_trace"] = row.get("_search_trace", [])
    return result


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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.2 * (attempt + 1))
    if last_error:
        raise last_error


def confidence_rank(value: Any) -> int:
    return {"high": 3, "medium": 2, "low": 1, "lead_only": 1, "none": 0}.get(clean_space(value).lower(), 0)


def result_rank(row: dict[str, Any]) -> tuple[int, int, float, int]:
    status = clean_space(row.get("verification_status"))
    similarity = 0.0
    try:
        similarity = float(row.get("title_similarity") or 0)
    except (TypeError, ValueError):
        similarity = 0.0
    return (
        STATUS_RANK.get(status, 0),
        confidence_rank(row.get("verification_confidence")),
        similarity,
        len(clean_space(row.get("abstract"))),
    )


def dedupe_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        key = row_key(row)
        if not key:
            key = json.dumps(row, ensure_ascii=False, sort_keys=True)
        if key not in best:
            best[key] = row
            order.append(key)
        elif result_rank(row) > result_rank(best[key]):
            best[key] = row
    return [best[key] for key in order]


def canonical_version_title(row: dict[str, Any]) -> str:
    title = clean_space(row.get("title") or row.get("draft_candidate_title"))
    title = re.sub(r"^(publisher correction|correction|erratum|corrigendum|retraction)\s*:\s*", "", title, flags=re.IGNORECASE)
    return normalize_title(title)


def mark_duplicate_preprint_versions(rows: list[dict[str, Any]], min_similarity: float = 0.82) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    verified = [row for row in rows if row.get("verification_status") in VERIFIED_STATUSES and not is_correction_record(row)]
    suppressed: list[dict[str, Any]] = []
    for row in rows:
        if row.get("verification_status") != "preprint_lead":
            continue
        row_title = canonical_version_title(row)
        if not row_title:
            continue
        for formal in verified:
            formal_title = canonical_version_title(formal)
            if not formal_title:
                continue
            same_title = row_title == formal_title
            similar = title_similarity(row_title, formal_title) >= min_similarity
            if same_title or similar:
                row["verification_status"] = VERSION_DUPLICATE_STATUS
                row["verification_confidence"] = "duplicate"
                row["pool_status"] = "duplicate_formal_version_available"
                row["notes"] = clean_space(
                    f"{row.get('notes', '')} Formal version available: {formal.get('title', '')} DOI {formal.get('doi', '')} PMID {formal.get('pmid', '')}. Do not cite this preprint separately."
                )
                suppressed.append(row)
                break
    return rows, suppressed


def write_deduplication_report(path: Path, raw_rows: list[dict[str, Any]], canonical_rows: list[dict[str, Any]]) -> None:
    if len(raw_rows) == len(canonical_rows):
        if path.exists():
            path.unlink()
        return
    groups: dict[str, int] = {}
    for row in raw_rows:
        key = row_key(row)
        groups[key] = groups.get(key, 0) + 1
    duplicate_groups = [(key, count) for key, count in groups.items() if count > 1]
    lines = [
        "# Draft Reference Verification Deduplication",
        "",
        f"- Raw rows: {len(raw_rows)}",
        f"- Canonical rows: {len(canonical_rows)}",
        f"- Duplicate groups: {len(duplicate_groups)}",
        "",
        "Resume writes canonical CSVs, so reruns should not grow duplicates when API verification is interrupted.",
        "",
    ]
    for key, count in duplicate_groups[:50]:
        lines.append(f"- `{key}`: {count} rows")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def search_trace_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for row in rows:
        items = row.get("_search_trace")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            merged = dict(item)
            for field, source in [
                ("draft", "draft"),
                ("ref_number", "ref_number"),
                ("candidate_id", "candidate_id"),
                ("draft_candidate_title", "draft_candidate_title"),
            ]:
                if not clean_space(merged.get(field)):
                    merged[field] = clean_space(row.get(source))
            traces.append(merged)
    return traces


def dedupe_trace_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = "|".join(clean_space(row.get(field)) for field in SEARCH_TRACE_FIELDS)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def write_search_trace_outputs(out_dir: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trace_jsonl = out_dir / "verification_query_trace.jsonl"
    trace_rows = read_jsonl(trace_jsonl)
    trace_rows.extend(search_trace_from_rows(rows))
    trace_rows = dedupe_trace_rows(trace_rows)
    write_csv(out_dir / "verification_query_trace.csv", trace_rows, SEARCH_TRACE_FIELDS)
    return trace_rows


def split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    verified_rows = [row for row in rows if row.get("verification_status") in VERIFIED_STATUSES]
    preprints = [row for row in rows if row.get("verification_status") == "preprint_lead"]
    rejected = [row for row in rows if row.get("verification_status") in {"unverified_delete_or_replace", "needs_api_verification"}]
    return verified_rows, preprints, rejected


def write_checkpoint_outputs(out_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    raw_rows = [dict(row) for row in rows]
    rows = dedupe_results([dict(row) for row in rows])
    rows, version_duplicates = mark_duplicate_preprint_versions(rows)
    verified_rows, preprints, rejected = split_rows(rows)
    write_csv(out_dir / "all_draft_reference_verification.csv", rows, OUTPUT_FIELDS)
    write_csv(out_dir / "verified_draft_papers.csv", verified_rows, OUTPUT_FIELDS)
    write_csv(out_dir / "preprint_leads.csv", preprints, OUTPUT_FIELDS)
    write_csv(out_dir / "version_suppressed_duplicate_preprints.csv", version_duplicates, OUTPUT_FIELDS)
    write_csv(out_dir / "rejected_or_unverified_draft_sources.csv", rejected, OUTPUT_FIELDS)
    trace_rows = write_search_trace_outputs(out_dir, rows)
    write_report(out_dir / "draft_reference_verification_report.md", rows, args)
    write_deduplication_report(out_dir / "verification_deduplication_report.md", raw_rows, rows)
    summary = {
        "checked": len(rows),
        "raw_rows": len(raw_rows),
        "verified": len(verified_rows),
        "preprint_leads": len(preprints),
        "version_suppressed_duplicates": len(version_duplicates),
        "rejected_or_unverified": len(rejected),
        "query_trace_rows": len(trace_rows),
        "out_dir": str(out_dir),
        "workers": args.workers,
        "incremental": args.incremental,
        "resumed": args.resume,
    }
    (out_dir / "verification_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def write_report(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        status = clean_space(row.get("verification_status"))
        counts[status] = counts.get(status, 0) + 1
    lines = [
        "# Draft Reference Verification Report",
        "",
        f"- Generated at: {now_iso()}",
        f"- Input: `{args.candidate_csv}`",
        f"- Offline: {args.offline}",
        f"- Records checked: {len(rows)}",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"- {status}: {count}")
    lines.extend(
        [
            "",
            "## Gate Policy",
            "",
            "- Import only API-verified rows or `official_conference_paper`/publisher-page verified rows into the governed pool.",
            "- Keep arXiv/OpenReview-only `preprint_lead` rows out of final references unless the user explicitly changes the published-only policy.",
            "- Official OpenReview/NeurIPS/ICML/ICLR proceedings pages may be verified as conference papers when title/venue evidence matches.",
            "- Treat `unverified_delete_or_replace` rows as deletion/replacement tasks.",
            "- After import, still run topic filtering, final-gate checks, LLM/human relevance screening, and literature-card creation.",
            "- Inspect `verification_query_trace.csv` when PubMed appears to miss an obvious paper; it records query modes, returned candidates, title mismatches, and accepted ranked hits.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_verify(args: argparse.Namespace) -> int:
    candidate_csv = Path(args.candidate_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(candidate_csv)
    if args.max_records > 0:
        rows = rows[: args.max_records]

    all_rows: list[dict[str, Any]] = []
    completed_keys: set[str] = set()
    all_csv = out_dir / "all_draft_reference_verification.csv"
    if args.resume and all_csv.exists():
        all_rows = dedupe_results(read_csv(all_csv))
        completed_keys = {row_key(row) for row in all_rows}

    pending = [row for row in rows if row_key(row) not in completed_keys]
    progress_path = out_dir / "verification_progress.jsonl"
    trace_path = out_dir / "verification_query_trace.jsonl"
    if args.incremental and not args.resume:
        for path in [progress_path, trace_path]:
            if path.exists():
                path.unlink()
    total_unique = len({row_key(row) for row in rows})

    def handle_result(result: dict[str, Any]) -> None:
        result["verified_at"] = now_iso()
        all_rows.append(result)
        if args.incremental:
            append_jsonl(progress_path, result)
            for trace in result.get("_search_trace", []) if isinstance(result.get("_search_trace"), list) else []:
                append_jsonl(trace_path, trace)
        checked = len({row_key(row) for row in all_rows})
        total = total_unique
        status = clean_space(result.get("verification_status"))
        title = clean_space(result.get("draft_candidate_title") or result.get("title"))[:90]
        print(json.dumps({"checked": checked, "total": total, "status": status, "title": title}, ensure_ascii=False), flush=True)
        if args.incremental and (checked % max(args.flush_every, 1) == 0):
            write_checkpoint_outputs(out_dir, all_rows, args)

    if pending:
        if args.workers <= 1 or args.offline:
            for row in pending:
                handle_result(verify_row(row, args))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_to_index = {executor.submit(verify_row, row, args): index for index, row in enumerate(pending)}
                ordered_results: dict[int, dict[str, Any]] = {}
                next_index = 0
                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        ordered_results[index] = future.result()
                    except Exception as exc:
                        ordered_results[index] = rejected_result(
                            {
                                "draft": clean_space(pending[index].get("draft")),
                                "ref_number": clean_space(pending[index].get("ref_number")),
                                "draft_candidate_title": clean_space(pending[index].get("candidate_title")),
                                "raw_reference": clean_space(pending[index].get("raw_reference")),
                                "draft_doi": normalize_doi(pending[index].get("doi")),
                                "draft_pmid": clean_space(pending[index].get("pmid")),
                                "draft_urls": clean_space(pending[index].get("urls")),
                                "draft_source_kind": clean_space(pending[index].get("source_kind")),
                                "cited_in_body": clean_space(pending[index].get("cited_in_body")),
                                "candidate_id": clean_space(pending[index].get("candidate_id") or pending[index].get("ref_number")),
                                "claim_id": clean_space(pending[index].get("claim_id")),
                                "candidate_type": clean_space(pending[index].get("candidate_type")),
                                "pubmed_query": clean_space(pending[index].get("pubmed_query")),
                                "raw_evidence_excerpt": clean_space(pending[index].get("raw_evidence_excerpt")),
                                "why_relevant": clean_space(pending[index].get("why_relevant")),
                            },
                            pending[index],
                            f"Verifier worker failed: {exc}",
                        )
                    while next_index in ordered_results:
                        handle_result(ordered_results.pop(next_index))
                        next_index += 1

    summary = write_checkpoint_outputs(out_dir, all_rows, args)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify draft-native paper clues with PubMed-first metadata checks.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_verify = sub.add_parser("verify", help="Verify candidate paper clues and produce importable verified papers.")
    p_verify.add_argument("--candidate-csv", default="./review-data/02_literature/draft_assets/candidate_paper_clues.csv")
    p_verify.add_argument("--out-dir", default="./review-data/05_audit/draft_reference_verification")
    p_verify.add_argument("--email", default="")
    p_verify.add_argument("--api-key", default="")
    p_verify.add_argument("--timeout", type=int, default=20)
    p_verify.add_argument("--sleep", type=float, default=0.1)
    p_verify.add_argument("--pubmed-retmax", type=int, default=10, help="Number of PubMed hits to fetch and rank per query attempt.")
    p_verify.add_argument("--max-records", type=int, default=0)
    p_verify.add_argument("--min-title-similarity", type=float, default=0.58)
    p_verify.add_argument("--user-agent", default=DEFAULT_UA)
    p_verify.add_argument("--workers", type=int, default=1, help="Parallel verifier workers for independent API lookups. Use 1 for conservative PubMed rate limits.")
    p_verify.add_argument("--flush-every", type=int, default=5, help="When incremental output is enabled, rewrite summary CSVs after this many completed records.")
    p_verify.add_argument("--resume", action="store_true", help="Skip rows already present in all_draft_reference_verification.csv.")
    p_verify.add_argument("--no-incremental", dest="incremental", action="store_false", help="Disable per-record progress JSONL and checkpoint CSV writes.")
    p_verify.add_argument("--offline", action="store_true")
    p_verify.set_defaults(incremental=True)
    p_verify.set_defaults(func=cmd_verify)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
