#!/usr/bin/env python3
"""Codex-subtask or optional external adjudication for citation candidates.

This script runs after deterministic metadata verification. APIs establish
candidate facts; Codex subtasks adjudicate title fit, publication-status risk,
claim-source fit, and human decisions by default. Optional external LLM calls
must not invent identifiers or promote unverified papers.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import json
import os
import re
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_ENV_FILES = (".env.local", ".env")
PREPRINT_DOI_MARKERS = ("10.1101/", "10.48550/arxiv", "10.21203/", "10.2139/ssrn")


FIELDS = [
    "ref_key",
    "draft",
    "ref_number",
    "deterministic_status",
    "llm_decision",
    "confidence",
    "title_match",
    "publication_status",
    "claim_fit",
    "keep_as",
    "supported_claim_ids",
    "rationale",
    "human_question",
    "required_next_action",
]
ACCEPTED_DECISIONS = {"accept_verified"}
VERIFIED_STATUSES = {
    "verified_pubmed",
    "verified_crossref",
    "verified_openalex",
    "verified_openreview",
    "verified_conference_page",
    "verified_publisher_url",
}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def clean_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_doi(value: Any) -> str:
    text = clean_space(value).lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi\s*:\s*", "", text)
    return text.rstrip(".,;)")


def ref_key(row: dict[str, Any]) -> str:
    draft = re.sub(r"[^A-Za-z0-9_.-]+", "-", clean_space(row.get("draft")) or "draft").strip("-")
    ref_number = clean_space(row.get("ref_number")) or "no-ref"
    return f"{draft}::ref-{ref_number}"


def is_preprint_doi(doi: str) -> bool:
    value = normalize_doi(doi)
    return any(marker in value for marker in PREPRINT_DOI_MARKERS)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def claim_index(claim_rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    index: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in claim_rows:
        draft = clean_space(row.get("draft"))
        for number in re.split(r";\s*|,\s*", clean_space(row.get("mapped_ref_numbers"))):
            if not number:
                continue
            index.setdefault((draft, number), []).append(row)
    return index


def claim_id_index(claim_rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    index: dict[str, list[dict[str, str]]] = {}
    for row in claim_rows:
        claim_id = clean_space(row.get("claim_id"))
        if claim_id:
            index.setdefault(claim_id, []).append(row)
    return index


def compact_claims(claims: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in claims[:limit]:
        out.append(
            {
                "claim_id": clean_space(row.get("claim_id")),
                "claim": clean_space(row.get("claim"))[:420],
                "citation_markers": clean_space(row.get("citation_markers")),
            }
        )
    return out


def build_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    verification_rows = read_csv(Path(args.verification_csv))
    claim_rows = read_csv(Path(args.claim_map))
    claims = claim_index(claim_rows)
    claims_by_id = claim_id_index(claim_rows)
    cases: list[dict[str, Any]] = []
    for row in verification_rows:
        status = clean_space(row.get("verification_status"))
        if args.status and status not in set(args.status.split(",")):
            continue
        key = ref_key(row)
        linked = claims_by_id.get(clean_space(row.get("claim_id")), [])
        if not linked:
            linked = claims.get((clean_space(row.get("draft")), clean_space(row.get("ref_number"))), [])
        linked_claims = compact_claims(linked, args.max_claims_per_ref)
        if not linked_claims and (clean_space(row.get("raw_evidence_excerpt")) or clean_space(row.get("why_relevant"))):
            linked_claims = [
                {
                    "claim_id": clean_space(row.get("claim_id") or row.get("candidate_id") or row.get("ref_number")),
                    "claim": clean_space(row.get("raw_evidence_excerpt") or row.get("why_relevant"))[:420],
                    "citation_markers": clean_space(row.get("candidate_type") or row.get("draft_source_kind")),
                }
            ]
        doi = normalize_doi(row.get("doi"))
        cases.append(
            {
                "ref_key": key,
                "draft": clean_space(row.get("draft")),
                "ref_number": clean_space(row.get("ref_number")),
                "candidate_id": clean_space(row.get("candidate_id")),
                "claim_id": clean_space(row.get("claim_id")),
                "candidate_type": clean_space(row.get("candidate_type")),
                "deterministic_status": status,
                "draft_candidate_title": clean_space(row.get("draft_candidate_title")),
                "raw_reference": clean_space(row.get("raw_reference"))[:700],
                "raw_evidence_excerpt": clean_space(row.get("raw_evidence_excerpt"))[:700],
                "why_relevant": clean_space(row.get("why_relevant"))[:500],
                "pubmed_query": clean_space(row.get("pubmed_query")),
                "verified_title": clean_space(row.get("title")),
                "authors": clean_space(row.get("authors")),
                "year": clean_space(row.get("year")),
                "journal": clean_space(row.get("journal")),
                "source": clean_space(row.get("source")),
                "pmid": clean_space(row.get("pmid")),
                "doi": doi,
                "url": clean_space(row.get("url")),
                "publication_type": clean_space(row.get("publication_type")),
                "abstract": clean_space(row.get("abstract"))[:900],
                "preprint_doi_marker": "yes" if is_preprint_doi(doi) else "no",
                "linked_claims": linked_claims,
            }
        )
    if args.max_records > 0:
        cases = cases[: args.max_records]
    return cases


def chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def adjudication_prompt(cases: list[dict[str, Any]], topic: str) -> str:
    schema = {
        "ref_key": "string from input",
        "llm_decision": "accept_verified | demote_preprint | reject_mismatch | reject_nonpaper | needs_manual | needs_published_version_search",
        "confidence": "high | medium | low",
        "title_match": "exact | strong | partial | weak | mismatch | not_applicable",
        "publication_status": "published_paper | preprint_only | nonpaper_or_guideline | unverifiable | unclear",
        "claim_fit": "direct | indirect | background | none | unclear",
        "keep_as": "citation_pool_candidate | background_only | supplemental_search_lead | delete_or_replace | human_decision",
        "supported_claim_ids": ["claim ids from linked_claims only"],
        "rationale": "one concise reason using only supplied metadata",
        "human_question": "blank if not needed",
        "required_next_action": "none | import_to_pool | keep_out_of_final_refs | search_published_version | ask_user | delete_or_replace",
    }
    return textwrap.dedent(
        f"""
        You are the Citation Adjudication Agent for a top-journal review.

        Review topic:
        {topic or "[not specified]"}

        Your job:
        - APIs already produced deterministic metadata candidates.
        - You adjudicate whether each candidate is safe to import, demote, reject, or ask a human about.
        - You must NOT invent DOI, PMID, authors, title, venue, results, or publication status.
        - Use only the provided metadata and linked claims.
        - Preprint DOI markers such as 10.1101 or 10.48550/arXiv are NOT final published papers unless a separate published DOI/PMID or official conference-paper/proceedings page is provided.
        - If title match is weak or the source is a book chapter, official webpage, guideline, policy page, product page, repository, or generic journal fragment, do not mark it as a published paper citation candidate.
        - A paper may be a real paper but still unsuitable for the linked claim; mark claim_fit accordingly.
        - If a preprint likely has a published version but the provided candidate is only preprint, choose needs_published_version_search.

        Return strict JSON only as one object with a `results` array. Each array item must use this schema:
        {json.dumps(schema, ensure_ascii=False, indent=2)}

        Cases:
        {json.dumps(cases, ensure_ascii=False, indent=2)}
        """
    ).strip()


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists() or path.is_dir():
        return values
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def is_placeholder_secret(value: str) -> bool:
    if not value:
        return True
    lowered = value.lower()
    return any(token in lowered for token in ["your_", "placeholder", "example", "replace", "<", ">"])


def project_env(project_dir: Path, env_file: str = "") -> tuple[dict[str, str], list[str]]:
    paths = [Path(env_file)] if env_file else [project_dir / name for name in DEFAULT_ENV_FILES]
    merged: dict[str, str] = {}
    loaded: list[str] = []
    for path in paths:
        if not path.is_absolute():
            path = project_dir / path
        if path.exists():
            merged.update(parse_env_file(path))
            loaded.append(str(path))
    return merged, loaded


def api_config(args: argparse.Namespace) -> dict[str, Any]:
    project_dir = Path(args.project_dir).resolve()
    env_values, loaded = project_env(project_dir, args.env_file)
    api_key = (
        args.api_key
        or os.environ.get("REVIEW_AGENT_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or env_values.get("REVIEW_AGENT_API_KEY")
        or env_values.get("DEEPSEEK_API_KEY")
        or ""
    )
    base_url = (
        args.base_url
        or os.environ.get("REVIEW_AGENT_BASE_URL")
        or os.environ.get("DEEPSEEK_BASE_URL")
        or env_values.get("REVIEW_AGENT_BASE_URL")
        or env_values.get("DEEPSEEK_BASE_URL")
        or DEFAULT_BASE_URL
    )
    model = (
        args.model
        or os.environ.get("REVIEW_AGENT_MODEL")
        or os.environ.get("DEEPSEEK_MODEL")
        or env_values.get("REVIEW_AGENT_MODEL")
        or env_values.get("DEEPSEEK_MODEL")
        or DEFAULT_MODEL
    )
    if is_placeholder_secret(api_key):
        api_key = ""
    return {"api_key": api_key, "base_url": base_url.rstrip("/"), "model": model, "loaded_env_files": loaded}


def chat_completion_once(prompt: str, args: argparse.Namespace, user_id: str = "") -> str:
    cfg = api_config(args)
    if not cfg["api_key"]:
        raise SystemExit("No API key. Set DEEPSEEK_API_KEY/REVIEW_AGENT_API_KEY or run `packet` mode.")
    payload: dict[str, Any] = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": "You are a rigorous citation adjudication agent. Return strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": args.temperature,
        "response_format": {"type": "json_object"} if args.json_object else {"type": "json_object"},
    }
    if user_id:
        payload["user_id"] = user_id
    if args.thinking != "omit":
        payload["thinking"] = {"type": args.thinking}
    if args.reasoning_effort:
        payload["reasoning_effort"] = args.reasoning_effort
    request = urllib.request.Request(
        cfg["base_url"] + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        body = json.loads(response.read().decode("utf-8", errors="replace"))
    return body["choices"][0]["message"]["content"]


def chat_completion(prompt: str, args: argparse.Namespace, user_id: str = "") -> str:
    last_error: Exception | None = None
    for attempt in range(max(1, args.max_retries + 1)):
        try:
            return chat_completion_once(prompt, args, user_id=user_id)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt >= args.max_retries:
                raise
            time.sleep(args.retry_backoff * (2**attempt))
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt >= args.max_retries:
                raise
            time.sleep(args.retry_backoff * (2**attempt))
    raise RuntimeError(f"LLM call failed after retries: {last_error}")


def call_prompt_batch(item: tuple[int, str], args: argparse.Namespace) -> tuple[int, str]:
    idx, prompt = item
    user_id = ""
    if args.user_id_prefix:
        safe_prefix = re.sub(r"[^A-Za-z0-9_-]+", "-", args.user_id_prefix).strip("-")[:480]
        user_id = f"{safe_prefix}-{idx:03d}"
    return idx, chat_completion(prompt, args, user_id=user_id)


def coerce_results(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\[.*\]|\{.*\})", text, flags=re.S)
        if not match:
            raise
        data = json.loads(match.group(1))
    if isinstance(data, dict):
        for key in ["results", "adjudications", "items"]:
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    if isinstance(data, list):
        return data
    return []


def write_packet_outputs(out_dir: Path, cases: list[dict[str, Any]], prompts: list[str]) -> None:
    packet_dir = out_dir / "packets"
    packet_dir.mkdir(parents=True, exist_ok=True)
    for idx, prompt in enumerate(prompts, 1):
        (packet_dir / f"citation_adjudication_packet_{idx:03d}.md").write_text(prompt + "\n", encoding="utf-8")
    with (out_dir / "citation_adjudication_cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    lines = [
        "# Citation Adjudication Packet",
        "",
        f"- Generated at: {now_iso()}",
        f"- Cases: {len(cases)}",
        f"- Packet files: {len(prompts)}",
        "",
        "Use these packets with Codex subtasks by default. Optional external LLM review is allowed only after approval. The adjudicator must return strict JSON only and must not invent identifiers.",
        "",
    ]
    for idx in range(1, len(prompts) + 1):
        lines.append(f"- `packets/citation_adjudication_packet_{idx:03d}.md`")
    (out_dir / "citation_adjudication_packet.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_decision_for_case(case: dict[str, Any]) -> dict[str, str]:
    status = clean_space(case.get("deterministic_status"))
    if status.startswith("verified_") and case.get("preprint_doi_marker") == "no":
        decision = "needs_manual"
        keep_as = "human_decision"
        action = "ask_user"
    elif status == "preprint_lead" or case.get("preprint_doi_marker") == "yes":
        decision = "demote_preprint"
        keep_as = "supplemental_search_lead"
        action = "search_published_version"
    else:
        decision = "reject_mismatch"
        keep_as = "delete_or_replace"
        action = "delete_or_replace"
    return {
        "ref_key": clean_space(case.get("ref_key")),
        "draft": clean_space(case.get("draft")),
        "ref_number": clean_space(case.get("ref_number")),
        "deterministic_status": status,
        "llm_decision": decision,
        "confidence": "low",
        "title_match": "not_applicable",
        "publication_status": "unclear",
        "claim_fit": "unclear",
        "keep_as": keep_as,
        "supported_claim_ids": "",
        "rationale": "Packet-only placeholder; requires Codex/DeepSeek adjudication before import.",
        "human_question": "Should this source be accepted, demoted, or replaced after LLM/subagent review?",
        "required_next_action": action,
    }


def write_adjudication_outputs(out_dir: Path, cases: list[dict[str, Any]], results: list[dict[str, Any]], raw_outputs: list[str]) -> None:
    case_by_key = {case["ref_key"]: case for case in cases}
    rows: list[dict[str, str]] = []
    for item in results:
        key = clean_space(item.get("ref_key"))
        case = case_by_key.get(key, {})
        rows.append(
            {
                "ref_key": key,
                "draft": clean_space(case.get("draft")),
                "ref_number": clean_space(case.get("ref_number")),
                "deterministic_status": clean_space(case.get("deterministic_status")),
                "llm_decision": clean_space(item.get("llm_decision")),
                "confidence": clean_space(item.get("confidence")),
                "title_match": clean_space(item.get("title_match")),
                "publication_status": clean_space(item.get("publication_status")),
                "claim_fit": clean_space(item.get("claim_fit")),
                "keep_as": clean_space(item.get("keep_as")),
                "supported_claim_ids": "; ".join(item.get("supported_claim_ids") or []) if isinstance(item.get("supported_claim_ids"), list) else clean_space(item.get("supported_claim_ids")),
                "rationale": clean_space(item.get("rationale")),
                "human_question": clean_space(item.get("human_question")),
                "required_next_action": clean_space(item.get("required_next_action")),
            }
        )
    seen = {row["ref_key"] for row in rows}
    for case in cases:
        if case["ref_key"] not in seen:
            rows.append(default_decision_for_case(case))
    write_csv(out_dir / "llm_citation_adjudication.csv", rows, FIELDS)
    with (out_dir / "raw_llm_outputs.jsonl").open("w", encoding="utf-8") as handle:
        for raw in raw_outputs:
            handle.write(json.dumps({"raw": raw}, ensure_ascii=False) + "\n")
    lines = [
        "# LLM Citation Adjudication",
        "",
        f"- Generated at: {now_iso()}",
        f"- Cases: {len(cases)}",
        f"- Decisions: {len(rows)}",
        "",
        "## Decision Counts",
        "",
    ]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["llm_decision"]] = counts.get(row["llm_decision"], 0) + 1
    for key, count in sorted(counts.items()):
        lines.append(f"- {key or '[blank]'}: {count}")
    lines.extend(["", "## Human Decisions Needed", ""])
    needs = [row for row in rows if row["human_question"] or row["required_next_action"] in {"ask_user", "search_published_version"}]
    if not needs:
        lines.append("- [none]")
    for row in needs[:80]:
        lines.append(f"- `{row['ref_key']}`: {row['llm_decision']} / {row['required_next_action']} - {row['human_question'] or row['rationale']}")
    (out_dir / "llm_citation_adjudication.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    verification_rows = read_csv(Path(args.verification_csv))
    adjudication_rows = read_csv(Path(args.adjudication_csv))
    adjudication_by_key = {clean_space(row.get("ref_key")): row for row in adjudication_rows}
    accepted: list[dict[str, Any]] = []
    demoted: list[dict[str, Any]] = []
    claim_fit: list[dict[str, Any]] = []
    preprint_resolution: list[dict[str, Any]] = []
    for row in verification_rows:
        key = ref_key(row)
        adj = adjudication_by_key.get(key, {})
        if not adj:
            adj = {
                "llm_decision": "needs_llm_adjudication",
                "confidence": "low",
                "title_match": "not_applicable",
                "publication_status": "unclear",
                "claim_fit": "unclear",
                "keep_as": "human_decision",
                "supported_claim_ids": "",
                "rationale": "No LLM/subagent adjudication record found for this source.",
                "human_question": "Run LLM/subagent adjudication before importing, citing, or deleting this source.",
                "required_next_action": "run_llm_adjudication",
            }
        status = clean_space(row.get("verification_status"))
        decision = clean_space(adj.get("llm_decision"))
        publication_status = clean_space(adj.get("publication_status"))
        keep_as = clean_space(adj.get("keep_as"))
        can_accept = status in VERIFIED_STATUSES and decision in ACCEPTED_DECISIONS and publication_status == "published_paper"
        merged = dict(row)
        merged.update(
            {
                "key": clean_space(row.get("key")),
                "pool_status": "candidate" if can_accept else clean_space(row.get("pool_status") or "maybe"),
                "decision_rationale": clean_space(adj.get("rationale")),
                "claim_supported": clean_space(adj.get("supported_claim_ids")),
                "limitations": clean_space(row.get("limitations")),
                "use_in_review": "",
                "llm_decision": decision,
                "title_match": clean_space(adj.get("title_match")),
                "publication_status": publication_status,
                "claim_fit": clean_space(adj.get("claim_fit")),
                "keep_as": keep_as,
                "required_next_action": clean_space(adj.get("required_next_action")),
                "human_question": clean_space(adj.get("human_question")),
                "origin": clean_space(row.get("origin") or origin_from_row(row)),
            }
        )
        if can_accept:
            accepted.append(merged)
        else:
            demoted.append(merged)
        claim_fit.append(
            {
                "ref_key": key,
                "draft": clean_space(row.get("draft")),
                "ref_number": clean_space(row.get("ref_number")),
                "title": clean_space(row.get("title") or row.get("draft_candidate_title")),
                "doi": normalize_doi(row.get("doi")),
                "deterministic_status": status,
                "llm_decision": decision,
                "claim_fit": clean_space(adj.get("claim_fit")),
                "supported_claim_ids": clean_space(adj.get("supported_claim_ids")),
                "rationale": clean_space(adj.get("rationale")),
                "human_question": clean_space(adj.get("human_question")),
            }
        )
        if status == "preprint_lead" or normalize_doi(row.get("doi")).startswith(PREPRINT_DOI_MARKERS):
            preprint_resolution.append(
                {
                    "ref_key": key,
                    "draft": clean_space(row.get("draft")),
                    "ref_number": clean_space(row.get("ref_number")),
                    "title": clean_space(row.get("title") or row.get("draft_candidate_title")),
                    "doi": normalize_doi(row.get("doi")),
                    "llm_decision": decision,
                    "required_next_action": clean_space(adj.get("required_next_action")),
                    "human_question": clean_space(adj.get("human_question")),
                    "rationale": clean_space(adj.get("rationale")),
                }
            )
    return accepted, demoted, claim_fit, preprint_resolution


def origin_from_row(row: dict[str, Any]) -> str:
    source_kind = clean_space(row.get("draft_source_kind") or row.get("source_kind") or row.get("candidate_type")).lower()
    if any(token in source_kind for token in ["subagent", "literature_discovery", "ai_mined", "hidden", "reference_recovery"]):
        return "draft-prose-ai-discovered-adjudicated"
    return "draft-native-citation-assets-llm-adjudicated"


def cmd_merge(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    accepted, demoted, claim_fit, preprint_resolution = merge_rows(args)
    verification_rows = read_csv(Path(args.verification_csv))
    accepted_fields = sorted({key for row in accepted for key in row.keys()} | {"title", "authors", "year", "source", "paper_id", "doi", "url", "abstract", "pool_status", "decision_rationale", "claim_supported", "origin"})
    demoted_fields = sorted({key for row in demoted for key in row.keys()} | set(accepted_fields))
    write_csv(out_dir / "accepted_verified_draft_papers.csv", accepted, accepted_fields)
    write_csv(out_dir / "rejected_or_demoted_sources.csv", demoted, demoted_fields)
    write_csv(out_dir / "claim_source_fit.csv", claim_fit, ["ref_key", "draft", "ref_number", "title", "doi", "deterministic_status", "llm_decision", "claim_fit", "supported_claim_ids", "rationale", "human_question"])
    write_csv(out_dir / "preprint_resolution.csv", preprint_resolution, ["ref_key", "draft", "ref_number", "title", "doi", "llm_decision", "required_next_action", "human_question", "rationale"])
    human_rows = [
        row
        for row in demoted
        if clean_space(row.get("human_question")) or clean_space(row.get("required_next_action")) in {"ask_user", "search_published_version", "run_llm_adjudication"}
    ]
    lines = [
        "# Human Confirmation Queue",
        "",
        f"- Generated at: {now_iso()}",
        f"- Verification records: {len(verification_rows)}",
        f"- Accepted verified papers: {len(accepted)}",
        f"- Rejected/demoted sources: {len(demoted)}",
        f"- Human/action queue: {len(human_rows)}",
        "",
    ]
    if not human_rows:
        lines.append("- [none]")
    for row in human_rows[:120]:
        lines.append(
            f"- `{ref_key(row)}` {clean_space(row.get('draft_candidate_title') or row.get('title'))}: "
            f"{clean_space(row.get('llm_decision'))} / {clean_space(row.get('required_next_action'))}. "
            f"{clean_space(row.get('human_question') or row.get('decision_rationale'))}"
        )
    (out_dir / "human_confirmation_queue.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"accepted": len(accepted), "demoted": len(demoted), "claim_fit": len(claim_fit), "preprint_resolution": len(preprint_resolution), "out_dir": str(out_dir)}, ensure_ascii=False))
    return 0


def cmd_packet(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases(args)
    prompts = [adjudication_prompt(batch, args.topic) for batch in chunked(cases, args.batch_size)]
    write_packet_outputs(out_dir, cases, prompts)
    placeholders = [default_decision_for_case(case) for case in cases]
    write_csv(out_dir / "llm_citation_adjudication.todo.csv", placeholders, FIELDS)
    print(json.dumps({"cases": len(cases), "packets": len(prompts), "out_dir": str(out_dir)}, ensure_ascii=False))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases(args)
    prompts = [adjudication_prompt(batch, args.topic) for batch in chunked(cases, args.batch_size)]
    write_packet_outputs(out_dir, cases, prompts)
    all_results: list[dict[str, Any]] = []
    raw_outputs_by_idx: dict[int, str] = {}
    prompt_items = list(enumerate(prompts, 1))
    if args.max_workers > 1 and len(prompt_items) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_map = {executor.submit(call_prompt_batch, item, args): item[0] for item in prompt_items}
            for future in concurrent.futures.as_completed(future_map):
                idx, raw = future.result()
                raw_outputs_by_idx[idx] = raw
    else:
        for item in prompt_items:
            idx, raw = call_prompt_batch(item, args)
            raw_outputs_by_idx[idx] = raw
    raw_outputs: list[str] = []
    for idx in sorted(raw_outputs_by_idx):
        raw = raw_outputs_by_idx[idx]
        raw_outputs.append(raw)
        all_results.extend(coerce_results(raw))
    write_adjudication_outputs(out_dir, cases, all_results, raw_outputs)
    print(json.dumps({"cases": len(cases), "results": len(all_results), "out_dir": str(out_dir)}, ensure_ascii=False))
    return 0


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verification-csv", default="./review-data/05_audit/draft_reference_verification/all_draft_reference_verification.csv")
    parser.add_argument("--claim-map", default="./review-data/02_literature/draft_assets/claim_evidence_map.csv")
    parser.add_argument("--out-dir", default="./review-data/05_audit/citation_llm_adjudication")
    parser.add_argument("--topic", default="")
    parser.add_argument("--status", default="verified_pubmed,verified_crossref,verified_openalex,verified_openreview,verified_conference_page,verified_publisher_url,preprint_lead,unverified_delete_or_replace")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-claims-per-ref", type=int, default=5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM/subagent citation adjudication after deterministic metadata verification.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_packet = sub.add_parser("packet", aliases=["build-cases"], help="Create LLM/subagent packets and TODO CSV without calling an API.")
    add_common(p_packet)
    p_packet.set_defaults(func=cmd_packet)

    p_run = sub.add_parser("run", aliases=["adjudicate"], help="Optional: call an approved OpenAI-compatible external LLM for citation adjudication.")
    add_common(p_run)
    p_run.add_argument("--project-dir", default=".")
    p_run.add_argument("--env-file", default="")
    p_run.add_argument("--api-key", default="")
    p_run.add_argument("--base-url", default="")
    p_run.add_argument("--model", default="")
    p_run.add_argument("--timeout", type=int, default=180)
    p_run.add_argument("--temperature", type=float, default=0.0)
    p_run.add_argument("--thinking", default="enabled", choices=["enabled", "disabled", "omit"])
    p_run.add_argument("--reasoning-effort", default="high")
    p_run.add_argument("--json-object", action="store_true", default=True)
    p_run.add_argument("--max-workers", type=int, default=1, help="Parallel LLM calls across prompt batches. Use 4-8 by default for stability; increase only if budget and API limits allow.")
    p_run.add_argument("--max-retries", type=int, default=3)
    p_run.add_argument("--retry-backoff", type=float, default=2.0)
    p_run.add_argument("--user-id-prefix", default="topjournal-review", help="Optional provider user_id prefix for scheduling/cache isolation; no private data.")
    p_run.set_defaults(func=cmd_run)

    p_merge = sub.add_parser("merge", help="Merge deterministic verification and LLM adjudication into importable accepted CSV plus human queues.")
    p_merge.add_argument("--verification-csv", default="./review-data/05_audit/draft_reference_verification/all_draft_reference_verification.csv")
    p_merge.add_argument("--adjudication-csv", default="./review-data/05_audit/citation_llm_adjudication/llm_citation_adjudication.csv")
    p_merge.add_argument("--out-dir", default="./review-data/05_audit/citation_llm_adjudication")
    p_merge.set_defaults(func=cmd_merge)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
