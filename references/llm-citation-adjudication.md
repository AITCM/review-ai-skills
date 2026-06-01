# LLM Citation Adjudication

Use this workflow after deterministic draft-reference verification and before importing draft-derived sources into the governed main pool.

## Quick Navigation

- Principle: APIs verify identity; Codex/LLM adjudicates fit within bounded fields.
- Pipeline position: run after deterministic verification and before pool import.
- Packet-only mode: default to Codex subagents with supplied metadata and claims.
- Optional external mode: use DeepSeek/OpenAI-compatible calls only after data-egress approval.
- Merge gate and guardrails: never invent identifiers or upgrade unverified records.

## Principle

APIs decide whether a candidate source exists. Codex subtasks decide by default whether the verified metadata is the right source for the draft claim. External LLMs are optional accelerators only when external data egress is explicitly approved.

Do not ask an LLM to create DOI, PMID, authors, year, title, or publication status. Give it only PubMed/Crossref/OpenAlex/arXiv/OpenReview outputs and ask for bounded adjudication: title match, preprint risk, claim-source fit, inclusion/exclusion rationale, and human questions.

## Position In The Draft-First Pipeline

Run:

```text
draft_citation_assets
  -> draft_reference_verifier
  -> draft_source_adjudicator build-cases
  -> Codex-subtask adjudication rows
  -> draft_source_adjudicator merge
  -> merge accepted lanes with literature_pool.py merge-verified
  -> official citation export from verified_papers_union.csv
  -> import verified_papers_union.csv
  -> topic-filter
  -> final-gate
```

Run a second claim-source adjudication after full text/cards/RAG are available and before final manuscript drafting.

## Packet-Only Or Codex Subagent Mode

Use this by default. It requires no external API key and creates human-inspectable packets for Codex subtasks.

```bash
python $SKILL_DIR/scripts/draft_source_adjudicator.py build-cases \
  --verification-csv ./review-data/05_audit/draft_reference_verification/all_draft_reference_verification.csv \
  --claim-map ./review-data/02_literature/draft_assets/claim_evidence_map.csv \
  --out-dir ./review-data/05_audit/draft_source_adjudication \
  --topic "review topic" \
  --batch-size 12
```

Outputs:

- `citation_adjudication_cases.jsonl`: machine-readable cases.
- `packets/citation_adjudication_packet_*.md`: packets for Codex subtasks or manual review.
- `llm_citation_adjudication.todo.csv`: placeholders showing that adjudication is still needed.

Codex subtasks may review packet files in parallel. Their outputs should be saved as structured adjudication rows, then merged with the deterministic verifier output.

## Optional External Parallel Mode

Use DeepSeek/OpenAI-compatible APIs only after the user has approved external LLM calls, external data egress is allowed, and an API key is configured. The script supports parallel batch calls:

```bash
python $SKILL_DIR/scripts/draft_source_adjudicator.py adjudicate \
  --project-dir . \
  --verification-csv ./review-data/05_audit/draft_reference_verification/all_draft_reference_verification.csv \
  --claim-map ./review-data/02_literature/draft_assets/claim_evidence_map.csv \
  --out-dir ./review-data/05_audit/draft_source_adjudication \
  --topic "review topic" \
  --batch-size 12 \
  --max-workers 4 \
  --max-retries 3 \
  --retry-backoff 2
```

For optional external DeepSeek batch adjudication, start with `--max-workers 4` or `8` for stability. DeepSeek currently documents high account-level concurrency limits for `deepseek-v4-pro` and `deepseek-v4-flash`, but projects should still use retries and backoff for 429 or transient server errors. Use `--user-id-prefix` without private data if you need request isolation.

## Merge Gate

Never import raw LLM decisions directly. Merge them with deterministic verification:

```bash
python $SKILL_DIR/scripts/draft_source_adjudicator.py merge \
  --verification-csv ./review-data/05_audit/draft_reference_verification/all_draft_reference_verification.csv \
  --adjudication-csv ./review-data/05_audit/draft_source_adjudication/llm_citation_adjudication.csv \
  --out-dir ./review-data/05_audit/draft_source_adjudication
```

Outputs:

- `accepted_verified_draft_papers.csv`: only API-verified rows accepted by LLM adjudication in this lane; merge all accepted lanes with `literature_pool.py merge-verified` before main-pool import.
- `rejected_or_demoted_sources.csv`: mismatch, preprint-only, non-paper, background-only, or unadjudicated rows.
- `claim_source_fit.csv`: claim-to-source fit decisions.
- `preprint_resolution.csv`: preprint rows requiring published-version search or human decision.
- `human_confirmation_queue.md`: user-facing decisions and missing evidence.

## Allowed LLM Fields

Use fixed enumerations:

- `title_match`: `exact | strong | partial | weak | mismatch | not_applicable`
- `publication_status`: `published_paper | preprint_only | nonpaper_or_guideline | unverifiable | unclear`
- `claim_fit`: `direct | indirect | background | none | unclear`
- `llm_decision`: `accept_verified | demote_preprint | reject_mismatch | reject_nonpaper | needs_manual | needs_published_version_search`
- `keep_as`: `citation_pool_candidate | background_only | supplemental_search_lead | delete_or_replace | human_decision`
- `required_next_action`: `none | import_to_pool | keep_out_of_final_refs | search_published_version | ask_user | delete_or_replace`

## Guardrails

- LLMs cannot upgrade `unverified_delete_or_replace` or `needs_api_verification` into verified sources.
- LLMs cannot turn an arXiv/OpenReview/bioRxiv-only lead into a final reference without a verified published DOI/PMID or publisher metadata.
- LLMs cannot cite from memory. If a paper is known but absent from the provided metadata, the next action is metadata search or user confirmation.
- `accepted_verified_draft_papers.csv` can only be derived from `verified_pubmed`, `verified_crossref`, or `verified_openalex` rows.
- Agent memory stores decisions, pending questions, and file pointers only; it is not an evidence source.
- If `claim_fit` is indirect, background, none, or unclear, rewrite, demote, or send the claim to the human queue before drafting.
