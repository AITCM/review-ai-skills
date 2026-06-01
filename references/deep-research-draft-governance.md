# Deep Research Draft Governance

Use this workflow when the user starts from GPT, Gemini, Perplexity, Claude, or other Deep Research drafts rather than a blank page.

## Quick Navigation

- Principle: reuse draft insight while treating every source as untrusted.
- Source audit: separate papers, web/blog/product sources, malformed citations, and unsupported claims.
- Frame extraction: preserve useful frameworks, controversies, and terminology.
- Paper filtering: verify real publications before promotion.
- Claim replacement and governance output: replace weak support and record decisions.

## Principle

Treat AI-generated drafts as high-value but untrusted scaffolds:

- Reuse useful viewpoints, frameworks, terminology, controversy maps, and candidate sources.
- Do not trust cited references until verified through APIs or publisher/index pages.
- Demote webpages, blogs, product docs, and news to background context unless the target article explicitly needs non-paper evidence.
- Replace blog/web support for scientific claims with verified peer-reviewed papers, preprints, guidelines, datasets, or technical reports as appropriate.

## Source Audit

First build draft-native citation assets. This is a no-recall health check:

```bash
python $SKILL_DIR/scripts/draft_citation_assets.py build \
  --draft-dir ./review-data/01_inputs/drafts_raw \
  --topic "<topic>" \
  --out-dir ./review-data/02_literature/draft_assets
```

Review `draft_citation_health.md`, `claim_evidence_map.csv`, `candidate_paper_clues.csv`, `body_mentioned_systems.csv`, `unmapped_claims.csv`, `non_academic_sources.csv`, `uncited_references.csv`, and `unresolved_intext_markers.csv` before any supplemental search.

Then verify paper-like clues with multi-signal identity resolution. PMID/DOI values from drafts are strong clues, not truth; if the resolved title conflicts with the draft title, fall back to title/bibliographic search and flag the mismatch:

```bash
python $SKILL_DIR/scripts/draft_reference_verifier.py verify \
  --candidate-csv ./review-data/02_literature/draft_assets/candidate_paper_clues.csv \
  --out-dir ./review-data/05_audit/draft_reference_verification \
  --email <email>
```

Then adjudicate the verified/preprint/unverified rows with Codex subtasks by default. APIs establish candidate metadata; the Codex subtask only judges whether the candidate matches the draft reference and linked claim. DeepSeek/OpenAI-compatible adjudication is optional only when external data egress is explicitly approved:

```bash
python $SKILL_DIR/scripts/draft_source_adjudicator.py build-cases \
  --verification-csv ./review-data/05_audit/draft_reference_verification/all_draft_reference_verification.csv \
  --claim-map ./review-data/02_literature/draft_assets/claim_evidence_map.csv \
  --out-dir ./review-data/05_audit/draft_source_adjudication \
  --topic "review topic"
```

After Codex subtasks write `llm_citation_adjudication.csv`, merge the decisions:

```bash
python $SKILL_DIR/scripts/draft_source_adjudicator.py merge \
  --verification-csv ./review-data/05_audit/draft_reference_verification/all_draft_reference_verification.csv \
  --adjudication-csv ./review-data/05_audit/draft_source_adjudication/llm_citation_adjudication.csv \
  --out-dir ./review-data/05_audit/draft_source_adjudication
```

Use `accepted_verified_draft_papers.csv` as a lane-level source only. Merge accepted explicit-reference, literature-discovery, hidden/identity-normalized, and screened PubMed-deep-dive lanes with `literature_pool.py merge-verified` before official citation export or main-pool import. Treat `preprint_leads.csv` and `preprint_resolution.csv` as lead/background only by default, and treat rejected/unverified rows as deletion/replacement work.

Run:

```bash
python $SKILL_DIR/scripts/draft_source_audit.py \
  --draft ./gpt_deep_research.md \
  --draft ./gemini_deep_research.md \
  --out-dir ./review-data/05_audit/draft_source_audit \
  --mailto <email>
```

Use `--offline` when network/API calls are not available:

```bash
python $SKILL_DIR/scripts/draft_source_audit.py \
  --draft ./gpt_deep_research.md \
  --out-dir ./review-data/05_audit/draft_source_audit \
  --offline
```

Outputs:

- `source_audit.md`: framework/headings, claim-like sentences, non-paper sources, and unverified paper candidates
- `verified_papers.csv`: paper-only rows suitable for import into the canonical literature pool; includes Crossref/OpenAlex-normalized metadata when online verification succeeds
- `all_reference_audit.csv`: all extracted reference-like entries, including non-paper web/blog rows
- `web_sources.csv`: URLs classified as publisher, preprint, DOI, biomedical index, web, or blog
- `paper_title_queries.txt`: title queries for follow-up `paper_recall.py`
- `draft_frames.json`: machine-readable headings and claim-like sentences

Import verified papers into the pool only after all lanes are merged:

```bash
python $SKILL_DIR/scripts/literature_pool.py import-csv \
  --pool-dir ./review-data/02_literature/pool \
  --csv ./review-data/05_audit/verified_papers_union/verified_papers_union.csv \
  --origin draft-derived-verified-union

python $SKILL_DIR/scripts/literature_pool.py topic-filter \
  --pool-dir ./review-data/02_literature/pool \
  --topic "<topic>" \
  --include-keywords "<keyword-1>,<keyword-2>,<domain-term>,<method-term>" \
  --out ./review-data/02_literature/pool/logs/topic_filter_report.csv \
  --apply
```

Do not run broad title recall automatically. If the draft-native citation assets reveal an approved evidence gap, run strict supplemental recall into `supplemental_pool` first:

```bash
python $SKILL_DIR/scripts/pubmed_recall.py \
  --query "confirmed evidence gap query" \
  --max-results 20 \
  --out-dir ./review-work/recall_runs/strict_supplemental_pubmed

python $SKILL_DIR/scripts/literature_pool.py import-recall \
  --pool-dir ./review-data/02_literature/supplemental_pool \
  --recall-dir ./review-work/recall_runs/strict_supplemental_pubmed \
  --origin strict-supplemental-gap-recall
```

## Frame Extraction

Before source recall or manuscript drafting, use draft frames as proposal material, not final structure. When multiple drafts exist, first run the bundled `draft_logic_framework.py` to produce the architecture brief and user-alignment questions:

```bash
python $SKILL_DIR/scripts/draft_logic_framework.py \
  --draft-dir ./review-data/01_inputs/drafts_raw \
  --topic "<topic>" \
  --target-journal "<target-journal>" \
  --out-dir ./review-data/03_framework/logic_framework
```

Compare multiple drafts:

- repeated section headings imply consensus framing
- unique headings may suggest missing dimensions
- claim-like sentences become audit targets
- web/blog-supported claims need paper replacement
- speculative future agendas must be labeled as expert interpretation

Do not let a merged outline inherit every draft's top-level heading. For top-journal reviews, collapse related headings into a central thesis and 3-5 major body movements whenever possible.

## Paper Filtering

Classification rules:

- `verified paper`: can enter the canonical literature pool; may enter selected citation-pool records only after topic filtering and relevance reasoning
- `paper_unverified`: search again by title/DOI before use
- `publisher/preprint/biomedical_index URL`: good lead, still verify metadata
- `web_or_blog`: background only; do not use as scientific support for top-journal review claims
- `company/product documentation`: context for tools/systems, not evidence for scientific effectiveness
- `news`: background for timeline only

## Claim Replacement

For every draft claim supported by a web/blog source:

1. Extract the claim.
2. Search for peer-reviewed, preprint, guideline, or dataset evidence.
3. If found, cite verified papers and rewrite the claim at the supported strength.
4. If not found, mark `[CITATION NEEDED]`, `[WEB-SUPPORTED ONLY]`, or remove the claim.

## Governance Output

Before drafting, produce:

- `draft_logic_framework.md`: thesis candidates, architecture options, heading warnings, and section-spine recommendations
- `argument_evidence_map.csv`: claim-like sentences and citation markers to be verified
- `user_alignment_questions.md`: questions for the user before drafting
- accepted frame elements
- rejected frame elements and reasons
- verified papers imported into the pool
- non-paper sources demoted to background
- claims requiring paper replacement
- missing evidence that needs new search
