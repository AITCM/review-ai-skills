# Literature Pool Workflow

Use this workflow when the user wants reference download, a persistent literature pool, citation selection, or literature-card memory for future manuscript writing.

## Quick Navigation

- Conceptual layers and directory structure: pool lanes, citation pool, cards, and progressive disclosure.
- Steps 1-2: inspect draft citation assets, then verify draft-native papers.
- Steps 3-4: filter/gate the main pool and export a candidate packet for screening.
- Step 5: run argument-driven supplemental recall only for confirmed gaps.
- Steps 6-7: build the citation pool, fetch full text, and create user handoff lists.
- Steps 8-10: build literature cards, export compact context, and audit before drafting.
- Citation sequence rule: after drafting, deterministic first-appearance numbering rebuilds the final bibliography.
- Writing-time rule: cite only selected, gated, carded sources.

## Conceptual Layers

- `review-data/02_literature/draft_assets`: first-pass citation assets extracted from GPT/Gemini/Deep Research drafts. This is a health-check lane, not a verified literature pool.
- `review-data/02_literature/pool`: governed main pool for verified draft-native papers, user-approved sources, and promoted supplemental papers.
- `review-data/02_literature/supplemental_pool`: strict supplemental records found only after confirmed framework, claim, table, or figure evidence gaps.
- `citation_pool`: selected high-relevance papers that support the review thesis, evidence matrix, figures, tables, or major claims.
- `cards`: compact per-paper memory files. Load only the cards needed for the current section.

This is progressive disclosure for literature: inspect draft assets first, import only verified papers into the main pool, keep supplemental search isolated, load selected cards only when drafting, and let the citation-sequence manager assign final numeric order after prose stabilizes.

## Directory Structure

Initialize the main pool:

```bash
python $SKILL_DIR/scripts/literature_pool.py init --pool-dir ./review-data/02_literature/pool
```

Initialize the supplemental pool when a confirmed gap exists:

```bash
python $SKILL_DIR/scripts/literature_pool.py init --pool-dir ./review-data/02_literature/supplemental_pool
```

Each pool contains:

```text
pool.json
queries/
recall_runs/
papers/
cards/
contexts/
decisions/
logs/
```

## Step 1: Inspect Draft Citation Assets Before Search

When drafts exist, do not start with broad recall. First extract draft-native references, citation markers, unsupported claims, and non-academic sources:

```bash
python $SKILL_DIR/scripts/draft_citation_assets.py build \
  --draft-dir ./review-data/01_inputs/drafts_raw \
  --topic "<topic>" \
  --out-dir ./review-data/02_literature/draft_assets
```

Inspect:

- `draft_citation_health.md`
- `draft_reference_inventory.csv`
- `claim_evidence_map.csv`
- `candidate_paper_clues.csv`
- `non_academic_sources.csv`
- `uncited_references.csv`
- `unresolved_intext_markers.csv`
- `supplemental_recall_tasks.csv` and `supplemental_recall_queries.txt`

Treat web/product/blog sources as background leads. Treat uncited references and unresolved markers as cleanup tasks before preserving numbering.

## Step 2: Verify Draft-Native Papers

Use multi-signal identity resolution: PubMed/PMID, DOI, title, author/year/venue, and official URL clues. PMID/DOI values from drafts are not self-validating; if they resolve to a nonmatching title, fall back to title/bibliographic search:

```bash
python $SKILL_DIR/scripts/draft_reference_verifier.py verify \
  --candidate-csv ./review-data/02_literature/draft_assets/candidate_paper_clues.csv \
  --out-dir ./review-data/05_audit/draft_reference_verification \
  --email <email>
```

The verifier writes:

- `verified_draft_papers.csv`: rows verified by PubMed, Crossref, or OpenAlex; adjudicate and merge before import.
- `preprint_leads.csv`: arXiv/OpenReview/bioRxiv/medRxiv leads; do not use as final citations by default.
- `rejected_or_unverified_draft_sources.csv`: delete, replace, or ask the user to confirm.
- `draft_reference_verification_report.md`: status counts and gate policy.

Before import, adjudicate verified and preprint/unverified rows with Codex subtasks by default. APIs provide existence facts; the subtask only judges title match, preprint risk, claim-source fit, and human questions. DeepSeek/OpenAI-compatible adjudication is optional only when external data egress is explicitly approved.

```bash
python $SKILL_DIR/scripts/draft_source_adjudicator.py build-cases \
  --verification-csv ./review-data/05_audit/draft_reference_verification/all_draft_reference_verification.csv \
  --claim-map ./review-data/02_literature/draft_assets/claim_evidence_map.csv \
  --out-dir ./review-data/05_audit/draft_source_adjudication \
  --topic "review topic" \
  --batch-size 12
```

After Codex subtasks write `llm_citation_adjudication.csv`, merge the decisions:

```bash
python $SKILL_DIR/scripts/draft_source_adjudicator.py merge \
  --verification-csv ./review-data/05_audit/draft_reference_verification/all_draft_reference_verification.csv \
  --adjudication-csv ./review-data/05_audit/draft_source_adjudication/llm_citation_adjudication.csv \
  --out-dir ./review-data/05_audit/draft_source_adjudication
```

Import only accepted verified draft-native papers into the main pool:

```bash
python $SKILL_DIR/scripts/literature_pool.py import-csv \
  --pool-dir ./review-data/02_literature/pool \
  --csv ./review-data/05_audit/draft_source_adjudication/accepted_verified_draft_papers.csv \
  --origin draft-native-citation-assets-llm-adjudicated
```

## Step 3: Filter And Gate The Main Pool

Run a topic filter before candidate export. This rejects obvious pollution such as supplementary files, decision letters, figure captions, database records, records with no topical overlap, and preprint-only sources.

```bash
python $SKILL_DIR/scripts/literature_pool.py topic-filter \
  --pool-dir ./review-data/02_literature/pool \
  --topic "<topic>" \
  --include-keywords "<keyword-1>,<keyword-2>,<domain-term>,<method-term>" \
  --out ./review-data/02_literature/pool/logs/topic_filter_report.csv \
  --apply

python $SKILL_DIR/scripts/literature_pool.py final-gate \
  --pool-dir ./review-data/02_literature/pool \
  --status candidate,background,maybe \
  --out ./review-data/02_literature/pool/contexts/final_citation_gate.md \
  --apply
```

Inspect both reports before relying on the candidate packet. Deterministic filtering is only a first pass; LLM/human relevance decisions still govern citation promotion.

## Step 4: Export Candidate Packet For Screening

```bash
python $SKILL_DIR/scripts/literature_pool.py export-candidates \
  --pool-dir ./review-data/02_literature/pool \
  --out ./review-data/02_literature/pool/contexts/candidate_packet.md \
  --limit 120
```

Screen against:

- review thesis and approved framework
- inclusion/exclusion criteria
- target journal audience
- evidence hierarchy and publication status
- claim-source fit
- figure/table evidence needs
- novelty, coverage, and citation risk

For each selected paper, record relevance score, rationale, claim supported, limitation, and intended section.

## Step 5: Use Argument-Driven Supplemental Recall Only For Confirmed Gaps

Supplemental search is allowed only after the draft assets and framework discussion identify a real gap. Put results in `supplemental_pool` first:

```bash
python $SKILL_DIR/scripts/argument_literature_expander.py plan \
  --framework-dir ./review-data/03_framework/logic_framework \
  --topic "<topic>" \
  --out-dir ./review-data/02_literature/argument_literature_expansion

python $SKILL_DIR/scripts/argument_literature_expander.py pubmed \
  --tasks-csv ./review-data/02_literature/argument_literature_expansion/argument_literature_tasks.csv \
  --out-dir ./review-data/02_literature/argument_literature_expansion \
  --email <email> \
  --workers 2 \
  --resume

python $SKILL_DIR/scripts/literature_pool.py import-csv \
  --pool-dir ./review-data/02_literature/supplemental_pool \
  --csv ./review-data/02_literature/argument_literature_expansion/pubmed_argument_candidates.csv \
  --origin argument-driven-supplemental-pubmed
```

The argument expander preserves the linked claim, evidence need, PMID, journal, publication type, abstract, candidate-fit hint, and full-text need in the pool record. Use `abstract_screening_packet.md` for Codex/DeepSeek/human screening. Abstracts can justify triage, but detailed method/result/metric claims still require full text.

Older query-file recall remains a fallback:

```bash
python $SKILL_DIR/scripts/pubmed_recall.py \
  --query-file ./review-data/02_literature/draft_assets/supplemental_recall_queries.txt \
  --max-results 20 \
  --out-dir ./review-work/recall_runs/supplemental_pubmed

python $SKILL_DIR/scripts/literature_pool.py import-recall \
  --pool-dir ./review-data/02_literature/supplemental_pool \
  --recall-dir ./review-work/recall_runs/supplemental_pubmed \
  --origin strict-supplemental-gap-recall
```

For non-biomedical gaps, use Crossref/OpenAlex or paper-search with narrow, user-approved queries:

```bash
python $SKILL_DIR/scripts/paper_recall.py \
  --query "confirmed narrow evidence gap query" \
  --sources openalex,crossref,semantic,pubmed \
  --max-results 10 \
  --out-dir ./review-work/recall_runs/strict_supplemental
```

Promote supplemental records to the main pool only after relevance, publication status, claim fit, and source quality are checked.

## Step 6: Build The Citation Pool

Record selection decisions:

```bash
python $SKILL_DIR/scripts/literature_pool.py decide \
  --pool-dir ./review-data/02_literature/pool \
  --key doi-10-0000-example \
  --status citation_pool \
  --score 5 \
  --rationale "Directly supports the review's core framework" \
  --claim "<specific claim supported by this source>" \
  --limitation "<source-specific limitation>" \
  --use "<section or display item>"
```

Use statuses:

- `citation_pool`: core citation candidate
- `seminal`: foundational paper
- `recent`: important new development
- `method`: methods/reporting/source standard
- `background`: context only
- `maybe`: needs full-text inspection
- `rejected`: not relevant or unsafe to cite

## Step 7: Download Selected Papers

Use Europe PMC Open Access `fullTextXML` first for biomedical papers:

```bash
python $SKILL_DIR/scripts/fulltext_manager.py fetch-europepmc \
  --pool-dir ./review-data/02_literature/pool \
  --status citation_pool,seminal,method,recent
```

Then use OA-first `paper-search` download where source and paper ID are available:

```bash
python $SKILL_DIR/scripts/literature_pool.py download \
  --pool-dir ./review-data/02_literature/pool \
  --status citation_pool,seminal,method,recent \
  --limit 50
```

After automatic attempts, run the full-text audit:

```bash
python $SKILL_DIR/scripts/fulltext_manager.py audit \
  --pool-dir ./review-data/02_literature/pool
```

If `contexts/needs_user_fulltext.md` is non-empty, tell the user to add requested PDFs/TXT/MD/DOCX files to `review-data/01_inputs/user_fulltext`, then run:

```bash
python $SKILL_DIR/scripts/fulltext_manager.py import \
  --pool-dir ./review-data/02_literature/pool \
  --source-dir ./review-data/01_inputs/user_fulltext
```

For PDFs that still do not yield usable text, parse them with MinerU:

```bash
python $SKILL_DIR/scripts/fulltext_manager.py mineru-agent \
  --pool-dir ./review-data/02_literature/pool \
  --language ch \
  --is-ocr \
  --enable-table \
  --enable-formula
```

Do not proceed to detailed method/result/metric comparison until the key papers for that claim have extracted full text, unless the user explicitly approves abstract-only treatment.

## Step 8: Build Literature Cards

```bash
python $SKILL_DIR/scripts/literature_pool.py build-cards \
  --pool-dir ./review-data/02_literature/pool
```

Each card includes exact identifiers, provenance, claim support, limitations, citation cautions, and where to use the paper in the review.

## Step 9: Export Compact Citation Context

```bash
python $SKILL_DIR/scripts/literature_pool.py export-context \
  --pool-dir ./review-data/02_literature/pool \
  --out ./review-data/02_literature/pool/contexts/citation_context.md \
  --max-cards 40
```

When drafting, first load `cards/index.md` or `contexts/citation_context.md`. Load individual cards only for the section being written.

## Step 10: Audit Before Drafting

```bash
python $SKILL_DIR/scripts/literature_pool.py audit \
  --pool-dir ./review-data/02_literature/pool \
  --out ./review-data/02_literature/pool/contexts/pool_audit.md
```

Fix before drafting:

- selected papers missing literature cards
- selected papers missing supported-claim notes
- records missing DOI/URL/PDF URL
- unresolved `maybe` records needed for a key claim
- records that fail the final citation gate because they are unverifiable or only available as preprints/review-platform submissions

## Writing-Time Rule

Do not cite directly from draft assets, the noisy full pool, or the supplemental pool. Cite from selected citation-pool records and literature cards only after the final citation gate passes. If a needed source is only in the full pool or supplemental pool, promote it with a decision rationale, verify publication status, create a card, and rerun the final gate before using it in manuscript prose.
