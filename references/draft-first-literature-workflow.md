# Draft-First Literature Workflow

Use this reference when the project starts from multiple GPT/Gemini/Deep Research drafts. The default order is: build draft memory, extract draft-native citation assets, have chief Codex read all drafts and citation assets, state a provisional thesis/claim map, create literature-discovery packets, dispatch Codex subagents to reason over full draft chunks and API routes, collect subagent candidates/API tasks with reference-overlap marking, normalize informal system names into official paper identities, verify real papers, and only then decide whether strict supplemental recall is needed. External LLM APIs such as DeepSeek are optional accelerators, not the default path.

Set `$SKILL_DIR` to the local `review-ai-skills` skill directory before running command examples.

## Quick Navigation

- Principle, lanes, and provenance: how draft-first evidence stays separated from later supplemental recall.
- Step 1-2: build draft memory/citation assets, ingest any user PubMed seed sets, build a candidate-board checkpoint, then create literature-discovery packets before API verification.
- Step 3: optional extra draft-prose candidate mining and identity normalization.
- Step 4-5: files chief Codex must load, deterministic verification, adjudication, union, and official citation export.
- Step 6-8: claim-evidence binding, argument-driven supplemental evidence, and final drafting/citation gates.

## Principle

Do not expand the corpus before you understand the draft corpus.

Deep Research drafts often mix papers, webpages, product pages, blogs, figure captions, supplementary files, and uncited reference leftovers. Broad recall from those strings amplifies noise. Start by reading the actual draft prose and cleaning/auditing the draft-native references; use supplemental search only for confirmed evidence gaps.

The order of authority is:

1. Chief Codex reads the drafts and extracts the paper's candidate logic.
2. Draft-native citation assets establish what is explicit, polluted, uncited, unresolved, or non-academic before any agent claims novelty.
3. A literature-discovery orchestrator turns the chief map into subagent packets and route-specific API tasks.
4. A draft-prose literature miner extracts named systems, venue clues, title fragments, and narrow PubMed/Crossref/arXiv/OpenReview queries from the body text, while marking overlap with explicit references as reference recovery.
5. The user-facing framework checkpoint defines the thesis, 3-5 section spine, argument map, historical foundations, current evidence claims, and future agenda.
6. Literature agents then retrieve, verify, screen, and manage sources as evidence for that argument map.

Literature is the proof layer, not the steering layer. New records must not dilute or replace the strongest logic already present in the user's drafts unless they provide verified counterevidence that the framework must be revised.

## Lanes

- `review-data/02_literature/draft_assets`: citation assets extracted from the drafts themselves. This is the first health check and should not be treated as a verified paper pool.
- `review-data/02_literature/user_pubmed_sets`: user-supplied PubMed Summary/Abstract exports parsed into screened seed candidates. This is curated search material, not accepted evidence until screened and verified.
- `review-data/02_literature/candidate_board`: read-only chief-editor board across candidate, screening, verifier-ready, verification, and repair/delete lanes.
- `review-data/02_literature/ai_candidate_mining`: AI-mined paper clues from draft prose, PubMed deep-dive tasks, and abstract screening packets. This is a candidate layer, not a verified pool.
- `review-data/02_literature/pool`: governed main pool for verified draft-native papers and user-approved seeds.
- `review-data/02_literature/supplemental_pool`: strict supplemental records added only after framework gaps, claim gaps, or table/figure needs are confirmed.
- `review-data/02_literature/pool/cards`: literature memory cards for selected citation-pool records.

## Provenance Categories

- `explicit_reference`: appears in a draft reference section and is extracted by `draft_citation_assets.py`.
- `body_cited_reference`: appears in the body and maps to a reference entry.
- `user_pubmed_seed`: appears in a user-prepared PubMed export and must be screened against the framework before verification or promotion.
- `reference_recovery`: appears in the reference assets but needs correction, official-title normalization, DOI/PMID repair, or publisher identity recovery.
- `hidden_body_candidate`: appears in prose as a system, method, author/year hint, venue clue, or title fragment but is not clearly present in the reference assets.
- `supplemental_gap_candidate`: newly searched paper for a confirmed framework/claim/display-item gap; keep in `supplemental_pool` until screened.

Do not describe `reference_recovery` rows as hidden discoveries. For example, if a draft reference list already names a Nature article but gives an informal title or wrong year, the subagent recovered the identity; it did not discover a missing paper.

## Step 1: Build Draft Memory And Citation Assets

Run draft memory first, then citation assets:

```bash
python $SKILL_DIR/scripts/draft_memory.py build \
  --draft-dir ./review-data/01_inputs/drafts_raw \
  --topic "<topic>" \
  --out-dir ./review-data/03_framework/draft_memory

python $SKILL_DIR/scripts/draft_citation_assets.py build \
  --draft-dir ./review-data/01_inputs/drafts_raw \
  --topic "<topic>" \
  --out-dir ./review-data/02_literature/draft_assets
```

Chief Codex must load `chief_editor_reading_checkpoint.md`, `draft_memory_index.md`, `agent_reading_packet.md`, relevant `cards/*.md`, `draft_citation_health.md`, `candidate_paper_clues.csv`, and `claim_evidence_map.csv` before delegating. It should then state a provisional thesis/claim map: past foundations, present systems, future agenda, disputed claims, and the citation lanes that need recovery or verification.

## Step 2: Orchestrate Literature Discovery Before API Verification

Create chief-Codex and subagent discovery packets:

```bash
python $SKILL_DIR/scripts/draft_literature_discovery_orchestrator.py plan \
  --draft-dir ./review-data/01_inputs/drafts_raw \
  --draft-assets-dir ./review-data/02_literature/draft_assets \
  --topic "<topic>" \
  --out-dir ./review-data/02_literature/literature_discovery
```

The current Codex conversation must read `chief_literature_brief_packet.md` and write/update the project-level chief literature brief before relying on subagents. Then dispatch Codex subagents over `subagent_discovery_packets/*.md`. Subagents should return CSV rows with candidate paper/system, claim role, time role, evidence need, and API route.

Collect the subagent outputs:

```bash
python $SKILL_DIR/scripts/draft_literature_discovery_orchestrator.py collect \
  --input-dir ./review-data/02_literature/literature_discovery/subagent_results \
  --reference-candidates-csv ./review-data/02_literature/draft_assets/candidate_paper_clues.csv \
  --out-dir ./review-data/02_literature/literature_discovery/collected
```

Outputs:

- `chief_literature_brief_packet.md`: mandatory main-Codex reading and literature strategy packet.
- `subagent_discovery_packets/`: full-draft chunk packets for subagent reasoning.
- `collected/literature_discovery_candidates.csv`: deduped AI-discovered/recovered candidates.
- `collected/candidate_paper_clues_for_verification.csv`: deterministic verifier input.
- `collected/api_search_tasks.csv`: PubMed/Crossref/OpenAlex/arXiv/OpenReview/publisher/human route tasks.
- `collected/queries/*.txt`: route-specific query files for API recall scripts.
- `collected/human_literature_discovery_checkpoint.md`: checkpoint before promotion.

This is the main defense against under-counting literature. It is not optional when the review relies on multiple AI-generated drafts.

## Step 2a: Ingest User PubMed Seed Sets When Provided

If the user provides PubMed-generated Summary or Abstract exports, parse them after draft citation assets and before broad supplemental recall:

```bash
python $SKILL_DIR/scripts/pubmed_user_set_ingestor.py parse \
  --input ./review-data/01_inputs/drafts_raw/abstract-<set>.txt \
  --input ./review-data/01_inputs/drafts_raw/summary-<set>.txt \
  --topic "<topic>" \
  --set-name "<set-name>" \
  --out-dir ./review-data/02_literature/user_pubmed_sets/<set-name>
```

Chief Codex must inspect `pubmed_user_set_ingestion_summary.md`, `user_pubmed_screening_guide.md`, and top rows of `pubmed_user_set_candidates.csv` before judging whether the PubMed set fills current, historical, governance, evaluation, or application evidence gaps.

Policy:

- A user PubMed set is curated search material, not a final reference list.
- Keep it separate from draft-native references and broad recall.
- Assign `screening_packets/` to Codex subagents or human reviewers before deterministic verification.
- Promote only screened inclusions to verification/adjudication; keep off-frame or low-quality rows in the seed-set folder for audit.
- Abstracts are enough for triage, but method/result/metric claims still require full text.

After subagent/human screening decisions are saved, create verifier-ready user PubMed aliases:

```bash
python $SKILL_DIR/scripts/pubmed_user_set_ingestor.py collect-screening \
  --set-dir ./review-data/02_literature/user_pubmed_sets/<set-name> \
  --screening-csv ./review-data/02_literature/user_pubmed_sets/<set-name>/agent_screening_decisions.csv
```

Then run deterministic verification on `screened/user_pubmed_candidates_for_verification.csv`.

## Step 2b: Build The Candidate Board

Build the board after initial draft assets/user PubMed ingestion, after literature-discovery collection, after screening collection, and after verification:

```bash
python $SKILL_DIR/scripts/literature_candidate_board.py build \
  --project-dir . \
  --out-dir ./review-data/02_literature/candidate_board
```

If the review root contains old smoke tests, archived candidate-mining experiments, or previous verification runs, scope the board before sending it to agents:

```bash
python $SKILL_DIR/scripts/literature_candidate_board.py build \
  --project-dir . \
  --out-dir ./review-data/02_literature/candidate_board \
  --exclude-path-contains ai_candidate_mining_test \
  --exclude-path-contains review-data/05_audit/old_smoke
```

```bash
python $SKILL_DIR/scripts/literature_candidate_board.py packets \
  --project-dir . \
  --board-dir ./review-data/02_literature/candidate_board \
  --out-dir ./review-data/02_literature/candidate_board/agent_packets \
  --packet-size 25

python $SKILL_DIR/scripts/literature_candidate_board.py collect \
  --project-dir . \
  --board-dir ./review-data/02_literature/candidate_board \
  --input-dir ./review-data/02_literature/candidate_board/agent_results \
  --out-dir ./review-data/02_literature/candidate_board/collected
```

Chief Codex must use `literature_candidate_board.md`, `screening_worklist_deduped.csv`, and `verification_worklist_deduped.csv` to decide whether the next bottleneck is more draft reading, screening, deterministic verification, repair/delete decisions, or full-text/citation export. Do not launch another broad recall wave until the board shows why the existing candidate lanes are insufficient.

Use `agent_packets/agent_packet_index.md` to dispatch `candidate_board_curator`, `literature_screener`, and `citation_verifier` subtasks. Keep the active batch small; close subagents after their CSV is saved. The curator packet decides the next bottleneck; screener packets produce inclusion/exclusion CSVs with claim-evidence bindings; verifier packets preflight verifier-ready rows before API calls. Save returned CSVs under `candidate_board/agent_results/`, run `collect`, then inspect `candidate_board_next_actions.md`. The collector tolerates concise natural-language next-action notes but always converts them back into explicit verifier, repair, full-text, rejection, duplicate, and human-decision queues. The collected queue is still pre-promotion: run deterministic verification and adjudication before any import into the main literature pool.

When a candidate is only a system name, acronym, or informal preprint clue, keep it in the queue instead of deleting it too early. The verifier performs PubMed system-name deep dives and phrase-contained title matching, which can recover formal records from clues such as `CRISPR-GPT` even when the draft supplied only a preprint-style reference.

For literature-flow retrospectives, use verifier traces rather than guessing what happened. `review-data/05_audit/<verification-run>/verification_query_trace.csv` records every PubMed query mode, returned ID list, title-mismatch exclusion, ranked candidate, and accepted hit, so the chief Codex can decide whether a miss was caused by a weak candidate clue, an overly strict title gate, a PubMed indexing gap, or the need for Crossref/OpenAlex/publisher repair.

## Step 3: Optional Extra AI-Mine Draft Prose For Candidate Papers

Run this after the literature-discovery orchestrator when system-name/title-fragment clues still look under-extracted. It is designed for the failure mode where the draft mentions a key Nature/PubMed paper through a system name, method name, author/year hint, or venue clue, but the reference list is incomplete, informal, or polluted. The default is to generate packets and let Codex subtasks mine the candidates locally.

Codex subtask packet mode:

```bash
python $SKILL_DIR/scripts/draft_literature_candidate_miner.py packet \
  --draft-dir ./review-data/01_inputs/drafts_raw \
  --topic "<topic>" \
  --out-dir ./review-data/02_literature/ai_candidate_mining
```

After Codex subtasks read the packets, save their CSV/markdown outputs under a folder such as `./review-data/02_literature/ai_candidate_mining/subagent_results`, then collect and normalize:

```bash
python $SKILL_DIR/scripts/draft_literature_candidate_miner.py collect \
  --input-dir ./review-data/02_literature/ai_candidate_mining/subagent_results \
  --reference-candidates-csv ./review-data/02_literature/draft_assets/candidate_paper_clues.csv \
  --out-dir ./review-data/02_literature/ai_candidate_mining/subagent_collected

python $SKILL_DIR/scripts/draft_literature_candidate_miner.py normalize-packet \
  --candidate-csv ./review-data/02_literature/ai_candidate_mining/subagent_collected/subagent_hidden_candidate_clues.csv \
  --out-dir ./review-data/02_literature/ai_candidate_mining/identity_normalization \
  --topic "<topic>"

python $SKILL_DIR/scripts/draft_literature_candidate_miner.py merge-normalized \
  --candidate-csv ./review-data/02_literature/ai_candidate_mining/subagent_collected/subagent_hidden_candidate_clues.csv \
  --normalized-csv ./review-data/02_literature/ai_candidate_mining/identity_normalization/normalized_identities.csv \
  --out-dir ./review-data/02_literature/ai_candidate_mining/identity_normalization/merged \
  --topic "<topic>"
```

This identity-normalization gate is required because drafts often use system names, lab nicknames, news headlines, or informal titles instead of official paper titles. Rows that remain ambiguous should stay in the human queue rather than going directly to verification or the literature pool.

Optional external DeepSeek mode, only when external data egress is explicitly approved:

```bash
python $SKILL_DIR/scripts/draft_literature_candidate_miner.py mine \
  --project-dir . \
  --draft-dir ./review-data/01_inputs/drafts_raw \
  --topic "<topic>" \
  --out-dir ./review-data/02_literature/ai_candidate_mining \
  --max-workers 4
```

Then run PubMed deep-dive queries from the AI-mined clues:

```bash
python $SKILL_DIR/scripts/draft_literature_candidate_miner.py pubmed \
  --tasks-csv ./review-data/02_literature/ai_candidate_mining/ai_pubmed_deep_dive_tasks.csv \
  --out-dir ./review-data/02_literature/ai_candidate_mining/pubmed_deep_dive \
  --email <email>
```

After Codex/human screening of `ai_pubmed_abstract_screening_packet.md`, convert eligible abstract hits back into deterministic verifier input:

```bash
python $SKILL_DIR/scripts/draft_literature_candidate_miner.py pubmed-to-verifier \
  --abstracts-csv ./review-data/02_literature/ai_candidate_mining/pubmed_deep_dive/ai_pubmed_candidate_abstracts.csv \
  --tasks-csv ./review-data/02_literature/ai_candidate_mining/ai_pubmed_deep_dive_tasks.csv \
  --out-dir ./review-data/02_literature/ai_candidate_mining/pubmed_deep_dive/verifier_input
```

Outputs:

- `ai_candidate_paper_clues.csv`: candidate clues mined from draft body text.
- `candidate_paper_clues_for_verification.csv`: input for deterministic verification.
- `subagent_collected/subagent_hidden_candidate_clues.csv`: structured Codex-subtask candidates; rows overlapping draft-native `candidate_paper_clues.csv` are marked as `explicit_reference_recovered_by_subagent` / `codex_subagent_reference_recovery`.
- `identity_normalization/identity_normalization_packets/`: Codex-subtask packets for official paper identity normalization.
- `identity_normalization/merged/candidate_paper_clues_for_verification.csv`: verifier input after title/identity normalization.
- `ai_pubmed_deep_dive_tasks.csv`: narrow PubMed searches linked to draft chunks.
- `human_ai_candidate_checkpoint.md`: keep/verify/search/delete/merge decisions.
- `pubmed_deep_dive/ai_pubmed_candidate_abstracts.csv`: PubMed abstract candidates for screening.
- `pubmed_deep_dive/verifier_input/pubmed_abstract_candidates_for_verification.csv`: screened abstract candidates ready for the verifier.

The miner may suggest a key paper, but it cannot promote it. Run deterministic verification and claim-fit screening before import or citation.

## Step 4: Draft Citation Asset Files To Load

```bash
python $SKILL_DIR/scripts/draft_citation_assets.py build \
  --draft-dir ./review-data/01_inputs/drafts_raw \
  --topic "<topic>" \
  --out-dir ./review-data/02_literature/draft_assets
```

Outputs:

- `draft_citation_health.md`: human-readable blockers and file list.
- `draft_reference_inventory.csv`: all reference entries, DOI/PMID/URL clues, source kind, and whether cited in the body.
- `draft_intext_citations.csv`: body citation markers and whether they resolve to reference entries.
- `claim_evidence_map.csv`: claim snippets, original text, citation markers, mapped reference numbers, and candidate evidence titles.
- `candidate_paper_clues.csv`: paper-like clues for PubMed/DOI verification.
- `non_academic_sources.csv`: web/product/blog/unknown sources to demote.
- `uncited_references.csv`: references that appear only in the reference list.
- `unresolved_intext_markers.csv`: body markers without a matching reference.
- `body_mentioned_systems.csv`: system/method names in prose that should seed subagent identity normalization, not direct citation.
- `unmapped_claims.csv`: claims with no mapped reference number.
- `supplemental_recall_tasks.csv`: unsupported claims that may need search after user/LLM triage.
- `supplemental_recall_queries.txt`: plain-text query seeds for strict supplemental recall after human/LLM approval.

Important: `candidate_paper_clues.csv` is a reference-section/mapped-citation asset, not the complete paper count for the drafts. Use it as the explicit-reference lane; use literature-discovery packets and subagents to recover hidden body candidates, claim-gap candidates, and missing seminal/current papers.

## Step 5: Verify Draft-Native And AI-Discovered Papers First

Use PubMed/PMID, DOI, title, author/year/venue, and official URL clues as a multi-signal identity loop. Official publisher URLs can themselves contain DOI evidence, for example Nature/Springer article paths such as `nature.com/articles/s41586-...` imply `10.1038/s41586-...`; however, URL-derived DOI evidence must still resolve to a title compatible with the candidate paper through Crossref/OpenAlex or official export. For biomedical topics, do not start with broad OpenAlex topic recall.

PMID/DOI values from drafts are not self-validating. A DOI may be copied from the wrong paper, point to a preprint, or be attached to a nonmatching title. If PMID/DOI resolution returns a title that conflicts with the draft title, reject that identifier match, record the mismatch, and fall back to exact-title or bibliographic search.

Recommended order:

1. PMID/DOI rows from `candidate_paper_clues.csv`, accepted only when resolved metadata matches the draft title or other bibliographic clues.
2. PubMed exact-title and title/bibliographic search when identifiers are missing or mismatched.
3. Crossref/OpenAlex title search when PubMed is not appropriate, fails, or exposes a mismatched draft DOI.
4. Official publisher/conference pages where PubMed/Crossref/OpenAlex misses a formal paper.
5. arXiv/OpenReview only as lead discovery unless official venue evidence confirms a conference paper.

Run the draft-reference verifier on the paper-like clues:

```bash
python $SKILL_DIR/scripts/draft_reference_verifier.py verify \
  --candidate-csv ./review-data/02_literature/draft_assets/candidate_paper_clues.csv \
  --out-dir ./review-data/05_audit/draft_reference_verification \
  --email <email> \
  --workers 4 \
  --resume
```

This produces `verified_draft_papers.csv`, `preprint_leads.csv`, `rejected_or_unverified_draft_sources.csv`, and `draft_reference_verification_report.md`. Do not import yet; first run LLM/subagent adjudication and merge.

Run the same verifier on AI-mined clues:

```bash
python $SKILL_DIR/scripts/draft_reference_verifier.py verify \
  --candidate-csv ./review-data/02_literature/ai_candidate_mining/identity_normalization/merged/candidate_paper_clues_for_verification.csv \
  --out-dir ./review-data/05_audit/ai_candidate_reference_verification \
  --email <email> \
  --workers 4 \
  --resume
```

Also verify literature-discovery candidates:

```bash
python $SKILL_DIR/scripts/draft_reference_verifier.py verify \
  --candidate-csv ./review-data/02_literature/literature_discovery/collected/candidate_paper_clues_for_verification.csv \
  --out-dir ./review-data/05_audit/literature_discovery_reference_verification \
  --email <email> \
  --workers 4 \
  --resume
```

The verifier writes per-record progress to `verification_progress.jsonl` and periodically rewrites the CSV/report outputs. If a long PubMed/Crossref/OpenAlex run times out, rerun with `--resume`; completed rows are not repeated. Use `--workers 2-4` for moderate parallelism and keep PubMed `--email`/`--api-key` configured when available. For unstable networks, use `--workers 1 --resume --timeout 40`.

Before import, run Codex-subtask adjudication. This step judges title match, preprint-to-published risk, and claim-source fit without inventing identifiers:

```bash
python $SKILL_DIR/scripts/draft_source_adjudicator.py build-cases \
  --verification-csv ./review-data/05_audit/draft_reference_verification/all_draft_reference_verification.csv \
  --claim-map ./review-data/02_literature/draft_assets/claim_evidence_map.csv \
  --out-dir ./review-data/05_audit/draft_source_adjudication \
  --topic "review topic" \
  --batch-size 12

python $SKILL_DIR/scripts/draft_source_adjudicator.py merge \
  --verification-csv ./review-data/05_audit/draft_reference_verification/all_draft_reference_verification.csv \
  --adjudication-csv ./review-data/05_audit/draft_source_adjudication/llm_citation_adjudication.csv \
  --out-dir ./review-data/05_audit/draft_source_adjudication
```

Run the same adjudication chain for `literature_discovery_reference_verification` and `ai_candidate_reference_verification`. Do not stop at verification; verified AI-discovered rows must pass claim-fit adjudication, official citation export, and pool import with origin `draft-prose-ai-discovered-adjudicated`.

If screened PubMed deep-dive candidates were converted to verifier input, run the same verification and adjudication chain for `pubmed_deep_dive_reference_verification` as well.

Use `build-cases` packet files with Codex subtasks as the default. Use `adjudicate --max-workers 4` or `8` for DeepSeek parallel batch review only when API use and external data egress are explicitly approved.

After source adjudication accepts verified paper identities, merge all accepted lanes into one deduped union:

```bash
python $SKILL_DIR/scripts/literature_pool.py merge-verified \
  --csv ./review-data/05_audit/draft_source_adjudication/accepted_verified_draft_papers.csv \
  --csv ./review-data/05_audit/literature_discovery_source_adjudication/accepted_verified_draft_papers.csv \
  --csv ./review-data/05_audit/ai_candidate_source_adjudication/accepted_verified_draft_papers.csv \
  --csv ./review-data/05_audit/pubmed_deep_dive_source_adjudication/accepted_verified_draft_papers.csv \
  --out-dir ./review-data/05_audit/verified_papers_union \
  --origin draft-derived-verified-union \
  --skip-missing
```

This closes the old side-channel failure mode where hidden/subagent recovery increased the evidence pool but never entered official export/import. The union report separates input rows, unique papers, duplicate rows, and source lanes.

Then fetch official citation-manager exports. These exports, not the original draft strings, are the source for the final bibliography format.

```bash
python $SKILL_DIR/scripts/official_citation_exporter.py export \
  --verified-csv ./review-data/05_audit/verified_papers_union/verified_papers_union.csv \
  --out-dir ./review-data/05_audit/official_citations \
  --style vancouver \
  --email <email> \
  --dedupe-by doi,pmid,title
```

The exporter writes BibTeX/RIS/NBIB/format-text when PubMed, Crossref/DOI, or arXiv exposes an official export. Rows without an official export are written to `official_citation_export_report.md` as a manual handoff list. Use derived citation strings only as an internal draft aid, not as final citation formatting.

Import verified paper rows into the main pool only from the deduped union:

```bash
python $SKILL_DIR/scripts/literature_pool.py import-csv \
  --pool-dir ./review-data/02_literature/pool \
  --csv ./review-data/05_audit/verified_papers_union/verified_papers_union.csv \
  --origin draft-derived-verified-union
```

Then run topic filtering and published-only checks:

```bash
python $SKILL_DIR/scripts/literature_pool.py topic-filter \
  --pool-dir ./review-data/02_literature/pool \
  --topic "review topic" \
  --include-keywords "domain-specific keywords" \
  --out ./review-data/02_literature/pool/logs/topic_filter_report.csv \
  --apply

python $SKILL_DIR/scripts/literature_pool.py final-gate \
  --pool-dir ./review-data/02_literature/pool \
  --status candidate,background \
  --out ./review-data/02_literature/pool/contexts/final_citation_gate.md \
  --apply
```

## Step 6: Claim-Evidence Binding

Before outline approval, inspect `claim_evidence_map.csv` with the framework blackboard:

- Keep claims whose cited papers are verified and relevant.
- Demote claims supported only by webpages, blogs, product pages, or preprints.
- Mark claims with no mapped reference as `needs_user_or_llm_triage_before_search`.
- Do not preserve reference numbering from the original draft until unresolved markers and uncited references are handled.
- Preserve draft-derived logic even when evidence is missing: mark the claim as a search task, counterevidence task, or user-decision task instead of deleting it automatically.

## Step 7: Argument-Driven Supplemental Evidence

Supplemental recall is allowed only for:

- a framework claim approved by the user but not supported by draft-native papers
- a table row or figure panel that needs evidence
- a missing seminal/recent paper identified by the literature strategist
- a citation-verifier replacement for a polluted or unverifiable draft source
- a past-to-present bridge needed to explain why the current topic emerged

First convert the confirmed framework gaps into bounded retrieval tasks:

```bash
python $SKILL_DIR/scripts/argument_literature_expander.py plan \
  --framework-dir ./review-data/03_framework/logic_framework \
  --topic "<topic>" \
  --out-dir ./review-data/02_literature/argument_literature_expansion
```

Review `human_argument_literature_checkpoint.md` before running a large search. The output tasks must each name a draft-derived claim, evidence need, allowed destination, and promotion gate. The current Codex conversation or a literature strategist subagent may narrow or merge tasks, but it must not turn this into broad topic recall.

Then fetch PubMed metadata and abstracts for the accepted tasks:

```bash
python $SKILL_DIR/scripts/argument_literature_expander.py pubmed \
  --tasks-csv ./review-data/02_literature/argument_literature_expansion/argument_literature_tasks.csv \
  --out-dir ./review-data/02_literature/argument_literature_expansion \
  --email <email> \
  --workers 2 \
  --resume
```

This writes `pubmed_argument_candidates.csv`, `abstract_screening_packet.md`, `claim_candidate_links.csv`, and incremental JSONL progress logs. Abstracts are triage material: they can support inclusion/exclusion screening and help discover replacement evidence, but detailed method, metric, result, or clinical-effect claims require full text.

Write supplemental records to `review-data/02_literature/supplemental_pool` first:

```bash
python $SKILL_DIR/scripts/literature_pool.py import-csv \
  --pool-dir ./review-data/02_literature/supplemental_pool \
  --csv ./review-data/02_literature/argument_literature_expansion/pubmed_argument_candidates.csv \
  --origin argument-driven-supplemental-pubmed
```

For non-PubMed or method papers, run the same claim-bounded task list through `paper_recall.py`, `crossref_openalex_recall.py`, or `arxiv_openreview_recall.py` and keep those outputs in `supplemental_pool` until screened.

If the verified draft-derived union remains too small for the target journal, or if the framework still lacks historical foundations, current landmarks, counterevidence, governance, or methods/evaluation papers, run the broad supplemental recall screening layer:

```bash
python $SKILL_DIR/scripts/supplemental_recall_screening.py plan \
  --topic "<topic>" \
  --draft-assets-dir ./review-data/02_literature/draft_assets \
  --literature-discovery-dir ./review-data/02_literature/literature_discovery/collected \
  --out-dir ./review-data/02_literature/supplemental_recall_screening \
  --max-tasks 160

python $SKILL_DIR/scripts/supplemental_recall_screening.py pubmed \
  --tasks-csv ./review-data/02_literature/supplemental_recall_screening/supplemental_recall_tasks.csv \
  --out-dir ./review-data/02_literature/supplemental_recall_screening/pubmed \
  --max-results 20 \
  --workers 2 \
  --resume \
  --email <email>

python $SKILL_DIR/scripts/supplemental_recall_screening.py collect-candidates \
  --recall-dir ./review-data/02_literature/supplemental_recall_screening/pubmed \
  --out-dir ./review-data/02_literature/supplemental_recall_screening/collected
```

Then dispatch Codex subagents over `collected/agent_screening_packet.md`. Their `agent_screening_decisions.csv` must use decisions such as `include_core`, `include_support`, `include_counterevidence`, `include_background`, `reject`, or `needs_fulltext`. Collect screened inclusions:

```bash
python $SKILL_DIR/scripts/supplemental_recall_screening.py collect-screening \
  --candidates-csv ./review-data/02_literature/supplemental_recall_screening/collected/supplemental_recall_candidates.csv \
  --screening-csv ./review-data/02_literature/supplemental_recall_screening/collected/agent_screening_decisions.csv \
  --out-dir ./review-data/02_literature/supplemental_recall_screening/screened
```

Only `screened/supplemental_candidates_for_verification.csv` can move into deterministic verification. Broad recall is intentionally noisy; agent screening protects the review's confirmed logic from being diluted by interesting but off-map papers.

Older broad recall remains a fallback, not the default:

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

Promote from supplemental pool to the main pool only after LLM/human screening, topic filtering, final-gate checks, and claim-source fit review.

After promotion, run Europe PMC full-text fetching and then a full-text audit. If full text cannot be fetched, tell the user exactly which PDFs/TXT/MD/DOCX files to add to `review-data/01_inputs/user_fulltext` before detailed synthesis.

## Step 8: Draft And Citation Gates

Before drafting, run the workflow gatekeeper:

```bash
python $SKILL_DIR/scripts/workflow_gatekeeper.py check \
  --project-dir . \
  --stage pre-draft \
  --fail-on-blocker
```

If it reports `blocked`, stop and show the report to the user. Do not continue into a polished manuscript merely because checkpoint files exist; the checkpoint must be discussed or explicitly waived. The gate is designed to catch common false-completion states: subagent packets generated but not run, candidate-board rows still waiting for screening, verified rows not exported as official citations, no governed pool, no full-text handoff, no RAG index, or display items not approved.

Generate new manuscript prose only after:

- user confirms the central thesis and 3-5 section spine
- literature-discovery and candidate-board packets have either subagent/human results or an explicit local-only fallback note
- verified papers have official citation export or manual official-export handoff
- verified papers have been imported into the governed literature pool
- full-text/RAG status has been presented to the user; detailed method/result/metric claims wait for full text or explicit abstract-only approval
- citation-pool records have cards or full-text notes
- display items have human decisions
- each major claim has a verified support path

During revision, keep these gates:

- claim-support gate: every important sentence must map to verified evidence, scoped interpretation, or explicit agenda
- reference-cleaning gate: no uncited references, no unresolved markers, no duplicate numbering, no web/blog scientific support
- published-only gate: preprints remain background unless the user explicitly changes policy
- narrative guard: no internal workflow artifacts in main text; no "草稿核验显示", `final-gate`, candidate-board, RAG-state, literature-pool count, or source-audit bookkeeping language in the article body

After the Markdown manuscript exists, first run the narrative-contamination guard:

```bash
python $SKILL_DIR/scripts/manuscript_narrative_guard.py audit \
  --manuscript ./review-output/manuscript/<draft>.md \
  --out-dir ./review-data/05_audit/manuscript_narrative \
  --fail-on-blocker
```

Rewrite blockers as field-level synthesis: genealogy, present evidence pattern, evidence ladder, boundary condition, controversy map, or future agenda. Do not polish a manuscript that still reads like a workflow report.

Then run the deterministic citation-sequence gate before Word/PDF export:

```bash
python $SKILL_DIR/scripts/citation_sequence_manager.py renumber \
  --manuscript ./review-output/manuscript/<draft>.md \
  --official-citations-csv ./review-data/05_audit/official_citations/official_citations.csv \
  --out-dir ./review-data/05_audit/citation_sequence \
  --compress-ranges \
  --require-official \
  --fail-on-blocker
```

Use `review-data/05_audit/citation_sequence/manuscript_renumbered.md` as the source for final DOCX/PDF generation. The writing model's original citation numbers are only temporary anchors; final numbering comes from first appearance in the rebuilt manuscript. Resolve `missing_reference_entries.csv`, `missing_official_citations.csv`, and `unresolved_citation_markers.csv` before treating the manuscript as submission-ready.
