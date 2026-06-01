# Scripted Literature Management

Use these bundled scripts when the user needs repeatable literature recall, auditable screening, or a saved evidence matrix.

Before running examples, set `$SKILL_DIR` to the local skill directory. In this reference, script paths under `$SKILL_DIR/scripts/` mean bundled skill scripts, never project-local `./scripts/` files.

## Quick Navigation

- Script 0-0h: project setup, draft memory, draft literature-discovery orchestration, draft citation assets, user-supplied PubMed seed-set ingestion, candidate-board control surface, AI draft-prose literature candidate mining, draft-reference verification, LLM/source adjudication, verified-lane union, official citation export, draft logic framework, argument-driven evidence expansion, and broad supplemental recall screening.
- Script 1-1d: paper-search, PubMed fallback, Crossref/OpenAlex metadata lookup, and arXiv/OpenReview lead discovery.
- Script 2: long-lived review library and evidence-matrix export.
- Direct MCP Tool Shape and Screening Statuses: when `paper-search-mcp` is available as a live MCP server.
- Practical Pattern: default draft-first sequence.
- Script 3: governed literature pool, citation pool, downloads, cards, and final gate.
- Script 4: legacy draft source audit.
- Script 5-7d: full text, Europe PMC, user handoff, MinerU, structured RAG, display-item planning, workflow gates, narrative contamination guard, and deterministic citation sequencing.
- Script 8-9: agent memory/orchestration and cover-letter package.

## Script 0: `scripts/project_bootstrap.py`

Purpose: inspect and initialize the review root for tool-heavy, multi-session, or multi-agent projects.

Check the current project:

```bash
python $SKILL_DIR/scripts/project_bootstrap.py check-env --project-dir .
```

If `.venv`, `pyproject.toml`, `src/review_project`, `review-data`, `review-work`, `review-output`, or the canonical literature pool are missing, ask the user before running `init` unless the user already explicitly asked for setup.

Initialize standard folders, `.env.example`, `pyproject.toml`, `src/review_project`, and optional `.venv`:

```bash
python $SKILL_DIR/scripts/project_bootstrap.py init \
  --project-dir . \
  --package-name review_project \
  --create-venv
```

Use dry-run first if the project may contain important existing files:

```bash
python $SKILL_DIR/scripts/project_bootstrap.py init --project-dir . --dry-run
```

The script never prints API key values. It only reports whether `DEEPSEEK_API_KEY` or `REVIEW_AGENT_API_KEY` is configured.

## Script 0a: `scripts/draft_memory.py`

Purpose: deeply read one or more AI-generated review drafts into persistent memory cards before framework discussion.

```bash
python $SKILL_DIR/scripts/draft_memory.py build \
  --draft-dir ./review-data/01_inputs/drafts_raw \
  --topic "<topic>" \
  --out-dir ./review-data/03_framework/draft_memory
```

Outputs:

- `draft_memory_index.md`: compact memory index for Codex and agents.
- `chief_editor_reading_checkpoint.md`: mandatory chief-Codex reading/task-board checkpoint before delegation or framework drafting.
- `cards/*.md`: per-draft memory cards.
- `agent_reading_packet.md`: condensed packet for draft-reading and framework agents.
- `claims_for_discussion.csv`: draft-derived claims to verify, preserve, demote, or delete.
- `draft_memory.sqlite`: searchable draft memory.

Use this before `draft_citation_assets.py`, `draft_literature_candidate_miner.py`, and `draft_logic_framework.py`.

## Script 0a-discovery: `scripts/draft_literature_discovery_orchestrator.py`

Purpose: force a reasoning-first literature stage after draft memory and citation assets. Chief Codex reads all drafts and writes a provisional thesis/claim map, then Codex subagents read full-draft packets to recover explicit references, reference-recovery cases, hidden body candidates, claim-gap candidates, seminal background needs, and current landmark papers. The output is a candidate/API-task layer; it does not certify papers.

Create chief and subagent packets:

```bash
python $SKILL_DIR/scripts/draft_literature_discovery_orchestrator.py plan \
  --draft-dir ./review-data/01_inputs/drafts_raw \
  --draft-assets-dir ./review-data/02_literature/draft_assets \
  --topic "<topic>" \
  --out-dir ./review-data/02_literature/literature_discovery
```

Collect subagent CSV/markdown outputs:

```bash
python $SKILL_DIR/scripts/draft_literature_discovery_orchestrator.py collect \
  --input-dir ./review-data/02_literature/literature_discovery/subagent_results \
  --reference-candidates-csv ./review-data/02_literature/draft_assets/candidate_paper_clues.csv \
  --out-dir ./review-data/02_literature/literature_discovery/collected
```

Outputs:

- `chief_literature_brief_packet.md`: mandatory main-Codex reading packet.
- `subagent_discovery_packets/`: one packet per draft body/reference chunk.
- `collected/literature_discovery_candidates.csv`: deduped subagent candidates.
- `collected/candidate_paper_clues_for_verification.csv`: deterministic verifier input.
- `collected/api_search_tasks.csv`: route-specific API tasks for PubMed, Crossref/OpenAlex, arXiv/OpenReview, publisher pages, or human lookup.
- `collected/queries/*.txt`: query files for API recall scripts.

This stage is mandatory when multiple GPT/Gemini/Deep Research drafts are the starting point. It prevents the workflow from under-counting papers by relying only on reference sections.

## Script 0b-ai: `scripts/draft_literature_candidate_miner.py`

Purpose: create Codex-subtask packets to read draft body prose and extract paper candidates that are not reliably captured by the reference list alone. This catches named systems, author/year/venue hints, title fragments, recent landmark clues, and narrow PubMed deep-dive queries. It also collects Codex-subtask outputs and creates paper-identity normalization packets so informal system names can be mapped to official paper titles before verification. It does not certify papers. External DeepSeek/OpenAI-compatible mining is optional and should be used only when external data egress is explicitly approved.

Codex-subtask packet mode:

```bash
python $SKILL_DIR/scripts/draft_literature_candidate_miner.py packet \
  --draft-dir ./review-data/01_inputs/drafts_raw \
  --topic "<topic>" \
  --out-dir ./review-data/02_literature/ai_candidate_mining
```

Collect Codex-subtask CSV/markdown outputs:

```bash
python $SKILL_DIR/scripts/draft_literature_candidate_miner.py collect \
  --input-dir ./review-data/02_literature/ai_candidate_mining/subagent_results \
  --reference-candidates-csv ./review-data/02_literature/draft_assets/candidate_paper_clues.csv \
  --out-dir ./review-data/02_literature/ai_candidate_mining/subagent_collected
```

Create paper-identity normalization packets:

```bash
python $SKILL_DIR/scripts/draft_literature_candidate_miner.py normalize-packet \
  --candidate-csv ./review-data/02_literature/ai_candidate_mining/subagent_collected/subagent_hidden_candidate_clues.csv \
  --out-dir ./review-data/02_literature/ai_candidate_mining/identity_normalization \
  --topic "<topic>"
```

After Codex subtasks produce `normalized_identities.csv`, merge to verifier input:

```bash
python $SKILL_DIR/scripts/draft_literature_candidate_miner.py merge-normalized \
  --candidate-csv ./review-data/02_literature/ai_candidate_mining/subagent_collected/subagent_hidden_candidate_clues.csv \
  --normalized-csv ./review-data/02_literature/ai_candidate_mining/identity_normalization/normalized_identities.csv \
  --out-dir ./review-data/02_literature/ai_candidate_mining/identity_normalization/merged \
  --topic "<topic>"
```

Optional DeepSeek mining mode:

```bash
python $SKILL_DIR/scripts/draft_literature_candidate_miner.py mine \
  --project-dir . \
  --draft-dir ./review-data/01_inputs/drafts_raw \
  --topic "<topic>" \
  --out-dir ./review-data/02_literature/ai_candidate_mining \
  --max-workers 4
```

PubMed deep-dive from the AI-mined queries:

```bash
python $SKILL_DIR/scripts/draft_literature_candidate_miner.py pubmed \
  --tasks-csv ./review-data/02_literature/ai_candidate_mining/ai_pubmed_deep_dive_tasks.csv \
  --out-dir ./review-data/02_literature/ai_candidate_mining/pubmed_deep_dive \
  --email <email>
```

After screening PubMed abstract hits, convert eligible rows back into verifier input:

```bash
python $SKILL_DIR/scripts/draft_literature_candidate_miner.py pubmed-to-verifier \
  --abstracts-csv ./review-data/02_literature/ai_candidate_mining/pubmed_deep_dive/ai_pubmed_candidate_abstracts.csv \
  --tasks-csv ./review-data/02_literature/ai_candidate_mining/ai_pubmed_deep_dive_tasks.csv \
  --out-dir ./review-data/02_literature/ai_candidate_mining/pubmed_deep_dive/verifier_input
```

Outputs:

- `ai_candidate_paper_clues.csv`: AI-mined candidate clues from draft prose.
- `candidate_paper_clues_for_verification.csv`: verifier-compatible input for `draft_reference_verifier.py`.
- `subagent_collected/subagent_hidden_candidate_clues.csv`: structured candidates collected from Codex-subtask outputs; rows overlapping draft-native `candidate_paper_clues.csv` are marked as reference recovery rather than hidden-only.
- `identity_normalization/identity_normalization_packets/`: Codex-subtask paper identity normalization packets.
- `identity_normalization/merged/candidate_paper_clues_for_verification.csv`: verifier-compatible input after official-title normalization.
- `ai_pubmed_deep_dive_tasks.csv`: narrow PubMed queries linked to draft chunks.
- `human_ai_candidate_checkpoint.md`: human/Codex decisions for verify/search/delete/merge.
- `pubmed_deep_dive/ai_pubmed_candidate_abstracts.csv`: PubMed abstracts returned from the AI-mined queries.
- `pubmed_deep_dive/ai_pubmed_abstract_screening_packet.md`: packet for claim-fit screening.
- `pubmed_deep_dive/verifier_input/pubmed_abstract_candidates_for_verification.csv`: screened abstract hits converted back into deterministic verifier input.

Use this after draft memory, draft citation assets, and the chief reading checkpoint, and before deterministic verification. AI-mined rows are not evidence until metadata verification and Codex/human claim-fit screening pass.

## Script 0b: `scripts/draft_citation_assets.py`

Purpose: inspect draft-native citation assets before any supplemental recall. It cleans the reference section out of body text, maps in-text citation markers, separates non-academic sources, extracts claim-evidence clues, and writes strict supplemental-search tasks for later triage.

```bash
python $SKILL_DIR/scripts/draft_citation_assets.py build \
  --draft-dir ./review-data/01_inputs/drafts_raw \
  --topic "<topic>" \
  --out-dir ./review-data/02_literature/draft_assets
```

Outputs:

- `draft_citation_health.md`
- `draft_reference_inventory.csv`
- `draft_intext_citations.csv`
- `claim_evidence_map.csv`
- `candidate_paper_clues.csv`
- `non_academic_sources.csv`
- `uncited_references.csv`
- `unresolved_intext_markers.csv`
- `body_mentioned_systems.csv`: named systems/methods found in draft prose for subagent identity normalization; not verified references.
- `unmapped_claims.csv`: claim snippets without mapped reference numbers.
- `supplemental_recall_tasks.csv`
- `supplemental_recall_queries.txt`
- `cleaned_drafts/*.body.md`

Use this before any broad `paper_recall.py` run when AI-generated drafts exist.

## Script 0b-user: `scripts/pubmed_user_set_ingestor.py`

Purpose: parse user-supplied PubMed Summary/Abstract text exports into a separate screened candidate lane. These files are valuable because the user has already searched the topic in PubMed, but they are not automatically accepted references. They must still pass agent/human screening, deterministic verification, claim-fit adjudication, and the published-only final gate.

Typical inputs are PubMed-generated text files such as `abstract-<set>.txt` and `summary-<set>.txt` placed in the draft/input folder.

```bash
python $SKILL_DIR/scripts/pubmed_user_set_ingestor.py parse \
  --input ./review-data/01_inputs/drafts_raw/abstract-<set>.txt \
  --input ./review-data/01_inputs/drafts_raw/summary-<set>.txt \
  --topic "<topic>" \
  --set-name "<set-name>" \
  --out-dir ./review-data/02_literature/user_pubmed_sets/<set-name>
```

Outputs:

- `pubmed_user_set_records.csv`: parsed PubMed records with PMID, PMCID, DOI, title, journal, year, raw citation, and abstract when available.
- `pubmed_user_set_candidates.csv`: candidate rows in the same schema used by supplemental recall.
- `screening_queue.csv`: ranked rows for subagent/human screening.
- `agent_screening_packet.md` and `screening_packets/`: packet index and chunked screening packets.
- `user_pubmed_screening_guide.md`: screening policy for this seed lane.
- `agent_screening_template.csv`: decision template compatible with `supplemental_recall_screening.py collect-screening`.
- `pubmed_user_set_ingestion_summary.md` and `pubmed_user_set_manifest.json`: run summary and machine-readable manifest.

Recommended next command after agent/human decisions:

```bash
python $SKILL_DIR/scripts/supplemental_recall_screening.py collect-screening \
  --candidates-csv ./review-data/02_literature/user_pubmed_sets/<set-name>/pubmed_user_set_candidates.csv \
  --screening-csv ./review-data/02_literature/user_pubmed_sets/<set-name>/agent_screening_decisions.csv \
  --out-dir ./review-data/02_literature/user_pubmed_sets/<set-name>/screened
```

Only `screened/supplemental_candidates_for_verification.csv` should proceed to `draft_reference_verifier.py verify`. Keep rejected/off-frame rows in the seed-set folder for audit, not in the main pool.

For a user PubMed seed set, use the wrapper to create lane-specific verifier aliases and a verification handoff:

```bash
python $SKILL_DIR/scripts/pubmed_user_set_ingestor.py collect-screening \
  --set-dir ./review-data/02_literature/user_pubmed_sets/<set-name> \
  --screening-csv ./review-data/02_literature/user_pubmed_sets/<set-name>/agent_screening_decisions.csv \
  --verification-out-dir ./review-data/05_audit/user_pubmed_seed_verification
```

This writes `user_pubmed_candidates_for_verification.csv`, `user_pubmed_core_candidates_for_verification.csv`, `user_pubmed_framework_candidates_for_verification.csv`, and `user_pubmed_verification_handoff.md`.

Screening decisions should fill the extended claim-evidence fields: `argument_role`, `evidence_level`, `evidence_strength`, `fulltext_need`, `citation_role`, and `key_supported_claim`. These fields are carried into accepted-candidate outputs and verifier rationale so chief Codex can see why a row is being promoted.

## Script 0b-board: `scripts/literature_candidate_board.py`

Purpose: build a read-only chief-editor control surface across all candidate lanes. Use it after major candidate-producing stages, screening collection, and verification runs so Codex can see whether the limiting factor is candidate extraction, screening backlog, verifier-ready rows, unverified repair tasks, or full-text/citation export.

```bash
python $SKILL_DIR/scripts/literature_candidate_board.py build \
  --project-dir . \
  --out-dir ./review-data/02_literature/candidate_board
```

For real-project reruns, keep the board scoped. If earlier smoke tests or archived experiments live under the same review root, use repeatable path filters so old candidate lanes do not pollute the current board:

```bash
python $SKILL_DIR/scripts/literature_candidate_board.py build \
  --project-dir . \
  --out-dir ./review-data/02_literature/candidate_board \
  --exclude-path-contains ai_candidate_mining_test \
  --exclude-path-contains review-data/05_audit/old_smoke
```

Outputs:

- `literature_candidate_board.md`: human-readable board for chief Codex and user discussion.
- `literature_candidate_board.csv`: normalized row-level view of draft-native, user PubMed, AI-mined, argument expansion, supplemental recall, and verification lanes.
- `candidate_lane_summary.csv`: counts by source channel, stage, and next action.
- `screening_worklist.csv`: candidates that still need Codex subagent/human screening.
- `verification_worklist.csv`: screened or normalized rows ready for deterministic verification.
- `screening_worklist_deduped.csv` and `verification_worklist_deduped.csv`: unique-paper worklists for subagent assignment.
- `literature_candidate_board.json`: machine-readable manifest.

The board does not promote, reject, or edit source rows. It only makes the current literature asset state visible.

Generate Codex-subtask packets from the board:

```bash
python $SKILL_DIR/scripts/literature_candidate_board.py packets \
  --project-dir . \
  --board-dir ./review-data/02_literature/candidate_board \
  --out-dir ./review-data/02_literature/candidate_board/agent_packets \
  --packet-size 25
```

Outputs:

- `agent_packet_index.md`: packet assignment index for chief Codex.
- `candidate_board_curator_packet.md`: whole-board triage packet.
- `literature_screener_packets/*.md`: chunked screening packets built from `screening_worklist_deduped.csv`.
- `citation_verifier_packets/*.md`: chunked verification preflight packets built from `verification_worklist_deduped.csv`.
- `candidate_board_triage_template.csv`, `literature_screener_decision_template.csv`, and `citation_verifier_triage_template.csv`: schemas for agent returns.

After Codex subtasks or human screeners return CSV decisions, collect them into auditable next-step queues. The collector accepts both strict action labels and short natural-language `next_action` instructions from the curator packet; it still writes explicit verifier/repair/fulltext/rejection queues for the chief Codex to review.

```bash
python $SKILL_DIR/scripts/literature_candidate_board.py collect \
  --project-dir . \
  --board-dir ./review-data/02_literature/candidate_board \
  --input-dir ./review-data/02_literature/candidate_board/agent_results \
  --out-dir ./review-data/02_literature/candidate_board/collected
```

Decision collection outputs:

- `candidate_board_agent_decisions.csv`: nonblank agent decisions joined back to board metadata.
- `candidate_board_candidates_for_verification.csv`: deterministic-verifier input generated from screened inclusions and verifier preflight rows.
- `candidate_board_core_candidates_for_verification.csv` and `candidate_board_framework_candidates_for_verification.csv`: split queues for core biomedical evidence versus historical/method/governance/framework evidence.
- `candidate_board_repair_queue.csv`: rows needing identity repair, replacement, or deletion before any API verification.
- `candidate_board_fulltext_queue.csv`: rows whose next useful step is official citation export and full-text acquisition after verification.
- `candidate_board_manual_questions.csv`: human-in-the-loop decisions.
- `candidate_board_next_actions.md`: chief-editor summary and next commands.

This collection step still does not promote papers into the main pool. It only converts agent reasoning into verifier, repair, full-text, rejection, duplicate, and human-decision queues.

## Script 0c: `scripts/draft_reference_verifier.py`

Purpose: verify draft-native and AI-mined paper clues through multi-signal identity resolution. PMID/DOI values are strong clues, not truth: when an identifier resolves to a title that conflicts with the draft title, the verifier falls back to title/bibliographic search and records the mismatch. PubMed verification fetches and ranks multiple returned PMID hits per PMID/DOI/title attempt rather than accepting the first result; formal non-correction journal records are preferred over preprints and correction notices when titles match. Title search now includes system-name deep dives and compact Title/Abstract query variants, so acronym-only clues such as `CRISPR-GPT` or `AI co-scientist` can resolve to formal PubMed records even when the draft title is incomplete. It also uses Crossref/OpenAlex and official publisher/conference URLs when present, including DOI inference from publisher URLs such as Nature/Springer article paths (`nature.com/articles/s41586-...` -> `10.1038/s41586-...`). URL-derived DOI evidence is guarded by Crossref/OpenAlex title checks; if the DOI resolves to a different paper, the candidate is rejected or routed to repair instead of being promoted. It creates an importable verified-paper CSV, keeps preprints/unverified rows out of the main pool, rejects correction/erratum records as target-paper substitutes, and suppresses preprint duplicates when a formal published version is available.

```bash
python $SKILL_DIR/scripts/draft_reference_verifier.py verify \
  --candidate-csv ./review-data/02_literature/draft_assets/candidate_paper_clues.csv \
  --out-dir ./review-data/05_audit/draft_reference_verification \
  --email <email> \
  --pubmed-retmax 10 \
  --workers 4 \
  --resume
```

Outputs:

- `verified_draft_papers.csv`: verified by APIs or official publisher/conference pages, but still run LLM/subagent adjudication before import.
- `preprint_leads.csv`: search/background leads only by default.
- `version_suppressed_duplicate_preprints.csv`: preprints that match an already verified formal paper; do not cite separately.
- `rejected_or_unverified_draft_sources.csv`: delete, replace, or ask the user.
- `all_draft_reference_verification.csv`
- `draft_reference_verification_report.md`
- `verification_deduplication_report.md`: written when interrupted/resumed runs collapse duplicate rows.
- `verification_progress.jsonl`: per-record checkpoint stream for long or interrupted runs.
- `verification_query_trace.csv` and `verification_query_trace.jsonl`: PubMed query audit trail showing PMID/DOI/title/system-name query modes, returned IDs, title-mismatch exclusions, ranked candidates, and accepted hits.

Use `--offline` only for extraction/testing; offline rows are not citable. The verifier writes incremental canonical CSV outputs by default, so a timeout no longer loses all progress. Rerun with `--resume` after interruption. Use `--pubmed-retmax 10-20` when a query may return preprint/formal/correction variants of the same paper. Use `--workers 2-4` for moderate parallel API calls; use `--workers 1 --timeout 40 --resume` on unstable networks or when PubMed rate limits are a concern.

When a paper looks findable manually but the verifier misses it, inspect `verification_query_trace.csv` before changing strategy. It shows whether the script searched by DOI, PMID, exact title, system name, compact title tokens, or broad bibliographic text; it also shows which returned PubMed records were rejected for title mismatch.

To verify AI-mined clues:

```bash
python $SKILL_DIR/scripts/draft_reference_verifier.py verify \
  --candidate-csv ./review-data/02_literature/ai_candidate_mining/identity_normalization/merged/candidate_paper_clues_for_verification.csv \
  --out-dir ./review-data/05_audit/ai_candidate_reference_verification \
  --email <email> \
  --workers 4 \
  --resume
```

Also verify literature-discovery and screened PubMed deep-dive candidates:

```bash
python $SKILL_DIR/scripts/draft_reference_verifier.py verify \
  --candidate-csv ./review-data/02_literature/literature_discovery/collected/candidate_paper_clues_for_verification.csv \
  --out-dir ./review-data/05_audit/literature_discovery_reference_verification \
  --email <email> \
  --workers 4 \
  --resume

python $SKILL_DIR/scripts/draft_reference_verifier.py verify \
  --candidate-csv ./review-data/02_literature/ai_candidate_mining/pubmed_deep_dive/verifier_input/pubmed_abstract_candidates_for_verification.csv \
  --out-dir ./review-data/05_audit/pubmed_deep_dive_reference_verification \
  --email <email> \
  --workers 4 \
  --resume
```

Each lane must then pass `draft_source_adjudicator.py build-cases` and `merge` before it can enter the verified union.

## Script 0c-union: `scripts/literature_pool.py merge-verified`

Purpose: merge accepted verified papers from all draft-derived lanes into one deduped union before official citation export and pool import. This prevents hidden/subagent recovery from staying in a side channel and prevents duplicate rows from being mistaken for unique papers.

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

Outputs:

- `verified_papers_union.csv`: deduped accepted verified papers from all lanes.
- `verified_papers_union_duplicates.csv`: collapsed duplicate rows and retained identity.
- `verified_papers_union_report.md`: input counts, unique-paper counts, and lane counts.

Use this union as the accepted draft-derived source of truth. Importing separate branch CSVs directly can undercount hidden recoveries, overcount duplicates, or pollute provenance.

## Script 0d: `scripts/official_citation_exporter.py`

Purpose: after identity verification and citation adjudication, fetch official citation-manager exports rather than reusing draft reference strings. This is where "real citation format" is produced: PubMed NBIB/MEDLINE, Crossref/DOI BibTeX/RIS, DOI content-negotiated bibliography text, arXiv BibTeX, or a manual official-export handoff when no export is available.

The exporter performs a final title-consistency guard on structured official exports. If a DOI returns BibTeX/RIS/NBIB for a different title, the automatic export is cleared and the row is routed to manual citation checking. This prevents a wrong DOI that slipped through earlier stages from becoming a final reference.

```bash
python $SKILL_DIR/scripts/official_citation_exporter.py export \
  --verified-csv ./review-data/05_audit/verified_papers_union/verified_papers_union.csv \
  --out-dir ./review-data/05_audit/official_citations \
  --style vancouver \
  --email <email> \
  --dedupe-by doi,pmid,title
```

Outputs:

- `official_citations.csv`: per-paper official export status, source, BibTeX/RIS/NBIB, and formatted bibliography text when available.
- `official_citations.bib`: combined BibTeX from Crossref/DOI/arXiv official exports.
- `official_citations.ris`: combined RIS from Crossref/DOI official exports.
- `official_citations.nbib`: combined PubMed NBIB/MEDLINE exports.
- `official_citations_<style>.txt`: DOI content-negotiated formatted bibliography text.
- `official_citation_export_report.md`: manual official-export handoff list for rows without machine-readable official export.
- `official_citation_duplicate_report.csv`: duplicate input rows collapsed before export.

Use `--offline` only for dry-run validation; it will not create official exports. Use `--allow-derived` only for an internal working bibliography, because derived strings are not a substitute for official citation-manager export. Report `input_rows`, `unique_rows`, and `duplicates_collapsed` separately.

## Script 0e: `scripts/draft_source_adjudicator.py`

Purpose: create Codex-subtask packets to adjudicate API-verified draft sources before import. It does not verify existence; it judges title match, preprint-to-published risk, claim-source fit, inclusion/exclusion rationale, and human confirmation items from supplied metadata only. External DeepSeek/OpenAI-compatible adjudication is optional and should be used only when external data egress is explicitly approved.

Codex-subtask packet mode:

```bash
python $SKILL_DIR/scripts/draft_source_adjudicator.py build-cases \
  --verification-csv ./review-data/05_audit/draft_reference_verification/all_draft_reference_verification.csv \
  --claim-map ./review-data/02_literature/draft_assets/claim_evidence_map.csv \
  --out-dir ./review-data/05_audit/draft_source_adjudication \
  --topic "<topic>" \
  --batch-size 12
```

Optional DeepSeek parallel mode:

```bash
python $SKILL_DIR/scripts/draft_source_adjudicator.py adjudicate \
  --project-dir . \
  --verification-csv ./review-data/05_audit/draft_reference_verification/all_draft_reference_verification.csv \
  --claim-map ./review-data/02_literature/draft_assets/claim_evidence_map.csv \
  --out-dir ./review-data/05_audit/draft_source_adjudication \
  --topic "<topic>" \
  --batch-size 12 \
  --max-workers 4 \
  --max-retries 3
```

Merge gate:

```bash
python $SKILL_DIR/scripts/draft_source_adjudicator.py merge \
  --verification-csv ./review-data/05_audit/draft_reference_verification/all_draft_reference_verification.csv \
  --adjudication-csv ./review-data/05_audit/draft_source_adjudication/llm_citation_adjudication.csv \
  --out-dir ./review-data/05_audit/draft_source_adjudication
```

Outputs:

- `accepted_verified_draft_papers.csv`: lane-level accepted papers; merge all lanes with `literature_pool.py merge-verified` before official export or pool import.
- `rejected_or_demoted_sources.csv`
- `claim_source_fit.csv`
- `preprint_resolution.csv`
- `human_confirmation_queue.md`

## Script 0f: `scripts/draft_logic_framework.py`

Purpose: read one or more AI-generated review drafts and produce a discussion-ready thesis-to-outline framework before drafting.

Directory mode:

```bash
python $SKILL_DIR/scripts/draft_logic_framework.py \
  --draft-dir ./review-data/01_inputs/drafts_raw \
  --topic "<topic>" \
  --target-journal "<target-journal>" \
  --out-dir ./review-data/03_framework/logic_framework
```

Individual draft mode:

```bash
python $SKILL_DIR/scripts/draft_logic_framework.py \
  --draft ./gpt-1.md \
  --draft ./Gemini\ 1.docx \
  --out-dir ./review-data/03_framework/logic_framework
```

Outputs:

- `draft_logic_framework.md`: architecture warnings, recurring frames, thematic weights, and 2-3 candidate section spines.
- `argument_evidence_map.csv`: claim-like sentences with citation markers for later verification.
- `framework_evidence_blackboard.md`: shared state for thesis candidates, evidence claims, gaps, literature tasks, and pending decisions.
- `literature_search_tasks.csv` / `.txt`: recall tasks derived from draft claims, reference leads, and heading themes.
- `framework_material_passport.json`: raw draft artifacts, framework artifacts, verification status, and pending architecture checkpoint.
- `abstract_diagnosis.md`: flags abstracts that read like search logs or internal audit records.
- `user_alignment_questions.md`: questions to discuss with the user before manuscript drafting.
- `draft_inventory.json` and `logic_framework_summary.json`: machine-readable extraction records.

Use this after `draft_memory.py` and `draft_citation_assets.py`, and before manuscript drafting when the user supplies multiple Deep Research drafts.

## Script 0g: `scripts/argument_literature_expander.py`

Purpose: convert the draft-derived argument map into bounded supplemental literature tasks, fetch PubMed metadata/abstracts by claim, and write screening packets before any new paper enters the main pool.

Plan tasks after the chief Codex/user framework checkpoint:

```bash
python $SKILL_DIR/scripts/argument_literature_expander.py plan \
  --framework-dir ./review-data/03_framework/logic_framework \
  --topic "<topic>" \
  --out-dir ./review-data/02_literature/argument_literature_expansion
```

Run PubMed abstract retrieval with incremental progress and resume support:

```bash
python $SKILL_DIR/scripts/argument_literature_expander.py pubmed \
  --tasks-csv ./review-data/02_literature/argument_literature_expansion/argument_literature_tasks.csv \
  --out-dir ./review-data/02_literature/argument_literature_expansion \
  --email <email> \
  --workers 2 \
  --resume
```

Outputs:

- `argument_literature_tasks.csv` / `.json`: claim-linked retrieval tasks.
- `human_argument_literature_checkpoint.md`: user/Codex decision sheet for keeping, merging, deleting, or narrowing tasks.
- `pubmed_argument_candidates.csv` / `.jsonl`: candidates with PMID, DOI, journal, publication type, abstract, linked claim, evidence need, and fit hint.
- `pubmed_task_progress.jsonl` and `pubmed_candidate_progress.jsonl`: incremental checkpoints for long runs.
- `abstract_screening_packet.md`: packet for Codex subtasks or human screening; DeepSeek is optional when approved.
- `claim_candidate_links.csv`: claim-to-paper link worksheet.
- `supplemental_import_and_fulltext_next_steps.md`: commands for importing screened candidates and fetching full text.

Default policy: import these rows to `review-data/02_literature/supplemental_pool`, not the main pool. Promote only after claim-fit screening, topic filtering, published-only final gate, and full-text readiness for detailed claims.

## Script 0h: `scripts/supplemental_recall_screening.py`

Purpose: build a larger candidate pool after the draft-derived framework is understood. It uses `body_mentioned_systems.csv`, `unmapped_claims.csv`, literature-discovery API tasks, and topic/framework terms to create broad-but-bounded PubMed, Crossref/OpenAlex, arXiv/OpenReview, paper-search, publisher, and human recall tasks. It then seed-locks already verified/curated draft assets, quarantines obvious metadata noise, and creates ranked chunked agent screening packets. Screened inclusions are converted back into deterministic verifier input; broad-recall hits never enter the main pool directly.

Evidence lanes:

- `core_biomedical_system_evidence`: papers that can support the main evidence table for biomedical/medical self-evolving or agentic systems.
- `framework_verified`: conceptual or taxonomy papers that shape the review architecture.
- `historical_foundation`: earlier AI-scientist, robot-scientist, closed-loop, ReAct/Reflexion-style method lineage papers.
- `method_foundation`: general agent/memory/reflection/tool-use methods that support mechanism explanation but are not biomedical evidence.
- `governance_background`: safety, evaluation, reporting, regulation, and clinical governance context.
- `background_hold`: interesting background that is not citable until a human/Codex decision promotes it.

Framework, historical, method, and governance records still require formal source verification before citation; they are simply kept out of the core biomedical-system evidence table.

Plan broad supplemental recall:

```bash
python $SKILL_DIR/scripts/supplemental_recall_screening.py plan \
  --topic "<topic>" \
  --draft-assets-dir ./review-data/02_literature/draft_assets \
  --literature-discovery-dir ./review-data/02_literature/literature_discovery/collected \
  --out-dir ./review-data/02_literature/supplemental_recall_screening \
  --max-tasks 160
```

Run PubMed recall for PubMed-route tasks:

```bash
python $SKILL_DIR/scripts/supplemental_recall_screening.py pubmed \
  --tasks-csv ./review-data/02_literature/supplemental_recall_screening/supplemental_recall_tasks.csv \
  --out-dir ./review-data/02_literature/supplemental_recall_screening/pubmed \
  --max-results 20 \
  --workers 2 \
  --resume \
  --email <email>
```

Run other route query files with existing recall scripts:

```bash
python $SKILL_DIR/scripts/crossref_openalex_recall.py \
  --query-file ./review-data/02_literature/supplemental_recall_screening/queries/crossref_openalex_queries.txt \
  --out-dir ./review-data/02_literature/supplemental_recall_screening/crossref_openalex

python $SKILL_DIR/scripts/arxiv_openreview_recall.py \
  --query-file ./review-data/02_literature/supplemental_recall_screening/queries/arxiv_openreview_queries.txt \
  --out-dir ./review-data/02_literature/supplemental_recall_screening/arxiv_openreview
```

Collect candidates and create agent screening packets:

```bash
python $SKILL_DIR/scripts/supplemental_recall_screening.py collect-candidates \
  --topic "<topic>" \
  --seed-csv ./review-data/05_audit/draft_source_adjudication/accepted_verified_draft_papers_curated.csv \
  --recall-dir ./review-data/02_literature/supplemental_recall_screening/pubmed \
  --recall-dir ./review-data/02_literature/supplemental_recall_screening/crossref_openalex \
  --recall-dir ./review-data/02_literature/supplemental_recall_screening/arxiv_openreview \
  --out-dir ./review-data/02_literature/supplemental_recall_screening/collected \
  --packet-size 40 \
  --min-screening-score 3 \
  --max-screening-candidates 240
```

After Codex subagents or humans fill `agent_screening_decisions.csv`, collect screened inclusions:

```bash
python $SKILL_DIR/scripts/supplemental_recall_screening.py collect-screening \
  --candidates-csv ./review-data/02_literature/supplemental_recall_screening/collected/supplemental_recall_candidates.csv \
  --screening-csv ./review-data/02_literature/supplemental_recall_screening/collected/agent_screening_decisions.csv \
  --out-dir ./review-data/02_literature/supplemental_recall_screening/screened
```

Use these screening decisions for non-core but citable architecture papers: `include_framework`, `include_historical_foundation`, `include_method_foundation`, and `include_governance_background`. Legacy `include_background` rows are held by default unless their rationale/next action clearly indicates framework, historical, method, governance, or lineage use.

Outputs:

- `supplemental_recall_tasks.csv`: large, route-specific recall task list.
- `queries/*.txt`: route-specific query files.
- `pubmed/pubmed_supplemental_candidates.csv`: PubMed candidates from broad recall.
- `collected/supplemental_recall_candidates.csv`: deduped candidates across recall routes.
- `collected/seed_locked_candidates.csv`: verified/curated draft-derived assets that must not be lost during broad recall.
- `collected/quarantined_candidates.csv`: metadata artifacts and likely noise; sample or rescue only when needed.
- `collected/screening_queue.csv`: ranked first-pass queue for Codex subagents.
- `collected/screening_packets/`: chunked packets, usually 40-70 candidates each.
- `collected/screening_packet_index.csv`: packet assignment index for subagents.
- `collected/agent_screening_packet.md`: packet for Codex subagents/DeepSeek/human screening.
- `screened/supplemental_candidates_for_verification.csv`: screened inclusions ready for `draft_reference_verifier.py`.
- `screened/core_candidates_for_verification.csv`: core biomedical-system evidence only.
- `screened/framework_candidates_for_verification.csv`: framework, historical, method, and governance papers that still need formal source verification.
- `screened/background_hold_candidates.csv`: background records not yet citable.

Audit the broad recall pool before deciding whether to screen more packets:

```bash
python $SKILL_DIR/scripts/supplemental_recall_screening.py audit-pool \
  --collection-dir ./review-data/02_literature/supplemental_recall_screening/collected \
  --screened-dir ./review-data/02_literature/supplemental_recall_screening/screened \
  --verification-dir ./review-data/05_audit/supplemental_reference_verification \
  --out-dir ./review-data/02_literature/supplemental_recall_screening/collected \
  --top-n 40
```

Audit outputs:

- `recall_pool_audit.md`: explains how the pool formed, route/lane/priority counts, screened coverage, verification outcomes, and next backlog sample.
- `screening_backlog.csv`: queue rows not yet covered by screening decisions.
- `deferred_not_queued_candidates.csv`: non-quarantined candidates left outside the current packet set because of `--max-screening-candidates`.
- `quarantine_rescue_sample.csv`: lower-priority records to sample for possible rescue.

Run agent-supervised recall after audit or after every 1-2 screening packets:

```bash
python $SKILL_DIR/scripts/supplemental_recall_screening.py supervise-pool \
  --collection-dir ./review-data/02_literature/supplemental_recall_screening/collected \
  --screened-dir ./review-data/02_literature/supplemental_recall_screening/screened \
  --out-dir ./review-data/02_literature/supplemental_recall_screening/recall_supervision \
  --topic "<topic>" \
  --backlog-n 80 \
  --deferred-n 80 \
  --quarantine-n 40 \
  --max-pubmed-tasks 60 \
  --strategic-pubmed-tasks 12
```

Supervision outputs:

- `recall_supervision_packet.md`: Codex/subagent packet that asks the agent to inspect pool health, query drift, PubMed underuse, framework/history lanes, and rescue candidates.
- `recall_supervision_seed_decisions.csv`: prefilled supervision rows that the subagent can edit, promote, reject, or refine.
- `agent_supervised_pubmed_tasks.csv`: bounded PubMed tasks generated from supervised candidates plus a small strategic PubMed-deep-dive supplement when candidate-derived tasks under-cover biomedical-agent evidence.
- `recall_supervision_template.csv`: blank decision template.
- `recall_supervision_manifest.json`: task quotas, task group counts, and coverage warnings. Treat deep-dive coverage below 30% or formal-version searches above 45% as a queue-design problem, not a literature conclusion.

Collect edited subagent supervision decisions:

```bash
python $SKILL_DIR/scripts/supplemental_recall_screening.py collect-supervision \
  --supervision-csv ./review-data/02_literature/supplemental_recall_screening/recall_supervision/agent_supervision_decisions.csv \
  --candidates-csv ./review-data/02_literature/supplemental_recall_screening/collected/supplemental_recall_candidates.csv \
  --out-dir ./review-data/02_literature/supplemental_recall_screening/recall_supervision/collected \
  --topic "<topic>" \
  --max-pubmed-tasks 80 \
  --strategic-pubmed-tasks 12
```

Optional quota overrides:

```bash
  --deep-dive-task-quota 30 \
  --formal-version-task-quota 15 \
  --framework-task-quota 12 \
  --governance-task-quota 5 \
  --rescue-task-quota 5
```

Then run PubMed recall on `agent_supervised_pubmed_tasks.csv` and merge the returned candidates back through `collect-candidates`. This closes the loop: Codex/subagents read the pool, refine recall, PubMed retrieves biomedical metadata/abstracts, and deterministic verification checks formal citation identity. Formal-version searches should not consume the whole PubMed budget; rows with DOI/title clues should also go through Crossref/OpenAlex/publisher verification.

PubMed zero-hit policy: a zero-hit query is a route signal, not exclusion. The agent should either relax the query, route to Crossref/OpenAlex/publisher, or mark the case for human lookup. DOI/AID PubMed searches must quote the DOI, for example `"10.xxxx/xxxxx"[AID]`; unquoted DOI strings can be tokenized and create false hits. Returned PubMed abstracts are screening material only; broad strategic hits without an agent/LLM/autonomous title signal are downgraded and must not be promoted without subagent/human claim-fit reasoning.

Use this when the draft-derived verified union is too small for the target review, or when the framework needs historical foundations, current landmarks, counterevidence, governance, or method/evaluation papers beyond the draft references. This is the intended "large recall, strict agent screening" layer.

## Script 1: `scripts/paper_recall.py`

Purpose: call the `paper-search` CLI, run one or more query variants, merge source outputs, deduplicate, and save review-ready files.

The script uses the CLI because Codex skills cannot assume that the MCP server is exposed as a callable tool in every session. The CLI wraps the same `paper-search-mcp` library and returns JSON for search/download operations. If an MCP tool is available in the current session, prefer the native MCP `search_papers` tool for interactive calls and still use this script to persist results.

When AI-generated drafts exist, do not use this as the first literature step. Run `draft_citation_assets.py` and `draft_reference_verifier.py` first, then use `paper_recall.py` only for confirmed supplemental gaps and write results to `review-data/02_literature/supplemental_pool`.

Basic command:

```bash
python $SKILL_DIR/scripts/paper_recall.py \
  --query "<topic search query>" \
  --sources openalex,crossref,semantic,pubmed \
  --max-results 10 \
  --out-dir ./review-work/recall_runs/general
```

Multiple query variants:

```bash
python $SKILL_DIR/scripts/paper_recall.py \
  --query "<primary topic query>" \
  --query "<mechanism or method query>" \
  --query "<clinical or translational query>" \
  --sources openalex,crossref,semantic,pubmed,pmc,europepmc \
  --max-results 10 \
  --year <start-year>-<current-year> \
  --out-dir ./review-work/recall_runs/general
```

Use a cloned `openags/paper-search-mcp` repository:

```bash
python $SKILL_DIR/scripts/paper_recall.py \
  --query "query" \
  --paper-search-repo D:/path/to/paper-search-mcp \
  --sources arxiv,semantic,crossref \
  --out-dir ./review-work/recall_runs/general
```

Outputs:

- `search_log.json`: commands, query variants, source counts, errors, raw/deduped totals
- `papers.jsonl`: normalized deduplicated records
- `papers.csv`: spreadsheet-friendly normalized records
- `screening.csv`: initial screening worksheet
- `evidence_matrix.md`: compact markdown evidence table
- `run_summary.md`: human-readable recall summary

## Script 1b: `scripts/pubmed_recall.py`

Purpose: fallback recall through NCBI PubMed E-utilities when `paper-search` is unavailable.

```bash
python $SKILL_DIR/scripts/pubmed_recall.py \
  --query "<topic search query>" \
  --mindate 2020 \
  --maxdate <current-year> \
  --max-results 20 \
  --out-dir ./review-work/pubmed_recall \
  --email <email>
```

Use `--query-file` for user/LLM-approved supplemental gap queries such as `draft_assets/supplemental_recall_queries.txt`. Outputs match `paper_recall.py` enough to import into `literature_pool.py`.

## Script 1c: `scripts/crossref_openalex_recall.py`

Purpose: verify DOI/title metadata and find published versions through Crossref and OpenAlex.

```bash
python $SKILL_DIR/scripts/crossref_openalex_recall.py \
  --query "<paper title or metadata query>" \
  --doi <doi-if-known> \
  --sources crossref,openalex \
  --max-results 5 \
  --out-dir ./review-work/metadata_recall \
  --email <email>
```

Outputs match the recall import schema.

## Script 1d: `scripts/arxiv_openreview_recall.py`

Purpose: discover preprints and OpenReview submissions that may not appear in PubMed or Crossref yet.

```bash
python $SKILL_DIR/scripts/arxiv_openreview_recall.py \
  --query "<preprint title or project name>" \
  --arxiv-id <arxiv-id-if-known> \
  --openreview-id <openreview-id-if-known> \
  --sources arxiv,openreview \
  --max-results 5 \
  --out-dir ./review-work/preprint_recall
```

Use these results as leads. The final citation gate demotes arXiv/OpenReview-only records unless the user explicitly allows preprints.

## Script 2: `scripts/review_library.py`

Purpose: keep one long-lived review library across multiple recall runs and user-provided corpora.

Initialize:

```bash
python $SKILL_DIR/scripts/review_library.py init \
  --library ./review-work/library.json
```

Import a recall run:

```bash
python $SKILL_DIR/scripts/review_library.py import-recall \
  --library ./review-work/library.json \
  --recall-dir ./review-work/recall_runs/general
```

Import a user CSV:

```bash
python $SKILL_DIR/scripts/review_library.py import-csv \
  --library ./review-work/library.json \
  --csv ./user-corpus.csv
```

Screen one paper:

```bash
python $SKILL_DIR/scripts/review_library.py screen \
  --library ./review-work/library.json \
  --key doi-10-0000-example \
  --status include \
  --reason "Directly supports the review's inclusion criteria"
```

Update evidence fields:

```bash
python $SKILL_DIR/scripts/review_library.py update \
  --library ./review-work/library.json \
  --key doi-10-0000-example \
  --set "claim_supported=<specific claim supported by this source>" \
  --set "limitations=<source-specific limitation>" \
  --set "use_in_review=<section or display item>"
```

Export matrix and audit:

```bash
python $SKILL_DIR/scripts/review_library.py export-matrix \
  --library ./review-work/library.json \
  --out ./review-work/evidence_matrix.md \
  --all

python $SKILL_DIR/scripts/review_library.py audit \
  --library ./review-work/library.json \
  --out ./review-work/library_audit.md
```

## Direct MCP Tool Shape

When `paper-search-mcp` is actually configured as an MCP server in the current client, use its tool names directly:

- `search_papers(query, max_results_per_source, sources, year)`
- Source-specific search tools such as `search_arxiv`, `search_pubmed`, `search_semantic`, `search_crossref`, `search_openalex`, `search_pmc`, and `search_europepmc`
- Source-specific `download_*` and `read_*_paper` tools when full text is needed
- `download_with_fallback` when a DOI/title is available and OA-first fallback retrieval is appropriate

Persist MCP outputs by writing them into the same library schema used by `review_library.py`: title, authors, year, source, paper ID, DOI, URL/PDF URL, abstract, recall query, screening status, claim supported, limitations, and use in review.

## Screening Statuses

Allowed statuses:

- `unscreened`: not evaluated yet
- `include`: core evidence
- `exclude`: fails criteria
- `maybe`: needs abstract/full-text inspection
- `background`: useful context, not core evidence
- `seminal`: foundational work
- `recent`: important new development
- `method`: reporting standard, review method, or technical method source

## Practical Pattern

1. If drafts exist, run `draft_memory.py`, `draft_citation_assets.py`, chief Codex reading/provisional claim map, `draft_literature_discovery_orchestrator.py plan`, Codex-subtask discovery, `collect`, optional `draft_literature_candidate_miner.py packet`, `normalize-packet`, `merge-normalized`, `draft_reference_verifier.py`, and `draft_source_adjudicator.py build-cases`.
2. Merge all accepted verified lanes with `literature_pool.py merge-verified`, export official citations from `verified_papers_union.csv`, and import only that union into the main pool or `library.json`.
3. Discuss framework, claim gaps, and citation risks with the user.
4. Run strict supplemental recall only for confirmed gaps; keep those records in `supplemental_pool` until screened.
5. Screen records in batches and update evidence fields only for included records.
6. Export evidence matrix and literature cards.
7. Run final-gate and claim-level audit before drafting; after drafting, run deterministic citation sequencing before DOCX/PDF export.

## Script 3: `scripts/literature_pool.py`

Purpose: manage the progressive-disclosure literature memory layer: `review-data/02_literature/pool` (`文献池` legacy alias), selected citation-pool records, downloaded PDFs/full text, and literature cards.

Core commands:

```bash
python $SKILL_DIR/scripts/literature_pool.py init --pool-dir ./review-data/02_literature/pool
python $SKILL_DIR/scripts/literature_pool.py init --pool-dir ./review-data/02_literature/supplemental_pool
python $SKILL_DIR/scripts/literature_pool.py merge-verified --csv ./review-data/05_audit/draft_source_adjudication/accepted_verified_draft_papers.csv --csv ./review-data/05_audit/literature_discovery_source_adjudication/accepted_verified_draft_papers.csv --csv ./review-data/05_audit/ai_candidate_source_adjudication/accepted_verified_draft_papers.csv --out-dir ./review-data/05_audit/verified_papers_union --origin draft-derived-verified-union --skip-missing
python $SKILL_DIR/scripts/literature_pool.py import-csv --pool-dir ./review-data/02_literature/pool --csv ./review-data/05_audit/verified_papers_union/verified_papers_union.csv --origin draft-derived-verified-union
python $SKILL_DIR/scripts/literature_pool.py import-recall --pool-dir ./review-data/02_literature/supplemental_pool --recall-dir ./review-work/recall_runs/strict_supplemental --origin strict-supplemental-gap-recall
python $SKILL_DIR/scripts/literature_pool.py topic-filter --pool-dir ./review-data/02_literature/pool --topic "<topic>" --include-keywords "<keyword-1>,<keyword-2>,<domain-term>,<method-term>" --out ./review-data/02_literature/pool/logs/topic_filter_report.csv --apply
python $SKILL_DIR/scripts/literature_pool.py export-candidates --pool-dir ./review-data/02_literature/pool --out ./review-data/02_literature/pool/contexts/candidate_packet.md
python $SKILL_DIR/scripts/literature_pool.py decide --pool-dir ./review-data/02_literature/pool --key doi-10-0000-example --status citation_pool --score 5 --rationale "Core evidence"
python $SKILL_DIR/scripts/literature_pool.py download --pool-dir ./review-data/02_literature/pool --status citation_pool,seminal,method,recent
python $SKILL_DIR/scripts/literature_pool.py build-cards --pool-dir ./review-data/02_literature/pool
python $SKILL_DIR/scripts/literature_pool.py export-context --pool-dir ./review-data/02_literature/pool --out ./review-data/02_literature/pool/contexts/citation_context.md
python $SKILL_DIR/scripts/literature_pool.py final-gate --pool-dir ./review-data/02_literature/pool --out ./review-data/02_literature/pool/contexts/final_citation_gate.md --apply
python $SKILL_DIR/scripts/literature_pool.py audit --pool-dir ./review-data/02_literature/pool --out ./review-data/02_literature/pool/contexts/pool_audit.md
```

Read `references/literature-pool-workflow.md` for the full download-and-card workflow.

## Script 4: `scripts/draft_source_audit.py`

Purpose: process GPT/Gemini/Deep Research drafts, extract framework headings and claim-like sentences, separate paper-like references from webpages/blogs, and verify papers through Crossref/OpenAlex.

```bash
python $SKILL_DIR/scripts/draft_source_audit.py \
  --draft ./gpt_deep_research.md \
  --draft ./gemini_deep_research.md \
  --out-dir ./review-data/05_audit/draft_source_audit \
  --mailto <email>
```

Use `--offline` to extract without API verification. Import `verified_papers.csv` into the canonical literature pool with `literature_pool.py import-csv`; keep `all_reference_audit.csv` and `web_sources.csv` for source-risk review.

## Script 5: `scripts/evidence_fetch.py` and `scripts/fulltext_manager.py`

Purpose: save abstracts/full text for selected citation-pool records, prefer Europe PMC fullTextXML, parse user/PDF files, and create a human handoff list for inaccessible papers.

Europe PMC fullTextXML first:

```bash
python $SKILL_DIR/scripts/fulltext_manager.py fetch-europepmc \
  --pool-dir ./review-data/02_literature/pool \
  --status citation_pool,seminal,method,recent
```

```bash
python $SKILL_DIR/scripts/evidence_fetch.py \
  --pool-dir ./review-data/02_literature/pool \
  --status citation_pool,seminal,method,recent
```

Then audit and request user-supplied full text where needed:

```bash
python $SKILL_DIR/scripts/fulltext_manager.py audit \
  --pool-dir ./review-data/02_literature/pool
```

If `review-data/02_literature/pool/contexts/needs_user_fulltext.md` lists missing papers, ask the user to place files in `review-data/01_inputs/user_fulltext`, then run:

```bash
python $SKILL_DIR/scripts/fulltext_manager.py import \
  --pool-dir ./review-data/02_literature/pool \
  --source-dir ./review-data/01_inputs/user_fulltext
```

For PDFs that need OCR, table, or formula parsing:

```bash
python $SKILL_DIR/scripts/fulltext_manager.py mineru-agent \
  --pool-dir ./review-data/02_literature/pool \
  --language ch \
  --is-ocr \
  --enable-table \
  --enable-formula
```

Outputs are stored in `review-data/02_literature/pool/evidence/`, `logs/`, `contexts/needs_user_download.md`, `contexts/needs_user_fulltext.md`, and `contexts/fulltext_audit.md`.

## Script 6: `scripts/structured_lit_rag.py`

Purpose: build a small TreeSearch-inspired SQLite FTS5 index over cards, abstracts, and fetched full text.

```bash
python $SKILL_DIR/scripts/structured_lit_rag.py build --pool-dir ./review-data/02_literature/pool
python $SKILL_DIR/scripts/structured_lit_rag.py search \
  --db ./review-data/02_literature/pool/indexes/lit_rag.sqlite \
  --query "external validation prospective dataset"
```

If `pytreesearch` is installed, prefer real TreeSearch for richer structure-aware retrieval. Use this script as the no-dependency fallback.

## Script 7: `scripts/display_item_planner.py`

Purpose: create figure, table, and box blueprints from the framework and literature state before drafting. Tables are evidence-first and must be filled from verified literature, cards, full text, RAG, or explicit user decisions. Figures are concept-first: the script creates blueprints, GPT image-ready prompts, generation queues, and post-generation audit checklists, but generated images remain discussion drafts until human and evidence review pass.

```bash
python $SKILL_DIR/scripts/display_item_planner.py plan \
  --project-dir . \
  --framework-dir ./review-data/03_framework/logic_framework \
  --pool-dir ./review-data/02_literature/pool \
  --rag-db ./review-data/02_literature/pool/indexes/lit_rag.sqlite \
  --out-dir ./review-data/03_framework/display_items \
  --target-journal "Nature Reviews-style journal"
```

For evidence review:

```bash
python $SKILL_DIR/scripts/display_item_planner.py evidence-pack \
  --display-items ./review-data/03_framework/display_items/display_items.json \
  --item all \
  --rag-db ./review-data/02_literature/pool/indexes/lit_rag.sqlite \
  --out ./review-data/03_framework/display_items/display_item_evidence_pack.md
```

For conceptual image prompts:

```bash
python $SKILL_DIR/scripts/display_item_planner.py prompt-pack \
  --display-items ./review-data/03_framework/display_items/display_items.json \
  --item F1,F2,F3 \
  --out ./review-data/03_framework/display_items/reviewed_figure_prompt_pack.md
```

The `plan` command also writes:

- `gpt_image_generation_queue.csv`: per-figure prompt, aspect ratio, output stub, status, and forbidden items.
- `gpt_image_generation_brief.md`: Codex/GPT image generation instructions after the human checkpoint approves the prompt.

Use Codex/GPT image generation one figure at a time from the reviewed prompt pack. Store generated drafts under `review-data/03_framework/display_items/generated_drafts/` when the runtime provides a file artifact, then audit against the checklist before drafting around the image. Do not use generated images to introduce claims, numbers, citations, or table content.

The script does not invent final figure content or table evidence. It marks unsupported rows as `[EVIDENCE GAP]` and stops at `human_display_item_checkpoint.md`.

## Script 7b: `scripts/workflow_gatekeeper.py`

Purpose: enforce the interactive pre-draft gate. This script checks whether Codex has crossed mandatory checkpoints too early: framework/user confirmation, subagent packet returns, candidate-board screening decisions, official citation export, verified-pool import, full-text handoff, RAG index, and display-item approval.

Run before any substantial manuscript draft:

```bash
python $SKILL_DIR/scripts/workflow_gatekeeper.py check \
  --project-dir . \
  --stage pre-draft \
  --fail-on-blocker
```

Default output:

```text
review-data/05_audit/workflow_gate/pre-draft_workflow_gate.md
```

If the report status is `blocked`, Codex must stop and present the report to the user. It should not write a polished manuscript unless the user explicitly authorizes a provisional abstract-only/local-only draft. A blocked report commonly means:

- framework questions exist but no user decision file exists
- literature-discovery or candidate-board packets exist but no subagent CSV results were collected
- verified papers exist but official citation export has not run
- verified papers have not been imported into the governed pool
- full-text handoff or RAG is missing
- display items exist but no human approval file exists

## Script 7c: `scripts/manuscript_narrative_guard.py`

Purpose: audit the drafted manuscript body for internal workflow contamination. Top-journal prose should synthesize the field's past, present, and future; it should not expose review-engine artifacts such as literature-pool counts, `final-gate`, candidate-board status, RAG readiness, subagent packet state, official-citation export status, or "草稿核验显示" language. Evidence maturity can be discussed, but it must become a field-level evidence ladder, taxonomy, boundary condition, controversy, or future agenda rather than a running complaint that "the evidence is weak."

Run after a Markdown draft exists and before polishing, citation sequencing, DOCX/PDF export, or cover-letter finalization:

```bash
python $SKILL_DIR/scripts/manuscript_narrative_guard.py audit \
  --manuscript ./review-output/manuscript/<draft>.md \
  --out-dir ./review-data/05_audit/manuscript_narrative \
  --fail-on-blocker
```

Outputs:

- `manuscript_narrative_guard.md`: blocker/warning report.
- `manuscript_narrative_findings.csv`: line-level findings with section, snippet, and rewrite action.
- `manuscript_narrative_summary.json`: counts for blockers, warnings, evidence-negativity hits, and past/present/future arc signals.

Blocking examples:

- "本综述基于截至某日的文献池、筛选记录和 published-only final-gate..."
- "草稿核验显示 76 条候选中只有 7 条通过正式来源..."
- "candidate board 仍有 288 rows..."
- "RAG 尚未建立，因此..."

Rewrite these as field synthesis:

- "The field has moved from single-step tool use toward closed-loop scientific agents, but evidence maturity differs sharply across in silico benchmarks, wet-lab validation, and clinical workflow evaluation."
- "This review organizes the literature into historical foundations, current agent architectures, evidence ladders, and future governance requirements."

## Script 7d: `scripts/citation_sequence_manager.py`

Purpose: after a manuscript Markdown draft exists, renumber numeric citations by first appearance and rebuild the reference list from parsed manuscript references plus `official_citations.csv` when available. This is the citation-number manager: the LLM writes prose, but this script controls final numeric order, uncited-reference removal, missing-reference detection, and official-citation handoff.

Run after `official_citation_exporter.py`, `literature_pool.py final-gate`, and manuscript drafting, before DOCX/PDF export:

```bash
python $SKILL_DIR/scripts/citation_sequence_manager.py renumber \
  --manuscript ./review-output/manuscript/<draft>.md \
  --official-citations-csv ./review-data/05_audit/official_citations/official_citations.csv \
  --out-dir ./review-data/05_audit/citation_sequence \
  --compress-ranges \
  --require-official \
  --fail-on-blocker
```

Outputs:

- `manuscript_renumbered.md`: the draft body and bibliography rebuilt by first appearance.
- `citation_sequence_map.csv`: old citation number, new citation number, final reference text, DOI/PMID, official citation key, and export status.
- `missing_reference_entries.csv`: body markers whose original reference entry is missing.
- `missing_official_citations.csv`: cited papers that need official citation-manager export or manual handoff.
- `uncited_references.csv`: original reference-list entries removed because they do not appear in the body.
- `unresolved_citation_markers.csv`: markers that could not be safely rewritten.
- `citation_sequence_report.md`: final gate report for citation numbering.

If `official_citations.csv` is missing, the script can still renumber against the manuscript reference list for a provisional draft, but final submission must resolve `missing_official_citations.csv` and regenerate the manuscript from official exports. Do not manually renumber citations in prose after this step; rerun the script.

## Script 8: `scripts/agent_memory.py` and `scripts/review_agent_orchestrator.py`

Purpose: create persistent shared/per-agent memory, then create and optionally run specialist agents. Codex remains chief editor.

Initialize memory:

```bash
python $SKILL_DIR/scripts/agent_memory.py init \
  --project-dir . \
  --agents all

python $SKILL_DIR/scripts/agent_memory.py append-shared \
  --project-dir . \
  --file project_memory.md \
  --kind scope \
  --source user_alignment \
  --note "Project topic, target journal family, and user-approved workflow state."
```

Packet-only mode:

```bash
python $SKILL_DIR/scripts/review_agent_orchestrator.py plan \
  --project-dir . \
  --out-dir ./review-work/agent_orchestration \
  --agent-memory-dir ./review-data/06_agent_memory \
  --agents literature_retriever,citation_verifier,outline_architect,figure_table_designer,reviewer_auditor
```

Framework-only packet mode after `draft_logic_framework.py`:

```bash
python $SKILL_DIR/scripts/review_agent_orchestrator.py plan \
  --project-dir . \
  --out-dir ./review-work/agent_orchestration_framework \
  --agent-memory-dir ./review-data/06_agent_memory \
  --no-default-inputs \
  --agents draft_deep_reader,draft_memory_curator,draft_frame_reader,literature_strategist,argument_builder,outline_architect,framework_devils_advocate,framework_dialogue_moderator,blackboard_curator \
  --input ./review-data/03_framework/draft_memory/draft_memory_index.md \
  --input ./review-data/03_framework/draft_memory/agent_reading_packet.md \
  --input ./review-data/03_framework/logic_framework/draft_logic_framework.md \
  --input ./review-data/03_framework/logic_framework/framework_evidence_blackboard.md \
  --input ./review-data/03_framework/logic_framework/argument_evidence_map.csv \
  --input ./review-data/03_framework/logic_framework/literature_search_tasks.csv \
  --input ./review-data/03_framework/logic_framework/framework_material_passport.json
```

Optional external OpenAI-compatible mode, only after approval for external calls and data egress:

```powershell
$env:DEEPSEEK_API_KEY="..."
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:DEEPSEEK_MODEL="deepseek-v4-pro"
python $SKILL_DIR/scripts/review_agent_orchestrator.py run `
  --project-dir . `
  --out-dir ./review-work/agent_orchestration `
  --agents literature_retriever,literature_screener,citation_verifier,reviewer_auditor
```

Default manuscript workflows should use packet/Codex-subtask mode. External defaults are provider examples for the `run` command, not a requirement for normal review work.

External agent results are written to `review-work/agent_orchestration/agent_results`. With `--record-agent-results` enabled, the orchestrator also appends an excerpt to that agent's memory under `review-data/06_agent_memory/agents/<agent>/`.

Read `references/multi-agent-orchestration.md` for the full role map.

## Script 9: `scripts/cover_letter_builder.py`

Purpose: create a concise cover-letter package for journal submission after the manuscript thesis and journal fit are stable.

Minimal command:

```bash
python $SKILL_DIR/scripts/cover_letter_builder.py \
  --manuscript <manuscript-path> \
  --target-journal "<target-journal>" \
  --article-type "Review" \
  --out-dir ./review-output/cover_letter
```

Better command with editor-facing pitch fields:

```bash
python $SKILL_DIR/scripts/cover_letter_builder.py \
  --manuscript <manuscript-path> \
  --target-journal "<target-journal>" \
  --article-type "Review" \
  --background-question "<one-sentence field problem>" \
  --main-finding "<main thesis or organizing insight>" \
  --field-impact "<why this matters for the field>" \
  --broad-audience "<communities that will care>" \
  --journal-fit "<why this journal's readers need this synthesis>" \
  --comparison "<respectful comparison with competing reviews or alternatives>" \
  --corresponding-author "Name <email@example.com>" \
  --out-dir ./review-output/cover_letter
```

Reviewer fields are pipe-delimited:

```bash
--reviewer "Name|Affiliation|email@example.com|medical AI evaluation|No recent collaboration or institutional conflict"
--exclude-reviewer "Name|Recent direct collaboration with a coauthor"
```

Outputs:

- `cover_letter_draft.md`
- `cover_letter_brief.md`
- `cover_letter_checklist.md`
- `suggested_referees.csv`
- `excluded_referees.csv`
- `cover_letter_summary.json`

Read `references/cover-letter-workflow.md` before finalizing the letter.
