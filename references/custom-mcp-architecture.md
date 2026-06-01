# Custom MCP Architecture

Use this reference when turning the review-writing skill into a stable local MCP service.

## Quick Navigation

- Recommendation and Why MCP: what belongs in the skill versus the MCP service.
- When To Graduate: why complex review writing should become a dedicated engine rather than more prompt text.
- Proposed MCP Tools: literature recall, verification, pool management, RAG, display items, memory, drafting, and audit surfaces.
- Proposed MCP Resources: stable resource names for project state, literature pools, cards, final gates, and agent memory.
- External LLM Configuration: packet-only default, optional approved DeepSeek/OpenAI-compatible acceleration, and secret handling.
- MCP Implementation Plan and Boundary: wrapper-first architecture, native service roadmap, and what must remain in the skill.

## Recommendation

Yes: combine this skill with a dedicated MCP server for long-running review projects.

The skill should remain the cognitive playbook: when to search, screen, demote, synthesize, audit, and write. The MCP server should expose stable tools and resources: literature recall, metadata verification, document pool state, citation gates, card retrieval, RAG search, shared/per-agent memory, and external LLM agent calls.

Codex remains the chief editor/PI:

```text
User
  -> Codex with $review-ai-skills
     -> custom review MCP tools/resources
        -> PubMed/Crossref/OpenAlex/arXiv/OpenReview/DeepSeek/local files
     -> Codex adjudicates, rewrites, and delivers
```

## Why MCP

Use MCP when you need:

- stable tool names instead of remembering script paths
- persistent project state across sessions
- central API keys, rate limits, retries, and logs
- resources such as `literature://pool`, `literature://cards/{key}`, `review://audit/final-gate`
- callable tools for multiple agents without shell command glue
- dynamic loading of only the selected agent's memory instead of flooding every agent with all state
- a single interface for PubMed, Crossref/OpenAlex, arXiv/OpenReview, DeepSeek, and local RAG

Keep scripts as the deterministic backend. The MCP server can call the scripts first, then later replace them with native Python implementations as the project matures.

## When To Graduate From Skill To Engine

A skill is enough for a playbook, command catalog, and editorial policy. It is not enough as the long-term state machine for a top-journal review once the project needs:

- a governed database for literature identity, abstracts, full text, cards, RAG chunks, official citations, and manuscript citation usage
- resumable multi-agent queues with real status, retries, rate limits, and logs
- deterministic gates that block drafting, polishing, DOCX export, or submission when state is incomplete
- interactive checkpoints where the user can approve framework, papers, display items, and missing-fulltext handoffs
- narrative guards that keep internal evidence-management artifacts out of the article body

The recommended split is:

- Skill: editorial constitution and routing policy.
- Scripts: deterministic backend commands and smoke tests.
- MCP/review engine: persistent state, APIs, queues, resources, and UI-facing checkpoints.
- Codex: chief editor/PI that reads drafts, interprets state, decides tradeoffs, and writes/refines prose.

If a manuscript draft starts mentioning `文献池`, `final-gate`, candidate counts, RAG status, or "草稿核验显示", that is a signal that the engine state has leaked into prose. The fix is not just a better prompt; it is an explicit narrative guard plus a stateful engine that separates evidence governance from article narrative.

## Proposed MCP Tools

Literature recall:

- `build_draft_citation_assets(draft_paths, draft_dir, topic)`
- `build_draft_memory(draft_paths, draft_dir, topic)`
- `load_chief_reading_packet(project_dir)` to expose draft memory plus citation assets before delegation
- `build_literature_discovery_packets(draft_dir, draft_assets_dir, topic)` to force chief-Codex and subagent draft reading before API search
- `collect_literature_discovery_outputs(input_path, is_dir, reference_candidates_csv)` to emit verifier candidates and route-specific API tasks
- `ingest_user_pubmed_set(input_files, topic, set_name)` to parse user-prepared PubMed Summary/Abstract exports into a separate screened seed lane
- `collect_user_pubmed_screening(set_dir, screening_csvs)` to turn accepted user PubMed seed decisions into verifier-ready rows and a verification handoff
- `build_literature_candidate_board(project_dir)` to expose a chief-editor board across draft-native, user PubMed, AI-mined, supplemental, and verification lanes
- `build_candidate_board_agent_packets(board_dir, packet_size)` to generate `candidate_board_curator`, `literature_screener`, and `citation_verifier` Codex-subtask packets from deduped board worklists
- `collect_candidate_board_agent_decisions(board_dir, input_dirs, input_csvs)` to turn candidate-board subagent decisions into deterministic-verifier, repair, full-text, rejection, duplicate, and human-decision queues without promoting papers
- `build_hidden_candidate_packets(draft_dir, topic)`
- `collect_hidden_candidate_outputs(input_path, is_dir, reference_candidates_csv)` with reference-overlap provenance marking
- `build_identity_normalization_packets(candidate_csv, topic)`
- `merge_normalized_identities(candidate_csv, normalized_csv)`
- `pubmed_abstracts_to_verifier(abstracts_csv, tasks_csv)` to move screened PubMed abstract hits back into deterministic verification
- `verify_draft_reference_clues(candidate_csv, email, max_records)`
- `classify_publication_status(verification_csv)`
- `build_draft_source_adjudication_cases(verification_csv, claim_map)`
- `expose_draft_source_adjudication_packets(verification_csv, claim_map)`
- `adjudicate_draft_sources_external(cases, model, max_workers)` only for explicitly approved external acceleration
- `merge_draft_source_adjudications(verification_csv, adjudication_csv)`
- `merge_verified_literature_lanes(csv_paths)` to combine explicit, discovery, hidden/normalized, and PubMed-deep-dive accepted papers into one deduped union
- `export_official_citations(verified_csv, style)` for PubMed NBIB/MEDLINE, Crossref/DOI BibTeX/RIS, DOI bibliography text, arXiv BibTeX, and manual handoff rows
- `plan_argument_literature_tasks(framework_dir, topic, max_tasks)`
- `fetch_pubmed_argument_abstracts(tasks_csv, max_results, workers, resume)`
- `export_argument_abstract_screening_packet(run_dir)`
- `plan_supplemental_recall_screening(topic, draft_assets_dir, literature_discovery_dir, max_tasks)` for broad-but-bounded recall after framework confirmation
- `run_supplemental_pubmed_recall(tasks_csv, max_tasks, max_results, workers)` for larger PubMed candidate pools
- `collect_supplemental_recall_candidates(candidate_csvs, recall_dirs)` to dedupe PubMed/Crossref/OpenAlex/arXiv/paper-search outputs and create agent screening packets
- `collect_supplemental_screening_decisions(candidates_csv, screening_csv)` to convert screened inclusions into verifier-ready rows
- `audit_supplemental_recall_pool(collection_dir, screened_dir, verification_dir)` to show route composition, backlog, deferred rows, quarantine, and verification coverage before more recall
- `build_recall_supervision_packet(collection_dir, screened_dir, max_pubmed_tasks, quotas, strategic_pubmed_tasks)` to expose the current recall pool to Codex/subagents and create quota-preserving PubMed tasks
- `collect_recall_supervision_decisions(supervision_csvs, max_pubmed_tasks, quotas, strategic_pubmed_tasks)` to turn edited supervision rows into PubMed tasks and rescue candidates without letting formal-version checks dominate
- `recall_pubmed(query, query_file, mindate, maxdate, max_results)`
- `recall_crossref_openalex(query, doi, sources, from_year, to_year, max_results)`
- `recall_arxiv_openreview(query, arxiv_id, openreview_id, sources, max_results)`
- `import_recall_to_pool(recall_dir, origin)`

Candidate provenance should be explicit in MCP responses:

- `explicit_reference`: draft-native reference-section paper clue
- `body_cited_reference`: in-text marker linked to a draft reference
- `user_pubmed_seed`: user-prepared PubMed export row that still needs framework screening before promotion
- `reference_recovery`: subagent recovered/corrected a paper already present in draft citation assets
- `hidden_body_candidate`: subagent found a paper clue in prose that is absent from citation assets
- `supplemental_gap_candidate`: new search result for an approved claim/display-item gap

The MCP should refuse to promote `user_pubmed_seed`, `reference_recovery`, `hidden_body_candidate`, or `supplemental_gap_candidate` rows to the main pool until screening, deterministic verification, and claim-fit adjudication pass.

Literature management:

- `init_literature_pool(pool_dir)`
- `export_draft_asset_health()`
- `list_unverified_draft_sources()`
- `list_strict_supplemental_queries()`
- `import_argument_candidates_to_supplemental_pool(candidates_csv)`
- `export_candidate_packet(limit, status)`
- `record_screening_decision(key, status, score, rationale, claim, limitation, use)`
- `build_literature_cards(status)`
- `export_citation_context(max_cards)`
- `fetch_evidence(status, limit)`
- `final_citation_gate(apply)`

Evidence/RAG:

- `build_structured_lit_rag(pool_dir)`
- `search_lit_rag(query, top_k)`
- `get_literature_card(key)`
- `get_paper_evidence(key)`

Display items:

- `plan_display_items(framework_dir, pool_dir, rag_db, target_journal)`
- `get_display_item_plan()`
- `build_display_item_evidence_pack(item_id)`
- `export_figure_prompt_pack(item_ids)`

Agent memory:

- `init_agent_memory(agents)`
- `get_agent_memory_packet(agent_name)`
- `append_agent_memory(agent_name, kind, note, source)`
- `ingest_agent_result(agent_name, result_path)`
- `list_agent_memory(agents)`

Draft and audit:

- `audit_deep_research_draft(draft_paths)`
- `run_review_agent(agent_name, goal, inputs)`
- `run_agent_pack(agents, goal)`
- `audit_manuscript_narrative(draft_path)` to block internal workflow/audit-count leakage and check the past-present-future arc
- `audit_claim_citation_fit(draft_path)`
- `renumber_citation_sequence(draft_path, official_citations_csv)` to rebuild numeric citations and references by first appearance
- `generate_revision_roadmap(draft_path)`

## Proposed MCP Resources

- `review://project/config`
- `review://draft-citation-assets`
- `review://draft-citation-assets/health`
- `review://draft-citation-assets/claim-evidence-map`
- `review://draft-source-adjudication`
- `review://draft-source-adjudication/human-queue`
- `review://argument-literature/tasks`
- `review://argument-literature/abstract-packet`
- `review://argument-literature/claim-candidate-links`
- `literature://pool`
- `literature://supplemental-pool`
- `literature://candidate-packet`
- `literature://citation-context`
- `literature://cards/index`
- `literature://cards/{key}`
- `literature://final-citation-gate`
- `review://display-items`
- `review://display-items/{item}`
- `review://agent-memory/shared`
- `review://agent-memory/{agent}`
- `review://agent-results/{agent}`
- `review://draft-audit`
- `review://manuscript/narrative-guard`
- `review://manuscript/citation-sequence`

## External LLM Configuration

The default custom-MCP design should expose packet-only resources and let Codex subtasks perform draft mining, citation adjudication, framework critique, and audits without requiring external LLM keys.

Before a custom MCP or specialist-agent runner makes optional external LLM calls, inspect local configuration with:

```bash
python $SKILL_DIR/scripts/project_bootstrap.py check-env --project-dir .
```

If a key is missing, keep using packet-only tools/resources and Codex subtasks. Ask the user to set a local environment variable or secure MCP config only when they explicitly request external acceleration. Do not store API keys in MCP resources, project JSON files, literature cards, or chat transcripts.

Generic provider variables:

```powershell
$env:REVIEW_AGENT_API_KEY="..."
$env:REVIEW_AGENT_BASE_URL="https://provider.example.com"
$env:REVIEW_AGENT_MODEL="provider-model"
```

Optional DeepSeek-compatible example:

```powershell
$env:DEEPSEEK_API_KEY="..."
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:DEEPSEEK_MODEL="deepseek-v4-pro"
```

Model policy: Codex-subtask packet mode is the default. If an external backend is explicitly approved, use a strong reasoning model for citation verification, reviewer audit, outline architecture, and final synthesis.

For faster/cheaper optional screening agents, use a lower-cost model only after the same approval:

```powershell
$env:DEEPSEEK_MODEL="deepseek-v4-flash"
```

For high-stakes optional external agents such as `citation_verifier`, `reviewer_auditor`, and `outline_architect`, if DeepSeek is selected after explicit approval, prefer `deepseek-v4-pro` with thinking enabled and high reasoning effort.

Do not store API keys inside skill files or project outputs. Keep secrets in environment variables or the MCP server's secure config.

## MCP Implementation Plan

Phase 1: wrapper MCP

- Implement MCP tools as thin wrappers around existing scripts.
- Return file paths plus compact JSON summaries.
- Start from the bundled prototype under `mcp/review_literature_mcp/`, which wraps draft citation extraction, hidden-candidate packets, Codex-subtask output collection, identity-normalization packets, verifier-ready merges, deterministic verification, pool import, and final gate.
- The prototype also wraps literature-discovery packets, PubMed-abstract-to-verifier conversion, verified-lane union, and deduped official citation export.
- Expose pool/cards/gate reports as resources.
- Expose `review-data/06_agent_memory` shared and per-agent packets as resources.
- Keep run logs in `review-work/logs`; keep governed literature-pool logs in `review-data/02_literature/pool/logs`.

Phase 2: native MCP

- Move PubMed/Crossref/OpenAlex/arXiv/OpenReview calls into server functions.
- Add retry, cache, and rate-limit middleware.
- Add project-level SQLite store for pool metadata and source provenance.
- Add typed memory records for agent handoffs, decisions, open questions, and result pointers.
- Add optional OpenAI-compatible model router for specialist agents while keeping packet-only/Codex-subtask resources as the default.

Phase 3: review cockpit

- Add tools for claim-citation alignment, final reference formatting, figure/table planning, and DOCX/LaTeX export QA.
- Add resources for human display-item checkpoints and reviewed figure prompt packs.
- Add dashboards/resources for unresolved evidence gaps and human-download handoffs.

## Boundary

Do not put the whole review-writing policy only in MCP. Keep the skill as the human-readable operating doctrine and use MCP as the reliable execution surface.
