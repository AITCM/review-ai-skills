---
name: top-journal-review-writer
description: "Top-journal review article workflow for high-impact review manuscripts: draft-first ingestion, literature verification, evidence synthesis, citation audit, multi-agent orchestration, display-item planning, manuscript revision, cover-letter preparation, and this skill's own maintenance. Use for review-project setup only when tied to a review manuscript, review-agent external-LLM configuration, specialist-agent orchestration for review writing, or maintaining this skill package. Do not trigger for generic Python environment, package metadata, credential, or agent questions outside review writing or this skill maintenance. Triggers include top journal review, Nature Reviews style, Q1 review, review manuscript, scoping review, systematic review, narrative review, cover letter, journal pitch, referee list, top-journal-review-writer maintenance, 顶刊综述, 高水平综述, 文献综述, 系统综述, 综述论文, 投稿信, 推荐审稿人, 综述项目环境, 综述智能体编排."
---

# Top Journal Review Writer

## Operating Rule

Use this skill to help build a rigorous, auditable review manuscript. The current Codex conversation remains the chief editor: it must read the user drafts, understand the central argument, decide what to accept from agents, and keep the human researcher in the loop for framework, evidence, and display-item decisions.

When running bundled scripts, always resolve paths from the skill directory, not from the manuscript project. In PowerShell:

```powershell
$SKILL_DIR = "<skill-dir>"
python "$SKILL_DIR/scripts/project_bootstrap.py" check-env --project-dir .
```

In this file, `<S>` means `<skill-dir>/scripts`. Never assume project-local `./scripts`.

## Mode Selection

Choose the lightest mode that fits:

- `plan`: clarify topic, review type, target journal family, audience, and contribution thesis.
- `skill-maintenance`: improve this skill package itself; edit skill files, run `<S>/skill_audit.py`, run `quick_validate.py` when available, and avoid writing manuscript project data unless the user explicitly asks for a smoke test.
- `data-governance`: inspect or initialize canonical folders and legacy path aliases.
- `bootstrap`: inspect or initialize a review-root Python/project environment only when the request is about this review workflow; ask before creating files if required project files are missing.
- `draft-memory`: deeply read GPT/Gemini/Deep Research drafts into persistent memory cards and `chief_editor_reading_checkpoint.md`.
- `literature-discovery-orchestration`: after draft memory and citation assets, force chief Codex to read all drafts, state a provisional thesis/claim map, create subagent literature-discovery packets, collect subagent reasoning outputs, and route bounded API search tasks before deterministic verification.
- `draft-literature-mining`: use Codex subtask packets by default to read draft prose and mine hidden/partial paper candidates, named systems, venue clues, and PubMed deep-dive queries; collect subtask outputs, normalize informal system names to official paper identities, then run deterministic verification before promotion; external LLM APIs are optional accelerators only when explicitly approved.
- `agent-memory`: initialize, inspect, dynamically load, and update `review-data/06_agent_memory` for shared and per-agent state.
- `architecture`: turn one or more drafts into a thesis-to-outline framework, argument-evidence map, and user alignment checkpoint.
- `draft-citation-assets`: inspect draft-native reference sections, in-text markers, claim evidence clues, non-academic sources, and strict supplemental-search gaps before any broad recall.
- `user-pubmed-set`: ingest user-supplied PubMed Summary/Abstract exports as a separate screened seed lane; parse PMID/DOI/title/abstract metadata, generate subagent screening packets, and do not promote rows until screening, verification, adjudication, and final-gate checks pass.
- `candidate-board`: build a read-only chief-editor board across draft-native, user PubMed seed, AI-mined, supplemental, and verification lanes so Codex can see screening backlog, verifier-ready rows, verified rows, and repair/delete tasks before launching more recall.
- `draft-governance`: verify real papers through PubMed/PMID/DOI and metadata APIs, separate webpages/blogs from papers, and decide whether supplemental recall is needed.
- `citation-adjudication`: use Codex specialist subtasks by default after API verification to judge title match, claim-source fit, preprint risk, inclusion rationale, and human confirmation items; use DeepSeek/OpenAI-compatible APIs only as an optional approved accelerator.
- `official-citation-export`: after paper identity verification/adjudication, fetch official citation-manager exports from PubMed, Crossref/DOI, arXiv, or publisher/conference pages; unresolved exports go to a manual handoff list.
- `citation-sequence`: after drafting and official citation export, run deterministic citation numbering and bibliography rebuild; never trust LLM-assigned numeric order as final.
- `argument-literature-expansion`: after the draft-derived framework checkpoint, convert specific claims, past-present-future bridges, display-item needs, and evidence gaps into bounded PubMed/Crossref/OpenAlex/arXiv/OpenReview tasks; save abstracts and candidates to `supplemental_pool` for screening before any promotion.
- `supplemental-recall-screening`: after the framework/user checkpoint and draft-derived verified union, run broad-but-bounded multi-source recall from body systems, unmapped claims, framework gaps, and discovery API tasks; seed-lock already verified/curated draft assets; quarantine obvious metadata noise; generate ranked, chunked agent screening packets; only screened inclusions return to deterministic verification and the verified union.
- `recall` / `pool` / `fulltext` / `rag`: build governed literature pools, fetch Europe PMC/OA/user full text, parse PDFs/OCR, create cards, and build a structure-aware RAG index.
- `display-items`: after the logic framework and literature state exist, plan evidence-grounded figures, tables, and boxes under `review-data/03_framework/display_items`.
- `orchestrate`: coordinate specialist agents for draft reading, literature, screening, framework, figures/tables, synthesis, governance, and final audit.
- `synthesis`: turn verified materials into themes, controversy maps, taxonomies, gaps, and future agenda.
- `draft`: write manuscript sections only from verified evidence and explicit gaps.
- `narrative-guard`: after drafting, audit the article body for internal workflow contamination, audit-count leakage, overuse of "weak evidence" language, and missing past-present-future arc.
- `audit`: check citation truth, claim support, overreach, missing evidence, display-item gaps, and journal-fit risk.
- `revise`: improve an existing draft or respond to reviewer/editor comments.
- `format`: adapt manuscript, abstract, highlights, disclosures, and submission materials to a journal style.
- `cover-letter`: prepare a concise editor-facing cover letter, related-manuscript disclosures, suggested referees, and excluded-referee rationale.

## Default Pipelines

For AI-generated drafts, run:

1. Read first: build draft memory, citation assets, and any user PubMed seed-set lane; chief Codex states the provisional thesis, claim map, and citation-lane health before broad recall.
2. Recover and verify papers: use literature-discovery packets, bounded subagents, candidate-board checkpoints, identity repair, PubMed/DOI/title/publisher verification, adjudication, and official citation export.
3. Discuss architecture: build the draft logic framework, evidence-aware argument map, historical/current/future spine, and human checkpoint before drafting.
4. Fill evidence gaps: screen user PubMed seeds and strict supplemental recall only for confirmed framework, claim, figure, table, or controversy gaps.
5. Build memory: merge verified lanes into the governed pool, fetch Europe PMC/OA or user-supplied full text, parse/OCR as needed, build cards, and create TreeRAG.
6. Produce and audit: plan display items, orchestrate specialist agents in small batches, synthesize, draft, claim-support audit, revise, format, and prepare the cover letter.

In draft-first projects, the drafts supply the candidate paper logic. Literature retrieval supplies evidence, counterevidence, historical foundations, and boundary conditions for that logic. Do not let broad supplemental recall rewrite the review's center of gravity before the user has discussed the thesis and section spine.

For the full executable chain and lane-specific command variants, read `references/draft-first-literature-workflow.md` and `references/scripted-literature-management.md`.

For a review from scratch, run:

`plan -> recall -> pool -> screen -> synthesize -> outline -> display-items -> draft -> audit -> revise -> format`.

For this skill package, run:

`skill-maintenance scope -> inspect skill files -> patch scripts/references/SKILL.md -> run skill_audit.py to stdout or scratch -> run quick_validate.py if present -> summarize changed capability`.

## Core Workflow

1. If the user is improving this skill, stay in `skill-maintenance` mode. Do not continue manuscript drafting or project initialization unless explicitly requested.
2. For review projects that will use many local tools or agents, run `<S>/path_schema.py check --project-dir .` and `<S>/project_bootstrap.py check-env --project-dir .`. If required files/folders are missing, ask before running `init` unless the user already requested setup.
3. If drafts exist, run `<S>/draft_memory.py build` first, then run `<S>/draft_citation_assets.py build`. If the user also supplies PubMed Summary/Abstract exports, run `<S>/pubmed_user_set_ingestor.py parse` and keep the output under `review-data/02_literature/user_pubmed_sets/<set-name>`. Run `<S>/literature_candidate_board.py build` after major candidate-producing or verification stages. The current Codex conversation must load `chief_editor_reading_checkpoint.md`, the memory index, the agent reading packet, relevant per-draft cards, `draft_citation_health.md`, `candidate_paper_clues.csv`, `claim_evidence_map.csv`, `body_mentioned_systems.csv`, `unmapped_claims.csv`, any `pubmed_user_set_ingestion_summary.md` / `user_pubmed_screening_guide.md`, and `literature_candidate_board.md` before delegating or drafting.
4. Chief Codex must state a provisional thesis/claim map from all drafts before spawning literature subtasks. The map should identify past foundations, current systems/results, future agenda, disputed claims, and which references are explicit, missing, polluted, or need identity normalization.
5. Run `<S>/draft_literature_discovery_orchestrator.py plan` and use Codex subagents on the generated packets. Each subagent must return candidate papers plus route-specific API tasks (`pubmed`, `crossref_openalex`, `arxiv_openreview`, `publisher`, or `human`) linked to a claim/argument role. Then run `collect` with `--reference-candidates-csv ./review-data/02_literature/draft_assets/candidate_paper_clues.csv`. This is the main literature discovery layer; it exists to prevent under-reading the drafts and under-counting papers.
6. Use `<S>/draft_literature_candidate_miner.py packet` as an optional legacy/extra chunk miner when discovery packets miss system-name or title-fragment clues. Collect with reference-overlap marking, then run `normalize-packet` and `merge-normalized` so informal names such as system names are converted into verifier-ready official paper identities. Use `mine` with DeepSeek/OpenAI-compatible APIs only if the user explicitly approves external data egress.
7. Run `<S>/draft_logic_framework.py` before writing and before broad supplemental recall. Stop for user discussion of thesis, 3-5 major section spine, argument-evidence map, historical foundations, missing evidence, and disputed claims unless the user requests autonomous drafting.
8. Keep draft-native citation assets, user-supplied PubMed seed sets, AI-discovered reference-recovery rows, hidden body candidates, claim-gap candidates, and supplemental search outputs in separate lanes.
9. Verify draft-derived and AI-discovered paper clues with `<S>/draft_reference_verifier.py verify`. Treat PMID/DOI as strong clues, not truth: if an identifier resolves to a title that conflicts with the draft title, fall back to title/bibliographic search and record the mismatch. Use PubMed/PMID, DOI, title, author/year/venue clues, Crossref/OpenAlex, and official publisher/conference pages as a multi-signal identity-resolution loop. Treat arXiv/OpenReview-only records as leads by default; official ICLR/NeurIPS/ICML/OpenReview proceedings pages can be verified as conference papers when title/venue evidence matches.
10. Route discovery `api_search_tasks.csv` through PubMed/Crossref/OpenAlex/arXiv/OpenReview recall when candidate titles are incomplete or when the claim needs missing foundational/current evidence. PubMed abstract hits must be screened and converted back to verifier input with `<S>/draft_literature_candidate_miner.py pubmed-to-verifier` before promotion; save broader results to `supplemental_pool` until screened.
11. Run `<S>/draft_source_adjudicator.py build-cases` and use Codex subtasks for citation adjudication by default. Use `adjudicate` with DeepSeek/OpenAI-compatible APIs only if the user explicitly approves external data egress. The adjudicator may judge fit from supplied facts; it may not invent identifiers, upgrade unverified records, or bypass API verification.
12. Merge accepted verified rows from all lanes with `<S>/literature_pool.py merge-verified`: explicit draft references, literature-discovery candidates, identity-normalized hidden candidates, and screened PubMed deep-dive candidates. This deduped union is the source of truth for accepted draft-derived papers.
13. Run `<S>/official_citation_exporter.py export` on the deduped verified union to fetch official citation-manager records: PubMed NBIB/MEDLINE, Crossref/DOI BibTeX/RIS, DOI bibliography text, arXiv BibTeX, or a manual official-export handoff. Do not treat draft reference strings as final formatted references, and do not confuse input rows with unique papers.
14. Import only the deduped verified union into `review-data/02_literature/pool`. User PubMed seed rows with PMID/DOI are still candidate rows until agent/human screening has accepted their framework role and deterministic verification/adjudication confirms identity and claim fit.
15. For approved framework gaps, run `<S>/argument_literature_expander.py plan` and then `pubmed` to save initial PubMed metadata/abstract candidates by claim. Treat abstracts as triage material only: they can justify screening, replacement, or background framing, but detailed method/result/metric claims require full text.
16. If the accepted verified union is too small for a top-journal review, or if the framework has broad historical/current/future gaps, run `<S>/supplemental_recall_screening.py plan`. This is the large candidate-pool layer: it creates PubMed, Crossref/OpenAlex, arXiv/OpenReview, paper-search, and human query tasks from `body_mentioned_systems.csv`, `unmapped_claims.csv`, literature-discovery API tasks, and the confirmed topic/frame. Run route-specific recall, then `collect-candidates` with `--seed-csv` pointing to the accepted verified/curated draft-derived union. Dispatch Codex subagents over `screening_packets/`, not the full raw candidate table. `seed_locked_candidates.csv` is an不可漏清单; `quarantined_candidates.csv` is for sampling or human rescue, not first-pass promotion. Run `collect-screening` after subagent/human decisions. After every broad recall audit, candidate-board checkpoint, or 1-2 screening packets, run `supervise-pool` / `collect-supervision`; PubMed supervision must preserve quotas for deep-dive, formal-version, framework/history, governance, and rescue tasks, and should add a small strategic PubMed supplement when candidate-derived deep dives are sparse.
17. Verify screened supplemental inclusions with `<S>/draft_reference_verifier.py verify`, adjudicate with `<S>/draft_source_adjudicator.py`, and merge accepted supplemental papers into the verified union with `<S>/literature_pool.py merge-verified`. Do not promote broad-recall hits directly to the main pool.
18. Put newly recalled papers in `review-data/02_literature/supplemental_pool` until screened. Promote only after relevance, publication status, claim fit, argument-map role, and human/LLM screening are checked. Interesting papers that do not serve the confirmed framework remain `background_or_defer`.
19. Run the published-only final gate before final drafting. If PubMed, Crossref/OpenAlex, arXiv, OpenReview, and official publisher/conference pages cannot verify a source at all, remove it from citation candidates. If only arXiv/OpenReview can verify it without official venue evidence, keep it out of final references unless the user explicitly allows preprints.
20. Fetch full text for selected papers with Europe PMC fullTextXML first, then OA/paper-search fallback, then user handoff. Do not make detailed method/result/metric claims from abstracts alone.
21. Build literature cards and, when needed, a TreeSearch-inspired RAG index over cards, abstracts, and full text.
22. Run display-item planning before substantial drafting when figures, tables, boxes, or graphical abstract ideas matter. Tables are evidence-first; unsupported cells must remain `[EVIDENCE GAP]`. Figures are concept-first: create blueprints, GPT image-ready prompts, and a human checkpoint; use Codex/GPT image generation only for approved conceptual drafts, then audit labels, claims, and evidence before drafting around them.
23. Before any substantial manuscript draft, run `<S>/workflow_gatekeeper.py check --stage pre-draft`. If it reports `blocked`, stop and present the report to the user. Do not silently skip framework confirmation, subagent screening, candidate-board decisions, official citation export, full-text handoff, RAG, or display-item approval unless the user explicitly authorizes a provisional abstract-only/local-only draft.
24. Use Codex specialist subtasks for bounded parallel work after the chief Codex pass has scoped the task and selected input files. Packet/subtask mode is the default and requires no API key. If packets exist but no subagent result CSVs are collected, say it was a local-only fallback rather than a multi-agent run. DeepSeek `deepseek-v4-pro` remains an optional accelerator only when external calls and data egress are explicitly approved.
25. Draft around a clear narrative, not a paper-by-paper list. Every section must advance an argument backed by verified evidence.
26. After a manuscript draft exists, run `<S>/manuscript_narrative_guard.py audit`. Remove internal workflow language from the article body: literature-pool counts, final-gate bookkeeping, candidate-board status, RAG state, "草稿核验显示", and repeated generic "evidence is weak" complaints. Convert evidence status into a field-level past-present-future synthesis, evidence ladder, taxonomy, limitations paragraph, or future agenda.
27. Run `<S>/citation_sequence_manager.py renumber` against the Markdown draft and `official_citations.csv`. Treat `manuscript_renumbered.md`, `citation_sequence_map.csv`, `missing_official_citations.csv`, and `uncited_references.csv` as the source for final DOCX/PDF export; do not hand-fix citation numbers in prose.
28. Audit every citation and important claim before polishing. Lead audits with blocking issues: fabricated or unverifiable references, unsupported claims, overgeneralization, weak structure, figure/table evidence gaps, narrative workflow contamination, wrong citation order, uncited references, and journal-fit risks.
29. For submission, prepare a short cover letter: main finding, field impact, broad appeal, respectful comparison with alternatives, related-manuscript disclosures, suggested referees, and reasonable excluded-referee requests.

## Non-Negotiables

- Never fabricate citations, identifiers, authors, venues, methods, sample sizes, results, or full-text evidence.
- Never accept a draft DOI/PMID when the resolved official title conflicts with the draft title; fall back to title/bibliographic search and flag the mismatch.
- Never use a draft reference string as final citation formatting when an official citation-manager export is available.
- Never let the writing model choose final numeric citation order; run deterministic first-appearance renumbering before final export.
- Never put internal workflow state, audit counts, candidate-board status, RAG status, or "草稿核验显示" style process language into the main article body.
- Never let agents replace the chief Codex reading pass.
- Never cite a paper for a claim it does not support.
- Never print, save, commit, or echo real API keys.
- Final references should be formally published papers or official guidelines/standards by default.
- Keep provenance for every source: title, DOI/URL, database/source, query or corpus origin, and retrieval date when available.
- Separate evidence from interpretation. Label hypotheses and expert judgment clearly.
- For systematic/scoping reviews, keep PRISMA-style reporting. For narrative reviews, still keep a transparent search and selection rationale.
- For Chinese users, discuss and draft in Chinese by default unless the manuscript target is English; preserve paper titles and DOI/URL values exactly.

## References

Read only the reference needed for the current task:

- `references/review-pipeline.md`: phase workflow, deliverables, and quality gates.
- `references/skill-maintenance-and-forward-test.md`: maintenance, self-audit, and forward-test rules.
- `references/data-governance-folder-schema.md`: canonical folders, aliases, and legacy path handling.
- `references/project-bootstrap-and-secrets.md`: review-root setup and secret detection.
- `references/draft-logic-framework.md`: draft-to-thesis framework and user checkpoint.
- `references/fulltext-and-draft-memory.md`: draft memory, full-text acquisition, user handoff, and MinerU parsing.
- `references/review-blackboard-passport.md`: blackboard, passport, and phase boundaries.
- `references/agent-memory-workflow.md`: shared/per-agent memory and dynamic loading.
- `references/deep-research-draft-governance.md`: Deep Research draft use, source filtering, and verification.
- `references/draft-first-literature-workflow.md`: draft-native citation assets and strict supplemental recall.
- `references/llm-citation-adjudication.md`: Codex-subtask citation adjudication after metadata verification; external LLM mode is optional.
- `references/literature-recall-integration.md`: `paper-search-mcp` and fallback recall.
- `references/scripted-literature-management.md`: full command catalog for bundled scripts.
- `references/literature-pool-workflow.md`: governed pools, downloads, and literature-card memory.
- `references/published-only-citation-policy.md`: final-reference gate.
- `references/multi-agent-orchestration.md`: specialist roles, packet mode, external API mode, and traceback.
- `references/custom-mcp-architecture.md`: dedicated MCP architecture and external model configuration.
- `references/cover-letter-workflow.md`: Nature-style cover-letter package.
- `references/structured-rag-workflow.md`: TreeSearch-inspired retrieval over evidence.
- `references/display-items-workflow.md`: figure/table/box planning and human display-item checkpoint.
- `references/integrity-and-review-gates.md`: citation verification, claim audit, peer-review simulation, and revision checklist.

Use web or local repository search when current facts matter, when a specific paper/journal/guideline is referenced, or when the user asks for latest literature.

## Minimal Command Index

Set the script root once and keep detailed command chains in `references/scripted-literature-management.md`:

```powershell
$SKILL_DIR = "<skill-dir>"
```

Core entry points only:

```powershell
python "$SKILL_DIR/scripts/project_bootstrap.py" check-env --project-dir .
python "$SKILL_DIR/scripts/path_schema.py" check --project-dir .
python "$SKILL_DIR/scripts/draft_memory.py" build --draft-dir ./review-data/01_inputs/drafts_raw --topic "<topic>" --out-dir ./review-data/03_framework/draft_memory
python "$SKILL_DIR/scripts/draft_citation_assets.py" build --draft-dir ./review-data/01_inputs/drafts_raw --topic "<topic>" --out-dir ./review-data/02_literature/draft_assets
python "$SKILL_DIR/scripts/pubmed_user_set_ingestor.py" parse --input ./review-data/01_inputs/drafts_raw/abstract-<set>.txt --input ./review-data/01_inputs/drafts_raw/summary-<set>.txt --topic "<topic>" --set-name "<set-name>" --out-dir ./review-data/02_literature/user_pubmed_sets/<set-name>
python "$SKILL_DIR/scripts/literature_candidate_board.py" build --project-dir . --out-dir ./review-data/02_literature/candidate_board
python "$SKILL_DIR/scripts/draft_literature_discovery_orchestrator.py" plan --draft-dir ./review-data/01_inputs/drafts_raw --draft-assets-dir ./review-data/02_literature/draft_assets --topic "<topic>" --out-dir ./review-data/02_literature/literature_discovery
python "$SKILL_DIR/scripts/draft_reference_verifier.py" verify --candidate-csv <candidate-csv> --out-dir ./review-data/05_audit/<verification-run> --email <email> --workers 4 --resume
python "$SKILL_DIR/scripts/draft_logic_framework.py" --draft-dir ./review-data/01_inputs/drafts_raw --topic "<topic>" --target-journal "<target-journal>" --out-dir ./review-data/03_framework/logic_framework
python "$SKILL_DIR/scripts/display_item_planner.py" plan --project-dir . --framework-dir ./review-data/03_framework/logic_framework --pool-dir ./review-data/02_literature/pool --out-dir ./review-data/03_framework/display_items --target-journal "<target-journal>"
python "$SKILL_DIR/scripts/workflow_gatekeeper.py" check --project-dir . --stage pre-draft --fail-on-blocker
python "$SKILL_DIR/scripts/cover_letter_builder.py" --manuscript <manuscript-path> --target-journal "<target-journal>" --article-type "Review" --out-dir ./review-output/cover_letter
```

For lane-specific verification, adjudication, official citation export, supplemental recall execution, full text, RAG, display-item evidence packs, and agent-memory commands, read `references/scripted-literature-management.md`.

For optional approved external LLM acceleration, read `references/multi-agent-orchestration.md` and run external `mine`/`adjudicate` only after data egress is allowed.

## Output Standard

For planning, return a review blueprint: scope, review type, target journal fit, thesis candidates, inclusion/exclusion criteria, search plan, and expected article structure.

For architecture synthesis from drafts, return: thesis candidates, recurring and unique frames, argument-evidence map location, proposed 3-5 major section spine, headings to merge/delete, abstract risks, display-item plan, and user alignment questions before drafting.

For recall, return: search log, raw/deduped counts, included/excluded/skipped list with reasons, and a compact evidence matrix.

For drafting, return manuscript prose with citation placeholders only where verified sources are missing, plus a short material-gaps list.

For cover letters, return: concise editor-facing letter, one-page brief, missing fields, related-manuscript/conflict disclosures, suggested referees, excluded-referee rationale, and a checklist confirming that the letter is short, specific, respectful, and not a duplicate abstract.

For audits, lead with blocking issues and file locations when available.
