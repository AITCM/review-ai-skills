# Multi-Agent Orchestration

Use this workflow when a review project needs more than linear drafting. Codex acts as the chief editor/PI and coordinates specialist agents.

## Quick Navigation

- Agent roles: specialist responsibilities and output contracts.
- Chief Codex first: mandatory draft-reading checkpoint before delegation.
- Codex-subtask mode: default path; create packets and delegate bounded decisions to Codex subtasks.
- External LLM mode: optional DeepSeek/OpenAI-compatible acceleration only when data egress is explicitly approved.
- Recommended passes: memory, framework, evidence strategy, retrieval, synthesis, safety, and final review.
- Final citation traceback: claim-source audit before final delivery.
- Literature-agent guardrails: draft-first, claim-bound retrieval boundaries.
- Codex as chief editor: how Codex accepts, rejects, and integrates agent findings.

## Agent Roles

- `draft_deep_reader`: reads draft memory cards and raw draft packets carefully so Codex does not underuse the user's starting drafts.
- `draft_memory_curator`: keeps accepted, rejected, and pending draft ideas in a persistent memory layer.
- `draft_frame_reader`: reads multiple AI-generated drafts as untrusted scaffolds, extracts consensus frames, unique insights, thesis candidates, and headings to merge/delete.
- `draft_literature_candidate_miner`: reads draft body prose and extracts hidden/partial paper candidates, named systems, venue clues, title fragments, and narrow PubMed deep-dive queries before deterministic verification.
- `literature_strategist`: converts framework claims into corpus-first/search-fills-gap recall plans and evidence sufficiency criteria before outline approval.
- `literature_retriever`: finds missing evidence and replaces web/blog claims with scholarly sources.
- `recall_supervisor`: reads the recall pool, backlog, deferred rows, quarantine sample, and verification outcomes; proposes query refinements, PubMed deep dives, lane promotions, duplicate merges, and rescue candidates before more API calls are launched.
- `pubmed_query_strategist`: converts agent-observed system names, partial titles, preprint records, and framework lineage gaps into narrow PubMed title/PMID/DOI/bibliographic queries, then checks whether PubMed is underused relative to the biomedical scope.
- `user_pubmed_set_screener`: reads user-supplied PubMed Summary/Abstract seed sets, separates core/support/framework/background/off-frame papers, and flags full-text needs before deterministic verification.
- `candidate_board_curator`: reads the candidate board, identifies screening backlog, verifier-ready rows, repair/delete rows, verified rows needing citation/full text, and whether additional recall is justified.
- `literature_screener`: decides include/exclude/background/preprint-only based on scope and evidence hierarchy.
- `literature_manager`: checks the canonical literature pool, selected citation-pool records, cards, full text, RAG, and handoff lists.
- `citation_verifier`: performs final citation traceback and claim-source fit audit.
- `outline_architect`: first reads multi-draft logic-framework outputs, then builds central thesis, section logic, and top-journal contribution frame.
- `argument_builder`: creates claim-evidence-reasoning chains, counterarguments, and argument-strength scores.
- `framework_devils_advocate`: challenges the proposed architecture for frame-lock, catalogue drift, weak novelty, and evidence voids.
- `framework_dialogue_moderator`: turns draft memory, literature strategy, and argument checks into a user-facing framework discussion.
- `blackboard_curator`: maintains framework blackboard/passport state and pending user decisions.
- `figure_table_designer`: proposes evidence-grounded figures, boxes, taxonomies, and tables; it must expose evidence gaps, GPT image prompt readiness, post-generation audit needs, and human decisions before image generation or drafting.
- `synthesis_writer`: turns verified sources into consensus, controversies, mechanisms, gaps, and future agenda.
- `methods_reporting_editor`: aligns methods/reporting with PRISMA-style transparency and AI reporting standards.
- `ethics_regulatory_agent`: audits safety, governance, regulation, AI-use disclosure, and domain-specific risks.
- `reviewer_auditor`: simulates editor and peer reviewers and produces a revision roadmap.

## Persistent Agent Memory

## Chief Codex First

Before any multi-agent run that affects the framework, the current Codex conversation must complete the chief-editor draft-reading checkpoint:

```text
draft_memory.py build
  -> read chief_editor_reading_checkpoint.md
  -> read draft_memory_index.md
  -> open relevant per-draft cards
  -> run and inspect draft_citation_assets.py build outputs
  -> parse and inspect user PubMed seed-set outputs when provided
  -> build/read literature_candidate_board.py outputs
  -> state provisional thesis/claim map and citation lanes
  -> run draft_literature_candidate_miner packet for prose-level paper clues
  -> only then spawn/rely on subagents
```

Codex subtasks are parallel specialists, not the paper's brain. They can read draft packets, challenge the architecture, adjudicate citations, and audit claims, but the chief Codex conversation must own the task plan, blackboard state, user-facing questions, and final framework decision.

The chief editor must also state what the drafts are trying to say before sending literature agents to search. Literature agents should receive a bounded argument map, not an open-ended topic prompt. Their job is to support, falsify, historicize, or narrow those arguments.

If a script generated subagent packets but Codex did not actually spawn/send them to specialist subtasks, the run is not a multi-agent run. Record it as `local_only_fallback`, and do not proceed to polished drafting until either a small packet batch is completed or the user explicitly waives subagent screening. `workflow_gatekeeper.py` flags packet directories without collected CSV outputs.

For literature recall, do not let deterministic API output and subagent reasoning become separate worlds. After each broad recall merge, run a recall-supervision pass so Codex/subagents can read pool statistics, top backlog rows, deferred rows, and quarantine samples before deciding which PubMed deep dives or rescue screens should happen next. PubMed should be treated as a biomedical identity and abstract source, not merely as one source among many; if Crossref/OpenAlex dominate the pool, the recall supervisor must ask which high-value rows need PubMed title/PMID/DOI follow-up.

The recall supervisor must preserve task diversity before any API run. Do not let formal-version exact-title searches crowd out discovery. A healthy PubMed-supervision batch should include PubMed deep dives, formal-version checks, framework/history/method foundations, governance/validation queries, and a small quarantine-rescue sample. If candidate-derived deep dives are sparse, add a bounded strategic PubMed supplement from the confirmed framework, such as biomedical AI agents, single-cell/transcriptomics agents, drug-discovery agents, CRISPR/gene-editing agents, protein-design agents, autonomous scientific discovery, and clinical validation/governance. Broad strategic hits without an agent/LLM/autonomous title signal remain screening candidates, not evidence.

Use Codex subtasks for bounded parallel work:

- one subtask can mine candidate papers from `draft_literature_candidate_miner.py packet`
- one subtask can supervise recall-pool backlog/deferred/quarantine rows and produce `recall_supervision_seed_decisions.csv`
- one subtask can screen `review-data/02_literature/user_pubmed_sets/<set-name>/screening_packets/` and produce `agent_screening_decisions.csv`
- one subtask can read `review-data/02_literature/candidate_board/literature_candidate_board.md` and produce a next-action triage memo before more recall
- one subtask can challenge the proposed section spine
- one subtask can adjudicate a citation batch from `draft_source_adjudicator.py build-cases`
- one subtask can inspect figure/table evidence gaps
- one subtask can audit final citation fit

Do not spawn subagents before the chief Codex pass has identified the concrete task boundary and the files each subagent should use.

Before a serious multi-agent run, initialize shared and per-agent memory:

```bash
python $SKILL_DIR/scripts/agent_memory.py init \
  --project-dir . \
  --agents all

python $SKILL_DIR/scripts/agent_memory.py append-shared \
  --project-dir . \
  --file handoff.md \
  --kind handoff \
  --source codex \
  --note "Current stage, blockers, and next files to load before the next agent run."
```

The canonical memory root is `review-data/06_agent_memory`. It stores shared project memory and one folder per specialist agent. The orchestrator dynamically loads shared memory plus the selected agent's own memory into each packet. This keeps literature, framework, figure/table, synthesis, and audit agents from losing prior decisions while avoiding a giant always-loaded context.

Memory is state, not evidence. If memory conflicts with the framework blackboard, material passport, literature pool, final citation gate, or current user instruction, the agent must flag the conflict and prefer the newer verified source.

## Codex-Subtask Mode

Use this by default. It creates prompts/task packets that Codex can inspect or delegate to Codex subtasks without configuring external API keys or sending draft text to third-party services:

```bash
python $SKILL_DIR/scripts/review_agent_orchestrator.py plan \
  --project-dir . \
  --out-dir ./review-work/agent_orchestration \
  --agent-memory-dir ./review-data/06_agent_memory \
  --goal "Optimize <topic> for <target-journal-family>"
```

When testing a framework board from another project or a deliberately narrowed evidence packet, use `--no-default-inputs` so the agent packets contain only files named with `--input`:

```bash
python $SKILL_DIR/scripts/review_agent_orchestrator.py plan \
  --project-dir . \
  --out-dir ./review-work/agent_orchestration_framework \
  --agent-memory-dir ./review-data/06_agent_memory \
  --no-default-inputs \
  --agents draft_deep_reader,draft_memory_curator,draft_frame_reader,literature_strategist,argument_builder,outline_architect,framework_devils_advocate,framework_dialogue_moderator,blackboard_curator \
  --input ./review-data/03_framework/draft_memory/draft_memory_index.md \
  --input ./review-data/03_framework/draft_memory/chief_editor_reading_checkpoint.md \
  --input ./review-data/03_framework/draft_memory/agent_reading_packet.md \
  --input ./review-data/03_framework/logic_framework/draft_logic_framework.md \
  --input ./review-data/03_framework/logic_framework/framework_evidence_blackboard.md \
  --input ./review-data/03_framework/logic_framework/argument_evidence_map.csv \
  --input ./review-data/03_framework/logic_framework/literature_search_tasks.csv \
  --input ./review-data/03_framework/logic_framework/framework_material_passport.json
```

Outputs:

- `agent_manifest.json`
- `agent_packets/*.md`

## Optional External LLM Mode

The orchestrator supports OpenAI-compatible Chat Completions endpoints as an optional accelerator. It can use DeepSeek `deepseek-v4-pro`, `https://api.deepseek.com`, thinking enabled, and high reasoning effort when external model calls and data egress are explicitly approved.

Before external calls, check configuration:

```bash
python $SKILL_DIR/scripts/review_agent_orchestrator.py verify-config
python $SKILL_DIR/scripts/project_bootstrap.py check-env --project-dir .
```

Do not ask for an API key by default. If the user explicitly requests external LLM acceleration and data egress is allowed, then check configuration. If no key is present, either remain in Codex-subtask mode or ask the user to configure `DEEPSEEK_API_KEY`/`REVIEW_AGENT_API_KEY` locally. Do not ask for a plaintext key by default and do not write keys to project files.

For DeepSeek-compatible usage, configure environment variables:

```powershell
$env:DEEPSEEK_API_KEY="..."
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:DEEPSEEK_MODEL="deepseek-v4-pro"
```

Or use generic names:

```powershell
$env:REVIEW_AGENT_API_KEY="..."
$env:REVIEW_AGENT_BASE_URL="https://api.deepseek.com"
$env:REVIEW_AGENT_MODEL="deepseek-v4-pro"
```

Then run selected agents:

```bash
python $SKILL_DIR/scripts/review_agent_orchestrator.py run \
  --project-dir . \
  --out-dir ./review-work/agent_orchestration \
  --agents literature_retriever,literature_screener,citation_verifier,reviewer_auditor \
  --goal "Optimize <topic>; enforce published-only final citations"
```

Results are saved in `agent_results/*.md`.

When `--record-agent-results` is enabled, external agent outputs are also summarized into that agent's `handoff.md`, `events.jsonl`, and `state.json`. Codex should still review the agent's `Memory updates to write back` before promoting decisions to shared memory or the framework blackboard.

## Recommended Passes

Pass -1, memory setup:

```text
draft_memory build -> chief Codex reading checkpoint -> agent_memory init -> shared memory update -> per-agent retrieval plans
```

Pass -0.5, draft-prose literature mining:

```text
draft_literature_candidate_miner packet -> Codex candidate-mining subtask -> human_ai_candidate_checkpoint -> PubMed deep-dive abstracts -> deterministic verifier -> Codex citation-adjudication subtask
```

Pass 0, framework blackboard:

```text
draft_logic_framework -> draft_frame_reader -> outline_architect -> framework_devils_advocate -> framework_dialogue_moderator -> user checkpoint
```

Pass 1, evidence strategy:

```text
literature_strategist reads the confirmed/near-confirmed framework plus AI-mined candidate checkpoint -> maps each claim to evidence needs -> separates history, current evidence, future agenda, counterevidence, and display-item needs -> reviews/edits argument_literature_expander tasks
```

Pass 2, evidence retrieval and screening:

```text
argument_literature_expander PubMed abstract packet -> literature_retriever for non-PubMed gaps -> literature_screener -> literature_manager -> argument_builder
```

Pass 3, architecture refinement:

```text
argument_builder -> blackboard_curator -> user checkpoint if evidence changes the thesis -> display_item_planner -> figure_table_designer -> synthesis_writer
```

Display-item planning command:

```bash
python $SKILL_DIR/scripts/display_item_planner.py plan \
  --project-dir . \
  --framework-dir ./review-data/03_framework/logic_framework \
  --pool-dir ./review-data/02_literature/pool \
  --rag-db ./review-data/02_literature/pool/indexes/lit_rag.sqlite \
  --out-dir ./review-data/03_framework/display_items
```

The `figure_table_designer` must return `display_item_inventory`, `figure_blueprints`, `table_blueprints`, `evidence_needed`, `human_decisions_needed`, `image_prompts_ready_for_review`, `gpt_image_generation_queue`, and `post_generation_audit_findings`. It must not invent visual claims, citations, numerical results, or final labels.

Pass 4, safety and compliance:

```text
methods_reporting_editor -> ethics_regulatory_agent -> citation_verifier
```

Pass 5, final review:

```text
reviewer_auditor -> citation_verifier
```

## Final Citation Traceback

Before final delivery:

1. Run all recall/verification sources: PubMed, Crossref/OpenAlex, and arXiv/OpenReview for leads.
2. Run `literature_pool.py final-gate --apply`.
3. Build/update cards and RAG.
4. Run `citation_verifier`.
5. Rewrite any sentence whose citation support is partial, indirect, preprint-only, or missing.

The citation verifier must check:

- reference exists and is published or official
- citation metadata matches DOI/PMID/publisher
- sentence claim does not exceed the paper's evidence
- preprints/OpenReview-only cases are not treated as established evidence
- web/blog sources are not used as scientific evidence

## Literature-Agent Guardrails

Literature agents must not search from the broad topic alone once drafts are available. Every retrieval task should name:

- the draft-derived claim or gap it serves
- whether the need is historical foundation, current evidence, method support, counterevidence, display-item support, or future-agenda grounding
- the allowed destination: draft-native verification, main pool, or supplemental pool
- the acceptance criteria for promotion to the main pool
- whether abstract-only evidence is sufficient for screening or whether full text is required before the claim can be written

If an agent finds interesting papers that do not serve the argument map, it records them as `background_or_defer` in the supplemental pool and does not let them change the manuscript structure.

## Codex As Chief Editor

Do not let external agents directly rewrite the manuscript without Codex review. Codex should:

- read and own the initial draft-memory checkpoint before delegation
- keep the visible task board and goal state aligned with the framework blackboard/passport
- choose which findings to accept
- maintain the framework blackboard/passport as the authoritative shared state
- maintain `review-data/06_agent_memory` as curated per-agent state, not as a citation source
- run local scripts and update the literature pool
- preserve the published-only citation policy
- resolve conflicts between agents
- perform the final human-readable synthesis
