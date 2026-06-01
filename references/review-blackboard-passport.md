# Review Blackboard And Material Passport

Use this reference when a review project has multiple drafts, many literature artifacts, or specialist agents.

## Quick Navigation

- Borrowed pattern: use phase boundaries, checkpoints, provenance, and narrow agent roles.
- Blackboard: shared thesis, claims, evidence, disputes, and decisions.
- Chief editor task board: track next actions without losing the architecture.
- Material passport: summarize every major artifact's provenance and verification status.
- Checkpoints and phase boundaries: promote only user-approved, verified decisions.

## Borrowed Pattern

The academic-research-skills project separates orchestration, literature strategy, structure architecture, argument building, state tracking, integrity gates, and material passports. The design lesson is not to copy its files verbatim, but to preserve its contracts:

- every agent has a narrow phase boundary
- user-facing decisions become checkpoints
- literature corpus is handled with the same criteria as external search
- artifacts carry provenance and verification status
- a material passport can act as a reset boundary across long sessions
- per-agent memory can preserve role-specific handoffs without overloading every packet
- reviewers/auditors should see verified material, not raw untrusted drafts only

## Blackboard

The blackboard is the shared working surface for the framework stage. It should answer:

- What is the candidate central thesis?
- What are the 3-5 candidate first-level sections?
- Which claim supports each section?
- Which papers or reference leads support each claim?
- Which claims still need recall, source audit, or demotion?
- What has the user approved?

Required file:

```text
review-data/03_framework/logic_framework/framework_evidence_blackboard.md
```

The blackboard is not final prose. It is a decision surface.

Agent memory is not the blackboard. Use `review-data/06_agent_memory` for role-specific handoffs and working state; promote only reviewed decisions into the blackboard or passport.

## Chief Editor Task Board

Codex should keep a short task board, either in the conversation plan or in the blackboard, whenever a review project starts from drafts. This task board is the operational version of the goal state:

- `draft_reading`: chief Codex read `chief_editor_reading_checkpoint.md`, `draft_memory_index.md`, `agent_reading_packet.md`, and relevant cards
- `draft_assets`: draft citation assets built before supplemental recall
- `source_verification`: multi-signal PubMed/DOI/title/source verification complete or blocked
- `source_adjudication`: LLM/subagent adjudication merged; only accepted verified rows may enter the main pool
- `framework`: central thesis and 3-5 section spine pending/approved/rejected
- `display_items`: figure/table/box checkpoint pending/approved
- `fulltext_rag`: key papers have full text or user handoff is open
- `final_audit`: citation traceback and claim-support audit pending/complete

The task board is not evidence. It prevents the project from losing state across long sessions and makes sure subagents are spawned only after the chief Codex pass has identified scoped work.

## Material Passport

The material passport records what the project is allowed to rely on.

Required file:

```text
review-data/03_framework/logic_framework/framework_material_passport.json
```

Minimum fields per material:

- `material_id`
- `kind`
- `path`
- `version_label`
- `data_access_level`: `raw`, `redacted`, or `verified_only`
- `verification_status`: `unverified`, `needs_user_approval`, `verified`, `stale`, or `rejected`
- `produced_by`
- `consumed_by`
- `notes`

Use raw drafts as `raw/unverified`. Use framework briefs as `redacted/needs_user_approval`. Use final citation gates and verified literature cards as `verified_only/verified`.

## Checkpoint Rule

Before drafting, the passport must show:

- chief-editor draft-reading checkpoint exists and has been consulted by Codex
- framework checkpoint exists
- user has approved one central thesis or requested a redo
- first-level section count is within the agreed architecture
- each section has at least one candidate evidence stream or an explicit search task
- major claims without evidence have been demoted, softened, or converted into recall tasks

If the checkpoint is pending, do not write the full manuscript. Return the blackboard and ask for a decision.

## Agent Phase Boundaries

- `draft_frame_reader` reads raw drafts and extracts frames; it must not write the final outline.
- `literature_strategist` designs recall and screening; it must not invent the central thesis.
- `literature_retriever` fetches/validates sources; it must not decide final narrative.
- `argument_builder` builds CER chains; it must not draft sections.
- `outline_architect` proposes section spine; it must not ignore evidence gaps.
- `framework_devils_advocate` attacks the chosen frame; it must not rewrite the whole paper.
- `blackboard_curator` records state and pending decisions; it must not perform substantive synthesis.
- Codex remains chief editor and resolves conflicts.

## Literature-First Framework Loop

Run this loop until the framework is defensible:

1. Read drafts and extract claims.
2. Convert claims to literature tasks.
3. Recall/verify core sources.
4. Bind claims to evidence or demote them.
5. Propose 2-3 architecture spines.
6. Attack the spines for novelty, evidence sufficiency, and catalogue drift.
7. Ask the user to choose or revise the spine.

Only then draft.
