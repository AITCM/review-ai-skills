# Agent Memory Workflow

Use this reference when a review project uses multiple specialist agents across several turns or days. The goal is progressive disclosure: each agent sees the shared project state plus only its own durable memory, while Codex remains the chief editor.

## Quick Navigation

- Memory layers: shared project memory, per-agent memory, retrieval plans, and event logs.
- Initialize: dry-run and create memory folders before long multi-agent work.
- Dynamic loading: decide which shared files, cards, RAG queries, and agent memories to load.
- Write back: update durable memory without treating it as verified evidence.
- Agent expectations and governance: role-specific state, conflict handling, and source priority.

## Memory Layers

- `review-data/06_agent_memory/shared/project_memory.md`: stable project facts, scope, target journal, and user preferences.
- `review-data/06_agent_memory/shared/decisions.md`: approved thesis, section spine, citation policy, display-item decisions, and rejected options.
- `review-data/06_agent_memory/shared/open_questions.md`: questions that require user input, literature retrieval, or later audit.
- `review-data/06_agent_memory/shared/style_and_scope.md`: tone, audience, section constraints, terminology, and journal-family expectations.
- `review-data/06_agent_memory/shared/handoff.md`: current stage, blockers, and files that future agents should load first.
- `review-data/06_agent_memory/agents/<agent>/profile.md`: role, focus, and dynamic-loading rule.
- `review-data/06_agent_memory/agents/<agent>/memory.md`: durable role-specific observations and decisions.
- `review-data/06_agent_memory/agents/<agent>/handoff.md`: what the next run of this exact agent must know first.
- `review-data/06_agent_memory/agents/<agent>/retrieval_plan.md`: files, cards, RAG queries, and blackboard sections this agent should load.
- `review-data/06_agent_memory/agents/<agent>/events.jsonl`: append-only recent events.
- `review-data/06_agent_memory/agents/<agent>/state.json`: lightweight status and last-result pointer.

Do not treat memory as verified evidence. It is state. When memory conflicts with the current framework blackboard, material passport, literature pool, final citation gate, or user instruction, flag the conflict and prefer the newer verified source.

## Initialize

Dry run first:

```bash
python $SKILL_DIR/scripts/agent_memory.py init \
  --project-dir . \
  --agents all \
  --dry-run
```

Create memory after user approval or when the user explicitly asks to initialize:

```bash
python $SKILL_DIR/scripts/agent_memory.py init \
  --project-dir . \
  --agents all
```

Initialize selected agents when the project is narrow:

```bash
python $SKILL_DIR/scripts/agent_memory.py init \
  --project-dir . \
  --agents outline_architect,figure_table_designer,citation_verifier
```

## Dynamic Loading

The orchestrator loads shared memory plus the selected agent's own memory into each packet:

```bash
python $SKILL_DIR/scripts/review_agent_orchestrator.py plan \
  --project-dir . \
  --out-dir ./review-work/agent_orchestration \
  --agent-memory-dir ./review-data/06_agent_memory \
  --agents outline_architect,figure_table_designer,citation_verifier
```

Use `--no-agent-memory` only for clean-room checks or when you suspect stale memory is biasing the agent.

To inspect one agent's memory packet directly:

```bash
python $SKILL_DIR/scripts/agent_memory.py packet \
  --project-dir . \
  --agent figure_table_designer \
  --out ./review-work/agent_orchestration/figure_table_designer_memory.md
```

## Write Back

Every agent prompt asks for `Memory updates to write back`. Codex should review those updates before accepting them.

Append a shared project note:

```bash
python $SKILL_DIR/scripts/agent_memory.py append-shared \
  --project-dir . \
  --file project_memory.md \
  --kind scope \
  --source user_alignment \
  --note "Project topic, target journal family, and user-approved workflow state."
```

Append a curated decision:

```bash
python $SKILL_DIR/scripts/agent_memory.py append \
  --project-dir . \
  --agent outline_architect \
  --kind decision \
  --source user_alignment \
  --note "Approved central thesis and 3-5 section spine."
```

Append a figure/table idea:

```bash
python $SKILL_DIR/scripts/agent_memory.py append \
  --project-dir . \
  --agent figure_table_designer \
  --kind figure \
  --source framework_blackboard \
  --note "Figure 1 should show knowledge base, multimodal perception, deliberative agent, and clinical governance as a closed loop."
```

Ingest an external agent's result into that agent's handoff memory:

```bash
python $SKILL_DIR/scripts/agent_memory.py ingest-result \
  --project-dir . \
  --agent reviewer_auditor \
  --result ./review-work/agent_orchestration/agent_results/reviewer_auditor.md
```

When `review_agent_orchestrator.py run` calls an external LLM and `--record-agent-results` is enabled, it writes the result excerpt to that agent's `events.jsonl`, `handoff.md`, and `state.json` automatically.

## Agent-Specific Memory Expectations

- Literature agents remember search gaps, include/exclude logic, source replacements, missing full text, and RAG readiness.
- Framework agents remember thesis options, approved 3-5 section spine, rejected headings, user decisions, and claims needing stronger evidence.
- Figure/table agents remember display-item inventory, visual claims, data needed, and which figures/tables were merged or rejected.
- Synthesis agents remember consensus/controversy patterns, evidence strength, and claims to demote.
- Audit agents remember blocking citation issues, claim-source fit failures, and final-gate risks.

## Governance Rules

- Keep secrets out of memory files.
- Keep raw papers and PDFs in the literature pool or user-fulltext folders, not agent memory.
- Keep memory short and curated. Prefer decisions, unresolved questions, and pointers to authoritative files over copied long excerpts.
- Promote accepted agent findings to the framework blackboard, material passport, literature pool, or manuscript only after Codex review.
- Archive stale or superseded memory in `review-archive` only after user approval; do not silently delete it.
