# ReviewAI Skills

**Evidence-governed AI skills for top-journal literature reviews.**

`review-ai-skills` is an AI-agent skill package for turning GPT/Gemini/Deep Research drafts into auditable, literature-grounded review manuscripts.

It is designed for researchers who already have one or more AI-generated drafts, PubMed exports, notes, or earlier manuscript versions, and need a governed workflow for framework discussion, literature verification, multi-agent screening, full-text/RAG handoff, evidence-grounded figures and tables, citation auditing, and cover-letter preparation.

ReviewAI Skills treats the AI assistant as a **chief editor plus specialist-agent orchestra**, not a one-shot manuscript generator. Drafts provide the starting logic; verified papers, official citations, full text, and human checkpoints determine what survives into the final review.

## What It Does

- Reads AI-generated drafts into persistent draft memory before writing.
- Extracts draft-native citation assets and separates paper leads from webpages/blogs.
- Orchestrates literature-discovery packets for Codex subagents.
- Verifies candidate papers through PubMed, Crossref/OpenAlex, arXiv/OpenReview, and publisher/conference sources.
- Ingests user-provided PubMed Summary/Abstract exports as screened seed sets.
- Builds candidate boards, governed literature pools, literature cards, and TreeRAG-style evidence indexes.
- Fetches full text through Europe PMC/OA/user handoff workflows.
- Plans evidence-grounded figures, tables, boxes, and GPT image-ready figure prompts.
- Guards against workflow/audit language leaking into the manuscript body.
- Renumbers numeric citations by first appearance and rebuilds final references from official citation exports.
- Prepares concise Nature-style cover-letter materials.

## Quick Install With An AI Agent

Give this repository link to your coding agent:

```text
https://github.com/AITCM/review-ai-skills
```

Then ask the agent:

```text
Install ReviewAI Skills from https://github.com/AITCM/review-ai-skills as a local AI-agent skill named review-ai-skills. After installing, validate the skill and show me the activation command.
```

This works well with Codex, Claude Code, Cursor, VS Code AI coding agents, and other tools that can clone a GitHub repository and place reusable instructions in a local skills or prompts directory.

## Install Manually For Codex

Clone or copy this repository into your Codex skills directory:

```powershell
git clone https://github.com/AITCM/review-ai-skills "$HOME\.codex\skills\review-ai-skills"
```

Then use the skill in Codex:

```text
$review-ai-skills
```

For other platforms, keep the repository as a reusable instruction/tool folder and point the agent to `SKILL.md`. If your tool supports persistent project rules, add a short rule such as: "Use ReviewAI Skills for draft-first literature review projects, evidence governance, citation auditing, and top-journal review workflows."

## Recommended Workflow

1. Put raw GPT/Gemini/Deep Research drafts in `review-data/01_inputs/drafts_raw`.
2. Build draft memory and citation assets.
3. Let the chief Codex pass read all drafts and generate a provisional thesis, claim map, and 3-5 section spine.
4. Run literature-discovery packets and candidate-board screening before broad recall.
5. Verify/adjudicate paper identities and export official citations.
6. Build the governed literature pool, full-text handoff, cards, and RAG index.
7. Discuss the framework and display items with the user.
8. Draft only from verified evidence and explicit gaps.
9. Run narrative guard, citation-sequence manager, claim audit, and final gates.

## Key Scripts

- `scripts/draft_memory.py`
- `scripts/draft_citation_assets.py`
- `scripts/draft_literature_discovery_orchestrator.py`
- `scripts/literature_candidate_board.py`
- `scripts/draft_reference_verifier.py`
- `scripts/official_citation_exporter.py`
- `scripts/literature_pool.py`
- `scripts/fulltext_manager.py`
- `scripts/structured_lit_rag.py`
- `scripts/display_item_planner.py`
- `scripts/manuscript_narrative_guard.py`
- `scripts/citation_sequence_manager.py`
- `scripts/workflow_gatekeeper.py`

Run the built-in audit:

```bash
python scripts/skill_audit.py --skill-dir .
```

## MCP Prototype

The `mcp/review_literature_mcp` folder contains a wrapper-style MCP prototype for exposing the literature and review workflow as tools/resources. The intended architecture is:

- Skill: editorial doctrine and routing policy.
- Scripts: deterministic backend commands.
- MCP/review engine: persistent state, queues, APIs, and resources.
- Codex: chief editor/PI.

## Secret Policy

Do not commit real API keys. External LLM acceleration is optional; Codex subtask packet mode is the default. Keep `DEEPSEEK_API_KEY`, `REVIEW_AGENT_API_KEY`, and related variables in your shell, secret manager, or local `.env.local` files excluded by `.gitignore`.

## Disclaimer

This project supports scientific writing and literature governance. It does not replace human expert review, journal policy checks, clinical judgment, legal review, or research ethics oversight.

## License

MIT License.
