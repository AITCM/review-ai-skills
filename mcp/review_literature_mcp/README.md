# Review Literature MCP

Thin-wrapper MCP server for the `top-journal-review-writer` literature workflow.

The skill remains the operating doctrine. This MCP server exposes stable tools
that call the bundled deterministic scripts and return compact JSON summaries.

Default mode is local/Codex-subtask first:

- no external LLM key is required;
- draft text stays local unless the user explicitly approves external calls;
- PubMed/Crossref/OpenAlex/official publisher verification can be run through
  normal network-approved tooling;
- Codex subtasks handle paper-identity normalization and claim-fit reasoning.
- `collect_hidden_candidate_outputs` accepts `reference_candidates_csv` so
  subagent rows already present in draft-native citation assets are treated as
  reference recovery, not hidden discoveries.

## Prototype Tools

- `extract_draft_citation_assets`
- `ingest_user_pubmed_set`
- `collect_user_pubmed_screening`
- `build_literature_candidate_board`
- `build_candidate_board_agent_packets`
- `collect_candidate_board_agent_decisions`
- `build_hidden_candidate_packets`
- `collect_hidden_candidate_outputs`
- `build_identity_normalization_packets`
- `merge_normalized_identities`
- `verify_paper_candidates`
- `export_official_citations`
- `init_literature_pool`
- `import_verified_to_pool`
- `final_citation_gate`

## Local Use

Install MCP dependencies in the environment that runs the server:

```bash
pip install mcp
```

Run:

```bash
python "$SKILL_DIR/mcp/review_literature_mcp/server.py"
```

Then configure the local Codex MCP entry to launch this server with the target
review project as the working directory.
