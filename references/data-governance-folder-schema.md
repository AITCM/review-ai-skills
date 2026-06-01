# Data Governance Folder Schema

Use this reference when creating or reorganizing a long-running review project. The goal is to keep raw materials, governed evidence, working artifacts, final outputs, and frozen snapshots separate enough that agents can reason about provenance.

## Quick Navigation

- Naming principles: canonical ASCII paths, immutable raw inputs, governed derived assets.
- Canonical folders: `review-data`, `review-work`, `review-output`, `review-archive`, and optional `src`.
- Data layout: where inputs, literature, framework, manuscript, audit, and agent memory live.
- Path schema script: check or initialize aliases before running bundled tools.
- Legacy handling: tolerate old display aliases but write new outputs to canonical paths.

## Naming Principles

- Use ASCII folder names for canonical paths to avoid CLI, encoding, and cross-platform issues.
- Use numeric prefixes only for stable ordering, not for workflow status.
- Keep user/raw materials immutable after import; write derived artifacts elsewhere.
- Treat `review-data` as the source-of-truth area and `review-work` as regenerable.
- Keep deliverables in `review-output`.
- Keep dated freezes in `review-archive`.
- `文献池` is allowed as a legacy/display alias, but new projects should use `review-data/02_literature/pool`.

## Canonical Top-Level Folders

```text
review-data/      governed persistent research assets
review-work/      regenerable work products, agent packets, scratch outputs
review-output/    manuscript-facing deliverables
review-archive/   frozen dated snapshots
src/              optional local Python package
.venv/            optional project virtual environment
```

## Canonical Data Layout

```text
review-data/
  00_project/
    folder_schema.md
  01_inputs/
    drafts_raw/
    user_fulltext/
  02_literature/
    draft_assets/
      cleaned_drafts/
      draft_citation_health.md
      draft_reference_inventory.csv
      claim_evidence_map.csv
    pool/
      pool.json
      recall_runs/
      evidence/
      cards/
      contexts/
      downloads/
      indexes/
      logs/
    supplemental_pool/
      pool.json
      recall_runs/
      cards/
      contexts/
      logs/
  03_framework/
    draft_memory/
    logic_framework/
    display_items/
  04_rag/
  05_audit/
  06_agent_memory/
    shared/
    agents/
review-work/
  recall_runs/
  agent_orchestration/
  logs/
review-output/
  manuscript/
  figures/
  tables/
  cover_letter/
  submission/
review-archive/
```

## Folder Semantics

- `review-data/01_inputs/drafts_raw`: original user/GPT/Gemini/Deep Research drafts. Preserve as received.
- `review-data/01_inputs/user_fulltext`: user-supplied PDFs/TXT/MD/DOCX before import.
- `review-data/02_literature/draft_assets`: first-pass citation assets extracted from the drafts themselves. This is not a recall pool; use it to inspect reference health, in-text markers, claim-evidence clues, non-academic sources, and missing evidence before searching.
- `review-data/02_literature/pool`: the governed literature pool and citation-pool memory. Pass this path as `--pool-dir`.
- `review-data/02_literature/supplemental_pool`: strictly screened supplemental papers found after user/framework gap confirmation. Promote records to the main pool only after relevance, publication status, and claim fit are checked.
- `review-data/03_framework/draft_memory`: persistent memory cards and SQLite index from initial drafts.
- `review-data/03_framework/logic_framework`: framework blackboard, argument map, material passport, and user-alignment checkpoint.
- `review-data/03_framework/display_items`: figure/table/box blueprints, image prompts, table evidence gaps, and human display-item checkpoint.
- `review-data/04_rag`: RAG indexes, retrieval traces, and claim lookup outputs.
- `review-data/05_audit`: citation, claim, source, full-text, and data-lineage audits.
- `review-data/06_agent_memory`: shared and per-agent persistent state for literature, framework, figure/table, synthesis, and audit agents.
- `review-work`: temporary or repeatable run outputs. It can be regenerated from `review-data`.
- `review-output`: files intended for human reading, journal submission, or export.

## Path Schema Script

Check a project:

```bash
python $SKILL_DIR/scripts/path_schema.py check --project-dir .
```

Create canonical folders and path alias files:

```bash
python $SKILL_DIR/scripts/path_schema.py init --project-dir .
```

Print the schema and write Markdown:

```bash
python $SKILL_DIR/scripts/path_schema.py show \
  --project-dir . \
  --markdown ./review-data/00_project/folder_schema.md
```

The script writes:

- `review_path_aliases.json`: machine-readable aliases and legacy path status.
- `review-data/00_project/folder_schema.md`: human-readable folder contract.

## Default Command Paths

Prefer these paths in new work:

```powershell
$SKILL_DIR = "<skill-dir>"

python "$SKILL_DIR/scripts/draft_memory.py" build \
  --draft-dir ./review-data/01_inputs/drafts_raw \
  --out-dir ./review-data/03_framework/draft_memory

python "$SKILL_DIR/scripts/draft_logic_framework.py" \
  --draft-dir ./review-data/01_inputs/drafts_raw \
  --out-dir ./review-data/03_framework/logic_framework

python "$SKILL_DIR/scripts/literature_pool.py" init \
  --pool-dir ./review-data/02_literature/pool

python "$SKILL_DIR/scripts/fulltext_manager.py" audit \
  --pool-dir ./review-data/02_literature/pool

python "$SKILL_DIR/scripts/structured_lit_rag.py" build \
  --pool-dir ./review-data/02_literature/pool

python "$SKILL_DIR/scripts/display_item_planner.py" plan \
  --project-dir . \
  --framework-dir ./review-data/03_framework/logic_framework \
  --pool-dir ./review-data/02_literature/pool \
  --out-dir ./review-data/03_framework/display_items

python "$SKILL_DIR/scripts/agent_memory.py" init \
  --project-dir . \
  --agents all
```

## Legacy Path Handling

Existing projects may contain:

- `文献池`
- `review-work/literature_pool`
- `review-work/draft_memory`
- `review-work/draft_logic_framework`
- `review-work/cover_letter`

Do not delete or move these automatically. Treat them as legacy inputs, inspect contents, and only migrate after the user approves a backup/migration plan.

Recommended mapping:

```text
文献池                         -> review-data/02_literature/pool
review-work/literature_pool    -> review-data/02_literature/pool
review-work/draft_memory       -> review-data/03_framework/draft_memory
review-work/draft_logic_framework -> review-data/03_framework/logic_framework
review-work/cover_letter       -> review-output/cover_letter
```

## Agent Rule

Before a multi-agent run, provide canonical paths in the packet. If an agent sees both canonical and legacy paths, it must prefer canonical outputs and mention legacy inputs only as provenance or migration candidates. Agent memory belongs in `review-data/06_agent_memory` and should be loaded dynamically by role; do not copy full papers, API keys, or raw draft files into memory.
