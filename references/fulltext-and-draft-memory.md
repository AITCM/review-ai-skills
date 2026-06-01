# Full Text And Draft Memory

Use this reference when the review needs detailed evidence support, TreeRAG over full text, user-supplied PDFs, OCR/Markdown parsing, or careful reuse of GPT/Gemini/Deep Research drafts.

## Quick Navigation

- Principle: keep draft memory and literature full-text memory separate but linked.
- Draft-reading gate: build durable draft memory before framework discussion.
- Full-text priority: decide which verified papers need full text before detailed claims.
- Europe PMC and user handoff: fetch OA XML first, then request missing PDFs from the user.
- MinerU/OCR and TreeRAG: parse PDFs, build structured RAG, and limit claims without full text.
- Agent orchestration: make full-text status visible to literature, framework, and synthesis agents.

## Principle

A top-journal review needs two persistent memory layers:

- Draft memory: what the user's initial drafts actually argued, proposed, cited, and overlooked.
- Literature full-text memory: what verified papers actually say in their methods, results, limitations, and discussion.

Do not let either layer evaporate across turns. Drafts guide framework discussion; verified full text supports manuscript claims.

## Chief Editor Draft-Reading Gate

Before framework discussion, run:

```bash
python $SKILL_DIR/scripts/draft_memory.py build \
  --draft-dir ./review-data/01_inputs/drafts_raw \
  --topic "review topic" \
  --out-dir ./review-data/03_framework/draft_memory
```

Outputs:

- `draft_memory_index.md`: compact index of draft cards and cross-draft signals.
- `chief_editor_reading_checkpoint.md`: mandatory checkpoint for the current Codex conversation before delegation, framework synthesis, or drafting.
- `cards/*.md`: one memory card per draft.
- `agent_reading_packet.md`: condensed packet for draft-reading and architecture agents.
- `claims_for_discussion.csv`: draft-derived claims to verify, preserve, demote, or delete.
- `draft_memory.sqlite`: searchable memory for later discussion.

Use rule:

- The current Codex conversation is the chief editor/PI and must personally read `chief_editor_reading_checkpoint.md`, `draft_memory_index.md`, `agent_reading_packet.md`, and the relevant per-draft cards before spawning or relying on subagents.
- Load `draft_memory_index.md` and relevant cards before discussing the framework.
- Treat draft claims as scaffolds, not evidence.
- Convert unsupported draft claims into literature tasks.
- Ask the user which draft ideas are must-keep or out-of-scope before drafting.

Subagents can parallelize careful reading, disagreement, evidence mapping, and critique, but they cannot replace the chief Codex reading pass. The chief editor should maintain a visible task board through the blackboard/passport: draft reading done, citation assets built, framework choices pending, user decisions pending, and next subagent packets.

## Full-Text Acquisition Priority

For selected papers in the citation pool, use this priority:

1. Europe PMC Open Access `fullTextXML`: best for biomedical papers because it gives structured XML that can be converted into Markdown for TreeRAG.
2. Existing local full text in the pool or user-supplied files.
3. `paper-search`/OA download outputs.
4. MinerU parsing for PDFs, scanned PDFs, or complex layouts that need OCR/table/formula extraction.
5. User handoff: if full text still cannot be obtained, generate a clear list and ask the user to supply the files.

Do not build detailed method/result/metric comparisons from abstracts alone unless the user explicitly approves abstract-only treatment for that claim.

## Europe PMC FullTextXML

Run:

```bash
python $SKILL_DIR/scripts/fulltext_manager.py fetch-europepmc \
  --pool-dir ./review-data/02_literature/pool \
  --status citation_pool,seminal,method,recent
```

The script:

- finds PMCID from pool metadata or Europe PMC search by DOI/PMID/title
- calls Europe PMC `/{PMCID}/fullTextXML`
- saves raw XML as `evidence/<key>/europepmc_fulltext.xml`
- converts JATS sections to `europepmc_fulltext.md`
- updates `pool.json` with `fulltext_text_path`, `fulltext_xml_path`, `pmcid`, and provenance

Reference: https://europepmc.org/RestfulWebService

## User Full-Text Handoff

After automatic attempts, run:

```bash
python $SKILL_DIR/scripts/fulltext_manager.py audit \
  --pool-dir ./review-data/02_literature/pool
```

Outputs:

- `contexts/fulltext_audit.md`: readiness summary.
- `contexts/needs_user_fulltext.md`: user-facing list of missing full text.
- `contexts/missing_fulltext_manifest.csv`: machine-readable missing list.
- `review-data/01_inputs/user_fulltext/`: folder where the user should place PDFs, TXT, MD, or DOCX before import.

When missing files remain, tell the user exactly which PDFs/full texts to add to `review-data/01_inputs/user_fulltext`. Do not bury this as a minor note.

After the user supplies files:

```bash
python $SKILL_DIR/scripts/fulltext_manager.py import \
  --pool-dir ./review-data/02_literature/pool \
  --source-dir ./review-data/01_inputs/user_fulltext
```

This matches files by DOI, citation key, or title tokens, copies them to the paper's evidence folder, extracts text when possible, and updates `pool.json`.

## MinerU OCR / Markdown Parsing

Use MinerU when a selected paper has a PDF but no usable extracted text, especially scanned PDFs, Chinese/English mixed layouts, tables, formulas, or two-column PDFs.

```bash
python $SKILL_DIR/scripts/fulltext_manager.py mineru-agent \
  --pool-dir ./review-data/02_literature/pool \
  --status citation_pool,seminal,method,recent \
  --language ch \
  --is-ocr \
  --enable-table \
  --enable-formula
```

The script uses the MinerU Agent lightweight API to submit local PDFs or URLs, polls until parsing is complete, downloads Markdown, saves `evidence/<key>/mineru_fulltext.md`, and updates `pool.json`.

If the Agent lightweight limit is insufficient for a large paper, use the precise MinerU API workflow from the official docs and then import the resulting Markdown with `fulltext_manager.py import`.

Reference: https://mineru.net/apiManage/docs

## TreeRAG Gate

Only build TreeRAG after full-text audit is acceptable:

```bash
python $SKILL_DIR/scripts/structured_lit_rag.py build \
  --pool-dir ./review-data/02_literature/pool
```

Before manuscript drafting, run:

```bash
python $SKILL_DIR/scripts/workflow_gatekeeper.py check \
  --project-dir . \
  --stage pre-draft
```

If the report says `rag_not_built`, `fulltext_handoff_missing`, or `user_fulltext_needed`, stop and present the handoff to the user. A manuscript may still be drafted as a clearly marked provisional/abstract-only version only when the user explicitly asks for that tradeoff.

The RAG index includes:

- literature cards
- saved abstracts
- Europe PMC XML-derived Markdown
- MinerU Markdown
- user-supplied TXT/MD/DOCX-derived text

When a RAG hit is used for writing, open the source file, verify the paper identity, and record claim support or limitation in the literature card/evidence matrix.

## Agent Orchestration

Before manuscript prose, the following agents should participate in framework discussion:

- `draft_deep_reader`: makes sure the original drafts were actually read.
- `draft_memory_curator`: keeps accepted/rejected/pending draft ideas persistent.
- `literature_strategist`: turns draft/framework claims into search and full-text tasks.
- `argument_builder`: creates claim-evidence-reasoning chains from verified materials.
- `outline_architect`: compresses the review into one thesis and 3-5 major movements.
- `framework_dialogue_moderator`: turns agent outputs into a user-facing decision discussion.

Run a framework-only packet:

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
