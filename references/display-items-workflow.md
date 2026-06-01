# Display Items Workflow

Use this reference when designing figures, tables, boxes, graphical abstracts, or visual prompts for a top-journal review.

## Quick Navigation

- Principle: figures, tables, and boxes must serve the central argument.
- Human-in-the-loop rule: confirm roles, claims, evidence, and exclusions before drafting.
- Figure workflow: blueprint, GPT image prompt/brief, label audit, and professional refinement.
- Table workflow: evidence packs, RAG/full-text checks, gap list, and confirmation.
- Script outputs and agent role: use `display_item_planner.py` and figure/table memory.

## Principle

Display items are argument infrastructure. They should make the review's central thesis, evidence map, taxonomy, controversy, roadmap, or governance logic easier to reuse and cite. Do not add figures or tables as decoration.

Run display-item planning after the draft/framework stage and before drafting substantial prose:

```text
draft_memory -> draft_logic_framework -> literature recall/pool/cards/RAG -> display_items -> user checkpoint -> figure/table agent -> drafting
```

## Human-In-The-Loop Rule

Before writing manuscript sections around a figure/table or generating images, stop for human confirmation:

- keep / merge / delete / redesign each display item
- confirm the central claim the item supports
- confirm which citations may support each table row or visual claim
- mark missing full text, missing RAG, and unsupported cells
- approve image prompts as conceptual drafts, not final submission art

The checkpoint file is:

```text
review-data/03_framework/display_items/human_display_item_checkpoint.md
```

## Figure Workflow

Default strategy: concept-first, GPT-image-assisted, evidence-constrained.

1. Draft a figure blueprint: argument role, panels, labels, visual logic, caption skeleton.
2. Audit linked claims and evidence requirements.
3. Generate an image prompt and `gpt_image_generation_queue.csv` entry for Codex/GPT image generation only after the blueprint is coherent.
4. Stop at `human_display_item_checkpoint.md`; the user must approve keep/merge/delete/redesign and the prompt before image generation.
5. Use Codex/GPT image generation one figure at a time to create conceptual raster drafts for discussion.
6. Audit the generated draft against the figure role, labels, caption skeleton, forbidden items, and evidence status.
7. After user approval, refine in a professional tool or redraw manually if the target journal requires editable/vector art.

Generated images should live, when the runtime provides file artifacts, under:

```text
review-data/03_framework/display_items/generated_drafts/
```

Codex should use GPT image generation for concept exploration, not for final evidence creation. If a figure needs exact labels, equations, axes, table-like content, citation markers, or journal-ready vector output, generate a clean conceptual layout first and then redraw or edit manually.

Figure prompts must not contain:

- numerical results not already verified
- author names, paper titles, DOI strings, or citation markers
- unsupported superiority claims
- clinical claims stronger than the evidence
- journal or company logos
- dense paragraph text that should be typeset later

## Table Workflow

Default strategy: evidence-first.

1. Build table blueprints from the literature pool, literature cards, full text, and RAG.
2. Each row should be a verified paper, claim-evidence unit, taxonomy unit, or checklist item.
3. Use `[EVIDENCE GAP]` when a cell lacks verified evidence.
4. Use RAG to retrieve detailed methods/results/limitations only after full text or cards exist.
5. Keep candidate citations visible until `final-gate` and claim-source audit pass.
6. Do not fill table cells from the figure prompt or image. Tables are authored from evidence packs, not from generated visuals.

Useful table types:

- evidence table for core studies and systems
- taxonomy table for model/agent task types
- method comparison table
- validation/readiness ladder
- risk and governance matrix
- research agenda and open evidence gaps

## Script

Plan display items:

```bash
python $SKILL_DIR/scripts/display_item_planner.py plan \
  --project-dir . \
  --framework-dir ./review-data/03_framework/logic_framework \
  --pool-dir ./review-data/02_literature/pool \
  --rag-db ./review-data/02_literature/pool/indexes/lit_rag.sqlite \
  --out-dir ./review-data/03_framework/display_items \
  --target-journal "Nature Reviews-style journal"
```

Build evidence packets for one or all items:

```bash
python $SKILL_DIR/scripts/display_item_planner.py evidence-pack \
  --display-items ./review-data/03_framework/display_items/display_items.json \
  --item all \
  --rag-db ./review-data/02_literature/pool/indexes/lit_rag.sqlite \
  --out ./review-data/03_framework/display_items/display_item_evidence_pack.md
```

Export reviewed conceptual image prompts:

```bash
python $SKILL_DIR/scripts/display_item_planner.py prompt-pack \
  --display-items ./review-data/03_framework/display_items/display_items.json \
  --item F1,F2,F3 \
  --out ./review-data/03_framework/display_items/reviewed_figure_prompt_pack.md
```

For GPT image generation, use the reviewed prompt pack together with:

```text
review-data/03_framework/display_items/gpt_image_generation_brief.md
review-data/03_framework/display_items/gpt_image_generation_queue.csv
```

In Codex, after human approval, call GPT image generation with one reviewed prompt at a time. Treat the result as a concept image for discussion, then run the post-generation audit before drafting around it.

## Outputs

- `display_item_plan.md`: human-readable inventory and checkpoint summary.
- `display_items.json`: machine-readable figure/table/box blueprints.
- `figure_blueprints.md`: figure argument, panels, labels, and caption skeletons.
- `figure_image_prompts.md`: conceptual image prompts for discussion drafts.
- `gpt_image_generation_brief.md`: human-approved workflow for using Codex/GPT image generation on concept drafts.
- `gpt_image_generation_queue.csv`: per-figure status, aspect ratio, output stub, prompt, and forbidden items.
- `table_blueprints.csv`: table/box structure and evidence requirements.
- `table_evidence_gaps.csv`: missing evidence, missing full text, RAG queries, and human actions.
- `human_display_item_checkpoint.md`: human approval surface before drafting or image generation.

## Figure/Table Agent

The `figure_table_designer` agent should consume the display-item outputs and return:

- `display_item_inventory`
- `figure_blueprints`
- `table_blueprints`
- `evidence_needed`
- `human_decisions_needed`
- `image_prompts_ready_for_review`
- `gpt_image_generation_queue`
- `post_generation_audit_findings`

It must not invent citations, numerical results, or final visual claims. It should ask for literature, full text, RAG, or user decisions when evidence is missing.
