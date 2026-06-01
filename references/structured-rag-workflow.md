# Structured Literature RAG Workflow

This workflow is inspired by `shibing624/TreeSearch`, which emphasizes structure-aware retrieval over document trees, SQLite FTS5 keyword matching, and no required embeddings or vector database. TreeSearch preserves document hierarchy rather than relying on arbitrary chunk splitting.

Repository: https://github.com/shibing624/TreeSearch

## When To Use

Use a structured literature RAG index after the scope has narrowed and the pool has selected papers, cards, abstracts, or full text. It is especially useful for:

- finding exact evidence for a section
- comparing methods, datasets, metrics, and limitations
- retrieving claim-supporting passages
- checking whether a paper discusses a term, intervention, or outcome
- drafting nuanced controversy or comparison paragraphs

Do not use RAG retrieval to bypass citation verification. Retrieved text supports local reasoning, but cited metadata must still be verified.

Before building the index, run the full-text gate:

```bash
python $SKILL_DIR/scripts/fulltext_manager.py fetch-europepmc \
  --pool-dir ./review-data/02_literature/pool
python $SKILL_DIR/scripts/fulltext_manager.py audit \
  --pool-dir ./review-data/02_literature/pool
```

If selected papers still lack extracted full text, ask the user to add files listed in `review-data/02_literature/pool/contexts/needs_user_fulltext.md`, then run `fulltext_manager.py import`. For difficult PDFs, run `fulltext_manager.py mineru-agent` to parse with MinerU/OCR into Markdown.

## Preferred Options

Option A: if `pytreesearch` or `treesearch` is installed, use TreeSearch directly over `review-data/02_literature/pool/cards`, `review-data/02_literature/pool/evidence`, and selected full-text folders.

```bash
treesearch index --paths ./review-data/02_literature/pool/cards ./review-data/02_literature/pool/evidence
treesearch search --db ./index.db --query "prospective validation external dataset"
```

Option B: use the bundled no-dependency fallback:

```bash
python $SKILL_DIR/scripts/structured_lit_rag.py build \
  --pool-dir ./review-data/02_literature/pool

python $SKILL_DIR/scripts/structured_lit_rag.py search \
  --db ./review-data/02_literature/pool/indexes/lit_rag.sqlite \
  --query "external validation prospective dataset" \
  --top-k 10 \
  --out ./review-data/02_literature/pool/contexts/rag_hits.json
```

The fallback script:

- indexes literature cards (`cards/*.md`)
- indexes saved abstracts and `evidence/**/*.txt`/`.md`
- indexes `fulltext_text_path` / `fulltext_path` values recorded in `pool.json`, including Europe PMC XML-derived Markdown and MinerU Markdown
- parses Markdown headings as document nodes
- stores nodes in SQLite FTS5
- returns document path, node path, title, snippet, and score

## RAG-To-Writing Rule

When using retrieved passages:

1. Open the source card or full-text file.
2. Verify the exact paper identity and citation metadata.
3. Record the claim supported, limitation, and locator in the literature card or evidence matrix.
4. Draft the sentence with appropriate strength.
5. Cite only papers in `引文参考池`.

## Query Patterns

For detailed scientific writing, query by:

- result term plus method: `external validation transformer diagnosis`
- metric plus dataset: `accuracy F1 benchmark dataset`
- limitation: `small sample single center retrospective`
- mechanism: `knowledge graph syndrome differentiation`
- comparison: `retrieval augmented generation fine tuning`
- controversy: `hallucination safety clinical decision support`

For Chinese topics, try bilingual terms and abbreviations.
