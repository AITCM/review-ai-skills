# Literature Recall Integration

Use this reference when the task requires real literature search, evidence recall, PDF retrieval, or corpus expansion.

## Quick Navigation

- Preferred retrieval stack: local literature-recall, paper-search MCP, PubMed, Crossref/OpenAlex, arXiv/OpenReview.
- Local availability: check `paper-search` and fallback scripts before assuming the MCP exists.
- Source selection: choose PubMed, Crossref/OpenAlex, arXiv/OpenReview, or web by paper type.
- Query expansion: turn claims and system names into bounded recall tasks.
- Search logs, screening, dedupe, and evidence matrix: keep provenance auditable.

## Preferred Retrieval Stack

1. Use the local `literature-recall` skill if available.
2. Use `paper-search-mcp` CLI or MCP server for multi-source search.
3. If `paper-search` is unavailable, use bundled `<S>/pubmed_recall.py` for PubMed E-utilities recall and abstract retrieval.
4. Use bundled `<S>/crossref_openalex_recall.py` for DOI, title, publication metadata, and published-version lookup.
5. Use bundled `<S>/arxiv_openreview_recall.py` for preprint/OpenReview lead discovery, then verify whether a published version exists before final citation.
6. Use web search only for specific papers, journal instructions, publisher pages, or latest information not covered by the local tools.

`paper-search-mcp` repository: https://github.com/openags/paper-search-mcp

## Check Local Availability

```powershell
paper-search sources
where.exe paper-search
Test-Path "$HOME/.codex/skills/literature-recall/SKILL.md"
```

If installed:

```bash
paper-search search "query" -n 5 -s openalex,crossref,semantic,pubmed
paper-search search "query" -n 10 -s semantic -y <start-year>-<current-year>
paper-search read arxiv 2106.12345 -o ./downloads
paper-search download arxiv 2106.12345 -o ./downloads
```

If `paper-search` is missing, use PubMed fallback:

```bash
python $SKILL_DIR/scripts/pubmed_recall.py \
  --query "<topic search query>" \
  --mindate <start-year> \
  --maxdate <current-year> \
  --max-results 20 \
  --out-dir ./review-work/pubmed_recall
```

The PubMed fallback writes `papers.jsonl`, `papers.csv`, `screening.csv`, `evidence_matrix.md`, `search_log.json`, and `run_summary.md`, so it can be imported through `literature_pool.py import-recall`.

For DOI/title metadata verification:

```bash
python $SKILL_DIR/scripts/crossref_openalex_recall.py \
  --query "<paper title or metadata query>" \
  --sources crossref,openalex \
  --max-results 5 \
  --out-dir ./review-work/metadata_recall
```

For preprint or OpenReview leads:

```bash
python $SKILL_DIR/scripts/arxiv_openreview_recall.py \
  --query "<preprint title or project name>" \
  --sources arxiv,openreview \
  --max-results 5 \
  --out-dir ./review-work/preprint_recall
```

Preprint/OpenReview outputs are leads only by default. Run Crossref/OpenAlex or publisher verification before adding them to the final reference list.

If running from a clone:

```bash
uv run --directory /absolute/path/to/paper-search-mcp paper-search search "query" -n 5 -s arxiv,semantic,crossref
```

Ask before installing or cloning because network access may be required:

```bash
uv tool install paper-search-mcp
```

## Source Selection

Use targeted source bundles:

- Broad metadata: `openalex,crossref,semantic,dblp`
- Biomedical: `pubmed,pmc,europepmc`
- Preprints: `arxiv,biorxiv,medrxiv`
- Open-access full text: `pmc,europepmc,core,openaire,doaj,zenodo,hal`
- Computer science/AI: `arxiv,semantic,dblp`
- DOI/OA lookup: `crossref,unpaywall`
- Discovery fallback: `google_scholar` only when needed; report bot-detection limitations

Do not use Sci-Hub by default. Keep retrieval OA-first and publisher-permitted.

## Query Expansion Pattern

For Chinese-language topics, create bilingual queries:

1. Chinese concept terms
2. English normalized domain terms
3. Synonyms and abbreviations
4. Methods/intervention/model terms
5. Target population or application domain
6. Review-specific terms: `review`, `systematic review`, `scoping review`, `meta-analysis`, `bibliometric`

Template:

```text
Topic: <Chinese or bilingual topic>
Queries:
- "<English method term>" "<domain term>"
- "<model or intervention term>" "<application domain>"
- "<domain synonym>" review
- "<abbreviation>" "<task or population>"
- "<knowledge representation term>" "<model term>"
```

## Search Log Template

```markdown
## Search Log

- Date run:
- Topic:
- Review type:
- Inclusion criteria:
- Exclusion criteria:
- Sources:
- Query variants:
- Year filters:
- Raw counts by source:
- Deduplication rules:
- Deduped total:
- Retrieval errors:
- Known limitations:
```

## Screening Rules

Apply the same criteria to user corpus and external search results.

Screening labels:

- `include`: directly relevant and meets criteria
- `exclude`: fails criteria; record why
- `maybe`: needs abstract/full-text inspection
- `background`: useful context but not part of core evidence
- `seminal`: foundational source regardless of age
- `recent`: important new development
- `method`: useful for method/reporting standards

## Deduplication

Deduplicate by:

1. DOI
2. PMID/PMCID/arXiv ID
3. Normalized title plus first author/year
4. Manual check for preprint-to-journal versions

When both preprint and published article exist, prefer the peer-reviewed version unless the preprint contains unique material. Record the relationship.

## Evidence Matrix Template

```markdown
| Key | Title | Year | Source | DOI/URL | Type | Finding | Claim Supported | Limitation | Use In Review |
|---|---|---:|---|---|---|---|---|---|---|
```

Keep exact identifiers because later `read` and `download` commands need source-specific paper IDs.
