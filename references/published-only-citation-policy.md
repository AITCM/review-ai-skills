# Published-Only Citation Policy

Use this reference when preparing the final reference list for a top-journal review.

## Default Rule

Final manuscript references should be:

- formally published journal articles, books, or conference proceedings with stable metadata
- official guidelines, standards, laws, regulatory documents, or trial registry records when the claim specifically needs them
- verified by PubMed, Crossref, OpenAlex, publisher pages, or official institutional pages

Do not include in the final reference list by default:

- arXiv-only preprints
- OpenReview-only or withdrawn submissions
- ResearchGate pages
- blogs, Medium posts, company pages, marketing pages, newsletters, or news articles
- cases or model names with no verifiable scholarly source

## Source Resolution Cascade

For each draft/source/case:

1. Treat PMID/DOI/title/author/year/venue/URL as clues, not as single-source truth.
2. Try PubMed/PMC for biomedical papers. If a draft PMID/DOI resolves to a nonmatching title, record the mismatch and fall back to title/bibliographic search.
3. Try Crossref/OpenAlex for DOI, venue, year, and published version. If a draft DOI is wrong, do not accept it; search by the paper name and bibliographic clues.
4. Try arXiv/OpenReview only as lead discovery.
5. If only arXiv/OpenReview is found, keep as `preprint_only` or `background`; do not cite in final manuscript unless the user explicitly allows preprints.
6. If none of PubMed, Crossref/OpenAlex, arXiv, OpenReview, publisher, or official pages can verify it, delete or reject the case from citation candidates.

## Version And Correction Gate

- If a preprint and formal paper share the same paper title or system identity, cite only the formal paper by default.
- PubMed verification must inspect and rank multiple PMID hits when one title/system has preprint, formal article, and correction records; do not stop at the first PubMed hit.
- Mark the preprint as `duplicate_formal_version_available` and keep it only as a discovery/history lead.
- Do not cite `Publisher Correction`, `Correction`, `Erratum`, `Corrigendum`, `Retraction`, or `Withdrawn` records as if they were the target paper.
- If an API resolves a preprint/title query to a correction record, continue searching for the target paper; if the formal article is already verified, suppress the preprint/correction duplicate.

## Official Citation Export

After final paper identity and inclusion are stable, generate the bibliography from official citation-manager exports rather than from draft reference strings:

```bash
python $SKILL_DIR/scripts/official_citation_exporter.py export \
  --verified-csv ./review-data/05_audit/draft_source_adjudication/accepted_verified_draft_papers.csv \
  --out-dir ./review-data/05_audit/official_citations \
  --style vancouver \
  --email <email>
```

Rows that cannot be exported from PubMed, Crossref/DOI, arXiv, or an official page should remain in the manual official-export handoff list until resolved.

## Enforcement Command

```bash
python $SKILL_DIR/scripts/literature_pool.py final-gate \
  --pool-dir ./review-data/02_literature/pool \
  --out ./review-data/02_literature/pool/contexts/final_citation_gate.md \
  --apply
```

Review `final_citation_gate.md` before final drafting.

## Writing Rule

When a strong but unpublished preprint is important, discuss the idea without treating it as established evidence, or replace it with a published adjacent paper. If the manuscript must mention the preprint, label it clearly as a preprint and keep it outside the final citation pool unless the user overrides this policy.
