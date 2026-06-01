# Integrity And Review Gates

Use this reference before claiming that a review manuscript is ready.

## Quick Navigation

- Citation integrity: verify every source and never cite from memory.
- Claim audit: check claim-source fit, overreach, and missing evidence.
- Evidence hierarchy: distinguish formal papers, guidelines, preprints, datasets, and web context.
- Failure checklist and top-journal rubric: stress-test novelty, synthesis, and balance.
- Reviewer simulation and revision roadmap: produce actionable fixes before submission.

## Citation Integrity

Never create references from memory. For every cited source:

- Verify that the paper exists.
- Verify title, authors, year, venue, DOI/URL, and publication status.
- Prefer programmatic metadata via `paper-search-mcp`, Crossref, Semantic Scholar, PubMed, OpenAlex, DOI resolver, or publisher pages.
- Retrieve BibTeX or citation metadata from source APIs when possible.
- Mark uncertain entries explicitly instead of filling details.
- After drafting, run `citation_sequence_manager.py renumber` so numeric citations follow first appearance and the bibliography is rebuilt from cited entries.
- After drafting, run `manuscript_narrative_guard.py audit` so workflow logs, candidate counts, RAG status, and "草稿核验显示" process language do not leak into the article body.

Default final-reference policy:

- Keep final manuscript references to formally published papers and official guidelines/standards.
- Use PubMed, Crossref/OpenAlex, publisher pages, or official guideline pages to confirm final references.
- Use arXiv/OpenReview/preprints only as search leads or background signals unless the user explicitly requests that preprints be cited.
- Delete or demote any case/source that cannot be verified in PubMed, Crossref/OpenAlex, arXiv, OpenReview, or a publisher/official source.
- Run `literature_pool.py final-gate` before final drafting and again before final reference formatting.
- Run `manuscript_narrative_guard.py audit` before polishing; a top-journal review can discuss evidence maturity, but it must not sound like an internal audit report.
- Run `citation_sequence_manager.py renumber` before final DOCX/PDF export; resolve `missing_reference_entries.csv`, `missing_official_citations.csv`, and `unresolved_citation_markers.csv`.

Placeholder format:

```text
[CITATION NEEDED: exact paper/source not verified]
[VERIFY REFERENCE: title/authors/year/DOI uncertain]
```

For LaTeX:

```latex
% EXPLICIT PLACEHOLDER - requires human verification
\cite{PLACEHOLDER_author_year_verify}
% VERIFY: Confirm this citation exists and supports the sentence.
```

## Claim Audit

Make a claim table for important claims:

```markdown
| Claim | Source(s) | Support Level | Locator | Risk | Action |
|---|---|---|---|---|---|
```

Support levels:

- `direct`: source explicitly supports the claim
- `partial`: source supports part of the claim; qualify the sentence
- `indirect`: source provides background only; do not cite as proof
- `conflicting`: sources disagree; present as controversy
- `unsupported`: remove, rewrite, or mark as material gap

Audit questions:

- Does the cited source really make this claim?
- Is the manuscript stronger than the evidence?
- Are negative or null findings omitted?
- Is a preprint treated as established fact?
- Are review articles cited for primary evidence where primary studies are needed?
- Are claims about causality, clinical efficacy, safety, policy, or guidelines supported by appropriate evidence?

## Evidence Hierarchy

For biomedical or clinical topics, distinguish:

- Guidelines and consensus statements
- Systematic reviews/meta-analyses
- Randomized trials
- Prospective cohorts
- Retrospective/observational studies
- Case reports/series
- Mechanistic/basic science
- Expert opinion
- Preprints

For AI/engineering topics, distinguish:

- Peer-reviewed papers
- arXiv preprints
- benchmark reports
- code repositories
- technical docs
- blog posts
- vendor claims

Do not give all sources equal evidentiary weight.

## AI Failure Mode Checklist

Run this blocking checklist before finalization:

1. Implementation or interpretation error passed self-review.
2. Hallucinated citation.
3. Hallucinated result or unsupported quantitative claim.
4. LLM-assigned citation numbers not ordered by first appearance.
5. Internal workflow/audit metadata leaked into the manuscript body.
6. Shortcut reliance or cherry-picked proxy metric.
7. Error reframed as novel insight.
8. Methodology fabrication or missing reproducibility detail.
9. Frame-lock: early framing prevents better interpretations.

If any item is suspected, stop polishing and fix the evidence or framing first.

## Top-Journal Review Rubric

Score each dimension 1-5:

- Timeliness: why the review is needed now
- Novel organizing lens
- Search transparency
- Coverage of seminal and recent work
- Evidence hierarchy and balance
- Treatment of controversies
- Conceptual framework or taxonomy
- Figures/tables as reusable field tools
- Limits and uncertainty
- Future agenda specificity
- Writing clarity and section logic
- Journal fit

Decision guidance:

- 4.5-5 average: submission polish
- 3.8-4.4: strong but needs targeted revision
- 3.0-3.7: major revision; strengthen thesis/evidence
- below 3.0: reframe before drafting more prose

## Reviewer Simulation

Generate three reviewer personas plus editor:

- Editor: journal fit, novelty, scope, audience
- Method reviewer: search strategy, screening, evidence grading, reproducibility
- Domain reviewer: missing key sources, field interpretation, controversies
- Skeptical reviewer: overclaims, weak causal logic, unsupported future agenda

The review report should lead with blocking issues, then major comments, minor comments, and an actionable revision roadmap.

## Revision Roadmap

For every issue:

- Severity: blocking, major, minor
- Location: section/table/figure/claim
- Problem
- Required evidence or rewrite
- Suggested action
- Verification step

After revision, rerun citation integrity and claim audit on changed sections.
