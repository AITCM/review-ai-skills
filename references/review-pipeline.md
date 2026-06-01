# Top-Journal Review Pipeline

This reference fuses three source patterns:

- `openags/paper-search-mcp`: free-first, multi-source academic search, standard metadata output, deduplication, and OA-first full-text retrieval.
- `Imbad0202/academic-research-skills`: staged academic pipeline, Socratic planning, material provenance, integrity gates, peer-review/revision loops, style calibration, and process records.
- `Orchestra-Research/AI-Research-SKILLs`: two-loop research orchestration, proactive drafting, top-venue paper narrative, citation verification, and figure-first writing.

## Quick Navigation

- Phase 0: intake, setup checks, and user constraints.
- Phase 1: thesis, top-journal contribution, and draft-derived framework.
- Phase 2: search strategy, draft-first literature governance, and corpus assembly.
- Phase 3: evidence matrix and literature cards.
- Phase 4: synthesis, controversies, taxonomy, and gap map.
- Phase 4.5: display items before drafting.
- Phase 5-6: structure and manuscript drafting.
- Phase 7-8: citation integrity and editorial review.
- Phase 9-10: revision, cover letter, and submission package.

## Phase 0: Intake

Collect only what changes the work:

- Topic and intended contribution
- Review type: narrative, systematic, scoping, umbrella, methodological, bibliometric, perspective, or state-of-the-art
- Target journal or journal family
- Discipline and audience
- Desired language
- Time window and geography
- Must-include or must-exclude papers
- Whether the user has a corpus, Zotero export, PDFs, notes, or a draft
- Output format: outline, search report, evidence matrix, manuscript, DOCX/LaTeX/PDF-ready text
- Existing GPT/Gemini/Deep Research drafts and whether their sources should be audited
- Whether this is a tool-heavy, multi-session, multi-agent, or MCP-backed project that should be bootstrapped with `.venv`, `pyproject.toml`, a local package, and `review-data/06_agent_memory`
- Whether external agents should be called directly and whether `DEEPSEEK_API_KEY` or `REVIEW_AGENT_API_KEY` is already configured

If the topic is vague, run a short Socratic pass before searching:

1. What is the field-level problem?
2. What existing reviews fail to explain?
3. What controversy, transition, or opportunity makes the review timely?
4. What audience needs this review now?
5. What would make the paper citable in five years?

If the project will use many Python tools or external subagents, run `<S>/path_schema.py check --project-dir .` and `<S>/project_bootstrap.py check-env --project-dir .` before starting heavy work. If `.venv`, `pyproject.toml`, the standard `review-data`/`review-work`/`review-output` folders, or `review-data/06_agent_memory` are missing, ask the user before initializing the review root, unless the user already explicitly requested setup.

## Phase 1: Review Thesis

Define the one-sentence review contribution before drafting:

```text
This review argues that [field/problem] should be understood through [organizing lens],
because [evidence pattern], which changes [research/practice/policy agenda].
```

Strong top-journal reviews usually offer at least one of:

- A new conceptual framework
- A field map or taxonomy
- A synthesis resolving conflicting evidence
- A methodological critique with standards
- A translational roadmap
- A future research agenda tied to concrete gaps
- A re-framing of an emerging technology or clinical/scientific problem

If the thesis is only "summarize recent studies", strengthen it before writing.

When multiple GPT/Gemini/Deep Research drafts are supplied, do not infer the thesis from only the latest draft. First run `<S>/draft_memory.py build` and `<S>/draft_citation_assets.py build`, then chief Codex must read the memory checkpoint, per-draft cards, citation health report, candidate clues, and claim-evidence map before delegation. State a provisional thesis/claim map, then run `<S>/draft_literature_discovery_orchestrator.py plan` and dispatch Codex subagents over the generated packets. Run `<S>/draft_logic_framework.py` only after the draft-derived literature discovery state exists, then discuss the architecture brief with the user before drafting. The goal is to choose one spine, not to concatenate all draft headings. Draft memory prevents Codex from forgetting the user's strongest starting material; citation assets and literature-discovery packets prevent noisy supplemental recall from polluting the evidence base.

## Phase 2: Search And Corpus

If AI-generated drafts exist, audit them before searching from scratch. The framework and argument map come from careful draft reading; literature search then supplies proof, counterproof, and historical context. Inspect draft-native citation assets: reference section health, in-text markers, uncited references, non-academic sources, claim snippets, and candidate paper clues. Then run literature-discovery/subagent reading packets so papers implied in the body prose are not missed. Verify draft-native, literature-discovery, hidden/identity-normalized, and screened PubMed-deep-dive papers through multi-signal PubMed/PMID, DOI, title, author/year/venue, metadata APIs, and official pages; do not trust a draft DOI/PMID when it resolves to a nonmatching title. Merge accepted lanes into a deduped verified union before official citation export or pool import.

Only after framework gaps are confirmed should supplemental recall run. Prefer `<S>/argument_literature_expander.py plan` to convert specific claims, past-present-future bridges, display-item needs, and polluted-source replacements into claim-linked search tasks. Then use `argument_literature_expander.py pubmed` to retrieve PubMed metadata and abstracts as a triage packet. If the verified draft-derived union is still too small or the framework needs a broader historical/current/method/governance evidence base, run `<S>/supplemental_recall_screening.py plan` to create broad-but-bounded multi-source recall tasks from body-mentioned systems, unmapped claims, and literature-discovery API tasks. Screen the large recall set with Codex subagents before deterministic verification. These rows first enter `review-data/02_literature/supplemental_pool`; they are promoted only after claim-fit screening, publication-status checks, and full-text readiness for detailed claims. See `references/draft-first-literature-workflow.md` and `references/deep-research-draft-governance.md`.

Run high-recall search before synthesis. Keep a search log with:

- Date run
- Query variants
- Sources/databases
- Filters and years
- Raw results per source
- Deduped total
- Inclusion/exclusion criteria
- Excluded/skipped records and reasons
- Retrieval errors and known upstream limitations

Store recalled records in `review-data/02_literature/pool` when the project will continue across multiple turns or drafting sessions. Promote only high-relevance, claim-supporting records into selected citation-pool records, then build literature cards so future drafting can load selected memory rather than the full noisy pool. See `references/literature-pool-workflow.md`.

When a user provides a corpus, use a corpus-first, search-fills-gap flow:

1. Screen user corpus with the same criteria as external results.
2. Record included, excluded, and skipped corpus items with reasons.
3. Identify uncovered concepts, years, methods, populations, or key authors.
4. Search externally to fill gaps.
5. Merge corpus and external results into one neutral included set.

Do not mutate the user's corpus. Do not silently discard papers.

## Phase 3: Evidence Matrix

Create an evidence matrix before writing substantial prose.

Recommended columns:

- Citation key
- Title
- Authors/year
- DOI/URL
- Source/database
- Study/review type
- Domain/population/material/system
- Methods/data
- Main finding
- Claim supported
- Limitations/risk of bias
- Why included
- Notes for figures/tables

For systematic reviews, add PRISMA fields: record ID, duplicate status, screening decision, full-text decision, exclusion reason, and quality/risk-of-bias rating.

## Phase 4: Synthesis

Synthesize before drafting. Produce:

- Field map or taxonomy
- Timeline of major shifts
- Consensus claims
- Contradictions and competing explanations
- Methodological weaknesses
- Evidence gaps
- Translational or practical implications
- Future research agenda

Use a "claim-evidence-reasoning" chain:

```text
Claim: what the review says.
Evidence: which verified sources support it.
Reasoning: why the evidence supports the claim and where it is limited.
Risk: what would make the claim weaker or false.
```

When the evidence pool includes cards, abstracts, or full text, use the structured RAG workflow for detailed lookups such as metrics, dataset names, result comparisons, limitations, and exact mechanism statements. Before TreeRAG, run `fulltext_manager.py fetch-europepmc`, `audit`, and if needed `import`/`mineru-agent`; show `needs_user_fulltext.md` to the user when selected papers still lack usable full text. Retrieved passages are working evidence, not automatically citable claims.

## Phase 4.5: Display Items

Before drafting around figures, tables, or boxes, run display-item planning. Display items must carry the framework, evidence ladder, taxonomy, controversy map, governance loop, or research agenda.

Use:

```bash
python $SKILL_DIR/scripts/display_item_planner.py plan \
  --project-dir . \
  --framework-dir ./review-data/03_framework/logic_framework \
  --pool-dir ./review-data/02_literature/pool \
  --rag-db ./review-data/02_literature/pool/indexes/lit_rag.sqlite \
  --out-dir ./review-data/03_framework/display_items
```

Stop at `human_display_item_checkpoint.md` before generating images or writing around table claims. Conceptual image prompts are discussion drafts only. Tables must be evidence-first and mark unsupported cells as `[EVIDENCE GAP]`.

## Phase 5: Structure

Choose structure by review type:

- Narrative/topical: introduction -> organizing framework -> thematic sections -> controversies -> future agenda -> conclusion
- Systematic: introduction -> methods -> results -> synthesis -> discussion -> limitations -> conclusion
- Scoping: introduction -> methods -> mapping results -> conceptual gaps -> research agenda
- Perspective: problem -> reframing -> evidence anchors -> implications -> objections -> agenda
- Methods review: field problem -> current methods -> comparison framework -> standards -> recommended workflow

Top-journal review front matter should be unusually strong:

- Title: precise, memorable, not inflated
- Abstract: problem, scope, organizing lens, key synthesis, implications
- Figure 1: core conceptual framework or field map
- Box/table: definitions, taxonomy, key studies, or practical checklist

Architecture check:

- Prefer 3-5 major body movements for a Nature Reviews-style narrative.
- Avoid 8-9 equal first-level sections unless the article is deliberately encyclopedic or systematic.
- Merge technology-specific headings when they serve the same argument.
- Make every first-level heading answer a question or advance the central thesis.
- If a draft's abstract includes search counts, final-gate counts, or internal workflow details, move those to methods/reporting notes unless the target journal explicitly requires a systematic/scoping abstract.

## Phase 6: Drafting

Draft from the synthesis, not from memory. Write around section-level claims and evidence chains.

Rules:

- Each section has one job.
- Start each section with the question or claim it answers.
- Group papers by idea, mechanism, method, or controversy, not by one-paper-per-paragraph chronology.
- Use explicit contrast language: "in contrast", "however", "this pattern is limited by".
- Put limitations and uncertainty close to the claims they qualify.
- Avoid "recent years have witnessed" and generic AI-style openings.

For top-journal polish:

- Make the introduction answer: Why this review, why now, why this lens?
- Use figures/tables as argument infrastructure, not decoration.
- Give readers reusable concepts, not just a summary.
- End with a concrete research agenda.

## Phase 7: Integrity Gate

Before peer-review simulation, run an integrity pass:

- Every reference exists and matches title/authors/year/DOI.
- Every central claim has at least one supporting source.
- No citation is used to support a stronger claim than it makes.
- No invented result, sample size, comparison, or mechanism.
- Negative findings and uncertainty are represented fairly.
- User-supplied corpus and external results are treated with consistent criteria.

Any unresolved item becomes `[MATERIAL GAP]`, `[CITATION NEEDED]`, or `[VERIFY CLAIM]`.

## Phase 8: Editorial Review

Simulate a top-journal editorial decision:

- Novelty and timeliness
- Fit to journal audience
- Completeness of search and coverage
- Balance of seminal and recent sources
- Strength of organizing framework
- Evidence sufficiency for major claims
- Handling of controversies and limitations
- Figure/table utility
- Writing clarity
- Ethical, clinical, or policy overreach

Use a decision scale:

- Accept-ready: rare; only copyediting and formatting remain
- Minor revision: structure and claims are sound; gaps are local
- Major revision: thesis or evidence structure needs rebuilding
- Reject/reframe: scope, novelty, or evidence base is insufficient

## Phase 9: Revision And Finalization

Convert review findings into a revision roadmap:

- Blocking integrity fixes
- Structure/narrative changes
- Missing sources or source classes
- Claim softening or strengthening
- Figure/table additions
- Language and journal formatting
- Abstract/title sharpening
- Disclosure, data availability, and AI-use statement if needed

After revision, rerun the integrity gate on all changed claims.

## Phase 10: Cover Letter And Submission Package

Prepare a short cover letter after the manuscript's central thesis, journal fit, and final citation gate are stable.

The cover letter should help the editor answer:

- What is the main finding, synthesis, or argument?
- Why does it matter now?
- Why will the journal's broad readership care?
- How does it differ from existing reviews, frameworks, or alternative solutions?
- Are there related manuscripts, conflicts, or prior editor discussions the editor must know?
- Who could review it independently, and who should reasonably be excluded?

For Nature-style submissions, do not repeat the abstract. Keep the letter short, direct, respectful toward competing work, and explicit about confidential issues. Use `references/cover-letter-workflow.md` and bundled `<S>/cover_letter_builder.py` to produce:

- `cover_letter_draft.md`
- `cover_letter_brief.md`
- `cover_letter_checklist.md`
- suggested and excluded referee tables

Run a cover-letter audit before submission: journal name, editor name, article type, title, related-manuscript disclosure, originality declaration, author approval, competing interests, AI-use disclosure, and referee independence.
