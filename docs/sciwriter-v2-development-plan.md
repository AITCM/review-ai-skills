# SciWriter V2 Development Plan

This document turns the previous citation-normalization experience and the `review-ai-skills` workflow into an implementation plan for an evidence-first AI review-writing workbench.

## 0. Product Positioning

SciWriter V2 is not a one-shot paper generator. It is a governed manuscript compiler:

```txt
DeepResearch drafts
  -> DraftMemory
  -> candidate claims
  -> Zotero / imported literature lanes
  -> Reference Authority
  -> full-text / RAG evidence layer
  -> claim-evidence verification
  -> manuscript blueprint
  -> controlled drafting
  -> deterministic citation compiler
  -> quality gates
  -> import trace
```

Core rule:

```txt
LLM proposes. Evidence constrains. Backend compiles. Quality gates decide.
```

## 1. Non-Negotiable Invariants

1. The global project literature ledger is the only formal citation source.
2. Zotero is an ingestion and user-library integration layer, not the final citation-numbering authority.
3. LLMs may create candidate claims, citation intents, draft paragraphs, and diagnostic notes.
4. LLMs must not create final bibliography entries, final numeric citation order, or citation-ready references.
5. A reference being real does not mean it supports the claim using it.
6. RAG hits are working evidence, not manuscript claims. A RAG hit must become an EvidenceAtom linked to a source span before it can support a claim.
7. Any agent failure is a quality event and may block importable manuscript status.
8. Fallback outputs are `diagnostic_only`, never importable.
9. Import must be by stored draft ID only, with hash verification and trace logging.
10. Frontend may display status but must not decide manuscript readiness.

## 2. Architecture Layers

### Layer A: Zotero & Material Ingestion

Zotero is the user-facing literature manager integration.

Supported ingestion modes, in priority order:

1. Zotero Local API read-only ingest from `localhost:23119/api/`.
2. Zotero Web API for synced personal or group libraries.
3. Export-based fallback: BibTeX, BibLaTeX, RIS, CSL-JSON, NBIB, CSV.
4. Direct SQLite read-only fallback only for advanced local recovery and never for writes.

Zotero responsibilities:

- Import selected collections, tags, saved searches, items, notes, and attachment metadata.
- Record Zotero item keys, library IDs, collection paths, item versions, and attachment hashes.
- Pull local attachment file paths where available.
- Preserve Zotero provenance but normalize every imported paper into `PaperEntity` and `ProjectReference` before use.
- Never mutate Zotero in the MVP.

New tables:

```txt
zotero_connections
zotero_libraries
zotero_collections
zotero_items
zotero_item_attachments
zotero_sync_states
zotero_import_runs
```

Key rule:

```txt
ZoteroItem -> PaperEntity -> ProjectReference
```

A Zotero item can seed or enrich a project reference. It cannot by itself become manuscript-ready.

### Layer B: Reference Authority

This layer resolves paper identity and official metadata.

Providers:

- PubMed / NCBI E-utilities
- Europe PMC
- Crossref
- OpenAlex
- DOI resolver
- Optional Semantic Scholar
- Zotero as user-library provenance

Core tables:

```txt
paper_entities
project_references
external_lookup_attempts
provider_cache_entries
citation_resolution_attempts
official_citation_exports
```

Required logging per external query:

```txt
provider
query
normalized_query_hash
response_hash
retrieved_at
status
failure_reason
rate_limit_context
source_mode = live | fixture
```

Readiness states:

```txt
ProjectReference.status:
  raw | resolving | resolved | ambiguous | failed | rejected | deprecated

ProjectReference.citation_readiness:
  not_ready | metadata_ready | anchor_ready | evidence_ready | manuscript_ready
```

Rule:

```txt
No DOI / PMID / PMCID / OpenAlex ID / trusted official anchor => not citation-ready by default.
```

### Layer C: Full Text & RAG Evidence Layer

This layer parses verified papers and makes their content searchable, but does not write manuscript claims directly.

Full-text sources, in priority order:

1. Europe PMC fullTextXML when available.
2. Zotero/local attachment PDF or linked file.
3. User-supplied PDF/TXT/MD/DOCX.
4. OA/publisher/paper-search output.
5. MinerU/OCR for complex PDF, scanned PDF, formula/table-heavy PDF, or Chinese-English mixed layouts.

Core tables:

```txt
full_text_records
full_text_parse_runs
evidence_source_spans
rag_indexes
rag_documents
rag_chunks
rag_retrieval_runs
rag_retrieval_hits
evidence_atoms
```

Source span contract:

```txt
evidence_source_spans:
  id
  full_text_record_id
  paper_entity_id
  section
  page_start
  page_end
  char_start
  char_end
  raw_text
  raw_text_hash
  parser
  parser_version
```

Evidence atom contract:

```txt
evidence_atoms:
  id
  paper_entity_id
  project_reference_id
  source_span_id
  atom_type = finding | method | limitation | dataset | metric | definition | background | contradiction
  normalized_text
  polarity = supports | contradicts | limits | contextualizes
  confidence
  source_mode = full_text | abstract_only | note | zotero_note
  created_by_agent_run_id
```

RAG gates:

- Do not build detailed method/result/metric claims from abstracts alone unless explicitly marked `abstract_only`.
- Every RAG retrieval used for writing must record query, topK, filters, index version, selected spans, and output hash.
- RAG results must be converted into EvidenceAtoms before claim verification.
- If selected papers lack usable full text, produce a user handoff manifest instead of silently drafting strong claims.

### Layer D: Claim & Verification Layer

DeepResearch drafts, Zotero notes, and user notes become candidate claims.

Core tables:

```txt
draft_memories
claims
claim_groups
claim_evidence_links
claim_reviews
citation_intents
```

Claim status:

```txt
candidate
normalized
deduped
needs_evidence
linked_to_evidence
reviewed_supported
reviewed_partial
reviewed_unsupported
reviewed_contradicted
approved_for_blueprint
used_in_draft
```

ClaimEvidenceLink contract:

```txt
claim_evidence_links:
  claim_id
  evidence_atom_id
  project_reference_id
  support_type = supports | partially_supports | contradicts | background_only
  support_strength = strong | medium | weak
  quote_span_id
  explanation
  created_by_agent_run_id
```

Rules:

- `unsupported` and `contradicted` claims cannot enter a manuscript blueprint.
- `partial` claims can enter only after rewriting/demotion.
- Every central manuscript claim must trace to at least one EvidenceAtom.

### Layer E: Synthesis & Blueprint Layer

This layer produces the review architecture before drafting.

Core tables:

```txt
review_protocols
synthesis_matrices
topic_clusters
gap_analyses
manuscript_blueprints
```

Blueprint inputs:

- review protocol
- verified references
- approved claims
- evidence matrices
- gap map
- controversy map
- display-item plan

Blueprint outputs:

- thesis
- 3-5 major section spine
- section-level jobs
- claim IDs per section
- required figures/tables/boxes
- unresolved material gaps

### Layer F: Manuscript Compiler & Quality Gates

Core tables:

```txt
full_manuscript_drafts
manuscript_sections
manuscript_citation_occurrences
citation_sequences
manuscript_imports
quality_gate_results
project_events
agent_runs
```

Citation flow:

```txt
CitationIntent -> ProjectReference -> ManuscriptCitationOccurrence -> CitationSequence -> Bibliography
```

The LLM writes stable markers such as:

```txt
{{cite:project_ref_abc123}}
{{cite_intent:claim_foo_supporting_sources}}
```

The backend compiler resolves those markers to numeric citations and reference lists.

Quality gates:

1. Structure gate: title, abstract, required sections, minimum length.
2. Citation gate: no unknown markers, no unresolved intents, stable order, no local reference blocks.
3. Evidence gate: central claims supported, partial claims demoted, unsupported claims blocked.
4. Agent gate: required agents succeeded, no invalid schema output, no diagnostic fallback promoted.
5. Trace gate: reference provenance, evidence span trace, draft hash, import hash.
6. Style/risk gate: overclaiming, workflow leakage, citation stacking, missing uncertainty.

## 3. Development Milestones

### M0: Repository & Contract Baseline

Deliverables:

- Create clean monorepo or app package boundary.
- Decide whether V2 lives in a new repo or under this repo as `apps/sciwriter-v2`.
- Add shared type contracts with Zod.
- Add fixture DB and deterministic test harness.
- Add status command that prints system readiness.

Exit criteria:

```powershell
npm.cmd run lint
npm.cmd run build
npm.cmd run test
npm.cmd run pipeline:status
```

### M1: Core Schema & Fixture Manuscript Compiler

Deliverables:

- Implement migrations for project, material, paper, reference, claim, citation, manuscript, quality, and event tables.
- Import 20 fixture ProjectReferences.
- Compile fixture manuscript with stable citation markers into numeric references.
- Store FullManuscriptDraft.
- Import by stored draft ID only.

Exit criteria:

```powershell
npm.cmd run pipeline:manuscript:compile-fixture
npm.cmd run pipeline:manuscript:quality
npm.cmd run smoke:citations
npm.cmd run smoke:import-trace
```

### M2: Zotero Read-Only Ingestion

Deliverables:

- Implement Zotero local API adapter.
- Implement Zotero Web API adapter behind explicit credentials.
- Implement import from BibTeX/RIS/CSL-JSON export files.
- Map Zotero items to PaperEntity candidates.
- Map Zotero attachments to Material/full_text candidates.
- Cache every Zotero response and record import run hashes.

CLI commands:

```powershell
npm.cmd run pipeline:zotero:local-status
npm.cmd run pipeline:zotero:import-fixture
npm.cmd run pipeline:zotero:sync-fixture
npm.cmd run pipeline:zotero:attachments-fixture
```

Quality rules:

- Zotero unavailable is a recoverable ingestion issue, not a manuscript-quality pass.
- Zotero imported items remain candidates until resolved by Reference Authority.
- Direct SQLite mode is read-only and blocked from writes.

### M3: Reference Authority & Official Citation Export

Deliverables:

- Provider abstraction for PubMed, Europe PMC, Crossref, OpenAlex, DOI resolver, optional Semantic Scholar.
- Live/fixture separation.
- Provider cache.
- Metadata conflict adjudication.
- Official citation export into CSL-JSON/BibTeX/RIS/NBIB where available.

CLI commands:

```powershell
npm.cmd run pipeline:references:resolve-fixture
npm.cmd run pipeline:references:dedupe-fixture
npm.cmd run pipeline:references:export-official-fixture
npm.cmd run smoke:citations
```

Blockers:

- No authoritative anchor.
- Metadata mismatch between provided DOI/PMID and title.
- Ambiguous title/author/year match.
- Unsupported preprint policy.

### M4: Full Text, Parsing, and RAG Index

Deliverables:

- FullTextRecord ingestion from Europe PMC XML, Zotero attachments, user files, MinerU Markdown.
- Source span extraction with section/page/char provenance.
- Structured RAG index over literature cards, abstracts, full text, and notes.
- Retrieval run logging.
- Full-text readiness audit and user handoff manifest.

CLI commands:

```powershell
npm.cmd run pipeline:fulltext:audit-fixture
npm.cmd run pipeline:fulltext:import-fixture
npm.cmd run pipeline:rag:build-fixture
npm.cmd run pipeline:rag:query-fixture
```

Blockers:

- Detailed claims requested from abstract-only source.
- Missing full text for papers marked `seminal`, `method`, or `core_evidence`.
- RAG hit without source span.

### M5: Claim Extraction and Claim-Evidence Verification

Deliverables:

- DraftMemory import for GPT/Gemini/Kimi/DeepResearch drafts.
- Atomic claim extraction with strict JSON schema.
- Claim deduplication and claim grouping.
- EvidenceAtom extraction from source spans.
- ClaimEvidenceLink verifier.
- ClaimReview demotion/rewrite workflow.

CLI commands:

```powershell
npm.cmd run pipeline:drafts:memory-fixture
npm.cmd run pipeline:claims:extract-fixture
npm.cmd run pipeline:evidence:atoms-fixture
npm.cmd run pipeline:evidence:verify-fixture
npm.cmd run pipeline:quality:explain
```

Blockers:

- Unsupported or contradicted claim enters blueprint.
- Claim uses a source only as background but presents it as empirical support.
- Agent failure treated as warning.

### M6: Blueprint, Controlled Drafting, and Final Gates

Deliverables:

- ReviewProtocol editor/data model.
- Synthesis matrices: theme, method, timeline, controversy, gap.
- ManuscriptBlueprint generation.
- Section drafting from approved claims only.
- Global manuscript assembly.
- Citation compiler.
- Full quality gate.

CLI commands:

```powershell
npm.cmd run pipeline:blueprint:create-fixture
npm.cmd run pipeline:manuscript:draft-fixture
npm.cmd run pipeline:manuscript:compile-fixture
npm.cmd run pipeline:manuscript:quality
```

Importable manuscript condition:

```txt
FullManuscriptDraft.status = importable
all block gates passed
all required hashes recorded
no unsupported claims in body
no unknown citation markers
bibliography generated by compiler
```

### M7: Minimal UI

Deliverables:

- Project dashboard.
- Zotero import panel.
- Reference authority board.
- Full-text/RAG readiness panel.
- Claim/evidence audit board.
- Blueprint and manuscript preview.
- Quality gate explanation view.

UI rules:

- Frontend never renumbers citations.
- Frontend never marks a manuscript ready.
- Frontend shows backend quality results and trace links.

### M8: Live Agent Orchestration

Deliverables:

- Provider abstraction for OpenAI-compatible APIs and fixture provider.
- Agent run ledger.
- Strict JSON schema validation.
- Retry with degradation policy.
- Diagnostic-only fallback.
- Optional DeepSeek/OpenAI-compatible acceleration after explicit user approval.

Core agents:

```txt
DraftClaimExtractorAgent
ReferenceMentionExtractorAgent
ClaimDedupAgent
EvidenceAtomExtractorAgent
ClaimEvidenceVerifierAgent
CitationIntentExtractorAgent
BlueprintPlannerAgent
SectionDraftAgent
RiskReviewerAgent
```

Do not implement first:

```txt
Full autonomous paper writer
Auto-import agent
Unbounded multi-agent literature crawler
Frontend-only citation fixer
```

## 4. Recommended First Build Slice

Implement one deterministic end-to-end path before live agents:

```txt
fixture DeepResearch draft
  -> 30 candidate claims
  -> 20 fixture ProjectReferences
  -> 20 fixture EvidenceAtoms
  -> claim-evidence verification
  -> blueprint using supported/partial claims only
  -> draft with stable citation markers
  -> deterministic citation compiler
  -> quality gate
  -> import trace
```

Expected outputs:

```txt
references_resolved.json
claims_raw.csv
claims_verified.csv
evidence_matrix.csv
manuscript_blueprint.json
full_manuscript_draft.md
compiled_manuscript.md
bibliography.json
quality_gate_report.json
import_trace.json
```

## 5. Open Design Decisions

1. V2 location: new repo vs `apps/sciwriter-v2` under this repo.
2. Local database: SQLite + Drizzle/Kysely first, PostgreSQL-compatible schema discipline.
3. RAG storage: SQLite FTS + local vector extension first, pgvector-compatible interface later.
4. Zotero write-back: defer; MVP should be read-only.
5. Citation style target: numeric/Vancouver first, CSL style switch later.
6. PDF parsing: Europe PMC XML and user text first, MinerU next, OCR fallback only when needed.
7. Agent provider policy: fixture provider required for tests; live models not required for regression.

## 6. Suggested Test Commands

```powershell
npm.cmd run lint
npm.cmd run build
npm.cmd run test
npm.cmd run pipeline:status
npm.cmd run pipeline:zotero:import-fixture
npm.cmd run pipeline:references:resolve-fixture
npm.cmd run pipeline:fulltext:audit-fixture
npm.cmd run pipeline:rag:build-fixture
npm.cmd run pipeline:claims:extract-fixture
npm.cmd run pipeline:evidence:verify-fixture
npm.cmd run pipeline:blueprint:create-fixture
npm.cmd run pipeline:manuscript:compile-fixture
npm.cmd run pipeline:manuscript:quality
npm.cmd run pipeline:quality:explain
npm.cmd run smoke:citations
npm.cmd run smoke:import-trace
```

## 7. Final Implementation Priority

```txt
schema
> Zotero/material ingestion
> reference authority
> full text/source spans
> RAG index
> evidence atoms
> claim-evidence verification
> citation compiler
> quality gate
> import trace
> fixture pipeline
> blueprint/drafting
> UI
> live agents
```
