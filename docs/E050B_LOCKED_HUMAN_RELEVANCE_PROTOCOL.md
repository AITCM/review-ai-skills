# E050B Locked Human-Relevance Analysis Protocol

Locked before Phase 2 human labels are loaded.

## Purpose
Evaluate whether the frozen T001 C2 case-evidence selector aligns better than the lexical TF-IDF baseline with PMC-Patients expert patient-patient similarity judgments.

This is a **construct-validity / hard-candidate discrimination** study, not a full retrieval benchmark.

## Frozen mapping population
Primary population:
- patient-patient annotated pairs whose candidate source maps to MedCaseReasoning train;
- patient-level mapping is high confidence under the E050A rule:
  - single-patient source article, OR
  - candidate is the unique best patient-text match to the MedCase case prompt with char-5gram cosine margin >= 0.05.
- expected locked population: 133 annotated pairs across 118 queries, representing 121 unique candidate patient UIDs.

Human relevance labels are not used to define this mapping population.

## Frozen model
Exactly the T001 C2-debiased-final-two-feature configuration:
- semantic case similarity
- predicted reasoning similarity
- training candidate K=40
- deployment semantic candidate K=50
- utility margin=0.03
- case SVD=256
- reasoning SVD=128
- ridge alpha=10
- pairwise top=5 / bottom=10
- logistic C=1
- seed=20261004

Same-source MedCase train reference is excluded for each human query before ranks are computed.

## Primary human label
Binary:
- Dissimilar: raw human label == 0
- Similar: any non-empty label containing one or more of 1/2/3

PMC-Patients meanings:
- 1 = Features
- 2 = Outcomes
- 3 = Exposure
- combined strings = similar on multiple dimensions.

## Primary score
Normalized full-library rank score: 1 - (rank - 1) / (N_allowed - 1).

C2 reranks only the frozen semantic Top-50; all candidates after Top-50 retain baseline TF-IDF order.

## Primary endpoints
1. AUROC for Similar vs Dissimilar using rank score:
   - TF-IDF
   - Frozen C2
   - delta AUROC (C2 - TF-IDF)
2. Average precision:
   - TF-IDF
   - Frozen C2
   - delta AP
3. Query-cluster bootstrap 95% CIs for deltas, resampling query IDs.

## Pre-specified structural sensitivity
**Actionable subset:** annotated high-confidence pairs with baseline TF-IDF rank <= 50.

Rationale: Frozen C2 can only change order inside the baseline semantic Top-50. This subset is defined by method structure, without human labels.

Report AUROC/AP and rank-score promotion for this subset separately; it is a sensitivity analysis, not a replacement primary endpoint.

## Additional analyses
- source-disjoint query sensitivity: query source PMCID absent from all MedCaseReasoning train/validation/test splits;
- dimension-specific descriptive discrimination:
  - Features vs Dissimilar
  - Outcomes vs Dissimilar
  - Exposure vs Dissimilar
- selective promotion:
  mean (C2 rank score - TF-IDF rank score) for human Similar pairs minus the corresponding mean for Dissimilar pairs, with query-cluster bootstrap CI.
- report the number of mixed-label queries with >=1 Similar and >=1 Dissimilar mapped candidate; do not overinterpret within-query tests if this number is small.

## Interpretation constraints
- Human annotations cover five candidates per query, not the entire MedCaseReasoning candidate library.
- Unannotated MedCaseReasoning cases must not be treated as irrelevant.
- This study cannot estimate full-corpus Recall@K or nDCG.
- Patient-patient similarity is related to, but not identical to, diagnostic reasoning utility.
- No Frozen C2 parameter may be changed using these human labels.