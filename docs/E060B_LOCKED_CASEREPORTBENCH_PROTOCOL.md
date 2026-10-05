# E060B Locked CaseReportBench Structured-Evidence Validation Protocol

Locked after E060A compatibility audit and before any Frozen-C2 CaseReportBench performance result.

## Purpose
Test whether the frozen T001 C2 selector retrieves historical cases whose diagnostic reasoning better covers independently expert-annotated clinical facts in CaseReportBench.

This is a structured clinical-evidence stress test, not diagnostic-accuracy validation and not a direct clinician utility judgment.

## Dataset
- CaseReportBench Clinical Dense Extraction Benchmark
- pinned revision: 3297c5b2db51872b70645a75f43dc0b849f77621
- 138 cases, 138 unique PMCIDs
- all 138 are PMCID-disjoint from every MedCaseReasoning split
- 14 expert clinical-fact categories

## Query leakage policy
Raw CaseReportBench text is used for ranking; expert category fields and confirmed diagnosis are unavailable to the ranker.

E060A found:
- 122 cases with >=1 confirmed diagnosis
- 69 cases with an exact confirmed-diagnosis phrase in raw text
- 53 cases with a confirmed diagnosis but no exact confirmed-diagnosis phrase in raw text
- 16 cases without a confirmed diagnosis

### Primary cohort
53 cases with >=1 confirmed diagnosis AND no exact confirmed-diagnosis phrase in the raw query text.

The primary cohort is fixed before ranking results are evaluated. The ranker receives no diagnosis field or cohort outcome fields.

### Secondary cohorts
- 69 cases with no exact confirmed-diagnosis phrase, including 16 with no confirmed diagnosis
- all 138 cases, explicitly labeled as leakage-sensitive secondary analysis

## Frozen ranker
Exactly the T001 C2-debiased-final-two-feature method:
- semantic case similarity
- predicted reasoning similarity
- training candidate K 40
- deployment candidate K 50
- utility margin 0.03
- case SVD 256
- reasoning SVD 128
- ridge alpha 10
- pairwise top/bottom 5/10
- logistic C 1
- seed 20261004

TF-IDF semantic ranking is the comparator.

## Independent evaluation representation
CaseReportBench expert clinical facts are loaded only after rankings are locked.

Target facts are drawn from 14 expert categories:
Vitals_Hema, GI, History, Neuro, Lab_Image, CVS, ENDO, GU, RESP, MSK, EENT, DERM, Pregnancy, LYMPH.

Confirmed-diagnosis phrases and diagnosis-specific tokens are aggressively masked from target expert facts before evaluation.
Historical MedCaseReasoning diagnostic-reasoning points are likewise diagnosis-masked for the independent evaluator.

Independent semantic evaluator:
- pritamdeka/S-PubMedBert-MS-MARCO
- pinned revision 96786c7024f95c5aac7f2b9a18086c7b97b23036
- normalized embeddings, max sequence length 256

## Endpoints
### Primary
Category-macro clinical-fact coverage at Top-3 in the 53-case primary cohort.

For each non-empty target clinical category:
1. compute, for every expert fact item, its maximum cosine similarity to any reasoning point in the selected historical evidence set;
2. average within category;
3. average equally across non-empty categories.

Compare Frozen C2 versus TF-IDF with paired query bootstrap 95% CI.

### Secondary
- category-macro coverage @1 and @10
- micro fact coverage @1/@3/@10
- symmetric fact/reasoning-set soft-F1 @1/@3/@10
- results in the 69-case no-exact-diagnosis cohort
- all-138 leakage-sensitive results
- descriptive per-category deltas

## Interpretation constraints
- Expert dense-extraction facts are clinical-evidence annotations, not pairwise ratings that one historical case is diagnostically useful for another.
- CaseReportBench focuses on rare/inherited metabolic cases and is not a population-representative clinical cohort.
- Absence of an exact final-diagnosis phrase does not guarantee absence of aliases or indirect diagnostic clues.
- No C2 parameters, cohort rule, Top-k, or endpoint may be modified after CaseReportBench results are observed.
