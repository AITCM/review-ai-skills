# E050B PMC-Patients Human-Relevance Validation — Result Note

## Formal run
`E050B V5 Final PMC Human Relevance Validation`

Frozen C2 and all cohort/mapping rules were fixed before human labels were loaded.

## Final source-clean cohort
- 129 high-confidence annotated pairs
- 115 human-evaluation queries
- 120 unique mapped candidate patients
- 4 same-PMC-source query-candidate pairs excluded before scoring
- 121 pairs / 108 queries are source-disjoint from all MedCaseReasoning splits at the query-source level

## Human label construct
PMC-Patients patient-patient expert labels:
- 0 = Dissimilar
- 1 = Features
- 2 = Outcomes
- 3 = Exposure
- combinations = similarity on multiple dimensions

Binary primary label: any nonzero similarity versus Dissimilar.

## Primary result
- TF-IDF AUROC: 0.5441
- Frozen C2 AUROC: 0.5339
- delta: -0.0102
- query-cluster bootstrap 95% CI: [-0.0687, 0.0482]

- TF-IDF average precision: 0.7882
- Frozen C2 average precision: 0.7618
- delta: -0.0265
- query-cluster bootstrap 95% CI: [-0.0704, 0.0189]

Interpretation: no evidence that Frozen C2 improves alignment with generic expert patient-patient similarity labels.

## Pre-specified actionable subset
Baseline TF-IDF rank <= 50, where Frozen C2 is structurally able to rerank.

- n=94 pairs / 85 queries
- TF-IDF AUROC: 0.5735
- Frozen C2 AUROC: 0.5539
- delta: -0.0196
- 95% CI: [-0.1291, 0.0911]

No evidence of improvement here either.

## Source-disjoint sensitivity
- TF-IDF AUROC: 0.5418
- Frozen C2 AUROC: 0.5350
- delta: -0.0069

Direction remains neutral/slightly negative.

## Selective promotion
Mean C2-minus-TFIDF rank-score change:
- human-similar pairs: -0.000202
- dissimilar pairs: -0.000222
- difference: +0.0000206
- 95% CI: [-0.000270, 0.000297]

No selective promotion signal.

## How this should be used
This is a negative/neutral auxiliary validation and must be reported as such.

It should NOT be used to claim that C2 matches generic human patient similarity better than TF-IDF.

The result is not inconsistent with E031 because the constructs differ:
- E031 evaluates cross-schema diagnostic reasoning-set utility.
- E050B evaluates PMC-Patients patient-patient similarity dimensions (Features / Outcomes / Exposure).

Thus E050B cannot replace the planned E040 clinician judgment of diagnostic reasoning utility.

## Important limitation
Human labels cover only five pre-retrieved candidates per query, not exhaustive relevance judgments over the MedCaseReasoning library. Only 129 high-confidence annotated pairs could be mapped after source and patient-level quality controls.

## Engineering audit
- initial E050B run falsely appeared successful because shell pipefail/output assertions were missing; invalid
- V2 exposed same-source query-candidate leakage and stopped before labels
- V3/V4 were implementation/debug runs
- V5 is the first complete formal run with source-clean cohort, frozen score lock, locked evaluation, and output assertions
