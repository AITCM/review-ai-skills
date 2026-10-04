# E020 Final Confirmatory Analysis Plan

Project: CASE-EVID-001 — Learning to Select and Use Clinical Case Evidence for Diagnostic Reasoning

Status: frozen before confirmatory test evaluation.

## Locked method
- Working name: C2-debiased-final-two-feature.
- Query-time input: case_prompt only.
- Historical privileged information: diagnostic_reasoning with exact normalized final-diagnosis phrase masked.
- Utility supervision: symmetric reasoning-set soft F1.
- Ranking features: semantic case-text similarity + predicted reasoning similarity only.
- Primary meta-split seed: 20261004.
- Case SVD: 256; reasoning SVD: 128; ridge alpha: 10.
- Train utility candidate pool: 40.
- Pairwise utility margin: 0.03.
- Top utility examples per meta-query: 5; bottom utility examples: 10.
- Pairwise logistic C: 1.0.
- Deployment candidate pool: TF-IDF Top-50.
- No diversity/MMR module.

## Confirmatory test protocol
The test split is opened exactly once after D11-D14 are complete and acceptable. No parameter, feature, masking, utility definition, candidate K, seed, or metric may be changed based on test results.

### Primary endpoint
Paired difference in diagnosis-masked symmetric reasoning-set soft F1 for the retrieved Top-10 evidence set:
C2-debiased-final-two-feature minus canonical TF-IDF.

Report:
- mean in each method
- paired mean difference
- 95% paired bootstrap CI over test queries (10,000 resamples, seed 20261004)
- number of queries improved, worsened, tied

### Secondary endpoints
- symmetric reasoning-set soft F1 at Top-1 and Top-3
- exact normalized diagnosis Hit@1 / Hit@3 / Hit@10 as a secondary proxy only
- results stratified by whether the exact normalized diagnosis is represented in the training library
- reasoning-point count selected by each method, to monitor annotation-density bias

### Independent robustness endpoint
Repeat symmetric reasoning-set soft F1 with S-PubMedBERT reasoning-point embeddings:
- model: pritamdeka/S-PubMedBert-MS-MARCO
- revision: 96786c7024f95c5aac7f2b9a18086c7b97b23036
- max_seq_length: 256
- Top-1 / Top-3 / Top-10
- paired bootstrap CI, same test queries

## Interpretation rules
- Exact-label mismatch is not equated with clinically useless evidence.
- Reasoning annotations are dataset-derived rationales, not exhaustive clinical truth.
- The method is an evidence-selection algorithm, not a diagnostic decision system.
- No causal or patient-care benefit claim is permitted from this benchmark alone.
- If the primary Top-10 test CI crosses zero, the method is not claimed to improve evidence utility on the confirmatory test set.
- Secondary endpoints cannot rescue a failed primary endpoint.
