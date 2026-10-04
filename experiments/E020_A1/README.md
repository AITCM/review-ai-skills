# E020-A1 Case Utility Landscape

Dataset revision: 469a5365bc534b5b2b9cfbc52b2ef2a10f43a339

## Main findings
- 299/500 validation cases have an exact normalized diagnosis label represented in the training split; 201/500 do not.
- Among these 299 eligible cases, TF-IDF retrieves an exact-label case at rank 1 for 25 (8.36%), rank <=10 for 85 (28.43%), and rank <=50 for 145 (48.49%).
- BM25 is similar: rank 1 for 29 (9.70%), rank <=10 for 87 (29.10%), and rank <=50 for 148 (49.50%).
- In 274/299 eligible queries (91.64%), the most text-similar different-label case is more similar than the most text-similar same-label case.
- The same-label candidate with the strongest privileged reasoning similarity has median case-text rank 533; only 14.38% lie in the top 10 by case-text similarity.
- Median Spearman correlation between case-text similarity and reasoning similarity within the TF-IDF top-10 is 0.164.

## Interpretation boundary
Exact diagnosis-string equality is only a proxy for diagnostic compatibility. Different diagnoses may still be useful for differential diagnosis. Validation reasoning is used here only for retrospective landscape analysis and is not available at deployment. The test set was not used.

## Execution
GitHub Actions run: 37169386694
Artifact: E020-A1-Fast-Case-Utility-Landscape
