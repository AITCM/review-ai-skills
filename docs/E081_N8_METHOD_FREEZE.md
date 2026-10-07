# E081 N8 Method Freeze

Date: 2026-10-07

## Frozen method

**N8-RRF-LEX-XMOD-STRONGMASK**

New-query input:
- `case_summary` / `case_prompt` only.

Historical library:
- case prompt;
- diagnostic reasoning with the exact normalized final-diagnosis phrase masked;
- final-diagnosis tokens of length >=4 additionally masked, excluding generic clinical terms.

Retrieval routes:
1. lexical case-to-case TF-IDF;
2. direct sparse cross-modal retrieval from new case text to historical masked diagnostic reasoning.

Fusion:
- reciprocal rank fusion;
- fixed constant `c=60`;
- no learned validation-dependent fusion weight.

Primary endpoint:
- Top-10 diagnosis-masked reasoning-set Soft-F1.

Secondary endpoints:
- Top-1 and Top-3 reasoning-set Soft-F1.

## Development evidence

MedCaseReasoning validation (n=500), strong historical mask:
- Top-1 delta vs lexical: +0.00592; 95% CI +0.00176 to +0.01016.
- Top-3 delta: +0.00719; 95% CI +0.00413 to +0.01019.
- Top-10 delta: +0.00998; 95% CI +0.00804 to +0.01189.
- Excluding the 25 validation queries with literal final-diagnosis exposure: Top-10 delta +0.00969; 95% CI +0.00765 to +0.01160.

Independent validation evaluators also supported the fused route:
- S-PubMedBERT evaluator: positive at Top-1/3/10.
- char-wb 3–5 gram evaluator: positive Top-3/10.

## Frozen external validation

MedR-Bench diagnosis:
- original 957 cases;
- 117 PMCID source overlaps excluded;
- source-disjoint external cohort n=840;
- ranking used only `pmcid + case_summary`;
- external differential diagnosis and final diagnosis were loaded only after the ranking lock.

Primary Top-10:
- lexical baseline 0.05155349;
- N8-RRF 0.06043724;
- delta +0.00888375 (+17.23%);
- 95% CI +0.00808530 to +0.00968265;
- 680 improved / 160 worsened.

External char evaluator Top-10:
- lexical 0.16519647;
- N8-RRF 0.17698501;
- delta +0.01178854;
- 95% CI +0.01035377 to +0.01325619.

## Governance

The method is frozen after E081. No subsequent external evaluator, subgroup, error analysis, or test outcome may be used to alter:
- mask policy;
- vectorizer specification;
- RRF constant;
- retrieval routes;
- fusion rule.

Further work is restricted to:
- independent evaluation;
- source/provenance and near-duplicate audit;
- failure-mode analysis;
- clinician assessment;
- downstream diagnostic-impact experiments;
- manuscript/reproducibility packaging.

## Invalid exploratory result

The original N6 set-aware run used current target-set utility as an inference feature and is invalid due to outcome leakage. Its Drive artifact was explicitly renamed `E080_N6_LEAKY_INVALID_DO_NOT_USE_2026-10-07.zip`. Leakage-free N6C was rerun separately.
