# CASE-EVID-001 — Algorithm Decision Ledger (2026-10-09)

**Canonical governance:** [CASE_EVID_RESEARCH_SOP_v1.md](CASE_EVID_RESEARCH_SOP_v1.md) and `configs/case_evid_policy_v1.json`.

## Frozen, exploratory and invalid results must not be conflated

| Run | Stage / population | Primary reasoning Soft-F1@10 | Paired Δ vs contemporaneous baseline | Independent evaluator | Decision |
|---|---|---:|---:|---|---|
| E081 N8 | Frozen source-disjoint MedR (n=840) | TF-IDF 0.0515535; N8-RRF 0.0604372 | +0.0088838; CI [+0.0080853,+0.0096827] | char +0.0117885; biomedical dense +0.0022472 | **FROZEN EVIDENCE, NO RE-TUNING** |
| E080 N9 | Reused validation (n=500) | N8 RRF 0.0870983; HGB 0.0883058 | +0.0012075; CI [+0.0003540,+0.0020287] | No new untouched external confirmation | **EXPLORATORY ONLY** |
| E083 N10 | Train-internal meta (n=480 PMCID-disjoint) | N8 0.0833973; point-tri RRF 0.0828695 | −0.0005279; CI [−0.0015519,+0.0005016] | independent char −0.0046529; CI [−0.0064686,−0.0028939] | **REJECT / ARCHIVE NEGATIVE** |
| E084 N11 | New train-internal meta (n=360 groups, disjoint from E083 heldout PMCID groups) | N8 0.0850119; learned marginal set-policy 0.0859934 | +0.0009814; CI [−0.0012364,+0.0031275] | independent char −0.0004927; CI [−0.0043009,+0.0034350] | **NO PROMOTION: INCONCLUSIVE** |

All metrics refer to masked reasoning-set similarity, **not clinical diagnosis accuracy**. The N8 external study was frozen only for the original N8 method; N9, N10 and N11 cannot claim independent evidence using that same external set. Differences from E083 and E084 should not be pooled because reference libraries, training, heldout groups and evaluator fitting differ.

## Leakage/metric caveats
- **INVALID**: early N6 used query true `current_u` as an inference feature; use N6C leakage-free results only.
- N8's point-style metric and its cross-modal retrieval share lexical roots. Char n-gram is an independent vectorizer/tokenization but still lexical; clinician annotation and prospective diagnosis utility are independent constructs.
- Exact diagnosis label retrieval can decline even as masked reasoning alignment rises. Maintain diagnosis-retrieval and safety guardrails instead of hiding negative secondary findings.
- Prior T001 test is not an available future algorithm-development split. Prior MedR-Bench is not an unused external confirmation set.
- E082 expert-blinded review pack exists; actual independent clinician scoring has not occurred.

## Decision and next allowed work
Keep **N8-RRF-LEX-XMOD-STRONGMASK** as the latest externally supported method. Do not promote N10 or N11. No additional tuning on their same meta holdout subsets, previously used 500-case validation, historical 897 test, or already examined MedR outcomes.

The next high-value task is construct-level validation: clinician usefulness and downstream diagnostic impact, supported by new independent groups and a genuinely different evidence label/measurement. A future model generation must have its own preregistered manifest, independent train-group query split, and frozen ranking lock under SOP v1.0.

## Permanent locations
- Study root: [Google Drive CASE-EVID-001](https://drive.google.com/drive/folders/16-pVpnvOYV9ekAMzowRM6vROpM64AkHh)
- SOP: [SOP_Method_Governance](https://drive.google.com/drive/folders/1d1BfvUZqQOteJm_DUnaJ8r1GbCdX2Auo)
- E083 N10: [E083_N10_Pointwise_TrainInternal](https://drive.google.com/drive/folders/1pQ3CIejafFZGQA8Gu_toe0DX0Zy6ZuKx)
- E084 N11: [E084_N11_SetAware_TrainInternal](https://drive.google.com/drive/folders/1qBdoEuQama5X_Ws6hSifqqUKUQDBYcJk)
- Clinical review E082: [reviewer-only folder](https://drive.google.com/drive/folders/1Sq0oiMCS2tUNNkVUTXPgyK2-AFz80CDl), [allocation key restricted](https://drive.google.com/drive/folders/1FTc77TC9mWWTfJOquzvIeekhKBFEnAeG)

## Review and reproducibility rules
Every experiment archived with: code commit SHA, pinned dataset revision, preregistered method configuration, rank lock hash, query-level outcomes, paired uncertainty, independent evaluator, failure logs and decision JSON. Changed experimental assumptions require a new manifest/version, not a silent overwrite.
