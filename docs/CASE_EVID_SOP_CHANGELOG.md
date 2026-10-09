# CASE-EVID-001 SOP Changelog

## SOP v1.0 — 2026-10-09
Initial enforceable policy and machine guard.
- Freeze original N8 / E081 as historical evidence (no retroactive overwriting).
- Lock train/meta inference outputs before reading gold reasoning.
- Reserve T001 and previously viewed MedR external outcomes from future tuning.
- Fix query input contract: `pmcid + case_prompt` only; historical masked reasoning may be indexed.
- Standardize exact budgets, paired PMCID bootstrap, independent evaluation, negatives, reproducibility bundles, project folder structure.
- Require a pre-registered manifest and archived negative results.
- Define exploratory versus independent external and clinical validation claims.

Revisions must create new `SOP-vX.Y`, explain backward compatibility and state which experiments are affected; never silently mutate older experiment records.
