# E040 Locked Statistical Analysis Protocol

**External Validation 2:** AgentClinic-NEJM clinician-perceived case-evidence utility.

Locked before any clinician ratings are collected.

- 120 AgentClinic-NEJM cases total.
- Three cases have identical unordered TF-IDF and C2 Top-3 evidence sets and are structural automatic ties.
- 117 cases are rated independently by Rater A and Rater B.
- Raters receive independently randomized A/B pack identities.
- Target gold diagnosis and method identity are hidden.
- Primary question: which evidence set is more useful for diagnostic reasoning?
- Response options: A clearly better; A slightly better; about equal; B slightly better; B clearly better; neither useful.
- C2/TF-IDF identities are decoded only after both raters submit complete response files.

## Primary endpoint

After adjudication of discordant directional judgments, compare:
- number of cases preferring C2
- number of cases preferring TF-IDF

"About equal", "Neither useful", and the three structural identical-set cases are ties and excluded from the directional denominator.

Report:
- C2 win fraction among directional non-tie cases
- exact Clopper-Pearson 95% CI
- two-sided exact binomial test against 0.5

## Agreement

Before adjudication:
- directional percent agreement across C2 / tie / TF-IDF
- unweighted Cohen kappa on C2 / tie / TF-IDF
- exact six-category agreement

## Adjudication

Use a third clinician only for cases where Rater A and Rater B do not agree on direction, including one directional choice vs tie. The adjudicator returns C2 / TF-IDF / tie after reviewing the same blinded evidence. Method identities remain hidden during adjudication.

## Secondary descriptive outcomes

- ordinal preference strength from -2 (TF-IDF clearly better) to +2 (C2 clearly better)
- confidence (1–5)
- potentially misleading evidence flags
- qualitative comments

No changes to retrieval method, Top-k, cases, randomization, or primary analysis are permitted after clinician labels are observed.
