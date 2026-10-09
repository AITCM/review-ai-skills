#!/usr/bin/env python3
"""Post-lock exploratory decision: do not promote to confirmatory without new cohort."""
import json,pathlib
base=pathlib.Path("e083-n10-output")
s=json.loads((base/"E083_N10_SUMMARY.json").read_text())
assert s["status"]=="train_internal_exploratory_outcome_opened_only_after_rank_lock"
d=s["result"]["primary_sparse"]["N10_RRF_LEX_XMOD_POINT"]["10"]
c=s["result"]["independent_char"]["N10_RRF_LEX_XMOD_POINT"]["10"]
ok=(d["delta_vs_N8"]>0 and c["delta_vs_N8"]>0)
stat=(d["cluster_ci95_delta_vs_N8"][0]>0 and c["cluster_ci95_delta_vs_N8"][0]>0)
decision={
 "study_id":"CASE-EVID-001","experiment_id":"E083_N10_v1",
 "sop":"SOP-1.0",
 "decision":"DEVELOPMENT_CANDIDATE_FOR_NEW_EXTERNAL_COHORT" if ok and stat else "HYPOTHESIS_NOT_SUPPORTED_OR_INCONCLUSIVE",
 "exploratory":True,"new_external_confirmation_done":False,
 "primary_delta":d["delta_vs_N8"],"primary_ci95":d["cluster_ci95_delta_vs_N8"],
 "char_delta":c["delta_vs_N8"],"char_ci95":c["cluster_ci95_delta_vs_N8"],
 "no_confirmatory_claim":True,
 "followup":"Do not tune to T001, reused val, or existing MedR; archive this run; if supported, preregister completely new external dataset."
}
(base/"E083_N10_DECISION.json").write_text(json.dumps(decision,indent=2)+"\n")
print(json.dumps(decision,indent=2))
