#!/usr/bin/env python3
"""CASE-EVID-001 SOP v1 manifest/source guard.
This cannot prove absence of outcome leakage: a reviewer must inspect dataflow.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, pathlib, re, sys

REQUIRED=["experiment_id","study_id","sop_version","status","hypothesis","data",
          "method","evaluation","governance"]
FORBIDDEN_SOURCE=[
    r"test-00000-of-00001\.parquet",
    r"E081_EXTERNAL_RESULTS",
    r"E081_BLIND_RANKINGS",
    r"CASE_EVID_E081\.zip",
    r"T001_TEST_RESULTS",
    r"T001_BLIND_RANKINGS"
]
ALLOWED_STATUS={"exploratory","train_internal_blind","validation_reused"}
ALLOWED_QUERY_FIELDS={"pmcid","case_prompt","case_summary"}
BAD_QUERY_FIELDS={"diagnostic_reasoning","final_diagnosis","differential_diagnosis","gold_reasoning","outcome","true_utility"}
def check_experiment(policy,exp,source_text):
    problems=[]
    for key in REQUIRED:
        if key not in exp: problems.append("missing required experiment."+key)
    if problems: return problems
    if exp["study_id"]!=policy["study_id"]:problems.append("study ID mismatch")
    if exp["sop_version"]!=policy["sop_version"]:problems.append("SOP version mismatch")
    if exp["status"] not in ALLOWED_STATUS:problems.append("wrong phase for a development experiment")
    if not re.fullmatch(r"E\d{3}_N\d+_v\d+",exp["experiment_id"]):
        problems.append("experiment_id must be E###_N#_v#")
    data=exp["data"];method=exp["method"];ev=exp["evaluation"];gov=exp["governance"]
    if data.get("dataset_revision")!=policy["dataset"]["revision"]:problems.append("dataset revision mismatch")
    if data.get("query_split")!="train_internal_meta":problems.append("new algorithm may use only train-internal meta queries")
    if data.get("test_access") is not False or data.get("old_external_access") is not False:
        problems.append("test and old external outcomes must be inaccessible")
    if not set(data.get("query_fields",[])).issubset(ALLOWED_QUERY_FIELDS):
        problems.append("query fields include forbidden gold or unknown fields")
    if any(x in BAD_QUERY_FIELDS for x in data.get("query_fields",[])):
        problems.append("query outcome field access")
    if not (data.get("group_key")=="pmcid" and data.get("grouped_split") is True):
        problems.append("PMCID grouping required")
    if method.get("baseline")!="N8-RRF-LEX-XMOD-STRONGMASK":
        problems.append("frozen N8 baseline required")
    if not method.get("candidate_budget"):
        problems.append("missing predeclared candidate budget")
    if method.get("random_seed") is None:problems.append("random seed required")
    if ev.get("primary")!=policy["metrics"]["primary"]:problems.append("primary metric mismatch")
    if not ev.get("blind_ranking_lock"):problems.append("ranking must be SHA locked before gold")
    if "char_ngram_softF1@10" not in ev.get("independent",[]):
        problems.append("independent char evaluator required")
    if not (gov.get("n8_frozen") is True and gov.get("t001_untouched") is True
            and gov.get("medr_not_used_for_tuning") is True):
        problems.append("frozen/test/external governance constraints violated")
    if not ev.get("predeclared_primary_comparison"):
        problems.append("missing named primary comparison")
    for pattern in FORBIDDEN_SOURCE:
        if re.search(pattern,source_text,re.I):
            problems.append("forbidden sealed artifact or test path referenced: "+pattern)
    try:ast.parse(source_text)
    except SyntaxError as e:problems.append(f"invalid Python: {e}")
    return problems

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--policy",required=True)
    p.add_argument("--experiment",required=True)
    p.add_argument("--source",required=True)
    args=p.parse_args()
    po=json.loads(pathlib.Path(args.policy).read_text())
    ex=json.loads(pathlib.Path(args.experiment).read_text())
    src=pathlib.Path(args.source).read_text()
    issues=check_experiment(po,ex,src)
    print(json.dumps({"experiment":ex.get("experiment_id"),"pass":not issues,
                      "problems":issues,"source_sha256":hashlib.sha256(src.encode()).hexdigest(),
                      "note":"static guard only; rank/eval source and dataflow still require human review"},
                     indent=2))
    return 1 if issues else 0
if __name__=="__main__":
    sys.exit(main())
