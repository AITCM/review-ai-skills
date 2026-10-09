#!/usr/bin/env python3
"""Repository-level guard: every pre-registered E08* experiment must pass SOP.
Runs without dataset access; refuses missing/ambiguous code entrypoints.
"""
from __future__ import annotations
import json,pathlib,sys
from case_evid_policy_gate import check_experiment

root=pathlib.Path(__file__).resolve().parents[1]
policy=json.loads((root/"configs/case_evid_policy_v1.json").read_text())
files=sorted((root/"configs/experiments").glob("E*_N*_v*.json"))
if not files:
    raise SystemExit("No registered experiments present")
failures=[]
for manifest in files:
    exp=json.loads(manifest.read_text())
    expid=exp.get("experiment_id","")
    stem=expid.rsplit("_v",1)[0].lower()
    matches=sorted(p for p in (root/"tools").glob(stem+"_*.py") if not p.name.endswith(("_decision.py","_report.py","_eval.py")))
    if len(matches)!=1:
        failures.append(f"{manifest.name}: expected one source file with prefix {stem}_, found {len(matches)}")
        continue
    errors=check_experiment(policy,exp,matches[0].read_text())
    if errors:failures.append(f"{manifest.name}: "+ "; ".join(errors))
    else:print("SOP PASS",expid,"source",matches[0].relative_to(root))
if failures:
    print("SOP FAILURES:\n"+"\n".join(failures),file=sys.stderr)
    raise SystemExit(1)
print("PASS",len(files),"registered experiments; static guard does not replace human leakage audit")
