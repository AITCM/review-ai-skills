from __future__ import annotations
import json,pathlib,urllib.request,pandas as pd,hashlib
MEDR_COMMIT="ff60ab440afd2f2bc0c603b3a65d715ec83138a7"
MEDR_URL=f"https://raw.githubusercontent.com/MAGIC-AI4Med/MedRBench/{MEDR_COMMIT}/data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json"
MC_REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
HF=f"https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/{MC_REV}/data"
OUT=pathlib.Path("e081-n8-medr");OUT.mkdir(exist_ok=True)

# Build source-exclusion set from all MedCaseReasoning splits using PMCID only.
used=set()
for split in ("train","val","test"):
    df=pd.read_parquet(f"{HF}/{split}-00000-of-00001.parquet",columns=["pmcid"])
    used |= set(df.pmcid.astype(str).str.upper())

# This preparation stage reads the official source once and emits a query-only file.
# The ranking stage will have no access to the original MedR-Bench JSON.
req=urllib.request.Request(MEDR_URL,headers={"User-Agent":"CASE-EVID-001/1.0"})
with urllib.request.urlopen(req,timeout=180) as r:
    raw=r.read()
data=json.loads(raw)
queries=[]
overlap=[]
for pmcid,rec in data.items():
    pm=str(pmcid).upper()
    if pm in used:
        overlap.append(pm);continue
    g=rec["generate_case"]
    queries.append({"pmcid":pm,"case_summary":g["case_summary"]})
assert len(data)==957
assert len(overlap)==117
assert len(queries)==840
with (OUT/"E081_QUERY_ONLY.jsonl").open("w",encoding="utf-8") as f:
    for i,x in enumerate(queries):
        f.write(json.dumps({"query_index":i,**x},ensure_ascii=False)+"\n")
lock={
 "status":"query_only_external_cohort_locked",
 "medrbench_commit":MEDR_COMMIT,
 "medrbench_source_sha256":hashlib.sha256(raw).hexdigest(),
 "medcasereasoning_revision":MC_REV,
 "n_original":len(data),"n_source_overlap_excluded":len(overlap),"n_external":len(queries),
 "source_disjoint_rule":"exclude MedR-Bench PMCID appearing in any MedCaseReasoning train/val/test split",
 "query_fields":["query_index","pmcid","case_summary"],
 "outcome_fields_retained_in_query_lock":[],
 "overlap_pmcids_sha256":hashlib.sha256("\n".join(sorted(overlap)).encode()).hexdigest()
}
(OUT/"E081_QUERY_LOCK.json").write_text(json.dumps(lock,indent=2)+"\n")
print(json.dumps(lock,indent=2))
