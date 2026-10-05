from __future__ import annotations
import json,re,unicodedata,pathlib,hashlib
import numpy as np,pandas as pd

CRB_REV="3297c5b2db51872b70645a75f43dc0b849f77621"
CRB=f"https://huggingface.co/datasets/cxyzhang/caseReportBench_ClinicalDenseExtraction_Benchmark/resolve/{CRB_REV}/data/train-00000-of-00001.parquet?download=true"
MC_REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
HF=f"https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/{MC_REV}/data"
OUT=pathlib.Path("e060b");OUT.mkdir(exist_ok=True)

def flatten(v):
 if v is None:return ""
 if isinstance(v,np.ndarray):v=v.tolist()
 if isinstance(v,(list,tuple,set)):return " ".join(flatten(x) for x in v if flatten(x))
 if isinstance(v,dict):return " ".join(flatten(x) for x in v.values() if flatten(x))
 try:
  if bool(pd.isna(v)):return ""
 except Exception:pass
 return str(v).strip()
def as_items(v):
 if v is None:return []
 if isinstance(v,np.ndarray):v=v.tolist()
 if isinstance(v,(list,tuple,set)):return [flatten(x) for x in v if flatten(x)]
 if isinstance(v,dict):return [flatten(x) for x in v.values() if flatten(x)]
 try:
  if bool(pd.isna(v)):return []
 except Exception:pass
 s=str(v).strip();return [s] if s and s.lower() not in {"nan","none","[]","{}","null"} else []
def has_exact(text,items):
 low=unicodedata.normalize("NFKC",text).casefold()
 return any(unicodedata.normalize("NFKC",x).casefold().strip() in low for x in items if x.strip())

df=pd.read_parquet(CRB)
assert len(df)==138
# Source disjointness asserted independently from outcome.
used=set()
for split in ("train","val","test"):
 x=pd.read_parquet(f"{HF}/{split}-00000-of-00001.parquet",columns=["pmcid"])
 used|=set(x.pmcid.astype(str).str.upper())
assert sum(str(x).upper() in used for x in df.pmcid)==0

queries=[];primary=[];noexact=[];nodx=[];explicit=[]
for i,r in df.iterrows():
 text=flatten(r["text"]);dx=as_items(r["Confirmed_Diagnosis(IEM)"]);ex=has_exact(text,dx)
 queries.append({"query_index":int(i),"pmcid":str(r.pmcid).upper(),"text":text})
 if dx and not ex:primary.append(int(i))
 if not ex:noexact.append(int(i))
 if not dx:nodx.append(int(i))
 if ex:explicit.append(int(i))
assert len(primary)==53 and len(noexact)==69 and len(nodx)==16 and len(explicit)==69
with (OUT/"E060B_QUERY_ONLY.jsonl").open("w",encoding="utf-8") as f:
 for x in queries:f.write(json.dumps(x,ensure_ascii=False)+"\n")
subset={"status":"cohorts_locked_before_rankings","primary_confirmed_dx_no_exact_phrase":primary,
 "secondary_no_exact_phrase":noexact,"no_confirmed_dx":nodx,"explicit_dx_phrase":explicit,
 "counts":{"all":138,"primary":53,"no_exact":69,"no_confirmed_dx":16,"explicit_dx":69}}
(OUT/"E060B_COHORT_LOCK.json").write_text(json.dumps(subset,indent=2)+"\n")
lock={"status":"query_only_case_report_bench_locked","dataset_revision":CRB_REV,"n_queries":138,
 "query_fields":["query_index","pmcid","text"],"expert_fact_fields_available_to_ranker":[],
 "confirmed_diagnosis_available_to_ranker":[],"source_disjoint_from_all_medcase":True,
 "query_sha256":hashlib.sha256((OUT/"E060B_QUERY_ONLY.jsonl").read_bytes()).hexdigest()}
(OUT/"E060B_QUERY_LOCK.json").write_text(json.dumps(lock,indent=2)+"\n")
print(json.dumps({"query_lock":lock,"cohorts":subset["counts"]},indent=2))
