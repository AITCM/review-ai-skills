from __future__ import annotations
import json,re,unicodedata,pathlib,urllib.request,hashlib
import numpy as np,pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer

MEDR_COMMIT="ff60ab440afd2f2bc0c603b3a65d715ec83138a7"
MEDR_URL=f"https://raw.githubusercontent.com/MAGIC-AI4Med/MedRBench/{MEDR_COMMIT}/data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json"
MC_REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
HF=f"https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/{MC_REV}/data"
OUT=pathlib.Path("e031-v2");SEED=20261004;RNG=np.random.default_rng(SEED)
PAT=re.compile(r'^\s*(?:\((\d+)\)|(\d+)[.)])\s*',re.M)
GENERIC={"with","without","acute","chronic","disease","syndrome","secondary","primary","type","and","the","of","disorder"}
def nw(s):return " ".join(re.findall(r"\w+",unicodedata.normalize("NFKC",str(s)).casefold()))
def split_reason(s):
 s=str(s);ms=list(PAT.finditer(s))
 if not ms or s[:ms[0].start()].strip():return [s.strip()]
 nums=[int(m.group(1) or m.group(2)) for m in ms]
 if nums!=list(range(1,len(ms)+1)):return [s.strip()]
 out=[]
 for i,m in enumerate(ms):
  e=ms[i+1].start() if i+1<len(ms) else len(s);x=s[m.end():e].strip()
  if x:out.append(x)
 return out or [s.strip()]
def pts_medcase(r,d):
 dd=nw(d);o=[]
 for x in split_reason(r):
  xx=nw(x)
  if dd and dd in xx:xx=xx.replace(dd," diagnosismask ")
  o.append(xx)
 return o
def split_medr_diff(s):
 s=str(s);chunks=re.split(r'(?m)(?=^\s*\d+\.\s+)',s);out=[]
 for x in chunks:
  x=x.strip()
  if not x:continue
  x=re.sub(r'^\d+\.\s*','',x)
  x=re.sub(r'^\*\*[^*]{1,160}\*\*\s*:\s*','',x)
  x=nw(x)
  if x:out.append(x)
 return out or [nw(s)]
def mask_final_tokens(points,final_dx):
 toks={t for t in nw(final_dx).split() if len(t)>=4 and t not in GENERIC}
 return [" ".join("diagnosismask" if t in toks else t for t in p.split()) for p in points]
def sf(q,c):
 S=(q@c.T).toarray()
 if not S.size:return 0.
 r=float(np.max(S,axis=1).mean());p=float(np.max(S,axis=0).mean())
 return 2*r*p/(r+p) if r+p else 0.

lock=json.load(open(OUT/"E031V2_RANKING_LOCK.json"))
assert lock["status"]=="external_rankings_locked_before_outcomes_loaded"
assert lock["external_outcome_fields_available_to_ranker"]==[]
queries=[json.loads(x) for x in open(OUT/"E031V2_QUERY_ONLY.jsonl",encoding="utf-8") if x.strip()]
ranks=[json.loads(x) for x in open(OUT/"E031V2_BLIND_RANKINGS.jsonl",encoding="utf-8") if x.strip()]
assert len(queries)==len(ranks)==840
assert all(q["pmcid"]==r["pmcid"] for q,r in zip(queries,ranks))

# Outcomes are loaded only here, after lock validation.
req=urllib.request.Request(MEDR_URL,headers={"User-Agent":"CASE-EVID-001/1.0"})
with urllib.request.urlopen(req,timeout=180) as rr: raw=rr.read()
data=json.loads(raw)
qmap={x["pmcid"]:x for x in queries}
rows=[]
for q in queries:
 rec=data[q["pmcid"]];g=rec["generate_case"]
 rows.append({"pmcid":q["pmcid"],"case_summary":q["case_summary"],"differential_diagnosis":g["differential_diagnosis"],
              "final_diagnosis":g["final_diagnosis"],"rare":bool(rec.get("checked_rare_disease")),"body_category":rec.get("body_category")})
ext=pd.DataFrame(rows)

tr=pd.read_parquet(HF+"/train-00000-of-00001.parquet",columns=["diagnostic_reasoning","final_diagnosis"])
tp=[pts_medcase(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
ext_pts=[];parse_counts=[];query_label_explicit=[]
for _,r in ext.iterrows():
 ps=mask_final_tokens(split_medr_diff(r.differential_diagnosis),r.final_diagnosis)
 ext_pts.append(ps);parse_counts.append(len(ps))
 dx=nw(r.final_diagnosis);cs=nw(r.case_summary)
 query_label_explicit.append(bool(dx and dx in cs))

vec=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
T=vec.fit_transform([p for z in tp for p in z]);offs=[];o=0
for z in tp:offs.append((o,o+len(z)));o+=len(z)
V=vec.transform([p for z in ext_pts for p in z]);vo=[];o=0
for z in ext_pts:vo.append((o,o+len(z)));o+=len(z)
def score(i,cands):
 s,e=vo[i];q=V[s:e];m=[]
 for j in cands:a,b=offs[j];m.append(T[a:b])
 return sf(q,vstack(m))

summary={"status":"strict_two_phase_source_disjoint_external_validation",
 "dataset":"MedR-Bench diagnosis","medrbench_commit":MEDR_COMMIT,
 "medrbench_source_sha256":hashlib.sha256(raw).hexdigest(),
 "n_original":957,"n_source_overlap_excluded":117,"n_external":840,
 "method":"T001 frozen C2-debiased-final-two-feature",
 "query_fields_used_for_ranking":["pmcid","case_summary"],
 "outcomes_loaded_only_after_ranking_lock":True,
 "target_reasoning":"differential_diagnosis; numbered items parsed; leading bold diagnosis label stripped; final-diagnosis tokens aggressively masked",
 "parse_reason_points":{"median":float(np.median(parse_counts)),"min":int(min(parse_counts)),"max":int(max(parse_counts))},
 "query_final_diagnosis_explicit_count":int(sum(query_label_explicit)),
 "results":{},"subgroups":{}}
for k in (1,3,10):
 b=np.array([score(i,ranks[i]["baseline_top50"][:k]) for i in range(len(ext))])
 m=np.array([score(i,ranks[i]["method_top50"][:k]) for i in range(len(ext))]);d=m-b;boots=[]
 for _ in range(10000):
  ids=RNG.integers(0,len(d),len(d));boots.append(float(d[ids].mean()))
 summary["results"][str(k)]={"baseline":float(b.mean()),"method":float(m.mean()),"delta":float(d.mean()),
  "relative_delta_pct":float(100*d.mean()/b.mean()) if b.mean()!=0 else None,
  "ci95":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))],
  "improved":int((d>0).sum()),"worsened":int((d<0).sum()),"tied":int((d==0).sum())}
 groups={
  "rare":[i for i,x in enumerate(ext.rare) if x],
  "not_marked_rare":[i for i,x in enumerate(ext.rare) if not x],
  "query_dx_not_explicit":[i for i,x in enumerate(query_label_explicit) if not x],
  "query_dx_explicit":[i for i,x in enumerate(query_label_explicit) if x]}
 for name,ids in groups.items():
  if ids:
   dd=d[ids];summary["subgroups"].setdefault(name,{})[str(k)]={"n":len(ids),"mean_delta":float(dd.mean())}
summary["limits"]=[
 "MedR-Bench and MedCaseReasoning both derive from published case reports; PMCID disjointness removes exact source overlap but not domain similarity.",
 "MedR-Bench differential_diagnosis and MedCaseReasoning diagnostic_reasoning are different annotation schemas.",
 "The metric is cross-schema reasoning-set alignment, not clinician-adjudicated clinical utility.",
 "Query summaries may occasionally contain the final diagnosis; a diagnosis-not-explicit subgroup is reported.",
 "No parameters or method components were changed using MedR-Bench outcomes."
]
(OUT/"E031V2_EXTERNAL_RESULTS.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
with (OUT/"E031V2_PER_CASE.jsonl").open("w",encoding="utf-8") as f:
 for i in range(len(ext)):
  f.write(json.dumps({"query_index":i,"pmcid":ext.iloc[i].pmcid,"rare":bool(ext.iloc[i].rare),
   "query_final_diagnosis_explicit":query_label_explicit[i]},ensure_ascii=False)+"\n")
print(json.dumps(summary,indent=2,ensure_ascii=False))
