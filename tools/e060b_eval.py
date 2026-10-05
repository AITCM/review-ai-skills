from __future__ import annotations
import json,re,unicodedata,pathlib
from collections import defaultdict
import numpy as np,pandas as pd
from sentence_transformers import SentenceTransformer

CRB_REV="3297c5b2db51872b70645a75f43dc0b849f77621"
CRB=f"https://huggingface.co/datasets/cxyzhang/caseReportBench_ClinicalDenseExtraction_Benchmark/resolve/{CRB_REV}/data/train-00000-of-00001.parquet?download=true"
MC_REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";HF=f"https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/{MC_REV}/data"
OUT=pathlib.Path("e060b");SEED=20261004;RNG=np.random.default_rng(SEED)
MODEL="pritamdeka/S-PubMedBert-MS-MARCO";MODEL_REV="96786c7024f95c5aac7f2b9a18086c7b97b23036"
CATS=["Vitals_Hema","GI","History","Neuro","Lab_Image","CVS","ENDO","GU","RESP","MSK","EENT","DERM","Pregnancy","LYMPH"]
GENERIC={"disease","syndrome","disorder","type","deficiency","defect","with","without","of","and","the","primary","secondary","congenital","acute","chronic"}
PAT=re.compile(r'^\s*(?:\((\d+)\)|(\d+)[.)])\s*',re.M)

def norm(s):return " ".join(re.findall(r"\w+",unicodedata.normalize("NFKC",str(s)).casefold()))
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
def aggressive_mask(text,dx_items):
 x=str(text)
 for d in sorted(dx_items,key=len,reverse=True):
  if d.strip():x=re.sub(re.escape(d.strip())," diagnosismask ",x,flags=re.I)
 toks=set()
 for d in dx_items:toks|={t for t in norm(d).split() if len(t)>=4 and t not in GENERIC}
 return " ".join("diagnosismask" if t in toks else t for t in norm(x).split())
def reason_points(reason,dx):
 return [aggressive_mask(x,[str(dx)]) for x in split_reason(reason)]

ranklock=json.load(open(OUT/"E060B_RANKING_LOCK.json"));assert ranklock["status"]=="case_report_bench_rankings_locked_before_expert_facts_loaded"
coh=json.load(open(OUT/"E060B_COHORT_LOCK.json"))
ranks=[json.loads(x) for x in open(OUT/"E060B_BLIND_RANKINGS.jsonl") if x.strip()]
assert len(ranks)==138
# Expert facts loaded only here.
df=pd.read_parquet(CRB)
tr=pd.read_parquet(HF+"/train-00000-of-00001.parquet",columns=["diagnostic_reasoning","final_diagnosis"])

target={}
all_target_text=[];target_keys=[]
for i,r in df.iterrows():
 dx=as_items(r["Confirmed_Diagnosis(IEM)"])
 target[i]={}
 for cat in CATS:
  facts=[aggressive_mask(x,dx) for x in as_items(r[cat])]
  facts=[x for x in facts if x.strip()]
  target[i][cat]=[]
  for j,x in enumerate(facts):target_keys.append((int(i),cat,j));all_target_text.append(x)

needed=set()
for r in ranks:
 for key in ("baseline_top50","method_top50"):needed.update(r[key][:10])
hist={};hist_text=[];hist_keys=[]
for j in sorted(needed):
 ps=reason_points(tr.iloc[j].diagnostic_reasoning,tr.iloc[j].final_diagnosis);hist[j]=[]
 for k,p in enumerate(ps):hist_keys.append((j,k));hist_text.append(p)

texts=hist_text+all_target_text
model=SentenceTransformer(MODEL,revision=MODEL_REV,device="cpu");model.max_seq_length=256
E=model.encode(texts,batch_size=64,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True)
off=0
for z,key in zip(E[:len(hist_text)],hist_keys):
 j,k=key;hist[j].append(z)
for z,key in zip(E[len(hist_text):],target_keys):
 i,c,k=key;target[i][c].append(z)

def score(i,cands):
 H=np.vstack([np.asarray(hist[j]) for j in cands])
 catvals=[];allfacts=[]
 for c in CATS:
  if not target[i][c]:continue
  T=np.asarray(target[i][c]);sim=T@H.T
  vals=np.max(sim,axis=1);catvals.append(float(vals.mean()));allfacts.extend(list(T))
 macro=float(np.mean(catvals)) if catvals else np.nan
 T=np.asarray(allfacts)
 S=T@H.T
 rec=float(np.max(S,axis=1).mean())
 prec=float(np.max(S,axis=0).mean())
 f1=2*rec*prec/(rec+prec) if rec+prec else 0.
 return macro,rec,f1

scores={m:{k:{"macro":[],"micro":[],"f1":[]} for k in (1,3,10)} for m in ("tfidf","C2")}
for i,r in enumerate(ranks):
 for m,key in (("tfidf","baseline_top50"),("C2","method_top50")):
  for k in (1,3,10):
   ma,mi,f=score(i,r[key][:k]);scores[m][k]["macro"].append(ma);scores[m][k]["micro"].append(mi);scores[m][k]["f1"].append(f)

def summarize(ids):
 out={}
 for k in (1,3,10):
  out[str(k)]={}
  for metric in ("macro","micro","f1"):
   a=np.asarray(scores["tfidf"][k][metric])[ids];b=np.asarray(scores["C2"][k][metric])[ids];d=b-a
   keep=np.isfinite(d);a=a[keep];b=b[keep];d=d[keep];boots=[]
   for _ in range(10000):
    ix=RNG.integers(0,len(d),len(d));boots.append(float(d[ix].mean()))
   out[str(k)][metric]={"n":len(d),"TFIDF":float(a.mean()),"C2":float(b.mean()),"delta":float(d.mean()),
     "ci95":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))],
     "improved":int((d>0).sum()),"worsened":int((d<0).sum()),"tied":int((d==0).sum())}
 return out

primary=np.array(coh["primary_confirmed_dx_no_exact_phrase"],dtype=int)
noexact=np.array(coh["secondary_no_exact_phrase"],dtype=int)
allids=np.arange(138)
summary={"status":"locked_structured_clinical_evidence_external_stress_test",
 "dataset":"CaseReportBench","dataset_revision":CRB_REV,"n_all":138,
 "primary_cohort":{"definition":"confirmed diagnosis present but exact diagnosis phrase absent from raw query text","n":len(primary)},
 "independent_evaluator":{"model":MODEL,"revision":MODEL_REV,"max_seq_length":256},
 "primary_endpoint":"category-macro expert clinical-fact coverage at Top-3",
 "results_primary_53":summarize(primary),
 "results_no_exact_69":summarize(noexact),
 "results_all_138_leakage_sensitive":summarize(allids),
 "limits":["Expert dense-extraction facts are independent structured clinical evidence, not pairwise ratings of historical-case diagnostic utility.",
  "No exact final-diagnosis phrase does not exclude aliases or indirect diagnostic clues.",
  "All-138 results are leakage-sensitive and secondary because 69 raw texts contain an exact confirmed-diagnosis phrase.",
  "CaseReportBench is enriched for rare/inherited metabolic disorders and is not population representative."]}
(OUT/"E060B_STRUCTURED_EVIDENCE_RESULTS.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
with (OUT/"E060B_PER_CASE_SCORES.jsonl").open("w") as f:
 for i in range(138):
  x={"query_index":i}
  for m in ("tfidf","C2"):
   x[m]={str(k):{metric:scores[m][k][metric][i] for metric in ("macro","micro","f1")} for k in (1,3,10)}
  f.write(json.dumps(x)+"\n")
print(json.dumps(summary,indent=2,ensure_ascii=False))
