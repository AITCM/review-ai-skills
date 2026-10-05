from __future__ import annotations
import json,re,unicodedata,pathlib,urllib.request
import numpy as np,pandas as pd
from sentence_transformers import SentenceTransformer

MEDR_COMMIT="ff60ab440afd2f2bc0c603b3a65d715ec83138a7"
MEDR_URL=f"https://raw.githubusercontent.com/MAGIC-AI4Med/MedRBench/{MEDR_COMMIT}/data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json"
MC_REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";HF=f"https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/{MC_REV}/data"
OUT=pathlib.Path("e032-output");OUT.mkdir(exist_ok=True);SEED=20261004
MODEL="pritamdeka/S-PubMedBert-MS-MARCO";MODEL_REV="96786c7024f95c5aac7f2b9a18086c7b97b23036"
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
def softf1(q,c):
 S=q@c.T
 rec=float(np.max(S,axis=1).mean());prec=float(np.max(S,axis=0).mean())
 return 2*rec*prec/(rec+prec) if rec+prec else 0.

# Consume locked E031-v2 rankings only.
locks=list(pathlib.Path("locked").rglob("E031V2_RANKING_LOCK.json"))
ranksf=list(pathlib.Path("locked").rglob("E031V2_BLIND_RANKINGS.jsonl"))
queryf=list(pathlib.Path("locked").rglob("E031V2_QUERY_ONLY.jsonl"))
assert len(locks)==len(ranksf)==len(queryf)==1
lock=json.load(open(locks[0]));assert lock["status"]=="external_rankings_locked_before_outcomes_loaded"
ranks=[json.loads(x) for x in open(ranksf[0]) if x.strip()]
queries=[json.loads(x) for x in open(queryf[0]) if x.strip()]
assert len(ranks)==len(queries)==840

req=urllib.request.Request(MEDR_URL,headers={"User-Agent":"CASE-EVID-001/1.0"})
with urllib.request.urlopen(req,timeout=180) as rr:data=json.load(rr)
tr=pd.read_parquet(HF+"/train-00000-of-00001.parquet",columns=["diagnostic_reasoning","final_diagnosis"])

needed=set()
for r in ranks:
 for key in ("baseline_top50","method_top50"): needed.update(r[key][:10])

texts=[];keys=[];tp={};vp={};explicit={}
for j in sorted(needed):
 ps=pts_medcase(tr.iloc[j].diagnostic_reasoning,tr.iloc[j].final_diagnosis);tp[j]=[]
 for k,p in enumerate(ps):keys.append(("tr",j,k));texts.append(p)
for i,q in enumerate(queries):
 g=data[q["pmcid"]]["generate_case"];ps=mask_final_tokens(split_medr_diff(g["differential_diagnosis"]),g["final_diagnosis"]);vp[i]=[]
 explicit[i]=bool(nw(g["final_diagnosis"]) and nw(g["final_diagnosis"]) in nw(q["case_summary"]))
 for k,p in enumerate(ps):keys.append(("va",i,k));texts.append(p)

model=SentenceTransformer(MODEL,revision=MODEL_REV,device="cpu");model.max_seq_length=256
E=model.encode(texts,batch_size=64,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True)
for z,key in zip(E,keys):
 typ,i,k=key;(tp if typ=="tr" else vp)[i].append(z)

scores={"baseline":{k:[] for k in (1,3,10)},"method":{k:[] for k in (1,3,10)}}
for i,r in enumerate(ranks):
 for name,key in [("baseline","baseline_top50"),("method","method_top50")]:
  for k in (1,3,10):
   q=np.asarray(vp[i]);c=np.vstack([np.asarray(tp[j]) for j in r[key][:k]])
   scores[name][k].append(softf1(q,c))

rng=np.random.default_rng(SEED)
summary={"status":"independent_evaluator_on_locked_E031V2_rankings","n_external":840,
 "model":{"name":MODEL,"revision":MODEL_REV,"max_seq_length":256},"results":{},"subgroup_query_dx_not_explicit":{}}
for k in (1,3,10):
 b=np.asarray(scores["baseline"][k]);m=np.asarray(scores["method"][k]);d=m-b;boots=[]
 for _ in range(5000):
  ids=rng.integers(0,len(d),len(d));boots.append(float(d[ids].mean()))
 summary["results"][str(k)]={"baseline":float(b.mean()),"method":float(m.mean()),"delta":float(d.mean()),
  "ci95":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))],
  "improved":int((d>0).sum()),"worsened":int((d<0).sum()),"tied":int((d==0).sum())}
 ids=[i for i in range(len(d)) if not explicit[i]];dd=d[ids]
 summary["subgroup_query_dx_not_explicit"][str(k)]={"n":len(ids),"mean_delta":float(dd.mean())}
summary["limits"]=["Independent dense evaluator is not clinician judgment.","The two benchmarks use different reasoning annotation schemas.","Rankings are inherited unchanged from E031-v2."]
(OUT/"E032_INDEPENDENT_EXTERNAL_RESULTS.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
