from __future__ import annotations
import json,re,unicodedata,pathlib
import numpy as np,pandas as pd
from sentence_transformers import SentenceTransformer
REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e020-c5-output");OUT.mkdir(exist_ok=True);SEED=20261004
MODEL="pritamdeka/S-PubMedBert-MS-MARCO";MODEL_REV="96786c7024f95c5aac7f2b9a18086c7b97b23036"
PAT=re.compile(r'^\s*(?:\((\d+)\)|(\d+)[.)])\s*',re.M)
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
def mask_point(p,dx):
 pt=nw(p);dd=nw(dx)
 return pt.replace(dd," diagnosismask ") if dd and dd in pt else pt
def load_rank(path,key):
 rows={}
 for line in open(path,encoding="utf-8"):
  x=json.loads(line);rows[int(x["query_index"])]=x[key]
 return rows

tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["diagnostic_reasoning","final_diagnosis"]]
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["diagnostic_reasoning","final_diagnosis"]]
base=load_rank("c2/e020-c2-output/E020_C2_RANKINGS.jsonl","baseline_top50")
c2=load_rank("c2/e020-c2-output/E020_C2_RANKINGS.jsonl","C2_top50")
c4=load_rank("c4/e020-c4-output/E020_C4_RANKINGS.jsonl","C4_ranked_union")
methods={"tfidf":base,"C2":c2,"C4":c4}
needed=set()
for R in methods.values():
 for i in range(len(va)):needed.update(R[i][:10])
train_points={};texts=[];keys=[]
for j in sorted(needed):
 ps=[mask_point(x,tr.iloc[j].final_diagnosis) for x in split_reason(tr.iloc[j].diagnostic_reasoning)]
 train_points[j]=[]
 for k,p in enumerate(ps):keys.append(("tr",j,k));texts.append(p)
val_points={}
for i in range(len(va)):
 ps=[mask_point(x,va.iloc[i].final_diagnosis) for x in split_reason(va.iloc[i].diagnostic_reasoning)]
 val_points[i]=[]
 for k,p in enumerate(ps):keys.append(("va",i,k));texts.append(p)
model=SentenceTransformer(MODEL,revision=MODEL_REV,device="cpu");model.max_seq_length=256
E=model.encode(texts,batch_size=64,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True)
for z,key in zip(E,keys):
 typ,i,k=key;(train_points if typ=="tr" else val_points)[i].append(z)
def coverage(qi,cands):
 q=np.asarray(val_points[qi]);c=np.vstack([np.asarray(train_points[j]) for j in cands])
 return float(np.max(q@c.T,axis=1).mean())
scores={m:{k:[] for k in (1,3,10)} for m in methods}
for m,R in methods.items():
 for i in range(len(va)):
  for k in (1,3,10):scores[m][k].append(coverage(i,R[i][:k]))
summary={"metric":"independent diagnosis-name-masked reasoning-point coverage via S-PubMedBERT",
 "model":{"name":MODEL,"revision":MODEL_REV,"max_seq_length":256},
 "n_validation":len(va),"unique_train_cases_embedded":len(needed),"results":{},"paired_bootstrap_vs_tfidf":{}}
for m in methods:summary["results"][m]={f"coverage@{k}_mean":float(np.mean(scores[m][k])) for k in (1,3,10)}
rng=np.random.default_rng(SEED)
for m in ("C2","C4"):
 summary["paired_bootstrap_vs_tfidf"][m]={}
 for k in (1,3,10):
  d=np.asarray(scores[m][k])-np.asarray(scores["tfidf"][k]);boots=[]
  for _ in range(5000):
   ids=rng.integers(0,len(d),len(d));boots.append(float(d[ids].mean()))
  summary["paired_bootstrap_vs_tfidf"][m][str(k)]={"delta":float(d.mean()),
   "ci95":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))],
   "improved":int(np.sum(d>0)),"worsened":int(np.sum(d<0)),"tied":int(np.sum(d==0))}
summary["limits"]=["Reasoning annotations are dataset-derived rationales, not exhaustive clinical truth.",
 "Exact diagnosis phrases are masked, but aliases/acronyms may remain.",
 "S-PubMedBERT is independent from the TF-IDF utility supervision but is not human judgment.",
 "Validation only; test remains sealed."]
(OUT/"E020_C5_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
with (OUT/"E020_C5_PER_CASE.jsonl").open("w") as f:
 for i in range(len(va)):f.write(json.dumps({"query_index":i,**{m:{str(k):scores[m][k][i] for k in (1,3,10)} for m in methods}})+"\n")
print(json.dumps(summary,indent=2))
