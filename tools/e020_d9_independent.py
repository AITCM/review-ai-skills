from __future__ import annotations
import json,re,unicodedata,pathlib
import numpy as np,pandas as pd
from sentence_transformers import SentenceTransformer
REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data";OUT=pathlib.Path("e020-d9-output");OUT.mkdir(exist_ok=True);RNG=np.random.default_rng(20261004)
MODEL="pritamdeka/S-PubMedBert-MS-MARCO";MODEL_REV="96786c7024f95c5aac7f2b9a18086c7b97b23036";PAT=re.compile(r'^\s*(?:\((\d+)\)|(\d+)[.)])\s*',re.M)
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
def mask(p,d):
 x=nw(p);dd=nw(d);return x.replace(dd," diagnosismask ") if dd and dd in x else x
def load(path,key):
 R={}
 for l in open(path):x=json.loads(l);R[int(x["query_index"])]=x[key]
 return R
def sf(q,c):
 S=q@c.T;r=float(np.max(S,axis=1).mean());p=float(np.max(S,axis=0).mean());return 2*r*p/(r+p) if r+p else 0.
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["diagnostic_reasoning","final_diagnosis"]];va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["diagnostic_reasoning","final_diagnosis"]]
base=load("d5/e020-d5-output/E020_D5_RANKINGS.jsonl","baseline_top50");method=load("d5/e020-d5-output/E020_D5_RANKINGS.jsonl","D5_top50")
needed=set()
for R in (base,method):
 for i in range(len(va)):needed.update(R[i][:10])
train_pts={};texts=[];keys=[]
for j in sorted(needed):
 ps=[mask(x,tr.iloc[j].final_diagnosis) for x in split_reason(tr.iloc[j].diagnostic_reasoning)];train_pts[j]=[]
 for k,p in enumerate(ps):keys.append(("tr",j,k));texts.append(p)
val_pts={}
for i in range(len(va)):
 ps=[mask(x,va.iloc[i].final_diagnosis) for x in split_reason(va.iloc[i].diagnostic_reasoning)];val_pts[i]=[]
 for k,p in enumerate(ps):keys.append(("va",i,k));texts.append(p)
model=SentenceTransformer(MODEL,revision=MODEL_REV,device="cpu");model.max_seq_length=256
E=model.encode(texts,batch_size=64,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True)
for z,key in zip(E,keys):
 typ,i,k=key;(train_pts if typ=="tr" else val_pts)[i].append(z)
def score(i,cands):
 q=np.asarray(val_pts[i]);c=np.vstack([np.asarray(train_pts[j]) for j in cands]);return sf(q,c)
summary={"metric":"independent S-PubMedBERT diagnosis-masked symmetric reasoning-set softF1","model":{"name":MODEL,"revision":MODEL_REV,"max_seq_length":256},"results":{}}
for k in (1,3,10):
 b=np.array([score(i,base[i][:k]) for i in range(len(va))]);m=np.array([score(i,method[i][:k]) for i in range(len(va))]);d=m-b;boots=[]
 for _ in range(5000):ids=RNG.integers(0,len(d),len(d));boots.append(float(d[ids].mean()))
 summary["results"][str(k)]={"baseline":float(b.mean()),"method":float(m.mean()),"delta":float(d.mean()),"ci95":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))],"improved":int((d>0).sum()),"worsened":int((d<0).sum()),"tied":int((d==0).sum())}
summary["limits"]=["Independent dense evaluator is not human judgment.","Exact diagnosis phrase masking does not remove aliases/acronyms.","Validation only; test sealed."]
(OUT/"E020_D9_INDEPENDENT_DEBIASED.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
