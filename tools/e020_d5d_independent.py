from __future__ import annotations
import json,re,unicodedata,pathlib
import numpy as np,pandas as pd
from sentence_transformers import SentenceTransformer
REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e020-d5d-output");OUT.mkdir(exist_ok=True);SEED=20261004
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
def mask_point(p,d):
 x=nw(p);dd=nw(d);return x.replace(dd," diagnosismask ") if dd and dd in x else x
def load(path,key):
 R={}
 for l in open(path):x=json.loads(l);R[int(x["query_index"])]=x[key]
 return R
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["diagnostic_reasoning","final_diagnosis"]]
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["diagnostic_reasoning","final_diagnosis"]]
base=load("d5/e020-d5-output/E020_D5_RANKINGS.jsonl","baseline_top50");d5=load("d5/e020-d5-output/E020_D5_RANKINGS.jsonl","D5_top50")
methods={"tfidf":base,"D5_debiased":d5}
needed=set()
for R in methods.values():
 for i in range(len(va)):needed.update(R[i][:10])
texts=[];keys=[];tp={};vp={}
for j in sorted(needed):
 ps=[mask_point(x,tr.iloc[j].final_diagnosis) for x in split_reason(tr.iloc[j].diagnostic_reasoning)];tp[j]=[]
 for k,p in enumerate(ps):keys.append(("tr",j,k));texts.append(p)
for i in range(len(va)):
 ps=[mask_point(x,va.iloc[i].final_diagnosis) for x in split_reason(va.iloc[i].diagnostic_reasoning)];vp[i]=[]
 for k,p in enumerate(ps):keys.append(("va",i,k));texts.append(p)
m=SentenceTransformer(MODEL,revision=MODEL_REV,device="cpu");m.max_seq_length=256
E=m.encode(texts,batch_size=64,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True)
for z,key in zip(E,keys):
 typ,i,k=key;(tp if typ=="tr" else vp)[i].append(z)
def f1(i,cands):
 q=np.asarray(vp[i]);c=np.vstack([np.asarray(tp[j]) for j in cands]);S=q@c.T
 rec=float(np.max(S,axis=1).mean());prec=float(np.max(S,axis=0).mean())
 return 2*rec*prec/(rec+prec) if rec+prec else 0.
summary={"metric":"independent S-PubMedBERT diagnosis-masked reasoning-set softF1","model":{"name":MODEL,"revision":MODEL_REV,"max_seq_length":256},"results":{},"paired":{}}
scores={}
for name,R in methods.items():
 scores[name]={}
 for k in (1,3,10):
  a=np.array([f1(i,R[i][:k]) for i in range(len(va))]);scores[name][k]=a;summary["results"].setdefault(name,{})[str(k)]={"mean":float(a.mean())}
rng=np.random.default_rng(SEED)
for k in (1,3,10):
 d=scores["D5_debiased"][k]-scores["tfidf"][k];boots=[]
 for _ in range(5000):
  ids=rng.integers(0,len(d),len(d));boots.append(float(d[ids].mean()))
 summary["paired"][str(k)]={"delta":float(d.mean()),"ci95":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))],"improved":int((d>0).sum()),"worsened":int((d<0).sum())}
summary["limits"]=["Independent dense evaluator is not human expert judgment.","Exact diagnosis phrases are masked, aliases may remain.","Validation only; test sealed."]
(OUT/"E020_D5D_INDEPENDENT_DEBIASED.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
