from __future__ import annotations
import argparse,json,re,unicodedata,pathlib
import numpy as np,pandas as pd
from sentence_transformers import SentenceTransformer
REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
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
ap=argparse.ArgumentParser();ap.add_argument("--shard",type=int,required=True);ap.add_argument("--nshards",type=int,default=5);ap.add_argument("--out",required=True);a=ap.parse_args()
OUT=pathlib.Path(a.out);OUT.mkdir(exist_ok=True)
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["diagnostic_reasoning","final_diagnosis"]]
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["diagnostic_reasoning","final_diagnosis"]]
base=load("locked/final-output/rankings.jsonl","baseline_top50");method=load("locked/final-output/rankings.jsonl","method_top50")
ids=[i for i in range(len(va)) if i%a.nshards==a.shard]
needed=set()
for R in (base,method):
 for i in ids:needed.update(R[i][:10])
texts=[];keys=[];TP={};VP={}
for j in sorted(needed):
 TP[j]=[]
 for k,x in enumerate(split_reason(tr.iloc[j].diagnostic_reasoning)):keys.append(("t",j,k));texts.append(mask(x,tr.iloc[j].final_diagnosis))
for i in ids:
 VP[i]=[]
 for k,x in enumerate(split_reason(va.iloc[i].diagnostic_reasoning)):keys.append(("v",i,k));texts.append(mask(x,va.iloc[i].final_diagnosis))
m=SentenceTransformer(MODEL,revision=MODEL_REV,device="cpu");m.max_seq_length=256
E=m.encode(texts,batch_size=64,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True)
for z,(typ,i,k) in zip(E,keys):(TP if typ=="t" else VP)[i].append(z)
rows=[]
for i in ids:
 q=np.asarray(VP[i]);row={"query_index":i}
 for name,R in (("baseline",base),("method",method)):
  for k in (1,3,10):
   c=np.vstack([np.asarray(TP[j]) for j in R[i][:k]])
   row[f"{name}_{k}"]=sf(q,c)
 rows.append(row)
with (OUT/"per_query.jsonl").open("w") as f:
 for x in rows:f.write(json.dumps(x)+"\n")
(OUT/"receipt.json").write_text(json.dumps({"shard":a.shard,"nshards":a.nshards,"n_queries":len(ids),"n_train_cases":len(needed),"model":MODEL,"revision":MODEL_REV},indent=2)+"\n")
print(json.dumps({"shard":a.shard,"n_queries":len(ids),"n_train_cases":len(needed)}))
