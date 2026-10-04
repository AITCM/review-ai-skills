from __future__ import annotations
import json,re,unicodedata,pathlib,glob
import numpy as np,pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data";OUT=pathlib.Path("d8-agg");OUT.mkdir(exist_ok=True);RNG=np.random.default_rng(20261004)
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
def mask(p,d):
 x=nw(p);dd=nw(d);return x.replace(dd," diagnosismask ") if dd and dd in x else x
def load(path,key):
 R={}
 for l in open(path):x=json.loads(l);R[int(x["query_index"])]=x[key]
 return R
def sf(q,c):
 S=(q@c.T).toarray();r=float(np.max(S,axis=1).mean());p=float(np.max(S,axis=0).mean());return 2*r*p/(r+p) if r+p else 0.
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["diagnostic_reasoning","final_diagnosis"]];va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["diagnostic_reasoning","final_diagnosis"]]
tp=[[mask(x,d) for x in split_reason(r)] for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)];vp=[[mask(x,d) for x in split_reason(r)] for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
vec=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32);T=vec.fit_transform([p for z in tp for p in z]);offs=[];o=0
for z in tp:offs.append((o,o+len(z)));o+=len(z)
V=vec.transform([p for z in vp for p in z]);vo=[];o=0
for z in vp:vo.append((o,o+len(z)));o+=len(z)
def cov(i,cands):
 s,e=vo[i];q=V[s:e];m=[]
 for j in cands:a,b=offs[j];m.append(T[a:b])
 return sf(q,vstack(m))
summary={"evaluation":"common exact-mask symmetric reasoning-set softF1","variants":{}}
for sd in glob.glob("downloads/d8-*"):
 mode=json.load(open(sd+"/summary.json"))["mode"];b=load(sd+"/rankings.jsonl","baseline_top50");m=load(sd+"/rankings.jsonl","method_top50");summary["variants"][mode]={}
 for k in (1,3,10):
  x=np.array([cov(i,b[i][:k]) for i in range(len(va))]);y=np.array([cov(i,m[i][:k]) for i in range(len(va))]);d=y-x;boots=[]
  for _ in range(3000):ids=RNG.integers(0,len(d),len(d));boots.append(float(d[ids].mean()))
  summary["variants"][mode][str(k)]={"delta":float(d.mean()),"ci95":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))]}
(OUT/"E020_D8_DEBIASED_MASK_CONTROLS.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
