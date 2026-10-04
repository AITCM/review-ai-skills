from __future__ import annotations
import json,re,unicodedata,pathlib
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e020-c6-output");OUT.mkdir(exist_ok=True);SEED=20261004
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
 for line in open(path):
  x=json.loads(line);rows[int(x["query_index"])]=x[key]
 return rows
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["diagnostic_reasoning","final_diagnosis"]]
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["diagnostic_reasoning","final_diagnosis"]]
base=load_rank("c2/e020-c2-output/E020_C2_RANKINGS.jsonl","baseline_top50")
c2=load_rank("c2/e020-c2-output/E020_C2_RANKINGS.jsonl","C2_top50")
c4=load_rank("c4/e020-c4-output/E020_C4_RANKINGS.jsonl","C4_ranked_union")
methods={"tfidf":base,"C2":c2,"C4":c4}
train_pts=[[mask_point(x,d) for x in split_reason(r)] for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
val_pts=[[mask_point(x,d) for x in split_reason(r)] for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
flat_train=[p for ps in train_pts for p in ps]
vec=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
T=vec.fit_transform(flat_train)
offs=[];o=0
for ps in train_pts:offs.append((o,o+len(ps)));o+=len(ps)
V=vec.transform([p for ps in val_pts for p in ps])
voffs=[];o=0
for ps in val_pts:voffs.append((o,o+len(ps)));o+=len(ps)
def cov(qi,cands):
 qs,qe=voffs[qi]; q=V[qs:qe]
 mats=[]
 for j in cands:
  s,e=offs[j];mats.append(T[s:e])
 from scipy.sparse import vstack
 c=vstack(mats)
 S=(q@c.T).toarray()
 return float(np.max(S,axis=1).mean())
scores={m:{k:[] for k in (1,3,10)} for m in methods}
for m,R in methods.items():
 for i in range(len(va)):
  for k in (1,3,10):scores[m][k].append(cov(i,R[i][:k]))
summary={"metric":"held-out validation diagnosis-masked reasoning-point TF-IDF coverage; vectorizer fit on training reasoning points only",
 "n_validation":len(va),"results":{},"paired_bootstrap_vs_tfidf":{}}
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
summary["limits"]=["Metric family matches C2/C4 train-only utility supervision but validation reasoning is held out from training/ranking.",
 "Diagnosis phrase masking is exact normalized phrase only.","Reasoning annotations are dataset-derived and not exhaustive clinical truth.","Test sealed."]
(OUT/"E020_C6_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
