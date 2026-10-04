from __future__ import annotations
import json,re,unicodedata,pathlib
import numpy as np,pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e020-d5b-output");OUT.mkdir(exist_ok=True);RNG=np.random.default_rng(20261004)
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
def scores(q,c):
 S=(q@c.T).toarray();rec=float(np.max(S,axis=1).mean());prec=float(np.max(S,axis=0).mean());f=2*rec*prec/(rec+prec) if rec+prec else 0.
 return rec,prec,f
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["diagnostic_reasoning","final_diagnosis"]]
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["diagnostic_reasoning","final_diagnosis"]]
tp=[[mask_point(x,d) for x in split_reason(r)] for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
vp=[[mask_point(x,d) for x in split_reason(r)] for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
vec=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
T=vec.fit_transform([p for ps in tp for p in ps]);offs=[];o=0
for ps in tp:offs.append((o,o+len(ps)));o+=len(ps)
V=vec.transform([p for ps in vp for p in ps]);vo=[];o=0
for ps in vp:vo.append((o,o+len(ps)));o+=len(ps)
def eset(i,cands):
 s,e=vo[i];q=V[s:e];m=[]
 for j in cands:a,b=offs[j];m.append(T[a:b])
 return scores(q,vstack(m))
base=load("c2/e020-c2-output/E020_C2_RANKINGS.jsonl","baseline_top50")
old=load("c2/e020-c2-output/E020_C2_RANKINGS.jsonl","C2_top50")
new=load("d5/e020-d5-output/E020_D5_RANKINGS.jsonl","D5_top50")
methods={"tfidf":base,"C2_original":old,"C2_debiased":new};summary={"metric":"diagnosis-masked symmetric reasoning-set softF1","results":{},"reason_point_count":{}}
for m,R in methods.items():
 summary["results"][m]={};summary["reason_point_count"][m]={}
 for k in (1,3,10):
  arr=np.array([eset(i,R[i][:k])[2] for i in range(len(va))]);cnt=np.array([sum(len(tp[j]) for j in R[i][:k]) for i in range(len(va))])
  summary["results"][m][str(k)]={"mean":float(arr.mean())};summary["reason_point_count"][m][str(k)]={"mean":float(cnt.mean()),"median":float(np.median(cnt))}
  if m!="tfidf":
   b=np.array([eset(i,base[i][:k])[2] for i in range(len(va))]);d=arr-b;boots=[]
   for _ in range(5000):ids=RNG.integers(0,len(d),len(d));boots.append(float(d[ids].mean()))
   summary["results"][m][str(k)].update({"delta":float(d.mean()),"ci95":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))]})
   cd=cnt-np.array([sum(len(tp[j]) for j in base[i][:k]) for i in range(len(va))])
   summary["results"][m][str(k)]["corr_delta_utility_vs_delta_reason_count"]=float(np.corrcoef(d,cd)[0,1])
summary["limits"]=["SoftF1 reduces but does not eliminate all annotation-density effects.","Reasoning point count is reported explicitly to diagnose selection bias.","Validation reasoning is evaluation only; test sealed."]
(OUT/"E020_D5B_REAUDIT.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
