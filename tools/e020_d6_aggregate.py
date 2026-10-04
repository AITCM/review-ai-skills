from __future__ import annotations
import json,re,unicodedata,pathlib,glob
import numpy as np,pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data";OUT=pathlib.Path("d6-aggregate");OUT.mkdir(exist_ok=True);RNG=np.random.default_rng(20261004)
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
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["case_prompt","diagnostic_reasoning","final_diagnosis"]];va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["case_prompt","diagnostic_reasoning","final_diagnosis"]]
tp=[[mask(x,d) for x in split_reason(r)] for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)];vp=[[mask(x,d) for x in split_reason(r)] for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
vec=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32);T=vec.fit_transform([p for ps in tp for p in ps]);offs=[];o=0
for ps in tp:offs.append((o,o+len(ps)));o+=len(ps)
V=vec.transform([p for ps in vp for p in ps]);vo=[];o=0
for ps in vp:vo.append((o,o+len(ps)));o+=len(ps)
def cov(i,cands):
 s,e=vo[i];q=V[s:e];m=[]
 for j in cands:a,b=offs[j];m.append(T[a:b])
 return sf(q,vstack(m))
tl=[nw(x) for x in tr.final_diagnosis];vl=[nw(x) for x in va.final_diagnosis];freq={}
for y in tl:freq[y]=freq.get(y,0)+1
dirs=sorted(glob.glob("downloads/d6-*"));seedres=[];matbyk={k:[] for k in (1,3,10)}
for sd in dirs:
 seed=json.load(open(sd+"/summary.json"))["seed"];b=load(sd+"/rankings.jsonl","baseline_top50");m=load(sd+"/rankings.jsonl","method_top50");row={"seed":seed}
 for k in (1,3,10):
  x=np.array([cov(i,b[i][:k]) for i in range(len(va))]);y=np.array([cov(i,m[i][:k]) for i in range(len(va))]);d=y-x;matbyk[k].append(d);row[f"delta@{k}"]=float(d.mean())
 row["exact1"]=int(sum(any(tl[j]==vl[i] for j in m[i][:1]) for i in range(len(va))));row["exact10"]=int(sum(any(tl[j]==vl[i] for j in m[i][:10]) for i in range(len(va))));seedres.append(row)
summary={"n_seeds":len(seedres),"seed_results":seedres,"aggregate":{},"label_subgroups":{}}
for k in (1,3,10):
 M=np.vstack(matbyk[k]);means=M.mean(1);qm=M.mean(0);boots=[]
 for _ in range(5000):ids=RNG.integers(0,len(qm),len(qm));boots.append(float(qm[ids].mean()))
 summary["aggregate"][str(k)]={"mean_delta":float(M.mean()),"seed_sd":float(means.std(ddof=1)),"min_seed":float(means.min()),"max_seed":float(means.max()),"positive_seeds":int((means>0).sum()),"query_bootstrap_ci95":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))]}
 for label,ids in {"present":[i for i,y in enumerate(vl) if freq.get(y,0)>0],"absent":[i for i,y in enumerate(vl) if freq.get(y,0)==0]}.items():
  summary["label_subgroups"].setdefault(label,{})[str(k)]={"n":len(ids),"mean_delta":float(M[:,ids].mean())}
(OUT/"E020_D6_DEBIASED_STABILITY.json").write_text(json.dumps(summary,indent=2)+"\n");pd.DataFrame(seedres).to_csv(OUT/"seed_results.csv",index=False);print(json.dumps(summary,indent=2))
