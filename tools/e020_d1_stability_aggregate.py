from __future__ import annotations
import json,re,unicodedata,pathlib,glob
from collections import defaultdict
import numpy as np,pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("stability-aggregate");OUT.mkdir(exist_ok=True);RNG=np.random.default_rng(20261004)
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
 pt=nw(p);dd=nw(dx);return pt.replace(dd," diagnosismask ") if dd and dd in pt else pt
def load(path,key):
 rows={}
 for line in open(path):x=json.loads(line);rows[int(x["query_index"])]=x[key]
 return rows
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["case_prompt","diagnostic_reasoning","final_diagnosis"]]
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["case_prompt","diagnostic_reasoning","final_diagnosis"]]
train_pts=[[mask_point(x,d) for x in split_reason(r)] for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
val_pts=[[mask_point(x,d) for x in split_reason(r)] for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
flat=[p for ps in train_pts for p in ps]
vec=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
T=vec.fit_transform(flat);offs=[];o=0
for ps in train_pts:offs.append((o,o+len(ps)));o+=len(ps)
V=vec.transform([p for ps in val_pts for p in ps]);voffs=[];o=0
for ps in val_pts:voffs.append((o,o+len(ps)));o+=len(ps)
def cov(i,cands):
 s,e=voffs[i];q=V[s:e];m=[]
 for j in cands:
  a,b=offs[j];m.append(T[a:b])
 c=vstack(m);return float(np.max((q@c.T).toarray(),axis=1).mean())
tl=[nw(x) for x in tr.final_diagnosis];vl=[nw(x) for x in va.final_diagnosis]
freq=defaultdict(int)
for y in tl:freq[y]+=1
seeddirs=sorted(glob.glob("downloads/stability-*"))
seed_rows=[];per_seed_delta={};per_query=[]
base_cache=None
for sd in seeddirs:
 summ=json.load(open(sd+"/summary.json"));seed=summ["seed"]
 base=load(sd+"/rankings.jsonl","baseline_top50");c2=load(sd+"/rankings.jsonl","C2_top50")
 if base_cache is None:base_cache=base
 vals={k:[] for k in (1,3,10)};exact={}
 for k in (1,3,10):
  b=np.array([cov(i,base[i][:k]) for i in range(len(va))])
  c=np.array([cov(i,c2[i][:k]) for i in range(len(va))])
  d=c-b;vals[k]=d
  exact[k]={"baseline":int(sum(any(tl[j]==vl[i] for j in base[i][:k]) for i in range(len(va)))),
            "C2":int(sum(any(tl[j]==vl[i] for j in c2[i][:k]) for i in range(len(va))))}
 seed_rows.append({"seed":seed,**{f"delta@{k}":float(np.mean(vals[k])) for k in (1,3,10)},**{f"exact@{k}_baseline":exact[k]["baseline"] for k in exact},**{f"exact@{k}_C2":exact[k]["C2"] for k in exact}})
 per_seed_delta[seed]=vals
# mean-over-seeds per query, bootstrap queries
summary={"n_seeds":len(seed_rows),"seed_results":seed_rows,"aggregate":{},"subgroups":{}}
for k in (1,3,10):
 mat=np.vstack([per_seed_delta[x["seed"]][k] for x in seed_rows])
 qmean=mat.mean(0);boots=[]
 for _ in range(5000):
  ids=RNG.integers(0,len(qmean),len(qmean));boots.append(float(qmean[ids].mean()))
 means=mat.mean(1)
 summary["aggregate"][str(k)]={"mean_delta_across_seed_case":float(mat.mean()),"seed_mean_sd":float(means.std(ddof=1)),
   "seed_mean_min":float(means.min()),"seed_mean_max":float(means.max()),"positive_seeds":int((means>0).sum()),
   "query_bootstrap_ci95":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))]}
 # label availability subgroup
 for label,ids in {"label_present":[i for i,y in enumerate(vl) if freq[y]>0],"label_absent":[i for i,y in enumerate(vl) if freq[y]==0]}.items():
  arr=mat[:,ids];summary["subgroups"].setdefault(label,{})[str(k)]={"n":len(ids),"mean_delta":float(arr.mean())}
# case length quartiles
length=np.array([len(str(x).split()) for x in va.case_prompt])
qs=np.quantile(length,[.25,.5,.75])
groups={"len_Q1":np.where(length<=qs[0])[0],"len_Q2":np.where((length>qs[0])&(length<=qs[1]))[0],"len_Q3":np.where((length>qs[1])&(length<=qs[2]))[0],"len_Q4":np.where(length>qs[2])[0]}
for g,ids in groups.items():
 summary["subgroups"][g]={str(k):{"n":len(ids),"mean_delta":float(np.vstack([per_seed_delta[x["seed"]][k] for x in seed_rows])[:,ids].mean())} for k in (1,3,10)}
(OUT/"E020_D1_STABILITY_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
pd.DataFrame(seed_rows).to_csv(OUT/"E020_D1_SEED_RESULTS.csv",index=False)
print(json.dumps(summary,indent=2))
