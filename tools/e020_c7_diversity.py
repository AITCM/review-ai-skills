from __future__ import annotations
import json,re,unicodedata,pathlib
import numpy as np,pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e020-c7-output");OUT.mkdir(exist_ok=True)
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
def load_rank():
 rows={}
 for line in open("c2/e020-c2-output/E020_C2_RANKINGS.jsonl"):
  x=json.loads(line);rows[int(x["query_index"])]=x["C2_top50"]
 return rows
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["diagnostic_reasoning","final_diagnosis"]]
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["diagnostic_reasoning","final_diagnosis"]]
R=load_rank()
tr_pts=[[mask_point(x,d) for x in split_reason(r)] for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
va_pts=[[mask_point(x,d) for x in split_reason(r)] for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
tr_docs=[" ".join(x) for x in tr_pts]
docvec=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
D=docvec.fit_transform(tr_docs)
# point metric as C6, fit on train only
flat=[p for ps in tr_pts for p in ps]
pv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
TP=pv.fit_transform(flat);offs=[];o=0
for ps in tr_pts:offs.append((o,o+len(ps)));o+=len(ps)
VP=pv.transform([p for ps in va_pts for p in ps]);voffs=[];o=0
for ps in va_pts:voffs.append((o,o+len(ps)));o+=len(ps)
def coverage(i,cands):
 s,e=voffs[i];q=VP[s:e]; mats=[]
 for j in cands:
  a,b=offs[j];mats.append(TP[a:b])
 c=vstack(mats);S=(q@c.T).toarray()
 return float(np.max(S,axis=1).mean())
def diversify(cands,lam,k=10):
 cands=list(cands);sel=[];rel=np.array([1/np.log2(r+2) for r in range(len(cands))],float)
 while len(sel)<min(k,len(cands)):
  best=None;bs=-1e9
  for pos,j in enumerate(cands):
   if j in sel:continue
   red=0.0
   if sel:
    red=max(float((D[j]@D[s].T).toarray()[0,0]) for s in sel)
   score=rel[pos]-lam*red
   if score>bs or (score==bs and (best is None or j<best)):bs=score;best=j
  sel.append(best)
 return sel
lams=[0.0,0.05,0.1,0.2,0.3,0.5]
results={}
for lam in lams:
 vals={1:[],3:[],10:[]};seqs={}
 for i in range(len(va)):
  seq=diversify(R[i],lam,10);seqs[i]=seq
  for k in (1,3,10):vals[k].append(coverage(i,seq[:k]))
 results[str(lam)]={f"coverage@{k}":float(np.mean(vals[k])) for k in (1,3,10)}
 results[str(lam)].update({f"delta_vs_lam0@{k}":None for k in (1,3,10)})
 if lam==0:base=vals
 else:
  for k in (1,3,10):
   d=np.asarray(vals[k])-np.asarray(base[k]);results[str(lam)][f"delta_vs_lam0@{k}"]=float(d.mean())
best3=max(lams,key=lambda l:results[str(l)]["coverage@3"])
best10=max(lams,key=lambda l:results[str(l)]["coverage@10"])
summary={"experiment":"C7 exploratory diversity-aware evidence set selection over locked C2 top50",
 "lambda_grid":lams,"results":results,"best_lambda_for_coverage3":best3,"best_lambda_for_coverage10":best10,
 "status":"exploratory validation-only; any selected lambda must be re-estimated on train-only meta-split before test",
 "limits":["Candidate redundancy uses historical diagnosis-masked reasoning; query reasoning is never used for selection.",
 "Validation reasoning is used for evaluation and lambda exploration, so these are development results, not confirmatory test results."]}
(OUT/"E020_C7_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
