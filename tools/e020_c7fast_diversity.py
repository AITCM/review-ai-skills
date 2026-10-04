from __future__ import annotations
import json,re,unicodedata,pathlib
import numpy as np,pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e020-c7fast-output");OUT.mkdir(exist_ok=True)
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
R={}
for line in open("c2/e020-c2-output/E020_C2_RANKINGS.jsonl"):
 x=json.loads(line);R[int(x["query_index"])]=x["C2_top50"]
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["diagnostic_reasoning","final_diagnosis"]]
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["diagnostic_reasoning","final_diagnosis"]]
tr_pts=[[mask_point(x,d) for x in split_reason(r)] for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
va_pts=[[mask_point(x,d) for x in split_reason(r)] for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
tr_docs=[" ".join(x) for x in tr_pts]
docvec=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
D=docvec.fit_transform(tr_docs)
flat=[p for ps in tr_pts for p in ps]
pv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
TP=pv.fit_transform(flat);offs=[];o=0
for ps in tr_pts:offs.append((o,o+len(ps)));o+=len(ps)
VP=pv.transform([p for ps in va_pts for p in ps]);voffs=[];o=0
for ps in va_pts:voffs.append((o,o+len(ps)));o+=len(ps)
lams=[0.0,0.05,0.1,0.2,0.3,0.5];scores={str(l):{k:[] for k in (1,3,10)} for l in lams}
def cov_from_matrix(qi,cands):
 s,e=voffs[qi];q=VP[s:e];mats=[]
 for j in cands:
  a,b=offs[j];mats.append(TP[a:b])
 c=vstack(mats);return float(np.max((q@c.T).toarray(),axis=1).mean())
for i in range(len(va)):
 cands=np.array(R[i],dtype=int);sim=(D[cands]@D[cands].T).toarray()
 rel=np.array([1/np.log2(r+2) for r in range(len(cands))])
 for lam in lams:
  sel=[];remaining=set(range(len(cands)))
  while len(sel)<10:
   best=None;bs=-1e99
   for p in remaining:
    red=max((sim[p,s] for s in sel),default=0.0);score=rel[p]-lam*red
    if score>bs or (score==bs and (best is None or cands[p]<cands[best])):best=p;bs=score
   sel.append(best);remaining.remove(best)
  ids=cands[sel]
  for k in (1,3,10):scores[str(lam)][k].append(cov_from_matrix(i,ids[:k]))
results={}
base=scores["0.0"]
for lam in lams:
 results[str(lam)]={f"coverage@{k}":float(np.mean(scores[str(lam)][k])) for k in (1,3,10)}
 for k in (1,3,10):
  d=np.asarray(scores[str(lam)][k])-np.asarray(base[k])
  results[str(lam)][f"delta@{k}"]=float(d.mean())
summary={"experiment":"C7fast exploratory diversity-aware evidence use over locked C2 top50","lambda_grid":lams,"results":results,
 "best_lambda_3":max(lams,key=lambda x:results[str(x)]["coverage@3"]),
 "best_lambda_10":max(lams,key=lambda x:results[str(x)]["coverage@10"]),
 "status":"validation exploration only; lambda must be chosen using train-only meta-split before test",
 "limits":["Query reasoning is never used for selection; historical masked reasoning is used only for redundancy.","Validation reasoning is evaluation only."]}
(OUT/"E020_C7FAST_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
