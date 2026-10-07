from __future__ import annotations
import json,re,unicodedata,pathlib
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e080-n8d-output");OUT.mkdir(exist_ok=True);SEED=20261007
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
def mask_phrase(p,d):
 x=nw(p);dd=nw(d);return x.replace(dd," diagnosismask ") if dd and dd in x else x
def mask_tokens(p,d):
 x=mask_phrase(p,d); toks=[t for t in nw(d).split() if len(t)>=4 and t not in {"with","without","acute","chronic","syndrome","disease","disorder"}]
 for t in sorted(set(toks),key=len,reverse=True):
  x=re.sub(r"\b"+re.escape(t)+r"\b"," diagnosistokenmask ",x)
 return " ".join(x.split())
def rank(s):
 ids=np.arange(len(s));return np.lexsort((ids,-np.asarray(s)))
def rrf(a,b):
 n=len(a);sc=np.zeros(n,float)
 for o in (a,b):
  pos=np.empty(n,int);pos[o]=np.arange(n);sc+=1/(61+pos)
 return rank(sc)
def f1(rec,prec):
 den=rec+prec;return np.where(den>0,2*rec*prec/den,0.)
def set_u(qmax,psum,cnt,ids):
 ids=np.asarray(ids,int);rec=float(np.max(qmax[:,ids],1).mean());prec=float(psum[ids].sum()/cnt[ids].sum());return 2*rec*prec/(rec+prec) if rec+prec else 0.
def boot(d,seed,n=3000):
 d=np.asarray(d,float);rng=np.random.default_rng(seed);z=np.empty(n)
 for t in range(n):
  ii=rng.integers(0,len(d),len(d));z[t]=d[ii].mean()
 return [float(np.percentile(z,2.5)),float(np.percentile(z,97.5))]
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["case_prompt","diagnostic_reasoning","final_diagnosis"]]
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["case_prompt","diagnostic_reasoning","final_diagnosis"]]
tp_phrase=[[mask_phrase(p,d) for p in split_reason(r)] for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
tp_strong=[[mask_tokens(p,d) for p in split_reason(r)] for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
vp=[[mask_phrase(p,d) for p in split_reason(r)] for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
doc_phrase=[" ".join(x) for x in tp_phrase];doc_strong=[" ".join(x) for x in tp_strong]
# Query diagnosis exposure audit, no outcome-based masking applied to retrieval.
exposure=np.asarray([bool(nw(d)) and nw(d) in nw(q) for q,d in zip(va.case_prompt,va.final_diagnosis)],bool)
train_exposure=np.asarray([bool(nw(d)) and nw(d) in nw(q) for q,d in zip(tr.case_prompt,tr.final_diagnosis)],bool)

cv=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
X=cv.fit_transform(tr.case_prompt.astype(str));Q=cv.transform(va.case_prompt.astype(str));LEX=(Q@X.T).toarray()
def xmod(docs):
 v=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2,max_df=.98,max_features=140000,dtype=np.float32)
 v.fit(tr.case_prompt.astype(str).tolist()+docs);qq=v.transform(va.case_prompt.astype(str));rr=v.transform(docs);return (qq@rr.T).toarray()
XP=xmod(doc_phrase);XS=xmod(doc_strong)
# Formal primary evaluator uses phrase-masked reasoning, unchanged.
flat=[p for ps in tp_phrase for p in ps];ev=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
T=ev.fit_transform(flat);V=ev.transform([p for ps in vp for p in ps]);Tt=T.T.tocsr();cnt=np.asarray([len(x) for x in tp_phrase],int);starts=np.concatenate(([0],np.cumsum(cnt)[:-1])).astype(int)
vo=[];o=0
for ps in vp:vo.append((o,o+len(ps)));o+=len(ps)
methods=["LEX","RRF_PHRASE","RRF_STRONG"];scores={m:{k:[] for k in (1,3,10)} for m in methods}
for i in range(len(va)):
 l=rank(LEX[i]);rp=rrf(l,rank(XP[i]));rs=rrf(l,rank(XS[i]));orders={"LEX":l,"RRF_PHRASE":rp,"RRF_STRONG":rs}
 s,e=vo[i];S=(V[s:e]@Tt).toarray();qmax=np.stack([np.maximum.reduceat(row,starts) for row in S],0);pmax=S.max(0);psum=np.add.reduceat(pmax,starts)
 for m,o2 in orders.items():
  for k in (1,3,10):scores[m][k].append(set_u(qmax,psum,cnt,o2[:k]))
summary={"experiment":"E080-N8d diagnosis-exposure and stronger historical-label masking sensitivity",
 "n_train":len(tr),"n_validation":len(va),
 "query_exact_diagnosis_exposure":{"train_rate":float(train_exposure.mean()),"validation_rate":float(exposure.mean()),"validation_n":int(exposure.sum()),"clean_validation_n":int((~exposure).sum())},
 "historical_strong_mask":"exact normalized diagnosis phrase plus diagnosis tokens length>=4, excluding generic clinical words",
 "results":{},"policy":"No validation query is masked using its outcome. Clean-subset analysis only excludes queries whose case_prompt already literally contains the final diagnosis. Test untouched."}
for subset_name,idx in [("all",np.ones(len(va),bool)),("query_diagnosis_not_exposed",~exposure)]:
 summary["results"][subset_name]={}
 for k in (1,3,10):
  base=np.asarray(scores["LEX"][k])[idx];summary["results"][subset_name][str(k)]={}
  for m in methods:
   arr=np.asarray(scores[m][k])[idx];d=arr-base
   summary["results"][subset_name][str(k)][m]={"mean":float(arr.mean()),"delta_vs_LEX":float(d.mean()),"ci95":boot(d,SEED+k+len(m)+len(subset_name))}
(OUT/"E080_N8D_MASK_SENSITIVITY.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
