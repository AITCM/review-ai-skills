from __future__ import annotations
import json,re,unicodedata,pathlib
import numpy as np,pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e080-n8c-output");OUT.mkdir(exist_ok=True);SEED=20261007
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
def rank(s):
 ids=np.arange(len(s));return np.lexsort((ids,-np.asarray(s)))
def rrf_rank(a,b):
 n=len(a);sc=np.zeros(n,float)
 for o in (a,b):
  p=np.empty(n,int);p[o]=np.arange(n);sc+=1/(61+p)
 return rank(sc)
def sf1(q,c):
 S=(q@c.T).toarray();rec=float(np.max(S,1).mean());prec=float(np.max(S,0).mean());return 2*rec*prec/(rec+prec) if rec+prec else 0.
def boot(d,seed,n=3000):
 d=np.asarray(d,float);rng=np.random.default_rng(seed);z=np.empty(n)
 for t in range(n):
  ii=rng.integers(0,len(d),len(d));z[t]=d[ii].mean()
 return [float(np.percentile(z,2.5)),float(np.percentile(z,97.5))]
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["case_prompt","diagnostic_reasoning","final_diagnosis"]]
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["case_prompt","diagnostic_reasoning","final_diagnosis"]]
tp=[[mask_point(p,d) for p in split_reason(r)] for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
vp=[[mask_point(p,d) for p in split_reason(r)] for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
tdoc=[" ".join(x) for x in tp]
cv=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
X=cv.fit_transform(tr.case_prompt.astype(str));Q=cv.transform(va.case_prompt.astype(str));LEX=(Q@X.T).toarray()
xv=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2,max_df=.98,max_features=140000,dtype=np.float32)
xv.fit(tr.case_prompt.astype(str).tolist()+tdoc);QX=xv.transform(va.case_prompt.astype(str));RX=xv.transform(tdoc);XM=(QX@RX.T).toarray()
R=[]
for i in range(len(va)):
 l=rank(LEX[i]);x=rank(XM[i]);f=rrf_rank(l,x);R.append({"LEX":l[:10],"XMOD":x[:10],"RRF":f[:10]})
needed=sorted(set(int(j) for r in R for k in r for j in r[k]))
# Independent character n-gram evaluator, train fit only.
flat=[p for ps in tp for p in ps]
ev=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),sublinear_tf=True,min_df=3,max_features=220000,dtype=np.float32)
T=ev.fit_transform(flat);V=ev.transform([p for ps in vp for p in ps]);offs=[];o=0
for ps in tp:offs.append((o,o+len(ps)));o+=len(ps)
vo=[];o=0
for ps in vp:vo.append((o,o+len(ps)));o+=len(ps)
def score(i,ids):
 s,e=vo[i];q=V[s:e];m=[]
 for j in ids:
  a,b=offs[int(j)];m.append(T[a:b])
 return sf1(q,vstack(m))
methods=["LEX","XMOD","RRF"];summary={"experiment":"E080-N8c independent char-ngram reasoning evaluator","metric":"symmetric reasoning-set char_wb TF-IDF softF1, 3-5 grams","n_validation":500,"results":{},"policy":"Validation evaluation only; test untouched."}
for k in (1,3,10):
 vals={m:np.asarray([score(i,R[i][m][:k]) for i in range(500)]) for m in methods};base=vals["LEX"];summary["results"][str(k)]={}
 for m in methods:
  d=vals[m]-base;summary["results"][str(k)][m]={"mean":float(vals[m].mean()),"delta_vs_LEX":float(d.mean()),"ci95":boot(d,SEED+k+len(m))}
(OUT/"E080_N8C_CHAR_EVAL.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
