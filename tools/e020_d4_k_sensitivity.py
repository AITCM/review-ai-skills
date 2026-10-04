from __future__ import annotations
import json,re,unicodedata,pathlib,hashlib
import numpy as np,pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.linear_model import LogisticRegression

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e020-d4-output");OUT.mkdir(exist_ok=True);SEED=20261004
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
def mask_points(r,d):
 dd=nw(d);out=[]
 for x in split_reason(r):
  xx=nw(x)
  if dd and dd in xx:xx=xx.replace(dd," diagnosismask ")
  out.append(xx)
 return out
def rank(s,k):
 ids=np.arange(len(s));return np.lexsort((ids,-s))[:k]
def feat(sem,prd,pos,cnt):
 sem=np.asarray(sem,float);prd=np.asarray(prd,float);rr=1/(1+np.asarray(pos,float));rc=np.log1p(np.asarray(cnt,float))
 return np.column_stack([sem,prd,sem*prd,rr,rc])
def cov(q,c):
 S=(q@c.T).toarray();return float(np.max(S,axis=1).mean()) if S.size else 0.

tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]]
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["case_prompt","diagnostic_reasoning","final_diagnosis"]]
isdev=np.array([int(hashlib.sha256(("E020-C2:"+str(x)).encode()).hexdigest()[:8],16)%5==0 for x in tr.pmcid])
ref=tr.loc[~isdev].reset_index(drop=True);mq=tr.loc[isdev].reset_index(drop=True)

cv=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=cv.fit_transform(ref.case_prompt.astype(str));Q=cv.transform(mq.case_prompt.astype(str))
cs=TruncatedSVD(256,n_iter=7,random_state=SEED);XC=cs.fit_transform(X).astype(float);QC=cs.transform(Q).astype(float)
rpts=[mask_points(r,d) for r,d in zip(ref.diagnostic_reasoning,ref.final_diagnosis)]
mpts=[mask_points(r,d) for r,d in zip(mq.diagnostic_reasoning,mq.final_diagnosis)]
rv=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R=rv.fit_transform([" ".join(x) for x in rpts]);rs=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR=rs.fit_transform(R).astype(float)
xm=XC.mean(0,keepdims=True);ym=ZR.mean(0,keepdims=True);A=XC-xm;Y=ZR-ym;W=np.linalg.solve(A.T@A+10*np.eye(A.shape[1]),A.T@Y)
P=normalize((QC-xm)@W+ym);ZI=normalize(ZR);CS=(Q@X.T).tocsr();PS=P@ZI.T
allp=[p for ps in rpts for p in ps]+[p for ps in mpts for p in ps]
pv=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
PM=pv.fit_transform(allp);off=0;rpm=[]
for ps in rpts:rpm.append(PM[off:off+len(ps)]);off+=len(ps)
mpm=[]
for ps in mpts:mpm.append(PM[off:off+len(ps)]);off+=len(ps)

def train_ranker(train_k):
 D=[];YY=[]
 for i in range(len(mq)):
  s=CS.getrow(i).toarray().ravel();cand=rank(s,train_k);u=np.array([cov(mpm[i],rpm[int(j)]) for j in cand])
  F=feat(s[cand],PS[i,cand],np.arange(len(cand)),[len(rpts[int(j)]) for j in cand])
  hi=np.argsort(-u)[:min(5,len(cand))];lo=np.argsort(u)[:min(10,len(cand))]
  for p in hi:
   for n in lo:
    if u[p]-u[n]<.035:continue
    d=F[p]-F[n];D.append(d);YY.append(1);D.append(-d);YY.append(0)
 D=np.asarray(D);YY=np.asarray(YY);sc=StandardScaler();Z=sc.fit_transform(D);clf=LogisticRegression(C=1,max_iter=2000,random_state=SEED).fit(Z,YY)
 return clf.coef_[0]/sc.scale_,len(YY)

# all-train deployment representations
cv2=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X2=cv2.fit_transform(tr.case_prompt.astype(str));Q2=cv2.transform(va.case_prompt.astype(str))
cs2=TruncatedSVD(256,n_iter=7,random_state=SEED);XC2=cs2.fit_transform(X2).astype(float);QC2=cs2.transform(Q2).astype(float)
tpts=[mask_points(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
rv2=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R2=rv2.fit_transform([" ".join(x) for x in tpts]);rs2=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR2=rs2.fit_transform(R2).astype(float)
xm=XC2.mean(0,keepdims=True);ym=ZR2.mean(0,keepdims=True);AA=XC2-xm;YYR=ZR2-ym;WW=np.linalg.solve(AA.T@AA+10*np.eye(AA.shape[1]),AA.T@YYR)
PP=normalize((QC2-xm)@WW+ym);ZZ=normalize(ZR2);CSS=(Q2@X2.T).tocsr();PSS=PP@ZZ.T;cnt=[len(x) for x in tpts]

# held-out utility evaluator fit on training only
flat=[p for ps in tpts for p in ps];ev=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
TP=ev.fit_transform(flat);offs=[];o=0
for ps in tpts:offs.append((o,o+len(ps)));o+=len(ps)
vpts=[mask_points(r,d) for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
VP=ev.transform([p for ps in vpts for p in ps]);vo=[];o=0
for ps in vpts:vo.append((o,o+len(ps)));o+=len(ps)
def vc(i,cands):
 s,e=vo[i];q=VP[s:e];m=[]
 for j in cands:
  a,b=offs[j];m.append(TP[a:b])
 return cov(q,vstack(m))

configs=[]
for train_k in [20,40,80]:
 w,nrows=train_ranker(train_k)
 for deploy_k in [20,50,100]:
  baseline=[];rer=[]
  for i in range(len(va)):
   s=CSS.getrow(i).toarray().ravel();cand=rank(s,deploy_k);baseline.append(cand.tolist())
   F=feat(s[cand],PSS[i,cand],np.arange(len(cand)),[cnt[int(j)] for j in cand]);score=F@w;ids=np.asarray(cand);o=np.lexsort((ids,-score));rer.append(ids[o].tolist())
  rec={"train_k":train_k,"deploy_k":deploy_k,"pair_rows":nrows}
  for k in (1,3,10):
   b=np.array([vc(i,baseline[i][:k]) for i in range(len(va))]);c=np.array([vc(i,rer[i][:k]) for i in range(len(va))])
   d=c-b;rec[f"delta@{k}"]=float(d.mean())
  configs.append(rec)
summary={"seed":SEED,"utility_margin":.035,"ridge_alpha":10.0,"configs":configs,"test_accessed":False}
(OUT/"E020_D4_K_SENSITIVITY.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
