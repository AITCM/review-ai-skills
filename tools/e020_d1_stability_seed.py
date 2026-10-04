from __future__ import annotations
import argparse,json,re,unicodedata,pathlib,hashlib
from collections import defaultdict
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.linear_model import LogisticRegression

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
PAT=re.compile(r'^\s*(?:\((\d+)\)|(\d+)[.)])\s*',re.M)
def nw(s): return " ".join(re.findall(r"\w+",unicodedata.normalize("NFKC",str(s)).casefold()))
def split_reason(s):
 s=str(s);ms=list(PAT.finditer(s))
 if not ms or s[:ms[0].start()].strip(): return [s.strip()]
 nums=[int(m.group(1) or m.group(2)) for m in ms]
 if nums!=list(range(1,len(ms)+1)): return [s.strip()]
 out=[]
 for i,m in enumerate(ms):
  e=ms[i+1].start() if i+1<len(ms) else len(s);x=s[m.end():e].strip()
  if x: out.append(x)
 return out or [s.strip()]
def mask_points(reason,diag):
 dd=nw(diag);out=[]
 for x in split_reason(reason):
  xx=nw(x)
  if dd and dd in xx: xx=xx.replace(dd," diagnosismask ")
  out.append(xx)
 return out
def rank(s,k=50):
 ids=np.arange(len(s));return np.lexsort((ids,-s))[:k]
def make_features(sem,prd,rankpos,cnt):
 sem=np.asarray(sem,float);prd=np.asarray(prd,float);rr=1/(1+np.asarray(rankpos,float));rc=np.log1p(np.asarray(cnt,float))
 return np.column_stack([sem,prd,sem*prd,rr,rc])
def coverage(qmat,cmat):
 S=(qmat@cmat.T).toarray()
 return float(np.max(S,axis=1).mean()) if S.size else 0.0

ap=argparse.ArgumentParser();ap.add_argument("--seed",type=int,required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
SEED=a.seed;OUT=pathlib.Path(a.out);OUT.mkdir(parents=True,exist_ok=True)
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()

isdev=np.array([int(hashlib.sha256((f"E020-STABILITY-{SEED}:"+str(x)).encode()).hexdigest()[:8],16)%5==0 for x in tr.pmcid])
ref=tr.loc[~isdev].reset_index(drop=True);mq=tr.loc[isdev].reset_index(drop=True)

cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=cv.fit_transform(ref.case_prompt.astype(str));Q=cv.transform(mq.case_prompt.astype(str))
cs=TruncatedSVD(256,n_iter=7,random_state=SEED);XC=cs.fit_transform(X).astype(float);QC=cs.transform(Q).astype(float)
ref_pts=[mask_points(r,d) for r,d in zip(ref.diagnostic_reasoning,ref.final_diagnosis)]
mq_pts=[mask_points(r,d) for r,d in zip(mq.diagnostic_reasoning,mq.final_diagnosis)]
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R=rv.fit_transform([" ".join(x) for x in ref_pts]);rs=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR=rs.fit_transform(R).astype(float)
xm=XC.mean(0,keepdims=True);ym=ZR.mean(0,keepdims=True);A=XC-xm;Y=ZR-ym
W=np.linalg.solve(A.T@A+10*np.eye(A.shape[1]),A.T@Y);P=normalize((QC-xm)@W+ym);ZI=normalize(ZR)
CS=(Q@X.T).tocsr();PS=P@ZI.T

allpts=[p for ps in ref_pts for p in ps]+[p for ps in mq_pts for p in ps]
pv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
PM=pv.fit_transform(allpts);off=0;ref_pm=[]
for ps in ref_pts: ref_pm.append(PM[off:off+len(ps)]);off+=len(ps)
mq_pm=[]
for ps in mq_pts: mq_pm.append(PM[off:off+len(ps)]);off+=len(ps)

D=[];YY=[];usable=0;pairs=0
for i in range(len(mq)):
 s=CS.getrow(i).toarray().ravel();cand=rank(s,40);sem=s[cand];prd=PS[i,cand]
 util=np.array([coverage(mq_pm[i],ref_pm[int(j)]) for j in cand])
 F=make_features(sem,prd,np.arange(len(cand)),[len(ref_pts[int(j)]) for j in cand])
 hi=np.argsort(-util)[:5];lo=np.argsort(util)[:10];local=0
 for p in hi:
  for n in lo:
   if util[p]-util[n]<.035: continue
   d=F[p]-F[n];D.append(d);YY.append(1);D.append(-d);YY.append(0);local+=1
 if local: usable+=1;pairs+=local
D=np.asarray(D);YY=np.asarray(YY);sc=StandardScaler();Ds=sc.fit_transform(D)
clf=LogisticRegression(C=1,max_iter=2000,random_state=SEED).fit(Ds,YY);w=clf.coef_[0]/sc.scale_

# locked validation deployment representation
cv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X2=cv2.fit_transform(tr.case_prompt.astype(str));Q2=cv2.transform(va.case_prompt.astype(str))
cs2=TruncatedSVD(256,n_iter=7,random_state=SEED);XC2=cs2.fit_transform(X2).astype(float);QC2=cs2.transform(Q2).astype(float)
tr_pts=[mask_points(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
rv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R2=rv2.fit_transform([" ".join(x) for x in tr_pts]);rs2=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR2=rs2.fit_transform(R2).astype(float)
xm=XC2.mean(0,keepdims=True);ym=ZR2.mean(0,keepdims=True);AA=XC2-xm;YYR=ZR2-ym
WW=np.linalg.solve(AA.T@AA+10*np.eye(AA.shape[1]),AA.T@YYR);PP=normalize((QC2-xm)@WW+ym);ZZ=normalize(ZR2)
CSS=(Q2@X2.T).tocsr();PSS=PP@ZZ.T
counts=[len(x) for x in tr_pts];base=[];rer=[]
for i in range(len(va)):
 s=CSS.getrow(i).toarray().ravel();cand=rank(s,50);base.append(cand.tolist())
 F=make_features(s[cand],PSS[i,cand],np.arange(len(cand)),[counts[int(j)] for j in cand])
 score=F@w;ids=np.asarray(cand);o=np.lexsort((ids,-score));rer.append(ids[o].tolist())
receipt={"seed":SEED,"meta_reference":len(ref),"meta_queries":len(mq),"usable_queries":usable,"preference_pairs":pairs,
 "feature_weights":dict(zip(["semantic","predicted_reasoning","interaction","reciprocal_rank","log_reason_count"],[float(x) for x in w])),
 "dataset_revision":REV}
(OUT/"summary.json").write_text(json.dumps(receipt,indent=2)+"\n")
with (OUT/"rankings.jsonl").open("w") as f:
 for i in range(len(va)): f.write(json.dumps({"query_index":i,"baseline_top50":base[i],"C2_top50":rer[i]})+"\n")
print(json.dumps(receipt,indent=2))
