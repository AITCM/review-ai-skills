from __future__ import annotations
import json,re,unicodedata,pathlib,hashlib
import numpy as np,pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.linear_model import LogisticRegression
REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e020-d7-output");OUT.mkdir(exist_ok=True);SEED=20261004;RNG=np.random.default_rng(SEED)
PAT=re.compile(r'^\s*(?:\((\d+)\)|(\d+)[.)])\s*',re.M);NAMES=["semantic","predicted_reasoning","interaction","reciprocal_rank"]
AB={"full":[0,1,2,3],"no_reasoning_signal":[0,3],"no_interaction":[0,1,3],"no_rank":[0,1,2],"semantic_only":[0],"predicted_reasoning_only":[1]}
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
def pts(r,d):
 dd=nw(d);o=[]
 for x in split_reason(r):
  xx=nw(x)
  if dd and dd in xx:xx=xx.replace(dd," diagnosismask ")
  o.append(xx)
 return o
def rank(s,k):ids=np.arange(len(s));return np.lexsort((ids,-s))[:k]
def F(sem,prd,rp):
 sem=np.asarray(sem,float);prd=np.asarray(prd,float);rr=1/(1+np.asarray(rp,float));return np.column_stack([sem,prd,sem*prd,rr])
def sf(q,c):
 S=(q@c.T).toarray();r=float(np.max(S,axis=1).mean());p=float(np.max(S,axis=0).mean());return 2*r*p/(r+p) if r+p else 0.
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]];va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["case_prompt","diagnostic_reasoning","final_diagnosis"]]
isdev=np.array([int(hashlib.sha256(("E020-D5:"+str(x)).encode()).hexdigest()[:8],16)%5==0 for x in tr.pmcid]);ref=tr.loc[~isdev].reset_index(drop=True);mq=tr.loc[isdev].reset_index(drop=True)
cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32);X=cv.fit_transform(ref.case_prompt.astype(str));Q=cv.transform(mq.case_prompt.astype(str));cs=TruncatedSVD(256,n_iter=7,random_state=SEED);XC=cs.fit_transform(X).astype(float);QC=cs.transform(Q).astype(float)
rp=[pts(r,d) for r,d in zip(ref.diagnostic_reasoning,ref.final_diagnosis)];mp=[pts(r,d) for r,d in zip(mq.diagnostic_reasoning,mq.final_diagnosis)]
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32);R=rv.fit_transform([" ".join(x) for x in rp]);rs=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR=rs.fit_transform(R).astype(float)
xm=XC.mean(0,keepdims=True);ym=ZR.mean(0,keepdims=True);A=XC-xm;Y=ZR-ym;W=np.linalg.solve(A.T@A+10*np.eye(A.shape[1]),A.T@Y);P=normalize((QC-xm)@W+ym);ZI=normalize(ZR);CS=(Q@X.T).tocsr();PS=P@ZI.T
pv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32);PM=pv.fit_transform([p for z in rp for p in z]+[p for z in mp for p in z]);off=0;RPM=[]
for z in rp:RPM.append(PM[off:off+len(z)]);off+=len(z)
MPM=[]
for z in mp:MPM.append(PM[off:off+len(z)]);off+=len(z)
D=[];Yy=[]
for i in range(len(mq)):
 s=CS.getrow(i).toarray().ravel();cand=rank(s,40);f=F(s[cand],PS[i,cand],np.arange(40));u=np.array([sf(MPM[i],RPM[int(j)]) for j in cand])
 for p in np.argsort(-u)[:5]:
  for n in np.argsort(u)[:10]:
   if u[p]-u[n]<.03:continue
   d=f[p]-f[n];D.append(d);Yy.append(1);D.append(-d);Yy.append(0)
D=np.asarray(D);Yy=np.asarray(Yy);models={}
for name,cols in AB.items():
 sc=StandardScaler();z=sc.fit_transform(D[:,cols]);clf=LogisticRegression(C=1,max_iter=2000,random_state=SEED).fit(z,Yy)
 models[name]=(cols,clf.coef_[0]/sc.scale_,float(clf.intercept_[0]-np.dot(clf.coef_[0],sc.mean_/sc.scale_)))
# deployment
cv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32);X2=cv2.fit_transform(tr.case_prompt.astype(str));Q2=cv2.transform(va.case_prompt.astype(str));cs2=TruncatedSVD(256,n_iter=7,random_state=SEED);XC2=cs2.fit_transform(X2).astype(float);QC2=cs2.transform(Q2).astype(float)
tp=[pts(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)];rv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32);R2=rv2.fit_transform([" ".join(x) for x in tp]);rs2=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR2=rs2.fit_transform(R2).astype(float)
xm=XC2.mean(0,keepdims=True);ym=ZR2.mean(0,keepdims=True);AA=XC2-xm;YY=ZR2-ym;WW=np.linalg.solve(AA.T@AA+10*np.eye(AA.shape[1]),AA.T@YY);PP=normalize((QC2-xm)@WW+ym);ZZ=normalize(ZR2);CSS=(Q2@X2.T).tocsr();PSS=PP@ZZ.T
base=[];ranks={x:[] for x in AB}
for i in range(len(va)):
 s=CSS.getrow(i).toarray().ravel();cand=rank(s,50);base.append(cand.tolist());f=F(s[cand],PSS[i,cand],np.arange(50))
 for name,(cols,w,b) in models.items():
  score=f[:,cols]@w+b;ids=np.asarray(cand);o=np.lexsort((ids,-score));ranks[name].append(ids[o].tolist())
# eval symmetric softF1
ev=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32);T=ev.fit_transform([p for z in tp for p in z]);offs=[];o=0
for z in tp:offs.append((o,o+len(z)));o+=len(z)
vp=[pts(r,d) for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)];V=ev.transform([p for z in vp for p in z]);vo=[];o=0
for z in vp:vo.append((o,o+len(z)));o+=len(z)
def ec(i,cands):
 s,e=vo[i];q=V[s:e];m=[]
 for j in cands:a,b=offs[j];m.append(T[a:b])
 return sf(q,vstack(m))
methods={"tfidf":base,**ranks};summary={"features":NAMES,"ablations":{}}
for name,Rr in methods.items():
 summary["ablations"][name]={}
 for k in (1,3,10):
  x=np.array([ec(i,Rr[i][:k]) for i in range(len(va))]);summary["ablations"][name][f"softF1@{k}"]=float(x.mean())
  if name!="tfidf":
   b=np.array([ec(i,base[i][:k]) for i in range(len(va))]);d=x-b;boots=[]
   for _ in range(3000):ids=RNG.integers(0,len(d),len(d));boots.append(float(d[ids].mean()))
   summary["ablations"][name][f"delta@{k}"]={"mean":float(d.mean()),"ci95":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))]}
summary["weights"]={n:{NAMES[c]:float(x) for c,x in zip(cols,w)} for n,(cols,w,b) in models.items()}
summary["limits"]=["Debiased symmetric utility; no reason-count feature.","Same meta-split/candidates/training pairs across ablations.","Test sealed."]
(OUT/"E020_D7_DEBIASED_ABLATION.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
