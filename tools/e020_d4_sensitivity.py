from __future__ import annotations
import json,re,unicodedata,pathlib,hashlib
import numpy as np,pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.linear_model import LogisticRegression

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e020-d4-output");OUT.mkdir(exist_ok=True);SEED=20261004;RNG=np.random.default_rng(SEED)
MARGINS=[0.02,0.035,0.05,0.07];KS=[20,50,100]
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
def feats(sem,prd,rp,cnt):
 sem=np.asarray(sem,float);prd=np.asarray(prd,float);rr=1/(1+np.asarray(rp,float));rc=np.log1p(np.asarray(cnt,float))
 return np.column_stack([sem,prd,sem*prd,rr,rc])
def cov(q,c):
 S=(q@c.T).toarray();return float(np.max(S,axis=1).mean()) if S.size else 0

tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
isdev=np.array([int(hashlib.sha256(("E020-C2:"+str(x)).encode()).hexdigest()[:8],16)%5==0 for x in tr.pmcid])
ref=tr.loc[~isdev].reset_index(drop=True);mq=tr.loc[isdev].reset_index(drop=True)

cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=cv.fit_transform(ref.case_prompt.astype(str));Q=cv.transform(mq.case_prompt.astype(str));cs=TruncatedSVD(256,n_iter=7,random_state=SEED)
XC=cs.fit_transform(X).astype(float);QC=cs.transform(Q).astype(float)
rp= [mask_points(r,d) for r,d in zip(ref.diagnostic_reasoning,ref.final_diagnosis)]
mp= [mask_points(r,d) for r,d in zip(mq.diagnostic_reasoning,mq.final_diagnosis)]
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R=rv.fit_transform([" ".join(x) for x in rp]);rs=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR=rs.fit_transform(R).astype(float)
xm=XC.mean(0,keepdims=True);ym=ZR.mean(0,keepdims=True);A=XC-xm;Y=ZR-ym;W=np.linalg.solve(A.T@A+10*np.eye(A.shape[1]),A.T@Y)
P=normalize((QC-xm)@W+ym);ZI=normalize(ZR);CS=(Q@X.T).tocsr();PS=P@ZI.T

allpts=[p for ps in rp for p in ps]+[p for ps in mp for p in ps]
pv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
PM=pv.fit_transform(allpts);off=0;rpm=[]
for ps in rp:rpm.append(PM[off:off+len(ps)]);off+=len(ps)
mpm=[]
for ps in mp:mpm.append(PM[off:off+len(ps)]);off+=len(ps)
# cache top40 training candidate features/utilities
cache=[]
for i in range(len(mq)):
 s=CS.getrow(i).toarray().ravel();cand=rank(s,40);F=feats(s[cand],PS[i,cand],np.arange(40),[len(rp[int(j)]) for j in cand]);u=np.array([cov(mpm[i],rpm[int(j)]) for j in cand])
 cache.append((cand,F,u))
models={}
for margin in MARGINS:
 D=[];YY=[];pairs=0
 for cand,F,u in cache:
  for p in np.argsort(-u)[:5]:
   for n in np.argsort(u)[:10]:
    if u[p]-u[n]<margin:continue
    d=F[p]-F[n];D.append(d);YY.append(1);D.append(-d);YY.append(0);pairs+=1
 D=np.asarray(D);YY=np.asarray(YY);sc=StandardScaler();Z=sc.fit_transform(D);clf=LogisticRegression(C=1,max_iter=2000,random_state=SEED).fit(Z,YY)
 models[margin]={"w":clf.coef_[0]/sc.scale_,"b":float(clf.intercept_[0]-np.dot(clf.coef_[0],sc.mean_/sc.scale_)),"pairs":pairs}

# deployment representation
cv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X2=cv2.fit_transform(tr.case_prompt.astype(str));Q2=cv2.transform(va.case_prompt.astype(str));cs2=TruncatedSVD(256,n_iter=7,random_state=SEED)
XC2=cs2.fit_transform(X2).astype(float);QC2=cs2.transform(Q2).astype(float)
tp=[mask_points(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
rv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R2=rv2.fit_transform([" ".join(x) for x in tp]);rs2=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR2=rs2.fit_transform(R2).astype(float)
xm=XC2.mean(0,keepdims=True);ym=ZR2.mean(0,keepdims=True);AA=XC2-xm;YYR=ZR2-ym;WW=np.linalg.solve(AA.T@AA+10*np.eye(AA.shape[1]),AA.T@YYR)
PP=normalize((QC2-xm)@WW+ym);ZZ=normalize(ZR2);CSS=(Q2@X2.T).tocsr();PSS=PP@ZZ.T;cnt=[len(x) for x in tp]

# held-out utility evaluator fit on all train only
flat=[p for ps in tp for p in ps]
ev=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32);T=ev.fit_transform(flat);offs=[];o=0
for ps in tp:offs.append((o,o+len(ps)));o+=len(ps)
vp=[mask_points(r,d) for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
V=ev.transform([p for ps in vp for p in ps]);vo=[];o=0
for ps in vp:vo.append((o,o+len(ps)));o+=len(ps)
def ecov(i,cands):
 s,e=vo[i];q=V[s:e];m=[]
 for j in cands:a,b=offs[j];m.append(T[a:b])
 return float(np.max((q@vstack(m).T).toarray(),axis=1).mean())
# common baseline for each K
pre=[]
for i in range(len(va)):
 s=CSS.getrow(i).toarray().ravel();pre.append((s,rank(s,max(KS))))
results={}
for margin,m in models.items():
 for K in KS:
  key=f"margin={margin}_K={K}";vals={1:[],3:[],10:[]};basevals={1:[],3:[],10:[]}
  for i,(s,ordermax) in enumerate(pre):
   cand=ordermax[:K];F=feats(s[cand],PSS[i,cand],np.arange(K),[cnt[int(j)] for j in cand]);score=F@m["w"]+m["b"];ids=np.asarray(cand);o=np.lexsort((ids,-score));rr=ids[o]
   for k in (1,3,10):
    kk=min(k,K);vals[k].append(ecov(i,rr[:kk]));basevals[k].append(ecov(i,cand[:kk]))
  results[key]={"pairs":m["pairs"]}
  for k in (1,3,10):
   d=np.asarray(vals[k])-np.asarray(basevals[k]);boots=[]
   for _ in range(2000):
    ids=RNG.integers(0,len(d),len(d));boots.append(float(d[ids].mean()))
   results[key][f"delta@{k}"]={"mean":float(d.mean()),"ci95":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))]}
summary={"margins":MARGINS,"candidate_K":KS,"results":results,
 "main_config":"margin=0.035_K=50","selection_rule":"sensitivity only; no post-hoc retuning from validation",
 "limits":["Validation reasoning is evaluation only.","Training candidate pool remains fixed at top40 for all margin conditions.","Test sealed."]}
(OUT/"E020_D4_PARAMETER_SENSITIVITY.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
