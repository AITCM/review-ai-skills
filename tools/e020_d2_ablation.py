from __future__ import annotations
import json,re,unicodedata,pathlib,hashlib
from collections import defaultdict
import numpy as np,pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.linear_model import LogisticRegression

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e020-d2-output");OUT.mkdir(exist_ok=True);SEED=20261004;RNG=np.random.default_rng(SEED)
PAT=re.compile(r'^\s*(?:\((\d+)\)|(\d+)[.)])\s*',re.M)
NAMES=["semantic","predicted_reasoning","interaction","reciprocal_rank","log_reason_count"]
ABLATIONS={
 "full":[0,1,2,3,4],
 "no_reasoning_signal":[0,3,4],
 "no_interaction":[0,1,3,4],
 "no_rank":[0,1,2,4],
 "no_reason_count":[0,1,2,3],
 "semantic_only":[0],
 "predicted_reasoning_only":[1]
}
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
def mask_points(reason,diag):
 dd=nw(diag);out=[]
 for x in split_reason(reason):
  xx=nw(x)
  if dd and dd in xx:xx=xx.replace(dd," diagnosismask ")
  out.append(xx)
 return out
def rank(s,k=50):
 ids=np.arange(len(s));return np.lexsort((ids,-s))[:k]
def feats(sem,prd,rankpos,cnt):
 sem=np.asarray(sem,float);prd=np.asarray(prd,float);rr=1/(1+np.asarray(rankpos,float));rc=np.log1p(np.asarray(cnt,float))
 return np.column_stack([sem,prd,sem*prd,rr,rc])
def cov(qmat,cmat):
 S=(qmat@cmat.T).toarray();return float(np.max(S,axis=1).mean()) if S.size else 0.0

tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
isdev=np.array([int(hashlib.sha256(("E020-C2:"+str(x)).encode()).hexdigest()[:8],16)%5==0 for x in tr.pmcid])
ref=tr.loc[~isdev].reset_index(drop=True);mq=tr.loc[isdev].reset_index(drop=True)
cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=cv.fit_transform(ref.case_prompt.astype(str));Q=cv.transform(mq.case_prompt.astype(str))
cs=TruncatedSVD(256,n_iter=7,random_state=SEED);XC=cs.fit_transform(X).astype(float);QC=cs.transform(Q).astype(float)
ref_pts=[mask_points(r,d) for r,d in zip(ref.diagnostic_reasoning,ref.final_diagnosis)]
mq_pts=[mask_points(r,d) for r,d in zip(mq.diagnostic_reasoning,mq.final_diagnosis)]
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R=rv.fit_transform([" ".join(x) for x in ref_pts]);rs=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR=rs.fit_transform(R).astype(float)
xm=XC.mean(0,keepdims=True);ym=ZR.mean(0,keepdims=True);A=XC-xm;Y=ZR-ym;W=np.linalg.solve(A.T@A+10*np.eye(A.shape[1]),A.T@Y)
P=normalize((QC-xm)@W+ym);ZI=normalize(ZR);CS=(Q@X.T).tocsr();PS=P@ZI.T
allpts=[p for ps in ref_pts for p in ps]+[p for ps in mq_pts for p in ps]
pv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
PM=pv.fit_transform(allpts);off=0;ref_pm=[]
for ps in ref_pts:ref_pm.append(PM[off:off+len(ps)]);off+=len(ps)
mq_pm=[]
for ps in mq_pts:mq_pm.append(PM[off:off+len(ps)]);off+=len(ps)
D=[];YY=[]
for i in range(len(mq)):
 s=CS.getrow(i).toarray().ravel();cand=rank(s,40);u=np.array([cov(mq_pm[i],ref_pm[int(j)]) for j in cand])
 F=feats(s[cand],PS[i,cand],np.arange(len(cand)),[len(ref_pts[int(j)]) for j in cand])
 hi=np.argsort(-u)[:5];lo=np.argsort(u)[:10]
 for p in hi:
  for n in lo:
   if u[p]-u[n]<.035:continue
   d=F[p]-F[n];D.append(d);YY.append(1);D.append(-d);YY.append(0)
D=np.asarray(D);YY=np.asarray(YY)

models={}
for name,cols in ABLATIONS.items():
 sc=StandardScaler();Z=sc.fit_transform(D[:,cols]);clf=LogisticRegression(C=1,max_iter=2000,random_state=SEED).fit(Z,YY)
 w=clf.coef_[0]/sc.scale_;models[name]={"cols":cols,"weights":w,"intercept":float(clf.intercept_[0]-np.dot(clf.coef_[0],sc.mean_/sc.scale_))}

# Full-train deployment features
cv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X2=cv2.fit_transform(tr.case_prompt.astype(str));Q2=cv2.transform(va.case_prompt.astype(str))
cs2=TruncatedSVD(256,n_iter=7,random_state=SEED);XC2=cs2.fit_transform(X2).astype(float);QC2=cs2.transform(Q2).astype(float)
tr_pts=[mask_points(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
rv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R2=rv2.fit_transform([" ".join(x) for x in tr_pts]);rs2=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR2=rs2.fit_transform(R2).astype(float)
xm=XC2.mean(0,keepdims=True);ym=ZR2.mean(0,keepdims=True);AA=XC2-xm;YYR=ZR2-ym;WW=np.linalg.solve(AA.T@AA+10*np.eye(AA.shape[1]),AA.T@YYR)
PP=normalize((QC2-xm)@WW+ym);ZZ=normalize(ZR2);CSS=(Q2@X2.T).tocsr();PSS=PP@ZZ.T
cnt=[len(x) for x in tr_pts];base=[];ranks={n:[] for n in ABLATIONS}
for i in range(len(va)):
 s=CSS.getrow(i).toarray().ravel();cand=rank(s,50);base.append(cand.tolist());F=feats(s[cand],PSS[i,cand],np.arange(len(cand)),[cnt[int(j)] for j in cand])
 for name,m in models.items():
  score=F[:,m["cols"]]@m["weights"]+m["intercept"];ids=np.asarray(cand);o=np.lexsort((ids,-score));ranks[name].append(ids[o].tolist())

# held-out reasoning utility metric fit on training only
flat=[p for ps in tr_pts for p in ps]
ev=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
T=ev.fit_transform(flat);offs=[];o=0
for ps in tr_pts:offs.append((o,o+len(ps)));o+=len(ps)
vpts=[mask_points(r,d) for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
V=ev.transform([p for ps in vpts for p in ps]);vo=[];o=0
for ps in vpts:vo.append((o,o+len(ps)));o+=len(ps)
def ecov(i,cands):
 s,e=vo[i];q=V[s:e];m=[]
 for j in cands:
  a,b=offs[j];m.append(T[a:b])
 return float(np.max((q@vstack(m).T).toarray(),axis=1).mean())
tl=[nw(x) for x in tr.final_diagnosis];vl=[nw(x) for x in va.final_diagnosis]
methods={"tfidf":base,**ranks};score={m:{k:np.array([ecov(i,R[i][:k]) for i in range(len(va))]) for k in (1,3,10)} for m,R in methods.items()}
summary={"seed":SEED,"features":NAMES,"ablations":{}}
for name in methods:
 summary["ablations"][name]={}
 for k in (1,3,10):
  x=score[name][k];summary["ablations"][name][f"coverage@{k}"]=float(x.mean())
  summary["ablations"][name][f"exact_hit@{k}"]=int(sum(any(tl[j]==vl[i] for j in methods[name][i][:k]) for i in range(len(va))))
  if name!="tfidf":
   d=x-score["tfidf"][k];boots=[]
   for _ in range(3000):
    ids=RNG.integers(0,len(d),len(d));boots.append(float(d[ids].mean()))
   summary["ablations"][name][f"delta@{k}"]={"mean":float(d.mean()),"ci95":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))]}
summary["model_weights"]={n:{NAMES[c]:float(w) for c,w in zip(m["cols"],m["weights"])} for n,m in models.items()}
summary["limits"]=["Validation reasoning is evaluation only.","All ablations share the same meta-split, representations, pairwise examples, candidates, and evaluation metric.","Test remains sealed."]
(OUT/"E020_D2_ABLATION_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
