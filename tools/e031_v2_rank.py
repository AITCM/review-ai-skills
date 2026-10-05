from __future__ import annotations
import json,re,unicodedata,pathlib,hashlib
import numpy as np,pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.linear_model import LogisticRegression

MC_REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
HF=f"https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/{MC_REV}/data"
OUT=pathlib.Path("e031-v2")
SEED=20261004;CASE_DIM=256;REASON_DIM=128;RIDGE_ALPHA=10.;TRAIN_K=40;MARGIN=.03;DEPLOY_K=50;TOP_POS=5;BOTTOM_NEG=10;LOGISTIC_C=1.
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
def pts(r,d):
 dd=nw(d);o=[]
 for x in split_reason(r):
  xx=nw(x)
  if dd and dd in xx:xx=xx.replace(dd," diagnosismask ")
  o.append(xx)
 return o
def rank(s,k):ids=np.arange(len(s));return np.lexsort((ids,-s))[:k]
def F(sem,prd):return np.column_stack([np.asarray(sem,float),np.asarray(prd,float)])
def sf(q,c):
 S=(q@c.T).toarray()
 if not S.size:return 0.
 r=float(np.max(S,axis=1).mean());p=float(np.max(S,axis=0).mean())
 return 2*r*p/(r+p) if r+p else 0.

# Ranking stage reads the locked query-only file. Assert only permitted fields exist.
queries=[json.loads(x) for x in open(OUT/"E031V2_QUERY_ONLY.jsonl",encoding="utf-8") if x.strip()]
assert len(queries)==840
assert all(set(x)=={"query_index","pmcid","case_summary"} for x in queries)
assert [x["query_index"] for x in queries]==list(range(840))
ext=pd.DataFrame(queries)

# Assert method identity against the original one-shot T001 lock.
t001=json.load(open("t001/T001_RANKING_LOCK.json"))
expected={
 "seed":SEED,"case_svd_dim":CASE_DIM,"reasoning_svd_dim":REASON_DIM,"ridge_alpha":RIDGE_ALPHA,
 "training_candidate_k":TRAIN_K,"utility_margin":MARGIN,"deployment_candidate_k":DEPLOY_K,
 "features":["semantic_case_similarity","predicted_reasoning_similarity"],
 "pairwise_top":TOP_POS,"pairwise_bottom":BOTTOM_NEG,"logistic_C":LOGISTIC_C}
for k,v in expected.items():
    assert t001[k]==v,(k,t001[k],v)
assert t001["method"]=="C2-debiased-final-two-feature"

tr=pd.read_parquet(HF+"/train-00000-of-00001.parquet",columns=["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"])
isdev=np.array([int(hashlib.sha256((f"E020-FINAL-{SEED}:"+str(x)).encode()).hexdigest()[:8],16)%5==0 for x in tr.pmcid])
ref=tr.loc[~isdev].reset_index(drop=True);mq=tr.loc[isdev].reset_index(drop=True)

cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=cv.fit_transform(ref.case_prompt.astype(str));Q=cv.transform(mq.case_prompt.astype(str))
cs=TruncatedSVD(CASE_DIM,n_iter=7,random_state=SEED);XC=cs.fit_transform(X).astype(float);QC=cs.transform(Q).astype(float)
rp=[pts(r,d) for r,d in zip(ref.diagnostic_reasoning,ref.final_diagnosis)];mp=[pts(r,d) for r,d in zip(mq.diagnostic_reasoning,mq.final_diagnosis)]
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R=rv.fit_transform([" ".join(x) for x in rp]);rs=TruncatedSVD(REASON_DIM,n_iter=7,random_state=SEED);ZR=rs.fit_transform(R).astype(float)
xm=XC.mean(0,keepdims=True);ym=ZR.mean(0,keepdims=True);A=XC-xm;Y=ZR-ym
W=np.linalg.solve(A.T@A+RIDGE_ALPHA*np.eye(A.shape[1]),A.T@Y);P=normalize((QC-xm)@W+ym);ZI=normalize(ZR);CS=(Q@X.T).tocsr();PS=P@ZI.T
pv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
PM=pv.fit_transform([p for z in rp for p in z]+[p for z in mp for p in z]);off=0;RPM=[]
for z in rp:RPM.append(PM[off:off+len(z)]);off+=len(z)
MPM=[]
for z in mp:MPM.append(PM[off:off+len(z)]);off+=len(z)
D=[];Yy=[];pairs=0
for i in range(len(mq)):
 s=CS.getrow(i).toarray().ravel();cand=rank(s,TRAIN_K);f=F(s[cand],PS[i,cand]);u=np.array([sf(MPM[i],RPM[int(j)]) for j in cand])
 for p in np.argsort(-u)[:TOP_POS]:
  for n in np.argsort(u)[:BOTTOM_NEG]:
   if u[p]-u[n]<MARGIN:continue
   d=f[p]-f[n];D.append(d);Yy.append(1);D.append(-d);Yy.append(0);pairs+=1
D=np.asarray(D);Yy=np.asarray(Yy);sc=StandardScaler();z=sc.fit_transform(D)
clf=LogisticRegression(C=LOGISTIC_C,max_iter=2000,random_state=SEED).fit(z,Yy)
w=clf.coef_[0]/sc.scale_;b=float(clf.intercept_[0]-np.dot(clf.coef_[0],sc.mean_/sc.scale_))

# Refit representation on all MedCaseReasoning train, external query case_summary only.
cv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X2=cv2.fit_transform(tr.case_prompt.astype(str));QE=cv2.transform(ext.case_summary.astype(str))
cs2=TruncatedSVD(CASE_DIM,n_iter=7,random_state=SEED);XC2=cs2.fit_transform(X2).astype(float);QCE=cs2.transform(QE).astype(float)
tp=[pts(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
rv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R2=rv2.fit_transform([" ".join(x) for x in tp]);rs2=TruncatedSVD(REASON_DIM,n_iter=7,random_state=SEED);ZR2=rs2.fit_transform(R2).astype(float)
xm=XC2.mean(0,keepdims=True);ym=ZR2.mean(0,keepdims=True);AA=XC2-xm;YY=ZR2-ym
WW=np.linalg.solve(AA.T@AA+RIDGE_ALPHA*np.eye(AA.shape[1]),AA.T@YY);PP=normalize((QCE-xm)@WW+ym);ZZ=normalize(ZR2)
CSS=(QE@X2.T).tocsr();PSS=PP@ZZ.T
with (OUT/"E031V2_BLIND_RANKINGS.jsonl").open("w",encoding="utf-8") as f:
 for i in range(len(ext)):
  s=CSS.getrow(i).toarray().ravel();cand=rank(s,DEPLOY_K);score=F(s[cand],PSS[i,cand])@w+b;ids=np.asarray(cand);o=np.lexsort((ids,-score))
  f.write(json.dumps({"query_index":i,"pmcid":ext.iloc[i].pmcid,"baseline_top50":cand.tolist(),"method_top50":ids[o].tolist()})+"\n")
config={**expected,"method":"C2-debiased-final-two-feature","training_pairs":pairs}
config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
lock={"status":"external_rankings_locked_before_outcomes_loaded","n_external":len(ext),
 "query_lock_sha256":hashlib.sha256((OUT/"E031V2_QUERY_ONLY.jsonl").read_bytes()).hexdigest(),
 "ranking_fields_used":["pmcid","case_summary"],"external_outcome_fields_available_to_ranker":[],
 "frozen_method_source":"T001_RANKING_LOCK.json","method_config":config,"method_config_sha256":config_hash,
 "weights":[float(x) for x in w]}
(OUT/"E031V2_RANKING_LOCK.json").write_text(json.dumps(lock,indent=2)+"\n")
print(json.dumps(lock,indent=2))
