from __future__ import annotations
import argparse,json,re,unicodedata,pathlib,hashlib
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.sparse import vstack

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
PAT=re.compile(r'^\s*(?:\((\d+)\)|(\d+)[.)])\s*',re.M)
FEATURE_NAMES=["semantic","predicted_reasoning","interaction","reciprocal_rank","log_reason_count"]
ABLATIONS={
 "full":[0,1,2,3,4],
 "no_predicted_reasoning":[0,3,4],
 "reasoning_only":[1,4],
 "semantic_only":[0],
 "semantic_plus_rank":[0,3],
 "full_no_interaction":[0,1,3,4],
 "full_no_rank":[0,1,2,4],
 "full_no_reason_count":[0,1,2,3],
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
def mask_points(r,d):
 dd=nw(d);out=[]
 for x in split_reason(r):
  xx=nw(x)
  if dd and dd in xx:xx=xx.replace(dd," diagnosismask ")
  out.append(xx)
 return out
def rank(s,k=50):
 ids=np.arange(len(s));return np.lexsort((ids,-s))[:k]
def feats(sem,prd,pos,cnt):
 sem=np.asarray(sem,float);prd=np.asarray(prd,float);rr=1/(1+np.asarray(pos,float));rc=np.log1p(np.asarray(cnt,float))
 return np.column_stack([sem,prd,sem*prd,rr,rc])
def cov(q,c):
 S=(q@c.T).toarray();return float(np.max(S,axis=1).mean()) if S.size else 0.0

ap=argparse.ArgumentParser();ap.add_argument("--seed",type=int,required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
SEED=a.seed;OUT=pathlib.Path(a.out);OUT.mkdir(parents=True,exist_ok=True)
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]]
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["case_prompt","diagnostic_reasoning","final_diagnosis"]]
isdev=np.array([int(hashlib.sha256((f"E020-D2-{SEED}:"+str(x)).encode()).hexdigest()[:8],16)%5==0 for x in tr.pmcid])
ref=tr.loc[~isdev].reset_index(drop=True);mq=tr.loc[isdev].reset_index(drop=True)

cv=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=cv.fit_transform(ref.case_prompt.astype(str));Q=cv.transform(mq.case_prompt.astype(str))
cs=TruncatedSVD(256,n_iter=7,random_state=SEED);XC=cs.fit_transform(X).astype(float);QC=cs.transform(Q).astype(float)
rpts=[mask_points(r,d) for r,d in zip(ref.diagnostic_reasoning,ref.final_diagnosis)]
mpts=[mask_points(r,d) for r,d in zip(mq.diagnostic_reasoning,mq.final_diagnosis)]
rv=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R=rv.fit_transform([" ".join(x) for x in rpts]);rs=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR=rs.fit_transform(R).astype(float)
xm=XC.mean(0,keepdims=True);ym=ZR.mean(0,keepdims=True);AA=XC-xm;YYR=ZR-ym
W=np.linalg.solve(AA.T@AA+10*np.eye(AA.shape[1]),AA.T@YYR);P=normalize((QC-xm)@W+ym);ZI=normalize(ZR)
CS=(Q@X.T).tocsr();PS=P@ZI.T
allp=[p for ps in rpts for p in ps]+[p for ps in mpts for p in ps]
pv=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
PM=pv.fit_transform(allp);off=0;rpm=[]
for ps in rpts:rpm.append(PM[off:off+len(ps)]);off+=len(ps)
mpm=[]
for ps in mpts:mpm.append(PM[off:off+len(ps)]);off+=len(ps)

D=[];Y=[];usable=0
for i in range(len(mq)):
 s=CS.getrow(i).toarray().ravel();cand=rank(s,40);F=feats(s[cand],PS[i,cand],np.arange(len(cand)),[len(rpts[int(j)]) for j in cand])
 util=np.array([cov(mpm[i],rpm[int(j)]) for j in cand])
 local=0
 for p in np.argsort(-util)[:5]:
  for n in np.argsort(util)[:10]:
   if util[p]-util[n]<.035:continue
   d=F[p]-F[n];D.append(d);Y.append(1);D.append(-d);Y.append(0);local+=1
 if local:usable+=1
D=np.asarray(D);Y=np.asarray(Y)

models={}
for name,cols in ABLATIONS.items():
 sc=StandardScaler();Z=sc.fit_transform(D[:,cols]);clf=LogisticRegression(C=1,max_iter=2000,random_state=SEED).fit(Z,Y)
 models[name]=(cols,clf.coef_[0]/sc.scale_)

# validation representations refit on all training
cv2=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X2=cv2.fit_transform(tr.case_prompt.astype(str));Q2=cv2.transform(va.case_prompt.astype(str))
cs2=TruncatedSVD(256,n_iter=7,random_state=SEED);XC2=cs2.fit_transform(X2).astype(float);QC2=cs2.transform(Q2).astype(float)
tpts=[mask_points(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
rv2=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R2=rv2.fit_transform([" ".join(x) for x in tpts]);rs2=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR2=rs2.fit_transform(R2).astype(float)
xm=XC2.mean(0,keepdims=True);ym=ZR2.mean(0,keepdims=True);AA=XC2-xm;YYR=ZR2-ym
WW=np.linalg.solve(AA.T@AA+10*np.eye(AA.shape[1]),AA.T@YYR);PP=normalize((QC2-xm)@WW+ym);ZZ=normalize(ZR2)
CSS=(Q2@X2.T).tocsr();PSS=PP@ZZ.T;cnt=[len(x) for x in tpts]
ranks={"baseline":[]}; 
for name in ABLATIONS:ranks[name]=[]
for i in range(len(va)):
 s=CSS.getrow(i).toarray().ravel();cand=rank(s,50);ranks["baseline"].append(cand.tolist())
 F=feats(s[cand],PSS[i,cand],np.arange(len(cand)),[cnt[int(j)] for j in cand])
 for name,(cols,w) in models.items():
  score=F[:,cols]@w;ids=np.asarray(cand);o=np.lexsort((ids,-score));ranks[name].append(ids[o].tolist())

# independent held-out point coverage: vectorizer fit training only
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
results={}
for name,RK in ranks.items():
 results[name]={}
 for k in (1,3,10):
  arr=np.array([vc(i,RK[i][:k]) for i in range(len(va))])
  results[name][str(k)]=arr
summary={"seed":SEED,"usable_meta_queries":usable,"n_pair_rows":len(Y),"ablation_results":{}}
base=results["baseline"]
for name in ABLATIONS:
 summary["ablation_results"][name]={}
 for k in (1,3,10):
  d=results[name][str(k)]-base[str(k)]
  summary["ablation_results"][name][str(k)]={"mean_delta":float(d.mean()),"improved":int((d>0).sum()),"worsened":int((d<0).sum())}
summary["feature_weights"]={name:{FEATURE_NAMES[c]:float(wi) for c,wi in zip(cols,w)} for name,(cols,w) in models.items()}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
