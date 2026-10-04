from __future__ import annotations
import argparse,json,re,unicodedata,pathlib,hashlib
import numpy as np,pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.linear_model import LogisticRegression

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
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
def rank(s,k=50):
 ids=np.arange(len(s));return np.lexsort((ids,-s))[:k]
def feats(sem,prd,pos):
 sem=np.asarray(sem,float);prd=np.asarray(prd,float);rr=1/(1+np.asarray(pos,float))
 return np.column_stack([sem,prd,sem*prd,rr])
def set_scores(q,c):
 S=(q@c.T).toarray()
 if not S.size:return 0.,0.,0.
 rec=float(np.max(S,axis=1).mean());prec=float(np.max(S,axis=0).mean());f=2*rec*prec/(rec+prec) if rec+prec else 0.
 return rec,prec,f
ap=argparse.ArgumentParser();ap.add_argument("--seed",type=int,required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
SEED=a.seed;OUT=pathlib.Path(a.out);OUT.mkdir(parents=True,exist_ok=True)
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]]
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["case_prompt","diagnostic_reasoning","final_diagnosis"]]
isdev=np.array([int(hashlib.sha256((f"E020-D5C-{SEED}:"+str(x)).encode()).hexdigest()[:8],16)%5==0 for x in tr.pmcid]);ref=tr.loc[~isdev].reset_index(drop=True);mq=tr.loc[isdev].reset_index(drop=True)
cv=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32);X=cv.fit_transform(ref.case_prompt.astype(str));Q=cv.transform(mq.case_prompt.astype(str));cs=TruncatedSVD(256,n_iter=7,random_state=SEED);XC=cs.fit_transform(X).astype(float);QC=cs.transform(Q).astype(float)
rp=[mask_points(r,d) for r,d in zip(ref.diagnostic_reasoning,ref.final_diagnosis)];mp=[mask_points(r,d) for r,d in zip(mq.diagnostic_reasoning,mq.final_diagnosis)]
rv=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32);R=rv.fit_transform([" ".join(x) for x in rp]);rs=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR=rs.fit_transform(R).astype(float)
xm=XC.mean(0,keepdims=True);ym=ZR.mean(0,keepdims=True);AA=XC-xm;YYR=ZR-ym;W=np.linalg.solve(AA.T@AA+10*np.eye(AA.shape[1]),AA.T@YYR);P=normalize((QC-xm)@W+ym);ZI=normalize(ZR);CS=(Q@X.T).tocsr();PS=P@ZI.T
allp=[p for ps in rp for p in ps]+[p for ps in mp for p in ps];pv=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32);PM=pv.fit_transform(allp);off=0;rpm=[]
for ps in rp:rpm.append(PM[off:off+len(ps)]);off+=len(ps)
mpm=[]
for ps in mp:mpm.append(PM[off:off+len(ps)]);off+=len(ps)
D=[];Y=[];pairs=0
for i in range(len(mq)):
 s=CS.getrow(i).toarray().ravel();cand=rank(s,40);F=feats(s[cand],PS[i,cand],np.arange(len(cand)))
 u=np.array([set_scores(mpm[i],rpm[int(j)])[2] for j in cand])
 for p in np.argsort(-u)[:5]:
  for n in np.argsort(u)[:10]:
   if u[p]-u[n]<.03:continue
   d=F[p]-F[n];D.append(d);Y.append(1);D.append(-d);Y.append(0);pairs+=1
D=np.asarray(D);Y=np.asarray(Y);sc=StandardScaler();Z=sc.fit_transform(D);clf=LogisticRegression(C=1,max_iter=2000,random_state=SEED).fit(Z,Y);w=clf.coef_[0]/sc.scale_
# deployment
cv2=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32);X2=cv2.fit_transform(tr.case_prompt.astype(str));Q2=cv2.transform(va.case_prompt.astype(str));cs2=TruncatedSVD(256,n_iter=7,random_state=SEED);XC2=cs2.fit_transform(X2).astype(float);QC2=cs2.transform(Q2).astype(float)
tp=[mask_points(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)];rv2=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32);R2=rv2.fit_transform([" ".join(x) for x in tp]);rs2=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR2=rs2.fit_transform(R2).astype(float)
xm=XC2.mean(0,keepdims=True);ym=ZR2.mean(0,keepdims=True);AA=XC2-xm;YYR=ZR2-ym;WW=np.linalg.solve(AA.T@AA+10*np.eye(AA.shape[1]),AA.T@YYR);PP=normalize((QC2-xm)@WW+ym);ZZ=normalize(ZR2);CSS=(Q2@X2.T).tocsr();PSS=PP@ZZ.T
base=[];rer=[]
for i in range(len(va)):
 s=CSS.getrow(i).toarray().ravel();cand=rank(s,50);base.append(cand.tolist());F=feats(s[cand],PSS[i,cand],np.arange(len(cand)));score=F@w;ids=np.asarray(cand);o=np.lexsort((ids,-score));rer.append(ids[o].tolist())
# heldout symmetric evaluator train-fit only
flat=[p for ps in tp for p in ps];ev=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32);T=ev.fit_transform(flat);offs=[];o=0
for ps in tp:offs.append((o,o+len(ps)));o+=len(ps)
vp=[mask_points(r,d) for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)];V=ev.transform([p for ps in vp for p in ps]);vo=[];o=0
for ps in vp:vo.append((o,o+len(ps)));o+=len(ps)
def score(i,cands):
 s,e=vo[i];q=V[s:e];m=[]
 for j in cands:a,b=offs[j];m.append(T[a:b])
 return set_scores(q,vstack(m))[2]
res={"seed":SEED,"pairs":pairs,"weights":[float(x) for x in w]}
for k in (1,3,10):
 b=np.array([score(i,base[i][:k]) for i in range(len(va))]);c=np.array([score(i,rer[i][:k]) for i in range(len(va))]);d=c-b;res[f"delta@{k}"]=float(d.mean())
(OUT/"summary.json").write_text(json.dumps(res,indent=2)+"\n")
with (OUT/"rankings.jsonl").open("w") as f:
 for i in range(len(va)):f.write(json.dumps({"query_index":i,"baseline_top50":base[i],"D5_top50":rer[i]})+"\n")
print(json.dumps(res,indent=2))
