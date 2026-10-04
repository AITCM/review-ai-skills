from __future__ import annotations
import json,re,unicodedata,pathlib,hashlib
from collections import defaultdict
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import binomtest

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e020-c2-output");OUT.mkdir(exist_ok=True);SEED=20261004
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
def mask_reason_points(reason,diag):
 dd=nw(diag);out=[]
 for x in split_reason(reason):
  xx=nw(x)
  if dd and dd in xx:xx=xx.replace(dd," diagnosismask ")
  out.append(xx)
 return out
def rank(s,k=50):
 ids=np.arange(len(s));return np.lexsort((ids,-s))[:k]
def make_features(sem,prd,rankpos,cand_reason_count):
 sem=np.asarray(sem,float);prd=np.asarray(prd,float);rr=1/(1+np.asarray(rankpos,float));rc=np.log1p(np.asarray(cand_reason_count,float))
 return np.column_stack([sem,prd,sem*prd,rr,rc])
def exact_metrics(name,ranks,tl,vl,mp):
 elig=[i for i,y in enumerate(vl) if y in mp];o={"method":name,"eligible":len(elig)}
 for k in (1,3,10,50):
  a=sum(any(tl[j]==vl[i] for j in ranks[i][:k]) for i in range(len(vl)))
  e=sum(any(tl[j]==vl[i] for j in ranks[i][:k]) for i in elig)
  o[f"hit{k}_all"]=a;o[f"hit{k}_all_rate"]=a/len(vl);o[f"hit{k}_eligible"]=e;o[f"hit{k}_eligible_rate"]=e/len(elig)
 return o

tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()

# fixed train-only meta split
isdev=np.array([int(hashlib.sha256(("E020-C2:"+str(x)).encode()).hexdigest()[:8],16)%5==0 for x in tr.pmcid])
ref=tr.loc[~isdev].reset_index(drop=True); mq=tr.loc[isdev].reset_index(drop=True)

# Case lexical representation and case->masked-reasoning latent projection (training only)
cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=cv.fit_transform(ref.case_prompt.astype(str));Q=cv.transform(mq.case_prompt.astype(str))
case_svd=TruncatedSVD(256,n_iter=7,random_state=SEED);XC=case_svd.fit_transform(X).astype(float);QC=case_svd.transform(Q).astype(float)
ref_rdoc=[" ".join(mask_reason_points(r,d)) for r,d in zip(ref.diagnostic_reasoning,ref.final_diagnosis)]
mq_rdoc=[" ".join(mask_reason_points(r,d)) for r,d in zip(mq.diagnostic_reasoning,mq.final_diagnosis)]
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R=rv.fit_transform(ref_rdoc); rsvd=TruncatedSVD(128,n_iter=7,random_state=SEED); ZR=rsvd.fit_transform(R).astype(float)
xm=XC.mean(0,keepdims=True);ym=ZR.mean(0,keepdims=True);A=XC-xm;Y=ZR-ym
W=np.linalg.solve(A.T@A+10*np.eye(A.shape[1]),A.T@Y);P=normalize((QC-xm)@W+ym);ZI=normalize(ZR)
CS=(Q@X.T).tocsr();PS=P@ZI.T

# masked reasoning-point TFIDF oracle ONLY to construct train-only utility preferences
ref_points=[mask_reason_points(r,d) for r,d in zip(ref.diagnostic_reasoning,ref.final_diagnosis)]
mq_points=[mask_reason_points(r,d) for r,d in zip(mq.diagnostic_reasoning,mq.final_diagnosis)]
allpts=[p for ps in ref_points for p in ps]+[p for ps in mq_points for p in ps]
pv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
PM=pv.fit_transform(allpts)
off=0;ref_pm=[]
for ps in ref_points:ref_pm.append(PM[off:off+len(ps)]);off+=len(ps)
mq_pm=[]
for ps in mq_points:mq_pm.append(PM[off:off+len(ps)]);off+=len(ps)
def coverage(qmat,cmat):
 S=(qmat@cmat.T).toarray()
 return float(np.max(S,axis=1).mean()) if S.size else 0.0

D=[];YY=[];usable=0;pair_count=0
for i in range(len(mq)):
 s=CS.getrow(i).toarray().ravel();cand=rank(s,40);sem=s[cand];prd=PS[i,cand]
 util=np.array([coverage(mq_pm[i],ref_pm[int(j)]) for j in cand])
 feats=make_features(sem,prd,np.arange(len(cand)),[len(ref_points[int(j)]) for j in cand])
 # Preference pairs: top utility vs bottom utility, but require nontrivial utility gap.
 hi=np.argsort(-util)[:5];lo=np.argsort(util)[:10]
 local=0
 for p in hi:
  for n in lo:
   if util[p]-util[n] < .035:continue
   d=feats[p]-feats[n];D.append(d);YY.append(1);D.append(-d);YY.append(0);local+=1
 if local:usable+=1;pair_count+=local
D=np.asarray(D);YY=np.asarray(YY)
sc=StandardScaler();Ds=sc.fit_transform(D);clf=LogisticRegression(C=1.0,max_iter=2000,random_state=SEED).fit(Ds,YY)
w=clf.coef_[0]/sc.scale_

# Refit representation on ALL train, use validation only as query at inference.
cv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X2=cv2.fit_transform(tr.case_prompt.astype(str));Q2=cv2.transform(va.case_prompt.astype(str))
cs2=TruncatedSVD(256,n_iter=7,random_state=SEED);XC2=cs2.fit_transform(X2).astype(float);QC2=cs2.transform(Q2).astype(float)
tr_rdoc=[" ".join(mask_reason_points(r,d)) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
rv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R2=rv2.fit_transform(tr_rdoc);rs2=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR2=rs2.fit_transform(R2).astype(float)
xm=XC2.mean(0,keepdims=True);ym=ZR2.mean(0,keepdims=True);AA=XC2-xm;YYR=ZR2-ym
WW=np.linalg.solve(AA.T@AA+10*np.eye(AA.shape[1]),AA.T@YYR);PP=normalize((QC2-xm)@WW+ym);ZZ=normalize(ZR2)
CSS=(Q2@X2.T).tocsr();PSS=PP@ZZ.T
base=[];rer=[]
tr_counts=[len(mask_reason_points(r,d)) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
for i in range(len(va)):
 s=CSS.getrow(i).toarray().ravel();cand=rank(s,50);base.append(cand.tolist())
 F=make_features(s[cand],PSS[i,cand],np.arange(len(cand)),[tr_counts[int(j)] for j in cand])
 score=F@w;ids=np.asarray(cand);o=np.lexsort((ids,-score));rer.append(ids[o].tolist())

tl=[nw(x) for x in tr.final_diagnosis];vl=[nw(x) for x in va.final_diagnosis];mp=defaultdict(list)
for i,y in enumerate(tl):mp[y].append(i)
paired={}
for k in (1,3,10):
 a=np.array([any(tl[j]==vl[i] for j in base[i][:k]) for i in range(len(va))]);b=np.array([any(tl[j]==vl[i] for j in rer[i][:k]) for i in range(len(va))])
 h=int(np.sum((~a)&b));m=int(np.sum(a&(~b)));paired[str(k)]={"baseline":int(a.sum()),"C2":int(b.sum()),"helped":h,"harmed":m,"p":float(binomtest(min(h,m),h+m,.5).pvalue) if h+m else 1.0}
summary={"method":"C2 train-only reasoning-utility pairwise ranker",
 "utility_supervision":"masked diagnostic-reasoning point TF-IDF coverage, train meta-split only",
 "training":{"meta_reference":len(ref),"meta_queries":len(mq),"usable_queries":usable,"preference_pairs":pair_count,"symmetric_rows":len(YY),
             "utility_margin":0.035,"feature_weights":dict(zip(["semantic","predicted_reasoning","interaction","reciprocal_rank","log_reason_count"],[float(x) for x in w]))},
 "deployment":{"new_query_inputs":"case_prompt only","historical_library_fields":"case_prompt + diagnosis-masked diagnostic_reasoning"},
 "validation_exact_label_secondary":[exact_metrics("tfidf",base,tl,vl,mp),exact_metrics("C2",rer,tl,vl,mp)],
 "paired_exact_label_secondary":paired,
 "limits":["Primary intended endpoint is independent reasoning coverage (C1), not exact diagnosis equality.","Train utility labels use TF-IDF reasoning-point coverage; independent dense evaluation is required.","Diagnosis masking removes exact normalized phrase only.","No validation-label optimization and no test access."]}
(OUT/"E020_C2_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
with (OUT/"E020_C2_RANKINGS.jsonl").open("w") as f:
 for i,r in enumerate(rer):f.write(json.dumps({"query_index":i,"baseline_top50":base[i],"C2_top50":r})+"\n")
print(json.dumps(summary,indent=2))
