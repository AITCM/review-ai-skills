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
OUT=pathlib.Path("e020-c4-output");OUT.mkdir(exist_ok=True);SEED=20261004
PAT=re.compile(r'^\s*(?:\((\d+)\)|(\d+)[.)])\s*',re.M)
SEM_K=50; PRD_K=50; UTILITY_MARGIN=.035

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
def rank_full(scores,k):
 ids=np.arange(len(scores));return np.lexsort((ids,-scores))[:k]
def features(sem,prd,semrank,prdrank,reason_count,sem_route,prd_route):
 sem=np.asarray(sem,float);prd=np.asarray(prd,float)
 sr=np.where(np.asarray(semrank)>=0,1/(1+np.asarray(semrank,float)),0)
 rr=np.where(np.asarray(prdrank)>=0,1/(1+np.asarray(prdrank,float)),0)
 rc=np.log1p(np.asarray(reason_count,float))
 return np.column_stack([sem,prd,sem*prd,sr,rr,rc,np.asarray(sem_route,float),np.asarray(prd_route,float)])
def exact_metrics(name,ranks,tl,vl,mp):
 elig=[i for i,y in enumerate(vl) if y in mp];o={"method":name,"eligible":len(elig)}
 for k in (1,3,10,50):
  a=sum(any(tl[j]==vl[i] for j in ranks[i][:k]) for i in range(len(vl)))
  e=sum(any(tl[j]==vl[i] for j in ranks[i][:k]) for i in elig)
  o[f"hit{k}_all"]=a;o[f"hit{k}_all_rate"]=a/len(vl);o[f"hit{k}_eligible"]=e;o[f"hit{k}_eligible_rate"]=e/len(elig)
 return o
def candidate_union(sem_scores,prd_scores):
 sr=rank_full(sem_scores,SEM_K);rr=rank_full(prd_scores,PRD_K)
 ids=sorted(set(sr.tolist())|set(rr.tolist()))
 spos={int(x):i for i,x in enumerate(sr)};rpos={int(x):i for i,x in enumerate(rr)}
 return np.array(ids,dtype=int),spos,rpos
def coverage(qmat,cmat):
 S=(qmat@cmat.T).toarray()
 return float(np.max(S,axis=1).mean()) if S.size else 0.0

tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()

# deterministic meta split for training the ranker
isdev=np.array([int(hashlib.sha256(("E020-C4:"+str(x)).encode()).hexdigest()[:8],16)%5==0 for x in tr.pmcid])
ref=tr.loc[~isdev].reset_index(drop=True);mq=tr.loc[isdev].reset_index(drop=True)

# Learn case representation and case->masked reasoning projection on meta-reference
cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=cv.fit_transform(ref.case_prompt.astype(str));Q=cv.transform(mq.case_prompt.astype(str))
cS=TruncatedSVD(256,n_iter=7,random_state=SEED);XC=cS.fit_transform(X).astype(float);QC=cS.transform(Q).astype(float)
ref_pts=[mask_points(r,d) for r,d in zip(ref.diagnostic_reasoning,ref.final_diagnosis)]
mq_pts=[mask_points(r,d) for r,d in zip(mq.diagnostic_reasoning,mq.final_diagnosis)]
ref_rdoc=[" ".join(x) for x in ref_pts]
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R=rv.fit_transform(ref_rdoc);rS=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR=rS.fit_transform(R).astype(float)
xm=XC.mean(0,keepdims=True);ym=ZR.mean(0,keepdims=True);A=XC-xm;Y=ZR-ym
W=np.linalg.solve(A.T@A+10*np.eye(A.shape[1]),A.T@Y)
P=normalize((QC-xm)@W+ym);ZI=normalize(ZR)
CS=(Q@X.T).tocsr();PS=P@ZI.T

# Train-only utility oracle from masked reasoning points
allpts=[p for ps in ref_pts for p in ps]+[p for ps in mq_pts for p in ps]
pv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
PM=pv.fit_transform(allpts);off=0;ref_pm=[]
for ps in ref_pts:ref_pm.append(PM[off:off+len(ps)]);off+=len(ps)
mq_pm=[]
for ps in mq_pts:mq_pm.append(PM[off:off+len(ps)]);off+=len(ps)

D=[];YY=[];usable=0;pairs=0;union_sizes=[]
for i in range(len(mq)):
 ss=CS.getrow(i).toarray().ravel();ps=PS[i]
 cand,spos,rpos=candidate_union(ss,ps);union_sizes.append(len(cand))
 sem=ss[cand];prd=ps[cand]
 util=np.array([coverage(mq_pm[i],ref_pm[int(j)]) for j in cand])
 F=features(sem,prd,[spos.get(int(j),-1) for j in cand],[rpos.get(int(j),-1) for j in cand],
            [len(ref_pts[int(j)]) for j in cand],[int(j) in spos for j in cand],[int(j) in rpos for j in cand])
 hi=np.argsort(-util)[:5];lo=np.argsort(util)[:10];local=0
 for p in hi:
  for n in lo:
   if util[p]-util[n]<UTILITY_MARGIN:continue
   d=F[p]-F[n];D.append(d);YY.append(1);D.append(-d);YY.append(0);local+=1
 if local:usable+=1;pairs+=local
D=np.asarray(D);YY=np.asarray(YY)
sc=StandardScaler();Ds=sc.fit_transform(D);clf=LogisticRegression(C=1,max_iter=2000,random_state=SEED).fit(Ds,YY)
w=clf.coef_[0]/sc.scale_

# Refit representations on all training cases
cv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X2=cv2.fit_transform(tr.case_prompt.astype(str));Q2=cv2.transform(va.case_prompt.astype(str))
cS2=TruncatedSVD(256,n_iter=7,random_state=SEED);XC2=cS2.fit_transform(X2).astype(float);QC2=cS2.transform(Q2).astype(float)
tr_pts=[mask_points(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
tr_doc=[" ".join(x) for x in tr_pts]
rv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R2=rv2.fit_transform(tr_doc);rS2=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR2=rS2.fit_transform(R2).astype(float)
xm=XC2.mean(0,keepdims=True);ym=ZR2.mean(0,keepdims=True);AA=XC2-xm;YYR=ZR2-ym
WW=np.linalg.solve(AA.T@AA+10*np.eye(AA.shape[1]),AA.T@YYR)
PP=normalize((QC2-xm)@WW+ym);ZZ=normalize(ZR2);CSS=(Q2@X2.T).tocsr();PSS=PP@ZZ.T

baseline=[];c4=[];union_recall=[];semantic_recall=[];prd_recall=[];val_union_sizes=[]
tl=[nw(x) for x in tr.final_diagnosis];vl=[nw(x) for x in va.final_diagnosis];mp=defaultdict(list)
for i,y in enumerate(tl):mp[y].append(i)
for i in range(len(va)):
 ss=CSS.getrow(i).toarray().ravel();ps=PSS[i]
 semrank=rank_full(ss,50);prdrank=rank_full(ps,50);baseline.append(semrank.tolist())
 cand,spos,rpos=candidate_union(ss,ps);val_union_sizes.append(len(cand))
 F=features(ss[cand],ps[cand],[spos.get(int(j),-1) for j in cand],[rpos.get(int(j),-1) for j in cand],
            [len(tr_pts[int(j)]) for j in cand],[int(j) in spos for j in cand],[int(j) in rpos for j in cand])
 score=F@w;order=np.lexsort((cand,-score));ranked=cand[order].tolist();c4.append(ranked)
 semantic_recall.append(any(tl[j]==vl[i] for j in semrank))
 prd_recall.append(any(tl[j]==vl[i] for j in prdrank))
 union_recall.append(any(tl[j]==vl[i] for j in cand))

paired={}
for k in (1,3,10,50):
 a=np.array([any(tl[j]==vl[i] for j in baseline[i][:k]) for i in range(len(va))])
 b=np.array([any(tl[j]==vl[i] for j in c4[i][:k]) for i in range(len(va))])
 h=int(np.sum((~a)&b));m=int(np.sum(a&(~b)))
 paired[str(k)]={"baseline":int(a.sum()),"C4":int(b.sum()),"helped":h,"harmed":m,
                 "p":float(binomtest(min(h,m),h+m,.5).pvalue) if h+m else 1.0}

summary={
 "method":"C4 dual-route candidate generation + train-only reasoning-utility pairwise ranking",
 "routes":{"semantic_topk":SEM_K,"predicted_reasoning_topk":PRD_K,"union_before_ranking":True},
 "training":{"meta_reference":len(ref),"meta_queries":len(mq),"usable_queries":usable,
             "preference_pairs":pairs,"symmetric_rows":len(YY),"utility_margin":UTILITY_MARGIN,
             "mean_union_size":float(np.mean(union_sizes)),
             "feature_weights":dict(zip(["semantic","predicted_reasoning","interaction","semantic_rr","reasoning_rr","log_reason_count","semantic_route","reasoning_route"],[float(x) for x in w]))},
 "deployment":{"new_query_inputs":"case_prompt only","historical_library_fields":"case_prompt + diagnosis-masked diagnostic_reasoning"},
 "validation_candidate_recall_secondary":{"semantic_top50":int(sum(semantic_recall)),"predicted_reasoning_top50":int(sum(prd_recall)),
   "union_top100_max":int(sum(union_recall)),"mean_union_size":float(np.mean(val_union_sizes))},
 "validation_exact_label_secondary":[exact_metrics("tfidf",baseline,tl,vl,mp),exact_metrics("C4",c4,tl,vl,mp)],
 "paired_exact_label_secondary":paired,
 "limits":["Primary endpoint requires independent reasoning-coverage evaluation; exact diagnosis equality is secondary proxy.",
           "Utility supervision is masked reasoning-point TF-IDF coverage from train meta-split only.",
           "Diagnosis phrase masking does not remove aliases/acronyms.","No validation-label tuning; test remains sealed."]
}
(OUT/"E020_C4_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
with (OUT/"E020_C4_RANKINGS.jsonl").open("w") as f:
 for i,r in enumerate(c4):f.write(json.dumps({"query_index":i,"baseline_top50":baseline[i],"C4_ranked_union":r})+"\n")
print(json.dumps(summary,indent=2))
