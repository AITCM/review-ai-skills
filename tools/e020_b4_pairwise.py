from __future__ import annotations
import json,re,unicodedata,pathlib,hashlib
from collections import defaultdict
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import binomtest
REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e020-b4-output");OUT.mkdir(exist_ok=True);SEED=20261004
def nw(s):return " ".join(re.findall(r"\w+",unicodedata.normalize("NFKC",str(s)).casefold()))
def mask(r,d):
 rr=nw(r);dd=nw(d);return rr.replace(dd," diagnosismask ") if dd and dd in rr else rr
def rank(s,k=50):
 ids=np.arange(len(s));return np.lexsort((ids,-s))[:k]
def feat(sem,prd,pos):
 sem=np.asarray(sem,float);prd=np.asarray(prd,float);rr=1/(1+np.asarray(pos,float))
 return np.column_stack([sem,prd,sem*prd,rr,sem-prd])
def met(name,ranks,tl,vl,mp):
 elig=[i for i,y in enumerate(vl) if y in mp];o={"method":name,"eligible":len(elig)}
 for k in (1,3,10,50):
  a=sum(any(tl[j]==vl[i] for j in ranks[i][:k]) for i in range(len(vl)))
  e=sum(any(tl[j]==vl[i] for j in ranks[i][:k]) for i in elig)
  o[f"hit{k}_all"]=a;o[f"hit{k}_all_rate"]=a/len(vl);o[f"hit{k}_eligible"]=e;o[f"hit{k}_eligible_rate"]=e/len(elig)
 return o
def reprs(ref,q):
 cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
 X=cv.fit_transform(ref.case_prompt.astype(str));Q=cv.transform(q.case_prompt.astype(str))
 cs=TruncatedSVD(256,n_iter=7,random_state=SEED);XC=cs.fit_transform(X).astype(float);QC=cs.transform(Q).astype(float)
 rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
 R=rv.fit_transform([mask(r,d) for r,d in zip(ref.diagnostic_reasoning,ref.final_diagnosis)])
 rs=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR=rs.fit_transform(R).astype(float)
 xm=XC.mean(0,keepdims=True);ym=ZR.mean(0,keepdims=True);A=XC-xm;Y=ZR-ym
 W=np.linalg.solve(A.T@A+10*np.eye(A.shape[1]),A.T@Y)
 P=normalize((QC-xm)@W+ym);ZI=normalize(ZR)
 return X,Q,P,ZI
def cands(ref,q,X,Q,P,ZI,k=50):
 CS=(Q@X.T).tocsr();PS=P@ZI.T;rr=[];ss=[];pp=[]
 for i in range(len(q)):
  s=CS.getrow(i).toarray().ravel();r=rank(s,k);rr.append(r.tolist());ss.append(s[r]);pp.append(PS[i,r])
 return rr,ss,pp
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
isdev=np.array([int(hashlib.sha256(("E020-B4:"+str(x)).encode()).hexdigest()[:8],16)%5==0 for x in tr.pmcid])
ref=tr.loc[~isdev].reset_index(drop=True);q=tr.loc[isdev].reset_index(drop=True)
rl=[nw(x) for x in ref.final_diagnosis];ql=[nw(x) for x in q.final_diagnosis]
X,Q,P,ZI=reprs(ref,q);cr,sv,pv=cands(ref,q,X,Q,P,ZI,50)
D=[];Y=[];groups=0;positive_pairs=0
for i,c in enumerate(cr):
 labs=np.array([rl[j] for j in c],object);pos=np.where(labs==ql[i])[0];neg=np.where(labs!=ql[i])[0]
 if not len(pos) or not len(neg):continue
 groups+=1
 F=feat(sv[i],pv[i],np.arange(len(c)))
 # hardest positive = lowest lexical among first three positives if available; hard negatives = top 5 lexical mismatches
 chosen_pos=pos[:min(3,len(pos))];chosen_neg=neg[:5]
 for p in chosen_pos:
  for n in chosen_neg:
   diff=F[p]-F[n];D.append(diff);Y.append(1);D.append(-diff);Y.append(0);positive_pairs+=1
D=np.asarray(D);Y=np.asarray(Y)
sc=StandardScaler();Ds=sc.fit_transform(D);clf=LogisticRegression(C=1,max_iter=2000,random_state=SEED).fit(Ds,Y)
# raw candidate scoring direction; centering/intercept do not affect within-query ordering
w=clf.coef_[0]/sc.scale_
# full train -> validation
tl=[nw(x) for x in tr.final_diagnosis];vl=[nw(x) for x in va.final_diagnosis];mp=defaultdict(list)
for i,y in enumerate(tl):mp[y].append(i)
X,Q,P,ZI=reprs(tr,va);base,sv,pv=cands(tr,va,X,Q,P,ZI,50)
rer=[]
for i,c in enumerate(base):
 F=feat(sv[i],pv[i],np.arange(len(c)));score=F@w;ids=np.asarray(c);o=np.lexsort((ids,-score));rer.append(ids[o].tolist())
paired={}
for k in (1,3,10):
 a=np.array([any(tl[j]==vl[i] for j in base[i][:k]) for i in range(len(va))])
 b=np.array([any(tl[j]==vl[i] for j in rer[i][:k]) for i in range(len(va))])
 h=int(np.sum((~a)&b));m=int(np.sum(a&(~b)));paired[str(k)]={"baseline":int(a.sum()),"reranked":int(b.sum()),"helped":h,"harmed":m,
 "p":float(binomtest(min(h,m),h+m,.5).pvalue) if h+m else 1.0}
# masked reasoning posthoc
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R=rv.fit_transform([mask(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]);RV=rv.transform([mask(r,d) for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)])
rs=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR=normalize(rs.fit_transform(R).astype(float));ZV=normalize(rs.transform(RV).astype(float));TRUE=ZV@ZR.T
rq=lambda rr,k:float(np.mean([np.mean(TRUE[i,r[:k]]) for i,r in enumerate(rr)]))
summary={"method":"B4 pairwise hard-negative utility ranker","training":{"meta_ref":len(ref),"meta_queries":len(q),"usable_query_groups":groups,
"pair_comparisons":positive_pairs,"symmetric_rows":len(Y),"raw_feature_weights":dict(zip(["semantic","predicted_masked_reasoning","interaction","reciprocal_rank","semantic_minus_reasoning"],[float(x) for x in w]))},
"validation":[met("tfidf",base,tl,vl,mp),met("B4_pairwise",rer,tl,vl,mp)],"paired":paired,
"masked_reasoning":{"tfidf_top1":rq(base,1),"B4_top1":rq(rer,1),"tfidf_top3":rq(base,3),"B4_top3":rq(rer,3)},
"limits":["Exact diagnosis equality is proxy pairwise supervision.","Historical candidates use diagnosis-masked reasoning; new query uses case text only.","No validation-label tuning; test remains sealed."]}
(OUT/"E020_B4_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
with (OUT/"E020_B4_RANKINGS.jsonl").open("w") as f:
 for i,r in enumerate(rer):f.write(json.dumps({"query_index":i,"baseline":base[i],"B4":r})+"\n")
print(json.dumps(summary,indent=2))
