from __future__ import annotations
import json,re,unicodedata,pathlib,hashlib
from collections import defaultdict
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from scipy.stats import binomtest

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e020-b3-output");OUT.mkdir(exist_ok=True);SEED=20261004
def nw(s):return " ".join(re.findall(r"\w+",unicodedata.normalize("NFKC",str(s)).casefold()))
def mask(r,d):
 rr=nw(r);dd=nw(d);return rr.replace(dd," diagnosismask ") if dd and dd in rr else rr
def rank(s,k=50):
 idx=np.arange(len(s));return np.lexsort((idx,-s))[:k]
def features(sem,prd,rankpos):
 sem=np.asarray(sem,float);prd=np.asarray(prd,float);rr=1.0/(1.0+np.asarray(rankpos,float))
 return np.column_stack([sem,prd,sem*prd,rr])
def metrics(name,ranks,tl,vl,mp):
 elig=[i for i,y in enumerate(vl) if y in mp];o={"method":name,"eligible":len(elig)}
 for k in (1,3,10,50):
  a=sum(any(tl[j]==vl[i] for j in ranks[i][:k]) for i in range(len(vl)))
  e=sum(any(tl[j]==vl[i] for j in ranks[i][:k]) for i in elig)
  o[f"hit{k}_all"]=a;o[f"hit{k}_all_rate"]=a/len(vl);o[f"hit{k}_eligible"]=e;o[f"hit{k}_eligible_rate"]=e/len(elig)
 return o
def fit_repr(ref,query):
 cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
 X=cv.fit_transform(ref.case_prompt.astype(str));Q=cv.transform(query.case_prompt.astype(str))
 cs=TruncatedSVD(256,n_iter=7,random_state=SEED);XC=cs.fit_transform(X).astype(float);QC=cs.transform(Q).astype(float)
 rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
 R=rv.fit_transform([mask(r,d) for r,d in zip(ref.diagnostic_reasoning,ref.final_diagnosis)])
 rs=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR=rs.fit_transform(R).astype(float)
 xm=XC.mean(0,keepdims=True);ym=ZR.mean(0,keepdims=True);A=XC-xm;Y=ZR-ym
 W=np.linalg.solve(A.T@A+10*np.eye(A.shape[1]),A.T@Y)
 P=normalize((QC-xm)@W+ym);ZI=normalize(ZR)
 return X,Q,P,ZI
def candidate_data(ref,queries,X,Q,P,ZI,k=50):
 CS=(Q@X.T).tocsr();PS=P@ZI.T
 ranks=[];semvals=[];prdvals=[]
 for i in range(len(queries)):
  s=CS.getrow(i).toarray().ravel();r=rank(s,k)
  ranks.append(r.tolist());semvals.append(s[r]);prdvals.append(PS[i,r])
 return ranks,semvals,prdvals
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
# fixed hash meta split: 80% representation/reference pool, 20% reranker-training queries
is_dev=np.array([int(hashlib.sha256(("E020-B3:"+str(x)).encode()).hexdigest()[:8],16)%5==0 for x in tr.pmcid])
meta_ref=tr.loc[~is_dev].reset_index(drop=True);meta_q=tr.loc[is_dev].reset_index(drop=True)
ref_labels=[nw(x) for x in meta_ref.final_diagnosis];q_labels=[nw(x) for x in meta_q.final_diagnosis]
X,Q,P,ZI=fit_repr(meta_ref,meta_q);cr,sv,pv=candidate_data(meta_ref,meta_q,X,Q,P,ZI,50)
FX=[];FY=[];groups_with_positive=0
for i,c in enumerate(cr):
 y=q_labels[i];labs=[ref_labels[j] for j in c];yy=np.array([int(z==y) for z in labs])
 if yy.sum()>0:groups_with_positive+=1
 # keep all positives plus first 12 hard negatives to control imbalance while preserving hard cases
 keep=list(np.where(yy==1)[0])+list(np.where(yy==0)[0][:12])
 if not keep:continue
 cidx=np.array(keep);FX.append(features(np.array(sv[i])[cidx],np.array(pv[i])[cidx],cidx));FY.extend(yy[cidx].tolist())
FX=np.vstack(FX);FY=np.asarray(FY)
clf=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,class_weight="balanced",max_iter=2000,random_state=SEED))
clf.fit(FX,FY)
# refit representations on complete training data, then validation reranking
tl=[nw(x) for x in tr.final_diagnosis];vl=[nw(x) for x in va.final_diagnosis];mp=defaultdict(list)
for i,y in enumerate(tl):mp[y].append(i)
X,Q,P,ZI=fit_repr(tr,va);base,sv,pv=candidate_data(tr,va,X,Q,P,ZI,50)
rer=[]
for i,c in enumerate(base):
 f=features(sv[i],pv[i],np.arange(len(c)));prob=clf.predict_proba(f)[:,1];ids=np.asarray(c);o=np.lexsort((ids,-prob));rer.append(ids[o].tolist())
# posthoc masked reasoning evaluation, validation reasoning never used to rank
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R=rv.fit_transform([mask(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)])
RV=rv.transform([mask(r,d) for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)])
rs=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR=normalize(rs.fit_transform(R).astype(float));ZV=normalize(rs.transform(RV).astype(float));TRUE=ZV@ZR.T
def rq(ranks,k=1):return float(np.mean([np.mean(TRUE[i,r[:k]]) for i,r in enumerate(ranks)]))
paired={}
for k in (1,3,10):
 a=np.array([any(tl[j]==vl[i] for j in base[i][:k]) for i in range(len(va))])
 b=np.array([any(tl[j]==vl[i] for j in rer[i][:k]) for i in range(len(va))])
 h=int(np.sum((~a)&b));m=int(np.sum(a&(~b)))
 paired[str(k)]={"baseline":int(a.sum()),"reranked":int(b.sum()),"helped":h,"harmed":m,
  "paired_binomial_p":float(binomtest(min(h,m),h+m,.5).pvalue) if h+m else 1.0}
summary={"method":"B3 learned clinical-case utility reranker",
 "training":{"meta_reference_rows":len(meta_ref),"meta_query_rows":len(meta_q),"meta_queries_with_positive_in_top50":groups_with_positive,
             "pair_rows":len(FY),"positive_pairs":int(FY.sum()),"features":["case_tfidf_similarity","predicted_masked_reasoning_similarity","interaction","reciprocal_semantic_rank"],
             "model":"StandardScaler + balanced LogisticRegression(C=1)"},
 "deployment_query_inputs":"case_prompt only",
 "candidate_side_information":"historical candidate case_prompt + diagnosis-masked diagnostic_reasoning index",
 "validation":[metrics("tfidf_candidate_order",base,tl,vl,mp),metrics("B3_learned_rerank",rer,tl,vl,mp)],
 "paired_exact_label_transitions":paired,
 "masked_reasoning_similarity":{"baseline_top1_mean":rq(base,1),"B3_top1_mean":rq(rer,1),"baseline_top3_mean":rq(base,3),"B3_top3_mean":rq(rer,3)},
 "limits":["Exact diagnosis equality is proxy supervision/evaluation, not clinical equivalence.","Candidate reasoning is available on the historical evidence side; the new query uses case text only.","Exact diagnosis phrase masking does not remove aliases/acronyms.","No validation-label tuning and no test-set use."]}
(OUT/"E020_B3_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
with (OUT/"E020_B3_RANKINGS.jsonl").open("w") as f:
 for i,r in enumerate(rer):f.write(json.dumps({"query_index":i,"baseline_top50":base[i],"B3_top50":r})+"\n")
print(json.dumps(summary,indent=2))
