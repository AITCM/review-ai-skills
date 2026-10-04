import json,re,unicodedata,pathlib
from collections import defaultdict
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e020-b1-output");OUT.mkdir(exist_ok=True);SEED=20261004
def nl(s):
 s=unicodedata.normalize("NFKC",str(s)).casefold()
 return " ".join(re.findall(r"\w+",s))
def rank(scores,k=50):
 idx=np.arange(len(scores));return np.lexsort((idx,-scores))[:k]
def met(name,ranks,tl,vl,mp):
 elig=[i for i,y in enumerate(vl) if y in mp];o={"method":name,"eligible":len(elig)}
 for k in (1,3,10,50):
  a=sum(any(tl[j]==vl[i] for j in ranks[i][:k]) for i in range(len(vl)))
  e=sum(any(tl[j]==vl[i] for j in ranks[i][:k]) for i in elig)
  o[f"hit{k}_all"]=a;o[f"hit{k}_all_rate"]=a/len(vl);o[f"hit{k}_eligible"]=e;o[f"hit{k}_eligible_rate"]=e/len(elig)
 return o
def minmax(x):
 x=np.asarray(x,float);lo=x.min();hi=x.max()
 return np.zeros_like(x) if hi<=lo else (x-lo)/(hi-lo)
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["case_prompt","diagnostic_reasoning","final_diagnosis"]]
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["case_prompt","diagnostic_reasoning","final_diagnosis"]]
tl=[nl(x) for x in tr.final_diagnosis];vl=[nl(x) for x in va.final_diagnosis];mp=defaultdict(list)
for i,y in enumerate(tl):mp[y].append(i)
# case view
cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=cv.fit_transform(tr.case_prompt.astype(str));Q=cv.transform(va.case_prompt.astype(str));CS=(Q@X.T).tocsr()
case_rank=[];case_scores=[]
for i in range(len(va)):
 s=CS.getrow(i).toarray().ravel();r=rank(s,50);case_rank.append(r.tolist());case_scores.append(s[r])
# privileged reasoning projection
case_svd=TruncatedSVD(256,n_iter=7,random_state=SEED);XC=case_svd.fit_transform(X).astype(np.float64);QC=case_svd.transform(Q).astype(np.float64)
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R=rv.fit_transform(tr.diagnostic_reasoning.astype(str));RV=rv.transform(va.diagnostic_reasoning.astype(str))
rsvd=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR=rsvd.fit_transform(R).astype(np.float64);ZRV=rsvd.transform(RV).astype(np.float64)
xm=XC.mean(0,keepdims=True);ym=ZR.mean(0,keepdims=True);A=XC-xm;Y=ZR-ym
W=np.linalg.solve(A.T@A+10*np.eye(A.shape[1]),A.T@Y)
PQ=normalize((QC-xm)@W+ym);ZI=normalize(ZR);VT=normalize(ZRV);PS=PQ@ZI.T;TRUE=VT@ZI.T
alphas=[0.0,0.25,0.5,0.75,1.0];res=[];allr={}
for a in alphas:
 ranks=[]
 for i,cands in enumerate(case_rank):
  c=np.array(cands,dtype=int);lex=minmax(case_scores[i]);prd=minmax(PS[i,c]);score=a*lex+(1-a)*prd
  order=np.lexsort((c,-score));ranks.append(c[order].tolist())
 m=met(f"semantic_weight_{a:.2f}",ranks,tl,vl,mp)
 m["true_reasoning_top1_mean"]=float(np.mean([TRUE[i,r[0]] for i,r in enumerate(ranks)]))
 m["true_reasoning_top3_mean"]=float(np.mean([np.mean(TRUE[i,r[:3]]) for i,r in enumerate(ranks)]))
 m["true_reasoning_top10_max_mean"]=float(np.mean([np.max(TRUE[i,r[:10]]) for i,r in enumerate(ranks)]))
 res.append(m);allr[str(a)]=ranks
summary={"method":"two-stage semantic candidate generation + privileged-reasoning reranking","candidate_pool_k":50,"semantic_weight_grid":alphas,
"parameters_fixed_except_reported_weight_grid":{"case_lsa":256,"reason_lsa":128,"ridge_alpha":10.0,"seed":SEED},
"results":res,
"limitations":["Weight grid is exploratory on validation and must not be evaluated as a locked test-set choice.","Exact diagnosis equality is a proxy.","Validation reasoning is used only for post-hoc quality metrics, not reranking.","Training reasoning may contain diagnosis names."]}
(OUT/"E020_B1_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
with (OUT/"E020_B1_RANKINGS.jsonl").open("w") as f:
 for i in range(len(va)):f.write(json.dumps({"query_index":i,**{a:allr[a][i] for a in allr}})+"\n")
print(json.dumps(summary,indent=2))
