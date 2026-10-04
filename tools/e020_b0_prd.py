import json,re,unicodedata,pathlib,hashlib
from collections import defaultdict
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e020-b0-output");OUT.mkdir(exist_ok=True)
SEED=20261004
def nl(s):
 s=unicodedata.normalize("NFKC",str(s)).casefold()
 return " ".join(re.findall(r"\w+",s))
def rank(scores,k=50):
 idx=np.arange(len(scores));return np.lexsort((idx,-scores))[:k]
def metrics(name,ranks,tl,vl,mp):
 elig=[i for i,y in enumerate(vl) if y in mp];o={"method":name,"eligible":len(elig)}
 for k in (1,3,10,50):
  a=sum(any(tl[j]==vl[i] for j in ranks[i][:k]) for i in range(len(vl)))
  e=sum(any(tl[j]==vl[i] for j in ranks[i][:k]) for i in elig)
  o[f"hit{k}_all"]=a;o[f"hit{k}_all_rate"]=a/len(vl);o[f"hit{k}_eligible"]=e;o[f"hit{k}_eligible_rate"]=e/len(elig)
 return o
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]]
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]]
tl=[nl(x) for x in tr.final_diagnosis];vl=[nl(x) for x in va.final_diagnosis];mp=defaultdict(list)
for i,y in enumerate(tl):mp[y].append(i)
# fixed representations, no validation tuning
case_vec=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=case_vec.fit_transform(tr.case_prompt.astype(str));Q=case_vec.transform(va.case_prompt.astype(str))
case_svd=TruncatedSVD(n_components=256,n_iter=7,random_state=SEED)
XC=case_svd.fit_transform(X).astype(np.float64);QC=case_svd.transform(Q).astype(np.float64)
reason_vec=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R=reason_vec.fit_transform(tr.diagnostic_reasoning.astype(str));RV=reason_vec.transform(va.diagnostic_reasoning.astype(str))
reason_svd=TruncatedSVD(n_components=128,n_iter=7,random_state=SEED)
ZR=reason_svd.fit_transform(R).astype(np.float64);ZRV=reason_svd.transform(RV).astype(np.float64)
# centered ridge map case-LSA -> reasoning-LSA
alpha=10.0
xm=XC.mean(0,keepdims=True);ym=ZR.mean(0,keepdims=True)
A=XC-xm;Y=ZR-ym
W=np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]),A.T@Y)
pred=(QC-xm)@W+ym
# index uses observed training reasoning; query uses predicted reasoning only
ZI=normalize(ZR);PQ=normalize(pred);VT=normalize(ZRV)
S=PQ@ZI.T
ranks=[rank(S[i],50).tolist() for i in range(len(va))]
# lexical deterministic comparison, same case representation
LX=normalize(X);LQ=normalize(Q);LS=(LQ@LX.T).tocsr()
lex=[]
for i in range(len(va)):lex.append(rank(LS.getrow(i).toarray().ravel(),50).tolist())
# privileged diagnostic: actual validation reasoning only for post-hoc evaluation
oracleS=VT@ZI.T
def reason_quality(rankings,k):
 vals=[]
 for i,r in enumerate(rankings):
  ids=r[:k];vals.append(float(np.mean(oracleS[i,ids])))
 return float(np.mean(vals))
def max_reason_quality(rankings,k):
 vals=[]
 for i,r in enumerate(rankings):
  ids=r[:k];vals.append(float(np.max(oracleS[i,ids])))
 return float(np.mean(vals))
summary={"method":"Privileged Reasoning Distillation prototype B0",
 "dataset_revision":REV,"seed":SEED,"case_lsa_dim":256,"reason_lsa_dim":128,"ridge_alpha":alpha,
 "query_features_at_inference":"case_prompt only","reference_index":"training diagnostic_reasoning latent vectors",
 "metrics":[metrics("tfidf_case",lex,tl,vl,mp),metrics("prd_b0",ranks,tl,vl,mp)],
 "posthoc_reasoning_similarity":{
  "tfidf_top1_mean":reason_quality(lex,1),"prd_top1_mean":reason_quality(ranks,1),
  "tfidf_top3_mean":reason_quality(lex,3),"prd_top3_mean":reason_quality(ranks,3),
  "tfidf_top10_max_mean":max_reason_quality(lex,10),"prd_top10_max_mean":max_reason_quality(ranks,10)},
 "limitations":[
  "Validation diagnostic_reasoning is used only for post-hoc evaluation, never to create query embeddings.",
  "Training reasoning often contains diagnosis names; B0 is a privileged-information prototype, not yet a diagnosis-masked sensitivity analysis.",
  "Exact diagnosis equality is a proxy retrieval endpoint, not clinical equivalence.",
  "No hyperparameter tuning was performed and the test split is sealed."
 ]}
(OUT/"E020_B0_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
with (OUT/"E020_B0_TOP50.jsonl").open("w") as f:
 for i,r in enumerate(ranks):f.write(json.dumps({"query_index":i,"rankings":r})+"\n")
print(json.dumps(summary,indent=2))
