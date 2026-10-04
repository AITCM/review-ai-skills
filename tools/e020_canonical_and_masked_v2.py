from __future__ import annotations
import json,re,unicodedata,pathlib
from collections import defaultdict
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from scipy.stats import spearmanr
from rank_bm25 import BM25Okapi
REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e020-canonical-v2-output");OUT.mkdir(exist_ok=True)
TOKEN_RE=re.compile(r"(?u)\b\w\w+\b")
def tok(s):return TOKEN_RE.findall(unicodedata.normalize("NFKC",str(s)).casefold())
def norm_label(s):return " ".join(re.findall(r"\w+",unicodedata.normalize("NFKC",str(s)).casefold()))
def norm_text(s):return " ".join(tok(s))
def mask_reason(r,d):
 rt=tok(r);dt=tok(d);out=[];i=0
 while i<len(rt):
  if dt and rt[i:i+len(dt)]==dt:out.append("diagnosismask");i+=len(dt)
  else:out.append(rt[i]);i+=1
 return " ".join(out)
def rank(scores,k=50):
 ids=np.arange(len(scores),dtype=np.int64);return np.lexsort((ids,-scores))[:k]
def met(name,rr,tl,vl,mp):
 elig=[i for i,y in enumerate(vl) if y in mp];o={"method":name,"eligible":len(elig)}
 for k in (1,3,10,50):
  a=sum(any(tl[j]==vl[i] for j in rr[i][:k]) for i in range(len(vl)))
  e=sum(any(tl[j]==vl[i] for j in rr[i][:k]) for i in elig)
  o[f"hit{k}_all"]=a;o[f"hit{k}_all_rate"]=a/len(vl);o[f"hit{k}_eligible"]=e;o[f"hit{k}_eligible_rate"]=e/len(elig)
 return o
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["case_prompt","diagnostic_reasoning","final_diagnosis"]]
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["case_prompt","diagnostic_reasoning","final_diagnosis"]]
tl=[norm_label(x) for x in tr.final_diagnosis];vl=[norm_label(x) for x in va.final_diagnosis];mp=defaultdict(list)
for i,y in enumerate(tl):mp[y].append(i)
elig=[i for i,y in enumerate(vl) if y in mp]
# deterministic TFIDF
tv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=tv.fit_transform(tr.case_prompt.astype(str));Q=tv.transform(va.case_prompt.astype(str));TR=[]
for i in range(len(va)):TR.append(rank((Q[i]@X.T).toarray().ravel()).tolist())
# canonical BM25Okapi: normalize strings first, then CountVectorizer and rank_bm25 see identical tokens.
train_norm=[norm_text(x) for x in tr.case_prompt];val_norm=[norm_text(x) for x in va.case_prompt]
cv=CountVectorizer(lowercase=False,token_pattern=r"(?u)\b\w\w+\b",dtype=np.float64)
C=cv.fit_transform(train_norm).tocsr();CQ=cv.transform(val_norm).tocsr()
N=C.shape[0];df=np.diff(C.tocsc().indptr).astype(float)
idf=np.log(N-df+0.5)-np.log(df+0.5);avg_idf=float(idf.mean());eps=.25*avg_idf;idf[idf<0]=eps
dl=np.asarray(C.sum(axis=1)).ravel();avgdl=float(dl.mean());k1=1.5;b=.75
B=C.copy().astype(float)
for i in range(N):
 s,e=B.indptr[i],B.indptr[i+1];f=B.data[s:e]
 B.data[s:e]=idf[B.indices[s:e]]*(f*(k1+1)/(f+k1*(1-b+b*dl[i]/avgdl)))
M=(CQ@B.T).tocsr();BR=[]
for i in range(len(va)):BR.append(rank(M.getrow(i).toarray().ravel()).tolist())
# exact reference validation
corpus=[x.split() for x in train_norm];ref=BM25Okapi(corpus,k1=1.5,b=.75,epsilon=.25)
checks=[]
for i in list(range(10))+list(range(490,500)):
 a=M.getrow(i).toarray().ravel();z=np.asarray(ref.get_scores(val_norm[i].split()),float)
 ar=rank(a,50);zr=rank(z,50)
 checks.append({"q":i,"max_abs_diff":float(np.max(np.abs(a-z))),"top50_identical":bool(np.array_equal(ar,zr))})
if not all(x["max_abs_diff"]<1e-6 and x["top50_identical"] for x in checks):
 raise RuntimeError("BM25 reference mismatch: "+json.dumps(checks))
# diagnosis-masked reasoning landscape
trm=[mask_reason(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
vam=[mask_reason(r,d) for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=False,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
RX=rv.fit_transform(trm);RQ=rv.transform(vam)
rranks=[];corrs=[];rsims=[]
for i in elig:
 same=np.array(mp[vl[i]]);rs=(RQ[i]@RX[same].T).toarray().ravel();br=int(same[int(np.argmax(rs))]);rsims.append(float(rs.max()))
 cs=(Q[i]@X.T).toarray().ravel();rranks.append(1+int(np.sum(cs>cs[br])))
 top=np.array(TR[i][:10]);a=cs[top];z=(RQ[i]@RX[top].T).toarray().ravel()
 if np.std(a)>0 and np.std(z)>0:
  r=float(spearmanr(a,z).statistic)
  if np.isfinite(r):corrs.append(r)
summary={"dataset_revision":REV,"canonical_metrics":[met("tfidf",TR,tl,vl,mp),met("bm25_okapi_rank_bm25",BR,tl,vl,mp)],
 "bm25_reference":{"rank_bm25_version":"0.2.2","checked_queries":20,"all_top50_identical":True,"max_abs_diff":max(x["max_abs_diff"] for x in checks)},
 "diagnosis_masked_reasoning":{"eligible":len(elig),"median_reasoning_aligned_case_text_rank":float(np.median(rranks)),
 "p25":float(np.percentile(rranks,25)),"p75":float(np.percentile(rranks,75)),"top10_rate":float(np.mean(np.array(rranks)<=10)),
 "median_masked_reasoning_similarity":float(np.median(rsims)),"median_top10_case_vs_masked_reasoning_spearman":float(np.median(corrs)),
 "mask_note":"exact normalized diagnosis token sequence removed; aliases/acronyms can remain"},
 "limits":["Exact diagnosis equality is a proxy, not clinical equivalence.","Validation reasoning is privileged post-hoc analysis only.","Test split is not used."]}
(OUT/"E020_CANONICAL_V2_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
(OUT/"BM25_REFERENCE_CHECK.json").write_text(json.dumps(checks,indent=2)+"\n")
print(json.dumps(summary,indent=2))
