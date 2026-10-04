import json,re,unicodedata,pathlib,math
from collections import Counter,defaultdict
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e020-a1d-output");OUT.mkdir(exist_ok=True)
def nl(s):
 s=unicodedata.normalize("NFKC",str(s)).casefold()
 return " ".join(re.findall(r"\w+",s))
def deterministic_rank(scores,k=50):
 idx=np.arange(len(scores),dtype=np.int64)
 return np.lexsort((idx,-scores))[:k]
def ev(name,ranks,tl,vl,mp):
 eligible=[i for i,y in enumerate(vl) if y in mp];o={"method":name,"eligible":len(eligible)}
 for k in (1,3,10,50):
  a=sum(any(tl[j]==vl[i] for j in ranks[i][:k]) for i in range(len(vl)))
  e=sum(any(tl[j]==vl[i] for j in ranks[i][:k]) for i in eligible)
  o[f"hit{k}_all"]=a;o[f"hit{k}_all_rate"]=a/len(vl);o[f"hit{k}_eligible"]=e;o[f"hit{k}_eligible_rate"]=e/len(eligible)
 return o
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["case_prompt","final_diagnosis"]]
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["case_prompt","final_diagnosis"]]
tl=[nl(x) for x in tr.final_diagnosis];vl=[nl(x) for x in va.final_diagnosis];mp=defaultdict(list)
for i,y in enumerate(tl):mp[y].append(i)
# Canonical TFIDF deterministic full sort
v=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=v.fit_transform(tr.case_prompt.astype(str));Q=v.transform(va.case_prompt.astype(str))
tranks=[]
for i in range(len(va)):
 s=(Q[i]@X.T).toarray().ravel();tranks.append(deterministic_rank(s).tolist())
# Canonical sparse BM25 matching reference regex token rule, no vocab cap
cv=CountVectorizer(lowercase=True,token_pattern=r"(?u)\b\w+\b",dtype=np.float32)
C=cv.fit_transform(tr.case_prompt.astype(str)).tocsr();CQ=cv.transform(va.case_prompt.astype(str)).tocsr()
N=C.shape[0];df=np.diff(C.tocsc().indptr).astype(np.float32);idf=np.log1p((N-df+.5)/(df+.5)).astype(np.float32)
dl=np.asarray(C.sum(axis=1)).ravel().astype(np.float32);avg=float(dl.mean());k1=1.5;b=.75;base=k1*(1-b+b*dl/avg)
B=C.copy().astype(np.float32)
for i in range(N):
 st,en=B.indptr[i],B.indptr[i+1];vals=B.data[st:en]
 B.data[st:en]=(vals*(k1+1)/(vals+base[i]))*idf[B.indices[st:en]]
QB=CQ.copy().astype(np.float32);QB.data[:]=1
branks=[]
for i in range(len(va)):
 s=(QB[i]@B.T).toarray().ravel();branks.append(deterministic_rank(s).tolist())
out={"dataset_revision":REV,"tie_break":"score descending, training-row index ascending","bm25_token_rule":"lowercase + (?u)\\\\b\\\\w+\\\\b; unique query terms; no vocabulary cap","metrics":[ev("tfidf_canonical",tranks,tl,vl,mp),ev("bm25_canonical",branks,tl,vl,mp)]}
(OUT/"E020_A1D_CANONICAL_SUMMARY.json").write_text(json.dumps(out,indent=2)+"\n")
with (OUT/"E020_A1D_TOP50.jsonl").open("w") as f:
 for i in range(len(va)):f.write(json.dumps({"query_index":i,"tfidf":tranks[i],"bm25":branks[i]})+"\n")
print(json.dumps(out,indent=2))
