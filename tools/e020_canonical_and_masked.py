from __future__ import annotations
import json, re, unicodedata, pathlib, math
from collections import defaultdict
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.stats import spearmanr
from rank_bm25 import BM25Okapi

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e020-canonical-output"); OUT.mkdir(exist_ok=True)

def norm_label(s):
    s=unicodedata.normalize("NFKC",str(s)).casefold()
    return " ".join(re.findall(r"\w+",s))

TOKEN_RE=re.compile(r"(?u)\b\w\w+\b")
def tokenize(s): return TOKEN_RE.findall(unicodedata.normalize("NFKC",str(s)).casefold())

def mask_label(reasoning,label):
    rt=tokenize(reasoning); lt=tokenize(label)
    if not lt: return " ".join(rt)
    out=[]; i=0
    while i<len(rt):
        if rt[i:i+len(lt)]==lt:
            out.extend(["[diagnosis_masked]"])
            i+=len(lt)
        else:
            out.append(rt[i]); i+=1
    return " ".join(out)

def ranks_from_dense(scores,k=50):
    idx=np.lexsort((np.arange(len(scores),dtype=np.int64),-scores))
    return idx[:k].tolist(), scores[idx[:k]].astype(float).tolist()

def evalm(name,rankings,tl,vl,idx):
    eligible=[i for i,y in enumerate(vl) if y in idx]
    d={"method":name,"n":len(vl),"eligible":len(eligible)}
    for k in (1,3,10,50):
        h=sum(any(tl[j]==vl[i] for j in rankings[i][:k]) for i in range(len(vl)))
        e=sum(any(tl[j]==vl[i] for j in rankings[i][:k]) for i in eligible)
        d[f"hit@{k}_all"]=h; d[f"hit@{k}_all_rate"]=h/len(vl)
        d[f"hit@{k}_eligible"]=e; d[f"hit@{k}_eligible_rate"]=e/len(eligible)
    return d

tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
tl=[norm_label(x) for x in tr.final_diagnosis]; vl=[norm_label(x) for x in va.final_diagnosis]
idx=defaultdict(list)
for i,y in enumerate(tl):idx[y].append(i)
eligible=[i for i,y in enumerate(vl) if y in idx]

# Deterministic TF-IDF
tv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=tv.fit_transform(tr.case_prompt.astype(str)); Q=tv.transform(va.case_prompt.astype(str))
tf_rank=[]; tf_score=[]
for i in range(len(va)):
    s=(Q[i]@X.T).toarray().ravel()
    r,v=ranks_from_dense(s,50); tf_rank.append(r); tf_score.append(v)

# Canonical BM25Okapi sparse implementation matching rank_bm25 0.2.2
cv=CountVectorizer(lowercase=True,token_pattern=r"(?u)\b\w\w+\b",dtype=np.float32)
C=cv.fit_transform(tr.case_prompt.astype(str)).tocsr()
CQ=cv.transform(va.case_prompt.astype(str)).tocsr()
N=C.shape[0]; df=np.diff(C.tocsc().indptr).astype(np.float64)
idf=np.log(N-df+0.5)-np.log(df+0.5)
avg_idf=float(idf.mean()); epsilon=0.25*avg_idf
idf[idf<0]=epsilon
dl=np.asarray(C.sum(axis=1)).ravel().astype(np.float64); avgdl=float(dl.mean())
k1=1.5; b=0.75
B=C.copy().astype(np.float64)
for i in range(N):
    s,e=B.indptr[i],B.indptr[i+1]
    f=B.data[s:e]
    B.data[s:e]=(idf[B.indices[s:e]]*(f*(k1+1)/(f+k1*(1-b+b*dl[i]/avgdl))))
M=(CQ.astype(np.float64)@B.T).tocsr()
bm_rank=[]; bm_score=[]
for i in range(len(va)):
    s=M.getrow(i).toarray().ravel()
    r,v=ranks_from_dense(s,50); bm_rank.append(r); bm_score.append(v)

# Reference agreement check against rank_bm25 for fixed 20 queries.
docs=[tokenize(x) for x in tr.case_prompt.astype(str)]
ref=BM25Okapi(docs,k1=1.5,b=0.75,epsilon=0.25)
agreement=[]
for i in list(range(10))+list(range(490,500)):
    a=M.getrow(i).toarray().ravel()
    z=np.asarray(ref.get_scores(tokenize(va.iloc[i].case_prompt)),dtype=np.float64)
    ar=np.lexsort((np.arange(len(a)),-a))[:50]
    zr=np.lexsort((np.arange(len(z)),-z))[:50]
    agreement.append({"query":i,"max_abs_score_diff":float(np.max(np.abs(a-z))),
                      "top50_identical":bool(np.array_equal(ar,zr))})
if not all(x["max_abs_score_diff"]<1e-5 and x["top50_identical"] for x in agreement):
    raise RuntimeError("Sparse BM25 does not match rank_bm25 reference")

# masked reasoning privileged audit
tr_mask=[mask_label(r,l) for r,l in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
va_mask=[mask_label(r,l) for r,l in zip(va.diagnostic_reasoning,va.final_diagnosis)]
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=False,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
RX=rv.fit_transform(tr_mask); RQ=rv.transform(va_mask)
masked_ranks=[]; masked_sims=[]; top10_corr=[]
examples=[]
for i in eligible:
    same=np.array(idx[vl[i]],dtype=int)
    rs=(RQ[i]@RX[same].T).toarray().ravel()
    br=int(same[int(np.argmax(rs))]); masked_sims.append(float(rs.max()))
    cs=(Q[i]@X.T).toarray().ravel()
    rank=1+int(np.sum(cs>cs[br])); masked_ranks.append(rank)
    t10=np.array(tf_rank[i][:10],dtype=int)
    c10=np.array(tf_score[i][:10],dtype=float)
    r10=(RQ[i]@RX[t10].T).toarray().ravel()
    if np.std(c10)>0 and np.std(r10)>0:
        rho=float(spearmanr(c10,r10).statistic)
        if np.isfinite(rho):top10_corr.append(rho)
    if rank>100 and float(rs.max())>=0.10:
        examples.append({"query_index":i,"diagnosis":str(va.iloc[i].final_diagnosis),
                         "masked_reasoning_similarity":float(rs.max()),"case_text_rank":rank,
                         "case_text_similarity":float(cs[br]),"reference_index":br,
                         "target_case":str(va.iloc[i].case_prompt),"target_reasoning_masked":va_mask[i],
                         "reference_case":str(tr.iloc[br].case_prompt),"reference_reasoning_masked":tr_mask[br]})
examples=sorted(examples,key=lambda x:(x["masked_reasoning_similarity"],x["case_text_rank"]),reverse=True)[:30]

summary={
 "dataset_revision":REV,"train_rows":len(tr),"validation_rows":len(va),
 "exact_label_present":len(eligible),"exact_label_absent":len(va)-len(eligible),
 "canonical_metrics":[evalm("tfidf_deterministic",tf_rank,tl,vl,idx),evalm("bm25_okapi_rank_bm25_equivalent",bm_rank,tl,vl,idx)],
 "bm25_reference_validation":{"package":"rank_bm25==0.2.2","queries_checked":20,
   "all_top50_identical":all(x["top50_identical"] for x in agreement),
   "max_score_abs_diff":max(x["max_abs_score_diff"] for x in agreement)},
 "masked_reasoning_privileged_analysis":{
   "diagnosis_phrase_masking":"exact normalized final-diagnosis token sequence only; aliases/acronyms may remain",
   "median_best_masked_reasoning_same_label_case_rank_by_case_text":float(np.median(masked_ranks)),
   "p25":float(np.percentile(masked_ranks,25)),"p75":float(np.percentile(masked_ranks,75)),
   "top10_rate":float(np.mean(np.array(masked_ranks)<=10)),
   "median_masked_reasoning_similarity":float(np.median(masked_sims)),
   "median_top10_case_vs_masked_reasoning_spearman":float(np.median(top10_corr))
 },
 "limits":[
  "Exact normalized diagnosis equality is a proxy label, not clinical equivalence.",
  "Masked reasoning remains privileged retrospective information and may retain diagnosis aliases/acronyms.",
  "Different-diagnosis cases may be useful differential evidence; exact-label miss is not clinical uselessness.",
  "No test-set method selection."
 ]}
(OUT/"E020_CANONICAL_SUMMARY.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
(OUT/"BM25_REFERENCE_CHECK.json").write_text(json.dumps(agreement,indent=2)+"\n")
(OUT/"MASKED_REASONING_HIDDEN_TOP30.json").write_text(json.dumps(examples,indent=2,ensure_ascii=False)+"\n")
with (OUT/"CANONICAL_RANKINGS.jsonl").open("w",encoding="utf-8") as f:
    for i in range(len(va)):
        f.write(json.dumps({"query_index":i,"pmcid":str(va.iloc[i].pmcid),"diagnosis":str(va.iloc[i].final_diagnosis),
                            "tfidf_top50":tf_rank[i],"bm25_top50":bm_rank[i]},ensure_ascii=False)+"\n")
print(json.dumps(summary,indent=2))
