from __future__ import annotations
import json, math, re, unicodedata, pathlib
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy import sparse
from scipy.stats import spearmanr

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e020-a1-fast-output"); OUT.mkdir(exist_ok=True)

def norm_label(s):
    s=unicodedata.normalize("NFKC",str(s)).casefold()
    return " ".join(re.findall(r"\w+",s))
def text_norm(s): return " ".join(str(s).split())
def save(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def topk_rows(mat,k=50):
    rankings=[]
    scores_out=[]
    for i in range(mat.shape[0]):
        row=mat.getrow(i)
        dense=row.toarray().ravel()
        kk=min(k,len(dense))
        idx=np.argpartition(-dense,kk-1)[:kk]
        idx=idx[np.argsort(-dense[idx],kind="mergesort")]
        rankings.append(idx.tolist()); scores_out.append(dense[idx].astype(float).tolist())
    return rankings,scores_out
def eval_method(name,rankings,train_labels,val_labels,label_to_train):
    eligible=[i for i,y in enumerate(val_labels) if y in label_to_train]
    out={"method":name,"queries":len(val_labels),"eligible_exact_label_in_train":len(eligible)}
    for k in (1,3,10,50):
        a=sum(any(train_labels[j]==val_labels[i] for j in rankings[i][:k]) for i in range(len(val_labels)))
        e=sum(any(train_labels[j]==val_labels[i] for j in rankings[i][:k]) for i in eligible)
        out[f"hit_at_{k}_all"]=a; out[f"hit_at_{k}_all_rate"]=a/len(val_labels)
        out[f"hit_at_{k}_eligible"]=e; out[f"hit_at_{k}_eligible_rate"]=e/len(eligible)
    return out

train=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
val=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
val_pmc=set(val.pmcid.astype(str)); val_text={text_norm(x) for x in val.case_prompt}
mask=np.array([(str(r.pmcid) not in val_pmc) and (text_norm(r.case_prompt) not in val_text) for r in train.itertuples()])
train=train.loc[mask].reset_index(drop=True)
train_labels=[norm_label(x) for x in train.final_diagnosis]
val_labels=[norm_label(x) for x in val.final_diagnosis]
label_to_train=defaultdict(list)
for i,y in enumerate(train_labels): label_to_train[y].append(i)

# TF-IDF
tfv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=tfv.fit_transform(train.case_prompt.astype(str)); Q=tfv.transform(val.case_prompt.astype(str))
T=(Q@X.T).tocsr()
tf_rank,tf_scores=topk_rows(T,50)

# Sparse BM25
cv=CountVectorizer(lowercase=True,ngram_range=(1,1),min_df=1,max_features=120000,dtype=np.float32)
C=cv.fit_transform(train.case_prompt.astype(str)).tocsr()
CQ=cv.transform(val.case_prompt.astype(str)).tocsr()
N=C.shape[0]
df=np.diff(C.tocsc().indptr).astype(np.float32)
idf=np.log1p((N-df+0.5)/(df+0.5)).astype(np.float32)
dl=np.asarray(C.sum(axis=1)).ravel().astype(np.float32)
avgdl=float(dl.mean()); k1=1.5; b=.75
den_base=k1*(1-b+b*dl/avgdl)
B=C.copy().astype(np.float32)
for i in range(N):
    s,e=B.indptr[i],B.indptr[i+1]
    vals=B.data[s:e]
    B.data[s:e]=(vals*(k1+1)/(vals+den_base[i]))*idf[B.indices[s:e]]
BQ=CQ.copy().astype(np.float32)
BQ.data[:]=1.0
M=(BQ@B.T).tocsr()
bm_rank,bm_scores=topk_rows(M,50)

metrics=[eval_method("tfidf",tf_rank,train_labels,val_labels,label_to_train),
         eval_method("bm25",bm_rank,train_labels,val_labels,label_to_train)]

# Reasoning privileged space
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
RX=rv.fit_transform(train.diagnostic_reasoning.astype(str)); RQ=rv.transform(val.diagnostic_reasoning.astype(str))
eligible=[i for i,y in enumerate(val_labels) if y in label_to_train]
land=[]; prefer_wrong=0; rranks=[]; corrs=[]
for i in range(len(val)):
    y=val_labels[i]; top=np.array(tf_rank[i]); ts=np.array(tf_scores[i])
    item={"query_index":i,"pmcid":str(val.iloc[i].pmcid),"diagnosis":str(val.iloc[i].final_diagnosis),
          "exact_label_in_train":bool(y in label_to_train),"tfidf_top1_index":int(top[0]),
          "tfidf_top1_diagnosis":str(train.iloc[top[0]].final_diagnosis),
          "tfidf_top1_exact_label":bool(train_labels[top[0]]==y),"tfidf_top1_score":float(ts[0])}
    same=np.array(label_to_train.get(y,[]),dtype=int)
    if len(same):
        ss=(Q[i]@X[same].T).toarray().ravel()
        hp=int(same[int(np.argmax(ss))]); hp_s=float(ss.max())
        diff=[j for j in top if train_labels[j]!=y]
        hn=int(diff[0]) if diff else None
        hn_s=float((Q[i]@X[hn].T).toarray().ravel()[0]) if hn is not None else None
        if hn is not None and hn_s>hp_s: prefer_wrong+=1
        rs=(RQ[i]@RX[same].T).toarray().ravel()
        br=int(same[int(np.argmax(rs))]); brs=float(rs.max())
        all_case=T.getrow(i).toarray().ravel()
        rank=1+int(np.sum(all_case>all_case[br])); rranks.append(rank)
        item.update({"best_same_label_index":hp,"best_same_label_case_similarity":hp_s,
                     "best_diff_label_index":hn,"best_diff_label_case_similarity":hn_s,
                     "semantic_prefers_diff_over_same":bool(hn is not None and hn_s>hp_s),
                     "best_reasoning_same_label_index":br,"best_reasoning_same_label_similarity":brs,
                     "best_reasoning_same_label_case_rank":rank})
    t10=np.array(tf_rank[i][:10]); cs=np.array(tf_scores[i][:10],dtype=float)
    rs=(RQ[i]@RX[t10].T).toarray().ravel()
    if np.std(cs)>0 and np.std(rs)>0:
        rho=float(spearmanr(cs,rs).statistic)
        if np.isfinite(rho): corrs.append(rho)
    land.append(item)
summary={
 "dataset_revision":REV,"train_rows":len(train),"validation_rows":len(val),
 "exact_validation_diagnosis_present_in_train":len(eligible),
 "exact_validation_diagnosis_absent_from_train":len(val)-len(eligible),
 "metrics":metrics,
 "eligible_queries_where_best_different_label_has_higher_case_similarity_than_best_same_label":prefer_wrong,
 "eligible_rate_semantic_prefers_wrong_label":prefer_wrong/len(eligible),
 "best_reasoning_aligned_same_label_case_rank_median":float(np.median(rranks)),
 "best_reasoning_aligned_same_label_case_rank_p25":float(np.percentile(rranks,25)),
 "best_reasoning_aligned_same_label_case_rank_p75":float(np.percentile(rranks,75)),
 "best_reasoning_aligned_same_label_case_rank_top10_rate":float(np.mean(np.array(rranks)<=10)),
 "top10_case_vs_reasoning_similarity_spearman_median":float(np.median(corrs)),
 "interpretation_limits":[
  "Exact normalized diagnosis equality is a proxy label, not clinical equivalence.",
  "Reasoning similarity is privileged retrospective analysis only.",
  "Different-diagnosis cases may still be clinically useful for differential diagnosis.",
  "Test set was not used for method selection."
 ]}
save("E020_A1_SUMMARY.json",summary)
pd.DataFrame(land).to_csv(OUT/"E020_A1_CASE_LANDSCAPE.csv",index=False)
with (OUT/"E020_A1_CASE_LANDSCAPE.jsonl").open("w",encoding="utf-8") as f:
    for x in land:f.write(json.dumps(x,ensure_ascii=False)+"\n")
eligible_items=[x for x in land if x["exact_label_in_train"]]
hard=sorted(eligible_items,key=lambda x:(x.get("best_diff_label_case_similarity",-1)-x.get("best_same_label_case_similarity",-1)),reverse=True)[:30]
hidden=sorted(eligible_items,key=lambda x:x.get("best_reasoning_same_label_case_rank",0),reverse=True)[:30]
save("E020_A1_HARD_NEGATIVE_TOP30.json",hard); save("E020_A1_REASONING_POSITIVE_HIDDEN_TOP30.json",hidden)
print(json.dumps(summary,indent=2))
