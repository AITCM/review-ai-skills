from __future__ import annotations
import json, math, re, unicodedata, hashlib, pathlib, statistics, os
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.stats import spearmanr

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e020-a1-output"); OUT.mkdir(exist_ok=True)

def norm_label(s):
    s=unicodedata.normalize("NFKC", str(s)).casefold()
    return " ".join(re.findall(r"\w+", s))

def text_norm(s):
    return " ".join(str(s).split())

def write_json(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")

def topk_from_scores(scores,k=50):
    if k>=len(scores):
        idx=np.arange(len(scores))
    else:
        idx=np.argpartition(-scores,k-1)[:k]
    return idx[np.argsort(-scores[idx],kind="mergesort")]

def bm25_prepare(docs):
    toks=[re.findall(r"\b\w+\b", str(x).casefold()) for x in docs]
    N=len(toks); df=Counter()
    for d in toks:
        df.update(set(d))
    avgdl=sum(map(len,toks))/N
    idf={t: math.log(1+(N-f+0.5)/(f+0.5)) for t,f in df.items()}
    tf=[Counter(d) for d in toks]
    lens=np.array([len(d) for d in toks],dtype=np.float32)
    return toks,tf,lens,avgdl,idf

def bm25_scores(query, tf, lens, avgdl, idf, k1=1.5,b=0.75):
    q=set(re.findall(r"\b\w+\b", str(query).casefold()))
    s=np.zeros(len(tf),dtype=np.float32)
    for term in q:
        it=idf.get(term)
        if it is None: continue
        for i,ctr in enumerate(tf):
            f=ctr.get(term,0)
            if f:
                denom=f+k1*(1-b+b*lens[i]/avgdl)
                s[i]+=it*(f*(k1+1)/denom)
    return s

def eval_method(name, rankings, train_labels, val_labels, label_to_train):
    ks=(1,3,10,50)
    eligible=[i for i,y in enumerate(val_labels) if y in label_to_train]
    out={"method":name,"queries":len(val_labels),"eligible_exact_label_in_train":len(eligible)}
    for k in ks:
        hit=sum(any(train_labels[j]==val_labels[i] for j in rankings[i][:k]) for i in range(len(val_labels)))
        ehit=sum(any(train_labels[j]==val_labels[i] for j in rankings[i][:k]) for i in eligible)
        out[f"hit_at_{k}_all"]=hit
        out[f"hit_at_{k}_all_rate"]=hit/len(val_labels)
        out[f"hit_at_{k}_eligible"]=ehit
        out[f"hit_at_{k}_eligible_rate"]=ehit/len(eligible) if eligible else None
    return out

train=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")
val=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")
need=["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]
train=train[need].copy(); val=val[need].copy()
train_labels=[norm_label(x) for x in train.final_diagnosis]
val_labels=[norm_label(x) for x in val.final_diagnosis]
label_to_train=defaultdict(list)
for i,y in enumerate(train_labels): label_to_train[y].append(i)

# Exact source/duplicate exclusions are already absent across official train/val, but retain defensive filtering in rankings.
val_pmc=set(val.pmcid.astype(str))
val_prompt_norm=set(text_norm(x) for x in val.case_prompt)
allowed=np.array([(str(r.pmcid) not in val_pmc) and (text_norm(r.case_prompt) not in val_prompt_norm) for r in train.itertuples()],dtype=bool)
if not allowed.all():
    train=train.loc[allowed].reset_index(drop=True)
    train_labels=[norm_label(x) for x in train.final_diagnosis]
    label_to_train=defaultdict(list)
    for i,y in enumerate(train_labels): label_to_train[y].append(i)

# TF-IDF case representation
vec=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=0.98,max_features=80000,dtype=np.float32)
X=vec.fit_transform(train.case_prompt.astype(str))
Q=vec.transform(val.case_prompt.astype(str))
tfidf_rank=[]; tfidf_top50_scores=[]
for i in range(len(val)):
    scores=(Q[i]@X.T).toarray().ravel()
    idx=topk_from_scores(scores,50)
    tfidf_rank.append(idx.tolist())
    tfidf_top50_scores.append([float(scores[j]) for j in idx])

# BM25 lexical baseline
_,tf,lens,avgdl,idf=bm25_prepare(train.case_prompt.astype(str).tolist())
bm25_rank=[]
for q in val.case_prompt.astype(str):
    scores=bm25_scores(q,tf,lens,avgdl,idf)
    bm25_rank.append(topk_from_scores(scores,50).tolist())

metrics=[
    eval_method("tfidf",tfidf_rank,train_labels,val_labels,label_to_train),
    eval_method("bm25",bm25_rank,train_labels,val_labels,label_to_train)
]

# Reasoning-space TF-IDF used ONLY as privileged analysis, never as deployment input.
rvec=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=0.98,max_features=80000,dtype=np.float32)
RX=rvec.fit_transform(train.diagnostic_reasoning.astype(str))
RQ=rvec.transform(val.diagnostic_reasoning.astype(str))

eligible=[i for i,y in enumerate(val_labels) if y in label_to_train]
landscape=[]
prefer_wrong=0
best_reasoning_ranks=[]
top10_corrs=[]
for i in range(len(val)):
    y=val_labels[i]
    ridx=np.array(tfidf_rank[i],dtype=int)
    rscores=np.array(tfidf_top50_scores[i],dtype=float)
    same=np.array(label_to_train.get(y,[]),dtype=int)
    item={"query_index":i,"pmcid":str(val.iloc[i].pmcid),"diagnosis":str(val.iloc[i].final_diagnosis),
          "exact_label_in_train":bool(len(same)),"tfidf_top1_index":int(ridx[0]),
          "tfidf_top1_diagnosis":str(train.iloc[ridx[0]].final_diagnosis),
          "tfidf_top1_exact_label":bool(train_labels[ridx[0]]==y),"tfidf_top1_score":float(rscores[0])}
    if len(same):
        # highest text-sim same-label candidate
        ss=(Q[i]@X[same].T).toarray().ravel()
        p=int(np.argmax(ss)); hp=int(same[p]); hp_s=float(ss[p])
        # top text-sim different-label candidate
        diff=[j for j in ridx if train_labels[j]!=y]
        hn=int(diff[0]) if diff else None
        hn_s=float((Q[i]@X[hn].T).toarray().ravel()[0]) if hn is not None else None
        if hn is not None and hn_s>hp_s: prefer_wrong+=1
        # reasoning-privileged best same-label candidate and its case-text rank
        r_same=(RQ[i]@RX[same].T).toarray().ravel()
        rp=int(np.argmax(r_same)); best_r=int(same[rp]); best_rsim=float(r_same[rp])
        all_case=(Q[i]@X.T).toarray().ravel()
        rank=1+int(np.sum(all_case>all_case[best_r]))
        best_reasoning_ranks.append(rank)
        item.update({"best_same_label_index":hp,"best_same_label_case_similarity":hp_s,
                     "best_diff_label_index":hn,"best_diff_label_case_similarity":hn_s,
                     "semantic_prefers_diff_over_same": bool(hn is not None and hn_s>hp_s),
                     "best_reasoning_same_label_index":best_r,
                     "best_reasoning_same_label_similarity":best_rsim,
                     "best_reasoning_same_label_case_rank":rank})
    # top10 case-vs-reasoning similarity correlation
    top10=np.array(tfidf_rank[i][:10],dtype=int)
    case=np.array(tfidf_top50_scores[i][:10],dtype=float)
    reason=(RQ[i]@RX[top10].T).toarray().ravel()
    if np.std(case)>0 and np.std(reason)>0:
        rho=float(spearmanr(case,reason).statistic)
        if not math.isnan(rho): top10_corrs.append(rho)
    landscape.append(item)

summary={
 "dataset_revision":REV,
 "train_rows":len(train),"validation_rows":len(val),
 "exact_validation_diagnosis_present_in_train":len(eligible),
 "exact_validation_diagnosis_absent_from_train":len(val)-len(eligible),
 "metrics":metrics,
 "eligible_queries_where_best_different_label_has_higher_case_similarity_than_best_same_label":prefer_wrong,
 "eligible_rate_semantic_prefers_wrong_label":prefer_wrong/len(eligible),
 "best_reasoning_aligned_same_label_case_rank_median":float(np.median(best_reasoning_ranks)),
 "best_reasoning_aligned_same_label_case_rank_p25":float(np.percentile(best_reasoning_ranks,25)),
 "best_reasoning_aligned_same_label_case_rank_p75":float(np.percentile(best_reasoning_ranks,75)),
 "best_reasoning_aligned_same_label_case_rank_top10_rate":float(np.mean(np.array(best_reasoning_ranks)<=10)),
 "top10_case_vs_reasoning_similarity_spearman_median":float(np.median(top10_corrs)),
 "interpretation_limits":[
   "Exact normalized diagnosis equality is a proxy label, not clinical equivalence.",
   "Reasoning similarity uses validation diagnostic_reasoning only for retrospective landscape analysis, never as a deployable query feature.",
   "A different diagnosis can still be clinically useful for differential diagnosis; mismatch is not synonymous with useless evidence.",
   "No test-set model selection or method tuning is performed."
 ]
}
write_json("E020_A1_SUMMARY.json",summary)
with (OUT/"E020_A1_CASE_LANDSCAPE.jsonl").open("w",encoding="utf-8") as f:
    for x in landscape: f.write(json.dumps(x,ensure_ascii=False)+"\n")
pd.DataFrame(landscape).to_csv(OUT/"E020_A1_CASE_LANDSCAPE.csv",index=False)

# Curated extreme examples for later qualitative review, selected algorithmically.
eligible_items=[x for x in landscape if x.get("exact_label_in_train")]
hard_neg=sorted(eligible_items,key=lambda x:(x.get("best_diff_label_case_similarity",-1)-x.get("best_same_label_case_similarity",-1)),reverse=True)[:30]
reason_hidden=sorted(eligible_items,key=lambda x:x.get("best_reasoning_same_label_case_rank",0),reverse=True)[:30]
write_json("E020_A1_HARD_NEGATIVE_TOP30.json",hard_neg)
write_json("E020_A1_REASONING_POSITIVE_HIDDEN_TOP30.json",reason_hidden)
print(json.dumps(summary,indent=2))
