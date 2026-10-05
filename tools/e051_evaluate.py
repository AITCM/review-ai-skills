from __future__ import annotations
import json,math,pathlib,urllib.request
from collections import Counter
import numpy as np,pandas as pd
from scipy.stats import spearmanr

META_REV="0887286c5e7940545c6d3fcb907c70edc1d156b3"
HUMAN_URL=f"https://huggingface.co/datasets/zhengyun21/PMC-Patients-MetaData/resolve/{META_REV}/PMC-Patients_human_eval.json?download=true"
OUT=pathlib.Path("e051");RNG=np.random.default_rng(20261005)

lock=json.load(open(OUT/"E051_SCORE_LOCK.json"))
assert lock["status"]=="baseline_scores_locked_before_human_labels_loaded"
scores=pd.read_csv(OUT/"E051_BLIND_SCORES.csv",dtype={"query_uid":str,"candidate_uid":str})

# Labels loaded only here.
req=urllib.request.Request(HUMAN_URL,headers={"User-Agent":"CASE-EVID-001/1.0"})
with urllib.request.urlopen(req,timeout=180) as r:human=json.load(r)
labels={}
for qi,q in enumerate(human):
 for uid,l in q["similar_patients"].items():labels[(qi,uid)]=str(l)
assert len(labels)==3030

def dims(label):
 return {x for x in str(label) if x in "123"}
scores["raw_label"]=[labels[(int(r.query_index),r.candidate_uid)] for r in scores.itertuples()]
scores["valid_label"]=[bool(str(x)) and (str(x)=="0" or bool(dims(x))) for x in scores.raw_label]
scores["any_similarity"]=[str(x)!="0" and bool(dims(x)) for x in scores.raw_label]
scores["features"]=["1" in dims(x) for x in scores.raw_label]
scores["outcomes"]=["2" in dims(x) for x in scores.raw_label]
scores["exposure"]=["3" in dims(x) for x in scores.raw_label]
scores["dimension_breadth"]=[len(dims(x)) for x in scores.raw_label]
scores.to_csv(OUT/"E051_SCORES_WITH_HUMAN_LABELS.csv",index=False)

def dcg(rel):
 return sum((2**float(r)-1)/math.log2(i+2) for i,r in enumerate(rel))
def metrics(method,target):
 per=[]
 for qi,g in scores[scores.valid_label].groupby("query_index"):
  g=g.copy();y=g[target].astype(int).to_numpy()
  if y.min()==y.max():continue
  s=g[method+"_score"].astype(float).to_numpy()
  # pairwise accuracy positive vs negative; ties=0.5
  vals=[]
  for p in np.where(y==1)[0]:
   for n in np.where(y==0)[0]:
    vals.append(1.0 if s[p]>s[n] else (0.5 if s[p]==s[n] else 0.0))
  order=np.lexsort((g.original_candidate_rank.astype(int).to_numpy(),-s))
  top1=int(y[order[0]])
  rel=y[order];ideal=np.sort(y)[::-1]
  nd=dcg(rel)/dcg(ideal) if dcg(ideal)>0 else np.nan
  per.append({"query_index":int(qi),"pairwise_accuracy":float(np.mean(vals)),"top1_positive":top1,"ndcg5":float(nd),
              "hard_negative_top1":bool(top1==0 and y.sum()>0)})
 return pd.DataFrame(per)

targets=["features","any_similarity","outcomes","exposure"]
methods=["tfidf","bm25"]
result={"status":"human_labels_loaded_only_after_text_and_score_locks",
 "dataset":"PMC-Patients human evaluation",
 "n_queries":606,"n_pairs":3030,
 "label_semantics":{"0":"Dissimilar","1":"Features","2":"Outcomes","3":"Exposure","multi_digit":"similar on multiple dimensions"},
 "primary_target":"features","secondary_targets":["any_similarity","outcomes","exposure"],
 "note":"This is candidate-set construct validation, not full-corpus retrieval and not direct Frozen-C2 validation.",
 "metrics":{},"paired_comparison":{},"hard_negative_examples":[]}

per={}
for t in targets:
 result["metrics"][t]={};per[t]={}
 for m in methods:
  d=metrics(m,t);per[t][m]=d
  result["metrics"][t][m]={"eligible_queries":len(d),
   "mean_pairwise_accuracy":float(d.pairwise_accuracy.mean()),
   "top1_positive_rate":float(d.top1_positive.mean()),
   "mean_ndcg5":float(d.ndcg5.mean()),
   "hard_negative_top1_rate":float(d.hard_negative_top1.mean())}
 # paired BM25-TFIDF across same eligible queries
 a=per[t]["tfidf"].set_index("query_index");b=per[t]["bm25"].set_index("query_index")
 common=a.index.intersection(b.index)
 comp={}
 for metric in ["pairwise_accuracy","top1_positive","ndcg5"]:
  diff=b.loc[common,metric].to_numpy(float)-a.loc[common,metric].to_numpy(float)
  boots=[]
  for _ in range(5000):
   ids=RNG.integers(0,len(diff),len(diff));boots.append(float(diff[ids].mean()))
  comp[metric]={"bm25_minus_tfidf":float(diff.mean()),"ci95":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))]}
 result["paired_comparison"][t]=comp

# Human-labeled hard negatives for primary Features target, selected by TF-IDF only.
gdf=scores[scores.valid_label].copy()
examples=[]
for qi,g in gdf.groupby("query_index"):
 y=g.features.astype(bool).to_numpy()
 if not (y.any() and (~y).any()):continue
 order=np.lexsort((g.original_candidate_rank.astype(int).to_numpy(),-g.tfidf_score.astype(float).to_numpy()))
 if not y[order[0]]:
  best_pos=max(g[y].itertuples(),key=lambda r:r.tfidf_score)
  top=g.iloc[order[0]]
  examples.append({"query_index":int(qi),"tfidf_top1_candidate_uid":top.candidate_uid,
    "tfidf_top1_human_label":top.raw_label,"tfidf_top1_score":float(top.tfidf_score),
    "best_feature_positive_uid":best_pos.candidate_uid,"best_feature_positive_label":best_pos.raw_label,
    "best_feature_positive_score":float(best_pos.tfidf_score),
    "similarity_gap":float(top.tfidf_score-best_pos.tfidf_score)})
result["hard_negative_examples"]=sorted(examples,key=lambda x:x["similarity_gap"],reverse=True)[:30]
result["invalid_or_empty_label_pairs"]=int((~scores.valid_label).sum())
result["limits"]=[
 "Human labels cover five preselected candidate patients per query rather than the full PMC-Patients corpus.",
 "The primary Features label is the closest available human annotation to clinical presentation similarity, but it is not a direct rating of diagnostic reasoning utility.",
 "This experiment validates the problem construct (surface similarity versus human clinical similarity), not the frozen C2 algorithm.",
 "Label strings with repeated digits are collapsed to dimension presence; two empty labels are excluded."
]
(OUT/"E051_HUMAN_CONSTRUCT_VALIDATION.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
print(json.dumps(result,indent=2,ensure_ascii=False))
