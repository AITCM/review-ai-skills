from __future__ import annotations
import json,pathlib,urllib.request
from collections import Counter
import numpy as np,pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score

OUT=pathlib.Path("e050b-output");RNG=np.random.default_rng(20261005)
lock=json.load(open(OUT/"E050B_SCORE_LOCK.json"));assert lock["status"]=="human_labels_not_loaded_during_scoring"
scores=[json.loads(x) for x in open(OUT/"E050B_LABEL_FREE_SCORES.jsonl") if x.strip()]
# Labels are loaded only now.
META_REV="0887286c5e7940545c6d3fcb907c70edc1d156b3"
URL=f"https://huggingface.co/datasets/zhengyun21/PMC-Patients-MetaData/resolve/{META_REV}/PMC-Patients_human_eval.json?download=true"
req=urllib.request.Request(URL,headers={"User-Agent":"CASE-EVID-001/1.0"})
with urllib.request.urlopen(req,timeout=180) as r:human=json.load(r)
label_map={}
for qi,q in enumerate(human):
 for uid,label in q["similar_patients"].items():
  label_map[(qi,str(uid))]=str(label)
rows=[]
for x in scores:
 lab=label_map[(x["query_index"],x["candidate_uid"])]
 if lab=="":continue
 y=0 if lab=="0" else 1
 dims={"features":int("1" in lab),"outcomes":int("2" in lab),"exposure":int("3" in lab)}
 rows.append({**x,"human_label":lab,"human_relevant":y,**dims})
df=pd.DataFrame(rows)
df["strict_primary"]=(~df.query_source_in_medcase_train.astype(bool)) & (~df.same_source_article.astype(bool))
df["operational"] = df.in_baseline_top50.astype(bool)

def cluster_boot_diff(sub,col="delta_reciprocal_rank",n=10000):
 # difference positive - negative, resample queries as clusters
 qids=sub.query_index.unique()
 vals=[]
 for _ in range(n):
  qs=RNG.choice(qids,size=len(qids),replace=True)
  chunks=[sub[sub.query_index==q] for q in qs]
  z=pd.concat(chunks,ignore_index=True)
  p=z[z.human_relevant==1][col];n0=z[z.human_relevant==0][col]
  if len(p) and len(n0):vals.append(float(p.mean()-n0.mean()))
 return [float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))] if vals else [None,None]

def cohort_stats(sub,name):
 pos=sub[sub.human_relevant==1];neg=sub[sub.human_relevant==0]
 out={"name":name,"n_pairs":len(sub),"n_queries":int(sub.query_index.nunique()),"positive_pairs":len(pos),"negative_pairs":len(neg),
  "in_baseline_top50":int(sub.in_baseline_top50.sum()),
  "mean_rank_improvement_positive":float(pos.rank_improvement.mean()) if len(pos) else None,
  "mean_rank_improvement_negative":float(neg.rank_improvement.mean()) if len(neg) else None,
  "mean_delta_rr_positive":float(pos.delta_reciprocal_rank.mean()) if len(pos) else None,
  "mean_delta_rr_negative":float(neg.delta_reciprocal_rank.mean()) if len(neg) else None,
  "difference_delta_rr_positive_minus_negative":float(pos.delta_reciprocal_rank.mean()-neg.delta_reciprocal_rank.mean()) if len(pos) and len(neg) else None,
  "cluster_bootstrap95_difference_delta_rr":cluster_boot_diff(sub)}
 # Operational Top50-only comparisons
 op=sub[sub.in_baseline_top50]
 if len(op) and op.human_relevant.nunique()==2:
  out["operational_top50_n"]=len(op)
  out["operational_positive"]=int(op.human_relevant.sum())
  out["operational_negative"]=int((1-op.human_relevant).sum())
  out["AUC_baseline_rank"]=float(roc_auc_score(op.human_relevant,-op.baseline_rank))
  out["AUC_method_rank"]=float(roc_auc_score(op.human_relevant,-op.method_rank))
  # Promotion contingency human relevance vs promoted/not
  promoted=op.rank_improvement>0
  table=[[int(((op.human_relevant==1)&promoted).sum()),int(((op.human_relevant==1)&(~promoted)).sum())],
         [int(((op.human_relevant==0)&promoted).sum()),int(((op.human_relevant==0)&(~promoted)).sum())]]
  odds,p=fisher_exact(table)
  out["promotion_table_positive_negative_by_promoted_not"]=table
  out["promotion_fisher_odds_ratio"]=float(odds);out["promotion_fisher_p"]=float(p)
  # Top-k positive and negative placements, paired counts
  out["topk"]={}
  for k in (1,3,10):
   out["topk"][str(k)]={
    "human_positive_baseline":int(((op.human_relevant==1)&(op.baseline_rank<=k)).sum()),
    "human_positive_method":int(((op.human_relevant==1)&(op.method_rank<=k)).sum()),
    "human_negative_baseline":int(((op.human_relevant==0)&(op.baseline_rank<=k)).sum()),
    "human_negative_method":int(((op.human_relevant==0)&(op.method_rank<=k)).sum())}
 return out

strict=df[df.strict_primary].copy()
allhc=df.copy()
summary={"status":"auxiliary_human_construct_validation","primary_interpretation":"candidate-level human relevance alignment, not exhaustive retrieval evaluation",
 "label_definition":{"0":"Dissimilar","positive":"any of Features(1), Outcomes(2), Exposure(3)"},
 "cohorts":[cohort_stats(strict,"strict_primary"),cohort_stats(allhc,"all_high_conf_sensitivity")],
 "label_distribution":dict(Counter(df.human_label)),
 "dimension_counts":{"features":int(df.features.sum()),"outcomes":int(df.outcomes.sum()),"exposure":int(df.exposure.sum())},
 "limits":[
  "Human annotations cover only five preselected candidate patients per query, not the full MedCaseReasoning library.",
  "Only a small subset of annotated candidate patients maps to MedCaseReasoning train, so this is construct validation rather than a full retrieval benchmark.",
  "C2 can change rank only within the semantic Top-50 candidate set; mapped candidates outside Top-50 retain their baseline rank.",
  "Patient-level mapping is restricted to the pre-specified high-confidence E050-A rule.",
  "The PMC-Patients human labels measure Features/Outcomes/Exposure similarity, which is related to but not identical to diagnostic reasoning utility."
 ]}
(OUT/"E050B_HUMAN_RELEVANCE_RESULTS.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
df.to_csv(OUT/"E050B_LABELED_PAIR_RESULTS.csv",index=False)
print(json.dumps(summary,indent=2,ensure_ascii=False))
