from __future__ import annotations
import json,pathlib,urllib.request
from collections import Counter,defaultdict
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from scipy.stats import spearmanr

META_REV="0887286c5e7940545c6d3fcb907c70edc1d156b3"
HUMAN=f"https://huggingface.co/datasets/zhengyun21/PMC-Patients-MetaData/resolve/{META_REV}/PMC-Patients_human_eval.json?download=true"
OUT=pathlib.Path("e050b");RNG=np.random.default_rng(20261004)

lock=json.load(open(OUT/"E050B_SCORE_LOCK.json"))
assert lock["status"]=="frozen_scores_locked_before_human_labels_loaded" and lock["human_relevance_labels_available_to_scorer"]==[]
scores=[json.loads(x) for x in open(OUT/"E050B_BLIND_PAIR_SCORES.jsonl") if x.strip()]
assert len(scores)==129

# Human labels are loaded only after score lock.
req=urllib.request.Request(HUMAN,headers={"User-Agent":"CASE-EVID-001/1.0"})
with urllib.request.urlopen(req,timeout=180) as r: human=json.load(r)
lab={}
for qi,q in enumerate(human):
 for uid,label in q["similar_patients"].items():lab[(qi,str(uid))]=str(label)

rows=[]
for x in scores:
 label=lab[(x["query_index"],x["candidate_uid"])]
 if label=="" or label not in {"0","1","2","3","12","13","23","123","112","132","11"}:continue
 rows.append({**x,"human_label":label,"human_similar":int(label!="0"),
              "feature_similarity":int("1" in label),"outcome_similarity":int("2" in label),"exposure_similarity":int("3" in label),
              "dimension_count":0 if label=="0" else len(set(label))})
df=pd.DataFrame(rows)
assert len(df)==129
assert df.human_similar.nunique()==2

def metrics(d):
 y=d.human_similar.to_numpy()
 res={}
 for name,col in [("TFIDF","baseline_rank_score"),("C2","method_rank_score")]:
  s=d[col].to_numpy()
  res[name]={"AUROC":float(roc_auc_score(y,s)),"average_precision":float(average_precision_score(y,s)),
             "mean_score_positive":float(s[y==1].mean()),"mean_score_dissimilar":float(s[y==0].mean()),
             "top50_positive_rate":float(d.loc[y==1, "baseline_top50" if name=="TFIDF" else "method_top50"].mean())}
 res["delta_AUROC_C2_minus_TFIDF"]=res["C2"]["AUROC"]-res["TFIDF"]["AUROC"]
 res["delta_AP_C2_minus_TFIDF"]=res["C2"]["average_precision"]-res["TFIDF"]["average_precision"]
 return res

def with_cluster_bootstrap(d):
 res=metrics(d)
 qids=np.array(sorted(d.query_index.unique()))
 q_to_pos={q:i for i,q in enumerate(qids)}
 pair_qpos=np.array([q_to_pos[q] for q in d.query_index])
 y=d.human_similar.to_numpy()
 sb=d.baseline_rank_score.to_numpy(float);sm=d.method_rank_score.to_numpy(float)
 ba=[];bp=[]
 for _ in range(10000):
  pick=RNG.integers(0,len(qids),len(qids))
  counts=np.bincount(pick,minlength=len(qids)).astype(float)
  w=counts[pair_qpos]
  keep=w>0
  if len(np.unique(y[keep]))<2:continue
  au_b=roc_auc_score(y[keep],sb[keep],sample_weight=w[keep])
  au_m=roc_auc_score(y[keep],sm[keep],sample_weight=w[keep])
  ap_b=average_precision_score(y[keep],sb[keep],sample_weight=w[keep])
  ap_m=average_precision_score(y[keep],sm[keep],sample_weight=w[keep])
  ba.append(float(au_m-au_b));bp.append(float(ap_m-ap_b))
 res["cluster_bootstrap_delta_AUROC_CI95"]=[float(np.percentile(ba,2.5)),float(np.percentile(ba,97.5))]
 res["cluster_bootstrap_delta_AP_CI95"]=[float(np.percentile(bp,2.5)),float(np.percentile(bp,97.5))]
 return res

main=with_cluster_bootstrap(df)
actionable=df[df.baseline_rank<=50].copy()
actionable_metrics=with_cluster_bootstrap(actionable) if len(actionable) and actionable.human_similar.nunique()==2 else None

# Source-disjoint query sensitivity.
dis=df[df.query_source_disjoint==True].copy()
dis_metrics=metrics(dis) if len(dis) and dis.human_similar.nunique()==2 else None

# Human-dimension descriptive discrimination vs dissimilar labels.
dims={}
neg=df[df.human_label=="0"]
for dim,col in [("Features","feature_similarity"),("Outcomes","outcome_similarity"),("Exposure","exposure_similarity")]:
 sub=pd.concat([neg,df[df[col]==1]],ignore_index=True).drop_duplicates(subset=["query_index","candidate_uid"])
 y=(sub.human_label!="0").astype(int).to_numpy()
 dims[dim]={"n_positive":int(y.sum()),"n_dissimilar":int((y==0).sum()),
            "TFIDF_AUROC":float(roc_auc_score(y,sub.baseline_rank_score)),
            "C2_AUROC":float(roc_auc_score(y,sub.method_rank_score))}
 dims[dim]["delta_AUROC"]=dims[dim]["C2_AUROC"]-dims[dim]["TFIDF_AUROC"]

# Does C2 selectively promote human-similar pairs?
df["rank_score_delta"]=df.method_rank_score-df.baseline_rank_score
pos=df[df.human_similar==1].rank_score_delta.to_numpy()
negd=df[df.human_similar==0].rank_score_delta.to_numpy()
selective=float(pos.mean()-negd.mean())
boot=[]
qids_arr=np.array(qids);q_to_pos={q:i for i,q in enumerate(qids_arr)}
pair_qpos=np.array([q_to_pos[q] for q in df.query_index])
yv=df.human_similar.to_numpy();dv=df.rank_score_delta.to_numpy(float)
for _ in range(10000):
 pick=RNG.integers(0,len(qids_arr),len(qids_arr));counts=np.bincount(pick,minlength=len(qids_arr)).astype(float);w=counts[pair_qpos]
 kp=(yv==1)&(w>0);kn=(yv==0)&(w>0)
 if kp.any() and kn.any():
  pm=np.average(dv[kp],weights=w[kp]);nm=np.average(dv[kn],weights=w[kn]);boot.append(float(pm-nm))

# Very small within-query discrimination subset is reported but not inferentially emphasized.
mixed=[]
for q,g in df.groupby("query_index"):
 if g.human_similar.nunique()==2:
  mixed.append({"query_index":int(q),"n":len(g),
    "TFIDF_pos_minus_neg":float(g[g.human_similar==1].baseline_rank_score.mean()-g[g.human_similar==0].baseline_rank_score.mean()),
    "C2_pos_minus_neg":float(g[g.human_similar==1].method_rank_score.mean()-g[g.human_similar==0].method_rank_score.mean())})

summary={"status":"human_relevance_candidate_discrimination_validation",
 "construct":"PMC-Patients expert patient-patient similarity labels; 0=Dissimilar, 1=Features, 2=Outcomes, 3=Exposure; combined strings denote multiple dimensions",
 "analysis_population":{"high_conf_pairs":len(df),"queries":int(df.query_index.nunique()),"human_similar":int(df.human_similar.sum()),"dissimilar":int((df.human_similar==0).sum()),
                        "source_disjoint_pairs":int(len(dis)),"source_disjoint_queries":int(dis.query_index.nunique())},
 "primary":main,
 "actionable_subset":{"definition":"baseline TF-IDF rank <= 50; Frozen C2 can only rerank this region",
                      "n_pairs":int(len(actionable)),"n_queries":int(actionable.query_index.nunique()),
                      "human_similar":int(actionable.human_similar.sum()),"dissimilar":int((actionable.human_similar==0).sum()),
                      "metrics":actionable_metrics},
 "source_disjoint_sensitivity":dis_metrics,"dimension_descriptive":dims,
 "selective_promotion":{"mean_rank_score_delta_positive":float(pos.mean()),"mean_rank_score_delta_dissimilar":float(negd.mean()),
                        "difference_positive_minus_dissimilar":selective,
                        "cluster_bootstrap_CI95":[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))]},
 "within_query_mixed_label_queries":{"n":len(mixed),"records":mixed},
 "limits":[
  "Human annotations cover only five pre-retrieved candidate patients per query and are not exhaustive relevance judgments over the MedCaseReasoning library.",
  "Only 133 high-confidence annotated pairs map to MedCaseReasoning train; this is a construct-validity / hard-candidate discrimination test, not a full retrieval benchmark.",
  "Most queries contribute one mapped annotated candidate, so primary AUROC/AP are pooled candidate-level measures with query-cluster bootstrap uncertainty.",
  "Human similarity dimensions are patient-patient relevance constructs, not diagnostic-reasoning utility labels.",
  "No method parameters are changed using these human labels."
 ]}
(OUT/"E050B_HUMAN_RELEVANCE_RESULTS.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
df.to_csv(OUT/"E050B_HUMAN_RELEVANCE_PAIRS.csv",index=False)
print(json.dumps(summary,indent=2,ensure_ascii=False))
