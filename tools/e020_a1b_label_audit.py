import json,re,unicodedata,pathlib
from collections import Counter,defaultdict
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e020-a1b-output");OUT.mkdir(exist_ok=True)
def n(s):
 s=unicodedata.normalize("NFKC",str(s)).casefold()
 return " ".join(re.findall(r"\w+",s))
def jt(a,b):
 A=set(n(a).split());B=set(n(b).split())
 return len(A&B)/len(A|B) if A|B else 0
def save(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]]
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]]
tl=[n(x) for x in tr.final_diagnosis]; vl=[n(x) for x in va.final_diagnosis]
idx=defaultdict(list)
for i,y in enumerate(tl): idx[y].append(i)
freq=[len(idx[y]) for y in vl if y in idx]
# recreate TF-IDF ranking and exact best-same pairs
v=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=v.fit_transform(tr.case_prompt.astype(str));Q=v.transform(va.case_prompt.astype(str))
rows=[]; mismatch_token_overlap=[]
for i,y in enumerate(vl):
 scores=(Q[i]@X.T).toarray().ravel()
 top=int(np.argmax(scores))
 rec={"query_index":i,"pmcid":str(va.iloc[i].pmcid),"target_diagnosis":str(va.iloc[i].final_diagnosis),
      "train_exact_label_count":len(idx.get(y,[])),"top1_index":top,"top1_diagnosis":str(tr.iloc[top].final_diagnosis),
      "top1_similarity":float(scores[top]),"top1_exact":bool(tl[top]==y),
      "diagnosis_token_jaccard":jt(va.iloc[i].final_diagnosis,tr.iloc[top].final_diagnosis)}
 if tl[top]!=y:mismatch_token_overlap.append(rec["diagnosis_token_jaccard"])
 if y in idx:
  same=np.array(idx[y]);ss=scores[same];j=int(same[int(np.argmax(ss))])
  diff_order=np.argsort(-scores)
  d=next(int(x) for x in diff_order if tl[x]!=y)
  rec.update({"best_same_index":j,"best_same_similarity":float(scores[j]),"best_diff_index":d,
              "best_diff_diagnosis":str(tr.iloc[d].final_diagnosis),"best_diff_similarity":float(scores[d]),
              "diff_minus_same":float(scores[d]-scores[j])})
 rows.append(rec)
# frequency strata
eligible=[r for r in rows if r["train_exact_label_count"]]
strata={}
for lo,hi,name in [(1,1,"1"),(2,2,"2"),(3,5,"3-5"),(6,10,"6-10"),(11,10**9,">10")]:
 xs=[r for r in eligible if lo<=r["train_exact_label_count"]<=hi]
 strata[name]={"n":len(xs),"top1_exact_rate":sum(r["top1_exact"] for r in xs)/len(xs) if xs else None,
               "diff_beats_same_rate":sum(r["diff_minus_same"]>0 for r in xs)/len(xs) if xs else None}
summary={"eligible_n":len(freq),"train_label_frequency":{"median":float(np.median(freq)),"p25":float(np.percentile(freq,25)),
 "p75":float(np.percentile(freq,75)),"min":int(min(freq)),"max":int(max(freq)),
 "singleton_queries":sum(x==1 for x in freq),"singleton_query_rate":sum(x==1 for x in freq)/len(freq)},
 "frequency_strata":strata,
 "top1_mismatch_diagnosis_token_jaccard":{"median":float(np.median(mismatch_token_overlap)),
  "any_token_overlap_rate":float(np.mean(np.array(mismatch_token_overlap)>0))},
 "note":"Token overlap is only a crude naming audit; it is not ontology equivalence."}
save("E020_A1B_LABEL_AUDIT_SUMMARY.json",summary)
pd.DataFrame(rows).to_csv(OUT/"E020_A1B_ROWS.csv",index=False)
# algorithmic detailed samples
hard=sorted([r for r in eligible if not r["top1_exact"]],key=lambda r:r["diff_minus_same"],reverse=True)[:15]
details=[]
for r in hard:
 i=r["query_index"]; d=r["best_diff_index"]; s=r["best_same_index"]
 details.append({**r,
  "target_case_prompt":str(va.iloc[i].case_prompt),"target_reasoning":str(va.iloc[i].diagnostic_reasoning),
  "diff_case_prompt":str(tr.iloc[d].case_prompt),"diff_reasoning":str(tr.iloc[d].diagnostic_reasoning),
  "same_case_prompt":str(tr.iloc[s].case_prompt),"same_reasoning":str(tr.iloc[s].diagnostic_reasoning)})
save("E020_A1B_HARD_NEGATIVE_DETAILS.json",details)
print(json.dumps(summary,indent=2))
