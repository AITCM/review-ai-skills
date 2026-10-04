from __future__ import annotations
import json,re,unicodedata,pathlib
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher
REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e020-qa1-output");OUT.mkdir(exist_ok=True)
def norm(s):
 s=unicodedata.normalize("NFKC",str(s)).casefold()
 s=re.sub(r"\s+"," ",s).strip()
 return s
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["pmcid","case_prompt","final_diagnosis"]]
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["pmcid","case_prompt","final_diagnosis"]]
# character 5-gram cosine as broad candidate detector
vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(5,5),min_df=2,max_features=150000,dtype=np.float32)
X=vec.fit_transform(tr.case_prompt.astype(str));Q=vec.transform(va.case_prompt.astype(str))
flags=[]; top_ratios=[]
for i in range(len(va)):
 s=(Q[i]@X.T).toarray().ravel()
 cand=np.lexsort((np.arange(len(s)),-s))[:5]
 best=None
 for j in cand:
  a=norm(va.iloc[i].case_prompt);b=norm(tr.iloc[j].case_prompt)
  ratio=SequenceMatcher(None,a,b,autojunk=False).ratio()
  rec={"query_index":i,"val_pmcid":str(va.iloc[i].pmcid),"train_index":int(j),"train_pmcid":str(tr.iloc[j].pmcid),
       "char5_cosine":float(s[j]),"sequence_ratio":float(ratio),
       "val_diagnosis":str(va.iloc[i].final_diagnosis),"train_diagnosis":str(tr.iloc[j].final_diagnosis),
       "val_chars":len(a),"train_chars":len(b)}
  if best is None or ratio>best["sequence_ratio"]:best=rec
  if ratio>=0.85 or s[j]>=0.90:flags.append(rec)
 top_ratios.append(best)
summary={
 "dataset_revision":REV,"validation_rows":len(va),"train_rows":len(tr),
 "pmcid_overlap":len(set(va.pmcid.astype(str))&set(tr.pmcid.astype(str))),
 "normalized_exact_prompt_overlap":len(set(map(norm,va.case_prompt))&set(map(norm,tr.case_prompt))),
 "best_sequence_ratio_quantiles":{str(q):float(np.quantile([x["sequence_ratio"] for x in top_ratios],q)) for q in [0.5,0.9,0.95,0.99,1.0]},
 "best_char5_cosine_quantiles":{str(q):float(np.quantile([x["char5_cosine"] for x in top_ratios],q)) for q in [0.5,0.9,0.95,0.99,1.0]},
 "flagged_pairs_threshold":"sequence_ratio>=0.85 OR char5_cosine>=0.90",
 "flagged_pair_count":len(flags),
 "flagged_validation_queries":len(set(x["query_index"] for x in flags)),
 "limits":["High text overlap is a near-duplicate signal, not proof of the same patient.","Cross-article re-reporting can remain even without high textual overlap."]
}
(OUT/"E020_QA1_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
(OUT/"E020_QA1_FLAGS.json").write_text(json.dumps(sorted(flags,key=lambda x:max(x["sequence_ratio"],x["char5_cosine"]),reverse=True),indent=2,ensure_ascii=False)+"\n")
pd.DataFrame(top_ratios).to_csv(OUT/"E020_QA1_BEST_MATCH_PER_VALIDATION.csv",index=False)
print(json.dumps(summary,indent=2))
