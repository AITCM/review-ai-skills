from __future__ import annotations
import json,re,unicodedata,pathlib,urllib.request
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher

MEDR_COMMIT="ff60ab440afd2f2bc0c603b3a65d715ec83138a7"
URL=f"https://raw.githubusercontent.com/MAGIC-AI4Med/MedRBench/{MEDR_COMMIT}/data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json"
MC_REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";HF=f"https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/{MC_REV}/data"
OUT=pathlib.Path("e031-qa-output");OUT.mkdir(exist_ok=True)
def norm(s):
 s=unicodedata.normalize("NFKC",str(s)).casefold()
 return re.sub(r"\s+"," ",s).strip()

frames=[]
used=set()
for split in ("train","val","test"):
 df=pd.read_parquet(f"{HF}/{split}-00000-of-00001.parquet",columns=["pmcid","case_prompt"])
 df["split"]=split;frames.append(df);used |= set(df.pmcid.astype(str).str.upper())
lib=pd.concat(frames,ignore_index=True)
req=urllib.request.Request(URL,headers={"User-Agent":"CASE-EVID-001/1.0"})
with urllib.request.urlopen(req,timeout=180) as r:data=json.load(r)
ext=[]
for pmcid,rec in data.items():
 pm=str(pmcid).upper()
 if pm in used:continue
 ext.append((pm,rec["generate_case"]["case_summary"]))
assert len(ext)==840

vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(5,5),min_df=2,max_features=180000,dtype=np.float32)
X=vec.fit_transform(lib.case_prompt.astype(str));Q=vec.transform([x[1] for x in ext])
flags=[];best=[]
for i,(pm,qtext) in enumerate(ext):
 s=(Q[i]@X.T).toarray().ravel()
 cand=np.lexsort((np.arange(len(s)),-s))[:5]
 b=None
 for j in cand:
  ratio=SequenceMatcher(None,norm(qtext),norm(lib.iloc[j].case_prompt),autojunk=False).ratio()
  rec={"query_index":i,"medr_pmcid":pm,"medcase_index":int(j),"medcase_pmcid":str(lib.iloc[j].pmcid),
       "medcase_split":str(lib.iloc[j]["split"]),"char5_cosine":float(s[j]),"sequence_ratio":float(ratio)}
  if b is None or ratio>b["sequence_ratio"]:b=rec
  if ratio>=0.85 or s[j]>=0.90:flags.append(rec)
 best.append(b)
summary={"n_external":840,"library_cases":len(lib),"pmcid_overlap_after_exclusion":0,
 "threshold":"sequence_ratio>=0.85 OR char5_cosine>=0.90","flagged_pairs":len(flags),
 "flagged_external_queries":len(set(x["query_index"] for x in flags)),
 "best_sequence_ratio_quantiles":{str(q):float(np.quantile([x["sequence_ratio"] for x in best],q)) for q in [.5,.9,.95,.99,1]},
 "best_char5_cosine_quantiles":{str(q):float(np.quantile([x["char5_cosine"] for x in best],q)) for q in [.5,.9,.95,.99,1]},
 "limits":["High text overlap is a screening signal, not proof of the same patient.","Low text overlap does not rule out cross-article re-reporting."]}
(OUT/"E031_QA_NEAR_DUPLICATE_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
(OUT/"E031_QA_NEAR_DUPLICATE_FLAGS.json").write_text(json.dumps(flags,indent=2)+"\n")
pd.DataFrame(best).to_csv(OUT/"E031_QA_BEST_MATCH_PER_EXTERNAL.csv",index=False)
print(json.dumps(summary,indent=2))
