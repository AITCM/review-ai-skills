import json,re,unicodedata,pathlib
from collections import defaultdict
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"; BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e020-a1c-output");OUT.mkdir(exist_ok=True)
def n(s):
 s=unicodedata.normalize("NFKC",str(s)).casefold()
 return " ".join(re.findall(r"\w+",s))
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]]
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]]
tl=[n(x) for x in tr.final_diagnosis];vl=[n(x) for x in va.final_diagnosis]
idx=defaultdict(list)
for i,y in enumerate(tl):idx[y].append(i)
cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=cv.fit_transform(tr.case_prompt.astype(str));Q=cv.transform(va.case_prompt.astype(str))
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
RX=rv.fit_transform(tr.diagnostic_reasoning.astype(str));RQ=rv.transform(va.diagnostic_reasoning.astype(str))
candidates=[]
for i,y in enumerate(vl):
 same=np.array(idx.get(y,[]),dtype=int)
 if len(same)<2:continue
 cs=(Q[i]@X[same].T).toarray().ravel(); rs=(RQ[i]@RX[same].T).toarray().ravel()
 lex=int(same[int(np.argmax(cs))]); rr=int(same[int(np.argmax(rs))])
 allcs=(Q[i]@X.T).toarray().ravel(); rank=1+int(np.sum(allcs>allcs[rr]))
 # focus on a reasoning-aligned same-label case hidden by lexical retrieval
 if rank>100 and float(rs.max())>=0.12:
  candidates.append({"query_index":i,"target_diagnosis":str(va.iloc[i].final_diagnosis),"target_pmcid":str(va.iloc[i].pmcid),
   "reasoning_case_rank_by_case_text":rank,"reasoning_similarity":float(rs.max()),"reasoning_case_text_similarity":float(allcs[rr]),
   "best_lexical_same_similarity":float(allcs[lex]),"target_case":str(va.iloc[i].case_prompt),"target_reasoning":str(va.iloc[i].diagnostic_reasoning),
   "reasoning_aligned_case_diagnosis":str(tr.iloc[rr].final_diagnosis),"reasoning_aligned_case":str(tr.iloc[rr].case_prompt),"reasoning_aligned_reasoning":str(tr.iloc[rr].diagnostic_reasoning),
   "lexical_same_case":str(tr.iloc[lex].case_prompt),"lexical_same_reasoning":str(tr.iloc[lex].diagnostic_reasoning)})
candidates=sorted(candidates,key=lambda x:(x["reasoning_similarity"],x["reasoning_case_rank_by_case_text"]),reverse=True)[:15]
(OUT/"E020_A1C_REASONING_HIDDEN_DETAILS.json").write_text(json.dumps(candidates,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
summary={"n_selected":len(candidates),"criteria":"same exact normalized diagnosis, >=2 train cases, reasoning similarity >=0.12, case-text rank >100",
"examples":[{k:x[k] for k in ["query_index","target_diagnosis","reasoning_case_rank_by_case_text","reasoning_similarity","reasoning_case_text_similarity","best_lexical_same_similarity"]} for x in candidates]}
(OUT/"E020_A1C_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
