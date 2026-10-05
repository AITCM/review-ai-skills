from __future__ import annotations
import json,pathlib,re,unicodedata,math
from collections import Counter
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

OUT=pathlib.Path("e051")
queries=[json.loads(x) for x in open(OUT/"E051_TEXT_ONLY_QUERY_CANDIDATE_LOCK.jsonl",encoding="utf-8") if x.strip()]
cand=pd.read_csv(OUT/"E051_CANDIDATE_TEXTS.csv",dtype=str).fillna("")
cmap=dict(zip(cand.patient_uid,cand.patient))
uids=cand.patient_uid.tolist();uid_idx={u:i for i,u in enumerate(uids)}
docs=cand.patient.astype(str).tolist()

# TF-IDF scores fit on candidate corpus only.
v=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
X=v.fit_transform(docs);Q=v.transform([x["query_text"] for x in queries])
# BM25 over the same full human-annotated candidate corpus, not per-query 5-doc sets.
tok=lambda s:re.findall(r"(?u)\b\w\w+\b",unicodedata.normalize("NFKC",str(s)).casefold())
bm=BM25Okapi([tok(x) for x in docs],k1=1.5,b=0.75,epsilon=0.25)

rows=[]
for qi,q in enumerate(queries):
 ts=(Q[qi]@X.T).toarray().ravel()
 bs=np.asarray(bm.get_scores(tok(q["query_text"])),dtype=float)
 for original_rank,uid in enumerate(q["candidate_uids"],1):
  j=uid_idx[uid]
  rows.append({"query_index":qi,"query_uid":q["query_uid"],"candidate_uid":uid,
   "original_candidate_rank":original_rank,"tfidf_score":float(ts[j]),"bm25_score":float(bs[j])})
df=pd.DataFrame(rows)
df.to_csv(OUT/"E051_BLIND_SCORES.csv",index=False)
lock=json.load(open(OUT/"E051_TEXT_LOCK.json"))
scorelock={"status":"baseline_scores_locked_before_human_labels_loaded",
 "text_lock_sha256":__import__("hashlib").sha256((OUT/"E051_TEXT_ONLY_QUERY_CANDIDATE_LOCK.jsonl").read_bytes()).hexdigest(),
 "n_pairs":len(df),"methods":{
  "tfidf":{"ngram_range":[1,2],"sublinear_tf":True,"min_df":2,"max_df":0.98,"max_features":100000},
  "bm25":{"implementation":"rank_bm25 0.2.2 BM25Okapi","k1":1.5,"b":0.75,"epsilon":0.25,"corpus":"2672 unique human-eval candidate patients"}}
}
(OUT/"E051_SCORE_LOCK.json").write_text(json.dumps(scorelock,indent=2)+"\n")
print(json.dumps(scorelock,indent=2))
