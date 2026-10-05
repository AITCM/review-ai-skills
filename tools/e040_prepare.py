from __future__ import annotations
import json,hashlib,pathlib,urllib.request,re,unicodedata
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher

AG_COMMIT="b6fbe22300e99a267a7ac94eaa465ab552eef741"
AG_URL=f"https://raw.githubusercontent.com/SamuelSchmidgall/AgentClinic/{AG_COMMIT}/agentclinic_nejm_extended.jsonl"
MC_REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
HF=f"https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/{MC_REV}/data"
OUT=pathlib.Path("e040");OUT.mkdir(exist_ok=True)

def norm(s):
 s=unicodedata.normalize("NFKC",str(s)).casefold()
 return re.sub(r"\s+"," ",s).strip()

req=urllib.request.Request(AG_URL,headers={"User-Agent":"CASE-EVID-001/1.0"})
with urllib.request.urlopen(req,timeout=120) as r: raw=r.read()
lines=[x for x in raw.decode("utf-8").splitlines() if x.strip()]
data=[json.loads(x) for x in lines]
assert len(data)==120,len(data)

queries=[]
for i,x in enumerate(data):
 image_url=str(x.get("image_url",""))
 case_id=image_url.split("id=")[-1].split("&")[0] if "id=" in image_url else f"NEJM-{i:03d}"
 queries.append({"query_index":i,"case_id":case_id,"question":x["question"]})
with (OUT/"E040_QUERY_ONLY.jsonl").open("w",encoding="utf-8") as f:
 for q in queries:f.write(json.dumps(q,ensure_ascii=False)+"\n")
lock={
 "status":"agentclinic_query_only_cohort_locked",
 "agentclinic_commit":AG_COMMIT,
 "source_sha256":hashlib.sha256(raw).hexdigest(),
 "n_cases":len(queries),
 "query_fields":["query_index","case_id","question"],
 "outcome_fields_retained_in_query_lock":[],
 "source_file":"agentclinic_nejm_extended.jsonl"}
(OUT/"E040_QUERY_LOCK.json").write_text(json.dumps(lock,indent=2)+"\n")

# Near-duplicate audit against all MedCaseReasoning splits, using query text only.
frames=[]
for split in ("train","val","test"):
 df=pd.read_parquet(f"{HF}/{split}-00000-of-00001.parquet",columns=["pmcid","case_prompt"])
 df["split"]=split;frames.append(df)
lib=pd.concat(frames,ignore_index=True)
vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(5,5),min_df=2,max_features=180000,dtype=np.float32)
X=vec.fit_transform(lib.case_prompt.astype(str));Q=vec.transform([q["question"] for q in queries])
flags=[];best=[]
for i,q in enumerate(queries):
 s=(Q[i]@X.T).toarray().ravel();cand=np.lexsort((np.arange(len(s)),-s))[:5];b=None
 for j in cand:
  ratio=SequenceMatcher(None,norm(q["question"]),norm(lib.iloc[j].case_prompt),autojunk=False).ratio()
  rec={"query_index":i,"case_id":q["case_id"],"medcase_index":int(j),"medcase_pmcid":str(lib.iloc[j].pmcid),
       "medcase_split":str(lib.iloc[j]["split"]),"char5_cosine":float(s[j]),"sequence_ratio":float(ratio)}
  if b is None or ratio>b["sequence_ratio"]:b=rec
  if ratio>=0.85 or s[j]>=0.90:flags.append(rec)
 best.append(b)
qa={"n_agentclinic":120,"medcase_library":len(lib),"threshold":"sequence_ratio>=0.85 OR char5_cosine>=0.90",
 "flagged_pairs":len(flags),"flagged_queries":len(set(x["query_index"] for x in flags)),
 "best_sequence_ratio_quantiles":{str(q):float(np.quantile([x["sequence_ratio"] for x in best],q)) for q in [.5,.9,.95,.99,1]},
 "best_char5_cosine_quantiles":{str(q):float(np.quantile([x["char5_cosine"] for x in best],q)) for q in [.5,.9,.95,.99,1]}}
(OUT/"E040_NEAR_DUPLICATE_AUDIT.json").write_text(json.dumps(qa,indent=2)+"\n")
pd.DataFrame(best).to_csv(OUT/"E040_NEAR_DUPLICATE_BEST.csv",index=False)
print(json.dumps({"lock":lock,"near_duplicate_audit":qa},indent=2))
