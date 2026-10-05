from __future__ import annotations
import hashlib,json,pathlib,re,unicodedata,urllib.request
import pandas as pd

META_REV="0887286c5e7940545c6d3fcb907c70edc1d156b3"
HUMAN_URL=f"https://huggingface.co/datasets/zhengyun21/PMC-Patients-MetaData/resolve/{META_REV}/PMC-Patients_human_eval.json?download=true"
PMC_URL="https://huggingface.co/datasets/THUMedInfo/PMC-Patients/resolve/9d5b14a/PMC-Patients.csv?download=true"
PMC_SHA256="97f41501c8fb05e1c5e0ec35b6509d0f354cca86777ae8db9ff2f204ce2eef20"
OUT=pathlib.Path("e051");OUT.mkdir(exist_ok=True)

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
 return h.hexdigest()

req=urllib.request.Request(HUMAN_URL,headers={"User-Agent":"CASE-EVID-001/1.0"})
with urllib.request.urlopen(req,timeout=180) as r:
 raw=r.read()
human=json.loads(raw)
assert len(human)==606

# Lock text-only query/candidate identities; labels intentionally omitted.
queries=[];uids=set()
for qi,q in enumerate(human):
 cands=list(q["similar_patients"].keys())
 assert len(cands)==5
 uids.update(cands)
 queries.append({"query_index":qi,"query_uid":q["human_patient_uid"],"query_text":q["patient"],"candidate_uids":cands})
with (OUT/"E051_TEXT_ONLY_QUERY_CANDIDATE_LOCK.jsonl").open("w",encoding="utf-8") as f:
 for x in queries:f.write(json.dumps(x,ensure_ascii=False)+"\n")
assert all(set(x)=={"query_index","query_uid","query_text","candidate_uids"} for x in queries)
assert all(not any(k in x for k in ["similar_patients","relevant_articles","answers","label","gold"]) for x in queries)

# Download pinned original PMC-Patients CSV and verify immutable content hash.
p=pathlib.Path("PMC-Patients.csv")
if not p.exists():
 req=urllib.request.Request(PMC_URL,headers={"User-Agent":"CASE-EVID-001/1.0"})
 with urllib.request.urlopen(req,timeout=1200) as r,p.open("wb") as f:
  while True:
   b=r.read(4*1024*1024)
   if not b:break
   f.write(b)
actual=sha(p);assert actual==PMC_SHA256,(actual,PMC_SHA256)

# Extract exact candidate patient records only.
sel=[]
for chunk in pd.read_csv(p,usecols=["patient_uid","patient","title","PMID","file_path"],chunksize=5000,dtype=str):
 keep=chunk.patient_uid.astype(str).isin(uids)
 if keep.any():sel.append(chunk.loc[keep].copy())
cand=pd.concat(sel,ignore_index=True)
assert cand.patient_uid.nunique()==len(uids),(cand.patient_uid.nunique(),len(uids))
assert not cand.patient_uid.duplicated().any()
cand.to_csv(OUT/"E051_CANDIDATE_TEXTS.csv",index=False)

lock={"status":"text_only_candidate_set_locked_before_human_labels_loaded",
 "human_eval_revision":META_REV,"human_eval_source_sha256":hashlib.sha256(raw).hexdigest(),
 "pmc_patients_csv_sha256":actual,"n_queries":len(queries),"unique_candidate_uids":len(uids),
 "pairs":sum(len(x["candidate_uids"]) for x in queries),
 "fields_available_for_scoring":["query_index","query_uid","query_text","candidate_uids","candidate patient text"],
 "human_similarity_labels_available_for_scoring":[]}
(OUT/"E051_TEXT_LOCK.json").write_text(json.dumps(lock,indent=2)+"\n")
print(json.dumps(lock,indent=2))
