from __future__ import annotations
import json, pathlib, urllib.request, hashlib
import pandas as pd

MC_REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
HF=f"https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/{MC_REV}/data"
META_REV="0887286c5e7940545c6d3fcb907c70edc1d156b3"
HUMAN=f"https://huggingface.co/datasets/zhengyun21/PMC-Patients-MetaData/resolve/{META_REV}/PMC-Patients_human_eval.json?download=true"
OUT=pathlib.Path("e050b");OUT.mkdir(exist_ok=True)

# Mapping artifact contains patient_uid->MedCase train index audit, but no retrieval outcome.
maps=list(pathlib.Path("mapping").rglob("E050_A_MAPPING_AUDIT.csv"))
assert len(maps)==1,maps
m=pd.read_csv(maps[0])
high=m[(m.patients_in_source_article==1) | ((m.candidate_is_best_text_match==True)&(m.candidate_vs_runner_margin.fillna(1)>=0.05))].copy()
assert len(high)==121
uid_to_idx=dict(zip(high.candidate_uid.astype(str),high.medcase_train_index.astype(int)))

# Query source disjointness metadata.
split_sets={}
for split in ("train","val","test"):
    x=pd.read_parquet(f"{HF}/{split}-00000-of-00001.parquet",columns=["pmcid"])
    split_sets[split]=set(x.pmcid.astype(str).str.upper())
all_mc=set().union(*split_sets.values())

def uid_pmc(uid):
    p=str(uid).split("-")[0]
    return ("PMC"+p).upper() if not p.upper().startswith("PMC") else p.upper()

# Preparation reads the official human-eval source only to create a label-free lock.
# The scoring job receives no human relevance labels.
req=urllib.request.Request(HUMAN,headers={"User-Agent":"CASE-EVID-001/1.0"})
with urllib.request.urlopen(req,timeout=180) as r: raw=r.read()
data=json.loads(raw)
assert len(data)==606
queries=[];pairs=[]
for qi,q in enumerate(data):
    qpm=uid_pmc(q["human_patient_uid"])
    queries.append({"query_index":qi,"query_uid":q["human_patient_uid"],"query_pmcid":qpm,
                    "patient":q["patient"],"source_disjoint_from_all_medcase":qpm not in all_mc,
                    "source_in_medcase_train":qpm in split_sets["train"]})
    # use only dict KEYS here; label values are intentionally not emitted.
    for uid in q["similar_patients"].keys():
        if str(uid) in uid_to_idx:
            pairs.append({"query_index":qi,"query_uid":q["human_patient_uid"],"candidate_uid":str(uid),
                          "candidate_pmcid":uid_pmc(uid),"medcase_train_index":uid_to_idx[str(uid)]})
assert len(pairs)==133
with (OUT/"E050B_QUERY_ONLY.jsonl").open("w",encoding="utf-8") as f:
    for x in queries:f.write(json.dumps(x,ensure_ascii=False)+"\n")
with (OUT/"E050B_HIGHCONF_PAIR_LOCK.jsonl").open("w",encoding="utf-8") as f:
    for x in pairs:f.write(json.dumps(x,ensure_ascii=False)+"\n")
lock={"status":"human_relevance_labels_removed_before_scoring",
      "n_queries":len(queries),"n_highconf_pairs":len(pairs),"n_unique_candidates":len(set(x["candidate_uid"] for x in pairs)),
      "query_source_disjoint_count":sum(x["source_disjoint_from_all_medcase"] for x in queries),
      "pair_queries_source_disjoint_count":len(set(x["query_index"] for x in pairs if queries[x["query_index"]]["source_disjoint_from_all_medcase"])),
      "human_eval_revision":META_REV,"human_eval_source_sha256":hashlib.sha256(raw).hexdigest(),
      "medcasereasoning_revision":MC_REV,
      "outcome_fields_in_scoring_lock":[],
      "high_conf_mapping_rule":"single-patient source article OR unique best text match with char5 margin >=0.05"}
(OUT/"E050B_COHORT_LOCK.json").write_text(json.dumps(lock,indent=2)+"\n")
print(json.dumps(lock,indent=2))
