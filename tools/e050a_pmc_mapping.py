from __future__ import annotations
import csv,hashlib,json,pathlib,re,unicodedata,urllib.request
from collections import Counter,defaultdict
from difflib import SequenceMatcher
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

MC_REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
HF_MC=f"https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/{MC_REV}/data"
META_REV="0887286c5e7940545c6d3fcb907c70edc1d156b3"
HUMAN_URL=f"https://huggingface.co/datasets/zhengyun21/PMC-Patients-MetaData/resolve/{META_REV}/PMC-Patients_human_eval.json?download=true"
PMC_URL="https://huggingface.co/datasets/THUMedInfo/PMC-Patients/resolve/9d5b14a/PMC-Patients.csv?download=true"
PMC_SHA256_EXPECTED="97f41501c8fb05e1c5e0ec35b6509d0f354cca86777ae8db9ff2f204ce2eef20"
OUT=pathlib.Path("e050-a-output");OUT.mkdir(exist_ok=True)

def norm(s):
    s=unicodedata.normalize("NFKC",str(s)).casefold()
    return re.sub(r"\s+"," ",s).strip()
def uid_pmc(uid):
    p=str(uid).split("-")[0]
    return ("PMC"+p).upper() if not p.upper().startswith("PMC") else p.upper()
def sha256_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

# Human-eval labels
req=urllib.request.Request(HUMAN_URL,headers={"User-Agent":"CASE-EVID-001/1.0"})
with urllib.request.urlopen(req,timeout=180) as r: human=json.load(r)
assert len(human)==606
pairs=[]
for qi,q in enumerate(human):
    assert len(q["similar_patients"])==5
    for uid,label in q["similar_patients"].items():
        pairs.append({"query_index":qi,"query_uid":q["human_patient_uid"],"candidate_uid":uid,
                      "label":str(label),"candidate_pmcid":uid_pmc(uid)})
assert len(pairs)==3030
candidate_uids=sorted(set(x["candidate_uid"] for x in pairs))

# MedCase train
tr=pd.read_parquet(HF_MC+"/train-00000-of-00001.parquet",columns=["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"])
tr["pmcid_norm"]=tr.pmcid.astype(str).str.upper()
assert tr.pmcid_norm.nunique()==len(tr)
pm_to_idx=dict(zip(tr.pmcid_norm,range(len(tr))))
matched_pmids={x["candidate_pmcid"] for x in pairs if x["candidate_pmcid"] in pm_to_idx}
mapped_pairs=[x for x in pairs if x["candidate_pmcid"] in pm_to_idx]

# Download original PMC-Patients CSV and verify known file hash.
csv_path=pathlib.Path("PMC-Patients.csv")
if not csv_path.exists():
    req=urllib.request.Request(PMC_URL,headers={"User-Agent":"CASE-EVID-001/1.0"})
    with urllib.request.urlopen(req,timeout=1200) as r, csv_path.open("wb") as f:
        while True:
            b=r.read(4*1024*1024)
            if not b:break
            f.write(b)
actual=sha256_file(csv_path)
assert actual==PMC_SHA256_EXPECTED,(actual,PMC_SHA256_EXPECTED)

# Extract all patient records from source articles that map to MedCase train.
wanted_numeric={p.replace("PMC","") for p in matched_pmids}
selected=[]
usecols=["patient_uid","patient","title","PMID","file_path"]
for chunk in pd.read_csv(csv_path,usecols=usecols,chunksize=5000,dtype=str):
    # patient_uid prefix is PMC numeric id in this release.
    pmnum=chunk.patient_uid.astype(str).str.split("-").str[0]
    keep=pmnum.isin(wanted_numeric)
    if keep.any():
        selected.append(chunk.loc[keep].copy())
subset=pd.concat(selected,ignore_index=True) if selected else pd.DataFrame(columns=usecols)
subset["pmcid_norm"]="PMC"+subset.patient_uid.astype(str).str.split("-").str[0]
subset.to_csv(OUT/"PMC_PATIENTS_MAPPED_SOURCE_ARTICLE_SUBSET.csv",index=False)

# Map each human candidate UID to its exact patient summary and audit alignment with MedCase case_prompt.
uid_rows={str(r.patient_uid):r for r in subset.itertuples()}
article_uids=defaultdict(list)
for r in subset.itertuples():article_uids[r.pmcid_norm].append(str(r.patient_uid))

audit=[]
missing_candidate_text=[]
for uid in sorted({x["candidate_uid"] for x in mapped_pairs}):
    pm=uid_pmc(uid);idx=pm_to_idx[pm]
    row=uid_rows.get(uid)
    if row is None:
        missing_candidate_text.append(uid);continue
    med=str(tr.iloc[idx].case_prompt)
    # Compare candidate and all patient summaries from same article to MedCase prompt.
    siblings=article_uids[pm]
    texts=[str(uid_rows[u].patient) for u in siblings]
    docs=[med]+texts
    vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(5,5),min_df=1,dtype=np.float32)
    Z=vec.fit_transform(docs)
    sims=(Z[1:]@Z[0].T).toarray().ravel()
    order=np.argsort(-sims)
    pos=siblings.index(uid)
    rank=1+int(np.sum(sims>sims[pos]))
    runner=float(np.sort(sims)[-2]) if len(sims)>1 else None
    margin=float(sims[pos]-runner) if runner is not None else None
    seq=SequenceMatcher(None,norm(str(row.patient)),norm(med),autojunk=False).ratio()
    audit.append({
      "candidate_uid":uid,"candidate_pmcid":pm,"medcase_train_index":int(idx),
      "patients_in_source_article":len(siblings),
      "candidate_is_best_text_match":bool(rank==1),
      "candidate_match_rank":rank,
      "candidate_char5_cosine":float(sims[pos]),
      "candidate_vs_runner_margin":margin,
      "candidate_sequence_ratio":float(seq),
      "medcase_final_diagnosis":str(tr.iloc[idx].final_diagnosis)
    })

adf=pd.DataFrame(audit)
adf.to_csv(OUT/"E050_A_MAPPING_AUDIT.csv",index=False)

# Pair/query coverage and human label distribution.
mapped_uid=set(adf.candidate_uid)
for x in pairs:x["candidate_text_available"]=x["candidate_uid"] in mapped_uid
pairdf=pd.DataFrame(pairs)
pairdf.to_csv(OUT/"E050_A_ALL_HUMAN_PAIRS_WITH_MAPPING.csv",index=False)
query_counts=pairdf.groupby("query_index").candidate_text_available.sum()
labels_all=Counter(x["label"] for x in pairs)
labels_mapped=Counter(x["label"] for x in pairs if x["candidate_uid"] in mapped_uid)

# Purely mapping-quality descriptive strata; no retrieval outcome used.
single=adf[adf.patients_in_source_article==1]
multi=adf[adf.patients_in_source_article>1]
high_conf=adf[(adf.patients_in_source_article==1) | ((adf.candidate_is_best_text_match)&(adf.candidate_vs_runner_margin.fillna(1)>=0.05))]
summary={
 "status":"mapping_audit_only_no_C2_outcome_examined",
 "human_queries":606,"annotated_patient_pairs":3030,"unique_annotated_candidate_uids":len(candidate_uids),
 "medcase_train_rows":len(tr),
 "candidate_pairs_source_pmcid_in_medcase_train":len(mapped_pairs),
 "unique_candidates_source_pmcid_in_medcase_train":len({x["candidate_uid"] for x in mapped_pairs}),
 "mapped_candidate_uids_with_patient_text":len(mapped_uid),
 "missing_candidate_uids_after_full_corpus_lookup":len(missing_candidate_text),
 "queries_by_number_of_mapped_candidates":{str(k):int((query_counts==k).sum()) for k in range(6)},
 "queries_with_at_least_2_mapped_candidates":int((query_counts>=2).sum()),
 "queries_with_all_5_mapped_candidates":int((query_counts==5).sum()),
 "human_label_distribution_all":dict(labels_all),
 "human_label_distribution_mapped":dict(labels_mapped),
 "mapping_quality":{
   "single_patient_source_articles":int(len(single)),
   "multi_patient_source_articles":int(len(multi)),
   "multi_patient_candidate_is_best_match_rate":float(multi.candidate_is_best_text_match.mean()) if len(multi) else None,
   "char5_cosine_quantiles":{str(q):float(adf.candidate_char5_cosine.quantile(q)) for q in [.1,.25,.5,.75,.9]},
   "sequence_ratio_quantiles":{str(q):float(adf.candidate_sequence_ratio.quantile(q)) for q in [.1,.25,.5,.75,.9]},
   "predefined_high_conf_rule":"single-patient source article OR unique best text match with char5 margin >=0.05",
   "high_conf_candidate_uids":int(len(high_conf))
 },
 "source":{
   "human_eval_revision":META_REV,
   "pmc_patients_csv_sha256":actual,
   "medcasereasoning_revision":MC_REV
 },
 "limits":[
   "PMC-Patients patient_uid prefix behaves as PMC article identifier in the released data despite README wording referring to PMID.",
   "Exact source-article overlap is necessary but not sufficient for patient-level alignment in multi-patient reports.",
   "This step audits mapping only and does not inspect whether C2 or TF-IDF ranks human-relevant pairs better."
 ]}
(OUT/"E050_A_MAPPING_SUMMARY.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
(OUT/"SOURCE_RECEIPT.json").write_text(json.dumps(summary["source"],indent=2)+"\n")
print(json.dumps(summary,indent=2,ensure_ascii=False))
