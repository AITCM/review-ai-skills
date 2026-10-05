import json,time,urllib.parse,urllib.request,pathlib
from collections import Counter
import pandas as pd

ROOT=pathlib.Path("e051a2");ROOT.mkdir(exist_ok=True)
MC="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
TRAIN=f"https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/{MC}/data/train-00000-of-00001.parquet"
META="0887286c5e7940545c6d3fcb907c70edc1d156b3"
HUMAN=f"https://huggingface.co/datasets/zhengyun21/PMC-Patients-MetaData/resolve/{META}/PMC-Patients_human_eval.json?download=true"
UA={"User-Agent":"CASE-EVID-001/1.0"}

tr=pd.read_parquet(TRAIN,columns=["pmcid"])
pmcids=tr.pmcid.astype(str).str.upper().tolist()
pmcid_to_idx={p:i for i,p in enumerate(pmcids)}

mapping={}
for start in range(0,len(pmcids),180):
    ids=pmcids[start:start+180]
    q=urllib.parse.urlencode({"ids":",".join(ids),"format":"json"})
    url="https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?"+q
    req=urllib.request.Request(url,headers=UA)
    with urllib.request.urlopen(req,timeout=120) as r:data=json.load(r)
    for rec in data.get("records",[]):
        pmcid=str(rec.get("pmcid","")).upper();pmid=str(rec.get("pmid",""))
        if pmcid and pmid:mapping[pmcid]=pmid
    time.sleep(0.4)

assert len(mapping)>12000,len(mapping)
pmid_to_train={pmid:pmcid_to_idx[pmcid] for pmcid,pmid in mapping.items()}
(ROOT/"MEDCASE_TRAIN_PMCID_TO_PMID.json").write_text(json.dumps(mapping,indent=2)+"\n")

with urllib.request.urlopen(urllib.request.Request(HUMAN,headers=UA),timeout=180) as r:human=json.load(r)
rows=[]
for qi,q in enumerate(human):
    q_pmcid="PMC"+str(q["human_patient_uid"]).split("-")[0]
    q_pmid=str(q["PMID"])
    for pmid,label in q["relevant_articles"].items():
        pmid=str(pmid);idx=pmid_to_train.get(pmid)
        rows.append({"query_index":qi,"query_pmcid":q_pmcid,"query_pmid":q_pmid,
                     "candidate_pmid":pmid,"label":str(label),"medcase_train_index":idx,
                     "candidate_in_medcase_train":idx is not None,
                     "same_source_article":pmid==q_pmid,
                     "query_source_in_medcase_train":q_pmcid in pmcid_to_idx})
df=pd.DataFrame(rows);mapped=df[df.candidate_in_medcase_train].copy()
mapped.to_csv(ROOT/"E051A2_ARTICLE_PAIRS.csv",index=False)
qcount=mapped.groupby("query_index").size()
strict=(~mapped.query_source_in_medcase_train.astype(bool))&(~mapped.same_source_article.astype(bool))
summary={"status":"official_idconv_article_mapping_audit_only",
 "human_queries":len(human),"annotated_article_pairs":len(df),
 "medcase_train_pmcids":len(pmcids),"pmcids_with_pmid_mapping":len(mapping),
 "mapped_article_pairs":len(mapped),"mapped_unique_candidate_pmids":mapped.candidate_pmid.nunique(),
 "queries_ge1":int((qcount>=1).sum()),"queries_ge2":int((qcount>=2).sum()),"queries_ge3":int((qcount>=3).sum()),"queries_all5":int((qcount==5).sum()),
 "mapped_label_distribution":dict(Counter(mapped.label)),
 "diagnosis_relevant_pairs":int(mapped.label.str.contains("1").sum()),
 "strict_pairs":int(strict.sum()),"strict_queries":int(mapped.loc[strict,"query_index"].nunique()),
 "source":{"medcase_revision":MC,"human_eval_revision":META,"id_mapping":"NCBI PMC ID Converter v1.0"}}
(ROOT/"E051A2_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))