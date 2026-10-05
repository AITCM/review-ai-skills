import json,urllib.request,hashlib,pathlib
from collections import Counter
import pandas as pd

ROOT=pathlib.Path("e051a");ROOT.mkdir(exist_ok=True)
MC="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
TRAIN=f"https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/{MC}/data/train-00000-of-00001.parquet"
META="0887286c5e7940545c6d3fcb907c70edc1d156b3"
HUMAN=f"https://huggingface.co/datasets/zhengyun21/PMC-Patients-MetaData/resolve/{META}/PMC-Patients_human_eval.json?download=true"
CSV="https://huggingface.co/datasets/THUMedInfo/PMC-Patients/resolve/9d5b14a/PMC-Patients.csv?download=true"
UA={"User-Agent":"CASE-EVID-001/1.0"}

def dl(url,p):
    req=urllib.request.Request(url,headers=UA)
    with urllib.request.urlopen(req,timeout=1200) as r, open(p,"wb") as f:
        while True:
            b=r.read(4*1024*1024)
            if not b: break
            f.write(b)

tr=pd.read_parquet(TRAIN,columns=["pmcid"])
train_pm=set(tr.pmcid.astype(str).str.upper())
with urllib.request.urlopen(urllib.request.Request(HUMAN,headers=UA),timeout=180) as r: human=json.load(r)
p=pathlib.Path("PMC-Patients.csv")
if not p.exists(): dl(CSV,p)

# Exact article mapping PMCID -> PMID from PMC-Patients release.
want={x.replace("PMC","") for x in train_pm}
pmcid_to_pmid={}
for chunk in pd.read_csv(p,usecols=["patient_uid","PMID"],dtype=str,chunksize=10000):
    pref=chunk.patient_uid.astype(str).str.split("-").str[0]
    z=chunk[pref.isin(want)].copy()
    for uid,pmid in zip(z.patient_uid,z.PMID):
        pmcid="PMC"+str(uid).split("-")[0]
        if pmcid in pmcid_to_pmid and pmcid_to_pmid[pmcid]!=str(pmid):
            raise RuntimeError((pmcid,pmcid_to_pmid[pmcid],pmid))
        pmcid_to_pmid[pmcid]=str(pmid)
pmid_to_train={pmid:i for i,pmcid in enumerate(tr.pmcid.astype(str).str.upper()) if (pmid:=pmcid_to_pmid.get(pmcid))}

rows=[]
for qi,q in enumerate(human):
    q_pmcid="PMC"+str(q["human_patient_uid"]).split("-")[0]
    q_pmid=str(q["PMID"])
    for pmid,label in q["relevant_articles"].items():
        pmid=str(pmid)
        rows.append({"query_index":qi,"query_pmcid":q_pmcid,"query_pmid":q_pmid,
                     "candidate_pmid":pmid,"label":str(label),
                     "medcase_train_index":pmid_to_train.get(pmid),
                     "candidate_in_medcase_train":pmid in pmid_to_train,
                     "same_source_article":pmid==q_pmid,
                     "query_source_in_medcase_train":q_pmcid in train_pm})
df=pd.DataFrame(rows)
mapped=df[df.candidate_in_medcase_train].copy()
mapped.to_csv(ROOT/"E051A_ARTICLE_PAIRS.csv",index=False)
qcount=mapped.groupby("query_index").size()
summary={
 "status":"article_mapping_audit_only_no_C2_results",
 "human_queries":len(human),"annotated_article_pairs":len(df),
 "medcase_train_articles_with_pmid_mapping":len(pmid_to_train),
 "mapped_article_pairs":len(mapped),"mapped_unique_candidate_pmids":mapped.candidate_pmid.nunique(),
 "queries_ge1":int((qcount>=1).sum()),"queries_ge2":int((qcount>=2).sum()),"queries_ge3":int((qcount>=3).sum()),"queries_all5":int((qcount==5).sum()),
 "mapped_label_distribution":dict(Counter(mapped.label)),
 "diagnosis_relevant_pairs":int(mapped.label.str.contains("1").sum()),
 "strict_pairs":int((~mapped.query_source_in_medcase_train.astype(bool)&~mapped.same_source_article.astype(bool)).sum()),
 "source":{"medcase_revision":MC,"human_eval_revision":META,"pmc_patients_csv":"9d5b14a"},
 "note":"Labels: 0=Irrelevant, 1=Diagnosis, 2=Test, 3=Treatment; combinations allowed."
}
(ROOT/"E051A_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))