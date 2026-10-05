import json,urllib.request,pandas as pd
from collections import Counter
MC_REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
HF_MC=f"https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/{MC_REV}/data"
META_REV="0887286c5e7940545c6d3fcb907c70edc1d156b3"
URL=f"https://huggingface.co/datasets/zhengyun21/PMC-Patients-MetaData/resolve/{META_REV}/PMC-Patients_human_eval.json?download=true"
with urllib.request.urlopen(urllib.request.Request(URL,headers={"User-Agent":"CASE-EVID-001/1.0"}),timeout=180) as r:h=json.load(r)
tr=pd.read_parquet(HF_MC+"/train-00000-of-00001.parquet",columns=["pmcid"])
S=set(tr.pmcid.astype(str).str.upper())
def pm(uid):return "PMC"+str(uid).split("-")[0]
rows=[]
for qi,q in enumerate(h):
 for uid,label in q["similar_patients"].items():
  rows.append((qi,uid,str(label),pm(uid),pm(uid) in S))
df=pd.DataFrame(rows,columns=["q","uid","label","pmcid","mapped"])
qc=df.groupby("q").mapped.sum()
# binary relevant
df["positive"]=df.label!="0"
eligible=0
for q,g in df[df.mapped].groupby("q"):
 if len(g)>=2 and g.positive.any() and (~g.positive).any():eligible+=1
out={"queries":606,"pairs":3030,"mapped_pairs":int(df.mapped.sum()),"mapped_unique_uids":int(df[df.mapped].uid.nunique()),
"queries_ge1":int((qc>=1).sum()),"queries_ge2":int((qc>=2).sum()),"queries_ge3":int((qc>=3).sum()),"queries_all5":int((qc==5).sum()),
"queries_binary_discrimination_eligible_before_patient_level_alignment":eligible,
"mapped_label_distribution":dict(Counter(df[df.mapped].label))}
print(json.dumps(out,indent=2))
