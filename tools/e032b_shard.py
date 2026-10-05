from __future__ import annotations
import argparse,json,re,unicodedata,pathlib,urllib.request
import numpy as np,pandas as pd
from sentence_transformers import SentenceTransformer

MEDR_COMMIT="ff60ab440afd2f2bc0c603b3a65d715ec83138a7"
URL=f"https://raw.githubusercontent.com/MAGIC-AI4Med/MedRBench/{MEDR_COMMIT}/data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json"
REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";HF=f"https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/{REV}/data"
MODEL="pritamdeka/S-PubMedBert-MS-MARCO";MODEL_REV="96786c7024f95c5aac7f2b9a18086c7b97b23036"
PAT=re.compile(r'^\s*(?:\((\d+)\)|(\d+)[.)])\s*',re.M);GENERIC={"with","without","acute","chronic","disease","syndrome","secondary","primary","type","and","the","of","disorder"}
def nw(s):return " ".join(re.findall(r"\w+",unicodedata.normalize("NFKC",str(s)).casefold()))
def split_reason(s):
 s=str(s);ms=list(PAT.finditer(s))
 if not ms or s[:ms[0].start()].strip():return [s.strip()]
 nums=[int(m.group(1) or m.group(2)) for m in ms]
 if nums!=list(range(1,len(ms)+1)):return [s.strip()]
 out=[]
 for i,m in enumerate(ms):
  e=ms[i+1].start() if i+1<len(ms) else len(s);x=s[m.end():e].strip()
  if x:out.append(x)
 return out or [s.strip()]
def pts_mc(r,d):
 dd=nw(d);return [nw(x).replace(dd," diagnosismask ") if dd and dd in nw(x) else nw(x) for x in split_reason(r)]
def pts_medr(s,dx):
 chunks=re.split(r'(?m)(?=^\s*\d+\.\s+)',str(s));o=[]
 for x in chunks:
  x=x.strip()
  if not x:continue
  x=re.sub(r'^\d+\.\s*','',x);x=re.sub(r'^\*\*[^*]{1,160}\*\*\s*:\s*','',x);x=nw(x)
  if x:o.append(x)
 if not o:o=[nw(s)]
 toks={t for t in nw(dx).split() if len(t)>=4 and t not in GENERIC}
 return [" ".join("diagnosismask" if t in toks else t for t in p.split()) for p in o]
def sf(q,c):
 S=q@c.T;r=float(np.max(S,axis=1).mean());p=float(np.max(S,axis=0).mean())
 return 2*r*p/(r+p) if r+p else 0.

ap=argparse.ArgumentParser();ap.add_argument("--shard",type=int,required=True);ap.add_argument("--nshards",type=int,default=6);ap.add_argument("--out",required=True);a=ap.parse_args()
OUT=pathlib.Path(a.out);OUT.mkdir(parents=True,exist_ok=True)
locks=list(pathlib.Path("locked").rglob("E031V2_RANKING_LOCK.json"));rfs=list(pathlib.Path("locked").rglob("E031V2_BLIND_RANKINGS.jsonl"));qfs=list(pathlib.Path("locked").rglob("E031V2_QUERY_ONLY.jsonl"))
assert len(locks)==len(rfs)==len(qfs)==1
assert json.load(open(locks[0]))["status"]=="external_rankings_locked_before_outcomes_loaded"
ranks=[json.loads(x) for x in open(rfs[0]) if x.strip()];queries=[json.loads(x) for x in open(qfs[0]) if x.strip()]
ids=[i for i in range(len(queries)) if i%a.nshards==a.shard]
req=urllib.request.Request(URL,headers={"User-Agent":"CASE-EVID-001/1.0"})
with urllib.request.urlopen(req,timeout=180) as rr:data=json.load(rr)
tr=pd.read_parquet(HF+"/train-00000-of-00001.parquet",columns=["diagnostic_reasoning","final_diagnosis"])
needed=set()
for i in ids:
 for key in ("baseline_top50","method_top50"):needed.update(ranks[i][key][:10])
texts=[];keys=[];tp={};vp={};explicit={}
for j in sorted(needed):
 ps=pts_mc(tr.iloc[j].diagnostic_reasoning,tr.iloc[j].final_diagnosis);tp[j]=[]
 for k,p in enumerate(ps):keys.append(("tr",j,k));texts.append(p)
for i in ids:
 q=queries[i];g=data[q["pmcid"]]["generate_case"];ps=pts_medr(g["differential_diagnosis"],g["final_diagnosis"]);vp[i]=[]
 explicit[i]=bool(nw(g["final_diagnosis"]) and nw(g["final_diagnosis"]) in nw(q["case_summary"]))
 for k,p in enumerate(ps):keys.append(("va",i,k));texts.append(p)
m=SentenceTransformer(MODEL,revision=MODEL_REV,device="cpu");m.max_seq_length=256
E=m.encode(texts,batch_size=64,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True)
for z,key in zip(E,keys):
 typ,i,k=key;(tp if typ=="tr" else vp)[i].append(z)
rows=[]
for i in ids:
 rec={"query_index":i,"pmcid":queries[i]["pmcid"],"query_dx_explicit":explicit[i]}
 q=np.asarray(vp[i])
 for k in (1,3,10):
  for name,key in (("baseline","baseline_top50"),("method","method_top50")):
   c=np.vstack([np.asarray(tp[j]) for j in ranks[i][key][:k]])
   rec[f"{name}_{k}"]=sf(q,c)
 rows.append(rec)
with (OUT/f"shard_{a.shard}.jsonl").open("w") as f:
 for x in rows:f.write(json.dumps(x)+"\n")
print(json.dumps({"shard":a.shard,"n":len(rows),"unique_train_cases":len(needed),"texts_embedded":len(texts)}))
