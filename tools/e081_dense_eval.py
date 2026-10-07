from __future__ import annotations
import json,re,unicodedata,pathlib,urllib.request,zipfile
import numpy as np,pandas as pd
from sentence_transformers import SentenceTransformer

MEDR_COMMIT="ff60ab440afd2f2bc0c603b3a65d715ec83138a7"
MEDR_URL=f"https://raw.githubusercontent.com/MAGIC-AI4Med/MedRBench/{MEDR_COMMIT}/data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json"
MC_REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
HF=f"https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/{MC_REV}/data"
MODEL="pritamdeka/S-PubMedBert-MS-MARCO";MODEL_REV="96786c7024f95c5aac7f2b9a18086c7b97b23036"
OUT=pathlib.Path("e081-dense");OUT.mkdir(exist_ok=True);RNG=np.random.default_rng(20261007)
PAT=re.compile(r'^\s*(?:\((\d+)\)|(\d+)[.)])\s*',re.M)
GENERIC={"with","without","acute","chronic","disease","syndrome","secondary","primary","type","and","the","of","disorder"}
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
def mask_mc_point(p,d):
 x=nw(p);dd=nw(d);return x.replace(dd," diagnosismask ") if dd and dd in x else x
def split_medr(s):
 out=[]
 for x in re.split(r'(?m)(?=^\s*\d+\.\s+)',str(s)):
  x=x.strip()
  if not x:continue
  x=re.sub(r'^\d+\.\s*','',x);x=re.sub(r'^\*\*[^*]{1,160}\*\*\s*:\s*','',x);x=nw(x)
  if x:out.append(x)
 return out or [nw(s)]
def mask_final(ps,dx):
 toks={t for t in nw(dx).split() if len(t)>=4 and t not in GENERIC}
 return [" ".join("diagnosismask" if t in toks else t for t in p.split()) for p in ps]
def sf(q,c):
 S=q@c.T;r=float(S.max(1).mean());p=float(S.max(0).mean())
 return 2*r*p/(r+p) if r+p else 0.

with zipfile.ZipFile("e081_artifact.zip") as z:z.extractall("e081")
root=pathlib.Path("e081/e081-n8-medr")
lock=json.load(open(root/"E081_RANKING_LOCK.json"))
assert lock["status"]=="E081_external_rankings_locked_before_outcomes_loaded"
ranks=[json.loads(x) for x in open(root/"E081_BLIND_RANKINGS.jsonl")]
queries=[json.loads(x) for x in open(root/"E081_QUERY_ONLY.jsonl")]
assert len(ranks)==len(queries)==840

req=urllib.request.Request(MEDR_URL,headers={"User-Agent":"CASE-EVID-001/1.0"})
with urllib.request.urlopen(req,timeout=180) as rr:data=json.loads(rr.read())
tr=pd.read_parquet(HF+"/train-00000-of-00001.parquet",columns=["diagnostic_reasoning","final_diagnosis"])

needed=set()
for r in ranks:
 for key in ("baseline_top50","xmod_top50","method_top50"):needed.update(r[key][:10])
texts=[];keys=[];train_pts={}
for j in sorted(needed):
 ps=[mask_mc_point(p,tr.iloc[j].final_diagnosis) for p in split_reason(tr.iloc[j].diagnostic_reasoning)]
 train_pts[j]=[]
 for k,p in enumerate(ps):keys.append(("tr",j,k));texts.append(p)
ext_pts={}
for i,q in enumerate(queries):
 g=data[q["pmcid"]]["generate_case"];ps=mask_final(split_medr(g["differential_diagnosis"]),g["final_diagnosis"]);ext_pts[i]=[]
 for k,p in enumerate(ps):keys.append(("ex",i,k));texts.append(p)

model=SentenceTransformer(MODEL,revision=MODEL_REV,device="cpu");model.max_seq_length=256
E=model.encode(texts,batch_size=64,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True)
for z,key in zip(E,keys):
 typ,i,k=key;(train_pts if typ=="tr" else ext_pts)[i].append(z)
def score(i,cands):
 q=np.asarray(ext_pts[i]);c=np.vstack([np.asarray(train_pts[int(j)]) for j in cands])
 return sf(q,c)

summary={"experiment":"E081 independent external S-PubMedBERT evaluator","n_external":840,
 "model":{"name":MODEL,"revision":MODEL_REV,"max_seq_length":256},"results":{},
 "policy":"Frozen E081 rankings only; dense evaluator is post-ranking evaluation and cannot rerank."}
for k in (1,3,10):
 b=np.array([score(i,ranks[i]["baseline_top50"][:k]) for i in range(840)])
 x=np.array([score(i,ranks[i]["xmod_top50"][:k]) for i in range(840)])
 m=np.array([score(i,ranks[i]["method_top50"][:k]) for i in range(840)])
 for name,arr in [("XMOD",x),("N8_RRF",m)]:
  d=arr-b;boots=[]
  for _ in range(5000):
   ids=RNG.integers(0,840,840);boots.append(float(d[ids].mean()))
  summary["results"].setdefault(str(k),{})[name]={"baseline":float(b.mean()),"method":float(arr.mean()),
   "delta":float(d.mean()),"ci95":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))],
   "improved":int((d>1e-12).sum()),"worsened":int((d<-1e-12).sum()),"tied":int((np.abs(d)<=1e-12).sum())}
(OUT/"E081_DENSE_RESULTS.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
