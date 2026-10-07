from __future__ import annotations
import json,re,unicodedata,pathlib,zipfile
import numpy as np,pandas as pd
from sentence_transformers import SentenceTransformer

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
MODEL="pritamdeka/S-PubMedBert-MS-MARCO";MODEL_REV="96786c7024f95c5aac7f2b9a18086c7b97b23036"
OUT=pathlib.Path("e080-n6-independent-output");OUT.mkdir(exist_ok=True)
RNG=np.random.default_rng(20261007)
PAT=re.compile(r'^\s*(?:\((\d+)\)|(\d+)[.)])\s*',re.M)

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
def mask_point(p,d):
    x=nw(p);dd=nw(d)
    return x.replace(dd," diagnosismask ") if dd and dd in x else x
def sf1(q,c):
    S=q@c.T
    rec=float(np.max(S,axis=1).mean());prec=float(np.max(S,axis=0).mean())
    return 2*rec*prec/(rec+prec) if rec+prec else 0.0

with zipfile.ZipFile("n6_artifact.zip") as z:z.extractall("n6")
path=pathlib.Path("n6/e080-n6-output/E080_N6_RANKINGS.jsonl")
rows=[json.loads(x) for x in open(path)]
assert len(rows)==500
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["diagnostic_reasoning","final_diagnosis"]]
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["diagnostic_reasoning","final_diagnosis"]]

needed=set()
for x in rows:
    needed.update(x["lexical_top100"][:10])
    needed.update(x["n6_selected10"][:10])

train_pts={};texts=[];keys=[]
for j in sorted(needed):
    ps=[mask_point(p,tr.iloc[j].final_diagnosis) for p in split_reason(tr.iloc[j].diagnostic_reasoning)]
    train_pts[j]=[]
    for k,p in enumerate(ps):keys.append(("tr",j,k));texts.append(p)
val_pts={}
for i in range(len(va)):
    ps=[mask_point(p,va.iloc[i].final_diagnosis) for p in split_reason(va.iloc[i].diagnostic_reasoning)]
    val_pts[i]=[]
    for k,p in enumerate(ps):keys.append(("va",i,k));texts.append(p)

model=SentenceTransformer(MODEL,revision=MODEL_REV,device="cpu");model.max_seq_length=256
E=model.encode(texts,batch_size=64,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True)
for z,key in zip(E,keys):
    typ,i,k=key
    (train_pts if typ=="tr" else val_pts)[i].append(z)

def score(i,cands):
    q=np.asarray(val_pts[i]);c=np.vstack([np.asarray(train_pts[int(j)]) for j in cands])
    return sf1(q,c)

summary={
 "experiment":"E080-N6 independent dense reasoning evaluator",
 "ranking_source":"immutable E080-N6 validation rankings; no reranking under dense evaluator",
 "model":{"name":MODEL,"revision":MODEL_REV,"max_seq_length":256},
 "n_validation":500,"results":{},
 "policy":"Dense evaluator is evaluation-only. No N6 model fitting or selection uses dense validation outcomes. Test/T001 untouched."
}
for k in (1,3,10):
    b=np.array([score(i,rows[i]["lexical_top100"][:k]) for i in range(500)])
    m=np.array([score(i,rows[i]["n6_selected10"][:k]) for i in range(500)])
    d=m-b;boots=[]
    for _ in range(5000):
        ids=RNG.integers(0,500,500);boots.append(float(d[ids].mean()))
    summary["results"][str(k)]={
      "lexical":float(b.mean()),"N6_set":float(m.mean()),"delta":float(d.mean()),
      "ci95":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))],
      "improved":int((d>1e-12).sum()),"worsened":int((d<-1e-12).sum()),"tied":int((np.abs(d)<=1e-12).sum())
    }
(OUT/"E080_N6_INDEPENDENT_DENSE.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
