from __future__ import annotations
import json,re,unicodedata,pathlib,urllib.request,zipfile
import numpy as np,pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer

MEDR_COMMIT="ff60ab440afd2f2bc0c603b3a65d715ec83138a7"
MEDR_URL=f"https://raw.githubusercontent.com/MAGIC-AI4Med/MedRBench/{MEDR_COMMIT}/data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json"
MC_REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
HF=f"https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/{MC_REV}/data"
OUT=pathlib.Path("e081-char");OUT.mkdir(exist_ok=True);RNG=np.random.default_rng(20261007)
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
def pts_mc(r,d):
 dd=nw(d);out=[]
 for p in split_reason(r):
  x=nw(p)
  if dd and dd in x:x=x.replace(dd," diagnosismask ")
  out.append(x)
 return out
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
 S=(q@c.T).toarray();r=float(S.max(1).mean());p=float(S.max(0).mean())
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
tp=[pts_mc(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
vp=[]
for q in queries:
 g=data[q["pmcid"]]["generate_case"]
 vp.append(mask_final(split_medr(g["differential_diagnosis"]),g["final_diagnosis"]))

flat=[p for z in tp for p in z]
vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),lowercase=True,sublinear_tf=True,
                    min_df=2,max_df=.995,max_features=150000,dtype=np.float32)
T=vec.fit_transform(flat);V=vec.transform([p for z in vp for p in z])
offs=[];o=0
for z in tp:offs.append((o,o+len(z)));o+=len(z)
vo=[];o=0
for z in vp:vo.append((o,o+len(z)));o+=len(z)
def score(i,cands):
 s,e=vo[i];q=V[s:e];m=[]
 for j in cands:
  a,b=offs[int(j)];m.append(T[a:b])
 return sf(q,vstack(m))
summary={"experiment":"E081 independent external char-ngram evaluator","n_external":840,
 "evaluator":{"analyzer":"char_wb","ngram_range":[3,5],"max_features":150000},"results":{},
 "policy":"Frozen E081 rankings only; no reranking."}
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
   "improved":int((d>0).sum()),"worsened":int((d<0).sum()),"tied":int((d==0).sum())}
(OUT/"E081_CHAR_RESULTS.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
