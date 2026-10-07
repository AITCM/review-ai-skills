from __future__ import annotations
import json,re,unicodedata,pathlib,zipfile
import numpy as np,pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e080-n6c-char-output");OUT.mkdir(exist_ok=True)
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
 x=nw(p);dd=nw(d);return x.replace(dd," diagnosismask ") if dd and dd in x else x
def sf1(q,c):
 S=(q@c.T).toarray();rec=float(np.max(S,axis=1).mean());prec=float(np.max(S,axis=0).mean())
 return 2*rec*prec/(rec+prec) if rec+prec else 0.

with zipfile.ZipFile("n6c_artifact.zip") as z:z.extractall("n6c")
rows=[json.loads(x) for x in open("n6c/e080-n6c-output/E080_N6C_RANKINGS.jsonl")]
assert len(rows)==500
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["diagnostic_reasoning","final_diagnosis"]]
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["diagnostic_reasoning","final_diagnosis"]]
tp=[[mask_point(p,d) for p in split_reason(r)] for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
vp=[[mask_point(p,d) for p in split_reason(r)] for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
flat=[p for ps in tp for p in ps]
vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),lowercase=True,sublinear_tf=True,min_df=2,max_df=.995,max_features=150000,dtype=np.float32)
T=vec.fit_transform(flat);V=vec.transform([p for ps in vp for p in ps])
offs=[];o=0
for ps in tp:offs.append((o,o+len(ps)));o+=len(ps)
vo=[];o=0
for ps in vp:vo.append((o,o+len(ps)));o+=len(ps)
def score(i,cands):
 s,e=vo[i];q=V[s:e];m=[]
 for j in cands:
  a,b=offs[int(j)];m.append(T[a:b])
 return sf1(q,vstack(m))
summary={"experiment":"E080-N6C independent char-ngram reasoning evaluator","evaluator":{"analyzer":"char_wb","ngram_range":[3,5],"max_features":150000},
 "n_validation":500,"results":{},"policy":"Frozen N6C rankings only; no reranking or model selection."}
for k in (1,3,10):
 b=np.array([score(i,rows[i]["lexical_top100"][:k]) for i in range(500)])
 m=np.array([score(i,rows[i]["n6_selected10"][:k]) for i in range(500)])
 d=m-b;boots=[]
 for _ in range(5000):
  ids=RNG.integers(0,500,500);boots.append(float(d[ids].mean()))
 summary["results"][str(k)]={"lexical":float(b.mean()),"N6C":float(m.mean()),"delta":float(d.mean()),
  "ci95":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))],
  "improved":int((d>1e-12).sum()),"worsened":int((d<-1e-12).sum()),"tied":int((np.abs(d)<=1e-12).sum())}
(OUT/"E080_N6C_CHAR.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
