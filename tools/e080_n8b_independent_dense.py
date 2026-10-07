from __future__ import annotations
import json,re,unicodedata,pathlib
import numpy as np,pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
MODEL="pritamdeka/S-PubMedBert-MS-MARCO";MODEL_REV="96786c7024f95c5aac7f2b9a18086c7b97b23036"
OUT=pathlib.Path("e080-n8b-output");OUT.mkdir(exist_ok=True)
SEED=20261007;RNG=np.random.default_rng(SEED)
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
def rank(s):
 ids=np.arange(len(s));return np.lexsort((ids,-np.asarray(s)))
def rrf_rank(a,b):
 n=len(a);score=np.zeros(n,np.float64)
 for order in (a,b):
  pos=np.empty(n,np.int32);pos[order]=np.arange(n,dtype=np.int32);score+=1/(60+pos+1)
 return rank(score)
def sf1(q,c):
 S=q@c.T;rec=float(np.max(S,axis=1).mean());prec=float(np.max(S,axis=0).mean())
 return 2*rec*prec/(rec+prec) if rec+prec else 0.
def boot(d,seed,n=3000):
 rng=np.random.default_rng(seed);d=np.asarray(d,float);z=np.empty(n)
 for t in range(n):
  ii=rng.integers(0,len(d),len(d));z[t]=d[ii].mean()
 return [float(np.percentile(z,2.5)),float(np.percentile(z,97.5))]

tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
tp=[[mask_point(p,d) for p in split_reason(r)] for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
vp=[[mask_point(p,d) for p in split_reason(r)] for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
tr_doc=[" ".join(x) for x in tp]

# Reconstruct N8 rankings deterministically.
cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
XC=cv.fit_transform(tr.case_prompt.astype(str));QV=cv.transform(va.case_prompt.astype(str))
LEX=(QV@XC.T).toarray().astype(np.float32)
xv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=140000,dtype=np.float32)
xv.fit(tr.case_prompt.astype(str).tolist()+tr_doc)
QX=xv.transform(va.case_prompt.astype(str));RX=xv.transform(tr_doc)
XMOD=(QX@RX.T).toarray().astype(np.float32)

rankings=[]
needed=set()
for i in range(len(va)):
 lex=rank(LEX[i]);xm=rank(XMOD[i]);fus=rrf_rank(lex,xm)
 row={"query_index":i,"LEX":lex[:10].tolist(),"XMOD":xm[:10].tolist(),"RRF_LEX_XMOD":fus[:10].tolist()}
 rankings.append(row)
 for k in ("LEX","XMOD","RRF_LEX_XMOD"):needed.update(row[k])
with (OUT/"E080_N8B_RANKINGS_TOP10.jsonl").open("w") as f:
 for x in rankings:f.write(json.dumps(x)+"\n")

# Embed only cases actually used in the top-10 comparisons.
train_pts={};texts=[];keys=[]
for j in sorted(needed):
 train_pts[int(j)]=[]
 for pno,p in enumerate(tp[int(j)]):
  keys.append(("tr",int(j),pno));texts.append(p)
val_pts={}
for i,ps in enumerate(vp):
 val_pts[i]=[]
 for pno,p in enumerate(ps):
  keys.append(("va",i,pno));texts.append(p)

model=SentenceTransformer(MODEL,revision=MODEL_REV,device="cpu");model.max_seq_length=256
E=model.encode(texts,batch_size=64,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True)
for z,key in zip(E,keys):
 typ,i,pno=key;(train_pts if typ=="tr" else val_pts)[i].append(z)
def score(i,cands):
 q=np.asarray(val_pts[i]);c=np.vstack([np.asarray(train_pts[int(j)]) for j in cands])
 return sf1(q,c)

methods=["LEX","XMOD","RRF_LEX_XMOD"];summary={
 "experiment":"E080-N8b independent dense evaluator of direct cross-modal retrieval",
 "status":"validation robustness only; test untouched",
 "model":{"name":MODEL,"revision":MODEL_REV,"max_seq_length":256},
 "n_validation":len(va),"unique_train_cases_embedded":len(needed),"reason_points_embedded":len(texts),
 "results":{},"policy":"Rankings use case_prompt only for the new query and masked historical reasoning. Dense model is evaluation only."
}
for k in (1,3,10):
 vals={m:np.asarray([score(i,rankings[i][m][:k]) for i in range(len(va))]) for m in methods}
 base=vals["LEX"]
 summary["results"][str(k)]={}
 for m in methods:
  d=vals[m]-base
  summary["results"][str(k)][m]={
   "mean":float(vals[m].mean()),"delta_vs_LEX":float(d.mean()),"ci95":boot(d,SEED+k+len(m)),
   "improved":int((d>0).sum()),"worsened":int((d<0).sum()),"tied":int((d==0).sum())
  }
(OUT/"E080_N8B_INDEPENDENT_DENSE.json").write_text(json.dumps(summary,indent=2)+"\n")
md=["# E080-N8b Independent Dense Evaluation","",
"Validation-only robustness check. The retrieval rankings are sparse TF-IDF based; evaluation uses frozen S-PubMedBERT reasoning-point embeddings.","",
"| k | LEX | XMOD | Δ XMOD | RRF LEX+XMOD | Δ RRF |","|---:|---:|---:|---:|---:|---:|"]
for k in (1,3,10):
 x=summary["results"][str(k)]
 md.append(f"| {k} | {x['LEX']['mean']:.5f} | {x['XMOD']['mean']:.5f} | {x['XMOD']['delta_vs_LEX']:+.5f} | {x['RRF_LEX_XMOD']['mean']:.5f} | {x['RRF_LEX_XMOD']['delta_vs_LEX']:+.5f} |")
(OUT/"E080_N8B_REPORT.md").write_text("\n".join(md)+"\n")
print(json.dumps(summary,indent=2))
