from __future__ import annotations
import json,re,unicodedata,pathlib
from collections import defaultdict
import numpy as np,pandas as pd
from sentence_transformers import SentenceTransformer

CRB_REV="3297c5b2db51872b70645a75f43dc0b849f77621"
CRB=f"https://huggingface.co/datasets/cxyzhang/caseReportBench_ClinicalDenseExtraction_Benchmark/resolve/{CRB_REV}/data/train-00000-of-00001.parquet?download=true"
MC_REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
HF=f"https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/{MC_REV}/data"
OUT=pathlib.Path("e060c");OUT.mkdir(exist_ok=True)
MODEL="pritamdeka/S-PubMedBert-MS-MARCO";MODEL_REV="96786c7024f95c5aac7f2b9a18086c7b97b23036"
SEED=20261004;RNG=np.random.default_rng(SEED)
CATS=["Vitals_Hema","GI","History","Neuro","Lab_Image","CVS","ENDO","GU","RESP","MSK","EENT","DERM","Pregnancy","LYMPH"]
GENERIC={"disease","syndrome","disorder","type","deficiency","defect","with","without","of","and","the","primary","secondary","congenital","acute","chronic"}
PAT=re.compile(r'^\s*(?:\((\d+)\)|(\d+)[.)])\s*',re.M)

def norm(s):return " ".join(re.findall(r"\w+",unicodedata.normalize("NFKC",str(s)).casefold()))
def flatten(v):
 if v is None:return ""
 if isinstance(v,np.ndarray):v=v.tolist()
 if isinstance(v,(list,tuple,set)):return " ".join(flatten(x) for x in v if flatten(x))
 if isinstance(v,dict):return " ".join(flatten(x) for x in v.values() if flatten(x))
 try:
  if bool(pd.isna(v)):return ""
 except Exception:pass
 return str(v).strip()
def as_items(v):
 if v is None:return []
 if isinstance(v,np.ndarray):v=v.tolist()
 if isinstance(v,(list,tuple,set)):return [flatten(x) for x in v if flatten(x)]
 if isinstance(v,dict):return [flatten(x) for x in v.values() if flatten(x)]
 try:
  if bool(pd.isna(v)):return []
 except Exception:pass
 s=str(v).strip();return [s] if s and s.lower() not in {"nan","none","[]","{}","null"} else []
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
def aggressive_mask(text,dx_items):
 x=str(text)
 for d in sorted(dx_items,key=len,reverse=True):
  if d.strip():x=re.sub(re.escape(d.strip())," diagnosismask ",x,flags=re.I)
 toks=set()
 for d in dx_items:toks|={t for t in norm(d).split() if len(t)>=4 and t not in GENERIC}
 return " ".join("diagnosismask" if t in toks else t for t in norm(x).split())
def reason_points(reason,dx):
 return [aggressive_mask(x,[str(dx)]) for x in split_reason(reason)]

# consume locked E060B rankings/cohorts only
cohfiles=list(pathlib.Path("locked").rglob("E060B_COHORT_LOCK.json"))
rankfiles=list(pathlib.Path("locked").rglob("E060B_BLIND_RANKINGS.jsonl"))
lockfiles=list(pathlib.Path("locked").rglob("E060B_RANKING_LOCK.json"))
assert len(cohfiles)==len(rankfiles)==len(lockfiles)==1
coh=json.load(open(cohfiles[0]));ranks=[json.loads(x) for x in open(rankfiles[0]) if x.strip()]
rl=json.load(open(lockfiles[0]));assert rl["status"]=="case_report_bench_rankings_locked_before_expert_facts_loaded"

df=pd.read_parquet(CRB)
tr=pd.read_parquet(HF+"/train-00000-of-00001.parquet",columns=["diagnostic_reasoning","final_diagnosis"])

# expert facts loaded only for evaluation
target={};texts=[];keys=[]
for i,r in df.iterrows():
 dx=as_items(r["Confirmed_Diagnosis(IEM)"]);target[i]={}
 for c in CATS:
  vals=[aggressive_mask(x,dx) for x in as_items(r[c])]
  vals=[x for x in vals if x.strip()];target[i][c]=[]
  for j,x in enumerate(vals):keys.append(("q",int(i),c,j));texts.append(x)
needed=set()
for r in ranks:
 for k in ("baseline_top50","method_top50"):needed.update(r[k][:10])
hist={}
for j in sorted(needed):
 ps=reason_points(tr.iloc[j].diagnostic_reasoning,tr.iloc[j].final_diagnosis);hist[j]=[]
 for z,p in enumerate(ps):keys.append(("h",j,"",z));texts.append(p)

model=SentenceTransformer(MODEL,revision=MODEL_REV,device="cpu");model.max_seq_length=256
E=model.encode(texts,batch_size=64,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True)
for vec,key in zip(E,keys):
 typ,i,c,j=key
 if typ=="q":target[i][c].append(vec)
 else:hist[i].append(vec)

def cat_score(i,cands,c):
 if not target[i][c]:return np.nan
 H=np.vstack([np.asarray(hist[j]) for j in cands])
 T=np.asarray(target[i][c]);S=T@H.T
 return float(np.max(S,axis=1).mean())

primary=np.array(coh["primary_confirmed_dx_no_exact_phrase"],dtype=int)
noexact=np.array(coh["secondary_no_exact_phrase"],dtype=int)
groups={"primary53":primary,"noexact69":noexact}
summary={"status":"posthoc_category_decomposition_on_locked_E060B_rankings",
 "dataset":"CaseReportBench","evaluator":{"model":MODEL,"revision":MODEL_REV},
 "note":"Interpretive sensitivity analysis only; not used to change E060B endpoint or method.","groups":{}}
per=[]
for g,ids in groups.items():
 summary["groups"][g]={}
 for k in (1,3,10):
  gout={}
  for c in CATS:
   a=[];b=[];caseids=[]
   for i in ids:
    x=cat_score(int(i),ranks[int(i)]["baseline_top50"][:k],c)
    y=cat_score(int(i),ranks[int(i)]["method_top50"][:k],c)
    if np.isfinite(x) and np.isfinite(y):
     a.append(x);b.append(y);caseids.append(int(i))
   if not a:continue
   a=np.asarray(a);b=np.asarray(b);d=b-a
   boots=[]
   for _ in range(5000):
    ix=RNG.integers(0,len(d),len(d));boots.append(float(d[ix].mean()))
   rec={"n":len(d),"TFIDF":float(a.mean()),"C2":float(b.mean()),"delta":float(d.mean()),
        "ci95":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))],
        "improved":int((d>0).sum()),"worsened":int((d<0).sum()),"tied":int((d==0).sum())}
   gout[c]=rec
   for ii,xx,yy in zip(caseids,a,b):per.append({"group":g,"k":k,"category":c,"query_index":ii,"TFIDF":float(xx),"C2":float(yy),"delta":float(yy-xx)})
  summary["groups"][g][str(k)]=gout
(OUT/"E060C_CATEGORY_RESULTS.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
pd.DataFrame(per).to_csv(OUT/"E060C_PER_CASE_CATEGORY.csv",index=False)
print(json.dumps(summary,indent=2,ensure_ascii=False))
