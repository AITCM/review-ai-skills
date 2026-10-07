from __future__ import annotations
import json,re,unicodedata,pathlib
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e080-n8-output");OUT.mkdir(exist_ok=True)
SEED=20261007;KS=(1,3,10);BUDGETS=(50,100,200);RRF_C=60.0
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
def mask_points(r,d):
 dd=nw(d);out=[]
 for x in split_reason(r):
  xx=nw(x)
  if dd and dd in xx:xx=xx.replace(dd," diagnosismask ")
  out.append(xx)
 return out
def f1(rec,prec):
 den=rec+prec
 return np.where(den>0,2*rec*prec/den,0.0)
def rank(s):
 ids=np.arange(len(s));return np.lexsort((ids,-np.asarray(s)))
def rrf_rank(*orders):
 n=len(orders[0]);score=np.zeros(n,np.float64)
 for order in orders:
  pos=np.empty(n,np.int32);pos[order]=np.arange(n,dtype=np.int32);score+=1/(RRF_C+pos+1)
 return rank(score)
def set_u(qmax,psum,cnt,ids):
 ids=np.asarray(ids,int);rec=float(np.max(qmax[:,ids],axis=1).mean());prec=float(psum[ids].sum()/cnt[ids].sum())
 return float(2*rec*prec/(rec+prec)) if rec+prec else 0.
def greedy(qmax,psum,cnt,pool,kmax=10):
 pool=np.asarray(pool,int);alive=np.ones(len(pool),bool);recvec=np.zeros(qmax.shape[0],np.float32);ps=0.;pc=0;out={}
 for step in range(1,min(kmax,len(pool))+1):
  recs=np.maximum(recvec[:,None],qmax[:,pool]).mean(0);precs=(ps+psum[pool])/(pc+cnt[pool]);vals=f1(recs,precs);vals[~alive]=-np.inf
  p=int(np.lexsort((pool,-vals))[0]);j=int(pool[p]);alive[p]=False;recvec=np.maximum(recvec,qmax[:,j]);ps+=float(psum[j]);pc+=int(cnt[j])
  if step in KS:out[step]=float(vals[p])
 return out
def boot_ci(x,seed,n=3000):
 x=np.asarray(x,float);rng=np.random.default_rng(seed);z=np.empty(n)
 for t in range(n):
  ii=rng.integers(0,len(x),len(x));z[t]=x[ii].mean()
 return [float(np.percentile(z,2.5)),float(np.percentile(z,97.5))]

tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
tp=[mask_points(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
vp=[mask_points(r,d) for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
tr_rdoc=[" ".join(x) for x in tp]

# Standard lexical case-to-case route.
cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
XC=cv.fit_transform(tr.case_prompt.astype(str));QV=cv.transform(va.case_prompt.astype(str))
LEX=(QV@XC.T).toarray().astype(np.float32)

# Direct cross-modal shared-vocabulary route:
# validation/new query is case_prompt; indexed target is historical masked diagnostic reasoning.
joint_corpus=tr.case_prompt.astype(str).tolist()+tr_rdoc
xv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=140000,dtype=np.float32)
xv.fit(joint_corpus)
QX=xv.transform(va.case_prompt.astype(str));RX=xv.transform(tr_rdoc)
XMOD=(QX@RX.T).toarray().astype(np.float32)

# Formal evaluator unchanged and fit only on train reasoning points.
flat=[p for ps in tp for p in ps]
ev=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
T=ev.fit_transform(flat);V=ev.transform([p for ps in vp for p in ps]);Tt=T.T.tocsr()
cnt=np.asarray([len(x) for x in tp],np.int32);starts=np.concatenate(([0],np.cumsum(cnt)[:-1])).astype(np.int64)
vo=[];o=0
for ps in vp:vo.append((o,o+len(ps)));o+=len(ps)
allids=np.arange(len(tr));train_dx=[nw(x) for x in tr.final_diagnosis];val_dx=[nw(x) for x in va.final_diagnosis];dxset=set(train_dx)

routes=["LEX","XMOD","RRF_LEX_XMOD"]
ordered={m:{k:[] for k in KS} for m in routes}
poolstats={m:{B:{"contain":[],"oracle":{k:[] for k in KS},"exact":[]} for B in BUDGETS} for m in routes}
globalor={k:[] for k in KS};rows=[]
for i in range(len(va)):
 lex=rank(LEX[i]);xm=rank(XMOD[i]);fus=rrf_rank(lex,xm);orders={"LEX":lex,"XMOD":xm,"RRF_LEX_XMOD":fus}
 s,e=vo[i];S=(V[s:e]@Tt).toarray().astype(np.float32,copy=False);qmax=np.stack([np.maximum.reduceat(row,starts) for row in S],0);pmax=S.max(0);psum=np.add.reduceat(pmax,starts);u=f1(qmax.mean(0),psum/cnt)
 gb=int(np.lexsort((allids,-u))[0]);gg=greedy(qmax,psum,cnt,allids,10)
 for k in KS:globalor[k].append(gg[k])
 row={"query_index":i,"global_best_single":gb}
 for name,order in orders.items():
  for k in KS:ordered[name][k].append(set_u(qmax,psum,cnt,order[:k]))
  for B in BUDGETS:
   pool=order[:B];z=poolstats[name][B];z["contain"].append(gb in set(pool.tolist()));gp=greedy(qmax,psum,cnt,pool,10)
   for k in KS:z["oracle"][k].append(gp[k])
   z["exact"].append(any(train_dx[j]==val_dx[i] for j in pool) if val_dx[i] in dxset else False)
  row[f"{name}_xmod_score_at1"]=float(XMOD[i,order[0]])
 rows.append(row)
 if (i+1)%50==0:print(f"VALIDATION {i+1}/{len(va)}",flush=True)

pd.DataFrame(rows).to_csv(OUT/"E080_N8_PER_QUERY.csv",index=False)
summary={"experiment":"E080-N8 direct sparse cross-modal case-to-reasoning retrieval",
 "status":"validation-development only; test untouched","dataset_revision":REV,
 "n_train":len(tr),"n_validation":len(va),
 "route":"shared-vocabulary TF-IDF: new case_prompt -> historical diagnosis-masked diagnostic_reasoning",
 "methods":{},"global_greedy_oracle":{str(k):float(np.mean(globalor[k])) for k in KS},
 "policy":"New-query input is case_prompt only. Historical diagnosis is not indexed; exact normalized diagnosis phrase is masked from reasoning. Validation reasoning is evaluation only. Test/T001 untouched."}
elig=np.asarray([d in dxset for d in val_dx],bool)
for name in routes:
 x={"ordered":{},"candidate":{}}
 for k in KS:
  arr=np.asarray(ordered[name][k]);base=np.asarray(ordered["LEX"][k]);d=arr-base
  x["ordered"][str(k)]={"softF1":float(arr.mean()),"delta_vs_LEX":float(d.mean()),"ci95":boot_ci(d,SEED+k+len(name))}
 for B in BUDGETS:
  z=poolstats[name][B];q={"global_best_containment_rate":float(np.mean(z["contain"]))}
  ex=np.asarray(z["exact"],bool);q["exact_label_recall_representable"]=float(ex[elig].mean())
  for k in KS:q[f"oracle_{k}"]=float(np.mean(z["oracle"][k]))
  x["candidate"][str(B)]=q
 summary["methods"][name]=x
(OUT/"E080_N8_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
md=["# E080-N8 Direct Sparse Cross-modal Retrieval","",
"Validation-development only; MedCaseReasoning test untouched.","",
"## Ordered retrieval","",
"| method | @1 | @3 | @10 |","|---|---:|---:|---:|"]
for m in routes:
 x=summary["methods"][m]["ordered"];md.append(f"| {m} | {x['1']['softF1']:.5f} | {x['3']['softF1']:.5f} | {x['10']['softF1']:.5f} |")
md += ["","## Candidate100","",
"| method | global-best recall | oracle@1 | oracle@3 | oracle@10 |","|---|---:|---:|---:|---:|"]
for m in routes:
 x=summary["methods"][m]["candidate"]["100"];md.append(f"| {m} | {100*x['global_best_containment_rate']:.1f}% | {x['oracle_1']:.5f} | {x['oracle_3']:.5f} | {x['oracle_10']:.5f} |")
(OUT/"E080_N8_REPORT.md").write_text("\n".join(md)+"\n")
print(json.dumps(summary,indent=2))
