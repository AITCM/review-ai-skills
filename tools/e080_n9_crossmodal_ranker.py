from __future__ import annotations
import json,re,unicodedata,pathlib,hashlib
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingRegressor

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e080-n9-output");OUT.mkdir(exist_ok=True)
SEED=20261007;KS=(1,3,10);TOPK=100
PAT=re.compile(r'^\s*(?:\((\d+)\)|(\d+)[.)])\s*',re.M)
FEATURES=["lex","xmod","lex_x_xmod","lex_rr","xmod_rr","rrf","route_count","score_spread"]
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
def rank(s,k=TOPK):
 ids=np.arange(len(s));return np.lexsort((ids,-np.asarray(s)))[:k]
def pool2(lex,xmod):
 lr=rank(lex);xr=rank(xmod);pool=np.asarray(sorted(set(lr)|set(xr)),dtype=int);return pool,lr,xr
def features(pool,lex,xmod,lr,xr):
 lp={int(j):r for r,j in enumerate(lr,1)};xp={int(j):r for r,j in enumerate(xr,1)};out=[]
 for j in pool:
  j=int(j);a=float(lex[j]);b=float(xmod[j]);r1=1/lp[j] if j in lp else 0.;r2=1/xp[j] if j in xp else 0.
  rr=sum(1/(60+r) for r in (lp.get(j),xp.get(j)) if r is not None);rc=int(j in lp)+int(j in xp)
  out.append([a,b,a*b,r1,r2,rr,rc,abs(a-b)])
 return np.asarray(out,np.float32)
def rrf_order(pool,F):
 ids=np.asarray(pool,int);s=F[:,5];return ids[np.lexsort((ids,-s))]
def utility_all(q,Tt,starts,cnt):
 S=(q@Tt).toarray().astype(np.float32,copy=False);qmax=np.stack([np.maximum.reduceat(row,starts) for row in S],0);pmax=S.max(0);psum=np.add.reduceat(pmax,starts)
 return f1(qmax.mean(0),psum/cnt),qmax,psum
def set_u(qmax,psum,cnt,ids):
 ids=np.asarray(ids,int);rec=float(np.max(qmax[:,ids],1).mean());prec=float(psum[ids].sum()/cnt[ids].sum());return 2*rec*prec/(rec+prec) if rec+prec else 0.
def greedy(qmax,psum,cnt,pool,kmax=10):
 pool=np.asarray(sorted(set(int(x) for x in pool)),int);alive=np.ones(len(pool),bool);rv=np.zeros(qmax.shape[0],np.float32);ps=0.;pc=0;out={}
 for step in range(1,min(kmax,len(pool))+1):
  rec=np.maximum(rv[:,None],qmax[:,pool]).mean(0);prec=(ps+psum[pool])/(pc+cnt[pool]);v=f1(rec,prec);v[~alive]=-np.inf;p=int(np.lexsort((pool,-v))[0]);j=int(pool[p]);alive[p]=False;rv=np.maximum(rv,qmax[:,j]);ps+=float(psum[j]);pc+=int(cnt[j])
  if step in KS:out[step]=float(v[p])
 return out
def boot(d,seed,n=3000):
 d=np.asarray(d,float);rng=np.random.default_rng(seed);z=np.empty(n)
 for t in range(n):
  ii=rng.integers(0,len(d),len(d));z[t]=d[ii].mean()
 return [float(np.percentile(z,2.5)),float(np.percentile(z,97.5))]

tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
isdev=np.array([int(hashlib.sha256(("E080-N9:"+str(x)).encode()).hexdigest()[:8],16)%5==0 for x in tr.pmcid])
ref=tr.loc[~isdev].reset_index(drop=True);mq=tr.loc[isdev].reset_index(drop=True)
rp=[mask_points(r,d) for r,d in zip(ref.diagnostic_reasoning,ref.final_diagnosis)]
mp=[mask_points(r,d) for r,d in zip(mq.diagnostic_reasoning,mq.final_diagnosis)]

# Meta split lexical case->case.
cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
XC=cv.fit_transform(ref.case_prompt.astype(str));QM=cv.transform(mq.case_prompt.astype(str));LEX=(QM@XC.T).toarray().astype(np.float32)
# Meta split direct case->masked reasoning cross-modal.
rdoc=[" ".join(x) for x in rp]
xv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=140000,dtype=np.float32)
xv.fit(ref.case_prompt.astype(str).tolist()+rdoc);QX=xv.transform(mq.case_prompt.astype(str));RX=xv.transform(rdoc);XMOD=(QX@RX.T).toarray().astype(np.float32)
# Train-only utility labels.
flat=[p for ps in rp for p in ps];ev=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
T=ev.fit_transform(flat);M=ev.transform([p for ps in mp for p in ps]);Tt=T.T.tocsr();cnt=np.asarray([len(x) for x in rp],np.int32);starts=np.concatenate(([0],np.cumsum(cnt)[:-1])).astype(np.int64)
mo=[];o=0
for ps in mp:mo.append((o,o+len(ps)));o+=len(ps)
Xs=[];Ys=[];sizes=[]
for i in range(len(mq)):
 pool,lr,xr=pool2(LEX[i],XMOD[i]);F=features(pool,LEX[i],XMOD[i],lr,xr);s,e=mo[i];u,_,_=utility_all(M[s:e],Tt,starts,cnt);Xs.append(F);Ys.append(u[pool]);sizes.append(len(pool))
 if (i+1)%250==0:print(f"META {i+1}/{len(mq)}",flush=True)
Xreg=np.vstack(Xs);Yreg=np.concatenate(Ys)
model=HistGradientBoostingRegressor(loss="squared_error",learning_rate=.05,max_iter=300,max_leaf_nodes=31,min_samples_leaf=40,l2_regularization=1.0,early_stopping=False,random_state=SEED).fit(Xreg,Yreg)

# Refit exact same representations on full train for validation inference.
tp=[mask_points(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)];vp=[mask_points(r,d) for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)];tdoc=[" ".join(x) for x in tp]
cv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32);XA=cv2.fit_transform(tr.case_prompt.astype(str));QV=cv2.transform(va.case_prompt.astype(str));LEXS=(QV@XA.T).toarray().astype(np.float32)
xv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=140000,dtype=np.float32);xv2.fit(tr.case_prompt.astype(str).tolist()+tdoc);QXV=xv2.transform(va.case_prompt.astype(str));RXA=xv2.transform(tdoc);XMS=(QXV@RXA.T).toarray().astype(np.float32)

flat=[p for ps in tp for p in ps];ee=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32);TT=ee.fit_transform(flat);VV=ee.transform([p for ps in vp for p in ps]);TTt=TT.T.tocsr();cnt2=np.asarray([len(x) for x in tp],np.int32);st=np.concatenate(([0],np.cumsum(cnt2)[:-1])).astype(np.int64)
vo=[];o=0
for ps in vp:vo.append((o,o+len(ps)));o+=len(ps)

methods=["LEX","RRF_LEX_XMOD","N9_HGB","UNION_ORACLE"];scores={m:{k:[] for k in KS} for m in methods};rankings=[];vsizes=[]
tl=[nw(x) for x in tr.final_diagnosis];vl=[nw(x) for x in va.final_diagnosis];exact={m:{k:0 for k in KS} for m in methods if m!="UNION_ORACLE"}
for i in range(len(va)):
 pool,lr,xr=pool2(LEXS[i],XMS[i]);F=features(pool,LEXS[i],XMS[i],lr,xr);vsizes.append(len(pool));rr=rrf_order(pool,F);sh=model.predict(F);hgb=pool[np.lexsort((pool,-sh))]
 s,e=vo[i];_,qmax,psum=utility_all(VV[s:e],TTt,st,cnt2);go=greedy(qmax,psum,cnt2,pool,10)
 orders={"LEX":lr,"RRF_LEX_XMOD":rr,"N9_HGB":hgb}
 for m,o2 in orders.items():
  for k in KS:
   scores[m][k].append(set_u(qmax,psum,cnt2,o2[:k]));exact[m][k]+=int(any(tl[int(j)]==vl[i] for j in o2[:k]))
 for k in KS:scores["UNION_ORACLE"][k].append(go[k])
 rankings.append({"query_index":i,"union_pool":pool.tolist(),"lex_ranked":lr.tolist(),"rrf_ranked":rr.tolist(),"n9_ranked":hgb.tolist()})
 if (i+1)%50==0:print(f"VALIDATION {i+1}/{len(va)}",flush=True)
with (OUT/"E080_N9_RANKINGS.jsonl").open("w") as f:
 for x in rankings:f.write(json.dumps(x)+"\n")
summary={"experiment":"E080-N9 learned cross-modal ranker","status":"validation-development only; test untouched","dataset_revision":REV,
 "candidate_generator":"union of lexical Top-100 and direct case->masked-reasoning XMOD Top-100",
 "training":{"meta_reference":len(ref),"meta_queries":len(mq),"rows":len(Yreg),"mean_pool_size":float(np.mean(sizes)),"features":FEATURES,
             "hgb":{"learning_rate":.05,"max_iter":300,"max_leaf_nodes":31,"min_samples_leaf":40,"l2_regularization":1.0}},
 "validation":{"mean_pool_size":float(np.mean(vsizes)),"methods":{}},"policy":"All learning train-only. Validation reasoning evaluation only. Test/T001 untouched."}
for m in methods:
 summary["validation"]["methods"][m]={}
 for k in KS:
  a=np.asarray(scores[m][k]);b=np.asarray(scores["LEX"][k]);rr=np.asarray(scores["RRF_LEX_XMOD"][k]);d=a-b;dr=a-rr
  summary["validation"]["methods"][m][str(k)]={"softF1":float(a.mean()),"delta_vs_LEX":float(d.mean()),"ci95_vs_LEX":boot(d,SEED+k+len(m)),"delta_vs_RRF":float(dr.mean()),"ci95_vs_RRF":boot(dr,SEED+100+k+len(m))}
  if m in exact:summary["validation"]["methods"][m][str(k)]["exact_label_hit"]=int(exact[m][k])
(OUT/"E080_N9_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
