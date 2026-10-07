from __future__ import annotations
import json,re,unicodedata,pathlib,hashlib,random
import numpy as np,pandas as pd,torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e080-n5b-output");OUT.mkdir(exist_ok=True)
SEED=20261007;KS=(1,3,10);CAND_K=100;TOP_POS=5;NEG_PER_POS=4;LEX_HARD_K=200;RRF_C=60.0
PAT=re.compile(r'^\s*(?:\((\d+)\)|(\d+)[.)])\s*',re.M)
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)

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
def rrf_rank(a,b):
 n=len(a);score=np.zeros(n,np.float64)
 for order in (a,b):
  pos=np.empty(n,np.int32);pos[order]=np.arange(n,dtype=np.int32);score+=1/(RRF_C+pos+1)
 return rank(score)
def set_u(qmax,psum,counts,ids):
 ids=np.asarray(ids,int);rec=float(np.max(qmax[:,ids],axis=1).mean());prec=float(psum[ids].sum()/counts[ids].sum())
 return float(2*rec*prec/(rec+prec)) if rec+prec else 0.
def greedy(qmax,psum,counts,pool,kmax=10):
 pool=np.asarray(pool,int);alive=np.ones(len(pool),bool);recvec=np.zeros(qmax.shape[0],np.float32);ps=0.;pc=0;scores={}
 for step in range(1,min(kmax,len(pool))+1):
  recs=np.maximum(recvec[:,None],qmax[:,pool]).mean(0);precs=(ps+psum[pool])/(pc+counts[pool]);vals=f1(recs,precs);vals[~alive]=-np.inf
  pos=int(np.lexsort((pool,-vals))[0]);j=int(pool[pos]);alive[pos]=False;recvec=np.maximum(recvec,qmax[:,j]);ps+=float(psum[j]);pc+=int(counts[j])
  if step in KS:scores[step]=float(vals[pos])
 return scores
def boot_ci(x,seed,n=3000):
 x=np.asarray(x,float);rng=np.random.default_rng(seed);z=np.empty(n)
 for t in range(n):
  ii=rng.integers(0,len(x),len(x));z[t]=x[ii].mean()
 return [float(np.percentile(z,2.5)),float(np.percentile(z,97.5))]

class QT(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Linear(128,64))
 def forward(self,x):return F.normalize(self.net(x),dim=-1)
class CT(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(384,160),nn.ReLU(),nn.Linear(160,64))
 def forward(self,x):return F.normalize(self.net(x),dim=-1)

tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
isdev=np.array([int(hashlib.sha256(("E080-N5B:"+str(x)).encode()).hexdigest()[:8],16)%5==0 for x in tr.pmcid])
ref=tr.loc[~isdev].reset_index(drop=True);mq=tr.loc[isdev].reset_index(drop=True)
hold=np.array([int(hashlib.sha256(("E080-N5B-HOLD:"+str(x)).encode()).hexdigest()[:8],16)%5==0 for x in mq.pmcid])
fit_idx=np.where(~hold)[0];hold_idx=np.where(hold)[0]

cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
XR=cv.fit_transform(ref.case_prompt.astype(str));XMQ=cv.transform(mq.case_prompt.astype(str));XV=cv.transform(va.case_prompt.astype(str));XALL=cv.transform(tr.case_prompt.astype(str))
sv=TruncatedSVD(256,n_iter=7,random_state=SEED);CR=sv.fit_transform(XR).astype(np.float32);CMQ=sv.transform(XMQ).astype(np.float32);CV=sv.transform(XV).astype(np.float32);CALL=sv.transform(XALL).astype(np.float32)
CRN=normalize(CR);CMQN=normalize(CMQ);CVN=normalize(CV);CALLN=normalize(CALL)
rp=[mask_points(r,d) for r,d in zip(ref.diagnostic_reasoning,ref.final_diagnosis)]
mp=[mask_points(r,d) for r,d in zip(mq.diagnostic_reasoning,mq.final_diagnosis)]
tp=[mask_points(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
vp=[mask_points(r,d) for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
RR=rv.fit_transform([" ".join(x) for x in rp]);rsvd=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR=rsvd.fit_transform(RR).astype(np.float32);ZALL=rsvd.transform(rv.transform([" ".join(x) for x in tp])).astype(np.float32)
ZRN=normalize(ZR);ZALLN=normalize(ZALL);CREF=np.concatenate([CRN,ZRN],1).astype(np.float32);CALLF=np.concatenate([CALLN,ZALLN],1).astype(np.float32)

flat=[p for ps in rp for p in ps];uv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
RT=uv.fit_transform(flat);MQT=uv.transform([p for ps in mp for p in ps]);RTt=RT.T.tocsr()
cnt=np.asarray([len(x) for x in rp],np.int32);starts=np.concatenate(([0],np.cumsum(cnt)[:-1])).astype(np.int64)
mo=[];o=0
for ps in mp:mo.append((o,o+len(ps)));o+=len(ps)
LEXM=(XMQ@XR.T).toarray().astype(np.float32)

qidx=[];pidx=[];nidx=[];hold_utils={};hold_best={}
for i in range(len(mq)):
 s,e=mo[i];S=(MQT[s:e]@RTt).toarray().astype(np.float32,copy=False);qmax=np.stack([np.maximum.reduceat(row,starts) for row in S],0);pmax=S.max(0);psum=np.add.reduceat(pmax,starts);util=f1(qmax.mean(0),psum/cnt)
 uorder=np.lexsort((np.arange(len(ref)),-util))
 if hold[i]:
  hold_utils[i]=util.astype(np.float16);hold_best[i]=int(uorder[0]);continue
 pos=uorder[:TOP_POS];lexord=np.lexsort((np.arange(len(ref)),-LEXM[i]))[:LEX_HARD_K];hard=lexord[np.argsort(util[lexord])[:32]]
 for p in pos:
  cand=hard[hard!=p]
  if len(cand)<NEG_PER_POS:cand=uorder[-64:]
  for n in cand[:NEG_PER_POS]:
   qidx.append(i);pidx.append(int(p));nidx.append(int(n))
 if (i+1)%250==0:print(f"LABEL {i+1}/{len(mq)}",flush=True)

Q=torch.from_numpy(CMQ[np.asarray(qidx)]);P=torch.from_numpy(CREF[np.asarray(pidx)]);N=torch.from_numpy(CREF[np.asarray(nidx)])
loader=DataLoader(TensorDataset(Q,P,N),batch_size=2048,shuffle=True,generator=torch.Generator().manual_seed(SEED))
qt=QT();ct=CT();opt=torch.optim.AdamW(list(qt.parameters())+list(ct.parameters()),lr=1e-3,weight_decay=1e-4)
best=-1.;best_epoch=0;best_state=None;history=[]
hold_list=sorted(hold_utils)
HQ=torch.from_numpy(CMQ[hold_list])
for epoch in range(20):
 qt.train();ct.train();tot=0.;num=0
 for q,p,n in loader:
  opt.zero_grad();qs=qt(q);ps=ct(p);ns=ct(n);margin=(qs*ps).sum(1)-(qs*ns).sum(1);loss=F.softplus(-margin/.15).mean();loss.backward();opt.step();tot+=float(loss)*len(q);num+=len(q)
 qt.eval();ct.eval()
 with torch.no_grad():
  qh=qt(HQ).numpy();ch=ct(torch.from_numpy(CREF)).numpy();sc=qh@ch.T
 vals=[];rec=[]
 for row,i in enumerate(hold_list):
  o=rank(sc[row]);u=np.asarray(hold_utils[i],dtype=np.float32);vals.append(float(u[o[0]]));rec.append(hold_best[i] in set(o[:100].tolist()))
 metric=float(np.mean(vals));r100=float(np.mean(rec));history.append({"epoch":epoch+1,"train_loss":tot/num,"hold_utility_at1":metric,"hold_global_best_recall100":r100})
 print(f"EPOCH {epoch+1} loss={tot/num:.5f} hold_u1={metric:.5f} hold_r100={r100:.3f}",flush=True)
 if metric>best+1e-8:
  best=metric;best_epoch=epoch+1;best_state={"q":{k:v.detach().clone() for k,v in qt.state_dict().items()},"c":{k:v.detach().clone() for k,v in ct.state_dict().items()}}
 if epoch+1-best_epoch>=5:break
qt.load_state_dict(best_state["q"]);ct.load_state_dict(best_state["c"]);qt.eval();ct.eval()
with torch.no_grad():
 QVEM=qt(torch.from_numpy(CV)).numpy().astype(np.float32);CEM=ct(torch.from_numpy(CALLF)).numpy().astype(np.float32)
S5=(QVEM@CEM.T).astype(np.float32)

cvall=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
XA=cvall.fit_transform(tr.case_prompt.astype(str));QV=cvall.transform(va.case_prompt.astype(str));LEXS=(QV@XA.T).toarray().astype(np.float32)
flat=[p for ps in tp for p in ps];ev=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
T=ev.fit_transform(flat);V=ev.transform([p for ps in vp for p in ps]);Tt=T.T.tocsr();cnt2=np.asarray([len(x) for x in tp],np.int32);st=np.concatenate(([0],np.cumsum(cnt2)[:-1])).astype(np.int64)
vo=[];o=0
for ps in vp:vo.append((o,o+len(ps)));o+=len(ps)
allids=np.arange(len(tr));methods=["LEX","N5B","RRF_LEX_N5B"];ordered={m:{k:[] for k in KS} for m in methods};cand={m:{"contain":[],"oracle":{k:[] for k in KS}} for m in methods};glob={k:[] for k in KS}
for i in range(len(va)):
 lex=rank(LEXS[i]);n5=rank(S5[i]);fus=rrf_rank(lex,n5);routes={"LEX":lex,"N5B":n5,"RRF_LEX_N5B":fus}
 s,e=vo[i];SS=(V[s:e]@Tt).toarray().astype(np.float32,copy=False);qmax=np.stack([np.maximum.reduceat(row,st) for row in SS],0);pmax=SS.max(0);psum=np.add.reduceat(pmax,st);u=f1(qmax.mean(0),psum/cnt2);gb=int(np.lexsort((allids,-u))[0]);gg=greedy(qmax,psum,cnt2,allids,10)
 for k in KS:glob[k].append(gg[k])
 for m,o2 in routes.items():
  for k in KS:ordered[m][k].append(set_u(qmax,psum,cnt2,o2[:k]))
  pool=o2[:CAND_K];cand[m]["contain"].append(gb in set(pool.tolist()));gp=greedy(qmax,psum,cnt2,pool,10)
  for k in KS:cand[m]["oracle"][k].append(gp[k])
summary={"experiment":"E080-N5b nested-holdout direct utility metric","status":"validation-development only; test untouched",
 "training":{"meta_reference":len(ref),"meta_queries":len(mq),"fit_queries":int((~hold).sum()),"holdout_queries":int(hold.sum()),"pairs":len(qidx),"best_epoch":best_epoch,"history":history},
 "methods":{},"global_oracle":{str(k):float(np.mean(glob[k])) for k in KS},
 "policy":"Nested train-only holdout chooses epoch. Validation reasoning evaluation only. Test/T001 untouched."}
for m in methods:
 summary["methods"][m]={}
 for k in KS:
  a=np.asarray(ordered[m][k]);b=np.asarray(ordered["LEX"][k]);d=a-b;summary["methods"][m][str(k)]={"softF1":float(a.mean()),"delta_vs_LEX":float(d.mean()),"ci95":boot_ci(d,SEED+k+len(m))}
 summary["methods"][m]["candidate100"]={"global_best_containment_rate":float(np.mean(cand[m]["contain"])),"oracle10":float(np.mean(cand[m]["oracle"][10]))}
(OUT/"E080_N5B_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
torch.save({"query":qt.state_dict(),"candidate":ct.state_dict(),"best_epoch":best_epoch},OUT/"E080_N5B_MODEL.pt")
print(json.dumps(summary,indent=2))
