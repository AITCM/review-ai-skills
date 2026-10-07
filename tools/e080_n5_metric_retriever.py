from __future__ import annotations
import json,re,unicodedata,pathlib,hashlib,random
import numpy as np,pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e080-n5-output");OUT.mkdir(exist_ok=True)
SEED=20261007;KS=(1,3,10);CAND_K=100;RRF_C=60.0
TOP_POS=5;NEG_PER_POS=4;LEX_HARD_K=200
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
def rank(scores):
    ids=np.arange(len(scores),dtype=int)
    return np.lexsort((ids,-np.asarray(scores)))
def rrf_rank(a,b):
    n=len(a);score=np.zeros(n,dtype=np.float64)
    for order in (a,b):
        pos=np.empty(n,dtype=np.int32);pos[order]=np.arange(n,dtype=np.int32)
        score+=1.0/(RRF_C+pos+1.0)
    return rank(score)
def set_u(qmax,psum,counts,ids):
    ids=np.asarray(ids,dtype=int)
    rec=float(np.max(qmax[:,ids],axis=1).mean());prec=float(psum[ids].sum()/counts[ids].sum())
    return float(2*rec*prec/(rec+prec)) if rec+prec else 0.
def greedy(qmax,psum,counts,pool,kmax=10):
    pool=np.asarray(pool,dtype=int);alive=np.ones(len(pool),bool)
    recvec=np.zeros(qmax.shape[0],np.float32);ps=0.;pc=0;scores={}
    for step in range(1,min(kmax,len(pool))+1):
        recs=np.maximum(recvec[:,None],qmax[:,pool]).mean(axis=0)
        precs=(ps+psum[pool])/(pc+counts[pool]);vals=f1(recs,precs);vals[~alive]=-np.inf
        pos=int(np.lexsort((pool,-vals))[0]);j=int(pool[pos]);alive[pos]=False
        recvec=np.maximum(recvec,qmax[:,j]);ps+=float(psum[j]);pc+=int(counts[j])
        if step in KS:scores[step]=float(vals[pos])
    return scores
def boot_ci(x,seed,n=3000):
    x=np.asarray(x,float);rng=np.random.default_rng(seed);z=np.empty(n,float)
    for t in range(n):
        ii=rng.integers(0,len(x),len(x));z[t]=x[ii].mean()
    return [float(np.percentile(z,2.5)),float(np.percentile(z,97.5))]

class QueryTower(nn.Module):
    def __init__(self,din=256,dout=64):
        super().__init__();self.net=nn.Sequential(nn.Linear(din,128),nn.ReLU(),nn.Linear(128,dout))
    def forward(self,x):return F.normalize(self.net(x),dim=-1)
class CandTower(nn.Module):
    def __init__(self,din=384,dout=64):
        super().__init__();self.net=nn.Sequential(nn.Linear(din,160),nn.ReLU(),nn.Linear(160,dout))
    def forward(self,x):return F.normalize(self.net(x),dim=-1)

tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
assert len(tr)==13092 and len(va)==500

isdev=np.array([int(hashlib.sha256(("E080-N5:"+str(x)).encode()).hexdigest()[:8],16)%5==0 for x in tr.pmcid])
ref=tr.loc[~isdev].reset_index(drop=True);mq=tr.loc[isdev].reset_index(drop=True)
assert len(ref)+len(mq)==len(tr)

# Case latent space on train meta-reference.
cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
XR=cv.fit_transform(ref.case_prompt.astype(str));XMQ=cv.transform(mq.case_prompt.astype(str));XV=cv.transform(va.case_prompt.astype(str));XALL=cv.transform(tr.case_prompt.astype(str))
csvd=TruncatedSVD(256,n_iter=7,random_state=SEED)
CR=csvd.fit_transform(XR).astype(np.float32);CMQ=csvd.transform(XMQ).astype(np.float32);CV=csvd.transform(XV).astype(np.float32);CALL=csvd.transform(XALL).astype(np.float32)
CRN=normalize(CR);CMQN=normalize(CMQ);CVN=normalize(CV);CALLN=normalize(CALL)

# Masked historical reasoning latent space.
rp=[mask_points(r,d) for r,d in zip(ref.diagnostic_reasoning,ref.final_diagnosis)]
mp=[mask_points(r,d) for r,d in zip(mq.diagnostic_reasoning,mq.final_diagnosis)]
tp=[mask_points(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
vp=[mask_points(r,d) for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
RR=rv.fit_transform([" ".join(x) for x in rp]);rsvd=TruncatedSVD(128,n_iter=7,random_state=SEED)
ZR=rsvd.fit_transform(RR).astype(np.float32);ZALL=rsvd.transform(rv.transform([" ".join(x) for x in tp])).astype(np.float32)
ZRN=normalize(ZR);ZALLN=normalize(ZALL)
CREF=np.concatenate([CRN,ZRN],axis=1).astype(np.float32)
CALLF=np.concatenate([CALLN,ZALLN],axis=1).astype(np.float32)

# Train-only reasoning utility labels for meta queries vs meta-reference.
flat_r=[p for ps in rp for p in ps];flat_m=[p for ps in mp for p in ps]
uv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
RT=uv.fit_transform(flat_r);MQT=uv.transform(flat_m);RTt=RT.T.tocsr()
rcount=np.asarray([len(x) for x in rp],np.int32);rstarts=np.concatenate(([0],np.cumsum(rcount)[:-1])).astype(np.int64)
mo=[];o=0
for ps in mp:mo.append((o,o+len(ps)));o+=len(ps)

# Lexical hard-negative route in the meta split.
LEX_META=(XMQ@XR.T).toarray().astype(np.float32)
q_idx=[];p_idx=[];n_idx=[];train_stats=[]

batches=[];qa=0
while qa<len(mq):
    qb=qa;npts=0
    while qb<len(mq):
        n=mo[qb][1]-mo[qb][0]
        if qb>qa and npts+n>192:break
        npts+=n;qb+=1
    batches.append((qa,qb));qa=qb

rng=np.random.default_rng(SEED)
for bno,(qa,qb) in enumerate(batches,1):
    p0=mo[qa][0];p1=mo[qb-1][1]
    SB=(MQT[p0:p1]@RTt).toarray().astype(np.float32,copy=False)
    for i in range(qa,qb):
        s,e=mo[i];S=SB[s-p0:e-p0]
        qmax=np.stack([np.maximum.reduceat(row,rstarts) for row in S],axis=0)
        pmax=S.max(axis=0);psum=np.add.reduceat(pmax,rstarts)
        util=f1(qmax.mean(axis=0),psum/rcount)
        uorder=np.lexsort((np.arange(len(ref)),-util));pos=uorder[:TOP_POS]
        lex_order=np.lexsort((np.arange(len(ref)),-LEX_META[i]))[:LEX_HARD_K]
        # hard negatives: lexically close but low utility; fallback to global low utility.
        hard=lex_order[np.argsort(util[lex_order])[:max(20,TOP_POS*NEG_PER_POS)]]
        global_low=uorder[-max(100,TOP_POS*NEG_PER_POS*4):]
        neg_pool=np.unique(np.concatenate([hard,global_low]))
        for p in pos:
            # sample a fixed number of negatives, favor hard negatives deterministically.
            candidates=hard[hard!=p]
            if len(candidates)<NEG_PER_POS:
                candidates=neg_pool[neg_pool!=p]
            take=candidates[:NEG_PER_POS] if len(candidates)>=NEG_PER_POS else rng.choice(neg_pool[neg_pool!=p],NEG_PER_POS,replace=True)
            for n in take:
                q_idx.append(i);p_idx.append(int(p));n_idx.append(int(n))
        train_stats.append({"best":float(util[pos[0]]),"pos5_mean":float(util[pos].mean()),"hardneg_mean":float(util[hard].mean())})
    print(f"LABEL BATCH {bno}/{len(batches)} {qa}:{qb}",flush=True)

q_idx=np.asarray(q_idx,np.int64);p_idx=np.asarray(p_idx,np.int64);n_idx=np.asarray(n_idx,np.int64)
QTRAIN=torch.from_numpy(CMQ[q_idx]);PTRAIN=torch.from_numpy(CREF[p_idx]);NTRAIN=torch.from_numpy(CREF[n_idx])
loader=DataLoader(TensorDataset(QTRAIN,PTRAIN,NTRAIN),batch_size=2048,shuffle=True,generator=torch.Generator().manual_seed(SEED))

qt=QueryTower();ct=CandTower();opt=torch.optim.AdamW(list(qt.parameters())+list(ct.parameters()),lr=1e-3,weight_decay=1e-4)
history=[]
for epoch in range(12):
    qt.train();ct.train();tot=0.;num=0
    for q,p,n in loader:
        opt.zero_grad()
        qs=qt(q);ps=ct(p);ns=ct(n)
        margin=(qs*ps).sum(1)-(qs*ns).sum(1)
        loss=F.softplus(-margin/0.15).mean()
        loss.backward();opt.step()
        tot+=float(loss.item())*len(q);num+=len(q)
    history.append(tot/num);print(f"EPOCH {epoch+1} loss={history[-1]:.6f}",flush=True)

qt.eval();ct.eval()
with torch.no_grad():
    QVEM=qt(torch.from_numpy(CV)).numpy().astype(np.float32)
    CEM=ct(torch.from_numpy(CALLF)).numpy().astype(np.float32)
N5S=(QVEM@CEM.T).astype(np.float32)

# Standard lexical comparator fitted on all train.
cvall=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
XA=cvall.fit_transform(tr.case_prompt.astype(str));QV=cvall.transform(va.case_prompt.astype(str))
LEXS=(QV@XA.T).toarray().astype(np.float32)

# Validation reasoning utility evaluator.
flat_t=[p for ps in tp for p in ps];flat_v=[p for ps in vp for p in ps]
ev=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
T=ev.fit_transform(flat_t);V=ev.transform(flat_v);Tt=T.T.tocsr()
counts=np.asarray([len(x) for x in tp],np.int32);starts=np.concatenate(([0],np.cumsum(counts)[:-1])).astype(np.int64)
vo=[];o=0
for ps in vp:vo.append((o,o+len(ps)));o+=len(ps)
all_ids=np.arange(len(tr),dtype=int)
train_dx=[nw(x) for x in tr.final_diagnosis];val_dx=[nw(x) for x in va.final_diagnosis];dxset=set(train_dx)

methods=["LEX","N5_METRIC","RRF_LEX_N5"]
ordered={m:{k:[] for k in KS} for m in methods}
cand={m:{"contain":[],"oracle":{k:[] for k in KS},"exact":[]} for m in methods}
global_or={k:[] for k in KS};rows=[]

for i in range(len(va)):
    lex=rank(LEXS[i]);n5=rank(N5S[i]);fusion=rrf_rank(lex,n5)
    routes={"LEX":lex,"N5_METRIC":n5,"RRF_LEX_N5":fusion}
    s,e=vo[i];S=(V[s:e]@Tt).toarray().astype(np.float32,copy=False)
    qmax=np.stack([np.maximum.reduceat(row,starts) for row in S],axis=0)
    pmax=S.max(axis=0);psum=np.add.reduceat(pmax,starts)
    util=f1(qmax.mean(axis=0),psum/counts);gbest=int(np.lexsort((all_ids,-util))[0])
    gg=greedy(qmax,psum,counts,all_ids,10)
    for k in KS:global_or[k].append(gg[k])
    row={"query_index":i,"global_best_single":gbest}
    for name,order in routes.items():
        for k in KS:ordered[name][k].append(set_u(qmax,psum,counts,order[:k]))
        pool=order[:CAND_K];cand[name]["contain"].append(gbest in set(int(x) for x in pool))
        gp=greedy(qmax,psum,counts,pool,10)
        for k in KS:cand[name]["oracle"][k].append(gp[k])
        ex=any(train_dx[j]==val_dx[i] for j in pool) if val_dx[i] in dxset else False
        cand[name]["exact"].append(ex);row[f"{name}_contains_global_best"]=bool(cand[name]["contain"][-1])
    rows.append(row)
    if (i+1)%50==0:print(f"VALIDATION {i+1}/{len(va)}",flush=True)

pd.DataFrame(rows).to_csv(OUT/"E080_N5_PER_QUERY.csv",index=False)
summary={
 "experiment":"E080-N5 direct utility metric retriever",
 "status":"validation-development only; MedCaseReasoning test untouched",
 "dataset_revision":REV,"n_train":len(tr),"n_validation":len(va),
 "training":{"meta_reference":len(ref),"meta_queries":len(mq),"pairs":int(len(q_idx)),
             "top_positive_per_query":TOP_POS,"negative_per_positive":NEG_PER_POS,"lexical_hard_negative_k":LEX_HARD_K,
             "query_features":"256-d case latent","candidate_features":"256-d case latent + 128-d masked-reasoning latent",
             "architecture":"two-tower MLP 256->128->64 and 384->160->64; cosine; pairwise softplus",
             "loss_history":[float(x) for x in history],
             "label_stats":{"mean_best":float(np.mean([x["best"] for x in train_stats])),
                            "mean_pos5":float(np.mean([x["pos5_mean"] for x in train_stats])),
                            "mean_hardneg":float(np.mean([x["hardneg_mean"] for x in train_stats]))}},
 "candidate_budget":CAND_K,"global_greedy_oracle":{str(k):float(np.mean(global_or[k])) for k in KS},"methods":{},
 "policy":"All supervision is from a deterministic train-only meta split. Validation query input is case_prompt only. Validation reasoning is evaluation only. Test/T001 is untouched."
}
elig=np.asarray([d in dxset for d in val_dx],bool)
for name in methods:
    x={"ordered":{},"candidate100":{}}
    for k in KS:
        arr=np.asarray(ordered[name][k]);base=np.asarray(ordered["LEX"][k]);d=arr-base
        x["ordered"][str(k)]={"softF1":float(arr.mean()),"delta_vs_LEX":float(d.mean()),"delta_ci95":boot_ci(d,SEED+k+len(name))}
        oc=np.asarray(cand[name]["oracle"][k])
        x["candidate100"][f"greedy_oracle_{k}"]=float(oc.mean())
        x["candidate100"][f"oracle_retention_{k}"]=float(oc.mean()/np.mean(global_or[k]))
    x["candidate100"]["global_best_containment_rate"]=float(np.mean(cand[name]["contain"]))
    ex=np.asarray(cand[name]["exact"],bool)
    x["candidate100"]["exact_label_recall_representable"]=float(ex[elig].mean()) if elig.any() else None
    summary["methods"][name]=x
(OUT/"E080_N5_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
torch.save({"query_tower":qt.state_dict(),"candidate_tower":ct.state_dict(),"seed":SEED},OUT/"E080_N5_MODEL.pt")
md=["# E080-N5 Direct Utility Metric Retriever","",
"Validation-development only; MedCaseReasoning test untouched.","",
"| method | @1 | @3 | @10 | candidate100 global-best | oracle@10 |",
"|---|---:|---:|---:|---:|---:|"]
for name in methods:
    x=summary["methods"][name]
    md.append(f"| {name} | {x['ordered']['1']['softF1']:.5f} | {x['ordered']['3']['softF1']:.5f} | {x['ordered']['10']['softF1']:.5f} | {100*x['candidate100']['global_best_containment_rate']:.1f}% | {x['candidate100']['greedy_oracle_10']:.5f} |")
md += ["","The learned metric is trained on pairwise train-only utility preferences. No validation reasoning enters fitting or pair construction."]
(OUT/"E080_N5_REPORT.md").write_text("\n".join(md)+"\n")
print(json.dumps(summary,indent=2))
