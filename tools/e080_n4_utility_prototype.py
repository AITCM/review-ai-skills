from __future__ import annotations
import json,re,unicodedata,pathlib,hashlib
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e080-n4-output");OUT.mkdir(exist_ok=True)
SEED=20261007; TOPMS=(5,10,20); KS=(1,3,10); CAND_K=100; RRF_C=60.0
PAT=re.compile(r'^\s*(?:\((\d+)\)|(\d+)[.)])\s*',re.M)

def nw(s):return " ".join(re.findall(r"\w+",unicodedata.normalize("NFKC",str(s)).casefold()))
def split_reason(s):
    s=str(s);ms=list(PAT.finditer(s))
    if not ms or s[:ms[0].start()].strip(): return [s.strip()]
    nums=[int(m.group(1) or m.group(2)) for m in ms]
    if nums!=list(range(1,len(ms)+1)): return [s.strip()]
    out=[]
    for i,m in enumerate(ms):
        e=ms[i+1].start() if i+1<len(ms) else len(s)
        x=s[m.end():e].strip()
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
def rank(scores,k=None):
    ids=np.arange(len(scores),dtype=int)
    o=np.lexsort((ids,-np.asarray(scores)))
    return o if k is None else o[:k]
def rrf_rank(*orders):
    n=len(orders[0]);score=np.zeros(n,dtype=np.float64)
    for order in orders:
        pos=np.empty(n,dtype=np.int32);pos[order]=np.arange(n,dtype=np.int32)
        score+=1.0/(RRF_C+pos+1.0)
    return rank(score)
def set_u(qmax,psum,counts,ids):
    ids=np.asarray(ids,dtype=int)
    rec=float(np.max(qmax[:,ids],axis=1).mean())
    prec=float(psum[ids].sum()/counts[ids].sum())
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
def ridge_map(X,Y,l2=10.0):
    xm=X.mean(0,keepdims=True);ym=Y.mean(0,keepdims=True)
    A=X-xm;B=Y-ym
    W=np.linalg.solve(A.T@A+l2*np.eye(A.shape[1],dtype=np.float32),A.T@B)
    return xm,ym,W
def predict_map(X,xm,ym,W):
    return normalize((X-xm)@W+ym)
def boot_ci(x,seed,n=3000):
    x=np.asarray(x,float);rng=np.random.default_rng(seed);z=np.empty(n,float)
    for b in range(n):
        ids=rng.integers(0,len(x),len(x));z[b]=x[ids].mean()
    return [float(np.percentile(z,2.5)),float(np.percentile(z,97.5))]

tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
assert len(tr)==13092 and len(va)==500

# Fixed train-only meta split. Validation is not used to learn any representation or weight.
isdev=np.array([int(hashlib.sha256(("E080-N4:"+str(x)).encode()).hexdigest()[:8],16)%5==0 for x in tr.pmcid])
ref=tr.loc[~isdev].reset_index(drop=True);mq=tr.loc[isdev].reset_index(drop=True)
assert len(ref)+len(mq)==len(tr)

# Case space fitted on meta-reference only.
cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
XR=cv.fit_transform(ref.case_prompt.astype(str));XMQ=cv.transform(mq.case_prompt.astype(str));XV=cv.transform(va.case_prompt.astype(str))
csvd=TruncatedSVD(256,n_iter=7,random_state=SEED)
CR=csvd.fit_transform(XR).astype(np.float32);CMQ=csvd.transform(XMQ).astype(np.float32);CV=csvd.transform(XV).astype(np.float32)

# Historical masked-reasoning document space, fitted on meta-reference only.
rp=[mask_points(r,d) for r,d in zip(ref.diagnostic_reasoning,ref.final_diagnosis)]
mp=[mask_points(r,d) for r,d in zip(mq.diagnostic_reasoning,mq.final_diagnosis)]
tp=[mask_points(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
vp=[mask_points(r,d) for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
RR=rv.fit_transform([" ".join(x) for x in rp])
rsvd=TruncatedSVD(128,n_iter=7,random_state=SEED)
ZR=rsvd.fit_transform(RR).astype(np.float32)
ZALL=rsvd.transform(rv.transform([" ".join(x) for x in tp])).astype(np.float32)
ZRn=normalize(ZR);ZALLn=normalize(ZALL)

# Train-only outcome utility labels: meta-query reasoning vs meta-reference reasoning.
flat_r=[p for ps in rp for p in ps];flat_m=[p for ps in mp for p in ps]
uv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
RT=uv.fit_transform(flat_r);MQT=uv.transform(flat_m);RTt=RT.T.tocsr()
rcount=np.asarray([len(x) for x in rp],np.int32);rstarts=np.concatenate(([0],np.cumsum(rcount)[:-1])).astype(np.int64)
mo=[];o=0
for ps in mp:mo.append((o,o+len(ps)));o+=len(ps)

targets={m:np.zeros((len(mq),ZR.shape[1]),dtype=np.float32) for m in TOPMS}
train_best=[]
batches=[];qa=0
while qa<len(mq):
    qb=qa;npnts=0
    while qb<len(mq):
        n=mo[qb][1]-mo[qb][0]
        if qb>qa and npnts+n>192:break
        npnts+=n;qb+=1
    batches.append((qa,qb));qa=qb

for bno,(qa,qb) in enumerate(batches,1):
    p0=mo[qa][0];p1=mo[qb-1][1]
    SB=(MQT[p0:p1]@RTt).toarray().astype(np.float32,copy=False)
    for i in range(qa,qb):
        s,e=mo[i];S=SB[s-p0:e-p0]
        qmax=np.stack([np.maximum.reduceat(row,rstarts) for row in S],axis=0)
        pmax=S.max(axis=0);psum=np.add.reduceat(pmax,rstarts)
        util=f1(qmax.mean(axis=0),psum/rcount)
        order=np.lexsort((np.arange(len(ref)),-util))
        train_best.append(float(util[order[0]]))
        for m in TOPMS:
            ids=order[:m];w=np.maximum(util[ids],1e-6)
            proto=(w[:,None]*ZRn[ids]).sum(axis=0)/w.sum()
            nrm=np.linalg.norm(proto)
            if nrm>0:proto=proto/nrm
            targets[m][i]=proto.astype(np.float32)
    print(f"UTILITY LABEL BATCH {bno}/{len(batches)} {qa}:{qb}",flush=True)

# Old idea: predict the query's own reasoning representation.
own_target=normalize(rsvd.transform(rv.transform([" ".join(x) for x in mp])).astype(np.float32))
own_xm,own_ym,own_W=ridge_map(CMQ,own_target,10.0)
OWNQ=predict_map(CV,own_xm,own_ym,own_W)
OWN_SCORE=(OWNQ@ZALLn.T).astype(np.float32)

# New idea: predict the utility-prototype representation.
UP_SCORE={};map_info={}
for m in TOPMS:
    xm,ym,W=ridge_map(CMQ,targets[m],10.0)
    QP=predict_map(CV,xm,ym,W)
    UP_SCORE[m]=(QP@ZALLn.T).astype(np.float32)
    pred=predict_map(CMQ,xm,ym,W)
    map_info[str(m)]={"mean_train_target_cosine":float(np.sum(pred*targets[m],axis=1).mean())}

# Standard all-train lexical route for validation comparison.
cvall=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
XA=cvall.fit_transform(tr.case_prompt.astype(str));QV=cvall.transform(va.case_prompt.astype(str))
LEX=(QV@XA.T).toarray().astype(np.float32)

# Formal validation utility evaluator fit on train reasoning points only.
flat_t=[p for ps in tp for p in ps];flat_v=[p for ps in vp for p in ps]
ev=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
T=ev.fit_transform(flat_t);V=ev.transform(flat_v);Tt=T.T.tocsr()
counts=np.asarray([len(x) for x in tp],np.int32);starts=np.concatenate(([0],np.cumsum(counts)[:-1])).astype(np.int64)
vo=[];o=0
for ps in vp:vo.append((o,o+len(ps)));o+=len(ps)
all_ids=np.arange(len(tr),dtype=int)
train_dx=[nw(x) for x in tr.final_diagnosis];val_dx=[nw(x) for x in va.final_diagnosis];dxset=set(train_dx)

route_names=["LEX","OWNPRED"]+[f"UP{m}" for m in TOPMS]+["RRF_LEX_UP10"]
ordered={name:{k:[] for k in KS} for name in route_names}
candidate={name:{"contain":[],"oracle":{k:[] for k in KS},"exact":[]} for name in route_names}
global_oracle={k:[] for k in KS};per=[]

for i in range(len(va)):
    lex=rank(LEX[i]);own=rank(OWN_SCORE[i]);ups={m:rank(UP_SCORE[m][i]) for m in TOPMS}
    fuse=rrf_rank(lex,ups[10])
    routes={"LEX":lex,"OWNPRED":own,**{f"UP{m}":ups[m] for m in TOPMS},"RRF_LEX_UP10":fuse}

    s,e=vo[i];S=(V[s:e]@Tt).toarray().astype(np.float32,copy=False)
    qmax=np.stack([np.maximum.reduceat(row,starts) for row in S],axis=0)
    pmax=S.max(axis=0);psum=np.add.reduceat(pmax,starts)
    util=f1(qmax.mean(axis=0),psum/counts)
    gbest=int(np.lexsort((all_ids,-util))[0]);gg=greedy(qmax,psum,counts,all_ids,10)
    for k in KS:global_oracle[k].append(gg[k])

    row={"query_index":i,"global_best_single":gbest}
    for name,order in routes.items():
        for k in KS:ordered[name][k].append(set_u(qmax,psum,counts,order[:k]))
        pool=order[:CAND_K];candidate[name]["contain"].append(gbest in set(int(x) for x in pool))
        gp=greedy(qmax,psum,counts,pool,10)
        for k in KS:candidate[name]["oracle"][k].append(gp[k])
        exact=any(train_dx[j]==val_dx[i] for j in pool) if val_dx[i] in dxset else False
        candidate[name]["exact"].append(exact);row[f"{name}_contains_global_best"]=bool(candidate[name]["contain"][-1])
    per.append(row)
    if (i+1)%50==0:print(f"VALIDATION {i+1}/{len(va)}",flush=True)

pd.DataFrame(per).to_csv(OUT/"E080_N4_PER_QUERY.csv",index=False)
summary={
 "experiment":"E080-N4 utility-prototype retriever",
 "status":"validation-development only; MedCaseReasoning test untouched",
 "dataset_revision":REV,"n_train":len(tr),"n_validation":len(va),
 "training":{"meta_reference":len(ref),"meta_queries":len(mq),"topM_variants":list(TOPMS),
             "mean_train_global_best_individual_utility":float(np.mean(train_best)),
             "prototype_map":map_info,
             "supervision":"train-only global masked-reasoning individual utility; no validation/test outcomes used to fit mappings"},
 "candidate_budget":CAND_K,"methods":{},
 "global_greedy_oracle":{str(k):float(np.mean(global_oracle[k])) for k in KS},
 "policy":"Validation case_prompt is the only new-query input. Historical train diagnostic_reasoning is diagnosis-masked and may be indexed. Validation reasoning is evaluation only. Test/T001 is not accessed."
}
elig=np.array([d in dxset for d in val_dx],bool)
for name in route_names:
    x={"ordered":{},"candidate100":{}}
    for k in KS:
        arr=np.asarray(ordered[name][k]);base=np.asarray(ordered["LEX"][k]);d=arr-base
        x["ordered"][str(k)]={"softF1":float(arr.mean()),"delta_vs_LEX":float(d.mean()),
                              "delta_ci95":boot_ci(d,SEED+k+len(name))}
        oc=np.asarray(candidate[name]["oracle"][k])
        x["candidate100"][f"greedy_oracle_{k}"]=float(oc.mean())
        x["candidate100"][f"oracle_retention_{k}"]=float(oc.mean()/np.mean(global_oracle[k]))
    x["candidate100"]["global_best_containment_rate"]=float(np.mean(candidate[name]["contain"]))
    ex=np.asarray(candidate[name]["exact"],bool)
    x["candidate100"]["exact_label_recall_representable"]=float(ex[elig].mean()) if elig.any() else None
    summary["methods"][name]=x
(OUT/"E080_N4_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
md=["# E080-N4 Utility-Prototype Retriever","",
"Validation-development only. Test/T001 remains untouched.","",
"| method | ordered @1 | @3 | @10 | candidate100 global-best | oracle@10 |",
"|---|---:|---:|---:|---:|---:|"]
for name in route_names:
    x=summary["methods"][name]
    md.append(f"| {name} | {x['ordered']['1']['softF1']:.5f} | {x['ordered']['3']['softF1']:.5f} | {x['ordered']['10']['softF1']:.5f} | {100*x['candidate100']['global_best_containment_rate']:.1f}% | {x['candidate100']['greedy_oracle_10']:.5f} |")
md += ["","## Key contrast",
"OWNPRED predicts the query's own reasoning representation. UP5/10/20 instead predict a train-only prototype formed from the reasoning representations of historically useful cases. This directly tests utility-targeted supervision.",
"",
"All prototype labels are constructed inside a deterministic train-only meta split."]
(OUT/"E080_N4_REPORT.md").write_text("\n".join(md)+"\n")
print(json.dumps(summary,indent=2))
