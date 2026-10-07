from __future__ import annotations
import json,re,unicodedata,pathlib,hashlib
import numpy as np,pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.ensemble import HistGradientBoostingRegressor

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e080-n6c-output");OUT.mkdir(exist_ok=True)
SEED=20261007;POOL_K=100;MAX_K=10;KS=(1,3,10)
PAT=re.compile(r'^\s*(?:\((\d+)\)|(\d+)[.)])\s*',re.M)
FEATURES=["lex","case_latent","pred_reason","lex_rr","step_frac","current_u",
          "max_case_red","mean_case_red","max_reason_red","mean_reason_red",
          "reason_novelty","lex_x_novelty","pred_x_novelty"]

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
def rank(scores,k=POOL_K):
    ids=np.arange(len(scores),dtype=int)
    return np.lexsort((ids,-np.asarray(scores)))[:k]
def set_u(qmax,psum,counts,ids):
    if len(ids)==0:return 0.0
    ids=np.asarray(ids,dtype=int)
    rec=float(np.max(qmax[:,ids],axis=1).mean())
    prec=float(psum[ids].sum()/counts[ids].sum())
    return float(2*rec*prec/(rec+prec)) if rec+prec else 0.0
def greedy(qmax,psum,counts,pool,kmax=MAX_K):
    pool=np.asarray(pool,dtype=int);alive=np.ones(len(pool),bool)
    recvec=np.zeros(qmax.shape[0],np.float32);ps=0.;pc=0;scores={};chosen=[]
    for step in range(1,min(kmax,len(pool))+1):
        recs=np.maximum(recvec[:,None],qmax[:,pool]).mean(axis=0)
        precs=(ps+psum[pool])/(pc+counts[pool]);vals=f1(recs,precs);vals[~alive]=-np.inf
        pos=int(np.lexsort((pool,-vals))[0]);j=int(pool[pos]);alive[pos]=False;chosen.append(j)
        recvec=np.maximum(recvec,qmax[:,j]);ps+=float(psum[j]);pc+=int(counts[j])
        if step in KS:scores[step]=float(vals[pos])
    return chosen,scores
def boot_ci(x,seed,n=3000):
    x=np.asarray(x,float);rng=np.random.default_rng(seed);z=np.empty(n,float)
    for b in range(n):
        ii=rng.integers(0,len(x),len(x));z[b]=x[ii].mean()
    return [float(np.percentile(z,2.5)),float(np.percentile(z,97.5))]
def candidate_features(pool,lex,lat,pred,lex_rank,step,selected,case_emb,reason_emb):
    pool=np.asarray(pool,dtype=int)
    if selected:
        S=np.asarray(selected,dtype=int)
        case_sim=case_emb[pool]@case_emb[S].T
        reas_sim=reason_emb[pool]@reason_emb[S].T
        max_case=case_sim.max(1);mean_case=case_sim.mean(1)
        max_reas=reas_sim.max(1);mean_reas=reas_sim.mean(1)
    else:
        max_case=np.zeros(len(pool),np.float32);mean_case=max_case.copy()
        max_reas=np.zeros(len(pool),np.float32);mean_reas=max_reas.copy()
    nov=1.0-max_reas
    rr=np.asarray([1.0/(1.0+lex_rank[int(j)]) for j in pool],np.float32)
    sf=np.full(len(pool),step/MAX_K,np.float32);cu=np.full(len(pool),current_u,np.float32)
    return np.column_stack([
        lex[pool],lat[pool],pred[pool],rr,sf,cu,
        max_case,mean_case,max_reas,mean_reas,nov,
        lex[pool]*nov,pred[pool]*nov
    ]).astype(np.float32)

tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
assert len(tr)==13092 and len(va)==500

# deterministic train-only meta split
isdev=np.array([int(hashlib.sha256(("E080-N6:"+str(x)).encode()).hexdigest()[:8],16)%5==0 for x in tr.pmcid])
ref=tr.loc[~isdev].reset_index(drop=True);mq=tr.loc[isdev].reset_index(drop=True)

# ---------- train-only representations ----------
cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=cv.fit_transform(ref.case_prompt.astype(str));Q=cv.transform(mq.case_prompt.astype(str))
cs=TruncatedSVD(256,n_iter=7,random_state=SEED);XC=cs.fit_transform(X).astype(np.float32);QC=cs.transform(Q).astype(np.float32)
XCN=normalize(XC);QCN=normalize(QC)
rp=[mask_points(r,d) for r,d in zip(ref.diagnostic_reasoning,ref.final_diagnosis)]
mp=[mask_points(r,d) for r,d in zip(mq.diagnostic_reasoning,mq.final_diagnosis)]
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R=rv.fit_transform([" ".join(x) for x in rp]);rs=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR=rs.fit_transform(R).astype(np.float32);ZRN=normalize(ZR)
xm=XC.mean(0,keepdims=True);ym=ZR.mean(0,keepdims=True);A=XC-xm;Y=ZR-ym
W=np.linalg.solve(A.T@A+10*np.eye(A.shape[1],dtype=np.float32),A.T@Y)
P=normalize((QC-xm)@W+ym)

# formal train-only utility evaluator
flat=[p for ps in rp for p in ps]
ev=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
T=ev.fit_transform(flat);M=ev.transform([p for ps in mp for p in ps]);Tt=T.T.tocsr()
counts=np.asarray([len(z) for z in rp],np.int32);starts=np.concatenate(([0],np.cumsum(counts)[:-1])).astype(np.int64)
mo=[];o=0
for ps in mp:mo.append((o,o+len(ps)));o+=len(ps)

trainX=[];trainY=[];rng=np.random.default_rng(SEED);traj_stats=[]
for i in range(len(mq)):
    lex=(Q.getrow(i)@X.T).toarray().ravel().astype(np.float32)
    lat=np.asarray(QCN[i]@XCN.T).ravel().astype(np.float32)
    pred=np.asarray(P[i]@ZRN.T).ravel().astype(np.float32)
    pool=rank(lex,POOL_K);lexpos={int(j):r for r,j in enumerate(pool)}
    s,e=mo[i];S=(M[s:e]@Tt).toarray().astype(np.float32,copy=False)
    qmax=np.stack([np.maximum.reduceat(row,starts) for row in S],axis=0)
    pmax=S.max(axis=0);psum=np.add.reduceat(pmax,starts)
    selected=[];recvec=np.zeros(qmax.shape[0],np.float32);cur_ps=0.;cur_pc=0;cur_u=0.
    qrows=0
    for step in range(1,MAX_K+1):
        rem=np.asarray([j for j in pool if int(j) not in selected],dtype=int)
        recs=np.maximum(recvec[:,None],qmax[:,rem]).mean(axis=0)
        precs=(cur_ps+psum[rem])/(cur_pc+counts[rem]);newu=f1(recs,precs);marg=newu-cur_u
        F=candidate_features(rem,lex,lat,pred,lexpos,step,selected,XCN,ZRN)
        # training sample: top 5 marginal, bottom 5, and 10 random mids
        order=np.lexsort((rem,-marg));top=order[:5];bottom=order[-5:]
        mid=order[5:-5]
        rand=rng.choice(mid,size=min(10,len(mid)),replace=False) if len(mid) else np.array([],dtype=int)
        take=np.unique(np.concatenate([top,bottom,rand]))
        trainX.append(F[take]);trainY.append(marg[take].astype(np.float32));qrows+=len(take)
        # teacher-force along the true greedy oracle trajectory
        best=int(order[0]);j=int(rem[best]);selected.append(j)
        recvec=np.maximum(recvec,qmax[:,j]);cur_ps+=float(psum[j]);cur_pc+=int(counts[j]);cur_u=float(newu[best])
    traj_stats.append({"final_oracle10":cur_u,"sample_rows":qrows})
    if (i+1)%250==0:print(f"TRAIN TRAJECTORIES {i+1}/{len(mq)}",flush=True)

TX=np.vstack(trainX);TY=np.concatenate(trainY)
model=HistGradientBoostingRegressor(loss="squared_error",learning_rate=.05,max_iter=300,max_leaf_nodes=31,
    min_samples_leaf=80,l2_regularization=2.0,early_stopping=False,random_state=SEED).fit(TX,TY)

# ---------- full-train inference representations ----------
cv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X2=cv2.fit_transform(tr.case_prompt.astype(str));Q2=cv2.transform(va.case_prompt.astype(str))
cs2=TruncatedSVD(256,n_iter=7,random_state=SEED);XC2=cs2.fit_transform(X2).astype(np.float32);QC2=cs2.transform(Q2).astype(np.float32)
X2N=normalize(XC2);Q2N=normalize(QC2)
tp=[mask_points(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
vp=[mask_points(r,d) for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
rv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R2=rv2.fit_transform([" ".join(x) for x in tp]);rs2=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR2=rs2.fit_transform(R2).astype(np.float32);Z2N=normalize(ZR2)
xm2=XC2.mean(0,keepdims=True);ym2=ZR2.mean(0,keepdims=True);AA=XC2-xm2;YY=ZR2-ym2
WW=np.linalg.solve(AA.T@AA+10*np.eye(AA.shape[1],dtype=np.float32),AA.T@YY)
PP=normalize((QC2-xm2)@WW+ym2)

# validation evaluator
flat2=[p for ps in tp for p in ps]
ee=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
TT=ee.fit_transform(flat2);VV=ee.transform([p for ps in vp for p in ps]);TTt=TT.T.tocsr()
cnt=np.asarray([len(z) for z in tp],np.int32);st=np.concatenate(([0],np.cumsum(cnt)[:-1])).astype(np.int64)
vo=[];o=0
for ps in vp:vo.append((o,o+len(ps)));o+=len(ps)

scores={m:{k:[] for k in KS} for m in ["LEX","N6_SET","LEX100_ORACLE"]}
recover={k:[] for k in KS};rows=[]
for i in range(len(va)):
    lex=(Q2.getrow(i)@X2.T).toarray().ravel().astype(np.float32)
    lat=np.asarray(Q2N[i]@X2N.T).ravel().astype(np.float32)
    pred=np.asarray(PP[i]@Z2N.T).ravel().astype(np.float32)
    pool=rank(lex,POOL_K);lexpos={int(j):r for r,j in enumerate(pool)}

    # Freeze the learned ranking using inference-time information only.
    selected=[]
    for step in range(1,MAX_K+1):
        rem=np.asarray([j for j in pool if int(j) not in selected],dtype=int)
        F=candidate_features(rem,lex,lat,pred,lexpos,step,selected,X2N,Z2N)
        pr=model.predict(F)
        pos=int(np.lexsort((rem,-pr))[0]);selected.append(int(rem[pos]))

    # Only after ranking is frozen do we access validation reasoning for evaluation.
    s,e=vo[i];S=(VV[s:e]@TTt).toarray().astype(np.float32,copy=False)
    qmax=np.stack([np.maximum.reduceat(row,st) for row in S],axis=0)
    pmax=S.max(axis=0);psum=np.add.reduceat(pmax,st)
    _,oracle=greedy(qmax,psum,cnt,pool,MAX_K)
    for k in KS:
        b=set_u(qmax,psum,cnt,pool[:k]);m=set_u(qmax,psum,cnt,selected[:k]);o=float(oracle[k])
        scores["LEX"][k].append(b);scores["N6_SET"][k].append(m);scores["LEX100_ORACLE"][k].append(o)
        denom=o-b;recover[k].append((m-b)/denom if denom>1e-12 else 0.0)
    rows.append({"query_index":i,"lexical_top100":pool.tolist(),"n6_selected10":selected,
                 **{f"lex_{k}":scores["LEX"][k][-1] for k in KS},
                 **{f"n6_{k}":scores["N6_SET"][k][-1] for k in KS},
                 **{f"oracle_{k}":scores["LEX100_ORACLE"][k][-1] for k in KS}})
    if (i+1)%50==0:print(f"VALIDATION {i+1}/{len(va)}",flush=True)

with (OUT/"E080_N6C_RANKINGS.jsonl").open("w") as f:
    for x in rows:f.write(json.dumps(x)+"\n")
summary={
 "experiment":"E080-N6C leakage-free set-aware marginal-utility policy",
 "status":"validation-development only; MedCaseReasoning test untouched",
 "dataset_revision":REV,"n_train":len(tr),"n_validation":len(va),"candidate_pool":"lexical Top-100 fixed",
 "features":FEATURES,
 "training":{"meta_reference":len(ref),"meta_queries":len(mq),"rows":int(len(TY)),
             "teacher":"train-only greedy oracle trajectories",
             "target":"true marginal change in diagnosis-masked reasoning-set softF1",
             "model":{"type":"HistGradientBoostingRegressor","learning_rate":.05,"max_iter":300,
                      "max_leaf_nodes":31,"min_samples_leaf":80,"l2_regularization":2.0}},
 "results":{},"policy":"Validation ranking is frozen before validation reasoning is accessed. Inference uses case_prompt plus historical-library case/masked-reasoning representations only. Test/T001 untouched."
}
for k in KS:
    b=np.asarray(scores["LEX"][k]);m=np.asarray(scores["N6_SET"][k]);o=np.asarray(scores["LEX100_ORACLE"][k]);d=m-b
    summary["results"][str(k)]={
      "lexical":float(b.mean()),"n6_set":float(m.mean()),"delta":float(d.mean()),"delta_ci95":boot_ci(d,SEED+k),
      "lex100_oracle":float(o.mean()),
      "fraction_of_recoverable_gap":float((m.mean()-b.mean())/(o.mean()-b.mean())) if o.mean()>b.mean() else None,
      "querywise_fraction_recovered_median":float(np.median(recover[k]))
    }
(OUT/"E080_N6C_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
md=["# E080-N6C Leakage-free Set-aware Marginal-Utility Policy","",
"Validation-development only. The MedCaseReasoning test split remains untouched.","",
"| k | lexical | N6 set-aware | delta | lexical-100 oracle | fraction gap recovered |",
"|---:|---:|---:|---:|---:|---:|"]
for k in KS:
    x=summary["results"][str(k)]
    md.append(f"| {k} | {x['lexical']:.5f} | {x['n6_set']:.5f} | {x['delta']:+.5f} | {x['lex100_oracle']:.5f} | {100*x['fraction_of_recoverable_gap']:.1f}% |")
md += ["","N6 isolates set selection: the candidate pool is exactly the same lexical Top-100 for baseline, learned policy, and oracle."]
(OUT/"E080_N6C_REPORT.md").write_text("\n".join(md)+"\n")
print(json.dumps(summary,indent=2))
