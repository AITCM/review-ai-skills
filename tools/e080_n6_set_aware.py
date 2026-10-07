from __future__ import annotations
import json,re,unicodedata,pathlib,hashlib
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.ensemble import HistGradientBoostingRegressor

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e080-n6-output");OUT.mkdir(exist_ok=True)
SEED=20261007;POOL_K=200;KS=(1,3,10);STEPS=10;SAMPLE_PER_STEP=48
PAT=re.compile(r'^\s*(?:\((\d+)\)|(\d+)[.)])\s*',re.M)
FEATURES=["lex","case_lat","pred_reason","lex_x_pred","lex_rr","step_frac",
          "reason_red_max","reason_red_mean","case_red_max","case_red_mean"]

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
def f1v(rec,prec):
    den=rec+prec
    return np.where(den>0,2*rec*prec/den,0.0)
def set_u(qmax,psum,counts,ids):
    ids=np.asarray(ids,dtype=int)
    rec=float(np.max(qmax[:,ids],axis=1).mean());prec=float(psum[ids].sum()/counts[ids].sum())
    return float(2*rec*prec/(rec+prec)) if rec+prec else 0.
def boot_ci(x,seed,n=3000):
    x=np.asarray(x,float);rng=np.random.default_rng(seed);z=np.empty(n,float)
    for t in range(n):
        ii=rng.integers(0,len(x),len(x));z[t]=x[ii].mean()
    return [float(np.percentile(z,2.5)),float(np.percentile(z,97.5))]
def make_features(pool,lex,lat,pred,lexpos,reason_emb,case_emb,selected,step):
    pool=np.asarray(pool,dtype=int)
    if selected:
        rr=reason_emb[pool]@reason_emb[np.asarray(selected,dtype=int)].T
        cc=case_emb[pool]@case_emb[np.asarray(selected,dtype=int)].T
        rmax=rr.max(1);rmean=rr.mean(1);cmax=cc.max(1);cmean=cc.mean(1)
    else:
        rmax=rmean=cmax=cmean=np.zeros(len(pool),dtype=np.float32)
    lp=np.asarray([lexpos[int(j)] for j in pool],dtype=np.float32)
    return np.column_stack([
        lex[pool],lat[pool],pred[pool],lex[pool]*pred[pool],1.0/lp,
        np.full(len(pool),step/STEPS,dtype=np.float32),rmax,rmean,cmax,cmean
    ]).astype(np.float32)

tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
assert len(tr)==13092 and len(va)==500
isdev=np.array([int(hashlib.sha256(("E080-N6:"+str(x)).encode()).hexdigest()[:8],16)%5==0 for x in tr.pmcid])
ref=tr.loc[~isdev].reset_index(drop=True);mq=tr.loc[isdev].reset_index(drop=True)

# Meta-reference representations.
cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=cv.fit_transform(ref.case_prompt.astype(str));Q=cv.transform(mq.case_prompt.astype(str))
cs=TruncatedSVD(256,n_iter=7,random_state=SEED);XC=cs.fit_transform(X).astype(np.float32);QC=cs.transform(Q).astype(np.float32)
XCN=normalize(XC);QCN=normalize(QC)
rp=[mask_points(r,d) for r,d in zip(ref.diagnostic_reasoning,ref.final_diagnosis)]
mp=[mask_points(r,d) for r,d in zip(mq.diagnostic_reasoning,mq.final_diagnosis)]
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R=rv.fit_transform([" ".join(x) for x in rp]);rs=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR=rs.fit_transform(R).astype(np.float32);ZRN=normalize(ZR)
xm=XC.mean(0,keepdims=True);ym=ZR.mean(0,keepdims=True);A=XC-xm;Y=ZR-ym
W=np.linalg.solve(A.T@A+10*np.eye(A.shape[1],dtype=np.float32),A.T@Y);P=normalize((QC-xm)@W+ym)

# Train-only formal utility matrix ingredients.
flat=[p for ps in rp for p in ps]
ev=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
T=ev.fit_transform(flat);M=ev.transform([p for ps in mp for p in ps]);Tt=T.T.tocsr()
cnt=np.asarray([len(z) for z in rp],np.int32);starts=np.concatenate(([0],np.cumsum(cnt)[:-1])).astype(np.int64)
mo=[];o=0
for ps in mp:mo.append((o,o+len(ps)));o+=len(ps)

rng=np.random.default_rng(SEED);rowsX=[];rowsY=[];oracle_train=[]
for i in range(len(mq)):
    lex=(Q.getrow(i)@X.T).toarray().ravel();lat=np.asarray(QCN[i]@XCN.T).ravel();pred=np.asarray(P[i]@ZRN.T).ravel()
    ids=np.arange(len(ref));pool=np.lexsort((ids,-lex))[:POOL_K];lexpos={int(j):r+1 for r,j in enumerate(pool)}
    s,e=mo[i];S=(M[s:e]@Tt).toarray().astype(np.float32,copy=False)
    qmax=np.stack([np.maximum.reduceat(row,starts) for row in S],axis=0)
    pmax=S.max(axis=0);psum=np.add.reduceat(pmax,starts)
    selected=[];recvec=np.zeros(qmax.shape[0],np.float32);ps=0.;pc=0;cur=0.;seq=[]
    remaining=pool.copy()
    for step in range(STEPS):
        cand=np.asarray([j for j in remaining if int(j) not in set(selected)],dtype=int)
        recs=np.maximum(recvec[:,None],qmax[:,cand]).mean(axis=0)
        precs=(ps+psum[cand])/(pc+cnt[cand]);new=f1v(recs,precs);marg=new-cur
        order=np.lexsort((cand,-marg));chosen=int(cand[order[0]]);seq.append(chosen)

        # training sample: chosen + best competitors + worst + deterministic random
        take=set([chosen])
        for pos in order[:12]:take.add(int(cand[pos]))
        for pos in order[-8:]:take.add(int(cand[pos]))
        rest=np.asarray([j for j in cand if int(j) not in take],dtype=int)
        if len(rest):
            for j in rng.choice(rest,size=min(max(0,SAMPLE_PER_STEP-len(take)),len(rest)),replace=False):take.add(int(j))
        samp=np.asarray(sorted(take),dtype=int)
        idx={int(j):p for p,j in enumerate(cand)}
        Fm=make_features(samp,lex,lat,pred,lexpos,ZRN,XCN,selected,step)
        rowsX.append(Fm);rowsY.append(np.asarray([marg[idx[int(j)]] for j in samp],dtype=np.float32))

        cpos=idx[chosen];recvec=np.maximum(recvec,qmax[:,chosen]);ps+=float(psum[chosen]);pc+=int(cnt[chosen]);cur=float(new[cpos]);selected.append(chosen)
    oracle_train.append(cur)
    if (i+1)%250==0:print(f"TRAIN META {i+1}/{len(mq)}",flush=True)

Xreg=np.vstack(rowsX);Yreg=np.concatenate(rowsY)
model=HistGradientBoostingRegressor(loss="squared_error",learning_rate=.05,max_iter=300,max_leaf_nodes=31,
    min_samples_leaf=60,l2_regularization=1.0,early_stopping=False,random_state=SEED).fit(Xreg,Yreg)

# Full-train representations for validation inference.
cv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X2=cv2.fit_transform(tr.case_prompt.astype(str));Q2=cv2.transform(va.case_prompt.astype(str))
cs2=TruncatedSVD(256,n_iter=7,random_state=SEED);XC2=cs2.fit_transform(X2).astype(np.float32);QC2=cs2.transform(Q2).astype(np.float32)
X2N=normalize(XC2);Q2N=normalize(QC2)
tp=[mask_points(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
vp=[mask_points(r,d) for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
rv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R2=rv2.fit_transform([" ".join(x) for x in tp]);rs2=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR2=rs2.fit_transform(R2).astype(np.float32);Z2N=normalize(ZR2)
xm=XC2.mean(0,keepdims=True);ym=ZR2.mean(0,keepdims=True);AA=XC2-xm;YY=ZR2-ym
WW=np.linalg.solve(AA.T@AA+10*np.eye(AA.shape[1],dtype=np.float32),AA.T@YY);PP=normalize((QC2-xm)@WW+ym)

# Validation evaluator.
flat=[p for ps in tp for p in ps]
ee=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
TT=ee.fit_transform(flat);VV=ee.transform([p for ps in vp for p in ps]);TTt=TT.T.tocsr()
cnt2=np.asarray([len(z) for z in tp],np.int32);st=np.concatenate(([0],np.cumsum(cnt2)[:-1])).astype(np.int64)
vo=[];o=0
for ps in vp:vo.append((o,o+len(ps)));o+=len(ps)

scores={"LEX":{k:[] for k in KS},"N6_SET":{k:[] for k in KS},"LEX200_ORACLE":{k:[] for k in KS}}
tl=[nw(x) for x in tr.final_diagnosis];vl=[nw(x) for x in va.final_diagnosis]
exact={"LEX":{k:0 for k in KS},"N6_SET":{k:0 for k in KS}}
rankings=[]
for i in range(len(va)):
    lex=(Q2.getrow(i)@X2.T).toarray().ravel();lat=np.asarray(Q2N[i]@X2N.T).ravel();pred=np.asarray(PP[i]@Z2N.T).ravel()
    ids=np.arange(len(tr));pool=np.lexsort((ids,-lex))[:POOL_K];lexpos={int(j):r+1 for r,j in enumerate(pool)}
    selected=[]
    for step in range(STEPS):
        cand=np.asarray([j for j in pool if int(j) not in set(selected)],dtype=int)
        Fm=make_features(cand,lex,lat,pred,lexpos,Z2N,X2N,selected,step)
        predm=model.predict(Fm);order=np.lexsort((cand,-predm));selected.append(int(cand[order[0]]))
    lexorder=pool;setorder=np.asarray(selected,dtype=int)
    s,e=vo[i];S=(VV[s:e]@TTt).toarray().astype(np.float32,copy=False)
    qmax=np.stack([np.maximum.reduceat(row,st) for row in S],axis=0)
    pmax=S.max(axis=0);psum=np.add.reduceat(pmax,st)
    for k in KS:
        scores["LEX"][k].append(set_u(qmax,psum,cnt2,lexorder[:k]))
        scores["N6_SET"][k].append(set_u(qmax,psum,cnt2,setorder[:k]))
        exact["LEX"][k]+=int(any(tl[j]==vl[i] for j in lexorder[:k]))
        exact["N6_SET"][k]+=int(any(tl[j]==vl[i] for j in setorder[:k]))
    # true post-hoc pool oracle for ceiling only
    alive=np.ones(len(pool),bool);recvec=np.zeros(qmax.shape[0],np.float32);ps=0.;pc=0;cur=0.
    for step in range(STEPS):
        recs=np.maximum(recvec[:,None],qmax[:,pool]).mean(axis=0);precs=(ps+psum[pool])/(pc+cnt2[pool]);vals=f1v(recs,precs);vals[~alive]=-np.inf
        p=int(np.lexsort((pool,-vals))[0]);j=int(pool[p]);alive[p]=False;recvec=np.maximum(recvec,qmax[:,j]);ps+=float(psum[j]);pc+=int(cnt2[j]);cur=float(vals[p])
        if step+1 in KS:scores["LEX200_ORACLE"][step+1].append(cur)
    rankings.append({"query_index":i,"lex200":pool.tolist(),"n6_selected":selected})
    if (i+1)%50==0:print(f"VALIDATION {i+1}/{len(va)}",flush=True)

with (OUT/"E080_N6_RANKINGS.jsonl").open("w") as f:
    for x in rankings:f.write(json.dumps(x)+"\n")
summary={"experiment":"E080-N6 set-aware marginal-utility imitation policy",
 "status":"validation-development only; test untouched","dataset_revision":REV,
 "n_train":len(tr),"n_validation":len(va),"candidate_pool":"lexical Top-200",
 "training":{"meta_reference":len(ref),"meta_queries":len(mq),"rows":int(len(Yreg)),"features":FEATURES,
             "target":"true train-only marginal gain in symmetric reasoning-set softF1 under oracle trajectory",
             "mean_oracle_train_softF1_at10":float(np.mean(oracle_train)),
             "hgb":{"learning_rate":.05,"max_iter":300,"max_leaf_nodes":31,"min_samples_leaf":60,"l2_regularization":1.0}},
 "validation":{"methods":{}},"policy":"New-query input is case_prompt only. Historical masked reasoning is used only for candidate redundancy features. Validation reasoning is evaluation only; test/T001 untouched."}
for m in scores:
    summary["validation"]["methods"][m]={}
    for k in KS:
        arr=np.asarray(scores[m][k]);b=np.asarray(scores["LEX"][k]);d=arr-b
        summary["validation"]["methods"][m][str(k)]={"softF1":float(arr.mean()),"delta_vs_LEX":float(d.mean()),"delta_ci95":boot_ci(d,SEED+k+len(m))}
        if m in exact:summary["validation"]["methods"][m][str(k)]["exact_label_hit"]=int(exact[m][k])
(OUT/"E080_N6_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
md=["# E080-N6 Set-aware Policy","",
"Validation-development only; test untouched.","",
"| method | @1 | @3 | @10 |","|---|---:|---:|---:|"]
for m in scores:
    x=summary["validation"]["methods"][m]
    md.append(f"| {m} | {x['1']['softF1']:.5f} | {x['3']['softF1']:.5f} | {x['10']['softF1']:.5f} |")
(OUT/"E080_N6_REPORT.md").write_text("\n".join(md)+"\n")
print(json.dumps(summary,indent=2))
