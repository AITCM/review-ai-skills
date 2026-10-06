from __future__ import annotations
import json,re,unicodedata,pathlib,hashlib,pickle
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.sparse import vstack

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("g2-004-output");OUT.mkdir(exist_ok=True)
SEED=20261006;K=500;GAMMAS=(0.0,0.25,0.5,0.75,1.0,1.5,2.0,3.0)
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
def pts(r,d):
    dd=nw(d);o=[]
    for x in split_reason(r):
        xx=nw(x)
        if dd and dd in xx:xx=xx.replace(dd," diagnosismask ")
        o.append(xx)
    return o
def sf(q,c):
    S=(q@c.T).toarray()
    if not S.size:return 0.
    r=float(np.max(S,axis=1).mean());p=float(np.max(S,axis=0).mean())
    return 2*r*p/(r+p) if r+p else 0.
def rank(scores):
    ids=np.arange(len(scores));return np.lexsort((ids,-scores))
def z(v):
    v=np.asarray(v,float);sd=v.std();return (v-v.mean())/(sd if sd>1e-12 else 1.0)
def features(sem,pr):
    sem=np.asarray(sem,float);pr=np.asarray(pr,float);n=len(sem)
    sz=z(sem);pz=z(pr)
    sr=np.linspace(1,0,n,endpoint=True)
    pro=np.argsort(-pr);prank=np.empty(n,int);prank[pro]=np.arange(n)
    prn=1.0-prank/max(n-1,1)
    return np.column_stack([sz,pz,sr,prn])

tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet",columns=["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"])
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet",columns=["case_prompt","diagnostic_reasoning","final_diagnosis"])
h=np.array([int(hashlib.sha256(("G2-004:"+str(x)).encode()).hexdigest()[:8],16) for x in tr.pmcid],dtype=np.uint64)
mqmask=(h%5==0);ref=tr.loc[~mqmask].reset_index(drop=True);mq=tr.loc[mqmask].reset_index(drop=True)
h2=np.array([int(hashlib.sha256(("G2-004-SPLIT:"+str(x)).encode()).hexdigest()[:8],16) for x in mq.pmcid],dtype=np.uint64)
train_idx=np.where(h2%4<=1)[0];tune_idx=np.where(h2%4==2)[0];hold_idx=np.where(h2%4==3)[0]
print("META",len(ref),len(mq),len(train_idx),len(tune_idx),len(hold_idx),flush=True)

# Reference-fitted query-available signals.
cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=cv.fit_transform(ref.case_prompt.astype(str));Q=cv.transform(mq.case_prompt.astype(str));SEM=(Q@X.T).toarray().astype(np.float32)
cs=TruncatedSVD(256,n_iter=7,random_state=SEED);XC=cs.fit_transform(X).astype(float);QC=cs.transform(Q).astype(float)
rp=[pts(r,d) for r,d in zip(ref.diagnostic_reasoning,ref.final_diagnosis)]
mp=[pts(r,d) for r,d in zip(mq.diagnostic_reasoning,mq.final_diagnosis)]
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R=rv.fit_transform([" ".join(x) for x in rp]);rs=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR=rs.fit_transform(R).astype(float)
xm=XC.mean(0,keepdims=True);ym=ZR.mean(0,keepdims=True);A=XC-xm;Y=ZR-ym;W=np.linalg.solve(A.T@A+10*np.eye(A.shape[1]),A.T@Y)
PR=normalize((QC-xm)@W+ym)@normalize(ZR).T

flat=[p for ps in rp for p in ps];ev=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
T=ev.fit_transform(flat);Tt=T.T.tocsr();counts=np.asarray([len(x) for x in rp],int);starts=np.concatenate(([0],np.cumsum(counts)[:-1]))
M=ev.transform([p for ps in mp for p in ps]);mo=[];o=0
for ps in mp:mo.append((o,o+len(ps)));o+=len(ps)

# Build monotonic utility-regression training table on meta-train only.
XF=[];YT=[]
cache={}
for qi in range(len(mq)):
    sem=SEM[qi];base=rank(sem);cand=base[:K];F=features(sem[cand],PR[qi,cand])
    s,e=mo[qi];qmat=M[s:e];S=(qmat@Tt).toarray().astype(np.float32,copy=False)
    qmax=np.stack([np.maximum.reduceat(row,starts) for row in S],axis=0)
    pmax=S.max(axis=0);psum=np.add.reduceat(pmax,starts);rec=qmax.mean(axis=0);prec=psum/counts
    den=rec+prec;util=np.divide(2*rec*prec,den,out=np.zeros_like(rec),where=den>0)
    cache[qi]=(cand,F,qmat,util)
    if qi in set(train_idx.tolist()):
        XF.append(F.astype(np.float32));YT.append(util[cand].astype(np.float32))
    if (qi+1)%250==0:print("CACHE",qi+1,flush=True)
XF=np.vstack(XF);YT=np.concatenate(YT)
print("TRAIN_ROWS",len(YT),flush=True)

model=HistGradientBoostingRegressor(loss="squared_error",learning_rate=.05,max_iter=250,max_leaf_nodes=31,
    min_samples_leaf=80,l2_regularization=1.0,random_state=SEED,monotonic_cst=[1,1,1,1]).fit(XF,YT)

def eval_group(indices,gamma):
    vals={1:[],3:[],10:[]};basevals={1:[],3:[],10:[]}
    for qi in indices:
        cand,F,qmat,util=cache[int(qi)]
        pu=model.predict(F);score=F[:,0]+gamma*z(pu);oo=np.lexsort((cand,-score));rr=cand[oo]
        for kk in (1,3,10):
            basevals[kk].append(sf(qmat,vstack([T[starts[j]:starts[j]+counts[j]] for j in cand[:kk]])))
            vals[kk].append(sf(qmat,vstack([T[starts[j]:starts[j]+counts[j]] for j in rr[:kk]])))
    return {kk:float(np.mean(vals[kk])-np.mean(basevals[kk])) for kk in (1,3,10)}

tune=[]
for g in GAMMAS:
    d=eval_group(tune_idx,g);obj=float(np.mean(list(d.values())))
    tune.append({"gamma":g,"d1":d[1],"d3":d[3],"d10":d[10],"objective":obj})
tunedf=pd.DataFrame(tune).sort_values(["objective","gamma"],ascending=[False,True]);G=float(tunedf.iloc[0].gamma)
hold=eval_group(hold_idx,G)
print("SELECTED_GAMMA",G,"HOLD",hold,flush=True)

# Final validation with all-train query-available signals. Utility estimator itself remains frozen.
cv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X2=cv2.fit_transform(tr.case_prompt.astype(str));Q2=cv2.transform(va.case_prompt.astype(str));SEM2=(Q2@X2.T).toarray().astype(np.float32)
cs2=TruncatedSVD(256,n_iter=7,random_state=SEED);XC2=cs2.fit_transform(X2).astype(float);QC2=cs2.transform(Q2).astype(float)
tp=[pts(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)];vp=[pts(r,d) for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
rv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R2=rv2.fit_transform([" ".join(x) for x in tp]);rs2=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR2=rs2.fit_transform(R2).astype(float)
xm=XC2.mean(0,keepdims=True);ym=ZR2.mean(0,keepdims=True);AA=XC2-xm;YY=ZR2-ym;WW=np.linalg.solve(AA.T@AA+10*np.eye(AA.shape[1]),AA.T@YY)
PR2=normalize((QC2-xm)@WW+ym)@normalize(ZR2).T

flat2=[p for ps in tp for p in ps];ev2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
TT=ev2.fit_transform(flat2);starts2=[];counts2=[];o=0
for ps in tp:starts2.append(o);counts2.append(len(ps));o+=len(ps)
VV=ev2.transform([p for ps in vp for p in ps]);vo=[];o=0
for ps in vp:vo.append((o,o+len(ps)));o+=len(ps)
starts2=np.asarray(starts2);counts2=np.asarray(counts2)

tl=[nw(x) for x in tr.final_diagnosis];vl=[nw(x) for x in va.final_diagnosis]
methods={"tfidf":{1:[],3:[],10:[]},"g2_003":{1:[],3:[],10:[]},"g2_004":{1:[],3:[],10:[]}}
hits={m:{1:0,3:0,10:0} for m in methods};rankings=[]
for i in range(len(va)):
    sem=SEM2[i];base=rank(sem);cand=base[:K];F=features(sem[cand],PR2[i,cand])
    r3=cand[np.lexsort((cand,-(F[:,0]+1.5*F[:,1])))]
    pu=model.predict(F);r4=cand[np.lexsort((cand,-(F[:,0]+G*z(pu))))]
    s,e=vo[i];qmat=VV[s:e]
    rr={"tfidf":base,"g2_003":r3,"g2_004":r4}
    for m,r in rr.items():
        for kk in (1,3,10):
            methods[m][kk].append(sf(qmat,vstack([TT[starts2[j]:starts2[j]+counts2[j]] for j in r[:kk]])))
            hits[m][kk]+=int(any(tl[j]==vl[i] for j in r[:kk]))
    rankings.append({"query_index":i,"tfidf":base[:10].tolist(),"g2_003":r3[:10].tolist(),"g2_004":r4[:10].tolist()})

summary={"experiment":"G2-004 monotonic learned utility residual","status":"completed_validation_development",
 "policy":"Utility model fit on train-only meta-train; gamma chosen on train-only meta-tune; checked on meta-holdout; one validation evaluation; test not used.",
 "meta":{"n_ref":len(ref),"n_queries":len(mq),"n_train":len(train_idx),"n_tune":len(tune_idx),"n_holdout":len(hold_idx),
         "training_rows":len(YT),"selected_gamma":G,"meta_holdout_delta":hold},
 "validation":{}}
for m in methods:
    summary["validation"][m]={}
    for kk in (1,3,10):
        a=np.asarray(methods[m][kk]);summary["validation"][m][str(kk)]={"mean":float(a.mean()),"exact_hit_rate":hits[m][kk]/len(va)}
for m in ("g2_003","g2_004"):
    for kk in (1,3,10):
        b=np.asarray(methods["tfidf"][kk]);a=np.asarray(methods[m][kk]);d=a-b;rng=np.random.default_rng(SEED+kk+(100 if m=="g2_004" else 0));boots=[]
        for _ in range(5000):
            ids=rng.integers(0,len(d),len(d));boots.append(float(d[ids].mean()))
        summary["validation"][m][str(kk)]["delta_vs_tfidf"]=float(d.mean())
        summary["validation"][m][str(kk)]["ci95_vs_tfidf"]=[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))]

tunedf.to_csv(OUT/"G2_004_GAMMA_SEARCH.csv",index=False)
with (OUT/"G2_004_RANKINGS.jsonl").open("w") as f:
    for x in rankings:f.write(json.dumps(x)+"\n")
(OUT/"G2_004_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
with open(OUT/"G2_004_MODEL.pkl","wb") as f:pickle.dump(model,f)

md=["# G2-004 — Monotonic Learned Utility Residual","",
f"Selected gamma on train-only meta-tune: **{G}**. Meta-holdout deltas: {hold}.","",
"| Method | U@1 | U@3 | U@10 | Δ@10 vs TF-IDF | exact-hit@10 |",
"|---|---:|---:|---:|---:|---:|"]
for m in ("tfidf","g2_003","g2_004"):
    x=summary["validation"][m]
    d=x["10"].get("delta_vs_tfidf",0.0)
    md.append(f"| {m} | {x['1']['mean']:.5f} | {x['3']['mean']:.5f} | {x['10']['mean']:.5f} | {d:+.5f} | {100*x['10']['exact_hit_rate']:.1f}% |")
(OUT/"G2_004_REPORT.md").write_text("\n".join(md)+"\n")
print(json.dumps(summary,indent=2))
