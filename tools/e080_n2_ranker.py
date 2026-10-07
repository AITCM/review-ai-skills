from __future__ import annotations
import json,re,unicodedata,pathlib,hashlib
from collections import defaultdict
import numpy as np,pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e080-n2-output");OUT.mkdir(exist_ok=True)
SEED=20261007;KS=(1,3,10)
PAT=re.compile(r'^\s*(?:\((\d+)\)|(\d+)[.)])\s*',re.M)
FEATURES=["lex","latent","pred_reason","lex_x_pred","lex_x_latent","latent_x_pred",
          "lex_rr","latent_rr","pred_rr","rrf","route_count","score_spread"]

def nw(s): return " ".join(re.findall(r"\w+",unicodedata.normalize("NFKC",str(s)).casefold()))
def split_reason(s):
    s=str(s);ms=list(PAT.finditer(s))
    if not ms or s[:ms[0].start()].strip(): return [s.strip()]
    nums=[int(m.group(1) or m.group(2)) for m in ms]
    if nums!=list(range(1,len(ms)+1)): return [s.strip()]
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
def rank(s,k):
    ids=np.arange(len(s));return np.lexsort((ids,-np.asarray(s)))[:k]
def f1(rec,prec):
    den=rec+prec
    return np.where(den>0,2*rec*prec/den,0.0)
def utility_all(qmat,Tt,starts,counts):
    S=(qmat@Tt).toarray().astype(np.float32,copy=False)
    qmax=np.stack([np.maximum.reduceat(row,starts) for row in S],axis=0)
    pmax=S.max(axis=0);psum=np.add.reduceat(pmax,starts)
    rec=qmax.mean(axis=0);prec=psum/counts
    return f1(rec,prec),qmax,psum
def set_u(qmax,psum,counts,ids):
    ids=np.asarray(ids,dtype=int)
    rec=float(np.max(qmax[:,ids],axis=1).mean());prec=float(psum[ids].sum()/counts[ids].sum())
    return float(2*rec*prec/(rec+prec)) if rec+prec else 0.0
def greedy(qmax,psum,counts,pool,kmax=10):
    pool=np.asarray(sorted(set(int(x) for x in pool)),dtype=int)
    alive=np.ones(len(pool),bool);recvec=np.zeros(qmax.shape[0],np.float32);ps=0.;pc=0;scores={}
    for step in range(1,min(kmax,len(pool))+1):
        recs=np.maximum(recvec[:,None],qmax[:,pool]).mean(axis=0)
        precs=(ps+psum[pool])/(pc+counts[pool]);vals=f1(recs,precs);vals[~alive]=-np.inf
        pos=int(np.lexsort((pool,-vals))[0]);j=int(pool[pos]);alive[pos]=False
        recvec=np.maximum(recvec,qmax[:,j]);ps+=float(psum[j]);pc+=int(counts[j])
        if step in KS:scores[step]=float(vals[pos])
    return scores
def tri_pool(lex,lat,pred,k=100):
    lr=rank(lex,k);ar=rank(lat,k);pr=rank(pred,k)
    pool=np.asarray(sorted(set(lr)|set(ar)|set(pr)),dtype=int)
    return pool,lr,ar,pr
def make_features(pool,lex,lat,pred,lr,ar,pr):
    lpos={int(j):r for r,j in enumerate(lr,1)};apos={int(j):r for r,j in enumerate(ar,1)};ppos={int(j):r for r,j in enumerate(pr,1)}
    out=[]
    for j in pool:
        j=int(j);a=float(lex[j]);b=float(lat[j]);c=float(pred[j])
        rr1=1/lpos[j] if j in lpos else 0.;rr2=1/apos[j] if j in apos else 0.;rr3=1/ppos[j] if j in ppos else 0.
        rrf=sum(1/(60+r) for r in (lpos.get(j),apos.get(j),ppos.get(j)) if r is not None)
        rc=int(j in lpos)+int(j in apos)+int(j in ppos)
        spread=float(np.std([a,b,c]))
        out.append([a,b,c,a*c,a*b,b*c,rr1,rr2,rr3,rrf,rc,spread])
    return np.asarray(out,np.float32)
def rrf_order(pool,F):
    ids=np.asarray(pool,dtype=int);s=F[:,9]
    return ids[np.lexsort((ids,-s))]
def boot_ci(d,seed,n=3000):
    d=np.asarray(d,float);rng=np.random.default_rng(seed);z=[]
    for _ in range(n):
        ids=rng.integers(0,len(d),len(d));z.append(float(d[ids].mean()))
    return [float(np.percentile(z,2.5)),float(np.percentile(z,97.5))]

tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
assert len(tr)==13092 and len(va)==500
isdev=np.array([int(hashlib.sha256(("E080-N2:"+str(x)).encode()).hexdigest()[:8],16)%5==0 for x in tr.pmcid])
ref=tr.loc[~isdev].reset_index(drop=True);mq=tr.loc[isdev].reset_index(drop=True)

# ---------- train rankers on a train-only meta split ----------
cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=cv.fit_transform(ref.case_prompt.astype(str));Q=cv.transform(mq.case_prompt.astype(str))
cs=TruncatedSVD(256,n_iter=7,random_state=SEED);XC=cs.fit_transform(X).astype(np.float32);QC=cs.transform(Q).astype(np.float32)
XCN=normalize(XC);QCN=normalize(QC)
rp=[mask_points(r,d) for r,d in zip(ref.diagnostic_reasoning,ref.final_diagnosis)]
mp=[mask_points(r,d) for r,d in zip(mq.diagnostic_reasoning,mq.final_diagnosis)]
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R=rv.fit_transform([" ".join(x) for x in rp]);rs=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR=rs.fit_transform(R).astype(np.float32)
xm=XC.mean(0,keepdims=True);ym=ZR.mean(0,keepdims=True);A=XC-xm;Y=ZR-ym
W=np.linalg.solve(A.T@A+10*np.eye(A.shape[1],dtype=np.float32),A.T@Y)
P=normalize((QC-xm)@W+ym);ZN=normalize(ZR)

flat=[p for ps in rp for p in ps]
ev=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
T=ev.fit_transform(flat);M=ev.transform([p for ps in mp for p in ps])
counts=np.asarray([len(z) for z in rp],np.int32);starts=np.concatenate(([0],np.cumsum(counts)[:-1])).astype(np.int64)
mo=[];o=0
for ps in mp:mo.append((o,o+len(ps)));o+=len(ps)
Tt=T.T.tocsr()

rowsX=[];rowsY=[];groups=[];pairD=[];pairY=[];train_pool_sizes=[]
for i in range(len(mq)):
    lex=(Q.getrow(i)@X.T).toarray().ravel();lat=np.asarray(QCN[i]@XCN.T).ravel();pred=np.asarray(P[i]@ZN.T).ravel()
    pool,lr,ar,pr=tri_pool(lex,lat,pred,100);F=make_features(pool,lex,lat,pred,lr,ar,pr)
    s,e=mo[i];util,_,_=utility_all(M[s:e],Tt,starts,counts);y=util[pool].astype(np.float32)
    rowsX.append(F);rowsY.append(y);groups.append(len(pool));train_pool_sizes.append(len(pool))
    hi=np.argsort(-y)[:5];lo=np.argsort(y)[:10]
    for p in hi:
        for n in lo:
            if y[p]-y[n] < .02:continue
            d=F[p]-F[n];pairD.append(d);pairY.append(1);pairD.append(-d);pairY.append(0)
Xreg=np.vstack(rowsX);Yreg=np.concatenate(rowsY)
D=np.asarray(pairD,np.float32);PY=np.asarray(pairY,np.int8)
sc=StandardScaler();DZ=sc.fit_transform(D)
lin=LogisticRegression(C=1.0,max_iter=2000,random_state=SEED).fit(DZ,PY)
lin_w=lin.coef_[0]/sc.scale_
lin_b=float(lin.intercept_[0]-np.dot(lin.coef_[0],sc.mean_/sc.scale_))
hgb=HistGradientBoostingRegressor(loss="squared_error",learning_rate=.05,max_iter=250,max_leaf_nodes=31,
    min_samples_leaf=40,l2_regularization=1.0,early_stopping=False,random_state=SEED).fit(Xreg,Yreg)

# ---------- refit representations on full train; validation is inference/evaluation only ----------
cv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X2=cv2.fit_transform(tr.case_prompt.astype(str));Q2=cv2.transform(va.case_prompt.astype(str))
cs2=TruncatedSVD(256,n_iter=7,random_state=SEED);XC2=cs2.fit_transform(X2).astype(np.float32);QC2=cs2.transform(Q2).astype(np.float32)
X2N=normalize(XC2);Q2N=normalize(QC2)
tp=[mask_points(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
vp=[mask_points(r,d) for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
rv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R2=rv2.fit_transform([" ".join(x) for x in tp]);rs2=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR2=rs2.fit_transform(R2).astype(np.float32)
xm=XC2.mean(0,keepdims=True);ym=ZR2.mean(0,keepdims=True);AA=XC2-xm;YY=ZR2-ym
WW=np.linalg.solve(AA.T@AA+10*np.eye(AA.shape[1],dtype=np.float32),AA.T@YY)
PP=normalize((QC2-xm)@WW+ym);ZZ=normalize(ZR2)

flat=[p for ps in tp for p in ps]
ee=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
TT=ee.fit_transform(flat);VV=ee.transform([p for ps in vp for p in ps])
cnt=np.asarray([len(z) for z in tp],np.int32);st=np.concatenate(([0],np.cumsum(cnt)[:-1])).astype(np.int64)
vo=[];o=0
for ps in vp:vo.append((o,o+len(ps)));o+=len(ps)
TTt=TT.T.tocsr()

methods=["LEX","RRF_TRI100","N2_LINEAR","N2_HGB","TRI100_ORACLE"]
scores={m:{k:[] for k in KS} for m in methods};rankings=[];pool_sizes=[]
tl=[nw(x) for x in tr.final_diagnosis];vl=[nw(x) for x in va.final_diagnosis]
exact={m:{k:0 for k in KS} for m in methods if m!="TRI100_ORACLE"}

for i in range(len(va)):
    lex=(Q2.getrow(i)@X2.T).toarray().ravel();lat=np.asarray(Q2N[i]@X2N.T).ravel();pred=np.asarray(PP[i]@ZZ.T).ravel()
    pool,lr,ar,pr=tri_pool(lex,lat,pred,100);F=make_features(pool,lex,lat,pred,lr,ar,pr);pool_sizes.append(len(pool))
    lexord=lr
    rrf=rrf_order(pool,F)
    sl=F@lin_w+lin_b;ids=np.asarray(pool);linord=ids[np.lexsort((ids,-sl))]
    sh=hgb.predict(F);hgbord=ids[np.lexsort((ids,-sh))]
    s,e=vo[i];_,qmax,psum=utility_all(VV[s:e],TTt,st,cnt)
    gs=greedy(qmax,psum,cnt,pool,10)
    orders={"LEX":lexord,"RRF_TRI100":rrf,"N2_LINEAR":linord,"N2_HGB":hgbord}
    for m,order in orders.items():
        for k in KS:
            scores[m][k].append(set_u(qmax,psum,cnt,order[:k]))
            if any(tl[int(j)]==vl[i] for j in order[:k]):exact[m][k]+=1
    for k in KS:scores["TRI100_ORACLE"][k].append(gs[k])
    rankings.append({"query_index":i,"tri100_pool":pool.tolist(),"rrf_ranked":rrf.tolist(),"linear_ranked":linord.tolist(),"hgb_ranked":hgbord.tolist()})

with (OUT/"E080_N2_RANKINGS.jsonl").open("w") as f:
    for x in rankings:f.write(json.dumps(x)+"\n")
summary={
 "experiment":"E080-N2 TRI100 ranker development",
 "status":"validation-development only; MedCaseReasoning test untouched",
 "dataset_revision":REV,"n_train":len(tr),"n_validation":len(va),
 "candidate_generator":"TRI100 = union of lexical Top-100, case-latent Top-100, predicted-reasoning Top-100",
 "training":{"meta_reference":len(ref),"meta_queries":len(mq),"mean_tri100_pool":float(np.mean(train_pool_sizes)),
             "regression_rows":int(len(Yreg)),"pairwise_symmetric_rows":int(len(PY)),"features":FEATURES,
             "linear_weights":{k:float(v) for k,v in zip(FEATURES,lin_w)},
             "hgb":{"learning_rate":.05,"max_iter":250,"max_leaf_nodes":31,"min_samples_leaf":40,"l2_regularization":1.0}},
 "validation":{"mean_tri100_pool":float(np.mean(pool_sizes)),"methods":{}},
 "policy":"All ranker supervision comes from a deterministic train-only meta split. Validation reasoning is evaluation only. Test/T001 is not accessed."
}
base=np.asarray(scores["LEX"][1],float)
for m in methods:
    summary["validation"]["methods"][m]={}
    for k in KS:
        arr=np.asarray(scores[m][k],float);b=np.asarray(scores["LEX"][k],float);d=arr-b
        summary["validation"]["methods"][m][str(k)]={
          "softF1":float(arr.mean()),"delta_vs_LEX":float(d.mean()),"delta_ci95":boot_ci(d,SEED+k+len(m))
        }
        if m in exact:summary["validation"]["methods"][m][str(k)]["exact_label_hit"]=int(exact[m][k])
(OUT/"E080_N2_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
md=["# E080-N2 TRI100 Ranker Development","",
"Validation-development only; the 897-case MedCaseReasoning test set is untouched.","",
"| method | @1 | Δ@1 | @3 | Δ@3 | @10 | Δ@10 |","|---|---:|---:|---:|---:|---:|---:|"]
for m in methods:
    x=summary["validation"]["methods"][m]
    md.append(f"| {m} | {x['1']['softF1']:.5f} | {x['1']['delta_vs_LEX']:+.5f} | {x['3']['softF1']:.5f} | {x['3']['delta_vs_LEX']:+.5f} | {x['10']['softF1']:.5f} | {x['10']['delta_vs_LEX']:+.5f} |")
(OUT/"E080_N2_REPORT.md").write_text("\n".join(md)+"\n")
print(json.dumps(summary,indent=2))
