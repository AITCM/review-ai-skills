from __future__ import annotations
import json,re,unicodedata,pathlib,hashlib
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from scipy.sparse import vstack

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("g2-003-output");OUT.mkdir(exist_ok=True)
SEED=20261006;KS=(50,100,200,500);LAMBDAS=(0.0,0.1,0.25,0.5,0.75,1.0,1.5,2.0,3.0)
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
    v=np.asarray(v,float);sd=v.std()
    return (v-v.mean())/(sd if sd>1e-12 else 1.0)

tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet",columns=["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"])
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet",columns=["case_prompt","diagnostic_reasoning","final_diagnosis"])
assert len(tr)==13092 and len(va)==500

# Train-only meta split.
h=np.array([int(hashlib.sha256(("G2-003:"+str(x)).encode()).hexdigest()[:8],16) for x in tr.pmcid],dtype=np.uint64)
mqmask=(h%5==0);ref=tr.loc[~mqmask].reset_index(drop=True);mq=tr.loc[mqmask].reset_index(drop=True)
h2=np.array([int(hashlib.sha256(("G2-003-HOLD:"+str(x)).encode()).hexdigest()[:8],16) for x in mq.pmcid],dtype=np.uint64)
dev_idx=np.where(h2%2==0)[0];hold_idx=np.where(h2%2==1)[0]
print("META",len(ref),len(mq),len(dev_idx),len(hold_idx),flush=True)

# Representation/predicted-reasoning fit on meta-reference only.
cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=cv.fit_transform(ref.case_prompt.astype(str));Q=cv.transform(mq.case_prompt.astype(str))
SEM=(Q@X.T).toarray().astype(np.float32)
cs=TruncatedSVD(256,n_iter=7,random_state=SEED);XC=cs.fit_transform(X).astype(float);QC=cs.transform(Q).astype(float)
rp=[pts(r,d) for r,d in zip(ref.diagnostic_reasoning,ref.final_diagnosis)]
mp=[pts(r,d) for r,d in zip(mq.diagnostic_reasoning,mq.final_diagnosis)]
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R=rv.fit_transform([" ".join(x) for x in rp]);rs=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR=rs.fit_transform(R).astype(float)
xm=XC.mean(0,keepdims=True);ym=ZR.mean(0,keepdims=True);A=XC-xm;Y=ZR-ym
W=np.linalg.solve(A.T@A+10*np.eye(A.shape[1]),A.T@Y)
PR=normalize((QC-xm)@W+ym)@normalize(ZR).T

# Utility evaluator fit on meta-reference reasoning only.
flat=[p for ps in rp for p in ps]
ev=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
T=ev.fit_transform(flat);offs=[];o=0
for ps in rp:offs.append((o,o+len(ps)));o+=len(ps)
M=ev.transform([p for ps in mp for p in ps]);mo=[];o=0
for ps in mp:mo.append((o,o+len(ps)));o+=len(ps)

cfg={(K,l):{"dev":[[],[],[]],"hold":[[],[],[]],"dev_base":[[],[],[]],"hold_base":[[],[],[]]} for K in KS for l in LAMBDAS}
for qi in range(len(mq)):
    s,e=mo[qi];qmat=M[s:e]
    sem=SEM[qi];base=rank(sem)
    subset="dev" if qi in set(dev_idx.tolist()) else "hold"
    for K in KS:
        cand=base[:K];semz=z(sem[cand]);prz=z(PR[qi,cand])
        # baseline top-k within original semantic order
        for ix,kk in enumerate((1,3,10)):
            mats=[T[offs[j][0]:offs[j][1]] for j in cand[:kk]]
            cfg[(K,0.0)][subset+"_base"][ix].append(sf(qmat,vstack(mats)))
        for lam in LAMBDAS:
            sc=semz+lam*prz
            oo=np.lexsort((cand,-sc));rr=cand[oo]
            for ix,kk in enumerate((1,3,10)):
                mats=[T[offs[j][0]:offs[j][1]] for j in rr[:kk]]
                cfg[(K,lam)][subset][ix].append(sf(qmat,vstack(mats)))
    if (qi+1)%250==0:print("META_DONE",qi+1,flush=True)

rows=[]
for (K,lam),d in cfg.items():
    dev=[float(np.mean(x)) for x in d["dev"]];hold=[float(np.mean(x)) for x in d["hold"]]
    # baseline does not depend on lambda; use same K record
    # objective emphasizes consistent gain across 1/3/10
    base_dev=[]
    # retrieve from lambda=0 holder where stored
    bd=cfg[(K,0.0)]["dev_base"];bh=cfg[(K,0.0)]["hold_base"]
    base_dev=[float(np.mean(x)) for x in bd];base_hold=[float(np.mean(x)) for x in bh]
    delta=[dev[i]-base_dev[i] for i in range(3)]
    obj=float(np.mean(delta))
    rows.append({"K":K,"lambda":lam,"dev_u1":dev[0],"dev_u3":dev[1],"dev_u10":dev[2],
                 "dev_d1":delta[0],"dev_d3":delta[1],"dev_d10":delta[2],"dev_objective":obj,
                 "hold_u1":hold[0],"hold_u3":hold[1],"hold_u10":hold[2],
                 "hold_d1":hold[0]-base_hold[0],"hold_d3":hold[1]-base_hold[1],"hold_d10":hold[2]-base_hold[2]})
conf=pd.DataFrame(rows)
# deterministic selection: best dev objective, then smaller K, smaller lambda
conf=conf.sort_values(["dev_objective","K","lambda"],ascending=[False,True,True]).reset_index(drop=True)
best=conf.iloc[0].to_dict();BESTK=int(best["K"]);BESTL=float(best["lambda"])
print("SELECTED",BESTK,BESTL,best,flush=True)

# Refit query-available representations on all training, freeze K/lambda, evaluate validation once.
cv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X2=cv2.fit_transform(tr.case_prompt.astype(str));Q2=cv2.transform(va.case_prompt.astype(str));SEM2=(Q2@X2.T).toarray().astype(np.float32)
cs2=TruncatedSVD(256,n_iter=7,random_state=SEED);XC2=cs2.fit_transform(X2).astype(float);QC2=cs2.transform(Q2).astype(float)
tp=[pts(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)];vp=[pts(r,d) for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
rv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R2=rv2.fit_transform([" ".join(x) for x in tp]);rs2=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR2=rs2.fit_transform(R2).astype(float)
xm=XC2.mean(0,keepdims=True);ym=ZR2.mean(0,keepdims=True);AA=XC2-xm;YY=ZR2-ym
WW=np.linalg.solve(AA.T@AA+10*np.eye(AA.shape[1]),AA.T@YY);PR2=normalize((QC2-xm)@WW+ym)@normalize(ZR2).T

flat2=[p for ps in tp for p in ps];ev2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
TT=ev2.fit_transform(flat2);to=[];o=0
for ps in tp:to.append((o,o+len(ps)));o+=len(ps)
VV=ev2.transform([p for ps in vp for p in ps]);vo=[];o=0
for ps in vp:vo.append((o,o+len(ps)));o+=len(ps)

tl=[nw(x) for x in tr.final_diagnosis];vl=[nw(x) for x in va.final_diagnosis]
basevals={1:[],3:[],10:[]};gvals={1:[],3:[],10:[]};basehit={1:0,3:0,10:0};ghit={1:0,3:0,10:0}
rankings=[]
for i in range(len(va)):
    sem=SEM2[i];base=rank(sem);cand=base[:BESTK]
    score=z(sem[cand])+BESTL*z(PR2[i,cand]);oo=np.lexsort((cand,-score));rr=cand[oo]
    s,e=vo[i];qmat=VV[s:e]
    for kk in (1,3,10):
        basevals[kk].append(sf(qmat,vstack([TT[to[j][0]:to[j][1]] for j in base[:kk]])))
        gvals[kk].append(sf(qmat,vstack([TT[to[j][0]:to[j][1]] for j in rr[:kk]])))
        basehit[kk]+=int(any(tl[j]==vl[i] for j in base[:kk]))
        ghit[kk]+=int(any(tl[j]==vl[i] for j in rr[:kk]))
    rankings.append({"query_index":i,"baseline_top10":base[:10].tolist(),"g2_003_top10":rr[:10].tolist()})

summary={"experiment":"G2-003 semantic-anchored wide-pool utility fusion","status":"completed_validation_development",
 "policy":"K/lambda selected on train-only meta-dev, checked on train-only meta-holdout, then frozen for one validation evaluation. Test not used.",
 "meta":{"n_ref":len(ref),"n_queries":len(mq),"n_dev":len(dev_idx),"n_holdout":len(hold_idx),
         "selected_K":BESTK,"selected_lambda":BESTL,"selected_dev_objective":float(best["dev_objective"]),
         "selected_train_holdout_delta":[float(best["hold_d1"]),float(best["hold_d3"]),float(best["hold_d10"])]},
 "validation":{}}
for kk in (1,3,10):
    b=np.array(basevals[kk]);g=np.array(gvals[kk]);d=g-b;rng=np.random.default_rng(SEED+kk);boots=[]
    for _ in range(5000):
        ids=rng.integers(0,len(d),len(d));boots.append(float(d[ids].mean()))
    summary["validation"][str(kk)]={"tfidf":float(b.mean()),"g2_003":float(g.mean()),"delta":float(d.mean()),
      "ci95":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))],
      "improved":int((d>0).sum()),"worsened":int((d<0).sum()),
      "exact_hit_tfidf":basehit[kk]/len(va),"exact_hit_g2_003":ghit[kk]/len(va)}

conf.to_csv(OUT/"G2_003_CONFIG_SEARCH.csv",index=False)
with (OUT/"G2_003_RANKINGS.jsonl").open("w") as f:
    for x in rankings:f.write(json.dumps(x)+"\n")
(OUT/"G2_003_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")

md=["# G2-003 — Semantic-Anchored Wide-Pool Utility Fusion","",
f"Selected entirely on train meta-development: **K={BESTK}, lambda={BESTL}**.","",
f"Train meta-holdout deltas @1/@3/@10: {best['hold_d1']:+.5f} / {best['hold_d3']:+.5f} / {best['hold_d10']:+.5f}.","",
"| k | TF-IDF | G2-003 | Delta | 95% CI | exact-hit baseline → G2 |",
"|---:|---:|---:|---:|---|---:|"]
for kk in (1,3,10):
    x=summary["validation"][str(kk)]
    md.append(f"| {kk} | {x['tfidf']:.5f} | {x['g2_003']:.5f} | {x['delta']:+.5f} | [{x['ci95'][0]:.5f}, {x['ci95'][1]:.5f}] | {100*x['exact_hit_tfidf']:.1f}% → {100*x['exact_hit_g2_003']:.1f}% |")
(OUT/"G2_003_REPORT.md").write_text("\n".join(md)+"\n")
print(json.dumps(summary,indent=2))
