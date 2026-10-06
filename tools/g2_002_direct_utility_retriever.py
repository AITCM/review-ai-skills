from __future__ import annotations
import json,re,unicodedata,pathlib,hashlib
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("g2-002-output");OUT.mkdir(exist_ok=True)
SEED=20261006; RNG=np.random.default_rng(SEED)
DCASE=256;DREASON=128;N_META=1000;EPOCHS=12;BATCH=2048;LR=.01;L2=1e-4
PAT=re.compile(r'^\s*(?:\((\d+)\)|(\d+)[.)])\s*',re.M)

def nw(s): return " ".join(re.findall(r"\w+",unicodedata.normalize("NFKC",str(s)).casefold()))
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
    dd=nw(d);out=[]
    for x in split_reason(r):
        xx=nw(x)
        if dd and dd in xx:xx=xx.replace(dd," diagnosismask ")
        out.append(xx)
    return out
def f1(rec,prec):
    den=np.asarray(rec)+np.asarray(prec);out=np.zeros_like(den,dtype=float)
    np.divide(2*np.asarray(rec)*np.asarray(prec),den,out=out,where=den>0)
    return out
def set_u(qmax,psum,counts,ids):
    ids=np.asarray(ids,int);rec=float(np.max(qmax[:,ids],axis=1).mean());prec=float(psum[ids].sum()/counts[ids].sum())
    return float(2*rec*prec/(rec+prec)) if rec+prec else 0.
def rank(scores,k=None):
    ids=np.arange(len(scores));o=np.lexsort((ids,-scores));return o if k is None else o[:k]
def sigmoid_neg_margin(m):
    # sigmoid(-m), numerically stable
    x=np.clip(m,-40,40);return 1.0/(1.0+np.exp(x))

tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet",columns=["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"])
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet",columns=["case_prompt","diagnostic_reasoning","final_diagnosis"])
assert len(tr)==13092 and len(va)==500

# Train-only representation spaces; validation outcomes never enter fitting.
cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
Xsp=cv.fit_transform(tr.case_prompt.astype(str));Qsp=cv.transform(va.case_prompt.astype(str))
cs=TruncatedSVD(DCASE,n_iter=7,random_state=SEED)
X=normalize(cs.fit_transform(Xsp).astype(np.float32));Q=normalize(cs.transform(Qsp).astype(np.float32))
SEM_T=(Xsp@Xsp.T).tocsr();SEM_V=(Qsp@Xsp.T).toarray().astype(np.float32)

tp=[pts(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
vp=[pts(r,d) for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
Rsp=rv.fit_transform([" ".join(x) for x in tp])
rs=TruncatedSVD(DREASON,n_iter=7,random_state=SEED)
Z=normalize(rs.fit_transform(Rsp).astype(np.float32))

# Ridge reconstruction initializer = first-generation predicted-reasoning signal.
W0=np.linalg.solve(X.T@X+10*np.eye(DCASE,dtype=np.float32),X.T@Z).astype(np.float32)

# Train-only oracle utility evaluator.
flat=[p for ps in tp for p in ps]
ev=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
T=ev.fit_transform(flat);Tt=T.T.tocsr()
counts=np.asarray([len(x) for x in tp],np.int32);starts=np.concatenate(([0],np.cumsum(counts)[:-1])).astype(np.int64)
offs=[];o=0
for x in tp:offs.append((o,o+len(x)));o+=len(x)

# Deterministic meta-query sample by PMCID hash.
hv=np.array([int(hashlib.sha256(("G2-002:"+str(x)).encode()).hexdigest()[:16],16) for x in tr.pmcid],dtype=np.uint64)
meta_ids=np.argsort(hv)[:N_META].astype(int)

pairs_q=[];pairs_p=[];pairs_n=[];pair_du=[]
allids=np.arange(len(tr),dtype=int)
for n,qi in enumerate(meta_ids,1):
    s,e=offs[qi];qmat=T[s:e];S=(qmat@Tt).toarray().astype(np.float32,copy=False)
    qmax=np.stack([np.maximum.reduceat(row,starts) for row in S],axis=0)
    pmax=S.max(axis=0);psum=np.add.reduceat(pmax,starts)
    rec=qmax.mean(axis=0);prec=psum/counts;util=f1(rec,prec)
    util[qi]=-1.0
    sem=SEM_T.getrow(qi).toarray().ravel().astype(np.float32);sem[qi]=-1.0
    pos=rank(util,5)

    sem100=rank(sem,100)
    sem100=np.array([j for j in sem100 if j not in set(pos)],int)
    hard=sem100[np.argsort(util[sem100])[:10]]

    low_pool=np.where(util<=np.percentile(util[util>=0],20))[0]
    rr=np.random.default_rng(SEED+int(qi))
    rand=rr.choice(low_pool,size=min(10,len(low_pool)),replace=False)

    neg=np.unique(np.concatenate([hard,rand]))
    for p in pos:
        for ng in neg:
            du=float(util[p]-util[ng])
            if du<.03:continue
            pairs_q.append(qi);pairs_p.append(int(p));pairs_n.append(int(ng));pair_du.append(du)
    if n%100==0: print("META",n,"pairs",len(pairs_q),flush=True)

pq=np.asarray(pairs_q,np.int32);pp=np.asarray(pairs_p,np.int32);pn=np.asarray(pairs_n,np.int32);pdu=np.asarray(pair_du,np.float32)
assert len(pq)>10000

# Direct utility ranker: score(q,c)=x_q W z_c + beta*semantic(q,c)
# Initialize W from ridge reconstruction, then optimize pairwise utility preferences.
W=W0.copy();beta=np.float32(1.0)
mW=np.zeros_like(W);vW=np.zeros_like(W);mb=np.float32(0);vb=np.float32(0)
b1=.9;b2=.999;eps=1e-8;t=0
history=[]
for ep in range(1,EPOCHS+1):
    order=RNG.permutation(len(pq));loss_sum=0.;margin_sum=0.
    for st in range(0,len(order),BATCH):
        ids=order[st:st+BATCH];q=pq[ids];p=pp[ids];n=pn[ids]
        Xq=X[q];DZ=Z[p]-Z[n]
        dsem=np.asarray([SEM_T[qj,pj]-SEM_T[qj,nj] for qj,pj,nj in zip(q,p,n)],dtype=np.float32)
        margin=np.einsum("bi,ij,bj->b",Xq,W,DZ)+beta*dsem
        g=sigmoid_neg_margin(margin)
        wt=np.clip(pdu[ids]/.08,.5,2.0)
        gw=(g*wt).astype(np.float32)
        gradW=-(Xq.T@(gw[:,None]*DZ))/len(ids)+L2*W
        gradb=-np.mean(gw*dsem)

        t+=1
        mW=b1*mW+(1-b1)*gradW;vW=b2*vW+(1-b2)*(gradW*gradW)
        mh=mW/(1-b1**t);vh=vW/(1-b2**t);W-=LR*mh/(np.sqrt(vh)+eps)
        mb=b1*mb+(1-b1)*gradb;vb=b2*vb+(1-b2)*(gradb*gradb)
        beta-=LR*(mb/(1-b1**t))/(np.sqrt(vb/(1-b2**t))+eps)
        loss_sum+=float(np.logaddexp(0,-np.clip(margin,-40,40)).mean())*len(ids)
        margin_sum+=float(margin.mean())*len(ids)
    history.append({"epoch":ep,"loss":loss_sum/len(pq),"mean_margin":margin_sum/len(pq),"beta":float(beta)})
    print("EPOCH",ep,history[-1],flush=True)

# Validation global rankings.
ridge_pred=normalize(Q@W0)@Z.T
g2_bilin=(Q@W)@Z.T
G2=g2_bilin+float(beta)*SEM_V

# Validation outcome evaluator: fit on TRAIN points only.
V=ev.transform([p for ps in vp for p in ps]);vo=[];o=0
for ps in vp:vo.append((o,o+len(ps)));o+=len(ps)

tl=[nw(x) for x in tr.final_diagnosis];vl=[nw(x) for x in va.final_diagnosis]
methods={"tfidf":SEM_V,"ridge_pred":ridge_pred.astype(np.float32),"g2_direct_utility":G2.astype(np.float32)}
ranks={m:[rank(scores[i]) for i in range(len(va))] for m,scores in methods.items()}

summary={"experiment":"G2-002 direct pairwise utility-aware global retrieval","status":"completed_validation_development",
 "policy":"Train-only utility supervision; validation used only for development evaluation. MedCaseReasoning test not read.",
 "training":{"meta_queries":len(meta_ids),"pairwise_preferences":len(pq),"epochs":EPOCHS,"case_dim":DCASE,"reason_dim":DREASON,
             "final_beta_semantic":float(beta),"history":history},
 "results":{},"candidate_reachability":{}}

per=[]
for i in range(len(va)):
    s,e=vo[i];qmat=V[s:e];S=(qmat@Tt).toarray().astype(np.float32,copy=False)
    qmax=np.stack([np.maximum.reduceat(row,starts) for row in S],axis=0)
    pmax=S.max(axis=0);psum=np.add.reduceat(pmax,starts)
    rec=qmax.mean(axis=0);prec=psum/counts;util=f1(rec,prec)
    glob=rank(util);gb=int(glob[0]);gt10=set(map(int,glob[:10]))
    row={"query_index":i,"global_best":gb,"global_best_utility":float(util[gb])}
    for name in ranks:
        r=ranks[name][i]
        for k in (1,3,10):
            row[f"{name}_u{k}"]=set_u(qmax,psum,counts,r[:k])
        for k in (20,50,100,200):
            pool=r[:k];row[f"{name}_bestrec{k}"]=float(gb in set(map(int,pool)))
            row[f"{name}_top10rec{k}"]=len(gt10 & set(map(int,pool)))/10.0
    per.append(row)

df=pd.DataFrame(per)
for name in ranks:
    summary["results"][name]={}
    for k in (1,3,10):
        vals=df[f"{name}_u{k}"].to_numpy()
        summary["results"][name][str(k)]={"mean_reasoning_utility":float(vals.mean())}
    summary["candidate_reachability"][name]={}
    for k in (20,50,100,200):
        summary["candidate_reachability"][name][str(k)]={
          "global_best_single_recall":float(df[f"{name}_bestrec{k}"].mean()),
          "oracle_top10_candidate_recall":float(df[f"{name}_top10rec{k}"].mean())
        }
    for k in (1,3,10):
        hit=sum(any(tl[j]==vl[i] for j in ranks[name][i][:k]) for i in range(len(va)))
        summary["results"][name][str(k)]["exact_label_hit_rate"]=hit/len(va)

for k in (1,3,10):
    b=df[f"tfidf_u{k}"].to_numpy();g=df[f"g2_direct_utility_u{k}"].to_numpy();d=g-b
    rng=np.random.default_rng(SEED+k);boots=[]
    for _ in range(5000):
        ids=rng.integers(0,len(d),len(d));boots.append(float(d[ids].mean()))
    summary["results"]["g2_direct_utility"][str(k)]["delta_vs_tfidf"]=float(d.mean())
    summary["results"]["g2_direct_utility"][str(k)]["ci95_vs_tfidf"]=[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))]
    summary["results"]["g2_direct_utility"][str(k)]["improved"]=int((d>0).sum())
    summary["results"]["g2_direct_utility"][str(k)]["worsened"]=int((d<0).sum())

pd.DataFrame(per).to_csv(OUT/"G2_002_PER_QUERY.csv",index=False)
(OUT/"G2_002_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
np.savez_compressed(OUT/"G2_002_MODEL.npz",W=W,beta=np.asarray([beta],np.float32),W0=W0,meta_ids=meta_ids,pq=pq,pp=pp,pn=pn,pdu=pdu)

md=["# G2-002 — Direct Utility-Aware Global Retrieval","",
"Train-only pairwise utility supervision; validation-only development evaluation. MedCaseReasoning test is not used.","",
f"Training pairs: **{len(pq):,}**; final semantic coefficient beta: **{float(beta):.4f}**.","",
"| Method | U@1 | U@3 | U@10 | exact-hit@10 | best-global recall@50 |",
"|---|---:|---:|---:|---:|---:|"]
for name in ("tfidf","ridge_pred","g2_direct_utility"):
    x=summary["results"][name];cr=summary["candidate_reachability"][name]["50"]["global_best_single_recall"]
    md.append(f"| {name} | {x['1']['mean_reasoning_utility']:.5f} | {x['3']['mean_reasoning_utility']:.5f} | {x['10']['mean_reasoning_utility']:.5f} | {100*x['10']['exact_label_hit_rate']:.1f}% | {100*cr:.1f}% |")
g=summary["results"]["g2_direct_utility"]
md += ["","## Delta vs TF-IDF",
f"- @1: {g['1']['delta_vs_tfidf']:+.5f}, 95% CI {g['1']['ci95_vs_tfidf']}",
f"- @3: {g['3']['delta_vs_tfidf']:+.5f}, 95% CI {g['3']['ci95_vs_tfidf']}",
f"- @10: {g['10']['delta_vs_tfidf']:+.5f}, 95% CI {g['10']['ci95_vs_tfidf']}",
"",
"Interpretation must remain development-only until a new external confirmatory evaluation is prespecified."
]
(OUT/"G2_002_REPORT.md").write_text("\n".join(md)+"\n")
print(json.dumps(summary,indent=2))
