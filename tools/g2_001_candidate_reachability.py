from __future__ import annotations
import json,re,unicodedata,pathlib
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("g2-001-output");OUT.mkdir(exist_ok=True)
SEED=20261006
BUDGETS=(20,50,100,200)
PAT=re.compile(r'^\s*(?:\((\d+)\)|(\d+)[.)])\s*',re.M)

def nw(s):
    return " ".join(re.findall(r"\w+",unicodedata.normalize("NFKC",str(s)).casefold()))

def split_reason(s):
    s=str(s);ms=list(PAT.finditer(s))
    if not ms or s[:ms[0].start()].strip(): return [s.strip()]
    nums=[int(m.group(1) or m.group(2)) for m in ms]
    if nums!=list(range(1,len(ms)+1)): return [s.strip()]
    out=[]
    for i,m in enumerate(ms):
        e=ms[i+1].start() if i+1<len(ms) else len(s)
        x=s[m.end():e].strip()
        if x: out.append(x)
    return out or [s.strip()]

def pts(r,d):
    dd=nw(d);out=[]
    for x in split_reason(r):
        xx=nw(x)
        if dd and dd in xx: xx=xx.replace(dd," diagnosismask ")
        out.append(xx)
    return out

def f1(rec,prec):
    den=rec+prec
    out=np.zeros_like(np.asarray(den,dtype=float),dtype=float)
    np.divide(2*np.asarray(rec)*np.asarray(prec),den,out=out,where=np.asarray(den)>0)
    return out

def set_u(qmax,psum,counts,ids):
    ids=np.asarray(ids,dtype=int)
    rec=float(np.max(qmax[:,ids],axis=1).mean())
    prec=float(psum[ids].sum()/counts[ids].sum())
    return float(2*rec*prec/(rec+prec)) if rec+prec else 0.0

def greedy10(qmax,psum,counts,pool):
    pool=np.asarray(pool,dtype=int)
    chosen=[];alive=np.ones(len(pool),bool)
    recvec=np.zeros(qmax.shape[0],np.float32); ps=0.0; pc=0
    score10=0.0
    for step in range(1,11):
        recs=np.maximum(recvec[:,None],qmax[:,pool]).mean(axis=0)
        precs=(ps+psum[pool])/(pc+counts[pool])
        vals=f1(recs,precs); vals[~alive]=-np.inf
        order=np.lexsort((pool,-vals));pos=int(order[0]);j=int(pool[pos])
        chosen.append(j);alive[pos]=False
        recvec=np.maximum(recvec,qmax[:,j]);ps+=float(psum[j]);pc+=int(counts[j])
        if step==10: score10=float(vals[pos])
    return chosen,score10

def interleave(a,b,k):
    out=[];seen=set();ia=ib=0
    while len(out)<k and (ia<len(a) or ib<len(b)):
        if ia<len(a):
            j=int(a[ia]);ia+=1
            if j not in seen:seen.add(j);out.append(j)
            if len(out)>=k:break
        if ib<len(b):
            j=int(b[ib]);ib+=1
            if j not in seen:seen.add(j);out.append(j)
    return np.asarray(out,dtype=int)

tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet",columns=["case_prompt","diagnostic_reasoning","final_diagnosis"])
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet",columns=["case_prompt","diagnostic_reasoning","final_diagnosis"])
assert len(tr)==13092 and len(va)==500

# Query-available candidate-generation signals only.
cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=cv.fit_transform(tr.case_prompt.astype(str));Q=cv.transform(va.case_prompt.astype(str))
SEM=(Q@X.T).toarray().astype(np.float32)

case_svd=TruncatedSVD(256,n_iter=7,random_state=SEED)
XC=case_svd.fit_transform(X).astype(float); QC=case_svd.transform(Q).astype(float)
tp=[pts(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
vp=[pts(r,d) for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R=rv.fit_transform([" ".join(x) for x in tp])
rsvd=TruncatedSVD(128,n_iter=7,random_state=SEED)
ZR=rsvd.fit_transform(R).astype(float)
xm=XC.mean(0,keepdims=True); ym=ZR.mean(0,keepdims=True)
A=XC-xm;Y=ZR-ym
W=np.linalg.solve(A.T@A+10*np.eye(A.shape[1]),A.T@Y)
P=normalize((QC-xm)@W+ym);Z=normalize(ZR)
PR=(P@Z.T).astype(np.float32)

# Outcome-only utility evaluator, fitted on train reasoning points.
flat_t=[p for z in tp for p in z]; flat_v=[p for z in vp for p in z]
ev=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
T=ev.fit_transform(flat_t);V=ev.transform(flat_v)
counts=np.asarray([len(z) for z in tp],dtype=np.int32)
starts=np.concatenate(([0],np.cumsum(counts)[:-1])).astype(np.int64)
vo=[];o=0
for z in vp:vo.append((o,o+len(z)));o+=len(z)
Tt=T.T.tocsr();all_ids=np.arange(len(tr),dtype=int)

records=[]
agg={g:{str(k):{"best":[],"best_recall":[],"top10_recall":[],"greedy10":[]} for k in BUDGETS}
     for g in ("tfidf","pred_reason","interleave","rrf")}
global_best_vals=[];global_greedy10_vals=[];tfidf_global_ranks=[]

for i in range(len(va)):
    s,e=vo[i]
    S=(V[s:e]@Tt).toarray().astype(np.float32,copy=False)
    qmax=np.stack([np.maximum.reduceat(row,starts) for row in S],axis=0)
    pmax=S.max(axis=0); psum=np.add.reduceat(pmax,starts)
    rec=qmax.mean(axis=0);prec=psum/counts;util=f1(rec,prec)
    glob_order=np.lexsort((all_ids,-util))
    gbest=int(glob_order[0]); gtop10=set(map(int,glob_order[:10]))
    _,gg10=greedy10(qmax,psum,counts,all_ids)
    global_best_vals.append(float(util[gbest]));global_greedy10_vals.append(gg10)

    sem_order=np.lexsort((all_ids,-SEM[i]))
    pr_order=np.lexsort((all_ids,-PR[i]))
    inv_sem=np.empty(len(tr),int);inv_sem[sem_order]=np.arange(len(tr))
    inv_pr=np.empty(len(tr),int);inv_pr[pr_order]=np.arange(len(tr))
    rrf=1.0/(60.0+inv_sem)+1.0/(60.0+inv_pr)
    rrf_order=np.lexsort((all_ids,-rrf))
    tfidf_global_ranks.append(int(inv_sem[gbest]+1))

    for k in BUDGETS:
        pools={
          "tfidf":sem_order[:k],
          "pred_reason":pr_order[:k],
          "interleave":interleave(sem_order,pr_order,k),
          "rrf":rrf_order[:k]
        }
        for g,pool in pools.items():
            best=float(util[pool].max())
            _,g10=greedy10(qmax,psum,counts,pool)
            agg[g][str(k)]["best"].append(best)
            agg[g][str(k)]["best_recall"].append(float(gbest in set(map(int,pool))))
            agg[g][str(k)]["top10_recall"].append(len(gtop10 & set(map(int,pool)))/10.0)
            agg[g][str(k)]["greedy10"].append(g10)
    records.append({"query_index":i,"global_best_train_index":gbest,
                    "global_best_utility":float(util[gbest]),
                    "tfidf_rank_of_global_best":int(inv_sem[gbest]+1),
                    "pred_reason_rank_of_global_best":int(inv_pr[gbest]+1)})

summary={
  "experiment":"G2-001 validation-only candidate-generator reachability",
  "status":"completed_validation_development",
  "dataset_revision":REV,
  "policy":"No MedCaseReasoning test outcomes read or used. Validation is development-only.",
  "n_validation":len(va),
  "generators":{
    "tfidf":"case_prompt TF-IDF unigram+bigram cosine",
    "pred_reason":"global retrieval by train-fitted case->reasoning projection similarity",
    "interleave":"deterministic alternating union of TF-IDF and predicted-reasoning ranks",
    "rrf":"reciprocal-rank fusion of full TF-IDF and predicted-reasoning ranks, k0=60"
  },
  "global_oracle":{
    "mean_best_single_utility":float(np.mean(global_best_vals)),
    "mean_greedy10_utility":float(np.mean(global_greedy10_vals)),
    "median_tfidf_rank_of_global_best":float(np.median(tfidf_global_ranks)),
    "iqr_tfidf_rank_of_global_best":[float(np.percentile(tfidf_global_ranks,25)),float(np.percentile(tfidf_global_ranks,75))]
  },
  "results":{}
}
for g in agg:
    summary["results"][g]={}
    for k in BUDGETS:
        x=agg[g][str(k)]
        summary["results"][g][str(k)]={
          "global_best_single_recall":float(np.mean(x["best_recall"])),
          "oracle_top10_candidate_recall":float(np.mean(x["top10_recall"])),
          "mean_best_single_utility_in_pool":float(np.mean(x["best"])),
          "fraction_of_global_best_single_utility":float(np.mean(x["best"])/np.mean(global_best_vals)),
          "mean_greedy10_oracle_utility_in_pool":float(np.mean(x["greedy10"])),
          "fraction_of_global_greedy10_utility":float(np.mean(x["greedy10"])/np.mean(global_greedy10_vals))
        }

pd.DataFrame(records).to_csv(OUT/"G2_001_PER_QUERY.csv",index=False)
(OUT/"G2_001_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")

md=["# G2-001 — Validation-only Candidate Generator Reachability","",
"Development-only analysis. MedCaseReasoning test is not used.","",
f"Global oracle mean best-single utility: **{summary['global_oracle']['mean_best_single_utility']:.5f}**.",
f"Global greedy Top-10 utility: **{summary['global_oracle']['mean_greedy10_utility']:.5f}**.",
f"Median lexical rank of global best single candidate: **{summary['global_oracle']['median_tfidf_rank_of_global_best']:.0f}**.",
"",
"| Generator | Budget | Global-best recall | Oracle-top10 recall | Best utility in pool | Greedy-set@10 |",
"|---|---:|---:|---:|---:|---:|"]
for g in ("tfidf","pred_reason","interleave","rrf"):
    for k in BUDGETS:
        x=summary["results"][g][str(k)]
        md.append(f"| {g} | {k} | {100*x['global_best_single_recall']:.1f}% | {100*x['oracle_top10_candidate_recall']:.1f}% | {x['mean_best_single_utility_in_pool']:.5f} | {x['mean_greedy10_oracle_utility_in_pool']:.5f} |")
(OUT/"G2_001_REPORT.md").write_text("\n".join(md)+"\n")
print(json.dumps(summary,indent=2))
