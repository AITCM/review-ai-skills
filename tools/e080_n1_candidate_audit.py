from __future__ import annotations
import json,re,unicodedata,pathlib,hashlib
import numpy as np,pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e080-n1-output");OUT.mkdir(exist_ok=True)
SEED=20261007
PAT=re.compile(r'^\s*(?:\((\d+)\)|(\d+)[.)])\s*',re.M)
KS=(1,3,10)

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
def mask_points(r,d):
    dd=nw(d);out=[]
    for x in split_reason(r):
        xx=nw(x)
        if dd and dd in xx: xx=xx.replace(dd," diagnosismask ")
        out.append(xx)
    return out
def rank(scores,k):
    ids=np.arange(len(scores))
    return np.lexsort((ids,-np.asarray(scores)))[:k]
def f1(rec,prec):
    den=rec+prec
    return np.where(den>0,2*rec*prec/den,0.0)
def set_u(qmax,psum,counts,ids):
    ids=np.asarray(ids,dtype=int)
    rec=float(np.max(qmax[:,ids],axis=1).mean())
    prec=float(psum[ids].sum()/counts[ids].sum())
    return float(2*rec*prec/(rec+prec)) if rec+prec else 0.0
def greedy(qmax,psum,counts,pool,kmax=10):
    pool=np.asarray(sorted(set(int(x) for x in pool)),dtype=int)
    chosen=[];alive=np.ones(len(pool),dtype=bool)
    recvec=np.zeros(qmax.shape[0],dtype=np.float32);ps=0.0;pc=0;scores={}
    for step in range(1,min(kmax,len(pool))+1):
        recs=np.maximum(recvec[:,None],qmax[:,pool]).mean(axis=0)
        precs=(ps+psum[pool])/(pc+counts[pool])
        vals=f1(recs,precs);vals[~alive]=-np.inf
        order=np.lexsort((pool,-vals));pos=int(order[0]);j=int(pool[pos])
        chosen.append(j);alive[pos]=False
        recvec=np.maximum(recvec,qmax[:,j]);ps+=float(psum[j]);pc+=int(counts[j])
        if step in KS:scores[step]=float(vals[pos])
    return chosen,scores
def rrf_order(routes,kconst=60):
    scores={}
    for arr in routes:
        for r,j in enumerate(arr,1):
            scores[int(j)]=scores.get(int(j),0.0)+1.0/(kconst+r)
    ids=np.asarray(sorted(scores),dtype=int)
    vals=np.asarray([scores[int(j)] for j in ids])
    return ids[np.lexsort((ids,-vals))]
def boot_ci(x,seed=20261007,n=3000):
    x=np.asarray(x,float);rng=np.random.default_rng(seed);vals=[]
    for _ in range(n):
        z=rng.integers(0,len(x),len(x));vals.append(float(x[z].mean()))
    return [float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))]

tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
assert len(tr)==13092 and len(va)==500

# Case-space representations. Validation contributes no fitting information.
cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=cv.fit_transform(tr.case_prompt.astype(str));Q=cv.transform(va.case_prompt.astype(str))
csvd=TruncatedSVD(256,n_iter=7,random_state=SEED)
XC=csvd.fit_transform(X).astype(np.float32);QC=csvd.transform(Q).astype(np.float32)
XCN=normalize(XC);QCN=normalize(QC)

# Privileged historical reasoning representation; diagnosis phrase masked before fitting.
tp=[mask_points(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
vp=[mask_points(r,d) for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
rdocs=[" ".join(x) for x in tp]
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R=rv.fit_transform(rdocs)
rsvd=TruncatedSVD(128,n_iter=7,random_state=SEED)
ZR=rsvd.fit_transform(R).astype(np.float32)
xm=XC.mean(0,keepdims=True);ym=ZR.mean(0,keepdims=True)
A=XC-xm;Y=ZR-ym
W=np.linalg.solve(A.T@A+10*np.eye(A.shape[1],dtype=np.float32),A.T@Y)
P=normalize((QC-xm)@W+ym)
ZN=normalize(ZR)

# Formal validation evaluator, fitted on training reasoning points only.
flat=[p for ps in tp for p in ps]
ev=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
T=ev.fit_transform(flat);V=ev.transform([p for ps in vp for p in ps])
counts=np.asarray([len(z) for z in tp],dtype=np.int32)
starts=np.concatenate(([0],np.cumsum(counts)[:-1])).astype(np.int64)
vo=[];o=0
for ps in vp:vo.append((o,o+len(ps)));o+=len(ps)
Tt=T.T.tocsr();all_ids=np.arange(len(tr),dtype=int)

route_names=["LEX50","LEX100","LAT50","LAT100","PRED50","PRED100","DUAL50","TRI50","TRI100"]
rows=[];ordered_scores={name:{k:[] for k in KS} for name in ["LEX","LAT","PRED","RRF_DUAL50","RRF_TRI50","RRF_TRI100"]}
oracle_scores={name:{k:[] for k in KS} for name in route_names+["GLOBAL"]}
contain={name:[] for name in route_names}
sizes={name:[] for name in route_names}

for i in range(len(va)):
    lex=(Q.getrow(i)@X.T).toarray().ravel()
    lat=np.asarray(QCN[i]@XCN.T).ravel()
    pred=np.asarray(P[i]@ZN.T).ravel()
    l50=rank(lex,50);l100=rank(lex,100)
    a50=rank(lat,50);a100=rank(lat,100)
    p50=rank(pred,50);p100=rank(pred,100)
    pools={
      "LEX50":l50,"LEX100":l100,"LAT50":a50,"LAT100":a100,"PRED50":p50,"PRED100":p100,
      "DUAL50":np.asarray(sorted(set(l50)|set(p50)),dtype=int),
      "TRI50":np.asarray(sorted(set(l50)|set(a50)|set(p50)),dtype=int),
      "TRI100":np.asarray(sorted(set(l100)|set(a100)|set(p100)),dtype=int)
    }
    rrfd=rrf_order([l50,p50]);rrft=rrf_order([l50,a50,p50]);rrft100=rrf_order([l100,a100,p100])

    s,e=vo[i];S=(V[s:e]@Tt).toarray().astype(np.float32,copy=False)
    qmax=np.stack([np.maximum.reduceat(row,starts) for row in S],axis=0)
    pmax=S.max(axis=0);psum=np.add.reduceat(pmax,starts)
    rec=qmax.mean(axis=0);prec=psum/counts;util=f1(rec,prec)
    gbest=int(np.lexsort((all_ids,-util))[0])

    for name,pool in pools.items():
        sizes[name].append(len(pool));contain[name].append(gbest in set(int(x) for x in pool))
        _,gs=greedy(qmax,psum,counts,pool,10)
        for k in KS: oracle_scores[name][k].append(gs[k])
    _,gg=greedy(qmax,psum,counts,all_ids,10)
    for k in KS: oracle_scores["GLOBAL"][k].append(gg[k])

    orders={"LEX":l100,"LAT":a100,"PRED":p100,"RRF_DUAL50":rrfd,"RRF_TRI50":rrft,"RRF_TRI100":rrft100}
    for name,order in orders.items():
        for k in KS: ordered_scores[name][k].append(set_u(qmax,psum,counts,order[:k]))

    rows.append({"query_index":i,"global_best_single_train_index":gbest,"global_best_single_utility":float(util[gbest]),
                 **{f"{name}_contains_global_best":bool(contain[name][-1]) for name in route_names},
                 **{f"{name}_size":int(sizes[name][-1]) for name in route_names}})

pd.DataFrame(rows).to_csv(OUT/"E080_N1_PER_QUERY.csv",index=False)
summary={
 "experiment":"E080-N1 second-generation candidate-generation audit",
 "status":"validation-development only; MedCaseReasoning test untouched",
 "dataset_revision":REV,"n_train":len(tr),"n_validation":len(va),
 "query_inputs_at_retrieval":"case_prompt only",
 "historical_privileged_fields":"diagnosis-masked diagnostic_reasoning used to fit reasoning space",
 "candidate_routes":{},
 "ordered_retrieval":{},
 "interpretation_policy":"Validation reasoning is used only for evaluation/oracle analysis. No test data or T001 outcomes are used for development."
}
for name in route_names:
    summary["candidate_routes"][name]={
      "mean_pool_size":float(np.mean(sizes[name])),
      "global_best_single_containment_rate":float(np.mean(contain[name])),
      "global_best_single_containment_ci95":boot_ci(np.asarray(contain[name],float),SEED+len(name)),
      **{f"greedy_oracle_softF1@{k}":float(np.mean(oracle_scores[name][k])) for k in KS}
    }
summary["global_greedy_oracle"]={f"softF1@{k}":float(np.mean(oracle_scores["GLOBAL"][k])) for k in KS}
for name in ordered_scores:
    summary["ordered_retrieval"][name]={f"softF1@{k}":float(np.mean(ordered_scores[name][k])) for k in KS}
(OUT/"E080_N1_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")

md=["# E080-N1 Second-Generation Candidate Audit","",
"Validation-development only. The 897-case MedCaseReasoning test set is not read.","",
"## Candidate pools","",
"| route | mean pool | contains global-best single | oracle@1 | oracle@3 | oracle@10 |",
"|---|---:|---:|---:|---:|---:|"]
for name in route_names:
    x=summary["candidate_routes"][name]
    md.append(f"| {name} | {x['mean_pool_size']:.1f} | {100*x['global_best_single_containment_rate']:.1f}% | {x['greedy_oracle_softF1@1']:.5f} | {x['greedy_oracle_softF1@3']:.5f} | {x['greedy_oracle_softF1@10']:.5f} |")
md += ["","## Zero-learned ordered retrieval","",
"| ranking | @1 | @3 | @10 |","|---|---:|---:|---:|"]
for name,x in summary["ordered_retrieval"].items():
    md.append(f"| {name} | {x['softF1@1']:.5f} | {x['softF1@3']:.5f} | {x['softF1@10']:.5f} |")
md += ["","## Global oracle",json.dumps(summary["global_greedy_oracle"],indent=2)]
(OUT/"E080_N1_REPORT.md").write_text("\n".join(md)+"\n")
print(json.dumps(summary,indent=2))
