from __future__ import annotations
import json,re,unicodedata,pathlib
import numpy as np,pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
MODEL="pritamdeka/S-PubMedBert-MS-MARCO";MODEL_REV="96786c7024f95c5aac7f2b9a18086c7b97b23036"
OUT=pathlib.Path("e080-n3-output");OUT.mkdir(exist_ok=True)
SEED=20261007;KS=(1,3,10)
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
def set_u(qmax,psum,counts,ids):
    ids=np.asarray(ids,dtype=int)
    rec=float(np.max(qmax[:,ids],axis=1).mean());prec=float(psum[ids].sum()/counts[ids].sum())
    return float(2*rec*prec/(rec+prec)) if rec+prec else 0.
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
def rrf(routes,kconst=60):
    s={}
    for arr in routes:
        for r,j in enumerate(arr,1):s[int(j)]=s.get(int(j),0.)+1/(kconst+r)
    ids=np.asarray(sorted(s),dtype=int);v=np.asarray([s[int(j)] for j in ids])
    return ids[np.lexsort((ids,-v))]
def boot_ci(x,seed,n=3000):
    x=np.asarray(x,float);rng=np.random.default_rng(seed);z=[]
    for _ in range(n):
        q=rng.integers(0,len(x),len(x));z.append(float(x[q].mean()))
    return [float(np.percentile(z,2.5)),float(np.percentile(z,97.5))]

tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
assert len(tr)==13092 and len(va)==500

# Sparse lexical route for reference.
cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=cv.fit_transform(tr.case_prompt.astype(str));Q=cv.transform(va.case_prompt.astype(str))

# Dense biomedical representations. Query uses case_prompt only.
tp=[mask_points(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
vp=[mask_points(r,d) for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
tr_rdoc=[" ".join(x) for x in tp]
model=SentenceTransformer(MODEL,revision=MODEL_REV,device="cpu");model.max_seq_length=256
case_texts=tr.case_prompt.astype(str).tolist()+va.case_prompt.astype(str).tolist()
CE=model.encode(case_texts,batch_size=64,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True).astype(np.float32)
TC=CE[:len(tr)];VQ=CE[len(tr):]
TR=model.encode(tr_rdoc,batch_size=64,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True).astype(np.float32)

# Formal utility evaluator remains the same diagnosis-masked TF-IDF point metric used in N1/N2.
flat=[p for ps in tp for p in ps]
ev=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
T=ev.fit_transform(flat);V=ev.transform([p for ps in vp for p in ps]);Tt=T.T.tocsr()
counts=np.asarray([len(x) for x in tp],np.int32);starts=np.concatenate(([0],np.cumsum(counts)[:-1])).astype(np.int64)
vo=[];o=0
for ps in vp:vo.append((o,o+len(ps)));o+=len(ps)
all_ids=np.arange(len(tr),dtype=int)

routes=["LEX100","DCASE100","DREASON100","HYBRID100","HYBRID200"]
sizes={x:[] for x in routes};contain={x:[] for x in routes};oracle={x:{k:[] for k in KS} for x in routes+["GLOBAL"]}
ordered={x:{k:[] for k in KS} for x in ["LEX","DCASE","DREASON","RRF_HYBRID100","RRF_HYBRID200"]}
rows=[]

for i in range(len(va)):
    lex=(Q.getrow(i)@X.T).toarray().ravel()
    dc=np.asarray(VQ[i]@TC.T).ravel()
    dr=np.asarray(VQ[i]@TR.T).ravel()
    l100=rank(lex,100);c100=rank(dc,100);r100=rank(dr,100)
    l200=rank(lex,200);c200=rank(dc,200);r200=rank(dr,200)
    pools={
      "LEX100":l100,"DCASE100":c100,"DREASON100":r100,
      "HYBRID100":np.asarray(sorted(set(l100)|set(c100)|set(r100)),dtype=int),
      "HYBRID200":np.asarray(sorted(set(l200)|set(c200)|set(r200)),dtype=int)
    }
    rr100=rrf([l100,c100,r100]);rr200=rrf([l200,c200,r200])
    s,e=vo[i];S=(V[s:e]@Tt).toarray().astype(np.float32,copy=False)
    qmax=np.stack([np.maximum.reduceat(row,starts) for row in S],axis=0)
    pmax=S.max(axis=0);psum=np.add.reduceat(pmax,starts)
    util=f1(qmax.mean(axis=0),psum/counts)
    gbest=int(np.lexsort((all_ids,-util))[0])
    for name,pool in pools.items():
        sizes[name].append(len(pool));contain[name].append(gbest in set(int(x) for x in pool))
        gs=greedy(qmax,psum,counts,pool,10)
        for k in KS:oracle[name][k].append(gs[k])
    gg=greedy(qmax,psum,counts,all_ids,10)
    for k in KS:oracle["GLOBAL"][k].append(gg[k])
    orders={"LEX":l200,"DCASE":c200,"DREASON":r200,"RRF_HYBRID100":rr100,"RRF_HYBRID200":rr200}
    for name,arr in orders.items():
        for k in KS:ordered[name][k].append(set_u(qmax,psum,counts,arr[:k]))
    rows.append({"query_index":i,"global_best_single_train_index":gbest,
                 **{f"{x}_contains_global_best":bool(contain[x][-1]) for x in routes},
                 **{f"{x}_size":int(sizes[x][-1]) for x in routes}})
pd.DataFrame(rows).to_csv(OUT/"E080_N3_PER_QUERY.csv",index=False)
summary={
 "experiment":"E080-N3 biomedical dense candidate-route audit","status":"validation-development only; test untouched",
 "model":{"name":MODEL,"revision":MODEL_REV,"max_seq_length":256},
 "n_train":len(tr),"n_validation":len(va),
 "retrieval_routes":{"DCASE":"S-PubMedBERT query case -> historical case cosine",
                     "DREASON":"S-PubMedBERT query case -> diagnosis-masked historical reasoning-document cosine"},
 "candidate_routes":{},"ordered_retrieval":{},
 "global_greedy_oracle":{f"softF1@{k}":float(np.mean(oracle["GLOBAL"][k])) for k in KS},
 "policy":"Query retrieval input is case_prompt only. Historical reasoning is diagnosis-masked. Validation reasoning is evaluation only; MedCaseReasoning test is not accessed."
}
for name in routes:
    summary["candidate_routes"][name]={
      "mean_pool_size":float(np.mean(sizes[name])),
      "global_best_single_containment_rate":float(np.mean(contain[name])),
      "containment_ci95":boot_ci(np.asarray(contain[name],float),SEED+len(name)),
      **{f"greedy_oracle_softF1@{k}":float(np.mean(oracle[name][k])) for k in KS}
    }
for name in ordered:
    summary["ordered_retrieval"][name]={f"softF1@{k}":float(np.mean(ordered[name][k])) for k in KS}
(OUT/"E080_N3_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
md=["# E080-N3 Biomedical Dense Candidate Audit","",
"Validation-development only; MedCaseReasoning test untouched.","",
"| route | mean pool | contains global-best | oracle@1 | oracle@3 | oracle@10 |",
"|---|---:|---:|---:|---:|---:|"]
for name in routes:
    x=summary["candidate_routes"][name]
    md.append(f"| {name} | {x['mean_pool_size']:.1f} | {100*x['global_best_single_containment_rate']:.1f}% | {x['greedy_oracle_softF1@1']:.5f} | {x['greedy_oracle_softF1@3']:.5f} | {x['greedy_oracle_softF1@10']:.5f} |")
md += ["","## Ordered retrieval","",
"| ranker | @1 | @3 | @10 |","|---|---:|---:|---:|"]
for name,x in summary["ordered_retrieval"].items():
    md.append(f"| {name} | {x['softF1@1']:.5f} | {x['softF1@3']:.5f} | {x['softF1@10']:.5f} |")
(OUT/"E080_N3_REPORT.md").write_text("\n".join(md)+"\n")
print(json.dumps(summary,indent=2))
