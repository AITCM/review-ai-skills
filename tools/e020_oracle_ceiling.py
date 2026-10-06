from __future__ import annotations
import json,re,unicodedata,pathlib,zipfile,hashlib,os
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("oracle-output");OUT.mkdir(exist_ok=True)
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

def pts(r,d):
    dd=nw(d);o=[]
    for x in split_reason(r):
        xx=nw(x)
        if dd and dd in xx: xx=xx.replace(dd," diagnosismask ")
        o.append(xx)
    return o

def f1(rec,prec):
    den=rec+prec
    return np.where(den>0,2*rec*prec/den,0.0)

def set_u(qmax,psum,counts,ids):
    ids=np.asarray(ids,dtype=int)
    rec=float(np.max(qmax[:,ids],axis=1).mean())
    prec=float(psum[ids].sum()/counts[ids].sum())
    return float(2*rec*prec/(rec+prec)) if rec+prec else 0.0

def greedy(qmax,psum,counts,pool,kmax=10):
    pool=np.asarray(pool,dtype=int)
    chosen=[]
    recvec=np.zeros(qmax.shape[0],dtype=np.float32)
    ps=0.0;pc=0
    scores={}
    alive=np.ones(len(pool),dtype=bool)
    for step in range(1,kmax+1):
        ids=pool
        recs=np.maximum(recvec[:,None],qmax[:,ids]).mean(axis=0)
        precs=(ps+psum[ids])/(pc+counts[ids])
        vals=f1(recs,precs)
        vals[~alive]=-np.inf
        # deterministic tie break: lower train index
        order=np.lexsort((ids,-vals))
        pos=int(order[0]);j=int(ids[pos])
        chosen.append(j);alive[pos]=False
        recvec=np.maximum(recvec,qmax[:,j])
        ps+=float(psum[j]);pc+=int(counts[j])
        if step in KS: scores[step]=float(vals[pos])
    return chosen,scores

def boot_ci(x,seed=20261006,n=5000):
    x=np.asarray(x,float);rng=np.random.default_rng(seed)
    vals=[]
    for _ in range(n):
        ids=rng.integers(0,len(x),len(x));vals.append(float(x[ids].mean()))
    return [float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))]

artifact=pathlib.Path("t001_artifact.zip")
if not artifact.exists(): raise FileNotFoundError("t001_artifact.zip")
with zipfile.ZipFile(artifact) as z:
    z.extractall("t001-artifact")
rankfile=pathlib.Path("t001-artifact/t001-output/T001_BLIND_RANKINGS.jsonl")
resultfile=pathlib.Path("t001-artifact/t001-output/T001_TEST_RESULTS.json")
lockfile=pathlib.Path("t001-artifact/t001-output/T001_RANKING_LOCK.json")
lock=json.load(open(lockfile))
assert lock["status"]=="rankings_locked_before_test_outcomes_loaded"
assert lock["method"]=="C2-debiased-final-two-feature"
R=[json.loads(x) for x in open(rankfile)]
assert len(R)==897 and [x["query_index"] for x in R]==list(range(897))
published=json.load(open(resultfile))

tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet",columns=["case_prompt","diagnostic_reasoning","final_diagnosis"])
te=pd.read_parquet(BASE+"/test-00000-of-00001.parquet",columns=["case_prompt","diagnostic_reasoning","final_diagnosis"])
assert len(tr)==13092 and len(te)==897

# Reproduce the exact T001 reasoning metric representation.
tp=[pts(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
vp=[pts(r,d) for r,d in zip(te.diagnostic_reasoning,te.final_diagnosis)]
flat_t=[p for z in tp for p in z]
flat_v=[p for z in vp for p in z]
vec=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
T=vec.fit_transform(flat_t)
V=vec.transform(flat_v)
counts=np.asarray([len(z) for z in tp],dtype=np.int32)
starts=np.concatenate(([0],np.cumsum(counts)[:-1])).astype(np.int64)
vo=[];o=0
for z in vp:
    vo.append((o,o+len(z)));o+=len(z)

# Candidate positions are taken ONLY from the immutable T001 ranking lock.
# We intentionally do not reconstruct full lexical ranks post hoc.
all_ids=np.arange(len(tr),dtype=int)
rank_mismatch=None

per=[]
Tt=T.T.tocsr()
# Batch whole queries while capping total reasoning points per sparse multiply.
batches=[];start_q=0
while start_q<len(te):
    end_q=start_q;pts_n=0
    while end_q<len(te):
        n=vo[end_q][1]-vo[end_q][0]
        if end_q>start_q and pts_n+n>192: break
        pts_n+=n;end_q+=1
    batches.append((start_q,end_q));start_q=end_q

for bno,(qa,qb) in enumerate(batches,1):
    ps0=vo[qa][0];ps1=vo[qb-1][1]
    SB=(V[ps0:ps1]@Tt).toarray().astype(np.float32,copy=False)
    for i in range(qa,qb):
        s,e=vo[i];S=SB[s-ps0:e-ps0]
        # qmax[:,j] = best match for each target reasoning point within train case j
        qmax=np.stack([np.maximum.reduceat(row,starts) for row in S],axis=0)
        pmax=S.max(axis=0)
        psum=np.add.reduceat(pmax,starts)
        rec=qmax.mean(axis=0)
        prec=psum/counts
        util=f1(rec,prec)

        base=np.asarray(R[i]["baseline_top50"],dtype=int)
        meth=np.asarray(R[i]["method_top50"],dtype=int)
        top50_oracle_order=base[np.lexsort((base,-util[base]))]
        global_order=np.lexsort((all_ids,-util))
        gbest=int(global_order[0]);tbest=int(top50_oracle_order[0])

        # true lexical rank of globally best single-case utility candidate
        cs=case_scores[i];v=float(cs[gbest])
        lexrank=1+int(np.sum(cs>v))+int(np.sum((cs==v)&(all_ids<gbest)))

        row={"query_index":i,"global_best_single_train_index":gbest,
             "global_best_single_utility":float(util[gbest]),
             "global_best_single_locked_tfidf_position":locked_pos,
             "top50_best_single_train_index":tbest,
             "top50_best_single_utility":float(util[tbest])}
        _,g50=greedy(qmax,psum,counts,base,10)
        _,gg=greedy(qmax,psum,counts,all_ids,10)
        for k in KS:
            row[f"baseline_{k}"]=set_u(qmax,psum,counts,base[:k])
            row[f"c2_{k}"]=set_u(qmax,psum,counts,meth[:k])
            row[f"top50_individual_oracle_{k}"]=set_u(qmax,psum,counts,top50_oracle_order[:k])
            row[f"top50_greedy_oracle_{k}"]=g50[k]
            row[f"global_greedy_oracle_{k}"]=gg[k]
        per.append(row)
    print(f"BATCH {bno}/{len(batches)} queries {qa}:{qb}",flush=True)

df=pd.DataFrame(per).sort_values("query_index")
df.to_csv(OUT/"E020_ORACLE_CEILING_PER_QUERY.csv",index=False)

summary={
 "experiment":"E020-O1 Oracle/Candidate-Ceiling Audit",
 "status":"completed_post_test_analysis_only",
 "policy":"Outcome-informed oracle analysis only. Must not modify or select the frozen T001 method.",
 "dataset_revision":REV,
 "n_test":len(df),
 "t001_artifact_sha256":hashlib.sha256(artifact.read_bytes()).hexdigest(),
 "ranking_lock":lock,
 "tfidf_rank_reconstruction":"not_attempted_after_initial mismatch; immutable T001 Top-50 lock used as sole candidate-position authority",
 "metric":"T001 diagnosis-masked symmetric reasoning-set TF-IDF softF1",
 "oracle_definitions":{
   "single_global_oracle":"Exact best single historical case over all 13,092 training cases using test reasoning outcomes.",
   "individual_oracle_top50":"Sort the frozen TF-IDF Top-50 by each candidate's individual outcome-defined utility.",
   "greedy_oracle_top50":"Outcome-informed greedy set selection from the frozen TF-IDF Top-50, maximizing set softF1 at each step.",
   "greedy_oracle_global":"Outcome-informed greedy set selection from all training cases; achievable post-hoc ceiling, not a proof of combinatorial global optimum for k>1."
 },
 "results":{},
 "candidate_generation":{}
}

for k in KS:
    b=df[f"baseline_{k}"].to_numpy();m=df[f"c2_{k}"].to_numpy()
    o50=df[f"top50_greedy_oracle_{k}"].to_numpy();og=df[f"global_greedy_oracle_{k}"].to_numpy()
    gain=float((m-b).mean());within=float((o50-b).mean());globalgap=float((og-b).mean())
    summary["results"][str(k)]={
      "baseline_mean":float(b.mean()),"c2_mean":float(m.mean()),"c2_delta":gain,
      "c2_delta_ci95":boot_ci(m-b,20261006+k),
      "top50_greedy_oracle_mean":float(o50.mean()),
      "top50_recoverable_gain_over_baseline":within,
      "fraction_of_top50_recoverable_gain_captured_by_c2":float(gain/within) if within>0 else None,
      "global_greedy_oracle_mean":float(og.mean()),
      "global_recoverable_gain_over_baseline":globalgap,
      "fraction_of_global_greedy_gain_captured_by_c2":float(gain/globalgap) if globalgap>0 else None,
      "fraction_of_global_greedy_gain_accessible_within_tfidf_top50":float(within/globalgap) if globalgap>0 else None
    }

pos=df["global_best_single_locked_tfidf_position"]
inside=pos.notna().to_numpy()
contained=pos.dropna().to_numpy(dtype=float)
summary["candidate_generation"]={
 "global_best_single_mean_utility":float(df.global_best_single_utility.mean()),
 "top50_best_single_mean_utility":float(df.top50_best_single_utility.mean()),
 "global_best_single_in_locked_top1_rate":float(np.mean(pos.fillna(999999).to_numpy()<=1)),
 "global_best_single_in_locked_top10_rate":float(np.mean(pos.fillna(999999).to_numpy()<=10)),
 "global_best_single_in_locked_top50_rate":float(np.mean(inside)),
 "conditional_median_locked_position_if_in_top50":float(np.median(contained)) if len(contained) else None,
 "n_global_best_single_missing_from_locked_top50":int((~inside).sum())
}

# Hard validation: same frozen rankings + same primary evaluator must reproduce T001 means.
for k in KS:
    pk=published["results"][str(k)]
    if abs(summary["results"][str(k)]["baseline_mean"]-pk["baseline"])>2e-6:
        raise RuntimeError(f"baseline reproduction failed at k={k}")
    if abs(summary["results"][str(k)]["c2_mean"]-pk["method"])>2e-6:
        raise RuntimeError(f"C2 reproduction failed at k={k}")
summary["t001_metric_reproduction_passed"]=True

(OUT/"E020_ORACLE_CEILING_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
md=["# E020-O1 Oracle / Candidate-Ceiling Audit","",
"**Status:** post-test diagnostic analysis only; the frozen T001 method is unchanged.","",
"## Why this analysis","This audit asks how much reasoning utility was recoverable by reranking the frozen TF-IDF Top-50, and how much potential utility was excluded by TF-IDF candidate generation.","",
"## Main results","",
"| k | TF-IDF | Frozen C2 | Top-50 greedy oracle | Global greedy oracle | C2 / Top-50 recoverable | C2 / global recoverable |",
"|---:|---:|---:|---:|---:|---:|---:|"]
for k in KS:
    x=summary["results"][str(k)]
    md.append(f"| {k} | {x['baseline_mean']:.5f} | {x['c2_mean']:.5f} | {x['top50_greedy_oracle_mean']:.5f} | {x['global_greedy_oracle_mean']:.5f} | {100*x['fraction_of_top50_recoverable_gain_captured_by_c2']:.1f}% | {100*x['fraction_of_global_greedy_gain_captured_by_c2']:.1f}% |")
c=summary["candidate_generation"]
md += ["","## Candidate-generation ceiling",
f"- Global best single case present in the immutable TF-IDF Top-50: **{100*c['global_best_single_in_locked_top50_rate']:.1f}%**.",
f"- Present in locked Top-10 / Top-1: **{100*c['global_best_single_in_locked_top10_rate']:.1f}% / {100*c['global_best_single_in_locked_top1_rate']:.1f}%**.",
f"- Global-best single cases missing entirely from the locked Top-50: **{c['n_global_best_single_missing_from_locked_top50']} / {len(df)}**.",
"",
"## Interpretation rule",
"Top-1 global oracle is exact. Top-3/Top-10 greedy oracles are outcome-informed achievable ceilings, not proofs of the combinatorial global optimum. These numbers are for diagnosis of the retrieval bottleneck and must not be used to retune T001."
]
(OUT/"E020_ORACLE_CEILING_REPORT.md").write_text("\n".join(md)+"\n")
print(json.dumps(summary,indent=2))
