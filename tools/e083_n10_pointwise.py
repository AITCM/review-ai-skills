from __future__ import annotations
"""E083 / N10: diagnosis-masked reasoning-point candidate route.
Ranking and evaluation are intentionally separated. No existing test/MedR gold read.
"""
import argparse, hashlib, json, pathlib, re, unicodedata
import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
SOURCE=BASE+"/train-00000-of-00001.parquet"
OUT=pathlib.Path("e083-n10-output");OUT.mkdir(exist_ok=True)
SEED=20261009;N_META_GROUPS=480;TOPK=(1,3,10);RRF_CONST=60
GENERIC={"with","without","acute","chronic","syndrome","disease","disorder"}
PAT=re.compile(r'^\s*(?:\((\d+)\)|(\d+)[.)])\s*',re.M)
CONFIG=pathlib.Path("configs/experiments/E083_N10_v1.json")
FIELDS=["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]
METHODS=("LEX","XMOD","N8_RRF_LEX_XMOD","POINT_ONLY","RRF_XMOD_POINT","N10_RRF_LEX_XMOD_POINT")
PRIMARY=("N10_RRF_LEX_XMOD_POINT","N8_RRF_LEX_XMOD")
def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def nw(s):return " ".join(re.findall(r"\w+",unicodedata.normalize("NFKC",str(s)).casefold()))
def points(s):
    s=str(s);ms=list(PAT.finditer(s))
    if not ms or s[:ms[0].start()].strip():return [s.strip()]
    nums=[int(m.group(1) or m.group(2)) for m in ms]
    if nums!=list(range(1,len(ms)+1)):return [s.strip()]
    out=[]
    for k,m in enumerate(ms):
        end=ms[k+1].start() if k+1<len(ms) else len(s)
        v=s[m.end():end].strip()
        if v:out.append(v)
    return out or [s.strip()]
def mask_hist_points(s,diagnosis):
    d=nw(diagnosis)
    tokens={t for t in d.split() if len(t)>=4 and t not in GENERIC}
    out=[]
    for p in points(s):
        p=nw(p)
        if d and d in p:p=p.replace(d," diagnosismask ")
        for t in sorted(tokens,key=lambda x:(-len(x),x)):
            p=re.sub(r"\b"+re.escape(t)+r"\b"," diagnosistokenmask ",p)
        out.append(" ".join(p.split()))
    return out or ["empty"]
def norm_full(scores):
    ids=np.arange(len(scores),dtype=np.int32)
    return np.lexsort((ids,-np.asarray(scores)))
def rrf(*rankings):
    n=len(rankings[0]);values=np.zeros(n,np.float64)
    for order in rankings:
        r=np.empty(n,np.int32);r[np.asarray(order)]=np.arange(n,dtype=np.int32)
        values+=1.0/(RRF_CONST+r+1.0)
    return norm_full(values)
def split_ids(all_ids):
    sorted_ids=sorted(set(all_ids),key=lambda s:sha(("E083:N10:"+s).encode()))
    return set(sorted_ids[:N_META_GROUPS])
def get_meta_ref_info():
    meta=pd.read_parquet(SOURCE,columns=["pmcid","case_prompt"])
    meta.pmcid=meta.pmcid.astype(str)
    train_meta=split_ids(meta.pmcid)
    isquery=meta.pmcid.isin(train_meta)
    qm=meta.loc[isquery].reset_index(drop=True)
    rm=meta.loc[~isquery].reset_index(drop=True)
    assert len(meta)==13092 and len(qm)>0 and len(rm)>0
    assert set(qm.pmcid).isdisjoint(set(rm.pmcid))
    return rm,qm,train_meta

def rank_phase():
    rm,qm,held_ids=get_meta_ref_info()
    # Rank phase reads gold only for historical reference group, not meta-query cases.
    # PyArrow predicate excludes held-out PMCID groups BEFORE outcomes are materialized.
    ref=pd.read_parquet(SOURCE,columns=FIELDS,filters=[("pmcid","not in",sorted(held_ids))])
    ref.pmcid=ref.pmcid.astype(str)
    ref=ref.reset_index(drop=True)
    assert len(ref)==len(rm) and ref.pmcid.tolist()==rm.pmcid.tolist()
    assert set(ref.pmcid).isdisjoint(set(qm.pmcid))
    # New-query object holds only PMCID and case_prompt.
    assert set(qm.columns)=={"pmcid","case_prompt"}

    cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,
       min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
    CX=cv.fit_transform(ref.case_prompt.astype(str))
    CQ=cv.transform(qm.case_prompt.astype(str))
    reason=[mask_hist_points(r,d) for r,d in zip(ref.diagnostic_reasoning,ref.final_diagnosis)]
    docs=[" ".join(arr) for arr in reason]
    xv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,
       min_df=2,max_df=.98,max_features=140000,dtype=np.float32)
    xv.fit(ref.case_prompt.astype(str).tolist()+docs)
    Q=xv.transform(qm.case_prompt.astype(str))
    DOC=xv.transform(docs)
    flat=[p for arr in reason for p in arr]
    TP=xv.transform(flat).tocsr()
    starts=np.concatenate(([0],np.cumsum([len(z) for z in reason])[:-1])).astype(np.int64)
    assert len(starts)==len(ref) and TP.shape[0]>=len(ref)
    config=json.loads(CONFIG.read_text(encoding="utf-8"))
    records=[]
    for i,row in qm.iterrows():
        lscore=(CQ.getrow(i)@CX.T).toarray().ravel()
        dscore=(Q.getrow(i)@DOC.T).toarray().ravel()
        point_scores=(Q.getrow(i)@TP.T).toarray().ravel()
        pscore=np.maximum.reduceat(point_scores,starts)
        a=norm_full(lscore);b=norm_full(dscore);c=norm_full(pscore)
        rankings={
           "LEX":a,"XMOD":b,
           "N8_RRF_LEX_XMOD":rrf(a,b),
           "POINT_ONLY":c,
           "RRF_XMOD_POINT":rrf(b,c),
           "N10_RRF_LEX_XMOD_POINT":rrf(a,b,c)
        }
        rec={"query_index":int(i),"query_pmcid":str(row.pmcid),
             "candidate_budget":50,
             "rankings":{k:v[:50].astype(int).tolist() for k,v in rankings.items()}}
        records.append(rec)
        if (i+1)%100==0:print("BLIND RANKINGS",i+1,"/",len(qm),flush=True)
    f=OUT/"E083_N10_BLIND_RANKINGS.jsonl"
    f.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records))
    lock={
      "status":"RANKINGS_LOCKED_BEFORE_META_QUERY_GOLD",
      "experiment_id":config["experiment_id"],
      "code_revision_expected":"git workflow run SHA",
      "dataset_revision":REV,
      "query_split":"train_internal_meta",
      "n_total":13092,"n_ref":len(ref),"n_query":len(qm),
      "n_query_pmcid_groups":len(held_ids),
      "query_fields_accessed":["pmcid","case_prompt"],
      "reference_fields_accessed":FIELDS,
      "query_pmcid_sha256":sha(json.dumps(sorted(held_ids)).encode()),
      "reference_pmcid_sha256":sha(json.dumps(ref.pmcid.tolist()).encode()),
      "query_pmcid_rowsha256":sha(json.dumps(qm.pmcid.tolist()).encode()),
      "config_sha256":sha(CONFIG.read_bytes()),
      "blind_ranking_sha256":sha(f.read_bytes()),
      "primary_variant":"N10_RRF_LEX_XMOD_POINT",
      "baseline_variant":"N8_RRF_LEX_XMOD",
      "n_queries_with_gold_loaded_in_rank_stage":0,
      "guarantee":"meta query gold excluded from ref outcome scan by PMCID predicate; test/MedR never accessed"
    }
    (OUT/"E083_N10_RANKING_LOCK.json").write_text(json.dumps(lock,indent=2)+"\n")
    print("LOCKED",json.dumps({k:lock[k] for k in ("status","n_ref","n_query","n_query_pmcid_groups","blind_ranking_sha256")},indent=2))

def softf1(q,C,ids):
    from scipy.sparse import vstack
    A=vstack([C[j] for j in ids],format="csr")
    S=(q@A.T).toarray()
    if not S.size:return 0.0
    rec=float(S.max(axis=1).mean());prec=float(S.max(axis=0).mean())
    return 2*rec*prec/(rec+prec) if (rec+prec)>0 else 0.0

def paired_cluster_ci(d,groups,seed):
    # Resample PMCID groups, not individual rows.
    groups=np.asarray(groups)
    unique=sorted(set(groups));idx={g:np.where(groups==g)[0] for g in unique}
    means=np.asarray([np.mean(d[idx[g]]) for g in unique],dtype=float)
    rng=np.random.default_rng(seed);boot=np.empty(3000,float)
    for b in range(3000):boot[b]=np.mean(means[rng.integers(0,len(means),len(means))])
    return [float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]
def evaluator_matrix(qpts,refpts,params):
    v=TfidfVectorizer(**params)
    flat=[p for arr in refpts for p in arr]
    T=v.fit_transform(flat)
    V=v.transform([p for arr in qpts for p in arr])
    starts=np.concatenate(([0],np.cumsum([len(z) for z in refpts])[:-1])).astype(np.int64)
    counts=np.asarray([len(z) for z in refpts],dtype=np.int32)
    qstarts=np.concatenate(([0],np.cumsum([len(z) for z in qpts])[:-1])).astype(np.int64)
    qcounts=np.asarray([len(z) for z in qpts],dtype=np.int32)
    return T,V,starts,counts,qstarts,qcounts
def f1_of_order(S,starts,counts,arr):
    # S is query-point x historical-point matrix; use all points of the selected cases.
    rr=[]
    for j in arr:
        rr.extend(range(int(starts[j]),int(starts[j]+counts[j])))
    sl=S[:,rr]
    if not sl.size:return 0.
    r=float(sl.max(axis=1).mean());p=float(sl.max(axis=0).mean())
    return 2*r*p/(r+p) if r+p else 0.

def eval_phase():
    lock=json.loads((OUT/"E083_N10_RANKING_LOCK.json").read_text())
    rf=OUT/"E083_N10_BLIND_RANKINGS.jsonl"
    assert lock["status"]=="RANKINGS_LOCKED_BEFORE_META_QUERY_GOLD"
    assert sha(rf.read_bytes())==lock["blind_ranking_sha256"]
    assert sha(CONFIG.read_bytes())==lock["config_sha256"]
    ranks=[json.loads(x) for x in rf.read_text().splitlines()]
    rm,qm,held_ids=get_meta_ref_info()
    assert len(ranks)==len(qm)==lock["n_query"]
    assert sha(json.dumps(qm.pmcid.tolist()).encode())==lock["query_pmcid_rowsha256"]
    assert all(z["query_index"]==i and z["query_pmcid"]==qm.pmcid.iloc[i] for i,z in enumerate(ranks))
    # GOLD ACCESS STARTS HERE, AFTER the read-only ranking SHA256 has been verified.
    outcomes=pd.read_parquet(SOURCE,columns=FIELDS)
    outcomes.pmcid=outcomes.pmcid.astype(str)
    ref=outcomes.loc[~outcomes.pmcid.isin(held_ids)].reset_index(drop=True)
    query=outcomes.loc[outcomes.pmcid.isin(held_ids)].reset_index(drop=True)
    assert ref.pmcid.tolist()==rm.pmcid.tolist()
    assert query.pmcid.tolist()==qm.pmcid.tolist()
    refpoints=[mask_hist_points(r,d) for r,d in zip(ref.diagnostic_reasoning,ref.final_diagnosis)]
    qpoints=[mask_hist_points(r,d) for r,d in zip(query.diagnostic_reasoning,query.final_diagnosis)]
    specs={
      "primary_sparse":dict(ngram_range=(1,2),lowercase=True,sublinear_tf=True,
                            min_df=2,max_df=.98,max_features=100000,dtype=np.float32),
      "independent_char":dict(analyzer="char_wb",ngram_range=(3,5),lowercase=True,
                            sublinear_tf=True,min_df=2,max_df=.995,max_features=120000,dtype=np.float32)
    }
    allstats={}
    records=[{"query_index":i,"query_pmcid":query.pmcid.iloc[i]} for i in range(len(query))]
    for metric,kwargs in specs.items():
        print("EVALUATOR",metric,flush=True)
        T,V,starts,counts,qstarts,qcounts=evaluator_matrix(qpoints,refpoints,kwargs)
        TT=T.T.tocsr()
        for i,r in enumerate(ranks):
            qmat=V[qstarts[i]:qstarts[i]+qcounts[i]]
            S=(qmat@TT).toarray().astype(np.float32,copy=False)
            for name in METHODS:
                order=r["rankings"][name]
                for k in TOPK:
                    records[i][f"{metric}_{name}_{k}"]=f1_of_order(S,starts,counts,order[:k])
            if (i+1)%100==0:print("EVALUATED",metric,i+1,"/",len(ranks),flush=True)
        summary={}
        for name in METHODS:
            summary[name]={}
            for k in TOPK:
                arr=np.asarray([x[f"{metric}_{name}_{k}"] for x in records],dtype=float)
                baseline=np.asarray([x[f"{metric}_N8_RRF_LEX_XMOD_{k}"] for x in records],dtype=float)
                delta=arr-baseline
                summary[name][str(k)]={
                    "mean":float(arr.mean()),
                    "delta_vs_N8":float(delta.mean()),
                    "cluster_ci95_delta_vs_N8":paired_cluster_ci(delta,query.pmcid.to_numpy(),SEED+k+len(name)+len(metric)),
                    "improved":int((delta>1e-12).sum()),
                    "worsened":int((delta<-1e-12).sum()),
                    "tied":int((np.abs(delta)<=1e-12).sum())
                }
        allstats[metric]=summary
    rows=pd.DataFrame(records)
    rows.to_csv(OUT/"E083_N10_PER_QUERY.csv",index=False)
    dx_ref=set(map(nw,ref.final_diagnosis))
    explicitly_exposed=int(sum(nw(d) in nw(p) for d,p in zip(query.final_diagnosis,query.case_prompt) if nw(d)))
    result={
        "experiment_id":"E083_N10_v1",
        "status":"train_internal_exploratory_outcome_opened_only_after_rank_lock",
        "rank_lock":lock,"n_query":len(query),
        "result":allstats,"query_diagnosis_phrase_exposure_n":explicitly_exposed,
        "independent_evaluator_lexically_related":True,
        "selection_warning":"This split and previously explored 500-case validation are development data only. Do not call any of these confirmation.",
        "primary_comparison":"N10_RRF_LEX_XMOD_POINT vs frozen-style N8_RRF_LEX_XMOD, Soft-F1@10"
    }
    (OUT/"E083_N10_SUMMARY.json").write_text(json.dumps(result,indent=2)+"\n")
    lines=["# E083 N10 — diagnosis-masked reasoning-point route",
       "","**Train-internal blind lock, exploratory. No original test or MedR outcome used.**",
       f"Evaluation meta queries: {len(query)}; PMCID groups: {len(set(query.pmcid))}.",
       "","| method | sparse@1 | sparse@3 | sparse@10 | Δ@10 vs N8 | char@10 | char Δ@10 |",
       "|---|---:|---:|---:|---:|---:|---:|"]
    for name in METHODS:
        s=allstats["primary_sparse"][name];c=allstats["independent_char"][name]
        lines.append(f"| {name} | {s['1']['mean']:.5f} | {s['3']['mean']:.5f} | {s['10']['mean']:.5f} | {s['10']['delta_vs_N8']:+.5f} | {c['10']['mean']:.5f} | {c['10']['delta_vs_N8']:+.5f} |")
    lines += ["","### Interpretation",
       "The point route is a zero-shot exact-TFIDF max-over-historical-reasoning-points route. It adds no learned query gold. All features of new queries come from case_prompt only. The primary comparison and RRF constant were fixed before the run.",
       "Char-ngram evaluation is independent of the retrieval tokenization, but both are lexical; full clinical validation requires independent clinicians and/or a genuinely new cohort.",
       "Do not tune on T001, on the historical 500-case validation, or the previously exposed E081 MedR set.",
       f"Literal diagnosis exposure in meta-query case_prompt: {explicitly_exposed}."]
    (OUT/"E083_N10_REPORT.md").write_text("\n".join(lines)+"\n")
    print("PRIMARY",json.dumps(allstats["primary_sparse"]["N10_RRF_LEX_XMOD_POINT"]["10"],indent=2),flush=True)
    print("CHAR",json.dumps(allstats["independent_char"]["N10_RRF_LEX_XMOD_POINT"]["10"],indent=2),flush=True)

if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--stage",required=True,choices=["rank","eval"])
    args=cli.parse_args()
    if args.stage=="rank":rank_phase()
    else:eval_phase()
