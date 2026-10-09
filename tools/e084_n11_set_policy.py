from __future__ import annotations
"""E084/N11. Train-only set-aware reranking over the frozen-style N8 top-100.
Train-reference + teacher-meta is used to learn marginal utility. Separate PMCID-
disjoint outer-meta scores remain unread until blinded rankings are SHA-locked.
"""
import argparse,hashlib,json,pathlib
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingRegressor
from e083_n10_pointwise import (REV,SOURCE,nw,mask_hist_points,sha,points,
   norm_full,rrf,paired_cluster_ci,evaluator_matrix,f1_of_order)
OUT=pathlib.Path("e084-n11-output");OUT.mkdir(exist_ok=True)
CONFIG=pathlib.Path("configs/experiments/E084_N11_v1.json")
SEED=20261009;N_OUTER_GROUPS=360;N_TEACHER_GROUPS=600;POOL=100
TOPKS=(1,3,10)
FEATURES=("lex","xmod","rrf","lex_rr","xmod_rr","pool_rr",
          "step_frac","max_case_redundancy","mean_case_redundancy",
          "max_reason_redundancy","mean_reason_redundancy",
          "reason_novelty","xmod_x_novelty","lex_x_novelty")
def ref_query_meta():
    all_meta=pd.read_parquet(SOURCE,columns=["pmcid","case_prompt"])
    all_meta.pmcid=all_meta.pmcid.astype(str)
    all_ids=set(all_meta.pmcid)
    old_n10=sorted(all_ids,key=lambda x:sha(("E083:N10:"+x).encode()))[:480]
    old_n10=set(old_n10)
    # New outer groups are disjoint from the entire previous N10 outer holdout.
    fresh=sorted(all_ids-old_n10,key=lambda x:sha(("E084:N11:OUTER:"+x).encode()))
    outer=set(fresh[:N_OUTER_GROUPS])
    hold=all_meta.loc[all_meta.pmcid.isin(outer)].reset_index(drop=True)
    hist=all_meta.loc[~all_meta.pmcid.isin(outer)].reset_index(drop=True)
    assert len(all_meta)==13092 and len(set(hold.pmcid))==N_OUTER_GROUPS
    assert not (outer & old_n10)
    assert set(hold.pmcid).isdisjoint(set(hist.pmcid))
    assert set(hold.columns)=={"pmcid","case_prompt"}
    return hist,hold,outer,old_n10

def read_historical(hist_metadata,outer):
    history=pd.read_parquet(SOURCE,
        columns=["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"],
        filters=[("pmcid","not in",sorted(outer))])
    history.pmcid=history.pmcid.astype(str)
    history=history.reset_index(drop=True)
    assert history.pmcid.tolist()==hist_metadata.pmcid.tolist()
    return history

def n8_scores(reference,queries):
    cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,
         min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
    X=cv.fit_transform(reference.case_prompt.astype(str))
    Q=cv.transform(queries.case_prompt.astype(str))
    masked=[mask_hist_points(r,d) for r,d in zip(reference.diagnostic_reasoning,reference.final_diagnosis)]
    xv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,
         min_df=2,max_df=.98,max_features=140000,dtype=np.float32)
    docs=[" ".join(z) for z in masked]
    xv.fit(reference.case_prompt.astype(str).tolist()+docs)
    R=xv.transform(docs);QR=xv.transform(queries.case_prompt.astype(str))
    lex=(Q@X.T).toarray().astype(np.float32)
    xmod=(QR@R.T).toarray().astype(np.float32)
    return X,R,masked,lex,xmod

def feature_matrix(locations,lex,xmod,ranks_lex,ranks_xmod,
                   case_sims,reason_sims,selected):
    n=len(locations)
    rr=1/(60.+np.arange(n,dtype=np.float32)+1.)
    poolscore=(1/(60+ranks_lex+1.)+1/(60+ranks_xmod+1.)).astype(np.float32)
    if selected:
        case_mat=case_sims[:,selected];reason_mat=reason_sims[:,selected]
        case_max=case_mat.max(axis=1);case_mean=case_mat.mean(axis=1)
        reason_max=reason_mat.max(axis=1);reason_mean=reason_mat.mean(axis=1)
    else:
        case_max=case_mean=reason_max=reason_mean=np.zeros(n,np.float32)
    novelty=(1.-reason_max).astype(np.float32)
    step=np.full(n,len(selected)/10.,np.float32)
    return np.column_stack([lex,xmod,poolscore,
      1./(ranks_lex+1.),1./(ranks_xmod+1.),rr,
      step,case_max,case_mean,reason_max,reason_mean,novelty,
      xmod*novelty,lex*novelty]).astype(np.float32)

def ranks_and_features(lex,xmod,X,R):
    a=norm_full(lex);b=norm_full(xmod);c=rrf(a,b)
    n=len(lex)
    pa=np.empty(n,dtype=np.int32);pb=np.empty(n,dtype=np.int32)
    pa[a]=np.arange(n);pb[b]=np.arange(n)
    pool=c[:POOL].astype(int)
    local_case=(X[pool]@X[pool].T).toarray().astype(np.float32)
    local_reason=(R[pool]@R[pool].T).toarray().astype(np.float32)
    return pool,lex[pool],xmod[pool],pa[pool],pb[pool],local_case,local_reason

def u_and_teacher(qmat,T,offs,counts,pool):
    S=(qmat@T.T).toarray().astype(np.float32,copy=False)
    qmax=np.stack([np.maximum.reduceat(row,offs) for row in S],axis=0)[:,pool]
    pmax=S.max(axis=0)
    psum=np.add.reduceat(pmax,offs)[pool]
    cc=counts[pool]
    selected=[];teacher=[];records=[]
    for step in range(10):
        if selected:
            recmax=qmax[:,selected].max(axis=1)
            old_prec=float(psum[selected].sum()/cc[selected].sum())
            old_rec=float(recmax.mean())
            old=(2*old_prec*old_rec/(old_prec+old_rec)) if old_prec+old_rec else 0.
        else:
            recmax=np.zeros(qmax.shape[0],np.float32);old=0.
        rec=np.maximum(recmax[:,None],qmax).mean(axis=0)
        if selected:
            prec=(psum+psum[selected].sum())/(cc+cc[selected].sum())
        else:prec=psum/cc
        vals=np.where(rec+prec>0,2*rec*prec/(rec+prec),0.)
        gain=(vals-old).astype(np.float32)
        if selected:gain[selected]=-np.inf
        nextj=int(np.lexsort((pool,-gain))[0])
        records.append(gain.copy())
        selected.append(nextj)
        teacher.append(nextj)
    return records,teacher

def learn_model(ref_all):
    # Inner teacher queries are from training data only; outer holdout is excluded.
    old_set=set(ref_all.pmcid)
    teacher_ids=sorted(old_set,key=lambda p:sha(("E084:N11:TEACHER:"+p).encode()))[:N_TEACHER_GROUPS]
    teacher_ids=set(teacher_ids)
    reference=ref_all.loc[~ref_all.pmcid.isin(teacher_ids)].reset_index(drop=True)
    mq=ref_all.loc[ref_all.pmcid.isin(teacher_ids)].reset_index(drop=True)
    assert len(set(mq.pmcid))==N_TEACHER_GROUPS
    assert set(reference.pmcid).isdisjoint(set(mq.pmcid))
    X,R,refpoints,L,DX=n8_scores(reference,mq)
    points_ref=[p for x in refpoints for p in x]
    ev=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,
        min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
    T=ev.fit_transform(points_ref).tocsr()
    qpoints=[mask_hist_points(r,d) for r,d in zip(mq.diagnostic_reasoning,mq.final_diagnosis)]
    V=ev.transform([p for x in qpoints for p in x]).tocsr()
    counts=np.asarray([len(p) for p in refpoints],dtype=int)
    offs=np.concatenate(([0],np.cumsum(counts)[:-1])).astype(int)
    vo=[];j=0
    for p in qpoints:vo.append((j,j+len(p)));j+=len(p)
    rng=np.random.default_rng(SEED)
    feats=[];labels=[]
    for i in range(len(mq)):
        pool,lex,xmod,pl,px,C,D=ranks_and_features(L[i],DX[i],X,R)
        a,b=vo[i]
        gains,teacher=u_and_teacher(V[a:b],T,offs,counts,pool)
        for step,utility in enumerate(gains):
            selected=teacher[:step]
            F=feature_matrix(pool,lex,xmod,pl,px,C,D,selected)
            valid=np.ones(len(pool),dtype=bool)
            if selected:valid[selected]=False
            ids=np.where(valid)[0]
            # teacher optimum, difficult negatives, and random middle ranks
            scores=utility[ids];order=ids[np.argsort(-scores,kind="stable")]
            ktop=order[:5];kbottom=order[-5:]
            middle=order[5:-5]
            sample=rng.choice(middle,size=min(10,len(middle)),replace=False) if len(middle) else np.asarray([],int)
            chosen=np.unique(np.concatenate((ktop,kbottom,sample)))
            feats.append(F[chosen]);labels.append(utility[chosen])
        if (i+1)%100==0:print("TRAIN-TEACHER",i+1,"/",len(mq),flush=True)
    xx=np.vstack(feats);yy=np.concatenate(labels)
    params=dict(loss="squared_error",learning_rate=.05,max_iter=180,
        max_leaf_nodes=15,min_samples_leaf=60,l2_regularization=2.,
        early_stopping=False,random_state=SEED)
    model=HistGradientBoostingRegressor(**params).fit(xx,yy)
    return model,{"n_inner_ref":len(reference),"n_teacher_queries":len(mq),
       "n_teacher_pmcid_groups":len(set(mq.pmcid)),"n_teacher_rows":len(yy),
       "params":params,"features":list(FEATURES)}

def rank_phase():
    rm,qm,outer,old=get_meta_ref_info_or_die()
    ref=read_historical(rm,outer)
    model,training=learn_model(ref)
    X,R,refpoints,L,DX=n8_scores(ref,qm)
    ranks=[]
    for i in range(len(qm)):
        pool,lex,xmod,pl,px,C,D=ranks_and_features(L[i],DX[i],X,R)
        chosen=[]
        for _ in range(10):
            F=feature_matrix(pool,lex,xmod,pl,px,C,D,chosen)
            score=model.predict(F).astype(np.float64)
            if chosen:score[chosen]=-np.inf
            pick=int(np.lexsort((pool,-score))[0]);chosen.append(pick)
        row={"query_index":i,"query_pmcid":str(qm.pmcid.iloc[i]),
             "n8_top50":pool[:50].tolist(),
             "n11_top10":pool[np.asarray(chosen)].tolist(),
             "pool100":pool.tolist()}
        ranks.append(row)
        if (i+1)%100==0:print("RANKED BLIND",i+1,"/",len(qm),flush=True)
    f=OUT/"E084_N11_BLIND_RANKINGS.jsonl"
    f.write_text("".join(json.dumps(x)+"\n" for x in ranks))
    lock={
      "status":"E084_N11_OUTER_META_RANKINGS_LOCKED_BEFORE_GOLD",
      "experiment_id":"E084_N11_v1","n_outer_queries":len(qm),
      "n_outer_groups":N_OUTER_GROUPS,"n_disjoint_from_e083_groups":480,
      "reference_rows":len(ref),"dataset_revision":REV,
      "training":training,"query_allowed_fields":["pmcid","case_prompt"],
      "reference_allowed_fields":["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"],
      "source_query_outcome_materialized_in_rank_stage":False,
      "rank_file_sha256":sha(f.read_bytes()),
      "config_sha256":sha(CONFIG.read_bytes()),
      "meta_row_pmcid_sha256":sha(json.dumps(qm.pmcid.tolist()).encode()),
      "ref_row_pmcid_sha256":sha(json.dumps(ref.pmcid.tolist()).encode()),
      "t001_and_medr_not_accessed":True
    }
    (OUT/"E084_N11_RANKING_LOCK.json").write_text(json.dumps(lock,indent=2)+"\n")
    print("RANKING LOCK:",json.dumps({"status":lock["status"],"n_outer_queries":len(qm),
              "n_inner_training":training["n_teacher_queries"],"rank_sha256":lock["rank_file_sha256"]},indent=2))

def get_meta_ref_info_or_die():
    meta=pd.read_parquet(SOURCE,columns=["pmcid","case_prompt"])
    meta.pmcid=meta.pmcid.astype(str)
    ids=set(meta.pmcid)
    old=set(sorted(ids,key=lambda x:sha(("E083:N10:"+x).encode()))[:480])
    available=ids-old
    outer=set(sorted(available,key=lambda x:sha(("E084:N11:OUTER:"+x).encode()))[:N_OUTER_GROUPS])
    qm=meta.loc[meta.pmcid.isin(outer)].reset_index(drop=True)
    rm=meta.loc[~meta.pmcid.isin(outer)].reset_index(drop=True)
    assert len(meta)==13092 and len(set(qm.pmcid))==N_OUTER_GROUPS
    assert not outer&old and set(qm.pmcid).isdisjoint(set(rm.pmcid))
    return rm,qm,outer,old

def eval_phase():
    lock=json.loads((OUT/"E084_N11_RANKING_LOCK.json").read_text())
    rf=OUT/"E084_N11_BLIND_RANKINGS.jsonl"
    assert lock["status"]=="E084_N11_OUTER_META_RANKINGS_LOCKED_BEFORE_GOLD"
    assert lock["rank_file_sha256"]==sha(rf.read_bytes())
    assert lock["config_sha256"]==sha(CONFIG.read_bytes())
    ranks=[json.loads(x) for x in rf.read_text().splitlines()]
    rm,qm,outer,old=get_meta_ref_info_or_die()
    assert lock["meta_row_pmcid_sha256"]==sha(json.dumps(qm.pmcid.tolist()).encode())
    # Outcome loading only after ranking lock checks above.
    all_data=pd.read_parquet(SOURCE,columns=["pmcid","diagnostic_reasoning","final_diagnosis"])
    all_data.pmcid=all_data.pmcid.astype(str)
    hist=all_data.loc[~all_data.pmcid.isin(outer)].reset_index(drop=True)
    gold=all_data.loc[all_data.pmcid.isin(outer)].reset_index(drop=True)
    assert hist.pmcid.tolist()==rm.pmcid.tolist()
    assert gold.pmcid.tolist()==qm.pmcid.tolist()
    rp=[mask_hist_points(r,d) for r,d in zip(hist.diagnostic_reasoning,hist.final_diagnosis)]
    qp=[mask_hist_points(r,d) for r,d in zip(gold.diagnostic_reasoning,gold.final_diagnosis)]
    specs={
      "primary_sparse":dict(ngram_range=(1,2),lowercase=True,sublinear_tf=True,
                            min_df=2,max_df=.98,max_features=100000,dtype=np.float32),
      "independent_char":dict(analyzer="char_wb",ngram_range=(3,5),lowercase=True,
                            sublinear_tf=True,min_df=2,max_df=.995,max_features=120000,dtype=np.float32)
    }
    results={};per=[{"query_index":i,"query_pmcid":qm.pmcid.iloc[i]} for i in range(len(qm))]
    for metric,params in specs.items():
        T,V,starts,counts,qstarts,qcounts=evaluator_matrix(qp,rp,params)
        TT=T.T.tocsr()
        for i,row in enumerate(ranks):
            assert row["query_index"]==i and row["query_pmcid"]==qm.pmcid.iloc[i]
            s=int(qstarts[i]);e=s+int(qcounts[i])
            S=(V[s:e]@TT).toarray().astype(np.float32,copy=False)
            for name,key in (("N8","n8_top50"),("N11","n11_top10")):
                order=row[key]
                for k in TOPKS:
                    per[i][f"{metric}_{name}_{k}"]=f1_of_order(S,starts,counts,order[:k])
            if (i+1)%100==0:print("EVALUATED",metric,i+1,"/",len(ranks),flush=True)
        metrics={}
        for k in TOPKS:
            b=np.array([x[f"{metric}_N8_{k}"] for x in per])
            m=np.array([x[f"{metric}_N11_{k}"] for x in per])
            d=m-b
            metrics[str(k)]={"N8":float(b.mean()),"N11":float(m.mean()),
               "delta":float(d.mean()),"cluster_ci95":paired_cluster_ci(d,qm.pmcid.to_numpy(),SEED+k+len(metric)),
               "improved":int(np.sum(d>1e-12)),"worsened":int(np.sum(d<-1e-12)),
               "ties":int(np.sum(abs(d)<=1e-12))}
        results[metric]=metrics
    df=pd.DataFrame(per)
    df.to_csv(OUT/"E084_N11_PER_QUERY.csv",index=False)
    out={"experiment_id":"E084_N11_v1","status":"train_internal_exploratory_after_blind_lock",
         "n_outer_queries":len(qm),"n_disjoint_meta_groups":len(set(qm.pmcid)),
         "ranking_lock":lock,"results":results,
         "selection_warning":"Exploratory only. No T001/test or previously used MedR external tuning. No confirmatory label.",
         "primary_comparison":"N11_HGB versus N8_RRF with same Top100 pool, SoftF1@10"}
    (OUT/"E084_N11_SUMMARY.json").write_text(json.dumps(out,indent=2)+"\n")
    primary=results["primary_sparse"]["10"];independent=results["independent_char"]["10"]
    candidate=(primary["delta"]>0 and primary["cluster_ci95"][0]>0 and
               independent["delta"]>0 and independent["cluster_ci95"][0]>0)
    decision={"experiment_id":"E084_N11_v1","sop_version":"1.0",
       "decision":"CANDIDATE_FOR_NEW_EXTERNAL_COHORT" if candidate else "NO_PROMOTION_ARCHIVE_EXPLORATORY",
       "new_external_confirmatory_validation":False,
       "primary":primary,"char":independent,
       "no_old_test_or_external_used":True}
    (OUT/"E084_N11_DECISION.json").write_text(json.dumps(decision,indent=2)+"\n")
    txt=["# E084/N11 N8-Conditioned Set-Aware Policy","",
         "Train-internal outer PMCID-group holdout, with ranking locked before gold.","",
         "| k | N8 sparse | N11 sparse | delta | N8 char | N11 char | char delta |",
         "|---:|---:|---:|---:|---:|---:|---:|"]
    for k in TOPKS:
        s=results["primary_sparse"][str(k)]
        c=results["independent_char"][str(k)]
        txt.append(f"| {k} | {s['N8']:.5f} | {s['N11']:.5f} | {s['delta']:+.5f} | {c['N8']:.5f} | {c['N11']:.5f} | {c['delta']:+.5f} |")
    txt+=["","This experiment predicts query-specific marginal reasoning-set utility from case and masked historical-reasoning similarities. The current query gold is never a ranking input.",
       "No test/external confirmation claim. Archive negative results, if any."]
    (OUT/"E084_N11_REPORT.md").write_text("\n".join(txt)+"\n")
    print("E084 PRIMARY",json.dumps(primary,indent=2),flush=True)
    print("E084 CHAR",json.dumps(independent,indent=2),flush=True)
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--stage",choices=["rank","eval"],required=True)
    args=a.parse_args()
    if args.stage=="rank":rank_phase()
    else:eval_phase()
