from __future__ import annotations
import argparse,csv,json,pathlib
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score

CHOICES={"A clearly better","A slightly better","About equal","B slightly better","B clearly better","Neither useful"}

def read_responses(path):
    df=pd.read_csv(path,dtype=str).fillna("")
    req={"query_index","case_id","preference","confidence","misleading","comment"}
    missing=req-set(df.columns)
    if missing: raise ValueError(f"missing response columns: {sorted(missing)}")
    if len(df)!=117: raise ValueError(f"expected 117 manually rated cases, got {len(df)}")
    if df.query_index.duplicated().any(): raise ValueError("duplicate query_index")
    bad=set(df.preference)-CHOICES
    if bad: raise ValueError(f"invalid preferences: {bad}")
    return df

def read_key(path):
    rows=[json.loads(x) for x in open(path,encoding="utf-8") if x.strip()]
    return {str(x["query_index"]):x for x in rows}

def standardize(pref,key):
    A=key["A_method"];B=key["B_method"]
    if pref=="About equal": return "equal",0
    if pref=="Neither useful": return "neither",0
    if pref=="A clearly better": return ("C2 clearly",2) if A=="C2" else ("TFIDF clearly",-2)
    if pref=="A slightly better": return ("C2 slightly",1) if A=="C2" else ("TFIDF slightly",-1)
    if pref=="B slightly better": return ("C2 slightly",1) if B=="C2" else ("TFIDF slightly",-1)
    if pref=="B clearly better": return ("C2 clearly",2) if B=="C2" else ("TFIDF clearly",-2)
    raise ValueError(pref)

def decode(df,key):
    out=[]
    for _,r in df.iterrows():
        q=str(r.query_index);k=key[q];cat,ordv=standardize(r.preference,k)
        out.append({**r.to_dict(),"decoded_category":cat,"ordinal_C2":ordv,
                    "direction":"C2" if ordv>0 else ("TFIDF" if ordv<0 else "tie")})
    return pd.DataFrame(out)

def rater_summary(df):
    direction=Counter(df.direction)
    cats=Counter(df.decoded_category)
    non_tie=direction["C2"]+direction["TFIDF"]
    bt=binomtest(direction["C2"],non_tie,p=.5) if non_tie else None
    ci=bt.proportion_ci(confidence_level=.95,method="exact") if bt else None
    return {
      "n":len(df),"direction_counts":dict(direction),"category_counts":dict(cats),
      "mean_ordinal_C2":float(df.ordinal_C2.astype(float).mean()),
      "C2_win_fraction_among_directional":float(direction["C2"]/non_tie) if non_tie else None,
      "C2_win_fraction_exact95CI":[float(ci.low),float(ci.high)] if ci else None,
      "two_sided_exact_binomial_p":float(bt.pvalue) if bt else None}

def consensus_status(a,b):
    da,db=a.direction,b.direction
    if da==db: return da
    if "tie" in (da,db): return "needs_adjudication"
    return "needs_adjudication"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--rater-a",required=True)
    ap.add_argument("--rater-b",required=True)
    ap.add_argument("--key-a",required=True)
    ap.add_argument("--key-b",required=True)
    ap.add_argument("--adjudication",default="")
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True)
    A=decode(read_responses(args.rater_a),read_key(args.key_a))
    B=decode(read_responses(args.rater_b),read_key(args.key_b))
    M=A.merge(B,on=["query_index","case_id"],suffixes=("_A","_B"),validate="one_to_one")
    M["consensus_pre_adjudication"]=[consensus_status(a,b) for a,b in zip(A.itertuples(),B.itertuples())]
    # Agreement before adjudication.
    directional_labels=["C2","tie","TFIDF"]
    percent=float(np.mean(M.direction_A==M.direction_B))
    kappa=float(cohen_kappa_score(M.direction_A,M.direction_B,labels=directional_labels))
    exact6=float(np.mean(M.decoded_category_A==M.decoded_category_B))
    result={
      "protocol":"E040 locked clinician pairwise analysis v1",
      "manual_cases_per_rater":117,
      "structural_ties_not_manually_rated":3,
      "rater_A":rater_summary(A),"rater_B":rater_summary(B),
      "agreement_pre_adjudication":{"directional_percent":percent,"directional_cohen_kappa":kappa,"exact_6category_percent":exact6},
      "pre_adjudication_consensus_counts":dict(Counter(M.consensus_pre_adjudication)),
      "primary_endpoint_status":"pending_adjudication",
      "primary_endpoint_definition":"C2 preferred vs TF-IDF preferred among adjudicated directional non-tie cases; structural identical-set cases are automatic ties."
    }
    # Optional adjudication CSV columns: query_index,decision where decision in C2/TFIDF/tie.
    if args.adjudication:
        J=pd.read_csv(args.adjudication,dtype=str).fillna("")
        if not {"query_index","decision"}<=set(J.columns): raise ValueError("adjudication requires query_index,decision")
        amap=dict(zip(J.query_index.astype(str),J.decision))
        final=[]
        for r in M.itertuples():
            pre=r.consensus_pre_adjudication
            if pre!="needs_adjudication": final.append(pre)
            else:
                d=amap.get(str(r.query_index),"")
                if d not in {"C2","TFIDF","tie"}: raise ValueError(f"missing/invalid adjudication for {r.query_index}")
                final.append(d)
        # Add the 3 structural ties.
        final += ["tie"]*3
        C=Counter(final);n=C["C2"]+C["TFIDF"];bt=binomtest(C["C2"],n,p=.5);ci=bt.proportion_ci(.95,method="exact")
        result["primary_endpoint_status"]="complete"
        result["primary_endpoint"]={"all_120_counts":dict(C),"directional_n":n,
          "C2_win_fraction":float(C["C2"]/n) if n else None,
          "exact95CI":[float(ci.low),float(ci.high)],"two_sided_exact_binomial_p":float(bt.pvalue)}
    A.to_csv(out/"decoded_rater_A.csv",index=False);B.to_csv(out/"decoded_rater_B.csv",index=False)
    M.to_csv(out/"paired_pre_adjudication.csv",index=False)
    (out/"E040_ANALYSIS_RESULTS.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=="__main__": main()
