from __future__ import annotations
import json,re,unicodedata,pathlib,hashlib
import numpy as np,pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e021-n1-output");OUT.mkdir(exist_ok=True)
SEED=20261007
BUDGETS=(50,100,200)
SET_KS=(1,3,10)
RRF_C=60.0
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

def mask_points(reason,diag):
    dd=nw(diag);out=[]
    for x in split_reason(reason):
        xx=nw(x)
        if dd and dd in xx: xx=xx.replace(dd," diagnosismask ")
        out.append(xx)
    return out

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
    ps=0.0; pc=0
    alive=np.ones(len(pool),dtype=bool)
    scores={}
    for step in range(1,kmax+1):
        recs=np.maximum(recvec[:,None],qmax[:,pool]).mean(axis=0)
        precs=(ps+psum[pool])/(pc+counts[pool])
        vals=f1(recs,precs)
        vals[~alive]=-np.inf
        order=np.lexsort((pool,-vals))
        pos=int(order[0]); j=int(pool[pos])
        chosen.append(j); alive[pos]=False
        recvec=np.maximum(recvec,qmax[:,j])
        ps+=float(psum[j]); pc+=int(counts[j])
        if step in SET_KS: scores[step]=float(vals[pos])
    return chosen,scores

def full_rank(scores):
    ids=np.arange(len(scores),dtype=int)
    return np.lexsort((ids,-scores))

def rrf_rank(*orders):
    n=len(orders[0])
    score=np.zeros(n,dtype=np.float64)
    for order in orders:
        pos=np.empty(n,dtype=np.int32)
        pos[order]=np.arange(n,dtype=np.int32)
        score += 1.0/(RRF_C+pos+1.0)
    return full_rank(score)

def boot_ci(x,seed,n=3000):
    x=np.asarray(x,float);rng=np.random.default_rng(seed)
    vals=np.empty(n,float)
    for b in range(n):
        ids=rng.integers(0,len(x),len(x)); vals[b]=x[ids].mean()
    return [float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))]

tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
assert len(tr)==13092 and len(va)==500

# ---------- Query-only retrieval representations ----------
# 1) lexical case TF-IDF
cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,
                   max_features=80000,dtype=np.float32)
X=cv.fit_transform(tr.case_prompt.astype(str))
Q=cv.transform(va.case_prompt.astype(str))
LEX=(Q@X.T).toarray().astype(np.float32)

# 2) low-rank case latent route
case_svd=TruncatedSVD(256,n_iter=7,random_state=SEED)
XC=case_svd.fit_transform(X).astype(np.float32)
QC=case_svd.transform(Q).astype(np.float32)
XL=normalize(XC); QL=normalize(QC)
LAT=(QL@XL.T).astype(np.float32)

# 3) predicted-reasoning route, learned only from historical train cases
tr_pts=[mask_points(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
va_pts=[mask_points(r,d) for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
rdoc=[" ".join(x) for x in tr_pts]
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,
                   max_features=80000,dtype=np.float32)
R=rv.fit_transform(rdoc)
rsvd=TruncatedSVD(128,n_iter=7,random_state=SEED)
ZR=rsvd.fit_transform(R).astype(np.float32)
xm=XC.mean(0,keepdims=True); ym=ZR.mean(0,keepdims=True)
A=XC-xm; Y=ZR-ym
W=np.linalg.solve(A.T@A+10*np.eye(A.shape[1],dtype=np.float32),A.T@Y)
P=normalize((QC-xm)@W+ym)
ZI=normalize(ZR)
PRD=(P@ZI.T).astype(np.float32)

# ---------- Independent outcome-defined utility evaluator (validation only) ----------
flat_t=[p for ps in tr_pts for p in ps]
flat_v=[p for ps in va_pts for p in ps]
ev=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,
                   max_features=100000,dtype=np.float32)
T=ev.fit_transform(flat_t)
V=ev.transform(flat_v)
counts=np.asarray([len(z) for z in tr_pts],dtype=np.int32)
starts=np.concatenate(([0],np.cumsum(counts)[:-1])).astype(np.int64)
vo=[];o=0
for z in va_pts:
    vo.append((o,o+len(z)));o+=len(z)
Tt=T.T.tocsr()
all_ids=np.arange(len(tr),dtype=int)

route_names=[
    "lexical","case_latent","predicted_reasoning",
    "rrf_lex_prd","rrf_lat_prd","rrf_lex_lat","rrf_tri"
]
records=[]
global_rows=[]

# Process validation queries in batches to keep sparse-point similarity bounded.
batches=[];start_q=0
while start_q<len(va):
    end_q=start_q;pts_n=0
    while end_q<len(va):
        n=vo[end_q][1]-vo[end_q][0]
        if end_q>start_q and pts_n+n>192: break
        pts_n+=n;end_q+=1
    batches.append((start_q,end_q)); start_q=end_q

train_dx=[nw(x) for x in tr.final_diagnosis]
val_dx=[nw(x) for x in va.final_diagnosis]
representable=np.array([x in set(train_dx) for x in val_dx],dtype=bool)

for bno,(qa,qb) in enumerate(batches,1):
    ps0=vo[qa][0]; ps1=vo[qb-1][1]
    SB=(V[ps0:ps1]@Tt).toarray().astype(np.float32,copy=False)
    for i in range(qa,qb):
        s,e=vo[i]; S=SB[s-ps0:e-ps0]
        qmax=np.stack([np.maximum.reduceat(row,starts) for row in S],axis=0)
        pmax=S.max(axis=0)
        psum=np.add.reduceat(pmax,starts)
        rec=qmax.mean(axis=0); prec=psum/counts
        util=f1(rec,prec)
        gorder=np.lexsort((all_ids,-util))
        gbest=int(gorder[0])
        _,gg=greedy(qmax,psum,counts,all_ids,10)
        global_rows.append({
            "query_index":i,
            "global_best_single_train_index":gbest,
            "global_best_single_utility":float(util[gbest]),
            "global_greedy_1":gg[1],"global_greedy_3":gg[3],"global_greedy_10":gg[10],
            "representable_exact_label":bool(representable[i])
        })

        lex=full_rank(LEX[i]); lat=full_rank(LAT[i]); prd=full_rank(PRD[i])
        routes={
            "lexical":lex,
            "case_latent":lat,
            "predicted_reasoning":prd,
            "rrf_lex_prd":rrf_rank(lex,prd),
            "rrf_lat_prd":rrf_rank(lat,prd),
            "rrf_lex_lat":rrf_rank(lex,lat),
            "rrf_tri":rrf_rank(lex,lat,prd),
        }
        for name,order in routes.items():
            for B in BUDGETS:
                pool=order[:B]
                _,gp=greedy(qmax,psum,counts,pool,10)
                label_hit=any(train_dx[j]==val_dx[i] for j in pool) if representable[i] else False
                records.append({
                    "query_index":i,"route":name,"budget":B,
                    "global_best_in_pool":bool(gbest in set(pool.tolist())),
                    "pool_best_single_utility":float(np.max(util[pool])),
                    "pool_greedy_1":gp[1],"pool_greedy_3":gp[3],"pool_greedy_10":gp[10],
                    "exact_label_hit":bool(label_hit),
                    "representable_exact_label":bool(representable[i])
                })
    print(f"BATCH {bno}/{len(batches)} queries {qa}:{qb}",flush=True)

df=pd.DataFrame(records)
gdf=pd.DataFrame(global_rows)
df.to_csv(OUT/"E021_N1_PER_QUERY_ROUTE_BUDGET.csv",index=False)
gdf.to_csv(OUT/"E021_N1_GLOBAL_ORACLE.csv",index=False)

summary={
  "experiment":"E021-N1 fixed-budget candidate-generation audit",
  "status":"validation development only; MedCaseReasoning test untouched",
  "dataset_revision":REV,
  "n_train":len(tr),"n_validation":len(va),
  "candidate_budgets":list(BUDGETS),
  "routes":{
    "lexical":"case-prompt TF-IDF unigram/bigram cosine",
    "case_latent":"256-d TruncatedSVD case-prompt cosine",
    "predicted_reasoning":"case latent -> 128-d masked-reasoning latent ridge projection",
    "rrf_lex_prd":"fixed-c=60 reciprocal-rank fusion of lexical + predicted reasoning",
    "rrf_lat_prd":"fixed-c=60 reciprocal-rank fusion of case latent + predicted reasoning",
    "rrf_lex_lat":"fixed-c=60 reciprocal-rank fusion of lexical + case latent",
    "rrf_tri":"fixed-c=60 reciprocal-rank fusion of all three routes"
  },
  "evaluation":{
    "candidate_selection_inputs":"validation case_prompt only",
    "historical_library_fields":"train case_prompt + diagnosis-masked diagnostic_reasoning",
    "outcome_use":"validation diagnostic_reasoning used only after candidate rankings are fixed, for utility/oracle evaluation",
    "primary":"global-best-single inclusion and candidate-pool greedy-oracle reasoning-set softF1",
    "secondary":"exact-label candidate recall among representable diagnoses"
  },
  "results":{}
}

for name in route_names:
    summary["results"][name]={}
    for B in BUDGETS:
        x=df[(df.route==name)&(df.budget==B)].sort_values("query_index")
        assert len(x)==len(va)
        res={
          "global_best_single_in_pool_rate":float(x.global_best_in_pool.mean()),
          "mean_pool_best_single_utility":float(x.pool_best_single_utility.mean()),
          "exact_label_recall_representable":float(x.loc[x.representable_exact_label,"exact_label_hit"].mean())
        }
        for k in SET_KS:
            pool=x[f"pool_greedy_{k}"].to_numpy()
            glob=gdf.sort_values("query_index")[f"global_greedy_{k}"].to_numpy()
            gap=pool-glob
            res[f"pool_greedy_oracle_{k}_mean"]=float(pool.mean())
            res[f"global_greedy_oracle_{k}_mean"]=float(glob.mean())
            res[f"oracle_retention_{k}"]=float(pool.mean()/glob.mean())
            res[f"gap_to_global_{k}"]=float(gap.mean())
            res[f"gap_to_global_{k}_ci95"]=boot_ci(gap,SEED+B+k+len(name))
        summary["results"][name][str(B)]=res

# Rank routes per budget using predeclared primary statistics, no hidden tuning.
summary["leaderboard"]={}
for B in BUDGETS:
    rows=[]
    for name in route_names:
        r=summary["results"][name][str(B)]
        rows.append({
          "route":name,
          "budget":B,
          "global_best_recall":r["global_best_single_in_pool_rate"],
          "oracle10":r["pool_greedy_oracle_10_mean"],
          "oracle3":r["pool_greedy_oracle_3_mean"],
          "oracle1":r["pool_greedy_oracle_1_mean"]
        })
    rows=sorted(rows,key=lambda z:(-z["oracle10"],-z["global_best_recall"],z["route"]))
    summary["leaderboard"][str(B)]=rows

(OUT/"E021_N1_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")

md=["# E021-N1 Fixed-budget Candidate-generation Audit","",
"**Policy:** validation-only development analysis. The 897-case MedCaseReasoning test split is not loaded or queried.","",
"Candidate generators are compared at equal budgets (50/100/200) so larger unions do not receive an automatic oracle advantage.","",
"## Leaderboard by budget",""]
for B in BUDGETS:
    md += [f"### Budget {B}","",
    "| route | global-best recall | greedy oracle @1 | @3 | @10 | exact-label recall* |",
    "|---|---:|---:|---:|---:|---:|"]
    for z in summary["leaderboard"][str(B)]:
        r=summary["results"][z["route"]][str(B)]
        md.append(f"| {z['route']} | {r['global_best_single_in_pool_rate']:.3f} | {r['pool_greedy_oracle_1_mean']:.5f} | {r['pool_greedy_oracle_3_mean']:.5f} | {r['pool_greedy_oracle_10_mean']:.5f} | {r['exact_label_recall_representable']:.3f} |")
    md.append("")
md += ["\* Exact-label recall is secondary and restricted to validation diagnoses represented in the training library.","",
"## Interpretation",
"- A higher pool oracle means the candidate generator supplies a better substrate for any downstream reranker.",
"- Global-best inclusion measures whether the single highest-utility historical case is reachable at all.",
"- RRF uses a fixed constant (60) and is not tuned on validation outcomes.",
"- No route is evaluated on the sealed test split in this experiment."
]
(OUT/"E021_N1_REPORT.md").write_text("\n".join(md)+"\n")
print(json.dumps(summary,indent=2))
