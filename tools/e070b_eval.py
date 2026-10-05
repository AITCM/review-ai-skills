from __future__ import annotations
import ast,csv,io,json,re,unicodedata,zipfile,pathlib
from collections import Counter
import numpy as np,pandas as pd

OUT=pathlib.Path("e070b");RNG=np.random.default_rng(20261005)
MC_REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";HF=f"https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/{MC_REV}/data"
root=pathlib.Path("ddx")

def norm(s):
 s=unicodedata.normalize("NFKC",str(s)).casefold()
 return " ".join(re.findall(r"\w+",s))
def parse_diff(s):
 try:return [(str(a),float(b)) for a,b in ast.literal_eval(str(s))]
 except Exception:return []

lock=json.load(open(OUT/"E070B_RANKING_LOCK.json"))
assert lock["status"]=="ddxplus_rankings_locked_before_outcomes_loaded" and lock["outcomes_available_to_ranker"]==[]
ranks=[json.loads(x) for x in open(OUT/"E070B_BLIND_RANKINGS.jsonl") if x.strip()]
assert len(ranks)==10000
row_to_q={int(x["source_row_index"]):int(x["query_index"]) for x in ranks}

# Outcomes loaded only after ranking lock.
outcomes={}
with zipfile.ZipFile(root/"release_test_patients.zip") as z:
 fn=z.namelist()[0]
 with z.open(fn) as raw:
  reader=csv.DictReader(io.TextIOWrapper(raw,encoding="utf-8"))
  for idx,row in enumerate(reader):
   if idx in row_to_q:
    qi=row_to_q[idx]
    outcomes[qi]={"pathology":str(row["PATHOLOGY"]),"differential":parse_diff(row["DIFFERENTIAL_DIAGNOSIS"])}
assert len(outcomes)==10000

tr=pd.read_parquet(HF+"/train-00000-of-00001.parquet",columns=["final_diagnosis"])
train_dx=[norm(x) for x in tr.final_diagnosis]
train_set=set(train_dx)

# Only exact normalized DDXPlus condition-name matches are counted; no synonym tuning.
cond=json.load(open(root/"release_conditions.json"))
condition_norm={norm(rec.get("cond-name-eng") or name):(rec.get("cond-name-eng") or name) for name,rec in cond.items()}
represented={d for d in condition_norm if d in train_set}

def metrics_for(qi,ids,k):
    diff=outcomes[qi]["differential"]
    pmap={norm(dx):float(p) for dx,p in diff}
    full_mass=sum(pmap.values())
    rep_mass=sum(p for d,p in pmap.items() if d in represented)
    retrieved=[]
    for j in ids[:k]:
        d=train_dx[int(j)]
        if d in condition_norm and d not in retrieved:retrieved.append(d)
    captured=sum(pmap.get(d,0.0) for d in retrieved)
    rep_recall=captured/rep_mass if rep_mass>0 else np.nan
    full_recall=captured/full_mass if full_mass>0 else np.nan
    true=norm(outcomes[qi]["pathology"])
    true_hit=int(true in retrieved) if true in represented else None
    return rep_recall,full_recall,captured,true_hit,len(retrieved),rep_mass

per=[]
for r in ranks:
 qi=int(r["query_index"])
 for method,key in (("TFIDF","baseline_top50"),("C2","method_top50")):
  for k in (1,3,10):
   rr,fr,cap,hit,nret,repm=metrics_for(qi,r[key],k)
   per.append({"query_index":qi,"method":method,"k":k,"representable_mass_recall":rr,
               "full_differential_mass_recall":fr,"captured_probability_mass":cap,
               "true_pathology_hit":hit,"unique_DDX_conditions_retrieved":nret,"representable_mass":repm})
df=pd.DataFrame(per)
df.to_csv(OUT/"E070B_PER_QUERY_METRICS.csv",index=False)

def paired_summary(k,col,cohort_mask=None,nboot=10000):
 a=df[(df.method=="TFIDF")&(df.k==k)].sort_values("query_index")
 b=df[(df.method=="C2")&(df.k==k)].sort_values("query_index")
 assert (a.query_index.to_numpy()==b.query_index.to_numpy()).all()
 x=a[col].to_numpy(float);y=b[col].to_numpy(float)
 keep=np.isfinite(x)&np.isfinite(y)
 if cohort_mask is not None:keep &= cohort_mask
 x=x[keep];y=y[keep];d=y-x
 boots=[]
 for _ in range(nboot):
  ix=RNG.integers(0,len(d),len(d));boots.append(float(d[ix].mean()))
 return {"n":len(d),"TFIDF":float(x.mean()),"C2":float(y.mean()),"delta":float(d.mean()),
         "ci95":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))],
         "improved":int((d>0).sum()),"worsened":int((d<0).sum()),"tied":int((d==0).sum())}

base10=df[(df.method=="TFIDF")&(df.k==10)].sort_values("query_index")
primary_mask=np.isfinite(base10.representable_mass_recall.to_numpy(float))
summary={"status":"locked_cross_domain_DDXPlus_stress_test",
 "dataset":"DDXPlus English test","sample_size":10000,
 "selection":"outcome-blind deterministic hash sample from 134,529 test patients",
 "adapter":"locked deterministic structured-evidence-to-text serialization; no learned adapter",
 "diagnosis_mapping":"exact normalized condition-name match only; no synonym/ontology tuning",
 "represented_DDX_conditions":len(represented),"of_49":49,
 "primary_endpoint":"representable differential-diagnosis probability-mass recall at Top-10",
 "primary_cohort":"queries with at least one differential diagnosis exactly representable in MedCaseReasoning train",
 "results":{"representable_mass_recall":{},"full_differential_mass_recall":{},"captured_probability_mass":{}},
 "secondary_true_pathology_hit":{}}

for k in (1,3,10):
 for col in ("representable_mass_recall","full_differential_mass_recall","captured_probability_mass"):
  summary["results"][col][str(k)]=paired_summary(k,col)
 # true pathology hit only on representable pathology queries
 a=df[(df.method=="TFIDF")&(df.k==k)].sort_values("query_index")
 b=df[(df.method=="C2")&(df.k==k)].sort_values("query_index")
 x=a.true_pathology_hit.to_numpy(float);y=b.true_pathology_hit.to_numpy(float);keep=np.isfinite(x)&np.isfinite(y)
 x=x[keep];y=y[keep];d=y-x
 # paired bootstrap for absolute hit-rate delta
 boots=[]
 for _ in range(10000):
  ix=RNG.integers(0,len(d),len(d));boots.append(float(d[ix].mean()))
 summary["secondary_true_pathology_hit"][str(k)]={"n":len(d),"TFIDF_hit_rate":float(x.mean()),"C2_hit_rate":float(y.mean()),
   "delta":float(d.mean()),"ci95":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))],
   "helped":int(((x==0)&(y==1)).sum()),"harmed":int(((x==1)&(y==0)).sum())}

# pathology-level descriptive heterogeneity for primary Top10
a=df[(df.method=="TFIDF")&(df.k==10)].set_index("query_index")
b=df[(df.method=="C2")&(df.k==10)].set_index("query_index")
path_rows=[]
for qi in range(10000):
 if not np.isfinite(a.loc[qi,"representable_mass_recall"]):continue
 path=outcomes[qi]["pathology"]
 path_rows.append({"query_index":qi,"pathology":path,"delta":float(b.loc[qi,"representable_mass_recall"]-a.loc[qi,"representable_mass_recall"])})
pdf=pd.DataFrame(path_rows)
path_summary=[]
for path,g in pdf.groupby("pathology"):
 path_summary.append({"pathology":path,"n":len(g),"mean_delta":float(g.delta.mean()),
                      "positive":int((g.delta>0).sum()),"negative":int((g.delta<0).sum())})
summary["pathology_descriptive"]=sorted(path_summary,key=lambda x:-x["n"])
summary["limits"]=[
 "DDXPlus is synthetic and structurally generated; this is a cross-domain stress test rather than real-world external clinical validation.",
 "Only 9 of 49 DDXPlus condition names exactly match normalized MedCaseReasoning training diagnoses.",
 "The primary endpoint conditions on differential diagnoses that are exactly representable in the historical library.",
 "No synonym or ontology mapping is introduced, to avoid outcome-tuned label reconciliation.",
 "The text adapter is deterministic and frozen before outcome loading, but DDXPlus structured evidence differs substantially from narrative case reports.",
 "Differential-diagnosis probability mass measures diagnostic-space coverage, not clinician-perceived case utility or patient outcome."
]
(OUT/"E070B_DDXPLUS_RESULTS.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
print(json.dumps({k:v for k,v in summary.items() if k!="pathology_descriptive"},indent=2,ensure_ascii=False))
