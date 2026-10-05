from __future__ import annotations
import ast,csv,io,json,re,unicodedata,zipfile,pathlib
from collections import Counter,defaultdict
import numpy as np,pandas as pd

OUT=pathlib.Path("e070c");OUT.mkdir(exist_ok=True)
MC_REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";HF=f"https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/{MC_REV}/data"
root=pathlib.Path("ddx")
def norm(s):
 s=unicodedata.normalize("NFKC",str(s)).casefold()
 return " ".join(re.findall(r"\w+",s))
def parse_diff(s):
 try:return [(norm(a),float(b)) for a,b in ast.literal_eval(str(s))]
 except Exception:return []

rankfiles=list(pathlib.Path("locked").rglob("E070B_BLIND_RANKINGS.jsonl"))
lockfiles=list(pathlib.Path("locked").rglob("E070B_RANKING_LOCK.json"))
assert len(rankfiles)==len(lockfiles)==1
lock=json.load(open(lockfiles[0]));assert lock["status"]=="ddxplus_rankings_locked_before_outcomes_loaded"
ranks=[json.loads(x) for x in open(rankfiles[0]) if x.strip()]
assert len(ranks)==10000
row_to_q={int(x["source_row_index"]):int(x["query_index"]) for x in ranks}

# load outcomes after ranking lock
outcomes={}
full_path_counts=Counter()
with zipfile.ZipFile(root/"release_test_patients.zip") as z:
 fn=z.namelist()[0]
 with z.open(fn) as raw:
  reader=csv.DictReader(io.TextIOWrapper(raw,encoding="utf-8"))
  for idx,row in enumerate(reader):
   full_path_counts[row["PATHOLOGY"]]+=1
   if idx in row_to_q:
    outcomes[row_to_q[idx]]={"pathology":str(row["PATHOLOGY"]),"differential":dict(parse_diff(row["DIFFERENTIAL_DIAGNOSIS"]))}
assert len(outcomes)==10000

tr=pd.read_parquet(HF+"/train-00000-of-00001.parquet",columns=["final_diagnosis"])
tdx=[norm(x) for x in tr.final_diagnosis]
cond=json.load(open(root/"release_conditions.json"))
conditions={norm(rec.get("cond-name-eng") or name):(rec.get("cond-name-eng") or name) for name,rec in cond.items()}
represented=sorted(d for d in conditions if d in set(tdx))

# sample representativeness
sample_counts=Counter(outcomes[i]["pathology"] for i in range(10000))
freq_rows=[]
for p,n in full_path_counts.items():
 full=n/sum(full_path_counts.values());sam=sample_counts[p]/10000
 freq_rows.append({"pathology":p,"full_rate":full,"sample_rate":sam,"delta":sam-full})
tv=.5*sum(abs(x["delta"]) for x in freq_rows)

summary={"status":"posthoc_concentration_and_precision_audit_on_locked_DDXPlus_rankings",
 "represented_conditions":[conditions[d] for d in represented],
 "sample_representativeness":{"total_variation_distance_pathology_distribution":float(tv),
   "max_absolute_pathology_rate_delta":float(max(abs(x["delta"]) for x in freq_rows))},
 "methods":{},"limits":["This audit explains the sparse E070B effect and does not alter rankings or the primary endpoint."]}

for method,key in (("TFIDF","baseline_top50"),("C2","method_top50")):
 occurrences=[];per_condition=defaultdict(lambda:{"retrieved_queries":0,"gold_relevant_queries":0,"prob_sum":0.0,"true_pathology_queries":0})
 queries_with_any=0
 for r in ranks:
  qi=int(r["query_index"]);diff=outcomes[qi]["differential"];path=norm(outcomes[qi]["pathology"])
  seen=[]
  for j in r[key][:10]:
   d=tdx[int(j)]
   if d in represented and d not in seen:seen.append(d)
  if seen:queries_with_any+=1
  for d in seen:
   rel=d in diff;p=float(diff.get(d,0.0))
   occurrences.append((qi,d,rel,p,d==path))
   z=per_condition[d];z["retrieved_queries"]+=1;z["gold_relevant_queries"]+=int(rel);z["prob_sum"]+=p;z["true_pathology_queries"]+=int(d==path)
 nocc=len(occurrences);nrel=sum(x[2] for x in occurrences);prob=sum(x[3] for x in occurrences);ntrue=sum(x[4] for x in occurrences)
 condout=[]
 for d in represented:
  z=per_condition[d]
  condout.append({"condition":conditions[d],**z,
    "gold_relevance_precision":z["gold_relevant_queries"]/z["retrieved_queries"] if z["retrieved_queries"] else None,
    "mean_gold_probability_per_occurrence":z["prob_sum"]/z["retrieved_queries"] if z["retrieved_queries"] else None})
 summary["methods"][method]={
  "queries_with_any_represented_condition_in_top10":queries_with_any,
  "unique_condition_occurrences":nocc,
  "gold_differential_relevant_occurrences":nrel,
  "micro_gold_relevance_precision":nrel/nocc if nocc else None,
  "captured_probability_mass_per_occurrence":prob/nocc if nocc else None,
  "true_pathology_occurrences":ntrue,
  "condition_breakdown":condout}

# paired query-level retrieval burden and relevant condition count
paired=[]
for r in ranks:
 qi=int(r["query_index"]);diff=outcomes[qi]["differential"]
 row={"query_index":qi}
 for method,key in (("TFIDF","baseline_top50"),("C2","method_top50")):
  seen=[]
  for j in r[key][:10]:
   d=tdx[int(j)]
   if d in represented and d not in seen:seen.append(d)
  row[f"{method}_represented_count"]=len(seen)
  row[f"{method}_relevant_count"]=sum(d in diff for d in seen)
  row[f"{method}_irrelevant_count"]=sum(d not in diff for d in seen)
 row["delta_relevant"]=row["C2_relevant_count"]-row["TFIDF_relevant_count"]
 row["delta_irrelevant"]=row["C2_irrelevant_count"]-row["TFIDF_irrelevant_count"]
 paired.append(row)
pdf=pd.DataFrame(paired);pdf.to_csv(OUT/"E070C_PER_QUERY_CONCENTRATION.csv",index=False)
summary["paired_top10"]={
 "mean_delta_relevant_condition_count":float(pdf.delta_relevant.mean()),
 "mean_delta_irrelevant_condition_count":float(pdf.delta_irrelevant.mean()),
 "queries_more_relevant":int((pdf.delta_relevant>0).sum()),"queries_fewer_relevant":int((pdf.delta_relevant<0).sum()),
 "queries_more_irrelevant":int((pdf.delta_irrelevant>0).sum()),"queries_fewer_irrelevant":int((pdf.delta_irrelevant<0).sum())}
summary["sample_pathology_distribution"]=sorted(freq_rows,key=lambda x:-abs(x["delta"]))
(OUT/"E070C_CONCENTRATION_PRECISION.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
print(json.dumps({k:v for k,v in summary.items() if k!="sample_pathology_distribution"},indent=2,ensure_ascii=False))
