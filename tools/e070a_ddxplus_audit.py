from __future__ import annotations
import ast,csv,io,json,re,unicodedata,zipfile,pathlib
from collections import Counter,defaultdict
import numpy as np,pandas as pd

OUT=pathlib.Path("e070a");OUT.mkdir(exist_ok=True)
MC_REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
HF=f"https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/{MC_REV}/data"

def norm(s):
 s=unicodedata.normalize("NFKC",str(s)).casefold()
 return " ".join(re.findall(r"\w+",s))
def parse_diff(s):
 try:x=ast.literal_eval(str(s))
 except Exception:return []
 return [(str(a),float(b)) for a,b in x]
def parse_evid(s):
 try:return list(ast.literal_eval(str(s)))
 except Exception:return []

# verified DDXPlus files downloaded from E030 artifact
root=pathlib.Path("ddx")
cond=json.load(open(root/"release_conditions.json"))
evid=json.load(open(root/"release_evidences.json"))
assert len(cond)==49 and len(evid)==223

tr=pd.read_parquet(HF+"/train-00000-of-00001.parquet",columns=["final_diagnosis"])
train_labels=[norm(x) for x in tr.final_diagnosis]
label_counts=Counter(train_labels)

conditions=[]
for name,rec in cond.items():
 n=norm(rec.get("cond-name-eng") or name)
 conditions.append({"condition":name,"norm":n,"icd10":rec.get("icd10-id"),
                    "exact_train_count":label_counts.get(n,0)})
cdf=pd.DataFrame(conditions).sort_values(["exact_train_count","condition"],ascending=[False,True])
cdf.to_csv(OUT/"E070A_CONDITION_OVERLAP.csv",index=False)

# scan DDXPlus test without loading full file to memory
with zipfile.ZipFile(root/"release_test_patients.zip") as z:
 fn=z.namelist()[0]
 with z.open(fn) as raw:
  txt=io.TextIOWrapper(raw,encoding="utf-8")
  reader=csv.DictReader(txt)
  nrows=0; path_counts=Counter(); diff_counts=Counter(); diff_len=[]; evidence_len=[]
  outcome_exact_representable=0; diff_any_exact_representable=0
  examples=[]
  for row in reader:
   nrows+=1
   path=str(row["PATHOLOGY"]);path_counts[path]+=1
   dif=parse_diff(row["DIFFERENTIAL_DIAGNOSIS"]);diff_len.append(len(dif))
   ev=parse_evid(row["EVIDENCES"]);evidence_len.append(len(ev))
   for dx,p in dif:diff_counts[dx]+=1
   if label_counts.get(norm(path),0)>0:outcome_exact_representable+=1
   if any(label_counts.get(norm(dx),0)>0 for dx,_ in dif):diff_any_exact_representable+=1
   if len(examples)<5:examples.append({"AGE":row["AGE"],"SEX":row["SEX"],"PATHOLOGY":path,
      "DIFFERENTIAL_DIAGNOSIS":dif,"EVIDENCES":ev[:20],"INITIAL_EVIDENCE":row["INITIAL_EVIDENCE"]})

# deterministic evidence serialization spec
serialization={
 "age":"Age: <AGE> years.",
 "sex":"Sex: male/female.",
 "evidence_boolean_positive":"<question_en>: yes.",
 "evidence_boolean_negative":"Negative/default evidence is not serialized unless explicitly present in EVIDENCES.",
 "evidence_categorical":"<question_en>: <English value meaning>.",
 "evidence_numeric_or_other":"<question_en>: <observed value>.",
 "initial_evidence":"Serialized identically to EVIDENCES; no special weight in Frozen C2.",
 "outcome_excluded":["PATHOLOGY","DIFFERENTIAL_DIAGNOSIS"],
 "note":"Deterministic template adapter only; no learned or outcome-tuned text generation."
}

summary={
 "status":"compatibility_audit_only_no_C2_result",
 "ddxplus":{"conditions":len(cond),"evidences":len(evid),"test_rows":nrows,
            "test_pathology_counts":dict(path_counts),
            "differential_size":{"median":float(np.median(diff_len)),"p25":float(np.percentile(diff_len,25)),"p75":float(np.percentile(diff_len,75)),
                                 "min":int(min(diff_len)),"max":int(max(diff_len))},
            "evidence_count":{"median":float(np.median(evidence_len)),"p25":float(np.percentile(evidence_len,25)),"p75":float(np.percentile(evidence_len,75))}},
 "diagnosis_space":{
   "ddx_conditions_exactly_present_as_medcase_train_labels":int((cdf.exact_train_count>0).sum()),
   "of_49":49,
   "test_patients_whose_true_pathology_exactly_present_in_medcase_train":outcome_exact_representable,
   "test_pathology_exact_representability_rate":outcome_exact_representable/nrows,
   "test_patients_with_any_differential_diagnosis_exactly_present_in_medcase_train":diff_any_exact_representable,
   "test_any_differential_exact_representability_rate":diff_any_exact_representable/nrows},
 "query_serialization_spec":serialization,
 "recommended_evaluation":{
   "primary_candidate":"differential-diagnosis utility among retrieved historical-case diagnoses, restricted to exact normalized DDXPlus disease-name matches",
   "secondary":"true-pathology exact-label hit among retrieved diagnoses",
   "warning":"If exact diagnosis-space overlap is sparse, exact-label outcomes will be underpowered and should not be treated as the main cross-domain endpoint."},
 "limits":["DDXPlus is synthetic and structurally generated, not a real clinical case-report corpus.",
           "Exact disease-name normalization does not resolve synonyms or diagnostic hierarchy.",
           "A deterministic text adapter is needed because Frozen C2 consumes textual case presentations."]
}
(OUT/"E070A_COMPATIBILITY_SUMMARY.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
(OUT/"E070A_SERIALIZATION_SPEC.json").write_text(json.dumps(serialization,indent=2,ensure_ascii=False)+"\n")
(OUT/"E070A_EXAMPLES.json").write_text(json.dumps(examples,indent=2,ensure_ascii=False)+"\n")
print(json.dumps(summary,indent=2,ensure_ascii=False))
