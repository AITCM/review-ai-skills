from __future__ import annotations
import json,re,unicodedata,pathlib,urllib.request,hashlib
from collections import Counter
import numpy as np,pandas as pd

CRB_REV="3297c5b2db51872b70645a75f43dc0b849f77621"
CRB=f"https://huggingface.co/datasets/cxyzhang/caseReportBench_ClinicalDenseExtraction_Benchmark/resolve/{CRB_REV}/data/train-00000-of-00001.parquet?download=true"
MC_REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
HF=f"https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/{MC_REV}/data"
OUT=pathlib.Path("e060a");OUT.mkdir(exist_ok=True)

CATS=["Vitals_Hema","GI","History","Neuro","Lab_Image","CVS","ENDO","GU","RESP","MSK","EENT","DERM","Pregnancy","LYMPH"]
GENERIC={"disease","syndrome","disorder","type","deficiency","defect","with","without","of","and","the","primary","secondary","congenital"}

def norm(s):
 s=unicodedata.normalize("NFKC",str(s)).casefold()
 return " ".join(re.findall(r"\w+",s))
def words(s): return len(re.findall(r"\b\w+\b",str(s)))
def nonempty(v):
 if pd.isna(v):return False
 s=str(v).strip()
 return bool(s and s.lower() not in {"nan","none","[]","{}","null","n/a","na"})
def mask_diag(text,diag):
 x=str(text);d=str(diag).strip()
 if d:x=re.sub(re.escape(d)," [DIAGNOSIS_MASKED] ",x,flags=re.I)
 toks={t for t in norm(diag).split() if len(t)>=4 and t not in GENERIC}
 out=[]
 for token in re.split(r'(\W+)',x):
  nt=norm(token)
  if nt and nt in toks:out.append("[DXTOKEN]")
  else:out.append(token)
 return "".join(out)

df=pd.read_parquet(CRB)
assert len(df)==138
assert set(CATS).issubset(df.columns)
assert "pmcid" in df.columns and "text" in df.columns and "Confirmed_Diagnosis(IEM)" in df.columns

# MedCase PMCID overlap
splits={}
for split in ("train","val","test"):
 x=pd.read_parquet(f"{HF}/{split}-00000-of-00001.parquet",columns=["pmcid"])
 splits[split]=set(x.pmcid.astype(str).str.upper())
all_mc=set().union(*splits.values())

rows=[]
for i,r in df.iterrows():
 pm=str(r.pmcid).upper()
 text=str(r["text"]);dx=str(r["Confirmed_Diagnosis(IEM)"])
 nt=norm(text);nd=norm(dx)
 exact=bool(nd and nd in nt)
 # character location based on case-folded raw exact substring
 low=text.casefold();dl=dx.strip().casefold();pos=low.find(dl) if dl else -1
 frac=(pos/len(text)) if pos>=0 and len(text) else None
 prefix=text[:pos] if pos>=0 else text
 masked=mask_diag(text,dx)
 toks={t for t in nd.split() if len(t)>=4 and t not in GENERIC}
 residual=sum(1 for t in toks if re.search(r'\b'+re.escape(t)+r'\b',norm(masked))) if toks else 0
 rows.append({
  "row_index":int(i),"pmcid":pm,"diagnosis":dx,"text_words":words(text),
  "diagnosis_exact_in_text":exact,"first_exact_dx_char_fraction":frac,
  "prefix_before_dx_words":words(prefix),"masked_text_words":words(masked),
  "diagnosis_specific_token_count":len(toks),"residual_specific_dx_tokens_after_mask":residual,
  "overlap_train":pm in splits["train"],"overlap_val":pm in splits["val"],"overlap_test":pm in splits["test"],
  "source_disjoint":pm not in all_mc,
  **{f"has_{c}":nonempty(r[c]) for c in CATS}
 })
audit=pd.DataFrame(rows)
audit.to_csv(OUT/"E060A_CASE_AUDIT.csv",index=False)

cat_counts={c:int(audit[f"has_{c}"].sum()) for c in CATS}
cat_per_case=audit[[f"has_{c}" for c in CATS]].sum(axis=1)

# Save small raw examples for schema audit only.
examples=[]
for i in range(min(5,len(df))):
 r=df.iloc[i]
 examples.append({"pmcid":str(r.pmcid),"diagnosis":str(r["Confirmed_Diagnosis(IEM)"]),"text":str(r.text)[:3000],
                  "categories":{c:str(r[c])[:1200] for c in CATS if nonempty(r[c])}})
(OUT/"E060A_SCHEMA_EXAMPLES.json").write_text(json.dumps(examples,ensure_ascii=False,indent=2)+"\n")

summary={
 "status":"compatibility_audit_only_no_C2_result",
 "dataset_revision":CRB_REV,"medcasereasoning_revision":MC_REV,
 "n_cases":len(df),"unique_pmcids":int(df.pmcid.astype(str).nunique()),
 "source_overlap":{"train":int(audit.overlap_train.sum()),"validation":int(audit.overlap_val.sum()),"test":int(audit.overlap_test.sum()),
                   "any":int((~audit.source_disjoint).sum()),"source_disjoint":int(audit.source_disjoint.sum())},
 "diagnosis_leakage":{
   "exact_diagnosis_in_raw_text":int(audit.diagnosis_exact_in_text.sum()),
   "rate":float(audit.diagnosis_exact_in_text.mean()),
   "first_exact_dx_position_fraction_median":float(audit.first_exact_dx_char_fraction.dropna().median()) if audit.first_exact_dx_char_fraction.notna().any() else None,
   "prefix_before_first_exact_dx_ge100_words":int((audit.prefix_before_dx_words>=100).sum()),
   "prefix_before_first_exact_dx_ge200_words":int((audit.prefix_before_dx_words>=200).sum()),
   "prefix_before_first_exact_dx_word_median":float(audit.prefix_before_dx_words.median()),
   "cases_with_residual_specific_dx_tokens_after_aggressive_mask":int((audit.residual_specific_dx_tokens_after_mask>0).sum())
 },
 "text_length_words":{"median":float(audit.text_words.median()),"p25":float(audit.text_words.quantile(.25)),"p75":float(audit.text_words.quantile(.75)),
                      "min":int(audit.text_words.min()),"max":int(audit.text_words.max())},
 "expert_annotation":{"categories":CATS,"nonempty_by_category":cat_counts,
                      "nonempty_categories_per_case_median":float(cat_per_case.median()),
                      "nonempty_categories_per_case_p25":float(cat_per_case.quantile(.25)),
                      "nonempty_categories_per_case_p75":float(cat_per_case.quantile(.75))},
 "candidate_query_definitions_for_future_audit":[
   "raw text (only if diagnosis leakage is negligible)",
   "text prefix before first exact diagnosis mention",
   "aggressively diagnosis-masked raw text"
 ],
 "limits":[
   "Prefix before exact diagnosis mention is a mechanical leakage-control heuristic and is not guaranteed to correspond to the clinical presentation section.",
   "Exact/token masking cannot remove indirect diagnostic clues.",
   "Expert category fields describe dense clinical facts, not direct judgments of cross-case diagnostic utility."
 ]}
(OUT/"E060A_COMPATIBILITY_SUMMARY.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(summary,ensure_ascii=False,indent=2))
