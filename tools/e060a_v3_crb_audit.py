from __future__ import annotations
import json,re,unicodedata,pathlib,ast
from collections import Counter
import numpy as np,pandas as pd

CRB_REV="3297c5b2db51872b70645a75f43dc0b849f77621"
CRB=f"https://huggingface.co/datasets/cxyzhang/caseReportBench_ClinicalDenseExtraction_Benchmark/resolve/{CRB_REV}/data/train-00000-of-00001.parquet?download=true"
MC_REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
HF=f"https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/{MC_REV}/data"
OUT=pathlib.Path("e060a-v3");OUT.mkdir(exist_ok=True)

CATS=["Vitals_Hema","GI","History","Neuro","Lab_Image","CVS","ENDO","GU","RESP","MSK","EENT","DERM","Pregnancy","LYMPH"]
GENERIC={"disease","syndrome","disorder","type","deficiency","defect","with","without","of","and","the","primary","secondary","congenital"}

def norm(s):
 s=unicodedata.normalize("NFKC",str(s)).casefold()
 return " ".join(re.findall(r"\w+",s))
def flatten(v):
 if v is None:return ""
 if isinstance(v,np.ndarray):v=v.tolist()
 if isinstance(v,(list,tuple,set)):
  return " ".join(flatten(x) for x in v if flatten(x))
 if isinstance(v,dict):
  return " ".join(flatten(x) for x in v.values() if flatten(x))
 try:
  if bool(pd.isna(v)):return ""
 except Exception:pass
 return str(v).strip()
def as_items(v):
 if v is None:return []
 if isinstance(v,np.ndarray):v=v.tolist()
 if isinstance(v,(list,tuple,set)):
  return [flatten(x) for x in v if flatten(x)]
 if isinstance(v,dict):
  return [flatten(x) for x in v.values() if flatten(x)]
 try:
  if bool(pd.isna(v)):return []
 except Exception:pass
 s=str(v).strip()
 return [s] if s and s.lower() not in {"nan","none","[]","{}","null","n/a","na"} else []
def words(s):return len(re.findall(r"\b\w+\b",str(s)))
def find_earliest_dx(text,dx_items):
 low=unicodedata.normalize("NFKC",text).casefold()
 hits=[]
 for d in dx_items:
  d0=unicodedata.normalize("NFKC",d).casefold().strip()
  if d0:
   p=low.find(d0)
   if p>=0:hits.append((p,d))
 return min(hits,key=lambda x:x[0]) if hits else (None,None)
def aggressive_mask(text,dx_items):
 x=text
 # exact phrases first
 for d in sorted(dx_items,key=len,reverse=True):
  if d.strip():x=re.sub(re.escape(d.strip())," [DIAGNOSIS_MASKED] ",x,flags=re.I)
 toks=set()
 for d in dx_items:
  toks|={t for t in norm(d).split() if len(t)>=4 and t not in GENERIC}
 parts=re.split(r'(\W+)',x)
 y=[]
 for token in parts:
  nt=norm(token)
  y.append("[DXTOKEN]" if nt and nt in toks else token)
 return "".join(y),toks

df=pd.read_parquet(CRB)
assert len(df)==138 and df.pmcid.astype(str).nunique()==138
splits={}
for split in ("train","val","test"):
 x=pd.read_parquet(f"{HF}/{split}-00000-of-00001.parquet",columns=["pmcid"])
 splits[split]=set(x.pmcid.astype(str).str.upper())
all_mc=set().union(*splits.values())

rows=[];schema=[]
for i,r in df.iterrows():
 pm=str(r.pmcid).upper()
 text=flatten(r["text"])
 dx_items=as_items(r["Confirmed_Diagnosis(IEM)"])
 earliest,which=find_earliest_dx(text,dx_items)
 prefix=text[:earliest] if earliest is not None else text
 masked,toks=aggressive_mask(text,dx_items)
 residual=sum(bool(re.search(r'\b'+re.escape(t)+r'\b',norm(masked))) for t in toks)
 rec={
  "row_index":int(i),"pmcid":pm,"diagnosis_items":len(dx_items),
  "diagnosis":" | ".join(dx_items),"has_confirmed_diagnosis":bool(dx_items),
  "text_words":words(text),
  "any_exact_dx_phrase_in_text":earliest is not None,
  "first_exact_dx_char_fraction":float(earliest/len(text)) if earliest is not None and len(text) else None,
  "prefix_before_first_exact_dx_words":words(prefix),
  "masked_text_words":words(masked),
  "diagnosis_specific_token_count":len(toks),
  "residual_specific_dx_tokens_after_aggressive_mask":int(residual),
  "overlap_train":pm in splits["train"],"overlap_val":pm in splits["val"],"overlap_test":pm in splits["test"],
  "source_disjoint":pm not in all_mc,
 }
 for c in CATS:
  rec[f"has_{c}"]=bool(as_items(r[c]))
  rec[f"n_{c}_items"]=len(as_items(r[c]))
 rows.append(rec)
 if len(schema)<5:
  schema.append({"pmcid":pm,"diagnoses":dx_items,"text_preview":text[:2000],
    "expert_categories":{c:as_items(r[c])[:10] for c in CATS if as_items(r[c])}})

audit=pd.DataFrame(rows);audit.to_csv(OUT/"E060A_V3_CASE_AUDIT.csv",index=False)
(OUT/"E060A_V3_SCHEMA_EXAMPLES.json").write_text(json.dumps(schema,ensure_ascii=False,indent=2)+"\n")
hasdx=audit[audit.has_confirmed_diagnosis]
pos=audit.first_exact_dx_char_fraction.dropna()
cat_counts={c:int(audit[f"has_{c}"].sum()) for c in CATS}
cat_items={c:int(audit[f"n_{c}_items"].sum()) for c in CATS}
cat_per=audit[[f"has_{c}" for c in CATS]].sum(axis=1)
summary={
 "status":"corrected_nested_schema_compatibility_audit_no_C2_result",
 "dataset_revision":CRB_REV,"medcasereasoning_revision":MC_REV,
 "n_cases":len(df),"source_overlap":{"train":int(audit.overlap_train.sum()),"validation":int(audit.overlap_val.sum()),
  "test":int(audit.overlap_test.sum()),"any":int((~audit.source_disjoint).sum()),"source_disjoint":int(audit.source_disjoint.sum())},
 "diagnosis_schema":{"cases_with_confirmed_diagnosis":int(audit.has_confirmed_diagnosis.sum()),
  "cases_without_confirmed_diagnosis":int((~audit.has_confirmed_diagnosis).sum()),
  "diagnosis_item_count_distribution":dict(Counter(audit.diagnosis_items.astype(int)))},
 "diagnosis_leakage":{
  "cases_with_exact_confirmed_dx_phrase_in_text":int(audit.any_exact_dx_phrase_in_text.sum()),
  "rate_among_all_cases":float(audit.any_exact_dx_phrase_in_text.mean()),
  "rate_among_cases_with_confirmed_dx":float(audit[ audit.has_confirmed_diagnosis ].any_exact_dx_phrase_in_text.mean()) if len(hasdx) else None,
  "first_exact_dx_position_fraction_median":float(pos.median()) if len(pos) else None,
  "first_exact_dx_position_fraction_p25":float(pos.quantile(.25)) if len(pos) else None,
  "first_exact_dx_position_fraction_p75":float(pos.quantile(.75)) if len(pos) else None,
  "prefix_before_first_exact_dx_words_median":float(audit.prefix_before_first_exact_dx_words.median()),
  "prefix_ge100_words":int((audit.prefix_before_first_exact_dx_words>=100).sum()),
  "prefix_ge200_words":int((audit.prefix_before_first_exact_dx_words>=200).sum()),
  "cases_with_residual_specific_dx_tokens_after_aggressive_mask":int((audit.residual_specific_dx_tokens_after_aggressive_mask>0).sum())
 },
 "text_length_words":{"median":float(audit.text_words.median()),"p25":float(audit.text_words.quantile(.25)),
  "p75":float(audit.text_words.quantile(.75)),"min":int(audit.text_words.min()),"max":int(audit.text_words.max())},
 "expert_annotation":{"categories":CATS,"nonempty_cases_by_category":cat_counts,"total_items_by_category":cat_items,
  "nonempty_categories_per_case_median":float(cat_per.median()),"p25":float(cat_per.quantile(.25)),"p75":float(cat_per.quantile(.75))},
 "query_definition_assessment":{
  "raw_text":"unsafe if confirmed diagnosis phrase appears in a large fraction of cases",
  "prefix_before_first_exact_diagnosis":"candidate only if enough pre-diagnosis text remains; cases without a diagnosis phrase use full text",
  "aggressive_diagnosis_mask":"candidate query; removes exact diagnosis phrases and diagnosis-specific tokens but cannot remove indirect clues"
 },
 "limits":["Dense category annotations are structured clinical facts, not direct pairwise case-utility labels.",
           "Mechanical diagnosis masking cannot remove all indirect diagnostic clues.",
           "Prefix truncation can remove post-diagnostic but clinically informative evidence and changes text length across cases."]}
(OUT/"E060A_V3_COMPATIBILITY_SUMMARY.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(summary,ensure_ascii=False,indent=2))
