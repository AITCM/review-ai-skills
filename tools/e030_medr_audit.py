from __future__ import annotations
import json, pathlib, re, urllib.request, hashlib
import pandas as pd

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
HF="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
MEDR="https://raw.githubusercontent.com/MAGIC-AI4Med/MedRBench/main/data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json"
OUT=pathlib.Path("external-medr-audit");OUT.mkdir(exist_ok=True)

# Load MedCaseReasoning IDs / prompts / diagnoses only
splits={}
for sp in ("train","val","test"):
    df=pd.read_parquet(f"{HF}/{sp}-00000-of-00001.parquet",columns=["pmcid","case_prompt","final_diagnosis"])
    splits[sp]=df
all_pm=set().union(*[set(x.pmcid.astype(str)) for x in splits.values()])
all_prompt=set(" ".join(str(x).casefold().split()) for df in splits.values() for x in df.case_prompt)

with urllib.request.urlopen(MEDR,timeout=120) as r:
    data=json.load(r)

def walk_keys(obj,prefix="",depth=0,acc=None):
    if acc is None:acc=set()
    if depth>4:return acc
    if isinstance(obj,dict):
        for k,v in obj.items():
            p=f"{prefix}.{k}" if prefix else str(k);acc.add(p);walk_keys(v,p,depth+1,acc)
    elif isinstance(obj,list) and obj:
        walk_keys(obj[0],prefix+"[]",depth+1,acc)
    return acc

def find_values(obj, key_fragments, path=""):
    out=[]
    if isinstance(obj,dict):
        for k,v in obj.items():
            p=f"{path}.{k}" if path else k
            lk=k.lower()
            if any(f in lk for f in key_fragments):out.append((p,v))
            out.extend(find_values(v,key_fragments,p))
    elif isinstance(obj,list):
        for i,v in enumerate(obj[:5]):out.extend(find_values(v,key_fragments,path+"[]"))
    return out

# normalize top-level records
if isinstance(data,dict):
    items=list(data.items())
    records=[{"__root_key__":k,"__record__":v} for k,v in items]
elif isinstance(data,list):
    records=[{"__root_key__":str(i),"__record__":v} for i,v in enumerate(data)]
else:
    raise TypeError(type(data))

schema=sorted(walk_keys(data))
first=records[0]["__record__"]
sample_matches={
 "pmcid_like":find_values(first,["pmcid","pmc_id","pmc"]),
 "diagnosis_like":find_values(first,["diagnos"]),
 "reasoning_like":find_values(first,["reason","rationale","logic"]),
 "case_like":find_values(first,["case","patient","history","present","symptom"]),
}

pmc_candidates=[];prompt_candidates=[];diag_candidates=[];reason_candidates=[]
for recx in records:
    root=recx["__root_key__"];rec=recx["__record__"]
    # collect scalar values from keys
    vals=find_values(rec,["pmcid","pmc_id"])
    pm=None
    for p,v in vals:
        if isinstance(v,(str,int)):
            m=re.search(r"PMC\d+",str(v),re.I)
            if m: pm=m.group(0).upper();break
    if pm is None:
        m=re.search(r"PMC\d+",root,re.I)
        if m:pm=m.group(0).upper()
    pmc_candidates.append(pm)
    # collect likely reasoning / diagnosis / case text for availability
    dm=find_values(rec,["diagnos"])
    rm=find_values(rec,["reason","rationale"])
    cm=find_values(rec,["patient","case_info","basic_info","history","presentation","clinical"])
    diag_candidates.append(any(isinstance(v,(str,list,dict)) and bool(v) for _,v in dm))
    reason_candidates.append(any(isinstance(v,(str,list,dict)) and bool(v) for _,v in rm))
    # conservative prompt-ish text
    texts=[]
    for _,v in cm:
        if isinstance(v,str) and len(v)>80:texts.append(v)
    prompt_candidates.append(max(texts,key=len) if texts else None)

medr_pm={x for x in pmc_candidates if x}
overlaps={sp:len(medr_pm & set(df.pmcid.astype(str).str.upper())) for sp,df in splits.items()}
overlap_union=len(medr_pm & {x.upper() for x in all_pm})
unique_external=medr_pm-{x.upper() for x in all_pm}

# prompt exact overlaps where a clear prompt candidate could be found
prompt_exact=0; prompt_available=0
for p in prompt_candidates:
    if p:
        prompt_available+=1
        if " ".join(p.casefold().split()) in all_prompt:prompt_exact+=1

summary={
 "medr_source_url":MEDR,
 "medr_record_count":len(records),
 "top_level_type":type(data).__name__,
 "schema_paths_first_levels":schema[:300],
 "sample_field_matches":{k:[(p, type(v).__name__, (str(v)[:300] if not isinstance(v,(dict,list)) else str(v)[:300])) for p,v in vals[:20]] for k,vals in sample_matches.items()},
 "pmcid_extraction":{"non_null":sum(x is not None for x in pmc_candidates),"unique":len(medr_pm),
                    "overlap_train":overlaps["train"],"overlap_val":overlaps["val"],"overlap_test":overlaps["test"],
                    "overlap_any_medcase":overlap_union,"unique_vs_all_medcase":len(unique_external)},
 "field_availability_proxy":{"diagnosis_like_records":sum(diag_candidates),"reasoning_like_records":sum(reason_candidates),
                             "clear_prompt_candidate_records":prompt_available,"exact_prompt_overlap_with_medcase":prompt_exact},
 "note":[
   "Field-availability counts are schema proxies and require semantic field mapping before use.",
   "PMCID overlap is the primary source-overlap audit; records without extractable PMCID need separate review.",
   "A MedR-Bench external validation should exclude any query whose PMCID appeared in any MedCaseReasoning split."
 ]
}
(OUT/"MEDRBENCH_COMPATIBILITY_AND_OVERLAP.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
print(json.dumps(summary,indent=2,ensure_ascii=False))
