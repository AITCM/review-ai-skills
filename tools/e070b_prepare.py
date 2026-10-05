from __future__ import annotations
import ast,csv,hashlib,io,json,pathlib,zipfile

OUT=pathlib.Path("e070b");OUT.mkdir(exist_ok=True)
N_SAMPLE=10000
SALT="E070B-DDXPLUS-CROSSDOMAIN-v1"

root=pathlib.Path("ddx")
evid=json.load(open(root/"release_evidences.json"))
assert len(evid)==223

def parse_evidences(s):
    try:return list(ast.literal_eval(str(s)))
    except Exception:return []

def evidence_to_text(code):
    code=str(code)
    if "_@_" in code:
        base,val=code.split("_@_",1)
    else:
        base,val=code,None
    meta=evid.get(base)
    if meta is None:
        return f"Clinical evidence {base}: {val if val is not None else 'present'}."
    q=str(meta.get("question_en") or base).strip()
    dtype=str(meta.get("data_type",""))
    if val is None:
        return f"{q} Answer: yes."
    vm=meta.get("value_meaning") or {}
    meaning=None
    if val in vm:
        z=vm[val]
        if isinstance(z,dict):meaning=z.get("en") or z.get("fr")
        else:meaning=str(z)
    value=str(meaning if meaning is not None else val)
    if dtype=="B" and value in {"1","True","true"}: value="yes"
    return f"{q} Answer: {value}."

def serialize(age,sex,codes):
    parts=[f"Age: {age} years.",f"Sex: {'female' if str(sex).upper()=='F' else 'male' if str(sex).upper()=='M' else str(sex)}."]
    seen=set()
    for code in codes:
        if code in seen:continue
        seen.add(code);parts.append(evidence_to_text(code))
    return " ".join(parts)

# Deterministic selection depends only on source row index, never outcomes.
with zipfile.ZipFile(root/"release_test_patients.zip") as z:
    fn=z.namelist()[0]
    with z.open(fn) as raw:
        reader=csv.DictReader(io.TextIOWrapper(raw,encoding="utf-8"))
        heap=[]
        import heapq
        for idx,row in enumerate(reader):
            h=int(hashlib.sha256(f"{SALT}:{idx}".encode()).hexdigest()[:16],16)
            item=(-h,idx,row["AGE"],row["SEX"],row["EVIDENCES"],row["INITIAL_EVIDENCE"])
            if len(heap)<N_SAMPLE: heapq.heappush(heap,item)
            elif item>heap[0]: heapq.heapreplace(heap,item)
selected=sorted([(-x[0],x[1],x[2],x[3],x[4],x[5]) for x in heap],key=lambda x:x[1])
assert len(selected)==N_SAMPLE

rows=[];word_counts=[];evidence_counts=[]
for sample_i,(_,src_i,age,sex,evs,initial) in enumerate(selected):
    codes=parse_evidences(evs)
    text=serialize(age,sex,codes)
    rows.append({"query_index":sample_i,"source_row_index":src_i,"query_text":text})
    word_counts.append(len(text.split()));evidence_counts.append(len(codes))

with (OUT/"E070B_QUERY_ONLY.jsonl").open("w",encoding="utf-8") as f:
    for x in rows:f.write(json.dumps(x,ensure_ascii=False)+"\n")
spec={
 "status":"ddxplus_query_only_sample_locked",
 "sample_size":N_SAMPLE,"selection":"lowest SHA256-derived ranks over source row_index with fixed salt",
 "selection_salt":SALT,"selection_uses_outcomes":False,
 "source_split":"release_test_patients",
 "query_fields":["query_index","source_row_index","query_text"],
 "outcome_fields_retained":[],
 "serialization":{
   "age":"Age: <AGE> years.","sex":"Sex: female/male.",
   "evidence":"Each explicitly observed evidence is rendered as '<question_en> Answer: <English value meaning or observed value>.'",
   "negative_defaults":"Not added unless explicitly present in EVIDENCES.",
   "outcomes_excluded":["PATHOLOGY","DIFFERENTIAL_DIAGNOSIS"]},
 "query_sha256":hashlib.sha256((OUT/"E070B_QUERY_ONLY.jsonl").read_bytes()).hexdigest(),
 "query_words":{"median":sorted(word_counts)[len(word_counts)//2],"min":min(word_counts),"max":max(word_counts)},
 "evidence_count":{"median":sorted(evidence_counts)[len(evidence_counts)//2],"min":min(evidence_counts),"max":max(evidence_counts)}
}
(OUT/"E070B_QUERY_LOCK.json").write_text(json.dumps(spec,indent=2)+"\n")
print(json.dumps(spec,indent=2))
