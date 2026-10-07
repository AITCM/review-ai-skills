from __future__ import annotations
import json,re,unicodedata,pathlib,hashlib
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

MC_REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
HF=f"https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/{MC_REV}/data"
OUT=pathlib.Path("e081-n8-medr");OUT.mkdir(exist_ok=True)
RRF_C=60.0
PAT=re.compile(r'^\s*(?:\((\d+)\)|(\d+)[.)])\s*',re.M)
GENERIC={"with","without","acute","chronic","syndrome","disease","disorder"}

def nw(s):
    return " ".join(re.findall(r"\w+",unicodedata.normalize("NFKC",str(s)).casefold()))
def split_reason(s):
    s=str(s);ms=list(PAT.finditer(s))
    if not ms or s[:ms[0].start()].strip():return [s.strip()]
    nums=[int(m.group(1) or m.group(2)) for m in ms]
    if nums!=list(range(1,len(ms)+1)):return [s.strip()]
    out=[]
    for i,m in enumerate(ms):
        e=ms[i+1].start() if i+1<len(ms) else len(s);x=s[m.end():e].strip()
        if x:out.append(x)
    return out or [s.strip()]
def mask_tokens(reason,diag):
    dd=nw(diag);toks=[t for t in dd.split() if len(t)>=4 and t not in GENERIC]
    out=[]
    for p in split_reason(reason):
        x=nw(p)
        if dd and dd in x:x=x.replace(dd," diagnosismask ")
        for t in sorted(set(toks),key=len,reverse=True):
            x=re.sub(r"\b"+re.escape(t)+r"\b"," diagnosistokenmask ",x)
        out.append(" ".join(x.split()))
    return out
def rank(scores):
    ids=np.arange(len(scores),dtype=int)
    return np.lexsort((ids,-np.asarray(scores)))
def rrf(a,b):
    n=len(a);score=np.zeros(n,dtype=np.float64)
    for o in (a,b):
        pos=np.empty(n,dtype=np.int32);pos[o]=np.arange(n,dtype=np.int32)
        score += 1.0/(RRF_C+pos+1.0)
    return rank(score)

queries=[json.loads(x) for x in open(OUT/"E081_QUERY_ONLY.jsonl",encoding="utf-8") if x.strip()]
assert len(queries)==840
assert all(set(x)=={"query_index","pmcid","case_summary"} for x in queries)
assert [x["query_index"] for x in queries]==list(range(840))
ext=pd.DataFrame(queries)

tr=pd.read_parquet(HF+"/train-00000-of-00001.parquet",
                   columns=["case_prompt","diagnostic_reasoning","final_diagnosis"])
assert len(tr)==13092
reason_pts=[mask_tokens(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
reason_docs=[" ".join(x) for x in reason_pts]

# Route 1: ordinary case-to-case lexical retrieval.
cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,
                   min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
X=cv.fit_transform(tr.case_prompt.astype(str));Q=cv.transform(ext.case_summary.astype(str))
LEX=(Q@X.T).toarray().astype(np.float32)

# Route 2: direct case -> historical diagnosis-masked reasoning in a shared sparse vocabulary.
xv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,
                   min_df=2,max_df=.98,max_features=140000,dtype=np.float32)
xv.fit(tr.case_prompt.astype(str).tolist()+reason_docs)
QX=xv.transform(ext.case_summary.astype(str));RX=xv.transform(reason_docs)
XMOD=(QX@RX.T).toarray().astype(np.float32)

with (OUT/"E081_BLIND_RANKINGS.jsonl").open("w",encoding="utf-8") as f:
    for i in range(len(ext)):
        lex=rank(LEX[i]);xm=rank(XMOD[i]);method=rrf(lex,xm)
        f.write(json.dumps({
          "query_index":i,"pmcid":ext.iloc[i].pmcid,
          "baseline_top50":lex[:50].tolist(),
          "xmod_top50":xm[:50].tolist(),
          "method_top50":method[:50].tolist()
        })+"\n")

config={
 "method":"N8-RRF-LEX-XMOD-STRONGMASK",
 "rrf_constant":RRF_C,
 "lexical":{"ngram_range":[1,2],"min_df":2,"max_df":.98,"max_features":100000},
 "xmod":{"ngram_range":[1,2],"min_df":2,"max_df":.98,"max_features":140000},
 "historical_reasoning_mask":"exact normalized final-diagnosis phrase plus diagnosis tokens length>=4 excluding generic clinical words",
 "new_query_input":["pmcid","case_summary"],
 "historical_index_fields":["case_prompt","diagnosis-masked diagnostic_reasoning"],
 "primary_endpoint":"Top-10 reasoning-set Soft-F1",
 "secondary_endpoints":["Top-1 reasoning-set Soft-F1","Top-3 reasoning-set Soft-F1"]
}
lock={
 "status":"E081_external_rankings_locked_before_outcomes_loaded",
 "n_external":len(ext),
 "query_lock_sha256":hashlib.sha256((OUT/"E081_QUERY_ONLY.jsonl").read_bytes()).hexdigest(),
 "ranking_fields_used":["pmcid","case_summary"],
 "external_outcome_fields_available_to_ranker":[],
 "method_config":config,
 "method_config_sha256":hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
}
(OUT/"E081_RANKING_LOCK.json").write_text(json.dumps(lock,indent=2)+"\n")
print(json.dumps(lock,indent=2))
