from __future__ import annotations
import csv,hashlib,html,json,pathlib,re,unicodedata,zipfile
import pandas as pd

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
SOURCE_ZIP=pathlib.Path("E081_SOURCE_LOCKED.zip")
ROOT=pathlib.Path("e082-clinical-review")
PUBLIC=ROOT/"reviewer_only";PRIVATE=ROOT/"analyst_private"
PUBLIC.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
N=120;SEED="E082-20261009";TOP=3
GENERIC={"with","without","acute","chronic","syndrome","disease","disorder"}
def digest(data):return hashlib.sha256(data).hexdigest()
def normalize(s):
    return " ".join(re.findall(r"\w+",unicodedata.normalize("NFKC",str(s)).casefold()))
def mask_human(text,diagnosis):
    s=str(text);d=normalize(diagnosis)
    if d:s=re.sub(re.escape(d),"[DIAGNOSIS REDACTED]",s,flags=re.I)
    for w in sorted({x for x in d.split() if len(x)>=4 and x not in GENERIC},key=lambda x:(-len(x),x)):
        s=re.sub(r"\b"+re.escape(w)+r"\b","[DIAGNOSIS TOKEN REDACTED]",s,flags=re.I)
    return s
def writecsv(path,rows,cols):
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader();writer.writerows(rows)
with zipfile.ZipFile(SOURCE_ZIP) as z:
    prefix="e081-n8-medr/"
    qbytes=z.read(prefix+"E081_QUERY_ONLY.jsonl")
    rbytes=z.read(prefix+"E081_BLIND_RANKINGS.jsonl")
    qlock=json.loads(z.read(prefix+"E081_QUERY_LOCK.json"))
    lock=json.loads(z.read(prefix+"E081_RANKING_LOCK.json"))
assert lock["status"]=="E081_external_rankings_locked_before_outcomes_loaded"
assert qlock["status"]=="query_only_external_cohort_locked"
assert digest(qbytes)==lock["query_lock_sha256"]
assert lock["method_config"]["method"]=="N8-RRF-LEX-XMOD-STRONGMASK"
Q=[json.loads(x) for x in qbytes.decode().splitlines()]
R=[json.loads(x) for x in rbytes.decode().splitlines()]
assert len(Q)==len(R)==840
assert all(q["query_index"]==i==r["query_index"] and q["pmcid"]==r["pmcid"] for i,(q,r) in enumerate(zip(Q,R)))
assert all(set(q)=={"query_index","pmcid","case_summary"} for q in Q)
tr=pd.read_parquet(
 "https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data/train-00000-of-00001.parquet",
 columns=["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"])
assert len(tr)==13092
assert set(tr.pmcid.astype(str)).isdisjoint({q["pmcid"] for q in Q})
sample=sorted(range(840),key=lambda i:digest((SEED+":sample:"+Q[i]["pmcid"]).encode()))[:N]
assign=sorted(sample,key=lambda i:digest((SEED+":arm:"+Q[i]["pmcid"]).encode()))
n8_on_A=set(assign[:N//2])
assert len(sample)==len(set(sample))==120
public_rows=[];private_rows=[];overlap=[]
for i in sample:
    b=R[i]["baseline_top50"][:TOP];n=R[i]["method_top50"][:TOP]
    assert len(set(b))==len(set(n))==TOP
    assert all(isinstance(j,int) and 0<=j<len(tr) for j in b+n)
    A,B=(n,b) if i in n8_on_A else (b,n)
    cid="EV-"+digest((SEED+":cid:"+Q[i]["pmcid"]).encode())[:12].upper()
    def historical(ids):
        return [{"case":mask_human(tr.case_prompt.iloc[j],tr.final_diagnosis.iloc[j]),
                 "reasoning":mask_human(tr.diagnostic_reasoning.iloc[j],tr.final_diagnosis.iloc[j])}
                for j in ids]
    public_rows.append({"case_id":cid,"new_case":Q[i]["case_summary"],
                        "panel_A":historical(A),"panel_B":historical(B)})
    common=len(set(b)&set(n));overlap.append(common)
    private_rows.append({"case_id":cid,"query_index":i,"query_pmcid":Q[i]["pmcid"],
       "arm_A":"N8_RRF" if i in n8_on_A else "TFIDF",
       "arm_B":"TFIDF" if i in n8_on_A else "N8_RRF",
       "A_train_indices":json.dumps(A),"B_train_indices":json.dumps(B),
       "common_top3":common,"query_sha256":digest(Q[i]["case_summary"].encode())})
with (PUBLIC/"E082_REVIEWER_CASES.jsonl").open("w",encoding="utf-8") as f:
    for x in public_rows:f.write(json.dumps(x,ensure_ascii=False)+"\n")
writecsv(PRIVATE/"E082_ALLOCATION_KEY.csv",private_rows,list(private_rows[0]))
rating_cols=["rater_id","case_id","A_relevance_1to5","B_relevance_1to5",
 "A_differential_helpfulness_1to5","B_differential_helpfulness_1to5",
 "A_nonredundancy_1to5","B_nonredundancy_1to5",
 "A_misleading_risk_1to5","B_misleading_risk_1to5",
 "preferred_panel_A_B_TIE","preference_confidence_1to5","comments","review_status"]
writecsv(PUBLIC/"E082_RATING_TEMPLATE.csv",
         [{"case_id":x["case_id"],"review_status":"not_scored"} for x in public_rows],rating_cols)
protocol="""# E082 — Clinician-blinded appraisal of retrieved case evidence

**Purpose:** Compare clinical diagnostic-reasoning usefulness of frozen N8-RRF and TF-IDF retrieved evidence. This is NOT a diagnostic accuracy experiment or a medical decision system.

**Sampling and concealment:** From 840 externally locked MedR-Bench cases, 120 were sampled by pre-specified SHA-256 order of query PMCID, independent of diagnostic outcomes or automated retrieval scores. Panel A is N8 in exactly 60 cases and TF-IDF in exactly 60; panel B is the complement. Each panel contains the frozen Top-3 historical cases, in rank order. Source PMCID, method identity, reference diagnosis, oracle results and performance numbers are not in the reviewer material. Do not distribute the analyst allocation key.

**Rater task:** Review the new published case and the two 3-case evidence panels. Independently score A and B, 1 (low) to 5 (high), on (a) clinical relevance, (b) differential diagnosis helpfulness, (c) nonredundancy, and (d) risk of misleading reasoning (5 indicates greater risk). Choose overall A, B or TIE, and record confidence 1–5. A TIE is legitimate. Please record suspected unmasked diagnostic labels or misleading analogies. At least two independent clinician raters are recommended.

**Locked primary endpoint:** clinician panel preference, decoded after all ratings have been signed/locked. Report A/B/TIE counts, N8/TF-IDF preference after decoding, per-query paired preference proportions and query-clustered 95% bootstrap intervals. A two-sided sign test among non-ties may be shown; repeated raters for a case are not independent observations. Secondary outcomes include paired differences in rating dimensions and inter-rater agreement.

**Important boundaries:** These human panels use Top-3, while external numeric primary Soft-F1 was Top-10. They test expert-perceived evidence usefulness, not the published numeric metric and not diagnostic accuracy. Literal diagnosis phrase/token masking is automatic and cannot guarantee synonym removal. Published-case contexts may contain distinctive diagnostic clues. Comply with local research oversight before recruiting clinicians; do not use non-public identifiable records.
"""
(PUBLIC/"E082_STUDY_PROTOCOL.md").write_text(protocol,encoding="utf-8")
css="""body{font-family:system-ui,Arial,sans-serif;background:#f4f7fb;color:#213246;line-height:1.5;margin:0}header{background:#14314d;color:white;padding:25px 5vw}main{max-width:1140px;margin:auto;padding:20px}article{background:white;padding:20px;border-radius:12px;border:1px solid #dce4ee;margin-bottom:22px}h2{margin-top:0}.question,.historical{white-space:pre-wrap;padding:12px;border-radius:7px}.question{background:#eaf1f9;font-size:14px}.panels{display:grid;grid-template-columns:1fr 1fr;gap:15px}.panel{padding:14px;border:1px solid #dae3ed;background:#f9fbfe;border-radius:10px}details{background:white;margin-top:9px;padding:9px;border-radius:6px}summary{cursor:pointer;font-weight:600}.historical{font-size:13px}@media(max-width:760px){.panels{grid-template-columns:1fr}}"""
parts=['<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E082 Blinded Clinical Evidence Review</title><style>'+css+'</style></head><body><header><h1>E082 · Clinical Evidence Review</h1><p>120 independently sampled published cases · Two blinded evidence panels · Top-3 per panel</p><p>Research only. Submit scores using the accompanying CSV.</p></header><main>']
for row in public_rows:
    parts.append('<article><h2>'+html.escape(row["case_id"])+'</h2><h3>New patient case</h3><div class="question">'+html.escape(row["new_case"])+'</div><div class="panels">')
    for arm in ["A","B"]:
        parts.append('<section class="panel"><h3>Evidence panel '+arm+'</h3>')
        for j,h in enumerate(row["panel_"+arm],1):
            parts.append('<details><summary>Historical case '+str(j)+'</summary><div class="historical"><strong>Case description</strong>\n'+html.escape(h["case"])+'</div><div class="historical"><strong>Historical reasoning (diagnosis masked)</strong>\n'+html.escape(h["reasoning"])+'</div></details>')
        parts.append('</section>')
    parts.append('</div><p>Score both panels independently; A / B / TIE preference is entered separately.</p></article>')
parts.append('</main></body></html>')
(PUBLIC/"E082_REVIEWER_BOOKLET.html").write_text("".join(parts),encoding="utf-8")
public_manifest={
 "study":"E082 independent clinician appraisal","status":"packaged_unscored",
 "source_query_sha256":digest(qbytes),"source_rank_sha256":digest(rbytes),
 "n_frame":840,"n_sample":N,"historical_cases_per_panel":TOP,
 "sampling_independent_of_outcome":True,"panel_identity_hidden":True,
 "identical_top3_sets":sum(v==TOP for v in overlap),
 "mean_overlap_top3":sum(overlap)/len(overlap),
 "reviewer_cases_sha256":digest((PUBLIC/"E082_REVIEWER_CASES.jsonl").read_bytes()),
 "masking":"historical case and reasoning exact diagnosis phrase + non-generic diagnosis tokens",
 "limitations":["automatic mask may miss aliases","Top-3 human utility differs from Top-10 numerical result","human ratings not yet conducted"]
}
(PUBLIC/"E082_PUBLIC_MANIFEST.json").write_text(json.dumps(public_manifest,indent=2)+"\n")
private_manifest={"allocation_key_sha256":digest((PRIVATE/"E082_ALLOCATION_KEY.csv").read_bytes()),
                  "n_N8_as_A":sum(x["arm_A"]=="N8_RRF" for x in private_rows),
                  "n_N8_as_B":sum(x["arm_B"]=="N8_RRF" for x in private_rows),
                  "source_rank_sha256":digest(rbytes),"sampling_seed":SEED,
                  "do_not_send_to_clinical_reviewers":True}
(PRIVATE/"E082_PRIVATE_MANIFEST.json").write_text(json.dumps(private_manifest,indent=2)+"\n")
assert private_manifest["n_N8_as_A"]==private_manifest["n_N8_as_B"]==60
assert all("N8_RRF" not in json.dumps(x) and "TFIDF" not in json.dumps(x) for x in public_rows)
for folder,name in [(PUBLIC,"E082_REVIEWER_BLINDED_ONLY.zip"),(PRIVATE,"E082_ANALYST_KEY_PRIVATE.zip")]:
    with zipfile.ZipFile(ROOT/name,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for f in folder.iterdir():z.write(f,arcname=f.name)
print(json.dumps({"public":public_manifest,"private_arm_counts":[60,60],
  "archives":[{"name":x.name,"bytes":x.stat().st_size} for x in ROOT.glob("*.zip")]},indent=2))
