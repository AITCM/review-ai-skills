from __future__ import annotations
import json,re,unicodedata,pathlib,urllib.request,hashlib,html
import pandas as pd, numpy as np

AG_COMMIT="b6fbe22300e99a267a7ac94eaa465ab552eef741"
AG_URL=f"https://raw.githubusercontent.com/SamuelSchmidgall/AgentClinic/{AG_COMMIT}/agentclinic_nejm_extended.jsonl"
MC_REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";HF=f"https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/{MC_REV}/data"
OUT=pathlib.Path("e040");SALT_A="E040-RATER-A-v1";SALT_B="E040-RATER-B-v1"

def nw(s):return " ".join(re.findall(r"\w+",unicodedata.normalize("NFKC",str(s)).casefold()))
def side(case_id,salt):
 return "method_left" if int(hashlib.sha256((salt+":"+case_id).encode()).hexdigest()[:8],16)%2==0 else "baseline_left"
def mask_reason(reason,dx):
 x=str(reason);d=str(dx).strip()
 if d: x=re.sub(re.escape(d),"[diagnosis masked]",x,flags=re.I)
 return x

lock=json.load(open(OUT/"E040_RANKING_LOCK.json"));assert lock["status"]=="agentclinic_rankings_locked_before_answers_loaded"
queries=[json.loads(x) for x in open(OUT/"E040_QUERY_ONLY.jsonl") if x.strip()]
ranks=[json.loads(x) for x in open(OUT/"E040_BLIND_RANKINGS.jsonl") if x.strip()]
assert len(queries)==len(ranks)==120

# Load answers only after ranking lock; used for audit/answer key, never displayed to raters.
req=urllib.request.Request(AG_URL,headers={"User-Agent":"CASE-EVID-001/1.0"})
with urllib.request.urlopen(req,timeout=120) as r: full=[json.loads(x) for x in r.read().decode().splitlines() if x.strip()]
assert len(full)==120
tr=pd.read_parquet(HF+"/train-00000-of-00001.parquet",columns=["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"])

records=[];keys=[];same_order=0;same_set=0
for i,(q,r,x) in enumerate(zip(queries,ranks,full)):
 correct=[a["text"] for a in x["answers"] if a.get("correct")]
 assert len(correct)==1
 b=r["baseline_top50"][:3];m=r["method_top50"][:3]
 same_order+=int(b==m);same_set+=int(set(b)==set(m))
 def ev(ids):
  out=[]
  for rank_pos,j in enumerate(ids,1):
   row=tr.iloc[j]
   out.append({"rank":rank_pos,"reference_id":f"MC-{int(j):05d}","pmcid":str(row.pmcid),
    "case_prompt":str(row.case_prompt),"historical_diagnosis":str(row.final_diagnosis),
    "diagnostic_reasoning":mask_reason(row.diagnostic_reasoning,row.final_diagnosis)})
  return out
 rec={"query_index":i,"case_id":q["case_id"],"target_case":q["question"],
      "baseline_ids":b,"method_ids":m,"baseline_evidence":ev(b),"method_evidence":ev(m),
      "same_ordered_top3":b==m,"same_unordered_top3":set(b)==set(m),
      "target_gold_diagnosis":correct[0],"target_gold_explicit_in_query":nw(correct[0]) in nw(q["question"])}
 records.append(rec)

def make_pack(salt,label):
 blind=[];answer=[]
 for rec in records:
  s=side(rec["case_id"],salt)
  left=rec["method_evidence"] if s=="method_left" else rec["baseline_evidence"]
  right=rec["baseline_evidence"] if s=="method_left" else rec["method_evidence"]
  blind.append({"query_index":rec["query_index"],"case_id":rec["case_id"],"target_case":rec["target_case"],
   "set_A":left,"set_B":right,"same_ordered_top3":rec["same_ordered_top3"],"same_unordered_top3":rec["same_unordered_top3"]})
  answer.append({"query_index":rec["query_index"],"case_id":rec["case_id"],"A_method":"C2" if s=="method_left" else "TFIDF",
   "B_method":"TFIDF" if s=="method_left" else "C2","target_gold_diagnosis":rec["target_gold_diagnosis"],
   "target_gold_explicit_in_query":rec["target_gold_explicit_in_query"]})
 (OUT/f"E040_{label}_BLIND.jsonl").write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in blind)+"\n")
 (OUT/f"E040_{label}_ANSWER_KEY.jsonl").write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in answer)+"\n")
 return blind

A=make_pack(SALT_A,"RATER_A");B=make_pack(SALT_B,"RATER_B")

def render(pack,label):
 rows=[]
 for rec in pack:
  if rec["same_unordered_top3"]: continue
  def evidence_html(items):
   z=[]
   for x in items:
    z.append(f"""<div class="ref"><div class="refhead">Reference {x['rank']} · {html.escape(x['reference_id'])}</div>
    <p><b>Case:</b> {html.escape(x['case_prompt'])}</p>
    <p><b>Historical diagnosis:</b> {html.escape(x['historical_diagnosis'])}</p>
    <p><b>Historical reasoning:</b> {html.escape(x['diagnostic_reasoning'])}</p></div>""")
   return "\n".join(z)
  rows.append(f"""<section class="case" id="case-{rec['query_index']}">
  <h2>Case {rec['query_index']+1} <span>{html.escape(rec['case_id'])}</span></h2>
  <div class="target">{html.escape(rec['target_case'])}</div>
  <div class="cols"><div><h3>Set A</h3>{evidence_html(rec['set_A'])}</div><div><h3>Set B</h3>{evidence_html(rec['set_B'])}</div></div>
  <div class="rating"><b>Which reference set is more useful for reasoning about the diagnosis of the target case?</b><br/>
  ☐ A clearly better &nbsp; ☐ A slightly better &nbsp; ☐ About equal &nbsp; ☐ B slightly better &nbsp; ☐ B clearly better &nbsp; ☐ Neither useful<br/>
  Confidence (1–5): ______ &nbsp;&nbsp; Misleading evidence? ☐ A ☐ B ☐ Both ☐ Neither<br/>
  Comment: __________________________________________________________________________________</div></section>""")
 css="""body{font-family:Arial,'Microsoft YaHei',sans-serif;line-height:1.55;margin:30px;color:#172033}h1{margin-bottom:6px}.note{background:#eef3ff;padding:14px;border-radius:10px}.case{border-top:2px solid #1e3a8a;padding:22px 0;page-break-inside:avoid}h2 span{font-size:12px;color:#667085}.target{background:#f8fafc;padding:15px;border:1px solid #d9e1ee;border-radius:8px}.cols{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:16px}.ref{border:1px solid #d9e1ee;border-radius:8px;padding:11px;margin-bottom:10px}.refhead{font-weight:700;color:#1e3a8a}.rating{background:#fff8e8;padding:13px;border-radius:8px;margin-top:14px}@media(max-width:900px){.cols{grid-template-columns:1fr}}@media print{body{margin:12mm}.case{break-before:page}}"""
 body="\n".join(rows)
 doc=f"""<!doctype html><html><head><meta charset="utf-8"><title>E040 {label} Blind Expert Evaluation</title><style>{css}</style></head><body>
 <h1>AgentClinic-NEJM Blind Clinical Case Evidence Evaluation — {label}</h1>
 <div class="note"><b>Blinding:</b> Set A/B identities are randomized. Target gold diagnosis and retrieval method names are hidden.
 <br/><b>Primary question:</b> Which set of historical cases would be more useful for diagnostic reasoning about the target case?
 <br/><b>Instruction:</b> Judge clinical usefulness, discriminative relevance, and risk of misleading analogy. Do not reward superficial word overlap alone.</div>{body}</body></html>"""
 (OUT/f"E040_{label}_BLIND_EXPERT_PACK.html").write_text(doc,encoding="utf-8")

render(A,"RATER_A");render(B,"RATER_B")

summary={"status":"expert_pairwise_pack_locked","n_cases":120,"top_k":3,
 "same_ordered_top3":same_order,"same_unordered_top3":same_set,
 "cases_requiring_human_review":120-same_set,
 "rater_randomization":{"RATER_A":SALT_A,"RATER_B":SALT_B},
 "primary_analysis_population":"all cases with non-identical unordered Top-3 evidence sets",
 "primary_outcome":"blinded pairwise preference for diagnostic reasoning utility",
 "predefined_choices":["A clearly better","A slightly better","About equal","B slightly better","B clearly better","Neither useful"],
 "secondary_fields":["confidence_1_to_5","misleading_evidence_A_B_both_neither","free_text_comment"],
 "analysis_plan":[
  "Decode A/B only after both raters complete all assigned cases.",
  "Report rater-specific C2 win/loss/tie counts and ordinal preference distributions.",
  "Primary consensus endpoint: C2 preferred vs TF-IDF preferred among adjudicated non-tie cases; report exact binomial 95% CI/test.",
  "Report agreement before adjudication; use a third clinician only for discordant directional preferences.",
  "Cases with identical unordered Top-3 evidence sets are recorded as automatic ties and are not manually rated."
 ],
 "target_gold_explicit_count":sum(x["target_gold_explicit_in_query"] for x in records),
 "note":"No human labels have been collected yet; this artifact is a locked blinded evaluation instrument."}
(OUT/"E040_EXPERT_PROTOCOL_LOCK.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
