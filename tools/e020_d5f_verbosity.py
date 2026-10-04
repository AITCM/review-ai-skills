from __future__ import annotations
import json,re,unicodedata,pathlib,glob
import numpy as np,pandas as pd
from scipy.stats import spearmanr

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("d5f-output");OUT.mkdir(exist_ok=True)
PAT=re.compile(r'^\s*(?:\((\d+)\)|(\d+)[.)])\s*',re.M)
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
def load(path,key):
 R={}
 for line in open(path):
  x=json.loads(line);R[int(x["query_index"])]=x[key]
 return R

tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["diagnostic_reasoning"]]
counts=np.array([len(split_reason(x)) for x in tr.diagnostic_reasoning],float)
words=np.array([len(str(x).split()) for x in tr.diagnostic_reasoning],float)

rows=[]; per_seed=[]
for d in sorted(glob.glob("downloads/d5c-*")):
 seed=int(d.rsplit("-",1)[-1])
 base=load(d+"/rankings.jsonl","baseline_top50");d5=load(d+"/rankings.jsonl","D5_top50")
 top_count_diff=[];top_word_diff=[];rho_count=[];rho_words=[]
 for i in range(len(base)):
  b=base[i];r=d5[i];top_count_diff.append(counts[r[0]]-counts[b[0]]);top_word_diff.append(words[r[0]]-words[b[0]])
  posb={j:p for p,j in enumerate(b)};posr={j:p for p,j in enumerate(r)}
  common=[j for j in b if j in posr]
  promo=np.array([posb[j]-posr[j] for j in common],float)
  c=counts[common];w=words[common]
  if np.std(promo)>0 and np.std(c)>0:
   z=float(spearmanr(promo,c).statistic)
   if np.isfinite(z):rho_count.append(z)
  if np.std(promo)>0 and np.std(w)>0:
   z=float(spearmanr(promo,w).statistic)
   if np.isfinite(z):rho_words.append(z)
 per_seed.append({"seed":seed,
  "mean_top1_reason_point_count_delta":float(np.mean(top_count_diff)),
  "mean_top1_reason_word_delta":float(np.mean(top_word_diff)),
  "median_within_query_promotion_vs_reason_count_spearman":float(np.median(rho_count)),
  "median_within_query_promotion_vs_reason_words_spearman":float(np.median(rho_words))})
summary={"n_seeds":len(per_seed),"per_seed":per_seed,
 "aggregate":{
  "mean_top1_reason_point_count_delta":float(np.mean([x["mean_top1_reason_point_count_delta"] for x in per_seed])),
  "mean_top1_reason_word_delta":float(np.mean([x["mean_top1_reason_word_delta"] for x in per_seed])),
  "mean_median_promotion_vs_reason_count_spearman":float(np.mean([x["median_within_query_promotion_vs_reason_count_spearman"] for x in per_seed])),
  "mean_median_promotion_vs_reason_words_spearman":float(np.mean([x["median_within_query_promotion_vs_reason_words_spearman"] for x in per_seed]))
 },
 "interpretation":"Positive deltas would indicate residual preference for more heavily annotated / longer-reasoning cases; near-zero values support the debiasing claim.",
 "limits":["Reason point count and word count are annotation-verbosity proxies, not complete measures of case complexity.","Validation cases are used only to inspect ranking behavior; no outcome tuning is performed."]}
(OUT/"E020_D5F_VERBOSITY_BIAS_AUDIT.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
