from __future__ import annotations
import json,re,unicodedata,pathlib
from collections import defaultdict
import numpy as np,pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("t001-output");RNG=np.random.default_rng(20261004)
PAT=re.compile(r'^\s*(?:\((\d+)\)|(\d+)[.)])\s*',re.M)
def nw(s):return " ".join(re.findall(r"\w+",unicodedata.normalize("NFKC",str(s)).casefold()))
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
def pts(r,d):
 dd=nw(d);o=[]
 for x in split_reason(r):
  xx=nw(x)
  if dd and dd in xx:xx=xx.replace(dd," diagnosismask ")
  o.append(xx)
 return o
def sf(q,c):
 S=(q@c.T).toarray()
 if not S.size:return 0.
 r=float(np.max(S,axis=1).mean());p=float(np.max(S,axis=0).mean())
 return 2*r*p/(r+p) if r+p else 0.
lock=json.load(open(OUT/"T001_RANKING_LOCK.json"))
assert lock["status"]=="rankings_locked_before_test_outcomes_loaded" and lock["test_columns_read"]==["pmcid","case_prompt"]
R=[]
for l in open(OUT/"T001_BLIND_RANKINGS.jsonl"):R.append(json.loads(l))
assert len(R)==897 and [x["query_index"] for x in R]==list(range(897))
# Outcomes are loaded only after rankings are present and validated.
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet",columns=["diagnostic_reasoning","final_diagnosis"])
te=pd.read_parquet(BASE+"/test-00000-of-00001.parquet",columns=["diagnostic_reasoning","final_diagnosis"])
tp=[pts(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)];vp=[pts(r,d) for r,d in zip(te.diagnostic_reasoning,te.final_diagnosis)]
vec=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
T=vec.fit_transform([p for z in tp for p in z]);offs=[];o=0
for z in tp:offs.append((o,o+len(z)));o+=len(z)
V=vec.transform([p for z in vp for p in z]);vo=[];o=0
for z in vp:vo.append((o,o+len(z)));o+=len(z)
def score(i,cands):
 s,e=vo[i];q=V[s:e];m=[]
 for j in cands:a,b=offs[j];m.append(T[a:b])
 return sf(q,vstack(m))
tl=[nw(x) for x in tr.final_diagnosis];yl=[nw(x) for x in te.final_diagnosis];freq=defaultdict(int)
for y in tl:freq[y]+=1
summary={"status":"one_shot_locked_test_evaluation","method":"C2-debiased-final-two-feature","n_test":len(te),"primary_metric":"diagnosis-masked symmetric reasoning-set softF1","results":{},"subgroups":{}}
for k in (1,3,10):
 b=np.array([score(i,R[i]["baseline_top50"][:k]) for i in range(len(te))]);m=np.array([score(i,R[i]["method_top50"][:k]) for i in range(len(te))]);d=m-b;boots=[]
 for _ in range(10000):
  ids=RNG.integers(0,len(d),len(d));boots.append(float(d[ids].mean()))
 exact_b=int(sum(any(tl[j]==yl[i] for j in R[i]["baseline_top50"][:k]) for i in range(len(te))))
 exact_m=int(sum(any(tl[j]==yl[i] for j in R[i]["method_top50"][:k]) for i in range(len(te))))
 summary["results"][str(k)]={"baseline":float(b.mean()),"method":float(m.mean()),"delta":float(d.mean()),"ci95":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))],
  "improved":int((d>0).sum()),"worsened":int((d<0).sum()),"tied":int((d==0).sum()),"exact_label_hit_baseline":exact_b,"exact_label_hit_method":exact_m}
 for label,ids in {"label_present":[i for i,y in enumerate(yl) if freq[y]>0],"label_absent":[i for i,y in enumerate(yl) if freq[y]==0]}.items():
  dd=d[ids];summary["subgroups"].setdefault(label,{})[str(k)]={"n":len(ids),"mean_delta":float(dd.mean())}
# annotation density audit
for k in (1,3,10):
 bc=np.array([sum(len(tp[j]) for j in R[i]["baseline_top50"][:k]) for i in range(len(te))])
 mc=np.array([sum(len(tp[j]) for j in R[i]["method_top50"][:k]) for i in range(len(te))])
 summary["results"][str(k)]["mean_reason_points_baseline"]=float(bc.mean())
 summary["results"][str(k)]["mean_reason_points_method"]=float(mc.mean())
 summary["results"][str(k)]["corr_utility_delta_vs_reason_count_delta"]=float(np.corrcoef(
  np.array([score(i,R[i]["method_top50"][:k])-score(i,R[i]["baseline_top50"][:k]) for i in range(len(te))]),mc-bc)[0,1])
summary["test_gate_pass"]=summary["results"]["10"]["ci95"][0]>0
summary["policy"]="No parameter or method changes are permitted based on this test result; any future method is a new preregistered experiment."
(OUT/"T001_TEST_RESULTS.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
