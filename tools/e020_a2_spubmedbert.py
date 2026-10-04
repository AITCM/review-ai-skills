from __future__ import annotations
import json,re,unicodedata,pathlib
from collections import defaultdict
import numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer
from huggingface_hub import HfApi

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
MODEL="pritamdeka/S-PubMedBert-MS-MARCO"
OUT=pathlib.Path("e020-a2-output"); OUT.mkdir(exist_ok=True)
def nl(s):
    s=unicodedata.normalize("NFKC",str(s)).casefold()
    return " ".join(re.findall(r"\w+",s))
def topk(sim,k=50):
    p=np.argpartition(-sim,k-1,axis=1)[:,:k]
    vals=np.take_along_axis(sim,p,axis=1)
    order=np.argsort(-vals,axis=1,kind="stable")
    return np.take_along_axis(p,order,axis=1),np.take_along_axis(vals,order,axis=1)
train=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["pmcid","case_prompt","final_diagnosis"]]
val=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["pmcid","case_prompt","final_diagnosis"]]
tl=[nl(x) for x in train.final_diagnosis]; vl=[nl(x) for x in val.final_diagnosis]
label_to_train=defaultdict(list)
for i,y in enumerate(tl):label_to_train[y].append(i)
sha=HfApi().model_info(MODEL).sha
model=SentenceTransformer(MODEL,revision=sha,device="cpu")
model.max_seq_length=256
texts=train.case_prompt.astype(str).tolist()+val.case_prompt.astype(str).tolist()
emb=model.encode(texts,batch_size=16,show_progress_bar=True,normalize_embeddings=True,convert_to_numpy=True)
X=emb[:len(train)].astype(np.float32); Q=emb[len(train):].astype(np.float32)
sim=Q@X.T
rank,scores=topk(sim,50)
eligible=[i for i,y in enumerate(vl) if y in label_to_train]
metrics={"method":"S-PubMedBert-MS-MARCO","model":MODEL,"model_revision":sha,"max_seq_length":256,
         "train_rows":len(train),"validation_rows":len(val),"eligible_exact_label":len(eligible)}
for k in (1,3,10,50):
    allhit=sum(any(tl[j]==vl[i] for j in rank[i,:k]) for i in range(len(val)))
    ehit=sum(any(tl[j]==vl[i] for j in rank[i,:k]) for i in eligible)
    metrics[f"hit_at_{k}_all"]=allhit;metrics[f"hit_at_{k}_all_rate"]=allhit/len(val)
    metrics[f"hit_at_{k}_eligible"]=ehit;metrics[f"hit_at_{k}_eligible_rate"]=ehit/len(eligible)
pref=0; margins=[]; cases=[]
for i in eligible:
    same=np.array(label_to_train[vl[i]],dtype=int)
    same_scores=sim[i,same]
    hp=int(same[int(np.argmax(same_scores))]); hp_s=float(same_scores.max())
    diff_idx=np.where(np.array(tl)!=vl[i])[0]
    # top-ranked different label
    d=next(int(j) for j in rank[i] if tl[int(j)]!=vl[i])
    ds=float(sim[i,d])
    if ds>hp_s: pref+=1
    margins.append(ds-hp_s)
    cases.append({"query_index":i,"pmcid":str(val.iloc[i].pmcid),"diagnosis":str(val.iloc[i].final_diagnosis),
                  "top1_index":int(rank[i,0]),"top1_diagnosis":str(train.iloc[int(rank[i,0])].final_diagnosis),
                  "top1_score":float(scores[i,0]),"top1_exact_label":bool(tl[int(rank[i,0])]==vl[i]),
                  "best_same_label_index":hp,"best_same_label_score":hp_s,
                  "best_diff_label_index":d,"best_diff_label_score":ds,"diff_minus_same_margin":ds-hp_s})
metrics["eligible_semantic_prefers_diff_over_best_same"]=pref
metrics["eligible_semantic_prefers_diff_rate"]=pref/len(eligible)
metrics["diff_minus_same_margin_median"]=float(np.median(margins))
metrics["limitations"]=[
 "Exact normalized diagnosis equality is a proxy, not clinical equivalence.",
 "The model was trained for medical/health information retrieval, not specifically case-to-case diagnosis retrieval.",
 "Inputs are truncated to 256 model tokens; truncation is an explicit benchmark choice.",
 "The test split is not used."
]
(OUT/"E020_A2_SUMMARY.json").write_text(json.dumps(metrics,indent=2,ensure_ascii=False)+"\n")
pd.DataFrame(cases).to_csv(OUT/"E020_A2_CASES.csv",index=False)
with (OUT/"E020_A2_TOP50.jsonl").open("w",encoding="utf-8") as f:
    for i in range(len(val)):
        f.write(json.dumps({"query_index":i,"indices":[int(x) for x in rank[i]],"scores":[float(x) for x in scores[i]]})+"\n")
print(json.dumps(metrics,indent=2))
