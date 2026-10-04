import json,re,unicodedata,pathlib
from collections import defaultdict
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e020-b2-output");OUT.mkdir(exist_ok=True);SEED=20261004
def norm_words(s):
 return " ".join(re.findall(r"\w+",unicodedata.normalize("NFKC",str(s)).casefold()))
def mask_reason(r,d):
 rr=norm_words(r);dd=norm_words(d)
 if dd and dd in rr:return rr.replace(dd," diagnosismask "),True
 return rr,False
def rank(s,k=50):
 idx=np.arange(len(s));return np.lexsort((idx,-s))[:k]
def mm(x):
 x=np.asarray(x,float);a=x.min();b=x.max();return np.zeros_like(x) if b<=a else (x-a)/(b-a)
def met(name,ranks,tl,vl,mp):
 elig=[i for i,y in enumerate(vl) if y in mp];o={"method":name}
 for k in (1,3,10,50):
  a=sum(any(tl[j]==vl[i] for j in ranks[i][:k]) for i in range(len(vl)))
  e=sum(any(tl[j]==vl[i] for j in ranks[i][:k]) for i in elig)
  o[f"hit{k}_all"]=a;o[f"hit{k}_all_rate"]=a/len(vl);o[f"hit{k}_eligible"]=e;o[f"hit{k}_eligible_rate"]=e/len(elig)
 return o
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["case_prompt","diagnostic_reasoning","final_diagnosis"]]
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["case_prompt","diagnostic_reasoning","final_diagnosis"]]
tl=[norm_words(x) for x in tr.final_diagnosis];vl=[norm_words(x) for x in va.final_diagnosis];mp=defaultdict(list)
for i,y in enumerate(tl):mp[y].append(i)
train_mask=[mask_reason(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
val_mask=[mask_reason(r,d) for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
trr=[x[0] for x in train_mask];var=[x[0] for x in val_mask]
# lexical candidate generator
cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=cv.fit_transform(tr.case_prompt.astype(str));Q=cv.transform(va.case_prompt.astype(str));CS=(Q@X.T).tocsr()
cr=[];cs=[]
for i in range(len(va)):
 s=CS.getrow(i).toarray().ravel();r=rank(s,50);cr.append(r.tolist());cs.append(s[r])
# masked privileged space
cSVD=TruncatedSVD(256,n_iter=7,random_state=SEED);XC=cSVD.fit_transform(X).astype(float);QC=cSVD.transform(Q).astype(float)
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R=rv.fit_transform(trr);RV=rv.transform(var)
rSVD=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR=rSVD.fit_transform(R).astype(float);ZV=rSVD.transform(RV).astype(float)
xm=XC.mean(0,keepdims=True);ym=ZR.mean(0,keepdims=True);A=XC-xm;Y=ZR-ym
W=np.linalg.solve(A.T@A+10*np.eye(A.shape[1]),A.T@Y)
P=normalize((QC-xm)@W+ym);ZI=normalize(ZR);VT=normalize(ZV);PS=P@ZI.T;TRUE=VT@ZI.T
results=[]
for a in [0.0,0.25,0.5,0.75,1.0]:
 ranks=[]
 for i,c0 in enumerate(cr):
  c=np.array(c0);score=a*mm(cs[i])+(1-a)*mm(PS[i,c]);order=np.lexsort((c,-score));ranks.append(c[order].tolist())
 m=met(f"masked_semantic_weight_{a:.2f}",ranks,tl,vl,mp)
 m["masked_reasoning_top1_mean"]=float(np.mean([TRUE[i,r[0]] for i,r in enumerate(ranks)]))
 m["masked_reasoning_top3_mean"]=float(np.mean([np.mean(TRUE[i,r[:3]]) for i,r in enumerate(ranks)]))
 results.append(m)
summary={"method":"B2 diagnosis-name-masked privileged reasoning sensitivity",
 "train_reasoning_exact_diagnosis_phrase_masked":sum(x[1] for x in train_mask),
 "train_mask_rate":sum(x[1] for x in train_mask)/len(train_mask),
 "validation_reasoning_exact_diagnosis_phrase_masked":sum(x[1] for x in val_mask),
 "validation_mask_rate":sum(x[1] for x in val_mask)/len(val_mask),
 "mask_rule":"NFKC+casefold+word-token normalization; exact normalized diagnosis phrase replaced by diagnosismask",
 "results":results,
 "limitations":["Exact phrase masking does not remove synonyms or indirect diagnostic clues.","Validation reasoning remains post-hoc evaluation only.","Weight grid is exploratory validation analysis."]}
(OUT/"E020_B2_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
