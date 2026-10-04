import json,re,unicodedata,pathlib
from collections import defaultdict
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from scipy.stats import binomtest
REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
OUT=pathlib.Path("e020-b2b-output");OUT.mkdir(exist_ok=True);SEED=20261004
def nw(s):return " ".join(re.findall(r"\w+",unicodedata.normalize("NFKC",str(s)).casefold()))
def mask(r,d):
 rr=nw(r);dd=nw(d);return rr.replace(dd," diagnosismask ") if dd and dd in rr else rr
def rank(s,k=50):
 idx=np.arange(len(s));return np.lexsort((idx,-s))[:k]
def mm(x):
 x=np.asarray(x,float);a=x.min();b=x.max();return np.zeros_like(x) if b<=a else (x-a)/(b-a)
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["case_prompt","diagnostic_reasoning","final_diagnosis"]]
va=pd.read_parquet(BASE+"/val-00000-of-00001.parquet")[["case_prompt","diagnostic_reasoning","final_diagnosis"]]
tl=[nw(x) for x in tr.final_diagnosis];vl=[nw(x) for x in va.final_diagnosis]
cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=cv.fit_transform(tr.case_prompt.astype(str));Q=cv.transform(va.case_prompt.astype(str));CS=(Q@X.T).tocsr()
cr=[];cs=[]
for i in range(len(va)):
 s=CS.getrow(i).toarray().ravel();r=rank(s,50);cr.append(r.tolist());cs.append(s[r])
cS=TruncatedSVD(256,n_iter=7,random_state=SEED);XC=cS.fit_transform(X).astype(float);QC=cS.transform(Q).astype(float)
trr=[mask(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
var=[mask(r,d) for r,d in zip(va.diagnostic_reasoning,va.final_diagnosis)]
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R=rv.fit_transform(trr);RV=rv.transform(var);rS=TruncatedSVD(128,n_iter=7,random_state=SEED)
ZR=rS.fit_transform(R).astype(float);ZV=rS.transform(RV).astype(float)
xm=XC.mean(0,keepdims=True);ym=ZR.mean(0,keepdims=True);A=XC-xm;Y=ZR-ym;W=np.linalg.solve(A.T@A+10*np.eye(A.shape[1]),A.T@Y)
P=normalize((QC-xm)@W+ym);ZI=normalize(ZR);VT=normalize(ZV);PS=P@ZI.T;TRUE=VT@ZI.T
fusion=[]
for i,c0 in enumerate(cr):
 c=np.array(c0);score=.5*mm(cs[i])+.5*mm(PS[i,c]);o=np.lexsort((c,-score));fusion.append(c[o].tolist())
def hit(r,i,k):return any(tl[j]==vl[i] for j in r[:k])
stats={}
for k in [1,3,10]:
 a=np.array([hit(cr[i],i,k) for i in range(len(va))]);b=np.array([hit(fusion[i],i,k) for i in range(len(va))])
 helpn=int(np.sum((~a)&b));harm=int(np.sum(a&(~b)));both=int(np.sum(a&b));neither=int(np.sum((~a)&(~b)))
 p=float(binomtest(min(helpn,harm),n=helpn+harm,p=.5).pvalue) if helpn+harm else 1.0
 stats[str(k)]={"baseline_hits":int(a.sum()),"fusion_hits":int(b.sum()),"helped":helpn,"harmed":harm,"both":both,"neither":neither,"exact_mcnemar_binomial_p":p}
# paired bootstrap reasoning top1 delta
base=np.array([TRUE[i,cr[i][0]] for i in range(len(va))]);fus=np.array([TRUE[i,fusion[i][0]] for i in range(len(va))]);delta=fus-base
rng=np.random.default_rng(SEED);boots=[]
for _ in range(5000):
 ids=rng.integers(0,len(delta),len(delta));boots.append(float(delta[ids].mean()))
summary={"comparison":"masked B2 semantic_weight=0.50 vs TF-IDF within same top50 candidate pools","paired_hit_transitions":stats,
"reasoning_top1_mean_baseline":float(base.mean()),"reasoning_top1_mean_fusion":float(fus.mean()),"reasoning_top1_mean_delta":float(delta.mean()),
"reasoning_delta_bootstrap_95pct":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))],
"interpretation":"Exploratory validation-set paired statistics; not confirmatory test-set evidence."}
(OUT/"E020_B2B_PAIRED_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
