from __future__ import annotations
import argparse,json,re,unicodedata,pathlib,hashlib
from collections import defaultdict
import numpy as np,pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.linear_model import LogisticRegression

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
BASE="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
SEED=20261004;TRAIN_K=40;DEPLOY_K=50;MARGIN=.03
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
def rank(s,k):
 ids=np.arange(len(s));return np.lexsort((ids,-s))[:k]
def feat(sem,prd):
 return np.column_stack([np.asarray(sem,float),np.asarray(prd,float)])
def softf1(q,c):
 S=(q@c.T).toarray()
 r=float(np.max(S,axis=1).mean());p=float(np.max(S,axis=0).mean())
 return 2*r*p/(r+p) if r+p else 0.

ap=argparse.ArgumentParser();ap.add_argument("--out",default="test-output");args=ap.parse_args()
OUT=pathlib.Path(args.out);OUT.mkdir(parents=True,exist_ok=True)

# ---- TRAIN ONLY: learn privileged utility ranker ----
tr=pd.read_parquet(BASE+"/train-00000-of-00001.parquet")[["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]].copy()
isdev=np.array([int(hashlib.sha256((f"E020-FINAL-{SEED}:"+str(x)).encode()).hexdigest()[:8],16)%5==0 for x in tr.pmcid])
ref=tr.loc[~isdev].reset_index(drop=True);mq=tr.loc[isdev].reset_index(drop=True)

cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=cv.fit_transform(ref.case_prompt.astype(str));Q=cv.transform(mq.case_prompt.astype(str))
cs=TruncatedSVD(256,n_iter=7,random_state=SEED);XC=cs.fit_transform(X).astype(float);QC=cs.transform(Q).astype(float)
rp=[pts(r,d) for r,d in zip(ref.diagnostic_reasoning,ref.final_diagnosis)]
mp=[pts(r,d) for r,d in zip(mq.diagnostic_reasoning,mq.final_diagnosis)]
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R=rv.fit_transform([" ".join(x) for x in rp]);rs=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR=rs.fit_transform(R).astype(float)
xm=XC.mean(0,keepdims=True);ym=ZR.mean(0,keepdims=True);A=XC-xm;Y=ZR-ym
W=np.linalg.solve(A.T@A+10*np.eye(A.shape[1]),A.T@Y)
P=normalize((QC-xm)@W+ym);ZI=normalize(ZR);CS=(Q@X.T).tocsr();PS=P@ZI.T

pv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
PM=pv.fit_transform([p for z in rp for p in z]+[p for z in mp for p in z]);off=0;RPM=[]
for z in rp:RPM.append(PM[off:off+len(z)]);off+=len(z)
MPM=[]
for z in mp:MPM.append(PM[off:off+len(z)]);off+=len(z)

D=[];Yy=[];pairs=0
for i in range(len(mq)):
 s=CS.getrow(i).toarray().ravel();cand=rank(s,TRAIN_K);F=feat(s[cand],PS[i,cand]);u=np.array([softf1(MPM[i],RPM[int(j)]) for j in cand])
 for p in np.argsort(-u)[:5]:
  for n in np.argsort(u)[:10]:
   if u[p]-u[n]<MARGIN:continue
   d=F[p]-F[n];D.append(d);Yy.append(1);D.append(-d);Yy.append(0);pairs+=1
D=np.asarray(D);Yy=np.asarray(Yy);sc=StandardScaler();Z=sc.fit_transform(D)
clf=LogisticRegression(C=1,max_iter=2000,random_state=SEED).fit(Z,Yy)
w=clf.coef_[0]/sc.scale_;b=float(clf.intercept_[0]-np.dot(clf.coef_[0],sc.mean_/sc.scale_))

# ---- DEPLOYMENT: fit library representations on all train; test inputs = case_prompt only ----
test_input=pd.read_parquet(BASE+"/test-00000-of-00001.parquet",columns=["pmcid","case_prompt"])
cv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X2=cv2.fit_transform(tr.case_prompt.astype(str));QT=cv2.transform(test_input.case_prompt.astype(str))
cs2=TruncatedSVD(256,n_iter=7,random_state=SEED);XC2=cs2.fit_transform(X2).astype(float);QCT=cs2.transform(QT).astype(float)
tp=[pts(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
rv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R2=rv2.fit_transform([" ".join(x) for x in tp]);rs2=TruncatedSVD(128,n_iter=7,random_state=SEED);ZR2=rs2.fit_transform(R2).astype(float)
xm=XC2.mean(0,keepdims=True);ym=ZR2.mean(0,keepdims=True);AA=XC2-xm;YY=ZR2-ym
WW=np.linalg.solve(AA.T@AA+10*np.eye(AA.shape[1]),AA.T@YY)
PP=normalize((QCT-xm)@WW+ym);ZZ=normalize(ZR2);CSS=(QT@X2.T).tocsr();PSS=PP@ZZ.T

baseline=[];method=[]
for i in range(len(test_input)):
 s=CSS.getrow(i).toarray().ravel();cand=rank(s,DEPLOY_K);baseline.append(cand.tolist())
 F=feat(s[cand],PSS[i,cand]);score=F@w+b;ids=np.asarray(cand);o=np.lexsort((ids,-score));method.append(ids[o].tolist())

ranking_receipt={
 "dataset_revision":REV,"test_queries":len(test_input),"seed":SEED,"training_candidate_k":TRAIN_K,
 "deployment_candidate_k":DEPLOY_K,"utility_margin":MARGIN,"features":["semantic_case_similarity","predicted_reasoning_similarity"],
 "weights":[float(x) for x in w],"intercept":b,"preference_pairs":pairs,
 "test_fields_loaded_before_ranking":["pmcid","case_prompt"],
 "test_answer_fields_loaded_before_ranking":False
}
(OUT/"TEST_RANKING_RECEIPT.json").write_text(json.dumps(ranking_receipt,indent=2)+"\n")
with (OUT/"TEST_LOCKED_RANKINGS.jsonl").open("w") as f:
 for i in range(len(test_input)):
  f.write(json.dumps({"query_index":i,"pmcid":str(test_input.iloc[i].pmcid),"baseline_top50":baseline[i],"method_top50":method[i]})+"\n")

# Explicit barrier: only after locked rankings are persisted do we load test outcomes for evaluation.
te=pd.read_parquet(BASE+"/test-00000-of-00001.parquet",columns=["diagnostic_reasoning","final_diagnosis"])
assert len(te)==len(test_input)

# Primary evaluation vectorizer fit on TRAIN reasoning only.
ev=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
T=ev.fit_transform([p for z in tp for p in z]);offs=[];o=0
for z in tp:offs.append((o,o+len(z)));o+=len(z)
tep=[pts(r,d) for r,d in zip(te.diagnostic_reasoning,te.final_diagnosis)]
V=ev.transform([p for z in tep for p in z]);vo=[];o=0
for z in tep:vo.append((o,o+len(z)));o+=len(z)
def evalsf(i,cands):
 s,e=vo[i];q=V[s:e];m=[]
 for j in cands:
  a,z=offs[j];m.append(T[a:z])
 return softf1(q,vstack(m))

tl=[nw(x) for x in tr.final_diagnosis];yl=[nw(x) for x in te.final_diagnosis];freq=defaultdict(int)
for y in tl:freq[y]+=1
RNG=np.random.default_rng(SEED)
summary={"analysis_plan_commit":"8f811906c38762bba0c172417827d3e8fc36e478",
 "method":"C2-debiased-final-two-feature","primary_endpoint":"paired delta symmetric reasoning-set softF1@10",
 "test_n":len(te),"results":{},"exact_label_secondary":{},"subgroups":{},"reason_point_count":{}}
per=[]
for k in (1,3,10):
 bv=np.array([evalsf(i,baseline[i][:k]) for i in range(len(te))]);mv=np.array([evalsf(i,method[i][:k]) for i in range(len(te))]);d=mv-bv
 boots=[]
 for _ in range(10000):
  ids=RNG.integers(0,len(d),len(d));boots.append(float(d[ids].mean()))
 summary["results"][str(k)]={"baseline":float(bv.mean()),"method":float(mv.mean()),"delta":float(d.mean()),
  "ci95":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))],
  "improved":int((d>0).sum()),"worsened":int((d<0).sum()),"tied":int((d==0).sum())}
 summary["exact_label_secondary"][str(k)]={
  "baseline":int(sum(any(tl[j]==yl[i] for j in baseline[i][:k]) for i in range(len(te)))),
  "method":int(sum(any(tl[j]==yl[i] for j in method[i][:k]) for i in range(len(te))))}
 bc=np.array([sum(len(tp[j]) for j in baseline[i][:k]) for i in range(len(te))]);mc=np.array([sum(len(tp[j]) for j in method[i][:k]) for i in range(len(te))])
 summary["reason_point_count"][str(k)]={"baseline_mean":float(bc.mean()),"method_mean":float(mc.mean()),"delta_mean":float((mc-bc).mean())}
 for label,ids in {"label_present":[i for i,y in enumerate(yl) if freq[y]>0],"label_absent":[i for i,y in enumerate(yl) if freq[y]==0]}.items():
  summary["subgroups"].setdefault(label,{})[str(k)]={"n":len(ids),"delta":float(d[ids].mean()) if ids else None}
 if k==10:
  per=[{"query_index":i,"baseline":float(bv[i]),"method":float(mv[i]),"delta":float(d[i]),"label_present":bool(freq[yl[i]]>0)} for i in range(len(te))]
summary["primary_pass"]=bool(summary["results"]["10"]["ci95"][0]>0)
summary["interpretation_rule"]="Secondary endpoints cannot rescue a failed primary endpoint."
(OUT/"E020_D15_CONFIRMATORY_TEST.json").write_text(json.dumps(summary,indent=2)+"\n")
with (OUT/"E020_D15_PER_QUERY_TOP10.jsonl").open("w") as f:
 for x in per:f.write(json.dumps(x)+"\n")
print(json.dumps(summary,indent=2))
