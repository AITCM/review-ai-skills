from __future__ import annotations
import json,re,unicodedata,pathlib,hashlib,urllib.request
import numpy as np,pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.linear_model import LogisticRegression

REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
HF="https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/"+REV+"/data"
MEDR="https://raw.githubusercontent.com/MAGIC-AI4Med/MedRBench/main/data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json"
OUT=pathlib.Path("e031-output");OUT.mkdir(exist_ok=True)
SEED=20261004;CASE_DIM=256;REASON_DIM=128;RIDGE_ALPHA=10.;TRAIN_K=40;MARGIN=.03;DEPLOY_K=50;TOP_POS=5;BOTTOM_NEG=10;LOGISTIC_C=1.
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
def pts_medcase(r,d):
 dd=nw(d);o=[]
 for x in split_reason(r):
  xx=nw(x)
  if dd and dd in xx:xx=xx.replace(dd," diagnosismask ")
  o.append(xx)
 return o
def split_medr_diff(s):
 s=str(s)
 # numbered markdown differential bullets; fallback to paragraph
 chunks=re.split(r'(?m)(?=^\s*\d+\.\s+)',s)
 out=[]
 for x in chunks:
  x=x.strip()
  if not x:continue
  x=re.sub(r'^\d+\.\s*','',x)
  # remove a leading explicit diagnosis label before colon, including markdown **...**
  x=re.sub(r'^\*\*[^*]{1,160}\*\*\s*:\s*','',x)
  out.append(nw(x))
 return [x for x in out if x] or [nw(s)]
def mask_final_tokens(points,final_dx):
 toks={t for t in nw(final_dx).split() if len(t)>=4 and t not in {"with","without","acute","chronic","disease","syndrome","secondary","primary"}}
 out=[]
 for p in points:
  out.append(" ".join("diagnosismask" if t in toks else t for t in p.split()))
 return out
def rank(s,k):
 ids=np.arange(len(s));return np.lexsort((ids,-s))[:k]
def F(sem,prd):return np.column_stack([np.asarray(sem,float),np.asarray(prd,float)])
def sf(q,c):
 S=(q@c.T).toarray()
 if not S.size:return 0.
 r=float(np.max(S,axis=1).mean());p=float(np.max(S,axis=0).mean())
 return 2*r*p/(r+p) if r+p else 0.

tr=pd.read_parquet(HF+"/train-00000-of-00001.parquet",columns=["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"])
va=pd.read_parquet(HF+"/val-00000-of-00001.parquet",columns=["pmcid"])
te0=pd.read_parquet(HF+"/test-00000-of-00001.parquet",columns=["pmcid"])
all_medcase_pm=set(tr.pmcid.astype(str).str.upper())|set(va.pmcid.astype(str).str.upper())|set(te0.pmcid.astype(str).str.upper())

with urllib.request.urlopen(MEDR,timeout=120) as rr:data=json.load(rr)
rows=[]
for pmcid,rec in data.items():
 pm=str(pmcid).upper()
 if pm in all_medcase_pm:continue
 g=rec["generate_case"]
 rows.append({"pmcid":pm,"case_summary":g["case_summary"],"differential_diagnosis":g["differential_diagnosis"],
              "final_diagnosis":g["final_diagnosis"],"diagnosis_results":g["diagnosis_results"],
              "body_category":rec.get("body_category"),"rare":rec.get("checked_rare_disease")})
ext=pd.DataFrame(rows)
assert len(ext)==840

# EXACT T001 frozen method training on MedCase train
isdev=np.array([int(hashlib.sha256((f"E020-FINAL-{SEED}:"+str(x)).encode()).hexdigest()[:8],16)%5==0 for x in tr.pmcid])
ref=tr.loc[~isdev].reset_index(drop=True);mq=tr.loc[isdev].reset_index(drop=True)
cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=cv.fit_transform(ref.case_prompt.astype(str));Q=cv.transform(mq.case_prompt.astype(str))
cs=TruncatedSVD(CASE_DIM,n_iter=7,random_state=SEED);XC=cs.fit_transform(X).astype(float);QC=cs.transform(Q).astype(float)
rp=[pts_medcase(r,d) for r,d in zip(ref.diagnostic_reasoning,ref.final_diagnosis)]
mp=[pts_medcase(r,d) for r,d in zip(mq.diagnostic_reasoning,mq.final_diagnosis)]
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R=rv.fit_transform([" ".join(x) for x in rp]);rs=TruncatedSVD(REASON_DIM,n_iter=7,random_state=SEED);ZR=rs.fit_transform(R).astype(float)
xm=XC.mean(0,keepdims=True);ym=ZR.mean(0,keepdims=True);A=XC-xm;Y=ZR-ym
W=np.linalg.solve(A.T@A+RIDGE_ALPHA*np.eye(A.shape[1]),A.T@Y);P=normalize((QC-xm)@W+ym);ZI=normalize(ZR);CS=(Q@X.T).tocsr();PS=P@ZI.T
pv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
PM=pv.fit_transform([p for z in rp for p in z]+[p for z in mp for p in z]);off=0;RPM=[]
for z in rp:RPM.append(PM[off:off+len(z)]);off+=len(z)
MPM=[]
for z in mp:MPM.append(PM[off:off+len(z)]);off+=len(z)
D=[];Yy=[]
for i in range(len(mq)):
 s=CS.getrow(i).toarray().ravel();cand=rank(s,TRAIN_K);f=F(s[cand],PS[i,cand]);u=np.array([sf(MPM[i],RPM[int(j)]) for j in cand])
 for p in np.argsort(-u)[:TOP_POS]:
  for n in np.argsort(u)[:BOTTOM_NEG]:
   if u[p]-u[n]<MARGIN:continue
   d=f[p]-f[n];D.append(d);Yy.append(1);D.append(-d);Yy.append(0)
D=np.asarray(D);Yy=np.asarray(Yy);sc=StandardScaler();z=sc.fit_transform(D)
clf=LogisticRegression(C=LOGISTIC_C,max_iter=2000,random_state=SEED).fit(z,Yy)
w=clf.coef_[0]/sc.scale_;b=float(clf.intercept_[0]-np.dot(clf.coef_[0],sc.mean_/sc.scale_))

# Refit on all MedCase train; external query = case_summary only
cv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X2=cv2.fit_transform(tr.case_prompt.astype(str));QE=cv2.transform(ext.case_summary.astype(str))
cs2=TruncatedSVD(CASE_DIM,n_iter=7,random_state=SEED);XC2=cs2.fit_transform(X2).astype(float);QCE=cs2.transform(QE).astype(float)
tp=[pts_medcase(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
rv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R2=rv2.fit_transform([" ".join(x) for x in tp]);rs2=TruncatedSVD(REASON_DIM,n_iter=7,random_state=SEED);ZR2=rs2.fit_transform(R2).astype(float)
xm=XC2.mean(0,keepdims=True);ym=ZR2.mean(0,keepdims=True);AA=XC2-xm;YY=ZR2-ym
WW=np.linalg.solve(AA.T@AA+RIDGE_ALPHA*np.eye(AA.shape[1]),AA.T@YY)
PP=normalize((QCE-xm)@WW+ym);ZZ=normalize(ZR2);CSS=(QE@X2.T).tocsr();PSS=PP@ZZ.T
rankings=[]
for i in range(len(ext)):
 s=CSS.getrow(i).toarray().ravel();cand=rank(s,DEPLOY_K);score=F(s[cand],PSS[i,cand])@w+b;ids=np.asarray(cand);o=np.lexsort((ids,-score))
 rankings.append({"query_index":i,"pmcid":ext.iloc[i].pmcid,"baseline_top50":cand.tolist(),"method_top50":ids[o].tolist()})
with (OUT/"E031_EXTERNAL_BLIND_RANKINGS.jsonl").open("w") as f:
 for x in rankings:f.write(json.dumps(x)+"\n")
lock={"status":"external_rankings_locked","method":"T001 frozen two-feature method","n_external":len(ext),
      "source_disjoint_rule":"exclude MedR-Bench PMCID if present in any MedCaseReasoning train/val/test split",
      "external_query_fields_used_for_ranking":["pmcid","case_summary"],"weights":[float(x) for x in w]}
(OUT/"E031_RANKING_LOCK.json").write_text(json.dumps(lock,indent=2)+"\n")

# External evaluation: MedR differential-diagnosis rationales, normalized and label-prefix stripped.
ext_pts=[]
for _,r in ext.iterrows():
 ps=split_medr_diff(r.differential_diagnosis)
 ps=mask_final_tokens(ps,r.final_diagnosis)
 ext_pts.append(ps)
ev=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
T=ev.fit_transform([p for z in tp for p in z]);offs=[];o=0
for z in tp:offs.append((o,o+len(z)));o+=len(z)
V=ev.transform([p for z in ext_pts for p in z]);vo=[];o=0
for z in ext_pts:vo.append((o,o+len(z)));o+=len(z)
def score(i,cands):
 s,e=vo[i];q=V[s:e];m=[]
 for j in cands:a,b0=offs[j];m.append(T[a:b0])
 return sf(q,vstack(m))
rng=np.random.default_rng(SEED)
summary={"status":"source_disjoint_external_validation","dataset":"MedR-Bench diagnosis set",
 "n_original":957,"n_excluded_source_overlap":117,"n_external":840,
 "query_representation":"MedR-Bench generate_case.case_summary",
 "target_reasoning":"MedR-Bench generate_case.differential_diagnosis; leading explicit diagnosis labels stripped; final-diagnosis tokens aggressively masked",
 "candidate_library":"MedCaseReasoning train only","method":"T001 frozen C2-debiased-final-two-feature","results":{},"subgroups":{}}
for k in (1,3,10):
 b=np.array([score(i,rankings[i]["baseline_top50"][:k]) for i in range(len(ext))])
 m=np.array([score(i,rankings[i]["method_top50"][:k]) for i in range(len(ext))]);d=m-b;boots=[]
 for _ in range(10000):
  ids=rng.integers(0,len(d),len(d));boots.append(float(d[ids].mean()))
 summary["results"][str(k)]={"baseline":float(b.mean()),"method":float(m.mean()),"delta":float(d.mean()),
  "ci95":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))],"improved":int((d>0).sum()),"worsened":int((d<0).sum()),"tied":int((d==0).sum())}
 for label,ids in {"rare":[i for i,x in enumerate(ext.rare) if bool(x)],"not_marked_rare":[i for i,x in enumerate(ext.rare) if not bool(x)]}.items():
  if ids:summary["subgroups"].setdefault(label,{})[str(k)]={"n":len(ids),"mean_delta":float(d[ids].mean())}
summary["limits"]=[
 "Both datasets ultimately derive from published case reports; PMCID disjointness removes exact source overlap but not domain similarity.",
 "MedR-Bench differential_diagnosis is a different annotation schema from MedCaseReasoning diagnostic_reasoning.",
 "The external utility metric is therefore a cross-schema reasoning-alignment measure, not expert-adjudicated clinical utility.",
 "No parameter or method changes are made using MedR-Bench outcomes."
]
(OUT/"E031_EXTERNAL_RESULTS.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
print(json.dumps(summary,indent=2,ensure_ascii=False))
