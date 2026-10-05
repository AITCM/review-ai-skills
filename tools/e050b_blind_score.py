from __future__ import annotations
import json,re,unicodedata,pathlib,hashlib
from collections import defaultdict
import numpy as np,pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.linear_model import LogisticRegression

MC_REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
HF=f"https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/{MC_REV}/data"
SEED=20261004;CASE_DIM=256;REASON_DIM=128;RIDGE_ALPHA=10.;TRAIN_K=40;MARGIN=.03;DEPLOY_K=50;TOP_POS=5;BOTTOM_NEG=10;LOGISTIC_C=1.
OUT=pathlib.Path("e050b-output");OUT.mkdir(exist_ok=True)
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
def rank(s,k): ids=np.arange(len(s));return np.lexsort((ids,-s))[:k]
def F(sem,prd):return np.column_stack([np.asarray(sem,float),np.asarray(prd,float)])
def sf(q,c):
 S=(q@c.T).toarray()
 if not S.size:return 0.
 rec=float(np.max(S,axis=1).mean());prec=float(np.max(S,axis=0).mean())
 return 2*rec*prec/(rec+prec) if rec+prec else 0.
def uid_pmc(uid):
 p=str(uid).split("-")[0]
 return ("PMC"+p).upper() if not p.upper().startswith("PMC") else p.upper()

# Load mapping-only artifact generated before any C2 outcome analysis.
maps=list(pathlib.Path("mapping").rglob("E050_A_MAPPING_AUDIT.csv"))
pairs=list(pathlib.Path("mapping").rglob("E050_A_ALL_HUMAN_PAIRS_WITH_MAPPING.csv"))
subset=list(pathlib.Path("mapping").rglob("PMC_PATIENTS_MAPPED_SOURCE_ARTICLE_SUBSET.csv"))
assert len(maps)==len(pairs)==len(subset)==1
audit=pd.read_csv(maps[0])
pairdf=pd.read_csv(pairs[0],dtype=str).fillna("")
src=pd.read_csv(subset[0],dtype=str).fillna("")
# High-confidence rule is copied exactly from E050-A pre-outcome mapping audit.
high=audit[(audit.patients_in_source_article==1) | ((audit.candidate_is_best_text_match==True)&(audit.candidate_vs_runner_margin.fillna(1)>=0.05))].copy()
assert len(high)==121
high_uids=set(high.candidate_uid.astype(str))
uid_to_train=dict(zip(high.candidate_uid.astype(str),high.medcase_train_index.astype(int)))

# Query text comes from the human-eval patient notes embedded in E050-A pair file? Need source human file for query text.
# E050-A all-pairs file has query UID but not query text, so use human-eval JSON here BUT DO NOT RETAIN relation labels.
META_REV="0887286c5e7940545c6d3fcb907c70edc1d156b3"
URL=f"https://huggingface.co/datasets/zhengyun21/PMC-Patients-MetaData/resolve/{META_REV}/PMC-Patients_human_eval.json?download=true"
import urllib.request
req=urllib.request.Request(URL,headers={"User-Agent":"CASE-EVID-001/1.0"})
with urllib.request.urlopen(req,timeout=180) as r: human=json.load(r)
assert len(human)==606

tr=pd.read_parquet(HF+"/train-00000-of-00001.parquet",columns=["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"])
tr_pm=set(tr.pmcid.astype(str).str.upper())

# Lock candidate panel WITHOUT human relation labels.
panel=[]
for qi,q in enumerate(human):
 q_uid=str(q["human_patient_uid"]);q_pm=uid_pmc(q_uid);qtext=str(q["patient"])
 for cand_uid in q["similar_patients"].keys(): # keys only, labels deliberately ignored
  cand_uid=str(cand_uid)
  if cand_uid not in high_uids:continue
  cpm=uid_pmc(cand_uid);idx=int(uid_to_train[cand_uid])
  panel.append({
   "query_index":qi,"query_uid":q_uid,"query_pmcid":q_pm,"query_text":qtext,
   "candidate_uid":cand_uid,"candidate_pmcid":cpm,"medcase_train_index":idx,
   "query_source_in_medcase_train":q_pm in tr_pm,
   "same_source_article":q_pm==cpm
  })
assert len(panel)>0
with (OUT/"E050B_LABEL_FREE_PANEL.jsonl").open("w",encoding="utf-8") as f:
 for x in panel:f.write(json.dumps(x,ensure_ascii=False)+"\n")
panel_hash=hashlib.sha256((OUT/"E050B_LABEL_FREE_PANEL.jsonl").read_bytes()).hexdigest()

# Assert exact T001 frozen method config.
locks=list(pathlib.Path("t001").rglob("T001_RANKING_LOCK.json"));assert len(locks)==1
t001=json.load(open(locks[0]))
expected={"seed":SEED,"case_svd_dim":CASE_DIM,"reasoning_svd_dim":REASON_DIM,"ridge_alpha":RIDGE_ALPHA,
 "training_candidate_k":TRAIN_K,"utility_margin":MARGIN,"deployment_candidate_k":DEPLOY_K,
 "features":["semantic_case_similarity","predicted_reasoning_similarity"],"pairwise_top":TOP_POS,"pairwise_bottom":BOTTOM_NEG,"logistic_C":LOGISTIC_C}
for k,v in expected.items():assert t001[k]==v,(k,t001[k],v)

# Train frozen C2 exactly as T001.
isdev=np.array([int(hashlib.sha256((f"E020-FINAL-{SEED}:"+str(x)).encode()).hexdigest()[:8],16)%5==0 for x in tr.pmcid])
ref=tr.loc[~isdev].reset_index(drop=True);mq=tr.loc[isdev].reset_index(drop=True)
cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=cv.fit_transform(ref.case_prompt.astype(str));Q=cv.transform(mq.case_prompt.astype(str))
cs=TruncatedSVD(CASE_DIM,n_iter=7,random_state=SEED);XC=cs.fit_transform(X).astype(float);QC=cs.transform(Q).astype(float)
rp=[pts(r,d) for r,d in zip(ref.diagnostic_reasoning,ref.final_diagnosis)];mp=[pts(r,d) for r,d in zip(mq.diagnostic_reasoning,mq.final_diagnosis)]
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
D=np.asarray(D);Yy=np.asarray(Yy);sc=StandardScaler();z=sc.fit_transform(D);clf=LogisticRegression(C=LOGISTIC_C,max_iter=2000,random_state=SEED).fit(z,Yy)
w=clf.coef_[0]/sc.scale_;b=float(clf.intercept_[0]-np.dot(clf.coef_[0],sc.mean_/sc.scale_))

# Refit representations on all train.
cv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X2=cv2.fit_transform(tr.case_prompt.astype(str))
cs2=TruncatedSVD(CASE_DIM,n_iter=7,random_state=SEED);XC2=cs2.fit_transform(X2).astype(float)
tp=[pts(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
rv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R2=rv2.fit_transform([" ".join(x) for x in tp]);rs2=TruncatedSVD(REASON_DIM,n_iter=7,random_state=SEED);ZR2=rs2.fit_transform(R2).astype(float)
xm=XC2.mean(0,keepdims=True);ym=ZR2.mean(0,keepdims=True);AA=XC2-xm;YY=ZR2-ym;WW=np.linalg.solve(AA.T@AA+RIDGE_ALPHA*np.eye(AA.shape[1]),AA.T@YY);ZZ=normalize(ZR2)

# Score all unique queries in the locked panel, no labels available here.
query_by_index={x["query_index"]:x["query_text"] for x in panel}
qids=sorted(query_by_index)
QH=cv2.transform([query_by_index[i] for i in qids])
QHC=cs2.transform(QH).astype(float);PP=normalize((QHC-xm)@WW+ym)
CSS=(QH@X2.T).tocsr();PSS=PP@ZZ.T
qid_pos={q:i for i,q in enumerate(qids)}

scored=[]
for x in panel:
 qi=x["query_index"];pos=qid_pos[qi];j=x["medcase_train_index"]
 sem=CSS.getrow(pos).toarray().ravel()
 full_order=np.lexsort((np.arange(len(sem)),-sem))
 inv=np.empty(len(sem),dtype=int);inv[full_order]=np.arange(1,len(sem)+1)
 baseline_rank=int(inv[j])
 cand50=full_order[:DEPLOY_K]
 f=F(sem[cand50],PSS[pos,cand50]);score=f@w+b;ids=np.asarray(cand50);o=np.lexsort((ids,-score));method50=ids[o]
 if baseline_rank<=50:
  method_rank=int(np.where(method50==j)[0][0])+1
 else:
  method_rank=baseline_rank
 scored.append({**{k:x[k] for k in x if k!="query_text"},
   "baseline_rank":baseline_rank,"method_rank":method_rank,
   "rank_improvement":baseline_rank-method_rank,
   "baseline_reciprocal_rank":1.0/baseline_rank,"method_reciprocal_rank":1.0/method_rank,
   "delta_reciprocal_rank":1.0/method_rank-1.0/baseline_rank,
   "in_baseline_top50":baseline_rank<=50,
   "semantic_similarity":float(sem[j]),
   "predicted_reasoning_similarity":float(PSS[pos,j]),
   "method_raw_score":float(F([sem[j]],[PSS[pos,j]])@w+b)
 })
with (OUT/"E050B_LABEL_FREE_SCORES.jsonl").open("w",encoding="utf-8") as f:
 for x in scored:f.write(json.dumps(x,ensure_ascii=False)+"\n")
lock={"status":"human_labels_not_loaded_during_scoring","panel_pairs":len(panel),"unique_queries":len(qids),
 "panel_sha256":panel_hash,"high_conf_mapping_rule":"E050-A locked rule",
 "strict_primary_rule":"high-confidence mapping AND query source not in MedCaseReasoning train AND query/candidate different source article",
 "method":"T001 frozen C2-debiased-final-two-feature","method_config":expected,
 "human_relation_labels_available_to_scorer":[]}
(OUT/"E050B_SCORE_LOCK.json").write_text(json.dumps(lock,indent=2)+"\n")
print(json.dumps({**lock,"pairs_in_baseline_top50":sum(x["in_baseline_top50"] for x in scored)},indent=2))
