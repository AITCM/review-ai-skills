from __future__ import annotations
import json,re,unicodedata,pathlib,hashlib
import numpy as np,pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

CRB_REV="3297c5b2db51872b70645a75f43dc0b849f77621"
CRB=f"https://huggingface.co/datasets/cxyzhang/caseReportBench_ClinicalDenseExtraction_Benchmark/resolve/{CRB_REV}/data/train-00000-of-00001.parquet?download=true"
MC_REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339";HF=f"https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/{MC_REV}/data"
OUT=pathlib.Path("e060d");OUT.mkdir(exist_ok=True)
SEED=20261004;CASE_DIM=256;REASON_DIM=128;RIDGE_ALPHA=10.;TRAIN_K=40;MARGIN=.03;DEPLOY_K=50;TOP_POS=5;BOTTOM_NEG=10;LOGISTIC_C=1.
MODEL="pritamdeka/S-PubMedBert-MS-MARCO";MODEL_REV="96786c7024f95c5aac7f2b9a18086c7b97b23036";RNG=np.random.default_rng(SEED)
CATS=["Vitals_Hema","GI","History","Neuro","Lab_Image","CVS","ENDO","GU","RESP","MSK","EENT","DERM","Pregnancy","LYMPH"]
GENERIC={"disease","syndrome","disorder","type","deficiency","defect","with","without","of","and","the","primary","secondary","congenital","acute","chronic"}
PAT=re.compile(r'^\s*(?:\((\d+)\)|(\d+)[.)])\s*',re.M)

def norm(s):return " ".join(re.findall(r"\w+",unicodedata.normalize("NFKC",str(s)).casefold()))
def flatten(v):
 if v is None:return ""
 if isinstance(v,np.ndarray):v=v.tolist()
 if isinstance(v,(list,tuple,set)):return " ".join(flatten(x) for x in v if flatten(x))
 if isinstance(v,dict):return " ".join(flatten(x) for x in v.values() if flatten(x))
 try:
  if bool(pd.isna(v)):return ""
 except Exception:pass
 return str(v).strip()
def as_items(v):
 if v is None:return []
 if isinstance(v,np.ndarray):v=v.tolist()
 if isinstance(v,(list,tuple,set)):return [flatten(x) for x in v if flatten(x)]
 if isinstance(v,dict):return [flatten(x) for x in v.values() if flatten(x)]
 try:
  if bool(pd.isna(v)):return []
 except Exception:pass
 s=str(v).strip();return [s] if s and s.lower() not in {"nan","none","[]","{}","null"} else []
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
def aggressive_mask(text,dx_items):
 x=str(text)
 for d in sorted(dx_items,key=len,reverse=True):
  if d.strip():x=re.sub(re.escape(d.strip())," diagnosismask ",x,flags=re.I)
 toks=set()
 for d in dx_items:toks|={t for t in norm(d).split() if len(t)>=4 and t not in GENERIC}
 return " ".join("diagnosismask" if t in toks else t for t in norm(x).split())
def reason_points(reason,dx):
 return [aggressive_mask(x,[str(dx)]) for x in split_reason(reason)]
def topk(s,k):ids=np.arange(len(s));return np.lexsort((ids,-s))[:k]
def F(sem,prd):return np.column_stack([np.asarray(sem,float),np.asarray(prd,float)])
def sf(q,c):
 S=q@c.T
 rec=float(np.max(S,axis=1).mean());prec=float(np.max(S,axis=0).mean())
 return 2*rec*prec/(rec+prec) if rec+prec else 0.

df=pd.read_parquet(CRB);tr=pd.read_parquet(HF+"/train-00000-of-00001.parquet",columns=["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"])
# frozen T001 method identity
locks=list(pathlib.Path("t001").rglob("T001_RANKING_LOCK.json"));assert len(locks)==1
t001=json.load(open(locks[0]))
expected={"seed":SEED,"case_svd_dim":CASE_DIM,"reasoning_svd_dim":REASON_DIM,"ridge_alpha":RIDGE_ALPHA,
 "training_candidate_k":TRAIN_K,"utility_margin":MARGIN,"deployment_candidate_k":DEPLOY_K,
 "features":["semantic_case_similarity","predicted_reasoning_similarity"],"pairwise_top":TOP_POS,"pairwise_bottom":BOTTOM_NEG,"logistic_C":LOGISTIC_C}
for k,v in expected.items():assert t001[k]==v

# Train frozen ranker exactly as T001
isdev=np.array([int(hashlib.sha256((f"E020-FINAL-{SEED}:"+str(x)).encode()).hexdigest()[:8],16)%5==0 for x in tr.pmcid])
ref=tr.loc[~isdev].reset_index(drop=True);mq=tr.loc[isdev].reset_index(drop=True)
cv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X=cv.fit_transform(ref.case_prompt.astype(str));Q=cv.transform(mq.case_prompt.astype(str))
cs=TruncatedSVD(CASE_DIM,n_iter=7,random_state=SEED);XC=cs.fit_transform(X);QC=cs.transform(Q)
rp=[reason_points(r,d) for r,d in zip(ref.diagnostic_reasoning,ref.final_diagnosis)];mp=[reason_points(r,d) for r,d in zip(mq.diagnostic_reasoning,mq.final_diagnosis)]
rv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R=rv.fit_transform([" ".join(x) for x in rp]);rs=TruncatedSVD(REASON_DIM,n_iter=7,random_state=SEED);ZR=rs.fit_transform(R)
xm=XC.mean(0,keepdims=True);ym=ZR.mean(0,keepdims=True);A=XC-xm;Y=ZR-ym
W=np.linalg.solve(A.T@A+RIDGE_ALPHA*np.eye(A.shape[1]),A.T@Y);P=normalize((QC-xm)@W+ym);ZI=normalize(ZR);CS=(Q@X.T).tocsr();PS=P@ZI.T
pv=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32)
PM=pv.fit_transform([p for z in rp for p in z]+[p for z in mp for p in z]);off=0;RPM=[]
for z in rp:RPM.append(PM[off:off+len(z)]);off+=len(z)
MPM=[]
for z in mp:MPM.append(PM[off:off+len(z)]);off+=len(z)
D=[];Yy=[]
for i in range(len(mq)):
 s=CS.getrow(i).toarray().ravel();cand=topk(s,TRAIN_K);f=F(s[cand],PS[i,cand]);u=np.array([sf(MPM[i].toarray(),RPM[int(j)].toarray()) for j in cand])
 for p in np.argsort(-u)[:TOP_POS]:
  for n in np.argsort(u)[:BOTTOM_NEG]:
   if u[p]-u[n]<MARGIN:continue
   d=f[p]-f[n];D.append(d);Yy.append(1);D.append(-d);Yy.append(0)
D=np.asarray(D);Yy=np.asarray(Yy);sc=StandardScaler();Z=sc.fit_transform(D);clf=LogisticRegression(C=LOGISTIC_C,max_iter=2000,random_state=SEED).fit(Z,Yy)
w=clf.coef_[0]/sc.scale_;b=float(clf.intercept_[0]-np.dot(clf.coef_[0],sc.mean_/sc.scale_))

# all-train representations
cv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
X2=cv2.fit_transform(tr.case_prompt.astype(str));cs2=TruncatedSVD(CASE_DIM,n_iter=7,random_state=SEED);XC2=cs2.fit_transform(X2)
tp=[reason_points(r,d) for r,d in zip(tr.diagnostic_reasoning,tr.final_diagnosis)]
rv2=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
R2=rv2.fit_transform([" ".join(x) for x in tp]);rs2=TruncatedSVD(REASON_DIM,n_iter=7,random_state=SEED);ZR2=rs2.fit_transform(R2)
xm=XC2.mean(0,keepdims=True);ym=ZR2.mean(0,keepdims=True);AA=XC2-xm;YY=ZR2-ym;WW=np.linalg.solve(AA.T@AA+RIDGE_ALPHA*np.eye(AA.shape[1]),AA.T@YY);ZZ=normalize(ZR2)

# masked query variants and target facts
queries=[];confirmed=[]
target_text=[];target_keys=[];target={}
for i,r in df.iterrows():
 dx=as_items(r["Confirmed_Diagnosis(IEM)"])
 raw=flatten(r["text"]);masked=aggressive_mask(raw,dx)
 queries.append(masked);confirmed.append(bool(dx));target[i]=[]
 for c in CATS:
  for x in as_items(r[c]):
   xx=aggressive_mask(x,dx)
   if xx.strip():target_keys.append((i,c));target_text.append(xx)
assert sum(confirmed)==122
QE=cv2.transform(queries);QC2=cs2.transform(QE);PP=normalize((QC2-xm)@WW+ym);CSS=(QE@X2.T).tocsr();PSS=PP@ZZ.T
rankings=[]
for i in range(138):
 s=CSS.getrow(i).toarray().ravel();cand=topk(s,DEPLOY_K);score=F(s[cand],PSS[i,cand])@w+b;ids=np.asarray(cand);o=np.lexsort((ids,-score));rankings.append((cand.tolist(),ids[o].tolist()))

needed=set()
for a,b0 in rankings:
 needed.update(a[:10]);needed.update(b0[:10])
hist={};hist_text=[];hist_keys=[]
for j in sorted(needed):
 hist[j]=[]
 for p in tp[j]:hist_keys.append(j);hist_text.append(p)
texts=hist_text+target_text
model=SentenceTransformer(MODEL,revision=MODEL_REV,device="cpu");model.max_seq_length=256
E=model.encode(texts,batch_size=64,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True)
for vec,j in zip(E[:len(hist_text)],hist_keys):hist[j].append(vec)
for vec,(i,c) in zip(E[len(hist_text):],target_keys):target[i].append(vec)

def f1score(i,cands):
 H=np.vstack([np.asarray(hist[j]) for j in cands]);T=np.asarray(target[i]);S=T@H.T
 rec=float(np.max(S,axis=1).mean());prec=float(np.max(S,axis=0).mean())
 return 2*rec*prec/(rec+prec) if rec+prec else 0.

ids=np.array([i for i,x in enumerate(confirmed) if x],dtype=int)
summary={"status":"posthoc_aggressive_diagnosis_mask_query_sensitivity","n_confirmed_dx":len(ids),
 "query_variant":"full CaseReportBench text with exact confirmed diagnosis phrases and diagnosis-specific tokens aggressively masked",
 "note":"Sensitivity analysis after E060B; not used for method selection.","results":{}}
for k in (1,3,10):
 a=[];m=[]
 for i in ids:
  br,mr=rankings[i];a.append(f1score(i,br[:k]));m.append(f1score(i,mr[:k]))
 a=np.asarray(a);m=np.asarray(m);d=m-a;boots=[]
 for _ in range(5000):
  ix=RNG.integers(0,len(d),len(d));boots.append(float(d[ix].mean()))
 summary["results"][str(k)]={"TFIDF":float(a.mean()),"C2":float(m.mean()),"delta":float(d.mean()),
  "ci95":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))],
  "improved":int((d>0).sum()),"worsened":int((d<0).sum()),"tied":int((d==0).sum())}
(OUT/"E060D_MASKED_QUERY_SENSITIVITY.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
