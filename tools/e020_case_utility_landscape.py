#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,re,unicodedata,hashlib,urllib.request
from pathlib import Path
from collections import defaultdict
import numpy as np
DATASET="zou-lab/MedCaseReasoning"; DATA_REV="469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
FILES={"train":(13092,99332305,"12b23b1d652caff931ea53012e8b84c2e74bfd3c6394c4fcdea63afa74158207"),"val":(500,3783083,"442a60139b8b8d0526690ce3773c47c654e19a979cb95e76750eabc0e0e53b4d")}
FIELDS=["pmcid","case_prompt","diagnostic_reasoning","final_diagnosis"]
MODEL_MAP={"neuml_pubmed":("NeuML/pubmedbert-base-embeddings","sentence_transformer"),"biomedbert_mean":("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext","mean_pool"),"bioclinicalbert_mean":("emilyalsentzer/Bio_ClinicalBERT","mean_pool")}
def norm_label(s):
    s=unicodedata.normalize("NFKC",str(s)).casefold(); return " ".join(re.findall(r"[a-z0-9]+",s))
def sha256(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def download(url,dest,size,digest):
    dest=Path(dest); dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and dest.stat().st_size==size and sha256(dest)==digest:return
    tmp=dest.with_suffix(dest.suffix+".part")
    req=urllib.request.Request(url,headers={"User-Agent":"case-utility-landscape/0.1"})
    with urllib.request.urlopen(req,timeout=120) as r,tmp.open("wb") as f:
        h=hashlib.sha256();n=0
        while True:
            b=r.read(1<<20)
            if not b:break
            n+=len(b);h.update(b);f.write(b)
    if n!=size or h.hexdigest()!=digest:
        tmp.unlink(missing_ok=True);raise RuntimeError("checksum mismatch")
    tmp.replace(dest)
def load_data(cache):
    import pyarrow.parquet as pq
    out={}
    for split,(rows,size,digest) in FILES.items():
        name=f"{split}-00000-of-00001.parquet";p=Path(cache)/name
        download(f"https://huggingface.co/datasets/{DATASET}/resolve/{DATA_REV}/data/{name}?download=true",p,size,digest)
        tab=pq.read_table(p,columns=FIELDS)
        if tab.num_rows!=rows:raise RuntimeError("row mismatch")
        out[split]=tab.to_pylist()
    return out
def dj(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def djl(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8") as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n")
def cohort(train,val):
    m=defaultdict(list)
    for i,r in enumerate(train):m[norm_label(r["final_diagnosis"])].append(i)
    seen=np.array([norm_label(r["final_diagnosis"]) in m for r in val],dtype=bool)
    return m,seen
def metrics(idx,sc,train,val,seen,name):
    ks=[1,3,5,10,50];ranks=[];per=[];hard=[];easy=[];labs=[norm_label(r["final_diagnosis"]) for r in train]
    for qi,r in enumerate(val):
        q=norm_label(r["final_diagnosis"]); ls=[labs[j] for j in idx[qi]]; same=[j for j,l in enumerate(ls) if l==q]; rank=same[0]+1 if same else None;ranks.append(rank)
        per.append({"query_index":qi,"pmcid":r["pmcid"],"diagnosis":r["final_diagnosis"],"label_seen_in_train":bool(seen[qi]),"first_exact_label_rank":rank,"top_candidates":[{"rank":k+1,"train_index":int(idx[qi,k]),"pmcid":train[int(idx[qi,k])]["pmcid"],"diagnosis":train[int(idx[qi,k])]["final_diagnosis"],"score":float(sc[qi,k])} for k in range(min(10,idx.shape[1]))]})
        if seen[qi] and ls and ls[0]!=q and rank is not None:
            hard.append({"query_index":qi,"query_pmcid":r["pmcid"],"query_diagnosis":r["final_diagnosis"],"wrong_top1_train_index":int(idx[qi,0]),"wrong_top1_diagnosis":train[int(idx[qi,0])]["final_diagnosis"],"wrong_top1_score":float(sc[qi,0]),"first_positive_rank":rank,"first_positive_train_index":int(idx[qi,rank-1]),"first_positive_diagnosis":train[int(idx[qi,rank-1])]["final_diagnosis"],"first_positive_score":float(sc[qi,rank-1]),"score_margin_wrong_minus_positive":float(sc[qi,0]-sc[qi,rank-1])})
        if seen[qi] and rank is not None and rank<=10: easy.append({"query_index":qi,"query_pmcid":r["pmcid"],"query_diagnosis":r["final_diagnosis"],"positive_rank":rank,"positive_train_index":int(idx[qi,rank-1]),"positive_score":float(sc[qi,rank-1])})
    n=int(seen.sum());s={"method":name,"queries_all":len(val),"queries_exact_label_seen_in_train":n,"exact_label_seen_fraction":n/len(val)}
    for k in ks:
        s[f"exact_label_hit_at_{k}_all"]=sum(x is not None and x<=k for x in ranks)/len(ranks)
        s[f"exact_label_hit_at_{k}_seen"]=sum(seen[i] and ranks[i] is not None and ranks[i]<=k for i in range(len(ranks)))/n
    s["mrr_at_50_seen"]=sum((1/ranks[i]) if seen[i] and ranks[i] and ranks[i]<=50 else 0 for i in range(len(ranks)))/n
    s["top1_wrong_with_recoverable_positive_at_50"]=len(hard)
    hard.sort(key=lambda x:x["score_margin_wrong_minus_positive"],reverse=True);easy.sort(key=lambda x:x["positive_score"],reverse=True)
    return s,per,hard,easy
def lexical(train,val,out):
    from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
    tr=[str(r["case_prompt"]) for r in train];va=[str(r["case_prompt"]) for r in val];labels,seen=cohort(train,val)
    v=TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True,min_df=2,max_df=.98,max_features=80000,dtype=np.float32)
    X=v.fit_transform(tr);Q=v.transform(va);S=(Q@X.T).toarray();I=np.argsort(-S,axis=1)[:,:50];SC=np.take_along_axis(S,I,axis=1)
    s,p,h,e=metrics(I,SC,train,val,seen,"tfidf");dj(out/"tfidf_summary.json",s);djl(out/"tfidf_per_query.jsonl",p);djl(out/"tfidf_hard_negatives.jsonl",h[:150]);djl(out/"tfidf_easy_positives.jsonl",e[:100])
    cv=CountVectorizer(lowercase=True,min_df=2,max_df=.98,max_features=100000,dtype=np.float32);C=cv.fit_transform(tr).tocsr();CQ=cv.transform(va).tocsr()
    N=C.shape[0];dl=np.asarray(C.sum(axis=1)).ravel();avg=float(dl.mean());df=np.asarray((C>0).sum(axis=0)).ravel();idf=np.log(1+(N-df+.5)/(df+.5)).astype(np.float32);k1=1.5;b=.75;dn=(k1*(1-b+b*dl/avg)).astype(np.float32)
    W=C.copy().astype(np.float32)
    for i in range(W.shape[0]):
        a,z=W.indptr[i],W.indptr[i+1];vals=W.data[a:z];W.data[a:z]=((vals*(k1+1))/(vals+dn[i]))*idf[W.indices[a:z]]
    CQ.data[:]=1.0;S=(CQ@W.T).toarray();I=np.argsort(-S,axis=1)[:,:50];SC=np.take_along_axis(S,I,axis=1)
    s,p,h,e=metrics(I,SC,train,val,seen,"bm25");dj(out/"bm25_summary.json",s);djl(out/"bm25_per_query.jsonl",p);djl(out/"bm25_hard_negatives.jsonl",h[:150]);djl(out/"bm25_easy_positives.jsonl",e[:100])
    dj(out/"cohort.json",{"train_rows":len(train),"val_rows":len(val),"seen_exact_label_queries":int(seen.sum()),"unseen_exact_label_queries":int((~seen).sum()),"unique_train_normalized_diagnoses":len(labels)})
def mean_encode(name,rev,texts,maxlen):
    import torch
    from transformers import AutoTokenizer,AutoModel
    tok=AutoTokenizer.from_pretrained(name,revision=rev);model=AutoModel.from_pretrained(name,revision=rev);model.eval();out=[];at=0;lens=[]
    with torch.inference_mode():
        for i in range(0,len(texts),16):
            enc=tok(texts[i:i+16],padding=True,truncation=True,max_length=maxlen,return_tensors="pt",return_length=True);lens+=enc["length"].tolist();at+=sum(int(x>=maxlen) for x in enc["length"].tolist());mi={k:v for k,v in enc.items() if k!="length"};z=model(**mi).last_hidden_state;m=mi["attention_mask"].unsqueeze(-1).float();e=(z*m).sum(1)/m.sum(1).clamp(min=1e-9);e=torch.nn.functional.normalize(e,p=2,dim=1);out.append(e.numpy().astype("float32"))
    return np.vstack(out),{"max_length":maxlen,"sequences_at_token_cap":at,"mean_encoded_length":float(np.mean(lens))}
def st_encode(name,rev,texts,maxlen):
    from sentence_transformers import SentenceTransformer
    m=SentenceTransformer(name,revision=rev,device="cpu");m.max_seq_length=maxlen;e=m.encode(texts,batch_size=32,show_progress_bar=True,normalize_embeddings=True,convert_to_numpy=True).astype("float32");lens=[];at=0
    for i in range(0,len(texts),128):
        x=m.tokenizer(texts[i:i+128],padding=False,truncation=True,max_length=maxlen,return_length=True);lens+=x["length"];at+=sum(int(v>=maxlen) for v in x["length"])
    return e,{"max_length":maxlen,"sequences_at_token_cap":at,"mean_encoded_length":float(np.mean(lens))}
def strip_reason(r):
    text=str(r["diagnostic_reasoning"]);diag=str(r["final_diagnosis"]).strip()
    if not diag:return text,False
    new,n=re.subn(re.escape(diag)," [DIAGNOSIS_REMOVED] ",text,flags=re.I);return new,bool(n)
def dense(train,val,out,key,maxlen):
    from huggingface_hub import HfApi
    name,kind=MODEL_MAP[key];rev=HfApi().model_info(name).sha;enc=st_encode if kind=="sentence_transformer" else mean_encode;labels,seen=cohort(train,val)
    A,sa=enc(name,rev,[str(r["case_prompt"]) for r in train],maxlen);B,sb=enc(name,rev,[str(r["case_prompt"]) for r in val],maxlen);S=B@A.T;I=np.argsort(-S,axis=1)[:,:50];SC=np.take_along_axis(S,I,axis=1)
    s,p,h,e=metrics(I,SC,train,val,seen,key);s["encoding"]={"model":name,"revision":rev,"kind":kind,"train":sa,"val":sb};dj(out/f"{key}_summary.json",s);djl(out/f"{key}_per_query.jsonl",p);djl(out/f"{key}_hard_negatives.jsonl",h[:150]);djl(out/f"{key}_easy_positives.jsonl",e[:100])
    if key=="neuml_pubmed":
        trr=[];var=[];rt=rv=0
        for r in train:x,ok=strip_reason(r);trr.append(x);rt+=ok
        for r in val:x,ok=strip_reason(r);var.append(x);rv+=ok
        RA,rsa=enc(name,rev,trr,maxlen);RB,rsb=enc(name,rev,var,maxlen);R=RB@RA.T;RI=np.argsort(-R,axis=1)[:,:50];RSC=np.take_along_axis(R,RI,axis=1);sr,pr,_,_=metrics(RI,RSC,train,val,seen,"neuml_reasoning_scrubbed")
        tl=[norm_label(r["final_diagnosis"]) for r in train];ov=[];hp=[]
        for qi,r in enumerate(val):
            a=set(map(int,I[qi,:10]));b=set(map(int,RI[qi,:10]));ov.append(len(a&b)/len(a|b) if a|b else 1);q=norm_label(r["final_diagnosis"]);case50=set(map(int,I[qi,:50]))
            for rk,j in enumerate(RI[qi,:10],1):
                j=int(j)
                if tl[j]==q and j not in case50:hp.append({"query_index":qi,"query_pmcid":r["pmcid"],"query_diagnosis":r["final_diagnosis"],"train_index":j,"candidate_diagnosis":train[j]["final_diagnosis"],"reasoning_rank":rk,"reasoning_score":float(R[qi,j]),"case_text_rank_gt":50});break
        sr.update({"case_vs_reasoning_top10_mean_jaccard":float(np.mean(ov)),"reasoning_exact_diagnosis_removed_train":rt,"reasoning_exact_diagnosis_removed_val":rv,"reasoning_encoding":{"train":rsa,"val":rsb}});dj(out/"neuml_reasoning_scrubbed_summary.json",sr);djl(out/"neuml_reasoning_scrubbed_per_query.jsonl",pr);djl(out/"reasoning_positive_candidates.jsonl",hp[:200])
def main():
    a=argparse.ArgumentParser();a.add_argument("--backend",required=True,choices=["lexical",*MODEL_MAP]);a.add_argument("--out",required=True);a.add_argument("--cache",default="cache");a.add_argument("--max-length",type=int,default=256);x=a.parse_args();out=Path(x.out);out.mkdir(parents=True,exist_ok=True);d=load_data(Path(x.cache));train,val=d["train"],d["val"];dj(out/"RECEIPT.json",{"backend":x.backend,"dataset":DATASET,"dataset_revision":DATA_REV,"train_rows":len(train),"val_rows":len(val),"test_accessed":False,"primary_label_definition":"NFKC+casefold+alphanumeric exact final_diagnosis","candidate_pair_labels_are_not_clinical_gold":True})
    lexical(train,val,out) if x.backend=="lexical" else dense(train,val,out,x.backend,x.max_length)
if __name__=="__main__":main()
