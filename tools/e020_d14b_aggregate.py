import glob,json,pathlib,numpy as np
OUT=pathlib.Path("d14b-aggregate");OUT.mkdir(exist_ok=True);RNG=np.random.default_rng(20261004)
rows=[]
for p in glob.glob("downloads/shard-*/per_query.jsonl"):
 for l in open(p):rows.append(json.loads(l))
rows=sorted(rows,key=lambda x:x["query_index"])
assert len(rows)==500 and [x["query_index"] for x in rows]==list(range(500))
summary={"method":"C2-debiased-final-two-feature","metric":"independent S-PubMedBERT diagnosis-masked symmetric reasoning-set softF1","n_validation":500,"results":{}}
for k in (1,3,10):
 b=np.array([x[f"baseline_{k}"] for x in rows]);m=np.array([x[f"method_{k}"] for x in rows]);d=m-b;boots=[]
 for _ in range(5000):
  ids=RNG.integers(0,len(d),len(d));boots.append(float(d[ids].mean()))
 summary["results"][str(k)]={"baseline":float(b.mean()),"method":float(m.mean()),"delta":float(d.mean()),"ci95":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))],"improved":int((d>0).sum()),"worsened":int((d<0).sum()),"tied":int((d==0).sum())}
summary["gate_pass"]=all(summary["results"][str(k)]["delta"]>0 for k in (1,3,10)) and summary["results"]["10"]["ci95"][0]>0
(OUT/"E020_D14B_INDEPENDENT_FINAL.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
