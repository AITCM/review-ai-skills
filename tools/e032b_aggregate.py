import glob,json,pathlib,numpy as np
OUT=pathlib.Path("e032b-aggregate");OUT.mkdir(exist_ok=True);rng=np.random.default_rng(20261004)
rows=[]
for p in glob.glob("downloads/shard-*/*.jsonl"):
 rows += [json.loads(x) for x in open(p) if x.strip()]
rows=sorted(rows,key=lambda x:x["query_index"])
assert len(rows)==840 and [x["query_index"] for x in rows]==list(range(840))
summary={"status":"independent_sharded_evaluator_on_locked_E031V2_rankings","n_external":840,
 "model":{"name":"pritamdeka/S-PubMedBert-MS-MARCO","revision":"96786c7024f95c5aac7f2b9a18086c7b97b23036","max_seq_length":256},"results":{},"query_dx_not_explicit":{}}
for k in (1,3,10):
 b=np.array([x[f"baseline_{k}"] for x in rows]);m=np.array([x[f"method_{k}"] for x in rows]);d=m-b;boots=[]
 for _ in range(10000):
  ids=rng.integers(0,len(d),len(d));boots.append(float(d[ids].mean()))
 summary["results"][str(k)]={"baseline":float(b.mean()),"method":float(m.mean()),"delta":float(d.mean()),
  "ci95":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))],"improved":int((d>0).sum()),"worsened":int((d<0).sum()),"tied":int((d==0).sum())}
 ids=[i for i,x in enumerate(rows) if not x["query_dx_explicit"]];summary["query_dx_not_explicit"][str(k)]={"n":len(ids),"mean_delta":float(d[ids].mean())}
summary["limits"]=["Independent dense evaluator is not clinician judgment.","Rankings are unchanged from E031-v2.","Cross-schema reasoning annotations differ between datasets."]
(OUT/"E032B_INDEPENDENT_EXTERNAL_RESULTS.json").write_text(json.dumps(summary,indent=2)+"\n")
with (OUT/"E032B_PER_CASE.jsonl").open("w") as f:
 for x in rows:f.write(json.dumps(x)+"\n")
print(json.dumps(summary,indent=2))
