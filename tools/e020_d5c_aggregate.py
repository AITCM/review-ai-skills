import json,glob,numpy as np,pathlib
OUT=pathlib.Path("d5c-aggregate");OUT.mkdir(exist_ok=True);rows=[json.load(open(p)) for p in glob.glob("downloads/d5c-*/summary.json")]
out={"n_seeds":len(rows),"seeds":[x["seed"] for x in rows],"metrics":{}}
for k in (1,3,10):
 a=np.array([x[f"delta@{k}"] for x in rows]);out["metrics"][str(k)]={"mean":float(a.mean()),"sd":float(a.std(ddof=1)),"min":float(a.min()),"max":float(a.max()),"positive_seeds":int((a>0).sum())}
(OUT/"E020_D5C_MULTISEED_DEBIASED.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
