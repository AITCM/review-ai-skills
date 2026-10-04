import json,glob,numpy as np,pathlib
OUT=pathlib.Path("d2-aggregate");OUT.mkdir(exist_ok=True)
rows=[json.load(open(p)) for p in sorted(glob.glob("downloads/d2-*/summary.json"))]
names=list(rows[0]["ablation_results"])
out={"n_seeds":len(rows),"seeds":[x["seed"] for x in rows],"ablations":{}}
for name in names:
 out["ablations"][name]={}
 for k in ("1","3","10"):
  vals=np.array([x["ablation_results"][name][k]["mean_delta"] for x in rows])
  out["ablations"][name][k]={"mean_delta":float(vals.mean()),"sd":float(vals.std(ddof=1)),"min":float(vals.min()),"max":float(vals.max()),"positive_seeds":int((vals>0).sum())}
(OUT/"E020_D2_ABLATION_SUMMARY.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
