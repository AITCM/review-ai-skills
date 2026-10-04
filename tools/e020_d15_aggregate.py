import glob,json,numpy as np,pathlib
OUT=pathlib.Path("d15-aggregate");OUT.mkdir(exist_ok=True);rows=[json.load(open(p)) for p in glob.glob("downloads/d15-*/summary.json")]
out={"n_runs":len(rows),"feature_modes":{}}
for mode in sorted(set(x["features"] for x in rows)):
 xs=[x for x in rows if x["features"]==mode];out["feature_modes"][mode]={"n_seeds":len(xs)}
 for k in (1,3,10):
  a=np.array([x[f"delta@{k}"] for x in xs]);out["feature_modes"][mode][str(k)]={"mean":float(a.mean()),"sd":float(a.std(ddof=1)),"min":float(a.min()),"max":float(a.max()),"positive_seeds":int((a>0).sum())}
(OUT/"E020_D15_FINAL_FEATURE_ABLATION.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
