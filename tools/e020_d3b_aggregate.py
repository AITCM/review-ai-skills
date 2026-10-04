import json,glob,numpy as np,pathlib
OUT=pathlib.Path("d3b-aggregate");OUT.mkdir(exist_ok=True);rows=[json.load(open(p)) for p in glob.glob("downloads/d3b-*/summary.json")]
out={"n_runs":len(rows),"variants":{}}
for mode in sorted(set(x["mode"] for x in rows)):
 xs=[x for x in rows if x["mode"]==mode];out["variants"][mode]={"n_seeds":len(xs)}
 for k in (1,3,10):
  a=np.array([x[f"delta@{k}"] for x in xs]);out["variants"][mode][str(k)]={"mean":float(a.mean()),"sd":float(a.std(ddof=1)),"min":float(a.min()),"max":float(a.max()),"positive_seeds":int((a>0).sum())}
(OUT/"E020_D3B_MULTISEED_MASK_CONTROLS.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
