import glob,json,numpy as np,pathlib
OUT=pathlib.Path("d16-aggregate");OUT.mkdir(exist_ok=True);rows=[json.load(open(p)) for p in glob.glob("downloads/d16-*/summary.json")]
out={"n_runs":len(rows),"mask_modes":{}}
for mode in sorted(set(x["mask"] for x in rows)):
 xs=[x for x in rows if x["mask"]==mode];out["mask_modes"][mode]={"n_seeds":len(xs)}
 for k in (1,3,10):
  a=np.array([x[f"delta@{k}"] for x in xs]);out["mask_modes"][mode][str(k)]={"mean":float(a.mean()),"sd":float(a.std(ddof=1)),"min":float(a.min()),"max":float(a.max()),"positive_seeds":int((a>0).sum())}
(OUT/"E020_D16_FINAL_MASK_MULTISEED.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
