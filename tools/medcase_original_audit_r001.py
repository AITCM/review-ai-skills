"""R001: source-grounded audit of MedCaseReasoning; no model scores are generated.
Downloads public primary sources and the pinned original dataset; records exact
file identities. Test-set access is restricted to schema and bibliographic IDs.
"""
from __future__ import annotations
import collections, concurrent.futures, hashlib, json, pathlib, re, sys, urllib.request
from datetime import datetime, timezone
import fitz
import pyarrow.parquet as pq

ROOT = pathlib.Path('r001-output')
ROOT.mkdir(exist_ok=True)
REV = '469a5365bc534b5b2b9cfbc52b2ef2a10f43a339'
UPSTREAM = 'a64ec5ff1cb1fde7914a7bab3b1d9ab2b0546794'
SPECS = {
 'train': (13092, 99332305, '12b23b1d652caff931ea53012e8b84c2e74bfd3c6394c4fcdea63afa74158207'),
 'val': (500, 3783083, '442a60139b8b8d0526690ce3773c47c654e19a979cb95e76750eabc0e0e53b4d'),
 'test': (897, 7346727, '353b33113cd504a870430f59d4149f1cc9e1c5561523ba745031719548b1c643')}
receipt = {'experiment': 'R001', 'scope': 'paper-source-and-released-data-audit', 'created_utc': datetime.now(timezone.utc).isoformat(), 'sources': [], 'errors': [], 'model_inference': False, 'paid_api_calls': 0}

def save_json(name, value):
 p = ROOT/name
 p.parent.mkdir(parents=True, exist_ok=True)
 p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def download(url, name, expected=None):
 p = ROOT/name
 p.parent.mkdir(parents=True, exist_ok=True)
 req = urllib.request.Request(url, headers={'User-Agent':'MedCaseReasoning-reproduction-source-audit/1.0'})
 h = hashlib.sha256()
 with urllib.request.urlopen(req, timeout=120) as r, p.open('wb') as f:
  while b := r.read(1048576):
   f.write(b); h.update(b)
 if expected and h.hexdigest()!=expected:
  raise RuntimeError('SHA256 mismatch: '+name)
 item = {'url':url,'path':name,'bytes':p.stat().st_size,'sha256':h.hexdigest()}
 receipt['sources'].append(item)
 return p

sources = {
 'sources/paper_arxiv_v2.pdf':'https://arxiv.org/pdf/2505.11733v2',
 'sources/Supplemental_Material.pdf':f'https://raw.githubusercontent.com/kevinwu23/Stanford-MedCaseReasoning/{UPSTREAM}/Supplemental_Material.pdf',
 'sources/upstream_README.md':f'https://raw.githubusercontent.com/kevinwu23/Stanford-MedCaseReasoning/{UPSTREAM}/README.md',
 'sources/upstream_tree.json':f'https://api.github.com/repos/kevinwu23/Stanford-MedCaseReasoning/git/trees/{UPSTREAM}?recursive=1',
 'sources/dataset_README.md':f'https://huggingface.co/datasets/zou-lab/MedCaseReasoning/raw/{REV}/README.md',
 'sources/verl_fsdp_sft_trainer.py':'https://raw.githubusercontent.com/volcengine/verl/v0.3.0.rc0/verl/trainer/fsdp_sft_trainer.py',
 'sources/verl_sft_trainer.yaml':'https://raw.githubusercontent.com/volcengine/verl/v0.3.0.rc0/verl/trainer/config/sft_trainer.yaml',
 'sources/qwen_base_metadata.json':'https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct',
 'sources/qwen_fp16_gguf_metadata.json':'https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct-GGUF?blobs=true',
 'sources/sft_medreason_metadata.json':'https://huggingface.co/api/models/zou-lab/MedCaseReasoning-medreason-8b?blobs=true',
 'sources/author_model_inventory.json':'https://huggingface.co/api/models?author=zou-lab&limit=100&full=true'}
for name, url in sources.items():
 try:
  download(url,name)
  print('SOURCE_OK',name,flush=True)
 except Exception as exc:
  receipt['errors'].append({'path':name,'error':type(exc).__name__+': '+str(exc)})
  print('SOURCE_FAILED',name,type(exc).__name__,flush=True)

for name in ['sources/paper_arxiv_v2.pdf','sources/Supplemental_Material.pdf']:
 p=ROOT/name
 if not p.exists(): continue
 doc=fitz.open(p)
 texts=[]
 for i,page in enumerate(doc):
  t=page.get_text(sort=True)
  texts.append(f'\n=== PDF PAGE {i+1} ===\n'+t)
  if p.stem=='paper_arxiv_v2' and (i in [0,1,3,4,5,10,11,12,17,18]):
   q=ROOT/'renders'/f'paper_page_{i+1:02d}.png';q.parent.mkdir(exist_ok=True)
   page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(q)
  if p.stem=='Supplemental_Material':
   q=ROOT/'renders'/f'supplement_page_{i+1:02d}.png';q.parent.mkdir(exist_ok=True)
   page.get_pixmap(matrix=fitz.Matrix(1.25,1.25)).save(q)
 out=p.with_suffix('.txt');out.write_text('\n'.join(texts),encoding='utf-8')
 receipt.setdefault('pdfs',[]).append({'file':name,'pages':len(doc),'text_file':str(out.relative_to(ROOT))})

schema_stats={}; bibliography=[]; trainval=[]
for split,(rows,size,sha) in SPECS.items():
 name=f'data/{split}-00000-of-00001.parquet'
 p=download(f'https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/{REV}/{name}?download=true', name, sha)
 if p.stat().st_size!=size: raise RuntimeError('Byte mismatch '+split)
 pf=pq.ParquetFile(p)
 if pf.metadata.num_rows!=rows: raise RuntimeError('Row mismatch '+split)
 columns=pf.schema_arrow.names
 meta=[c for c in columns if c=='pmcid' or 'journal' in c.lower() or 'date' in c.lower() or c.lower() in ['year','publication_year']]
 records=pf.read(columns=meta).to_pylist()
 for record in records: bibliography.append({'split':split,**record})
 schema_stats[split]={'rows':rows,'bytes':size,'sha256':sha,'columns':columns,'bibliographic_columns_read':meta}
 if split!='test':
  fields=['pmcid','case_prompt','diagnostic_reasoning','final_diagnosis']
  for idx,record in enumerate(pf.read(columns=fields).to_pylist()):
   trainval.append({'split':split,'source_row_index':idx,**record})
  schema_stats[split]['field_nulls']={c:sum(not r.get(c) for r in trainval if r['split']==split) for c in fields}
 p.unlink()  # originals are already archived elsewhere; do not duplicate in artifact
 print('DATA_VERIFIED',split,rows,flush=True)
save_json('results/schema_stats.json',schema_stats)
save_json('results/bibliographic_metadata.json',bibliography)
# Preserve the existing train/val field projection for local protocol preparation.
import gzip
with gzip.open(ROOT/'results/train_val_fields.jsonl.gz','wt',encoding='utf-8') as f:
 for r in trainval: f.write(json.dumps(r,ensure_ascii=False)+'\n')
# Compute journal prevalence without guessing or normalizing labels.
for col in sorted({k for r in bibliography for k in r if 'journal' in k.lower()}):
 counts=collections.Counter(str(r.get(col,'')) for r in bibliography)
 save_json('results/journal_'+re.sub(r'[^A-Za-z0-9_]','_',col)+'.json',{'column':col,'denominator':len(bibliography),'unique_including_missing':len(counts),'top20':[{'journal':k,'count':v,'percent':100*v/len(bibliography)} for k,v in counts.most_common(20)]})
paths=[]
tree_path=ROOT/'sources/upstream_tree.json'
if tree_path.exists():
 tree=json.loads(tree_path.read_text()); paths=[x['path'] for x in tree.get('tree',[])]
 required=['prompts.py','stitch_reasoning.py','evaluate.py','finetune/train_sft.py','environment.yml']
 save_json('results/upstream_availability.json',{'commit':UPSTREAM,'tree_truncated':tree.get('truncated'),'all_paths':paths,'required_files':{k:k in paths for k in required}})
receipt['status']='completed_source_and_data_audit'
receipt['test_access']='schema, row count and bibliographic metadata only; no test case prompts, diagnoses or reasoning loaded'
receipt['runtime']={'python':sys.version,'pyarrow':__import__('pyarrow').__version__,'pymupdf':fitz.VersionBind}
save_json('R001_RECEIPT.json',receipt)
print('R001_COMPLETE',flush=True)
