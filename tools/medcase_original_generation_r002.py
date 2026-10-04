"""R002: a bounded validation-only check of the paper's diagnostic generation protocol.
Uses the original Qwen2.5-7B-Instruct in the official unquantized FP16 GGUF format.
It is not a reproduction of the 897-case performance table. No judge/API/SFT run.
The paper does not specify backend, seed, output cap, or ancillary samplers;
these implementation choices are explicit in RECEIPT.json, not claimed original.
"""
from __future__ import annotations
import hashlib,json,os,platform,re,subprocess,tarfile,time,urllib.request,concurrent.futures
from pathlib import Path
from datetime import datetime,timezone
import fitz
import pyarrow.parquet as pq
OUT=Path('r002-output');CACHE=Path('r002-cache')
DATA_REV='469a5365bc534b5b2b9cfbc52b2ef2a10f43a339'
MODEL='Qwen/Qwen2.5-7B-Instruct-GGUF'
REV='bb5d59e06d9551d752d08b292a50eb208b07ab1f'
SPECS=[('00001',3951521376,'134386a760a6b4b69dfe7e7139d96b56c490455e0f66cfb53ce5ec406a5dca6f'),('00002',3864909312,'d62b2aebad493fc1dc37cf04f264a370d1557a710f5ed34fb4c66d180c823cc4'),('00003',3864894976,'09176e706fb2ef2bf4d65529ee32ae0549e00e517bdfd54fcc5cc9919b59995c'),('00004',3556527872,'cd60424ae5289f18906d36272d44e93de49c3ddd3f88826a2f24ce1e06502f97')]
SALT='MedCaseReasoning-R002-validation-protocol-check-v1:'
def now():return datetime.now(timezone.utc).isoformat()
def save(name,x):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def get(url):
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'MedCase-original-protocol/1.0'}),timeout=60) as r:return json.load(r)
def dl(url,path,sha=None,size=None):
 h=hashlib.sha256();n=0
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'MedCase-original-protocol/1.0'}),timeout=180) as r,path.open('wb') as f:
  while b:=r.read(1048576):
   f.write(b);h.update(b);n+=len(b)
   if size and n>size:raise ValueError('Unexpected response size')
 if sha and h.hexdigest()!=sha:raise ValueError('SHA mismatch '+path.name)
 if size and n!=size:raise ValueError('Size mismatch '+path.name)
 return {'file':path.name,'bytes':n,'sha256':h.hexdigest(),'source':url}
def post(path,x,timeout=600):
 req=urllib.request.Request('http://127.0.0.1:8080'+path,data=json.dumps(x).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=timeout) as r:return json.load(r)
def main():
 OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
 receipt={'experiment':'R002','status':'running','started_utc':now(),'model_name':'Qwen2.5-7B-Instruct','model_repo':MODEL,'model_revision':REV,'weight_precision':'official FP16 GGUF; no Q4/Q8 weight quantization','paper_settings':{'generations_per_case':10,'temperature':0.8,'top_p':0.95,'prompt':'arxiv v2 PDF Prompt 6','retrieval':False},'implementation_choices_not_specified_by_paper':{'backend':'llama.cpp b10964 CPU','seed_base':20261004,'n_predict':1024,'context':4096,'top_k':0,'min_p':0,'repeat_penalty':1.0,'chat_template':'official Qwen tokenizer chat template via server','prompt_cache':True},'scope':'1 prespecified validation case, not official test performance','selection_salt':SALT,'test_set_accessed':False,'diagnostic_judge':'not_called','reasoning_judge':'not_called','paid_api_calls':0,'predictions_completed':0,'completed_xml':0,'files':[]}
 save('RECEIPT.json',receipt)
 fn='val-00000-of-00001.parquet'
 receipt['files'].append(dl(f'https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/{DATA_REV}/data/{fn}',CACHE/fn,'442a60139b8b8d0526690ce3773c47c654e19a979cb95e76750eabc0e0e53b4d',3783083))
 rows=pq.read_table(CACHE/fn,columns=['pmcid','case_prompt']).to_pylist()
 if len(rows)!=500:raise ValueError('Unexpected validation rows')
 i=min(range(len(rows)),key=lambda j:hashlib.sha256((SALT+str(rows[j]['pmcid'])).encode()).hexdigest())
 target=rows[i];receipt['query']={'id':f'val:{i}','pmcid':target['pmcid']}
 pdf=CACHE/'paper.pdf';receipt['files'].append(dl('https://arxiv.org/pdf/2505.11733v2',pdf,'1272dc11813456d8bbe899db3bd0e652cae45fdbd4c73cd6d02df476d2db9cc6'))
 page=fitz.open(pdf)[18].get_text()
 q=page.split('Prompt 6: Diagnostic Question Template\n',1)[1].split('\n"""',1)[0].strip()
 if q.count('{case_presentation}')!=1 or '<answer>' not in q:raise ValueError('Prompt extraction mismatch')
 prompt=q.replace('{case_presentation}',target['case_prompt'])
 (OUT/'PAPER_PROMPT6.txt').write_text(q,encoding='utf-8')
 save('PROMPT.json',{'query_id':f'val:{i}','pmcid':target['pmcid'],'source_prompt_pdf_page':19,'messages':[{'role':'user','content':prompt}],'target_sha256':hashlib.sha256(target['case_prompt'].encode()).hexdigest(),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
 receipt['input_has_reference_cases']=False;receipt['target_gold_loaded']=False;save('RECEIPT.json',receipt)
 release=get('https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b10964')
 if release['target_commitish']!='b29c606e28a01b1bc8c1351026a0fa6e616bf6c4':raise ValueError('Backend commit changed')
 a=next(a for a in release['assets'] if a['name']=='llama-b10964-bin-ubuntu-x64.tar.gz')
 receipt['files'].append(dl(a['browser_download_url'],CACHE/'llama.tar.gz',a['digest'].split(':')[1],a['size']))
 with tarfile.open(CACHE/'llama.tar.gz') as tf:tf.extractall(CACHE/'llama',filter='data')
 def weight(spec):
  part,size,digest=spec;name=f'qwen2.5-7b-instruct-fp16-{part}-of-00004.gguf'
  info=dl(f'https://huggingface.co/{MODEL}/resolve/{REV}/{name}?download=true',CACHE/name,digest,size)
  print('WEIGHT_SHARD_VERIFIED',part,flush=True);return info
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:receipt['files']+=list(pool.map(weight,SPECS))
 server=next((CACHE/'llama').rglob('llama-server')).resolve();server.chmod(server.stat().st_mode|0o111)
 env=os.environ.copy();env['LD_LIBRARY_PATH']=str(server.parent)+':'+env.get('LD_LIBRARY_PATH','')
 threads=min(4,os.cpu_count() or 2)
 cmd=[str(server),'-m',str((CACHE/'qwen2.5-7b-instruct-fp16-00001-of-00004.gguf').resolve()),'--host','127.0.0.1','--port','8080','-c','4096','-np','1','-t',str(threads),'-tb',str(threads),'-ngl','0','--no-webui']
 receipt['command']=cmd;receipt['platform']=platform.platform();receipt['cpu_count']=os.cpu_count();receipt['memory_before_load']=Path('/proc/meminfo').read_text();save('RECEIPT.json',receipt)
 log=(OUT/'server.log').open('w');proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
 try:
  for _ in range(180):
   if proc.poll() is not None:raise RuntimeError('Server exited; see server.log')
   try:
    with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3) as r:
     if r.status==200:break
   except Exception:time.sleep(1)
  else:raise TimeoutError('Server did not become ready')
  props=get('http://127.0.0.1:8080/props');save('SERVER_PROPS.json',props)
  template=post('/apply-template',{'messages':[{'role':'user','content':prompt}],'add_generation_prompt':True})
  formatted=template['prompt'];tokens=post('/tokenize',{'content':formatted,'add_special':True})['tokens']
  if len(tokens)+1024>4096:raise ValueError('Untruncated target exceeds context; no silent clipping')
  (OUT/'FORMATTED_PROMPT.txt').write_text(formatted,encoding='utf-8')
  receipt['formatted_prompt_tokens']=len(tokens);receipt['generation_started_utc']=now();save('RECEIPT.json',receipt)
  start=time.monotonic()
  with (OUT/'PREDICTIONS.jsonl').open('w',encoding='utf-8') as f:
   for attempt in range(10):
    if time.monotonic()-start>900:raise TimeoutError('Bounded validation execution limit reached; preserve partial outputs')
    payload={'prompt':formatted,'n_predict':1024,'temperature':0.8,'top_p':0.95,'top_k':0,'min_p':0.0,'typical_p':1.0,'repeat_penalty':1.0,'presence_penalty':0.0,'frequency_penalty':0.0,'seed':20261004+attempt,'cache_prompt':True,'id_slot':0,'stop':['<|im_end|>','<|endoftext|>']}
    t=time.monotonic();resp=post('/completion',payload,timeout=600);text=resp.get('content','')
    thinks=re.findall(r'<think>(.*?)</think>',text,re.S);answers=re.findall(r'<answer>(.*?)</answer>',text,re.S)
    complete=len(thinks)==1 and len(answers)==1 and bool(answers[0].strip())
    item={'query_id':f'val:{i}','attempt':attempt,'seed':20261004+attempt,'prompt_sha256':hashlib.sha256(formatted.encode()).hexdigest(),'response':resp,'parse_status':'complete' if complete else 'invalid_or_incomplete','reasoning':thinks[0].strip() if len(thinks)==1 else None,'diagnosis':answers[0].strip() if len(answers)==1 else None,'latency_seconds':time.monotonic()-t}
    f.write(json.dumps(item,ensure_ascii=False)+'\n');f.flush()
    receipt['predictions_completed']+=1;receipt['completed_xml']+=int(complete);save('RECEIPT.json',receipt)
    print('ATTEMPT_COMPLETE',attempt,'xml_complete',complete,'seconds',round(item['latency_seconds'],2),flush=True)
  receipt['status']='ten_validation_generations_complete_not_semantically_evaluated';receipt['completed_utc']=now();save('RECEIPT.json',receipt)
 finally:
  proc.terminate()
  try:proc.wait(timeout=20)
  except subprocess.TimeoutExpired:proc.kill()
  log.close()
if __name__=='__main__':
 try:main()
 except Exception as exc:
  if (OUT/'RECEIPT.json').exists():
   r=json.loads((OUT/'RECEIPT.json').read_text());r.update(status='failed_or_partial',error=type(exc).__name__+': '+str(exc));save('RECEIPT.json',r)
  raise
