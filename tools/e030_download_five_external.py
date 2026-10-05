from __future__ import annotations
import csv, hashlib, json, pathlib, urllib.request, zipfile, os, time
from datetime import datetime, timezone

ROOT=pathlib.Path("external-five")
ROOT.mkdir(exist_ok=True)
UA={"User-Agent":"CASE-EVID-001-external-dataset-archiver/1.0"}

def fetch(url,dest,timeout=180):
    dest=pathlib.Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix(dest.suffix+".part")
    last=None
    for attempt in range(4):
        try:
            req=urllib.request.Request(url,headers=UA)
            with urllib.request.urlopen(req,timeout=timeout) as r, tmp.open("wb") as f:
                while True:
                    b=r.read(1024*1024)
                    if not b: break
                    f.write(b)
            tmp.replace(dest)
            return
        except Exception as e:
            last=e
            if tmp.exists(): tmp.unlink()
            if attempt==3: raise
            time.sleep(2**attempt)
    raise last

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def file_info(path,extra=None):
    p=pathlib.Path(path)
    x={"name":p.name,"bytes":p.stat().st_size,"sha256":sha256(p)}
    if extra:x.update(extra)
    return x

def write_manifest(folder,meta,files):
    out={"archived_utc":datetime.now(timezone.utc).isoformat(),"meta":meta,"files":files}
    (pathlib.Path(folder)/"SOURCE_MANIFEST.json").write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return out

# ---------- 1 MedR-Bench ----------
folder=ROOT/"01_MedRBench";folder.mkdir(exist_ok=True)
commit="ff60ab440afd2f2bc0c603b3a65d715ec83138a7"
base=f"https://raw.githubusercontent.com/MAGIC-AI4Med/MedRBench/{commit}"
urls={
 "diagnosis_957_cases_with_rare_disease_491.json":base+"/data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json",
 "treatment_496_cases_with_rare_disease_165.json":base+"/data/MedRBench/treatment_496_cases_with_rare_disease_165.json",
 "README.md":base+"/README.md","LICENSE":base+"/LICENSE"}
files=[]
for name,url in urls.items():
    fetch(url,folder/name)
    files.append(file_info(folder/name,{"source_url":url}))
dx=json.loads((folder/"diagnosis_957_cases_with_rare_disease_491.json").read_text())
tx=json.loads((folder/"treatment_496_cases_with_rare_disease_165.json").read_text())
assert len(dx)==957,(len(dx),"diagnosis")
assert len(tx)==496,(len(tx),"treatment")
write_manifest(folder,{
 "dataset":"MedR-Bench","source_repo":"MAGIC-AI4Med/MedRBench","source_commit":commit,
 "license_note":"Dataset stated as CC BY-NC-SA in paper/data availability; included upstream LICENSE file.",
 "validation":{"diagnosis_cases":len(dx),"treatment_cases":len(tx)}
},files)

# ---------- 2 AgentClinic NEJM ----------
folder=ROOT/"02_AgentClinic_NEJM";folder.mkdir(exist_ok=True)
commit="b6fbe22300e99a267a7ac94eaa465ab552eef741"
base=f"https://raw.githubusercontent.com/SamuelSchmidgall/AgentClinic/{commit}"
urls={
 "agentclinic_nejm.jsonl":base+"/agentclinic_nejm.jsonl",
 "agentclinic_nejm_extended.jsonl":base+"/agentclinic_nejm_extended.jsonl",
 "README.md":base+"/README.md","LICENSE.txt":base+"/LICENSE.txt"}
files=[]
for name,url in urls.items():
    fetch(url,folder/name);files.append(file_info(folder/name,{"source_url":url}))
def read_jsonl(path):
    rows=[]
    for i,line in enumerate(open(path,encoding="utf-8"),1):
        if line.strip(): rows.append(json.loads(line))
    return rows
nejm=read_jsonl(folder/"agentclinic_nejm.jsonl")
ext=read_jsonl(folder/"agentclinic_nejm_extended.jsonl")
assert len(ext)==120,(len(ext),"extended NEJM expected 120")
write_manifest(folder,{
 "dataset":"AgentClinic-NEJM","source_repo":"SamuelSchmidgall/AgentClinic","source_commit":commit,
 "license_note":"Upstream benchmark and data stated as MIT; included LICENSE.txt.",
 "validation":{"nejm_records":len(nejm),"nejm_extended_records":len(ext)}
},files)

# ---------- 3 PMC-Patients Human Evaluation ----------
folder=ROOT/"03_PMC_Patients_HumanEval";folder.mkdir(exist_ok=True)
hf_commit="0887286c5e7940545c6d3fcb907c70edc1d156b3"
hf_base=f"https://huggingface.co/datasets/zhengyun21/PMC-Patients-MetaData/resolve/{hf_commit}"
urls={
 "PMC-Patients_human_eval.json":hf_base+"/PMC-Patients_human_eval.json?download=true",
 "human_PMIDs.json":hf_base+"/human_PMIDs.json?download=true",
 "README.md":hf_base+"/README.md?download=true",
 "LICENSE": "https://raw.githubusercontent.com/zhao-zy15/PMC-Patients/20cba9b40b9c7b6462da9ba7f2149e17c6bd8221/LICENSE"}
files=[]
for name,url in urls.items():
    fetch(url,folder/name);files.append(file_info(folder/name,{"source_url":url}))
human=json.loads((folder/"PMC-Patients_human_eval.json").read_text())
pmids=json.loads((folder/"human_PMIDs.json").read_text())
assert len(pmids)==500,(len(pmids),"human_PMIDs")
write_manifest(folder,{
 "dataset":"PMC-Patients Human Evaluation Subset",
 "source_hf_repo":"zhengyun21/PMC-Patients-MetaData","source_hf_commit":hf_commit,
 "license_repo_commit":"20cba9b40b9c7b6462da9ba7f2149e17c6bd8221",
 "license_note":"CC BY-NC-SA 4.0 per upstream metadata/repository; included LICENSE.",
 "validation":{"sampled_article_pmids":len(pmids),"human_eval_records":len(human)}
},files)

# ---------- 4 CaseReportBench ----------
folder=ROOT/"04_CaseReportBench";folder.mkdir(exist_ok=True)
hf_commit="3297c5b2db51872b70645a75f43dc0b849f77621"
hf_base=f"https://huggingface.co/datasets/cxyzhang/caseReportBench_ClinicalDenseExtraction_Benchmark/resolve/{hf_commit}"
gh_commit="031699d7fa14040cf5952df8a391c34e2dd660f8"
gh_base=f"https://raw.githubusercontent.com/cindyzhangxy/CaseReportBench/{gh_commit}"
urls={
 "train-00000-of-00001.parquet":hf_base+"/data/train-00000-of-00001.parquet?download=true",
 "HF_README.md":hf_base+"/README.md?download=true",
 "DATA_LICENSE.txt":gh_base+"/DATA_LICENSE.txt",
 "CODE_LICENSE.txt":gh_base+"/LICENSE.txt",
 "UPSTREAM_README.md":gh_base+"/README.md"}
files=[]
for name,url in urls.items():
    fetch(url,folder/name);files.append(file_info(folder/name,{"source_url":url}))
import pyarrow.parquet as pq
pf=pq.ParquetFile(folder/"train-00000-of-00001.parquet")
rows=pf.metadata.num_rows
cols=pf.schema_arrow.names
assert rows>0
write_manifest(folder,{
 "dataset":"CaseReportBench: Clinical Dense Extraction Benchmark",
 "source_hf_repo":"cxyzhang/caseReportBench_ClinicalDenseExtraction_Benchmark","source_hf_commit":hf_commit,
 "source_code_repo_commit":gh_commit,
 "license_note":"Use stricter upstream DATA_LICENSE.txt (CC BY-NC 4.0) because HF metadata has varied across revisions.",
 "validation":{"rows":rows,"columns":cols}
},files)

# ---------- 5 DDXPlus English ----------
folder=ROOT/"05_DDXPlus_English";folder.mkdir(exist_ok=True)
article_id=22687585
with urllib.request.urlopen(urllib.request.Request(f"https://api.figshare.com/v2/articles/{article_id}",headers=UA),timeout=60) as r:
    meta=json.load(r)
(folder/"FIGSHARE_ARTICLE_METADATA.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
files=[file_info(folder/"FIGSHARE_ARTICLE_METADATA.json",{"source_url":f"https://api.figshare.com/v2/articles/{article_id}"})]
expected={"release_evidences.json","release_conditions.json","release_train_patients.zip","release_validate_patients.zip","release_test_patients.zip"}
available={x["name"]:x for x in meta.get("files",[])}
missing=expected-set(available)
assert not missing,missing
for name in sorted(expected):
    info=available[name]
    url=info["download_url"]
    fetch(url,folder/name,timeout=600)
    fi=file_info(folder/name,{"source_url":url,"figshare_file_id":info.get("id"),"figshare_supplied_md5":info.get("supplied_md5")})
    files.append(fi)
# docs
repo_commit="4f5ec677099ef99f7cc1dc0bfe2378bb29a29bda"
for name,path in [("README.md","README.md"),("CITATION.cff","CITATION.cff")]:
    url=f"https://raw.githubusercontent.com/mila-iqia/ddxplus/{repo_commit}/{path}"
    fetch(url,folder/name);files.append(file_info(folder/name,{"source_url":url}))
(folder/"LICENSE_NOTE.txt").write_text(
 "DDXPlus English release is labeled CC BY 4.0 on the official Figshare article and project documentation.\n"
 "Figshare article: https://figshare.com/articles/dataset/DDXPlus_Dataset_English_/22687585\n",encoding="utf-8")
files.append(file_info(folder/"LICENSE_NOTE.txt",{"source_url":"https://figshare.com/articles/dataset/DDXPlus_Dataset_English_/22687585"}))
evid=json.loads((folder/"release_evidences.json").read_text())
cond=json.loads((folder/"release_conditions.json").read_text())
assert len(evid)==223,(len(evid),"evidences")
assert len(cond)==49,(len(cond),"conditions")
split_rows={}
for name in ["release_train_patients.zip","release_validate_patients.zip","release_test_patients.zip"]:
    with zipfile.ZipFile(folder/name) as z:
        members=[m for m in z.namelist() if not m.endswith("/")]
        assert len(members)>=1
        # count CSV rows without extracting to disk
        total=0
        member=members[0]
        with z.open(member) as f:
            for _ in f: total+=1
        split_rows[name]=max(0,total-1)
write_manifest(folder,{
 "dataset":"DDXPlus English","source_figshare_article_id":article_id,"source_figshare_version":meta.get("version"),
 "source_figshare_doi":meta.get("doi"),"source_repo_commit":repo_commit,
 "license_note":"CC BY 4.0 per official Figshare/project documentation.",
 "validation":{"evidences":len(evid),"conditions":len(cond),"split_rows":split_rows}
},files)

# top-level receipt
summary={}
for d in sorted([p for p in ROOT.iterdir() if p.is_dir()]):
    m=json.loads((d/"SOURCE_MANIFEST.json").read_text())
    summary[d.name]={"file_count":len(m["files"]),"total_bytes":sum(x["bytes"] for x in m["files"]),"meta":m["meta"]}
(ROOT/"MASTER_MANIFEST.json").write_text(json.dumps({
 "project":"CASE-EVID-001","external_validation_data_bundle":"E030",
 "created_utc":datetime.now(timezone.utc).isoformat(),"datasets":summary
},indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
print(json.dumps(summary,indent=2,ensure_ascii=False))
