from __future__ import annotations
import json,html,pathlib,argparse

ap=argparse.ArgumentParser()
ap.add_argument("--input",required=True)
ap.add_argument("--rater",required=True)
ap.add_argument("--out",required=True)
a=ap.parse_args()

rows=[json.loads(x) for x in open(a.input,encoding="utf-8") if x.strip()]
rows=[x for x in rows if not x["same_unordered_top3"]]
assert len(rows)==117,len(rows)

def esc(x): return html.escape(str(x))
def evidence(items):
    blocks=[]
    for x in items:
        blocks.append(
        '<details class="ref"><summary>Reference %s · %s · historical diagnosis: <b>%s</b></summary>'
        '<div class="refbody"><p><b>Case presentation</b><br>%s</p>'
        '<p><b>Historical diagnostic reasoning</b><br>%s</p></div></details>' %
        (x["rank"],esc(x["reference_id"]),esc(x["historical_diagnosis"]),esc(x["case_prompt"]),esc(x["diagnostic_reasoning"])))
    return "\n".join(blocks)

cases=[]
for n,x in enumerate(rows,1):
    q=x["query_index"]
    cases.append(
    '<section class="case" data-q="%s" data-case="%s">'
    '<div class="casehead"><h2>Case %s <span>%s</span></h2><div class="badge">unrated</div></div>'
    '<div class="target"><b>Target case</b><br>%s</div>'
    '<div class="cols"><div><h3>Set A</h3>%s</div><div><h3>Set B</h3>%s</div></div>'
    '<div class="rating"><div class="prompt">Which reference set is more useful for reasoning about the diagnosis of the target case?</div>'
    '<div class="radios">'
    '<label><input type="radio" name="pref-%s" value="A clearly better"> A clearly better</label>'
    '<label><input type="radio" name="pref-%s" value="A slightly better"> A slightly better</label>'
    '<label><input type="radio" name="pref-%s" value="About equal"> About equal</label>'
    '<label><input type="radio" name="pref-%s" value="B slightly better"> B slightly better</label>'
    '<label><input type="radio" name="pref-%s" value="B clearly better"> B clearly better</label>'
    '<label><input type="radio" name="pref-%s" value="Neither useful"> Neither useful</label></div>'
    '<div class="fields"><label>Confidence <select class="confidence"><option value=""></option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></label>'
    '<label>Potentially misleading evidence <select class="misleading"><option value=""></option><option>Neither</option><option>A</option><option>B</option><option>Both</option></select></label></div>'
    '<label>Comment<textarea class="comment" rows="2"></textarea></label></div></section>' %
    (q,esc(x["case_id"]),n,esc(x["case_id"]),esc(x["target_case"]),evidence(x["set_A"]),evidence(x["set_B"]),q,q,q,q,q,q))

style="""
:root{--ink:#172033;--line:#d9e1ee;--blue:#1e3a8a;--pale:#f6f8fc;--gold:#fff8e8;--ok:#eaf7f0}
*{box-sizing:border-box}body{font-family:Arial,"Microsoft YaHei",sans-serif;color:var(--ink);line-height:1.55;margin:0;background:#f4f6fa}
header{position:sticky;top:0;z-index:10;background:white;border-bottom:1px solid var(--line);padding:14px 24px;display:flex;justify-content:space-between;gap:20px;align-items:center}
main{max-width:1300px;margin:auto;padding:24px}.intro{background:#eef3ff;padding:16px;border-radius:10px;margin-bottom:20px}
.progress{font-weight:700}.case{background:white;border:1px solid var(--line);border-radius:12px;padding:18px;margin:0 0 22px;box-shadow:0 4px 14px rgba(20,30,60,.04)}
.casehead{display:flex;justify-content:space-between;align-items:center}.casehead h2{margin:0 0 12px}.casehead span{font-size:12px;color:#667085}
.badge{font-size:12px;background:#eef2f6;padding:4px 8px;border-radius:999px}.case.done .badge{background:var(--ok);color:#137a4b}
.target{background:var(--pale);border:1px solid var(--line);padding:14px;border-radius:8px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:16px}.ref{border:1px solid var(--line);border-radius:8px;margin:8px 0;background:white}.ref summary{cursor:pointer;padding:10px;font-weight:600;color:var(--blue)}.refbody{padding:0 10px 10px}
.rating{background:var(--gold);padding:14px;border-radius:8px;margin-top:16px}.prompt{font-weight:700;margin-bottom:8px}.radios{display:flex;flex-wrap:wrap;gap:10px 18px}.radios label{white-space:nowrap}.fields{display:flex;gap:24px;margin:12px 0}.comment{display:block;width:100%;margin-top:4px}
button{border:0;border-radius:8px;padding:9px 13px;font-weight:700;cursor:pointer}button.primary{background:#1e3a8a;color:white}button.secondary{background:#e9eef8;color:#1e3a8a}
@media(max-width:900px){.cols{grid-template-columns:1fr}.fields{flex-direction:column;gap:8px}header{position:static;flex-direction:column;align-items:flex-start}}
"""

js="""
const PACK = "__PACK__";
const key = "case-evid-e040-" + PACK;
let state = {};
try { state = JSON.parse(localStorage.getItem(key) || "{}"); } catch(e) { state = {}; }

function getCaseData(sec){
  const q=sec.dataset.q;
  const checked=sec.querySelector('input[type=radio]:checked');
  return {query_index:q,case_id:sec.dataset.case,preference:checked?checked.value:"",
    confidence:sec.querySelector('.confidence').value,
    misleading:sec.querySelector('.misleading').value,
    comment:sec.querySelector('.comment').value};
}
function restore(sec){
  const x=state[sec.dataset.q]; if(!x) return;
  if(x.preference){const rs=[...sec.querySelectorAll('input[type=radio]')]; const r=rs.find(z=>z.value===x.preference); if(r)r.checked=true;}
  sec.querySelector('.confidence').value=x.confidence||"";
  sec.querySelector('.misleading').value=x.misleading||"";
  sec.querySelector('.comment').value=x.comment||"";
}
function save(sec){const x=getCaseData(sec);state[x.query_index]=x;localStorage.setItem(key,JSON.stringify(state));update();}
function update(){
  let done=0,total=0;
  document.querySelectorAll('.case').forEach(sec=>{total++;const x=getCaseData(sec);const ok=!!x.preference;sec.classList.toggle('done',ok);sec.querySelector('.badge').textContent=ok?'rated':'unrated';if(ok)done++;});
  document.getElementById('progress').textContent=done+'/'+total+' rated';
}
document.querySelectorAll('.case').forEach(sec=>{
 restore(sec);
 sec.querySelectorAll('input,select,textarea').forEach(el=>el.addEventListener('change',()=>save(sec)));
 sec.querySelector('.comment').addEventListener('input',()=>save(sec));
});
update();
function csvEscape(v){v=(v??"").toString();return '"'+v.replaceAll('"','""')+'"';}
function exportCSV(){
 const rows=[['rater_pack','query_index','case_id','preference','confidence','misleading','comment']];
 document.querySelectorAll('.case').forEach(sec=>{const x=getCaseData(sec);rows.push([PACK,x.query_index,x.case_id,x.preference,x.confidence,x.misleading,x.comment]);});
 const csv=rows.map(r=>r.map(csvEscape).join(',')).join('\\n');
 const blob=new Blob([csv],{type:'text/csv;charset=utf-8'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='E040_'+PACK+'_responses.csv';a.click();URL.revokeObjectURL(url);
}
function exportJSON(){
 const arr=[];document.querySelectorAll('.case').forEach(sec=>arr.push(getCaseData(sec)));
 const blob=new Blob([JSON.stringify({rater_pack:PACK,responses:arr},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='E040_'+PACK+'_responses.json';a.click();URL.revokeObjectURL(url);
}
""".replace("__PACK__",a.rater)

doc='<!doctype html><html><head><meta charset="utf-8"><title>E040 '+esc(a.rater)+' Interactive Blind Evaluation</title><style>'+style+'</style></head><body>'
doc+='<header><div><b>AgentClinic-NEJM Blind Evidence Evaluation — '+esc(a.rater)+'</b><div id="progress" class="progress"></div></div>'
doc+='<div><button class="secondary" onclick="exportJSON()">Export JSON</button> <button class="primary" onclick="exportCSV()">Export CSV</button></div></header>'
doc+='<main><div class="intro"><b>Blinding:</b> Set A/B identities and target gold diagnoses are hidden. This pack contains 117 cases; 3 cases with identical unordered Top-3 evidence sets were pre-specified as structural ties and are omitted from manual review.<br>'
doc+='<b>Judge:</b> Which historical evidence set would be more useful for diagnostic reasoning about the target case? Consider discriminative clinical relevance and risk of misleading analogy, not superficial word overlap. Progress is saved locally in this browser.</div>'
doc+=''.join(cases)+'</main><script>'+js+'</script></body></html>'
path=pathlib.Path(a.out);path.write_text(doc,encoding="utf-8");print(path)
