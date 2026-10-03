const C=window.CFD_CONFIG||{API_BASE_URL:""};
const API=(C.API_BASE_URL||"").replace(/\/$/,"");
const $=id=>document.getElementById(id);
const fmtPct=x=>(Number(x)*100).toFixed(1)+"%";
const band=p=>p<.33?"Lower":p<.66?"Intermediate":"Higher";
const cls=s=>s.toLowerCase().replace(" ","-");

async function api(path,options){
  if(!API) throw new Error("API_BASE_URL is not configured");
  const r=await fetch(API+path,{headers:{"Content-Type":"application/json"},...options});
  if(!r.ok) throw new Error((await r.text())||r.statusText);
  return r.json();
}

function payload(){
  return {
    model:$("model").value,
    current_ratio:+$("current_ratio").value,
    debt_to_assets:+$("debt_to_assets").value,
    debt_to_equity:+$("debt_to_equity").value,
    gross_margin:.30,
    operating_margin:+$("operating_margin").value,
    roa:+$("roa").value,
    fcf_margin:+$("fcf_margin").value,
    revenue_yoy:.05,
    operating_income_yoy:.03,
    market_volatility:.25
  };
}

function localScore(x){
  const z=-1.7-.75*(x.current_ratio-1)+2.2*x.debt_to_assets+.22*x.debt_to_equity-4*x.operating_margin-3.2*x.roa-2.5*x.fcf_margin;
  return 1/(1+Math.exp(-z));
}

function renderScore(p,model,mode){
  $("score").innerHTML='<div class="score-number">'+fmtPct(p)+'</div><div><span class="band '+cls(band(p))+'">'+band(p)+'</span><div class="muted" style="margin-top:4px">'+model.replaceAll("_"," ")+' · '+mode+'</div></div>';
}

async function runScore(){
  const x=payload();
  try{
    if(API){
      const d=await api("/score",{method:"POST",body:JSON.stringify(x)});
      renderScore(d.probability,d.model,d.mode);
    }else{
      renderScore(localScore(x),x.model,"browser synthetic fallback");
    }
  }catch(e){
    renderScore(localScore(x),x.model,"browser fallback · API unavailable");
  }
}

function renderCompanies(rows){
  $("companyRows").innerHTML=rows.map(r=>'<tr><td>'+r.company_id+'</td><td>'+r.period+'</td><td class="num">'+fmtPct(r.probability)+'</td><td><span class="pill '+cls(r.risk_band)+'">'+r.risk_band+'</span></td></tr>').join("");
  const counts={Lower:0,Intermediate:0,Higher:0}; rows.forEach(r=>counts[r.risk_band]=(counts[r.risk_band]||0)+1);
  const n=Math.max(rows.length,1);
  $("barLower").style.width=(counts.Lower/n*100)+"%"; $("barIntermediate").style.width=(counts.Intermediate/n*100)+"%"; $("barHigher").style.width=(counts.Higher/n*100)+"%";
  $("nLower").textContent=counts.Lower; $("nIntermediate").textContent=counts.Intermediate; $("nHigher").textContent=counts.Higher;
}

function browserCompanies(){
  return Array.from({length:12},(_,i)=>{
    const p=Math.min(.96,Math.max(.04,.08+i*.075+((i%3)-1)*.025));
    return {company_id:"DEMO-"+String(i+1).padStart(3,"0"),period:"Synthetic",probability:p,risk_band:band(p)};
  }).sort((a,b)=>b.probability-a.probability);
}

async function loadCompanies(){
  try{
    const d=API?await api("/demo/companies"):null;
    renderCompanies(d?d.companies:browserCompanies());
  }catch(e){renderCompanies(browserCompanies())}
}

async function health(){
  const dot=$("apiDot"),txt=$("apiText");
  if(!API){dot.className="dot";txt.textContent="Demo mode";return}
  try{const d=await api("/health");dot.className="dot ok";txt.textContent="API online · "+d.model_version}catch(e){dot.className="dot bad";txt.textContent="API unavailable"}
}

$("scoreBtn").addEventListener("click",runScore);
$("refreshBtn").addEventListener("click",loadCompanies);
loadCompanies();health();runScore();
