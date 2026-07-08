"use strict";
// Each disease keeps the SAME cluster layout (x/y are frozen); only the colour
// hue and the per-target probability change. c=hue, w=Wikipedia page title.
const DISEASES = [
  {t:"CHEMBL364", n:"Malaria",          c:[245,179,66],  w:"Malaria"},
  {t:"CHEMBL360", n:"Tuberculosis",     c:[139,124,255], w:"Tuberculosis"},
  {t:"CHEMBL352", n:"Staph / MRSA",     c:[255,92,122],  w:"Methicillin-resistant_Staphylococcus_aureus"},
  {t:"CHEMBL368", n:"Chagas disease",   c:[30,177,130],  w:"Chagas_disease"},
  {t:"CHEMBL612849", n:"Sleeping sickness", c:[74,163,255], w:"African_trypanosomiasis"},
  {t:"CHEMBL367", n:"Leishmaniasis",    c:[192,132,252], w:"Leishmaniasis"},
  {t:"CHEMBL366", n:"Thrush (Candida)", c:[255,143,63],  w:"Oral_candidiasis"},
  {t:"CHEMBL354", n:"E. coli",          c:[53,208,208],  w:"Escherichia_coli"},
  {t:"CHEMBL348", n:"Pseudomonas",      c:[156,204,74],  w:"Pseudomonas_aeruginosa"},
];
const BASECOL=[27,42,58];
const PROMISC_HUE=[245,179,66];   // broad-spectrum discovery mode
const PROMISC_THR=0.7;            // a "hit" = predicted active at >= this
let DATA=null, TID=null, DIS=null, MODE="disease", cvs, ctx, base, bctx,
    W=0, H=0, DPR=1, hover=null, strong=[], anim=0;

// how many of the shown diseases a molecule is predicted active against at once
function promiscN(pt){ const p=pt.p||{};
  return DISEASES.reduce((a,d)=>a+((p[d.t]||0)>=PROMISC_THR?1:0),0); }
function promiscTop3(pt){ const p=pt.p||{};
  const v=DISEASES.map(d=>p[d.t]||0).sort((a,b)=>b-a); return (v[0]+v[1]+v[2])/3; }

function rgb(a){ return `rgb(${a[0]},${a[1]},${a[2]})`; }
function lerp(a,b,f){ return a.map((v,k)=>Math.round(v+(b[k]-v)*f)); }
// dark base -> disease hue by likelihood; hot white tip for the strongest hits.
function color(p){
  if(p==null) return "#33404f";
  const hue=DIS.c, e=Math.pow(p,0.85);
  let c=lerp(BASECOL,hue,e);
  if(p>0.85) c=lerp(c,[255,248,225],(p-0.85)/0.15*0.6);
  return rgb(c);
}
function prob(pt){
  if(MODE==="promisc") return Math.min(promiscN(pt)/4, 1);  // 0..4+ hits -> 0..1
  return pt.p && (TID in pt.p) ? pt.p[TID] : null;
}

function resize(){
  const s=document.getElementById("stage").getBoundingClientRect();
  DPR=window.devicePixelRatio||1; W=s.width; H=s.height;
  cvs.width=W*DPR; cvs.height=H*DPR; cvs.style.width=W+"px"; cvs.style.height=H+"px";
  ctx.setTransform(DPR,0,0,DPR,0,0);
  base.width=W*DPR; base.height=H*DPR;
  bctx.setTransform(DPR,0,0,DPR,0,0);
  renderBase();
}
function px(pt){ return [28+pt.x*(W-56), 24+pt.y*(H-48)]; }

// Draw the whole galaxy ONCE per disease into the offscreen base layer. The
// live loop only blits this and twinkles the strong hits on top, so the
// clusters never move and 7600 points don't redraw every frame.
function renderBase(){
  if(!DATA) return;
  bctx.clearRect(0,0,W,H);
  for(const pt of DATA.points){ const p=prob(pt), [x,y]=px(pt);
    bctx.fillStyle=color(p); bctx.globalAlpha=p==null?0.5:0.9;
    bctx.beginPath(); bctx.arc(x,y,p!=null&&p>0.7?2.6:1.6,0,7); bctx.fill(); }
  bctx.globalAlpha=1;
  // Only the standouts breathe: top ~50 by likelihood. Keeps the glow meaningful
  // (the few real leads, not a fog of 500) and the loop cheap.
  strong=DATA.points.filter(pt=>{ const p=prob(pt); return p!=null&&p>=0.8; })
    .sort((a,b)=>prob(b)-prob(a)).slice(0,50);
}

function frame(t){
  anim=requestAnimationFrame(frame);
  if(!DATA||!DIS) return;
  ctx.clearRect(0,0,W,H);
  ctx.drawImage(base,0,0,W,H);
  const hue=DIS.c;
  for(const pt of strong){ const [x,y]=px(pt), p=prob(pt);
    const pulse=0.35+0.35*Math.sin(t/650+(pt.x+pt.y)*40);
    const r=12+4*Math.sin(t/650+(pt.x+pt.y)*40);
    const g=ctx.createRadialGradient(x,y,0,x,y,r);
    g.addColorStop(0,`rgba(${hue[0]},${hue[1]},${hue[2]},${pulse*(p-0.5)})`);
    g.addColorStop(1,`rgba(${hue[0]},${hue[1]},${hue[2]},0)`);
    ctx.fillStyle=g; ctx.beginPath(); ctx.arc(x,y,r,0,7); ctx.fill(); }
  if(hover){ const [x,y]=px(hover); ctx.strokeStyle="#fff"; ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.arc(x,y,5,0,7); ctx.stroke(); }
}

function nearest(mx,my){
  let best=null,bd=64;
  for(const pt of DATA.points){ const [x,y]=px(pt);
    const d=(x-mx)**2+(y-my)**2; if(d<bd){bd=d;best=pt;} }
  return best;
}

function applyHue(){
  document.documentElement.style.setProperty("--hue",rgb(DIS.c));
}

function selectDisease(t){
  MODE="disease"; TID=t; DIS=DISEASES.find(d=>d.t===t); applyHue();
  document.querySelectorAll("#diseases button").forEach(b=>{
    const on=b.dataset.t===t; b.classList.toggle("on",on);
    b.style.background=on?rgb(DIS.c):""; b.style.borderColor=on?rgb(DIS.c):""; });
  const pb=document.getElementById("promiscbtn");
  if(pb){ pb.classList.remove("on"); pb.style.background=""; pb.style.borderColor=""; }
  document.getElementById("legend").classList.remove("hidden");
  renderBase(); buildRail();
}

// Broad-spectrum discovery: colour and rank molecules by how many diseases they
// hit at once, not by one disease. A multi-hit is either a real broad-spectrum
// lead or a frequent-hitter artifact; low familiarity tilts toward artifact.
function selectPromisc(){
  MODE="promisc"; TID=null; DIS={n:"broad-spectrum", c:PROMISC_HUE}; applyHue();
  document.querySelectorAll("#diseases button").forEach(b=>{
    b.classList.remove("on"); b.style.background=""; b.style.borderColor=""; });
  const pb=document.getElementById("promiscbtn");
  if(pb){ pb.classList.add("on"); pb.style.background=rgb(PROMISC_HUE);
    pb.style.borderColor=rgb(PROMISC_HUE); }
  document.getElementById("legend").classList.remove("hidden");
  renderBase(); buildRail();
}

function wikiLink(title,label){
  return `<a href="https://en.wikipedia.org/wiki/${encodeURIComponent(title)}"
    target="_blank" rel="noopener">${label}</a>`;
}
// Wikipedia REST summary (CORS-open): photo thumbnail + one-line extract.
async function wiki(title){
  try{
    const r=await fetch("https://en.wikipedia.org/api/rest_v1/page/summary/"
      +encodeURIComponent(title));
    if(!r.ok) return null; return await r.json();
  }catch(e){ return null; }
}

function buildRail(){
  if(MODE==="promisc") return buildPromiscRail();
  const byOrg=new Map();
  for(const pt of DATA.points){ const p=prob(pt); if(p==null) continue;
    for(const o of pt.orgs){ const cur=byOrg.get(o);
      if(!cur||p>cur.p) byOrg.set(o,{p,pt}); } }
  const top=[...byOrg.entries()].sort((a,b)=>b[1].p-a[1].p).slice(0,24);
  document.getElementById("railhint").textContent=
    `most likely to fight ${DIS.n.toLowerCase()}, ranked. none of these were tested for it.`;
  const ol=document.getElementById("raillist"); ol.innerHTML="";
  for(const [org,{p,pt}] of top){ const li=document.createElement("li");
    li.innerHTML=`<div class="plant">${org}</div>
      <div class="meta"><span class="dot" style="background:${color(p)}"></span>
      ${Math.round(p*100)}% likely</div>`;
    li.onclick=()=>{ hover=pt; openDossier(pt,org); };
    ol.appendChild(li); }
}

function buildPromiscRail(){
  document.getElementById("railhint").textContent=
    "predicted active against several diseases at once. a broad-spectrum lead, "
    + "or a frequent-hitter that trips many assays. low familiarity = suspect.";
  const seen=new Set(), rows=[];
  for(const pt of DATA.points){ if(seen.has(pt.ik)) continue; seen.add(pt.ik);
    const n=promiscN(pt); if(n>=2) rows.push({pt,n}); }
  rows.sort((a,b)=> b.n-a.n || promiscTop3(b.pt)-promiscTop3(a.pt));
  const ol=document.getElementById("raillist"); ol.innerHTML="";
  for(const {pt,n} of rows.slice(0,30)){ const li=document.createElement("li");
    const susp=pt.ad<0.3;
    li.innerHTML=`<div class="plant">${pt.orgs[0]||"natural product"}</div>
      <div class="meta"><span class="dot" style="background:${color(Math.min(n/4,1))}"></span>
      hits ${n} diseases${susp?' · <span class="susp">low familiarity</span>':''}</div>`;
    li.onclick=()=>{ hover=pt; openDossier(pt, pt.orgs[0]); };
    ol.appendChild(li); }
}

async function openDossier(pt,org){
  if(MODE==="promisc") return openPromiscDossier(pt,org);
  const p=prob(pt), name=org||pt.orgs[0]||"a natural product";
  const fam=pt.orgs.slice(0,6).map(o=>wikiLink(o,`<i>${o}</i>`)).join(", ")
    +(pt.orgs.length>6?" and more":"");
  const conf=pt.ad>=0.55?"The model has seen very similar molecules, so this is a confident read."
    :pt.ad>=0.3?"The model has seen roughly similar molecules."
    :"This molecule is unlike anything the model learned from, so treat it as a long shot.";
  document.getElementById("dbody").innerHTML=`
    <h3>${name}</h3>
    <div class="sub">a molecule it makes, never tested against ${wikiLink(DIS.w,DIS.n.toLowerCase())}</div>
    <div class="wikibox" id="wikibox"></div>
    <div id="mol">drawing…</div>
    <p class="say">The model rates this molecule's chance of fighting
      <b>${DIS.n.toLowerCase()}</b> at <b>${Math.round(p*100)}%</b>. ${conf}</p>
    <div class="kv"><span>likelihood</span><b>${Math.round(p*100)}%</b></div>
    <div class="bar"><i style="width:${Math.round(p*100)}%"></i></div>
    <div class="kv"><span>familiarity to known molecules</span><b>${Math.round(pt.ad*100)}%</b></div>
    <div class="bar"><i style="width:${Math.round(pt.ad*100)}%"></i></div>
    <div class="plants">found in: ${fam}</div>
    <span class="flag">A high score means worth testing, not proven. Lab activity is not a medicine.</span>`;
  document.querySelector("main").classList.add("dossier-open");
  document.getElementById("dossier").classList.remove("hidden");
  fetch("api/depict?smiles="+encodeURIComponent(pt.smiles)).then(r=>r.text())
    .then(svg=>{ document.getElementById("mol").innerHTML=svg; });
  const wk=await wiki(name);
  const box=document.getElementById("wikibox");
  if(box&&wk&&(wk.thumbnail||wk.extract)){
    box.innerHTML=(wk.thumbnail?`<img src="${wk.thumbnail.source}" alt="">`:"")
      +`<div class="wx">${wk.extract?wk.extract.slice(0,180):""}
        ${wikiLink(name,"Wikipedia ↗")}</div>`;
  } else if(box){ box.remove(); }
}

async function openPromiscDossier(pt,org){
  const name=org||pt.orgs[0]||"a natural product", n=promiscN(pt), susp=pt.ad<0.3;
  const rows=DISEASES.map(d=>({n:d.n, p:(pt.p||{})[d.t]||0})).sort((a,b)=>b.p-a.p);
  document.getElementById("dbody").innerHTML=`
    <h3>${name}</h3>
    <div class="sub">a molecule it makes, scored across every disease at once</div>
    <div class="wikibox" id="wikibox"></div>
    <div id="mol">drawing…</div>
    <p class="say">Predicted active against <b>${n}</b> of ${DISEASES.length}
      diseases at once. ${susp
        ? "But the model has seen little like it, so this broad hit is as likely a "
          + "frequent-hitter artifact as a real broad-spectrum lead."
        : "The model has seen similar molecules, which makes a genuine broad-spectrum "
          + "reading more credible."}</p>
    <div class="kv"><span>familiarity to known molecules</span><b>${Math.round(pt.ad*100)}%</b></div>
    <div class="bar"><i style="width:${Math.round(pt.ad*100)}%"></i></div>
    <ul class="profile">${rows.map(r=>`<li><span class="nm">${r.n}</span>
      <span class="b"><i style="width:${Math.round(r.p*100)}%"></i></span>
      <span>${Math.round(r.p*100)}%</span></li>`).join("")}</ul>
    <div class="plants">found in: ${pt.orgs.slice(0,6).map(o=>wikiLink(o,`<i>${o}</i>`)).join(", ")}</div>
    <span class="flag">Broad activity is a triage signal, not proof. Frequent-hitters
      trip many assays without being a medicine.</span>`;
  document.querySelector("main").classList.add("dossier-open");
  document.getElementById("dossier").classList.remove("hidden");
  fetch("api/depict?smiles="+encodeURIComponent(pt.smiles)).then(r=>r.text())
    .then(svg=>{ document.getElementById("mol").innerHTML=svg; });
  const wk=await wiki(name);
  const box=document.getElementById("wikibox");
  if(box&&wk&&(wk.thumbnail||wk.extract)){
    box.innerHTML=(wk.thumbnail?`<img src="${wk.thumbnail.source}" alt="">`:"")
      +`<div class="wx">${wk.extract?wk.extract.slice(0,180):""} ${wikiLink(name,"Wikipedia ↗")}</div>`;
  } else if(box){ box.remove(); }
}

async function scoreSmiles(smiles){
  const r=await fetch("api/score",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({smiles})});
  const j=await r.json();
  if(j.error||!j.predictions){ alert(j.error||"could not score"); return; }
  const res=j.predictions[0];
  if(res.error){ alert(res.error); return; }
  const rows=Object.values(res.predictions).filter(v=>v.kind==="amr")
    .sort((a,b)=>b.prob-a.prob);
  const dom=res.in_domain?"the model has seen molecules like this":
    "OUT of range — the model has seen nothing like this, every score is a guess";
  document.getElementById("dbody").innerHTML=`
    <h3>your molecule</h3><div class="sub">scored live against every disease</div>
    <div id="mol">drawing…</div>
    <div class="kv"><span>familiarity</span><b>${Math.round(res.ad_score*100)}%</b></div>
    <p class="say">${dom}.</p>
    <ul class="profile">${rows.map(v=>`<li><span class="nm">${v.label}</span>
      <span class="b"><i style="width:${Math.round(v.prob*100)}%"></i></span>
      <span>${Math.round(v.prob*100)}%</span></li>`).join("")}</ul>
    <span class="flag">Predictions, not prescriptions.</span>`;
  document.querySelector("main").classList.add("dossier-open");
  document.getElementById("dossier").classList.remove("hidden");
  fetch("api/depict?smiles="+encodeURIComponent(smiles)).then(r=>r.text())
    .then(svg=>{ document.getElementById("mol").innerHTML=svg; });
}

function init(){
  cvs=document.getElementById("sky"); ctx=cvs.getContext("2d");
  base=document.createElement("canvas"); bctx=base.getContext("2d");
  const dd=document.getElementById("diseases");
  for(const d of DISEASES){ const b=document.createElement("button");
    b.innerHTML=`<span class="sw" style="background:${rgb(d.c)}"></span>${d.n}`;
    b.dataset.t=d.t; b.onclick=()=>selectDisease(d.t); dd.appendChild(b); }
  const pb=document.createElement("button");
  pb.id="promiscbtn"; pb.className="promisc";
  pb.innerHTML=`<span class="sw" style="background:${rgb(PROMISC_HUE)}"></span>⚡ broad-spectrum`;
  pb.onclick=selectPromisc; dd.appendChild(pb);
  const tip=document.getElementById("tip");
  cvs.addEventListener("mousemove",e=>{ const r=cvs.getBoundingClientRect();
    const mx=e.clientX-r.left,my=e.clientY-r.top; hover=nearest(mx,my);
    if(hover&&(TID||MODE==="promisc")){
      const label=MODE==="promisc" ? `hits ${promiscN(hover)} diseases`
        : (prob(hover)!=null?Math.round(prob(hover)*100)+"% likely":"—");
      tip.innerHTML=`<b>${hover.orgs[0]||"natural product"}</b><br>${label}`;
      tip.style.left=Math.min(mx+12,W-250)+"px"; tip.style.top=(my+12)+"px";
      tip.classList.remove("hidden"); } else tip.classList.add("hidden"); });
  cvs.addEventListener("mouseleave",()=>{ hover=null; tip.classList.add("hidden"); });
  cvs.addEventListener("click",()=>{ if(hover&&(TID||MODE==="promisc")) openDossier(hover,hover.orgs[0]); });
  document.getElementById("dclose").onclick=()=>{
    document.querySelector("main").classList.remove("dossier-open");
    document.getElementById("dossier").classList.add("hidden"); };
  document.getElementById("scorebtn").onclick=()=>{
    const s=document.getElementById("smiles").value.trim(); if(s) scoreSmiles(s); };
  document.getElementById("smiles").addEventListener("keydown",e=>{
    if(e.key==="Enter"){ const s=e.target.value.trim(); if(s) scoreSmiles(s); } });
  window.addEventListener("resize",resize);
  fetch("static/mapdata.json").then(r=>r.json()).then(d=>{ DATA=d; resize();
    selectDisease("CHEMBL364"); if(!anim) frame(0); });
}
init();
