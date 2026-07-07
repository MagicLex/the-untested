"use strict";
const DISEASES = [
  {t:"CHEMBL364", n:"Malaria"}, {t:"CHEMBL360", n:"Tuberculosis"},
  {t:"CHEMBL352", n:"Staph / MRSA"}, {t:"CHEMBL368", n:"Chagas disease"},
  {t:"CHEMBL612849", n:"Sleeping sickness"}, {t:"CHEMBL367", n:"Leishmaniasis"},
  {t:"CHEMBL366", n:"Thrush (Candida)"}, {t:"CHEMBL354", n:"E. coli"},
  {t:"CHEMBL348", n:"Pseudomonas"},
];
let DATA=null, TID=null, cvs, ctx, W=0, H=0, DPR=1, hover=null;

function color(p){
  if(p==null) return "#33404f";
  const stops=[[0.0,[27,42,58]],[0.45,[31,90,84]],[0.7,[30,177,130]],
               [0.85,[143,209,79]],[1.0,[245,179,66]]];
  for(let i=1;i<stops.length;i++){
    if(p<=stops[i][0]){
      const a=stops[i-1],b=stops[i],f=(p-a[0])/(b[0]-a[0]);
      const c=a[1].map((v,k)=>Math.round(v+(b[1][k]-v)*f));
      return `rgb(${c[0]},${c[1]},${c[2]})`;
    }
  }
  return "rgb(245,179,66)";
}
function prob(pt){ return pt.p && (TID in pt.p) ? pt.p[TID] : null; }

function resize(){
  const s=document.getElementById("stage").getBoundingClientRect();
  DPR=window.devicePixelRatio||1; W=s.width; H=s.height;
  cvs.width=W*DPR; cvs.height=H*DPR; cvs.style.width=W+"px"; cvs.style.height=H+"px";
  ctx.setTransform(DPR,0,0,DPR,0,0); draw();
}
function px(pt){ return [28+pt.x*(W-56), 24+pt.y*(H-48)]; }

function draw(){
  if(!DATA) return;
  ctx.clearRect(0,0,W,H);
  const pts=DATA.points;
  // glow pass for the strong hits
  for(const pt of pts){ const p=prob(pt); if(p==null||p<0.85) continue;
    const [x,y]=px(pt); const g=ctx.createRadialGradient(x,y,0,x,y,14);
    g.addColorStop(0,"rgba(245,179,66,.35)"); g.addColorStop(1,"rgba(245,179,66,0)");
    ctx.fillStyle=g; ctx.beginPath(); ctx.arc(x,y,14,0,7); ctx.fill(); }
  for(const pt of pts){ const p=prob(pt); const [x,y]=px(pt);
    ctx.fillStyle=color(p); ctx.globalAlpha=p==null?0.5:0.9;
    ctx.beginPath(); ctx.arc(x,y,p!=null&&p>0.7?2.6:1.6,0,7); ctx.fill(); }
  ctx.globalAlpha=1;
  if(hover){ const [x,y]=px(hover); ctx.strokeStyle="#fff"; ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.arc(x,y,5,0,7); ctx.stroke(); }
}

function nearest(mx,my){
  let best=null,bd=64;
  for(const pt of DATA.points){ const [x,y]=px(pt);
    const d=(x-mx)**2+(y-my)**2; if(d<bd){bd=d;best=pt;} }
  return best;
}

function selectDisease(t){
  TID=t;
  document.querySelectorAll("#diseases button").forEach(b=>
    b.classList.toggle("on", b.dataset.t===t));
  document.getElementById("legend").classList.remove("hidden");
  draw(); buildRail();
}

function buildRail(){
  const byOrg=new Map();
  for(const pt of DATA.points){ const p=prob(pt); if(p==null) continue;
    for(const o of pt.orgs){ const cur=byOrg.get(o);
      if(!cur||p>cur.p) byOrg.set(o,{p,pt}); } }
  const top=[...byOrg.entries()].sort((a,b)=>b[1].p-a[1].p).slice(0,24);
  const dn=DISEASES.find(d=>d.t===TID).n;
  document.getElementById("railhint").textContent=
    `most likely to fight ${dn.toLowerCase()}, ranked. none of these were tested for it.`;
  const ol=document.getElementById("raillist"); ol.innerHTML="";
  for(const [org,{p,pt}] of top){ const li=document.createElement("li");
    li.innerHTML=`<div class="plant">${org}</div>
      <div class="meta"><span class="dot" style="background:${color(p)}"></span>
      ${Math.round(p*100)}% likely</div>`;
    li.onclick=()=>{ hover=pt; draw(); openDossier(pt,org); };
    ol.appendChild(li); }
}

async function openDossier(pt,org){
  const p=prob(pt), dn=DISEASES.find(d=>d.t===TID).n;
  const fam=pt.orgs.slice(0,5).join(", ")+(pt.orgs.length>5?" and more":"");
  const conf=pt.ad>=0.55?"The model has seen very similar molecules, so this is a confident read."
    :pt.ad>=0.3?"The model has seen roughly similar molecules."
    :"This molecule is unlike anything the model learned from, so treat it as a long shot.";
  document.getElementById("dbody").innerHTML=`
    <h3>${org||pt.orgs[0]||"a natural product"}</h3>
    <div class="sub">a molecule it makes, never tested against ${dn.toLowerCase()}</div>
    <div id="mol">drawing…</div>
    <p class="say">The model rates this molecule's chance of fighting
      <b>${dn.toLowerCase()}</b> at <b>${Math.round(p*100)}%</b>. ${conf}</p>
    <div class="kv"><span>likelihood</span><b>${Math.round(p*100)}%</b></div>
    <div class="bar"><i style="width:${Math.round(p*100)}%"></i></div>
    <div class="kv"><span>familiarity to known molecules</span><b>${Math.round(pt.ad*100)}%</b></div>
    <div class="bar"><i style="width:${Math.round(pt.ad*100)}%"></i></div>
    <div class="plants">found in: ${fam}</div>
    <span class="flag">A high score means worth testing, not proven. Lab activity is not a medicine.</span>`;
  document.querySelector("main").classList.add("dossier-open");
  document.getElementById("dossier").classList.remove("hidden");
  const svg=await (await fetch("api/depict?smiles="+encodeURIComponent(pt.smiles))).text();
  document.getElementById("mol").innerHTML=svg;
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
  const svg=await (await fetch("api/depict?smiles="+encodeURIComponent(smiles))).text();
  document.getElementById("mol").innerHTML=svg;
}

function init(){
  cvs=document.getElementById("sky"); ctx=cvs.getContext("2d");
  const dd=document.getElementById("diseases");
  for(const d of DISEASES){ const b=document.createElement("button");
    b.textContent=d.n; b.dataset.t=d.t; b.onclick=()=>selectDisease(d.t); dd.appendChild(b); }
  const tip=document.getElementById("tip");
  cvs.addEventListener("mousemove",e=>{ const r=cvs.getBoundingClientRect();
    const mx=e.clientX-r.left,my=e.clientY-r.top; hover=nearest(mx,my); draw();
    if(hover&&TID){ const p=prob(hover);
      tip.innerHTML=`<b>${hover.orgs[0]||"natural product"}</b><br>${p!=null?Math.round(p*100)+"% likely":"—"}`;
      tip.style.left=Math.min(mx+12,W-250)+"px"; tip.style.top=(my+12)+"px";
      tip.classList.remove("hidden"); } else tip.classList.add("hidden"); });
  cvs.addEventListener("click",()=>{ if(hover&&TID) openDossier(hover,hover.orgs[0]); });
  document.getElementById("dclose").onclick=()=>{
    document.querySelector("main").classList.remove("dossier-open");
    document.getElementById("dossier").classList.add("hidden"); };
  document.getElementById("scorebtn").onclick=()=>{
    const s=document.getElementById("smiles").value.trim(); if(s) scoreSmiles(s); };
  window.addEventListener("resize",resize);
  fetch("static/mapdata.json").then(r=>r.json()).then(d=>{ DATA=d; resize();
    selectDisease("CHEMBL364"); });
}
init();
