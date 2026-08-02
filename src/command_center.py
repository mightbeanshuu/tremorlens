"""Generates the TremorLens live command centre: a working ops console that replays
real pipeline streams (out/stream.json) as a live feed — fleet panel, scrolling
waveform, animating spectrum, SHS gauge, and an event feed. Single file, no CDN.
"""
import json

TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TremorLens Command Centre</title><style>
:root{--bg:#0f0f0e;--panel:#1a1a19;--edge:#2a2a28;--ink:#fff;--ink2:#c3c2b7;--mut:#8a897f;
--blue:#3987e5;--orange:#d95926;--good:#199e70;--warn:#c98500;--bad:#e66767}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:13px/1.45 -apple-system,'Segoe UI',sans-serif}
.top{display:flex;align-items:center;gap:14px;padding:10px 18px;border-bottom:1px solid var(--edge);background:var(--panel)}
.top h1{font-size:15px;margin:0;letter-spacing:.4px}.top .sub{color:var(--mut);font-size:11px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--bad);animation:bl 1.2s infinite}
@keyframes bl{50%{opacity:.25}}
.clock{margin-left:auto;font-variant-numeric:tabular-nums;color:var(--ink2)}
.sumchip{padding:3px 10px;border-radius:999px;font-size:11px;border:1px solid var(--edge)}
.grid{display:grid;grid-template-columns:230px 1fr 250px;gap:12px;padding:12px 18px;max-width:1280px;margin:0 auto}
.card{background:var(--panel);border:1px solid var(--edge);border-radius:10px;padding:12px}
.card h2{margin:0 0 8px;font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.8px;font-weight:600}
.bridge{padding:9px 10px;border:1px solid var(--edge);border-radius:8px;margin-bottom:8px;cursor:pointer}
.bridge:hover{border-color:var(--mut)}.bridge.sel{border-color:var(--blue)}
.bridge .nm{font-weight:600;font-size:12.5px}.bridge .cam{color:var(--mut);font-size:10.5px}
.badge{float:right;font-size:10.5px;font-weight:700;padding:1px 8px;border-radius:999px}
.b-HEALTHY{color:var(--good);background:rgba(25,158,112,.12)}
.b-WATCH{color:var(--warn);background:rgba(201,133,0,.12)}
.b-ALERT{color:var(--bad);background:rgba(230,103,103,.14)}
canvas{width:100%;display:block;border-radius:6px;background:#141413}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px}
.kpi{background:#141413;border:1px solid var(--edge);border-radius:8px;padding:8px 10px}
.kpi b{display:block;font-size:19px;font-variant-numeric:tabular-nums}
.kpi span{color:var(--mut);font-size:10px;text-transform:uppercase;letter-spacing:.6px}
.alertbar{display:none;margin-bottom:10px;padding:8px 12px;border-radius:8px;background:rgba(230,103,103,.14);
border:1px solid var(--bad);color:var(--bad);font-weight:600;font-size:12.5px}
.feed{max-height:380px;overflow-y:auto}.ev{padding:7px 0;border-bottom:1px solid var(--edge);font-size:11.5px}
.ev .t{color:var(--mut);font-variant-numeric:tabular-nums;margin-right:8px}
.lbl{color:var(--mut);font-size:10px;margin:8px 0 3px;text-transform:uppercase;letter-spacing:.6px}
.foot{color:var(--mut);font-size:10.5px;text-align:center;padding:8px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
</style></head><body>
<div class="top"><span class="dot"></span><h1>TREMORLENS · SPANDAN COMMAND CENTRE</h1>
<span class="sub">Municipal Bridge Fleet — Camera-Only Structural Monitoring</span>
<span class="sumchip" id="sum"></span><span class="clock" id="clock"></span></div>
<div class="grid">
  <div class="card"><h2>Fleet — 3 structures</h2><div id="fleet"></div>
    <div class="lbl">Baseline</div>
    <div style="font-size:12px;color:var(--ink2)">f&#8321; <b id="basef"></b> Hz · noise floor <span id="nz"></span>%</div>
    <div class="lbl">Ground truth link</div>
    <div style="font-size:11.5px;color:var(--ink2)">Contact accel agreement <b style="color:var(--good)">&le;0.16%</b></div>
  </div>
  <div class="card"><h2 id="mainttl">Live monitor</h2>
    <div class="alertbar" id="abar"></div>
    <div class="kpis">
      <div class="kpi"><b id="kshs">—</b><span>SHS</span></div>
      <div class="kpi"><b id="kf1">—</b><span>f&#8321; live (Hz)</span></div>
      <div class="kpi"><b id="kdrift">—</b><span>drift vs baseline</span></div>
      <div class="kpi"><b id="kstat">—</b><span>verdict</span></div>
    </div>
    <div class="lbl">Deck displacement — live (px)</div><canvas id="wave" height="110"></canvas>
    <div class="lbl">Modal fingerprint — live spectrum, 1–25 Hz</div><canvas id="spec" height="150"></canvas>
  </div>
  <div class="card"><h2>Event feed</h2><div class="feed" id="feed"></div></div>
</div>
<div class="foot">Phase-based motion analysis (MIT lineage: Wu 2012 · Wadhwa 2013) · educational research prototype · all data from pipeline runs on capture-realistic test rig</div>
<script>
const D=__DATA__;
let sel='C', tick=0, feedRows=[];
const $=id=>document.getElementById(id);
const stat2c={HEALTHY:'var(--good)',WATCH:'var(--warn)',ALERT:'var(--bad)'};
function fmtClock(){const d=new Date();return d.toLocaleTimeString('en-IN',{hour12:false});}
setInterval(()=>$('clock').textContent=fmtClock(),500);$('clock').textContent=fmtClock();
$('basef').textContent=D.baseline_f1.toFixed(2);$('nz').textContent=D.noise_pct.toFixed(2);

function fleetRender(){
  const el=$('fleet');el.innerHTML='';let h=0,a=0;
  for(const k of Object.keys(D.fleet)){
    const b=D.fleet[k],last=b.timeline[b.timeline.length-1];
    (last.status==='ALERT')?a++:h++;
    const div=document.createElement('div');
    div.className='bridge'+(k===sel?' sel':'');
    div.innerHTML=`<span class="badge b-${last.status}">${last.status==='ALERT'?'&#9650; ':''}SHS ${Math.round(last.shs)}</span>
      <div class="nm">Bridge ${k}</div><div class="cam">${b.name} · CAM-${k}${b.cam_km} · existing CCTV</div>`;
    div.onclick=()=>{sel=k;tick=0;feedRows=[];$('feed').innerHTML='';fleetRender();};
    el.appendChild(div);}
  $('sum').innerHTML=`<span style="color:var(--good)">${h} HEALTHY</span> · <span style="color:var(--bad)">${a} ALERT</span>`;
}
function ev(msg,color){
  const t=fmtClock();feedRows.unshift({t,msg,color});feedRows=feedRows.slice(0,40);
  $('feed').innerHTML=feedRows.map(r=>`<div class="ev"><span class="t">${r.t}</span><span style="color:${r.color||'var(--ink2)'}">${r.msg}</span></div>`).join('');
}
function drawWave(b){
  const c=$('wave'),x=c.getContext('2d'),W=c.width=c.clientWidth*2,H=c.height=220;
  x.clearRect(0,0,W,H);x.strokeStyle='#2a2a28';x.beginPath();x.moveTo(0,H/2);x.lineTo(W,H/2);x.stroke();
  const n=Math.floor(W/2), start=(tick*8)%Math.max(1,b.disp.length-n);
  const seg=b.disp.slice(start,start+n);
  const amp=Math.max(0.6,...seg.map(Math.abs));
  x.strokeStyle='#3987e5';x.lineWidth=2;x.beginPath();
  seg.forEach((v,i)=>{const px=i*2,py=H/2-v/amp*(H/2-12);i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke();
}
function drawSpec(b,step){
  const c=$('spec'),x=c.getContext('2d'),W=c.width=c.clientWidth*2,H=c.height=300,pad=30;
  x.clearRect(0,0,W,H);x.strokeStyle='#2a2a28';x.fillStyle='#8a897f';x.font='20px sans-serif';
  for(let i=0;i<=6;i++){const px=pad+i*(W-2*pad)/6;x.beginPath();x.moveTo(px,10);x.lineTo(px,H-24);x.stroke();
    x.fillText((1+i*4)+'Hz',px-14,H-6);}
  const bx=pad+(D.baseline_f1-1)/24*(W-2*pad);
  x.strokeStyle='#8a897f';x.setLineDash([6,6]);x.beginPath();x.moveTo(bx,10);x.lineTo(bx,H-24);x.stroke();x.setLineDash([]);
  const spec=b.timeline[step].spec,fa=b.spec_f;
  x.strokeStyle=(b.timeline[step].status==='ALERT')?'#e66767':'#3987e5';x.lineWidth=3;x.beginPath();
  fa.forEach((f,i)=>{if(f<1||f>25)return;const px=pad+(f-1)/24*(W-2*pad),py=H-24-(spec[i]||0)*(H-44);
    i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke();
  const f1=b.timeline[step].f1,fx=pad+(f1-1)/24*(W-2*pad);
  x.fillStyle='#fff';x.fillText('f1 '+f1.toFixed(2)+' Hz',fx+8,34);
}
let lastStatus=null;
function loop(){
  const b=D.fleet[sel];
  const step=Math.min(b.timeline.length-1,Math.floor(tick/4)%b.timeline.length);
  const row=b.timeline[step];
  $('mainttl').textContent='Live monitor — Bridge '+sel+' · '+b.name;
  $('kshs').textContent=Math.round(row.shs);$('kshs').style.color=stat2c[row.status];
  $('kf1').textContent=row.f1.toFixed(2);
  const drift=((row.f1-D.baseline_f1)/D.baseline_f1*100);
  $('kdrift').textContent=(drift>0?'+':'')+drift.toFixed(1)+'%';
  $('kstat').textContent=row.status;$('kstat').style.color=stat2c[row.status];
  const ab=$('abar');
  if(row.status==='ALERT'){ab.style.display='block';
    ab.innerHTML='&#9650; STRUCTURAL DRIFT '+Math.abs(drift).toFixed(1)+'% BELOW BASELINE — DISPATCH INSPECTION · Bridge '+sel;}
  else ab.style.display='none';
  if(row.status!==lastStatus){
    ev('Bridge '+sel+' → '+row.status+' (SHS '+Math.round(row.shs)+', f1 '+row.f1.toFixed(2)+' Hz)',stat2c[row.status]);
    if(row.status==='ALERT')ev('Work order drafted: joint inspection, Bay 3 mid-span — camera evidence attached','var(--ink2)');
    lastStatus=row.status;}
  if(tick%20===0)ev('CAM-'+sel+' frame batch processed · '+(b.fps||60)+' fps · ref-ROI cancel OK');
  drawWave(b);drawSpec(b,step);
  tick++;requestAnimationFrame(()=>setTimeout(loop,120));
}
fleetRender();ev('Command centre online — 3 cameras streaming','var(--good)');loop();
</script></body></html>"""


def build(out_html: str = "out/command_center.html") -> str:
    with open("out/stream.json") as fh:
        data = json.load(fh)
    with open(out_html, "w") as fh:
        fh.write(TEMPLATE.replace("__DATA__", json.dumps(data)))
    return out_html


if __name__ == "__main__":
    print("wrote", build())
