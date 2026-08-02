"""Generates the TremorLens dashboard: one self-contained dark HTML file from out/results.json.

No CDNs, no external assets — fusion plots are embedded base64, the spectrum
chart is inline SVG with a hover crosshair, and the run table doubles as the
accessible table view of every chart.
"""
import base64
import json

SURFACE = "#141413"; PANEL = "#1a1a19"; EDGE = "#2a2a28"
INK = "#ffffff"; INK2 = "#c3c2b7"; MUT = "#8a897f"
BLUE = "#3987e5"; ORANGE = "#d95926"
GOOD = "#199e70"; WARNC = "#c98500"; BADC = "#e66767"


def _b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def _spectrum_svg(hero: dict, base_f1: float) -> str:
    w, h, pad = 860, 240, 34
    hm, dm = hero["test_healthy_motor"], hero["test_damaged_motor"]
    fmax = 25.0

    def pts(d):
        out = []
        for f, p in zip(d["f"], d["p"]):
            x = pad + (f - 1.0) / (fmax - 1.0) * (w - 2 * pad)
            y = h - pad - p * (h - 2 * pad - 8)
            out.append(f"{x:.1f},{y:.1f}")
        return " ".join(out)

    grid = "".join(
        f'<line x1="{pad + i * (w - 2 * pad) / 6:.0f}" y1="{pad}" x2="{pad + i * (w - 2 * pad) / 6:.0f}" '
        f'y2="{h - pad}" stroke="{EDGE}" stroke-width="1"/>'
        f'<text x="{pad + i * (w - 2 * pad) / 6:.0f}" y="{h - pad + 16}" fill="{MUT}" '
        f'font-size="10" text-anchor="middle">{1 + i * 4:.0f} Hz</text>'
        for i in range(7))
    bx = pad + (base_f1 - 1.0) / (fmax - 1.0) * (w - 2 * pad)
    return f'''
<svg id="spec" viewBox="0 0 {w} {h}" role="img" aria-label="Modal spectrum: healthy baseline vs damaged twin">
  {grid}
  <line x1="{bx:.0f}" y1="{pad}" x2="{bx:.0f}" y2="{h - pad}" stroke="{MUT}" stroke-width="1" stroke-dasharray="4 4"/>
  <text x="{bx + 6:.0f}" y="{pad + 12}" fill="{MUT}" font-size="10">baseline f1 {base_f1:.2f} Hz</text>
  <polyline points="{pts(hm)}" fill="none" stroke="{BLUE}" stroke-width="2"/>
  <polyline points="{pts(dm)}" fill="none" stroke="{ORANGE}" stroke-width="2"/>
  <line id="xh" x1="0" y1="{pad}" x2="0" y2="{h - pad}" stroke="{INK2}" stroke-width="1" opacity="0"/>
  <rect id="hit" x="{pad}" y="{pad}" width="{w - 2 * pad}" height="{h - 2 * pad}" fill="transparent"/>
</svg>
<div id="tip" style="position:absolute;display:none;background:{PANEL};border:1px solid {EDGE};
     border-radius:6px;padding:6px 9px;font-size:12px;color:{INK};pointer-events:none"></div>
<script>
const HM={json.dumps({"f": hm["f"], "p": hm["p"]})}, DM={json.dumps({"f": dm["f"], "p": dm["p"]})};
const svg=document.getElementById('spec'),hit=document.getElementById('hit'),
      xh=document.getElementById('xh'),tip=document.getElementById('tip');
hit.addEventListener('mousemove',e=>{{
  const r=svg.getBoundingClientRect(), w={w}, pad={pad};
  const px=(e.clientX-r.left)*w/r.width;
  const f=1+(px-pad)/(w-2*pad)*24; if(f<1||f>25)return;
  let i=HM.f.findIndex(x=>x>=f); if(i<1)i=1;
  xh.setAttribute('x1',px); xh.setAttribute('x2',px); xh.setAttribute('opacity',.6);
  tip.style.display='block'; tip.style.left=(e.pageX+14)+'px'; tip.style.top=(e.pageY-10)+'px';
  tip.innerHTML=f.toFixed(2)+' Hz<br><span style="color:{BLUE}">&#9632;</span> healthy '+HM.p[i].toFixed(2)
    +'<br><span style="color:{ORANGE}">&#9632;</span> damaged '+DM.p[i].toFixed(2);
}});
hit.addEventListener('mouseleave',()=>{{xh.setAttribute('opacity',0);tip.style.display='none';}});
</script>'''


def build(results_path: str = "out/results.json", out_html: str = "out/dashboard.html") -> str:
    with open(results_path) as fh:
        R = json.load(fh)
    dmg = R["runs"]["test_damaged_motor"]
    fus = R["fusion"]["damaged"]
    br = R["blind_reveal"]
    scolor = {"HEALTHY": GOOD, "WATCH": WARNC, "ALERT": BADC}

    rows = "".join(
        f"<tr><td>{n}</td><td>{r['gt']:.2f}</td><td>{r['f1']:.3f}</td><td>{r['err_pct']:.2f}%</td>"
        f"<td>{r['shs']:.1f}</td><td style='color:{scolor[r['status']]}'>&#9679; {r['status']}</td></tr>"
        for n, r in R["runs"].items())
    brrows = "".join(
        f"<tr><td>Bridge {x['bridge']}</td><td>{x['f1']:.3f}</td><td>{x['shs']:.1f}</td>"
        f"<td style='color:{scolor[x['status']]}'>&#9679; {x['status']}</td></tr>" for x in br["rows"])

    html = f'''<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TremorLens · SPANDAN — Structural Heartbeat</title><style>
body{{margin:0;background:{SURFACE};color:{INK};font:14px/1.5 -apple-system,'Segoe UI',sans-serif}}
.wrap{{max-width:960px;margin:0 auto;padding:28px 20px}}
.h{{display:flex;align-items:baseline;gap:14px}} .h small{{color:{MUT}}}
.badge{{margin-left:auto;padding:6px 14px;border-radius:999px;font-weight:600;
  background:{BADC}22;color:{BADC};border:1px solid {BADC}66}}
.kpis{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:22px 0}}
.kpi{{background:{PANEL};border:1px solid {EDGE};border-radius:10px;padding:14px}}
.kpi b{{display:block;font-size:22px;font-weight:650}} .kpi span{{color:{MUT};font-size:11.5px}}
.card{{background:{PANEL};border:1px solid {EDGE};border-radius:12px;padding:18px;margin:14px 0;position:relative}}
.card h2{{margin:0 0 4px;font-size:15px}} .card p.sub{{margin:0 0 12px;color:{MUT};font-size:12.5px}}
.leg{{display:flex;gap:16px;font-size:12px;color:{INK2};margin-bottom:6px}}
.leg i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
td,th{{padding:7px 10px;text-align:left;border-bottom:1px solid {EDGE}}} th{{color:{MUT};font-weight:500}}
img{{max-width:100%;border-radius:8px}}
@media(max-width:720px){{.kpis{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><div class="wrap">
<div class="h"><h1 style="font-size:20px;margin:0;display:flex;align-items:center"><svg width="34" height="34" viewBox="0 0 64 64" style="vertical-align:middle;margin-right:10px"><circle cx="32" cy="32" r="28" fill="none" stroke="#2DD4BF" stroke-width="4"/><path d="M10 32 h12 l5 -14 l9 26 l6 -12 h12" stroke="#fff" stroke-width="4" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>TremorLens <span style="color:#2DD4BF;font-weight:400;margin-left:10px">· SPANDAN Engine</span></h1>
<small>Model truss — Bay 3 test rig</small>
<span class="badge">&#9650; ALERT — SHS {dmg["shs"]:.0f}</span></div>

<div class="kpis">
<div class="kpi"><b>{dmg["shs"]:.0f}</b><span>Structural Heartbeat Score</span></div>
<div class="kpi"><b>{R["baseline_f1"]:.2f} Hz</b><span>Baseline f&#8321;</span></div>
<div class="kpi"><b>{dmg["f1"]:.2f} Hz</b><span>Current f&#8321;</span></div>
<div class="kpi"><b style="color:{BADC}">-{dmg["drift_pct"]:.1f}%</b><span>Frequency drift</span></div>
<div class="kpi"><b>{fus["agreement_pct"]:.2f}%</b><span>Camera vs accel gap</span></div>
</div>

<div class="card"><h2>Modal fingerprint</h2>
<p class="sub">Displacement spectrum from camera only — no contact sensors. Damaged twin's fundamental
has dropped {dmg["drift_pct"]:.1f}% below the healthy baseline.</p>
<div class="leg"><span><i style="background:{BLUE}"></i>Healthy baseline</span>
<span><i style="background:{ORANGE}"></i>Damaged twin</span></div>
{_spectrum_svg(R["hero_spectra"], R["baseline_f1"])}</div>

<div class="card"><h2>Ground truth: contact accelerometer agrees</h2>
<p class="sub">Phone accelerometer taped to the deck (phyphox export) vs TremorLens camera reading —
f&#8321; agreement {fus["agreement_pct"]:.2f}% on the damaged twin.</p>
<img alt="Camera vs accelerometer PSD overlay" src="data:image/png;base64,{_b64(fus["plot"])}"></div>

<div class="card"><h2>Blind reveal</h2>
<p class="sub">Three bridges, one secretly tampered. TremorLens picked
<b>Bridge {br["picked"]}</b> — {"correct" if br["correct"] else "wrong"}.</p>
<table><tr><th>Bridge</th><th>f&#8321; (Hz)</th><th>SHS</th><th>Status</th></tr>{brrows}</table></div>

<div class="card"><h2>All validation runs</h2>
<p class="sub">Full test matrix incl. stress variants (3&times; noise, 4&times; flicker, 4&times; jitter).
Calibrated noise floor {R["noise_pct"]:.2f}%.</p>
<table><tr><th>Run</th><th>GT (Hz)</th><th>Measured</th><th>Error</th><th>SHS</th><th>Status</th></tr>
{rows}</table></div>

<p style="color:{MUT};font-size:11.5px">TremorLens · SPANDAN Engine — modal fingerprint · dysphonia panel (jitter/HNR/THD) · Mahalanobis novelty · MAC/COMAC + transmissibility localization · DIN 4150-3 / IS-ISO 4866 screening · phase-based lineage (
Wu 2012 / Wadhwa 2013) + modal fingerprinting. Educational research prototype.</p>
</div></body></html>'''
    with open(out_html, "w") as fh:
        fh.write(html)
    return out_html


if __name__ == "__main__":
    print("wrote", build())
