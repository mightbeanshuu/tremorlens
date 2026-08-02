"""TremorLens · SPANDAN Engine — hosted analysis Space.

Wraps the exact pipeline from the repo's src/ (copied into this Space):
video -> auto-ROI -> sub-pixel phase correlation -> modal fingerprint ->
SPANDAN verdicts (SHS, guarded drift, novelty, transmissibility localization).
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fusion
import modal
import shs
import spandan

SURFACE, INK, INK2, BLUE, ORANGE, GRID = ("#1a1a19", "#ffffff", "#c3c2b7",
                                          "#3987e5", "#d95926", "#33332f")


def _spectrum_png(results: dict) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=140)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for (name, r), color in zip(results.items(), (BLUE, ORANGE, "#3fb950")):
        m = (r.freqs >= 1.0) & (r.freqs <= 25.0)
        if not m.any():
            continue
        psd = r.psd[m] / max(r.psd[m].max(), 1e-20)
        ax.plot(r.freqs[m], psd, color=color, lw=2,
                label=f"{name}: f1 {r.f1:.2f} Hz")
    ax.set_xlabel("Frequency (Hz)", color=INK2, fontsize=9)
    ax.set_ylabel("Normalized PSD", color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(color=GRID, lw=0.5, alpha=0.6)
    ax.legend(facecolor=SURFACE, edgecolor=GRID, labelcolor=INK, fontsize=8)
    fig.tight_layout()
    out = tempfile.mktemp(suffix=".png")
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def analyze(baseline_video, current_video, accel_csv, progress=gr.Progress()):
    if baseline_video is None:
        return ("Upload at least a **baseline (healthy) video** — a structure with "
                "high-contrast speckle targets, filmed from a fixed camera."), None, None
    md = ["## TremorLens · SPANDAN report"]
    results = {}

    progress(0.1, desc="Tracking baseline video (sub-pixel phase correlation)…")
    rb = modal.extract(baseline_video)
    results["baseline"] = rb
    boot = spandan.bootstrap_f1(rb.displacement, rb.fps)
    ci = (f" (95% CI [{boot['f1_ci95'][0]}, {boot['f1_ci95'][1]}] Hz)"
          if boot.get("f1_ci95") else "")
    z = f", damping ζ ≈ {rb.zeta*100:.1f}%" if rb.zeta else ""
    exc = spandan.classify_excitation(rb.displacement)
    md.append(f"**Baseline:** f₁ = **{rb.f1:.3f} Hz**{ci} @ {rb.fps:.0f} fps{z} · "
              f"excitation classified **{exc['kind']}**")

    heat = None
    if current_video is not None:
        progress(0.45, desc="Tracking current-state video…")
        rc = modal.extract(current_video)
        results["current"] = rc
        v = shs.score(rb, rc, noise_pct=1.0)
        gv = spandan.guarded_verdict(rb.f1, rc.f1, boot.get("sigma_pct"))
        icon = {"HEALTHY": "🟢", "WATCH": "🟠", "ALERT": "🔴"}[v.status]
        md.append(f"**Verdict:** {icon} **{v.status}** · SHS {v.shs} · drift "
                  f"{gv['drift_pct']}% vs guard {gv['guard_pct']}% "
                  f"(rule: {gv['rule']})")
        nov = spandan.novelty([(rb.displacement, rb.fps)], rc.displacement, rc.fps)
        md.append(f"**Novelty (Mahalanobis):** D² {nov['d2_median']} vs χ²-99% "
                  f"{nov['chi2_99']} → {'**NOVEL**' if nov['novel'] else 'normal'} "
                  f"({nov['windows_flagged']}/{nov['windows_total']} windows)")
        tf_b, tf_c = spandan.transmissibility(rb), spandan.transmissibility(rc)
        if tf_b and tf_c:
            tfd = spandan.tf_damage(tf_b, tf_c)
            md.append(f"**Transmissibility localization (excitation-independent):** "
                      f"strongest change at region **#{tfd['worst_roi_rank']+1}** "
                      f"(left→right) · per-region LSD {tfd['lsd']}")
        sh_b, sh_c = spandan.mode_shapes(rb), spandan.mode_shapes(rc)
        if sh_b and sh_c:
            cmp = spandan.compare_shapes(sh_b, sh_c)
            md.append(f"**Mode shapes:** MAC {cmp['mac_per_mode']} · shape-change "
                      f"heatmap below")
            rois_lr = [rc.rois[i] for i in sh_c.order]
            scores = cmp["comac"] or [1.0 - v / (max(cmp["nmsd"]) or 1.0)
                                      for v in cmp["nmsd"]]
            heat = tempfile.mktemp(suffix=".png")
            spandan.heatmap_overlay(current_video, rois_lr, scores,
                                    cmp["low_confidence_rois"], heat)

    if accel_csv is not None:
        progress(0.8, desc="Fusing with accelerometer ground truth…")
        try:
            fpng = tempfile.mktemp(suffix=".png")
            r = fusion.fuse(baseline_video, accel_csv, fpng)
            md.append(f"**Camera vs accelerometer:** camera {r['camera_f1']} Hz vs "
                      f"contact {r['accel_f1']} Hz → agreement "
                      f"**{r['agreement_pct']}%**")
        except Exception as e:
            md.append(f"*Accelerometer CSV could not be fused: {e}*")

    progress(0.95, desc="Rendering spectra…")
    spec = _spectrum_png(results)
    md.append("\n*Every number regenerates from the open pipeline: "
              "[github.com/mightbeanshuu/tremorlens]"
              "(https://github.com/mightbeanshuu/tremorlens)*")
    return "\n\n".join(md), spec, heat


demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.Video(label="Baseline (healthy) video — fixed camera, speckle targets"),
        gr.Video(label="Current-state video (optional — unlocks verdict + localization)"),
        gr.File(label="phyphox accelerometer CSV (optional — ground-truth fusion)"),
    ],
    outputs=[
        gr.Markdown(label="SPANDAN report"),
        gr.Image(label="Frequency fingerprint"),
        gr.Image(label="Damage localization heatmap"),
    ],
    title="TremorLens · SPANDAN Engine 🌉",
    description=("Every camera a structural health sensor: sub-pixel vibration from ordinary "
                 "video → modal fingerprint → damage verdict, novelty score, and localization. "
                 "Phone-sensor live demo: [tremorlens-live.vercel.app](https://tremorlens-live.vercel.app)"),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
