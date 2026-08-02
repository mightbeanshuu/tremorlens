"""Real-hardware run: videos + phyphox CSVs from data/real/ -> genuine artifact set.

Drop your captures in data/real/ (any subset works, healthy.mp4 is the minimum):
  healthy.mp4          baseline take (tap or hair-dryer excitation)
  healthy_take2.mp4    repeat of baseline, nothing changed -> measured noise floor
  damaged.mp4          take after the gross structural change
  accel_healthy.csv    phyphox export recorded during/for the healthy state
  accel_damaged.csv    phyphox export for the damaged state

Then:  python src/run_real.py
Outputs land in out/ as real_*.png / real_report.json, ready for docs & deck.

Phone videos are often variable-frame-rate; if ffmpeg is on PATH each video is
first normalized to constant-frame-rate into data/real/cfr/ so the PSD frequency
axis is trustworthy.
"""
import json
import os
import shutil
import subprocess

import numpy as np

import fusion
import modal
import shs
import spandan

REAL = "data/real"
CFR = os.path.join(REAL, "cfr")
VIDEOS = ["healthy", "healthy_take2", "damaged"]


def _cfr(name: str) -> str | None:
    """Return a constant-frame-rate path for data/real/<name>.{mp4,mov,MOV}, or None if absent."""
    src = next((p for ext in ("mp4", "mov", "MOV")
                if os.path.exists(p := os.path.join(REAL, f"{name}.{ext}"))), None)
    if not src:
        return None
    if not shutil.which("ffmpeg"):
        print(f"  [warn] ffmpeg not found -- using {src} as-is (VFR phones may smear the spectrum)")
        return src
    dst = os.path.join(CFR, f"{name}.mp4")
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return dst
    os.makedirs(CFR, exist_ok=True)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=avg_frame_rate", "-of", "csv=p=0", src],
        capture_output=True, text=True)
    num, _, den = probe.stdout.strip().partition("/")
    fps = round(float(num) / float(den or 1))
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", src, "-vsync", "cfr", "-r", str(fps),
         "-an", "-pix_fmt", "yuv420p", dst], check=True)
    print(f"  normalized {name}.mp4 -> CFR {fps} fps")
    return dst


def _frac(a: modal.ModalResult, b: modal.ModalResult,
          fmin: float = 1.0, fmax: float = 25.0) -> float:
    """FRAC-style spectral similarity: Pearson correlation of log-PSDs on a common grid."""
    grid = np.linspace(fmin, fmax, 200)
    pa = np.interp(grid, a.freqs, a.psd)
    pb = np.interp(grid, b.freqs, b.psd)
    la, lb = np.log10(pa + 1e-20), np.log10(pb + 1e-20)
    return float(np.corrcoef(la, lb)[0, 1])


def main() -> int:
    print("== TremorLens REAL-DATA run ==")
    os.makedirs("out", exist_ok=True)
    paths = {n: _cfr(n) for n in VIDEOS}
    if not paths["healthy"]:
        print(f"nothing to do: put at least healthy.mp4 in {REAL}/")
        return 1

    res = {}
    for name, p in paths.items():
        if not p:
            print(f"{name:14s} -- missing, skipped")
            continue
        r = modal.extract(p)
        res[name] = r
        peaks = ", ".join(f"{f:.2f}Hz({v:.2f})" for f, v in r.peaks[:3])
        z = f"  zeta {r.zeta * 100:.2f}%" if r.zeta else ""
        print(f"{name:14s} f1 = {r.f1:6.3f} Hz @ {r.fps:.0f} fps{z} | peaks: {peaks}")

    report: dict = {"f1": {k: round(v.f1, 3) for k, v in res.items()},
                    "zeta_pct": {k: round(v.zeta * 100, 2) for k, v in res.items() if v.zeta}}

    # metric scale: data/real/scale.json {"target_mm": 50.0, "target_px": 240}
    # (printed speckle target of known size, measured in one frame) -> px -> mm.
    # Unlocks amplitude in real units + fully non-contact DIN 4150-3 screening.
    scale_path = os.path.join(REAL, "scale.json")
    mm_per_px = None
    if os.path.exists(scale_path):
        with open(scale_path) as fh:
            sc = json.load(fh)
        mm_per_px = float(sc["target_mm"]) / float(sc["target_px"])
        report["scale_mm_per_px"] = round(mm_per_px, 5)
        print(f"metric scale: {mm_per_px:.4f} mm/px (target {sc['target_mm']} mm = {sc['target_px']} px)")
        for name, r in res.items():
            amp_mm = float(np.percentile(np.abs(r.displacement), 99) * mm_per_px)
            print(f"  {name}: peak displacement ~{amp_mm*1000:.0f} um")
            report.setdefault("amplitude_um", {})[name] = round(amp_mm * 1000, 1)
            din_cam = spandan.ppv_screening_disp(r.displacement * mm_per_px, r.fps)
            print(f"  {name}: NON-CONTACT DIN screening: PPV {din_cam['ppv_mm_s']} mm/s "
                  f"vs {din_cam['din_limit_mm_s']} -> {din_cam['verdict']}")
            report.setdefault("din_camera", {})[name] = din_cam

    if "healthy_take2" in res:
        noise = abs(res["healthy_take2"].f1 - res["healthy"].f1) / res["healthy"].f1 * 100
        print(f"measured noise floor (repeat takes): {noise:.2f}%")
    else:
        noise = 1.0
        print("no healthy_take2.mp4 -- assuming 1.0% noise floor (capture a repeat take to measure it)")
    report["noise_pct"] = round(noise, 2)

    if "healthy_take2" in res:
        v = shs.score(res["healthy"], res["healthy_take2"], noise_pct=max(noise, 0.2))
        print(f"healthy re-take: SHS {v.shs:5.1f}  {v.status}")
        report["retake"] = {"shs": v.shs, "status": v.status, "detail": v.detail}
    if "damaged" in res:
        v = shs.score(res["healthy"], res["damaged"], noise_pct=max(noise, 0.2))
        print(f"damaged       : SHS {v.shs:5.1f}  {v.status}  ({v.detail})")
        report["damaged"] = {"shs": v.shs, "status": v.status, "detail": v.detail}

    # supporting spectral-fingerprint indicators (secondary evidence, not the alarm)
    for name in ("healthy_take2", "damaged"):
        if name in res:
            frac = _frac(res["healthy"], res[name])
            print(f"log-PSD correlation healthy vs {name}: {frac:.3f}")
            report.setdefault("frac", {})[name] = round(frac, 3)

    # ---- SPANDAN engine: honesty layer + novelty + localization ----
    print("-- SPANDAN engine --")
    boot = spandan.bootstrap_f1(res["healthy"].displacement, res["healthy"].fps)
    report["bootstrap_healthy"] = boot
    if boot["f1_ci95"]:
        print(f"healthy f1 95% CI: [{boot['f1_ci95'][0]}, {boot['f1_ci95'][1]}] Hz "
              f"(sigma {boot['sigma_pct']}%)")
    if "damaged" in res:
        gv = spandan.guarded_verdict(res["healthy"].f1, res["damaged"].f1,
                                     boot["sigma_pct"])
        print(f"guarded verdict: drift {gv['drift_pct']}% vs guard {gv['guard_pct']}% "
              f"-> {'ALERT' if gv['alert'] else 'within guard band'}")
        report["guarded_verdict"] = gv

        baselines = [(res["healthy"].displacement, res["healthy"].fps)]
        if "healthy_take2" in res:
            baselines.append((res["healthy_take2"].displacement, res["healthy_take2"].fps))
        nov = spandan.novelty(baselines, res["damaged"].displacement, res["damaged"].fps)
        print(f"novelty (Mahalanobis): D2 {nov['d2_median']} vs chi2-99% {nov['chi2_99']} "
              f"-> {'NOVEL' if nov['novel'] else 'normal'} "
              f"({nov['windows_flagged']}/{nov['windows_total']} windows)")
        report["novelty"] = nov

        # adaptive layer: gated baseline + slow-drift CUSUM + excitation class
        base_feats = np.vstack([spandan._window_features(d, fps) for d, fps in baselines])
        ab = spandan.AdaptiveBaseline(base_feats)
        cur_feats = spandan._window_features(res["damaged"].displacement, res["damaged"].fps)
        novel_ct = sum(ab.step(x)["novel"] for x in cur_feats)
        print(f"adaptive baseline: {ab.accepted} windows learned, {ab.rejected} rejected "
              f"(damage never absorbed), {novel_ct}/{len(cur_feats)} damaged windows novel")
        report["adaptive"] = {"learned": ab.accepted, "rejected": ab.rejected,
                              "novel_windows": int(novel_ct), "total_windows": len(cur_feats)}
        cus = spandan.cusum_drift([float(x[0]) for x in cur_feats], res["healthy"].f1,
                                  boot["sigma_pct"])
        print(f"CUSUM slow-drift: up {cus['cusum_up']} / down {cus['cusum_down']} "
              f"-> {'ALARM' if cus['alarm'] else 'quiet'}")
        report["cusum"] = cus
        for nm in ("healthy", "damaged"):
            exc = spandan.classify_excitation(res[nm].displacement)
            print(f"excitation [{nm}]: {exc['kind']} (crest {exc['crest']}, kurtosis {exc['kurtosis']})")
            report.setdefault("excitation", {})[nm] = exc

        tf_h = spandan.transmissibility(res["healthy"])
        tf_d = spandan.transmissibility(res["damaged"])
        if tf_h and tf_d:
            tfd = spandan.tf_damage(tf_h, tf_d)
            print(f"transmissibility: mean log-spectral distance {tfd['mean_lsd']} "
                  f"(excitation-independent), worst ROI (L->R) #{tfd['worst_roi_rank'] + 1}")
            report["transmissibility"] = tfd

        # structural dysphonia panel: the structure's "voice quality"
        dys_h = spandan.dysphonia(res["healthy"].displacement, res["healthy"].fps,
                                  res["healthy"].f1)
        dys_d = spandan.dysphonia(res["damaged"].displacement, res["damaged"].fps,
                                  res["damaged"].f1)
        if dys_h and dys_d:
            print(f"dysphonia: jitter {dys_h['jitter_pct']}->{dys_d['jitter_pct']}% | "
                  f"HNR {dys_h['hnr_db']}->{dys_d['hnr_db']} dB | "
                  f"THD {dys_h['thd_pct']}->{dys_d['thd_pct']}%")
            report["dysphonia"] = {"healthy": dys_h, "damaged": dys_d}
        fap = spandan.ls_fap(res["healthy"].displacement, res["healthy"].fps,
                             res["healthy"].f1)
        print(f"Lomb-Scargle: f1 power {fap['ls_power']}, false-alarm prob {fap['fap']}")
        report["ls_fap"] = fap

        sh_h = spandan.mode_shapes(res["healthy"])
        sh_d = spandan.mode_shapes(res["damaged"])
        if sh_h and sh_d:
            cmp = spandan.compare_shapes(sh_h, sh_d)
            print(f"MAC per mode: {cmp['mac_per_mode']}  "
                  f"worst ROI (L->R rank): #{cmp['worst_roi_rank'] + 1}")
            report["localization"] = cmp
            rois_lr = [res["damaged"].rois[i] for i in sh_d.order]
            scores = cmp["comac"] or [1.0 - v / (max(cmp["nmsd"]) or 1.0) for v in cmp["nmsd"]]
            spandan.heatmap_overlay(paths["damaged"], rois_lr, scores,
                                    cmp["low_confidence_rois"], "out/real_localization.png")
            print("localization heatmap -> out/real_localization.png")

        # sim-trained damage-TYPE diagnosis + conformal set (model-assisted SHM).
        # GATED on detection: the Farrar-Worden hierarchy names a mechanism only
        # AFTER the novelty/drift layer has flagged change — running a damage
        # classifier on a healthy structure invites hallucinated diagnoses.
        if gv["alert"] or nov["novel"]:
            try:
                import sim_damage
                diag = sim_damage.diagnose(res["healthy"].displacement,
                                           res["damaged"].displacement,
                                           res["damaged"].fps)
                if diag:
                    print(f"damage-type diagnosis: {diag['prediction']} "
                          f"(conformal 90% set: {diag['conformal_set_90']}) "
                          f"proba {diag['proba']}")
                    report["diagnosis"] = diag
            except Exception as e:
                print(f"  [warn] diagnosis unavailable: {e}")
        else:
            print("damage-type diagnosis: skipped (no detection flag — "
                  "diagnosis only names mechanisms for flagged changes)")

        # Structural APGAR: five measured signs -> one 0-10 triage card
        if dys_h and dys_d:
            shape = (report.get("localization", {}).get("mac_per_mode") or [None])[0]
            ap = spandan.apgar(
                report["guarded_verdict"]["drift_pct"], noise,
                dys_h["jitter_pct"], dys_d["jitter_pct"],
                dys_h["hnr_db"], dys_d["hnr_db"],
                res["healthy"].zeta, res["damaged"].zeta, shape)
            print(f"Structural APGAR: {ap['signs']} -> {ap['total']}/10 ({ap['verdict']})")
            report["apgar"] = ap

    for state in ("healthy", "damaged"):
        acsv = os.path.join(REAL, f"accel_{state}.csv")
        if state in res and os.path.exists(acsv):
            r = fusion.fuse(paths[state], acsv, f"out/real_fusion_{state}.png",
                            f"out/real_fusion_{state}.json")
            print(f"fusion {state}: camera {r['camera_f1']} Hz vs accel {r['accel_f1']} Hz "
                  f"-> agreement {r['agreement_pct']}%")
            report[f"fusion_{state}"] = r
            a, fs_a = fusion.load_phyphox(acsv)
            din = spandan.ppv_screening(a, fs_a)
            print(f"DIN 4150-3 screening [{state}]: PPV {din['ppv_mm_s']} mm/s @ "
                  f"{din['dominant_hz']} Hz vs limit {din['din_limit_mm_s']} mm/s "
                  f"({din['category']}) -> {din['verdict']}")
            report[f"din_{state}"] = din

    with open("out/real_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print("report -> out/real_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
