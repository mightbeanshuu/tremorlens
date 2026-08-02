"""Regenerates every SPANDAN-layer number in Design Documentation §5.2–§5.7.

One command, no arguments:  python src/run_spandan_tests.py
(Requires the test set from run_verify.py; renders it if missing.)
"""
import numpy as np
from scipy.signal import lfilter, welch

import fusion
import modal
import spandan
import supernyquist as sn
import testdata


def _resonator_bank(local_shift: float, seed: int = 3):
    """Six-channel synthetic rig: shared mode, one channel's local path shifted."""
    rng = np.random.default_rng(seed)
    fps, n = 60.0, 3600
    base = rng.standard_normal(n)
    def resonator(x, f0, zeta=0.03):
        w = 2 * np.pi * f0 / fps
        a1 = -2 * np.exp(-zeta * w) * np.cos(w * np.sqrt(1 - zeta ** 2))
        a2 = np.exp(-2 * zeta * w)
        return lfilter([1.0], [1.0, a1, a2], x)
    sigs = [resonator(base, 7.3 + (local_shift if i == 3 else 0.0))
            + 0.05 * rng.standard_normal(n) for i in range(6)]
    return modal.ModalResult(7.3, [(7.3, 1.0)], np.array([]), np.array([]), fps,
                             sigs[0], [(i * 100, 0, 48, 48) for i in range(6)],
                             None, sigs)


def _bilinear(n, fps, f0, kratio, forcing, zeta=0.02):
    w2o, w2c = (2 * np.pi * f0) ** 2, (2 * np.pi * f0) ** 2 * kratio
    x = v = 0.0
    out = np.empty(n)
    dt, sub = 1.0 / fps, 8
    for i in range(n):
        for _ in range(sub):
            w2 = w2o if x >= 0 else w2c
            a = -w2 * x - 2 * zeta * np.sqrt(w2) * v + forcing[i]
            v += a * dt / sub
            x += v * dt / sub
        out[i] = x
    return out


def _f1_of(x, fps=60.0):
    f, p = welch(x, fs=fps, nperseg=int(fps * 4))
    m = (f >= 1) & (f <= 25)
    return float(f[m][np.argmax(p[m])])


def main() -> int:
    ok = True
    print("== SPANDAN test suite (regenerates doc §5.2–§5.7) ==")
    testdata.render_set()
    h = modal.extract("data/test_healthy_motor.mp4")
    h2 = modal.extract("data/test_healthy_take2.mp4")
    d = modal.extract("data/test_damaged_motor.mp4")

    print("\n[§5.2] pairwise-transmissibility localization (seeded LOCAL fault)")
    tfh = spandan.transmissibility(_resonator_bank(0.0))
    tfh2 = spandan.transmissibility(_resonator_bank(0.0))
    tfd = spandan.transmissibility(_resonator_bank(-1.1))
    same = spandan.tf_damage(tfh, tfh2)
    diff = spandan.tf_damage(tfh, tfd)
    print(f"  repeat LSD {same['lsd']}")
    print(f"  damaged LSD {diff['lsd']} -> worst region #{diff['worst_roi_rank'] + 1} (truth #4)")
    ok &= diff["worst_roi_rank"] == 3

    print("\n[§5.3] dysphonia panel, three regimes (bilinear breathing-crack physics)")
    fps, n = 60.0, int(60.0 * 90)
    rng = np.random.default_rng(5)
    Fr = rng.standard_normal(n) * 50
    t = np.arange(n) / fps
    Fh = 30 * np.sin(2 * np.pi * 8.0 * t)
    hh = _bilinear(n, fps, 7.3, 1.0, Fr)
    dd = _bilinear(n, fps, 7.3, 1.5, Fr)
    p_h, p_d = spandan.dysphonia(hh, fps, _f1_of(hh)), spandan.dysphonia(dd, fps, _f1_of(dd))
    hnr_drop = p_h["hnr_db"] - p_d["hnr_db"]
    print(f"  crack+random  : HNR {p_h['hnr_db']} -> {p_d['hnr_db']} dB (drop {hnr_drop:.1f})")
    ok &= hnr_drop > 2
    h2t = _bilinear(n, fps, 7.3, 1.0, Fh)
    d2t = _bilinear(n, fps, 7.3, 1.5, Fh)
    p_h2, p_d2 = spandan.dysphonia(h2t, fps, _f1_of(h2t)), spandan.dysphonia(d2t, fps, _f1_of(d2t))
    print(f"  crack+tonal   : THD {p_h2['thd_pct']} -> {p_d2['thd_pct']}%")
    ok &= p_d2["thd_pct"] > 2 * max(p_h2["thd_pct"], 0.01)
    imp = 0.5 * np.abs(h2t).max() * (rng.random(n) < 0.03) * rng.standard_normal(n)
    p_r = spandan.dysphonia(h2t + imp, fps, _f1_of(h2t))
    print(f"  rattle+steady : jitter {p_h2['jitter_pct']} -> {p_r['jitter_pct']}%")
    ok &= p_r["jitter_pct"] > 2 * p_h2["jitter_pct"]

    print("\n[§5.4] gated adaptive baseline + CUSUM")
    base_feats = np.vstack([spandan._window_features(r.displacement, r.fps) for r in (h, h2)])
    ab = spandan.AdaptiveBaseline(base_feats)
    drift_res = [ab.step(x) for x in spandan._window_features(h2.displacement * 1.02, h2.fps)]
    dam_res = [ab.step(x) for x in spandan._window_features(d.displacement, d.fps)]
    print(f"  drifted-healthy learned {sum(r['learned'] for r in drift_res)}/{len(drift_res)}; "
          f"damaged learned {sum(r['learned'] for r in dam_res)}/{len(dam_res)}, "
          f"novel {sum(r['novel'] for r in dam_res)}/{len(dam_res)}")
    ok &= sum(r["learned"] for r in dam_res) == 0
    cus = spandan.cusum_drift([h.f1 * (1 - 0.001 * i) for i in range(30)], h.f1, 0.05)
    print(f"  CUSUM on 0.1%/window creep: alarm={cus['alarm']} at window {cus['tripped_at_window']}")
    ok &= cus["alarm"]

    print("\n[§5.5] gated damage-type diagnosis (needs out/damage_model.joblib)")
    import sim_damage
    diag = sim_damage.diagnose(h.displacement, d.displacement, d.fps)
    if diag:
        print(f"  damaged twin -> {diag['prediction']} (conformal set {diag['conformal_set_90']})")
        ok &= diag["prediction"] == "support_loss"
    else:
        print("  model not trained — run: python src/sim_damage.py (~10 min); skipping assert")

    print("\n[§5.6] super-Nyquist rolling-shutter vibrometry")
    def render_rs(f_vib, fps_v, row_time, n_frames=60, n_rows=240, n_cols=64, amp=6.0):
        rng2 = np.random.default_rng(1)
        frames = np.empty((n_frames, n_rows, n_cols))
        xs = np.arange(n_cols)
        for nf in range(n_frames):
            for m in range(n_rows):
                tt = nf / fps_v + m * row_time
                edge = 32 + amp * np.sin(2 * np.pi * f_vib * tt)
                frames[nf, m] = 200 / (1 + np.exp(-(xs - edge) * 1.5)) + 20
        return frames + rng2.standard_normal(frames.shape) * 2.0
    for regime, r in [("slow-scan 28.5us", 28.5e-6), ("burst 4.9us", 4.9e-6)]:
        x = sn.track_edge_rows(render_rs(167.0, 60.0, r))
        f, p = sn.rs_spectrum(x, 60.0, r)
        fpk = sn.peak_freq(f, p)
        print(f"  {regime}: recovered {fpk:.2f} Hz (truth 167.00, frame Nyquist 30)")
        ok &= abs(fpk - 167) < 2

    print("\n[§5.7] standards screening calibration")
    tt = np.arange(0, 30, 1 / 200)
    din_a = spandan.ppv_screening(0.5 * np.sin(2 * np.pi * 8 * tt), 200.0)
    print(f"  accel channel : PPV {din_a['ppv_mm_s']} mm/s (analytic 9.95)")
    ok &= abs(din_a["ppv_mm_s"] - 9.95) < 1.0
    tt2 = np.arange(0, 30, 1 / 60.0)
    din_c = spandan.ppv_screening_disp(0.2 * np.sin(2 * np.pi * 8 * tt2), 60.0)
    print(f"  camera channel: PPV {din_c['ppv_mm_s']} mm/s (analytic 10.05)")
    ok &= abs(din_c["ppv_mm_s"] - 10.05) < 0.5

    print("\n[jitter-clock test] ±30% jittered sensor timestamps")
    rngj = np.random.default_rng(1)
    tj = np.cumsum(rngj.uniform(0.007, 0.013, 4000))
    import csv as _csv, tempfile, os as _os
    fpath = _os.path.join(tempfile.gettempdir(), "jitter_test.csv")
    with open(fpath, "w", newline="") as fh:
        w = _csv.writer(fh)
        for ti, ai in zip(tj, np.sin(2 * np.pi * 8 * tj)):
            w.writerow([ti, ai])
    f1j, _, _ = fusion.accel_f1(fpath)
    print(f"  recovered {f1j:.3f} Hz (truth 8.000)")
    ok &= abs(f1j - 8.0) < 0.1

    print("\nRESULT:", "PASS ✅" if ok else "FAIL ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
