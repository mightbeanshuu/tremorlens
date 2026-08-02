"""Test matrix + blind reveal: the Testing & Validation evidence for the submission.

Runs the full clip set (plus stress variants with harder noise/flicker/jitter),
scores every run against the healthy baseline, executes a seeded blind reveal
(3 bridges, one secretly damaged — pipeline must pick it), and writes:
  out/results.json   (machine-readable, consumed by dashboard.py)
  out/TEST_REPORT.md (human-readable table for the design documentation)
"""
import json
import os
import random

import numpy as np

import testdata
import modal
import shs
import fusion


def run_matrix() -> dict:
    os.makedirs("out", exist_ok=True)
    testdata.render_set()

    # stress variants: what bad demo conditions look like
    stress = {
        "stress_high_noise":  dict(f1=7.30, f2=18.4, mode="motor", seed=21, noise_sigma=3.0),
        "stress_flicker_x4":  dict(f1=7.30, f2=18.4, mode="motor", seed=22, flicker=4.0),
        "stress_jitter_x4":   dict(f1=7.30, f2=18.4, mode="motor", seed=23, jitter_px=0.6),
    }
    for name, kw in stress.items():
        testdata.render_clip(f"data/{name}.mp4", **kw)

    gt = dict(testdata.GT, **{k: v["f1"] for k, v in stress.items()})
    clips = list(testdata.SET) + list(stress)

    results = {}
    for name in clips:
        r = modal.extract(f"data/{name}.mp4")
        err = abs(r.f1 - gt[name]) / gt[name] * 100
        results[name] = {"f1": round(r.f1, 3), "gt": gt[name], "err_pct": round(err, 2),
                         "peaks": [[round(f, 2), round(p, 2)] for f, p in r.peaks[:3]]}

    base = results["test_healthy_motor"]["f1"]
    noise = abs(results["test_healthy_take2"]["f1"] - base) / base * 100
    noise = max(noise, 0.2)
    base_r = modal.ModalResult(base, [], None, None, 60, None, [])
    for name, row in results.items():
        cur = modal.ModalResult(row["f1"], [], None, None, 60, None, [])
        v = shs.score(base_r, cur, noise_pct=noise)
        row.update(shs=v.shs, drift_pct=v.drift_pct, status=v.status)

    # spectra of the two hero runs for the dashboard chart
    hero = {}
    for name in ("test_healthy_motor", "test_damaged_motor"):
        r = modal.extract(f"data/{name}.mp4")
        m = (r.freqs >= 1.0) & (r.freqs <= 25.0)
        hero[name] = {"f": [round(float(x), 3) for x in r.freqs[m]],
                      "p": [float(x) for x in (r.psd[m] / r.psd[m].max())]}

    # ---- blind reveal: 3 bridges, one secretly damaged (seeded shuffle) ----
    rng = random.Random(42)
    bridges = [("A", 7.30, 18.4, 31), ("B", 7.30, 18.4, 32), ("C", 6.20, 16.9, 33)]
    rng.shuffle(bridges)
    reveal_rows, truth = [], None
    for label, (tag, f1, f2, seed) in zip("ABC", bridges):
        if f1 < 7.0:
            truth = label
        testdata.render_clip(f"data/blind_{label}.mp4", f1=f1, f2=f2, mode="motor", seed=seed)
        r = modal.extract(f"data/blind_{label}.mp4")
        v = shs.score(base_r, modal.ModalResult(r.f1, [], None, None, 60, None, []), noise_pct=noise)
        reveal_rows.append({"bridge": label, "f1": round(r.f1, 3), "shs": v.shs,
                            "drift_pct": v.drift_pct, "status": v.status})
    pick = min(reveal_rows, key=lambda x: x["shs"])["bridge"]

    fus = {
        "healthy": fusion.fuse("data/test_healthy_motor.mp4", "data/test_accel_healthy.csv",
                               "out/fusion_healthy.png"),
        "damaged": fusion.fuse("data/test_damaged_motor.mp4", "data/test_accel_damaged.csv",
                               "out/fusion_damaged.png"),
    }

    out = {"baseline_f1": base, "noise_pct": round(noise, 2), "runs": results,
           "hero_spectra": hero, "fusion": fus,
           "blind_reveal": {"rows": reveal_rows, "picked": pick, "truth": truth,
                            "correct": pick == truth}}
    with open("out/results.json", "w") as fh:
        json.dump(out, fh, indent=2)

    lines = ["# TremorLens — Test & Validation Report", "",
             f"Baseline f1: **{base:.3f} Hz** · calibrated noise floor: **{noise:.2f}%**", "",
             "| Run | GT (Hz) | Measured (Hz) | Err % | SHS | Status |",
             "|---|---|---|---|---|---|"]
    for name, row in results.items():
        lines.append(f"| {name} | {row['gt']:.2f} | {row['f1']:.3f} | {row['err_pct']:.2f} | "
                     f"{row['shs']:.1f} | {row['status']} |")
    lines += ["", "## Sensor fusion (camera vs contact accelerometer)",
              f"- healthy: camera {fus['healthy']['camera_f1']} Hz vs accel {fus['healthy']['accel_f1']} Hz "
              f"→ agreement {fus['healthy']['agreement_pct']}%",
              f"- damaged: camera {fus['damaged']['camera_f1']} Hz vs accel {fus['damaged']['accel_f1']} Hz "
              f"→ agreement {fus['damaged']['agreement_pct']}%", "",
              "## Blind reveal (one of three bridges secretly damaged)",
              "| Bridge | f1 (Hz) | SHS | Drift % | Status |", "|---|---|---|---|---|"]
    for row in reveal_rows:
        lines.append(f"| {row['bridge']} | {row['f1']:.3f} | {row['shs']:.1f} | "
                     f"{row['drift_pct']:.2f} | {row['status']} |")
    lines += ["", f"**Pipeline picked bridge {pick}; truth was {truth} → "
              f"{'CORRECT ✅' if pick == truth else 'WRONG ❌'}**"]
    with open("out/TEST_REPORT.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    r = run_matrix()
    print(json.dumps({k: v for k, v in r.items() if k not in ("hero_spectra",)}, indent=2)[:2000])
