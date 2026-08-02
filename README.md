# TremorLens — Every Camera a Structural Health Sensor

AI Arena 3.0 prototype. Turns any phone/CCTV into a non-contact vibration sensor:
phase-registration displacement extraction → modal frequency fingerprint →
**Structural Heartbeat Score (SHS)** → damage alert when the fingerprint drifts
beyond the calibrated noise floor. Eulerian motion magnification (MIT lineage:
Wu 2012 / Wadhwa 2013) renders the invisible vibration visible for inspection.


## For judges — three ways to see it working in 60 seconds
1. **Live on your phone**: [tremorlens-live.vercel.app](https://tremorlens-live.vercel.app) — tap *Start sensor mode*, hold the phone on a table, tap the table.
2. **On any device, zero hardware**: same link → **▶ Watch a 40-second replay** — simulated ground-truth data streamed through the identical live pipeline (badged REPLAY; healthy baseline → monsoon → ALERT).
3. **Reproduce the numbers**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mightbeanshuu/tremorlens/blob/main/verify.ipynb) — one *Run All* re-derives the 0.31% proof from physics-true synthetic ground truth (~10 min; expected output shown in the notebook).

## Pipeline
```
video → auto-ROI (speckle targets) → sub-pixel registration (upsampled phase correlation)
      → displacement time series → Welch PSD → modal peaks (parabolic refine)
      → SHS vs stored healthy baseline → HEALTHY / WATCH / ALERT
```

## SPANDAN engine (स्पंदन — "heartbeat")
*Structural Pulse ANalysis via Displacement And Novelty-detection* — the upgraded
scoring layer (`src/spandan.py`), following the Farrar–Worden statistical
pattern-recognition paradigm for SHM:

- **Damage localization** — per-ROI mode shapes at each modal peak via
  cross-spectral density (coherence-gated ≥ 0.7), MAC per mode + per-ROI
  COMAC/NMSD → red/green heatmap over the actual video frame. One camera =
  six virtual sensors; a single contact accelerometer cannot localize.
- **Novelty scoring** — Mahalanobis distance of a windowed spectral fingerprint
  against the learned healthy distribution, alert at the χ² 99% quantile.
- **Honesty layer** — block-bootstrap 95% CI on f₁, damping (half-power ζ)
  reported with published-field-error caveats, ±2% environmental (temperature)
  guard band: `ALERT iff drift > max(3σ_bootstrap, EOV band)`.

Lineage: Pandey/Biswas/Samman 1991 (shape-change localization), Lieven & Ewins
1988 (COMAC), Farrar & Worden (novelty detection). Learned motion-magnification
lineage (LVMM '18 → STB-VMM '23 → EulerMormer AAAI'24 → FD4MM CVPR'24 →
diffusion/SSM '25–'26) was evaluated; the classical phase-based measurement
channel was kept — published deep trackers report 0.4–0.7% on their own benchmarks;
this classical channel reaches **0.31%** on its physics-true synthetic rig (different data — stated plainly).

## Quick start
```bash
python3 -m venv .venv   # Python >=3.10; 3.12 recommended (on macOS/Homebrew, if pip bootstrap fails use python3.12 or uv venv) && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python src/run_verify.py       # end-to-end proof vs synthetic ground truth
./.venv/bin/python src/magnify.py data/test_healthy_motor.mp4 out/mag.mp4 --lo 6 --hi 9 --alpha 25   # run run_verify.py first (renders the test data)
```

`run_verify.py` renders healthy/damaged model bridges with known physics
(7.30 Hz vs 6.20 Hz), recovers f₁ within **0.4% (forced) / 1.2% (tap decay)**, and shows the damaged twin
triggering **ALERT (SHS 28.5)** while a healthy re-take stays **HEALTHY (SHS 100)**.

## Real hardware
See **[CAPTURE_PROTOCOL.md](CAPTURE_PROTOCOL.md)** — household rig (plank bridge,
marker speckle targets, hair-dryer broadband excitation), smartphone video +
[phyphox](https://phyphox.org) accelerometer ground truth. Drop captures in
`data/real/` and run:
```bash
./.venv/bin/python src/run_real.py         # auto-CFR, fusion overlay, SPANDAN report
```

## Modules
| File | Role |
|---|---|
| `src/modal.py` | Auto-ROI + sub-pixel tracking + modal fingerprint (f₁, ζ, per-ROI series) |
| `src/spandan.py` | SPANDAN engine: MAC/COMAC localization, Mahalanobis novelty, bootstrap CI, EOV guard |
| `src/shs.py` | Baseline diff → SHS + verdict (noise floor calibrated, not assumed) |
| `src/fusion.py` | Camera vs phyphox accelerometer agreement (robust CSV loader) |
| `src/run_real.py` | Real-capture pipeline: ingest, fusion, localization heatmap, report |
| `src/magnify.py` | Eulerian bandpass magnification for visualization B-roll |
| `src/run_verify.py` | End-to-end verification harness |

## Method & prior-art note
Implements the open academic pipeline (Wu et al. 2012; Wadhwa et al. 2013;
phase-correlation registration). Educational/research prototype; no affiliation
with, and no use of, any vendor's proprietary workflow or trademarks.

## Live demo
**[tremorlens-live.vercel.app](https://tremorlens-live.vercel.app)** (open on a phone): your phone becomes a TremorLens contact sensor. Live spectrum, f₁, and SHS verdict computed on-device in the browser (`webapp/`).
