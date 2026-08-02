"""Sensor fusion validation: camera-derived spectrum vs contact accelerometer (phyphox CSV).

Produces the agreement number for the docs (target: f1 within ~1%) and the
overlay plot for the dashboard/deck.
"""
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch, detrend

import modal

SURFACE = "#1a1a19"
INK = "#ffffff"
INK2 = "#c3c2b7"
CAMERA = "#3987e5"   # series 1
ACCEL = "#d95926"    # series 2
GRID = "#33332f"


def load_phyphox(path: str) -> tuple[np.ndarray, float]:
    """Robust phyphox CSV loader.

    Handles both our 2-column synthetic exports and real phyphox exports
    (comma/semicolon/tab delimited, header row, x/y/z/absolute columns,
    decimal comma). Time = column named *time* (else col 0); signal = the
    single AXIS with the largest variance. The 'absolute' magnitude column is
    excluded: |a| rectifies the oscillation and doubles its apparent frequency.
    """
    with open(path) as fh:
        sample = fh.read(4096)
        fh.seek(0)
        delim = max(",;\t", key=sample.count)
        rows = list(csv.reader(fh, delimiter=delim))
    if not rows:
        raise ValueError(f"empty CSV: {path}")

    header, data_rows = None, rows
    def _num(s: str) -> float:
        return float(s.strip().strip('"').replace(",", "."))
    try:
        _num(rows[0][0])
    except (ValueError, IndexError):
        header, data_rows = [c.strip().strip('"').lower() for c in rows[0]], rows[1:]

    cols: list[list[float]] = []
    for row in data_rows:
        try:
            vals = [_num(c) for c in row]
        except (ValueError, IndexError):
            continue
        if not cols:
            cols = [[] for _ in vals]
        if len(vals) == len(cols):
            for c, v in zip(cols, vals):
                c.append(v)
    if len(cols) < 2:
        raise ValueError(f"need >=2 numeric columns in {path}")
    arr = [np.array(c) for c in cols]

    t_idx = 0
    if header:
        for i, name in enumerate(header[:len(arr)]):
            if "time" in name:
                t_idx = i
                break
    candidates = [i for i in range(len(arr)) if i != t_idx]
    if header:  # |a| magnitude rectifies the signal -> apparent frequency doubles
        candidates = [i for i in candidates
                      if i >= len(header) or "absolute" not in header[i]] or candidates
    sig_idx = max(candidates, key=lambda i: float(np.var(arr[i])))

    t, a = arr[t_idx], arr[sig_idx]
    # Android/iOS sensor timestamps jitter (non-real-time scheduling); FFT/Welch
    # assumes a uniform grid, and jitter leaks power across bins. Resample onto
    # a strict uniform grid at the median rate before any spectral analysis.
    fs = 1.0 / np.median(np.diff(t))
    tu = np.arange(t[0], t[-1], 1.0 / fs)
    au = np.interp(tu, t, a)
    return detrend(au), fs


def accel_f1(path: str, fmin=1.0, fmax=25.0) -> tuple[float, np.ndarray, np.ndarray]:
    a, fs = load_phyphox(path)
    nper = min(len(a), int(fs * 4))
    f, p = welch(a, fs=fs, nperseg=nper, noverlap=nper // 2)
    m = (f >= fmin) & (f <= fmax)
    fb, pb = f[m], p[m]
    # acceleration PSD scales as (2*pi*f)^4 x displacement PSD, so higher modes
    # dominate raw accel spectra; convert to displacement-equivalent before
    # comparing with the camera (which measures displacement directly)
    pb = pb / (2 * np.pi * np.clip(fb, 1e-6, None)) ** 4
    i = int(np.argmax(pb))
    if 0 < i < len(pb) - 1:
        c = 0.5 * (pb[i - 1] - pb[i + 1]) / (pb[i - 1] - 2 * pb[i] + pb[i + 1])
        f1 = fb[i] + c * (fb[1] - fb[0])
    else:
        f1 = fb[i]
    return float(f1), fb, pb


def fuse(video: str, accel_csv: str, out_png: str, out_json: str | None = None) -> dict:
    cam = modal.extract(video)
    fa, fb_a, pb_a = accel_f1(accel_csv)
    agree = abs(cam.f1 - fa) / fa * 100

    m = (cam.freqs >= 1.0) & (cam.freqs <= 25.0)
    fb_c, pb_c = cam.freqs[m], cam.psd[m]

    fig, ax = plt.subplots(figsize=(7.5, 3.4), dpi=160)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    ax.plot(fb_c, pb_c / pb_c.max(), color=CAMERA, lw=2, label="Camera (TremorLens)")
    ax.plot(fb_a, pb_a / pb_a.max(), color=ACCEL, lw=2, label="Accelerometer (contact)")
    ax.annotate(f"camera f1 {cam.f1:.2f} Hz", (cam.f1, 1.0), xytext=(cam.f1 + 1.2, 0.92),
                color=INK, fontsize=8)
    ax.annotate(f"accel f1 {fa:.2f} Hz", (fa, 0.8), xytext=(fa + 1.2, 0.72),
                color=INK2, fontsize=8)
    ax.set_xlabel("Frequency (Hz)", color=INK2, fontsize=9)
    ax.set_ylabel("Normalized PSD", color=INK2, fontsize=9)
    ax.set_title(f"Camera vs contact sensor — f1 agreement {agree:.2f}%",
                 color=INK, fontsize=10, loc="left")
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(color=GRID, lw=0.5, alpha=0.6)
    ax.legend(facecolor=SURFACE, edgecolor=GRID, labelcolor=INK, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, facecolor=SURFACE)
    plt.close(fig)

    result = {"camera_f1": round(cam.f1, 3), "accel_f1": round(fa, 3),
              "agreement_pct": round(agree, 2), "plot": out_png}
    if out_json:
        with open(out_json, "w") as fh:
            json.dump(result, fh, indent=2)
    return result


if __name__ == "__main__":
    import os
    os.makedirs("out", exist_ok=True)
    r1 = fuse("data/test_healthy_motor.mp4", "data/test_accel_healthy.csv",
              "out/fusion_healthy.png", "out/fusion_healthy.json")
    r2 = fuse("data/test_damaged_motor.mp4", "data/test_accel_damaged.csv",
              "out/fusion_damaged.png", "out/fusion_damaged.json")
    print("healthy:", r1)
    print("damaged:", r2)
