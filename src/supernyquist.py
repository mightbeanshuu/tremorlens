"""Super-Nyquist rolling-shutter vibrometry — kHz-band sensing from a 30/60 fps phone.

Every CMOS sensor row is exposed at a slightly different microsecond; a frame of
a vibrating edge is therefore ~1,000 time samples, not one. Per-row sub-pixel
edge tracking + the non-uniform timestamp model t = n/F + m*r (US11776238B2)
+ Lomb-Scargle recovers frequencies far above the frame-rate Nyquist —
55/59 Hz from 30 fps (Andre et al., MSSP 2021, SURVISHNO); 1 kHz from 60 fps
(Zhao et al., Opt. Eng. 2018). Frequency-only claim (the SHM-diagnostic
quantity); amplitude needs row-gap + inverse-sinc corrections (roadmap).

Capture rules (non-negotiable): stabilization OFF, shutter locked <= 1/1000 s,
AE/AF locked, high-contrast edge CROSSING the sensor rows, flood with light.

Self-calibration: sweep the row time r to maximize the Lomb-Scargle peak at a
speaker tone of known frequency (use 173 or 197 Hz — never a multiple of the
frame rate). The calibration IS part of the demo: the instrument proves its
own clock live.
"""
import numpy as np
from scipy.signal import lombscargle


def track_edge_rows(frames: np.ndarray) -> np.ndarray:
    """Per-row sub-pixel edge position for a stack of grayscale ROI frames.

    frames: (n_frames, n_rows, n_cols) float/uint8. Returns x[(row, frame)]:
    horizontal position of the strongest vertical edge in each row, refined by
    parabolic interpolation of the |gradient| peak (~0.02-0.05 px noise).
    """
    f = frames.astype(np.float64)
    g = np.abs(np.diff(f, axis=2))                 # |horizontal gradient|
    n, r, c = g.shape
    k = np.argmax(g, axis=2)                       # (n, r) integer peak
    k = np.clip(k, 1, c - 2)
    ii, jj = np.meshgrid(np.arange(n), np.arange(r), indexing="ij")
    gm1, g0, gp1 = g[ii, jj, k - 1], g[ii, jj, k], g[ii, jj, k + 1]
    denom = gm1 - 2 * g0 + gp1
    delta = np.where(np.abs(denom) > 1e-12, 0.5 * (gm1 - gp1) / denom, 0.0)
    x = (k + np.clip(delta, -1, 1)).T              # (rows, frames)
    return x - x.mean(axis=1, keepdims=True)       # remove static shape per row


def rs_spectrum(x: np.ndarray, fps: float, row_time: float,
                fmin: float = 5.0, fmax: float = 800.0,
                nfreq: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Lomb-Scargle over the full row-timestamped sample cloud.

    x: (n_rows, n_frames) detrended per-row positions.
    t[m, n] = n/fps + m*row_time  (blanking lives in row_time being < frame/rows).
    """
    n_rows, n_frames = x.shape
    m = np.arange(n_rows)[:, None]
    n = np.arange(n_frames)[None, :]
    t = (n / fps + m * row_time).ravel()
    d = x.ravel()
    d = d - d.mean()
    freqs = np.linspace(fmin, fmax, nfreq)
    p = lombscargle(t, d, 2 * np.pi * freqs, normalize=True)
    return freqs, p


def peak_freq(freqs: np.ndarray, p: np.ndarray) -> float:
    i = int(np.argmax(p))
    if 0 < i < len(p) - 1:
        d = 0.5 * (p[i - 1] - p[i + 1]) / (p[i - 1] - 2 * p[i] + p[i + 1] or 1e-12)
        return float(freqs[i] + d * (freqs[1] - freqs[0]))
    return float(freqs[i])


def self_calibrate(x: np.ndarray, fps: float, known_freq: float,
                   r_lo: float = 1e-6, r_hi: float = 35e-6,
                   n_r: int = 120) -> tuple[float, float]:
    """Sweep row time r to maximize LS power at a known speaker tone.
    Returns (best_row_time, peak_power). The published self-cal ritual."""
    best_r, best_p = r_lo, -1.0
    for r in np.linspace(r_lo, r_hi, n_r):
        freqs, p = rs_spectrum(x, fps, r, fmin=known_freq - 15,
                               fmax=known_freq + 15, nfreq=300)
        pk = float(p.max())
        if pk > best_p:
            best_p, best_r = pk, float(r)
    return best_r, best_p


def analyze_video(path: str, roi: tuple[int, int, int, int] | None = None,
                  row_time: float | None = None,
                  fmax: float = 800.0) -> dict:
    """Full pipeline on a real capture: video -> per-row edge -> RS spectrum.
    roi = (x, y, w, h) containing the vibrating edge; rows of the ROI keep
    their ABSOLUTE sensor-row spacing (crop does not change row timing)."""
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if roi:
            xr, yr, wr, hr = roi
            g = g[yr:yr + hr, xr:xr + wr]
        frames.append(g)
    cap.release()
    stack = np.stack(frames)
    n_sensor_rows = stack.shape[1] if roi is None else None
    if row_time is None:
        # conservative default: full-duty scan (calibrate for real numbers!)
        row_time = 1.0 / (fps * stack.shape[1])
    x = track_edge_rows(stack)
    freqs, p = rs_spectrum(x, fps, row_time, fmax=fmax)
    f_pk = peak_freq(freqs, p)
    return {"fps": fps, "row_time_us": round(row_time * 1e6, 3),
            "n_frames": stack.shape[0], "n_rows": stack.shape[1],
            "frame_nyquist_hz": fps / 2,
            "peak_hz": round(f_pk, 2),
            "super_nyquist_factor": round(f_pk / (fps / 2), 1),
            "note": ("frequency-only claim; amplitude needs row-gap + "
                     "inverse-sinc corrections (roadmap). Calibrate row_time "
                     "with a known speaker tone before quoting numbers.")}
