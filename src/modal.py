"""Modal extraction: video -> sub-pixel displacement time series -> frequency fingerprint.

Auto-ROI: picks the most textured grid cells (the high-contrast targets a user
tapes onto the structure), tracks each against the first frame with windowed
phase correlation (sub-pixel), then reads modal peaks from a Welch PSD.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.signal import welch, find_peaks, detrend
from skimage.registration import phase_cross_correlation


@dataclass
class ModalResult:
    f1: float                 # fundamental frequency, Hz
    peaks: list[tuple[float, float]]  # (freq, normalized power) strongest-first
    freqs: np.ndarray
    psd: np.ndarray
    fps: float
    displacement: np.ndarray  # mean deck displacement per frame, px
    rois: list[tuple[int, int, int, int]]
    zeta: float | None = None  # damping ratio via half-power bandwidth (secondary indicator)
    roi_disp: list[np.ndarray] | None = None  # per-ROI camera-cancelled displacement (for mode shapes)


def _auto_rois(gray: np.ndarray, cell: int = 48, top_k: int = 6,
               band: tuple[float, float] = (0.45, 0.85)) -> list[tuple[int, int, int, int]]:
    """Pick top-K high-texture cells inside the vertical band where the structure sits."""
    h, w = gray.shape
    y0, y1 = int(h * band[0]), int(h * band[1])
    scored = []
    for y in range(y0, y1 - cell, cell // 2):
        for x in range(0, w - cell, cell // 2):
            patch = gray[y:y + cell, x:x + cell]
            scored.append((cv2.Laplacian(patch, cv2.CV_64F).var(), (x, y, cell, cell)))
    scored.sort(key=lambda s: -s[0])
    picked: list[tuple[int, int, int, int]] = []
    for _, roi in scored:
        if all(abs(roi[0] - p[0]) > cell or abs(roi[1] - p[1]) > cell for p in picked):
            picked.append(roi)
        if len(picked) == top_k:
            break
    return picked


def _half_power_zeta(fb: np.ndarray, pb: np.ndarray, f1: float) -> float | None:
    """Damping ratio from the -3 dB bandwidth around the f1 peak: zeta = (fb-fa)/(2*f1).

    Damping scatters +/-20-30% run to run, so it is reported as a secondary
    indicator only, never a standalone alarm.
    """
    i = int(np.argmin(np.abs(fb - f1)))
    half = pb[i] / 2.0
    lo = hi = None
    for j in range(i, 0, -1):
        if pb[j - 1] < half:
            lo = fb[j - 1] + (fb[j] - fb[j - 1]) * (half - pb[j - 1]) / (pb[j] - pb[j - 1])
            break
    for j in range(i, len(pb) - 1):
        if pb[j + 1] < half:
            hi = fb[j] + (fb[j + 1] - fb[j]) * (pb[j] - half) / (pb[j] - pb[j + 1])
            break
    if lo is None or hi is None or hi <= lo:
        return None
    return float((hi - lo) / (2.0 * f1))


def extract(path: str, fmin: float = 1.0, fmax: float = 25.0) -> ModalResult:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0

    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"empty video: {path}")
    g0 = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    rois = _auto_rois(g0, band=(0.45, 0.74))          # structure (deck targets)
    ref_rois = _auto_rois(g0, band=(0.74, 0.97), top_k=3)  # static references (piers/table)
    all_rois = rois + ref_rois
    refs = [np.float32(g0[y:y + ch, x:x + cw]) for x, y, cw, ch in all_rois]

    series = [[0.0] for _ in all_rois]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for i, (x, y, cw, ch) in enumerate(all_rois):
            cur = np.float32(g[y:y + ch, x:x + cw])
            shift, _, _ = phase_cross_correlation(refs[i], cur, upsample_factor=50)
            dy = float(-shift[0])  # skimage returns shift to register cur onto ref
            if abs(dy) > 8.0:  # ambiguous lock — hold previous sample, don't spike
                dy = series[i][-1]
            series[i].append(dy)  # vertical motion (structure or camera, per ROI)
    cap.release()

    arr = np.array(series)
    deck = np.median(arr[:len(rois)], axis=0)      # median across ROIs resists one bad tracker
    cam = np.median(arr[len(rois):], axis=0) if ref_rois else 0.0
    disp = detrend(deck - cam)                     # static-reference cancellation of camera motion
    roi_disp = [detrend(arr[i] - cam) for i in range(len(rois))]  # per-ROI, for mode shapes
    nper = min(len(disp), int(fps * 4))
    freqs, psd = welch(disp, fs=fps, nperseg=nper, noverlap=nper // 2)

    mask = (freqs >= fmin) & (freqs <= fmax)
    fb, pb = freqs[mask], psd[mask]
    idx, _ = find_peaks(pb, height=pb.max() * 0.05)
    order = idx[np.argsort(pb[idx])[::-1]][:5]
    # parabolic interpolation for sub-bin frequency accuracy
    peaks = []
    for i in order:
        if 0 < i < len(pb) - 1:
            d = 0.5 * (pb[i - 1] - pb[i + 1]) / (pb[i - 1] - 2 * pb[i] + pb[i + 1])
            f = fb[i] + d * (fb[1] - fb[0])
        else:
            f = fb[i]
        peaks.append((float(f), float(pb[i] / pb.max())))
    if not peaks:
        raise RuntimeError("no modal peak found — check excitation/lighting")
    zeta = _half_power_zeta(fb, pb, peaks[0][0])
    return ModalResult(peaks[0][0], peaks, freqs, psd, fps, disp, rois, zeta, roi_disp)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("usage: python src/modal.py <video>")
    r = extract(sys.argv[1])
    print(f"f1 = {r.f1:.3f} Hz | peaks: " + ", ".join(f"{f:.2f}Hz({p:.2f})" for f, p in r.peaks))
