"""TremorLens live mode — hands-free: auto-ROI -> auto-baseline -> auto-monitor -> auto-alert.

Zero interaction after launch. Source is a webcam index or a video file
(the file path streams like a live camera, so test data exercises the exact
code path the hardware demo will use).

  python src/live.py --source data/test_healthy_motor.mp4 --headless out/live_healthy.mp4
  python src/live.py --source 0            # webcam, on-screen HUD

If out/baseline.json exists it monitors against it; otherwise it records the
baseline first — so run #1 (healthy) then run #2 (suspect) is the whole demo.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
from scipy.signal import welch, find_peaks, detrend
from skimage.registration import phase_cross_correlation

from modal import _auto_rois
from shs import score as shs_score
from modal import ModalResult

BASELINE_SECS = 8.0
WINDOW_SECS = 6.0
FMIN, FMAX = 1.0, 25.0

INK = (235, 235, 235)
DIM = (150, 150, 150)
GOOD = (120, 220, 120)
WARN = (60, 190, 255)
BAD = (80, 80, 255)


def _peak(disp: np.ndarray, fps: float) -> tuple[float, np.ndarray, np.ndarray]:
    d = detrend(disp)
    nper = min(len(d), int(fps * 4))
    freqs, psd = welch(d, fs=fps, nperseg=nper, noverlap=nper // 2)
    m = (freqs >= FMIN) & (freqs <= FMAX)
    fb, pb = freqs[m], psd[m]
    i = int(np.argmax(pb))
    if 0 < i < len(pb) - 1:
        c = 0.5 * (pb[i - 1] - pb[i + 1]) / (pb[i - 1] - 2 * pb[i] + pb[i + 1])
        f1 = fb[i] + c * (fb[1] - fb[0])
    else:
        f1 = fb[i]
    return float(f1), fb, pb


def _spectrum_strip(frame, fb, pb, x, y, w, h, f_mark=None, color=INK):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (45, 45, 45), -1)
    if pb is not None and pb.max() > 0:
        pts = np.column_stack([
            x + (fb - fb[0]) / (fb[-1] - fb[0]) * w,
            y + h - 4 - (pb / pb.max()) * (h - 10),
        ]).astype(np.int32)
        cv2.polylines(frame, [pts], False, color, 1, cv2.LINE_AA)
        if f_mark:
            mx = int(x + (f_mark - fb[0]) / (fb[-1] - fb[0]) * w)
            cv2.line(frame, (mx, y + 2), (mx, y + h - 2), WARN, 1)
    cv2.putText(frame, f"{FMIN:.0f}-{FMAX:.0f} Hz", (x + 4, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, DIM, 1, cv2.LINE_AA)


def run(source, headless_out: str | None, baseline_path: str = "out/baseline.json") -> None:
    cam = str(source).isdigit()
    cap = cv2.VideoCapture(int(source) if cam else source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    os.makedirs("out", exist_ok=True)
    baseline = None
    if os.path.exists(baseline_path):
        with open(baseline_path) as fh:
            baseline = json.load(fh)

    vw = None
    rois = ref_rois = refs = None
    deck: list[list[float]] = []
    refm: list[list[float]] = []
    t_series: list[float] = []
    phase = "CALIBRATING"
    verdict = None
    noise_pct = max(baseline.get("noise_pct", 0.5), 0.2) if baseline else 0.5
    f1_now, fb, pb = None, None, None

    k = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if rois is None:  # first frame: auto-ROI, no user input
            rois = _auto_rois(g, band=(0.45, 0.74))
            ref_rois = _auto_rois(g, band=(0.74, 0.97), top_k=3)
            refs = [np.float32(g[y:y + ch, x:x + cw]) for x, y, cw, ch in rois + ref_rois]
            deck = [[0.0] for _ in rois]
            refm = [[0.0] for _ in ref_rois]
            phase = "BASELINE" if baseline is None else "MONITOR"
        else:
            for i, (x, y, cw, ch) in enumerate(rois + ref_rois):
                cur = np.float32(g[y:y + ch, x:x + cw])
                shift, _, _ = phase_cross_correlation(refs[i], cur, upsample_factor=20)
                dy = float(-shift[0])
                tgt = deck[i] if i < len(rois) else refm[i - len(rois)]
                tgt.append(tgt[-1] if abs(dy) > 8.0 else dy)

        k += 1
        t_series.append(k / fps)

        n_need = int((BASELINE_SECS if phase == "BASELINE" else WINDOW_SECS) * fps)
        n_have = len(deck[0])
        if n_have > int(fps):  # ~1 s minimum before first spectrum
            d = np.median(np.array([s[-min(n_have, n_need):] for s in deck]), axis=0)
            c = np.median(np.array([s[-min(n_have, n_need):] for s in refm]), axis=0) if refm else 0.0
            f1_now, fb, pb = _peak(np.asarray(d) - np.asarray(c), fps)

        if phase == "BASELINE" and n_have >= n_need:
            baseline = {"f1": f1_now, "noise_pct": 0.5, "captured_s": n_have / fps}
            with open(baseline_path, "w") as fh:
                json.dump(baseline, fh, indent=2)
            phase = "MONITOR"
        if phase == "MONITOR" and baseline and f1_now and n_have >= int(WINDOW_SECS * fps):
            b = ModalResult(baseline["f1"], [], None, None, fps, None, [])
            cur = ModalResult(f1_now, [], None, None, fps, None, [])
            verdict = shs_score(b, cur, noise_pct=noise_pct)

        # ---- HUD (drawn every frame; zero user interaction) ----
        hud = frame.copy()
        for x, y, cw, ch in rois or []:
            cv2.rectangle(hud, (x, y), (x + cw, y + ch), GOOD, 1)
        for x, y, cw, ch in ref_rois or []:
            cv2.rectangle(hud, (x, y), (x + cw, y + ch), DIM, 1)
        bar_color = {"CALIBRATING": DIM, "BASELINE": WARN, "MONITOR": GOOD}[phase]
        status_txt = phase
        if verdict:
            bar_color = {"HEALTHY": GOOD, "WATCH": WARN, "ALERT": BAD}[verdict.status]
            status_txt = f"{verdict.status}  SHS {verdict.shs:.0f}"
        cv2.rectangle(hud, (0, 0), (hud.shape[1], 34), (25, 25, 25), -1)
        cv2.putText(hud, "TremorLens", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, INK, 1, cv2.LINE_AA)
        cv2.putText(hud, status_txt, (150, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 2, cv2.LINE_AA)
        if f1_now:
            txt = f"f1 {f1_now:5.2f} Hz"
            if baseline:
                txt += f"   baseline {baseline['f1']:5.2f} Hz"
            cv2.putText(hud, txt, (hud.shape[1] - 300, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK, 1, cv2.LINE_AA)
        if fb is not None:
            _spectrum_strip(hud, fb, pb, 10, hud.shape[0] - 70, 220, 60,
                            f_mark=baseline["f1"] if baseline else None,
                            color=bar_color if verdict else INK)
        if verdict and verdict.status == "ALERT":
            cv2.rectangle(hud, (0, 34), (hud.shape[1], 60), BAD, -1)
            cv2.putText(hud, f"STRUCTURAL DRIFT {verdict.drift_pct:.1f}% vs baseline - INSPECT",
                        (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if headless_out:
            if vw is None:
                vw = cv2.VideoWriter(headless_out, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (hud.shape[1], hud.shape[0]))
            vw.write(hud)
        else:
            cv2.imshow("TremorLens", hud)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    if vw:
        vw.release()
    if not headless_out:
        cv2.destroyAllWindows()
    if verdict:
        print(f"final: {verdict.status} SHS {verdict.shs} | {verdict.detail}")
    elif phase == "MONITOR" and baseline:
        print(f"baseline recorded: f1={baseline['f1']:.3f} Hz -> {baseline_path}")
    else:
        print(f"phase={phase} (source ended)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="webcam index or video file")
    ap.add_argument("--headless", default=None, metavar="OUT_MP4",
                    help="write annotated video instead of showing a window")
    ap.add_argument("--baseline", default="out/baseline.json")
    args = ap.parse_args()
    run(args.source, args.headless, args.baseline)
