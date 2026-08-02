"""Eulerian motion magnification (MIT-lineage: Wu et al. 2012) for the 'breathing bridge' B-roll.

Laplacian-pyramid temporal bandpass, amplified and recomposed. Offline tool:
  python src/magnify.py in.mp4 out.mp4 --lo 5 --hi 9 --alpha 25
"""
import argparse

import cv2
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


def _pyr_down_levels(frame: np.ndarray, levels: int) -> list[np.ndarray]:
    pyr = [frame]
    for _ in range(levels):
        pyr.append(cv2.pyrDown(pyr[-1]))
    lap = []
    for i in range(levels):
        up = cv2.pyrUp(pyr[i + 1], dstsize=(pyr[i].shape[1], pyr[i].shape[0]))
        lap.append(pyr[i] - up)
    lap.append(pyr[-1])
    return lap


def _collapse(lap: list[np.ndarray]) -> np.ndarray:
    out = lap[-1]
    for i in range(len(lap) - 2, -1, -1):
        out = cv2.pyrUp(out, dstsize=(lap[i].shape[1], lap[i].shape[0])) + lap[i]
    return out


def magnify(src: str, dst: str, lo: float, hi: float, alpha: float, levels: int = 4) -> None:
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    sos = butter(2, [lo, hi], btype="bandpass", fs=fps, output="sos")
    states: list[np.ndarray] | None = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        lap = _pyr_down_levels(frame.astype(np.float32) / 255.0, levels)
        if states is None:
            zi = sosfilt_zi(sos)
            states = [np.einsum("ij,...->ij...", zi, band) for band in lap]
        out_lap = []
        for i, band in enumerate(lap):
            filtered, states[i] = sosfilt(sos, band[None, ...], axis=0, zi=states[i])
            gain = alpha if i < levels else 0.0  # don't amplify the DC residual
            out_lap.append(band + gain * filtered[0])
        out = np.clip(_collapse(out_lap) * 255.0, 0, 255).astype(np.uint8)
        vw.write(out)
    cap.release()
    vw.release()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--lo", type=float, default=5.0)
    ap.add_argument("--hi", type=float, default=9.0)
    ap.add_argument("--alpha", type=float, default=25.0)
    args = ap.parse_args()
    magnify(args.src, args.dst, args.lo, args.hi, args.alpha)
    print(f"magnified -> {args.dst}")
