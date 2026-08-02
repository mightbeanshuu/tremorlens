"""Exports per-frame pipeline streams for the live command-centre dashboard.

For each fleet bridge (blind A/B/C clips) dumps: displacement series, rolling
1-s-step f1/SHS timeline, and final spectrum — real pipeline output, replayed
live in the browser.
"""
import json

import numpy as np
from scipy.signal import welch, detrend

import modal
import shs

WINDOW_S = 6.0
STEP_S = 0.5


def stream_for(path: str, baseline_f1: float, noise_pct: float) -> dict:
    r = modal.extract(path)
    fps = r.fps
    disp = r.displacement
    base = modal.ModalResult(baseline_f1, [], None, None, fps, None, [])

    timeline = []
    n_win = int(WINDOW_S * fps)
    for k in range(n_win, len(disp), int(STEP_S * fps)):
        seg = detrend(disp[k - n_win:k])
        nper = min(len(seg), int(fps * 4))
        f, p = welch(seg, fs=fps, nperseg=nper, noverlap=nper // 2)
        m = (f >= 1.0) & (f <= 25.0)
        fb, pb = f[m], p[m]
        i = int(np.argmax(pb))
        if 0 < i < len(pb) - 1:
            c = 0.5 * (pb[i - 1] - pb[i + 1]) / (pb[i - 1] - 2 * pb[i] + pb[i + 1])
            f1 = float(fb[i] + c * (fb[1] - fb[0]))
        else:
            f1 = float(fb[i])
        v = shs.score(base, modal.ModalResult(f1, [], None, None, fps, None, []), noise_pct)
        timeline.append({"t": round(k / fps, 2), "f1": round(f1, 3),
                         "shs": v.shs, "status": v.status,
                         "spec": [round(float(x), 4) for x in (pb / pb.max())]})
    m = (r.freqs >= 1.0) & (r.freqs <= 25.0)
    return {
        "fps": fps,
        "disp": [round(float(x), 4) for x in disp],
        "timeline": timeline,
        "spec_f": [round(float(x), 3) for x in r.freqs[m]],
        "final_f1": round(r.f1, 3),
    }


def main() -> None:
    with open("out/results.json") as fh:
        R = json.load(fh)
    base_f1, noise = R["baseline_f1"], max(R["noise_pct"], 0.2)
    fleet = {}
    meta = {"A": ("Gambhira Approach Span 3", 12.4), "B": ("Old City Flyover Bay 7", 8.1),
            "C": ("Riverside Truss Km 4", 4.7)}
    for label in "ABC":
        name, km = meta[label]
        fleet[label] = {"name": name, "cam_km": km,
                        **stream_for(f"data/blind_{label}.mp4", base_f1, noise)}
    out = {"baseline_f1": base_f1, "noise_pct": noise, "fleet": fleet,
           "fusion": R["fusion"], "blind": R["blind_reveal"]}
    with open("out/stream.json", "w") as fh:
        json.dump(out, fh)
    print("wrote out/stream.json",
          {k: len(v["timeline"]) for k, v in fleet.items()}, "timeline steps")


if __name__ == "__main__":
    main()
