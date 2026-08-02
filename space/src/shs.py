"""Structural Heartbeat Score: baseline fingerprint vs current reading -> severity + verdict."""
from dataclasses import dataclass

from modal import ModalResult


@dataclass
class Verdict:
    shs: float          # 0-100, 100 = matches healthy baseline
    drift_pct: float    # |Δf1| / f1_baseline * 100
    status: str         # HEALTHY / WATCH / ALERT
    detail: str


def score(baseline: ModalResult, current: ModalResult, noise_pct: float = 1.0) -> Verdict:
    """noise_pct: run-to-run scatter measured from repeated healthy takes (calibrated, not assumed)."""
    drift = abs(current.f1 - baseline.f1) / baseline.f1 * 100.0
    # drift inside measured noise costs nothing; beyond it, SHS falls fast (5 pts per % of true drift)
    excess = max(0.0, drift - noise_pct)
    shs = max(0.0, 100.0 - 5.0 * excess)
    if excess <= 0.5:
        status = "HEALTHY"
    elif excess <= 3.0:
        status = "WATCH"
    else:
        status = "ALERT"
    detail = (f"baseline f1={baseline.f1:.3f} Hz, current f1={current.f1:.3f} Hz, "
              f"drift={drift:.2f}% (noise floor {noise_pct:.2f}%)")
    return Verdict(round(shs, 1), round(drift, 2), status, detail)
