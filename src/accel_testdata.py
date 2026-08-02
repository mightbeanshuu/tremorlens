"""Accelerometer test data in phyphox export format, from the same test-rig physics.

The D4 hardware session tapes a phone running phyphox to the deck; its CSV export
has columns "Time (s)","Acceleration z (m/s^2)". This generates that exact format
so fusion.py exercises the real import path.
"""
import numpy as np

FS = 200.0  # phyphox typical accelerometer rate


def render_csv(path: str, f1: float, f2: float | None = None, seconds: float = 12.0,
               amp_mm: float = 0.12, noise: float = 0.03, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    # a(t) = -(2*pi*f)^2 * x(t); x in metres
    x = amp_mm * 1e-3 * np.sin(2 * np.pi * f1 * t)
    a = -(2 * np.pi * f1) ** 2 * x
    if f2:
        x2 = 0.3 * amp_mm * 1e-3 * np.sin(2 * np.pi * f2 * t + 0.7)
        a += -(2 * np.pi * f2) ** 2 * x2
    a += rng.normal(0, noise, a.shape) + 9.81  # sensor noise + gravity offset
    with open(path, "w") as fh:
        fh.write('"Time (s)","Acceleration z (m/s^2)"\n')
        for ti, ai in zip(t, a):
            fh.write(f"{ti:.5f},{ai:.6f}\n")
    return path


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    render_csv("data/test_accel_healthy.csv", f1=7.30, f2=18.4, seed=11)
    render_csv("data/test_accel_damaged.csv", f1=6.20, f2=16.9, seed=12)
    print("wrote data/test_accel_healthy.csv, data/test_accel_damaged.csv")
