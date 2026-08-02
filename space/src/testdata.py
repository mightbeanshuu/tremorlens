"""Test-data generator: renders bridge clips that mimic the hardware rig's camera output.

Faithful to what the real capture will contain:
  - static layer (background, table edge, piers with speckle reference targets)
  - moving layer (truss + deck speckle targets) vibrating with known modal physics
  - camera jitter (low-frequency tripod shake), mains-light flicker (100 Hz beat),
    slow exposure drift, sensor noise, lossy codec
  - two excitation modes: "motor" (steady sine) and "tap" (decaying impulses)

Sub-pixel motion is honest: layers are drawn at SS x resolution, shifted by
integer supersample pixels, then area-downsampled — the same anti-aliased
motion a real sensor sees.
"""
import numpy as np
import cv2
from scipy.signal import butter, sosfiltfilt

W, H = 640, 360
FPS = 60
SS = 8  # supersampling: motion quantized to 1/SS = 0.125 px


def _speckle(canvas: np.ndarray, cx: int, cy: int, s: int, seed: int, cells: int = 8, sq: int = 4) -> None:
    """Aperiodic DIC-style speckle target centered at (cx, cy). Periodic patterns
    (checkerboards) make phase correlation hop between lattice positions."""
    rng = np.random.default_rng(seed)
    pat = rng.integers(0, 2, (cells, cells))
    half = cells * sq * s // 2
    for r in range(cells):
        for c in range(cells):
            col = (0, 0, 0) if pat[r, c] else (255, 255, 255)
            cv2.rectangle(canvas,
                          (cx - half + c * sq * s, cy - half + r * sq * s),
                          (cx - half + (c + 1) * sq * s, cy - half + (r + 1) * sq * s),
                          col, -1)


def build_layers(s: int = SS):
    """Static layer (piers/background + reference targets) and moving layer (truss+deck targets) with mask."""
    h, w = H * s, W * s
    deck_y, top_y = int(h * 0.62), int(h * 0.30)
    x0, x1 = int(w * 0.12), int(w * 0.88)
    n_panels = 6
    xs = np.linspace(x0, x1, n_panels + 1).astype(int)

    static = np.full((h, w, 3), 200, np.uint8)
    cv2.rectangle(static, (0, int(h * 0.86)), (w, h), (170, 170, 170), -1)  # table edge
    cv2.rectangle(static, (x0 - 16 * s, deck_y), (x0 - 2 * s, int(h * 0.88)), (90, 90, 90), -1)  # pier L
    cv2.rectangle(static, (x1 + 2 * s, deck_y), (x1 + 16 * s, int(h * 0.88)), (90, 90, 90), -1)  # pier R
    # static speckle reference targets (taped to piers/table — cancels camera motion)
    _speckle(static, x0 - 9 * s, int(h * 0.78), s, seed=77)
    _speckle(static, x1 + 9 * s, int(h * 0.78), s, seed=78)
    _speckle(static, w // 2, int(h * 0.93), s, seed=79)

    moving = np.zeros_like(static)
    mask = np.zeros((h, w), np.uint8)

    def strokes(canvas, color, on_mask=False):
        tgt = mask if on_mask else canvas
        col = 255 if on_mask else color
        cv2.line(tgt, (x0, deck_y), (x1, deck_y), col, 6 * s)
        cv2.line(tgt, (xs[0], deck_y), (xs[1], top_y), col, 4 * s)
        for i in range(1, n_panels):
            cv2.line(tgt, (xs[i], top_y if i % 2 else deck_y),
                     (xs[i + 1], deck_y if i % 2 else top_y), col, 4 * s)
        for i in range(1, n_panels, 2):
            if i + 2 <= n_panels:
                cv2.line(tgt, (xs[i], top_y), (xs[min(i + 2, n_panels)], top_y), col, 4 * s)

    strokes(moving, (30, 30, 30))
    strokes(None, None, on_mask=True)
    for i in range(1, n_panels):
        _speckle(moving, xs[i], deck_y + 20 * s, s, seed=1000 + i)
        half = 8 * 4 * s // 2
        cv2.rectangle(mask, (xs[i] - half, deck_y + 20 * s - half),
                      (xs[i] + half, deck_y + 20 * s + half), 255, -1)
    return static, moving, mask


def _deck_motion(mode: str, f1: float, f2: float | None, amp_px: float, n: int) -> np.ndarray:
    t = np.arange(n) / FPS
    if mode == "motor":  # steady forced excitation
        d = amp_px * np.sin(2 * np.pi * f1 * t)
        if f2:
            d += 0.3 * amp_px * np.sin(2 * np.pi * f2 * t + 0.7)
    else:  # "tap": impulse every ~3 s, exponentially decaying free vibration
        d = np.zeros(n)
        for k0 in range(FPS, n, 3 * FPS):
            tt = np.arange(n - k0) / FPS
            ring = np.exp(-1.6 * tt) * np.sin(2 * np.pi * f1 * tt)
            if f2:
                ring += 0.25 * np.exp(-2.5 * tt) * np.sin(2 * np.pi * f2 * tt)
            d[k0:] += 2.2 * amp_px * ring
    return d


def render_clip(path: str, f1: float, amp_px: float = 0.45, seconds: float = 12.0,
                f2: float | None = None, mode: str = "motor", seed: int = 0,
                noise_sigma: float = 1.2, jitter_px: float = 0.15,
                flicker: float = 1.0, drift: float = 2.0) -> str:
    """Render one capture-realistic clip. Returns path."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FPS)
    static, moving, mask = build_layers()
    mask3 = cv2.merge([mask] * 3)

    disp = _deck_motion(mode, f1, f2, amp_px, n)
    # camera jitter: white noise low-passed below 3 Hz (tripod sway), both axes
    sos = butter(2, 3.0, btype="lowpass", fs=FPS, output="sos")
    jx = sosfiltfilt(sos, rng.normal(0, 1, n)) * jitter_px
    jy = sosfiltfilt(sos, rng.normal(0, 1, n)) * jitter_px
    t = np.arange(n) / FPS
    flick = flicker * np.sin(2 * np.pi * 100.0 * t)          # mains 100 Hz beat
    expo = drift * np.sin(2 * np.pi * t / t[-1] * 0.5)        # slow exposure drift

    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for k in range(n):
        dy = int(round(disp[k] * SS))
        mv = np.roll(moving, dy, axis=0)
        mk = np.roll(mask3, dy, axis=0)
        comp = np.where(mk > 0, mv, static)
        cjx, cjy = int(round(jx[k] * SS)), int(round(jy[k] * SS))
        if cjx or cjy:
            comp = np.roll(np.roll(comp, cjy, axis=0), cjx, axis=1)
        frame = cv2.resize(comp, (W, H), interpolation=cv2.INTER_AREA)
        out = frame.astype(np.float32) + flick[k] + expo[k] + rng.normal(0, noise_sigma, frame.shape)
        vw.write(np.clip(out, 0, 255).astype(np.uint8))
    vw.release()
    return path


# Canonical test set: mirrors what the hardware session will produce on D4
SET = {
    "test_healthy_motor":  dict(f1=7.30, f2=18.4, mode="motor", seed=1),
    "test_healthy_take2":  dict(f1=7.30, f2=18.4, mode="motor", seed=3),
    "test_healthy_tap":    dict(f1=7.30, f2=18.4, mode="tap",   seed=4),
    "test_damaged_motor":  dict(f1=6.20, f2=16.9, mode="motor", seed=2),
    "test_damaged_tap":    dict(f1=6.20, f2=16.9, mode="tap",   seed=5),
}
GT = {name: kw["f1"] for name, kw in SET.items()}


def render_set(outdir: str = "data") -> dict[str, str]:
    import os
    os.makedirs(outdir, exist_ok=True)
    return {name: render_clip(f"{outdir}/{name}.mp4", **kw) for name, kw in SET.items()}


if __name__ == "__main__":
    for name, p in render_set().items():
        print("wrote", p)
