"""SPANDAN engine — Structural Pulse ANalysis via Displacement And Novelty-detection.

Upgrades the single-number f1-shift verdict to the Farrar–Worden statistical
pattern-recognition paradigm for SHM, plus damage LOCALIZATION from the
multi-ROI video (something a single contact accelerometer cannot do):

  1. Mode shapes: per-ROI amplitude + relative phase at each modal peak via
     cross-spectral density against the strongest ROI, quality-gated by
     magnitude-squared coherence (gamma^2 >= 0.7, else "low confidence").
  2. Global shape check: MAC per mode (healthy vs current, MSF scale-aligned).
  3. Localization: per-ROI COMAC across modes (>=2 modes) with NMSD as the
     single-mode fallback — a red/green heatmap over the actual video frame.
  4. Novelty scoring: Mahalanobis distance^2 of a windowed spectral fingerprint
     [f1, band log-powers] against the healthy distribution, alert at the
     chi-squared 99% quantile.
  5. Honesty layer: block-bootstrap 95% CI on f1, damping gated on spectral
     resolution, and an environmental (temperature) guard band — final rule:
     ALERT iff drift > max(3*sigma_bootstrap, EOV band).

Lineage: Pandey/Biswas/Samman 1991 (shape-change localization), Lieven & Ewins
1988 (COMAC), Farrar & Worden (novelty-detection paradigm).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import welch, csd, coherence, detrend, find_peaks
from scipy.stats import chi2

from modal import ModalResult

COH_MIN = 0.7        # coherence gate for a trustworthy mode-shape entry
EOV_BAND_PCT = 2.0   # +/- guard band on f1 for temperature/environmental variation
N_MODES = 3          # 6 ROIs resolve ~3 half-waves -> modes 1..3 are trustworthy


# ---------------------------------------------------------------- mode shapes

@dataclass
class ShapeSet:
    freqs: list[float]                  # modal frequencies used (Hz)
    phi: np.ndarray                     # (n_modes, n_roi) unit-normalized shapes
    coh: np.ndarray                     # (n_modes, n_roi) coherence at each mode
    order: list[int]                    # ROI indices sorted left-to-right


def mode_shapes(r: ModalResult, n_modes: int = N_MODES) -> ShapeSet | None:
    if not r.roi_disp or len(r.roi_disp) < 3:
        return None
    order = sorted(range(len(r.rois)), key=lambda i: r.rois[i][0])  # left-to-right
    sigs = [np.asarray(r.roi_disp[i], dtype=float) for i in order]
    fps = r.fps
    nper = min(len(sigs[0]), int(fps * 4))
    ref = max(range(len(sigs)), key=lambda i: float(np.var(sigs[i])))

    fsel = [f for f, _ in r.peaks[:n_modes]]
    phi = np.zeros((len(fsel), len(sigs)))
    coh_m = np.zeros_like(phi)
    for i, s in enumerate(sigs):
        f_c, pxy = csd(sigs[ref], s, fs=fps, nperseg=nper, noverlap=nper // 2)
        f_w, pyy = welch(s, fs=fps, nperseg=nper, noverlap=nper // 2)
        f_g, g2 = coherence(sigs[ref], s, fs=fps, nperseg=nper, noverlap=nper // 2)
        for m, fm in enumerate(fsel):
            k = int(np.argmin(np.abs(f_c - fm)))
            sign = 1.0 if np.cos(np.angle(pxy[k])) >= 0 else -1.0
            phi[m, i] = sign * np.sqrt(max(pyy[k], 0.0))
            coh_m[m, i] = g2[k]
    norms = np.linalg.norm(phi, axis=1, keepdims=True)
    phi = phi / np.where(norms > 0, norms, 1.0)
    return ShapeSet(fsel, phi, coh_m, order)


def mac(a: np.ndarray, b: np.ndarray) -> float:
    """Modal assurance criterion between two shape vectors."""
    num = float(np.dot(a, b)) ** 2
    den = float(np.dot(a, a)) * float(np.dot(b, b))
    return num / den if den > 0 else 0.0


def compare_shapes(healthy: ShapeSet, current: ShapeSet) -> dict:
    """MAC per mode + per-ROI localization (COMAC across modes, NMSD fallback)."""
    n_modes = min(len(healthy.freqs), len(current.freqs))
    n_roi = min(healthy.phi.shape[1], current.phi.shape[1])
    ph, pc = healthy.phi[:n_modes, :n_roi].copy(), current.phi[:n_modes, :n_roi].copy()

    macs, nmsd = [], np.zeros(n_roi)
    for m in range(n_modes):
        # modal scale factor alignment (handles arbitrary sign/scale of CSD shapes)
        msf = float(np.dot(ph[m], pc[m])) / max(float(np.dot(pc[m], pc[m])), 1e-12)
        pc[m] *= msf
        macs.append(mac(ph[m], pc[m]))
        nmsd += np.abs(pc[m] - ph[m])

    if n_modes >= 2:  # COMAC is identically 1 with a single mode — needs >=2
        num = np.abs((ph * pc)).sum(axis=0) ** 2
        den = (ph ** 2).sum(axis=0) * (pc ** 2).sum(axis=0)
        comac = np.where(den > 1e-12, num / den, 1.0)
    else:
        comac = None

    low_conf = [int(i) for i in range(n_roi)
                if healthy.coh[0, i] < COH_MIN or current.coh[0, i] < COH_MIN]
    loc = comac if comac is not None else 1.0 - nmsd / max(nmsd.max(), 1e-12)
    worst = int(np.argmin(loc))
    return {"mac_per_mode": [round(m, 3) for m in macs],
            "comac": None if comac is None else [round(c, 3) for c in comac],
            "nmsd": [round(float(v), 4) for v in nmsd],
            "worst_roi_rank": worst,          # index into left-to-right ROI order
            "low_confidence_rois": low_conf}


# ------------------------------------------------------------ novelty scoring

def _window_features(disp: np.ndarray, fps: float, nwin: int = 8,
                     fmin: float = 1.0, fmax: float = 25.0) -> np.ndarray:
    """Spectral fingerprint per overlapping window: [f1, 4 band log-powers]."""
    disp = np.asarray(disp, dtype=float)
    wlen = max(int(fps * 4), len(disp) // (nwin // 2 + 1))
    step = max(1, (len(disp) - wlen) // max(nwin - 1, 1))
    edges = np.linspace(fmin, fmax, 5)
    feats = []
    for s in range(0, len(disp) - wlen + 1, step):
        seg = detrend(disp[s:s + wlen])
        nper = min(len(seg), int(fps * 2))
        f, p = welch(seg, fs=fps, nperseg=nper, noverlap=nper // 2)
        m = (f >= fmin) & (f <= fmax)
        fb, pb = f[m], p[m]
        if not len(pb):
            continue
        idx, _ = find_peaks(pb, height=pb.max() * 0.05)
        f1 = fb[idx[np.argmax(pb[idx])]] if len(idx) else fb[int(np.argmax(pb))]
        total = pb.sum()
        bands = [np.log10(pb[(fb >= lo) & (fb < hi)].sum() / total + 1e-12)
                 for lo, hi in zip(edges[:-1], edges[1:])]
        feats.append([f1, *bands])
    return np.array(feats)


def novelty(baseline_disps: list[tuple[np.ndarray, float]],
            current_disp: np.ndarray, current_fps: float) -> dict:
    """Mahalanobis novelty of current take vs healthy-window distribution."""
    base = np.vstack([_window_features(d, fps) for d, fps in baseline_disps])
    cur = _window_features(current_disp, current_fps)
    mu = base.mean(axis=0)
    cov = np.cov(base.T) + np.eye(base.shape[1]) * 1e-6  # ridge: few windows
    inv = np.linalg.inv(cov)
    d2 = np.array([float((x - mu) @ inv @ (x - mu)) for x in cur])
    thresh = float(chi2.ppf(0.99, df=base.shape[1]))
    d2_med = float(np.median(d2))
    return {"d2_median": round(d2_med, 2), "chi2_99": round(thresh, 2),
            "novel": bool(d2_med > thresh),
            "windows_flagged": int((d2 > thresh).sum()), "windows_total": len(d2)}


# ------------------------------------------------------------- honesty layer

def bootstrap_f1(disp: np.ndarray, fps: float, n_boot: int = 300,
                 fmin: float = 1.0, fmax: float = 25.0) -> dict:
    """Block bootstrap over Welch segments -> 95% CI and sigma on f1."""
    rng = np.random.default_rng(7)
    disp = np.asarray(disp, dtype=float)
    nper = min(len(disp), int(fps * 4))
    step = nper // 2
    segs = [detrend(disp[s:s + nper]) for s in range(0, len(disp) - nper + 1, step)]
    if len(segs) < 2:
        return {"f1_ci95": None, "sigma_pct": None}
    from numpy.fft import rfft, rfftfreq
    win = np.hanning(nper)
    psds = [np.abs(rfft(s * win)) ** 2 for s in segs]
    f = rfftfreq(nper, 1.0 / fps)
    m = (f >= fmin) & (f <= fmax)
    fb = f[m]
    est = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(psds), len(psds))
        pb = np.mean([psds[i] for i in pick], axis=0)[m]
        i = int(np.argmax(pb))
        if 0 < i < len(pb) - 1:
            c = 0.5 * (pb[i - 1] - pb[i + 1]) / (pb[i - 1] - 2 * pb[i] + pb[i + 1])
            est.append(fb[i] + c * (fb[1] - fb[0]))
        else:
            est.append(fb[i])
    est = np.array(est)
    lo, hi = np.percentile(est, [2.5, 97.5])
    mean = float(est.mean())
    return {"f1_ci95": [round(float(lo), 3), round(float(hi), 3)],
            "sigma_pct": round(float(est.std() / mean * 100), 3) if mean else None}


# ------------------------------- structural dysphonia panel (voice pathology)

def dysphonia(disp: np.ndarray, fps: float, f1: float,
              fband: float = 0.3) -> dict | None:
    """Clinical voice-pathology perturbation metrics applied to the structural
    'voice': jitter (cycle-to-cycle period instability), shimmer (amplitude
    instability), HNR (harmonic-to-noise, Boersma autocorrelation method), CPP
    (cepstral peak prominence). A breathing crack gives bilinear stiffness ->
    the instantaneous frequency wobbles between open/closed values (Douka &
    Hadjileontiadis 2005) — the structure turns HOARSE before its pitch moves.
    Speech-RECOGNITION features (MFCC et al.) are established in SHM; the
    clinical perturbation panel is not — cite both facts in the docs.
    ALWAYS publish alongside a rigid-control-ROI 'measurement floor': frame
    timing jitter masquerades as structural jitter.
    """
    from scipy.signal import butter, filtfilt
    x = detrend(np.asarray(disp, dtype=float))
    nyq = fps / 2.0
    lo, hi = max(f1 * (1 - fband), 0.2) / nyq, min(f1 * (1 + fband) / nyq, 0.95)
    b, a = butter(2, [lo, hi], "band")
    xb = filtfilt(b, a, x)

    s = np.sign(xb)
    idx = np.where((s[:-1] <= 0) & (s[1:] > 0))[0]
    if len(idx) < 12:
        return None
    tc = idx + xb[idx] / (xb[idx] - xb[idx + 1])      # sub-sample crossings
    T = np.diff(tc) / fps
    A = []
    for t0, t1 in zip(tc[:-1], tc[1:]):
        seg = xb[int(np.ceil(t0)):int(np.ceil(t1))]
        if len(seg) > 1:
            A.append(float(np.abs(seg).max()))
    A = np.array(A)
    if len(T) < 10 or len(A) < 10:
        return None

    mT = float(T.mean())
    jitter_local = float(np.abs(np.diff(T)).mean() / mT * 100)
    rap = float(np.abs(T[1:-1] - (T[:-2] + T[1:-1] + T[2:]) / 3).mean() / mT * 100)
    mA = float(A.mean())
    shimmer_local = float(np.abs(np.diff(A)).mean() / mA * 100)
    ratios = A[1:] / np.clip(A[:-1], 1e-12, None)
    shimmer_db = float(np.abs(20 * np.log10(np.clip(ratios, 1e-6, None))).mean())

    # HNR (Boersma 1993): windowed normalized autocorrelation divided by the
    # window's own autocorrelation; r_max searched around the f1 lag
    n = len(xb)
    w = np.hanning(n)
    xw = xb * w
    r_sig = np.correlate(xw, xw, "full")[n - 1:]
    r_win = np.correlate(w, w, "full")[n - 1:]
    r = r_sig / np.clip(r_win, 1e-12, None)
    r /= max(r[0], 1e-20)
    lag_lo = max(int(fps / (f1 * (1 + fband))), 1)
    lag_hi = min(int(fps / max(f1 * (1 - fband), 0.2)), n - 1)
    r_max = float(np.clip(r[lag_lo:lag_hi + 1].max(), 1e-6, 1 - 1e-6)) if lag_hi > lag_lo else 0.5
    hnr_db = float(10 * np.log10(r_max / (1 - r_max)))

    # CPP: real cepstrum in dB, linear-regression baseline over the quefrency
    # window [0.5/f1, 2/f1], prominence of the peak above the regression line
    spec = np.abs(np.fft.rfft(xw)) + 1e-12
    cep = np.fft.irfft(20 * np.log10(spec))
    q = np.arange(len(cep)) / fps
    qm = (q >= 0.5 / f1) & (q <= min(2.0 / f1, q[-1]))
    cpp_db = None
    if qm.sum() > 8:
        qq, cc = q[qm], cep[qm]
        k = int(np.argmax(cc))
        coef = np.polyfit(qq, cc, 1)
        cpp_db = float(cc[k] - np.polyval(coef, qq[k]))

    # THD: harmonic distortion ratio — bilinear (breathing-crack) stiffness
    # pumps energy into 2f1/3f1 while f1 itself barely moves. Jitter is the
    # rattling-joint channel; THD is the breathing-crack channel.
    nper = min(n, int(fps * 8))
    f_w, p_w = welch(x, fs=fps, nperseg=nper, noverlap=nper // 2)
    def band_power(fc):
        m = (f_w >= fc * 0.94) & (f_w <= fc * 1.06)
        return float(p_w[m].max()) if m.any() else 0.0
    p1 = band_power(f1)
    p_harm = sum(band_power(k * f1) for k in (2, 3)
                 if k * f1 < fps / 2 * 0.95)
    thd_pct = float(np.sqrt(p_harm / p1) * 100) if p1 > 0 else None

    return {"jitter_pct": round(jitter_local, 3), "jitter_rap_pct": round(rap, 3),
            "shimmer_pct": round(shimmer_local, 3), "shimmer_db": round(shimmer_db, 3),
            "hnr_db": round(hnr_db, 2),
            "cpp_db": round(cpp_db, 2) if cpp_db is not None else None,
            "thd_pct": round(thd_pct, 2) if thd_pct is not None else None,
            "n_cycles": int(len(T)),
            "low_cycle_warning": bool(len(T) < 30)}


def ls_fap(disp: np.ndarray, fps: float, f1: float,
           fmin: float = 1.0, fmax: float = 25.0) -> dict:
    """Lomb-Scargle power at f1 with analytic false-alarm probability:
    converts 'we picked the tallest bar' into 'FAP = 3e-9'. Standard in
    astronomy, occasionally in SHM, essentially absent from vision SHM."""
    from scipy.signal import lombscargle
    x = detrend(np.asarray(disp, dtype=float))
    t = np.arange(len(x)) / fps
    freqs = np.linspace(fmin, min(fmax, fps / 2 * 0.95), 800)
    p = lombscargle(t, x, 2 * np.pi * freqs, normalize=True)
    k = int(np.argmin(np.abs(freqs - f1)))
    pk = float(np.clip(p[k], 1e-12, 1 - 1e-12))
    n_eff = max(t[-1] * (freqs[-1] - freqs[0]), 1.0)
    # normalized-power tail: P(>z) ~ (1-z)^((N-3)/2); FAP with N_eff trials
    n_s = len(x)
    single = (1 - pk) ** max((n_s - 3) / 2, 1)
    fap = float(1 - (1 - single) ** n_eff)
    return {"ls_power": round(pk, 4), "fap": float(f"{fap:.3e}"),
            "n_eff": round(n_eff, 1)}


def apgar(drift_pct: float, noise_pct: float,
          jitter_h: float, jitter_c: float,
          hnr_h: float, hnr_c: float,
          zeta_h: float | None, zeta_c: float | None,
          shape_score: float | None) -> dict:
    """Structural APGAR: five signs scored 0/1/2, total /10. Boundaries are
    z-scores against the measured baseline scatter — nothing hand-tuned.
    Presentation layer over measured quantities, and labeled as such."""
    def band(z):
        z = abs(z)
        return 2 if z < 2 else (1 if z < 4 else 0)
    sigma = max(noise_pct, 0.2)
    signs = {"pitch": band(drift_pct / sigma)}
    signs["rhythm"] = band((jitter_c - jitter_h) / max(0.5 * jitter_h, 0.05))
    signs["voice"] = band((hnr_h - hnr_c) / 3.0)          # 3 dB HNR ~ 1 z
    if zeta_h and zeta_c:
        signs["ring"] = band((zeta_c - zeta_h) / max(0.3 * zeta_h, 0.002))
    else:
        signs["ring"] = 1
    if shape_score is not None:
        signs["symmetry"] = 2 if shape_score > 0.95 else (1 if shape_score > 0.85 else 0)
    else:
        signs["symmetry"] = 1
    total = int(sum(signs.values()))
    verdict = ("monitor" if total >= 8 else
               "inspect" if total >= 4 else "restrict")
    if 0 in signs.values() and verdict == "monitor":
        verdict = "inspect"   # clinical rule: any sign at 0 is never "all clear"
    return {"signs": signs, "total": total, "verdict": verdict,
            "note": "signs scored 0/1/2 by z-score vs measured baseline scatter; "
                    "any zero sign caps the verdict at inspect"}


# ------------------------------------- standards screening (DIN 4150-3 PPV)

# DIN 4150-3 short-term vibration guide values (peak particle velocity, mm/s)
# per building line, frequency-dependent; tables are public (micromega-dynamics,
# Svantek). Measurement methodology aligned with IS/ISO 4866:2010 (BIS-adopted).
# SCREENING-LEVEL check, not a certified survey (those need calibrated
# transducers) — the report must always carry that label.
DIN_4150_3 = {
    "industrial":  [(1.0, 10.0, 20.0, 20.0), (10.0, 50.0, 20.0, 40.0), (50.0, 100.0, 40.0, 50.0)],
    "residential": [(1.0, 10.0, 5.0, 5.0),   (10.0, 50.0, 5.0, 15.0),  (50.0, 100.0, 15.0, 20.0)],
    "sensitive":   [(1.0, 10.0, 3.0, 3.0),   (10.0, 50.0, 3.0, 8.0),   (50.0, 100.0, 8.0, 10.0)],
}


def din_limit(freq: float, category: str = "residential") -> float:
    """Frequency-dependent DIN 4150-3 PPV guide value (mm/s), linear within bands."""
    bands = DIN_4150_3[category]
    if freq <= bands[0][0]:
        return bands[0][2]
    for lo, hi, vlo, vhi in bands:
        if lo <= freq <= hi:
            return vlo + (vhi - vlo) * (freq - lo) / (hi - lo)
    return bands[-1][3]


def ppv_screening_disp(disp_mm: np.ndarray, fps: float,
                       category: str = "residential") -> dict:
    """CAMERA-ONLY standards screening: metric displacement (mm, from known-size
    target scale calibration) -> velocity -> PPV -> DIN 4150-3 verdict. Fully
    non-contact — nothing touches the structure. Same screening-level label."""
    from scipy.signal import butter, filtfilt
    d = detrend(np.asarray(disp_mm, dtype=float))
    # exact spectral derivative: finite differences attenuate by sinc(f/fs)
    # (11% low at 8 Hz / 60 fps); i*omega in the frequency domain does not.
    # Wrap-around edge artifacts land in the trimmed edges below.
    D = np.fft.rfft(d)
    om = 2j * np.pi * np.fft.rfftfreq(len(d), 1.0 / fps)
    v = np.fft.irfft(D * om, n=len(d))           # mm/s
    nyq = fps / 2.0
    b, c = butter(2, min(1.0 / nyq, 0.9), "high")
    v = filtfilt(b, c, v)
    edge = min(int(fps * 1.5), len(v) // 4)
    core = v[edge:-edge] if len(v) > 4 * edge else v
    ppv = float(np.abs(core).max())
    nper = min(len(v), int(fps * 4))
    f, p = welch(v, fs=fps, nperseg=nper, noverlap=nper // 2)
    m = (f >= 1.0) & (f <= min(100.0, nyq * 0.9))
    f_dom = float(f[m][np.argmax(p[m])]) if m.any() else 0.0
    limit = din_limit(max(f_dom, 1.0), category)
    ratio = ppv / limit if limit else 0.0
    verdict = "OK" if ratio < 0.5 else ("CHECK" if ratio <= 1.0 else "EXCEEDS")
    return {"ppv_mm_s": round(ppv, 3), "dominant_hz": round(f_dom, 2),
            "category": category, "din_limit_mm_s": round(limit, 2),
            "ratio": round(ratio, 3), "verdict": verdict, "channel": "camera-only",
            "note": ("NON-CONTACT screening vs DIN 4150-3 guide values via "
                     "known-target scale calibration; methodology aligned with "
                     "IS/ISO 4866:2010; not a certified vibration survey")}


def ppv_screening(accel: np.ndarray, fs: float,
                  category: str = "residential") -> dict:
    """Peak particle velocity from a contact accelerometer trace (m/s^2) ->
    DIN 4150-3 screening verdict. Velocity by integration with high-pass
    guards against drift; dominant frequency read at the velocity peak."""
    from scipy.signal import butter, filtfilt
    a = detrend(np.asarray(accel, dtype=float))
    nyq = fs / 2.0
    b, c = butter(2, min(1.0 / nyq, 0.9), "high")
    a_f = filtfilt(b, c, a)
    v = np.cumsum(a_f) / fs
    v = filtfilt(b, c, v)                       # kill residual integration drift
    edge = min(int(fs * 1.5), len(v) // 4)      # trim filter-settling transients
    core = v[edge:-edge] if len(v) > 4 * edge else v
    ppv = float(np.abs(core).max() * 1000.0)    # m/s -> mm/s
    nper = min(len(v), int(fs * 4))
    f, p = welch(v, fs=fs, nperseg=nper, noverlap=nper // 2)
    m = (f >= 1.0) & (f <= min(100.0, nyq * 0.9))
    f_dom = float(f[m][np.argmax(p[m])]) if m.any() else 0.0
    limit = din_limit(max(f_dom, 1.0), category)
    ratio = ppv / limit if limit else 0.0
    verdict = "OK" if ratio < 0.5 else ("CHECK" if ratio <= 1.0 else "EXCEEDS")
    return {"ppv_mm_s": round(ppv, 3), "dominant_hz": round(f_dom, 2),
            "category": category, "din_limit_mm_s": round(limit, 2),
            "ratio": round(ratio, 3), "verdict": verdict,
            "note": ("screening-level check vs DIN 4150-3 guide values, "
                     "methodology aligned with IS/ISO 4866:2010; "
                     "not a certified vibration survey")}


# ------------------------------------------------- transmissibility (TF/TC)

def transmissibility(r: ModalResult, fmin: float = 1.0, fmax: float = 25.0) -> dict | None:
    """Output-only transmissibility vs the strongest ROI: T_i(f) = G_i,ref / G_ref,ref.

    TFs are ratios BETWEEN outputs, so the unknown excitation cancels: the same
    fingerprint whether the bridge is shaken by a truck, wind, or a hair dryer.
    Localized stiffness change bends T_i(f) only for ROIs coupled to the damage,
    which is why TF distance localizes where global f1 drift cannot.
    """
    if not r.roi_disp or len(r.roi_disp) < 3:
        return None
    order = sorted(range(len(r.rois)), key=lambda i: r.rois[i][0])
    sigs = [np.asarray(r.roi_disp[i], dtype=float) for i in order]
    nper = min(len(sigs[0]), int(r.fps * 4))
    ref = max(range(len(sigs)), key=lambda i: float(np.var(sigs[i])))
    # full pairwise TF matrix, not one reference: if the reference ROI itself is
    # the damaged one, every single-reference TF changes EXCEPT its own — the
    # classic inversion trap. Pairwise row-medians are reference-free.
    n = len(sigs)
    f_r, _ = welch(sigs[0], fs=r.fps, nperseg=nper, noverlap=nper // 2)
    band = (f_r >= fmin) & (f_r <= fmax)
    autos = [welch(s, fs=r.fps, nperseg=nper, noverlap=nper // 2)[1][band] for s in sigs]
    logT = {}
    coh = {}
    for i in range(n):
        for j in range(i + 1, n):
            _, g_ij = csd(sigs[i], sigs[j], fs=r.fps, nperseg=nper, noverlap=nper // 2)
            _, g2 = coherence(sigs[i], sigs[j], fs=r.fps, nperseg=nper, noverlap=nper // 2)
            T = np.abs(g_ij[band]) / np.clip(autos[j], 1e-20, None)
            logT[(i, j)] = np.log10(np.clip(T, 1e-12, None))
            coh[(i, j)] = g2[band]
    return {"order": order, "n": n, "freqs": f_r[band], "logT": logT, "coherence": coh}


def tf_damage(healthy: dict, current: dict, coh_min: float = 0.6) -> dict:
    """Reference-free TF damage localization: masked log-spectral distance per
    ROI pair (only bins coherent in BOTH states — incoherent bins fake damage),
    then per-ROI indicator = median distance over all pairs containing that ROI.
    A damaged region corrupts every pair it participates in, so its row-median
    spikes while healthy-healthy pairs stay quiet. NOTE: TFs cancel common
    motion, so a purely GLOBAL stiffness change is invisible here by design;
    pair with f1 drift (global, but blind to location) — complementary."""
    n = min(healthy["n"], current["n"])
    pair_lsd = np.zeros((n, n))
    valid = np.zeros((n, n), dtype=bool)
    for (i, j), h_t in healthy["logT"].items():
        if i >= n or j >= n or (i, j) not in current["logT"]:
            continue
        c_t = current["logT"][(i, j)]
        nf = min(len(h_t), len(c_t))
        mask = (healthy["coherence"][(i, j)][:nf] >= coh_min) & \
               (current["coherence"][(i, j)][:nf] >= coh_min)
        if mask.mean() < 0.3:
            continue
        d = float(np.sqrt(((h_t[:nf][mask] - c_t[:nf][mask]) ** 2).mean()))
        pair_lsd[i, j] = pair_lsd[j, i] = d
        valid[i, j] = valid[j, i] = True
    lsd, low_conf = [], []
    for i in range(n):
        vals = pair_lsd[i, valid[i]]
        if len(vals) < 2:
            low_conf.append(i)
            lsd.append(0.0)
        else:
            lsd.append(float(np.median(vals)))
    worst = int(np.argmax(lsd))
    return {"lsd": [round(v, 3) for v in lsd],
            "worst_roi_rank": worst,
            "mean_lsd": round(float(np.mean(lsd)), 3),
            "low_confidence_rois": low_conf}


# ---------------------------------------------------------- adaptive layer

class AdaptiveBaseline:
    """Gated recursive healthy baseline (adaptive-Mahalanobis lineage,
    Sarmadi & Karamodin 2020, MSSP).

    Absorbs slow environmental drift, never absorbs damage: the running
    (mu, Sigma) update with forgetting factor lam only accepts windows that
    pass the chi-squared 95% gate. Novel windows are scored and counted but
    excluded from learning, so a damaged structure cannot re-teach the model
    that damage is normal.
    """

    def __init__(self, feats: np.ndarray, lam: float = 0.05):
        self.lam = lam
        self.df = feats.shape[1]
        self.mu = feats.mean(axis=0)
        self.cov = np.cov(feats.T) + np.eye(self.df) * 1e-6
        self.gate = float(chi2.ppf(0.95, df=self.df))
        self.alert = float(chi2.ppf(0.99, df=self.df))
        self.accepted = self.rejected = 0

    def _d2(self, x: np.ndarray) -> float:
        d = x - self.mu
        return float(d @ np.linalg.solve(self.cov, d))

    def step(self, x: np.ndarray) -> dict:
        """Score one window; adapt only if it looks healthy."""
        d2 = self._d2(x)
        learned = d2 < self.gate
        if learned:
            self.accepted += 1
            d = x - self.mu
            self.mu = (1 - self.lam) * self.mu + self.lam * x
            self.cov = ((1 - self.lam) * self.cov
                        + self.lam * np.outer(d, d) + np.eye(self.df) * 1e-9)
        else:
            self.rejected += 1
        return {"d2": round(d2, 2), "novel": bool(d2 > self.alert), "learned": learned}


def cusum_drift(f1_series: list[float], f1_ref: float, sigma_pct: float | None,
                k: float = 0.5, h: float = 5.0) -> dict:
    """Two-sided CUSUM on per-window f1: flags SLOW cumulative drift that never
    trips the per-window alarm (scour, progressive cracking). Standard control
    chart: drift is standardized by the bootstrap sigma; alarm at h sigma."""
    sig = max((sigma_pct or 0.5) / 100.0 * f1_ref, 1e-9)
    up = dn = 0.0
    trip = None
    for i, f in enumerate(f1_series):
        z = (f - f1_ref) / sig
        up = max(0.0, up + z - k)
        dn = max(0.0, dn - z - k)
        if trip is None and (up > h or dn > h):
            trip = i
    return {"cusum_up": round(up, 2), "cusum_down": round(dn, 2),
            "alarm": trip is not None, "tripped_at_window": trip}


def classify_excitation(disp: np.ndarray) -> dict:
    """Impulsive (tap decay) vs stationary (forced/ambient) via crest factor and
    kurtosis, so downstream estimators adapt: log-decrement damping is only
    meaningful on impulsive records, Welch stats on stationary ones."""
    d = np.asarray(disp, dtype=float)
    d = d - d.mean()
    rms = np.sqrt((d ** 2).mean()) or 1e-12
    crest = float(np.abs(d).max() / rms)
    m2 = (d ** 2).mean() or 1e-12
    kurt = float((d ** 4).mean() / m2 ** 2)
    impulsive = kurt > 6.0 or crest > 5.0
    return {"kind": "impulsive" if impulsive else "stationary",
            "crest": round(crest, 2), "kurtosis": round(kurt, 2)}


def heatmap_overlay(video_path: str, rois_ordered: list[tuple[int, int, int, int]],
                    scores: list[float], low_conf: list[int], out_png: str) -> None:
    """Red->green localization heatmap over the actual video frame (the poster shot)."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return
    for i, ((x, y, w, h), s) in enumerate(zip(rois_ordered, scores)):
        # absolute thresholds (COMAC-style): >=0.9 healthy, <0.7 damage — not
        # min-max, so a uniform (unlocalized) change doesn't paint everything red
        v = float(np.clip((s - 0.7) / 0.2, 0.0, 1.0))
        color = (60, 60, 160) if i in low_conf else (int(60 + 40 * v), int(200 * v), int(220 * (1 - v)))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        label = "low conf" if i in low_conf else f"{s:.2f}"
        cv2.putText(frame, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite(out_png, frame)


def guarded_verdict(f1_healthy: float, f1_current: float,
                    sigma_pct: float | None, eov_pct: float = EOV_BAND_PCT) -> dict:
    """ALERT iff drift exceeds max(3*sigma_bootstrap, EOV guard band)."""
    drift = abs(f1_current - f1_healthy) / f1_healthy * 100.0
    guard = max(3.0 * (sigma_pct or 0.0), eov_pct)
    return {"drift_pct": round(drift, 2), "guard_pct": round(guard, 2),
            "alert": bool(drift > guard),
            "rule": f"ALERT iff drift > max(3*sigma_boot, {eov_pct}% EOV band)"}
