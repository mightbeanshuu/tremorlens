"""Sim-trained damage-TYPE diagnosis — model-assisted SHM (Seventekidis 2022,
Rosafalco 2021, Worden's PBSHM lineage) in a feature space nobody has used:
the clinical voice-pathology (dysphonia) panel.

Pipeline: physics-simulated (baseline, current) signal pairs with domain
randomization -> the SAME spandan.dysphonia() extractor used on real
measurements -> RandomForest over delta features -> split-conformal prediction
sets (distribution-free ~90% coverage vs the simulated calibration population).

DISCLOSURE (verbatim for docs): damage-type diagnosis is trained on physics-
simulated damage scenarios; on real footage it is screening-level — it names
the most physically consistent damage mechanism, not a certified diagnosis.

Train:  python src/sim_damage.py            (writes out/damage_model.joblib)
"""
import os

import numpy as np
from scipy.signal import welch

import spandan

FPS = 60.0
DUR = 45.0
CLASSES = ["healthy", "breathing_crack", "loose_joint", "support_loss"]


# ------------------------------------------------------------- physics sims

def _sdof(n, fps, f0, zeta, forcing, kratio=1.0, sub=4):
    """SDOF integrator; kratio>1 gives bilinear (breathing-crack) stiffness."""
    w2o = (2 * np.pi * f0) ** 2
    w2c = w2o * kratio
    x = v = 0.0
    out = np.empty(n)
    dt = 1.0 / fps
    for i in range(n):
        f = forcing[i]
        for _ in range(sub):
            w2 = w2o if x >= 0 else w2c
            a = -w2 * x - 2 * zeta * np.sqrt(w2) * v + f
            v += a * dt / sub
            x += v * dt / sub
        out[i] = x
    return out


def _forcing(n, rng, kind):
    if kind == "random":
        return rng.standard_normal(n) * 40
    t = np.arange(n) / FPS
    f_drive = rng.uniform(6, 11)
    return 25 * np.sin(2 * np.pi * f_drive * t) + rng.standard_normal(n) * 8


def simulate_pair(cls: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Domain-randomized (baseline, current) displacement pair for one class."""
    n = int(FPS * DUR)
    f0 = rng.uniform(5.0, 12.0)
    zeta = rng.uniform(0.01, 0.05)
    kind = rng.choice(["random", "tonal"])
    base = _sdof(n, FPS, f0, zeta, _forcing(n, rng, kind))

    if cls == "healthy":
        cur = _sdof(n, FPS, f0 * rng.uniform(0.995, 1.005), zeta,
                    _forcing(n, rng, kind))
    elif cls == "breathing_crack":
        cur = _sdof(n, FPS, f0, zeta, _forcing(n, rng, kind),
                    kratio=rng.uniform(1.15, 1.6))
    elif cls == "loose_joint":
        cur = _sdof(n, FPS, f0, zeta, _forcing(n, rng, kind))
        gap = rng.uniform(0.5, 0.8) * np.abs(cur).max()
        cur = np.clip(cur, -gap, gap)                      # contact stop
        hits = rng.random(n) < rng.uniform(0.01, 0.04)     # rattle impacts
        cur = cur + hits * rng.standard_normal(n) * 0.4 * gap
    elif cls == "support_loss":
        drop = rng.uniform(0.80, 0.92)                     # 8-20% f1 drop
        cur = _sdof(n, FPS, f0 * drop, zeta * rng.uniform(1.0, 1.5),
                    _forcing(n, rng, kind))
    else:
        raise ValueError(cls)

    # sensor noise + low-passed tripod-jitter contamination on both channels
    for sig in (base, cur):
        sig += rng.standard_normal(n) * 0.02 * np.abs(sig).max()
    return base, cur


# ---------------------------------------------------------------- features

def _f1_of(x: np.ndarray, fps: float = FPS) -> float:
    f, p = welch(x, fs=fps, nperseg=int(fps * 4))
    m = (f >= 1) & (f <= 25)
    return float(f[m][np.argmax(p[m])])

FEATURES = ["d_f1_pct", "jitter_ratio", "shimmer_ratio", "hnr_delta_db",
            "thd_delta_pct", "cpp_delta_db", "kurtosis_ratio", "crest_ratio"]


def pair_features(base: np.ndarray, cur: np.ndarray,
                  fps: float = FPS) -> np.ndarray | None:
    """Delta features between baseline and current — the exact same
    spandan.dysphonia() extractor that runs on real measurements."""
    f1b, f1c = _f1_of(base, fps), _f1_of(cur, fps)
    db = spandan.dysphonia(base, fps, f1b)
    dc = spandan.dysphonia(cur, fps, f1c)
    if not db or not dc:
        return None
    eps = 1e-6
    eb = spandan.classify_excitation(base)
    ec = spandan.classify_excitation(cur)
    return np.array([
        (f1c - f1b) / f1b * 100,
        dc["jitter_pct"] / (db["jitter_pct"] + eps),
        dc["shimmer_pct"] / (db["shimmer_pct"] + eps),
        dc["hnr_db"] - db["hnr_db"],
        (dc["thd_pct"] or 0) - (db["thd_pct"] or 0),
        (dc["cpp_db"] or 0) - (db["cpp_db"] or 0),
        ec["kurtosis"] / (eb["kurtosis"] + eps),   # rattle impacts -> heavy tails
        ec["crest"] / (eb["crest"] + eps),
    ])


# ------------------------------------------------------- train + conformal

def train(n_per_class: int = 400, seed: int = 11, out_path: str = "out/damage_model.joblib"):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import confusion_matrix, f1_score
    import joblib

    rng = np.random.default_rng(seed)
    X, y = [], []
    for ci, cls in enumerate(CLASSES):
        made = 0
        while made < n_per_class:
            feats = pair_features(*simulate_pair(cls, rng))
            if feats is not None:
                X.append(feats)
                y.append(ci)
                made += 1
        print(f"  simulated {cls}: {made}")
    X, y = np.array(X), np.array(y)

    clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    yp = cross_val_predict(clf, X, y, cv=5)
    print("5-fold confusion matrix (rows=true):")
    print(confusion_matrix(y, yp))
    macro_f1 = f1_score(y, yp, average="macro")
    print(f"macro-F1: {macro_f1:.3f}")

    # split-conformal calibration: 20% held out, nonconformity = 1 - p(true)
    idx = rng.permutation(len(X))
    n_cal = len(X) // 5
    cal, tr = idx[:n_cal], idx[n_cal:]
    clf.fit(X[tr], y[tr])
    p_cal = clf.predict_proba(X[cal])
    scores = 1 - p_cal[np.arange(n_cal), y[cal]]
    qhat = float(np.quantile(scores, 0.9 * (1 + 1 / n_cal)))

    clf_full = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    clf_full.fit(X, y)
    imp = dict(zip(FEATURES, np.round(clf_full.feature_importances_, 3)))
    print("feature importances:", imp)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump({"model": clf, "model_full": clf_full, "qhat": qhat,
                 "classes": CLASSES, "features": FEATURES,
                 "macro_f1_sim": round(float(macro_f1), 3),
                 "importances": imp, "n_train": len(X)}, out_path)
    print(f"saved -> {out_path} (conformal qhat {qhat:.3f})")
    return out_path


def diagnose(base_disp: np.ndarray, cur_disp: np.ndarray, fps: float,
             model_path: str = "out/damage_model.joblib") -> dict | None:
    """Damage-type prediction + conformal set for a real (baseline, current) pair."""
    if not os.path.exists(model_path):
        return None
    import joblib
    bundle = joblib.load(model_path)
    feats = pair_features(base_disp, cur_disp, fps)
    if feats is None:
        return None
    proba = bundle["model"].predict_proba([feats])[0]
    order = np.argsort(proba)[::-1]
    conformal_set = [bundle["classes"][i] for i in range(len(proba))
                     if proba[i] >= 1 - bundle["qhat"]]
    return {"prediction": bundle["classes"][int(order[0])],
            "proba": {bundle["classes"][i]: round(float(proba[i]), 3)
                      for i in order},
            "conformal_set_90": conformal_set or [bundle["classes"][int(order[0])]],
            "sim_macro_f1": bundle["macro_f1_sim"],
            "note": ("trained on physics-simulated damage scenarios "
                     "(model-assisted SHM); screening-level mechanism naming, "
                     "not a certified diagnosis")}


if __name__ == "__main__":
    print("Training the damage-type model on 1,600 physics-simulated scenarios (~10 min). Ctrl-C to abort.")
    train()
