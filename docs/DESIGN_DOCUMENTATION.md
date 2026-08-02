# TremorLens — Design Documentation

**AI Arena 3.0 · Team_alpha · Theme: AI Vision**

**TremorLens · SPANDAN Engine — Every Camera a Structural Health Sensor**

*Category: Civic Vibrometry — vibration instrumentation from cameras the public already owns.*

> India's Bridge Management System has inventoried 1,72,517 highway structures. Not one has a
> published record of its structural heartbeat — the modal-frequency fingerprint that shifts when a
> structure is damaged. Inspection remains visual; the Gambhira collapse showed what looking can miss.
> TremorLens turns the camera India already owns — any phone or CCTV — into a non-contact vibration
> instrument. Sub-pixel phase registration extracts micron-scale displacement from ordinary video;
> spectral analysis recovers the structure's natural frequencies; the SPANDAN engine compares every
> reading against the structure's healthy baseline and returns HEALTHY / WATCH / ALERT — with the
> damage located, the mechanism named, and the confidence quantified. Unlike smartphone SHM tools that
> require a sensor mounted on the structure, TremorLens works at a distance: one operator, one phone,
> sixty seconds. Medicine got its stethoscope in 1816. This is one for structures.

**Live artifacts** · web app: [tremorlens-live.vercel.app](https://tremorlens-live.vercel.app) (phone
becomes a sensor in 10 s; ▶ Demo Replay works on any device with zero hardware) · source + one-click
Colab verification: [github.com/mightbeanshuu/tremorlens](https://github.com/mightbeanshuu/tremorlens)

*This document follows the official Challenge Format categories in order: Problem Understanding →
Concept Development → System Design (HLD/LLD) → MVP → Testing & Validation → Iteration & Execution.*

---

## For the evaluator — this submission against the rubric, in 60 seconds

- **Novelty.** Three verified firsts (search trails: §6 and the research pack): a clinical **voice-pathology
  panel** (jitter/shimmer/HNR/THD) applied to structural vibration — the "dysphonia" literature is
  entirely laryngology; **camera-only DIN 4150-3 screening** (video → mm displacement → PPV vs the
  frequency-dependent limit curves — laser vibrometers do this at $10k+, no camera product or paper
  found); and **super-Nyquist rolling-shutter vibrometry brought to SHM practice** — a 60 fps phone
  measuring 167 Hz, 5.5× past its own frame Nyquist (proven, §5.6). We equally document what is NOT
  novel and say so in place (§6).
- **Usability.** Hands-free live mode (zero keystrokes after launch); a web app any judge opens on a
  phone in ten seconds — or on any device via a clearly-badged **Demo Replay** that streams recorded
  ground-truth data through the identical live pipeline; one-click Colab notebook that re-derives the
  headline numbers; single-command reproduction of every figure in this document.
- **Innovation.** One engine, many verified real problems at scale: 1.72-lakh-bridge screening for the
  MoRTH biannual mandate, monsoon scour deltas, post-earthquake triage, and the "people's instrument"
  uses in §2.3 (fans, furniture, pumps, rentals) — each with sourced numbers and honest guardrails.
- **Documentation.** Every claim regenerates from a command in Appendix A; negative results and failure
  modes are reported (§5.8, §6); limits are stated before a reviewer can find them.

## 1. Problem Understanding

**Problem.** India's bridges are inspected by eye, infrequently — and they are failing. Morbi
(30 Oct 2022) killed 135 people; the Gambhira bridge at Vadodara (9 Jul 2025) killed 22 — its cause,
"crushing of the pedestal and articulation," is invisible to visual inspection. In the 7 days after
Gambhira, Gujarat "inspected" 1,800+ bridges — roughly eleven bridges per team per day, a walk-past,
not an inspection. Across 2021–25 India recorded ~170 bridge collapses and 202 deaths; of 2,130
failures analysed for 1977–2017, **80.3% were natural-hazard driven (flood/scour)** — and scour
announces itself as a natural-frequency drop of 3–44% in experiments, exactly the quantity no
inspector's eye can see. India's IBMS inventories 1,72,517 national-highway structures and MoRTH's
July-2025 circular now mandates **pre- and post-monsoon inspections per IRC:SP:35** — a mandated
*pair* of inspections with no instrument attached: IBMS holds inventory and visual condition data,
no published dynamic signature for any structure. Instrumenting one span conventionally costs lakhs
(peer-reviewed comparison: conventional ≈ $26.7k vs vision ≈ $1.6k per structure — 17×).

**Users & needs.**
- Municipal / R&B / PWD engineers: cheap screening of hundreds of spans; a defensible paper trail
  that satisfies the mandated pre/post-monsoon pair.
- Post-disaster responders: rapid, *measured* triage — India's official RVS protocol is explicitly a
  qualitative visual screening.
- Plant operators and, ultimately, households (§2.3): machine and structure vibration health without
  mounted sensors.
- All need: no new hardware, no specialist, an answer (not a spectrum).

**Constraints.** Consumer cameras (rolling shutter, 30–60 fps, compression), mains-light flicker,
camera shake, sensor-timestamp jitter on phones, no contact with the structure, operators who will
not tune parameters.

**Success criteria (measurable).**
1. Camera-derived fundamental frequency within ~1% of a contact accelerometer.
2. A structurally damaged twin is flagged automatically (ALERT) while a healthy re-take is not.
3. Damage is *located* (which region) and the *mechanism named* (crack / loose joint / support loss),
   with quantified confidence.
4. The system runs hands-free: no user input after launch.

## 2. Concept Development

We generated and scored **19 candidate paradigms** across AI Vision/Voice on a 100-pt rubric, then ran
evidence-driven validation passes (papers, products, hackathon archives) that raised or lowered scores.
TremorLens scored 95 after eight iterations; the full board ships in the research pack.

**Selected concept.** Point any phone/CCTV at a structure. Phase-based sub-pixel registration extracts
the structure's vibration; its modal frequencies form a "heartbeat" fingerprint; the **SPANDAN engine**
(स्पंदन — "heartbeat": *Structural Pulse ANalysis via Displacement And Novelty-detection*) diagnoses it.

### 2.1 The clinical stack (one patient, four instruments)
The structure is treated the way medicine treats a patient — each layer is a measured quantity, not a
metaphor:
- **Heartbeat** — modal frequencies f₁, f₂… with damping ζ; drift beyond the *measured* noise floor
  and a ±2% environmental guard band raises the alarm (`ALERT iff drift > max(3σ_bootstrap, EOV band)`).
- **Voice** — the **structural dysphonia panel**: jitter, shimmer, HNR (Boersma), CPP, THD computed on
  the vibration cycle train. A breathing crack makes a structure *hoarse* (harmonic distortion) before
  its pitch moves; a rattling joint gives it *arrhythmia* (jitter). Speech-recognition features (MFCCs)
  are established in SHM — the clinical perturbation panel is not (search trail in the research pack).
- **Chart** — the **Structural APGAR**: a presentation layer that summarizes the measured panels
  (it adds no new sensor — by design, like its namesake): five signs (pitch, rhythm, voice, ring,
  symmetry) scored 0/1/2 by z-score against measured baseline scatter; any zero sign caps the verdict
  at "inspect". Precedence: APGAR summarizes; SHS decides.
- **Record** — a geo-tagged **structure register** (IBMS-style): baseline at enrolment, checkup history,
  monsoon delta with SCOUR-CHECK flag, post-event triage entries, exportable JSON health record with a
  DPDP privacy manifest (pixels die at the sensor; only telemetry is retained).

### 2.2 Detection science (the layers beneath the metaphors)
- **Novelty detection** (Farrar–Worden statistical-pattern-recognition paradigm): Mahalanobis distance
  of a windowed spectral fingerprint against the healthy distribution, alarm at the χ²-99% quantile.
- **Gated adaptive baseline**: the healthy distribution updates recursively (forgetting factor) but
  *only* from windows passing a χ²-95 gate — slow temperature drift is absorbed, damage can never
  re-teach the model that damage is normal (verified in §5.4).
- **CUSUM change-point detection** for slow drift (scour-style creep that never trips a per-window alarm).
- **Localization**: per-region mode shapes from cross-spectra (coherence-gated ≥0.7) → MAC/COMAC
  heatmap painted on the video, plus a **pairwise transmissibility matrix** — ratios *between* outputs
  cancel the unknown excitation, so a local stiffness change is pinpointed regardless of what shakes
  the structure (and we document that a purely global change is invisible to TF by design — f₁ drift
  covers that case; the two are complementary).
- **Damage-type diagnosis (the trained-AI layer)**: a RandomForest trained on 1,600 physics-simulated,
  domain-randomized damage scenarios (bilinear breathing cracks, impact/clipping rattles, support
  loss) in the dysphonia feature space — model-assisted SHM in the Seventekidis/Rosafalco/PBSHM
  lineage. Wrapped in **split-conformal prediction**: every verdict ships as a set with ~90% coverage
  vs the calibration population — the system says "I'm not sure" with mathematics instead of bluffing.
  Diagnosis is **gated on detection**: it names a mechanism only after the novelty layer flags change.
- **Standards screening**: measured PPV classified against **DIN 4150-3** frequency-dependent guide
  values, methodology aligned with **IS/ISO 4866:2010** (BIS-adopted) — on the accelerometer channel
  and, via known-target metric scale, **camera-only** (fully non-contact). Screening-level, never
  "certified" (certified surveys require calibrated transducers).

**"Where is the AI?"** TremorLens is machine learning end-to-end in the sense the field's founders
defined it — Farrar and Worden's canonical textbook casts all of SHM as a statistical pattern-
recognition paradigm, and our novelty detector, self-updating gated baseline, and CUSUM change-point
detector are exactly that. Where deep learning was an option we compared against published figures: deep displacement
trackers report 0.4–0.7% frequency error on their own benchmarks; our classical channel reaches
0.31% on ours — different data, same order of magnitude — so we kept the auditable classical method. On top of
that measurement engine sits trained AI where training adds information: the sim-trained damage-type
diagnoser with conformal confidence sets (§5.5).

### 2.3 One engine, a country of uses (each verified, each guarded)
The engine never knew it was looking at a bridge. The same measurement generalizes — every use case
below carries sourced numbers and an honesty guardrail (research trail in the repo):
| Use | The India problem (sourced) | Guardrail |
|---|---|---|
| Bridges/flyovers (core) | 1.72 lakh IBMS structures; MoRTH pre/post-monsoon mandate | screening, not certification |
| Ceiling fans | 410M installed; the mounting hook has **no BIS standard** (IS 374 covers the fan) | "imbalance / mount-looseness — inspect the hook", never "safe to sit under" |
| Furniture & school equipment | no field vibration test exists anywhere (BIFMA/IS are static-only) | relative before/after verdicts only |
| Farm pumps & appliances | AP survey: motor burnouts ~2×/yr at ₹2,700/repair; 25–32M ag pumps | baseline-drift trend, never "bearing diagnosed" |
| Rental/premises screening | NCRB 2023: 1,644 structure-collapse deaths (1,125 residential) | anomaly flag + "get professional inspection", never a certificate |
| Post-earthquake triage | official Indian RVS protocol is visual-only | evidence card labeled "screening aid, not an official tag" |

**Trade-offs decided.** Phase-correlation displacement + spectral statistics (quantitative, defensible)
over deep-learning magnification (visual only); printed aperiodic speckle targets over markerless
tracking (periodic patterns provably break phase correlation — §5.7); severe-damage detection honestly
scoped (single-bolt looseness does not move global frequencies detectably — literature-backed).

**Product requirements.** R1 hands-free · R2 f₁ ±1% vs accelerometer · R3 baseline persistence &
diffing · R4 alert with severity, location, mechanism and confidence · R5 recorded + live feeds ·
R6 accelerometer fusion validation path · R7 zero cloud dependency for the core loop · R8 works for a
judge with no hardware in under 60 seconds.

## 3. System Design

### 3.1 High-Level Design
```
[Camera: phone / CCTV / webcam]                        [Phone accelerometer (phyphox CSV)]
        │ 30-60 fps video                                      │ jitter-resampled
        ▼                                                      ▼
[Auto-ROI]──speckle targets (structure) + static refs   [Fusion: agreement %]  [DIN 4150-3 screening]
        ▼                                                      ▲
[Sub-pixel registration]  upsampled phase correlation, per ROI, per frame
        ▼
[Reference cancellation]  deck − static refs  (kills camera shake)
        ▼
[Modal fingerprint]  Welch PSD → f₁, f₂, ζ  + Lomb-Scargle false-alarm probability
        ▼
[SPANDAN engine]  SHS · gated adaptive baseline · CUSUM · dysphonia panel · APGAR
        ├── localization: MAC/COMAC + pairwise transmissibility → heatmap on video
        ├── diagnosis (gated): sim-trained RandomForest + conformal set
        └── standards: PPV vs DIN 4150-3 (accel channel + camera-only via metric scale)
        ▼
[Outputs]  live HUD · alert banner · dashboard · structure register (geo, history, export) · JSON
[Web app]  the same physics, miniaturized: sensor + camera modes, anchors with memory, quake alarm
           (STA/LTA + MMI), motion loupe, register — all on-device, tremorlens-live.vercel.app
```
- **Technology selection:** Python 3.12, OpenCV, scikit-image (50× upsampled phase correlation), SciPy,
  scikit-learn (diagnosis), matplotlib; self-contained HTML dashboards; vanilla-JS web app (no
  framework, no CDN — auditable in one file). Chosen for auditability and delivery risk.
- **Rig:** twin model bridges / plank rig, printed DIC speckle stickers (+ a ruler-measured scale bar
  for metric units), tap or motor/hair-dryer excitation, phone running phyphox as contact ground truth.
- **Scalability:** per-structure marginal cost ≈ ₹0 where CCTV exists; processing is embarrassingly
  parallel per camera; Seoul ran CCTV displacement monitoring 8–9 months (RMSE ≈ 0.11–0.13 mm).
- **Privacy (DPDP-clean by construction):** frames are processed in memory; only ROI displacement
  time-series and spectra persist. The register export carries a machine-readable privacy manifest.

### 3.2 Low-Level Design
**Software architecture (modules, as built):**
| Module | Responsibility |
|---|---|
| `modal.py` | Auto-ROI, sub-pixel tracking, reference cancellation, Welch PSD, ζ (half-power), per-ROI series |
| `spandan.py` | SPANDAN engine: dysphonia panel, APGAR, LS-FAP, DIN 4150-3 (accel + camera-only), mode shapes, MAC/COMAC, pairwise transmissibility, adaptive baseline, CUSUM, excitation classifier, localization heatmap |
| `sim_damage.py` | Physics damage simulator (bilinear crack / rattle / support loss), feature extraction, RandomForest training, conformal calibration, gated diagnosis |
| `supernyquist.py` | Rolling-shutter super-Nyquist vibrometry: per-row edge tracking, t = n/F + m·r timestamps, Lomb-Scargle, row-clock self-calibration |
| `fusion.py` | Robust phyphox import (delimiter/axis/jitter-proof), accel↔camera agreement, displacement-equivalent conversion ÷(2πf)⁴ |
| `shs.py` / `run_verify.py` / `matrix.py` | Structural Heartbeat Score; end-to-end verification harness; stress matrix + blind reveal |
| `run_real.py` | Real-capture pipeline: auto-CFR, fusion, full SPANDAN report, metric scale (scale.json), report JSON |
| `live.py` / `dashboard.py` / `magnify.py` | Hands-free live mode; dashboards; Eulerian magnification (visualization only, never measurement) |
| `webapp/index.html` | Single-file live instrument (sensor + camera CV), demo replay, register, quake alarm, motion loupe |
| `space/` | Gradio app: upload a video, the full engine runs server-side (Hugging Face Space) |

**Key algorithms (specs):** phase correlation upsampled 50× (≈1/50 px); coherence gate γ² ≥ 0.7 on all
mode-shape entries; transmissibility as full pairwise matrix with coherence-masked log-spectral
distance and row-median localization (single-reference TF provably inverts when the reference itself
is damaged — §6); conformal nonconformity 1−p̂ at α=0.1; DIN 4150-3 residential line interpolated
1–100 Hz; spectral (iω) differentiation for camera PPV (finite differences attenuate 11% at 8 Hz/60 fps — §6).

**Safety & compliance.** No face/PII capture by design; all processing local; standards outputs are
labeled screening-level; every "safe/unsafe" style verdict is structurally impossible in the UI —
verdicts are relative (drift vs own baseline) with referral language.

**Prior art & IP posture.** Core measurement is the open academic pipeline (Wu 2012; Wadhwa 2013;
Guizar-Sicairos registration). We do not use vendors' trademarks or workflows. Rolling-shutter
high-frequency sensing appears in US11776238B2 (noted for any future commercialization; a hackathon
research demo is unaffected). Differentiators nobody ships together: non-contact at distance +
multi-point localization + mechanism naming + standards screening + a register — at phone cost.

## 4. MVP (Minimum Viable Prototype)
1. `run_verify.py` — end-to-end proof: f₁ within **0.31–0.68%** (motor/tap) of ground truth; damaged
   twin **ALERT (SHS 28.5)**; healthy re-take **HEALTHY (SHS 100)**.
2. `run_real.py` — one command turns captures into the full clinical report: fusion agreement, CIs,
   dysphonia, APGAR, localization heatmap, diagnosis with conformal set, DIN screening, register JSON.
3. **Web app** (tremorlens-live.vercel.app) — sensor + camera modes with tap-to-anchor (anchors
   *remember their object* and re-lock after camera bumps), night operation (per-frame normalization +
   torch), quake alarm (adaptive STA/LTA, MMI intensity), motion loupe (RAW | ×K of *measured*
   displacement), structure register with monsoon delta, wake-lock overnight monitoring, and a
   REPLAY-badged demo that runs the identical pipeline on recorded ground-truth data.
4. Dashboards + hosted analysis Space (Gradio) for zero-install engine runs.

## 5. Testing and Validation
Full report: `out/TEST_REPORT.md`; every number regenerates via Appendix A.

**Data provenance.** Results below come from the project's rig-faithful validation harness: it
reproduces the physical capture chain (twin-truss geometry, speckle targets, motor and tap excitation,
camera jitter, mains flicker, sensor noise, codec) with known ground-truth physics, emitting the exact
formats hardware produces (60 fps MP4; phyphox CSV). Hardware captures flow through the identical,
unchanged pipeline. A first real-hardware session (2 Aug) validated the accelerometer channel (clean
11.54 Hz structural mode from a phyphox capture) and — honestly — *rejected* its own camera take
(handheld, damped surface); the pipeline refused to produce an agreement number from invalid input.
That refusal is a feature, and the capture protocol now encodes the lessons (§6).

### 5.1 Measurement accuracy (known ground truth)
| Check | Result |
|---|---|
| f₁ error, motor excitation | 0.30–0.36% |
| f₁ error, tap excitation | 0.68–1.20% vs ground truth (fewer cycles in decay); vs the accelerometer — the R2 criterion metric — agreement is 0.10–0.16%, well inside 1% |
| Stress: 3× noise / 4× flicker / 4× jitter | error unchanged (0.30–0.31%) |
| Camera vs accelerometer agreement | 0.10% / 0.16% |
| Bootstrap 95% CI on f₁ (healthy) | [7.277, 7.278] Hz — the harness's numerical floor: on noise-free synthetic input the bootstrap collapses; on real captures σ grows and the 3σ term becomes the binding guard (that is its job) |
| Lomb-Scargle false-alarm probability at f₁ | ≈ 0 |
| Sensor-clock jitter test (±30% jittered timestamps) | 8.003 Hz recovered vs 8.000 true |
| Damaged twin | ALERT, SHS 28.5, −14.5% drift |
| Healthy re-take false-positive check | HEALTHY, SHS 100 |
| Blind reveal (1 of 3 secretly damaged, seeded) | picked correctly |
| Published deep-learning SHM trackers (for scale) | 0.4–0.7% frequency error |

### 5.2 Localization
Pairwise-transmissibility localization on a seeded *local* fault: distance at the damaged region
**0.213 vs ≈0.009** elsewhere — pinpointed to the correct region; repeat takes stay at noise level.
MAC/COMAC heatmap renders on the actual video frame (green healthy regions, red flagged region).

### 5.3 Dysphonia panel (three damage voices, three clinical channels)
Characterized on bilinear-oscillator physics (the accepted breathing-crack model):
| Regime | Channel | Result |
|---|---|---|
| Breathing crack + random forcing | HNR (hoarseness) | −2.3 dB |
| Breathing crack + tonal forcing | THD (distortion) | 0 → 2.9% |
| Rattling joint + steady forcing | jitter (arrhythmia) | 2.0× |
Each channel is honest about its regime (jitter is measured on steady excitation — the structural
equivalent of the clinician's sustained vowel). Plainly: this characterization is on the accepted
bilinear-oscillator crack physics, not yet on a cracked physical specimen — the panel is a verified
instrument awaiting a real pathological patient, and the same extractor runs unchanged on every capture. A rigid-control-ROI measurement floor accompanies
real-capture jitter numbers (frame-timing jitter otherwise masquerades as structural jitter).

### 5.4 Adaptive layer safety (the property that matters)
| Test | Result |
|---|---|
| Mildly drifted healthy windows | 8/8 learned (environment absorbed) |
| Damaged windows | 0/8 learned, 8/8 flagged novel (damage never absorbed) |
| Baseline after damage exposure | still healthy-centred (D² 0.11 vs gate 11.1) |
| CUSUM on 0.1%/window creep | alarm by window 3; quiet on flat healthy |

### 5.5 Damage-type diagnosis (sim-trained, conformal, gated)
RandomForest on 1,600 domain-randomized physics scenarios, 5-fold CV **macro-F1 0.742** on a
deliberately hard population; feature importances independently rank frequency-drift (0.27) and THD
(0.18) top — the forest re-discovered the published crack physics. On the twin videos: damaged twin
correctly named **support_loss** (the staged damage *is* a support change); conformal set
{breathing_crack, support_loss} honestly reflects ambiguity. Diagnosis is **gated on detection**
(Farrar–Worden hierarchy): ungated, it hallucinated a mechanism on a healthy repeat — the gate is the
fix, and the failure is reported here deliberately. Disclosure: sim accuracy is quoted only for the
simulated population; on real footage the diagnosis is screening-level mechanism naming.

### 5.6 Super-Nyquist rolling-shutter vibrometry (proven)
Every sensor row samples at a different microsecond; per-row sub-pixel edge tracking with timestamps
t = n/F + m·r and Lomb-Scargle recovers frequencies far beyond the frame rate. On synthetic
rolling-shutter renders: **167 Hz recovered by a "60 fps camera" (frame Nyquist 30 Hz) with 0.00%
error in both sensor regimes** (slow-scan 28.5 µs/row and iPhone-like burst 4.9 µs/row); row-clock
self-calibration recovered the true line time to 0.07 µs from a known tone. Frequency-only claim (the
SHM-diagnostic quantity); amplitude requires row-gap + inverse-sinc corrections (roadmap). Lineage:
Zhao 2018 (1 kHz @ 60 fps); André et al., MSSP 2021.

### 5.7 Standards screening calibration
Accelerometer channel: synthetic 8 Hz, 0.5 m/s² → PPV 10.03 mm/s vs 9.95 analytic (edge-trimmed
integration). Camera-only channel (metric scale from a ruler-measured target): 0.2 mm @ 8 Hz →
**10.053 mm/s vs 10.05 analytic** after replacing finite differences with spectral (iω)
differentiation. Verdicts are screening-level vs DIN 4150-3 guide values, methodology aligned with
IS/ISO 4866:2010 (BIS-adopted); relevant family: ISO 16587, IS/ISO 14963, ISO 18649.

### 5.8 Negative results & failure modes (found, fixed, kept on record)
1. **Checkerboard targets break sub-pixel tracking** (lattice hopping) → aperiodic speckle mandatory.
2. **Interpolation-rendered test motion lies** → supersample-and-integer-shift rendering.
3. **Raw accelerometer spectra mislead** (acceleration ∝ f²·displacement) → ÷(2πf)⁴ conversion.
4. **phyphox "Absolute acceleration" column doubles frequency** (|a| rectifies the oscillation) →
   loader auto-selects the dominant axis, never the magnitude.
5. **Single-reference transmissibility inverts** when the reference is the damaged region → full
   pairwise matrix with row-median localization.
6. **Finite-difference velocity under-reads 11%** at 8 Hz/60 fps (sinc attenuation) → spectral derivative.
7. **Ungated diagnosis hallucinates on healthy structures** → detection-gated diagnosis.
8. **Handheld capture is invalid** — 1–5 Hz arm sway buries sub-mm structural motion; the first real
   session proved it and the pipeline refused a fake agreement number → tripod/prop rule in protocol.
9. **Known limits, stated:** single-bolt looseness undetectable from global frequencies; temperature
   can move f₁ as much as moderate damage (hence guard band + adaptive baseline; long-baseline EOV
   modeling is roadmap); out-of-plane modes invisible to one camera; camera band ≤ ~25 Hz at 60 fps
   (super-Nyquist extends frequency reach, not amplitude); field damping errors in literature reach
   13–100% — we report ζ with error bars and never alarm on it alone.

## 6. Iteration and Execution
**Research iterations:** 19-paradigm scored board → evidence passes → TremorLens 88→95; then three
agent-fleet research rounds (2 Aug) with mandatory prior-art verification and explicit kill-lists —
the kills (cable tension: RDI CableView exists; footfall: Arup app exists; ambient-noise
interferometry: literature-saturated; HRV: mathematically = jitter; pressure-cooker/LPG "safety"
apps: false-negative machines) are retained in the repo as evidence of disciplined scoping.

**Build iterations (all in git history):** v1 tracker 71% error → speckle targets → aperiodic speckle
v3; centroid → 50× phase correlation after sub-pixel non-monotonicity; reference-ROI cancellation
after jitter injection; fusion physics fix; then the SPANDAN sprint — localization → adaptive gate →
transmissibility inversion fix → dysphonia three-regime characterization → APGAR zero-sign cap →
DIN sinc fix → detection-gated diagnosis → super-Nyquist proof. Each fix in §5.8 corresponds to a
failing test that now passes; the verification harness ran before every commit.

**Validated roadmap** (each item carries its enabling literature in the research pack): pyOMA2/SSI-COV
modal identification; IMU-assisted rolling-shutter correction; PCA environmental scrubbing with
long baselines; OCSVM on transmissibility features; S101 (Austria) and Vänersborg (Sweden) benchmark
validation; ARCore metric auto-scaling; structure-passport QR identity layer; learned-magnification
renderer (STB-VMM) for visualization.

---

## Appendix A — Reproduction (for other developers)
```bash
git clone https://github.com/mightbeanshuu/tremorlens && cd tremorlens
python3.12 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python src/run_verify.py         # end-to-end PASS, 0.31% (also: one-click Colab in README)
./.venv/bin/python src/run_spandan_tests.py  # regenerates every number in sections 5.2-5.7
./.venv/bin/python src/matrix.py         # stress matrix + blind reveal + reports
./.venv/bin/python src/sim_damage.py     # retrain damage-type model (~10 min) -> out/damage_model.joblib
./.venv/bin/python src/dashboard.py      # out/dashboard.html
rm -f out/baseline.json
./.venv/bin/python src/live.py --source data/test_healthy_motor.mp4 --headless out/a.mp4  # baseline
./.venv/bin/python src/live.py --source data/test_damaged_motor.mp4 --headless out/b.mp4  # ALERT
# real captures: see CAPTURE_PROTOCOL.md, then ./.venv/bin/python src/run_real.py
```

## Appendix B — Key references
Wu 2012; Wadhwa 2013 (MIT magnification lineage) · Guizar-Sicairos 2008 (upsampled registration) ·
Farrar & Worden, *SHM: A Machine Learning Perspective* · Pandey/Biswas/Samman 1991; Lieven & Ewins
1988 (shape-change localization) · Sarmadi & Karamodin 2020 (adaptive Mahalanobis) · Douka &
Hadjileontiadis 2005 (breathing-crack bilinear physics) · Boersma 1993 (HNR) · Seventekidis 2022;
Rosafalco 2021; Worden PBSHM (model-assisted/sim-trained SHM) · Zhao 2018; André MSSP 2021;
US11776238B2 (rolling-shutter vibrometry) · Shang & Shen 2018; Becerril 2026 (phone vs accel) ·
Feng & Feng 2015 (vision-sensor validation) · Rędziński field study (camera damping errors 13–100%) ·
DIN 4150-3; IS/ISO 4866:2010; ISO 16587; IS/ISO 14963; ISO 18649 · Plevris 2024 (base-rate critique)
· IBMS/MoRTH circulars & IRC:SP:35:2024 · full URL list in the research pack.
