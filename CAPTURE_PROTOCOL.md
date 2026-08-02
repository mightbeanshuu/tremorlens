# TremorLens — Real-Data Capture Protocol (Android phone + hair dryer, no motor)

Goal: genuine hardware takes that drop into `data/real/` and regenerate every
artifact via `python src/run_real.py`. Total time ~45–60 min.

## 1. The rig (household items)

**Structure — pick one:**
- **A. Plank bridge (preferred):** wooden plank / sturdy cutting board / shelf
  board spanning two stacks of books on two chairs, span 60–90 cm. Heavy enough
  that a phone lying on it barely shifts f1.
- **B. Ruler cantilever (fallback):** steel scale clamped under a stack of books
  at a table edge, 15–20 cm overhang. Light — do NOT put the phone on it.

**Targets (mandatory):** hand-draw dense random blobs/speckles with a black
marker on white paper — aperiodic, NOT checkerboards (periodic patterns make
phase correlation lattice-hop). Tape 4–6 patches along the structure's side
facing the camera, and 2–3 patches on the supports/floor (static references).

**Framing (matters — auto-ROI depends on it):** camera 1–1.5 m away, landscape.
The structure must cross the frame at roughly **45–74% of frame height**
(middle band); supports/table/floor visible in the **bottom quarter**. Camera
rigid on books/tripod — never touched during a take.

## 2. Camera + accelerometer plans

- **Plan A (two devices — best):** a second phone films **1080p @ 60fps**
  while your Android lies flat mid-span running phyphox → simultaneous
  camera + contact ground truth.
- **Plan B (solo):** your Android lies mid-span running phyphox; the Mac webcam
  films. Webcam is ~30fps → structure f1 must be < ~12 Hz (plank spans are).
- **Plan C (solo, one camera):** Android films first, then rides the plank for a
  phyphox take. Only valid on a heavy structure (phone < ~5% of plank mass).

If the phone rides the structure during video takes, it rides it in **all**
takes — the measured system is "plank + phone", kept consistent.

Camera settings: 60fps if available (Android camera app → video → 60), lock
focus/exposure (tap-hold), bright steady light.

**phyphox:** experiment **"Acceleration (without g)"**, phone flat mid-span
(tape it). Start phyphox, start video, sharp clap (sync marker), run the take,
stop both, Export Data → CSV.

## 3. Excitation (hair dryer = the motor substitute)

- **Tap:** sharp knuckle/pen tap mid-span, hands fully off, let it ring 10–15 s.
  Do 2–3 taps per take.
- **Forced:** hair dryer, **cold setting**, fixed aim at mid-span from
  10–20 cm for 30–45 s → broadband forced excitation. Hold it or rest it on a
  separate chair — not on the same table as the rig or camera.

## 4. The takes (each 45–60 s)

| # | File | What |
|---|------|------|
| 1 | `healthy.mp4` + `accel_healthy.csv` | baseline: taps, then hair-dryer stretch |
| 2 | `healthy_take2.mp4` | exact repeat, touch nothing → measured noise floor |
| 3 | — | **gross damage:** slide one support inward 15–20% (bearing-failure / scour narrative). Small changes are undetectable — be gross. |
| 4 | `damaged.mp4` + `accel_damaged.csv` | same excitation routine |

## 5. Run

```bash
mkdir -p data/real   # drop the files above in here (exact names)
source .venv/bin/activate
python src/run_real.py
```

Outputs: `out/real_report.json`, `out/real_fusion_*.png` — camera-vs-accel f1
agreement, measured noise floor, SHS healthy/damaged verdicts. Phone videos are
auto-normalized to constant frame rate if ffmpeg is installed (`brew install ffmpeg`).
