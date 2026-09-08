# F-08 MEASUREMENT + CONFIDENCE AUDIT — sidecar pointer — 2026-09-08

The full audit lives in the Unity repo: **`docs/F08_MEASUREMENT_CONFIDENCE_AUDIT_2026-09-08.md`**.
It is kept there because it sits in the same phase-report series as P0_ACCEPTANCE / P1_1 / P1_2 /
P1_3 / P1_4_CLOSEOUT. This note records what it means for **this** repo, which owns every file the
audit implicates.

## Verdict: ROOT CAUSE IDENTIFIED

**(A) No depth-quality signal exists.** `conf` (`rtmw3d_pose.py:131`) is the SimCC **x/y** peak and
excludes `simcc_z` by construction, so it describes the 2D landmark only. The one depth check,
`min_valid = 6 of 25` (`oak_depth.py:86`), passes **98.1–98.5%** of the time because windows are
24.0–24.5/25 valid — yet **15–32% of them straddle a >150 mm depth discontinuity**, and those produce
frame-to-frame depth jumps **10.4× larger at p99** (3945 mm vs 379 mm; up to 30× per joint).

**(B) An occluded limb yields a clean, confident, depth-valid measurement.** Sustained hand-behind-
torso, 952 frames: hidden L-wrist held median confidence **0.575**, was under the 0.3 gate only
**2.10%** of frames, had valid depth **99.89%** of frames, and a **perfectly uniform** window
(spread p50 = 0 mm) — *cleaner* than the visible wrist's 78 mm. The model relocates the wrist onto
the torso; the sampler faithfully measures the torso. Every signal reports healthy.

(A) explains the P1-4 failure. (B) is the more dangerous defect and **is not fixed by (A)**.

## What this repo must not do

- Any bone-length / segment threshold — ADR-P009; natural p99 (1.791) exceeds the fault (1.50).
- Tune the 0.3 `--conf` gate — only 3.0–5.6% of frames sit below it, and the wrong wrist read 0.575.
- Treat `conf` as depth or 3D quality — it is a 2D activation peak.
- Use `measured` / `min_valid` as a quality signal — it passes 98%+.
- Chase the ~12 ms RGB/depth offset — fixed offset, **0.02 ms** spread, ≈16 mm of motion. Not it.
- Add smoothing — the failure is a *wrong* value, not a noisy one.

## Single recommended next change (NOT implemented)

**Make `sample_depth_mm` (`oak_depth.py:67`) surface-aware and have it return a per-joint
depth-quality scalar** — cluster the window's valid pixels, take the body-consistent cluster, and
report a quality value from spread and cluster occupancy. Estimator study shows the headroom is real
and joint-specific: R-hip **8× steadier** with a 9×9 window, L-knee 1.6× with `5x5_min`, while
shoulders/elbows are estimator-invariant (180–280 mm sd — their error is the 2D landmark, not the
sampler).

**Emit the quality value; do not gate on it in the same change.** Validate it against the avatar the
way P1-4 was validated. It does **not** address (B).

## Instrumentation added (DIAG-ONLY, default OFF)

`--audit-log` writes `audit_log.jsonl`: per frame and joint, raw confidence, 2D pixel, depth-validity
flag, full depth-window statistics, plus an 11×11 raw depth crop every `--audit-crop-every` frames.
Cost measured: fps 21.18 vs 21.32 (−0.7%), `trackerMs` p50 0.176 vs 0.175 ms.

Baseline untouched: `--tracker` True, `--latest-frame` True, `--recovery` **False**, P1-1 **37/37**.

## Reproduce

```bash
python audit_f08_offline.py
python audit_f08_capture.py --dir pipeline_logs_f08
python visual_p14_capture.py --pass rollback --audit
```
