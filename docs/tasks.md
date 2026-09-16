# Python Tracking Sidecar — Tasks

Update whenever work starts or finishes. Small checkboxes an agent can claim.
Scope boundary: sensor → signal → datagram. See `../AGENTS.md` §2.

---

## ✅ Stability program — P0-2 / P1-1 / P1-2 (2026-09-07 → 2026-09-08) — COMPLETE

Driven by the Unity repo's `docs/AUDIT_FBT_2026-09-07.md`. ADRs **P003–P006**.
Every item was accepted on a **measured** result, never on code review.

### P0-2 — displacement caps + legs protected (ADR-P003)
- [x] `--arm-max-jump` / `--leg-max-jump` = 0.35 m for wrist/elbow/knee/ankle, **slewed not dropped**
- [x] Knees/ankles (WB 13–16) added to `limb_idx` → heavier depth cutoff + bounded hold
- [x] `holds_log.jsonl` event stream (RATE_LIMIT / HOLD / DROP), events only — never healthy frames
- [x] Verified on hardware: all 12 tracked joints appear, legs included
- [x] Live result: limb peak displacement **−67…−76%**, trunk *improved* 33–36%

### P1-1 — per-joint temporal tracking (ADR-P004)
- [x] `joint_tracker.py` — `JointTracker` / `SkeletonTracker`, TRACKED/WEAK/PREDICTED/LOST
- [x] Seven plausibility signals incl. **FROZEN** (the one that catches the real failure)
- [x] Causal-only design — no lookahead, so no added latency
- [x] Blended reacquisition, never a teleport
- [x] `test_joint_tracker.py` — **37/37 assertions**, self-contained (no pytest)
- [x] `evaluate_p1.py` — replay (false-rejection) + adversarial; **6/6 cases detected**
- [x] Integrated behind `--tracker` / `--no-tracker`
- [x] Cost **0.085 ms** median for 12 joints (8× under the 1 ms budget)

### P1-2 — frame freshness (ADR-P005)
- [x] Root cause proven arithmetically: FIFO `get()` + `maxSize=4` at 30 fps = **133 ms** (measured 131.4)
- [x] Latest-frame drain + **timestamp-matched depth** selection
- [x] Permanent metrics: `frameAgeMs`, `queueDepth`, `staleDropped`, `rgbDepthSyncMs`
- [x] Per-stage timings: `camLatMs`, `capToPoseMs`, `poseToDepthMs`, `capToSendMs`, `trackerMs`
- [x] Live human A/B: frame age **131.51 → 31.36 ms**, camera→UDP **161.81 → 62.03 ms**
- [x] RGB/depth pairing **improved**: max error 54.63 → 21.24 ms
- [x] 180 s soak — **no freshness accumulation**
- [x] Compute spikes root-caused to **ONNX/DirectML warm-up at frame 0–2**, not drain or GC
- [x] `--inject-load-ms` (TEST ONLY) to reproduce the loaded condition without a human

### Housekeeping
- [x] `perf_counter()` for sub-ms measurement (ADR-P006) — `time.time()` was quantising to 0/1 ms
- [x] Class style normalised to bare `class Foo:` (was `(object)` in `joint_tracker.py`)
- [x] `.gitignore` for capture artifacts (`pipeline_logs*/`, `p12*/`, `probe*/`, `lat_*/`, …)
- [x] `README.md` rewritten — 22 scripts documented, flag table, `seq`/`t` in the wire contract
- [x] `AGENTS.md` + `docs/` set created (this file)

### Tooling added (reusable)
| Script | Purpose |
|---|---|
| `test_joint_tracker.py` | P1-1 unit tests, 37 assertions |
| `evaluate_p1.py` | replay false-rejection + adversarial injection |
| `analyze_capture.py` | acceptance analyzer (displacement, gate, latency, packet loss) |
| `verify_gate.py` | intended vs observed Unity LimbGate holds |
| `inject_occlusion.py` | scripted occlusions, device-free, drives Unity directly |
| `stream_motion.py` | deterministic motion for interpolation A/B |
| `compare_p12.py` | P1-2 A/B, velocity-normalised |
| `guided_capture.py` | guided human capture, blocks A–J |
| `run_p0_acceptance.bat`, `run_p12_ab.bat` | one-shot human protocols |

---

## Gotchas (do not re-discover)

- **`time.time()` has ~1 ms resolution on Windows** — useless for sub-ms work. Use `perf_counter()`.
- **A human cannot repeat a performance** closely enough to A/B a filter. Use `stream_motion.py` or
  `--inject-load-ms` for anything where the input must be identical.
- **`--inject-load-ms` is TEST ONLY.** Never set it in production.
- **Capture output is git-ignored** and regenerable — never commit a log directory.
- **When a test fails, first ask whether the test is wrong.** One "failure" was a fixture comparing a
  value against itself.

---

## Open / next (NOT started)

- [ ] **Short-gap prediction + blended recovery** — P1-1's known limit (sustained high-confidence
      teleport past the 6-frame window → LOST → slow recovery during fast motion)
- [ ] **Confidence normalisation** — audit F-08; the `0.3` threshold is brittle across distance/lighting
- [ ] **Inference headroom** — RTMW3D-x ~21 ms is why the 30 fps sensor cannot be fully consumed
- [ ] **RGB/depth systematic offset** — stable ~12.1 ms device stereo lag remains

## Blocked

_None._
