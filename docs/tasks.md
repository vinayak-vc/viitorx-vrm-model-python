# Python Tracking Sidecar — Tasks

Update whenever work starts or finishes. Small checkboxes an agent can claim.
Scope boundary: sensor → signal → datagram. See `../AGENTS.md` §2.

---

## ✅ F-33 — per-person filter chains (2026-09-17) — BUILT AND MEASURED

Report and ADR live in the Unity repo: `docs/F33_PER_PERSON_FILTERS_2026-09-17.md`, **ADR-071**.
Evidence: `<unity-repo>/docs/evidence/f33/`.

- [x] `person_filters.py` — `PersonFilters` (one chain: P0 → mid-hip → P1-1 → P1-4-optional → F-22)
      and `PersonFilterPool`, keyed on the **track id**. Not on list position: the tracker re-sorts
      most-established-first every frame, so an index-keyed bank hands one person's One-Euro history
      to another.
- [x] Two clocks, deliberately: `last_seen_at` for lifecycle, `last_update_at` for the rate estimate.
      Conflating them released the chain of anyone whose hips went unconfident for 3 s mid-session —
      caught by this module's own test.
- [x] **Each person measures their own sample rate.** One-Euro derives velocity as `delta * freq`;
      told 30 while sampled at 16 it over-estimates speed by 1.9x and opens up when it should damp.
      Median of the last 15 gaps, so one skipped frame cannot move it.
- [x] `smoothing.set_freq()` — purely **additive**, zero deleted lines, never called by the
      single-person sender.
- [x] **ADR-071: the feet (17–22) and head (0–4) joined the filter group.** They were in none. Feet
      fail with `src=0` and need the bounded HOLD; the head fails with `src=1` and needs the
      displacement CAP. Off-switches `--no-filter-feet` / `--no-filter-head` restore the
      single-person grouping exactly, which is how the measurement was taken.
- [x] `tools/diagnostics/f33_filter_bench.py` — ground truth, so **lag** is measured next to jitter.
      Without it "the output moved less" cannot be told from over-smoothing.
- [x] `tests/test_person_filters.py` — **46/46**, covering AGENTS.md §8's required list plus the
      re-ordering trap, the per-person M11 hold, and a config-drift guard that reads
      `wholebody_udp_sender.py`'s SOURCE and fails if a shared default stops matching.
- [x] Fixed: `tools/video/f32_multiperson_video.py` drove `PersonTracker` from the **wall clock**, so
      a file replay was not reproducible and the same clip gave different ids on a faster machine.
      Both it and the filters now run on the video's own timeline. The live sender is unaffected —
      there the wall clock IS the frame clock.

### Measured
- [x] Implausible single-frame steps (>300 mm in 33 ms = 9 m/s, not a person), 2060 person-frames of
      identical input: **2158 → 630 → 77** (no filters → single-person grouping → +feet +head).
      Worst step 2091 mm → 960 mm.
- [x] Ground truth at 16 fps: jitter median 32.4 → **19.3 mm**, lag 250 ms. At 10 fps the naive port
      (rate pinned at 30) is **worse than no filter at all** on the median frame — 35.6 vs 33.5 mm —
      while costing 400 ms of lag.
- [x] Cost **0.88 ms per person-frame** against 20.7 ms for the pose solve (~4%).
- [x] Joints emitted 60.2% → 59.1% — the chain refusing what it does not believe, as `[0,0,0,0]`.
- [x] `wholebody_udp_sender.py` byte-identical; whole suite **270/270**.

### ⚠ Known, and say it alongside any jitter figure
- [ ] **Lag is 233 ms at 30 fps.** Not new — it is the accepted single-person tuning.
- [ ] **A slow confident drift is still followed** (847 mm of an injected 850 mm). P1-4 catches it
      (264 mm) and P1-4 is rejected for production. A test pins the LIMITATION rather than hiding it.
- [ ] **Never run with real people.** Recorded video and synthetic trajectories only.
- [ ] The video A/B's stereo depth is **synthetic** — a depth-less clip gives P0 nothing to smooth and
      starves P1-1 to LOST. Its noise model also assumes 1/8 subpixel while production runs subpixel
      OFF, so it UNDERSTATES the real noise.

---

## ✅ F-32 — multi-person detection and tracking (2026-09-17) — BUILT

Report and ADR in the Unity repo: `docs/F32_MULTIPERSON_2026-09-17.md`, **ADR-070**.

- [x] `assignment.py` — pure-numpy optimal rectangular assignment, verified against brute force on
      200 random matrices. Greedy is suboptimal on **54.6%** of random 4x4 cost matrices and its
      failure mode is exactly an ID swap between crossing people. scipy is not in the venv and is not
      worth ~30 MB for one function.
- [x] `person_tracker.py` — detections → persistent ids, associating in **3-D** using metric depth,
      which no IoU-based tracker can: two people overlapping on screen at different distances are
      trivially separable in Z. **19/19** tests including the crossing case.
- [x] `multiperson_udp_sender.py` — detector on the VPU, tracker on the host, RTMW3D per person,
      everyone in one datagram. A NEW file so the measured single-person stack is not put at risk.
- [x] `tools/video/f32_multiperson_video.py` — the only testable path with nobody in the room.
- [x] Backward compatible: the most-established person is republished at the payload ROOT in the
      single-person shape. Verified `root lm == persons[0].lm` on every packet.

### Measured
- [x] `person-detection-retail-0013` finds a median **7 of 7** dancers; the vendored BlazePose blob
      managed 1–5 (it is a single-prominent-subject detector).
- [x] The detector is **free** on the VPU: rgb 29.8 → 29.7 fps, depth 29.6 → 29.4, detector 11.4 fps.
- [x] RTMW3D **20.7 ms p50, fixed batch of 1** → 2 people 24 fps, 3 → 16, 4 → 12. `--max-poses` = 3.

### Traps this cost a day to find
- [x] **Letterbox, never squash.** Squashing a portrait frame into the detector's landscape input
      measured 0.95 people on a clip containing seven; letterboxing took the SAME detector to 7.06.
- [x] **The NN input must be non-blocking, queue size 1.** A blocking input back-pressures
      `ColorCamera.preview`, which the production RGB output also consumes → RGB collapses to 12.3 fps.
- [x] **Sample depth at chest height, not the box centroid** — the centroid of a standing person
      lands between the legs and reads the floor behind them.

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
| `verify_gate.py` | intended vs observed Unity LimbGate holds — **deleted in `29ec57e`** |
| `inject_occlusion.py` | scripted occlusions, device-free — **deleted in `29ec57e`** |
| `stream_motion.py` | deterministic motion for interpolation A/B |
| `compare_p12.py` | P1-2 A/B, velocity-normalised |
| `guided_capture.py` | guided human capture, blocks A–J — **deleted in `29ec57e`** |
| `run_p0_acceptance.bat`, `run_p12_ab.bat` | one-shot human protocols |

> Three of the scripts above no longer exist: they were deleted in `29ec57e`, before the ADR-065
> reorganisation moved the survivors into `tools/`. The measurements they produced stand; the
> scripts do not. Noted here because a table of tools that mostly do not exist is worse than no
> table, and the same dangling references were found in the Unity repo's `P0_ACCEPTANCE` report.

---

## Gotchas (do not re-discover)

- **`time.time()` has ~1 ms resolution on Windows** — useless for sub-ms work. Use `perf_counter()`.
- **A human cannot repeat a performance** closely enough to A/B a filter. Use `stream_motion.py` or
  `--inject-load-ms` for anything where the input must be identical.
- **`--inject-load-ms` is TEST ONLY.** Never set it in production.
- **Capture output is git-ignored** and regenerable — never commit a log directory.
- **When a test fails, first ask whether the test is wrong.** One "failure" was a fixture comparing a
  value against itself.
- **A file replay must not read the wall clock.** `PersonTracker`'s prediction is scaled by `dt`, so
  a harness timed on the wall clock gives different identities on a faster machine — an A/B whose two
  halves disagreed about which ids existed. Drive replays from the media's own timeline (F-33).
- **A temporal filter belongs to an IDENTITY, not a list position.** Anything keyed on the tracker's
  emitted index is wrong the moment it re-sorts, which it does every frame (F-33).
- **`freq` is not cosmetic.** One-Euro derives velocity as `delta * freq`; a filter told the wrong
  sample rate is worse on jitter AND lag at once, not one traded for the other.

---

## Open / next (NOT started)

- [ ] **TWO REAL PEOPLE IN FRONT OF THE CAMERA.** The top item. Every F-32/F-33 number comes from
      recorded video or synthetic trajectories; nothing about identity through a real occlusion has
      been observed.
- [ ] **Short-gap prediction + blended recovery** — P1-1's known limit (sustained high-confidence
      teleport past the 6-frame window → LOST → slow recovery during fast motion)
- [ ] **Confidence normalisation** — audit F-08; the `0.3` threshold is brittle across distance/lighting
- [ ] **Inference headroom** — RTMW3D-x ~21 ms is why the 30 fps sensor cannot be fully consumed
- [ ] **RGB/depth systematic offset** — stable ~12.1 ms device stereo lag remains

## Blocked

_None._
