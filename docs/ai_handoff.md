# Python Tracking Sidecar — AI Handoff

Last updated: 2026-09-08
Purpose: the next agent can continue without re-deriving context.

**Read first:** `../AGENTS.md` (rules + the repo boundary), then `architecture.md`, then `roadmap.md`.

---

## 📍 STATUS

| Stage | Status | ADR |
|---|---|---|
| Phase 1 — BlazePose on the OAK VPU (fallback) | **COMPLETE** | P001 |
| Phase 2 — RTMW3D + measured depth (primary) | **COMPLETE**, hardware-verified | P001 |
| **P0-2** — distal caps + legs protected | **COMPLETE**, live-verified | P003 |
| **P1-1** — per-joint temporal tracking + plausibility | **COMPLETE** | P004 |
| **P1-2** — latest-frame queue policy | **COMPLETE**, live human A/B | P005 |
| **P1-4** — skeleton constraints + long-horizon recovery | **REJECTED** — default OFF, retained for forensics only | P008 (superseded) |
| **F-08** — surface-aware depth sampling + advisory `depthQuality` | **COMPLETE**, A/B validated | P010 |
| Next | palm / foot work, or upstream 2D occlusion uncertainty — **NOT STARTED** | — |

**Sidecar half, measured:** camera→UDP **161.8 → 62.0 ms** · frame age **131.5 → 31.4 ms** · limb peak
displacement **−67…−76%** · RGB/depth pairing max error **54.6 → 21.2 ms** · P1-1 cost **0.085 ms**.
**Tests: 37/37 (`python test_joint_tracker.py`).**

Unity-side counterpart (P0 LimbGate, P1-3 pose buffer) is complete — see the Unity repo's
`docs/ai_handoff.md`. Its P1-3 buffer **consumes `seq`/`t` from this repo's datagram**, so those two
fields are now load-bearing and no longer log-only.

---

## The one-paragraph version

This repo turns an OAK-D into stable metric landmarks. Two conditioning layers sit between the model
and the wire: **P0** (magnitude limiting — One-Euro, 0.35 m distal caps, bounded hold) and **P1-1**
(temporal reasoning — per-joint state, seven plausibility signals, TRACKED/WEAK/PREDICTED/LOST). Before
either runs, **P1-2** guarantees the frame being processed is the *newest* one, not a stale queue entry.
Anything the pipeline cannot vouch for leaves as `[0,0,0,0]` — the invalid signal — and **Unity's
LimbGate decides what the avatar does about it.** This repo never invents a position.

---

## What was learned on hardware (do not re-discover)

1. **Confidence is not validity.** A wrist hidden behind the torso held a *wrong* position for
   **840 frames at ~0.63 confidence**; **0 frames** fell below the 0.3 threshold across a 45 s block.
   That is why P1-1's plausibility layer exists — and why the **FROZEN** check is the one that matters
   (a stuck joint has near-zero residual, speed *and* acceleration, so nothing else can see it).
2. **The 131 ms latency was not slowness — it was staleness.** `q.get()` returns the OLDEST packet;
   a full `maxSize=4` queue at 30 fps is exactly 133 ms. Compute never changed (20.85 → 20.70 ms).
3. **Timestamp-matched depth *improved* pairing**, it did not risk it. The old ordinal pairing was
   mis-associating depth by **54.56 ms ≈ 1.6 frames**.
4. **`time.time()` has ~1 ms resolution on Windows** and will quantise any sub-ms measurement into
   fake 0.000/1.010 values. Use `perf_counter()`.
5. **Compute spikes >80 ms happen only at frame index 0–2** — ONNX/DirectML warm-up, per *process*.
   A 3378-frame soak spiked only at frames 0 and 1. Not a steady-state problem.
6. **A plausibility signal must not veto unless provably decisive.** An uncapped neighbour/segment
   check caused **3170 false rejections** of legitimate motion before being capped at 0.5.
7. **A human cannot repeat a performance** closely enough to A/B a filter. Use `stream_motion.py` or
   `--inject-load-ms` when the input must be identical.

---

## Known limits (honest)

- **Sustained high-confidence teleport** longer than the 6-frame prediction window → LOST with slow
  recovery during fast motion. Needs short-gap prediction + blended recovery. **Not built.**
- **Confidence is an un-normalised SimCC peak** (Unity audit F-08); the `0.3` threshold is brittle
  across distance and lighting. P1-1 works around it kinematically rather than fixing the scale.
- **~12.1 ms RGB/depth offset** remains — stable device stereo lag, identical at median/p95/p99.
- **Inference (~21 ms) is why the 30 fps sensor cannot be fully consumed**; effective rate with a
  subject is ~21 fps and ~38.6% of captured frames are deliberately dropped as stale.
- **Facing/depth degrade past ~2.5 m** — a single-front-camera limit.

---

## Files modified in the last session

| File | Change |
|---|---|
| `wholebody_udp_sender.py` | P0-2 caps + legs; P1-1 integration (`--tracker`); P1-2 latest-frame + timestamp-matched depth; freshness + stage-timing metrics; knees/ankles in `sender_log`; `--inject-load-ms` (test-only) |
| `smoothing.py` | P0-2 per-index `max_jump_overrides`; `filter()` now returns `(x, y, z, valid, action, displacement)` |
| `joint_tracker.py` | **NEW** — P1-1 tracker; class style later normalised to bare `class Foo:` |
| `test_joint_tracker.py`, `evaluate_p1.py`, `analyze_capture.py`, `verify_gate.py`, `inject_occlusion.py`, `stream_motion.py`, `compare_p12.py`, `guided_capture.py` | **NEW** — test + analysis tooling |
| `run_p0_acceptance.bat`, `run_p12_ab.bat` | **NEW** — human capture protocols |
| `README.md` | rewritten: 22 scripts, flag table with rationale, `seq`/`t` documented, known limits |
| `AGENTS.md`, `docs/*` | **NEW** — this doc set |
| `.gitignore` | capture artifacts ignored |

**Note:** `pipeline_logs/{sender,recv,model}_log.jsonl` show as modified because they were tracked
before `pipeline_logs/` was added to `.gitignore`. They are regenerable capture output; consider
`git rm --cached` on them.

---

## Next recommended task

**Audit upstream measurement quality (F-08). Produce findings, not code.**

P1-4 was **REJECTED** on 2026-09-08 after live human validation — full evidence in
[`P1_4_CLOSEOUT_2026-09-08.md`](P1_4_CLOSEOUT_2026-09-08.md). The architectural conclusion:

> Skeleton-level kinematic reconstruction cannot currently be made reliable using the present
> landmark geometry, because the bone-length signal's natural variation **overlaps** the corruption
> signal we need to detect. An 0.85 m knee displacement yields `relLenErr` ≈ 1.50 while *clean*
> frames already reach p99 = 1.791. The safe threshold band is empty.

Therefore: **do not** continue threshold tuning, **do not** add more smoothing, and **do not** add
further heuristic rejection layers on the same unstable signal.

The audit must determine *why*:

- high-confidence wrong joints keep their high confidence;
- bone lengths fluctuate so strongly frame to frame;
- depth/landmark geometry can become physically impossible while confidence stays high.

and then say which of these is the useful next step:

```text
better confidence calibration
better RGB/depth temporal correspondence
better depth sampling
joint-specific measurement quality
another upstream signal entirely
```

**Do not implement any of them yet.** Produce the audit first.
