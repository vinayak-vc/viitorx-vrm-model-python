# Python Tracking Sidecar — Architecture Decision Records

Add a new ADR for every tuned constant, filter, queue policy or wire-format change.
Do not silently contradict an Accepted ADR.

**An ADR here must record: the measurement, the honest limit, and an explicit DO-NOT list.**
A decision without a number is an opinion.

IDs are `ADR-Pnnn` (P = Python) so they never collide with the Unity repo's `ADR-nnn`.
Cross-repo decisions are recorded in **both** files.

---

## ADR-P001 — Sidecar process, not a Unity native plugin

- **Status:** Accepted. Cross-repo (Unity ADR-016, "Option B2").
- **Context:** DepthAI and GPU inference are the least stable parts of the system. A native `abort()`
  inside the editor cannot be caught by managed `try/catch` — it takes Unity with it. This happened
  repeatedly during early OAK-D bring-up.
- **Decision:** run all model + device code in a separate Python process and stream results over local
  UDP.
- **Measured cost:** UDP send → Unity receive **0.30 ms median / 1.30 ms p95**; **0.00% packet loss**
  and 0 out-of-order over 55 000+ packets on loopback.
- **DO NOT:** move inference back into Unity to "avoid the copy" — the copy costs 0.3 ms and buys
  process isolation.

---

## ADR-P002 — Single smoothing owner: the sidecar

- **Status:** Accepted. Cross-repo (Unity ADR-020).
- **Context:** both sides had filters; double-filtering added lag without adding stability.
- **Decision:** the sidecar owns temporal smoothing. Unity's `JointFilterPipeline` is **bypassed** on
  the OAK path.
- **DO NOT:** add a filter on either side "just to be safe". More filtering buys stillness by spending
  responsiveness. **Add memory, not lag.**

---

## ADR-P003 — P0-2: tighter distal-limb displacement caps + legs into the protected set

- **Status:** Accepted 2026-09-07. Unity counterpart ADR-028. Evidence: Unity repo
  `docs/AUDIT_FBT_2026-09-07.md` (F-03/F-04), `docs/P0_ACCEPTANCE_2026-09-07.md`.
- **Context:** the global `--max-jump 1.5 m` sat **above** the observed 0.5–0.9 m limb spikes, so they
  passed untouched. Separately, knees/ankles (WB 13–16) were deliberately excluded from depth-smoothing
  and hold, so a bad knee got only light image-plane filtering.
- **Decision:**
  1. Per-index displacement caps: `--arm-max-jump` / `--leg-max-jump` = **0.35 m** for
     wrists/elbows/knees/ankles. Trunk + hands keep the global 1.5 m.
  2. Knees/ankles added to `limb_idx`, gaining the heavier depth cutoff and the bounded hold.
  3. Over-cap samples are **rate-limited (slewed)**, never dropped.
- **Why 0.35:** peak *legitimate* fast-arm displacement measured **0.0746 m** (4.7× headroom) while
  pre-P0 spikes reached 0.5–0.9 m. The cap sits between them, above the depth-quantisation step.
- **Measured result (live human):** wrist/elbow peak displacement **−67…−76%**; trunk *improved* 33–36%.
- **Why slew, not drop:** a hard reject freezes a joint permanently when the condition persists every
  frame. That regression happened before with a 0.5 m hard gate.
- **DO NOT:** raise the cap without a fresh measurement of legitimate peak motion; do not apply the
  distal cap to trunk or hands (different motion envelopes).

---

## ADR-P004 — P1-1: per-joint temporal tracking + plausibility

- **Status:** Accepted 2026-09-08. Unity counterpart ADR-029. Evidence: Unity repo
  `docs/P1_1_TRACKER_2026-09-08.md`.
- **Context:** confidence is not validity. Measured on hardware: a wrist hidden behind the torso held a
  **wrong position for 840 frames at ~0.63 confidence** — **0 frames** fell below the 0.3 threshold
  across a 45 s block. No confidence gate can fire on that.
- **Decision:** one reusable `JointTracker` per joint (TRACKED / WEAK / PREDICTED / LOST) with seven
  plausibility signals, placed **after** the P0 smoother and **before** the message build.
- **The decisive signal is FROZEN** — a joint pinned within 4 mm for ≥12 frames while its parent moved
  ≥5 cm. A frozen joint has near-zero residual, speed *and* acceleration, so no conventional check sees
  it. This is the failure that was actually observed.
- **Causal only:** an N−2…N+2 window needs lookahead, and lookahead is latency. Suspicious samples are
  down-weighted; consecutive agreeing samples promote back to TRACKED, distinguishing real fast motion
  from a spike at zero added cost.
- **Measured:** 37/37 unit assertions; **6/6 adversarial cases detected** (0.8 m error at confidence
  0.95 leaves a **2 mm** trace; frozen-wrist recovers with **0.0000 m** residual); legitimate dancing
  median/p95 **unchanged**; cost **0.085 ms** median for 12 joints (8× under budget).
- **Two calibration lessons, both measured:**
  - the neighbour/segment check must be **corroborating evidence capped at 0.5**, never a veto —
    uncapped it caused **3170 false rejections** of legitimate motion;
  - the output step limiter must budget from **measurement-to-measurement**, not from `lastValid` —
    during WEAK, `lastValid` holds the blended output, which inflates the budget by exactly the offset
    the limiter is meant to bound, so it never binds.
- **HONEST LIMIT:** a sustained high-confidence teleport longer than the 6-frame prediction window ends
  in LOST with slow recovery during fast motion.
- **DO NOT:** raise prediction horizons (bounded 6 frames / 0.30 m by design); add smoothing here;
  emit a predicted position once the joint is LOST.

---

## ADR-P005 — P1-2: latest-frame queue policy (freshness over completeness)

- **Status:** Accepted 2026-09-08. Unity counterpart ADR-030. Evidence: Unity repo
  `docs/P1_2_FRESHNESS_2026-09-08.md`.
- **Context:** measured camera→host latency of **131 ms**. `DataOutputQueue.get()` returns the
  **OLDEST** packet; with inference (~21 ms) slower than the 30 fps sensor the host queue sat full at
  `maxSize=4` → **4 × 33.3 = 133 ms** of pure staleness. Nothing was slow — the system was working on
  old frames.
- **Decision:** one blocking `get()` (liveness), then `tryGetAll()` keeping only the **newest** packet.
  Stale intermediates are **discarded, not processed**. Queue size unchanged. Toggle
  `--latest-frame` / `--no-latest-frame`.
- **RGB/depth pairing:** depth is chosen by **closest timestamp** to the selected RGB frame — never
  newest-RGB + oldest-depth. This **improved** pairing, which had been mis-associating depth by
  **54.56 ms ≈ 1.6 frames**: max sync error 54.63 → **21.24 ms**.
- **Measured (live human A/B):** frame age **131.51 → 31.36 ms**; camera→UDP **161.81 → 62.03 ms**;
  fps (21.35 vs 21.38) and compute (20.85 vs 20.70 ms) **unchanged** — proving the win was queue wait,
  not processing. 180 s soak: **no accumulation** (31.11 → 31.52 ms median).
- **Cost:** under load **38.6%** of captured frames are deliberately dropped — frames the estimator
  could never have consumed in time. Previously they were consumed *late* instead.
- **DO NOT:** enlarge the queue (that trades latency for frames nobody sees); set `--inject-load-ms` in
  production (test-only); assume device-side `XLinkOut` queues need changing — measured as
  non-contributing (unloaded frame age lands at 14.08 ms = pure sensor→host transfer).

---

## ADR-P006 — `time.perf_counter()` for sub-millisecond measurement

- **Status:** Accepted 2026-09-08.
- **Context:** the P1-1 tracker cost first measured as "0.000 ms median / 1.010 ms p95" in-pipeline —
  physically implausible values that were **`time.time()`'s ~1 ms Windows resolution quantising a
  sub-millisecond signal**, not data.
- **Decision:** use `time.perf_counter()` for anything under ~10 ms. `time.time()` remains fine for
  epoch timestamps on the wire (where absolute time, not resolution, is what matters).
- **Result after the fix:** tracker cost **0.085 ms median / 0.121 ms p99** on real dancing data.
- **DO NOT:** report a sub-millisecond number measured with `time.time()`.

---

## ADR-P007 — Invalid is expressed as invalid, never as a position

- **Status:** Accepted. Cross-repo invariant (Unity ADR-028 / ADR-031).
- **Context:** the sidecar emits a sub-threshold joint as `[0,0,0,0]`. That zero is an **invalid
  signal**, not a coordinate. Treating it as a position places the joint at the hip origin, which is
  exactly the F-01 limb-collapse bug.
- **Decision:** this repo never fabricates a position to keep a joint alive. A `LOST` joint has its
  **emit confidence zeroed**, producing the same `[0,0,0,0]` a real occlusion produces, and Unity's
  LimbGate decides what the avatar does.
- **DO NOT:** move safety into this repo; emit a "best guess" for a LOST joint; interpolate toward a
  zero-filled landmark anywhere in either repo.

---

## ADR-P008 — P1-4: skeleton constraints + long-horizon recovery; bone length is NOT a veto on this pipeline

> **SUPERSEDED 2026-09-08 by ADR-P009 — P1-4 is REJECTED.** Live human validation showed P1-4 MISSED
> the controlled 0.85 m knee corruption entirely (`GEOMETRIC_REJECT = 0`, `RECONSTRUCT = 0`) while
> raising P0 limb holds 0.47% → 17.83% and LOST episodes 1 → 122. The −68% replay win did NOT
> reproduce on hardware. Read the section below as the historical record of a rejected approach,
> not as guidance. See `docs/P1_4_CLOSEOUT_2026-09-08.md`.

- **Status:** Accepted 2026-09-08 (CONDITIONAL — see the limits). Unity counterpart: none required.
- **Context:** P1-1 reasons about one joint at a time, so it cannot catch a joint that is
  *confidently wrong and stays wrong* past its 6-frame prediction horizon — the hip-correct /
  ankle-correct / knee-wrong case. Only skeleton-level evidence can.
- **Decision:** new `kinematic_recovery.py` running AFTER P1-1 and BEFORE the message build:
  robust per-bone length estimator (median + MAD over a bounded window, trusted samples only),
  a two-anchor circle-intersection solver for a chain middle joint, scale-free interior-angle
  limits, a `GEOMETRICALLY_SUSPECT` observation computed **without reference to model
  confidence**, staged recovery (P1-1 predict → P1-4 reconstruct → LOST), and a bounded
  recovery blend. `RECOVERING` added to `TrackingState` (additive; P1-1 never enters it).

### THE MEASUREMENT THAT CHANGED THE DESIGN

Bone-length constancy is the obvious skeleton prior. **On this pipeline it is not usable as a
rejection criterion.** Measured over 853 frames of real human capture, relative deviation of each
bone from its OWN median:

| statistic | value |
|---|---|
| median | 0.101 |
| p95 | **0.589** |
| p99 | **1.791** |
| max | 3.528 |
| worst bone (R-shoulder→R-elbow) p95 | **1.514** |

A "bone" in this pipeline routinely varies 50–150% on perfectly healthy frames. An initial
`seg_reject = 0.45`, set by reasoning rather than measurement, therefore fired on **25.3% of clean
joint-frames**, drove TRACKED from 95.7% down to **67.6%**, pushed 11.4% of joints to LOST and moved
joints by up to 0.56 m on healthy data. It was a false-positive machine.

**Corrected:** `seg_reject = 1.80` (above the measured p99 of natural variation) →
false positives **25.3% → 1.39%**, joint-frames modified **14.97% → 0.54%**.

By contrast the **interior angle** is scale-free and clean: measured min 30.5°, max 179.3° over
3409 real chain-samples, so limits of [20°, 195°] produce **zero** false positives. **The angle
check carries the real detection load; bone length is corroborating evidence only.**

### Second measurement: reconstruction needs a trustworthy prior

Reconstructing a 20-frame knee gap from the learned lengths was **worse than holding**
(max error 0.170 m → 0.327 m). Cause: the hip-knee estimator accepted only **45 of 818** samples
(5.5%) — an estimate built on an unrepresentative 5% does not describe a real limb. Added a
`len_min_accept_ratio = 0.30` gate: a bone whose samples are mostly rejected may score plausibility
but **may not drive reconstruction**. The regression disappeared.

### Measured result (real capture, 853 frames, ground truth = raw un-injected measurement)

| adversarial case | P1-1 only | P1-1+P1-4 | verdict |
|---|---|---|---|
| slow impossible knee drift, 25 frames | 0.6265 m | **0.1993 m** | **BETTER (−68%)** |
| hi-conf knee teleport, 10 frames | 0.5746 m | 0.5701 m | same |
| missing knee, 5 / 20 frames | 0.1702 m | 0.1702 m | same |
| frozen wrist, 25 frames | 0.6622 m | 0.6617 m | same |
| bad depth on elbow, 10 frames | 0.8707 m | 0.9056 m | same |

**P1-4 is narrow: it delivers one large win on exactly the case it was specified for, and regresses
nothing.** The "same" rows are cases P1-1 already handles, where P1-4 correctly does not engage.

Cost: **0.093 ms median / 0.144 ms p99** (P1-4 alone); combined P1-1+P1-4 on hardware **0.219 ms
median / 0.303 ms p99**. `host→UDP` unchanged (29.81 vs 30.3 ms).

### HONEST LIMITS

- **Reconstruction is effectively dormant on real captures** because no bone estimator currently
  reaches the 30% acceptance ratio. P1-4's value today is the *rejection* path. Reconstruction is
  proven by deterministic tests but has not earned its place on live data.
- A rejected joint that cannot be reconstructed becomes **LOST** (1.0% vs P1-1's 0.0%), which hands
  it to P0's LimbGate to hold. Deliberately conservative, per the brief.
- **Not visually validated on the avatar** (P1-4 Part 21) — that needs a human in front of the camera.

### DO NOT

- Re-tighten `seg_reject` toward 0.45 without new evidence — that number is measured, not chosen.
- Use bone length as a veto on this pipeline; use the angle check.
- Let a bone estimator with a low acceptance ratio drive reconstruction.
- Treat P1-4 as a smoothing stage — a healthy joint passes through byte-identical.

---

## Open questions (no ADR yet — decide deliberately)

- **Confidence normalisation.** Confidence is an un-normalised SimCC peak (Unity audit F-08), so the
  `0.3` threshold is brittle across distance and lighting. P1-1 works around it kinematically rather
  than fixing the scale. Fixing it properly would let the confidence gate carry more weight again.
- **RGB/depth systematic offset.** A stable ~12.1 ms lag remains (device stereo processing), identical
  at median/p95/p99. Correcting it means touching the depth pipeline.
- **Inference cost.** RTMW3D-x at ~21 ms is the single largest stage and the reason the sensor cannot
  be consumed at 30 fps. A smaller model would trade accuracy for headroom — not yet evaluated.

---

## ADR-P009 — P1-4 REJECTED: the bone-length signal overlaps its own noise floor

- **Status:** Accepted 2026-09-08. Supersedes ADR-P008. Unity counterpart: ADR-032.
- **Context:** ADR-P008 accepted P1-4 conditionally on **offline replay** evidence. Live validation with
  a human subject and the real OAK-D contradicted that verdict on every axis that matters.
- **Decision:** **P1-4 is rejected for production.** `--recovery` now defaults to **False**;
  `kinematic_recovery.py` is retained on disk, marked `REJECTED / EXPERIMENTAL / NOT FOR SHIPPING`,
  and cannot execute unless explicitly opted in. No algorithm or constant was changed — this is a
  rollback, not another tuning round.

### The measurement that decided it

The controlled block-8 injection (0.85 m right-knee drift and teleport, high confidence, upstream of
P1-1 and P1-4) is the only fully controlled comparison in the run:

| pass | error reaching the wire | `GEOMETRIC_REJECT` | `RECONSTRUCT` |
|---|---|---|---|
| P1-3 baseline (drift / teleport) | 0.8388 / 0.8542 m | 0 | 0 |
| **P1-4 (drift / teleport)** | **0.8419 / 0.8439 m** | **0** | **0** |

**P1-4 never fired on the case it exists for.** The mechanism is arithmetic, not tuning:

```text
corruption to detect (0.85 m knee)  relLenErr ~ 1.50
natural variation, clean capture    p99       = 1.791
shipping threshold                            = 1.80
```

The corruption produces a *smaller* bone-length error than clean human motion already produces. The
safe threshold band is **empty**. P0-2's 0.35 m/frame leg cap compounds it by slewing a teleport so the
rolling-median estimator **learns** the corrupted length.

### Collateral damage measured on the avatar

| metric | P1-3 | P1-4 |
|---|---|---|
| frames with a limb HELD by the P0 gate | 0.47% | **17.83%** |
| LOST episodes | 1 | **122** (longest 0.607 s) |
| snaps > 45° per pose | 15 | **27** |
| reconstruction displacement, left_knee | — | mean **2.19 m**, max **4.16 m** |

Within-pass control (same performance, so free of the "danced harder" confound): frames within 250 ms
of a P1-4 event were **13×** more likely to exceed 20° of rotation (1.248% vs 0.095%).
87% of rejections came from the bone-length check (`segment` 1001 / `angle` 128).

CPU (0.329 vs 0.173 ms) and latency (42.2 vs 39.1 ms) were both fine. **P1-4 was rejected for
correctness, not cost.**

### Consequences

- Production path is `P1-1 → P1-2 → UDP → P1-3 → Kalidokit → P0 LimbGate → VRM`.
- Rollback verified live: P0 limb-held **0.14%**, LOST **1**, tracker CPU **0.175 ms**, 0 packet loss,
  P1-1 37/37, P1-4 39/39, Unity 47/47, 0 compile errors.
- **Skeleton-level reconstruction cannot be made reliable on the present landmark geometry.** The next
  work is an upstream measurement-quality audit (F-08), not another downstream heuristic.

### DO NOT

- Tune `seg_reject`, `angle_min_deg`, `len_min_accept_ratio` or any other P1-4 constant.
- Add another heuristic rejection layer on the bone-length signal.
- Re-enable `--recovery` in production, or ship a launcher that passes it.
- Treat ADR-P008's −68% replay result as valid — it did not reproduce on hardware.

---

## ADR-P010 — F-08: surface-aware depth sampling; `depthQuality` is advisory

- **Status:** Accepted 2026-09-08 (CONDITIONAL — see the open item). Report:
  `docs/F08_SURFACE_AWARE_DEPTH_IMPLEMENTATION_2026-09-08.md`.
- **Context:** ADR-P009 closed P1-4 and the F-08 audit found why: the pipeline carries a quality
  signal for the 2D landmark only. The single depth check counted valid pixels (passing 98%+) while
  15–32% of windows straddled a >150 mm discontinuity, producing 10.4× larger depth jumps at p99.
- **Decision:** `sample_depth_surface` splits the sampling window into depth surfaces (gap > 100 mm)
  and takes **the surface nearest the keypoint pixel** — not the nearest depth, not the largest
  cluster. It returns `depthQuality` in [0,1] and per-window diagnostics. `sample_depth_mm` stays as
  a wrapper; `backproject` keeps its 2-tuple default return.
  **`depthQuality` is ADVISORY: nothing gates, suppresses or reweights on it.**

### Measured

- **94.07%** of 22,745 real joint-windows produce a **byte-identical** depth; a single-surface window
  is provably unchanged (400/400 vs the legacy sampler). Delta p95 128.8 mm, p99 236.0 mm.
- Quality separates clean from contaminated windows **1.49×** (0.944 vs 0.632), 90.7% accuracy at a
  single threshold. Reported as evidence; **not** used as an operating point.
- **3× FASTER than the sampler it replaces** — 2.813 vs 8.430 ms for 133 keypoints. `np.percentile`
  re-sorts and has large fixed overhead on 25-element windows.
- Live: camera→apply p50 **40.8 → 37.1 ms**, fps 21.32 → 21.34, rotation p95 **17.66 → 14.85°**,
  p99 **36.28 → 30.05°**, packet loss 0.
- Tests **32/32**; P1-1 37/37, P1-4 39/39, Unity 47/47 unchanged.

### The negative result, recorded deliberately

A wrist physically hidden behind the torso scores `depthQuality` **0.881 — high** — with depth valid
100% of frames. **Depth quality cannot see a hallucinated limb**, because the measurement itself is
excellent; it is measuring the wrong object. The remaining defect is upstream 2D semantic/occlusion
uncertainty and no depth-side signal reaches it.

### Open item — CLOSED

The initially unattributable gate-hold/LOST delta was closed by a controlled A/B: two back-to-back
live runs, identical block wording, sampler as the only difference, in-block frames only, input
motion matched to **1.2%**.

| metric | legacy | surface-aware |
|---|---|---|
| P0 gate held | 0.58% | **0.45%** |
| LOST | 3 | **0** |
| snaps > 45° | 18 | **15** |
| snaps > 20° | 86 | 105 |
| camera→apply p50 | 45.2 ms | **39.6 ms** |
| fps | 20.18 | **21.25** |

The earlier apparent regression was a reworded block-2 instruction plus prep-gap frames where the
operator is repositioning. **Walk — the block that produced it — is now 0.00% held and 0 LOST in
both passes.** Per block, holds follow input motion, not the sampler: fast-legs had +32% motion under
surface-aware and gained holds; dance had −23% motion and lost them.

Residual: mid-range rotation tail (p95 +5%, >20° +22%) against *fewer* large snaps (>45° −17%) with
1.2% more motion — mixed in direction, within run-to-run variation, claimed neither way.

**Status: ACCEPTED. `--surface-depth` is the shipping default.**

### DO NOT

- Gate, suppress or reweight a joint on `depthQuality` until that A/B exists.
- Treat `depthQuality` as a probability — it is a geometry score, uncalibrated by design.
- Expect it to solve the hallucinated-limb case; it provably does not.
- Re-tune `SURFACE_GAP_MM` without evidence; it is reasoned from limb thickness vs background
  separation and validated by the 94% identical rate.
