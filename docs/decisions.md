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

## Open questions (no ADR yet — decide deliberately)

- **Confidence normalisation.** Confidence is an un-normalised SimCC peak (Unity audit F-08), so the
  `0.3` threshold is brittle across distance and lighting. P1-1 works around it kinematically rather
  than fixing the scale. Fixing it properly would let the confidence gate carry more weight again.
- **RGB/depth systematic offset.** A stable ~12.1 ms lag remains (device stereo processing), identical
  at median/p95/p99. Correcting it means touching the depth pipeline.
- **Inference cost.** RTMW3D-x at ~21 ms is the single largest stage and the reason the sensor cannot
  be consumed at 30 fps. A smaller model would trade accuracy for headroom — not yet evaluated.
