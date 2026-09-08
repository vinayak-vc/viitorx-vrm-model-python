# Python Tracking Sidecar — Roadmap

Scope boundary: this repo owns sensor → signal → datagram. Retarget, avatar and the final safety gate
are the Unity repo's. See `../AGENTS.md` §2.

---

## Delivered

| Phase | Focus | Status | ADR |
|---|---|---|---|
| **Phase 1** | BlazePose on the OAK-D VPU → 33 landmarks, GHUM-estimated depth | **done** (fallback path) | P001 |
| **Phase 2** | Host RTMW3D-x whole-body + **measured** per-keypoint OAK depth | **done**, hardware-verified | P001 |
| **P0-2** | 0.35 m distal caps + legs into the protected set | **done**, live-verified | P003 |
| **P1-1** | Per-joint temporal tracking + plausibility | **done**, 6/6 adversarial | P004 |
| **P1-2** | Latest-frame queue policy (frame freshness) | **done**, live A/B | P005 |

**Cumulative measured effect on the sidecar half:** camera→UDP **161.8 → 62.0 ms**; frame age
**131.5 → 31.4 ms**; limb peak displacement **−67…−76%**; RGB/depth pairing max error
**54.6 → 21.2 ms**; tracker cost **0.085 ms**.

*(Unity-side P1-3 — the timestamped pose buffer — consumes `seq`/`t` from this repo's datagram but
required no change here.)*

---

## Next candidates — NOT started

Pick one deliberately. Do not batch; each needs its own measurement and ADR.

| Candidate | Why | Owner repo | Prerequisite |
|---|---|---|---|
| **Short-gap prediction + blended recovery** | P1-1's known limit: a sustained high-confidence teleport past the 6-frame window ends LOST with slow recovery during fast motion | **this repo** | met — P1-1 supplies the per-joint state |
| **Confidence normalisation** | Root cause of "confident-but-wrong" (audit F-08). The `0.3` threshold is brittle across distance/lighting; P1-1 currently works around it kinematically | **this repo** | met, but larger scope |
| **Inference headroom** | RTMW3D-x ~21 ms is why the 30 fps sensor cannot be consumed; a smaller model trades accuracy for latency | **this repo** | not evaluated |
| **RGB/depth systematic offset** | stable ~12.1 ms device stereo lag remains after P1-2 | **this repo** | means touching the depth pipeline |
| Palm / wrist rotation | L-palm rate limiter saturates (median = p95 = 14.98° against a 15°/frame cap) | **Unity** | — |
| Foot / ground constraint | audit F-10, never implemented | **Unity** | — |

---

## Standing rules

- **Do not** add another smoothing stage (ADR-P002). Add memory, not lag.
- **Do not** raise a tuned constant without a fresh measurement and a new ADR.
- **Do not** move safety into this repo (ADR-P007) — Unity's LimbGate is the final gate by design.
- **Do not** change the UDP wire format without a coordinated ADR in **both** repos.
- Any new behaviour ships behind a flag with the safe default and an off-switch for A/B.
