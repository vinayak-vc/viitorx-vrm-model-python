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
| **P1-4** | Skeleton constraints + long-horizon recovery | **REJECTED 2026-09-08** — disabled by default; see `docs/P1_4_CLOSEOUT_2026-09-08.md` | P008 (superseded) |
| **F-08** | Surface-aware depth sampling + advisory `depthQuality` | **done**, A/B validated | P010 |
| **F-21** | Single-person target ownership | **CONDITIONAL** — a LOCKED-branch drift route is open and measured; do not ship as a safety property | Unity ADR-061 |
| **F-22** | Elbow/knee biomechanical validation on the emitted geometry | **done**, default on | Unity ADR-062 |
| **F-29** | Trust channel on the wire (`st` / `own` / `lat`) — additive, drives nothing | **done** | Unity ADR-066 |
| **F-32** | **Multi-person**: VPU person detector, optimal 3-D assignment, stable ids, additive wire | **done on recorded video**; never run with real people | Unity ADR-070 |
| **F-33** | **Per-person filter chains** keyed on the track id, with a self-measured sample rate | **done and measured**; same caveat | Unity ADR-071 |

**F-32/F-33 in one line each.** F-32 answered *who is in frame and which of them is which*: a
`person-detection-retail-0013` blob on the otherwise-idle VPU (free — rgb 29.8 → 29.7 fps), optimal
assignment rather than greedy (greedy is suboptimal on **54.6%** of random 4x4 matrices and its
failure mode is exactly an ID swap between crossing people), and association in **3-D**, which no
IoU-based tracker can do. F-33 answered *what each of them looks like*: one complete P0 + P1-1 + F-22
chain per identity, which took implausible single-frame steps from **2158 to 77** across 2060
person-frames for **0.88 ms per person-frame**.

**The correction they forced.** `456.webm` was treated by F-29 as one subject at 2.9 m. It contains
**seven dancers**, and the tracked body's shoulder width ranges 0.068–0.455 m — a chimera. Every
"degradation with distance" figure from that clip measured identity contamination. The single-person
pipeline does not fail loudly on multi-person input; it emits a plausible skeleton belonging to
nobody, and those numbers reached a report.

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
| **Upstream measurement-quality audit (F-08)** | **DO THIS FIRST.** P1-4 proved skeleton-level reconstruction cannot be made reliable on the present landmark geometry: the bone-length signal's natural variation *overlaps* the corruption it must detect. Audit only — no implementation | **this repo** | met |
| ~~Short-gap prediction + blended recovery~~ | Attempted as P1-4 and **REJECTED**. Do not retry on the same unstable signal | **this repo** | blocked on F-08 |
| **Confidence normalisation** | Root cause of "confident-but-wrong" (audit F-08). The `0.3` threshold is brittle across distance/lighting; P1-1 currently works around it kinematically | **this repo** | met, but larger scope |
| **Inference headroom** | RTMW3D-x ~21 ms is why the 30 fps sensor cannot be consumed; a smaller model trades accuracy for latency — and it is now the **binding constraint on how many people can be tracked at once** (F-32: 2 → 24 fps, 3 → 16, 4 → 12), so this candidate is worth more than it was | **this repo** | not evaluated |
| **Two real people in front of the camera** | **The top item for F-32/F-33.** Every multi-person number is from recorded video or synthetic trajectories. Nothing about identity through a real occlusion has been observed | **this repo** | needs two people, ~30 min |
| **Per-person `st` in Unity** | The wire now carries real per-joint tracking states PER PERSON (all `-1` before F-33). The root person's already reach the trust HUD; a crowd HUD needs `PersonPose` to carry them | **Unity** | met |
| **A ReID / appearance model** | Long occlusions still cost an identity. Deferred in ADR-070: a third network on an already-contended budget, and 3-D position is a stronger signal than appearance at this resolution. Revisit only if occlusion proves to be the dominant failure | **this repo** | needs the live session first |
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

## Project status (as of 2026-09-08)

```text
P0 safety                         OK
P1-1 temporal tracker             OK
P1-2 latest-frame freshness       OK
P1-3 Unity pose buffer            OK
P1-4 kinematic recovery           REJECTED
Palm robustness                   NEXT
Foot / ground locking             LATER
Confidence normalization          IMPORTANT
IK                                OFF
```
