# Python Tracking Sidecar — Architecture

Agent entry summary. Run instructions and the wire contract live in `../README.md`; the reasoning
behind each tuned value lives in `decisions.md`.

**There are TWO senders.** `wholebody_udp_sender.py` is the single-person production path and is what
the rest of this file describes. `multiperson_udp_sender.py` (F-32) is a separate program that tracks
several people at once; it imports the first sender's building blocks so there is one definition of
the landmark contract, and leaves it byte-identical. See *Multi-person* below.

---

## Stages

```
┌─ DEVICE (OAK-D-PRO-W, OV9782) ──────────────────────────────────────┐
│ ColorCamera 800p → ISP 1/2 → 640x400 BGR ──► XLinkOut "rgb"         │
│ MonoL + MonoR 400p → StereoDepth(HIGH_DENSITY, LR-check,            │
│                      setDepthAlign(CAM_A)) ──► XLinkOut "depth"     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ XLink
┌─ HOST ────────────────────────▼──────────────────────────────────────┐
│ 1. QUEUE POLICY      q.get() + tryGetAll() → keep NEWEST      P1-2   │
│                      depth chosen by CLOSEST TIMESTAMP               │
│ 2. INFERENCE         RTMW3D-x ONNX (DirectML) → uv, zrel, conf       │
│                      + person-box tracking with wedge recovery       │
│ 3. DEPTH FUSION      5x5 window, 30th percentile → back-project      │
│                      → xyz_cam (metres), measured[] flags            │
│ 4. P0 SMOOTHER       One-Euro per coord + per-index displacement     │
│                      caps + bounded hold-on-dropout           P0     │
│ 5. P1-1 TRACKER      TRACKED / WEAK / PREDICTED / LOST         P1-1  │
│                      + 7 plausibility signals                        │
│ 6. BUILD + SEND      hip-centre → 33-slot JointId array → UDP        │
└──────────────────────────────────────────────────────────────────────┘
                               │ UDP JSON 127.0.0.1:8899
                               ▼   { lm, lh?, rh?, xyz, src, seq, t }
                    Unity: PoseBuffer → Kalidokit → P0 LimbGate → VRM
```

---

## Module map

| Module | Owns | Key entry points |
|---|---|---|
| `oak_depth.py` | device pipeline, intrinsics, depth sampling, back-projection | `build_rgbd_pipeline`, `read_rgb_intrinsics`, `sample_depth_mm`, `backproject` |
| `rtmw3d_pose.py` | ONNX session, SimCC decode, person-box tracking | `RTMW3D.infer`, `bbox_from_keypoints`, `center_bbox` |
| `smoothing.py` | P0 signal conditioning | `OneEuro`, `KeypointSmoother.filter` |
| `joint_tracker.py` | P1-1 temporal state + plausibility | `JointTracker.update`, `SkeletonTracker.update` |
| `wholebody_udp_sender.py` | the frame loop, queue policy, message build, diagnostics | `main` |
| `pose_validation.py` | F-22 elbow/knee biomechanics on the emitted geometry | `PoseValidator.update` |
| `target_ownership.py` | F-21 single-person identity lock (single-person path only) | `TargetOwnership.update` |
| `kinematic_recovery.py` | P1-4 skeleton constraints — **rejected for production**, off by default | `KinematicRecovery.apply` |
| `assignment.py` | F-32 optimal rectangular assignment, pure numpy | `solve` |
| `person_tracker.py` | F-32 detections → persistent person ids, associating in **3-D** | `PersonTracker.update` |
| `person_filters.py` | F-33 one complete P0+P1-1+F-22 chain **per person**, pooled on track id | `PersonFilters`, `PersonFilterPool` |
| `multiperson_udp_sender.py` | the multi-person frame loop and payload build | `main`, `build_person_payload` |

Dependency direction is strictly **downward**: `wholebody_udp_sender` imports the others; none of them
import it, and `joint_tracker.py` imports **nothing but `math`** so it stays unit-testable in isolation.
`multiperson_udp_sender` sits one level above and imports `wholebody_udp_sender` for its builders —
the only upward edge in the repo, and deliberate: it is what keeps a single definition of a landmark.

---

## Coordinate spaces (get these wrong and nothing works)

| Space | Axes | Units | Where |
|---|---|---|---|
| Image | u right, v down | px | `uv` from RTMW3D |
| Depth map | aligned to RGB | **mm** | `sample_depth_mm` |
| Camera | X right, **Y down**, Z forward | **m** | after `backproject` |
| Wire | camera axes, **hip-relative** | **m** (except `xyz` = **mm**) | the datagram |

`xyz` (the mid-hip root) is the one field in millimetres — that asymmetry is historical and load-bearing
for the Unity consumer. **Always state units in comments.**

---

## The two conditioning layers

```
raw measurement
   │
   ├─ P0  ── One-Euro (velocity-adaptive)
   │        ── per-index displacement cap (0.35 m distal / 1.5 m trunk+hands), SLEWED not dropped
   │        ── bounded hold-on-dropout (<= 8 frames, limb set incl. legs)
   │
   └─ P1-1 ── per-joint state: position, velocity, acceleration, timestamps, invalid/predicted counts
            ── 7 plausibility signals → suspicion 0..1
            ── TRACKED (use) / WEAK (down-weight) / PREDICTED (extrapolate, bounded) / LOST (invalid)
            ── blended reacquisition, never a teleport
```

**They are not redundant.** P0 is stateless-per-frame magnitude limiting; P1-1 is temporal reasoning.
P0 cannot see a joint that is *smoothly wrong*; P1-1 can (the FROZEN check). P1-1 alone would let a
single huge spike through before it had history.

### The seven plausibility signals (P1-1)

confidence · residual vs kinematic prediction (adaptive per-joint scale) · implied speed · implied
acceleration · depth consistency · **FROZEN** (pinned while the parent moves) · segment length vs a
running median.

**FROZEN is the one that matters** — a stuck joint has near-zero residual, speed *and* acceleration, so
nothing else can detect it. It is the failure that was actually observed on hardware.

---

## Causality constraint

The tracker is **causal only**. Comparing frames N−2…N+2 would need lookahead, and lookahead *is*
latency. Instead a suspicious sample is down-weighted, and consecutive suspicious samples that agree
with each other promote back to TRACKED — so `A→B→C→D→E` (real fast motion) survives while
`A→B→X→B` (a spike) is damped, at zero added latency.

---

## Threading

**Single-threaded by design.** One frame loop; no worker threads, no locks. The queue policy (§P1-2)
means the loop never falls behind — it drops instead. Adding threads here would buy nothing and add a
class of bug the current design cannot have.

---

## Failure behaviour

| Failure | Behaviour |
|---|---|
| depth hole at a keypoint | `measured=False` → hip-plane / zrel fallback at halved confidence |
| keypoint below `--conf` | emitted as `[0,0,0,0]` — the invalid signal |
| limb dropout ≤ 8 frames | P0 holds last good value, reports it as measured |
| joint implausible / missing | P1-1 → PREDICTED (≤ 6 frames, ≤ 0.30 m) → LOST |
| joint LOST | emit confidence zeroed → `[0,0,0,0]` → **Unity's LimbGate holds the limb** |
| person lost | person-box re-acquires from frame centre after a 20-frame wedge guard |
| host slower than sensor | stale frames **discarded**, newest processed |

Note the invariant: **this repo never invents a position to keep a joint alive.** Invalid is expressed
as invalid, and the Unity gate decides what the avatar does about it.

---

## Multi-person (F-32 + F-33)

A second program, not a mode. RTMW3D-x is a single-person top-down model — one crop in, one skeleton
out — so N people cost N sequential inferences and the whole design follows from that one fact.

```
┌─ DEVICE ────────────────────────────────────────────────────────────┐
│ the production RGB-D pipeline, UNCHANGED                            │
│   + ImageManip 544x320 → MobileNetDetectionNetwork                  │
│     (person-detection-retail-0013)     ──► XLinkOut "det"           │
│     NON-BLOCKING, queue size 1 — a blocking NN input back-pressures  │
│     ColorCamera.preview and collapses the RGB stream to 12.3 fps     │
└──────────────────────────────┬──────────────────────────────────────┘
┌─ HOST ────────────────────────▼──────────────────────────────────────┐
│ 1. DETECTIONS      letterboxed, never squashed; depth sampled at     │
│                    CHEST height, not the box centroid (which lands   │
│                    between the legs and reads the floor behind)      │
│ 2. PersonTracker   3-D association (image plane + metric depth),     │
│                    optimal assignment, stable ids never reused F-32  │
│ 3. per person, capped by --max-poses (default 3):                    │
│       RTMW3D crop → depth fuse                                       │
│       → PersonFilters[track.id]:  P0 → mid-hip → P1-1 → F-22   F-33  │
│       → build_body_landmarks                                         │
│ 4. BUILD + SEND    { persons[], n, ndet, ntrack, ... } + the most-   │
│                    established person REPUBLISHED at the root in the │
│                    single-person shape                               │
└──────────────────────────────────────────────────────────────────────┘
```

**Why the root duplication is the whole compatibility story.** Every existing consumer — the Unity
provider, the VRM mirror app, all nine experience scenes — reads the root fields and is unaffected.
Verified: `root lm == persons[0].lm` on every packet, and the same person across 752 Unity frames.
Cost is one duplicated person, ~6 KB of a ~10 KB payload; `--no-legacy-primary` drops it.

### Why the filters are pooled on the track id

Every stage of the chain is **temporal**: One-Euro carries a velocity estimate, the displacement cap
carries the last accepted position, the hold carries a countdown, P1-1 carries a state machine, F-22
carries a bend history. `PersonTracker.update()` returns people most-established-first, so a person's
INDEX changes whenever somebody else gains a hit — an index-keyed bank hands person A's history to
person B, every joint teleports, and the cap then slews the two bodies into each other. Ids are
stable and never reused. Unity's `TrackedStage.UpdateCrowd` keys `SkeletonPose` the same way.

Lifecycle is **idle-based** (3 s after a person stops being seen), not driven by tracker events: a
caller that forgets to drain events would leak slowly over an evening, and a clock cannot be
forgotten. Two clocks are kept — `last_seen_at` for lifecycle, `last_update_at` for the rate
estimate — because a frame whose pose failed is still a person in the room but is not a sample.

### Why each person measures their own sample rate

`KeypointSmoother` takes `freq` once, and the single-person sender leaves it at 30 — true there.
Three people cost three 20.7 ms solves, so this loop runs at ~16 fps, and a person below
`--max-poses` is updated rarer still. One-Euro derives velocity as `delta * freq`, so a filter told
30 while sampled at 16 over-estimates speed by 1.9x, inflates its adaptive cutoff and OPENS UP
exactly when it should damp. Measured at 10 fps, the naive port is **worse than not filtering at
all** on the median frame (35.6 mm vs 33.5) while costing 400 ms of lag. Each person therefore takes
the median of their last 15 gaps and retunes. `smoothing.set_freq()` exists for this and is additive.

### The one place the chain differs from single-person

ADR-071: the **feet** (WholeBody 17-22) and **head** (0-4) are in the filter group. The single-person
limb set contains neither, so both got the light image-plane One-Euro and nothing else. They failed
for opposite reasons — feet with `src=0` (no depth, rebuilt from raw monocular z; the HOLD fixes it),
head with `src=1` (depth sampled from the wall behind the head; the CAP fixes it). Implausible
single-frame steps over 2060 person-frames: 2158 unfiltered → 630 with the single-person grouping →
**77**. `--no-filter-feet` / `--no-filter-head` restore that grouping exactly.

### Budget

RTMW3D is **20.7 ms p50 with a fixed batch of [1,3,384,288]** — no batch axis in the ONNX — so N
people cost N sequential inferences: 2 → 24 fps, 3 → 16, 4 → 12. The F-33 chain adds **0.88 ms per
person-frame** (~4%). The detector is free: it runs on the VPU, which was doing nothing but stereo
(rgb 29.8 → 29.7 fps, depth 29.6 → 29.4).

### What multi-person does NOT have

* **No F-21 ownership.** It is superseded here: ownership holds ONE person and refuses hand-off; this
  is an assignment problem, not a gating problem.
* **No P1-3 interpolation.** That buffer reconstructs one body at a presentation time and has no
  concept of identity.
* **P1-1's horizons are counted in FRAMES**, and this loop is slower, so a 6-frame prediction spans
  ~375 ms at 16 fps against ~200 ms in the single-person loop. Only P0 adapts to the real rate.
* **It has never run with real people in front of the camera.** All evidence is recorded video and
  synthetic trajectories.

---

## Extension points

- New pose model → implement the `RTMW3D.infer` shape (`uv`, `zrel`, `conf`) and swap it in
- New plausibility signal → add a term inside `JointTracker._suspicion`; **cap its contribution** unless
  it is provably decisive (see `decisions.md` ADR-P004)
- New diagnostic → add a field to `sender_log.jsonl`; mark it `DIAG-ONLY` and keep it additive
- New per-person state (a filter, a score, an accumulator) → key it on the **track id** and give it
  a lifecycle, never an index into the tracker's emitted list; see *Multi-person* above for why

**Do not** add a second smoothing stage, move safety into this repo, or change the wire format without
a coordinated ADR in both repos. **Do not** modify `wholebody_udp_sender.py` to add a multi-person
feature — it carries the whole measured single-person stack and is deliberately byte-identical;
`multiperson_udp_sender.py` imports its builders instead.
