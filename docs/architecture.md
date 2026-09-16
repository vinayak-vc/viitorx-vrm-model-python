# Python Tracking Sidecar — Architecture

Agent entry summary. Run instructions and the wire contract live in `../README.md`; the reasoning
behind each tuned value lives in `decisions.md`.

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

Dependency direction is strictly **downward**: `wholebody_udp_sender` imports the others; none of them
import it, and `joint_tracker.py` imports **nothing but `math`** so it stays unit-testable in isolation.

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

## Extension points

- New pose model → implement the `RTMW3D.infer` shape (`uv`, `zrel`, `conf`) and swap it in
- New plausibility signal → add a term inside `JointTracker._suspicion`; **cap its contribution** unless
  it is provably decisive (see `decisions.md` ADR-P004)
- New diagnostic → add a field to `sender_log.jsonl`; mark it `DIAG-ONLY` and keep it additive

**Do not** add a second smoothing stage, move safety into this repo, or change the wire format without
a coordinated ADR in both repos.
