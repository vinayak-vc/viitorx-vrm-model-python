# viitorx-vrm-model-python

Python **tracking sidecar** for the Virtual Mirror VRM avatar app. It runs pose/hand models on the
host + OAK-D depth camera and streams landmarks to the Unity client over a local UDP socket. Running
the model code in a separate process (not a native plugin inside Unity) keeps any DepthAI/GPU
instability out of the editor — that was the whole point of the sidecar design (ADR-016, "Option B2").

The Unity consumer lives in the `viitorx-vrm-avtar-unity` project
(`Runtime/Tracking/OakD/OakDUdpPoseProvider.cs`).

## Repository role (submodule)

This repo (`vinayak-vc/viitorx-vrm-model-python`) is consumed as a **git submodule** of the Unity
project repo `vinayak-vc/viitorx-vrm-avtar-unity`, checked out at:

```
Assets/Games/viitorx-vrm-avtar-unity/python-sidecar~/
```

The trailing `~` makes Unity's asset pipeline **ignore the whole folder** (same convention as
`Samples~`), so the model blobs and the `.venv` are never imported by the editor. The virtualenv is
named `.venv` (dot-hidden) as a second safeguard. **This supersedes the old `oak_sidecar/` folder at
the Unity-project root, which is deprecated — use this submodule from now on.**

---

## Pipeline stages owned by this repo

The sidecar owns everything from the sensor up to the UDP datagram. Unity owns everything after it.

**Docs for this repo** — read these before changing anything:

| File | Purpose |
|---|---|
| [`AGENTS.md`](AGENTS.md) | rules for AI agents working here + the **repo responsibility boundary** |
| [`docs/project-overview.md`](docs/project-overview.md) | what this repo is and is not |
| [`docs/architecture.md`](docs/architecture.md) | stages, module map, coordinate spaces, failure behaviour |
| [`docs/decisions.md`](docs/decisions.md) | **ADR-P001…P007** — why every tuned constant is that value |
| [`docs/roadmap.md`](docs/roadmap.md) | delivered vs next candidates |
| [`docs/tasks.md`](docs/tasks.md) | checklist + gotchas |
| [`docs/ai_handoff.md`](docs/ai_handoff.md) | **start here to resume work** |

The Unity counterpart decisions are **ADR-028…031** in the Unity project's `docs/decisions.md`, with
measured evidence in `docs/P0_ACCEPTANCE_2026-09-07.md`, `P1_1_TRACKER_*.md`, `P1_2_FRESHNESS_*.md`,
`P1_3_POSE_BUFFER_*.md`.

```
OAK-D RGB + RGB-aligned stereo depth
  → latest-frame queue policy        P1-2  (newest RGB; depth matched by TIMESTAMP)
  → RTMW3D-x inference (~21 ms, ONNX Runtime / DirectML)
  → per-keypoint depth sample (5x5 window, 30th percentile) + back-projection
  → P0 smoother     One-Euro + 0.35 m distal caps + bounded hold-on-dropout
  → P1-1 JointTracker   TRACKED / WEAK / PREDICTED / LOST + plausibility
  → UDP JSON  { lm[33], lh?, rh?, xyz, src, seq, t }
─────────────────────────────────────────────────────── process boundary → Unity
```

**Layering rule:** P1-1 improves the *signal*; the Unity-side **P0 LimbGate is the final safety
mechanism**. A joint P1-1 marks `LOST` has its **emit confidence zeroed**, which is byte-identical to a
real occlusion, so Unity's gate still makes the final call. Never move safety out of P0.

---

## Layout

```
.
├── wholebody_udp_sender.py     # PRIMARY sidecar: RTMW3D + measured depth + P0/P1-1/P1-2 → UDP
├── rtmw3d_pose.py              # RTMW3D ONNX wrapper (SimCC decode, person-box tracking)
├── oak_depth.py                # DepthAI pipeline, intrinsics, 5x5/p30 depth sampling, back-projection
├── smoothing.py                # P0: One-Euro + per-index displacement caps + bounded hold
├── joint_tracker.py            # P1-1: JointTracker / SkeletonTracker (temporal state + plausibility)
├── mock_udp_sender.py          # device-free: synthetic skeleton on the wire contract
│
├── test_joint_tracker.py       # P1-1 unit tests (37 assertions, no pytest needed)
├── evaluate_p1.py              # P1-1 replay (false-rejection) + adversarial harness
├── analyze_capture.py          # P0 acceptance analyzer (displacement, gate, latency, packet loss)
├── guided_capture.py           # guided human capture: prompts through blocks A–J, writes blocks.json
├── verify_gate.py              # P0-1 proof: intended vs observed Unity LimbGate holds
├── inject_occlusion.py         # scripted occlusion injector (device-free, drives Unity directly)
├── stream_motion.py            # deterministic motion streamer (P1-3 interpolation A/B)
├── compare_p12.py              # P1-2 A/B comparator (velocity-normalised, not per-frame)
├── compare_logs.py             # 3-stage pipeline log diff (sender / recv / model)
├── replay_video.py             # offline RGB replay harness
├── validate_rtmw3d.py          # draw keypoints on one RGB frame
├── validate_depth.py           # draw measured/hole depth + report metric XYZ
├── validate_p0.py              # P0 offline validation (spikes, dropouts, gate simulation)
│
├── run_capture.bat             # one-shot capture + compare report
├── run_p0_acceptance.bat       # P0 human acceptance capture (blocks A–J)
├── run_p12_ab.bat              # P1-2 human A/B (FIFO vs latest-frame)
│
├── requirements.txt
└── depthai_blazepose/          # vendored geaxgx/depthai_blazepose (MIT) + our OV9782 modifications
    ├── udp_pose_sender.py       # Phase-1 fallback: BlazePose on the OAK-D VPU → UDP
    ├── BlazeposeDepthaiEdge.py  # (modified) OV9782 native color mode / full-FOV fix
    └── models/*.blob            # BlazePose blobs (device NN, needed at runtime)
```

Capture output (`pipeline_logs*/`, `p12*/`, `probe*/`, …) is **git-ignored** — regenerate on demand.

---

## Setup (Windows, Python 3.10)

```bash
"C:\Program Files\Python310\python.exe" -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

> Use Python **3.10** to match the OAK-D `depthai` wheels (cp310). ONNX Runtime uses the DirectML
> execution provider on the RTX 3060 — no CUDA toolkit required. The RTMW3D ONNX (369 MB) lives at the
> Unity project's `Assets/SentisModel/rtmw3d-x.onnx` (git-ignored; download URL in ADR-015).

---

## Run

### Primary — whole-body RTMW3D + measured depth

```bash
.venv\Scripts\python wholebody_udp_sender.py --model ..\..\..\SentisModel\rtmw3d-x.onnx
.venv\Scripts\python wholebody_udp_sender.py --model <onnx> --show          # + cv2 preview
.venv\Scripts\python wholebody_udp_sender.py --model <onnx> --seconds 20    # auto-stop (testing)
.venv\Scripts\python wholebody_udp_sender.py --model <onnx> --log-dir pipeline_logs
```

Stand ~2–2.5 m out, full body in frame. Beyond ~2.5 m depth coverage and facing both degrade.

#### Stability flags (defaults are the shipping configuration — change only with evidence)

| Flag | Default | Stage | What it does |
|---|---|---|---|
| `--arm-max-jump` | `0.35` | P0-2 | per-frame displacement cap (m) for elbows+wrists; slewed, not dropped |
| `--leg-max-jump` | `0.35` | P0-2 | same for knees+ankles |
| `--max-jump` | `1.5` | P0 | global cap for trunk + hands (kept generous) |
| `--max-hold-frames` | `8` | P0 | bounded hold through depth dropouts |
| `--min-cutoff` / `--beta` | `0.5` / `0.4` | P0 | One-Euro stillness / reaction |
| `--depth-min-cutoff` / `--depth-beta` | `0.3` / `0.1` | P0 | heavier filtering on the noisy Z axis |
| `--tracker` / `--no-tracker` | ON | P1-1 | per-joint temporal tracking + plausibility |
| `--tracker-predict-frames` | `6` | P1-1 | max PREDICTED frames before a joint goes LOST |
| `--tracker-reacquire-frames` | `5` | P1-1 | blend frames on reacquisition (never teleport) |
| `--latest-frame` / `--no-latest-frame` | ON | P1-2 | drain queues to the NEWEST frame each iteration |
| `--conf` | `0.3` | P0 | emit-gate threshold (matches Unity's `limbConfidenceThreshold`) |
| `--inject-load-ms` | `0` | — | **TEST ONLY.** Adds artificial work to reproduce the consumer-slower-than-sensor condition without a human. Never set in production. |

> **Why `0.35 m`:** peak *legitimate* fast-arm displacement measured **0.0746 m** — 4.7× headroom —
> while pre-P0 spikes reached 0.5–0.9 m. **Why latest-frame:** `DataOutputQueue.get()` returns the
> **oldest** packet, so a full `maxSize=4` queue at 30 fps meant every pose was ~133 ms stale
> (measured 131 ms). Draining to newest took frame age to **31 ms** with no fps or compute change.

### Device-free — mock sender / scripted occlusion / deterministic motion

No camera required; these drive Unity directly over the wire contract.

```bash
python mock_udp_sender.py                     # synthetic animated skeleton (pure stdlib)
python inject_occlusion.py                    # scripted 3/5/8/12/20-frame occlusions per limb
python stream_motion.py --fps 21 --seconds 30 # smooth deterministic swing (interpolation A/B)
```

### Phase 1 — BlazePose on the OAK-D VPU (fallback)

```bash
cd depthai_blazepose
..\.venv\Scripts\python udp_pose_sender.py --lm full --show
```

Device note: this unit is an **OAK-D-PRO-W** (wide lens, **OV9782** 1280×800 color). Defaults
`--color_res 800p --color_scale 1/2` give the full ~120° FOV at 640×400.

---

## Tests

```bash
.venv\Scripts\python test_joint_tracker.py      # P1-1: 37 assertions, exit 0 on pass
.venv\Scripts\python evaluate_p1.py --dir pipeline_logs    # replay + adversarial (needs a capture)
```

`test_joint_tracker.py` is self-contained (no pytest). It covers stationary, constant velocity, fast
motion, isolated spike, **high-confidence spike**, 3/5/10-frame dropouts, prediction, prediction
expiry, reacquisition, large-error reacquisition, irregular timestamps, zero/invalid measurement,
depth inconsistency, neighbour constraint, and a per-frame cost budget check.

---

## Diagnostics

`--log-dir <dir>` writes JSONL streams, one row per frame:

| file | contents |
|---|---|
| `sender_log.jsonl` | per-frame key landmarks, `hipZ`, coverage, **stage timings** and **freshness metrics** |
| `holds_log.jsonl` | P0-2 spike/hold events + P1-1 state transitions and suspicious samples (events only — healthy frames are never logged) |

Freshness / latency fields (permanent production diagnostics, added in P1-2):

| field | meaning |
|---|---|
| `frameAgeMs` | host processing time − camera frame timestamp (OAK device clock) |
| `queueDepth` | packets waiting when the frame was sampled |
| `staleDropped` | frames discarded as stale this iteration |
| `rgbDepthSyncMs` | \|RGB timestamp − depth timestamp\| |
| `camLatMs`, `capToPoseMs`, `poseToDepthMs`, `capToSendMs` | per-stage cost |
| `trackerMs` | P1-1 cost (~0.085 ms for 12 joints) |

The console status line prints `age=NNms stale=N` every 2 s.

---

## UDP wire contract

Local UDP, default `127.0.0.1:8899`, one JSON datagram per frame:

```jsonc
{ "lm":  [[x, y, z, vis], ...33],   // body, hip-relative metres, JointId order (unmapped slots = zeros)
  "lh":  [[x, y, z], ...21],         // left hand landmarks, hip-relative metres  (omitted when untracked)
  "rh":  [[x, y, z], ...21],         // right hand landmarks                      (omitted when untracked)
  "xyz": [hipX, hipY, hipZ],         // measured mid-hip, millimetres, camera space (root position)
  "src": [0|1, ...33],               // 1 = depth-measured, 0 = hip-plane fallback (debug)
  "seq": 1234,                       // monotonic frame id
  "t":   1788846511.0234 }           // send epoch seconds
```

**A joint that fails the confidence gate — or that P1-1 marks `LOST` — is emitted as `[0,0,0,0]`.**
That zero is the "invalid" signal, **not** a position. Unity must never interpolate a position toward
it (doing so drags the limb toward the origin — the F-01 collapse bug); Unity's `PoseBuffer` and
`LimbGate` both special-case it.

`seq` and `t` are **load-bearing since P1-3**: Unity's timestamped pose buffer uses them for ordering,
duplicate/out-of-order rejection and interpolation. They are no longer log-only fields.

Axes are camera space (X right, Y down, Z forward), so the Unity `PoseSpaceConverter` +
`poseFlipX/Y/Z` tuning applies. Face blendshapes stay on MediaPipe (not in this stream).
Design detail: Unity project's `docs/26_OakDDepthPhase2.md` + ADR-015/016/018.

---

## Status

- **Phase 1 (done):** OAK-D BlazePose edge → 33 body landmarks, GHUM-estimated depth.
- **Phase 2 (done, hardware-verified):** host-side RTMW3D whole-body + **measured** per-keypoint OAK-D
  depth; body + both hands with metric depth over UDP.
- **Stability program (done, live-verified with a human subject):**
  - **P0** — 0.35 m distal caps, legs into the hold set (ADR-028). Limb spikes **−67…−76%**.
  - **P1-1** — per-joint temporal tracking + plausibility (ADR-029). Catches *confident-but-wrong*
    landmarks that no confidence gate can see; **6/6** adversarial cases detected; **0.085 ms** cost.
  - **P1-2** — latest-frame queue policy (ADR-030). Frame age **131 → 31 ms**, camera→UDP
    **162 → 62 ms**, RGB/depth pairing *improved*; 180 s soak with no accumulation.

### Known limits (do not re-discover these)

- **RTMW3D does not lower confidence for an occluded limb.** A hand hidden behind the torso keeps
  ~0.63 confidence and a *wrong* position (measured: 0 frames below 0.3 across a 45 s block). This is
  why P1-1's plausibility layer exists — confidence alone is not validity.
- **A sustained high-confidence teleport** longer than the 6-frame prediction window ends in `LOST`
  with slow recovery during fast motion. Needs short-gap prediction + blended recovery (not built).
- **Compute spikes >80 ms occur only at frame index 0–2** — ONNX/DirectML warm-up, per *process*.
  A 3378-frame soak spiked only at frames 0 and 1. Not a steady-state problem.
- **Confidence is an un-normalised SimCC peak** (audit F-08); the `0.3` threshold is tuned to that
  scale and is brittle across distance/lighting. P1-1 works around it kinematically.
- Facing/depth degrade past ~2.5 m — a single-front-camera limit (ADR-019/023/024/027).

## Licenses

- This project: see repository license.
- `depthai_blazepose/`: MIT © geaxgx (`depthai_blazepose/LICENSE.txt`).
