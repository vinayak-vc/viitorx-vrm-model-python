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

The root holds the **production path and nothing else**: the two entry points plus the nine modules
`wholebody_udp_sender.py` imports. Everything else is a harness, a self-test or a superseded
one-off, and lives under `tools/` or `tests/` (ADR-065).

That split is what lets the Unity build copy the root directory verbatim and get exactly what a
player needs — see `Editor/SidecarBuildPostprocessor.cs`.

```
.
├── sidecar_supervisor.py       # ENTRY: watchdog; restarts a dead sidecar. Unity launches this.
├── wholebody_udp_sender.py     # ENTRY: RTMW3D + measured depth + P0/P1 stages -> UDP 8899
│
├── rtmw3d_pose.py              # RTMW3D ONNX wrapper (SimCC decode, person-box tracking)
├── oak_depth.py                # DepthAI pipeline, intrinsics, depth sampling, back-projection
├── f18_portrait.py             # F-18/F-19 portrait transform (verified, imported - not re-derived)
├── smoothing.py                # P0: One-Euro + per-index displacement caps + bounded hold
├── joint_tracker.py            # P1-1: temporal state + plausibility
├── kinematic_recovery.py       # P1-4: skeleton constraints + long-horizon recovery
├── target_ownership.py         # F-21: single-person target ownership
├── pose_validation.py          # F-22: human / biomechanical pose validation
├── f21_cue_display.py          # F-21: on-screen subject cues
├── _sidecar_path.py            # import shim so harnesses in subfolders resolve the modules above
│
├── tests/                      # 8 self-tests. No pytest needed; each prints "N/N assertions passed"
│                               #   test_target_ownership 75, test_kinematic_recovery 39,
│                               #   test_joint_tracker 37, test_surface_depth 32, test_pose_validation 22
│
├── tools/
│   ├── capture/      (23)      # F-16 stereo config sweeps, F-18 portrait, F-19 production capture
│   ├── deployment/    (7)      # F-20A/F-20B supervisor, USB and failure-injection harnesses
│   ├── ownership/    (13)      # F-21 two-person ownership: protocols, replays, adversarial
│   ├── validation/    (6)      # F-22 pose validation: thresholds, rejections, replay soak
│   ├── video/         (4)      # F-23 video-driven pipeline and demo composition
│   ├── diagnostics/  (11)      # F-24..F-27 jitter, fidelity, humanized-skeleton analysis
│   └── armaim/        (9)      # ARM V2/V3 aiming analysis (was loose inside arm_v*_evidence/)
│
├── scripts/                    # double-clickable launchers. Each re-bases to the repo root with
│   ├── run_supervisor.bat      #   `cd /d "%~dp0.."`, so they work from this subfolder.
│   ├── run_capture.bat         # capture + compare_logs report
│   ├── oak_guided_v4.ps1       # guided LIVE capture, instructions SPOKEN aloud (the subject is
│   │                           #   2 m away and cannot read a console)
│   ├── run_p0_acceptance.bat   # DOES NOT RUN - needs analyze_capture.py, deleted in 29ec57e.
│   └── run_p12_ab.bat          # DOES NOT RUN - needs compare_p12.py, deleted in 29ec57e.
│                               #   Both kept because P0_ACCEPTANCE / P1_2_FRESHNESS cite them by
│                               #   name as the command that produced their numbers. They exit 1.
│
├── setup_sidecar.ps1           # one-time target setup; VERIFIES the DirectML provider
├── requirements.lock.txt       # exact pins - this is what setup installs
├── requirements.txt            # human-readable intent
│
├── docs/                       # sidecar-local notes + p0_human_report.txt
│
├── evidence/                   # ALL capture output, under one root. Resolve it with
│   ├── oak_v4/                 #   evidence_paths, never by hand - see below.
│   ├── arm_v1/                 # Tracked: the distilled .txt/.json analyses the reports cite.
│   ├── arm_v2/                 # Git-ignored: the per-frame captures, which are regenerable.
│   └── arm_v3/
│
├── evidence_paths.py           # the ONLY place that knows where evidence lives
│
└── depthai_blazepose/          # vendored geaxgx/depthai_blazepose (MIT), Phase-1 fallback.
                                # Superseded and NOT imported by the production path.
```

### Evidence paths

Every capture path resolves through `evidence_paths`, never through a literal:

```python
import evidence_paths as EV
EV.oak_v4("f21")            # <root>/evidence/oak_v4/f21
EV.evidence("oak_v4/f21")   # same thing; embedded separators are fine
EV.ensure_dir(EV.oak_v4("f21"))
```

It resolves from the module's own location, so the answer does not depend on the caller's working
directory or on how deep the calling script sits. Both had already caused silent bugs: the ADR-065
move broke thirteen harnesses that built paths from a per-file `HERE`, and seven others hardcoded an
absolute `C:\Unity\...` path to a checkout that exists on no machine - including this one, since the
project now lives on `D:`.

Set `VIRTUAL_MIRROR_EVIDENCE_DIR` to redirect the whole tree, e.g. to keep captures off the repo
volume.

The root holds 18 files: the 12 production `.py` above plus `.gitignore`, `README.md`, `AGENTS.md`,
`setup_sidecar.ps1` and the two requirements files — each of which belongs at a repository root by
convention. It was 96 before ADR-065.

**Running a harness.** They are still plain scripts, so only the path changed:

```powershell
.venv\Scripts\python.exe tests\test_target_ownership.py
.venv\Scripts\python.exe tools\diagnostics\f26_fidelity_analyze.py --selftest
.venv\Scripts\python.exe tools\ownership\f21_walkin_protocol.py --help
```

Anything under `tools/` or `tests/` that imports a project module carries a two-line prelude that
walks up to `_sidecar_path.py` and puts the root — and every `tools/` group — on `sys.path`. Python
only puts the *script's own* directory on the path, so without it a harness one level down cannot
`import rtmw3d_pose`.

Capture output (`pipeline_logs*/`, `evidence/oak_v4/`, `probe*/`, …) is **git-ignored** — regenerate
on demand; the commands are in the F-report that used them.

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

### Supervised — restart on a device crash (F-20B)

`sidecar_supervisor.py` owns the sidecar's lifecycle: readiness detection, backoff, crash-loop
protection and a single-instance guard. Use it for unattended running and for live test sessions,
where a DepthAI fault would otherwise end the experiment.

```bash
.venv\Scripts\python sidecar_supervisor.py --model ..\..\..\SentisModel\rtmw3d-x.onnx
.venv\Scripts\python sidecar_supervisor.py --model <onnx> --show --cue-file <path\to\cue.json>
```

`--show` and `--cue-file` are forwarded verbatim to the sidecar (F-21 §31). They are named
explicitly rather than accepted as a generic passthrough: a supervisor that forwards arbitrary
strings could also forward `--no-ownership`, and then the thing under test is not the thing that was
configured. With neither flag set the launched command line is byte-identical to before.

Run every command above with `.venv\Scripts\python.exe`, not the `python` on PATH — they are
different environments and only the venv has DirectML. See `docs/ai_handoff.md` §6.

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

### Device-free — drive Unity without a camera

`mock_udp_sender.py`, `inject_occlusion.py` and `stream_motion.py` were removed in `29ec57e`
("Remove legacy verification and video streaming scripts"). Their surviving replacements:

```bash
# Replay a video file through the real pipeline and stream the result to Unity (F-23).
.venv\Scripts\python.exe toolsideo23_video_to_unity.py --help

# A stand-in sidecar for exercising the supervisor's restart paths without hardware (F-20B).
.venv\Scripts\python.exe tools\deployment20b_fake_sidecar.py
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

Self-contained: no pytest, each prints `N/N assertions passed` and exits 0 on success. Last full run
after the ADR-065 reorganisation:

```bash
.venv\Scripts\python.exe tests	est_target_ownership.py       # F-21 ownership          75/75
.venv\Scripts\python.exe tests	est_kinematic_recovery.py     # P1-4 recovery           39/39
.venv\Scripts\python.exe tests	est_joint_tracker.py          # P1-1 tracker            37/37
.venv\Scripts\python.exe tests	est_surface_depth.py          # depth sampling          32/32
.venv\Scripts\python.exe tests	est_pose_validation.py        # F-22 validation         22/22

.venv\Scripts\python.exe tools\diagnostics26_fidelity_analyze.py --selftest    # 28 passed
.venv\Scripts\python.exe tools\diagnostics27_humanized_analyze.py --selftest   # 18 passed

# Replay + adversarial against a real capture (needs a --log-dir from a session).
.venv\Scripts\python.exe tools\diagnostics\evaluate_p1.py --dir <capture-dir>
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
  "t":   1788846511.0234,            // send epoch seconds
  "st":  [-1..4, ...33],             // F-29 trust channel: P1-1/P1-4 state per joint
  "own": "LOCKED",                   // F-29: F-21 ownership state  (OMITTED under --no-ownership)
  "lat": 41.3 }                      // F-29: measured camera->payload latency, ms
```

**Feet are inside `lm`, not a separate field.** `build_body_landmarks` emits `FOOT_TO_JOINTID`
alongside the COCO-17 body: heels at JointId **29/30**, big toes at **31/32**. A consumer that
ignores those four slots is choosing to, not being denied the data.

### The F-29 trust channel (`st` / `own` / `lat`)

Read-only: these echo decisions the pipeline has already made so a consumer can **display** them.
Nothing upstream reads them back and omitting them changes no behaviour, so a consumer written
before F-29 is unaffected.

| `st` | meaning |
|---|---|
| `-1` | **no tracker covers this joint.** Only the 12 joints in `joint_tracker.DEFAULT_TRACKED` have one. This is a claim, not padding — reporting the other 21 as healthy would overstate what the system knows about them. |
| `0`–`4` | `TrackingState`'s own ints: TRACKED, WEAK, PREDICTED, LOST, RECOVERING. Not a parallel enum, so the two cannot drift apart. |

`st` is captured **after** P1-4, so it describes the geometry actually emitted — reading it before
recovery would report `LOST` for a joint that was reconstructed and sent.

`own` **absent** means ownership is not running (`--no-ownership`); it does **not** mean "not
locked". Those mean opposite things to an operator and must not be rendered the same way.

`lat` is the sidecar's leg only. A consumer adds its own receive-to-present time for an end-to-end
figure — and must compare `t` against its own **epoch** clock to do so, never against a
monotonic/stopwatch clock.

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

- This project: MIT - see [`LICENSE`](LICENSE).
- `depthai_blazepose/`: MIT © geaxgx (`depthai_blazepose/LICENSE.txt`).
