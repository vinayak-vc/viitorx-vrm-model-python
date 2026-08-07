# viitorx-vrm-model-python

Python **tracking sidecar** for the Virtual Mirror VRM avatar app. It runs pose/hand/face
models on the host + OAK-D depth camera and streams landmarks to the Unity client over a local
UDP socket. Running the model code in a separate process (not a native plugin inside Unity) keeps
any DepthAI/GPU instability out of the editor — that was the whole point of the sidecar design
(ADR-016, "Option B2").

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

## Layout

```
.
├── mock_udp_sender.py          # device-free test: streams a synthetic skeleton on the wire contract
├── requirements.txt
└── depthai_blazepose/          # vendored geaxgx/depthai_blazepose (MIT) + our OV9782 modifications
    ├── udp_pose_sender.py       # Phase-1 sidecar: BlazePose on the OAK-D -> UDP
    ├── BlazeposeDepthaiEdge.py  # (modified) OV9782 native color mode / full-FOV fix
    ├── models/*.blob            # BlazePose blobs (device NN, needed at runtime)
    └── ...                      # geaxgx pipeline + utils
```

`depthai_blazepose/` is a vendored copy of [geaxgx/depthai_blazepose](https://github.com/geaxgx/depthai_blazepose)
(MIT — see `depthai_blazepose/LICENSE.txt`) carrying our device-specific modifications. The upstream
`examples/` and `img/` demo assets were removed; they are not needed to run the sidecar.

---

## Setup (Windows, Python 3.10)

```bash
"C:\Program Files\Python310\python.exe" -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

> The `.venv/` is git-ignored (and Unity-ignored). Use Python **3.10** to match the OAK-D `depthai`
> wheels (cp310). ONNX Runtime uses the DirectML execution provider on the RTX 3060 — no CUDA toolkit
> required.

---

## Run

### Phase 1 — BlazePose on the OAK-D (current, working)

Runs Google BlazePose on the OAK-D's own VPU (edge mode) and streams 33 body landmarks
(hip-relative metres, GHUM) to Unity.

```bash
cd depthai_blazepose
..\.venv\Scripts\python udp_pose_sender.py                 # 127.0.0.1:8899, lite model
..\.venv\Scripts\python udp_pose_sender.py --lm full       # more accurate, ~18 fps
..\.venv\Scripts\python udp_pose_sender.py --show          # + cv2 preview of the OAK view
```

Device note: this unit is an **OAK-D-PRO-W** (wide lens, **OV9782** 1280×800 color). The defaults
`--color_res 800p --color_scale 1/2` give the full ~120° FOV at 640×400. `--frame_height 200` is the
verified value for the legacy path; higher values can error on this sensor's ISP limits.

### Device-free — mock sender (no camera)

Streams a synthetic animated skeleton on the exact UDP contract, so the Unity path can be tested
with no hardware. Pure stdlib — no venv required.

```bash
python mock_udp_sender.py
```

---

## UDP wire contract

Local UDP, default `127.0.0.1:8899`, one JSON datagram per frame:

```jsonc
{ "lm":  [[x, y, z, vis], ...33],   // body landmarks, hip-relative metres (BlazePose/JointId order)
  "xyz": [hipX, hipY, hipZ] }        // measured mid-hip, millimetres, camera space (root position)
```

Phase 2 (in progress) extends this with `lh`/`rh` (per-hand landmarks) and `face` (blendshapes) — see
the Unity project's `docs/26_OakDDepthPhase2.md` and the whole-body-fusion ADR.

---

## Roadmap

- **Phase 1 (done):** OAK-D BlazePose edge → 33 body landmarks, GHUM-estimated depth.
- **Phase 2 (in progress):** host-side **RTMW3D** whole-body model (body + feet + hands) on the RTX 3060
  via ONNX Runtime (DirectML), fused with **measured** per-keypoint OAK-D depth; plus MediaPipe face
  blendshapes on the OAK RGB. Fixes wrist/finger tracking and the monocular front/back ("hands behind")
  ambiguity at the source.

## Licenses

- This project: see repository license.
- `depthai_blazepose/`: MIT © geaxgx (`depthai_blazepose/LICENSE.txt`).
