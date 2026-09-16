# Python Tracking Sidecar — Project Overview

**Status:** Production path. Stability program **P0 + P1-1 + P1-2 COMPLETE** (live-verified with a
human subject on real OAK-D hardware). See `ai_handoff.md` for the status table.
**Stack:** Python 3.10 + DepthAI (OAK-D-PRO-W) + RTMW3D-x ONNX (ONNX Runtime / DirectML) + NumPy + OpenCV
**Transport:** local UDP JSON, `127.0.0.1:8899`

---

## What It Is

The **sensing half** of the Virtual Mirror. It turns an OAK-D depth camera into a stream of stable,
metric, temporally-validated body landmarks, and hands them to the Unity avatar client over a local
UDP socket.

```
OAK-D (RGB + stereo depth) → RTMW3D → measured depth fusion
                           → signal conditioning (P0 + P1-1)
                           → UDP JSON → Unity
```

It runs as a **separate process**, not a native plugin inside Unity. That was deliberate (ADR-016,
"Option B2"): DepthAI and GPU inference are the least stable parts of the system, and a crash there
must not take the editor with it.

---

## Why a sidecar at all

| Concern | Consequence of in-process | Consequence of sidecar |
|---|---|---|
| DepthAI native `abort()` | kills the Unity editor | kills one Python process |
| ONNX/DirectML warm-up, driver stalls | editor hitch | absorbed before the datagram |
| Model swap / retune | Unity recompile + domain reload | restart a script |
| Language fit for CV work | poor | native |

The cost is a process boundary and a serialisation step — measured at **0.30 ms median** on loopback,
with **0.00% packet loss** over 55 000+ packets. That is a good trade.

---

## Scope

**In scope (this repo owns):**
- OAK-D pipeline construction, RGB/depth alignment, intrinsics
- Frame-freshness / queue policy
- RTMW3D whole-body inference and person-box tracking
- Per-keypoint depth sampling and back-projection to metric camera space
- Signal conditioning: One-Euro smoothing, displacement caps, bounded hold (**P0**)
- Per-joint temporal tracking and plausibility (**P1-1**)
- The UDP wire format, and every diagnostic that explains it

**Out of scope (the Unity repo owns):**
- Retargeting, Kalidokit, the VRM avatar, rendering
- **The P0 LimbGate** — the final safety gate lives in Unity, deliberately
- Pose buffering / interpolation on the render timeline (P1-3)
- Face blendshapes (MediaPipe, in Unity)

---

## Non-negotiables

1. **No measurement, no claim.** Every behavioural statement in these docs cites a number.
2. **Freshness beats completeness.** Dropping a stale frame is correct; processing it is not.
3. **This repo improves the signal; Unity owns safety.** A `LOST` joint is emitted as `[0,0,0,0]` —
   the "invalid" signal — never as a fabricated position.
4. **The UDP datagram is a cross-repo contract.** Changing it needs an ADR in both repos.

---

## Doc Map

| Need | Read |
|------|------|
| Run it / flags / wire contract | `../README.md` |
| Agent rules for this repo | `../AGENTS.md` |
| Stage-by-stage design | `architecture.md` |
| What is done / next | `roadmap.md`, `tasks.md` |
| Why a constant is that value | `decisions.md` |
| Resume work here | `ai_handoff.md` |
| Unity-side counterpart | Unity repo `docs/` + its `AGENTS.md` |
