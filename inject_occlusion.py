#!/usr/bin/env python3
"""P0-1 UNITY GATE PROOF — scripted occlusion injector (device-free, diagnostic-only).

Streams the SAME UDP JSON contract as the real sidecar to Unity, but with a scripted
confidence timeline, so the Unity LimbGate can be proven to fire WITHOUT inferring
anything from the Python-side hold (the sidecar is not running at all here).

An occluded joint is emitted EXACTLY as the real sidecar emits it when confidence
falls below --conf: the whole landmark becomes [0,0,0,0]
(see wholebody_udp_sender.build_body_landmarks.emit -> early return leaves the zero slot).
So Unity sees byte-for-byte what a real occlusion produces.

Scripted timeline (frames @30 Hz), per limb, with recovery between each:
    3, 5, 8, 12, 20 frame occlusions
which directly exercises the brief's 3-5 / 8 / 12+ requirements.

Writes ground_truth.json so the analyzer can compare INTENDED vs OBSERVED gate holds.

WHAT THIS PROVES : the Unity gate mechanism fires, holds for the right duration, never
                   emits zero, and re-acquires. Bone lengths stay constant under it.
WHAT IT DOES NOT : real confidence-degradation profiles, real depth behaviour, real
                   reacquisition dynamics, or how the avatar LOOKS. Those need a human.

Usage (Unity in PLAY, pipelineLogging = ON):
    python inject_occlusion.py
"""
import argparse
import json
import math
import socket
import time

NUM = 33

# Rough neutral standing pose, mid-hip origin, MediaPipe-world convention (x right, y DOWN,
# z forward). Matches mock_udp_sender.BASE so the avatar takes a plausible pose.
BASE = {
    0: (0.00, -0.62, 0.05),
    11: (0.18, -0.50, 0.00), 12: (-0.18, -0.50, 0.00),
    13: (0.34, -0.30, 0.00), 14: (-0.34, -0.30, 0.00),
    15: (0.42, -0.10, 0.00), 16: (-0.42, -0.10, 0.00),
    23: (0.10, 0.00, 0.00), 24: (-0.10, 0.00, 0.00),
    25: (0.12, 0.45, 0.00), 26: (-0.12, 0.45, 0.00),
    27: (0.13, 0.88, 0.00), 28: (-0.13, 0.88, 0.00),
}

# Kalidokit cross-map, mirrored exactly from KalidokitControlRigDriver.Apply:
#   leftArm gate  <- landmarks 12,14,16      rightArm gate <- 11,13,15
#   leftLeg gate  <- landmarks 24,26,28      rightLeg gate <- 23,25,27
LIMBS = {
    "lArm": [12, 14, 16],
    "rArm": [11, 13, 15],
    "lLeg": [24, 26, 28],
    "rLeg": [23, 25, 27],
}

OCCLUSION_LENGTHS = [3, 5, 8, 12, 20]
RECOVERY_FRAMES = 40      # clean frames between occlusions (gate must re-acquire and settle)
WARMUP_FRAMES = 360       # 12 s @30 Hz. MUST outlast Unity's scene + VRM avatar load:
                          # model_log only writes once boundAnimator != null (measured ~7 s),
                          # and Bind() calls LimbGate.Reset(), so an occlusion before that
                          # lands on a gate with no valid rotation yet (correctly refuses to
                          # apply one) and is unobservable anyway.


def build_timeline():
    """[(limb_or_None, n_frames, phase_label)] -> the full scripted run."""
    tl = [(None, WARMUP_FRAMES, "warmup")]
    for limb in ["lArm", "rArm", "lLeg", "rLeg"]:
        for n in OCCLUSION_LENGTHS:
            tl.append((limb, n, "occlude:%s:%d" % (limb, n)))
            tl.append((None, RECOVERY_FRAMES, "recover:%s:%d" % (limb, n)))
    return tl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--conf", type=float, default=0.9, help="confidence for VISIBLE joints")
    ap.add_argument("--out", default="pipeline_logs/ground_truth.json")
    a = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (a.host, a.port)
    dt = 1.0 / a.fps
    timeline = build_timeline()
    total = sum(n for _, n, _ in timeline)

    print("=" * 70)
    print(" P0-1 UNITY GATE PROOF - scripted occlusion injector")
    print("=" * 70)
    print(" target      : %s:%d @ %.0f Hz" % (a.host, a.port, a.fps))
    print(" frames      : %d  (%.1f s)" % (total, total * dt))
    print(" occlusions  : %s frames, per limb lArm/rArm/lLeg/rLeg" % OCCLUSION_LENGTHS)
    print(" occluded joint is emitted as [0,0,0,0] - exactly as the real sidecar does")
    print("=" * 70)
    print(" Unity must be in PLAY with pipelineLogging = ON.")
    print()

    events = []
    seq = 0
    t_start = time.time()
    phase_i = 0
    for limb, nframes, label in timeline:
        t_phase0 = time.time()
        seq0 = seq
        for _ in range(nframes):
            # gentle sway so the pose is not perfectly static (exercises the solver)
            sway = 0.03 * math.sin((seq / a.fps) * 1.2)
            lm = [[0.0, 0.0, 0.0, 0.0] for _ in range(NUM)]
            hidden = set(LIMBS[limb]) if limb else set()
            for idx, (x, y, z) in BASE.items():
                if idx in hidden:
                    continue          # leave [0,0,0,0] -> the real occlusion signature
                lm[idx] = [x + sway, y, z, a.conf]
            msg = {"lm": lm, "xyz": [0.0, 0.0, 2200.0],
                   "src": [1] * NUM, "seq": seq, "t": round(time.time(), 4)}
            sock.sendto(json.dumps(msg).encode("utf-8"), addr)
            seq += 1
            time.sleep(dt)
        events.append({"phase": label, "limb": limb, "frames": nframes,
                       "seqStart": seq0, "seqEnd": seq - 1,
                       "tStart": round(t_phase0, 4), "tEnd": round(time.time(), 4)})
        phase_i += 1
        if limb:
            print("  [%2d/%2d] OCCLUDE %-5s %2d frames  (seq %d..%d)"
                  % (phase_i, len(timeline), limb, nframes, seq0, seq - 1))

    sock.close()
    with open(a.out, "w") as f:
        json.dump({"events": events,
                   "occlusionLengths": OCCLUSION_LENGTHS,
                   "limbs": list(LIMBS.keys()),
                   "fps": a.fps}, f, indent=2)
    print()
    print(" done in %.1f s, %d frames sent" % (time.time() - t_start, seq))
    print(" ground truth -> %s" % a.out)
    print()
    print(" Now STOP Unity play mode and run:")
    print("   python verify_gate.py --dir pipeline_logs")


if __name__ == "__main__":
    main()
