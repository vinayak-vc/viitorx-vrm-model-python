#!/usr/bin/env python3
"""ARM RETARGET V2 forensics — deterministic six-pose driver over the REAL UDP wire.

Sends exact, hand-built 33-slot landmark frames on the same JSON contract the OAK-D sidecar uses
(`{"lm": [[x,y,z,vis] x33], "xyz": [...], "src": [...], "seq": n, "t": epoch}`), so the pose travels
the ENTIRE production path — OakDUdpPoseProvider, PoseSpaceConverter, the P1-3 pose buffer, the P0
LimbGate, ArmAimSolver, the control rig, UniVRM's Process() and the rendered humanoid bones — with
zero tracking noise. That is what makes the stage table in
docs/UNITY_ARM_RETARGET_V2_FORENSICS_2026-09-09.md attributable: any divergence between stages is the
CODE, because the input is exact by construction.

No camera, no model, no depth. Standard library only.

Landmark frame (as produced by the real sidecar's backprojection, hip-relative metres):
    +x = image right  = the SUBJECT'S LEFT      -x = the subject's right
    +y = DOWN                                    -y = up
    +z = away from the camera (the subject's back)

Kalidokit's cross-map is preserved downstream: the AVATAR's left arm is driven by landmarks 12/14/16
(the SUBJECT'S right shoulder/elbow/wrist), and vice versa.

    python pose_stage_udp.py --list
    python pose_stage_udp.py --pose tpose --hold 6
    python pose_stage_udp.py --all --hold 6
"""
import argparse
import json
import socket
import sys
import time

NUM = 33

# Static torso/legs shared by every pose (hip-relative metres). Deliberately upright and frontal so the
# torso contributes nothing to the arm comparison.
BASE = {
    0:  (0.00, -0.62, 0.00),   # nose
    11: (0.18, -0.50, 0.00),   # subject LEFT shoulder   -> drives the AVATAR's right arm
    12: (-0.18, -0.50, 0.00),  # subject RIGHT shoulder  -> drives the AVATAR's left arm
    23: (0.10, 0.00, 0.00),    # left hip
    24: (-0.10, 0.00, 0.00),   # right hip
    25: (0.11, 0.45, 0.00),    # left knee
    26: (-0.11, 0.45, 0.00),   # right knee
    27: (0.12, 0.90, 0.00),    # left ankle
    28: (-0.12, 0.90, 0.00),   # right ankle
}

UPPER = 0.28
FORE = 0.26


def arm(shoulder, upper_dir, fore_dir):
    """Return (elbow, wrist) from unit-ish direction vectors and the fixed segment lengths."""
    def unit(v):
        m = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5
        return (v[0] / m, v[1] / m, v[2] / m)
    u = unit(upper_dir)
    f = unit(fore_dir)
    elbow = (shoulder[0] + u[0] * UPPER, shoulder[1] + u[1] * UPPER, shoulder[2] + u[2] * UPPER)
    wrist = (elbow[0] + f[0] * FORE, elbow[1] + f[1] * FORE, elbow[2] + f[2] * FORE)
    return elbow, wrist


DOWN = (0.0, 1.0, 0.0)
UP = (0.0, -1.0, 0.0)
SUBJ_RIGHT = (-1.0, 0.0, 0.0)
SUBJ_LEFT = (1.0, 0.0, 0.0)
TOWARD_CAM = (0.0, 0.0, -1.0)
AWAY_CAM = (0.0, 0.0, 1.0)

# name -> (subject-right upper, subject-right fore, subject-left upper, subject-left fore, description)
POSES = {
    "relaxed": (DOWN, DOWN, DOWN, DOWN,
                "1. relaxed arms - both hanging straight down, elbows straight"),
    "tpose": (SUBJ_RIGHT, SUBJ_RIGHT, SUBJ_LEFT, SUBJ_LEFT,
              "2. T-pose - both arms straight out sideways, elbows straight"),
    "onehoriz": (SUBJ_RIGHT, SUBJ_RIGHT, DOWN, DOWN,
                 "3. one arm horizontal - subject's RIGHT arm out, left hanging"),
    "elbow90": (SUBJ_RIGHT, TOWARD_CAM, DOWN, DOWN,
                "4. elbow bent 90 deg - right upper arm out, forearm forward"),
    "overhead": ((-0.15, -1.0, 0.0), (-0.15, -1.0, 0.0), (0.15, -1.0, 0.0), (0.15, -1.0, 0.0),
                 "5. both arms overhead - straight up, slightly splayed"),
    "behind": ((-0.15, 0.85, 0.45), (0.75, 0.30, 0.55), DOWN, DOWN,
               "6. arm behind back - right hand tucked behind the torso"),
}
ORDER = ["relaxed", "tpose", "onehoriz", "elbow90", "overhead", "behind"]


def build(pose_name):
    ru, rf, lu, lf, _desc = POSES[pose_name]
    lm = [[0.0, 0.0, 0.0, 0.0] for _ in range(NUM)]
    src = [1] * NUM
    pts = dict(BASE)
    r_elbow, r_wrist = arm(BASE[12], ru, rf)     # subject's RIGHT arm  -> avatar's LEFT
    l_elbow, l_wrist = arm(BASE[11], lu, lf)     # subject's LEFT arm   -> avatar's RIGHT
    pts[14], pts[16] = r_elbow, r_wrist
    pts[13], pts[15] = l_elbow, l_wrist
    for idx, p in pts.items():
        lm[idx] = [round(p[0], 5), round(p[1], 5), round(p[2], 5), 0.95]
    for i in range(NUM):
        if lm[i][3] <= 0.0:
            src[i] = 0
    return lm, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--hold", type=float, default=6.0, help="seconds to hold each pose")
    ap.add_argument("--pose", default="", help="single pose name (see --list)")
    ap.add_argument("--all", action="store_true", help="walk every pose in order")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--marker", default="pose_stage_marks.json",
                    help="write each pose's epoch window here, for the Unity-side snapshot")
    a = ap.parse_args()

    if a.list:
        for n in ORDER:
            print("  %-10s %s" % (n, POSES[n][4]))
        return 0

    names = ORDER if a.all else ([a.pose] if a.pose in POSES else [])
    if not names:
        print("pick --pose <name> or --all  (see --list)")
        return 2

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (a.host, a.port)
    period = 1.0 / a.fps
    # Seed the sequence number from the wall clock, NOT from zero.
    #
    # P1-3's PoseBuffer.Push rejects any packet whose seq is below the newest it holds
    # ("a late packet cannot be inserted mid-ring") — correct for a real-time buffer, but it means a
    # driver that restarts at seq 0 while Unity stays in Play has EVERY packet dropped as out-of-order
    # until it counts back up past the previous run. Measured here: recv climbed to 6847 with
    # parseErr 0 while the applied frame sat 60 s stale. Seeding from the clock makes seq monotonic
    # across restarts, so each pose switch takes effect immediately.
    seq = int((time.time() - 1788900000.0) * 100.0)
    marks = []
    print("=" * 72)
    print(" ARM V2 FORENSICS - deterministic pose driver -> %s:%d" % (a.host, a.port))
    print("=" * 72)
    for name in names:
        lm, src = build(name)
        t0 = time.time()
        print("\n %s" % POSES[name][4])
        print("   holding %.1f s ..." % a.hold)
        # Write the marker at the START of the pose, not the end. The Unity-side snapshot reads this
        # file to label itself and must be able to do so WHILE the pose is still streaming: a snapshot
        # taken after the sender exits reads a STALE frame, because Apply stops at poseStaleSeconds
        # (0.5 s) and the driver then keeps the previous pose's landmarks.
        with open(a.marker, "w") as mf:
            json.dump({"poses": marks + [{"pose": name, "desc": POSES[name][4],
                                          "tStart": round(t0, 4), "tEnd": 0.0}]}, mf, indent=2)
        while time.time() - t0 < a.hold:
            msg = {"lm": lm, "xyz": [0.0, 0.0, 2500.0], "src": src,
                   "seq": seq, "t": round(time.time(), 4)}
            sock.sendto(json.dumps(msg).encode("utf-8"), addr)
            seq += 1
            time.sleep(period)
        t1 = time.time()
        marks.append({"pose": name, "desc": POSES[name][4],
                      "tStart": round(t0, 4), "tEnd": round(t1, 4)})
        print("   done (%d frames)" % seq)

    with open(a.marker, "w") as f:
        json.dump({"poses": marks}, f, indent=2)
    print("\npose windows -> %s" % a.marker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
