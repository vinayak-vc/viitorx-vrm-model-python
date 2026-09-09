#!/usr/bin/env python3
"""ARM RETARGET V3 — drive the six deterministic poses and snapshot EVERY layer on the same frame.

For each pose: stream it on the real UDP wire, let the P0 gate validate and the per-frame slerp settle,
then in one editor call capture source + control + skinned bone + skinned-vertex mesh + screen-space,
and separately render the production camera. The render and the measurement are taken in consecutive
editor calls with the pose still streaming and static, so they describe the same pose.

Usage: python run_poses.py [pose ...]     (default: all six, in the brief's order)
"""
import json
import os
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ubridge

REPO = r"C:\Unity\viitorx-vrm-avtar-unity-base-project\Assets\Games\viitorx-vrm-avtar-unity"
SIDECAR = os.path.join(REPO, "python-sidecar~")
EVID = os.path.join(SIDECAR, "arm_v3_evidence")
POSES = ["relaxed", "tpose", "onehoriz", "elbow90", "overhead", "behind"]
SETTLE_S = 5.0          # P0 LimbGate validation + lerpAmount=0.5 convergence (0.5^n, n>=100 frames)
HOLD_S = 22.0           # must outlast SETTLE_S + both editor calls


def measure(label):
    with open(os.path.join(HERE, "measure.cs.txt"), encoding="utf-8") as fh:
        body = fh.read().replace("LABEL", label)
    return ubridge.execute_code(body)


def shot(label):
    with open(os.path.join(HERE, "shot.cs.txt"), encoding="utf-8") as fh:
        body = fh.read().replace("SHOTNAME", label)
    return ubridge.execute_code(body)


def main():
    want = sys.argv[1:] or POSES
    os.makedirs(EVID, exist_ok=True)
    for pose in want:
        print(f"\n=== {pose} ===", flush=True)
        # Stream the pose for the whole window. pose_stage_udp.py seeds seq from the wall clock, so a
        # fresh process cannot be rejected as out-of-order by P1-3's PoseBuffer (V2 forensics §7.3).
        proc = subprocess.Popen(
            [sys.executable, "pose_stage_udp.py", "--pose", pose, "--hold", str(HOLD_S)],
            cwd=SIDECAR, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        try:
            time.sleep(SETTLE_S)
            r = measure(f"pose_{pose}")
            print("  measure:", (r.get("data") or {}).get("result", r), flush=True)
            s = shot(f"pose_{pose}")
            print("  shot   :", (s.get("data") or {}).get("result", s), flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        time.sleep(1.0)
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
