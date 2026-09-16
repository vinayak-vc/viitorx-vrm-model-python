#!/usr/bin/env python3
"""ARM RETARGET V3 TASK 6 — hold ONE recorded video frame on the wire and snapshot every layer.

The frame's landmark payload is replayed byte-for-byte from what video_udp_sender.py actually emitted
(see wire_record.py), so the avatar is driven by the real RTMW3D output for that exact video frame —
not by a re-derivation of it. Because the frame is HELD static, the per-frame slerp fully converges and
the capture is not a moving target: measurement and render describe the same pose.

    python run_frames.py 279 359 273 177 24
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ubridge

REPO = r"C:\Unity\viitorx-vrm-avtar-unity-base-project\Assets\Games\viitorx-vrm-avtar-unity"
EVID = os.path.join(REPO, "python-sidecar~", "arm_v3_evidence")
DUMP = os.path.join(HERE, "wire_p0.jsonl")
SETTLE_S, HOLD_S = 5.0, 22.0


def _seq0():
    with open(DUMP, encoding="utf-8") as fh:
        return json.loads(fh.readline())["msg"]["seq"]


def run(body, label):
    return ubridge.execute_code(body.replace("LABEL", label).replace("SHOTNAME", label))


def main():
    idxs = [int(x) for x in sys.argv[1:]]
    seq0 = _seq0()
    by_idx = {}
    with open(DUMP, encoding="utf-8") as fh:
        for l in fh:
            r = json.loads(l)
            by_idx[r["index"]] = r["msg"]["seq"]
    measure = open(os.path.join(HERE, "measure.cs.txt"), encoding="utf-8").read()
    shot = open(os.path.join(HERE, "shot.cs.txt"), encoding="utf-8").read()
    os.makedirs(EVID, exist_ok=True)
    for idx in idxs:
        if idx not in by_idx:
            print(f"index {idx} not recorded, skipping", flush=True)
            continue
        frame = 60 + (by_idx[idx] - seq0)      # 1-based video frame
        label = f"vid_f{frame:03d}_idx{idx:03d}"
        print(f"\n=== {label}  (annotation f={frame - 1}, t={(frame - 1) / 30.0:.2f}s) ===", flush=True)
        proc = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "wire_record.py"), "hold",
             "--dump", DUMP, "--index", str(idx), "--seconds", str(HOLD_S)],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        try:
            time.sleep(SETTLE_S)
            r = run(measure, label)
            print("  measure:", (r.get("data") or {}).get("result", r), flush=True)
            s = run(shot, label)
            print("  shot   :", str((s.get("data") or {}).get("result", s))[-90:], flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        time.sleep(0.5)
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
