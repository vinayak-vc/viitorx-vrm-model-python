#!/usr/bin/env python3
"""TORSO YAW V4 -- guided LIVE OAK-D capture for the brief's motion list.

Prompts the subject through each motion with a countdown and writes wall-clock block boundaries to
`oak_v4_evidence/guided_marks.json`. The OAK sidecar must ALREADY be streaming (it owns the camera)
and Unity must be in Play with the V4 sampler armed; this script only paces the human and records
when each motion happened, so the Unity trace and the sidecar log can both be cut per motion.

Blocks cover the brief exactly:
  torso    -- slow yaw, fast yaw, left/right turns, repeated turns, ~90 deg, ~180 deg
  arm safety -- torso static + arms, torso rotating + arms static, both, fast both

    python oak_guided_v4.py [--rehearse]
"""
import argparse
import json
import os
import sys
import time

BLOCKS = [
    ("rest",          8, "STAND STILL, face the camera squarely. Arms relaxed at your sides."),
    ("slow_yaw",     14, "Turn your torso SLOWLY left, then slowly back to centre, then slowly right."),
    ("fast_yaw",     12, "Turn your torso FAST left-right-left-right. Keep your feet planted."),
    ("turn_lr",      14, "Turn to face LEFT, hold 2s, return to centre, face RIGHT, hold 2s, return."),
    ("repeat_lr",    14, "REPEATEDLY turn left and right, about one full cycle per second."),
    ("yaw_90",       12, "Turn ~90 degrees to your LEFT, hold 4s, then return to centre."),
    ("yaw_180",      12, "Turn ~180 degrees (show your BACK to the camera), hold 4s, then return."),
    ("arm_only",     12, "TORSO STILL, facing camera. Move both ARMS freely - raise, lower, wave."),
    ("torso_only",   12, "ARMS STILL at your sides. Rotate your TORSO left and right."),
    ("torso_arms",   12, "BOTH: rotate your torso AND move your arms at the same time."),
    ("fast_both",    12, "FAST: rotate your torso quickly while waving both arms quickly."),
]


def countdown(msg, secs):
    print("\n" + "=" * 78)
    print("  %s" % msg)
    print("=" * 78)
    for i in (3, 2, 1):
        sys.stdout.write("\r  starting in %d ... " % i)
        sys.stdout.flush()
        time.sleep(1.0)
    print("\r  GO" + " " * 40)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearse", action="store_true", help="print the script without timing it")
    ap.add_argument("--out", default=os.path.join("oak_v4_evidence", "guided_marks.json"))
    a = ap.parse_args()

    total = sum(b[1] for b in BLOCKS) + 3 * len(BLOCKS)
    print("=" * 78)
    print(" GUIDED OAK-D TORSO YAW CAPTURE -- %d blocks, about %d s total" % (len(BLOCKS), total))
    print(" Stand about 2 m from the OAK-D, full body in frame.")
    print("=" * 78)
    if a.rehearse:
        for name, secs, instr in BLOCKS:
            print("  %-12s %3ds  %s" % (name, secs, instr))
        return 0

    print("\n  Get into position: stand about 2 m back, FULL BODY in frame, facing the camera.")
    for i in range(12, 0, -1):
        sys.stdout.write("\r  capture starts in %2d s ... " % i)
        sys.stdout.flush()
        time.sleep(1.0)
    print("\r  starting" + " " * 40)

    marks = []
    t_start = time.time()
    for name, secs, instr in BLOCKS:
        countdown(instr, secs)
        t0 = time.time()
        for r in range(secs, 0, -1):
            sys.stdout.write("\r  %-12s %2ds remaining " % (name, r))
            sys.stdout.flush()
            time.sleep(1.0)
        t1 = time.time()
        marks.append({"name": name, "t0": round(t0, 4), "t1": round(t1, 4), "instr": instr})
        print("\r  %-12s done (%.1fs)          " % (name, t1 - t0))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"start": round(t_start, 4), "end": round(time.time(), 4), "blocks": marks}, f, indent=2)
    print("\n" + "=" * 78)
    print(" DONE. %d blocks, %.0f s -> %s" % (len(marks), time.time() - t_start, a.out))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
