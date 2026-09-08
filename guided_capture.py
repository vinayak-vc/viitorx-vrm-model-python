#!/usr/bin/env python3
"""P0 HUMAN ACCEPTANCE — guided, block-segmented capture.

Runs the REAL OAK-D sidecar as a subprocess while walking the operator through test
blocks A-J on a countdown, recording each block's epoch start/end to blocks.json.
The analyzer then reports EVERY metric per block, which a single undifferentiated
log cannot do.

Nothing here touches the P0 implementation: the sidecar is launched unmodified with
the shipping arguments, and block boundaries are pure wall-clock bookkeeping against
the epoch `t` already present in sender_log.jsonl.

Usage (Unity must be in PLAY with pipelineLogging = ON):
    python guided_capture.py
    python guided_capture.py --dry-run          # rehearse prompts, no camera
    python guided_capture.py --blocks A,E,I     # only some blocks
"""
import argparse
import json
import os
import subprocess
import sys
import time

MODEL_DEFAULT = os.path.join("..", "..", "..", "SentisModel", "rtmw3d-x.onnx")

# Shipping P0 arguments. Do NOT tune these during an acceptance run.
SIDECAR_ARGS = [
    "--min-cutoff", "0.5",
    "--beta", "0.4",
    "--depth-min-cutoff", "0.3",
    "--depth-beta", "0.1",
    "--max-hold-frames", "8",
]

# (key, seconds, title, instruction lines)
BLOCKS = [
    ("A", 15, "STATIC",
     ["Stand still, arms relaxed at your sides.",
      "Full body in frame. Do not move."]),
    ("B", 20, "NORMAL WALKING",
     ["Walk TOWARD the camera, then AWAY.",
      "Then walk ACROSS the frame, left to right and back."]),
    ("C", 20, "FAST ARM MOVEMENT",
     ["Fast waving, both arms.",
      "Large arm swings. Then reach OVERHEAD repeatedly."]),
    ("D", 20, "FAST LEG MOVEMENT",
     ["Kicks, then squats.",
      "Rapid knee bends, then fast stepping."]),
    ("E", 30, "HAND BEHIND TORSO  *** CRITICAL ***",
     ["LEFT hand behind your back:  hold 1s, release.",
      "Again: hold 3s, release.   Again: hold 6s, release.",
      "Now repeat all three with the RIGHT hand."]),
    ("F", 20, "ELBOW OCCLUSION",
     ["Cross arms tightly over your torso, hold 5s, release.",
      "Repeat with the other arm on top. 3x each side."]),
    ("G", 20, "SIDE-ON POSE",
     ["Turn 90 degrees LEFT, hold 5s, return to front.",
      "Turn 90 degrees RIGHT, hold 5s, return to front."]),
    ("H", 20, "LEGS CROSSED",
     ["Cross LEFT leg behind RIGHT, hold 5s, release.",
      "Cross RIGHT leg behind LEFT, hold 5s, release. Repeat."]),
    ("I", 25, "LONG OCCLUSION  *** CRITICAL ***",
     ["Hide your LEFT arm fully behind your back for a FULL 10 SECONDS.",
      "Then bring it back out sharply.",
      "Repeat with the RIGHT arm for 10 seconds."]),
    ("J", 30, "DANCING  (production stress test)",
     ["Dance freely: fast arms, fast legs, body turns,",
      "crossed limbs, rapid direction changes."]),
]


def countdown(seconds, label):
    end = time.time() + seconds
    while True:
        left = end - time.time()
        if left <= 0:
            break
        sys.stdout.write("\r    %-52s %4.1fs remaining " % (label, left))
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r    %-52s %-20s\n" % (label, "done"))
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--log-dir", default="pipeline_logs")
    ap.add_argument("--blocks", default="", help="comma-separated subset, e.g. A,E,I")
    ap.add_argument("--dry-run", action="store_true", help="rehearse prompts without the camera")
    ap.add_argument("--lead-in", type=float, default=8.0, help="seconds to get into position")
    a = ap.parse_args()

    wanted = [b.strip().upper() for b in a.blocks.split(",") if b.strip()]
    blocks = [b for b in BLOCKS if (not wanted or b[0] in wanted)]
    if not blocks:
        print("no blocks selected")
        return 2

    total = sum(b[1] for b in blocks) + a.lead_in + 3.0
    py = os.path.join(".venv", "Scripts", "python.exe")
    if not os.path.exists(py):
        py = sys.executable

    print("=" * 66)
    print(" P0 HUMAN ACCEPTANCE - GUIDED CAPTURE")
    print("=" * 66)
    print(" Blocks   : %s" % ", ".join(b[0] for b in blocks))
    print(" Duration : %.0f s" % total)
    print(" Log dir  : %s" % a.log_dir)
    print()
    print(" CHECKLIST before you start:")
    print("   [ ] Unity is in PLAY mode, Bootstrap.unity loaded")
    print("   [ ] pipelineLogging = ON, the VRM avatar is visible on screen")
    print("   [ ] You are 2.0-2.5 m from the OAK-D, FULL BODY in frame")
    print("   [ ] Hands and feet visible; normal room lighting (do NOT optimise it)")
    print()
    print(" WATCH THE AVATAR (not the debug skeleton) throughout, and note per body")
    print(" part: JITTER / SPIKE / FREEZE / COLLAPSE / FLIP / TELEPORT / LATENCY / NORMAL")
    print("=" * 66)
    try:
        raw_input_fn = raw_input  # noqa: F821  (py2)
    except NameError:
        raw_input_fn = input
    raw_input_fn(" Press ENTER when you are in position and ready ... ")

    proc = None
    if not a.dry_run:
        if not os.path.exists(a.model):
            print("[ERROR] model not found: %s" % a.model)
            return 1
        os.makedirs(a.log_dir, exist_ok=True)
        cmd = [py, "wholebody_udp_sender.py", "--model", a.model,
               "--log-dir", a.log_dir, "--show",
               "--seconds", str(int(total) + 5)] + SIDECAR_ARGS
        print("\n[starting sidecar] %s\n" % " ".join(cmd))
        proc = subprocess.Popen(cmd)
        # The RTMW3D ONNX session needs to be up before block A is timed, otherwise
        # the lead-in silently absorbs model load and block A starts late.
        print("  waiting for the model to load and the camera to open ...")
        deadline = time.time() + 120.0
        sender_log = os.path.join(a.log_dir, "sender_log.jsonl")
        start_size = os.path.getsize(sender_log) if os.path.exists(sender_log) else -1
        while time.time() < deadline:
            if proc.poll() is not None:
                print("[ERROR] sidecar exited early (code %s)" % proc.returncode)
                return 1
            if os.path.exists(sender_log) and os.path.getsize(sender_log) != start_size:
                break
            time.sleep(0.5)
        print("  sidecar streaming.\n")

    countdown(a.lead_in, "LEAD-IN - get into position, full body in frame")

    records = []
    for key, secs, title, lines in blocks:
        print()
        print("-" * 66)
        print(" BLOCK %s - %s   (%d s)" % (key, title, secs))
        for ln in lines:
            print("   * %s" % ln)
        print("-" * 66)
        t0 = time.time()
        countdown(secs, "BLOCK %s" % key)
        t1 = time.time()
        records.append({"block": key, "title": title,
                        "tStart": round(t0, 4), "tEnd": round(t1, 4),
                        "seconds": round(t1 - t0, 2)})

    if proc is not None:
        print("\n[stopping sidecar] ...")
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.terminate()

    out = os.path.join(a.log_dir, "blocks.json")
    with open(out, "w") as f:
        json.dump(records, f, indent=2)
    print("\nblock boundaries -> %s" % out)

    print("\n" + "=" * 66)
    print(" STOP Unity play mode now, then the analysis runs.")
    print("=" * 66)
    if not a.dry_run:
        subprocess.call([py, "analyze_capture.py", "--dir", a.log_dir,
                         "--blocks", out,
                         "--baseline", "pipeline_logs_baseline_audit",
                         "--label", "P0 HUMAN ACCEPTANCE"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
