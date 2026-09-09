#!/usr/bin/env python3
"""ARM RETARGET V1 -- LIVE VALIDATION capture (docs/UNITY_ARM_RETARGET_V1_2026-09-08.md §12).

Runs the REAL OAK-D sidecar as a subprocess while walking the operator through the nine arm
motions from the brief, recording each block's epoch start/end to blocks.json so every metric can
be reported PER MOTION.

The A/B is done INSIDE ONE sidecar session so the camera, the model warm-up, the lighting and the
subject's position are identical across the two variants:

    variant NEW  = Unity kalidokitAimArms = true   (ArmAimSolver)
    variant OLD  = Unity kalidokitAimArms = false  (Kalidokit RigArm, untouched)

The script cannot talk to Unity, so between the two variants it writes `awaiting_toggle.json` and
blocks until `toggle_ok.json` appears -- the operator's agent flips the Unity flag and drops that
file. Nothing about the sidecar changes between the variants.

NOTHING here changes the pipeline: no --recovery (P1-4 stays rejected/off), surface-aware depth
stays at its shipping default, and the sidecar arguments are the same shipping set used by every
previous validation run.

Usage (Unity must be in PLAY, pipelineLogging = ON, kalidokitDirectionTrace = ON):
    python arm_v1_live_capture.py --log-dir pipeline_logs_armv1 --both
    python arm_v1_live_capture.py --variant new --blocks 1,3,5      # subset
    python arm_v1_live_capture.py --dry-run --both                  # rehearse, no camera
"""
import argparse
import json
import os
import subprocess
import sys
import time

MODEL_DEFAULT = os.path.join("..", "..", "..", "SentisModel", "rtmw3d-x.onnx")

# Shipping arguments, IDENTICAL across both variants. Do NOT tune these during a run.
SIDECAR_ARGS = [
    "--min-cutoff", "0.5",
    "--beta", "0.4",
    "--depth-min-cutoff", "0.3",
    "--depth-beta", "0.1",
    "--max-hold-frames", "8",
]

# (key, seconds, title, instruction lines)
# The nine motions of the brief, in order. Wording is deliberately numbered and plain: an earlier
# capture failed because the operator could not tell what a block was asking for.
BLOCKS = [
    ("1", 15, "RELAXED ARMS",
     ["1) Stand facing the camera, feet slightly apart.",
      "2) Let BOTH arms hang straight down at your sides, relaxed.",
      "3) Stand still. Do not move your arms at all until the timer ends."]),

    ("2", 20, "ONE ARM HORIZONTAL",
     ["1) Raise your RIGHT arm straight out to the SIDE, shoulder height.",
      "   Keep the elbow straight. Hold it there for about 8 seconds.",
      "2) Lower it back to your side.",
      "3) Now do the same with your LEFT arm. Hold about 8 seconds.",
      "4) Lower it. The other arm stays hanging down the whole time."]),

    ("3", 15, "BOTH ARMS HORIZONTAL  (T-pose)",
     ["1) Raise BOTH arms straight out to the sides, shoulder height.",
      "2) Keep both elbows straight, palms facing down.",
      "3) Hold that T shape, still, for the whole block."]),

    ("4", 20, "ONE ARM OVERHEAD",
     ["1) Reach your RIGHT arm straight UP above your head.",
      "   Keep the elbow straight. Hold about 8 seconds.",
      "2) Lower it back to your side.",
      "3) Now reach your LEFT arm straight UP. Hold about 8 seconds.",
      "4) Lower it."]),

    ("5", 15, "BOTH ARMS OVERHEAD",
     ["1) Reach BOTH arms straight UP above your head.",
      "2) Keep both elbows straight, hands roughly shoulder-width apart.",
      "3) Hold that up until the timer ends."]),

    ("6", 20, "FAST WAVING",
     ["1) Raise both arms up in front of you, elbows a little bent.",
      "2) Wave BOTH hands fast from side to side, like signalling someone.",
      "3) Keep waving FAST for the whole block. Do not slow down."]),

    ("7", 20, "ARM CIRCLES",
     ["1) For the first 10 seconds: swing BOTH arms in big FORWARD circles.",
      "   Big full circles - all the way up past your head and back down.",
      "2) For the last 10 seconds: swing them in big BACKWARD circles.",
      "   Keep the elbows straight so it is a clean circle."]),

    ("8", 20, "ASYMMETRIC ARM POSES",
     ["Four poses, about 5 seconds each. The two arms are always different.",
      "1) RIGHT arm straight UP,  LEFT arm straight OUT to the side.",
      "2) Swap: LEFT arm straight UP,  RIGHT arm straight OUT to the side.",
      "3) RIGHT arm straight FORWARD at the camera, LEFT arm hanging DOWN.",
      "4) RIGHT arm out to the side, LEFT hand touching your LEFT shoulder."]),

    ("9", 25, "DANCE",
     ["1) Just dance, using your ARMS a lot.",
      "2) Reach overhead, cross your arms in front, bend the elbows,",
      "   swing them out to the sides. Keep changing.",
      "3) Turn your body a little left and right as you go.",
      "The more arm movement the better - this is the stress test."]),
]

# Supplementary dense-trace pass (see the report's jitter section). Same three roll-critical
# motions, run with the Unity trace throttle turned down so a per-frame roll step is measurable.
DENSE_BLOCKS = [
    ("D1", 15, "ARM CIRCLES  (dense trace)",
     ["1) Swing BOTH arms in big FORWARD circles, all the way over your head.",
      "2) Keep going for the whole 15 seconds. Keep elbows straight."]),
    ("D2", 15, "STRAIGHT ARMS HELD OUT  (dense trace)",
     ["1) Hold BOTH arms straight out to the sides, shoulder height.",
      "2) Lock the elbows STRAIGHT and hold completely still.",
      "This is the near-straight-elbow case - it is meant to be boring."]),
    ("D3", 15, "FAST WAVING  (dense trace)",
     ["1) Both arms up, wave BOTH hands fast from side to side.",
      "2) Keep it fast for the whole 15 seconds."]),
]

AWAIT_FILE = "awaiting_toggle.json"
OK_FILE = "toggle_ok.json"


def countdown(seconds, label):
    t0 = time.time()
    end = t0 + seconds
    while True:
        left = end - time.time()
        if left <= 0:
            break
        sys.stdout.write("\r    %-52s %4.1fs remaining " % (label, left))
        sys.stdout.flush()
        time.sleep(0.05)
    sys.stdout.write("\r    %-52s %-20s\n" % (label, "done"))
    sys.stdout.flush()


def run_sequence(blocks, variant, prep, records, dry_run):
    """Walk the operator through one variant's blocks; append to `records`."""
    print()
    print("#" * 68)
    print(" VARIANT %s  --  %s" % (variant.upper(), VARIANT_DESC[variant]))
    print("#" * 68)
    for key, secs, title, lines in blocks:
        print()
        print("-" * 68)
        print(" MOTION %s - %s   (%d s)" % (key, title, secs))
        for ln in lines:
            print("   %s" % ln)
        print("-" * 68)
        if prep > 0:
            countdown(prep, ">>> READ THE STEPS - GET IN POSITION")
        print("   %s" % ("-" * 62))
        print("   *** GO - START DOING IT NOW ***")
        t0 = time.time()
        countdown(secs, "MOTION %s RECORDING" % key)
        t1 = time.time()
        records.append({"block": key, "variant": variant, "title": title,
                        "tStart": round(t0, 4), "tEnd": round(t1, 4),
                        "seconds": round(t1 - t0, 2)})


VARIANT_DESC = {
    "new": "kalidokitAimArms = TRUE   (ArmAimSolver)",
    "old": "kalidokitAimArms = FALSE  (Kalidokit RigArm, the old branch)",
    "dense": "kalidokitAimArms = TRUE, trace throttle lowered",
}


def wait_for_toggle(log_dir, want_variant, timeout=420.0):
    """Hand control to the agent driving Unity, then wait for it to confirm the flag flip."""
    await_path = os.path.join(log_dir, AWAIT_FILE)
    ok_path = os.path.join(log_dir, OK_FILE)
    for p in (await_path, ok_path):
        if os.path.exists(p):
            os.remove(p)
    with open(await_path, "w") as f:
        json.dump({"want": want_variant, "t": round(time.time(), 4)}, f)
    print()
    print("=" * 68)
    print(" PAUSED - switching the avatar to variant %s." % want_variant.upper())
    print(" You can rest your arms. This takes a few seconds.")
    print("=" * 68)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(ok_path):
            try:
                with open(ok_path) as f:
                    info = json.load(f)
            except Exception:
                info = {}
            print(" switched: %s" % json.dumps(info))
            os.remove(ok_path)
            if os.path.exists(await_path):
                os.remove(await_path)
            return True
        sys.stdout.write("\r    waiting for the switch ...  %4.0fs " % (deadline - time.time()))
        sys.stdout.flush()
        time.sleep(0.25)
    print("\n [ERROR] timed out waiting for the variant switch")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--log-dir", default="pipeline_logs_armv1")
    ap.add_argument("--variant", choices=["new", "old"], default="new",
                    help="which Unity arm branch this sequence is labelled as; the flag itself "
                         "is set in Unity, not here.")
    ap.add_argument("--both", action="store_true",
                    help="run NEW then OLD in ONE sidecar session, pausing between for the "
                         "kalidokitAimArms toggle. This is the controlled A/B.")
    ap.add_argument("--dense", action="store_true",
                    help="append the supplementary dense-trace sequence after the A/B.")
    ap.add_argument("--blocks", default="", help="comma-separated subset, e.g. 1,3,7")
    ap.add_argument("--dry-run", action="store_true", help="rehearse prompts without the camera")
    ap.add_argument("--lead-in", type=float, default=8.0)
    ap.add_argument("--prep", type=float, default=6.0,
                    help="seconds shown BEFORE each motion, with that motion's steps on screen, "
                         "so the operator can read them and get into position.")
    a = ap.parse_args()

    wanted = [b.strip() for b in a.blocks.split(",") if b.strip()]
    blocks = [b for b in BLOCKS if (not wanted or b[0] in wanted)]
    if not blocks:
        print("no blocks selected")
        return 2

    seq_count = 2 if a.both else 1
    per_seq = sum(b[1] for b in blocks) + a.prep * len(blocks)
    total = a.lead_in + per_seq * seq_count + 20.0
    if a.dense:
        total += sum(b[1] for b in DENSE_BLOCKS) + a.prep * len(DENSE_BLOCKS) + 20.0
    py = os.path.join(".venv", "Scripts", "python.exe")
    if not os.path.exists(py):
        py = sys.executable

    print("=" * 68)
    print(" ARM RETARGET V1 - LIVE VALIDATION CAPTURE")
    print("=" * 68)
    print(" Sequence : %s" % ("NEW then OLD (controlled A/B)" if a.both else a.variant.upper()))
    print(" Motions  : %s" % ", ".join(b[0] for b in blocks))
    print(" Dense    : %s" % ("yes, appended" if a.dense else "no"))
    print(" Duration : about %.0f s total (%.0f s per sequence)" % (total, per_seq))
    print(" Log dir  : %s" % a.log_dir)
    print(" Sidecar  : shipping default - NO --recovery, surface-aware depth ON")
    print(" Unity    : must be in PLAY, pipelineLogging ON, kalidokitDirectionTrace ON")
    print("=" * 68)

    proc = None
    if not a.dry_run:
        if not os.path.exists(a.model):
            print("[ERROR] model not found: %s" % a.model)
            return 1
        os.makedirs(a.log_dir, exist_ok=True)
        cmd = [py, "wholebody_udp_sender.py", "--model", a.model,
               "--log-dir", a.log_dir, "--show",
               "--seconds", str(int(total) + 30)] + SIDECAR_ARGS
        print("\n[starting sidecar] %s\n" % " ".join(cmd))
        proc = subprocess.Popen(cmd)
        print("  waiting for the model to load and the camera to open ...")
        deadline = time.time() + 180.0
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
    else:
        os.makedirs(a.log_dir, exist_ok=True)

    countdown(a.lead_in, "LEAD-IN - stand facing the camera, whole body in frame")

    records = []
    order = ["new", "old"] if a.both else [a.variant]
    for i, variant in enumerate(order):
        if i > 0:
            if not wait_for_toggle(a.log_dir, variant):
                break
        run_sequence(blocks, variant, a.prep, records, a.dry_run)

    if a.dense:
        if wait_for_toggle(a.log_dir, "dense"):
            run_sequence(DENSE_BLOCKS, "dense", a.prep, records, a.dry_run)

    if proc is not None:
        print("\n[stopping sidecar] ...")
        try:
            proc.wait(timeout=40)
        except Exception:
            proc.terminate()

    out = os.path.join(a.log_dir, "blocks.json")
    with open(out, "w") as f:
        json.dump({"task": "ARM_RETARGET_V1_LIVE",
                   "recovery": "off (shipping default)",
                   "sampler": "surface-aware (shipping default)",
                   "blocks": records}, f, indent=2)
    print("\nblock boundaries -> %s" % out)
    print("\n" + "=" * 68)
    print(" DONE. Leave Unity in PLAY until told otherwise.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
