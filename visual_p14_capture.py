#!/usr/bin/env python3
"""P1-4 VISUAL VALIDATION -- guided, block-segmented capture (P1-3 vs P1-4 A/B).

Runs the REAL OAK-D sidecar as a subprocess while walking the operator through the
nine P1-4 visual-validation blocks on a countdown, recording each block's epoch
start/end to blocks.json so every metric can be reported PER BLOCK.

The ONLY difference between the two passes is --no-recovery (P1-3 baseline) vs
--recovery (P1-4). Everything else -- P0 filtering, P0-2 caps, P1-1 tracker,
P1-2 latest-frame, P1-3 Unity pose buffer -- is identical and untouched.

Block 8 fires a TEST-ONLY landmark injection (see --inject-drift-file in the sender)
so the "confidently wrong" path is exercised on demand at a KNOWN timestamp, rather
than hoping nature produces one during the run.

Usage (Unity must be in PLAY with pipelineLogging = ON):
    python visual_p14_capture.py --pass p13
    python visual_p14_capture.py --pass p14
    python visual_p14_capture.py --pass p13 --dry-run
"""
import argparse
import json
import os
import subprocess
import sys
import time

MODEL_DEFAULT = os.path.join("..", "..", "..", "SentisModel", "rtmw3d-x.onnx")

# Shipping arguments, IDENTICAL across both passes. Do NOT tune these during a run.
SIDECAR_ARGS = [
    "--min-cutoff", "0.5",
    "--beta", "0.4",
    "--depth-min-cutoff", "0.3",
    "--depth-beta", "0.1",
    "--max-hold-frames", "8",
]

INJECT_FILE = "inject_spec.json"

# Right knee (COCO-wholebody 14). 0.85 m ~= 2x shin length -- the same magnitude as the
# corrected D2 fixture, i.e. unambiguously impossible rather than borderline.
INJECTIONS = [
    (4.0, {"mode": "drift", "joint": 14, "meters": 0.85, "frames": 25}),
    (14.0, {"mode": "teleport", "joint": 14, "meters": 0.85, "frames": 10}),
]

# (key, seconds, title, instruction lines)
BLOCKS = [
    ("1", 15, "STAND STILL",
     ["Stand facing the camera, feet slightly apart.",
      "Let both arms hang down at your sides.",
      "Stay as still as you can. Do not shift your weight."]),
    ("2", 20, "WALK TOWARD / AWAY",
     ["Walk SLOWLY toward the camera until you are about 1 metre away.",
      "Then walk SLOWLY backwards until your whole body is in frame again.",
      "Repeat that twice. Keep facing the camera the whole time."]),
    ("3", 20, "FAST ARMS",
     ["1) Wave BOTH arms fast, like you are signalling someone.",
      "2) Then swing both arms in big circles.",
      "3) Then reach both hands straight UP overhead and back down,",
      "   fast, over and over until the timer ends."]),
    ("4", 20, "FAST LEGS",
     ["1) Kick your RIGHT leg forward, then your LEFT. Keep alternating.",
      "2) Then do about 5 squats.",
      "3) Then march on the spot, lifting your knees HIGH and fast."]),
    ("5", 25, "DANCE  (hardest test - be energetic)",
     ["Just dance. Move arms and legs at the same time.",
      "Turn your body left and right. Change direction often.",
      "The more energetic the better - this is the stress test."]),
    ("6", 30, "HAND BEHIND TORSO",
     ["1) LEFT hand behind your back. Hold 3 seconds. Return it to your side.",
      "2) LEFT hand behind your back again. Hold 8 seconds. Return it.",
      "3) Now RIGHT hand behind your back. Hold 3 seconds. Return it.",
      "4) RIGHT hand behind your back again. Hold 8 seconds. Return it."]),
]


# F-08 AUDIT (Part 9): a SUSTAINED occlusion, long enough that the wrist stays wrong well past
# P1-1's 6-frame prediction horizon -- the "confident-but-wrong" case, on demand.
AUDIT_BLOCK = ("10", 45, "LONG HOLD - HAND BEHIND BACK",
               ["1) LEFT hand behind your back. KEEP IT THERE for 20 SECONDS.",
                "   Count slowly to 20. Do not bring it out early.",
                "   Stand otherwise still, facing the camera.",
                "2) Bring it out. Rest 5 seconds, arms at your sides.",
                "3) Now RIGHT hand behind your back for the last 15 seconds."])


def countdown(seconds, label, triggers=None, fired=None):
    """Countdown that also fires (offsetSeconds, callable) triggers exactly once."""
    t0 = time.time()
    end = t0 + seconds
    pending = list(triggers or [])
    while True:
        now = time.time()
        left = end - now
        if left <= 0:
            break
        el = now - t0
        while pending and el >= pending[0][0]:
            _off, fn = pending.pop(0)
            stamp = fn()
            if fired is not None and stamp is not None:
                fired.append(stamp)
        sys.stdout.write("\r    %-52s %4.1fs remaining " % (label, left))
        sys.stdout.flush()
        time.sleep(0.05)
    sys.stdout.write("\r    %-52s %-20s\n" % (label, "done"))
    sys.stdout.flush()


def make_injector(spec, log_dir):
    def fire():
        path = os.path.join(log_dir, INJECT_FILE)
        try:
            with open(path, "w") as f:
                json.dump(spec, f)
        except Exception as e:
            print("\n  [inject] write failed: %s" % e)
            return None
        t = time.time()
        print("\n  [inject] %s joint=%d %.2fm x%d frames  @ %.4f"
              % (spec["mode"], spec["joint"], spec["meters"], spec["frames"], t))
        return {"t": round(t, 4), "spec": spec}
    return fire


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--log-dir", default="pipeline_logs")
    ap.add_argument("--pass", dest="which", required=True,
                    choices=["p13", "p14", "rollback"],
                    help="p13 = explicit --no-recovery; p14 = explicit --recovery; "
                         "rollback = pass NO recovery flag at all, i.e. exercise the SHIPPING "
                         "DEFAULT (proves normal startup needs no --no-recovery).")
    ap.add_argument("--blocks", default="", help="comma-separated subset, e.g. 1,8")
    ap.add_argument("--dry-run", action="store_true", help="rehearse prompts without the camera")
    ap.add_argument("--lead-in", type=float, default=8.0)
    ap.add_argument("--legacy-depth", action="store_true",
                    help="F-08 A/B: run the sidecar with --no-surface-depth (the pre-F-08 whole-window "
                         "percentile sampler). Block wording is unchanged, so the two passes differ "
                         "ONLY in the sampler.")
    ap.add_argument("--prep", type=float, default=7.0,
                    help="seconds shown BEFORE each block, with that block's instructions on "
                         "screen, so the operator can read them and get into position.")
    ap.add_argument("--audit", action="store_true",
                    help="F-08 AUDIT: add --audit-log to the sidecar and append block 10, a "
                         "SUSTAINED hand-behind-torso hold for the confident-but-wrong root-cause "
                         "trace (Part 9).")
    a = ap.parse_args()

    all_blocks = list(BLOCKS)
    if a.audit:
        all_blocks.append(AUDIT_BLOCK)
    wanted = [b.strip() for b in a.blocks.split(",") if b.strip()]
    blocks = [b for b in all_blocks if (not wanted or b[0] in wanted)]
    if not blocks:
        print("no blocks selected")
        return 2

    # rollback deliberately passes NOTHING, so the run exercises the shipping default.
    recovery_flag = {"p14": "--recovery", "p13": "--no-recovery"}.get(a.which, "")
    total = sum(b[1] for b in blocks) + a.lead_in + 3.0 + a.prep * len(blocks)
    py = os.path.join(".venv", "Scripts", "python.exe")
    if not os.path.exists(py):
        py = sys.executable

    print("=" * 68)
    print(" P1-4 VISUAL VALIDATION - %s PASS  (%s)"
          % (a.which.upper(), recovery_flag or "SHIPPING DEFAULT, no recovery flag"))
    print("=" * 68)
    print(" Blocks   : %s" % ", ".join(b[0] for b in blocks))
    print(" Duration : %.0f s" % total)
    print(" Log dir  : %s" % a.log_dir)
    print(" Sampler  : %s" % ("LEGACY (--no-surface-depth)" if a.legacy_depth
                              else "SURFACE-AWARE (shipping default)"))
    print(" Unity    : must be in PLAY, pipelineLogging = ON, avatar visible")
    print("=" * 68)

    proc = None
    if not a.dry_run:
        if not os.path.exists(a.model):
            print("[ERROR] model not found: %s" % a.model)
            return 1
        os.makedirs(a.log_dir, exist_ok=True)
        stale = os.path.join(a.log_dir, INJECT_FILE)
        if os.path.exists(stale):
            os.remove(stale)
        cmd = [py, "wholebody_udp_sender.py", "--model", a.model,
               "--log-dir", a.log_dir, "--show"]
        if recovery_flag:
            cmd.append(recovery_flag)
        if a.audit:
            cmd.append("--audit-log")
        if a.legacy_depth:
            cmd.append("--no-surface-depth")
        cmd += ["--inject-drift-file", stale,
                "--seconds", str(int(total) + 5)] + SIDECAR_ARGS
        print("\n[starting sidecar] %s\n" % " ".join(cmd))
        proc = subprocess.Popen(cmd)
        # Wait out the ONNX/DirectML warm-up so block 1 is not timed against model load.
        print("  waiting for the model to load and the camera to open ...")
        deadline = time.time() + 150.0
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

    records, injections = [], []
    for key, secs, title, lines in blocks:
        print()
        print("-" * 68)
        print(" BLOCK %s - %s   (%d s)" % (key, title, secs))
        for ln in lines:
            print("   * %s" % ln)
        print("-" * 68)
        trig = None
        if key == "8" and not a.dry_run:
            trig = [(off, make_injector(spec, a.log_dir)) for off, spec in INJECTIONS]
        # Read-and-position gap. The block's own timer does not start until this ends, so the
        # operator is never mid-transition while a block is being measured.
        if a.prep > 0:
            countdown(a.prep, ">>> READ THE STEPS - GET IN POSITION")
        print("   %s" % ("-" * 62))
        print("   *** GO - START DOING IT NOW ***")
        t0 = time.time()
        countdown(secs, "BLOCK %s RECORDING" % key, triggers=trig, fired=injections)
        t1 = time.time()
        records.append({"block": key, "title": title,
                        "tStart": round(t0, 4), "tEnd": round(t1, 4),
                        "seconds": round(t1 - t0, 2)})

    if proc is not None:
        print("\n[stopping sidecar] ...")
        try:
            proc.wait(timeout=25)
        except Exception:
            proc.terminate()

    out = os.path.join(a.log_dir, "blocks.json")
    with open(out, "w") as f:
        json.dump({"pass": a.which, "recovery": recovery_flag,
                   "sampler": "legacy" if a.legacy_depth else "surface-aware",
                   "blocks": records, "injections": injections}, f, indent=2)
    print("\nblock boundaries -> %s" % out)
    print("\n" + "=" * 68)
    print(" STOP Unity play mode now so model_log.jsonl is flushed and closed.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
