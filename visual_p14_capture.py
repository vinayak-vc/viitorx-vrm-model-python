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
     ["Stand still, arms relaxed at your sides.",
      "Full body in frame. Do not move."]),
    ("2", 20, "WALK TOWARD / AWAY",
     ["Walk TOWARD the camera, then AWAY. Repeat.",
      "Stay within the frame at the near end."]),
    ("3", 20, "FAST ARMS",
     ["Fast waving, both arms. Large swings.",
      "Then reach OVERHEAD repeatedly, fast."]),
    ("4", 20, "FAST LEGS",
     ["Kicks, then squats.",
      "Rapid knee bends, then fast stepping."]),
    ("5", 25, "DANCE  (production stress test)",
     ["Dance freely: fast arms, fast legs, body turns,",
      "crossed limbs, rapid direction changes."]),
    ("6", 30, "HAND BEHIND TORSO  *** CRITICAL ***",
     ["LEFT hand behind your back:  hold 3s, release.",
      "Again: hold 8s, release.",
      "Now repeat both with the RIGHT hand."]),
    ("7", 25, "KNEE / LEG PARTIAL OCCLUSION  *** CRITICAL ***",
     ["Stand BEHIND a chair so one knee is hidden, hold 5s.",
      "No chair? Cross one leg fully behind the other, hold 5s.",
      "Repeat on the other side. 2x each."]),
    ("8", 25, "INJECTED DRIFT / TELEPORT  *** WATCH THE RIGHT KNEE ***",
     ["Stand still, full body in frame, and WATCH THE AVATAR RIGHT LEG.",
      "At ~4s a slow impossible drift is injected for 25 frames.",
      "At ~14s a teleport is injected for 10 frames.",
      "Note whether you SEE either one."]),
    ("9", 25, "REACQUISITION AFTER LOSS",
     ["Step FULLY out of frame for 5s, then step back in. Repeat.",
      "Then cover the lens with your hand for 5s and uncover."]),
]


# F-08 AUDIT (Part 9): a SUSTAINED occlusion, long enough that the wrist stays wrong well past
# P1-1's 6-frame prediction horizon -- the "confident-but-wrong" case, on demand.
AUDIT_BLOCK = ("10", 45, "SUSTAINED HAND BEHIND TORSO  *** F-08 PART 9 ***",
               ["LEFT hand behind your back. HOLD IT THERE for a FULL 20 SECONDS.",
                "Do not bring it out. Stand otherwise still.",
                "Then bring it out, pause 5s, and repeat with the RIGHT hand for 15s."])


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
    total = sum(b[1] for b in blocks) + a.lead_in + 3.0
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
        t0 = time.time()
        countdown(secs, "BLOCK %s" % key, triggers=trig, fired=injections)
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
                   "blocks": records, "injections": injections}, f, indent=2)
    print("\nblock boundaries -> %s" % out)
    print("\n" + "=" * 68)
    print(" STOP Unity play mode now so model_log.jsonl is flushed and closed.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
