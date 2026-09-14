#!/usr/bin/env python3
"""F-21 section 17 - live two-person protocol, run against the REAL sidecar.

REV 2: the first live run (2026-09-14) found that printed console instructions are unreadable while
physically coordinating two people and watching the camera preview at the same time - the operator
could not follow the protocol and the result was inconclusive. Fixed the same way F-20A/F-20B's
lessons say to: the instruction now gets drawn AS A BANNER directly on the SAME window the operator
is already watching (wholebody_udp_sender.py's --show preview, via --cue-file), right above the
F-21 HUD (state/owner/timers/switch count) - one window, not a window plus a scrolling terminal.

    python f21_live_protocol.py
"""
import io
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, ".venv", "Scripts", "python.exe")
SCRIPT = os.path.join(HERE, "wholebody_udp_sender.py")
MODEL = os.path.join(HERE, "..", "..", "..", "SentisModel", "rtmw3d-x.onnx")
EVIDENCE = os.path.join(HERE, "oak_v4_evidence", "f21", "live")
TIMELINE = os.path.join(EVIDENCE, "timeline.jsonl")
CUE_FILE = os.path.join(EVIDENCE, "cue.json")

# (title, short on-screen cue, seconds) - kept SHORT deliberately: this is read at a glance on a
# 400px-wide preview window while physically moving, not a paragraph.
PHASES = [
    ("SETUP",            "Both OUT of frame", 8),
    ("A_ENTER",          "A enters, stand still", 15),
    ("B_CENTERED",       "B enters, gets CLOSER + MORE CENTERED than A", 15),
    ("A_OCCLUDE",        "A hides / turns away. B stays visible", 6),
    ("A_RETURN",         "A comes back to the same spot", 10),
    ("A_EXIT",           "A walks OUT and stays out", 8),
    ("WAIT_RELEASE",     "Nobody in frame - just wait", 7),
    ("B_ACQUIRES",       "B should now become owner", 10),
    ("CROSSING",         "A and B walk toward each other and CROSS PATHS", 15),
    ("BOTH_LEAVE",       "Both walk out of frame", 5),
    ("SIMULTANEOUS",     "A and B walk back in TOGETHER, side by side", 15),
    ("REPEAT_A_OUT",     "Whoever is owner walks out", 5),
    ("REPEAT_OTHER_IN",  "The OTHER person walks in", 8),
    ("REPEAT_SWAP_BACK", "Swap again: that person out, first person in", 13),
    ("ENVELOPE_EDGE",    "Owner walks to the far edge / 2m+, holds, returns", 15),
    ("DONE",             "Protocol complete - press q in the preview to stop", 6),
]

events = []


def write_cue(title, instruction, seconds_left):
    tmp = CUE_FILE + ".tmp"
    with io.open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(title=title, instruction=instruction, seconds_left=seconds_left), f)
    os.replace(tmp, CUE_FILE)


def run_phase(title, instruction, seconds):
    events.append(dict(t=round(time.time(), 3), phase=title, instruction=instruction))
    print("[%-16s] %s (%ds)" % (title, instruction, seconds), flush=True)
    t0 = time.time()
    while True:
        left = seconds - (time.time() - t0)
        if left <= 0:
            break
        write_cue(title, instruction, left)
        time.sleep(0.15)
    events.append(dict(t=round(time.time(), 3), event="phase_end", phase=title))


def main():
    os.makedirs(EVIDENCE, exist_ok=True)
    cmd = [PY, "-u", SCRIPT, "--model", MODEL, "--portrait", "--portrait-dir", "ccw",
           "--subpixel-bits", "3", "--seconds", "0", "--show",
           "--ownership-log-dir", EVIDENCE, "--cue-file", CUE_FILE]
    print("Starting the real sidecar with --show + on-screen cues.")
    print("A window titled 'whole-body OAK sidecar' will open with:")
    print("  - a top banner: the CURRENT INSTRUCTION and a countdown")
    print("  - below it: the F-21 HUD (state / owner / timers / switch count)")
    print("Watch ONLY that window from here on. Press q or ESC in it to stop early.")
    print()
    write_cue("STARTING", "Loading the model - wait", 8)
    proc = subprocess.Popen(cmd, cwd=HERE)
    try:
        for title, instruction, seconds in PHASES:
            run_phase(title, instruction, seconds)
    finally:
        if proc.poll() is None:
            print("\nStopping the sidecar...")
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                proc.wait(timeout=8)
            except Exception:
                pass
        io.open(TIMELINE, "w", encoding="utf-8").write("".join(json.dumps(e) + "\n" for e in events))
        print("wrote %s" % TIMELINE)
        print("cross-reference against %s" % os.path.join(EVIDENCE, "target_events.jsonl"))


if __name__ == "__main__":
    main()
