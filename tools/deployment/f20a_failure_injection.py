#!/usr/bin/env python3
"""F-20A sections 7, 8, 12 - drive the real sidecar through crash/restart cycles while Unity renders.

This is the LIVE half of F-20A. The unit tests prove the session logic in isolation; this proves the
whole path: real camera, real sidecar process, real UDP, real Unity, real VRM.

Each phase writes its name into the same block file the Unity bone recorder polls, so the rendered
avatar and the transport state land on ONE timeline and the recovery can be measured rather than
asserted. Phases:

    LIVE_BASELINE   sidecar healthy; establishes what Live looks like
    KILL_FORCED     taskkill /F - the unclean case (no chance to signal anything)
    GAP_FORCED      nothing is sending; Unity must go Live -> Reconnecting/StaleHold -> StaleFailsafe
    RESTART_1       sidecar restarts; seq goes back to 1 with a NEW sid
    LIVE_AFTER_1    must be tracking again, with no Unity restart and no manual intervention
    KILL_CLEAN      terminate() - the clean-exit case
    GAP_CLEAN
    RESTART_2
    LIVE_AFTER_2

    python f20a_failure_injection.py
"""
import argparse
import io
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, ".venv", "Scripts", "python.exe")
MODEL = os.path.join(HERE, "..", "..", "..", "SentisModel", "rtmw3d-x.onnx")
BLOCK_FILE = os.path.join(HERE, "oak_v4_evidence", "f20a", "block.txt")
TIMELINE = os.path.join(HERE, "oak_v4_evidence", "f20a", "timeline.jsonl")

events = []


def label(name):
    io.open(BLOCK_FILE, "w", encoding="utf-8").write(name)
    rec = dict(t=round(time.time(), 3), phase=name)
    events.append(rec)
    print("[f20a] %-14s  %s" % (name, time.strftime("%H:%M:%S")), flush=True)


def note(kind, detail=""):
    rec = dict(t=round(time.time(), 3), event=kind, detail=detail)
    events.append(rec)
    print("[f20a]   %s %s" % (kind, detail), flush=True)


def start_sidecar(log_dir, subpixel_bits=3):
    os.makedirs(log_dir, exist_ok=True)
    cmd = [PY, "-u", "wholebody_udp_sender.py", "--model", MODEL, "--portrait",
           "--subpixel-bits", str(subpixel_bits), "--log-dir", log_dir, "--seconds", "0"]
    out = io.open(os.path.join(log_dir, "sidecar_stdout.txt"), "w", encoding="utf-8")
    p = subprocess.Popen(cmd, cwd=HERE, stdout=out, stderr=subprocess.STDOUT)
    note("sidecar_spawn", "pid=%d" % p.pid)
    return p, out


def wait_until_sending(log_dir, timeout=90.0):
    """Wait for the sidecar to print its first frame line, i.e. it is actually on the wire."""
    path = os.path.join(log_dir, "sidecar_stdout.txt")
    sid = None
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            txt = io.open(path, encoding="utf-8", errors="replace").read()
        except Exception:
            txt = ""
        if sid is None and "producer session id = " in txt:
            sid = txt.split("producer session id = ")[1].split()[0]
        if "frames=" in txt:
            note("sidecar_live", "sid=%s after %.1fs" % (sid, time.time() - t0))
            return sid
        time.sleep(0.5)
    note("sidecar_never_live", "after %.1fs" % (time.time() - t0))
    return sid


def hold(seconds, phase):
    label(phase)
    time.sleep(seconds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=float, default=15.0)
    ap.add_argument("--gap", type=float, default=8.0, help="dead time after a kill (failsafe is 2 s)")
    ap.add_argument("--after", type=float, default=15.0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(BLOCK_FILE), exist_ok=True)

    proc = None
    out = None
    try:
        # ---- phase 1: healthy -----------------------------------------------------------------
        label("STARTING")
        proc, out = start_sidecar(os.path.join("oak_v4_evidence", "f20a", "log_run1"))
        sid1 = wait_until_sending(os.path.join("oak_v4_evidence", "f20a", "log_run1"))
        hold(a.baseline, "LIVE_BASELINE")

        # ---- phase 2: FORCED kill (§12 forced termination) --------------------------------------
        note("kill_forced", "taskkill /F /PID %d" % proc.pid)
        subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        label("KILL_FORCED")
        time.sleep(0.5)
        hold(a.gap, "GAP_FORCED")

        # ---- phase 3: restart -> new session, seq back to 1 -------------------------------------
        label("RESTART_1")
        proc, out = start_sidecar(os.path.join("oak_v4_evidence", "f20a", "log_run2"))
        sid2 = wait_until_sending(os.path.join("oak_v4_evidence", "f20a", "log_run2"))
        note("session_ids", "run1=%s run2=%s distinct=%s" % (sid1, sid2, sid1 != sid2))
        hold(a.after, "LIVE_AFTER_1")

        # ---- phase 4: CLEAN shutdown (§12 clean exit) -------------------------------------------
        note("kill_clean", "terminate() pid=%d" % proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        label("KILL_CLEAN")
        time.sleep(0.5)
        hold(a.gap, "GAP_CLEAN")

        # ---- phase 5: restart again -------------------------------------------------------------
        label("RESTART_2")
        proc, out = start_sidecar(os.path.join("oak_v4_evidence", "f20a", "log_run3"))
        sid3 = wait_until_sending(os.path.join("oak_v4_evidence", "f20a", "log_run3"))
        note("session_ids", "run3=%s distinct_from_run2=%s" % (sid3, sid3 != sid2))
        hold(a.after, "LIVE_AFTER_2")
        label("DONE")
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except Exception:
                subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if out is not None:
            out.close()
        io.open(TIMELINE, "w", encoding="utf-8").write(
            "".join(json.dumps(e) + "\n" for e in events))
        print("[f20a] wrote %s (%d events)" % (TIMELINE, len(events)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
