#!/usr/bin/env python3

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(_os.path.join(_d, "_sidecar_path.py")):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import _sidecar_path  # noqa: F401  - puts the sidecar root and every tools/ group on sys.path
"""F-21 S31 / brief S15 - does the live protocol actually work UNDER F-20B supervision?

Runs the real supervisor, which runs the real sidecar, against the real OAK-D, with an EMPTY ROOM.
No physical person is required and none is claimed: this is a PROTOCOL/TOOLING integration test.

What it does NOT prove, stated before the results so it cannot be read the other way round:
  - nothing about live camera recovery. The sidecar is killed here with taskkill, which is a clean
    process death, not a DepthAI device fault. F-20B S17 covers real device failure separately.
  - nothing about two-person ownership behaviour. The room is empty on purpose.

What it does prove is the list the brief asks for, one check each:
  1  the supervisor starts a sidecar
  2  the sidecar is given --show and --cue-file
  3  the preview window really exists (enumerated from the OS, not inferred from a flag)
  4  the cue the protocol writes is the cue the sidecar reads, and it renders
  5  the sidecar writes its ownership log where f21_live_protocol looks for it, and the reader
     handles the empty-room case (no events yet) without raising
  6  the supervisor still notices the sidecar exiting, and restarts it
  7  there is never more than one sidecar process tree alive

    python f21_supervisor_integration.py
"""
import ctypes
import io
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, ".venv", "Scripts", "python.exe")
SUPERVISOR = os.path.join(HERE, "sidecar_supervisor.py")
MODEL = os.path.join(HERE, "..", "..", "..", "SentisModel", "rtmw3d-x.onnx")
EVID = os.path.join(HERE, "oak_v4_evidence", "f21", "supervisor_integration")
CUE = os.path.join(EVID, "cue.json")

from f20b_deployment_verify import producer_pids            # noqa: E402  (path-dependent import)

_results = []


def check(name, cond, detail=""):
    _results.append((name, bool(cond), detail))
    print("  %-4s %-58s %s" % ("PASS" if cond else "FAIL", name, detail))


def window_titles():
    """Top-level window titles, straight from user32 - so 'the HUD appears' is an observation of the
    operating system rather than a deduction from having passed --show."""
    out = []
    EnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))
    u = ctypes.windll.user32

    def cb(hwnd, _lparam):
        if not u.IsWindowVisible(hwnd):
            return True
        n = u.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        u.GetWindowTextW(hwnd, buf, n + 1)
        out.append(buf.value)
        return True

    u.EnumWindows(EnumProc(cb), None)
    return out


def wait_for(fn, timeout, interval=0.5):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = fn()
        if v:
            return v
        time.sleep(interval)
    return None


def main():
    if os.path.isdir(EVID):
        shutil.rmtree(EVID, ignore_errors=True)
    os.makedirs(EVID, exist_ok=True)

    print("=" * 96)
    print(" F-21 S31 / S15  supervisor + live-protocol integration   (EMPTY ROOM, real OAK-D)")
    print(" NOT a camera-recovery test and NOT a two-person test - see the module docstring.")
    print("=" * 96)

    # ---- 4a. the cue round-trip, using the REAL writer and the REAL reader -----------------------
    import f21_live_protocol as LP
    import wholebody_udp_sender as W
    LP.CUE_FILE = CUE
    LP.write_cue("SETUP", "Both OUT of frame", 7.5)
    got = W._read_cue(CUE)
    check("4a the protocol's cue is parsed by the sidecar's own reader",
          got is not None and got.get("title") == "SETUP"
          and got.get("instruction") == "Both OUT of frame",
          "read back %s" % json.dumps(got))

    view = np.zeros((760, 420, 3), dtype=np.uint8)
    y0 = W._draw_cue(view, got)
    banner = view[:max(1, y0)]
    png = os.path.join(EVID, "cue_render.png")
    import cv2
    cv2.imwrite(png, view[:max(40, y0) + 10])
    check("4b the cue renders a non-empty banner through the real _draw_cue",
          y0 > 0 and int(banner.max()) > 0,
          "banner height=%dpx brightest=%d -> %s" % (y0, int(banner.max()),
                                                     os.path.basename(png)))

    # ---- launch the real supervisor --------------------------------------------------------------
    cmd = [PY, "-u", SUPERVISOR, "--model", MODEL, "--portrait", "--portrait-dir", "ccw",
           "--subpixel-bits", "3", "--evidence-dir", EVID, "--port", "8891", "--lock-port", "8893",
           "--show", "--cue-file", CUE]
    print("\n  launching: sidecar_supervisor.py --show --cue-file ...")
    log = io.open(os.path.join(EVID, "supervisor_stdout.txt"), "w", encoding="utf-8")
    sup = subprocess.Popen(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
    peak_trees = 0
    try:
        # ---- 1. a sidecar appears ----------------------------------------------------------------
        pids = wait_for(lambda: producer_pids("wholebody_udp_sender"), 90)
        check("1  the supervisor started a sidecar", bool(pids), "pids=%s" % pids)
        if not pids:
            return 1

        # ---- 2. it was given the two flags -------------------------------------------------------
        supervisor_log = io.open(os.path.join(EVID, "supervisor.log"), encoding="utf-8").read()
        cmdline = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"ProcessId=%d\").CommandLine" % pids[0]],
            stderr=subprocess.DEVNULL).decode(errors="replace")
        check("2a the supervisor LOGGED the forwarded flags",
              "--show" in supervisor_log and "--cue-file" in supervisor_log)
        check("2b the RUNNING sidecar process really carries them",
              "--show" in cmdline and "--cue-file" in cmdline,
              "argv has --show=%s --cue-file=%s" % ("--show" in cmdline, "--cue-file" in cmdline))

        # ---- 3. the window exists ----------------------------------------------------------------
        title = wait_for(lambda: [t for t in window_titles() if "OAK" in t or "whole-body" in t], 90)
        check("3  the preview window exists (enumerated from user32)", bool(title),
              "title=%s" % (title[0] if title else "<none>"))

        # ---- 5. the ownership-state plumbing --------------------------------------------------
        # NOT "an ownership event appears": the room is empty on purpose, so F-21 correctly never
        # acquires and correctly writes no events. Asserting otherwise would be asserting that an
        # empty room produces a target. What IS testable here is the plumbing - that the sidecar
        # creates its ownership log at exactly the path f21_live_protocol reads in supervised mode,
        # and that the reader handles the empty file without raising instead of exploding on the
        # first live phase.
        LP._mode = "supervised"
        LP.OWNERSHIP_LOG_DIR["supervised"] = os.path.join(HERE, "oak_v4_evidence", "f21")
        own_log = os.path.join(LP.OWNERSHIP_LOG_DIR["supervised"], "target_events.jsonl")
        appeared = wait_for(lambda: os.path.isfile(own_log), 60)
        check("5a the sidecar writes its ownership log where the protocol looks for it",
              bool(appeared), own_log)
        try:
            st = LP.last_ownership_state()
            raised = None
        except Exception as ex:                                   # noqa: BLE001
            st, raised = None, ex
        check("5b the protocol reads it without raising, and reports 'nothing yet' honestly",
              raised is None, "returned %s (None is CORRECT with an empty room)" % json.dumps(st))

        # ---- 7a. exactly one tree so far ---------------------------------------------------------
        peak_trees = max(peak_trees, len(producer_pids("wholebody_udp_sender")))

        # ---- 6. kill the sidecar; the supervisor must notice and restart --------------------------
        before = producer_pids("wholebody_udp_sender")
        print("\n  killing sidecar tree pid=%d to test detection..." % before[0])
        subprocess.call(["taskkill", "/F", "/T", "/PID", str(before[0])],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        def restarted():
            now = producer_pids("wholebody_udp_sender")
            return now if (now and now[0] != before[0]) else None

        after = wait_for(restarted, 120)
        evpath = os.path.join(EVID, "supervisor_events.jsonl")
        evs = [json.loads(l) for l in io.open(evpath, encoding="utf-8") if l.strip()]
        # "SidecarExit" - checked against sidecar_supervisor.py's own self.event() calls
        # rather than guessed at; the first version of this test guessed "SidecarExited"
        # and reported a supervisor failure that had not happened.
        exits = [e for e in evs if e.get("event") in ("SidecarExit", "HeartbeatTimeout",
                                                      "StartupStuck")]
        check("6a the supervisor noticed the sidecar exiting", bool(exits),
              "events=%s" % [e["event"] for e in exits])
        check("6b it restarted with a NEW pid", bool(after),
              "%s -> %s" % (before, after))
        for _ in range(12):
            peak_trees = max(peak_trees, len(producer_pids("wholebody_udp_sender")))
            time.sleep(0.5)
        check("7  never more than one sidecar process tree alive", peak_trees <= 1,
              "peak=%d" % peak_trees)
    finally:
        if sup.poll() is None:
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(sup.pid)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                sup.wait(timeout=15)
            except Exception:
                pass
        log.close()
        for p in producer_pids("wholebody_udp_sender"):
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(p)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    n = len(_results)
    ok = sum(1 for _, c, _ in _results if c)
    print("-" * 96)
    print(" %d/%d checks passed   evidence -> %s" % (ok, n, EVID))
    io.open(os.path.join(EVID, "results.json"), "w", encoding="utf-8").write(
        json.dumps([dict(name=a, passed=b, detail=c) for a, b, c in _results], indent=2))
    print("=" * 96)
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
