#!/usr/bin/env python3

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(_os.path.join(_d, "_sidecar_path.py")):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import _sidecar_path  # noqa: F401  - puts the sidecar root and every tools/ group on sys.path
"""Does f21_live_protocol.py actually ABORT when the producer dies? (brief §14A / §18)

The second live F-21 attempt lost the DepthAI device part-way through and the protocol kept marching
through its phases, writing a timeline that reads like a completed protocol. §31.1 added
`proc.poll()` polling to stop that. Implementing a safety check and not testing it is how the same
class of bug comes back, so this exercises the real abort path rather than trusting it.

No camera and no model: the producer is a STUB that idles until it is killed. That is the right test
double here because the thing under test is subprocess-death detection, which is entirely agnostic to
what the subprocess was doing - and using the real sidecar would make this a four-minute camera test
of something that has nothing to do with the camera. Correspondingly, this proves NOTHING about
DepthAI failure modes; F-20B §17 covers those.

    python f21_live_protocol_abort_test.py
"""
import io
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STUB = os.path.join(HERE, "_f21_abort_stub.py")
STUB_SRC = '''import sys, time
# accepts and ignores any flags the protocol passes, then idles until killed
print("producer session id = stubstubstub", flush=True)
while True:
    print("[wb] frames=1 sent=1 idle", flush=True)
    time.sleep(0.2)
'''

_results = []


def check(name, cond, detail=""):
    _results.append((name, bool(cond), detail))
    print("  %-4s %-62s %s" % ("PASS" if cond else "FAIL", name, detail))


def main():
    import f21_live_protocol as LP

    io.open(STUB, "w", encoding="utf-8").write(STUB_SRC)
    evidence = os.path.join(HERE, "oak_v4_evidence", "f21", "abort_test")
    os.makedirs(evidence, exist_ok=True)

    # point the protocol at the stub and at its own evidence dir, and shorten the phases so the
    # test takes seconds. Nothing about the abort path itself is stubbed.
    LP.SCRIPT = STUB
    LP.MODEL = "unused"
    LP.EVIDENCE = evidence
    LP.TIMELINE = os.path.join(evidence, "timeline.jsonl")
    LP.CUE_FILE = os.path.join(evidence, "cue.json")
    LP.OWNERSHIP_LOG_DIR["direct"] = evidence
    LP.PHASES = [("P_ONE", "first", 3), ("P_TWO", "second", 30), ("P_THREE", "third", 3)]
    LP.events = []

    # a plausible last ownership state, so the abort record is checked for reading the REAL file
    io.open(os.path.join(evidence, "target_events.jsonl"), "w", encoding="utf-8").write(
        json.dumps(dict(event="TARGET_LOCKED", state="LOCKED", target_id=1)) + "\n")

    print("=" * 96)
    print(" f21_live_protocol.py dead-producer abort test  (stub producer, no camera, no model)")
    print("=" * 96)

    killed = {}

    def killer():
        # wait until the protocol is inside the long second phase, then kill the producer
        time.sleep(6.0)
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "
             "'*_f21_abort_stub.py*' } | ForEach-Object { $_.ProcessId }"],
            stderr=subprocess.DEVNULL).decode(errors="replace").split()
        killed["pids"] = out
        for pid in out:
            subprocess.call(["taskkill", "/F", "/T", "/PID", pid],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    t = threading.Thread(target=killer)
    t.daemon = True
    t.start()

    rc = LP.main()
    t.join(timeout=10)

    check("0 the stub producer was actually running and got killed", bool(killed.get("pids")),
          "pids=%s" % killed.get("pids"))
    check("1 the protocol returns non-zero", rc == 1, "rc=%s" % rc)

    rows = [json.loads(l) for l in io.open(LP.TIMELINE, encoding="utf-8") if l.strip()]
    ab = [r for r in rows if r.get("event") == "LIVE_TEST_ABORTED"]
    check("2 a LIVE_TEST_ABORTED record is written to the timeline", len(ab) == 1,
          "found=%d" % len(ab))
    if not ab:
        return report()
    a = ab[0]
    check("3 it records the return code", "returncode" in a, "returncode=%s" % a.get("returncode"))
    check("4 it names the phase the producer died in", a.get("phase") == "P_TWO",
          "phase=%s" % a.get("phase"))
    check("5 it records a wall-clock timestamp", bool(a.get("wall_clock")),
          "wall_clock=%s" % a.get("wall_clock"))
    check("6 it records the last F-21 ownership state, read from the sidecar's own log",
          (a.get("last_ownership_state") or {}).get("event") == "TARGET_LOCKED",
          "last=%s" % json.dumps(a.get("last_ownership_state")))
    check("7 it says which phases were actually completed",
          a.get("completed_phases") == ["P_ONE"],
          "completed=%s" % a.get("completed_phases"))
    check("8 and it did NOT go on to run the phases after the death",
          not any(r.get("phase") == "P_THREE" for r in rows),
          "phases seen=%s" % sorted(set(r.get("phase") for r in rows if r.get("phase"))))
    return report()


def report():
    n = len(_results)
    ok = sum(1 for _, c, _ in _results if c)
    print("-" * 96)
    print(" %d/%d checks passed" % (ok, n))
    print("=" * 96)
    try:
        os.remove(STUB)
    except OSError:
        pass
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
