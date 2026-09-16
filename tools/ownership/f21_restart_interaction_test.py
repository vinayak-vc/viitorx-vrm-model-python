#!/usr/bin/env python3
"""F-21 section 16/19 - sidecar restart while a person is present, and F-20A interaction.

Proves rather than asserts: a new sidecar process must not blindly inherit a potentially stale
identity (SS16 of the F-21 brief). Since ownership state lives only in the sidecar process's own
target_ownership.TargetOwnership instance (never serialized, never shared across processes), a
restart trivially cannot carry it over - this script demonstrates that with the REAL process, not
just by reading the code, matching this project's evidence-first convention.

Reuses f20a_failure_injection.py's exact taskkill-based forced-kill pattern (this is a producer
restart, not a new scenario) and layers an F-21-specific assertion on top: the NEW process's own
target_events.jsonl must show a FRESH TARGET_ACQUIRED at target_id=1 (a fresh epoch), not any
target_id carried over from before the kill.

Needs a person in front of the camera for both halves (before AND after the restart) - run it with
the operator standing in frame.

    python f21_restart_interaction_test.py
"""

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(_os.path.join(_d, "_sidecar_path.py")):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import _sidecar_path  # noqa: F401  - puts the sidecar root and every tools/ group on sys.path
import evidence_paths as EV

import io
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, ".venv", "Scripts", "python.exe")
SCRIPT = os.path.join(HERE, "wholebody_udp_sender.py")
MODEL = EV.DEFAULT_MODEL
EVIDENCE = EV.oak_v4("f21", "restart_test")
LOG_DIR = os.path.join(EVIDENCE, "target_events")

results = []


def record(name, ok, detail):
    results.append((name, ok, detail))
    print("[f21-restart] %-46s %s  %s" % (name, "PASS" if ok else "FAIL", detail))


def read_events(run_log_dir):
    path = os.path.join(run_log_dir, "target_events.jsonl")
    out = []
    try:
        for line in io.open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                out.append(json.loads(line))
    except Exception:
        pass
    return out


def start(run_log_dir, stdout_path):
    os.makedirs(run_log_dir, exist_ok=True)
    out = io.open(stdout_path, "w", encoding="utf-8")
    cmd = [PY, "-u", SCRIPT, "--model", MODEL, "--portrait", "--portrait-dir", "ccw",
           "--subpixel-bits", "3", "--seconds", "0", "--ownership-log-dir", run_log_dir]
    p = subprocess.Popen(cmd, cwd=HERE, stdout=out, stderr=subprocess.STDOUT)
    print("[f21-restart] started pid=%d run_log_dir=%s" % (p.pid, run_log_dir))
    return p, out


def wait_for_lock(run_log_dir, timeout=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        ev = read_events(run_log_dir)
        if any(e.get("event") == "TARGET_LOCKED" for e in ev):
            return ev
        time.sleep(0.3)
    return read_events(run_log_dir)


def main():
    os.makedirs(EVIDENCE, exist_ok=True)
    print("=" * 78)
    print(" F-21 sidecar-restart interaction - stand in front of the camera now")
    print("=" * 78)

    run1_dir = os.path.join(LOG_DIR, "run1")
    proc1, out1 = start(run1_dir, os.path.join(EVIDENCE, "run1_stdout.txt"))
    try:
        ev1 = wait_for_lock(run1_dir, timeout=30)
        locked1 = any(e.get("event") == "TARGET_LOCKED" for e in ev1)
        record("initial LOCK before restart", locked1,
               "events=%d" % len(ev1))
        sid1 = None
        for e in ev1:
            if "sid" in e:
                sid1 = e["sid"]
                break

        print("[f21-restart] forcibly killing pid=%d (stay in frame) ..." % proc1.pid)
        subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc1.pid)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            proc1.wait(timeout=8)
        except Exception:
            pass
    finally:
        if out1 is not None and not out1.closed:
            out1.close()

    time.sleep(1.0)
    run2_dir = os.path.join(LOG_DIR, "run2")
    proc2, out2 = start(run2_dir, os.path.join(EVIDENCE, "run2_stdout.txt"))
    try:
        ev2 = wait_for_lock(run2_dir, timeout=30)
        locked2 = any(e.get("event") == "TARGET_LOCKED" for e in ev2)
        record("re-locked after restart", locked2, "events=%d" % len(ev2))

        sid2 = None
        target_id2 = None
        for e in ev2:
            if "sid" in e and sid2 is None:
                sid2 = e["sid"]
            if e.get("event") == "TARGET_ACQUIRED" and target_id2 is None:
                target_id2 = e.get("target_id")

        record("NEW sidecar session id (F-20A semantics unchanged)",
               sid1 is not None and sid2 is not None and sid1 != sid2,
               "sid1=%s sid2=%s" % (sid1, sid2))
        record("fresh ownership epoch, NOT inherited from before the kill",
               target_id2 == 1, "new process target_id=%s (must be 1 - a fresh instance)" % target_id2)
        record("NO stale-identity carry-over event of any kind",
               not any(e.get("event") not in
                       ("TARGET_ACQUIRED", "TARGET_LOCKED", "TARGET_TEMP_LOST", "TARGET_REACQUIRED")
                       for e in ev2),
               "event kinds=%s" % sorted(set(e.get("event") for e in ev2)))
    finally:
        if proc2.poll() is None:
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc2.pid)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                proc2.wait(timeout=8)
            except Exception:
                pass
        if out2 is not None and not out2.closed:
            out2.close()

    print("=" * 78)
    n = len(results)
    p = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        print(" %-46s %s  %s" % (name, "PASS" if ok else "FAIL", detail))
    print("-" * 78)
    print(" %d/%d PASS" % (p, n))
    print("=" * 78)
    return 0 if p == n else 1


if __name__ == "__main__":
    sys.exit(main())
