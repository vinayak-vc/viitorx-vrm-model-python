#!/usr/bin/env python3
"""F-20B automated test harness - proves supervisor-level recovery WITHOUT needing a person at the
camera. Complements the live protocol in f20b_usb_test.py (real USB unplug, Unity restart), which
genuinely needs a human present per the F-20A precedent.

Each scenario launches the REAL sidecar_supervisor.py as its own subprocess (its own evidence dir,
its own lock port) and reads back supervisor_state.json / supervisor_events.jsonl / supervisor.log -
the same artifacts an operator would read - rather than reaching into the supervisor's internals.

  A  NORMAL_START        real sidecar + real camera -> SIDECAR READY
  B  FORCED_KILL x2       taskkill /F on the tracked sidecar pid, twice in a row -> supervisor
                          restarts each time with a NEW session id (not a one-shot fluke)
  D  DUPLICATE_GUARD      a second supervisor pointed at the same lock port refuses to start; the
                          first is left completely undisturbed
  E  CRASH_LOOP           a synthetic always-exit-1 test double (f20b_fake_sidecar.py --mode crash)
                          drives 5 failures inside the crash-loop window -> FAILED_PERMANENT, then
                          keeps retrying at the (short, for the test) failed-permanent interval
                          instead of giving up forever
  F  DEPENDENCY_FAILURE   --python points at a non-existent executable -> FAILED_PERMANENT
                          immediately, the supervisor process itself exits, no restart loop at all

A and B use the real camera (confirmed connected) because they are the evidence that the readiness
regex and the taskkill-based recovery work against the ACTUAL production process, not just the state
machine in isolation. D/E/F are about the supervisor's own logic and are deliberately kept off the
real camera so they run in seconds, not minutes, and don't cycle real hardware needlessly.

    python f20b_failure_tests.py
"""
import io
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, ".venv", "Scripts", "python.exe")
SUPERVISOR = os.path.join(HERE, "sidecar_supervisor.py")
REAL_SCRIPT = os.path.join(HERE, "wholebody_udp_sender.py")
FAKE_SCRIPT = os.path.join(HERE, "f20b_fake_sidecar.py")
REAL_MODEL = os.path.join(HERE, "..", "..", "..", "SentisModel", "rtmw3d-x.onnx")
BASE_EVIDENCE = os.path.join(HERE, "oak_v4_evidence", "f20b", "tests")

results = []


def read_state(evidence_dir):
    try:
        return json.load(io.open(os.path.join(evidence_dir, "supervisor_state.json"), encoding="utf-8"))
    except Exception:
        return None


def read_events(evidence_dir):
    out = []
    try:
        for line in io.open(os.path.join(evidence_dir, "supervisor_events.jsonl"), encoding="utf-8"):
            line = line.strip()
            if line:
                out.append(json.loads(line))
    except Exception:
        pass
    return out


def start_supervisor(evidence_dir, extra_args, label, env=None):
    # A stale supervisor_state.json from a PREVIOUS run of this same scenario can otherwise be read
    # as "already ready" on the very first poll, before this run's own process has written
    # anything - found live when a leftover dir made D_DUPLICATE_GUARD flake on a rerun.
    shutil.rmtree(evidence_dir, ignore_errors=True)
    os.makedirs(evidence_dir, exist_ok=True)
    cmd = [PY, "-u", SUPERVISOR, "--evidence-dir", evidence_dir] + extra_args
    log = io.open(os.path.join(evidence_dir, "harness_stdout.txt"), "w", encoding="utf-8")
    p = subprocess.Popen(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env=env)
    print("[f20b-test] %-24s supervisor pid=%d evidence=%s" % (label, p.pid, evidence_dir))
    return p, log


def stop_supervisor(proc, log):
    if proc.poll() is None:
        subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            proc.wait(timeout=8)
        except Exception:
            pass
    if log is not None and not log.closed:
        log.close()


def wait_for(evidence_dir, predicate, timeout, poll=0.3):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = read_state(evidence_dir)
        if st is not None and predicate(st):
            return st, time.time() - t0
        time.sleep(poll)
    return read_state(evidence_dir), time.time() - t0


def record(name, passed, detail):
    results.append((name, passed, detail))
    print("[f20b-test] %-28s %s  %s" % (name, "PASS" if passed else "FAIL", detail))


# ------------------------------------------------------------------------------------------- A
def test_a_normal_start():
    d = os.path.join(BASE_EVIDENCE, "a_normal_start")
    proc, log = start_supervisor(
        d, ["--python", PY, "--script", REAL_SCRIPT, "--model", REAL_MODEL,
            "--lock-port", "8920", "--port", "8899"], "A_NORMAL_START")
    try:
        st, elapsed = wait_for(d, lambda s: s.get("SidecarReady"), timeout=60)
        ok = bool(st and st.get("SidecarReady"))
        record("A_NORMAL_START", ok,
               "ready=%.1fs sid=%s" % (elapsed, st and st.get("CurrentSessionId")))
    finally:
        stop_supervisor(proc, log)


# ------------------------------------------------------------------------------------------- B
def test_b_forced_kill_recovery(repeat=2):
    d = os.path.join(BASE_EVIDENCE, "b_forced_kill")
    proc, log = start_supervisor(
        d, ["--python", PY, "--script", REAL_SCRIPT, "--model", REAL_MODEL,
            "--lock-port", "8921", "--port", "8899"], "B_FORCED_KILL")
    try:
        st, elapsed0 = wait_for(d, lambda s: s.get("SidecarReady"), timeout=60)
        if not (st and st.get("SidecarReady")):
            record("B_FORCED_KILL_x%d" % repeat, False, "initial start never became ready")
            return
        detail = ["initial sid=%s (%.1fs)" % (st["CurrentSessionId"], elapsed0)]
        ok_all = True
        for i in range(repeat):
            pid = st["SidecarPid"]
            prior_sid = st["CurrentSessionId"]
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(pid)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            st, t_recover = wait_for(
                d, lambda s: s.get("SidecarReady") and s.get("CurrentSessionId") != prior_sid,
                timeout=90)
            recovered = bool(st and st.get("SidecarReady") and st.get("CurrentSessionId") != prior_sid)
            ok_all = ok_all and recovered
            detail.append("kill#%d -> new_sid=%s T_recover=%.1fs"
                           % (i + 1, st and st.get("CurrentSessionId"), t_recover))
        record("B_FORCED_KILL_x%d" % repeat, ok_all, "; ".join(detail))
    finally:
        stop_supervisor(proc, log)


# ------------------------------------------------------------------------------------------- D
def test_d_duplicate_guard():
    d1 = os.path.join(BASE_EVIDENCE, "d_duplicate_guard_1st")
    d2 = os.path.join(BASE_EVIDENCE, "d_duplicate_guard_2nd")
    proc1, log1 = start_supervisor(
        d1, ["--python", PY, "--script", FAKE_SCRIPT, "--lock-port", "8922", "--port", "8988"],
        "D_DUP_1st")
    try:
        st1, _ = wait_for(d1, lambda s: s.get("SidecarReady"), timeout=20)
        if not (st1 and st1.get("SidecarReady")):
            record("D_DUPLICATE_GUARD", False, "first instance never became ready")
            return
        proc2, log2 = start_supervisor(
            d2, ["--python", PY, "--script", FAKE_SCRIPT, "--lock-port", "8922", "--port", "8988"],
            "D_DUP_2nd")
        try:
            rc2 = proc2.wait(timeout=15)
        finally:
            stop_supervisor(proc2, log2)
        st1_after = read_state(d1)
        first_untouched = bool(st1_after and st1_after.get("SidecarReady")
                                and st1_after.get("CurrentSessionId") == st1.get("CurrentSessionId"))
        record("D_DUPLICATE_GUARD", rc2 != 0 and first_untouched,
               "2nd_rc=%s first_still_ready_same_sid=%s" % (rc2, first_untouched))
    finally:
        stop_supervisor(proc1, log1)


# ------------------------------------------------------------------------------------------- E
def test_e_crash_loop():
    d = os.path.join(BASE_EVIDENCE, "e_crash_loop")
    env = dict(os.environ, F20B_FAKE_MODE="crash")
    proc, log = start_supervisor(
        d, ["--python", PY, "--script", FAKE_SCRIPT,
            "--lock-port", "8923", "--port", "8989",
            "--crash-loop-count", "5", "--crash-loop-window", "60",
            "--failed-permanent-retry", "3"],
        "E_CRASH_LOOP", env=env)
    try:
        st, elapsed = wait_for(d, lambda s: s.get("SupervisorState") == "FAILED_PERMANENT", timeout=90)
        tripped = bool(st and st.get("SupervisorState") == "FAILED_PERMANENT")
        events = read_events(d)
        exit_events = [e for e in events if e.get("event") == "SidecarExit"]
        crash_loop_events = [e for e in events if e.get("event") == "CrashLoop"]
        record("E_CRASH_LOOP_TRIPS", tripped and len(crash_loop_events) == 1,
               "tripped_after=%.1fs exits=%d crash_loop_events=%d restart_count=%s"
               % (elapsed, len(exit_events), len(crash_loop_events), st and st.get("RestartCount")))

        restart_count_at_trip = st.get("RestartCount") if st else None
        time.sleep(8)
        st2 = read_state(d)
        still_retrying = bool(proc.poll() is None and st2
                               and st2.get("RestartCount", 0) > (restart_count_at_trip or 0))
        record("E_CRASH_LOOP_KEEPS_RETRYING", still_retrying,
               "supervisor_alive=%s state=%s restart_count %s->%s"
               % (proc.poll() is None, st2 and st2.get("SupervisorState"),
                  restart_count_at_trip, st2 and st2.get("RestartCount")))
    finally:
        stop_supervisor(proc, log)


# ------------------------------------------------------------------------------------------- F
def test_f_dependency_failure():
    d = os.path.join(BASE_EVIDENCE, "f_dep_failure")
    bad_python = os.path.join(HERE, "does_not_exist_python.exe")
    proc, log = start_supervisor(
        d, ["--python", bad_python, "--script", REAL_SCRIPT, "--model", REAL_MODEL,
            "--lock-port", "8924", "--port", "8990"], "F_DEP_FAILURE")
    try:
        rc = proc.wait(timeout=15)
        events = read_events(d)
        st = read_state(d)
        terminal_events = [e for e in events if e.get("event") == "FailedPermanent"]
        no_restart_loop = not any(e.get("event") == "Backoff" for e in events)
        ok = (rc == 2 and st is not None and st.get("SupervisorState") == "FAILED_PERMANENT"
              and len(terminal_events) == 1 and terminal_events[0].get("terminal") is True
              and no_restart_loop)
        record("F_DEPENDENCY_FAILURE", ok,
               "rc=%s reason=%s no_restart_loop=%s"
               % (rc, st and st.get("LastFailureReason"), no_restart_loop))
    finally:
        stop_supervisor(proc, log)


ALL_TESTS = {
    "a": test_a_normal_start,
    "b": lambda: test_b_forced_kill_recovery(repeat=2),
    "d": test_d_duplicate_guard,
    "e": test_e_crash_loop,
    "f": test_f_dependency_failure,
}


def main():
    os.makedirs(BASE_EVIDENCE, exist_ok=True)
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else list(ALL_TESTS.keys())
    print("=" * 78)
    print(" F-20B AUTOMATED SUPERVISOR TEST HARNESS")
    print(" A/B against the real sidecar + connected camera; D/E/F against test doubles")
    print(" running: %s" % ",".join(only))
    print("=" * 78)

    for key in only:
        ALL_TESTS[key]()

    print("=" * 78)
    n_pass = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        print(" %-28s %s  %s" % (name, "PASS" if ok else "FAIL", detail))
    print("-" * 78)
    print(" %d/%d PASS" % (n_pass, len(results)))
    print("=" * 78)

    with io.open(os.path.join(BASE_EVIDENCE, "summary.json"), "w", encoding="utf-8") as f:
        json.dump([dict(name=n, passed=ok, detail=de) for n, ok, de in results], f, indent=2)
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
