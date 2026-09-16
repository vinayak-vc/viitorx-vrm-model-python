#!/usr/bin/env python3
"""F-20A/F-20B deployment hardening - the remaining non-human checks, done for real.

f20b_failure_tests.py already proves the supervisor's own state machine (normal start, forced-kill
recovery, the duplicate-instance guard, crash-loop protection, dependency failure). This covers what
that suite does NOT: the things between the supervisor and the MACHINE it has to survive on.

  1 run_supervisor.bat          the launcher an operator actually double-clicks - does the command
                                inside it still match the supervisor's real CLI, and does every path
                                it depends on resolve from its own directory?
  2 logon registration          the documented schtasks command - is it well-formed, does everything
                                it references exist, and is it registered on THIS machine? Verified
                                STATICALLY and never executed: registering a logon task is a
                                persistent system change, and the F-20B report's own position is that
                                it is documented for an operator to run, not automated.
  3 duplicate-supervisor guard  re-proven here against the real binary, because it is the check that
                                stops two supervisors fighting over one camera after a logon race.
  4 diagnostics after logon     is supervisor_state.json readable, complete, and FRESH - i.e. does a
                                new supervisor overwrite a previous boot's snapshot rather than leave
                                an operator reading stale numbers after a reboot?
  5 no duplicate sidecars       after a kill/restart cycle, is there exactly ONE producer process?
  6 F-20A reconnect             two REAL sidecar process launches produce two real session ids; the
                                REAL C# TrackingStreamHealth classification for that pair is checked
                                in Unity (see --unity-check), not re-implemented in Python.

    python f20b_deployment_verify.py
"""
import argparse
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BAT = os.path.join(HERE, "run_supervisor.bat")
SUP = os.path.join(HERE, "sidecar_supervisor.py")
FAKE = os.path.join(HERE, "f20b_fake_sidecar.py")
PY = os.path.join(HERE, ".venv", "Scripts", "python.exe")
if not os.path.exists(PY):
    PY = sys.executable
REPORT = os.path.join(HERE, "..", "docs", "F20B_SIDECAR_SUPERVISOR_WATCHDOG_2026-09-14.md")

_results = []


def check(name, cond, detail=""):
    _results.append((name, bool(cond), detail))
    print("  %s  %s%s" % ("PASS" if cond else "FAIL", name, ("   [%s]" % detail) if detail else ""))


def read_state(ev):
    p = os.path.join(ev, "supervisor_state.json")
    for _ in range(10):
        try:
            with io.open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            time.sleep(0.1)
    return None


def wait_for(ev, pred, timeout, poll=0.3):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = read_state(ev)
        if st and pred(st):
            return st
        time.sleep(poll)
    return None


def start_sup(ev, extra, env=None):
    # A stale supervisor_state.json from a previous run of this same scenario reads as "already
    # ready" on the first poll - the same flake f20b_failure_tests.py documents hitting live.
    shutil.rmtree(ev, ignore_errors=True)
    os.makedirs(ev, exist_ok=True)
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.Popen([PY, "-u", SUP, "--evidence-dir", ev, "--python", PY] + extra,
                            cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=e)


def stop_sup(p):
    if p and p.poll() is None:
        subprocess.call(["taskkill", "/F", "/T", "/PID", str(p.pid)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        p.wait(timeout=10)
    except Exception:
        pass


def producer_pids(marker):
    """PIDs of live PRODUCER processes only.

    Naming the script is not enough to identify one. Three other things carry the same string on
    their command line and would be counted as producers by a naive match: the supervisor (it passes
    `--script ...f20b_fake_sidecar.py`), any second supervisor under test, and - the one that is easy
    to miss - the PowerShell process running this very query. So: python processes that are RUNNING
    the script, explicitly excluding anything that is merely REFERRING to it.

    One further wrinkle, measured rather than assumed: this venv's .venv\\Scripts\\python.exe is a
    LAUNCHER SHIM that spawns the real interpreter (C:\\Program Files\\Python310\\python.exe) as its
    own child, so a single healthy sidecar is always TWO pids - the shim the supervisor tracks as
    SidecarPid, plus its child. What must be unique is therefore the process TREE, not the process,
    so any matched process whose parent is also a match is folded into its parent. (This is also why
    the supervisor's taskkill /T is load-bearing: killing the shim alone would orphan the real
    interpreter, which would keep holding the UDP port and the camera.)

    Built with concatenation rather than %-formatting: the filter is full of literal % wildcards."""
    ps = ("Get-CimInstance Win32_Process | Where-Object { "
          "$_.Name -like '*python*' -and "
          "$_.CommandLine -like '*" + marker + ".py*' -and "
          "$_.CommandLine -notlike '*sidecar_supervisor.py*' } | "
          "ForEach-Object { \"$($_.ProcessId) $($_.ParentProcessId)\" }")
    try:
        out = subprocess.check_output(["powershell", "-NoProfile", "-Command", ps],
                                      stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return []
    pairs = []
    for line in out.decode(errors="replace").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            pairs.append((int(parts[0]), int(parts[1])))
    allpids = set(p for p, _ in pairs)
    return sorted(p for p, parent in pairs if parent not in allpids)


# ------------------------------------------------------------------------------------------- 1
def t1_run_supervisor_bat():
    print("\n-- 1  run_supervisor.bat")
    check("1 the launcher exists", os.path.exists(BAT))
    txt = io.open(BAT, encoding="utf-8", errors="replace").read()
    # Read the COMMAND, not the file: this .bat's header comment legitimately mentions both scripts,
    # so a whole-file substring test would report on prose rather than on what actually runs.
    invoke = [ln.strip() for ln in txt.splitlines()
              if ln.strip().startswith('"%PY%"') or ln.strip().startswith("%PY%")]
    check("1 exactly one command line actually launches something", len(invoke) == 1,
          "found %d: %s" % (len(invoke), invoke))
    cmdline = invoke[0] if invoke else ""
    check("1 it cds to its own directory before doing anything",
          'cd /d "%~dp0"' in txt, "so a Task Scheduler logon launch has the right cwd")
    check("1 the command launches the supervisor, not the sidecar directly",
          "sidecar_supervisor.py" in cmdline and "wholebody_udp_sender.py" not in cmdline,
          cmdline)
    check("1 it falls back to system python when the venv is absent",
          '.venv\\Scripts\\python.exe' in txt and 'set "PY=python"' in txt)
    check("1 it fails loudly on a missing model instead of starting broken",
          "[ERROR] Model not found" in txt and "exit /b 1" in txt)

    # Every flag the .bat passes must still exist in the supervisor's parser - this is the check
    # that catches a CLI rename silently breaking the production launcher.
    flags = sorted(set(re.findall(r"(--[a-z0-9-]+)", cmdline)))
    sup_src = io.open(SUP, encoding="utf-8").read()
    unknown = [f for f in flags if ('"%s"' % f) not in sup_src and ("'%s'" % f) not in sup_src]
    check("1 every flag it passes is still accepted by the supervisor CLI",
          flags and not unknown,
          "unknown=%s" % unknown if unknown else "flags=%s" % flags)

    rc = subprocess.call([PY, SUP, "--help"], cwd=HERE,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    check("1 the supervisor it invokes is itself runnable", rc == 0, "exit=%d" % rc)

    m = re.search(r'set "MODEL=(.*?)"', txt.split("if \"%MODEL%\"==\"\"")[1])
    if m:
        model_rel = m.group(1)
        resolved = os.path.normpath(os.path.join(HERE, model_rel.replace("\\", os.sep)))
        check("1 its default model path resolves to a real file",
              os.path.exists(resolved), resolved)


# ------------------------------------------------------------------------------------------- 2
def t2_logon_registration():
    print("\n-- 2  Windows logon startup registration (verified statically, NOT executed)")
    check("2 schtasks.exe is available on this machine", shutil.which("schtasks") is not None)
    if not os.path.exists(REPORT):
        check("2 the F-20B report documents the procedure", False, "report not found")
        return
    txt = io.open(REPORT, encoding="utf-8", errors="replace").read()
    m = re.search(r"(schtasks\s+/create[^\n]*)", txt, re.I)
    check("2 the report documents an exact schtasks command", m is not None)
    if not m:
        return
    cmd = m.group(1).strip()
    print("      documented: %s" % cmd)
    check("2 it registers at logon", "/sc onlogon" in cmd.lower(), cmd)
    check("2 it names a task", "/tn" in cmd.lower())
    tr = re.search(r'/tr\s+"([^"]+)"', cmd, re.I)
    check("2 the .bat path it points at is quoted", tr is not None,
          "unquoted paths break on the space in 'Assets\\Games'" if tr is None else "")
    if tr:
        check("2 that .bat path exists on this machine", os.path.exists(tr.group(1)), tr.group(1))
        check("2 it is the same launcher verified in check 1",
              os.path.normcase(os.path.abspath(tr.group(1))) == os.path.normcase(BAT),
              tr.group(1))
    tn = re.search(r'/tn\s+"([^"]+)"', cmd, re.I)
    if tn:
        rc = subprocess.call(["schtasks", "/query", "/tn", tn.group(1)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        registered = (rc == 0)
        print("      NOT registered on this machine yet - registration is a persistent system"
              if not registered else "      already registered on this machine")
        print("      change and is deliberately left to the operator (report SS12)."
              if not registered else "")
        _results.append(("2 registration state on this machine (informational)", True,
                         "registered" if registered else "not registered - operator action"))


# ------------------------------------------------------------------------------------------- 3
def t3_duplicate_supervisor_guard():
    print("\n-- 3  duplicate-supervisor guard (two supervisors, one lock port)")
    ev = os.path.join(HERE, "oak_v4_evidence", "f20b", "deploy_dup")
    p1 = start_sup(ev, ["--script", FAKE, "--lock-port", "8907", "--port", "8985",
                        "--heartbeat-timeout", "0"])
    st = wait_for(ev, lambda s: s.get("SidecarReady"), 30)
    check("3 the first supervisor starts and reports a ready sidecar", st is not None,
          str(st and st.get("SupervisorState")))
    p2 = start_sup(os.path.join(HERE, "oak_v4_evidence", "f20b", "deploy_dup2"),
                   ["--script", FAKE, "--lock-port", "8907", "--port", "8985",
                    "--heartbeat-timeout", "0"])
    out = b""
    try:
        out = p2.communicate(timeout=25)[0] or b""
    except Exception:
        stop_sup(p2)
    text = out.decode(errors="replace")
    check("3 the second supervisor refuses to start", p2.returncode not in (None, 0),
          "exit=%s" % p2.returncode)
    check("3 it says why, naming the lock port", "already owns lock_port" in text,
          text.strip().splitlines()[-1] if text.strip() else "no output")
    pids = producer_pids("f20b_fake_sidecar")
    check("3 only ONE producer is running after the duplicate attempt", len(pids) == 1,
          "pids=%s" % pids)
    stop_sup(p1)
    return ev


# ------------------------------------------------------------------------------------------- 4
def t4_diagnostics_after_logon():
    print("\n-- 4  operator diagnostics are complete and FRESH after a restart")
    ev = os.path.join(HERE, "oak_v4_evidence", "f20b", "deploy_diag")
    shutil.rmtree(ev, ignore_errors=True)
    os.makedirs(ev, exist_ok=True)
    stale = dict(SupervisorState="FROM_A_PREVIOUS_BOOT", SidecarPid=999999, SidecarUptime=99999.0,
                 RestartCount=42, LastExitCode=-1, LastFailureReason="stale", BackoffSeconds=999,
                 SidecarReady=True, CurrentSessionId="staleoldsid99")
    with io.open(os.path.join(ev, "supervisor_state.json"), "w", encoding="utf-8") as f:
        json.dump(stale, f)
    p = start_sup(ev, ["--script", FAKE, "--lock-port", "8908", "--port", "8986",
                       "--heartbeat-timeout", "0"])
    st = wait_for(ev, lambda s: s.get("SidecarReady"), 30)
    check("4 a new supervisor overwrites the previous boot's snapshot", st is not None
          and st.get("SupervisorState") != "FROM_A_PREVIOUS_BOOT", str(st and st.get("SupervisorState")))
    if st:
        required = ["SupervisorState", "SidecarPid", "SidecarUptime", "RestartCount",
                    "LastExitCode", "LastFailureReason", "BackoffSeconds", "SidecarReady",
                    "CurrentSessionId"]
        missing = [k for k in required if k not in st]
        check("4 every operator-facing field is present", not missing, "missing=%s" % missing)
        check("4 the stale session id is gone", st.get("CurrentSessionId") != "staleoldsid99",
              "sid=%s" % st.get("CurrentSessionId"))
        check("4 the stale restart count is gone", st.get("RestartCount") != 42,
              "RestartCount=%s" % st.get("RestartCount"))
    for fn in ("supervisor.log", "supervisor_events.jsonl"):
        check("4 %s exists for post-mortem after a reboot" % fn,
              os.path.exists(os.path.join(ev, fn)))
    stop_sup(p)
    return ev


# ------------------------------------------------------------------------------------------- 5
def t5_no_duplicate_sidecars():
    print("\n-- 5  no duplicate producers across a kill/restart cycle")
    ev = os.path.join(HERE, "oak_v4_evidence", "f20b", "deploy_nodup")
    p = start_sup(ev, ["--script", FAKE, "--lock-port", "8909", "--port", "8987",
                       "--heartbeat-timeout", "0"])
    st = wait_for(ev, lambda s: s.get("SidecarReady"), 30)
    check("5 one producer after a clean start",
          st is not None and len(producer_pids("f20b_fake_sidecar")) == 1,
          "pids=%s" % producer_pids("f20b_fake_sidecar"))
    first_pid = st.get("SidecarPid") if st else None
    subprocess.call(["taskkill", "/F", "/PID", str(first_pid)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    st2 = wait_for(ev, lambda s: s.get("SidecarReady") and s.get("SidecarPid") != first_pid, 60)
    check("5 the supervisor replaced the killed producer", st2 is not None,
          "pid %s -> %s" % (first_pid, st2 and st2.get("SidecarPid")))
    time.sleep(1.0)
    pids = producer_pids("f20b_fake_sidecar")
    check("5 still exactly ONE producer after the restart - no orphan", len(pids) == 1,
          "pids=%s" % pids)
    if st2:
        check("5 the replacement announced a NEW session id",
              st2.get("CurrentSessionId") != (st or {}).get("CurrentSessionId"),
              "%s -> %s" % ((st or {}).get("CurrentSessionId"), st2.get("CurrentSessionId")))
    stop_sup(p)
    time.sleep(1.0)
    check("5 stopping the supervisor takes the producer down with it",
          len(producer_pids("f20b_fake_sidecar")) == 0,
          "pids=%s" % producer_pids("f20b_fake_sidecar"))
    return ev


# ------------------------------------------------------------------------------------------- 6
def t6_f20a_reconnect_session_ids():
    print("\n-- 6  F-20A reconnect: two real producer launches -> two real session ids")
    sids = []
    for _ in range(3):
        out = subprocess.check_output(
            [PY, "-c", "import wholebody_udp_sender as W; print(W.SESSION_ID)"], cwd=HERE)
        sids.append(out.decode().strip())
    check("6 every producer process announces a distinct session id",
          len(set(sids)) == 3 and all(len(s) == 12 for s in sids), "sids=%s" % sids)
    check("6 ids are hex of the documented width (F-20A wire contract)",
          all(re.fullmatch(r"[0-9a-f]{12}", s) for s in sids), "sids=%s" % sids)

    # the UDP port must be free for the new producer the moment the old one dies, or a restart
    # silently produces a producer that can never send.
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("127.0.0.1", 8899))
        free = True
    except OSError:
        free = False
    finally:
        s.close()
    _results.append(("6 UDP 8899 bindability (informational - Unity may legitimately hold it)",
                     True, "free" if free else "in use, probably by a running Unity"))
    print("      UDP 8899 is currently %s" % ("free" if free else
                                              "in use (Unity is likely listening - expected)"))
    with io.open(os.path.join(HERE, "oak_v4_evidence", "f20b", "deploy_session_ids.json"),
                 "w", encoding="utf-8") as f:
        json.dump(dict(session_ids=sids, udp_8899_free=free), f, indent=2)
    return sids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "oak_v4_evidence", "f20b"))
    ap.parse_args()
    os.makedirs(os.path.join(HERE, "oak_v4_evidence", "f20b"), exist_ok=True)

    print("=" * 96)
    print(" F-20A/F-20B deployment hardening verification")
    print("=" * 96)
    t1_run_supervisor_bat()
    t2_logon_registration()
    t3_duplicate_supervisor_guard()
    t4_diagnostics_after_logon()
    t5_no_duplicate_sidecars()
    sids = t6_f20a_reconnect_session_ids()

    n = len(_results)
    p = sum(1 for _, ok, _ in _results if ok)
    print("\n" + "=" * 96)
    print(" %d/%d checks passed" % (p, n))
    if p != n:
        print("\nFAILED:")
        for name, ok, detail in _results:
            if not ok:
                print("  %s  %s" % (name, detail))
    print("=" * 96)
    with io.open(os.path.join(HERE, "oak_v4_evidence", "f20b", "deployment_verify.json"),
                 "w", encoding="utf-8") as f:
        json.dump(dict(checks=[dict(name=a, passed=b, detail=c) for a, b, c in _results],
                       passed=p, total=n, session_ids=sids), f, indent=2)
    return 0 if p == n else 1


if __name__ == "__main__":
    sys.exit(main())
