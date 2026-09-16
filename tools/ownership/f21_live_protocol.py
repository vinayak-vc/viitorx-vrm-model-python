#!/usr/bin/env python3
"""F-21 section 17 - live two-person protocol, run against the REAL sidecar.

REV 2: the first live run (2026-09-14) found that printed console instructions are unreadable while
physically coordinating two people and watching the camera preview at the same time - the operator
could not follow the protocol and the result was inconclusive. Fixed the same way F-20A/F-20B's
lessons say to: the instruction gets drawn AS A BANNER directly on the SAME window the operator is
already watching (wholebody_udp_sender.py's --show preview, via --cue-file), right above the F-21 HUD
(state/owner/timers/switch count) - one window, not a window plus a scrolling terminal.

REV 3 (F-21 S31), both changes required before the next live session:

  A. DEAD PRODUCER DETECTION. The second live attempt lost the DepthAI device part-way through and
     this script kept marching through its phases, writing a timeline that reads like a completed
     protocol. Every phase after the crash was scored against a producer that no longer existed.
     Now the child is polled during every wait and between every phase, and a death ABORTS with the
     return code, the phase it died in, the wall-clock time, and the last ownership state read back
     out of the sidecar's own target_events.jsonl - so the record says what was actually observed
     and what was not.

  B. OPTIONAL F-20B SUPERVISION (--supervised). A device crash during a two-person protocol
     currently destroys the whole experiment; under the supervisor it becomes a restart. The two
     modes have DIFFERENT abort conditions, which is the part worth being careful about:

         direct      the sidecar exiting is fatal - nothing will bring it back.
         supervised  the sidecar exiting is EXPECTED and gets restarted; only the SUPERVISOR
                     exiting is fatal. Restarts are recorded from supervisor_events.jsonl and
                     reported at the end, because a phase that straddles a restart was observed
                     across an ownership epoch reset and must not be read as continuous.

    python f21_live_protocol.py                  # direct, as before
    python f21_live_protocol.py --supervised     # under the F-20B watchdog
"""
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(_os.path.join(_d, "_sidecar_path.py")):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import _sidecar_path  # noqa: F401  - puts the sidecar root and every tools/ group on sys.path
import evidence_paths as EV

import argparse
import io
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, ".venv", "Scripts", "python.exe")
SCRIPT = os.path.join(HERE, "wholebody_udp_sender.py")
SUPERVISOR = os.path.join(HERE, "sidecar_supervisor.py")
MODEL = os.path.join(HERE, "..", "..", "..", "SentisModel", "rtmw3d-x.onnx")
EVIDENCE = EV.oak_v4("f21", "live")
TIMELINE = os.path.join(EVIDENCE, "timeline.jsonl")
CUE_FILE = os.path.join(EVIDENCE, "cue.json")
SUP_EVIDENCE = os.path.join(EVIDENCE, "supervisor")

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
    # S30's gate is the reason this one exists: the impostor case and the returning-owner case are
    # geometrically identical from a hip position, and the gate resolves both towards a DECLARED
    # hand-over. This phase is what prices that decision against a real room.
    ("WALK_IN_IMPOSTOR", "Owner hides. OTHER person walks slowly to owner's exact spot", 18),
    ("WALK_IN_OWNER",    "Owner returns to that same spot the same way", 15),
    ("DONE",             "Protocol complete - press q in the preview to stop", 6),
]

events = []


class ProducerDied(Exception):
    def __init__(self, rc, phase, what):
        Exception.__init__(self, "%s exited rc=%s during %s" % (what, rc, phase))
        self.rc = rc
        self.phase = phase
        self.what = what


def write_cue(title, instruction, seconds_left):
    """Publish the cue atomically, and never die because a write lost a race.

    os.replace() raised PermissionError [WinError 5] mid-protocol during F-21 S32 bring-up and took
    the whole run down. On Windows a rename over a file another process holds open fails, and the
    sidecar re-reads cue.json on EVERY preview frame (~30 Hz) while this writes it at ~7 Hz. The
    collision is expected, not rare. A cue update is worth a fraction of a second of countdown; the
    session is worth two people's afternoon - so retry briefly, then skip."""
    tmp = CUE_FILE + ".tmp"
    try:
        with io.open(tmp, "w", encoding="utf-8") as f:
            json.dump(dict(title=title, instruction=instruction, seconds_left=seconds_left), f)
        for _ in range(5):
            try:
                os.replace(tmp, CUE_FILE)
                return
            except PermissionError:
                time.sleep(0.01)
    except (IOError, OSError):
        pass


# Where the SIDECAR writes its ownership log in each mode. Direct mode passes --ownership-log-dir
# explicitly; supervised mode does not (the supervisor forwards only --show/--cue-file, deliberately
# - see its build_command), so the sidecar uses its own argparse default of evidence/oak_v4/f21
# relative to its cwd, which the supervisor sets to HERE.
OWNERSHIP_LOG_DIR = {"direct": EVIDENCE, "supervised": EV.oak_v4("f21")}
_mode = "direct"


def last_ownership_state():
    """The last thing F-21 said before the producer went away. Diagnostic only - read from the
    sidecar's own log, never from memory, so it is true even if this process was busy sleeping."""
    path = os.path.join(OWNERSHIP_LOG_DIR[_mode], "target_events.jsonl")
    try:
        lines = [l for l in io.open(path, encoding="utf-8") if l.strip()]
    except IOError:
        return None
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except ValueError:
        return None


def count_restarts():
    """Sidecar restarts the supervisor performed, so a phase spanning one is not read as continuous."""
    path = os.path.join(SUP_EVIDENCE, "supervisor_events.jsonl")
    out = []
    try:
        for l in io.open(path, encoding="utf-8"):
            l = l.strip()
            if not l:
                continue
            try:
                e = json.loads(l)
            except ValueError:
                continue
            if e.get("event") in ("SidecarExited", "StartupStuck", "HeartbeatTimeout",
                                  "CrashLoop", "PortInUse"):
                out.append(e)
    except IOError:
        pass
    return out


def check_alive(proc, phase, what):
    rc = proc.poll()
    if rc is not None:
        raise ProducerDied(rc, phase, what)


def run_phase(proc, title, instruction, seconds, what):
    check_alive(proc, title, what)
    events.append(dict(t=round(time.time(), 3), phase=title, instruction=instruction))
    print("[%-17s] %s (%ds)" % (title, instruction, seconds), flush=True)
    t0 = time.time()
    while True:
        left = seconds - (time.time() - t0)
        if left <= 0:
            break
        check_alive(proc, title, what)          # <- REV 3A: not only between phases
        write_cue(title, instruction, left)
        time.sleep(0.15)
    check_alive(proc, title, what)
    events.append(dict(t=round(time.time(), 3), event="phase_end", phase=title))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--supervised", action="store_true",
                    help="run the sidecar under sidecar_supervisor.py (F-20B), so a DepthAI crash "
                         "mid-protocol is a restart instead of the end of the experiment")
    a = ap.parse_args()

    global _mode
    _mode = "supervised" if a.supervised else "direct"
    os.makedirs(EVIDENCE, exist_ok=True)
    if a.supervised:
        os.makedirs(SUP_EVIDENCE, exist_ok=True)
        what = "supervisor"
        cmd = [PY, "-u", SUPERVISOR, "--model", MODEL, "--portrait", "--portrait-dir", "ccw",
               "--subpixel-bits", "3", "--evidence-dir", SUP_EVIDENCE,
               "--show", "--cue-file", CUE_FILE]
        print("Starting the F-20B SUPERVISOR, which starts the real sidecar with --show + cues.")
        print("A sidecar crash will be RESTARTED; only the supervisor dying aborts this protocol.")
    else:
        what = "sidecar"
        cmd = [PY, "-u", SCRIPT, "--model", MODEL, "--portrait", "--portrait-dir", "ccw",
               "--subpixel-bits", "3", "--seconds", "0", "--show",
               "--ownership-log-dir", EVIDENCE, "--cue-file", CUE_FILE]
        print("Starting the real sidecar with --show + on-screen cues.")
        print("If it exits, this protocol ABORTS - it will not score phases against a dead producer.")
    print("A window titled 'whole-body OAK sidecar' will open with:")
    print("  - a top banner: the CURRENT INSTRUCTION and a countdown")
    print("  - below it: the F-21 HUD (state / owner / timers / switch count / PATH-BLOCKED)")
    print("Watch ONLY that window from here on. Press q or ESC in it to stop early.")
    print()

    write_cue("STARTING", "Loading the model - wait", 8)
    proc = subprocess.Popen(cmd, cwd=HERE)
    aborted = None
    try:
        for title, instruction, seconds in PHASES:
            run_phase(proc, title, instruction, seconds, what)
    except ProducerDied as e:
        aborted = dict(event="LIVE_TEST_ABORTED", reason="%s exited" % e.what,
                       returncode=e.rc, phase=e.phase,
                       t=round(time.time(), 3),
                       wall_clock=time.strftime("%Y-%m-%d %H:%M:%S"),
                       last_ownership_state=last_ownership_state(),
                       completed_phases=[p["phase"] for p in events if "instruction" in p][:-1])
        events.append(aborted)
        print("\n" + "!" * 78)
        print("LIVE TEST ABORTED: %s EXITED" % e.what.upper())
        print("  return code      : %s" % e.rc)
        print("  died during phase: %s" % e.phase)
        print("  wall clock       : %s" % aborted["wall_clock"])
        print("  last F-21 state  : %s" % json.dumps(aborted["last_ownership_state"]))
        print("  phases completed : %d of %d" % (len(aborted["completed_phases"]), len(PHASES)))
        print("  Everything after this point is UNOBSERVED. Do not score it.")
        print("!" * 78)
    finally:
        if proc.poll() is None:
            print("\nStopping...")
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                proc.wait(timeout=8)
            except Exception:
                pass
        if a.supervised:
            restarts = count_restarts()
            if restarts:
                events.append(dict(event="SUPERVISOR_INTERVENTIONS", count=len(restarts),
                                   detail=restarts))
                print("\nSupervisor intervened %d time(s) during the protocol:" % len(restarts))
                for e in restarts:
                    print("   t=%s %s %s" % (e.get("t"), e.get("event"),
                                             {k: v for k, v in e.items()
                                              if k not in ("t", "event")}))
                print("A phase spanning a restart was observed across an ownership epoch reset - "
                      "read those phases with that in mind.")
        io.open(TIMELINE, "w", encoding="utf-8").write(
            "".join(json.dumps(e) + "\n" for e in events))
        print("wrote %s" % TIMELINE)
        print("cross-reference against %s"
              % os.path.join(OWNERSHIP_LOG_DIR[_mode], "target_events.jsonl"))
        if a.supervised:
            print("  and the supervisor's own record: %s"
                  % os.path.join(SUP_EVIDENCE, "supervisor_events.jsonl"))
    return 1 if aborted else 0


if __name__ == "__main__":
    sys.exit(main())
