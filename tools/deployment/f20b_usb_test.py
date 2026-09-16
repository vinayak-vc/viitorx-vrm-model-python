#!/usr/bin/env python3
"""F-20B SS17 live protocol - the scenarios that genuinely need a person at the camera (D, E, F, H).
Same shape as f20a_usb_test.py (fullscreen HUD + beeps - the subject is ~0.9 m from the machine and
cannot read a console, per the live-capture-subject-needs-visible-feedback lesson), but this drives
the SUPERVISOR (sidecar_supervisor.py) instead of the raw sidecar, and adds the Unity-restart phase
F-20A's live protocol didn't need.

  --mode live     (default) supervisor+sidecar start healthy, then: Unity gets restarted while both
                  stay alive (H), then a real USB unplug -> gap -> replug (D, E).
  --mode absent   the camera is unplugged BEFORE this script even starts (F) - confirms the
                  supervisor retries with backoff instead of giving up, then recovers once plugged in.

    python tools/deployment/f20b_usb_test.py --mode live
    python tools/deployment/f20b_usb_test.py --mode absent    (unplug the OAK-D first, THEN run this)
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
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, ".venv", "Scripts", "python.exe")
SUPERVISOR = os.path.join(HERE, "sidecar_supervisor.py")
MODEL = os.path.join(HERE, "..", "..", "..", "SentisModel", "rtmw3d-x.onnx")
EVIDENCE_DIR = EV.oak_v4("f20b", "live")
BLOCK_FILE = os.path.join(EVIDENCE_DIR, "block.txt")
TIMELINE = os.path.join(EVIDENCE_DIR, "timeline.jsonl")
STATE_FILE = os.path.join(EVIDENCE_DIR, "supervisor_state.json")
W, H = 1400, 760
GREEN = (90, 230, 90)
AMBER = (0, 190, 255)
RED = (60, 60, 255)
WHITE = (255, 255, 255)

events = []


def label(name):
    io.open(BLOCK_FILE, "w", encoding="utf-8").write(name)
    events.append(dict(t=round(time.time(), 3), phase=name))
    print("[f20b-live] %-20s %s" % (name, time.strftime("%H:%M:%S")), flush=True)


def note(kind, detail=""):
    events.append(dict(t=round(time.time(), 3), event=kind, detail=detail))
    print("[f20b-live]   %s %s" % (kind, detail), flush=True)


def read_state():
    try:
        return json.load(io.open(STATE_FILE, encoding="utf-8"))
    except Exception:
        return None


def hud(cue, sub, secs, colour=GREEN, show_state=False):
    """Same fullscreen-prompt pattern as f20a_usb_test.py's hud(): a countdown bar, no keyboard
    interaction required, so the subject at the camera never needs to touch the machine."""
    t0 = time.time()
    while True:
        left = secs - (time.time() - t0)
        if left <= 0:
            return True
        img = np.zeros((H, W, 3), np.uint8)
        (tw, _), _ = cv2.getTextSize(cue, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 5)
        cv2.putText(img, cue, (max(10, (W - tw) // 2), 260), cv2.FONT_HERSHEY_SIMPLEX, 2.0, colour,
                    5, cv2.LINE_AA)
        (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(img, sub, (max(10, (W - sw) // 2), 330), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2,
                    cv2.LINE_AA)
        cv2.putText(img, "%.0f s" % left, (W // 2 - 40, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    (150, 150, 150), 3)
        bw = int((W - 80) * (1.0 - left / secs))
        cv2.rectangle(img, (40, H - 90), (W - 40, H - 50), (55, 55, 55), -1)
        cv2.rectangle(img, (40, H - 90), (40 + bw, H - 50), colour, -1)
        if show_state:
            st = read_state() or {}
            info = "supervisor=%s ready=%s sid=%s restarts=%s" % (
                st.get("SupervisorState"), st.get("SidecarReady"),
                st.get("CurrentSessionId"), st.get("RestartCount"))
            cv2.putText(img, info, (30, H - 20), cv2.FONT_HERSHEY_PLAIN, 1.1, (180, 180, 180), 1)
        cv2.imshow("F-20B SUPERVISOR LIVE TEST", img)
        if (cv2.waitKey(30) & 0xFF) in (27, ord("q")):
            return False


def start_supervisor():
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    log = io.open(os.path.join(EVIDENCE_DIR, "supervisor_stdout.txt"), "w", encoding="utf-8")
    cmd = [PY, "-u", SUPERVISOR, "--evidence-dir", EVIDENCE_DIR, "--model", MODEL,
           "--portrait", "--portrait-dir", "ccw", "--subpixel-bits", "3",
           "--host", "127.0.0.1", "--port", "8899"]
    p = subprocess.Popen(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
    note("supervisor_spawn", "pid=%d (this is the launcher pid, not the real interpreter's - see "
                              "taskkill /T notes in the report)" % p.pid)
    return p, log


def wait_ready(timeout, prior_sid=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = read_state()
        if st and st.get("SidecarReady") and st.get("CurrentSessionId") != prior_sid:
            return st, time.time() - t0
        time.sleep(0.3)
    return read_state(), time.time() - t0


def run_live(gap_seconds, unity_restart_pause):
    proc, log = start_supervisor()
    try:
        label("SUP_STARTING")
        if not hud("STARTING", "supervisor is launching the sidecar - please wait", 8):
            return
        st, elapsed = wait_ready(60)
        note("sup_ready", "sid=%s after %.1fs" % (st and st.get("CurrentSessionId"), elapsed))

        label("SUP_LIVE")
        if not hud("TRACKING - MOVE ABOUT", "confirming the avatar is live before the tests", 12,
                    show_state=True):
            return

        label("UNITY_RESTART_CUE")
        note("unity_restart_cue_start")
        if not hud("STOP AND RESTART UNITY PLAY MODE NOW",
                   "the sidecar must keep running the whole time - do not touch this window",
                   unity_restart_pause, AMBER, show_state=True):
            return
        note("unity_restart_cue_end", "supervisor_restart_count=%s"
             % (read_state() or {}).get("RestartCount"))

        label("UNITY_RESTART_CONFIRM")
        if not hud("TRACKING AGAIN IN UNITY?", "move about - the avatar should follow", 12,
                   show_state=True):
            return

        label("USB_UNPLUG")
        st_before = read_state() or {}
        note("unplug_cue_start", "sid_before=%s" % st_before.get("CurrentSessionId"))
        if not hud("UNPLUG THE CAMERA NOW", "pull the OAK-D USB cable out", 25, AMBER,
                   show_state=True):
            return

        label("USB_GAP")
        if not hud("LEAVE IT UNPLUGGED", "watching what the supervisor does", gap_seconds, RED,
                   show_state=True):
            return
        note("gap_ended", "state=%s" % (read_state() or {}))

        label("USB_REPLUG")
        if not hud("PLUG THE CAMERA BACK IN", "then wait", 15, AMBER, show_state=True):
            return

        label("USB_RESTARTING")
        if not hud("RESTARTING", "the supervisor is re-opening the camera", 15, AMBER,
                   show_state=True):
            return

        label("USB_RECOVERED")
        st_after, t_recover = wait_ready(60, prior_sid=st_before.get("CurrentSessionId"))
        note("recovered", "new_sid=%s T_recover_from_replug_cue=%.1fs"
             % (st_after and st_after.get("CurrentSessionId"), t_recover))
        hud("TRACKING AGAIN?", "move about - the avatar should follow", 12, GREEN, show_state=True)
        label("DONE")
    finally:
        cv2.destroyAllWindows()
        if proc.poll() is None:
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if log is not None and not log.closed:
            log.close()


def run_absent():
    label("CAMERA_SHOULD_BE_UNPLUGGED")
    if not hud("IS THE CAMERA UNPLUGGED?", "this tests startup with NO camera - unplug it now if not",
               10, AMBER):
        return
    proc, log = start_supervisor()
    try:
        label("SUP_RETRYING")
        if not hud("SUPERVISOR STARTING WITH NO CAMERA", "it should retry with backoff, not give up",
                   20, AMBER, show_state=True):
            return
        st = read_state() or {}
        note("absent_state", "state=%s restarts=%s" % (st.get("SupervisorState"), st.get("RestartCount")))

        label("USB_REPLUG")
        if not hud("PLUG THE CAMERA IN NOW", "then wait for recovery", 15, AMBER, show_state=True):
            return

        label("USB_RECOVERED")
        st_after, t_recover = wait_ready(60)
        note("recovered", "sid=%s T_recover=%.1fs" % (st_after and st_after.get("CurrentSessionId"),
                                                        t_recover))
        hud("TRACKING?", "move about - the avatar should follow", 12, GREEN, show_state=True)
        label("DONE")
    finally:
        cv2.destroyAllWindows()
        if proc.poll() is None:
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if log is not None and not log.closed:
            log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["live", "absent"], default="live")
    ap.add_argument("--gap", type=float, default=45.0, help="seconds the camera stays unplugged (D/E)")
    ap.add_argument("--unity-restart-pause", type=float, default=25.0,
                     help="seconds given to Stop+Play Unity again (H)")
    a = ap.parse_args()
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    cv2.namedWindow("F-20B SUPERVISOR LIVE TEST", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("F-20B SUPERVISOR LIVE TEST", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        if a.mode == "live":
            run_live(a.gap, a.unity_restart_pause)
        else:
            run_absent()
    finally:
        cv2.destroyAllWindows()
        io.open(TIMELINE, "w", encoding="utf-8").write("".join(json.dumps(e) + "\n" for e in events))
        print("[f20b-live] wrote %s" % TIMELINE)


if __name__ == "__main__":
    main()
