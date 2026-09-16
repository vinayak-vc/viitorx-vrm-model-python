#!/usr/bin/env python3
"""F-20A section 11 - camera disconnect / reconnect, with a real USB unplug.

Prompts the operator on a fullscreen HUD (they are at the camera, not the keyboard), starts the
sidecar, waits for the unplug, records what the sidecar process does when the device vanishes, then
waits for the replug and restarts the sidecar. Phase names go into the same block file the Unity
recorder polls.
"""

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(_os.path.join(_d, "_sidecar_path.py")):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import _sidecar_path  # noqa: F401  - puts the sidecar root and every tools/ group on sys.path
import evidence_paths as EV

import io, json, os, subprocess, time
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, ".venv", "Scripts", "python.exe")
MODEL = EV.DEFAULT_MODEL
BLOCK = EV.oak_v4("f20a", "block.txt")
OUT = EV.oak_v4("f20a", "usb_timeline.jsonl")
LOG = EV.oak_v4("f20a", "log_usb")
W, H = 1400, 700
ev = []


def label(n):
    io.open(BLOCK, "w", encoding="utf-8").write(n)
    ev.append(dict(t=round(time.time(), 3), phase=n))
    print("[usb] %s" % n, flush=True)


def note(k, d=""):
    ev.append(dict(t=round(time.time(), 3), event=k, detail=d))
    print("[usb]   %s %s" % (k, d), flush=True)


def hud(cue, sub, secs, colour=(90, 230, 90)):
    t0 = time.time()
    while time.time() - t0 < secs:
        img = np.zeros((H, W, 3), np.uint8)
        (tw, _), _ = cv2.getTextSize(cue, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 6)
        cv2.putText(img, cue, ((W - tw) // 2, 300), cv2.FONT_HERSHEY_SIMPLEX, 2.2, colour, 6, cv2.LINE_AA)
        (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(img, sub, ((W - sw) // 2, 380), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        left = secs - (time.time() - t0)
        cv2.putText(img, "%.0f s" % left, (W // 2 - 40, 470), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (150, 150, 150), 3)
        bw = int((W - 80) * (1.0 - left / secs))
        cv2.rectangle(img, (40, H - 80), (W - 40, H - 40), (55, 55, 55), -1)
        cv2.rectangle(img, (40, H - 80), (40 + bw, H - 40), colour, -1)
        cv2.imshow("F-20A USB TEST", img)
        if (cv2.waitKey(30) & 0xFF) in (27, ord("q")):
            return False
    return True


def start():
    os.makedirs(LOG, exist_ok=True)
    out = io.open(os.path.join(LOG, "stdout_%d.txt" % int(time.time())), "w", encoding="utf-8")
    p = subprocess.Popen([PY, "-u", "wholebody_udp_sender.py", "--model", MODEL, "--portrait",
                          "--subpixel-bits", "3", "--seconds", "0"],
                         cwd=HERE, stdout=out, stderr=subprocess.STDOUT)
    note("sidecar_spawn", "pid=%d log=%s" % (p.pid, out.name))
    return p, out


cv2.namedWindow("F-20A USB TEST", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("F-20A USB TEST", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
os.makedirs(os.path.dirname(BLOCK), exist_ok=True)
proc, fh = start()
label("USB_STARTING")
hud("STARTING", "waiting for the camera pipeline", 25)
label("USB_LIVE")
hud("TRACKING - MOVE ABOUT", "confirming the avatar is live before the unplug", 15)

label("USB_UNPLUG")
hud("UNPLUG THE CAMERA NOW", "pull the OAK-D USB cable out", 12, (0, 190, 255))
note("unplug_cue_ended", "sidecar alive=%s rc=%s" % (proc.poll() is None, proc.poll()))
label("USB_GAP")
hud("LEAVE IT UNPLUGGED", "watching what the avatar does", 12, (60, 60, 255))
note("gap_ended", "sidecar alive=%s rc=%s" % (proc.poll() is None, proc.poll()))

label("USB_REPLUG")
hud("PLUG THE CAMERA BACK IN", "then wait", 15, (0, 190, 255))
if proc.poll() is None:
    note("sidecar_survived_unplug", "terminating it so a clean restart can be tested")
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except Exception:
        subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
else:
    note("sidecar_died_on_unplug", "rc=%s" % proc.returncode)
proc, fh2 = start()
label("USB_RESTART")
hud("RESTARTING", "the sidecar is re-opening the camera", 30)
label("USB_RECOVERED")
hud("TRACKING AGAIN?", "move about - the avatar should follow", 15)
label("DONE")
cv2.destroyAllWindows()
if proc.poll() is None:
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except Exception:
        pass
io.open(OUT, "w", encoding="utf-8").write("".join(json.dumps(e) + "\n" for e in ev))
print("[usb] wrote %s" % OUT)
