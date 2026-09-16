#!/usr/bin/env python3

import evidence_paths as EV

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(_os.path.join(_d, "_sidecar_path.py")):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import _sidecar_path  # noqa: F401  - puts the sidecar root and every tools/ group on sys.path
"""Live OAK-D session: measure the trunk-yaw noise floor, and watch the avatar while doing it.

ONE SESSION, TWO ANSWERS. The same recording gives the deadzone its missing number AND exercises the
pelvic-roll change live, because the wire probe forwards every packet to Unity unchanged - so the
avatar is driven normally throughout and can be watched and recorded on the same clock.

    sidecar --port 8900  ->  f24_wire_probe  ->  127.0.0.1:8899  ->  Unity

BLOCKS. The subject reads them off the big cue panel on the preview window (the sidecar's own banner
is ~4x too small to read from the capture position - see f21_cue_display.py):

    SETTLE    get into position, not scored
    STILL     stand still, face the camera, breathe normally. THIS block is the noise floor.
    MOVE      normal mirror movement - turn a little, shift weight, reach. This is signal, and it
              is what the deadzone is spending itself against.
    HIPS      deliberately sway and tilt the hips. Exercises the pelvic-roll change specifically.

Stand at the documented operating distance (0.90 m, portrait). A noise floor measured at the wrong
distance is not the noise floor this product has - F-16 measured the error scaling as Z-squared.

    .venv\\Scripts\\python.exe f24_jitter_session.py
"""
import argparse
import io
import json
import os
import subprocess
import sys
import time

import f21_live_protocol as LP

HERE = os.path.dirname(os.path.abspath(__file__))
EVIDENCE = EV.oak_v4("f24")
CUE_FILE = os.path.join(EVIDENCE, "cue.json")
WIRE = os.path.join(EVIDENCE, "wire.jsonl")

BLOCKS = [
    ("SETTLE", "A", "Stand on the mark, face the camera", "getting ready", 8),
    ("STILL",  "A", "STAND STILL. Face the camera",       "breathe normally, do not shift", 30),
    ("MOVE",   "A", "Move normally. Turn a little",       "shift weight, reach, look around", 30),
    ("HIPS",   "A", "Sway and tilt your HIPS",            "shoulders steady, hips only", 25),
    ("DONE",   "NOBODY", "Finished - thank you",          "", 5),
]


def write_cue(**kw):
    tmp = CUE_FILE + ".tmp"
    try:
        with io.open(tmp, "w", encoding="utf-8") as f:
            json.dump(kw, f)
        for _ in range(5):
            try:
                os.replace(tmp, CUE_FILE)
                return
            except PermissionError:
                time.sleep(0.01)
    except (IOError, OSError):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen-port", type=int, default=8900)
    ap.add_argument("--unity-port", type=int, default=8899)
    a = ap.parse_args()

    os.makedirs(EVIDENCE, exist_ok=True)
    total = sum(b[4] for b in BLOCKS)

    import onnxruntime as ort
    eps = list(ort.get_available_providers())
    print("interpreter : %s" % sys.executable)
    print("providers   : %s" % ", ".join(eps))
    if "DmlExecutionProvider" not in eps:
        print("  !! no DirectML - frame rate will be ~6x low and every per-frame figure with it.")
        return 2

    write_cue(title="STARTING", instruction="Loading the model", detail="stand by",
              actor="NOBODY", seconds_left=8, seconds_total=8)

    sidecar = subprocess.Popen(
        [LP.PY, "-u", LP.SCRIPT, "--model", LP.MODEL, "--portrait", "--portrait-dir", "ccw",
         "--subpixel-bits", "3", "--seconds", "0", "--show", "--cue-file", CUE_FILE,
         "--cue-panel", "--port", str(a.listen_port),
         "--ownership-log-dir", EVIDENCE], cwd=HERE)

    probe = subprocess.Popen(
        [LP.PY, "-u", os.path.join(HERE, "f24_wire_probe.py"),
         "--listen-port", str(a.listen_port), "--forward-port", str(a.unity_port),
         "--seconds", str(total + 20), "--out", WIRE], cwd=HERE)

    print("\nStand at 0.90 m, portrait framing. Read the big panel on the preview window.\n")
    marks = []
    try:
        for title, actor, instruction, detail, secs in BLOCKS:
            t0 = time.time()
            marks.append(dict(block=title, t_start=round(t0, 4), seconds=secs))
            print("  [%-6s] %s (%ds)" % (title, instruction, secs), flush=True)
            while time.time() - t0 < secs:
                if sidecar.poll() is not None:
                    raise SystemExit("\nSIDECAR EXITED rc=%s during %s - nothing after this is "
                                     "observed. Do not score it." % (sidecar.returncode, title))
                write_cue(title=title, instruction=instruction, detail=detail, actor=actor,
                          seconds_left=secs - (time.time() - t0), seconds_total=secs,
                          case="TRUNK JITTER")
                time.sleep(0.1)
            marks[-1]["t_end"] = round(time.time(), 4)
    except KeyboardInterrupt:
        print("\ninterrupted by operator")
    finally:
        write_cue(title="DONE", instruction="Finished", detail="", actor="NOBODY",
                  seconds_left=0, seconds_total=1)
        for p in (sidecar,):
            if p.poll() is None:
                subprocess.call(["taskkill", "/F", "/T", "/PID", str(p.pid)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            probe.wait(timeout=25)
        except Exception:
            probe.kill()
        with io.open(os.path.join(EVIDENCE, "blocks.json"), "w", encoding="utf-8") as f:
            json.dump(marks, f, indent=2)
        print("\nwrote %s" % WIRE)
        print("blocks -> %s" % os.path.join(EVIDENCE, "blocks.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
