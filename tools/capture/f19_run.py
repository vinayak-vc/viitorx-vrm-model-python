#!/usr/bin/env python3
"""F-19 — arm the Unity rendered-bone recorder, replay a recorded portrait block, stop, analyse.

    python tools/capture/f19_run.py <label> <capture> [block] [loops]
"""

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(_os.path.join(_d, "_sidecar_path.py")):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import _sidecar_path  # noqa: F401  - puts the sidecar root and every tools/ group on sys.path
import evidence_paths as EV

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ubridge

SIDE = EV.PROJECT_ROOT
PY = os.path.join(SIDE, ".venv", "Scripts", "python.exe")

STOP = '''
var g = GameObject.Find("F19Recorder");
if (g == null) return "no recorder running";
UnityEngine.Object.DestroyImmediate(g);
return "recorder stop signalled";
'''


def run(body):
    r = ubridge.execute_code(body)
    return (r.get("data") or {}).get("result", r)


def main():
    label = sys.argv[1]
    cap = sys.argv[2]
    block = sys.argv[3] if len(sys.argv) > 3 else ""
    loops = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    run(STOP)                                   # clear any stale recorder
    time.sleep(0.5)
    with open(os.path.join(HERE, "rec_start.cs.txt"), encoding="utf-8") as fh:
        body = fh.read().replace("LABEL", label)
    print(run(body), flush=True)
    time.sleep(1.5)

    cmd = [PY, "f19_replay_portrait.py", "--capture", cap, "--rate", "30", "--loop", str(loops)]
    if block:
        cmd += ["--block", block]
    print("replaying ...", flush=True)
    p = subprocess.run(cmd, cwd=SIDE, capture_output=True, text=True)
    print(p.stdout.strip()[-1200:], flush=True)
    if p.returncode != 0:
        print(p.stderr.strip()[-800:], file=sys.stderr)

    time.sleep(1.0)
    print(run(STOP), flush=True)
    time.sleep(1.5)

    out = os.path.join(SIDE, "f19_evidence", label + ".jsonl")
    if os.path.exists(out):
        print(f"telemetry: {os.path.getsize(out)} bytes -> {out}", flush=True)
        a = subprocess.run([PY, "f19_analyze_bones.py", os.path.join("f19_evidence", label + ".jsonl")],
                           cwd=SIDE, capture_output=True, text=True)
        print(a.stdout, flush=True)
        if a.returncode != 0:
            print(a.stderr[-800:], file=sys.stderr)
    else:
        print("NO TELEMETRY WRITTEN", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
