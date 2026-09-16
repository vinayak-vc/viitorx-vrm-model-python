#!/usr/bin/env python3
"""TORSO YAW V4 — sweep kalidokitTorsoYawScale over the five recorded V3 video frames.

The scale is set at RUNTIME by reflecting AppBootstrap's serialized field, which AppBootstrap pushes
into the driver every frame (`SetTorsoYawScale(kalidokitTorsoYawScale)`). Nothing on disk is modified
during the sweep, so the whole thing is reversible by leaving Play mode.

Each (scale, frame) cell: set the scale, call Recalibrate() so the ADR-027 yaw conditioners
(rate limiter, low-pass, dead zone) re-seed and the cell does not inherit the previous cell's slew
state, hold the recorded frame on the real UDP wire, let it settle, then snapshot every layer.

    python sweep_yaw.py                    # default scales, all five frames
    python sweep_yaw.py 0.7 0.714 0.75     # extra scales
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ubridge

REPO = r"C:\Unity\viitorx-vrm-avtar-unity-base-project\Assets\Games\viitorx-vrm-avtar-unity"
EVID = os.path.join(REPO, "python-sidecar~", "arm_v3_evidence")
DUMP = os.path.join(HERE, "wire_p0.jsonl")
FRAMES = [24, 177, 273, 279, 359]          # f084 f237 f333 f339 f419
SCALES = [0.0, 0.25, 0.5, 0.75, 1.0]
SETTLE_S, HOLD_S = 5.0, 22.0

SET_SCALE = '''
var boot = UnityEngine.Object.FindObjectsByType<VirtualMirror.App.AppBootstrap>(UnityEngine.FindObjectsSortMode.None)[0];
var f = typeof(VirtualMirror.App.AppBootstrap).GetField("kalidokitTorsoYawScale",
    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
f.SetValue(boot, SCALEVALUEf);
var df = typeof(VirtualMirror.App.AppBootstrap).GetField("kalidokitControlRig",
    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
var drv = df.GetValue(boot) as VirtualMirror.Retargeting.KalidokitControlRigDriver;
drv.SetTorsoYawScale(SCALEVALUEf);
drv.Recalibrate();      // re-seed the ADR-027 yaw conditioners so this cell is independent
var rf = typeof(VirtualMirror.Retargeting.KalidokitControlRigDriver).GetField("torsoYawScale",
    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
return "torsoYawScale now " + rf.GetValue(drv);
'''


def seqmap():
    m, seq0 = {}, None
    with open(DUMP, encoding="utf-8") as fh:
        for l in fh:
            r = json.loads(l)
            if seq0 is None:
                seq0 = r["msg"]["seq"]
            m[r["index"]] = 60 + (r["msg"]["seq"] - seq0)
    return m


def main():
    scales = [float(x) for x in sys.argv[1:]] or SCALES
    frame_of = seqmap()
    measure = open(os.path.join(HERE, "measure.cs.txt"), encoding="utf-8").read()
    shot = open(os.path.join(HERE, "shot.cs.txt"), encoding="utf-8").read()
    for sc in scales:
        r = ubridge.execute_code(SET_SCALE.replace("SCALEVALUE", repr(sc)))
        print(f"\n########## scale {sc}  -> {(r.get('data') or {}).get('result', r)}", flush=True)
        for idx in FRAMES:
            vf = frame_of[idx]
            tag = f"yaw{sc:g}".replace(".", "p")
            label = f"{tag}_f{vf:03d}"
            proc = subprocess.Popen(
                [sys.executable, os.path.join(HERE, "wire_record.py"), "hold",
                 "--dump", DUMP, "--index", str(idx), "--seconds", str(HOLD_S)],
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            try:
                time.sleep(SETTLE_S)
                m = ubridge.execute_code(measure.replace("LABEL", label))
                sh = ubridge.execute_code(shot.replace("SHOTNAME", label))
                print(f"  {label}: {(m.get('data') or {}).get('result', m)}", flush=True)
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            time.sleep(0.3)
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
