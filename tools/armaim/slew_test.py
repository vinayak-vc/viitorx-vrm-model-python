#!/usr/bin/env python3
"""TORSO YAW V4 §4 — isolate PARENT CONTAMINATION from temporal lag.

Why the first dynamic attempt was invalid: it compared the provider's NEWEST source frame against
bones driven by a ~40 ms-older frame (P1-3's poseInterpolationDelayMs) further damped by
lerpAmount = 0.5, so on a 30 fps dance it measured temporal lag, not geometry. The tell is that the
case with NO torso rotation had the WORST error, i.e. the error anticorrelated with torso motion.
V2 already documented that lag as pre-existing and not a geometric error.

This test removes lag by construction: the SOURCE IS FROZEN on one recorded frame, and only the torso
moves — because Recalibrate() re-seeds the ADR-027 conditioner to zero, after which the damped yaw
slews from 0 to its target over ~1.3 s while the held source never changes.

  * source arm direction: CONSTANT (frozen frame)
  * torso: slewing through tens of degrees
  * therefore ANY change in the arm's world direction during the slew is parent contamination

ApplyBone(hips, ...) runs after the arms are written in the same Apply() call, so if the arm solver's
parent division were using a stale parent this is where it would show.
"""
import json
import os
import socket
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ubridge

DUMP = os.path.join(HERE, "wire_p0.jsonl")
OUT = r"C:\Unity\viitorx-vrm-avtar-unity-base-project\Assets\Games\viitorx-vrm-avtar-unity\python-sidecar~\arm_v3_evidence"
BODY = open(os.path.join(HERE, "dyn_light.cs.txt"), encoding="utf-8").read()


def hold(msg, stop, rate=40.0):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    seq = int((time.time() - 1788900000.0) * 100.0)
    while not stop.is_set():
        m = dict(msg)
        m["seq"] = seq
        m["t"] = round(time.time(), 4)
        sock.sendto(json.dumps(m).encode("utf-8"), ("127.0.0.1", 8899))
        seq += 1
        time.sleep(1.0 / rate)


RESET = '''
var boot = UnityEngine.Object.FindObjectsByType<VirtualMirror.App.AppBootstrap>(UnityEngine.FindObjectsSortMode.None)[0];
typeof(VirtualMirror.App.AppBootstrap).GetField("kalidokitTorsoYawScale",
    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).SetValue(boot, SCf);
var drv = typeof(VirtualMirror.App.AppBootstrap).GetField("kalidokitControlRig",
    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).GetValue(boot)
    as VirtualMirror.Retargeting.KalidokitControlRigDriver;
drv.SetTorsoYawScale(SCf);
drv.Recalibrate();          // zeroes the yaw conditioner so the torso must slew up from 0
return "reset at scale SC";
'''


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 359      # f419, the 50.8 deg turn
    sc = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    recs = {json.loads(l)["index"]: json.loads(l)["msg"] for l in open(DUMP, encoding="utf-8")}
    msg = recs[idx]

    stop = threading.Event()
    th = threading.Thread(target=hold, args=(msg, stop), daemon=True)
    th.start()
    try:
        time.sleep(4.0)                                        # settle the arms + gate at scale SC
        print(ubridge.execute_code(RESET.replace("SC", repr(sc))).get("data", {}).get("result"), flush=True)
        rows = []
        t0 = time.time()
        while time.time() - t0 < 8.0:                          # cover the whole slew and past it
            r = ubridge.execute_code(BODY)
            txt = (r.get("data") or {}).get("result")
            if txt:
                d = json.loads(txt)
                if d.get("ok"):
                    d["dt"] = round(time.time() - t0, 3)
                    rows.append(d)
    finally:
        stop.set()
        th.join(timeout=2)

    print(f"\nheld source frame index {idx} (f{60 + 0:d}-series), torsoYawScale {sc}, source FROZEN")
    print(f"{'t(s)':>6} {'avatarShYaw':>12} {'srcShYaw':>10} | {'armL_up':>8} {'armL_fo':>8} {'armR_up':>8} {'armR_fo':>8}")
    errs = []
    for d in rows:
        e = [d["left"]["errUpper"], d["left"]["errFore"], d["right"]["errUpper"], d["right"]["errFore"]]
        errs += [x for x in e if x is not None]
        print(f"{d['dt']:>6.2f} {d['shYawAvatar']:>12.3f} {d['shYawSrc']:>10.3f} | "
              f"{e[0]:>8.4f} {e[1]:>8.4f} {e[2]:>8.4f} {e[3]:>8.4f}")
    yaws = [d["shYawAvatar"] for d in rows]
    if yaws:
        print(f"\navatar shoulder yaw travelled {min(yaws):.2f} -> {max(yaws):.2f} deg "
              f"(span {max(yaws) - min(yaws):.2f}) while the source never changed")
    if errs:
        print(f"arm world-frame direction error over the whole slew: "
              f"mean {sum(errs) / len(errs):.5f}  max {max(errs):.5f} deg  (n={len(errs)})")
    with open(os.path.join(OUT, f"slew_idx{idx}_scale{sc:g}.json".replace(".", "p", 1)), "w", encoding="utf-8") as fh:
        json.dump({"index": idx, "scale": sc, "rows": rows}, fh, indent=2)


if __name__ == "__main__":
    main()
