#!/usr/bin/env python3
"""TORSO YAW V4 §4 — arm regression while the torso is ACTIVELY ROTATING.

The four cases the brief asks for, built from the recorded wire so every case is reproducible:

  1 torso stationary + arm movement  : frames whose torso yaw is ~constant, arms changing
  2 torso rotating  + arms stationary: a synthetic pair — one recorded frame's arms held while the
                                       torso landmarks are yawed progressively about the vertical
  3 torso rotating  + arm movement   : the recorded clip replayed at its own 30 fps
  4 fast torso      + fast arms      : the same clip replayed at 4x

For 1/3/4 the wire is replayed and the avatar sampled as fast as the editor round-trip allows. For 2
the torso is rotated synthetically so torso motion is isolated from arm motion, which the dance clip
never does on its own.

Reported KPI: world-frame source->skinned-bone arm direction error. If the arm solver's parent
division is correct this stays at the static floor even while the torso slews; if it inherits a stale
parent it will spike in proportion to torso angular velocity.
"""
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ubridge

DUMP = os.path.join(HERE, "wire_p0.jsonl")
OUT = r"C:\Unity\viitorx-vrm-avtar-unity-base-project\Assets\Games\viitorx-vrm-avtar-unity\python-sidecar~\arm_v3_evidence"
PORT = 8899
BODY = open(os.path.join(HERE, "dyn_light.cs.txt"), encoding="utf-8").read()


def set_scale(sc):
    code = f'''
var boot = UnityEngine.Object.FindObjectsByType<VirtualMirror.App.AppBootstrap>(UnityEngine.FindObjectsSortMode.None)[0];
typeof(VirtualMirror.App.AppBootstrap).GetField("kalidokitTorsoYawScale",
    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).SetValue(boot, {sc!r}f);
var drv = typeof(VirtualMirror.App.AppBootstrap).GetField("kalidokitControlRig",
    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).GetValue(boot)
    as VirtualMirror.Retargeting.KalidokitControlRigDriver;
drv.SetTorsoYawScale({sc!r}f); drv.Recalibrate();
return "scale {sc}";
'''
    return ubridge.execute_code(code)


def load():
    return [json.loads(l) for l in open(DUMP, encoding="utf-8")]


def yaw_landmarks(lm, deg):
    """Rotate the TORSO landmarks about the vertical, leaving each arm's shape relative to its own
    shoulder untouched, so torso rotation is isolated from arm motion."""
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    out = [list(p) for p in lm]
    # rotate every filled landmark about the hip-centre vertical: the whole body turns rigidly, so the
    # arms keep their body-relative pose and only the torso frame changes.
    for i, p in enumerate(out):
        if p[3] <= 0.0:
            continue
        x, y, z = p[0], p[1], p[2]
        out[i] = [x * ca + z * sa, y, -x * sa + z * ca, p[3]]
    return out


def streamer(packets, rate, stop, host="127.0.0.1", port=PORT):
    """Send `packets` (list of msg dicts) in a loop at `rate` Hz until stop is set, re-stamping seq/t."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    seq = int((time.time() - 1788900000.0) * 100.0)
    i = 0
    period = 1.0 / rate
    nxt = time.time()
    while not stop.is_set():
        m = dict(packets[i % len(packets)])
        m["seq"] = seq
        m["t"] = round(time.time(), 4)
        sock.sendto(json.dumps(m).encode("utf-8"), (host, port))
        seq += 1
        i += 1
        nxt += period
        d = nxt - time.time()
        if d > 0:
            time.sleep(d)
        else:
            nxt = time.time()


def sample(packets, rate, seconds, label):
    stop = threading.Event()
    th = threading.Thread(target=streamer, args=(packets, rate, stop), daemon=True)
    th.start()
    time.sleep(2.0)                      # let the gate validate and the slerp reach the moving pose
    rows = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            r = ubridge.execute_code(BODY)
        except Exception as e:
            print("   sample failed:", e, flush=True)
            break
        txt = (r.get("data") or {}).get("result")
        if txt:
            try:
                rows.append(json.loads(txt))
            except Exception:
                pass
    stop.set()
    th.join(timeout=2)
    good = [r for r in rows if r.get("ok")]
    errs = []
    yaws = []
    for r in good:
        for s in ("left", "right"):
            for k in ("errUpper", "errFore"):
                v = r[s].get(k)
                if v is not None:
                    errs.append(v)
        yaws.append(r["shYawAvatar"])
    dy = [abs(yaws[i] - yaws[i - 1]) for i in range(1, len(yaws))]
    print(f"   {label}: n={len(good)} samples | arm world err mean={_m(errs):.4f} "
          f"p95={_p(errs,95):.4f} max={max(errs) if errs else float('nan'):.4f} deg | "
          f"avatar yaw span={(max(yaws)-min(yaws)) if yaws else 0:.2f} deg, "
          f"max step between samples={max(dy) if dy else 0:.2f} deg", flush=True)
    return {"label": label, "n": len(good), "armErrMean": _m(errs), "armErrP95": _p(errs, 95),
            "armErrMax": max(errs) if errs else None,
            "avatarYawSpan": (max(yaws) - min(yaws)) if yaws else 0.0,
            "samples": good}


def _m(x):
    return sum(x) / len(x) if x else float("nan")


def _p(x, q):
    if not x:
        return float("nan")
    y = sorted(x)
    return y[min(len(y) - 1, int(len(y) * q / 100.0))]


def main():
    sc = float(sys.argv[1]) if len(sys.argv) > 1 else 0.75
    recs = load()
    msgs = [r["msg"] for r in recs]
    print(set_scale(sc).get("data", {}).get("result"), flush=True)
    results = []

    # CASE 1 — torso stationary, arms moving: the clip's first ~2 s sit near frontal (source yaw < the
    # 8 deg dead zone) while the arms swing, so the torso term contributes nothing by construction.
    results.append(sample(msgs[0:60], 30.0, 12.0, "case1_torsoStatic_armsMoving"))

    # CASE 2 — torso rotating, arms stationary: one frame's body yawed rigidly through +/-60 deg.
    base = msgs[24]["lm"]
    sweep = []
    for d in list(range(-60, 61, 3)) + list(range(57, -61, -3)):
        m = dict(msgs[24])
        m["lm"] = yaw_landmarks(base, d)
        sweep.append(m)
    results.append(sample(sweep, 30.0, 14.0, "case2_torsoRotating_armsStatic"))

    # CASE 3 — both moving, the clip at its own rate
    results.append(sample(msgs, 30.0, 14.0, "case3_torsoRotating_armsMoving"))

    # CASE 4 — both moving fast: same clip at 4x
    results.append(sample(msgs, 120.0, 14.0, "case4_fastTorso_fastArms"))

    with open(os.path.join(OUT, f"dyn_arm_scale{sc:g}.json".replace(".", "p", 1)), "w", encoding="utf-8") as fh:
        json.dump({"scale": sc, "cases": results}, fh, indent=2)
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
