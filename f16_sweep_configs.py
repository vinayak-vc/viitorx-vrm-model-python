#!/usr/bin/env python3
"""
F-16 sections 8/9/10 - stereo CONFIGURATION sweep at one fixed distance, with the on-screen HUD.

Each configuration needs its own device boot, so the camera is reopened per config while the
RTMW3D session stays loaded. The subject stands SQUARE at the target distance throughout; the HUD
guides them there once and then tells them to hold through each configuration.

Usage:
  python f16_sweep_configs.py --distance 1.33 --seconds 12 --configs baseline,sub3,...
"""
import argparse
import io
import json
import time

import numpy as np
import cv2
import depthai as dai

import rtmw3d_pose as R
import oak_depth as D
import f16_capture as CAP
import f16_configs as C
from f16_autosweep import hud, WIN, RED, AMBER, GREEN, FX640

DEFAULT = "baseline,sub3,sub5,mono800,mono800_sub3,rgb800,sub3_rgb800,best"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"..\..\..\SentisModel\rtmw3d-x.onnx")
    ap.add_argument("--configs", default=DEFAULT)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--distance", type=float, default=1.33)
    ap.add_argument("--width", type=float, default=333.1)
    ap.add_argument("--tol", type=float, default=0.08)
    ap.add_argument("--prep", type=float, default=5.0)
    ap.add_argument("--seek-timeout", type=float, default=45.0)
    ap.add_argument("--name", default="")
    a = ap.parse_args()

    names = [n.strip() for n in a.configs.split(",") if n.strip()]
    for n in names:
        if n not in C.CONFIGS:
            raise SystemExit("unknown config: %s" % n)

    print("[f16] loading RTMW3D ...")
    model = R.RTMW3D(a.model)
    path = CAP.open_out(a.name or ("cfgsweep_%03d_%s"
                                   % (int(a.distance * 100), time.strftime("%H%M%S"))))
    print("[f16] writing %s" % path)
    counts = {}
    tgt = a.distance

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    with io.open(path, "w", encoding="utf-8") as fh:
        for ci, name in enumerate(names):
            try:
                pipe, cfg, mw, mh, rw, rh = C.build(name)
            except Exception as e:
                print("  %-18s BUILD FAILED: %s" % (name, e))
                counts[name] = "build-failed"
                continue
            fx = FX640 * (rw / 640.0)
            K = a.width * fx
            try:
                with dai.Device(pipe) as dev:
                    q_rgb = dev.getOutputQueue("rgb", maxSize=4, blocking=False)
                    q_dep = dev.getOutputQueue("depth", maxSize=4, blocking=False)
                    intr = D.read_rgb_intrinsics(dev, rw, rh)
                    bbox = R.center_bbox(rw, rh)
                    seq = 0

                    def grab():
                        nonlocal bbox
                        rgb_pkt = q_rgb.get()
                        dep_pkt = q_dep.get()
                        frame = rgb_pkt.getCvFrame()
                        depth = dep_pkt.getFrame()
                        uv, zrel, conf = model.infer(frame, bbox)
                        refined = R.bbox_from_keypoints(uv, conf, rw, rh, thr=0.3)
                        if refined is not None and float(np.mean(conf[0:17])) >= 0.3:
                            bbox = tuple(0.7 * np.array(bbox) + 0.3 * np.array(refined))
                        else:
                            bbox = R.center_bbox(rw, rh)
                        L, Rr = CAP.WB_L_SHOULDER, CAP.WB_R_SHOULDER
                        span = abs(float(uv[L, 0]) - float(uv[Rr, 0]))
                        ok = min(float(conf[L]), float(conf[Rr])) >= 0.4 and span > 8
                        est = (K / span / 1000.0) if ok else None
                        try:
                            lat = (dai.Clock.now() - rgb_pkt.getTimestamp()).total_seconds() * 1000.0
                        except Exception:
                            lat = -1.0
                        return est, span, uv, conf, depth, frame, lat

                    # ---- position / prep -------------------------------------------------------
                    t_seek = time.time()
                    prompt = "STAND SQUARE  (%d/%d)" % (ci + 1, len(names))
                    while time.time() - t_seek < (a.seek_timeout if ci == 0 else a.prep):
                        est, span, uv, conf, depth, frame, lat = grab()
                        if est is None:
                            hud(frame, tgt, None, "STEP INTO VIEW", RED, sub="no person detected")
                            continue
                        err = est - tgt
                        if abs(err) <= a.tol:
                            hud(frame, tgt, est, prompt, GREEN,
                                sub="%s  -  starting" % name)
                            if ci > 0 or time.time() - t_seek > 2.0:
                                break
                        else:
                            cue = ("MOVE BACK    %4.0f cm" % (-err * 100.0)) if err < 0 else \
                                  ("MOVE FORWARD %4.0f cm" % (err * 100.0))
                            if hud(frame, tgt, est, cue,
                                   AMBER if abs(err) <= 0.20 else RED, span=span) == ord("q"):
                                return

                    # ---- record ----------------------------------------------------------------
                    CAP.beep(1200, 220)
                    t0 = time.time()
                    n = 0
                    while time.time() - t0 < a.seconds:
                        est, span, uv, conf, depth, frame, lat = grab()
                        seq += 1
                        n += 1
                        xyz, meas, qual, _dg = D.backproject(
                            uv, depth, rw, rh, intr, k=CAP.KWIN, with_quality=True, legacy=False)
                        sxd = depth.shape[1] / float(rw)
                        syd = depth.shape[0] / float(rh)
                        L, Rr = CAP.WB_L_SHOULDER, CAP.WB_R_SHOULDER
                        uL, vL = float(uv[L, 0]), float(uv[L, 1])
                        uR, vR = float(uv[Rr, 0]), float(uv[Rr, 1])
                        p30L, nvL = CAP.pct(depth, uL * sxd, vL * syd, 30.0)
                        p30R, nvR = CAP.pct(depth, uR * sxd, vR * syd, 30.0)
                        hip_ok = meas[CAP.WB_L_HIP] and meas[CAP.WB_R_HIP]
                        hipZ = (float(0.5 * (xyz[CAP.WB_L_HIP, 2] + xyz[CAP.WB_R_HIP, 2]))
                                if hip_ok else None)
                        aa = [float(xyz[L, 0]), float(xyz[L, 1]), float(xyz[L, 2])]
                        bb = [float(xyz[Rr, 0]), float(xyz[Rr, 1]), float(xyz[Rr, 2])]
                        yaw = CAP.line_yaw_deg(aa, bb) if (meas[L] and meas[Rr]) else None
                        fh.write(json.dumps({
                            "seq": seq, "t": round(time.time(), 4), "cfg": name,
                            "block": "d%03d" % int(round(tgt * 100)),
                            "target_m": tgt, "estRange_m": round(est, 3) if est else None,
                            "uL": round(uL, 2), "vL": round(vL, 2),
                            "uR": round(uR, 2), "vR": round(vR, 2),
                            "cL": round(float(conf[L]), 3), "cR": round(float(conf[Rr]), 3),
                            "uSpan": round(abs(uL - uR), 2),
                            "zL": round(float(xyz[L, 2]) * 1000.0, 1),
                            "zR": round(float(xyz[Rr, 2]) * 1000.0, 1),
                            "qL": round(float(qual[L]), 3), "qR": round(float(qual[Rr]), 3),
                            "mL": int(bool(meas[L])), "mR": int(bool(meas[Rr])),
                            "p30L": p30L, "p30R": p30R, "nvL": nvL, "nvR": nvR,
                            "wL": CAP.window_unique(depth, uL * sxd, vL * syd),
                            "wR": CAP.window_unique(depth, uR * sxd, vR * syd),
                            "hipZ": round(hipZ, 4) if hipZ is not None else None,
                            "dx": round((bb[0] - aa[0]) * 1000.0, 2),
                            "dz": round((bb[2] - aa[2]) * 1000.0, 2),
                            "yaw3D": round(yaw, 3) if yaw is not None else None,
                            "latMs": round(lat, 1),
                            "depthW": int(depth.shape[1]), "depthH": int(depth.shape[0]),
                        }) + "\n")
                        left = a.seconds - (time.time() - t0)
                        if hud(frame, tgt, est, "HOLD STILL", GREEN,
                               sub="%s   %.0f s   (%d/%d)" % (name, max(0.0, left),
                                                              ci + 1, len(names)),
                               bar=1.0 - left / a.seconds) == ord("q"):
                            return
                    CAP.beep(500, 120)
                    CAP.beep(500, 120)
                    counts[name] = n
                    print("  %-18s %4d frames   %s" % (name, n, C.CONFIGS[name]["note"]))
            except Exception as e:
                print("  %-18s RUN FAILED: %s" % (name, str(e)[:90]))
                counts[name] = "run-failed"

    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    except Exception:
        pass
    print("[f16] done: %s" % counts)
    print("[f16] %s" % path)


if __name__ == "__main__":
    main()
