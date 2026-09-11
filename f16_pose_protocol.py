#!/usr/bin/env python3
"""
F-16 pose protocol - fixed distance, scripted body headings, on-screen HUD.

Two jobs:

  1. SECTION 6 rotation set (0, +-30, +-45, +-60, +-90) at one distance, so measurement resolution
     can be compared across the working range.

  2. THE BIAS TEST. The sub-pixel distance sweep resolved a persistent shoulder depth difference of
     46-152 mm at true-square, which sub-pixel did NOT remove. That is either (a) the subject not
     actually being square, or (b) a systematic left/right depth bias in the sensor. Turning the
     subject 180 degrees separates them: a SUBJECT error flips sign with the body, a SENSOR error
     does not. Nothing else in the capture can tell these apart.

Self-terminating: every block is time-bounded and the seek phase has a timeout.
"""
import argparse
import io
import json
import os
import time

import numpy as np
import cv2
import depthai as dai

import rtmw3d_pose as R
import oak_depth as D
import f16_capture as CAP
import f16_configs as C
from f16_autosweep import hud, WIN, RED, AMBER, GREEN, FX640

# label, on-screen prompt, seconds, nominal heading (deg, + = right shoulder back), prep seconds
PROTOCOL = [
    ("sq_a",    "FACE CAMERA - SQUARE",    12, 0,    6),
    ("back",    "TURN AROUND - BACK TO ME", 12, 180,  8),
    ("sq_b",    "FACE CAMERA - SQUARE",    12, 0,    8),
    ("left30",  "TURN LEFT  30",           10, -30,  6),
    ("right30", "TURN RIGHT 30",           10, 30,   6),
    ("left45",  "TURN LEFT  45",           10, -45,  6),
    ("right45", "TURN RIGHT 45",           10, 45,   6),
    ("left60",  "TURN LEFT  60",           10, -60,  6),
    ("right60", "TURN RIGHT 60",           10, 60,   6),
    ("left90",  "TURN LEFT  90",           10, -90,  6),
    ("right90", "TURN RIGHT 90",           10, 90,   6),
    ("sq_c",    "FACE CAMERA - SQUARE",    12, 0,    8),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"..\..\..\SentisModel\rtmw3d-x.onnx")
    ap.add_argument("--config", default="sub3")
    ap.add_argument("--distance", type=float, default=1.33)
    ap.add_argument("--width", type=float, default=333.1)
    ap.add_argument("--tol", type=float, default=0.08)
    ap.add_argument("--only", default="", help="comma list of block labels to run")
    ap.add_argument("--seek-timeout", type=float, default=60.0)
    ap.add_argument("--name", default="")
    a = ap.parse_args()

    proto = PROTOCOL
    if a.only:
        want = {s.strip() for s in a.only.split(",") if s.strip()}
        proto = [p for p in PROTOCOL if p[0] in want]

    print("[f16] loading RTMW3D ...")
    model = R.RTMW3D(a.model)
    path = CAP.open_out(a.name or ("pose_%s_%s" % (a.config, time.strftime("%H%M%S"))))
    print("[f16] writing %s" % path)

    pipe, cfg, mw, mh, rw, rh = C.build(a.config)
    fx = FX640 * (rw / 640.0)
    K = a.width * fx
    tgt = a.distance
    counts = {}

    with io.open(path, "w", encoding="utf-8") as fh, dai.Device(pipe) as dev:
        q_rgb = dev.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_dep = dev.getOutputQueue("depth", maxSize=4, blocking=False)
        intr = D.read_rgb_intrinsics(dev, rw, rh)
        bbox = R.center_bbox(rw, rh)
        seq = [0]
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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
            return est, span, uv, conf, depth, frame

        def write(label, heading, uv, conf, depth, est):
            seq[0] += 1
            xyz, meas, qual, _dg = D.backproject(
                uv, depth, rw, rh, intr, k=CAP.KWIN, with_quality=True, legacy=False)
            sxd = depth.shape[1] / float(rw)
            syd = depth.shape[0] / float(rh)
            L, Rr = CAP.WB_L_SHOULDER, CAP.WB_R_SHOULDER
            uL, vL = float(uv[L, 0]), float(uv[L, 1])
            uR, vR = float(uv[Rr, 0]), float(uv[Rr, 1])
            p30L, nvL = CAP.pct(depth, uL * sxd, vL * syd, 30.0)
            p30R, nvR = CAP.pct(depth, uR * sxd, vR * syd, 30.0)
            hL, _ = CAP.pct(depth, float(uv[CAP.WB_L_HIP, 0]) * sxd,
                            float(uv[CAP.WB_L_HIP, 1]) * syd, 30.0)
            hR, _ = CAP.pct(depth, float(uv[CAP.WB_R_HIP, 0]) * sxd,
                            float(uv[CAP.WB_R_HIP, 1]) * syd, 30.0)
            hip_ok = meas[CAP.WB_L_HIP] and meas[CAP.WB_R_HIP]
            hipZ = float(0.5 * (xyz[CAP.WB_L_HIP, 2] + xyz[CAP.WB_R_HIP, 2])) if hip_ok else None
            aa = [float(xyz[L, 0]), float(xyz[L, 1]), float(xyz[L, 2])]
            bb = [float(xyz[Rr, 0]), float(xyz[Rr, 1]), float(xyz[Rr, 2])]
            yaw = CAP.line_yaw_deg(aa, bb) if (meas[L] and meas[Rr]) else None
            fh.write(json.dumps({
                "seq": seq[0], "t": round(time.time(), 4), "cfg": a.config, "block": label,
                "heading": heading, "target_m": tgt,
                "estRange_m": round(est, 3) if est else None,
                "uL": round(uL, 2), "vL": round(vL, 2), "uR": round(uR, 2), "vR": round(vR, 2),
                "cL": round(float(conf[L]), 3), "cR": round(float(conf[Rr]), 3),
                "uSpan": round(abs(uL - uR), 2),
                "zL": round(float(xyz[L, 2]) * 1000.0, 1),
                "zR": round(float(xyz[Rr, 2]) * 1000.0, 1),
                "qL": round(float(qual[L]), 3), "qR": round(float(qual[Rr]), 3),
                "mL": int(bool(meas[L])), "mR": int(bool(meas[Rr])),
                "p30L": p30L, "p30R": p30R, "nvL": nvL, "nvR": nvR,
                # hip-line depths: a left/right SENSOR bias must show on the hips too
                "hipL": hL, "hipR": hR,
                "wL": CAP.window_unique(depth, uL * sxd, vL * syd),
                "wR": CAP.window_unique(depth, uR * sxd, vR * syd),
                "hipZ": round(hipZ, 4) if hipZ is not None else None,
                "dx": round((bb[0] - aa[0]) * 1000.0, 2),
                "dz": round((bb[2] - aa[2]) * 1000.0, 2),
                "yaw3D": round(yaw, 3) if yaw is not None else None,
                "depthW": int(depth.shape[1]), "depthH": int(depth.shape[0]),
            }) + "\n")
            return yaw

        # ---------------------------------------------------------------- get them to the mark
        t_seek = time.time()
        while True:
            est, span, uv, conf, depth, frame = grab()
            if time.time() - t_seek > a.seek_timeout:
                print("[f16] seek timeout - proceeding anyway")
                break
            if est is None:
                hud(frame, tgt, None, "STEP INTO VIEW", RED, sub="no person detected")
                continue
            err = est - tgt
            if abs(err) <= a.tol:
                hud(frame, tgt, est, "GOOD - STAY THERE", GREEN, sub="starting ...")
                if time.time() - t_seek > 2.0:
                    break
            else:
                cue = ("MOVE BACK    %4.0f cm" % (-err * 100.0)) if err < 0 else \
                      ("MOVE FORWARD %4.0f cm" % (err * 100.0))
                col = AMBER if abs(err) <= 0.20 else RED
                if hud(frame, tgt, est, cue, col, span=span) == ord("q"):
                    return

        # ---------------------------------------------------------------- the protocol
        for label, prompt, secs, heading, prep in proto:
            t0 = time.time()
            while time.time() - t0 < prep:              # get-ready countdown
                est, span, uv, conf, depth, frame = grab()
                left = prep - (time.time() - t0)
                if hud(frame, tgt, est, prompt, AMBER,
                       sub="get ready ... %.0f" % max(1.0, left)) == ord("q"):
                    return
            CAP.beep(1200, 220)
            t0 = time.time()
            n = 0
            while time.time() - t0 < secs:
                est, span, uv, conf, depth, frame = grab()
                yaw = write(label, heading, uv, conf, depth, est)
                n += 1
                left = secs - (time.time() - t0)
                if hud(frame, tgt, est, "HOLD", GREEN,
                       sub="%s   %.0f s" % (prompt, max(0.0, left)),
                       bar=1.0 - left / secs) == ord("q"):
                    return
            CAP.beep(500, 120)
            CAP.beep(500, 120)
            counts[label] = n
            print("  %-9s %-26s %4d frames" % (label, prompt, n))

    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    except Exception:
        pass
    print("[f16] done: %s" % counts)
    print("[f16] %s" % path)


if __name__ == "__main__":
    main()
