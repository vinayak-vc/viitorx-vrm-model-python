#!/usr/bin/env python3
"""
F-17 raw multi-camera capture, HUD-guided (the F-16 closed-loop protocol, section 6).

Records SYNCHRONISED raw CAM_A / CAM_B / CAM_C frames plus the device's own sub-pixel depth at the
full-body working distances. Everything downstream - rectification, matcher tuning, both baselines,
the disparity-bias fit - then runs OFFLINE against these frames, so matcher choices cost no further
subject time and every baseline is compared on byte-identical input.

Frames are written as PNG (gray/BGR) and PNG-16 (depth) under
oak_v4_evidence/f17/raw/<tag>/<block>/, with a manifest.jsonl carrying the pose-derived range.

Self-terminating: per-target timeout plus a global wall-clock cap.
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
import f16_capture as CAP
import f17_stereo_rig as RIG
from f16_autosweep import hud, WIN, RED, AMBER, GREEN, FX640

OUTROOT = os.path.join("oak_v4_evidence", "f17", "raw")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"..\..\..\SentisModel\rtmw3d-x.onnx")
    ap.add_argument("--targets", default="1.24,1.33,1.50,1.80,2.00")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--rate", type=float, default=8.0, help="frames/s written to disk")
    ap.add_argument("--tol", type=float, default=0.07)
    ap.add_argument("--hold", type=float, default=1.0)
    ap.add_argument("--width", type=float, default=333.1)
    ap.add_argument("--target-timeout", type=float, default=45.0)
    ap.add_argument("--global-timeout", type=float, default=520.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    tag = a.tag or time.strftime("%H%M%S")
    root = os.path.join(OUTROOT, tag)
    n = 0
    while os.path.exists(root):                      # section 18: overwrite protection
        n += 1
        root = os.path.join(OUTROOT, "%s_%d" % (tag, n))
    os.makedirs(root)
    print("[f17] writing %s" % root)

    print("[f17] loading RTMW3D ...")
    model = R.RTMW3D(a.model)

    targets = [float(t) for t in a.targets.split(",") if t.strip()]
    pipe = RIG.build_pipeline(subpixel=True, bits=3)
    K = a.width * FX640
    manifest = io.open(os.path.join(root, "manifest.jsonl"), "w", encoding="utf-8")
    counts = {}
    t_global = time.time()

    with dai.Device(pipe) as dev:
        calib = dev.readCalibration()
        io.open(os.path.join(root, "calib.json"), "w", encoding="utf-8").write(json.dumps({
            "baseline_cm": calib.getBaselineDistance(),
            "intrinsics": {k: calib.getCameraIntrinsics(s, RIG.W, RIG.H)
                           for k, s in RIG.SOCK.items()},
            "distortion": {k: list(calib.getDistortionCoefficients(s))
                           for k, s in RIG.SOCK.items()},
            "extrinsics": {"%s->%s" % (x, y):
                           calib.getCameraExtrinsics(RIG.SOCK[x], RIG.SOCK[y], False)
                           for x, y in (("A", "B"), ("A", "C"), ("B", "C"))},
        }, indent=1))

        q = {nm: dev.getOutputQueue(nm, maxSize=4, blocking=False)
             for nm in ("rgb", "monoB", "monoC", "depth")}
        bbox = R.center_bbox(RIG.W, RIG.H)
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        def grab():
            nonlocal bbox
            rgb = q["rgb"].get().getCvFrame()
            gb = q["monoB"].get().getCvFrame()
            gc = q["monoC"].get().getCvFrame()
            dd = q["depth"].get().getFrame()
            uv, zrel, conf = model.infer(rgb, bbox)
            refined = R.bbox_from_keypoints(uv, conf, RIG.W, RIG.H, thr=0.3)
            if refined is not None and float(np.mean(conf[0:17])) >= 0.3:
                bbox = tuple(0.7 * np.array(bbox) + 0.3 * np.array(refined))
            else:
                bbox = R.center_bbox(RIG.W, RIG.H)
            L, Rr = CAP.WB_L_SHOULDER, CAP.WB_R_SHOULDER
            span = abs(float(uv[L, 0]) - float(uv[Rr, 0]))
            ok = min(float(conf[L]), float(conf[Rr])) >= 0.4 and span > 8
            est = (K / span / 1000.0) if ok else None
            return est, span, rgb, gb, gc, dd, uv, conf

        for tgt in targets:
            label = "d%03d" % int(round(tgt * 100))
            bdir = os.path.join(root, label)
            os.makedirs(bdir, exist_ok=True)
            t_tgt = time.time()
            stable = None
            while True:
                if time.time() - t_global > a.global_timeout:
                    print("\n[f17] GLOBAL TIMEOUT")
                    counts[label] = "global-timeout"
                    manifest.close()
                    return
                if time.time() - t_tgt > a.target_timeout:
                    print("  !!! %s SKIPPED" % label)
                    counts[label] = "skipped"
                    break
                est, span, rgb, gb, gc, dd, uv, conf = grab()
                if est is None:
                    stable = None
                    hud(rgb, tgt, None, "STEP INTO VIEW", RED, sub="no person detected")
                    continue
                err = est - tgt
                if abs(err) <= a.tol:
                    if stable is None:
                        stable = time.time()
                    elif time.time() - stable >= a.hold:
                        break
                    cue = "HOLD  %.1f" % (a.hold - (time.time() - stable))
                    col = GREEN
                else:
                    stable = None
                    cue = ("MOVE BACK    %4.0f cm" % (-err * 100.0)) if err < 0 else \
                          ("MOVE FORWARD %4.0f cm" % (err * 100.0))
                    col = AMBER if abs(err) <= 0.20 else RED
                if hud(rgb, tgt, est, cue, col, span=span) == ord("q"):
                    manifest.close()
                    return
            if counts.get(label) == "skipped":
                continue

            CAP.beep(1200, 220)
            t0 = time.time()
            written = 0
            next_write = 0.0
            while time.time() - t0 < a.seconds:
                est, span, rgb, gb, gc, dd, uv, conf = grab()
                now = time.time() - t0
                if now >= next_write:
                    next_write += 1.0 / a.rate
                    i = written
                    cv2.imwrite(os.path.join(bdir, "a_%04d.png" % i), rgb)
                    cv2.imwrite(os.path.join(bdir, "b_%04d.png" % i), gb)
                    cv2.imwrite(os.path.join(bdir, "c_%04d.png" % i), gc)
                    cv2.imwrite(os.path.join(bdir, "d_%04d.png" % i), dd.astype(np.uint16))
                    manifest.write(json.dumps({
                        "block": label, "target_m": tgt, "i": i,
                        "t": round(time.time(), 4),
                        "estRange_m": round(est, 3) if est else None,
                        "uSpan": round(span, 2),
                        "uv_sh": [[round(float(uv[CAP.WB_L_SHOULDER, 0]), 2),
                                   round(float(uv[CAP.WB_L_SHOULDER, 1]), 2)],
                                  [round(float(uv[CAP.WB_R_SHOULDER, 0]), 2),
                                   round(float(uv[CAP.WB_R_SHOULDER, 1]), 2)]],
                        "conf_sh": [round(float(conf[CAP.WB_L_SHOULDER]), 3),
                                    round(float(conf[CAP.WB_R_SHOULDER]), 3)],
                    }) + "\n")
                    written += 1
                left = a.seconds - now
                if hud(rgb, tgt, est, "HOLD STILL", GREEN,
                       sub="recording   %.0f s left" % max(0.0, left),
                       bar=1.0 - left / a.seconds) == ord("q"):
                    manifest.close()
                    return
            CAP.beep(500, 120)
            CAP.beep(500, 120)
            counts[label] = written
            print("  %-6s %3d frame sets" % (label, written))

    manifest.close()
    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    except Exception:
        pass
    print("[f17] done: %s" % counts)
    print("[f17] %s" % root)


if __name__ == "__main__":
    main()
