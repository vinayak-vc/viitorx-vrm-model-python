#!/usr/bin/env python3
"""
F-16 self-driving distance sweep.

The first attempt failed because the subject cannot judge 0.80 m from 1.33 m and cannot read a
console from across the room. This version closes the loop: it estimates the subject's range from
the SHOULDER PIXEL SPAN (which does not depend on the stereo depth under test), tells them which
way to move, and starts recording only once they are actually inside the tolerance band and steady.

    range_est = (W * fx) / uSpan        W = this subject's shoulder-keypoint separation (mm)

uSpan is a pure RGB/pose quantity, so using it to POSITION the subject keeps the depth measurement
under test independent of the positioning signal. The recorded blocks still carry every raw depth
value, and the analysis reports measured range per block regardless.

Self-terminating: per-target timeout, plus a global wall-clock cap.
"""
import argparse
import io
import json
import math
import os
import time

import numpy as np
import cv2
import depthai as dai

import rtmw3d_pose as R
import oak_depth as D
import f16_capture as CAP
import f16_configs as C

FX640 = 284.6272          # CAM_A @ 640x400 (device EEPROM)

WIN = "F-16 CAPTURE"
HUD_W, HUD_H = 1280, 800
RED = (60, 60, 235)
AMBER = (40, 180, 245)
GREEN = (80, 220, 90)
WHITE = (245, 245, 245)
GREY = (140, 140, 140)


def hud(frame, target_m, you_m, cue, colour, sub="", bar=None, span=None):
    """Full-screen operator display. The subject is metres from the machine and cannot read a
    console, so every instruction has to be legible from across the room."""
    img = cv2.resize(frame, (HUD_W, HUD_H), interpolation=cv2.INTER_LINEAR)
    img = (img * 0.35).astype(np.uint8)                       # dim the video behind the text

    cv2.rectangle(img, (0, 0), (HUD_W, 90), (18, 18, 18), -1)
    cv2.putText(img, "TARGET  %.2f m" % target_m if target_m else "TARGET  --",
                (40, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.6, WHITE, 3, cv2.LINE_AA)
    if you_m:
        cv2.putText(img, "YOU  %.2f m" % you_m, (HUD_W - 430, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, colour, 3, cv2.LINE_AA)

    # the instruction, as large as it will go
    scale = 3.4 if len(cue) <= 16 else 2.4
    (tw, th), _ = cv2.getTextSize(cue, cv2.FONT_HERSHEY_SIMPLEX, scale, 8)
    cv2.putText(img, cue, ((HUD_W - tw) // 2, HUD_H // 2 + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 8, cv2.LINE_AA)

    if sub:
        (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
        cv2.putText(img, sub, ((HUD_W - sw) // 2, HUD_H // 2 + 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, WHITE, 3, cv2.LINE_AA)

    # position ruler: where you are relative to the target, +-0.5 m across the screen
    if target_m and you_m:
        y = HUD_H - 150
        cv2.line(img, (140, y), (HUD_W - 140, y), GREY, 3)
        cx = HUD_W // 2
        cv2.line(img, (cx, y - 38), (cx, y + 38), GREEN, 5)
        px = int(cx + np.clip((you_m - target_m) / 0.5, -1, 1) * (HUD_W // 2 - 150))
        cv2.circle(img, (px, y), 26, colour, -1)
        cv2.putText(img, "TOO CLOSE", (150, y + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREY, 2)
        cv2.putText(img, "TOO FAR", (HUD_W - 300, y + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREY, 2)

    if bar is not None:
        w = int((HUD_W - 160) * max(0.0, min(1.0, bar)))
        cv2.rectangle(img, (80, HUD_H - 70), (HUD_W - 80, HUD_H - 30), (60, 60, 60), -1)
        cv2.rectangle(img, (80, HUD_H - 70), (80 + w, HUD_H - 30), GREEN, -1)

    if span:
        cv2.putText(img, "span %.0f px" % span, (40, HUD_H - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREY, 2)
    cv2.imshow(WIN, img)
    return cv2.waitKey(1) & 0xFF


def estimate_W(paths, fx640=FX640):
    """Calibrate the subject's shoulder-keypoint separation from an earlier capture."""
    vals = []
    for p in paths:
        if not os.path.exists(p):
            continue
        for line in io.open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not (r.get("mL") and r.get("mR")):
                continue
            if min(r.get("cL", 0), r.get("cR", 0)) < 0.5:
                continue
            z = r.get("zL"), r.get("zR")
            if not all(z) or r.get("uSpan", 0) < 10:
                continue
            vals.append(r["uSpan"] * (0.5 * (z[0] + z[1])) / fx640)
    return float(np.median(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"..\..\..\SentisModel\rtmw3d-x.onnx")
    ap.add_argument("--config", default="baseline")
    ap.add_argument("--targets", default="0.80,1.00,1.20,1.33,1.50,1.80,2.00")
    ap.add_argument("--seconds", type=float, default=12.0, help="record time per target")
    ap.add_argument("--tol", type=float, default=0.06, help="+-m tolerance to accept a position")
    ap.add_argument("--hold", type=float, default=1.2, help="s inside tolerance before recording")
    ap.add_argument("--width", type=float, default=0.0, help="subject shoulder separation mm (0=auto)")
    ap.add_argument("--calib", default="oak_v4_evidence/f16/dist_baseline.jsonl")
    ap.add_argument("--target-timeout", type=float, default=75.0)
    ap.add_argument("--global-timeout", type=float, default=900.0)
    ap.add_argument("--name", default="")
    a = ap.parse_args()

    W = a.width or estimate_W([a.calib]) or 331.0
    print("[f16] subject shoulder separation W = %.1f mm  (%s)"
          % (W, "given" if a.width else "calibrated from %s" % os.path.basename(a.calib)))

    targets = [float(t) for t in a.targets.split(",") if t.strip()]
    print("[f16] loading RTMW3D ...")
    model = R.RTMW3D(a.model)
    path = CAP.open_out(a.name or ("autosweep_%s_%s" % (a.config, time.strftime("%H%M%S"))))
    print("[f16] writing %s" % path)
    print("[f16] %d targets. Stand SQUARE. The prompt tells you which way to move;")
    print("[f16] recording starts on its own (high beep) and ends with two low beeps.\n")

    pipe, cfg, mw, mh, rw, rh = C.build(a.config)
    fx = FX640 * (rw / 640.0)
    K = W * fx                                   # range_mm = K / uSpan
    done = {}
    t_global = time.time()

    with io.open(path, "w", encoding="utf-8") as fh, dai.Device(pipe) as dev:
        q_rgb = dev.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_dep = dev.getOutputQueue("depth", maxSize=4, blocking=False)
        intr = D.read_rgb_intrinsics(dev, rw, rh)
        bbox = R.center_bbox(rw, rh)
        seq = 0
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        def grab():
            """One synchronised RGB+depth frame -> (est_range_m, uSpan, payload-or-None)."""
            nonlocal bbox, seq
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
            return est, span, (uv, conf, depth, lat, frame)

        def record(label, secs, tgt):
            nonlocal seq
            CAP.beep(1200, 220)
            print("\r  >>> RECORDING %-10s hold still ...%s" % (label, " " * 40))
            t0 = time.time()
            n = 0
            while time.time() - t0 < secs:
                est, span, payload = grab()
                uv, conf, depth, lat, frame = payload
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
                p50L, _ = CAP.pct(depth, uL * sxd, vL * syd, 50.0)
                p50R, _ = CAP.pct(depth, uR * sxd, vR * syd, 50.0)
                hip_ok = meas[CAP.WB_L_HIP] and meas[CAP.WB_R_HIP]
                hipZ = float(0.5 * (xyz[CAP.WB_L_HIP, 2] + xyz[CAP.WB_R_HIP, 2])) if hip_ok else None
                aa = [float(xyz[L, 0]), float(xyz[L, 1]), float(xyz[L, 2])]
                bb = [float(xyz[Rr, 0]), float(xyz[Rr, 1]), float(xyz[Rr, 2])]
                yaw = CAP.line_yaw_deg(aa, bb) if (meas[L] and meas[Rr]) else None
                fh.write(json.dumps({
                    "seq": seq, "t": round(time.time(), 4), "cfg": a.config, "block": label,
                    "target_m": tgt, "estRange_m": round(est, 3) if est else None,
                    "uL": round(uL, 2), "vL": round(vL, 2), "uR": round(uR, 2), "vR": round(vR, 2),
                    "cL": round(float(conf[L]), 3), "cR": round(float(conf[Rr]), 3),
                    "uSpan": round(abs(uL - uR), 2),
                    "zL": round(float(xyz[L, 2]) * 1000.0, 1),
                    "zR": round(float(xyz[Rr, 2]) * 1000.0, 1),
                    "qL": round(float(qual[L]), 3), "qR": round(float(qual[Rr]), 3),
                    "mL": int(bool(meas[L])), "mR": int(bool(meas[Rr])),
                    "p30L": p30L, "p30R": p30R, "p50L": p50L, "p50R": p50R,
                    "nvL": nvL, "nvR": nvR,
                    "wL": CAP.window_unique(depth, uL * sxd, vL * syd),
                    "wR": CAP.window_unique(depth, uR * sxd, vR * syd),
                    "hipZ": round(hipZ, 4) if hipZ is not None else None,
                    "dx": round((bb[0] - aa[0]) * 1000.0, 2),
                    "dz": round((bb[2] - aa[2]) * 1000.0, 2),
                    "yaw3D": round(yaw, 3) if yaw is not None else None,
                    "latMs": round(lat, 1),
                    "depthW": int(depth.shape[1]), "depthH": int(depth.shape[0]),
                }) + "\n")
                left = secs - (time.time() - t0)
                hud(frame, tgt, est, "HOLD STILL", GREEN,
                    sub="recording   %.0f s left" % max(0.0, left),
                    bar=1.0 - left / secs)
            CAP.beep(500, 120)
            CAP.beep(500, 120)
            print("\r  >>> %-10s DONE  %d frames%s" % (label, n, " " * 40))
            return n

        for tgt in targets:
            label = "d%03d" % int(round(tgt * 100))
            t_tgt = time.time()
            stable_since = None
            last_print = 0.0
            while True:
                if time.time() - t_global > a.global_timeout:
                    print("\n[f16] GLOBAL TIMEOUT - stopping.")
                    done[label] = "global-timeout"
                    return
                if time.time() - t_tgt > a.target_timeout:
                    print("\r  !!! %s SKIPPED (could not settle in %.0f s)%s"
                          % (label, a.target_timeout, " " * 30))
                    done[label] = "skipped"
                    break
                est, span, _payload = grab()
                if est is None:
                    stable_since = None
                    hud(_payload[4], tgt, None, "STEP INTO VIEW", RED, sub="no person detected")
                    continue
                err = est - tgt
                if abs(err) <= a.tol:
                    if stable_since is None:
                        stable_since = time.time()
                    elif time.time() - stable_since >= a.hold:
                        done[label] = record(label, a.seconds, tgt)
                        break
                    cue = "HOLD  %.1f" % (a.hold - (time.time() - stable_since))
                else:
                    stable_since = None
                    cue = ("MOVE BACK    %4.0f cm" % (-err * 100.0)) if err < 0 else \
                          ("MOVE FORWARD %4.0f cm" % (err * 100.0))
                col = GREEN if abs(err) <= a.tol else (AMBER if abs(err) <= 0.20 else RED)
                if hud(_payload[4], tgt, est, cue, col, span=span) == ord("q"):
                    print("\n[f16] aborted by operator")
                    return

    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    except Exception:
        pass
    print("\n[f16] done: %s" % done)
    print("[f16] %s" % path)


if __name__ == "__main__":
    main()
