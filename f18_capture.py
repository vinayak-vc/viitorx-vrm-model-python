#!/usr/bin/env python3
"""
F-18 portrait capture - framing envelope, torso measurement and interaction envelope.

Diagnostic only. Reuses production's pose model (rtmw3d_pose) and production's F-08 depth sampler
(oak_depth.backproject) unchanged; the ONLY difference from F-16's harness is that the RGB and the
depth are rotated together, with the intrinsics rotated to match (see f18_portrait.py).

Protocols:
  frame   distance sweep x arm poses  -> the framing/interaction envelope (brief sections 4,5,12)
  torso   rotation set at one distance -> torso measurement + sign validation (sections 7,8,11)

Per-frame JSONL carries the full COCO-17 keypoint set so body extents, margins and which part
leaves frame can all be computed offline without another capture.
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
import f18_portrait as PT

OUTDIR = os.path.join("oak_v4_evidence", "f18")
WIN = "F-18 PORTRAIT CAPTURE"
RED, AMBER, GREEN, WHITE, GREY = (60, 60, 235), (40, 180, 245), (80, 220, 90), (245, 245, 245), (140, 140, 140)

# COCO-17
NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4
LSH, RSH, LEL, REL, LWR, RWR = 5, 6, 7, 8, 9, 10
LHIP, RHIP, LKNEE, RKNEE, LANK, RANK = 11, 12, 13, 14, 15, 16
HEAD = (NOSE, LEYE, REYE, LEAR, REAR)
FEET = (LANK, RANK)
HANDS = (LWR, RWR)

FRAME_POSES = [
    ("relax",  "STAND STILL - ARMS DOWN",      8),
    ("arms45", "ARMS OUT AT 45 DEGREES",       8),
    ("tpose",  "ARMS STRAIGHT OUT - T POSE",   8),
    ("over",   "ARMS STRAIGHT UP",             8),
]
TORSO_POSES = [
    ("sq_a",    "FACE CAMERA - SQUARE",  10, 0),
    ("left30",  "TURN LEFT  30",          8, -30),
    ("right30", "TURN RIGHT 30",          8, 30),
    ("left45",  "TURN LEFT  45",          8, -45),
    ("right45", "TURN RIGHT 45",          8, 45),
    ("left60",  "TURN LEFT  60",          8, -60),
    ("right60", "TURN RIGHT 60",          8, 60),
    ("left90",  "TURN LEFT  90",          8, -90),
    ("right90", "TURN RIGHT 90",          8, 90),
    ("sq_b",    "FACE CAMERA - SQUARE",  10, 0),
]
MOVE_POSES = [
    ("step_fwd",  "STEP FORWARD, THEN BACK",     8),
    ("step_side", "STEP LEFT, THEN RIGHT",       8),
    ("crouch",    "CROUCH DOWN, THEN STAND",     8),
    ("bend",      "BEND FORWARD, THEN STAND",    8),
    ("reach",     "REACH OUT TO ONE SIDE",       8),
    ("natural",   "MOVE NATURALLY",             10),
]


def hud(img_portrait, target_m, you_m, cue, colour, sub="", bar=None, extra=None):
    """Letterboxes the portrait frame so the subject is not distorted on a landscape screen."""
    canvas = np.zeros((800, 1280, 3), dtype=np.uint8)
    h, w = img_portrait.shape[:2]
    scale = 760.0 / h
    disp = cv2.resize(img_portrait, (int(w * scale), 760), interpolation=cv2.INTER_LINEAR)
    disp = (disp * 0.45).astype(np.uint8)
    x0 = (1280 - disp.shape[1]) // 2
    canvas[20:780, x0:x0 + disp.shape[1]] = disp
    cv2.rectangle(canvas, (x0, 20), (x0 + disp.shape[1], 780), (70, 70, 70), 2)

    cv2.rectangle(canvas, (0, 0), (1280, 78), (18, 18, 18), -1)
    cv2.putText(canvas, ("TARGET %.2f m" % target_m) if target_m else "TARGET --",
                (30, 54), cv2.FONT_HERSHEY_SIMPLEX, 1.3, WHITE, 3, cv2.LINE_AA)
    if you_m:
        cv2.putText(canvas, "YOU %.2f m" % you_m, (1280 - 340, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, colour, 3, cv2.LINE_AA)

    scale_t = 2.6 if len(cue) <= 18 else 1.7
    (tw, th), _ = cv2.getTextSize(cue, cv2.FONT_HERSHEY_SIMPLEX, scale_t, 7)
    cv2.putText(canvas, cue, ((1280 - tw) // 2, 420), cv2.FONT_HERSHEY_SIMPLEX,
                scale_t, colour, 7, cv2.LINE_AA)
    if sub:
        (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        cv2.putText(canvas, sub, ((1280 - sw) // 2, 490), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, WHITE, 3, cv2.LINE_AA)
    if extra:
        cv2.putText(canvas, extra, (30, 770), cv2.FONT_HERSHEY_SIMPLEX, 0.75, GREY, 2)
    if bar is not None:
        bw = int(1180 * max(0.0, min(1.0, bar)))
        cv2.rectangle(canvas, (50, 726), (1230, 756), (55, 55, 55), -1)
        cv2.rectangle(canvas, (50, 726), (50 + bw, 756), GREEN, -1)
    cv2.imshow(WIN, canvas)
    return cv2.waitKey(1) & 0xFF


def extents(uv, conf, w, h, thr=0.3):
    """Body extents and per-part in-frame flags, in the ROTATED (portrait) frame."""
    ok = conf[:17] >= thr
    idx = [i for i in range(17) if ok[i]]
    if not idx:
        return None
    us = np.array([uv[i, 0] for i in idx])
    vs = np.array([uv[i, 1] for i in idx])
    out = {
        "topV": float(vs.min()), "botV": float(vs.max()),
        "leftU": float(us.min()), "rightU": float(us.max()),
        "bodyH": float(vs.max() - vs.min()), "bodyW": float(us.max() - us.min()),
        "marginTop": float(vs.min()), "marginBot": float(h - 1 - vs.max()),
        "marginLeft": float(us.min()), "marginRight": float(w - 1 - us.max()),
        "nKp": len(idx),
    }
    for name, group in (("head", HEAD), ("feet", FEET), ("hands", HANDS)):
        g = [i for i in group if ok[i]]
        out[name + "Seen"] = len(g)
        if g:
            gu = [uv[i, 0] for i in g]
            gv = [uv[i, 1] for i in g]
            out[name + "In"] = int(all(0 <= x < w for x in gu) and all(0 <= y < h for y in gv))
            out[name + "V"] = float(np.mean(gv))
            out[name + "U"] = float(np.mean(gu))
        else:
            out[name + "In"] = 0
    # hand span in pixels (the arm-spread question)
    if ok[LWR] and ok[RWR]:
        out["handSpan"] = float(abs(uv[LWR, 0] - uv[RWR, 0]))
    return out


def open_out(name):
    os.makedirs(OUTDIR, exist_ok=True)
    base = os.path.join(OUTDIR, name)
    path, n = base + ".jsonl", 0
    while os.path.exists(path):
        n += 1
        path = "%s_%d.jsonl" % (base, n)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"..\..\..\SentisModel\rtmw3d-x.onnx")
    ap.add_argument("--config", default="sub3", help="stereo config (F-16 names)")
    ap.add_argument("--protocol", default="frame", choices=["frame", "torso", "move", "probe"])
    ap.add_argument("--distances", default="0.70,0.75,0.78,0.80,0.85,0.90,1.00")
    ap.add_argument("--width", type=float, default=333.1)
    ap.add_argument("--tol", type=float, default=0.06)
    ap.add_argument("--hold", type=float, default=1.0)
    ap.add_argument("--prep", type=float, default=6.0)
    ap.add_argument("--pose-secs", type=float, default=0.0,
                    help="override the per-block record time")
    ap.add_argument("--seek-timeout", type=float, default=60.0)
    ap.add_argument("--global-timeout", type=float, default=560.0)
    ap.add_argument("--rotation", default="auto", choices=["auto", "cw", "ccw"])
    ap.add_argument("--name", default="")
    a = ap.parse_args()

    print("[f18] loading RTMW3D ...")
    model = R.RTMW3D(a.model)
    dists = [float(x) for x in a.distances.split(",") if x.strip()]
    if a.pose_secs > 0:
        for lst in (FRAME_POSES, MOVE_POSES):
            lst[:] = [(x[0], x[1], a.pose_secs) for x in lst]
        TORSO_POSES[:] = [(x[0], x[1], a.pose_secs, x[3]) for x in TORSO_POSES]
    path = open_out(a.name or ("f18_%s_%s" % (a.protocol, time.strftime("%H%M%S"))))
    print("[f18] writing %s" % path)

    pipe, cfg, mw, mh, rw, rh = C.build(a.config)
    t_global = time.time()
    counts = {}

    with io.open(path, "w", encoding="utf-8") as fh, dai.Device(pipe) as dev:
        q_rgb = dev.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_dep = dev.getOutputQueue("depth", maxSize=4, blocking=False)
        intr_land = D.read_rgb_intrinsics(dev, rw, rh)
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # ---------------- decide the rotation direction from the image itself ----------------
        for _ in range(15):
            q_rgb.tryGet()
            q_dep.tryGet()
        if a.rotation == "auto":
            # The direction can only be decided from a REAL upright person: on an empty room the
            # model hallucinates ~0.63 body confidence for both rotations, which decides nothing.
            # Wait for a confident detection, and say so if one never arrives.
            print("[f18] detecting rotation - STAND IN FRONT OF THE CAMERA ...")
            direction, rep, t_wait = None, None, time.time()
            while time.time() - t_wait < 60.0:
                probe = q_rgb.get().getCvFrame()
                d, r = PT.detect_rotation(model, R, probe)
                best = max(r[k]["body_conf"] for k in r)
                if best >= 0.55 and r[d]["upright"]:
                    direction, rep = d, r
                    break
                cv2.imshow(WIN, cv2.resize(probe, (960, 600)))
                cv2.waitKey(1)
            if direction is None:
                print("[f18] COULD NOT DETECT a confident upright person in either rotation.")
                print("[f18] Is the camera actually rotated, and is someone in frame?")
                return
            print("[f18] rotation auto-detected: %s" % direction)
            for k, v in rep.items():
                print("      %-4s %s" % (k, v))
        else:
            direction, rep = a.rotation, {"forced": a.rotation}
        intr = PT.rotate_intrinsics(intr_land, rw, rh, direction)
        pw, ph = rh, rw                                  # portrait frame size
        hfov = PT.fov_deg(intr[0], pw)
        vfov = PT.fov_deg(intr[1], ph)
        print("[f18] portrait %dx%d  fx %.3f fy %.3f cx %.3f cy %.3f   H %.1f deg  V %.1f deg"
              % (pw, ph, intr[0], intr[1], intr[2], intr[3], hfov, vfov))
        fh.write(json.dumps({"_meta": True, "rotation": direction, "detect": rep,
                             "config": a.config, "portrait_w": pw, "portrait_h": ph,
                             "intr_portrait": list(intr), "intr_landscape": list(intr_land),
                             "hfov": hfov, "vfov": vfov,
                             "protocol": a.protocol}) + "\n")

        K = a.width * intr[0]
        bbox = R.center_bbox(pw, ph)
        seq = [0]
        t_prev = [time.time()]

        def grab():
            nonlocal bbox
            rgb_pkt = q_rgb.get()
            dep_pkt = q_dep.get()
            try:
                lat = (dai.Clock.now() - rgb_pkt.getTimestamp()).total_seconds() * 1000.0
            except Exception:
                lat = -1.0
            rgb = PT.rotate_image(rgb_pkt.getCvFrame(), direction)
            depth = PT.rotate_image(dep_pkt.getFrame(), direction)
            uv, zrel, conf = model.infer(rgb, bbox)
            refined = R.bbox_from_keypoints(uv, conf, pw, ph, thr=0.3)
            if refined is not None and float(np.mean(conf[0:17])) >= 0.3:
                bbox = tuple(0.7 * np.array(bbox) + 0.3 * np.array(refined))
            else:
                bbox = R.center_bbox(pw, ph)
            span = abs(float(uv[LSH, 0]) - float(uv[RSH, 0]))
            okc = min(float(conf[LSH]), float(conf[RSH])) >= 0.4 and span > 8
            est = (K / span / 1000.0) if okc else None
            now = time.time()
            fps = 1.0 / max(1e-6, now - t_prev[0])
            t_prev[0] = now
            return est, span, rgb, depth, uv, conf, lat, fps

        def write(block, prompt, heading, tgt, uv, conf, depth, est, lat, fps):
            seq[0] += 1
            xyz, meas, qual, _dg = D.backproject(
                uv, depth, pw, ph, intr, k=CAP.KWIN, with_quality=True, legacy=False)
            sxd = depth.shape[1] / float(pw)
            syd = depth.shape[0] / float(ph)
            uL, vL = float(uv[LSH, 0]), float(uv[LSH, 1])
            uR, vR = float(uv[RSH, 0]), float(uv[RSH, 1])
            p30L, nvL = CAP.pct(depth, uL * sxd, vL * syd, 30.0)
            p30R, nvR = CAP.pct(depth, uR * sxd, vR * syd, 30.0)
            hip_ok = meas[LHIP] and meas[RHIP]
            hipZ = float(0.5 * (xyz[LHIP, 2] + xyz[RHIP, 2])) if hip_ok else None
            aa = [float(xyz[LSH, 0]), float(xyz[LSH, 1]), float(xyz[LSH, 2])]
            bb = [float(xyz[RSH, 0]), float(xyz[RSH, 1]), float(xyz[RSH, 2])]
            yaw = CAP.line_yaw_deg(aa, bb) if (meas[LSH] and meas[RSH]) else None
            ext = extents(uv, conf, pw, ph)
            rec = {
                "seq": seq[0], "t": round(time.time(), 4), "cfg": a.config,
                "block": block, "prompt": prompt, "heading": heading, "target_m": tgt,
                "estRange_m": round(est, 3) if est else None,
                "uL": round(uL, 2), "vL": round(vL, 2), "uR": round(uR, 2), "vR": round(vR, 2),
                "cL": round(float(conf[LSH]), 3), "cR": round(float(conf[RSH]), 3),
                "uSpan": round(abs(uL - uR), 2),
                "zL": round(float(xyz[LSH, 2]) * 1000.0, 1),
                "zR": round(float(xyz[RSH, 2]) * 1000.0, 1),
                "qL": round(float(qual[LSH]), 3), "qR": round(float(qual[RSH]), 3),
                "mL": int(bool(meas[LSH])), "mR": int(bool(meas[RSH])),
                "p30L": p30L, "p30R": p30R, "nvL": nvL, "nvR": nvR,
                "wL": CAP.window_unique(depth, uL * sxd, vL * syd),
                "wR": CAP.window_unique(depth, uR * sxd, vR * syd),
                "hipZ": round(hipZ, 4) if hipZ is not None else None,
                "dx": round((bb[0] - aa[0]) * 1000.0, 2),
                "dz": round((bb[2] - aa[2]) * 1000.0, 2),
                "yaw3D": round(yaw, 3) if yaw is not None else None,
                "latMs": round(lat, 1), "fps": round(fps, 2),
                "kp": [[round(float(uv[i, 0]), 1), round(float(uv[i, 1]), 1),
                        round(float(conf[i]), 3)] for i in range(17)],
                "ext": ext,
            }
            fh.write(json.dumps(rec) + "\n")
            return yaw, ext

        def seek(tgt, timeout):
            t0 = time.time()
            stable = None
            while True:
                if time.time() - t_global > a.global_timeout:
                    return False
                if time.time() - t0 > timeout:
                    return True
                est, span, rgb, depth, uv, conf, lat, fps = grab()
                if est is None:
                    stable = None
                    hud(rgb, tgt, None, "STEP INTO VIEW", RED, sub="no person detected")
                    continue
                err = est - tgt
                if abs(err) <= a.tol:
                    if stable is None:
                        stable = time.time()
                    elif time.time() - stable >= a.hold:
                        return True
                    cue, col = "HOLD %.1f" % (a.hold - (time.time() - stable)), GREEN
                else:
                    stable = None
                    cue = ("MOVE BACK    %4.0f cm" % (-err * 100.0)) if err < 0 else \
                          ("MOVE FORWARD %4.0f cm" % (err * 100.0))
                    col = AMBER if abs(err) <= 0.20 else RED
                if hud(rgb, tgt, est, cue, col, span and ("span %.0f px" % span)) == ord("q"):
                    return False
            return True

        def run_block(label, prompt, secs, heading, tgt):
            t0 = time.time()
            while time.time() - t0 < a.prep:
                est, span, rgb, depth, uv, conf, lat, fps = grab()
                if hud(rgb, tgt, est, prompt, AMBER,
                       sub="get ready ... %.0f" % max(1.0, a.prep - (time.time() - t0))) == ord("q"):
                    return None
            CAP.beep(1200, 220)
            t0 = time.time()
            n = 0
            while time.time() - t0 < secs:
                est, span, rgb, depth, uv, conf, lat, fps = grab()
                yaw, ext = write(label, prompt, heading, tgt, uv, conf, depth, est, lat, fps)
                n += 1
                left = secs - (time.time() - t0)
                warn = ""
                if ext:
                    miss = [k for k in ("head", "feet", "hands") if not ext.get(k + "In", 1)]
                    if miss:
                        warn = "OUT OF FRAME: " + ",".join(miss)
                if hud(rgb, tgt, est, "HOLD", GREEN,
                       sub="%s   %.0f s" % (prompt, max(0.0, left)),
                       bar=1.0 - left / secs, extra=warn) == ord("q"):
                    return None
            CAP.beep(500, 120)
            CAP.beep(500, 120)
            print("  %-10s %-30s %4d frames" % (label, prompt, n))
            return n

        # ---------------------------------------------------------------- protocols
        if a.protocol == "probe":
            print("[f18] probe only - rotation and intrinsics reported above")
        elif a.protocol == "frame":
            for tgt in dists:
                if not seek(tgt, a.seek_timeout):
                    break
                for lab, prompt, secs in FRAME_POSES:
                    n = run_block("%s@%03d" % (lab, int(tgt * 100)), prompt, secs, None, tgt)
                    if n is None:
                        break
                    counts["%s@%03d" % (lab, int(tgt * 100))] = n
        elif a.protocol == "torso":
            for tgt in dists:
                if not seek(tgt, a.seek_timeout):
                    break
                for lab, prompt, secs, heading in TORSO_POSES:
                    n = run_block("%s@%03d" % (lab, int(tgt * 100)), prompt, secs, heading, tgt)
                    if n is None:
                        break
                    counts["%s@%03d" % (lab, int(tgt * 100))] = n
        elif a.protocol == "move":
            tgt = dists[0]
            if seek(tgt, a.seek_timeout):
                for lab, prompt, secs in MOVE_POSES:
                    n = run_block("%s@%03d" % (lab, int(tgt * 100)), prompt, secs, None, tgt)
                    if n is None:
                        break
                    counts["%s@%03d" % (lab, int(tgt * 100))] = n

    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    except Exception:
        pass
    print("[f18] done: %d blocks" % len(counts))
    print("[f18] %s" % path)


if __name__ == "__main__":
    main()
