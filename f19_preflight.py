#!/usr/bin/env python3
"""F-19 section 8 - camera preflight, run against the REAL production pipeline.

Why this owns the camera directly rather than parsing the sidecar log: section 8 asks for
intrinsics, FOV, depth validity, left/right identity and torso-yaw sign, none of which the
production sidecar emits. It therefore builds PRODUCTION's own pipeline
(oak_depth.build_rgbd_pipeline) and PRODUCTION's own depth sampler (oak_depth.backproject),
applies the F-18 portrait transform imported from f18_portrait, and measures what comes out.
Nothing here is a re-implementation; the only thing this file adds is the measuring.

Stage `device` (this file) covers the geometry half of section 8. The FPS / frame-age /
RGB-depth-sync / UDP-rate half is covered by f19_udp_probe.py, which runs the real sidecar.

The subject stands ~0.9 m from the machine and cannot read a console (F-16 learned this the hard
way), so every instruction goes on a fullscreen HUD.

    python f19_preflight.py --seconds 12
"""
import argparse
import io
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import depthai as dai

import rtmw3d_pose as R
import oak_depth as D
import f18_portrait as PT
from f16_capture import line_yaw_deg

WIN = "F-19 PREFLIGHT"
HUD_W, HUD_H = 1600, 900
WHITE = (255, 255, 255)
GREEN = (90, 230, 90)
AMBER = (0, 190, 255)
GREY = (150, 150, 150)
OUT_DIR = os.path.join("oak_v4_evidence", "f19")
# F-16 measured this subject's shoulder-KEYPOINT separation; used only for the fallback range cue.
W_SHOULDER_MM = 333.1
TARGET_M = 0.90                      # F-18's preferred working distance
COUNTDOWN = 3.0                      # seconds between blocks, not recorded


def read_imu(seconds=2.0):
    """Gravity direction in the CAMERA frame.

    The device calibration reports getImuToCameraExtrinsics(CAM_A) as a pure translation with an
    IDENTITY rotation, so IMU axes and CAM_A axes coincide and no frame conversion is needed. That
    is recorded here rather than assumed, because a non-identity rotation would silently corrupt
    both angles below.
    """
    p = dai.Pipeline()
    imu = p.create(dai.node.IMU)
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 100)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)
    x = p.create(dai.node.XLinkOut)
    x.setStreamName("imu")
    imu.out.link(x.input)
    acc = []
    with dai.Device(p) as dev:
        M = np.array(dev.readCalibration().getImuToCameraExtrinsics(dai.CameraBoardSocket.CAM_A))
        rot_is_identity = bool(np.allclose(M[:3, :3], np.eye(3), atol=1e-6))
        q = dev.getOutputQueue("imu", maxSize=50, blocking=False)
        t0 = time.time()
        while time.time() - t0 < seconds:
            for pk in q.get().packets:
                a = pk.acceleroMeter
                acc.append((a.x, a.y, a.z))
    A = np.array(acc)
    mean = A.mean(axis=0)
    # At rest a MEMS accelerometer reads +1 g along the axis pointing UP, so DOWN is -mean.
    down = -mean / np.linalg.norm(mean)
    # Optical-axis elevation: the angle between +Z and the horizontal plane. 0 = level,
    # positive = the camera is looking DOWN.
    pitch = math.degrees(math.asin(float(np.clip(np.dot(down, [0.0, 0.0, 1.0]), -1.0, 1.0))))
    # Which sensor axis is world-vertical decides landscape vs portrait, independently of any image.
    vert_axis = "Z" if abs(down[2]) > 0.9 else "XYZ"[int(np.argmax(np.abs(down[:2])))]
    # Roll about the optical axis, measured in the sensor XY plane. 0 = landscape upright,
    # +-90 = portrait. What needs levelling in a portrait mount is the deviation from exactly 90.
    roll = math.degrees(math.atan2(float(down[0]), float(down[1])))
    roll_dev = roll - (90.0 if roll > 0 else -90.0)
    return dict(n=len(A), accel_mean=[round(float(v), 4) for v in mean],
                accel_std=[round(float(v), 4) for v in A.std(axis=0)],
                magnitude=round(float(np.linalg.norm(mean)), 3),
                rot_identity=rot_is_identity, down=[round(float(v), 5) for v in down],
                pitch_deg=round(pitch, 2), roll_deg=round(roll, 2),
                roll_dev_from_portrait_deg=round(roll_dev, 2), vertical_axis=vert_axis)


def hud(frame, cue, colour, sub="", you_m=None, bar=None, extra=None):
    """Legible from across the room; the subject is 0.9 m from the camera, not at the keyboard."""
    h, w = frame.shape[:2]
    s = min(HUD_W / float(w), HUD_H / float(h))
    view = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_LINEAR)
    img = np.zeros((HUD_H, HUD_W, 3), np.uint8)
    y0 = (HUD_H - view.shape[0]) // 2
    x0 = (HUD_W - view.shape[1]) // 2
    img[y0:y0 + view.shape[0], x0:x0 + view.shape[1]] = (view * 0.42).astype(np.uint8)

    cv2.rectangle(img, (0, 0), (HUD_W, 84), (18, 18, 18), -1)
    cv2.putText(img, "TARGET  %.2f m" % TARGET_M, (36, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, WHITE, 3, cv2.LINE_AA)
    if you_m:
        c = GREEN if abs(you_m - TARGET_M) <= 0.06 else AMBER
        cv2.putText(img, "YOU  %.2f m" % you_m, (HUD_W - 400, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, c, 3, cv2.LINE_AA)

    scale = 3.2 if len(cue) <= 17 else 2.0
    (tw, th), _ = cv2.getTextSize(cue, cv2.FONT_HERSHEY_SIMPLEX, scale, 8)
    cv2.putText(img, cue, ((HUD_W - tw) // 2, HUD_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 8, cv2.LINE_AA)
    if sub:
        (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.putText(img, sub, ((HUD_W - sw) // 2, HUD_H // 2 + 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 3, cv2.LINE_AA)
    if you_m:
        y = HUD_H - 150
        cv2.line(img, (140, y), (HUD_W - 140, y), GREY, 3)
        cxp = HUD_W // 2
        cv2.line(img, (cxp, y - 34), (cxp, y + 34), GREEN, 5)
        px = int(cxp + float(np.clip((you_m - TARGET_M) / 0.40, -1.0, 1.0)) * (HUD_W // 2 - 150))
        cv2.circle(img, (px, y), 24, GREEN if abs(you_m - TARGET_M) <= 0.06 else AMBER, -1)
        cv2.putText(img, "CLOSER", (150, y + 74), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREY, 2)
        cv2.putText(img, "FURTHER", (HUD_W - 300, y + 74), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREY, 2)
    if bar is not None:
        bw = int((HUD_W - 160) * max(0.0, min(1.0, bar)))
        cv2.rectangle(img, (80, HUD_H - 68), (HUD_W - 80, HUD_H - 30), (60, 60, 60), -1)
        cv2.rectangle(img, (80, HUD_H - 68), (80 + bw, HUD_H - 30), GREEN, -1)
    if extra:
        for i, line in enumerate(extra):
            cv2.putText(img, line, (36, 130 + 34 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, GREY, 2, cv2.LINE_AA)
    cv2.imshow(WIN, img)
    return cv2.waitKey(1) & 0xFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"..\..\..\SentisModel\rtmw3d-x.onnx")
    ap.add_argument("--seconds", type=float, default=12.0, help="square-stance hold to measure")
    ap.add_argument("--warmup", type=float, default=45.0, help="max positioning time before giving up")
    ap.add_argument("--dir", default="", help="force rotation direction; default = detect from the image")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "preflight.json"))
    ap.add_argument("--blocks", default="",
                    help="prompted block protocol, e.g. "
                         "'FACE CAMERA:12,TURN AROUND:12,FACE CAMERA:12'. Each block is cued on the "
                         "HUD and its name recorded per frame. Used for the 180-degree bias test: a "
                         "camera-fixed measurement artefact keeps its sign through a 180 turn, a "
                         "genuinely rotated torso reverses it. Empty = single square-stance block.")
    ap.add_argument("--subpixel-bits", type=int, default=-1,
                    help="DIAGNOSTIC ONLY: override oak_depth.STEREO_CONFIG sub-pixel for THIS "
                         "process (0 = off, 3 = 1/8 px). Changes nothing on disk and nothing in "
                         "production; it exists so the preflight can attribute a failure to the "
                         "stereo configuration instead of guessing. Default -1 = leave production "
                         "settings exactly as shipped.")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if a.subpixel_bits >= 0:
        D.STEREO_CONFIG["subpixel"] = a.subpixel_bits > 0
        D.STEREO_CONFIG["subpixelBits"] = a.subpixel_bits
        print("[f19] DIAGNOSTIC OVERRIDE (in-process only): %s" % D.stereo_config_str(), flush=True)

    print("[f19] reading IMU ...", flush=True)
    imu = read_imu()
    print("[f19] IMU: pitch %+.2f deg, roll %+.2f deg (portrait deviation %+.2f), vertical axis %s"
          % (imu["pitch_deg"], imu["roll_deg"], imu["roll_dev_from_portrait_deg"],
             imu["vertical_axis"]), flush=True)

    print("[f19] loading pose model ...", flush=True)
    model = R.RTMW3D(a.model)

    rec = dict(imu=imu, providers=list(model.active_providers), target_m=TARGET_M)
    with dai.Device(D.build_rgbd_pipeline()) as dev:
        q_rgb = dev.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_depth = dev.getOutputQueue("depth", maxSize=4, blocking=False)
        first = q_rgb.get().getCvFrame()
        rgb_h0, rgb_w0 = first.shape[:2]
        depth0 = q_depth.get().getFrame()
        dh0, dw0 = depth0.shape
        intr_l = D.read_rgb_intrinsics(dev, dw0, dh0)

        # ---- rotation direction, DETECTED, never assumed --------------------------------------
        if a.dir:
            direction, det = a.dir, {"forced": a.dir}
        else:
            direction, det = PT.detect_rotation(model, R, first)
        intr = PT.rotate_intrinsics(intr_l, dw0, dh0, direction)
        rgb_w, rgb_h = rgb_h0, rgb_w0
        dw, dh = dh0, dw0
        rec.update(rotation=direction, detect=det,
                   landscape=dict(rgb=[rgb_w0, rgb_h0], depth=[dw0, dh0],
                                  intr=[float(x) for x in intr_l]),
                   portrait=dict(rgb=[rgb_w, rgb_h], depth=[dw, dh],
                                 intr=[float(x) for x in intr],
                                 hfov=float(PT.fov_deg(intr[0], dw)),
                                 vfov=float(PT.fov_deg(intr[1], dh))),
                   stereo=D.stereo_config_str())
        print("[f19] rotation=%s  portrait %dx%d  fx=%.3f fy=%.3f cx=%.3f cy=%.3f  HFOV %.2f VFOV %.2f"
              % (direction, rgb_w, rgb_h, intr[0], intr[1], intr[2], intr[3],
                 rec["portrait"]["hfov"], rec["portrait"]["vfov"]), flush=True)
        print("[f19] stereo: %s" % rec["stereo"], flush=True)

        cv2.namedWindow(WIN, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        bbox = R.center_bbox(rgb_w, rgb_h)
        rows = []
        lat = []
        sync = []
        t_start = time.time()
        measuring = False
        t_ready0 = None
        # Block protocol. The default is the single square-stance hold section 8 asks for; --blocks
        # replaces it with a prompted sequence. A 3 s countdown precedes each block so the subject
        # has time to change pose without that transition landing in the measured data.
        if a.blocks:
            blocks = []
            for part in a.blocks.split(","):
                name, _, secs = part.rpartition(":")
                blocks.append((name.strip(), float(secs)))
        else:
            blocks = [("STAND SQUARE", a.seconds)]
        rec["blocks"] = [dict(name=n, seconds=s) for n, s in blocks]
        bi = 0
        t_block = None
        while True:
            rp = q_rgb.get()
            dp = q_depth.get()
            for extra in (q_rgb.tryGetAll() or []):
                rp = extra
            dex = q_depth.tryGetAll() or []
            if dex:
                dp = dex[-1]
            try:
                lat.append((dai.Clock.now() - rp.getTimestamp()).total_seconds() * 1000.0)
                sync.append(abs((rp.getTimestamp() - dp.getTimestamp()).total_seconds()) * 1000.0)
            except Exception:
                pass
            frame = PT.rotate_image(rp.getCvFrame(), direction)
            depth = PT.rotate_image(dp.getFrame(), direction)

            uv, zrel, conf = model.infer(frame, bbox)
            ref = R.bbox_from_keypoints(uv, conf, rgb_w, rgb_h, thr=0.3)
            if ref is not None and float(np.mean(conf[0:17])) >= 0.3:
                bbox = tuple(0.7 * np.array(bbox) + 0.3 * np.array(ref))
            else:
                bbox = R.center_bbox(rgb_w, rgb_h)
            xyz, meas, qual, _ = D.backproject(uv, depth, rgb_w, rgb_h, intr, k=5, with_quality=True)

            # COCO-17: 5 = left shoulder, 6 = right shoulder, 11/12 = hips.
            iL, iR = 5, 6
            okL, okR = bool(meas[iL]), bool(meas[iR])
            you = None
            if okL and okR:
                you = (float(xyz[iL, 2]) + float(xyz[iR, 2])) / 2.0
            else:
                span = abs(float(uv[iL, 0]) - float(uv[iR, 0]))
                if span > 4:
                    you = W_SHOULDER_MM * intr[0] / span / 1000.0

            elapsed = time.time() - t_start
            if not measuring:
                ready = you is not None and abs(you - TARGET_M) <= 0.06
                if ready:
                    if t_ready0 is None:
                        t_ready0 = time.time()
                    if time.time() - t_ready0 > 1.2:
                        measuring = True
                        t_start = time.time()
                        continue
                else:
                    t_ready0 = None
                if you is None:
                    cue, col = "STEP INTO VIEW", AMBER
                elif ready:
                    cue, col = "STAND SQUARE", GREEN
                elif you < TARGET_M:
                    cue, col = "STEP BACK", AMBER
                else:
                    cue, col = "STEP FORWARD", AMBER
                k = hud(frame, cue, col, "hold still - measuring starts automatically", you,
                        None if t_ready0 is None else min(1.0, (time.time() - t_ready0) / 1.2),
                        ["rotation %s   %dx%d" % (direction, rgb_w, rgb_h),
                         "pitch %+.1f deg   roll dev %+.1f deg"
                         % (imu["pitch_deg"], imu["roll_dev_from_portrait_deg"])])
                if k in (27, ord("q")):
                    break
                if elapsed > a.warmup:
                    print("[f19] subject never held the target distance; aborting", file=sys.stderr)
                    break
                continue

            name, secs = blocks[bi]
            if t_block is None:
                t_block = time.time()
            age = time.time() - t_block
            if age < COUNTDOWN:
                # the pose change itself must not land in the measured data
                k = hud(frame, name, AMBER, "starting in %.0f" % (COUNTDOWN - age), you,
                        age / COUNTDOWN, ["block %d/%d" % (bi + 1, len(blocks))])
                if k in (27, ord("q")):
                    break
                continue

            if okL and okR:
                pL = [float(xyz[iL, i]) * 1000.0 for i in range(3)]
                pR = [float(xyz[iR, i]) * 1000.0 for i in range(3)]
                rows.append(dict(
                    block=name,
                    t=round(time.time(), 4),
                    uL=round(float(uv[iL, 0]), 2), vL=round(float(uv[iL, 1]), 2),
                    uR=round(float(uv[iR, 0]), 2), vR=round(float(uv[iR, 1]), 2),
                    cL=round(float(conf[iL]), 3), cR=round(float(conf[iR]), 3),
                    xL=round(pL[0], 1), yL=round(pL[1], 1), zL=round(pL[2], 1),
                    xR=round(pR[0], 1), yR=round(pR[1], 1), zR=round(pR[2], 1),
                    qL=round(float(qual[iL]), 3), qR=round(float(qual[iR]), 3),
                    yaw=round(line_yaw_deg(pL, pR), 3),
                    sep=round(math.dist(pL, pR), 1),
                    hipOK=bool(meas[11]) and bool(meas[12]),
                    validPct=round(100.0 * float(np.count_nonzero(depth)) / depth.size, 2),
                    kpIn=int(sum(1 for i in range(17)
                                 if 10 <= uv[i, 0] < rgb_w - 10 and 10 <= uv[i, 1] < rgb_h - 10)),
                    conf17=round(float(np.mean(conf[0:17])), 3)))
            prog = (age - COUNTDOWN) / secs
            k = hud(frame, name, GREEN, "measuring %.0f%%" % (100 * prog), you, prog,
                    ["block %d/%d   frames %d" % (bi + 1, len(blocks), len(rows))])
            if k in (27, ord("q")):
                break
            if prog >= 1.0:
                bi += 1
                t_block = None
                if bi >= len(blocks):
                    break
    cv2.destroyAllWindows()

    if lat:
        rec["frame_age_ms"] = dict(p50=float(np.percentile(lat, 50)),
                                   p95=float(np.percentile(lat, 95)), n=len(lat))
        rec["rgb_depth_sync_ms"] = dict(p50=float(np.percentile(sync, 50)),
                                        p95=float(np.percentile(sync, 95)))
    if not rows:
        print("[f19] PREFLIGHT FAILED: no frames with both shoulders measured", file=sys.stderr)
        rec["result"] = "FAILED - no measured frames"
        io.open(a.out, "w", encoding="utf-8").write(json.dumps(rec, indent=2))
        return 1
    rec["frames"] = rows
    io.open(a.out, "w", encoding="utf-8").write(json.dumps(rec, indent=2))
    print("[f19] wrote %s  (%d measured frames)" % (a.out, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
