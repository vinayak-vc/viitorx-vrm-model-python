#!/usr/bin/env python3
"""F-19 section 7 - level the mount, using the OAK-D's own BNO086 rather than a protractor.

WHAT IS AND IS NOT TRUSTED HERE, because getting this wrong would silently mis-state the mount:

The device calibration returns getImuToCameraExtrinsics(CAM_A) with an IDENTITY rotation. That is a
PLACEHOLDER on this board, not a measured extrinsic - it was disproved directly: a raw frame shows
the subject's head at the +X edge (world-up = camera -X), while the IMU simultaneously reports
gravity along its own +Y. IMU X/Y therefore do NOT coincide with camera X/Y.

What IS established, empirically: rotating the camera 90 deg about its optical axis between two
readings changed the IMU X and Y components completely while leaving Z unchanged (-1.129 -> -1.126).
Only the roll axis itself is invariant under a roll, so the IMU's Z axis IS the optical axis.

Consequently:
  * TILT (the optical axis's elevation away from horizontal) is computed from the Z component and is
    trustworthy.
  * ROLL is reported as the in-plane gravity direction, against a reference ANCHORED TO TWO RAW
    FRAMES rather than to the calibration. Two orientations were captured and the images inspected
    directly:
        in-plane   +1.8 deg  <-> subject lying sideways, head at the +X edge  => PORTRAIT (ccw)
        in-plane  -90.7 deg  <-> subject upright in the raw frame             => LANDSCAPE
    so PORTRAIT_INPLANE_DEG below is a measured constant, not an assumption, and the HUD can name
    the orientation instead of inferring it.
  * The TILT SIGN is not asserted from the accelerometer either. The HUD is self-correcting: tilt
    toward zero and the number falls, tilt the wrong way and it grows.

Height is the one item the device cannot supply; it is passed in with --height-m and recorded.

    python tools/capture/f19_level.py --height-m 0.80
"""

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(_os.path.join(_d, "_sidecar_path.py")):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import _sidecar_path  # noqa: F401  - puts the sidecar root and every tools/ group on sys.path
import evidence_paths as EV

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

WIN = "F-19 LEVEL THE MOUNT"
HUD_W, HUD_H = 1600, 900
WHITE = (255, 255, 255)
GREEN = (90, 230, 90)
AMBER = (0, 190, 255)
RED = (60, 60, 255)
GREY = (150, 150, 150)
OUT = EV.oak_v4("f19", "mount.json")
TILT_TOL_DEG = 2.0
ROLL_TOL_DEG = 3.0
# Measured, image-anchored (see the module docstring). Not a guess and not from calibration.
PORTRAIT_INPLANE_DEG = 1.8
LANDSCAPE_INPLANE_DEG = -90.7


def angles(mean):
    """(tilt magnitude from horizontal, in-plane gravity angle) from a mean accelerometer vector.

    Only |tilt| is claimed; see the module docstring for why the sign and the roll zero are not.
    """
    v = np.asarray(mean, dtype=float)
    n = v / np.linalg.norm(v)
    tilt = math.degrees(math.asin(abs(float(n[2]))))
    inplane = math.degrees(math.atan2(float(n[0]), float(n[1])))
    return tilt, inplane


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height-m", type=float, default=None,
                    help="tape-measured height of the camera above the floor (metres)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    p = dai.Pipeline()
    imu = p.create(dai.node.IMU)
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 100)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)
    x = p.create(dai.node.XLinkOut)
    x.setStreamName("imu")
    imu.out.link(x.input)

    cv2.namedWindow(WIN, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    buf = []
    settled_since = None
    result = None
    with dai.Device(p) as dev:
        q = dev.getOutputQueue("imu", maxSize=50, blocking=False)
        t0 = time.time()
        while True:
            for pk in q.get().packets:
                acc = pk.acceleroMeter
                buf.append((acc.x, acc.y, acc.z))
            buf = buf[-60:]
            if len(buf) < 20:
                continue
            A = np.array(buf)
            mean = A.mean(axis=0)
            still = float(np.max(A.std(axis=0))) < 0.06
            tilt, inplane = angles(mean)
            roll_dev = (inplane - PORTRAIT_INPLANE_DEG + 180.0) % 360.0 - 180.0
            land_dev = (inplane - LANDSCAPE_INPLANE_DEG + 180.0) % 360.0 - 180.0
            portrait = abs(roll_dev) < abs(land_dev)

            level = portrait and tilt <= TILT_TOL_DEG and abs(roll_dev) <= ROLL_TOL_DEG
            if level and still:
                settled_since = settled_since or time.time()
            else:
                settled_since = None

            img = np.zeros((HUD_H, HUD_W, 3), np.uint8)
            ok_t = tilt <= TILT_TOL_DEG
            ok_r = portrait and abs(roll_dev) <= ROLL_TOL_DEG
            cv2.putText(img, "PORTRAIT" if portrait else "LANDSCAPE - ROTATE 90 deg", (120, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, GREEN if portrait else RED, 6, cv2.LINE_AA)
            cv2.putText(img, "TILT  %4.1f deg" % tilt, (120, 290),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.0, GREEN if ok_t else AMBER, 8, cv2.LINE_AA)
            cv2.putText(img, "ROLL  %+5.1f deg" % roll_dev, (120, 430),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.0, GREEN if ok_r else AMBER, 8, cv2.LINE_AA)
            msg = ("HOLD - RECORDING" if settled_since else
                   ("KEEP STILL" if not still else "BRING BOTH NUMBERS TO 0"))
            cv2.putText(img, msg, (120, 540), cv2.FONT_HERSHEY_SIMPLEX, 1.6, WHITE, 4, cv2.LINE_AA)
            cv2.putText(img, "each number FALLS when you move the right way, RISES when you do not",
                        (120, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.95, GREY, 2, cv2.LINE_AA)
            cv2.putText(img, "target: tilt <= %.0f, roll <= %.0f    ENTER = accept    Q = abort"
                        % (TILT_TOL_DEG, ROLL_TOL_DEG),
                        (120, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.95, GREY, 2, cv2.LINE_AA)
            cv2.putText(img, "FIX THE CAMERA FIRMLY - it moved between the last two runs",
                        (120, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.95, AMBER, 2, cv2.LINE_AA)
            # two spirit-level bubbles, so direction is readable without reading numbers
            for row, val, span in ((790, float(mean[2]), 3.0), (860, roll_dev, 25.0)):
                cv2.line(img, (200, row), (HUD_W - 200, row), GREY, 3)
                cv2.line(img, (HUD_W // 2, row - 22), (HUD_W // 2, row + 22), GREEN, 3)
                bx = int(HUD_W // 2 + np.clip(val / span, -1, 1) * (HUD_W // 2 - 220))
                cv2.circle(img, (bx, row), 20,
                           GREEN if (ok_t if row == 790 else ok_r) else AMBER, -1)
            if settled_since and time.time() - settled_since > 2.0:
                result = dict(auto=True)
            cv2.imshow(WIN, img)
            k = cv2.waitKey(1) & 0xFF
            if k in (13, 10):
                result = dict(auto=False)
            if result is not None:
                result.update(tilt_deg=round(tilt, 2), roll_dev_deg=round(roll_dev, 2),
                              orientation="PORTRAIT_CCW" if portrait else "LANDSCAPE",
                              inplane_deg=round(inplane, 2),
                              accel_mean=[round(float(v), 4) for v in mean],
                              accel_std=[round(float(v), 4) for v in A.std(axis=0)])
                break
            if k in (27, ord("q")):
                break
            if time.time() - t0 > 300:
                break
    cv2.destroyAllWindows()

    if result is None:
        print("[f19] levelling aborted; mount NOT recorded", file=sys.stderr)
        return 1
    result["height_m"] = a.height_m
    result["tilt_tolerance_deg"] = TILT_TOL_DEG
    result["note"] = ("tilt is the optical-axis elevation from horizontal (magnitude only; the IMU "
                      "sign convention is not asserted). roll is deviation from the portrait "
                      "orientation held at the start of this run, not an absolute.")
    io.open(a.out, "w", encoding="utf-8").write(json.dumps(result, indent=2))
    print("[f19] mount recorded: tilt %.2f deg, roll deviation %+.2f deg, height %s"
          % (result["tilt_deg"], result["roll_dev_deg"],
             ("%.3f m" % a.height_m) if a.height_m else "NOT MEASURED"))
    print("[f19] wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
