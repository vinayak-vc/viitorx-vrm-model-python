#!/usr/bin/env python3
"""
F-17 host-side multi-baseline stereo rig.

No wider-baseline camera is attached (f17_inventory.txt), so the brief's primary question cannot be
answered by measuring one. What this board DOES have is three cameras, hence three real stereo
baselines: CAM_A-CAM_B 37.36 mm, CAM_A-CAM_C 37.64 mm, CAM_B-CAM_C 74.99 mm.

That is enough to convert F-16's extrapolation into a MEASURED scaling law. The decisive question
(brief section 9) is whether the systematic +0.75 px matching bias is a property of the PIXEL domain
(then a wider baseline divides the angular error proportionally -> Case A) or of the DEPTH domain
(then it does not -> Case B). Halving the baseline answers that as well as doubling it would.

Design, so only the baseline varies:
  pair BC  (74.99 mm)  rectified on the host, host SGBM   <- reproduces the production geometry
  pair AC  (37.64 mm)  rectified on the host, host SGBM   <- half baseline, IDENTICAL matcher
  device   (74.99 mm)  the OAK StereoDepth, sub-pixel 1/8 <- ties the host rig back to F-16

The pose model runs on each pair's own rectified LEFT image, so each baseline is self-consistent and
needs no cross-camera keypoint mapping. Agreement between `device` and host `BC` is the control that
says the host matcher and the grayscale pose are trustworthy.

Nothing here touches production.
"""
import io
import json
import math
import os
import time

import numpy as np
import cv2
import depthai as dai

import rtmw3d_pose as R

OUTDIR = os.path.join("oak_v4_evidence", "f17")
W, H = 640, 400
WB_L_SHOULDER, WB_R_SHOULDER = 5, 6
WB_L_HIP, WB_R_HIP = 11, 12
KWIN = 5

SOCK = {"A": dai.CameraBoardSocket.CAM_A,
        "B": dai.CameraBoardSocket.CAM_B,
        "C": dai.CameraBoardSocket.CAM_C}

# (left, right) -> the pair is rectified with `left` as the reference frame
PAIRS = [("B", "C"), ("A", "C")]


def line_yaw_deg(a, b):
    """The shipped Kalidokit y-channel, identical to f16_capture.line_yaw_deg."""
    r = math.atan2(b[0] - a[0], b[2] - a[2])
    ang = math.fmod(r, 2 * math.pi)
    if ang > math.pi:
        ang -= 2 * math.pi
    elif ang < -math.pi:
        ang += 2 * math.pi
    y = ang / math.pi
    if y > 0.5:
        y -= 2.0
    y += 0.5
    return math.degrees(y * math.pi)


def calib_arrays(calib, name):
    K = np.array(calib.getCameraIntrinsics(SOCK[name], W, H), dtype=np.float64)
    d = np.array(calib.getDistortionCoefficients(SOCK[name]), dtype=np.float64)
    # OpenCV accepts 4/5/8/12/14 coefficients; depthai returns 14 for the perspective model.
    if d.size not in (4, 5, 8, 12, 14):
        d = d[:8] if d.size > 8 else d[:4]
    return K, d.reshape(1, -1)


def build_rectification(calib):
    """cv2 stereo rectification for every pair in PAIRS, from the device's own EEPROM."""
    rect = {}
    for (ln, rn) in PAIRS:
        K1, D1 = calib_arrays(calib, ln)
        K2, D2 = calib_arrays(calib, rn)
        E = np.array(calib.getCameraExtrinsics(SOCK[ln], SOCK[rn], False), dtype=np.float64)
        Rm = E[:3, :3]
        T = E[:3, 3] * 10.0                       # depthai gives CENTIMETRES
        R1, R2, P1, P2, Q, _v1, _v2 = cv2.stereoRectify(
            K1, D1, K2, D2, (W, H), Rm, T.reshape(3, 1),
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        m1x, m1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (W, H), cv2.CV_16SC2)
        m2x, m2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (W, H), cv2.CV_16SC2)
        f_rect = float(P1[0, 0])
        b_rect = abs(float(P2[0, 3]) / P2[0, 0])   # rectified baseline, mm
        rect["%s%s" % (ln, rn)] = dict(
            left=ln, right=rn, m1=(m1x, m1y), m2=(m2x, m2y),
            f=f_rect, B=b_rect, fB=f_rect * b_rect,
            cx=float(P1[0, 2]), cy=float(P1[1, 2]),
            baseline_raw=float(np.linalg.norm(T)))
    return rect


def make_matcher(num_disp=160, block=5):
    """One SGBM configuration, shared by every baseline, so the matcher is never the variable."""
    return cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=num_disp, blockSize=block,
        P1=8 * block * block, P2=32 * block * block,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=2,
        preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def depth_from_disp(disp16, fB):
    """SGBM returns fixed-point disparity x16. -> depth in mm, 0 where invalid."""
    d = disp16.astype(np.float32) / 16.0
    out = np.zeros_like(d, dtype=np.float32)
    m = d > 0.25
    out[m] = fB / d[m]
    out[(out < 200) | (out > 6000)] = 0.0
    return out


def sample_window(depth, u, v, k=KWIN, pct=30.0):
    """Percentile of the valid depths in a kxk window - the same shape as production's sampler."""
    h, w = depth.shape
    x, y = int(round(u)), int(round(v))
    r = k // 2
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    if x1 <= x0 or y1 <= y0:
        return None, 0, None
    win = depth[y0:y1, x0:x1].reshape(-1)
    win = win[win > 0]
    if win.size < 6:
        return None, int(win.size), None
    return float(np.percentile(win, pct)), int(win.size), float(win.max() - win.min())


def build_pipeline(subpixel=True, bits=3):
    """Raw A/B/C streams PLUS the device's own StereoDepth, so the host rig can be tied to F-16."""
    p = dai.Pipeline()

    cam = p.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
    cam.setIspScale(1, 2)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(30)

    ml = p.create(dai.node.MonoCamera)
    mr = p.create(dai.node.MonoCamera)
    for m, s in ((ml, dai.CameraBoardSocket.CAM_B), (mr, dai.CameraBoardSocket.CAM_C)):
        m.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        m.setBoardSocket(s)
        m.setFps(30)

    st = p.create(dai.node.StereoDepth)
    st.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    st.setLeftRightCheck(True)
    st.setSubpixel(subpixel)
    if subpixel:
        st.setSubpixelFractionalBits(bits)
    st.setExtendedDisparity(False)
    st.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    ml.out.link(st.left)
    mr.out.link(st.right)

    for node, out, name in ((cam, cam.isp, "rgb"), (ml, ml.out, "monoB"),
                            (mr, mr.out, "monoC"), (st, st.depth, "depth")):
        x = p.create(dai.node.XLinkOut)
        x.setStreamName(name)
        out.link(x.input)
    return p


def open_out(name):
    os.makedirs(OUTDIR, exist_ok=True)
    base = os.path.join(OUTDIR, name)
    path, n = base + ".jsonl", 0
    while os.path.exists(path):
        n += 1
        path = "%s_%d.jsonl" % (base, n)
    return path
