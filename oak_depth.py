#!/usr/bin/env python3
"""
OAK-D stereo-depth utilities for the whole-body sidecar (Stage C, measured per-keypoint depth).

Builds an RGB + RGB-aligned stereo-depth DepthAI pipeline, samples the aligned depth map at each
RTMW3D keypoint pixel, and back-projects through the RGB camera intrinsics to metric camera-space
XYZ. This is the "measured depth" that removes the monocular front/back ambiguity (doc 26).

DepthAI camera-space convention: X right, Y down, Z forward (away from camera), millimetres.
"""

import numpy as np
import depthai as dai


def build_rgbd_pipeline(color_res="800p", isp_num=1, isp_den=2, mono_res="400p"):
    """OAK-D-PRO-W: full-FOV color (OV9782 1280x800 -> 640x400 via ISP 1/2) + stereo depth aligned to RGB."""
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
    cam.setIspScale(isp_num, isp_den)  # 640x400 full FOV
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(30)

    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_left.setFps(30)
    mono_right.setFps(30)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # align depth -> RGB frame
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam.isp.link(xout_rgb.input)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    return pipeline


def read_rgb_intrinsics(device, width, height):
    """RGB (CAM_A) intrinsics scaled to (width,height) — the depth frame is aligned to CAM_A."""
    calib = device.readCalibration()
    matrix = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, width, height), dtype=np.float32)
    fx = float(matrix[0][0])
    fy = float(matrix[1][1])
    cx = float(matrix[0][2])
    cy = float(matrix[1][2])
    return fx, fy, cx, cy


def sample_depth_mm(depth, u, v, k=5, min_valid=6, percentile=30.0):
    """Robust depth (mm) of a KxK window at (u,v). 0.0 if too few valid (hole).

    Uses a low percentile rather than the median: a keypoint sits ON the person, who is always in
    FRONT of the background, so when the window straddles the person/background edge the nearer
    (person) surface is the correct pick. percentile=50 recovers the plain median.
    """
    dh, dw = depth.shape
    x = int(round(u))
    y = int(round(v))
    r = k // 2
    x0 = max(0, x - r)
    x1 = min(dw, x + r + 1)
    y0 = max(0, y - r)
    y1 = min(dh, y + r + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    window = depth[y0:y1, x0:x1].reshape(-1)
    window = window[window > 0]
    if window.size < min_valid:
        return 0.0
    return float(np.percentile(window, percentile))


def backproject(uv, depth, rgb_w, rgb_h, intr, k=5, min_valid=6):
    """Back-project RTMW3D keypoints (uv in RGB-frame px) to camera-space metres using measured depth.

    Returns (xyz[N,3] metres, measured[N] bool). Depth holes -> measured=False, xyz=0.
    """
    fx, fy, cx, cy = intr
    dh, dw = depth.shape
    sx = dw / float(rgb_w)
    sy = dh / float(rgb_h)
    n = uv.shape[0]
    xyz = np.zeros((n, 3), dtype=np.float32)
    measured = np.zeros(n, dtype=bool)
    for i in range(n):
        u_d = uv[i, 0] * sx
        v_d = uv[i, 1] * sy
        z_mm = sample_depth_mm(depth, u_d, v_d, k=k, min_valid=min_valid)
        if z_mm <= 0.0:
            continue
        z = z_mm / 1000.0
        x = (u_d - cx) * z / fx
        y = (v_d - cy) * z / fy
        xyz[i, 0] = x
        xyz[i, 1] = y
        xyz[i, 2] = z
        measured[i] = True
    return xyz, measured
