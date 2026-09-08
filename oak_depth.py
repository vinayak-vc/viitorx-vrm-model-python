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


# ---------------------------------------------------------------- F-08 surface-aware sampling
# The pre-F-08 sampler asked only "do I have >= min_valid pixels?". The F-08 audit
# (docs/F08_MEASUREMENT_CONFIDENCE_AUDIT_2026-09-08.md) measured that this is NOT depth quality:
# windows are 24.0-24.5 of 25 valid essentially always, yet 15-32% of them SPAN A DEPTH
# DISCONTINUITY > 150 mm, and those produce frame-to-frame depth jumps 10.4x larger at p99
# (3945 mm vs 379 mm). Counting valid pixels cannot see that; separating surfaces can.
#
# Gap above which two sorted depths are treated as different surfaces. A limb is a few cm thick at
# ~2 m, while person-vs-background is typically >= 0.5 m, so 100 mm sits well inside that margin.
SURFACE_GAP_MM = 100.0
# Robust spread (IQR, mm) at which a surface stops looking flat. Range (max-min) was tried first
# and rejected: a single 25-pixel window of genuinely flat wall carries +/-20 mm of stereo noise,
# whose RANGE is ~40 mm and which collapsed quality to 0.35 on a perfectly good surface (test E).
# IQR ignores the two tails that cause that. 200 mm is well above sensor noise at ~2 m and well
# below the >=500 mm person-vs-background step this is meant to separate.
SURFACE_SPREAD_TOL_MM = 200.0
# Separation (mm) from the nearest competing cluster at which the selection is unambiguous.
SURFACE_SEP_TOL_MM = 200.0


def _clamp01(v):
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return float(v)


def _pct_sorted(sv, q):
    """Linear-interpolated percentile of an ALREADY-SORTED 1-D array. np.percentile re-sorts and
    carries large fixed overhead, which dominated the cost on 25-element windows."""
    n = sv.size
    if n == 1:
        return float(sv[0])
    pos = (n - 1) * q / 100.0
    lo = int(pos)
    hi = lo + 1 if lo + 1 < n else lo
    frac = pos - lo
    return float(sv[lo]) + (float(sv[hi]) - float(sv[lo])) * frac


def _iqr(sv):
    """Robust spread of an already-sorted array."""
    return _pct_sorted(sv, 75.0) - _pct_sorted(sv, 25.0)


def _window_bounds(depth, u, v, k):
    dh, dw = depth.shape
    x = int(round(u))
    y = int(round(v))
    r = k // 2
    return (max(0, x - r), min(dw, x + r + 1),
            max(0, y - r), min(dh, y + r + 1), x, y)


def sample_depth_surface(depth, u, v, k=5, min_valid=6, percentile=30.0,
                         gap_mm=SURFACE_GAP_MM, spread_tol=SURFACE_SPREAD_TOL_MM,
                         sep_tol=SURFACE_SEP_TOL_MM, want_diag=True):
    """Surface-aware depth of a KxK window at (u,v).

    Returns (depth_mm, depth_quality, diag):
      depth_mm      float, 0.0 when unusable (the same failure value the old sampler used)
      depth_quality float in [0,1]; 0 = unusable/ambiguous, 1 = strong coherent surface.
                    NOT a probability -- it is a score over observable window geometry.
      diag          dict: clusterCount, selectedClusterOccupancy, selectedClusterSpread,
                    depthWindowSpread, validPixelCount, reason

    METHOD. The valid depths are sorted and split wherever the gap between adjacent values exceeds
    `gap_mm`; each run is one candidate surface. The surface is chosen by SPATIAL PROXIMITY TO THE
    WINDOW CENTRE -- the keypoint asserts the joint is at that pixel, so the surface at/nearest that
    pixel is the joint's surface. Deliberately NOT "nearest depth" (a chair in front of the leg would
    win) and NOT "largest cluster" (background usually wins at a limb edge). No temporal state is
    used, so this introduces no new temporal owner.

    COMPATIBILITY. The selected cluster is reduced with the SAME `percentile` the old sampler used,
    so a single-surface window returns a value identical to the previous implementation. Only
    genuinely multi-surface windows can differ -- which is the entire point of the change.

    AMBIGUITY. If the centre pixel is invalid AND more than one cluster reaches the centre equally
    closely, no depth is invented: the legacy whole-window percentile is returned with quality 0, so
    behaviour degrades to exactly what shipped before rather than to something new.
    """
    # The sender back-projects all 133 keypoints per frame, so building a diagnostics dict for each
    # is real cost for values nobody reads. `want_diag=False` skips it entirely.
    diag = ({"clusterCount": 0, "selectedClusterOccupancy": 0.0, "selectedClusterSpread": 0.0,
             "depthWindowSpread": 0.0, "validPixelCount": 0, "reason": "ok"}
            if want_diag else None)
    x0, x1, y0, y1, cx_px, cy_px = _window_bounds(depth, u, v, k)
    if x1 <= x0 or y1 <= y0:
        if diag is not None:
            diag["reason"] = "outside"
        return 0.0, 0.0, diag

    win = depth[y0:y1, x0:x1]
    n_total = int(win.size)
    mask = win > 0
    vals = win[mask].astype(np.float32)
    n_valid = int(vals.size)
    if diag is not None:
        diag["validPixelCount"] = n_valid
    if n_valid < min_valid:
        if diag is not None:
            diag["reason"] = "sparse"
        return 0.0, 0.0, diag

    order = np.argsort(vals, kind="stable")
    sv = vals[order]
    if diag is not None:
        diag["depthWindowSpread"] = float(sv[-1] - sv[0])
    valid_ratio = float(n_valid) / n_total

    # Split into surfaces wherever consecutive sorted depths jump by more than gap_mm.
    cuts = np.nonzero(np.diff(sv) > gap_mm)[0]
    n_clusters = int(cuts.size) + 1
    if diag is not None:
        diag["clusterCount"] = n_clusters

    # COMMON CASE (measured: the large majority of windows) -- one surface, no cluster work,
    # and the value is identical to what the pre-F-08 sampler returned.
    if n_clusters == 1:
        if diag is not None:
            diag["selectedClusterOccupancy"] = 1.0
            diag["selectedClusterSpread"] = float(sv[-1] - sv[0])
        q = valid_ratio * _clamp01(1.0 - _iqr(sv) / spread_tol)
        return _pct_sorted(sv, percentile), _clamp01(q), diag

    # Pixel distance from the window centre, valid pixels only (L1 is enough at k <= 11).
    ys, xs = np.nonzero(mask)
    dist = (np.abs(xs + x0 - cx_px) + np.abs(ys + y0 - cy_px))[order]
    sdist = dist
    starts = np.concatenate(([0], cuts + 1))
    ends = np.concatenate((cuts + 1, [sv.size]))
    legacy = _pct_sorted(sv, percentile)

    # Multiple surfaces: pick the one nearest the keypoint pixel.
    best, best_key = -1, None
    for ci in range(starts.size):
        a, b = starts[ci], ends[ci]
        key = (float(sdist[a:b].min()), -(b - a))     # nearest pixel; ties -> larger cluster
        if best_key is None or key < best_key:
            best_key, best = key, ci
    a, b = starts[best], ends[best]
    sel = sv[a:b]
    sel_min_dist = float(sdist[a:b].min())

    ties = 0
    for ci in range(starts.size):
        if float(sdist[starts[ci]:ends[ci]].min()) <= sel_min_dist:
            ties += 1
    center_valid = (0 <= cy_px < depth.shape[0] and 0 <= cx_px < depth.shape[1]
                    and depth[cy_px, cx_px] > 0)
    if ties > 1 and not center_valid:
        if diag is not None:
            diag["reason"] = "ambiguous"
            diag["selectedClusterOccupancy"] = float(sel.size) / max(1, n_valid)
            diag["selectedClusterSpread"] = _iqr(sel)
        return legacy, 0.0, diag

    spread = _iqr(sel)
    occupancy = float(sel.size) / max(1, n_valid)
    centre = _pct_sorted(sel, percentile)
    # Depth distance to the closest competing surface -- how separable the choice actually was.
    sep = None
    for ci in range(starts.size):
        if ci == best:
            continue
        other = sv[starts[ci]:ends[ci]]
        d = min(abs(float(other[0]) - centre), abs(float(other[-1]) - centre))
        sep = d if sep is None else min(sep, d)
    if diag is not None:
        diag["selectedClusterOccupancy"] = occupancy
        diag["selectedClusterSpread"] = spread
        diag["reason"] = "multi"
    q = (valid_ratio
         * occupancy
         * _clamp01(1.0 - spread / spread_tol)
         * _clamp01((sep or 0.0) / sep_tol))
    return centre, _clamp01(q), diag


def sample_depth_mm(depth, u, v, k=5, min_valid=6, percentile=30.0):
    """Depth (mm) of a KxK window at (u,v); 0.0 if unusable.

    Thin wrapper preserving the original call contract. See `sample_depth_surface` for the quality
    value and the per-window diagnostics.
    """
    z, _q, _d = sample_depth_surface(depth, u, v, k=k, min_valid=min_valid, percentile=percentile,
                                     want_diag=False)
    return z


def sample_depth_legacy_mm(depth, u, v, k=5, min_valid=6, percentile=30.0):
    """The PRE-F-08 sampler, unchanged, kept ONLY so the A/B in
    docs/F08_SURFACE_AWARE_DEPTH_IMPLEMENTATION_2026-09-08.md can be reproduced. Not used in
    production; do not call it from the pipeline."""
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


def backproject(uv, depth, rgb_w, rgb_h, intr, k=5, min_valid=6, with_quality=False,
                legacy=False):
    """Back-project RTMW3D keypoints (uv in RGB-frame px) to camera-space metres using measured depth.

    Returns (xyz[N,3] metres, measured[N] bool). Depth holes -> measured=False, xyz=0.

    With `with_quality=True` returns (xyz, measured, quality[N] float32, diags[list]) instead, where
    quality is the F-08 depth-quality scalar in [0,1] and diags carries the per-window diagnostics.
    The 2-tuple form is the default so existing callers are unaffected.
    """
    fx, fy, cx, cy = intr
    dh, dw = depth.shape
    sx = dw / float(rgb_w)
    sy = dh / float(rgb_h)
    n = uv.shape[0]
    xyz = np.zeros((n, 3), dtype=np.float32)
    measured = np.zeros(n, dtype=bool)
    quality = np.zeros(n, dtype=np.float32) if with_quality else None
    diags = [None] * n if with_quality else None
    for i in range(n):
        u_d = uv[i, 0] * sx
        v_d = uv[i, 1] * sy
        if legacy:
            # A/B only (--no-surface-depth): the exact pre-F-08 sampler, no quality available.
            z_mm = sample_depth_legacy_mm(depth, u_d, v_d, k=k, min_valid=min_valid)
            q, dg = 0.0, None
        else:
            z_mm, q, dg = sample_depth_surface(depth, u_d, v_d, k=k, min_valid=min_valid,
                                               want_diag=with_quality)
        if with_quality:
            quality[i] = q
            diags[i] = dg
        if z_mm <= 0.0:
            continue
        z = z_mm / 1000.0
        x = (u_d - cx) * z / fx
        y = (v_d - cy) * z / fy
        xyz[i, 0] = x
        xyz[i, 1] = y
        xyz[i, 2] = z
        measured[i] = True
    if with_quality:
        return xyz, measured, quality, diags
    return xyz, measured
