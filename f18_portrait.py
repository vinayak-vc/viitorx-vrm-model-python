#!/usr/bin/env python3
"""
F-18 portrait-orientation support (diagnostic only; production is untouched).

THE TRANSFORMATION, and why it is the part that can silently invalidate the whole experiment.

The camera is rotated 90 degrees about its own optical axis. Nothing changes INSIDE the device:
CAM_B and CAM_C are still horizontally separated in the sensor frame, rectification is unchanged,
depth is still computed and aligned to CAM_A exactly as production does. What changes is that a
standing person now lies sideways in the raw frame.

So the host must rotate the RGB **and the depth identically**, and - the step that is easy to miss -
rotate the INTRINSICS with them. For a 90 deg clockwise rotation of a WxH image into HxW:

    u' = H - 1 - v        fx' = fy      cx' = H - 1 - cy
    v' = u                fy' = fx      cy' = cx

and counter-clockwise:

    u' = v                fx' = fy      cx' = cy
    v' = W - 1 - u        fy' = fx      cy' = W - 1 - cx

Back-projecting in the rotated frame with the rotated intrinsics yields (X', Y', Z) where X' is the
subject's left-right, Y' is world-down and Z is depth - i.e. exactly the DepthAI convention the
sidecar already expects (X right, Y down, Z forward). The rotation is therefore SEMANTICS-PRESERVING
as long as the intrinsics are rotated too; rotating the image alone would corrupt every back-
projected X and silently rescale the yaw triangle.

What genuinely does change is the physical stereo geometry relative to the body: the baseline is now
VERTICAL in world terms, so the disparity search runs along the world-vertical and occlusion happens
at HORIZONTAL edges rather than at the left/right silhouette. That is the F-17 open question and the
thing F-18 has to measure, not assume.

Rotation DIRECTION is not assumed - `detect_rotation` tries both and keeps whichever makes the pose
model see an upright person.
"""
import numpy as np
import cv2

CW = "cw"
CCW = "ccw"


def rotate_image(img, direction):
    if direction == CW:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def rotate_intrinsics(intr, w, h, direction):
    """(fx, fy, cx, cy) for a WxH image -> the same for the rotated HxW image."""
    fx, fy, cx, cy = intr
    if direction == CW:
        return (fy, fx, (h - 1) - cy, cx)
    return (fy, fx, cy, (w - 1) - cx)


def rotate_point(u, v, w, h, direction):
    """Map a pixel from the original WxH frame into the rotated HxW frame."""
    if direction == CW:
        return ((h - 1) - v, u)
    return (v, (w - 1) - u)


def unrotate_point(u2, v2, w, h, direction):
    """Inverse of rotate_point, back into the original WxH frame."""
    if direction == CW:
        return (v2, (h - 1) - u2)
    return ((w - 1) - v2, u2)


def fov_deg(f_px, n_px):
    """Full field of view spanned by n_px at focal length f_px."""
    return 2.0 * np.degrees(np.arctan((n_px / 2.0) / f_px))


def detect_rotation(model, R, frame_bgr, tries=(CW, CCW)):
    """Decide the rotation direction from the image, not from an assumption about the mount.

    Runs the pose model on both rotations and keeps the one that yields the higher mean body
    confidence AND a head-above-hips geometry. Returns (direction, report dict).
    """
    report = {}
    best, best_score = None, -1.0
    for d in tries:
        rot = rotate_image(frame_bgr, d)
        h2, w2 = rot.shape[:2]
        bbox = R.center_bbox(w2, h2)
        uv, zrel, conf = model.infer(rot, bbox)
        # second pass, refined box - a sideways person often needs one iteration to lock
        refined = R.bbox_from_keypoints(uv, conf, w2, h2, thr=0.3)
        if refined is not None:
            uv, zrel, conf = model.infer(rot, refined)
        body = float(np.mean(conf[0:17]))
        nose_y = float(uv[0, 1])
        hip_y = float(0.5 * (uv[11, 1] + uv[12, 1]))
        upright = nose_y < hip_y                     # image y grows downward
        score = body + (0.5 if upright else 0.0)
        report[d] = dict(body_conf=round(body, 4), nose_y=round(nose_y, 1),
                         hip_y=round(hip_y, 1), upright=bool(upright), score=round(score, 4))
        if score > best_score:
            best, best_score = d, score
    return best, report
