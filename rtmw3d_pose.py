#!/usr/bin/env python3
"""
RTMW3D (RTMPose3D family) whole-body 3D pose — host-side inference for the Virtual Mirror sidecar.

Runs the RTMW3D-x ONNX (OpenMMLab, COCO-WholeBody, 133 keypoints = body 17 + feet 6 + face 68 +
hands 42) on the host GPU via ONNX Runtime (DirectML). This is the "GPU AI model" half of the
whole-body + depth plan (ADR-015 / ADR-018): it produces rich 2D keypoints (x,y in image pixels) plus
a root-relative metric depth z per keypoint. Measured metric depth from the OAK-D is fused on top in
the sender (Stage C); this module is depth-agnostic.

I/O (confirmed on-device, ADR-015):
  input  "input"  [1,3,384,288]  NCHW RGB, ImageNet mean/std normalized
  output "output" [1,133,576]  SimCC x bins  (288 * 2)
  output "1554"   [1,133,768]  SimCC y bins  (384 * 2)
  output "1556"   [1,133,576]  SimCC z bins  (root-relative metric depth)

Decode (mmpose SimCC / SimCC3DLabel):
  x_px = argmax(x) / 2.0            in [0,288)
  y_px = argmax(y) / 2.0            in [0,384)
  z_m  = (argmax(z)/(576/2) - 1) * 2.1744869   root-relative metres, centred at 0
"""

import numpy as np
import cv2
import onnxruntime as ort

# --- model constants ---
INPUT_W = 288
INPUT_H = 384
INPUT_ASPECT = INPUT_W / INPUT_H  # 0.75 (portrait 3:4)
SIMCC_SPLIT = 2.0
Z_BINS = 576
Z_RANGE = 2.1744869  # mmpose rtmpose3d z_range
# ImageNet normalization (RTMPose default), RGB order
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

# --- COCO-WholeBody 133-keypoint layout ---
WB_BODY = slice(0, 17)     # COCO-17 body
WB_FOOT = slice(17, 23)    # l_big_toe,l_small_toe,l_heel, r_big_toe,r_small_toe,r_heel
WB_FACE = slice(23, 91)    # 68 face
WB_LHAND = slice(91, 112)  # 21 left hand  (wrist + thumb..pinky, MediaPipe order)
WB_RHAND = slice(112, 133)  # 21 right hand

# COCO-17 body index -> Virtual Mirror JointId (BlazePose 33 order). Only joints COCO provides.
COCO17_TO_JOINTID = {
    0: 0,    # nose        -> Nose
    1: 2,    # left_eye    -> LeftEye
    2: 5,    # right_eye   -> RightEye
    3: 7,    # left_ear    -> LeftEar
    4: 8,    # right_ear   -> RightEar
    5: 11,   # left_shoulder  -> LeftShoulder
    6: 12,   # right_shoulder -> RightShoulder
    7: 13,   # left_elbow  -> LeftElbow
    8: 14,   # right_elbow -> RightElbow
    9: 15,   # left_wrist  -> LeftWrist
    10: 16,  # right_wrist -> RightWrist
    11: 23,  # left_hip    -> LeftHip
    12: 24,  # right_hip   -> RightHip
    13: 25,  # left_knee   -> LeftKnee
    14: 26,  # right_knee  -> RightKnee
    15: 27,  # left_ankle  -> LeftAnkle
    16: 28,  # right_ankle -> RightAnkle
}
# WholeBody foot index (absolute, 0..132) -> JointId
FOOT_TO_JOINTID = {
    17: 31,  # left_big_toe  -> LeftFootIndex
    19: 29,  # left_heel     -> LeftHeel
    20: 32,  # right_big_toe -> RightFootIndex
    22: 30,  # right_heel    -> RightHeel
}

# Skeleton edges (JointId space) for the debug overlay.
JOINTID_EDGES = [
    (11, 13), (13, 15), (12, 14), (14, 16), (11, 12),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (27, 31), (28, 30), (28, 32),
    (0, 11), (0, 12),
]


def adjust_bbox_to_aspect(cx, cy, w, h, aspect=INPUT_ASPECT):
    """Grow the shorter side so (w/h) == aspect, keeping the centre fixed."""
    if w <= 0 or h <= 0:
        return cx, cy, w, h
    if w / h > aspect:
        h = w / aspect
    else:
        w = h * aspect
    return cx, cy, w, h


class RTMW3D:
    def __init__(self, model_path, providers=None):
        if providers is None:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        options = ort.SessionOptions()
        self.session = ort.InferenceSession(model_path, sess_options=options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.active_providers = self.session.get_providers()

    def _preprocess(self, frame_bgr, bbox):
        """bbox = (cx,cy,w,h) already aspect-adjusted. Returns (blob, meta) with a translate+scale
        warp (no rotation) so keypoints invert back to frame pixels exactly."""
        cx, cy, w, h = bbox
        x0 = cx - w / 2.0
        y0 = cy - h / 2.0
        sx = INPUT_W / w
        sy = INPUT_H / h
        warp = np.array([[sx, 0.0, -x0 * sx],
                         [0.0, sy, -y0 * sy]], dtype=np.float32)
        crop = cv2.warpAffine(frame_bgr, warp, (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - MEAN) / STD
        blob = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
        return blob, (x0, y0, w, h)

    def _decode(self, outputs, meta):
        x0, y0, w, h = meta
        simcc_x = outputs[0][0]  # [133,576]
        simcc_y = outputs[1][0]  # [133,768]
        simcc_z = outputs[2][0]  # [133,576]
        ix = simcc_x.argmax(axis=1)
        iy = simcc_y.argmax(axis=1)
        iz = simcc_z.argmax(axis=1)
        conf = np.minimum(simcc_x.max(axis=1), simcc_y.max(axis=1))
        px = ix / SIMCC_SPLIT  # input-space pixels [0,288)
        py = iy / SIMCC_SPLIT  # [0,384)
        u = x0 + px / INPUT_W * w  # frame pixels
        v = y0 + py / INPUT_H * h
        z = (iz / (Z_BINS / 2.0) - 1.0) * Z_RANGE  # root-relative metres
        uv = np.stack([u, v], axis=1).astype(np.float32)  # [133,2]
        return uv, z.astype(np.float32), conf.astype(np.float32)

    def infer(self, frame_bgr, bbox):
        """Return (uv[133,2] frame pixels, z[133] root-rel metres, conf[133])."""
        blob, meta = self._preprocess(frame_bgr, bbox)
        outputs = self.session.run(None, {self.input_name: blob})
        return self._decode(outputs, meta)


def bbox_from_keypoints(uv, conf, frame_w, frame_h, thr=0.3, margin=0.25):
    """Person box (cx,cy,w,h, aspect-adjusted) from confident keypoints; None if too few."""
    mask = conf > thr
    if mask.sum() < 4:
        return None
    pts = uv[mask]
    x_min = float(pts[:, 0].min())
    x_max = float(pts[:, 0].max())
    y_min = float(pts[:, 1].min())
    y_max = float(pts[:, 1].max())
    w = (x_max - x_min) * (1.0 + margin)
    h = (y_max - y_min) * (1.0 + margin)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    return adjust_bbox_to_aspect(cx, cy, w, h)


def center_bbox(frame_w, frame_h):
    """Bootstrap box: the largest centred 3:4 portrait region of the frame."""
    h = float(frame_h)
    w = h * INPUT_ASPECT
    if w > frame_w:
        w = float(frame_w)
        h = w / INPUT_ASPECT
    return frame_w / 2.0, frame_h / 2.0, w, h
