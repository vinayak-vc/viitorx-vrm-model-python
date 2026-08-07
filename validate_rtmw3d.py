#!/usr/bin/env python3
"""
Stage-B validation: run RTMW3D on a live OAK-D RGB frame and draw the detected whole-body skeleton.

Saves an annotated PNG so the decode + landmark-pixel mapping can be eyeballed (the single most
error-prone step). No UDP, no depth yet — this only answers "do the keypoints land on the body?".

Usage:
  ..\.venv\Scripts\python validate_rtmw3d.py --model <path-to-rtmw3d-x.onnx> --out frame.png
"""

import argparse
import time

import numpy as np
import cv2
import depthai as dai

import rtmw3d_pose as R

# COCO-17 body skeleton edges (WholeBody indices 0..16).
COCO_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
]
# Hand connections (relative to a hand's 21-point base: wrist=0, thumb 1-4, index 5-8, ...).
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def build_pipeline():
    """OAK-D-PRO-W (OV9782 1280x800) full-FOV color at 640x400 via ISP 1/2."""
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
    cam.setIspScale(1, 2)  # 1280x800 -> 640x400, full FOV
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")
    cam.isp.link(xout.input)
    return pipeline


def draw(frame, uv, conf, thr=0.3):
    out = frame.copy()
    def pt(i):
        return (int(round(uv[i, 0])), int(round(uv[i, 1])))
    # body
    for a, b in COCO_EDGES:
        if conf[a] > thr and conf[b] > thr:
            cv2.line(out, pt(a), pt(b), (255, 180, 90), 2, cv2.LINE_AA)
    for i in range(0, 17):
        if conf[i] > thr:
            cv2.circle(out, pt(i), 3, (0, 255, 0), -1)
    # feet
    for i in range(17, 23):
        if conf[i] > thr:
            cv2.circle(out, pt(i), 2, (0, 200, 255), -1)
    # hands
    for base in (91, 112):
        for a, b in HAND_EDGES:
            ia, ib = base + a, base + b
            if conf[ia] > thr and conf[ib] > thr:
                cv2.line(out, pt(ia), pt(ib), (200, 120, 255), 1, cv2.LINE_AA)
        for i in range(base, base + 21):
            if conf[i] > thr:
                cv2.circle(out, pt(i), 2, (255, 0, 255), -1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="rtmw3d_validate.png")
    parser.add_argument("--warmup", type=int, default=20, help="frames to skip for auto-exposure")
    args = parser.parse_args()

    print("[validate] loading model ...")
    model = R.RTMW3D(args.model)
    print("[validate] providers:", model.active_providers)

    with dai.Device(build_pipeline()) as device:
        q = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        frame = None
        for _ in range(args.warmup):
            frame = q.get().getCvFrame()
        h, w = frame.shape[:2]
        print("[validate] frame %dx%d" % (w, h))

        # two-pass: centre bbox -> refine from keypoints -> final
        bbox = R.center_bbox(w, h)
        t0 = time.time()
        uv, z, conf = model.infer(frame, bbox)
        refined = R.bbox_from_keypoints(uv, conf, w, h)
        if refined is not None:
            bbox = refined
            uv, z, conf = model.infer(frame, bbox)
        dt = (time.time() - t0) * 1000

        body_conf = conf[0:17]
        lh_conf = conf[91:112]
        rh_conf = conf[112:133]
        print("[validate] infer %.0f ms | body conf mean=%.2f max=%.2f | Lhand mean=%.2f | Rhand mean=%.2f"
              % (dt, body_conf.mean(), body_conf.max(), lh_conf.mean(), rh_conf.mean()))
        print("[validate] nose uv=%s conf=%.2f | Lwrist(9) uv=%s conf=%.2f | Rwrist(10) uv=%s conf=%.2f"
              % (uv[0].round(1), conf[0], uv[9].round(1), conf[9], uv[10].round(1), conf[10]))
        print("[validate] z range: min=%.2f max=%.2f (root-rel metres)" % (float(z.min()), float(z.max())))

        annotated = draw(frame, uv, conf)
        cv2.rectangle(annotated,
                      (int(bbox[0] - bbox[2] / 2), int(bbox[1] - bbox[3] / 2)),
                      (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)),
                      (0, 255, 255), 1)
        cv2.imwrite(args.out, annotated)
        cv2.imwrite(args.out.replace(".png", "_raw.png"), frame)
        print("[validate] wrote", args.out)


if __name__ == "__main__":
    main()
