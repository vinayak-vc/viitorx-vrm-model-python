#!/usr/bin/env python3
"""
Stage-C validation: fuse OAK-D measured depth with RTMW3D keypoints and eyeball the result.

Captures RGB + RGB-aligned stereo depth, runs RTMW3D, samples depth at each keypoint, back-projects
to metric camera-space XYZ, and saves:
  <out>.png       RGB with keypoints coloured green=measured / red=depth-hole  (the pixel-mapping check)
  <out>_depth.png depth colormap with the same keypoints overlaid
It reports measured coverage and a few metric XYZ so the depth can be sanity-checked before wiring UDP.

Usage:
  ..\.venv\Scripts\python validate_depth.py --model <rtmw3d-x.onnx> --out depthcheck.png
"""

import argparse
import time

import numpy as np
import cv2
import depthai as dai

import rtmw3d_pose as R
import oak_depth as D

COCO_EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
              (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


def draw_points(img, uv, measured, conf, thr=0.3):
    for i in range(uv.shape[0]):
        if conf[i] <= thr:
            continue
        p = (int(round(uv[i, 0])), int(round(uv[i, 1])))
        color = (0, 255, 0) if measured[i] else (0, 0, 255)  # green=measured, red=hole
        cv2.circle(img, p, 3, color, -1)
    for a, b in COCO_EDGES:
        if conf[a] > thr and conf[b] > thr:
            cv2.line(img, (int(uv[a, 0]), int(uv[a, 1])), (int(uv[b, 0]), int(uv[b, 1])), (255, 180, 90), 1, cv2.LINE_AA)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="depthcheck.png")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--scan", type=float, default=8.0, help="seconds to scan, keeping the best-confidence frame")
    args = parser.parse_args()

    print("[depth] loading model ...")
    model = R.RTMW3D(args.model)

    with dai.Device(D.build_rgbd_pipeline()) as device:
        q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)
        frame = None
        depth = None
        for _ in range(args.warmup):
            frame = q_rgb.get().getCvFrame()
            depth = q_depth.get().getFrame()  # uint16 mm

        # Scan: repeatedly grab a frame, run a quick centre-bbox inference, and keep the frame with
        # the highest mean body confidence. Robust to the user still moving into position.
        rgb_h, rgb_w = frame.shape[:2]
        best_conf = -1.0
        best_frame = frame
        best_depth = depth
        print("[depth] scanning %.0fs for the best pose — get into position now ..." % args.scan)
        t_end = time.time() + args.scan
        while time.time() < t_end:
            f = q_rgb.get().getCvFrame()
            d = q_depth.get().getFrame()
            uv0, _, conf0 = model.infer(f, R.center_bbox(rgb_w, rgb_h))
            score = float(conf0[0:17].mean())
            if score > best_conf:
                best_conf = score
                best_frame = f
                best_depth = d
        frame = best_frame
        depth = best_depth
        print("[depth] best frame body-conf mean=%.2f" % best_conf)
        rgb_h, rgb_w = frame.shape[:2]
        dh, dw = depth.shape
        print("[depth] rgb %dx%d  depth %dx%d" % (rgb_w, rgb_h, dw, dh))

        intr = D.read_rgb_intrinsics(device, dw, dh)
        print("[depth] intrinsics fx=%.1f fy=%.1f cx=%.1f cy=%.1f" % intr)

        bbox = R.center_bbox(rgb_w, rgb_h)
        uv, zrel, conf = model.infer(frame, bbox)
        refined = R.bbox_from_keypoints(uv, conf, rgb_w, rgb_h)
        if refined is not None:
            bbox = refined
            uv, zrel, conf = model.infer(frame, bbox)

        xyz, measured = D.backproject(uv, depth, rgb_w, rgb_h, intr)

        # coverage among CONFIDENT keypoints
        conf_mask = conf > 0.3
        cov_all = int((measured & conf_mask).sum())
        n_conf = int(conf_mask.sum())
        body_meas = int((measured[0:17] & conf_mask[0:17]).sum())
        lh_meas = int((measured[91:112] & conf_mask[91:112]).sum())
        rh_meas = int((measured[112:133] & conf_mask[112:133]).sum())
        print("[depth] measured coverage: %d/%d confident kpts | body %d/17 | Lhand %d/21 | Rhand %d/21"
              % (cov_all, n_conf, body_meas, lh_meas, rh_meas))

        def report(name, i):
            tag = "measured" if measured[i] else "HOLE"
            print("   %-8s conf=%.2f uv=%s  xyz(m)=%s  %s" % (name, conf[i], uv[i].round(1), xyz[i].round(3), tag))
        report("nose", 0)
        report("Lshld", 5)
        report("Rshld", 6)
        report("Lwrist", 9)
        report("Rwrist", 10)
        report("Lhip", 11)
        report("Rhip", 12)
        report("Lankle", 15)

        if measured[11] and measured[12]:
            hip = (xyz[11] + xyz[12]) / 2.0
            print("[depth] measured mid-hip (m): %s   (depth %.2f m from camera)" % (hip.round(3), hip[2]))

        # overlays
        overlay = draw_points(frame.copy(), uv, measured, conf)
        cv2.imwrite(args.out, overlay)
        dnorm = np.clip(depth, 0, 4000).astype(np.float32) / 4000.0 * 255.0
        dcolor = cv2.applyColorMap(dnorm.astype(np.uint8), cv2.COLORMAP_JET)
        dcolor[depth == 0] = (0, 0, 0)
        # map keypoints into depth px for the depth overlay
        sx = dw / float(rgb_w)
        sy = dh / float(rgb_h)
        for i in range(uv.shape[0]):
            if conf[i] > 0.3:
                p = (int(uv[i, 0] * sx), int(uv[i, 1] * sy))
                cv2.circle(dcolor, p, 2, (255, 255, 255), -1)
        cv2.imwrite(args.out.replace(".png", "_depth.png"), dcolor)
        print("[depth] wrote", args.out, "and", args.out.replace(".png", "_depth.png"))


if __name__ == "__main__":
    main()
