#!/usr/bin/env python3
"""
Whole-body + measured-depth OAK-D sidecar -> UDP (Stage B + C).

Replaces the Phase-1 BlazePose sidecar (depthai_blazepose/udp_pose_sender.py). Pipeline:
  OAK-D RGB (640x400 full FOV) + RGB-aligned stereo depth
    -> RTMW3D-x whole-body 2D keypoints on the host GPU (ONNX Runtime / DirectML)
    -> sample measured depth per keypoint + back-project through RGB intrinsics -> metric camera XYZ
    -> hip-centre -> stream 33 body + 2x21 hand landmarks (hip-relative metres) + measured mid-hip.

Measured depth removes the monocular front/back ("hands behind body") ambiguity at the source — the
whole point of using the OAK-D (ADR-016 / ADR-018, doc 26).

Wire contract (JSON per UDP datagram, default 127.0.0.1:8899):
  { "lm":  [[x,y,z,vis] x33],   # body, hip-relative metres, JointId order (unmapped slots = zeros)
    "lh":  [[x,y,z] x21],        # left hand, hip-relative metres (MediaPipe/COCO-WB hand order)
    "rh":  [[x,y,z] x21],        # right hand
    "xyz": [hipX,hipY,hipZ],     # measured mid-hip, millimetres, camera space (avatar root)
    "src": [0|1 x33] }           # 1 = measured depth, 0 = hip-plane fallback (debug/coverage)

Axes are camera space (X right, Y down, Z forward) as with the Phase-1 GHUM stream, so the Unity
PoseSpaceConverter + poseFlipX/Y/Z tuning apply the same way. Run:
  cd python-sidecar~ && .venv\Scripts\python wholebody_udp_sender.py --model <rtmw3d-x.onnx> [--show]
"""

import argparse
import json
import socket
import time

import numpy as np
import cv2
import depthai as dai

import rtmw3d_pose as R
import oak_depth as D
import smoothing

NUM_BODY = 33

# Left<->right JointId pairs for mirroring (a true reflection swaps sides AND negates X).
MIRROR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
                (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]


def build_body_landmarks(uv, xyz_cam, measured, conf, mid_hip, hip_z, intr, conf_thr, flatten_trunk=True):
    """Map the 133 WholeBody keypoints -> a 33-slot JointId body array of [x,y,z,vis] hip-relative m.

    Measured keypoints use their back-projected XYZ minus the mid-hip. Confident-but-unmeasured
    keypoints (depth hole) fall back to the hip-plane depth so the limb stays plausible, with halved
    confidence. Returns (lm33, src33)."""
    fx, fy, cx, cy = intr
    lm = [[0.0, 0.0, 0.0, 0.0] for _ in range(NUM_BODY)]
    src = [0] * NUM_BODY

    def emit(wb_index, joint_id):
        c = float(conf[wb_index])
        if c < conf_thr:
            return
        if measured[wb_index]:
            p = xyz_cam[wb_index] - mid_hip
            lm[joint_id] = [float(p[0]), float(p[1]), float(p[2]), c]
            src[joint_id] = 1
        else:
            # hip-plane fallback: place the joint at the body depth plane (z_rel = 0), keep x,y from px.
            u_d = uv[wb_index, 0]
            v_d = uv[wb_index, 1]
            x = (u_d - cx) * hip_z / fx
            y = (v_d - cy) * hip_z / fy
            p = np.array([x, y, hip_z], dtype=np.float32) - mid_hip
            lm[joint_id] = [float(p[0]), float(p[1]), 0.0, c * 0.5]
            src[joint_id] = 0

    for coco_idx, joint_id in R.COCO17_TO_JOINTID.items():
        emit(coco_idx, joint_id)
    for foot_idx, joint_id in R.FOOT_TO_JOINTID.items():
        emit(foot_idx, joint_id)

    # Trunk-flatten: the torso "up" = mid-shoulder - mid-hip is hypersensitive to depth, and RTMW3D+stereo
    # measures the upper body ~0.5-0.7 m farther than the hips, which tips the near-vertical spine into a
    # permanent forward hunch. Zero the Z of the 4 trunk joints (shoulders + hips) so the trunk stays
    # vertical (same idea as the Unity "hips upright-only" StableUp). Limbs/hands keep full measured depth.
    if flatten_trunk:
        for joint_id in (11, 12, 23, 24):  # Left/RightShoulder, Left/RightHip (JointId)
            if lm[joint_id][3] > 0.0:
                lm[joint_id][2] = 0.0
    return lm, src


def build_hand(uv, xyz_cam, measured, conf, mid_hip, hip_z, intr, base, conf_thr):
    """21 hand landmarks (hip-relative metres). Uses measured depth where available, hip-plane else."""
    fx, fy, cx, cy = intr
    out = []
    for k in range(21):
        i = base + k
        if measured[i]:
            p = xyz_cam[i] - mid_hip
        else:
            u_d = uv[i, 0]
            v_d = uv[i, 1]
            x = (u_d - cx) * hip_z / fx
            y = (v_d - cy) * hip_z / fy
            p = np.array([x, y, hip_z], dtype=np.float32) - mid_hip
        out.append([round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--conf", type=float, default=0.3, help="keypoint confidence threshold")
    parser.add_argument("--kwin", type=int, default=5, help="depth sampling window (px)")
    parser.add_argument("--show", action="store_true", help="cv2 preview of the OAK view + skeleton")
    parser.add_argument("--flatten-trunk", action=argparse.BooleanOptionalAction, default=True,
                        help="zero the Z of shoulders+hips so the trunk stays vertical (fixes the forward hunch from noisy torso depth). --no-flatten-trunk to disable.")
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=False,
                        help="DEFAULT OFF. Mirroring the INPUT skeleton (negate X + swap sides) reflects the pose, but the FK retarget builds rotations with LookRotation and a reflected skeleton twists the torso/limbs. Leave off (clean 'copy' retarget); do the mirror on the Unity/avatar side instead. --mirror to experiment.")
    parser.add_argument("--smooth", action=argparse.BooleanOptionalAction, default=True,
                        help="temporal One-Euro smoothing + depth-outlier gate to kill jitter. --no-smooth to disable.")
    parser.add_argument("--min-cutoff", type=float, default=1.0, help="One-Euro min cutoff Hz (lower = smoother when still)")
    parser.add_argument("--beta", type=float, default=0.02, help="One-Euro beta (higher = more responsive to fast motion)")
    parser.add_argument("--max-jump", type=float, default=1.5, help="rate-limit a keypoint that jumps more than this many metres in one frame (only catches gross depth-spike garbage; keep it ABOVE the depth-quantization step so normal motion is untouched)")
    parser.add_argument("--seconds", type=float, default=0.0, help="auto-stop after N seconds (0 = run forever)")
    args = parser.parse_args()

    print("[wb] loading RTMW3D ...")
    model = R.RTMW3D(args.model)
    print("[wb] providers:", model.active_providers)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (args.host, args.port)

    with dai.Device(D.build_rgbd_pipeline()) as device:
        q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)
        first = q_rgb.get().getCvFrame()
        rgb_h, rgb_w = first.shape[:2]
        depth0 = q_depth.get().getFrame()
        dh, dw = depth0.shape
        intr = D.read_rgb_intrinsics(device, dw, dh)
        print("[wb] rgb %dx%d depth %dx%d intr fx=%.1f fy=%.1f cx=%.1f cy=%.1f  -> %s"
              % (rgb_w, rgb_h, dw, dh, intr[0], intr[1], intr[2], intr[3], str(addr)))

        bbox = R.center_bbox(rgb_w, rgb_h)
        body_smoother = None
        if args.smooth:
            body_smoother = smoothing.KeypointSmoother(133, min_cutoff=args.min_cutoff, beta=args.beta, max_jump=args.max_jump)
        frames = 0
        sent = 0
        t_log = time.time()
        t_start = time.time()
        while True:
            if args.seconds > 0.0 and (time.time() - t_start) > args.seconds:
                break
            frame = q_rgb.get().getCvFrame()
            depth = q_depth.get().getFrame()
            frames += 1

            uv, zrel, conf = model.infer(frame, bbox)
            # track: refine bbox from this frame's keypoints (smoothed), else re-centre
            refined = R.bbox_from_keypoints(uv, conf, rgb_w, rgb_h, thr=args.conf)
            if refined is not None:
                bbox = tuple(0.7 * np.array(bbox) + 0.3 * np.array(refined))
            else:
                bbox = R.center_bbox(rgb_w, rgb_h)

            xyz_cam, measured = D.backproject(uv, depth, rgb_w, rgb_h, intr, k=args.kwin)

            # Smooth the metric keypoints at the source (One-Euro + depth-outlier gate) to kill jitter,
            # before hip-centring / building the message. Unmeasured points reset their filter.
            if body_smoother is not None:
                si = 0
                sn = xyz_cam.shape[0]
                while si < sn:
                    sx, sy, sz = body_smoother.filter(si, float(xyz_cam[si, 0]), float(xyz_cam[si, 1]), float(xyz_cam[si, 2]), bool(measured[si]))
                    xyz_cam[si, 0] = sx
                    xyz_cam[si, 1] = sy
                    xyz_cam[si, 2] = sz
                    si = si + 1

            # measured mid-hip (metres). Require both hips measured for a stable root; else skip send.
            if not (measured[11] and measured[12]):
                if args.show:
                    _preview(frame, uv, conf, args.conf, frames, sent, "no hip depth")
                    if cv2.waitKey(1) in (27, ord("q")):
                        break
                continue
            mid_hip = (xyz_cam[11] + xyz_cam[12]) / 2.0
            hip_z = float(mid_hip[2])

            lm, src = build_body_landmarks(uv, xyz_cam, measured, conf, mid_hip, hip_z, intr, args.conf, args.flatten_trunk)
            lh = build_hand(uv, xyz_cam, measured, conf, mid_hip, hip_z, intr, 91, args.conf)
            rh = build_hand(uv, xyz_cam, measured, conf, mid_hip, hip_z, intr, 112, args.conf)
            # round body
            lm = [[round(v, 4) for v in p] for p in lm]
            message = {
                "lm": lm,
                "lh": lh,
                "rh": rh,
                "xyz": [round(float(mid_hip[0] * 1000.0), 1),
                        round(float(mid_hip[1] * 1000.0), 1),
                        round(float(mid_hip[2] * 1000.0), 1)],
                "src": src,
            }
            # Exact mirror = negate X AND swap left<->right. Negate-X ALONE crosses the limbs on a
            # rotational retarget (open arms read as crossed); swapping the anatomical sides too gives a
            # true reflection with correct open/close. Keep Unity poseFlipX OFF (single flip here).
            if args.mirror:
                lm_m = message["lm"]
                src_m = message["src"]
                for p in lm_m:
                    p[0] = -p[0]
                for p in message["lh"]:
                    p[0] = -p[0]
                for p in message["rh"]:
                    p[0] = -p[0]
                message["xyz"][0] = -message["xyz"][0]
                for a, b in MIRROR_PAIRS:
                    lm_m[a], lm_m[b] = lm_m[b], lm_m[a]
                    src_m[a], src_m[b] = src_m[b], src_m[a]
                message["lh"], message["rh"] = message["rh"], message["lh"]
            sock.sendto(json.dumps(message).encode("utf-8"), addr)
            sent += 1

            if time.time() - t_log > 2.0:
                cov = int(sum(src))
                print("[wb] frames=%d sent=%d hip_z=%.2fm measured_body=%d/33 fps~%.1f"
                      % (frames, sent, hip_z, cov, frames / (time.time() - t_log + 1e-6)))
                frames = 0
                t_log = time.time()

            if args.show:
                _preview(frame, uv, conf, args.conf, frames, sent, "hip %.2fm" % hip_z)
                if cv2.waitKey(1) in (27, ord("q")):
                    break

    sock.close()
    print("[wb] stopped.")


def _preview(frame, uv, conf, thr, frames, sent, status):
    view = frame.copy()
    for i in range(uv.shape[0]):
        if conf[i] > thr:
            cv2.circle(view, (int(uv[i, 0]), int(uv[i, 1])), 2,
                       (0, 255, 0) if i < 91 else (255, 0, 255), -1)
    cv2.putText(view, "sent=%d %s" % (sent, status), (8, 20), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 1)
    cv2.imshow("whole-body OAK sidecar", view)


if __name__ == "__main__":
    main()
