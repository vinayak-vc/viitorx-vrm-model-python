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
import os

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

# M16: sign of RTMW3D root-relative z (zrel) relative to the OAK measured camera depth (Z forward,
# farther = +). +1 assumes they agree (a joint the model puts farther gets a larger camera Z). This is
# convention-sensitive — if occluded limbs poke the WRONG way in depth on a live OAK-D, flip to -1.0.
ZREL_SIGN = 1.0


def _fallback_point(wb_index, uv, zrel, zrel_hip, mid_hip, hip_z, intr, use_zrel):
    """Hip-relative metric position for a keypoint with NO measured depth (a depth hole).

    M16: with use_zrel, offset the joint from the hip depth plane by the model's own root-relative z
    (zrel[i] - zrel_hip) so an occluded joint the model believes is forward/back is placed there,
    instead of flattening every hole to the hip plane (z_rel=0). Falls back to the hip plane when
    use_zrel is off. x,y come from back-projecting the pixel at the chosen depth."""
    fx, fy, cx, cy = intr
    if use_zrel:
        z_cam = hip_z + ZREL_SIGN * (float(zrel[wb_index]) - zrel_hip)
        if z_cam < 0.2:  # keep the depth positive so the pinhole back-projection stays sane
            z_cam = 0.2
    else:
        z_cam = hip_z
    u_d = uv[wb_index, 0]
    v_d = uv[wb_index, 1]
    x = (u_d - cx) * z_cam / fx
    y = (v_d - cy) * z_cam / fy
    return np.array([x, y, z_cam], dtype=np.float32) - mid_hip


def build_body_landmarks(uv, xyz_cam, measured, conf, zrel, zrel_hip, mid_hip, hip_z, intr, conf_thr,
                         flatten_trunk=True, use_zrel=True):
    """Map the 133 WholeBody keypoints -> a 33-slot JointId body array of [x,y,z,vis] hip-relative m.

    Measured keypoints use their back-projected XYZ minus the mid-hip. Confident-but-unmeasured
    keypoints (depth hole) fall back via `_fallback_point` (M16: model root-relative z, or the hip
    plane when use_zrel is off), with halved confidence. Returns (lm33, src33)."""
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
            p = _fallback_point(wb_index, uv, zrel, zrel_hip, mid_hip, hip_z, intr, use_zrel)
            lm[joint_id] = [float(p[0]), float(p[1]), float(p[2]), c * 0.5]
            src[joint_id] = 0

    for coco_idx, joint_id in R.COCO17_TO_JOINTID.items():
        emit(coco_idx, joint_id)
    for foot_idx, joint_id in R.FOOT_TO_JOINTID.items():
        emit(foot_idx, joint_id)

    # Trunk-flatten for a STABLE, UPRIGHT, FRONT-FACING avatar. The raw torso depth causes two problems:
    #  (1) RTMW3D+stereo measures the upper body ~0.5-0.7 m farther than the hips -> the near-vertical spine
    #      tips into a forward hunch;
    #  (2) the shoulder/hip lines carry noisy Z, and the Kalidokit hips/spine yaw is HYPERSENSITIVE to it
    #      (measured live in Unity: ~0.10 m of hip-Z rotates the avatar ~40 deg, ~0.25 m ~64 deg) -> the
    #      avatar swings into PROFILE even when the user faces the camera.
    # History: the 2026-08-07 fix zeroed ONLY the 4 trunk joints' Z, which left the elbows/wrists at measured
    # Z -> a false forward ARM-bend. A later "shift the whole upper body by mid-shoulder Z" attempt fixed the
    # arm-bend but left the hip Z measured -> it REGRESSED facing into profile (problem 2 above, 2026-08-10).
    # Fix (ADR-023): zero the shoulders' AND hips' Z so both lines stay horizontal -> upright trunk + a
    # stable frontal facing (single-camera depth can't drive facing reliably; ADR-019 precedent). To avoid
    # the false arm-bend, SHIFT each arm's joints by ITS shoulder's Z *before* zeroing the shoulder — this
    # preserves the arm's segment vectors EXACTLY (upper-arm/forearm unchanged, just re-based to a Z=0
    # shoulder). Head shifts with the shoulders; legs keep measured depth.
    # TRADE-OFF: facing is frontal-locked (turning is NOT tracked) — the single-front-camera limit; a real
    # turn needs multi-view or a stabilized-depth facing signal. A stable frontal avatar >> a profile one.
    if flatten_trunk:
        ms_z = 0.0
        n = 0
        for s in (11, 12):  # Left/RightShoulder (JointId)
            if lm[s][3] > 0.0:
                ms_z += lm[s][2]
                n += 1
        ms_z = ms_z / n if n > 0 else 0.0
        for shoulder, chain in ((11, (13, 15, 17, 19, 21)), (12, (14, 16, 18, 20, 22))):  # each arm w/ its shoulder
            if lm[shoulder][3] > 0.0:
                sz = lm[shoulder][2]
                for j in chain:
                    if lm[j][3] > 0.0:
                        lm[j][2] -= sz  # re-base the arm to a Z=0 shoulder (segment vectors preserved)
                lm[shoulder][2] = 0.0
        for j in range(0, 11):  # head + face: keep above the flattened shoulders
            if lm[j][3] > 0.0:
                lm[j][2] -= ms_z
        for h in (23, 24):  # hips: flat hip line -> frontal-stable facing (no profile swing)
            if lm[h][3] > 0.0:
                lm[h][2] = 0.0
    return lm, src


def build_hand(uv, xyz_cam, measured, conf, zrel, zrel_hip, mid_hip, hip_z, intr, base, conf_thr,
               min_points=6, use_zrel=True):
    """21 hand landmarks (hip-relative metres), or None if the hand is not confidently visible.

    Gates on the wrist (landmark 0) AND a quorum of confident points. Without this gate every frame
    emitted a full hand regardless of confidence, so an occluded/absent hand streamed as "tracked" and
    the avatar fingers twitched from RTMW3D noise (audit H7). Returning None makes the Unity
    OakDUdpPoseProvider.ReadHand yield null → that hand is marked untracked and holds its rest pose."""
    wrist_conf = float(conf[base])
    n_conf = int(np.sum(conf[base:base + 21] >= conf_thr))
    if wrist_conf < conf_thr or n_conf < min_points:
        return None
    out = []
    for k in range(21):
        i = base + k
        if measured[i]:
            p = xyz_cam[i] - mid_hip
        else:
            p = _fallback_point(i, uv, zrel, zrel_hip, mid_hip, hip_z, intr, use_zrel)  # M16
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
    parser.add_argument("--min-cutoff", type=float, default=0.7, help="One-Euro min cutoff Hz — sets smoothness WHEN STILL (lower = smoother/steadier when you hold still, but adds lag to slow moves). Unity's jointFilter is bypassed for OAK (single-owner smoothing, ADR-020), so this is the ONLY smoothing stage. Range ~0.3 (very steady) .. 1.0 (snappier when still).")
    parser.add_argument("--beta", type=float, default=0.4, help="One-Euro beta — sets REACTION SPEED during motion (higher = less lag on fast moves). The old 0.02 felt sluggish on metric keypoints; 0.4 reacts quickly while --min-cutoff keeps stillness steady. Raise toward ~1.0 if it still feels laggy, lower if fast moves look jittery (ADR-020).")
    parser.add_argument("--max-jump", type=float, default=1.5, help="rate-limit a keypoint that jumps more than this many metres in one frame (only catches gross depth-spike garbage; keep it ABOVE the depth-quantization step so normal motion is untouched)")
    parser.add_argument("--depth-min-cutoff", type=float, default=0.3, help="LIMB depth (z) gets a HEAVIER One-Euro min-cutoff than the image plane (limb depth is ~5x noisier for small/distant hands). Lower = steadier depth, more lag. Arms+hands only.")
    parser.add_argument("--depth-beta", type=float, default=0.1, help="One-Euro beta for LIMB depth (low = depth reacts slowly, since it is the noisy axis). Raise if reaching toward/away the camera feels laggy.")
    parser.add_argument("--max-hold-frames", type=int, default=8, help="hold a LIMB keypoint's last-good value through up to N depth-dropout frames instead of the noisy zrel fallback (kills the 8-12 m spikes). 0 = off. Bounded so a genuinely-gone limb still drops.")
    parser.add_argument("--zrel-fallback", action=argparse.BooleanOptionalAction, default=True,
                        help="M16: for keypoints with NO measured depth (holes), use the model's root-relative z as the depth offset from the hip plane instead of flattening to the plane. --no-zrel-fallback restores the flat hip-plane. If occluded limbs poke the wrong way in depth, flip ZREL_SIGN in this file.")
    parser.add_argument("--seconds", type=float, default=0.0, help="auto-stop after N seconds (0 = run forever)")
    parser.add_argument("--log-dir", default="", help="pipeline logging: write sender_log.jsonl (seq + key landmarks per SENT frame) to this dir, to diff against Unity's recv_log.jsonl / model_log.jsonl via compare_logs.py. Empty = off.")
    args = parser.parse_args()

    log_f = None
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_f = open(os.path.join(args.log_dir, "sender_log.jsonl"), "w", buffering=1)
        print("[wb] pipeline logging -> %s" % os.path.join(args.log_dir, "sender_log.jsonl"))

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
            # LIMB depth-smoothing + hold-on-dropout target: arms (elbows 7/8, wrists 9/10) + both hands
            # (WholeBody 91-132). Legs are intentionally EXCLUDED so an occluded lower body still drops.
            limb_idx = set([7, 8, 9, 10]) | set(range(91, 133))
            body_smoother = smoothing.KeypointSmoother(
                133, min_cutoff=args.min_cutoff, beta=args.beta, max_jump=args.max_jump,
                depth_min_cutoff=args.depth_min_cutoff, depth_beta=args.depth_beta,
                limb_indices=limb_idx, max_hold=args.max_hold_frames)
        frames = 0            # monotonic total frame count (never reset)
        win_frames = 0        # LOW-D: separate windowed counter for the fps estimate (reset each log window)
        sent = 0
        last_mid_hip = None   # M11: hold last good mid-hip through transient hip depth-holes
        lowconf_streak = 0    # M15: force full-frame re-acquire if the person-box wedges
        t_log = time.time()
        t_start = time.time()
        while True:
            if args.seconds > 0.0 and (time.time() - t_start) > args.seconds:
                break
            frame = q_rgb.get().getCvFrame()
            depth = q_depth.get().getFrame()
            frames += 1
            win_frames += 1

            uv, zrel, conf = model.infer(frame, bbox)
            # Person-box tracking WITH recovery (M15): follow the person from the previous frame's
            # confident keypoints; if detection is lost OR the box wedges (body confidence stays low for a
            # sustained run, e.g. it locked onto a false detection), re-acquire from the full-frame centre.
            refined = R.bbox_from_keypoints(uv, conf, rgb_w, rgb_h, thr=args.conf)
            body_conf_mean = float(np.mean(conf[0:17]))
            if refined is not None and body_conf_mean >= args.conf:
                lowconf_streak = 0
                bbox = tuple(0.7 * np.array(bbox) + 0.3 * np.array(refined))
            else:
                lowconf_streak = lowconf_streak + 1
                if refined is None or lowconf_streak >= 20:
                    bbox = R.center_bbox(rgb_w, rgb_h)
                    lowconf_streak = 0
                else:
                    bbox = tuple(0.7 * np.array(bbox) + 0.3 * np.array(refined))

            xyz_cam, measured = D.backproject(uv, depth, rgb_w, rgb_h, intr, k=args.kwin)

            # Smooth the metric keypoints at the source (One-Euro + depth-outlier gate) to kill jitter,
            # before hip-centring / building the message. Unmeasured points reset their filter.
            if body_smoother is not None:
                si = 0
                sn = xyz_cam.shape[0]
                while si < sn:
                    sx, sy, sz, eff = body_smoother.filter(si, float(xyz_cam[si, 0]), float(xyz_cam[si, 1]), float(xyz_cam[si, 2]), bool(measured[si]))
                    xyz_cam[si, 0] = sx
                    xyz_cam[si, 1] = sy
                    xyz_cam[si, 2] = sz
                    measured[si] = eff  # hold-on-dropout can report a held limb as measured -> build uses it, not the fallback
                    si = si + 1

            # Mid-hip origin (M11): both hips → midpoint; one hip → that hip; neither but a recent mid-hip
            # exists → HOLD it through the transient depth-hole (avoids a whole-body stall); else skip.
            if measured[11] and measured[12]:
                mid_hip = (xyz_cam[11] + xyz_cam[12]) / 2.0
                last_mid_hip = mid_hip
            elif measured[11]:
                mid_hip = xyz_cam[11].copy()
                last_mid_hip = mid_hip
            elif measured[12]:
                mid_hip = xyz_cam[12].copy()
                last_mid_hip = mid_hip
            elif last_mid_hip is not None:
                mid_hip = last_mid_hip
            else:
                if args.show:
                    _preview(frame, uv, conf, args.conf, frames, sent, "no hip depth")
                    if cv2.waitKey(1) in (27, ord("q")):
                        break
                continue
            hip_z = float(mid_hip[2])
            # M16: the hips' own root-relative z, so a hole's zrel is offset relative to the hips (origin).
            zrel_hip = float((zrel[11] + zrel[12]) / 2.0)

            lm, src = build_body_landmarks(uv, xyz_cam, measured, conf, zrel, zrel_hip, mid_hip, hip_z,
                                           intr, args.conf, args.flatten_trunk, args.zrel_fallback)
            lm = [[round(v, 4) for v in p] for p in lm]
            lh = build_hand(uv, xyz_cam, measured, conf, zrel, zrel_hip, mid_hip, hip_z, intr, 91,
                            args.conf, use_zrel=args.zrel_fallback)
            rh = build_hand(uv, xyz_cam, measured, conf, zrel, zrel_hip, mid_hip, hip_z, intr, 112,
                            args.conf, use_zrel=args.zrel_fallback)
            message = {
                "lm": lm,
                "xyz": [round(float(mid_hip[0] * 1000.0), 1),
                        round(float(mid_hip[1] * 1000.0), 1),
                        round(float(mid_hip[2] * 1000.0), 1)],
                "src": src,
                "seq": frames,                 # monotonic frame id (aligns sender/recv/model logs)
                "t": round(time.time(), 4),    # send epoch seconds
            }
            # H7: only include a hand when confidently tracked. Unity treats a missing lh/rh as
            # "not tracked" and holds the rest pose instead of curling fingers from noise.
            if lh is not None:
                message["lh"] = lh
            if rh is not None:
                message["rh"] = rh
            # Mirror = negate X AND swap left<->right (DEFAULT OFF — reflecting the input twists a
            # rotation retarget, see audit). Guarded for optional hands.
            if args.mirror:
                lm_m = message["lm"]
                src_m = message["src"]
                for p in lm_m:
                    p[0] = -p[0]
                message["xyz"][0] = -message["xyz"][0]
                for a, b in MIRROR_PAIRS:
                    lm_m[a], lm_m[b] = lm_m[b], lm_m[a]
                    src_m[a], src_m[b] = src_m[b], src_m[a]
                if "lh" in message:
                    for p in message["lh"]:
                        p[0] = -p[0]
                if "rh" in message:
                    for p in message["rh"]:
                        p[0] = -p[0]
                if "lh" in message and "rh" in message:
                    message["lh"], message["rh"] = message["rh"], message["lh"]
                elif "lh" in message:
                    message["rh"] = message.pop("lh")
                elif "rh" in message:
                    message["lh"] = message.pop("rh")
            sock.sendto(json.dumps(message).encode("utf-8"), addr)
            sent += 1

            if log_f is not None:
                # Key signals for pipeline diffing: shoulders(11,12), hips(23,24), wrists(15,16) xyz, plus the
                # palm-basis hand points (0 wrist, 9 middle-MCP) so a wrist-spin can be traced to its source.
                rec = {"seq": frames, "t": round(time.time(), 4),
                       "sh": [lm[11][:3], lm[12][:3]], "el": [lm[13][:3], lm[14][:3]],
                       "hip": [lm[23][:3], lm[24][:3]], "wr": [lm[15][:3], lm[16][:3]],
                       "hipZ": round(hip_z, 3), "cov": int(sum(src))}  # distance (m) + measured-coverage (0-33)
                if lh is not None:
                    rec["lh"] = [lh[0], lh[9]]
                if rh is not None:
                    rec["rh"] = [rh[0], rh[9]]
                log_f.write(json.dumps(rec) + "\n")

            if time.time() - t_log > 2.0:
                cov = int(sum(src))
                # LOW-D: fps uses the windowed counter; `frames` stays a monotonic total.
                print("[wb] frames=%d sent=%d hip_z=%.2fm measured_body=%d/33 fps~%.1f"
                      % (frames, sent, hip_z, cov, win_frames / (time.time() - t_log + 1e-6)))
                win_frames = 0
                t_log = time.time()

            if args.show:
                _preview(frame, uv, conf, args.conf, frames, sent, "hip %.2fm" % hip_z)
                if cv2.waitKey(1) in (27, ord("q")):
                    break

    sock.close()
    if log_f is not None:
        log_f.close()
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
