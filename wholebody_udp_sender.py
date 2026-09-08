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
import joint_tracker as JT   # P1-1 per-joint temporal tracking + plausibility
import kinematic_recovery as KR   # P1-4 skeleton constraints + long-horizon recovery

NUM_BODY = 33

# P0-2 diagnostics: WholeBody index -> readable name for the tracked body joints we log spike/hold events on.
JOINT_NAMES = {
    5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle",
}

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
    parser.add_argument("--flatten-trunk", action=argparse.BooleanOptionalAction, default=False,
                        help="DEFAULT OFF (Milestone-2): keep the measured trunk Z so the model can BEND at the waist and TURN like the skeleton. The trunk joints are now depth-smoothed (see limb_idx) so this is stable at ~2 m without the old profile swing. Pass --flatten-trunk to restore the frontal-locked fallback (zeroes shoulders+hips Z) if a noisy/farther setup swings into profile.")
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=False,
                        help="DEFAULT OFF. Mirroring the INPUT skeleton (negate X + swap sides) reflects the pose, but the FK retarget builds rotations with LookRotation and a reflected skeleton twists the torso/limbs. Leave off (clean 'copy' retarget); do the mirror on the Unity/avatar side instead. --mirror to experiment.")
    parser.add_argument("--smooth", action=argparse.BooleanOptionalAction, default=True,
                        help="temporal One-Euro smoothing + depth-outlier gate to kill jitter. --no-smooth to disable.")
    parser.add_argument("--min-cutoff", type=float, default=0.5, help="One-Euro min cutoff Hz — sets smoothness WHEN STILL (lower = smoother/steadier when you hold still, but adds lag to slow moves). Unity's jointFilter is bypassed for OAK (single-owner smoothing, ADR-020), so this is the ONLY smoothing stage. Range ~0.3 (very steady) .. 1.0 (snappier when still).")
    parser.add_argument("--beta", type=float, default=0.4, help="One-Euro beta — sets REACTION SPEED during motion (higher = less lag on fast moves). The old 0.02 felt sluggish on metric keypoints; 0.4 reacts quickly while --min-cutoff keeps stillness steady. Raise toward ~1.0 if it still feels laggy, lower if fast moves look jittery (ADR-020).")
    parser.add_argument("--max-jump", type=float, default=1.5, help="GLOBAL rate-limit (m/frame) for trunk + hands. Kept generous; the distal-limb caps below are tighter (P0-2).")
    parser.add_argument("--arm-max-jump", type=float, default=0.35, help="P0-2: tighter per-frame displacement cap (m) for elbows+wrists (WB 7,8,9,10). Below the observed ~0.8 m spikes, above the depth-quantization step. Rate-limited (slewed), not dropped, so genuine fast motion catches up in 1-2 frames.")
    parser.add_argument("--leg-max-jump", type=float, default=0.35, help="P0-2: tighter per-frame displacement cap (m) for knees+ankles (WB 13,14,15,16). Same rationale as --arm-max-jump; legs are now also depth-smoothed + hold-protected.")
    parser.add_argument("--depth-min-cutoff", type=float, default=0.3, help="LIMB depth (z) gets a HEAVIER One-Euro min-cutoff than the image plane (limb depth is ~5x noisier for small/distant hands). Lower = steadier depth, more lag. Arms+hands only.")
    parser.add_argument("--depth-beta", type=float, default=0.1, help="One-Euro beta for LIMB depth (low = depth reacts slowly, since it is the noisy axis). Raise if reaching toward/away the camera feels laggy.")
    parser.add_argument("--max-hold-frames", type=int, default=8, help="hold a LIMB keypoint's last-good value through up to N depth-dropout frames instead of the noisy zrel fallback (kills the 8-12 m spikes). 0 = off. Bounded so a genuinely-gone limb still drops.")
    parser.add_argument("--zrel-fallback", action=argparse.BooleanOptionalAction, default=True,
                        help="M16: for keypoints with NO measured depth (holes), use the model's root-relative z as the depth offset from the hip plane instead of flattening to the plane. --no-zrel-fallback restores the flat hip-plane. If occluded limbs poke the wrong way in depth, flip ZREL_SIGN in this file.")
    parser.add_argument("--seconds", type=float, default=0.0, help="auto-stop after N seconds (0 = run forever)")
    # ---- P1-1 ----
    # ---- P1-2 frame freshness ----
    parser.add_argument("--latest-frame", dest="latest_frame", action="store_true", default=True,
                        help="P1-2 (default ON): drain the RGB/depth queues to the NEWEST frame each "
                             "iteration instead of consuming them FIFO. q.get() returns the OLDEST "
                             "frame, so when inference is slower than the sensor the host queue "
                             "(maxSize=4) stays full and every pose is ~4 frames (~133 ms) stale.")
    parser.add_argument("--no-latest-frame", dest="latest_frame", action="store_false",
                        help="P1-2 OFF: original FIFO behaviour (for A/B).")
    parser.add_argument("--inject-drift-file", default="",
                        help="TEST-ONLY (P1-4 visual validation). Path polled once per frame; when the file "
                        "appears it is read as JSON {mode:drift|teleport, joint:N, meters:F, frames:N} and "
                        "DELETED, then that joint is corrupted for N frames at HIGH confidence upstream of "
                        "P1-1/P1-4. Never set this in production; it deliberately falsifies measurements.")
    parser.add_argument("--inject-load-ms", type=float, default=0.0,
                        help="TEST-ONLY (P1-2 diagnostics): add N ms of artificial work per frame to "
                             "reproduce the consumer-slower-than-sensor condition a tracked subject "
                             "causes (~21 fps), so the queue policy can be proven without a human. "
                             "0 = off. Never set this in production.")
    parser.add_argument("--tracker", dest="tracker", action="store_true", default=True,
                        help="P1-1 per-joint temporal tracking + plausibility (default ON). Catches "
                             "CONFIDENT-BUT-WRONG landmarks that the confidence gate cannot see.")
    parser.add_argument("--no-tracker", dest="tracker", action="store_false",
                        help="disable P1-1 (A/B against P0-only).")
    parser.add_argument("--tracker-predict-frames", type=int, default=6,
                        help="P1-1 max PREDICTED frames before a joint goes LOST (2-6 typical).")
    parser.add_argument("--tracker-reacquire-frames", type=int, default=5,
                        help="P1-1 frames to blend predicted -> measured on reacquisition (never teleport).")
    # ---- P1-4: REJECTED / EXPERIMENTAL -- NOT FOR SHIPPING (see docs/P1_4_CLOSEOUT_2026-09-08.md)
    # DEFAULT OFF. Live human validation on 2026-09-08 showed P1-4 MISSED the controlled
    # 0.85 m knee drift/teleport (GEOMETRIC_REJECT=0, RECONSTRUCT=0) while raising P0 limb
    # holds 0.47% -> 17.83% and LOST episodes 1 -> 122. The bone-length signal it rejects on
    # overlaps the natural noise floor of this pipeline, so no threshold separates them.
    # Retained for forensic/research use ONLY; do not enable in production.
    parser.add_argument("--recovery", dest="recovery", action="store_true", default=False,
                        help="RESEARCH ONLY -- P1-4 is REJECTED (default OFF). Skeleton-level "
                             "constraints + long-horizon recovery. It did NOT catch the controlled "
                             "corruption on hardware and it degraded the avatar; see "
                             "docs/P1_4_CLOSEOUT_2026-09-08.md before enabling.")
    parser.add_argument("--no-recovery", dest="recovery", action="store_false",
                        help="explicitly disable P1-4 (already the default).")
    parser.add_argument("--recovery-max-frames", type=int, default=30,
                        help="RESEARCH ONLY (P1-4 rejected): max frames a joint may be "
                             "geometrically reconstructed before it goes LOST.")
    parser.add_argument("--recovery-blend-frames", type=int, default=8,
                        help="RESEARCH ONLY (P1-4 rejected): frames to blend a recovered joint back.")
    parser.add_argument("--log-dir", default="", help="pipeline logging: write sender_log.jsonl (seq + key landmarks per SENT frame) to this dir, to diff against Unity's recv_log.jsonl / model_log.jsonl via compare_logs.py. Empty = off.")
    args = parser.parse_args()

    log_f = None
    holds_f = None
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_f = open(os.path.join(args.log_dir, "sender_log.jsonl"), "w", buffering=1)
        # P0-2: separate stream of rejected/held joint events (spike rate-limits + depth dropouts).
        holds_f = open(os.path.join(args.log_dir, "holds_log.jsonl"), "w", buffering=1)
        print("[wb] pipeline logging -> %s (+ holds_log.jsonl)" % os.path.join(args.log_dir, "sender_log.jsonl"))

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
            # Depth-smoothing + hold-on-dropout target: TRUNK (shoulders 5/6, hips 11/12) + arms (elbows 7/8,
            # wrists 9/10) + LEGS (knees 13/14, ankles 15/16) + both hands (WholeBody 91-132). P0-2 (audit
            # F-04): legs were previously EXCLUDED, so a bad/occluded knee got only the light image-plane
            # One-Euro (no heavy depth cutoff, no hold) → leg jitter/collapse. They now get the SAME depth
            # smoothing + bounded hold as the arms. The hold is bounded (max_hold), so a genuinely-gone
            # lower body still drops after N frames rather than freezing.
            limb_idx = set([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]) | set(range(91, 133))
            # P0-2: per-index displacement caps. Trunk (5,6,11,12) + hands keep the global --max-jump; the
            # spike-prone distal joints (wrist/elbow/knee/ankle) get tighter caps that sit BELOW the observed
            # ~0.8 m spikes but ABOVE the depth-quantization step, so garbage is slewed out while genuine fast
            # motion still catches up within a frame or two (rate-limited, never frozen).
            limb_max_jump = {}
            for _idx in (7, 8, 9, 10):
                limb_max_jump[_idx] = args.arm_max_jump
            for _idx in (13, 14, 15, 16):
                limb_max_jump[_idx] = args.leg_max_jump
            body_smoother = smoothing.KeypointSmoother(
                133, min_cutoff=args.min_cutoff, beta=args.beta, max_jump=args.max_jump,
                depth_min_cutoff=args.depth_min_cutoff, depth_beta=args.depth_beta,
                limb_indices=limb_idx, max_hold=args.max_hold_frames,
                max_jump_overrides=limb_max_jump)
        # P1-1: one reusable tracker per tracked joint. Runs AFTER the P0 smoother, so P0-2's
        # cap/hold behaviour is untouched; P1 only adds temporal state + plausibility on top.
        skel = None
        if args.tracker:
            skel = JT.SkeletonTracker(cfg=JT.TrackerConfig(
                max_predict_frames=args.tracker_predict_frames,
                reacquire_frames=args.tracker_reacquire_frames))
            print("[wb] P1-1 joint tracker ON (predict<=%d, reacquire=%d)"
                  % (args.tracker_predict_frames, args.tracker_reacquire_frames))
        recovery = None
        if args.tracker and args.recovery:
            # P1-4 sits AFTER P1-1 and consumes its output. It is meaningless without the
            # tracker, so it is gated on it.
            recovery = KR.KinematicRecovery(KR.RecoveryConfig(
                max_reconstruct_frames=args.recovery_max_frames,
                recover_frames=args.recovery_blend_frames))
            print("[wb] *** WARNING: P1-4 kinematic recovery ENABLED (reconstruct<=%d, blend=%d)"
                  % (args.recovery_max_frames, args.recovery_blend_frames))
            print("[wb] *** P1-4 is REJECTED for production -- research use only.")
            print("[wb] *** See docs/P1_4_CLOSEOUT_2026-09-08.md")
        stale_rgb = 0         # P1-2: RGB frames discarded as stale (never inferred on)
        stale_depth = 0       # P1-2: depth frames discarded as stale
        frames = 0            # monotonic total frame count (never reset)
        inject = None         # TEST-ONLY --inject-drift-file: active injection, or None
        win_frames = 0        # LOW-D: separate windowed counter for the fps estimate (reset each log window)
        sent = 0
        last_mid_hip = None   # M11: hold last good mid-hip through transient hip depth-holes
        lowconf_streak = 0    # M15: force full-frame re-acquire if the person-box wedges
        t_log = time.time()
        t_start = time.time()
        while True:
            if args.seconds > 0.0 and (time.time() - t_start) > args.seconds:
                break
            # DIAG-ONLY (P0 acceptance S16): stage timestamps so camera->pose->depth->send is MEASURED,
            # not estimated. `getTimestamp()` is the OAK device clock (synchronised to the host by
            # depthai), so `dai.Clock.now() - ts` is the true sensor->host latency. Read-only.
            # ---- P1-2 FRAME FRESHNESS -------------------------------------------
            # q.get() hands back the OLDEST queued packet. With inference (~21 ms) slower
            # than the 30 fps sensor the host queue (maxSize=4) stays full, so FIFO makes
            # every pose ~4 frames stale -- the measured ~131 ms camera->host latency.
            # Fix: take one blocking packet (guarantees liveness), then drain whatever else
            # has arrived and keep only the NEWEST. Stale intermediates are discarded.
            _rgb_pkt = q_rgb.get()
            _rgb_dropped = 0
            _depth_dropped = 0
            _depth_pkt = q_depth.get()
            if args.latest_frame:
                _extra = q_rgb.tryGetAll()
                if _extra:
                    _rgb_dropped = len(_extra)
                    _rgb_pkt = _extra[-1]
                _dextra = q_depth.tryGetAll()
                _depth_cands = [_depth_pkt] + list(_dextra)
                _depth_dropped = len(_depth_cands) - 1
                # RGB/depth pairing: pick the depth packet CLOSEST IN TIMESTAMP to the
                # selected RGB frame. Taking newest-RGB + oldest-depth (or vice versa)
                # would mis-associate depth, which the brief explicitly forbids.
                try:
                    _rts = _rgb_pkt.getTimestamp()
                    _depth_pkt = min(_depth_cands,
                                     key=lambda _d: abs((_d.getTimestamp() - _rts).total_seconds()))
                except Exception:
                    _depth_pkt = _depth_cands[-1]
            t_cap = time.time()
            try:
                _cam_lat_ms = (dai.Clock.now() - _rgb_pkt.getTimestamp()).total_seconds() * 1000.0
            except Exception:
                _cam_lat_ms = -1.0
            try:
                _sync_ms = abs((_rgb_pkt.getTimestamp() - _depth_pkt.getTimestamp()).total_seconds()) * 1000.0
            except Exception:
                _sync_ms = -1.0
            frame = _rgb_pkt.getCvFrame()
            depth = _depth_pkt.getFrame()
            t_depth_ready = time.time()
            stale_rgb += _rgb_dropped
            stale_depth += _depth_dropped
            frames += 1
            win_frames += 1

            uv, zrel, conf = model.infer(frame, bbox)
            if args.inject_load_ms > 0.0:      # TEST-ONLY, see --inject-load-ms
                _busy_until = time.perf_counter() + args.inject_load_ms / 1000.0
                while time.perf_counter() < _busy_until:
                    pass
            t_pose = time.time()
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
            t_backproj = time.time()   # DIAG-ONLY (S16)

            # Smooth the metric keypoints at the source (One-Euro + depth-outlier gate) to kill jitter,
            # before hip-centring / building the message. Unmeasured points reset their filter.
            if body_smoother is not None:
                si = 0
                sn = xyz_cam.shape[0]
                while si < sn:
                    raw_measured = bool(measured[si])
                    sx, sy, sz, eff, action, disp = body_smoother.filter(
                        si, float(xyz_cam[si, 0]), float(xyz_cam[si, 1]), float(xyz_cam[si, 2]), raw_measured)
                    xyz_cam[si, 0] = sx
                    xyz_cam[si, 1] = sy
                    xyz_cam[si, 2] = sz
                    measured[si] = eff  # hold-on-dropout can report a held limb as measured -> build uses it, not the fallback
                    # P0-2 diagnostics: log only the interesting events (spike rate-limited, dropout held/dropped)
                    # on the tracked body joints — never per-frame for healthy joints. Opt-in via --log-dir.
                    if holds_f is not None and action != 'ACCEPT' and si in JOINT_NAMES:
                        holds_f.write(json.dumps({
                            "seq": frames, "t": round(time.time(), 4), "joint": JOINT_NAMES[si],
                            "confidence": round(float(conf[si]), 3), "displacement": round(disp, 4),
                            "depthValid": raw_measured, "action": action,
                            "reason": ("DISPLACEMENT_OUTLIER" if action == 'RATE_LIMIT'
                                       else "DEPTH_DROPOUT_HOLD" if action == 'HOLD' else "MEASUREMENT_INVALID")}) + "\n")
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

            # ---- TEST-ONLY landmark injection (--inject-drift-file) -----------------
            # Falsifies ONE joint at high confidence BEFORE P1-1/P1-4 see it, so a live
            # run can exercise the "confidently wrong" path on demand instead of waiting
            # for nature to produce one. Off unless the flag is set AND the file appears.
            if args.inject_drift_file:
                if inject is None and os.path.exists(args.inject_drift_file):
                    try:
                        with open(args.inject_drift_file) as _f:
                            _spec = json.load(_f)
                        os.remove(args.inject_drift_file)
                        inject = {"mode": _spec.get("mode", "drift"),
                                  "joint": int(_spec.get("joint", 14)),
                                  "meters": float(_spec.get("meters", 0.85)),
                                  "frames": int(_spec.get("frames", 25)),
                                  "i": 0, "seq0": frames}
                        print("[INJECT] %s joint=%d %.2fm over %d frames (seq %d)"
                              % (inject["mode"], inject["joint"], inject["meters"],
                                 inject["frames"], frames))
                    except Exception as _e:
                        print("[INJECT] bad spec: %s" % _e)
                        inject = None
                if inject is not None:
                    _j = inject["joint"]
                    _i = inject["i"]
                    if _i >= inject["frames"]:
                        print("[INJECT] end (seq %d)" % frames)
                        inject = None
                    else:
                        # drift ramps linearly (P1-1 tracks it, only geometry catches it);
                        # teleport applies the full offset at once.
                        _f = (float(_i + 1) / inject["frames"]) if inject["mode"] == "drift" else 1.0
                        xyz_cam[_j, 0] += inject["meters"] * _f
                        conf[_j] = max(float(conf[_j]), 0.85)   # CONFIDENTLY wrong
                        measured[_j] = True
                        inject["i"] = _i + 1

            # ---- P1-1 tracker -------------------------------------------------------
            # Consumes the P0-smoothed metric keypoints and returns temporally-validated
            # ones. A LOST joint has its EMIT confidence zeroed so build_body_landmarks
            # drops it -- which hands the decision to the P0 LimbGate in Unity exactly as
            # a real occlusion would. The tracker never writes zeros itself.
            conf_emit = conf
            if skel is not None:
                t_track0 = time.perf_counter()
                pos_in, cnf_in, dv_in = {}, {}, {}
                for _j in skel.indices:
                    pos_in[_j] = (float(xyz_cam[_j, 0]), float(xyz_cam[_j, 1]), float(xyz_cam[_j, 2]))
                    cnf_in[_j] = float(conf[_j]) if bool(measured[_j]) else 0.0
                    dv_in[_j] = bool(measured[_j])
                res = skel.update(pos_in, cnf_in, time.time(), depth_valid=dv_in,
                                  collect_events=(holds_f is not None))
                # ---- P1-4: skeleton constraints + long-horizon recovery -----------------
                # Consumes P1-1's output and repairs joints that are geometrically wrong or
                # have outlived P1-1's prediction horizon. Healthy joints pass through
                # unchanged. Result keeps P1-1's tuple shape plus an observation code.
                if recovery is not None:
                    rec_out = recovery.apply(res, pos_in,
                                             collect_events=(holds_f is not None))
                    res = dict((k, v[:6]) for k, v in rec_out.items())
                conf_emit = conf.copy()
                for _j, (_x, _y, _z, _c, _st, _usable) in res.items():
                    if _usable:
                        xyz_cam[_j, 0] = _x
                        xyz_cam[_j, 1] = _y
                        xyz_cam[_j, 2] = _z
                        measured[_j] = True
                    else:
                        measured[_j] = False
                        conf_emit[_j] = 0.0      # LOST -> drop -> P0 LimbGate holds
                track_ms = (time.perf_counter() - t_track0) * 1000.0
                if holds_f is not None and skel.events:
                    for _e in skel.events:
                        _e["seq"] = frames
                        holds_f.write(json.dumps(_e) + chr(10))
                    del skel.events[:]
                if holds_f is not None and recovery is not None:
                    if recovery.events:
                        for _e in recovery.events:
                            _e["seq"] = frames
                            _e["stage"] = "P1-4"
                            holds_f.write(json.dumps(_e) + chr(10))
                        del recovery.events[:]
                    # periodic aggregate only -- never per frame (P1-4 Part 15)
                    if frames % 300 == 0:
                        _t = recovery.telemetry()
                        _t["seq"] = frames
                        _t["event"] = "P1-4_TELEMETRY"
                        holds_f.write(json.dumps(_t) + chr(10))
            else:
                track_ms = 0.0

            lm, src = build_body_landmarks(uv, xyz_cam, measured, conf_emit, zrel, zrel_hip, mid_hip, hip_z,
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
                       # DIAG-ONLY (S16) per-stage millisecond costs + true sensor->host latency.
                       "camLatMs": round(_cam_lat_ms, 2),
                       "capToPoseMs": round((t_pose - t_cap) * 1000.0, 2),
                       "poseToDepthMs": round((t_backproj - t_pose) * 1000.0, 2),
                       "depthWaitMs": round((t_depth_ready - t_cap) * 1000.0, 2),
                       "capToSendMs": round((time.time() - t_cap) * 1000.0, 2),
                       "trackerMs": round(track_ms, 3),   # P1-1 cost, measured not estimated
                       # P1-2 freshness diagnostics (permanent):
                       "frameAgeMs": round(_cam_lat_ms, 2),      # host processing - camera timestamp
                       "queueDepth": _rgb_dropped + 1,           # packets waiting when we sampled
                       "staleDropped": _rgb_dropped,             # discarded this iteration
                       "rgbDepthSyncMs": round(_sync_ms, 2),     # RGB/depth pairing error
                       "sh": [lm[11][:3], lm[12][:3]], "el": [lm[13][:3], lm[14][:3]],
                       "hip": [lm[23][:3], lm[24][:3]], "wr": [lm[15][:3], lm[16][:3]],
                       # P1-2 Phase 6: knees/ankles were never in sender_log, so the leg jitter
                       # regression could not be measured sidecar-side.
                       "kn": [lm[25][:3], lm[26][:3]], "an": [lm[27][:3], lm[28][:3]],
                       "hipZ": round(hip_z, 3), "cov": int(sum(src))}  # distance (m) + measured-coverage (0-33)
                if lh is not None:
                    rec["lh"] = [lh[0], lh[9]]
                if rh is not None:
                    rec["rh"] = [rh[0], rh[9]]
                log_f.write(json.dumps(rec) + "\n")

            if time.time() - t_log > 2.0:
                cov = int(sum(src))
                # LOW-D: fps uses the windowed counter; `frames` stays a monotonic total.
                print("[wb] frames=%d sent=%d hip_z=%.2fm measured_body=%d/33 fps~%.1f age=%.0fms stale=%d"
                      % (frames, sent, hip_z, cov, win_frames / (time.time() - t_log + 1e-6),
                         _cam_lat_ms, stale_rgb))
                win_frames = 0
                t_log = time.time()

            if args.show:
                _preview(frame, uv, conf, args.conf, frames, sent, "hip %.2fm" % hip_z)
                if cv2.waitKey(1) in (27, ord("q")):
                    break

    sock.close()
    if log_f is not None:
        log_f.close()
    if holds_f is not None:
        holds_f.close()
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
