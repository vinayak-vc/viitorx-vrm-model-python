#!/usr/bin/env python3

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(_os.path.join(_d, "_sidecar_path.py")):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import _sidecar_path  # noqa: F401  - puts the sidecar root and every tools/ group on sys.path
"""Drive the REAL Unity avatar from an ORDINARY VIDEO FILE, to see what the retarget does with
motion the OAK-D envelope has never been asked for.

READ THIS BEFORE QUOTING ANY RESULT FROM IT.

WHAT IS REAL:
  * RTMW3D, the production model, at the production confidence gate
  * R.center_bbox / R.bbox_from_keypoints - the production M15 crop loop
  * build_body_landmarks() - the PRODUCTION landmark builder, unmodified, called through its own
    documented depth-hole path
  * the wire payload and UDP port - the production contract, byte-identical in shape
  * everything downstream: P1-3 buffer, TrunkGate, V5 torso, V6 guard, Arm V2, VRM retarget

WHAT IS NOT REAL, and this is the whole caveat:
  * THERE IS NO STEREO DEPTH. A webm has one camera. Every joint is therefore an UNMEASURED
    keypoint, placed by _fallback_point() from RTMW3D's own monocular root-relative z (M16's
    use_zrel path) against an assumed hip plane. `measured` is False for all 133 keypoints, so
    every `src` flag on the wire reads 0 - the payload declares its own provenance honestly and
    Unity can see it is not looking at stereo.
  * Consequently this exercises the RETARGET AND THE AVATAR. It does NOT exercise the torso-yaw
    signal, which F-16/F-17/F-18 measured as the hard blocker and which is built from shoulder
    Delta-z. An avatar that looks right here says nothing about whether the SENSOR can resolve
    torso yaw at a usable distance - that question is settled separately, and negatively outside
    0.78-1.00 m in portrait.
  * The intrinsics of whatever phone shot the clip are unknown, so they are ESTIMATED (see --hfov)
    and the hip plane is chosen to make the subject a plausible human height rather than measured.
    Both only scale the pose; neither invents articulation.

So: a fair test of "does the avatar follow a human", and not a test of the product's measured
blockers. Useful precisely because dancing is FAR outside the supported envelope (full-body relaxed,
arms-45, normal movement, single user - T-pose, crouch, hands-near-face and reliable +/-90 deg torso
are all outside it), so it shows where the retarget degrades under motion nobody has yet fed it.

    .venv\\Scripts\\python.exe f23_video_to_unity.py --video ..\\video\\video.webm
"""
import argparse
import json
import math
import os
import socket
import sys
import time

import cv2
import numpy as np

import rtmw3d_pose as R
import smoothing
import wholebody_udp_sender as W

# F-20A treats a new sid as a new producer session; the real sender always stamps one.
SID = "vid%08x" % (int(time.time()) & 0xFFFFFFFF)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "..", "..", "..", "SentisModel", "rtmw3d-x.onnx")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--hfov", type=float, default=60.0,
                    help="assumed horizontal FOV of the source camera, degrees. Unknown for an "
                         "arbitrary clip; only scales the pose.")
    ap.add_argument("--subject-height", type=float, default=1.70,
                    help="assumed subject height (m). The hip plane is chosen so the observed "
                         "head-to-ankle pixel span corresponds to this - again, scale only.")
    ap.add_argument("--loop", type=int, default=1, help="replay the clip N times")
    ap.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True,
                    help="pace to the clip's own frame rate (default). --no-realtime runs flat out.")
    ap.add_argument("--dump", default="", help="also write each frame's payload to this jsonl")
    ap.add_argument("--mark", default="", help="write the wall clock of the FIRST packet here")
    ap.add_argument("--smooth-zrel", action=argparse.BooleanOptionalAction, default=True,
                    help="One-Euro filter the model's per-joint depth, with production's own trunk "
                         "parameters. Production smooths xyz_cam through KeypointSmoother BEFORE "
                         "build_body_landmarks; on this path there is no stereo to smooth, so the "
                         "raw per-frame zrel went straight to the retarget and swung the avatar "
                         "into profile. Milestone-2 turned --flatten-trunk off precisely BECAUSE "
                         "the trunk is depth-smoothed; without this the comparison is unfair.")
    ap.add_argument("--sway-gain", type=float, default=0.45,
                    help="scale applied to the ROOT's lateral/vertical translation before sending. "
                         "1.0 is physically faithful (the avatar moves exactly as far as the "
                         "person) but at the mirror camera's framing a real dancer's +/-0.20 m of "
                         "hip sway walks the avatar out of shot. Depth (Z) is never scaled - it "
                         "sets the avatar's size and must stay metric.")
    ap.add_argument("--flatten-trunk", action=argparse.BooleanOptionalAction, default=False,
                    help="mirror production's own default (OFF since Milestone-2). ON zeroes "
                         "shoulder and hip Z, which frontal-locks the avatar: no waist bend, no "
                         "pelvis or torso rotation. Only for a noisy setup that swings to profile.")
    a = ap.parse_args()

    import onnxruntime as ort
    eps = ort.get_available_providers()
    print("interpreter : %s" % sys.executable)
    print("providers   : %s" % ", ".join(eps))
    if "DmlExecutionProvider" not in eps:
        print("  !! no DirectML - inference will fall back to CPU (~5 fps) and the avatar will")
        print("     stutter for reasons that have nothing to do with the retarget.")

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        print("cannot open %s" % a.video)
        return 2
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fx = fy = 0.5 * w / math.tan(math.radians(a.hfov) * 0.5)
    intr = (fx, fy, w * 0.5, h * 0.5)
    print("video       : %dx%d @ %.2f fps   assumed fx=fy=%.1f (hfov %.0f deg)"
          % (w, h, src_fps, fx, a.hfov))

    print("loading RTMW3D ...")
    model = R.RTMW3D(a.model)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dump = open(a.dump, "w", encoding="utf-8") if a.dump else None

    # No stereo anywhere: every keypoint takes the M16 zrel fallback path.
    measured = np.zeros(133, dtype=bool)
    xyz_cam = np.zeros((133, 3), dtype=np.float32)
    # Production's TRUNK smoothing parameters (--min-cutoff 0.5 / --beta 0.4); the separate
    # depth-min-cutoff/depth-beta pair is documented as arms+hands only.
    zfilt = [smoothing.OneEuro(freq=src_fps, min_cutoff=0.5, beta=0.4) for _ in range(133)]

    seq = int((time.time() - 1788900000.0) * 100.0)
    sent = skipped = 0
    t_start = time.time()
    frame_i = 0

    for _ in range(max(1, a.loop)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        bbox = R.center_bbox(w, h)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_frame = time.time()
            uv, zrel, conf = model.infer(frame, bbox)

            # production M15 crop refinement, same blend as the sender
            refined = R.bbox_from_keypoints(uv, conf, w, h, thr=a.conf)
            body_conf = float(np.mean([conf[i] for i in R.COCO17_TO_JOINTID]))
            if refined is not None and body_conf >= a.conf:
                bbox = tuple(0.7 * np.array(bbox) + 0.3 * np.array(refined))
            else:
                bbox = R.center_bbox(w, h)

            # hip plane from the apparent body span, so the avatar is a plausible size
            ys = [uv[i, 1] for i in (0, 15, 16) if conf[i] >= a.conf]
            span_px = (max(ys) - min(ys)) if len(ys) >= 2 else 0.0
            hip_z = (a.subject_height * fy / span_px) if span_px > 40 else 2.0
            hip_z = min(max(hip_z, 0.6), 6.0)

            if conf[11] < a.conf or conf[12] < a.conf:
                skipped += 1
                frame_i += 1
                continue
            if a.smooth_zrel:
                zrel = np.array([zfilt[i].filter(float(zrel[i])) for i in range(len(zrel))],
                                dtype=np.float32)
            zrel_hip = 0.5 * (float(zrel[11]) + float(zrel[12]))
            hip_uv = 0.5 * (uv[11] + uv[12])
            mid_hip = np.array([(hip_uv[0] - intr[2]) * hip_z / fx,
                                (hip_uv[1] - intr[3]) * hip_z / fy, hip_z], dtype=np.float32)

            # flatten_trunk MUST be passed explicitly. build_body_landmarks' SIGNATURE default is
            # True, but production's argparse default is FALSE (Milestone-2: "keep the measured
            # trunk Z so the model can BEND at the waist and TURN like the skeleton"). Calling it
            # positionally therefore silently flattens the trunk - zeroing BOTH hip Z and shoulder Z
            # - which removes pelvis yaw and torso yaw entirely. That is what made the avatar's hips
            # dead in the first run of this harness.
            lm, src = W.build_body_landmarks(uv, xyz_cam, measured, conf, zrel, zrel_hip,
                                             mid_hip, hip_z, intr, a.conf,
                                             flatten_trunk=a.flatten_trunk, use_zrel=True)
            lh = W.build_hand(uv, xyz_cam, measured, conf, zrel, zrel_hip, mid_hip, hip_z, intr,
                              91, a.conf, use_zrel=True)
            rh = W.build_hand(uv, xyz_cam, measured, conf, zrel, zrel_hip, mid_hip, hip_z, intr,
                              112, a.conf, use_zrel=True)
            seq += 1
            # xyz is the avatar ROOT: the measured mid-hip in camera-space millimetres, all three
            # axes. Sending [0, 0, z] pinned the root laterally and vertically, so the dancer's hip
            # SWAY never reached Unity at all - a separate defect from the flatten above.
            # X/Y are scaled, Z is NOT: Z is the distance the avatar is sized from, so scaling it
            # would shrink or grow the figure rather than move it.
            msg = {"lm": lm,
                   "xyz": [round(float(mid_hip[0] * 1000.0 * a.sway_gain), 1),
                           round(float(mid_hip[1] * 1000.0 * a.sway_gain), 1),
                           round(float(mid_hip[2] * 1000.0), 1)],
                   "src": src, "seq": seq, "sid": SID, "t": round(time.time(), 4)}
            if lh is not None:
                msg["lh"] = lh
            if rh is not None:
                msg["rh"] = rh
            blob = json.dumps(msg).encode("utf-8")
            sock.sendto(blob, (a.host, a.port))
            if sent == 0 and a.mark:
                # The compose step aligns on this rather than assuming a fixed lead-in: model load
                # is ~10 s and varies, and a side-by-side that drifts is worse than none.
                with open(a.mark, "w", encoding="utf-8") as mf:
                    mf.write(json.dumps(dict(t_first=time.time(), fps=src_fps)))
            sent += 1
            if dump:
                dump.write(json.dumps(dict(msg, frame=frame_i, hip_z=round(hip_z, 3))) + "\n")

            frame_i += 1
            if a.realtime:
                lag = (frame_i / src_fps) - (time.time() - t_start)
                if lag > 0:
                    time.sleep(min(lag, 0.25))
            if sent % 60 == 0:
                print("  sent=%-5d frame=%-5d hip_z=%.2fm  %.1f fps"
                      % (sent, frame_i, hip_z, sent / max(1e-6, time.time() - t_start)), flush=True)

    cap.release()
    if dump:
        dump.close()
    dt = time.time() - t_start
    print("\nsent %d packets, skipped %d (no confident hip), %.1f s, %.1f fps"
          % (sent, skipped, dt, sent / max(1e-6, dt)))
    print("every src flag is 0: no joint on this wire carries a stereo measurement.")
    print("flatten_trunk=%s  (production default is False)" % a.flatten_trunk)
    return 0


if __name__ == "__main__":
    sys.exit(main())
