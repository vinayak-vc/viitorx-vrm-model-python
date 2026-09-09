#!/usr/bin/env python3
"""Drive the Unity avatar from a VIDEO FILE over the real UDP wire (ARM RETARGET V2 forensics).

Runs the SAME RTMW3D model the OAK-D sidecar runs, on every frame of a video, and streams the result
on the same JSON contract. Lets the whole Unity path be exercised repeatably, on identical input,
with no camera and no human — which a live capture can never provide.

IMPORTANT LIMITATION, and it is a real one: a video has NO DEPTH and NO CAMERA INTRINSICS, so there is
no measured metric XYZ. Following `replay_video.py`, an ESTIMATED metric signal is built instead:

  * x/y  : model pixels, scaled by a global torso normalisation (median |midShoulder - midHip| in px
           is mapped to NOMINAL_TORSO_M), so the skeleton has plausible human proportions;
  * z    : the model's own root-relative metric depth (zrel), which IS metres but is a monocular
           estimate, not a stereo measurement.

Everything downstream of the wire is therefore exercised exactly as in production, but anything this
run attributes to the SOURCE stage is a property of (monocular model + this estimator), NOT of the
OAK-D depth path. Do not read a source-stage error here as an OAK-D result.

    python video_udp_sender.py --video ../../video/video.webm --loop
"""
import argparse
import json
import os
import socket
import sys
import time

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rtmw3d_pose as R

NUM_BODY = 33
NOMINAL_TORSO_M = 0.5
CONF_THR = 0.3
CALIB_FRAMES = 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default=os.path.join("..", "..", "..", "SentisModel", "rtmw3d-x.onnx"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--loop", action="store_true", help="restart the video when it ends")
    ap.add_argument("--rate", type=float, default=0.0, help="send fps (0 = the video's own fps)")
    ap.add_argument("--log-dir", default="", help="write video_sender_log.jsonl here")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after N seconds (0 = forever)")
    a = ap.parse_args()

    if not os.path.exists(a.video):
        print("[ERROR] video not found: %s" % a.video)
        return 1
    if not os.path.exists(a.model):
        print("[ERROR] model not found: %s" % a.model)
        return 1

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        print("[ERROR] cannot open video")
        return 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    period = 1.0 / (a.rate if a.rate > 0 else vfps)

    print("=" * 72)
    print(" VIDEO -> UDP  (%dx%d @ %.1f fps)  ->  %s:%d" % (w, h, vfps, a.host, a.port))
    print(" %s" % a.video)
    print(" NOTE: estimated metric scale, monocular depth. No OAK-D depth in this path.")
    print("=" * 72)
    model = R.RTMW3D(a.model)
    print(" providers: %s" % (model.active_providers,))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (a.host, a.port)
    # Monotonic across restarts: P1-3's PoseBuffer drops any packet whose seq is below the newest it
    # already holds, so a driver restarting at 0 while Unity stays in Play is silently ignored.
    seq = int((time.time() - 1788900000.0) * 100.0)

    log_f = None
    if a.log_dir:
        os.makedirs(a.log_dir, exist_ok=True)
        log_f = open(os.path.join(a.log_dir, "video_sender_log.jsonl"), "w")

    bbox = R.center_bbox(w, h)
    torso_px = []
    scale = None
    lowconf = 0
    frames = 0
    sent = 0
    t_start = time.time()
    next_send = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            if a.loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        frames += 1
        uv, zrel, conf = model.infer(frame, bbox)
        refined = R.bbox_from_keypoints(uv, conf, w, h, thr=CONF_THR)
        if refined is not None and float(np.mean(conf[0:17])) >= CONF_THR:
            bbox = refined
            lowconf = 0
        else:
            lowconf += 1
            if refined is None or lowconf >= 20:
                bbox = R.center_bbox(w, h)
                lowconf = 0

        # --- estimated metric scale from torso length (frozen after calibration) ---------
        mid_sh = (uv[5] + uv[6]) * 0.5
        mid_hip = (uv[11] + uv[12]) * 0.5
        if scale is None:
            if min(conf[5], conf[6], conf[11], conf[12]) >= CONF_THR:
                torso_px.append(float(np.linalg.norm(mid_sh - mid_hip)))
            if len(torso_px) >= CALIB_FRAMES:
                med = float(np.median(torso_px))
                scale = (NOMINAL_TORSO_M / med) if med > 1e-6 else None
                print(" calibrated: torso %.1f px -> %.2f m  (scale %.6f m/px)" % (med, NOMINAL_TORSO_M, scale))
            if scale is None:
                continue

        hip_z = float(zrel[11] + zrel[12]) * 0.5
        lm = [[0.0, 0.0, 0.0, 0.0] for _ in range(NUM_BODY)]
        src = [0] * NUM_BODY
        for coco_idx, joint_id in R.COCO17_TO_JOINTID.items():
            c = float(conf[coco_idx])
            if c < CONF_THR:
                continue
            x = float(uv[coco_idx][0] - mid_hip[0]) * scale
            y = float(uv[coco_idx][1] - mid_hip[1]) * scale
            z = float(zrel[coco_idx]) - hip_z
            lm[joint_id] = [round(x, 5), round(y, 5), round(z, 5), c]
            src[joint_id] = 0        # 0 = not a measured depth; this path has none
        for foot_idx, joint_id in R.FOOT_TO_JOINTID.items():
            c = float(conf[foot_idx])
            if c < CONF_THR:
                continue
            x = float(uv[foot_idx][0] - mid_hip[0]) * scale
            y = float(uv[foot_idx][1] - mid_hip[1]) * scale
            z = float(zrel[foot_idx]) - hip_z
            lm[joint_id] = [round(x, 5), round(y, 5), round(z, 5), c]

        now = time.time()
        if now < next_send:
            time.sleep(max(0.0, next_send - now))
        next_send = time.time() + period
        msg = {"lm": lm, "xyz": [0.0, 0.0, 2500.0], "src": src,
               "seq": seq, "t": round(time.time(), 4)}
        sock.sendto(json.dumps(msg).encode("utf-8"), addr)
        if log_f is not None:
            log_f.write(json.dumps({"seq": seq, "t": round(time.time(), 4), "frame": frames,
                                    "conf": [round(float(conf[i]), 3) for i in (5, 6, 7, 8, 9, 10)]}) + "\n")
            log_f.flush()
        seq += 1
        sent += 1
        if sent % 100 == 0:
            print("  frames=%d sent=%d  %.1f fps" % (frames, sent, sent / (time.time() - t_start)))
        if a.seconds > 0 and (time.time() - t_start) >= a.seconds:
            break

    cap.release()
    if log_f is not None:
        log_f.close()
    print("\ndone: %d frames, %d sent" % (frames, sent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
