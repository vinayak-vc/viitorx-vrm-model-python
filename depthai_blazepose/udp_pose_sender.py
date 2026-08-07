#!/usr/bin/env python3
"""
OAK-D BlazePose -> UDP sender for the Unity Virtual Mirror (ADR-016, Option B2).

Runs Google BlazePose ON the OAK-D (edge mode: NN + post-processing on the device) via
geaxgx/depthai_blazepose, and streams the 33 body landmarks to Unity over a local UDP socket
as JSON. Because this runs as a SEPARATE process, any DepthAI/native instability stays here and
can never crash the Unity editor (that was the whole point of B2 vs the in-Unity plugin).

Landmarks streamed:
  body.landmarks_world -> 33 x [x, y, z] in METRES, origin at mid-hip (GHUM world; same convention
  as MediaPipe world landmarks, so the Unity retargeter consumes it 1:1). Per-keypoint z is the
  GHUM estimate for now; a later phase can add measured per-keypoint depth (body.xyz already carries
  the depth-anchored hip reference).

Run (from this folder, using the sidecar venv):
  ..\venv\Scripts\python udp_pose_sender.py            # defaults: 127.0.0.1:8899, lite model
  ..\venv\Scripts\python udp_pose_sender.py --lm full  # more accurate, lower fps
Stop with Ctrl+C.
"""

import argparse
import json
import socket

from BlazeposeDepthaiEdge import BlazeposeDepthai

NUM_KEYPOINTS = 33

# BlazePose skeleton connections (copied from BlazeposeRenderer.LINES_BODY) for the optional --show
# debug preview. Drawn with cv2 directly so we do NOT import BlazeposeRenderer (it pulls Open3D via
# o3d_utils, which is not installed in the sidecar venv).
LINES_BODY = [[9, 10], [4, 6], [1, 3],
              [12, 14], [14, 16], [16, 20], [20, 18], [18, 16],
              [12, 11], [11, 23], [23, 24], [24, 12],
              [11, 13], [13, 15], [15, 19], [19, 17], [17, 15],
              [24, 26], [26, 28], [32, 30],
              [23, 25], [25, 27], [29, 31]]


def draw_debug_overlay(cv2, tracker, frame, body, frames, sent, scale):
    """Draw the OAK-D RGB frame with the detected skeleton + status text (2D cv2 preview).

    Shows EXACTLY what the OAK camera sees and what BlazePose produced, so we can tell whether the
    camera is capturing motion (skeleton tracks the user well) independently of the Unity retarget.
    """
    view = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    if body is not None:
        landmarks = body.landmarks
        presence = getattr(body, "presence", None)
        i = 0
        while i < len(LINES_BODY):
            a = LINES_BODY[i][0]
            b = LINES_BODY[i][1]
            present_a = presence is None or presence[a] > tracker.presence_threshold
            present_b = presence is None or presence[b] > tracker.presence_threshold
            if present_a and present_b:
                pa = (int(landmarks[a][0] * scale), int(landmarks[a][1] * scale))
                pb = (int(landmarks[b][0] * scale), int(landmarks[b][1] * scale))
                cv2.line(view, pa, pb, (255, 180, 90), 2, cv2.LINE_AA)
            i += 1
        j = 0
        while j < NUM_KEYPOINTS:
            present = presence is None or presence[j] > tracker.presence_threshold
            if present:
                center = (int(landmarks[j][0] * scale), int(landmarks[j][1] * scale))
                color = (0, 255, 255) if j == 0 else ((0, 255, 0) if j % 2 == 0 else (0, 0, 255))
                cv2.circle(view, center, 4, color, -1)
            j += 1
        cv2.putText(view, "lm_score=%.2f" % body.lm_score, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 1)
        xyz = getattr(body, "xyz", None)
        if xyz is not None:
            cv2.putText(view, "hip mm x=%.0f y=%.0f z=%.0f" % (xyz[0], xyz[1], xyz[2]), (10, 40), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1)
    else:
        cv2.putText(view, "NO BODY DETECTED", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.4, (0, 0, 255), 2)
    cv2.putText(view, "frames=%d sent=%d" % (frames, sent), (10, view.shape[0] - 12), cv2.FONT_HERSHEY_PLAIN, 1.2, (240, 180, 100), 1)
    return view


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="Unity host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8899, help="UDP port (default 8899)")
    parser.add_argument("--lm", default="lite", help="landmark model: lite | full | heavy")
    parser.add_argument("--fps", type=int, default=0, help="internal color fps (0 = auto)")
    # OV9782-based OAK-Ds cap the color video at ~768x480 (800p) AND the ColorCamera ISP can only scale by
    # ratios inside the sensor's HW limits (MAX_N:16, MAX_D:63). find_isp_scale_params derives the ISP ratio
    # from this height, and some heights (e.g. 400 on this unit) yield a ratio the OV9782 rejects at runtime
    # (RuntimeError). 200 is verified working here (→352x198; the driver disables ISP scaling but the pipeline
    # runs). Depth resolution is independent of this value (depth comes from the 400P mono pair aligned to RGB),
    # so 200 is fine for Phase-2 per-keypoint depth sampling; only raise it if the RGB/NN crop looks too coarse,
    # and re-test — larger values may error on this OV9782.
    parser.add_argument("--frame_height", type=int, default=200, help="internal color frame height (legacy 1080p-derived path; used only when --color_res none)")
    parser.add_argument("--color_res", default="800p", choices=["800p", "720p", "none"], help="OV9782 native color mode for FULL FOV (default 800p). 'none' = legacy 1080p-derived geometry (crops on OV9782).")
    parser.add_argument("--color_scale", default="1/2", help="ISP downscale ratio n/d applied to the native sensor (default 1/2 → 640x400 full-FOV). Use 2/5 or 1/3 for less bandwidth.")
    parser.add_argument("--show", action="store_true", help="open a cv2 window with the OAK-D RGB + detected skeleton (debug)")
    args = parser.parse_args()

    color_res = None if args.color_res == "none" else args.color_res
    color_scale = None
    if color_res is not None and args.color_scale:
        scale_parts = args.color_scale.split("/")
        color_scale = (int(scale_parts[0]), int(scale_parts[1]))

    cv2 = None
    if args.show:
        import cv2 as cv2_module
        cv2 = cv2_module

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (args.host, args.port)
    print("[oak-udp] starting BlazePose on OAK-D (edge mode); streaming to %s model=%s show=%s" % (str(addr), args.lm, args.show))

    kwargs = dict(input_src="rgb", lm_model=args.lm, xyz=True, internal_frame_height=args.frame_height, stats=False, color_res=color_res, color_scale=color_scale)
    if args.fps > 0:
        kwargs["internal_fps"] = args.fps
    tracker = BlazeposeDepthai(**kwargs)

    frames = 0
    sent = 0
    try:
        while True:
            frame, body = tracker.next_frame()
            if frame is None:
                break
            frames += 1
            if body is not None:
                world = body.landmarks_world  # (>=33, 3) metres, hip-relative GHUM
                visibility = getattr(body, "visibility", None)
                count = min(NUM_KEYPOINTS, world.shape[0])
                points = []
                i = 0
                while i < count:
                    v = 1.0
                    if visibility is not None and i < len(visibility):
                        v = float(visibility[i])
                    points.append([
                        round(float(world[i, 0]), 4),
                        round(float(world[i, 1]), 4),
                        round(float(world[i, 2]), 4),
                        round(v, 3),
                    ])
                    i += 1
                message = {"lm": points}
                xyz = getattr(body, "xyz", None)
                if xyz is not None:
                    message["xyz"] = [round(float(xyz[0]), 1), round(float(xyz[1]), 1), round(float(xyz[2]), 1)]
                sock.sendto(json.dumps(message).encode("utf-8"), addr)
                sent += 1
                if sent <= 3 or sent % 90 == 0:
                    print("[oak-udp] frames=%d sent=%d kp0=%s xyz=%s" % (frames, sent, str(points[0]), str(message.get("xyz"))))
            if cv2 is not None:
                view = draw_debug_overlay(cv2, tracker, frame, body, frames, sent, 2)
                cv2.imshow("OAK-D BlazePose (sidecar debug)", view)
                key = cv2.waitKey(1)
                if key == 27 or key == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        tracker.exit()
        sock.close()
        if cv2 is not None:
            cv2.destroyAllWindows()
        print("[oak-udp] stopped. frames=%d sent=%d" % (frames, sent))


if __name__ == "__main__":
    main()
