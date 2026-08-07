#!/usr/bin/env python3
"""
MOCK OAK-D pose sender for DEVICE-FREE testing of the Unity B2 path (ADR-016 / doc 26).

Streams a synthetic, animated 33-landmark BlazePose skeleton over the SAME UDP JSON contract as the
real sidecar (`udp_pose_sender.py`) — `{"lm": [[x,y,z,vis] x33], "xyz":[...]}`, hip-relative metres —
but needs NO OAK-D and NO depthai/venv (Python standard library only). Use it to prove that
`OakDUdpPoseProvider` receives + parses + drives the avatar before any hardware is attached:

  1. In Unity, tick `AppBootstrap.useOakUdpTracking` (port 8899) and press Play.
  2. Run this:  python mock_udp_sender.py
  3. HUD reads "OAK-D 3D (UDP sidecar)"; console shows "OAK-D UDP rx=..."; the avatar waves.

This is a smoke test of the WIRE + provider only. The skeleton is anatomically rough (coordinates are
not from a real body); its only job is to move plausibly so you can see tracking respond. Axis/mirror
still tune via poseFlipX/Y/Z exactly as with the real sidecar.
"""

import argparse
import json
import math
import socket
import time

NUM_KEYPOINTS = 33

# BlazePose index -> rough neutral position in metres, mid-hip origin, MediaPipe-world convention
# (x right, y DOWN, z forward). Only the joints the retargeter uses are placed; the rest sit near origin.
# Up the body is negative-y. Left/right are the subject's, mirrored later by poseFlipX if needed.
BASE = {
    0: (0.00, -0.62, 0.05),   # nose
    11: (0.18, -0.50, 0.00),  # left shoulder
    12: (-0.18, -0.50, 0.00), # right shoulder
    13: (0.34, -0.30, 0.00),  # left elbow
    14: (-0.34, -0.30, 0.00), # right elbow
    15: (0.42, -0.10, 0.00),  # left wrist (animated)
    16: (-0.42, -0.10, 0.00), # right wrist (animated)
    23: (0.10, 0.00, 0.00),   # left hip
    24: (-0.10, 0.00, 0.00),  # right hip
    25: (0.12, 0.45, 0.00),   # left knee
    26: (-0.12, 0.45, 0.00),  # right knee
    27: (0.13, 0.88, 0.00),   # left ankle
    28: (-0.13, 0.88, 0.00),  # right ankle
}


def build_frame(t):
    """One animated pose at time t (seconds). Wrists wave up/down + a little forward/back in z."""
    wave = math.sin(t * 2.0)          # -1..1
    swing = math.cos(t * 2.0)
    points = []
    i = 0
    while i < NUM_KEYPOINTS:
        if i in BASE:
            x, y, z = BASE[i]
            if i == 15:  # left wrist
                y = y - 0.25 * (wave + 1.0)   # raise as wave goes positive
                z = 0.15 * swing
            elif i == 16:  # right wrist
                y = y - 0.25 * (1.0 - wave)   # opposite phase
                z = -0.15 * swing
            elif i == 13:  # left elbow follows a bit
                y = y - 0.10 * (wave + 1.0)
            elif i == 14:
                y = y - 0.10 * (1.0 - wave)
            points.append([round(x, 4), round(y, 4), round(z, 4), 1.0])
        else:
            points.append([0.0, 0.0, 0.0, 0.0])  # undefined joint, zero confidence
        i = i + 1
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="Unity host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8899, help="UDP port (default 8899, matches oakUdpPort)")
    parser.add_argument("--fps", type=float, default=30.0, help="send rate (default 30)")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (args.host, args.port)
    period = 1.0 / args.fps if args.fps > 0 else 0.0
    print("[mock-oak] streaming synthetic skeleton to %s at %.0f fps (Ctrl+C to stop)" % (str(addr), args.fps))

    sent = 0
    start = time.time()
    try:
        while True:
            t = time.time() - start
            points = build_frame(t)
            message = {"lm": points, "xyz": [0.0, 0.0, 1500.0]}
            sock.sendto(json.dumps(message).encode("utf-8"), addr)
            sent = sent + 1
            if sent <= 3 or sent % 90 == 0:
                print("[mock-oak] sent=%d t=%.1fs kp16(rwrist)=%s" % (sent, t, str(points[16])))
            if period > 0:
                time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        print("[mock-oak] stopped. sent=%d" % sent)


if __name__ == "__main__":
    main()
