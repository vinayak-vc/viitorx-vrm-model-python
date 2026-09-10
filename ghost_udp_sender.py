#!/usr/bin/env python3
"""TORSO V5 -- stream a LIVE but EMPTY pose, to test the false +90 deg torso command directly.

This is the exact failure the V4 report caught live: the UDP stream stays healthy (so Unity's
freshness/stale logic is satisfied and `Apply` keeps running) while the body landmarks collapse to
[0,0,0]. Kalidokit's `CalcHipsAndSpine` then computes `atan2(0,0) = 0`, which its `+0.5` jump-fix
turns into EXACTLY +90 deg of commanded torso yaw.

Phases, so the before/after is visible in one continuous trace:
  1. `--warm` seconds of a REAL, gently turning torso   -> the gate accepts, avatar follows
  2. `--ghost` seconds of all-zero landmarks             -> the gate must HOLD, not swing to +90
  3. `--warm` seconds of the real torso again            -> the gate must resume immediately

    python ghost_udp_sender.py --warm 6 --ghost 10
"""
import argparse
import json
import math
import socket
import time

NUM_BODY = 33
CONF = 0.75          # deliberately HIGH: the V4 report measured 0.71 accompanying a hallucination,
                     # so the gate must not be relying on confidence to reject this.


def real_pose(t):
    """A plausible standing torso, yawing slowly by +/-20 deg so 'following' is visible."""
    yaw = math.radians(20.0 * math.sin(t * 0.6))
    half_sh, half_hip = 0.18, 0.10
    lm = [[0.0, 0.0, 0.0, 0.0] for _ in range(NUM_BODY)]

    def put(i, x, y, z):
        lm[i] = [round(x, 5), round(y, 5), round(z, 5), CONF]

    c, s = math.cos(yaw), math.sin(yaw)
    # shoulders (11 = subject left, 12 = subject right) and hips (23/24) rotate together = RIGID turn
    put(11, half_sh * c, 0.50, half_sh * s)
    put(12, -half_sh * c, 0.50, -half_sh * s)
    put(23, half_hip * c, 0.0, half_hip * s)
    put(24, -half_hip * c, 0.0, -half_hip * s)
    # arms, so the arm-regression path has something to track
    put(13, 0.30, 0.30, 0.0)
    put(14, -0.30, 0.30, 0.0)
    put(15, 0.40, 0.10, 0.0)
    put(16, -0.40, 0.10, 0.0)
    # legs, so the avatar hip line is defined
    put(25, half_hip, -0.45, 0.0)
    put(26, -half_hip, -0.45, 0.0)
    put(27, half_hip, -0.90, 0.0)
    put(28, -half_hip, -0.90, 0.0)
    return lm


def ghost_pose():
    """What the sidecar actually emits with no detection: every body landmark [0,0,0]."""
    return [[0.0, 0.0, 0.0, 0.0] for _ in range(NUM_BODY)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--warm", type=float, default=6.0)
    ap.add_argument("--ghost", type=float, default=10.0)
    ap.add_argument("--rate", type=float, default=30.0)
    a = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (a.host, a.port)
    # Monotonic across restarts: P1-3's PoseBuffer drops any packet whose seq is below the newest it holds.
    seq = int((time.time() - 1788900000.0) * 100.0)
    period = 1.0 / a.rate
    t0 = time.time()
    phases = [("REAL   ", a.warm), ("GHOST  ", a.ghost), ("REAL   ", a.warm)]
    print("=" * 68)
    print(" GHOST SENDER -> %s:%d   real %.0fs / ghost %.0fs / real %.0fs @ %.0f Hz"
          % (a.host, a.port, a.warm, a.ghost, a.warm, a.rate))
    print(" ghost phase = all-zero landmarks with conf %.2f still on the wire" % CONF)
    print("=" * 68)
    for label, dur in phases:
        mark = time.time()
        print(" %s phase, %.0fs  (seq from %d)" % (label, dur, seq))
        while time.time() - mark < dur:
            lm = ghost_pose() if label.strip() == "GHOST" else real_pose(time.time() - t0)
            msg = {"lm": lm, "xyz": [0.0, 0.0, 2000.0], "src": [0] * NUM_BODY,
                   "seq": seq, "t": round(time.time(), 4)}
            sock.sendto(json.dumps(msg).encode("utf-8"), addr)
            seq += 1
            time.sleep(period)
    print(" done, %d packets" % (seq - int((t0 - 1788900000.0) * 100.0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
