#!/usr/bin/env python3
"""P1-3 A/B motion streamer — deterministic, repeatable input.

Streams a smoothly-moving 33-landmark pose over the real UDP contract at a controlled
packet rate. Interpolation quality must be measured with IDENTICAL input in both runs,
which a human cannot reproduce by hand, so the A/B uses this instead of live motion.

The motion is a continuous sinusoidal arm/leg swing (no discontinuities), so any
stair-stepping seen in the avatar comes from the render/packet rate mismatch, not the input.

    python stream_motion.py --fps 21 --seconds 30
"""
import argparse
import json
import math
import socket
import time

NUM = 33

BASE = {
    0: (0.00, -0.62, 0.05),
    11: (0.18, -0.50, 0.00), 12: (-0.18, -0.50, 0.00),
    13: (0.34, -0.30, 0.00), 14: (-0.34, -0.30, 0.00),
    15: (0.42, -0.10, 0.00), 16: (-0.42, -0.10, 0.00),
    23: (0.10, 0.00, 0.00), 24: (-0.10, 0.00, 0.00),
    25: (0.12, 0.45, 0.00), 26: (-0.12, 0.45, 0.00),
    27: (0.13, 0.88, 0.00), 28: (-0.13, 0.88, 0.00),
}

# joints that swing, and how far (metres)
SWING = {13: 0.18, 14: 0.18, 15: 0.30, 16: 0.30, 25: 0.10, 26: 0.10, 27: 0.16, 28: 0.16}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--fps", type=float, default=21.0, help="packet rate (match the real sidecar)")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--hz", type=float, default=0.6, help="swing frequency")
    ap.add_argument("--conf", type=float, default=0.9)
    a = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (a.host, a.port)
    dt = 1.0 / a.fps
    seq = 0
    t0 = time.time()
    print("streaming %.0f Hz for %.0f s -> %s:%d" % (a.fps, a.seconds, a.host, a.port))
    while time.time() - t0 < a.seconds:
        # phase is driven by ELAPSED TIME, not the packet index, so both A/B runs see the
        # same trajectory even if the packet rate wobbles slightly.
        el = time.time() - t0
        ph = 2.0 * math.pi * a.hz * el
        s = math.sin(ph)
        lm = [[0.0, 0.0, 0.0, 0.0] for _ in range(NUM)]
        for idx, (x, y, z) in BASE.items():
            amp = SWING.get(idx, 0.0)
            lm[idx] = [x + amp * s, y, z + amp * 0.4 * math.cos(ph), a.conf]
        msg = {"lm": lm, "xyz": [0.0, 0.0, 2200.0], "src": [1] * NUM,
               "seq": seq, "t": round(time.time(), 4)}
        sock.sendto(json.dumps(msg).encode("utf-8"), addr)
        seq += 1
        nxt = t0 + seq * dt
        sl = nxt - time.time()
        if sl > 0:
            time.sleep(sl)
    sock.close()
    print("sent %d packets in %.1f s (%.2f Hz)" % (seq, time.time() - t0, seq / (time.time() - t0)))


if __name__ == "__main__":
    main()
