#!/usr/bin/env python3
"""TORSO V5 -- controlled torso probe on the real UDP wire.

`video.webm` cannot test the composition: its source shoulder yaw is p50 4.2 deg and only ~2.7 % of
frames clear the ADR-027 dead-zone knee (22 deg), so almost every frame is deliberately zeroed by the
conditioner and any gain measured there measures the DEAD ZONE, not the decomposition.

This streams a synthetic but fully valid 4-point trunk whose hip and shoulder yaw are commanded
INDEPENDENTLY, so the three cases the V4 report proved the old form got wrong can each be held long
enough for the rate limiter (140 deg/s) and low-pass (tau 0.15 s) to settle:

  rigid   hip == shoulder == +/-45 deg   -> V5 must give Hips ~= 45, upper-trunk twist ~= 0, gain ~= 1
                                            (the OLD form summed both and over-rotated ~1.4x)
  twist   hip = 0, shoulder = +/-35 deg  -> Hips ~= 0, twist ~= 35
  oppose  hip = +30, shoulder = -30      -> Hips ~= 30, twist ~= -60; the OLD form CANCELLED here
                                            (measured gain +0.116) and inverted on a 52 deg twist

The landmarks are geometrically plausible (shoulder span 0.36 m, hip span 0.20 m) so the TrunkGate
accepts every frame -- this probes the composition, not the gate.

    python torso_probe_udp.py
"""
import argparse
import json
import math
import socket
import time

NUM_BODY = 33
CONF = 0.9
HALF_SH = 0.18
HALF_HIP = 0.10

# name, hold seconds, hip yaw deg, shoulder yaw deg
PHASES = [
    ("zero",       3.0,   0.0,   0.0),
    ("rigid+45",   5.0,  45.0,  45.0),
    ("zero",       3.0,   0.0,   0.0),
    ("rigid-45",   5.0, -45.0, -45.0),
    ("zero",       3.0,   0.0,   0.0),
    ("twist+35",   5.0,   0.0,  35.0),
    ("zero",       3.0,   0.0,   0.0),
    ("twist-35",   5.0,   0.0, -35.0),
    ("zero",       3.0,   0.0,   0.0),
    ("oppose",     5.0,  30.0, -30.0),
    ("zero",       3.0,   0.0,   0.0),
    ("rigid+90",   5.0,  90.0,  90.0),
    ("zero",       3.0,   0.0,   0.0),
]
RAMP = 1.2          # s to slew between phases, so the 140 deg/s rate limiter is never the bottleneck


def build(hip_deg, sh_deg):
    lm = [[0.0, 0.0, 0.0, 0.0] for _ in range(NUM_BODY)]

    def put(i, x, y, z):
        lm[i] = [round(x, 5), round(y, 5), round(z, 5), CONF]

    hs, hc = math.sin(math.radians(hip_deg)), math.cos(math.radians(hip_deg))
    ss, sc = math.sin(math.radians(sh_deg)), math.cos(math.radians(sh_deg))
    put(11, HALF_SH * sc, 0.50, HALF_SH * ss)      # subject LEFT shoulder
    put(12, -HALF_SH * sc, 0.50, -HALF_SH * ss)    # subject RIGHT shoulder
    put(23, HALF_HIP * hc, 0.0, HALF_HIP * hs)     # subject LEFT hip
    put(24, -HALF_HIP * hc, 0.0, -HALF_HIP * hs)
    # arms held rigidly in the SHOULDER frame, so a correct rig rotates them with the chest and the
    # body-relative arm error stays constant -- that is the TASK 4 invariant under torso motion.
    for idx, sgn in ((13, 1.0), (14, -1.0)):
        put(idx, (HALF_SH + 0.14) * sc * sgn, 0.30, (HALF_SH + 0.14) * ss * sgn)
    for idx, sgn in ((15, 1.0), (16, -1.0)):
        put(idx, (HALF_SH + 0.26) * sc * sgn, 0.12, (HALF_SH + 0.26) * ss * sgn)
    # legs follow the hips, so the avatar hip line is well defined
    for idx, sgn in ((25, 1.0), (26, -1.0)):
        put(idx, HALF_HIP * hc * sgn, -0.45, HALF_HIP * hs * sgn)
    for idx, sgn in ((27, 1.0), (28, -1.0)):
        put(idx, HALF_HIP * hc * sgn, -0.90, HALF_HIP * hs * sgn)
    return lm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--rate", type=float, default=30.0)
    ap.add_argument("--marks", default="oak_v4_evidence/torso_probe_marks.json")
    a = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (a.host, a.port)
    seq = int((time.time() - 1788900000.0) * 100.0)
    period = 1.0 / a.rate
    marks = []
    prev_hip, prev_sh = 0.0, 0.0

    print("=" * 76)
    print(" TORSO PROBE -> %s:%d  @ %.0f Hz   %d phases" % (a.host, a.port, a.rate, len(PHASES)))
    print("=" * 76)
    for name, hold, hip, sh in PHASES:
        # ramp so the rate limiter is never the limiting factor
        t_ramp = time.time()
        while time.time() - t_ramp < RAMP:
            k = (time.time() - t_ramp) / RAMP
            lm = build(prev_hip + (hip - prev_hip) * k, prev_sh + (sh - prev_sh) * k)
            sock.sendto(json.dumps({"lm": lm, "xyz": [0.0, 0.0, 2000.0], "src": [0] * NUM_BODY,
                                    "seq": seq, "t": round(time.time(), 4)}).encode("utf-8"), addr)
            seq += 1
            time.sleep(period)
        prev_hip, prev_sh = hip, sh
        t0, s0 = time.time(), seq
        while time.time() - t0 < hold:
            lm = build(hip, sh)
            sock.sendto(json.dumps({"lm": lm, "xyz": [0.0, 0.0, 2000.0], "src": [0] * NUM_BODY,
                                    "seq": seq, "t": round(time.time(), 4)}).encode("utf-8"), addr)
            seq += 1
            time.sleep(period)
        marks.append({"name": name, "hip": hip, "sh": sh, "seq0": s0, "seq1": seq - 1})
        print("  %-10s hip %+6.1f  shoulder %+6.1f   seq %d..%d" % (name, hip, sh, s0, seq - 1))

    with open(a.marks, "w") as f:
        json.dump({"phases": marks, "ramp_s": RAMP}, f, indent=2)
    print("\n marks -> %s" % a.marks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
