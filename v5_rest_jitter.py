#!/usr/bin/env python3
"""TORSO V5 -- rest jitter and over-twist, OLD vs NEW composition, on identical recorded OAK-D input.

V4 sec 5d measured, on the real depth corpus, that at torsoYawScale = 0.75 the avatar's torso wandered
with a 5.4-7.6 deg standard deviation WHILE THE SUBJECT WAS STILL, against exactly 0.000 deg at scale 0.
The brief requires that to improve, not merely stay unchanged.

Both compositions are driven from the SAME recorded yaw, through the SAME verified `DampYaw` port
(checked against the shipping C# to 6.7e-6 deg), so the only difference is the composition:

    OLD (V4):  chain = 0.70*Damp(hipYaw) + (0.45+0.25)*Damp(shoulderYaw)      two ABSOLUTE yaws summed
    NEW (V5):  hips  = Damp(hipYaw)
               twist = Damp(shoulderYaw) - Damp(hipYaw)                       weights sum to 1.0
               chain = hips + twist  ==  Damp(shoulderYaw)

The V5 trunk gate is also applied, so degenerate frames hold instead of injecting the +90 sentinel.

"Rest" = frames whose SOURCE shoulder yaw is inside the 8 deg dead-zone floor, i.e. the human is
square to the camera and the torso should not be moving at all.

    python v5_rest_jitter.py
"""
import io
import math
import os

from oak_yaw_v4 import DampYaw, load, pct, mean

MIN_SPAN = 0.10          # TrunkGate.MinSpan
MAX_SPAN = 1.0           # TrunkGate.MaxSpan
MAX_RATE = 500.0         # TrunkGate.MaxYawRateDegPerSec
REACQUIRE = 8            # TrunkGate.ReacquireFrames


def stdev(v):
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def wrapd(a):
    return (a + 180.0) % 360.0 - 180.0


def gated(rows):
    """Apply the V5 TrunkGate to the recorded stream; yields (hipYaw, shYaw, fresh) per frame."""
    held_h, held_s, has, streak = 0.0, 0.0, False, 0
    out = []
    prev_t = rows[0]["t"]
    for r in rows:
        dt = r["t"] - prev_t
        prev_t = r["t"]
        if not (0 < dt < 1.0):
            dt = 0.0465
        ok = (MIN_SPAN < r["sh_span"] <= MAX_SPAN) and (MIN_SPAN < r["hip_span"] <= MAX_SPAN)
        if ok and has:
            step = max(abs(wrapd(r["sh_yaw"] - held_s)), abs(wrapd(r["hip_yaw"] - held_h)))
            if step > MAX_RATE * dt:
                ok = False
                streak += 1
                if streak >= REACQUIRE:
                    ok, streak = True, 0
            else:
                streak = 0
        elif not ok:
            streak = 0
        if ok:
            held_h, held_s, has = r["hip_yaw"], r["sh_yaw"], True
        out.append((held_h if has else 0.0, held_s if has else 0.0, ok, r, dt))
    return out


def run(rows, scale, new):
    dh, ds = DampYaw(), DampYaw()
    chain, src, gate_on = [], [], []
    for h_raw, s_raw, ok, r, dt in gated(rows):
        h = math.degrees(dh.step(math.radians(h_raw), dt))
        s = math.degrees(ds.step(math.radians(s_raw), dt))
        if new:
            c = scale * (h + (s - h))          # hips + relative twist  ==  s
        else:
            c = scale * (0.70 * h + 0.70 * s)  # two absolute yaws summed
        chain.append(c)
        src.append(r["sh_yaw"] if ok else s_raw)
        gate_on.append(ok)
    return chain, src, gate_on


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    print("=" * 104)
    print(" TORSO V5 -- rest jitter / over-twist, OLD vs NEW composition, identical recorded OAK-D input")
    print("=" * 104)
    print(" %-22s %-14s | %9s | %8s %8s %8s | %8s" %
          ("capture", "composition", "rest sd", ">45deg", ">90deg", "out max", "err mean"))
    for d in ("pipeline_logs_armv1", "pipeline_logs_armv1b", "pipeline_logs_p14"):
        p = os.path.join(here, d, "sender_log.jsonl")
        if not os.path.exists(p):
            continue
        rows = load(p)
        if len(rows) < 100:
            continue
        for label, new, scale in (("OLD V4 @0.75", False, 0.75), ("NEW V5 @1.00", True, 1.0)):
            chain, src, ok = run(rows, scale, new)
            a = [abs(c) for c in chain]
            rest = [chain[i] for i in range(len(chain)) if ok[i] and abs(src[i]) < 8.0]
            err = [abs(wrapd(chain[i] - src[i])) for i in range(len(chain)) if ok[i]]
            print(" %-22s %-14s | %9.3f | %7.2f%% %7.2f%% %8.2f | %8.2f" %
                  (d if label.startswith("OLD") else "", label, stdev(rest),
                   100.0 * sum(1 for x in a if x > 45) / len(a),
                   100.0 * sum(1 for x in a if x > 90) / len(a),
                   max(a), mean(err)))
        print("")
    print(" 'rest sd' = stdev of the avatar's commanded torso yaw over frames whose SOURCE yaw is")
    print(" inside the 8 deg dead-zone floor -- what the torso does while the human stands still.")
    print(" NEW is measured at scale 1.00 (its physically correct setting) against OLD at 0.75")
    print(" (the value V4 shipped), i.e. the comparison each composition is actually used at.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
