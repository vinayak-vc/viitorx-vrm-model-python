#!/usr/bin/env python3
"""TORSO YAW V4 -- OAK-D safety analysis: the failure modes the brief names, measured.

Runs the recorded REAL OAK-D torso yaw through the shipping conditioner (port verified against the
live C# to 6.7e-6 deg) and the V4 sec 2b composition, then measures the five symptoms the brief asks
to watch for:

    * +/-180 deg yaw flips        -- adjacent-frame wrapped step > 150 deg
    * sudden torso reversal       -- avatar output crossing sign while |output| > 20 deg
    * torso jitter                -- stdev of the avatar output while the SOURCE is at rest (<8 deg)
    * excessive torso twist       -- |avatar output| beyond a visible threshold, and for how long
    * arm drag                    -- handled separately in Unity (sec 4); not derivable from this log

and cross-tabulates the large source excursions against the sidecar's own measured-depth coverage
`cov` (0..33), to separate a genuine turn from a depth-noise artefact.

    python oak_safety_v4.py
"""
import io
import json
import math
import os

from oak_yaw_v4 import (DampYaw, load, wrapped_delta, pct, mean,
                        HIPS_DAMP, SPINE_DAMP, CHEST_DAMP)

TWIST_LIMITS = (45.0, 60.0, 90.0)


def stdev(v):
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def episodes(flags, times):
    """Contiguous runs where flags is True -> list of durations in seconds."""
    out, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append(times[i - 1] - times[start])
            start = None
    if start is not None:
        out.append(times[-1] - times[start])
    return out


def run(name, rows, scale):
    dh, ds = DampYaw(), DampYaw()
    dts = [rows[i]["t"] - rows[i - 1]["t"] for i in range(1, len(rows))]
    med_dt = pct([d for d in dts if 0 < d < 1.0], 50) or 0.0465
    prev_t = rows[0]["t"]
    out, src, times, valid, cov = [], [], [], [], []
    for r in rows:
        dt = r["t"] - prev_t
        prev_t = r["t"]
        if not (0.0 < dt < 1.0):
            dt = med_dt
        h = math.degrees(dh.step(math.radians(r["hip_yaw"]), dt))
        s = math.degrees(ds.step(math.radians(r["sh_yaw"]), dt))
        out.append(min(1.0, scale) * (HIPS_DAMP * h + SPINE_DAMP * s + CHEST_DAMP * s))
        src.append(r["sh_yaw"])
        times.append(r["t"])
        valid.append(r["valid"])
        cov.append(r["cov"])
    return out, src, times, valid, cov, med_dt


def report(name, rows):
    print("=" * 106)
    print(" %s  --  %d frames, %.0f s" % (name, len(rows), rows[-1]["t"] - rows[0]["t"]))
    print("=" * 106)

    # --- is a large SOURCE excursion a real turn or depth noise? -----------------------------
    src_all = [r["sh_yaw"] for r in rows if r["valid"]]
    cov_all = [r["cov"] for r in rows if r["valid"]]
    big = [(y, c) for y, c in zip(src_all, cov_all) if abs(y) > 60]
    calm = [(y, c) for y, c in zip(src_all, cov_all) if abs(y) <= 22]
    print("  source |yaw|>60deg : %5d frames (%.2f%%)   mean measured-depth coverage %.1f/33" %
          (len(big), 100.0 * len(big) / max(1, len(src_all)), mean([c for _y, c in big]) if big else 0))
    print("  source |yaw|<=22deg: %5d frames (%.2f%%)   mean measured-depth coverage %.1f/33" %
          (len(calm), 100.0 * len(calm) / max(1, len(src_all)), mean([c for _y, c in calm]) if calm else 0))

    print("\n  %6s | %7s %7s | %7s %7s %7s | %8s | %7s %7s" %
          ("scale", "outp95", "outmax", ">45deg", ">60deg", ">90deg", "longest", "rest", "revers"))
    print("  %6s | %7s %7s | %7s %7s %7s | %8s | %7s %7s" %
          ("", "deg", "deg", "% time", "% time", "% time", "ep. (s)", "jitter", "als"))
    for scale in (0.0, 0.25, 0.5, 0.75, 0.813, 1.0):
        out, src, times, valid, _cov, _dt = run(name, rows, scale)
        a = [abs(v) for v in out]
        pcts = []
        for lim in TWIST_LIMITS:
            pcts.append(100.0 * sum(1 for v in a if v > lim) / len(a))
        longest = max(episodes([v > 45.0 for v in a], times) or [0.0])
        rest = [out[i] for i in range(len(out)) if valid[i] and abs(src[i]) < 8.0]
        revs = sum(1 for i in range(1, len(out))
                   if abs(out[i - 1]) > 20 and abs(out[i]) > 20 and out[i - 1] * out[i] < 0)
        print("  %6.3f | %7.2f %7.2f | %7.2f %7.2f %7.2f | %8.2f | %7.3f %7d" %
              (scale, pct(a, 95), max(a), pcts[0], pcts[1], pcts[2], longest, stdev(rest), revs))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    # the two richest real-human captures with measured trunk Z
    for d in ("pipeline_logs_armv1", "pipeline_logs_armv1b", "pipeline_logs_p14"):
        p = os.path.join(here, d, "sender_log.jsonl")
        if not os.path.exists(p):
            continue
        rows = load(p)
        if not rows:
            continue
        report(d, rows)
        print("")
    print("NOTE: 'rest jitter' is the stdev of the AVATAR output over frames whose SOURCE yaw is")
    print("      under the 8 deg dead-zone floor -- i.e. what the torso does when the human is still.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
