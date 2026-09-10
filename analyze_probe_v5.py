#!/usr/bin/env python3
"""TORSO V5 -- score the controlled torso probe, per commanded phase.

Each phase held its hip and shoulder yaw constant long enough for the ADR-027 rate limiter (140 deg/s)
and low-pass (tau 0.15 s) to settle, so the STEADY-STATE value is what the composition produces with
no filter lag. Only the last 60 % of each phase is scored, for that reason.

The measurement is on the BONE-LOCAL yaws the driver writes:
    Hips Y                        should track the ABSOLUTE hip yaw
    Spine+Chest+UpperChest        should track the RELATIVE twist (shoulder - hip)
    their sum                     should track the shoulder yaw (bones nest)

Sign note: `mirrorSagittal` negates X, which flips the sign of the WORLD shoulder-line yaw. It does NOT
flip the bone-local Y euler the driver writes, which is what is scored here — measured directly: a
commanded +45 deg produces Hips Y = +45.00. Gains are therefore taken against the commanded value as
it stands, and a correct decomposition reads 1.000.

    python analyze_probe_v5.py
"""
import io
import json
import math
import os

TRACE = os.path.join("oak_v4_evidence", "probe_trace.jsonl")
MARKS = os.path.join("oak_v4_evidence", "torso_probe_marks.json")
SETTLE = 0.40      # skip the first 40 % of each hold


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def mean(v):
    return sum(v) / len(v) if v else 0.0


def stdev(v):
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def main():
    rows = [json.loads(l) for l in io.open(TRACE, encoding="utf-8") if l.strip() and '"err"' not in l]
    marks = json.load(io.open(MARKS, encoding="utf-8"))["phases"]
    by = {}
    for r in rows:
        by.setdefault(r["seq"], []).append(r)

    print("=" * 118)
    print(" TORSO V5 -- controlled probe, steady state per phase (bone-local yaws, skinned rig)")
    print(" gain = bone-local yaw / commanded source yaw. A correct decomposition reads 1.000.")
    print("=" * 118)
    print(" %-10s | %7s %7s %8s | %8s %9s %9s | %7s %7s | %7s" %
          ("phase", "cmd hip", "cmd sh", "cmd twist", "Hips Y", "trunk sum", "chain sum",
           "hip gain", "sh gain", "armErr"))

    results = []
    for m in marks:
        lo, hi = m["seq0"], m["seq1"]
        span = hi - lo
        start = lo + int(span * SETTLE)
        sel = []
        for s in range(start, hi + 1):
            sel.extend(by.get(s, []))
        if len(sel) < 5:
            print(" %-10s | (no samples)" % m["name"])
            continue
        hipsY = mean([wrap(r["hipsY"]) for r in sel])
        trunk = mean([wrap(r["spineY"]) + wrap(r["chestY"]) + wrap(r["uchY"]) for r in sel])
        chain = hipsY + trunk
        arm = max(mean([r["armUpL"] for r in sel]), mean([r["armUpR"] for r in sel]))
        cmd_hip, cmd_sh = m["hip"], m["sh"]
        cmd_tw = cmd_sh - cmd_hip
        hg = (hipsY / cmd_hip) if abs(cmd_hip) > 1e-6 else float("nan")
        sg = (chain / cmd_sh) if abs(cmd_sh) > 1e-6 else float("nan")
        print(" %-10s | %7.1f %7.1f %8.1f | %8.2f %9.2f %9.2f | %7s %7s | %7.3f" %
              (m["name"], cmd_hip, cmd_sh, cmd_tw, hipsY, trunk, chain,
               ("%7.3f" % hg) if hg == hg else "    -- ",
               ("%7.3f" % sg) if sg == sg else "    -- ", arm))
        results.append((m["name"], cmd_hip, cmd_sh, hipsY, trunk, chain, hg, sg, arm))

    print("\n" + "-" * 118)
    print(" WHAT THE OLD (V4) COMPOSITION WOULD HAVE PRODUCED, from its own measured structure")
    print("   old:  Hips = 0.70*hipYaw ; Spine+Chest = (0.45+0.25)*shoulderYaw = 0.70*shoulderYaw")
    print("   so    chain = 0.70*hipYaw + 0.70*shoulderYaw   (two ABSOLUTE yaws summed)")
    print(" %-10s | %10s %10s | %10s %10s" %
          ("phase", "V5 chain", "V5 sh gain", "V4 chain", "V4 sh gain"))
    for name, ch, cs, hipsY, trunk, chain, hg, sg, arm in results:
        if abs(cs) < 1e-6:
            continue
        v4 = 0.70 * ch + 0.70 * cs
        print(" %-10s | %10.2f %10.3f | %10.2f %10.3f" %
              (name, chain, sg, v4, v4 / cs))

    print("\n" + "-" * 118)
    print(" ARM SAFETY under torso motion (source -> actual skinned upper arm, deg)")
    allarm = [max(r["armUpL"], r["armUpR"]) for r in rows]
    allarm.sort()
    print("   p50=%.4f  p95=%.4f  p99=%.4f  max=%.4f  over %d samples"
          % (allarm[len(allarm)//2], allarm[int(len(allarm)*0.95)],
             allarm[int(len(allarm)*0.99)], allarm[-1], len(allarm)))
    br = [max(r["bodyRelL"], r["bodyRelR"]) for r in rows]
    br.sort()
    print("   body-relative arm error: p50=%.2f p95=%.2f max=%.2f"
          % (br[len(br)//2], br[int(len(br)*0.95)], br[-1]))
    print("   (the probe holds the arms rigid IN THE SHOULDER FRAME, so a correct rig keeps this")
    print("    constant as the torso turns -- it is the 'arms stay attached to the body' invariant)")
    rej = [r["rej"] for r in rows]
    print("   trunk-gate rejections across the whole probe: %d (of %d applied frames)"
          % (max(rej) - min(rej), len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
