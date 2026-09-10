#!/usr/bin/env python3
"""F-10 (pre-capture) -- the two questions answerable WITHOUT ground truth.

The F-10 brief's decision criteria 1 and 2 are:
    1. does 2-D magnitude have materially better angular RESOLUTION than depth yaw?
    2. does the depth SIGN remain reliable?

Criterion 1 is a property of the sensors and the geometry, not of any particular subject, so it can be
computed from the recorded captures directly. Criterion 2 is a stability property that can be bounded
from the recordings (a sign that flips between adjacent frames at large |yaw| cannot be reliable,
whatever the ground truth says). Neither needs a protractor.

Everything that requires knowing the TRUE heading -- accuracy, bias, hybrid MAE/RMSE, distance
sensitivity, twist -- is NOT computed here and is left to the ground-truth capture.

RESOLUTION MODEL, derived rather than assumed
---------------------------------------------
2-D:  cos(t) = uSpan / uSpanMax        (at fixed distance; uSpanMax = the face-on pixel span)
      dt/d(uSpan) = -1 / (uSpanMax * sin t)
      -> one pixel of span error costs  1/(uSpanMax*sin t)  radians.
      BEST near 90 deg, WORST near 0 deg (cos is flat there).

3-D:  yaw = atan2(dx, dz), and dz moves only in whole disparity steps.
      One depth step across a shoulder line of length L costs  ~ atan(step/L)  radians near 0 deg.
      Roughly CONSTANT in yaw, and set by the disparity quantisation.

The two therefore fail in opposite regimes, which is the whole basis of the hybrid proposal.

    python f10_resolution_sign.py
"""
import io
import json
import math
import os

FEAT = os.path.join("oak_v4_evidence", "f09_features.jsonl")


def pct(v, p):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[int(round((len(s) - 1) * p / 100.0))]


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def wrapd(a):
    return (a + 180.0) % 360.0 - 180.0


def main():
    rows = [json.loads(l) for l in io.open(FEAT, encoding="utf-8") if l.strip()]
    caps = {}
    for r in rows:
        caps.setdefault(r["cap"], []).append(r)
    for c in caps:
        caps[c].sort(key=lambda r: r["seq"])

    print("=" * 106)
    print(" F-10 PRE-CAPTURE -- criteria 1 and 2, from %d recorded frames (no ground truth needed)" % len(rows))
    print("=" * 106)

    # ---------------- criterion 1: angular resolution -----------------------------------
    print("\n CRITERION 1 -- angular resolution of each estimator, at the recorded geometry")
    print(" (a 1-pixel span error for 2-D; one disparity step for 3-D)")
    uspan_max = pct([r["uSpanSh"] for r in rows], 98)
    span_m = pct([r["shSpan3D"] for r in rows], 50)
    # measured disparity step, per capture (from f09)
    steps = []
    for cap, rs in caps.items():
        lv = sorted(set(round(r["midShZraw"] * 2 - r["shDzRaw"], 1) / 2.0 for r in rs))
        d = [lv[i] - lv[i - 1] for i in range(1, len(lv)) if lv[i] - lv[i - 1] > 1.0]
        if d:
            steps.append(pct(d, 50))
    step_mm = mean(steps)
    print("   face-on shoulder pixel span (p98)      : %.1f px" % uspan_max)
    print("   median 3-D shoulder length             : %.3f m" % span_m)
    print("   median stereo depth step               : %.1f mm" % step_mm)
    print("\n   %8s | %-22s | %-22s | %s" %
          ("true yaw", "2-D  deg per 1 px", "3-D  deg per depth step", "which is finer"))
    for t in (0, 5, 10, 20, 30, 45, 60, 75, 90):
        st = math.sin(math.radians(t))
        r2 = (1.0 / (uspan_max * st)) * 180.0 / math.pi if st > 1e-6 else float("inf")
        # 3-D: d(yaw)/d(dz) with dz = L*sin(t), dx = L*cos(t): yaw = atan2(dz,dx) -> dyaw/ddz = cos(t)/L
        ct = math.cos(math.radians(t))
        r3 = (step_mm / 1000.0) * ct / span_m * 180.0 / math.pi
        better = "2-D" if r2 < r3 else "3-D"
        print("   %6d   | %20.2f   | %20.2f   | %s" %
              (t, r2 if r2 != float("inf") else 999.99, r3, better))
    print("\n   The two are complementary: 3-D is finer below ~%d deg, 2-D is finer above it." %
          _crossover(uspan_max, step_mm, span_m))
    print("   That crossover is the single most important number for the hybrid design, and it is")
    print("   a property of the GEOMETRY, so the ground-truth capture must confirm it, not discover it.")

    # ---------------- criterion 2: sign stability ---------------------------------------
    print("\n" + "-" * 106)
    print(" CRITERION 2 -- is the DEPTH SIGN stable? (a sign that flips between adjacent frames")
    print(" cannot be reliable regardless of ground truth)")
    print("\n %-12s | %8s | %10s | %10s | %s" %
          ("|yaw3D| band", "frames", "sign flips", "flip rate", "adjacent-frame sign reversals"))
    bands = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 45), (45, 90), (90, 180)]
    for lo, hi in bands:
        n = flips = 0
        for cap, rs in caps.items():
            for i in range(1, len(rs)):
                a, b = rs[i - 1], rs[i]
                m = min(abs(a["yaw3D"]), abs(b["yaw3D"]))
                if not (lo <= m < hi):
                    continue
                dt = b["t"] - a["t"]
                if not (0 < dt < 0.5):
                    continue
                n += 1
                if a["yaw3D"] * b["yaw3D"] < 0:
                    flips += 1
        if n < 20:
            print(" %5d-%-6d | %8d | (too few)" % (lo, hi, n))
            continue
        print(" %5d-%-6d | %8d | %10d | %9.3f%% |" % (lo, hi, n, flips, 100.0 * flips / n))
    print("\n A sign flip between adjacent frames while |yaw| is well away from 0 is a measurement")
    print(" failure, not motion: the subject cannot cross from -30 to +30 deg in one frame.")

    # sign vs the 2-D magnitude: when 2-D says 'clearly turned', how often does 3-D sign wobble?
    print("\n" + "-" * 106)
    print(" SIGN STABILITY restricted to frames where the IMAGE agrees a real turn is happening")
    print(" (yaw2D > 25 deg), i.e. the regime the hybrid would actually rely on the sign in:")
    n = flips = 0
    for cap, rs in caps.items():
        for i in range(1, len(rs)):
            a, b = rs[i - 1], rs[i]
            if a["yaw2D"] < 25 or b["yaw2D"] < 25:
                continue
            dt = b["t"] - a["t"]
            if not (0 < dt < 0.5):
                continue
            n += 1
            if a["yaw3D"] * b["yaw3D"] < 0:
                flips += 1
    if n:
        print("   %d frame pairs, %d sign reversals = %.3f%%" % (n, flips, 100.0 * flips / n))
        print("   NOTE: this bounds sign STABILITY only. Sign CORRECTNESS cannot be measured without")
        print("   knowing which way the subject actually faced -- that is what the capture is for.")
    return 0


def _crossover(uspan_max, step_mm, span_m):
    """Yaw at which 2-D resolution becomes finer than 3-D."""
    for t in range(1, 90):
        st, ct = math.sin(math.radians(t)), math.cos(math.radians(t))
        r2 = 1.0 / (uspan_max * st)
        r3 = (step_mm / 1000.0) * ct / span_m
        if r2 < r3:
            return t
    return 90


if __name__ == "__main__":
    raise SystemExit(main())
