#!/usr/bin/env python3
"""TORSO YAW V4 -- controlled scale sweep on the SAME video, measured in the LIVE Unity rig.

Five passes of `video.webm` were streamed over the real UDP wire, one per torsoYawScale, so every
scale sees BYTE-IDENTICAL source input (376 packets, same frames, same order). That is what a live
human A/B cannot give: in V4's original held-frame sweep the mechanism was measured at steady state,
which flatters it ~9x; here it is measured in motion.

Alignment: the sender seeds `seq` from the wall clock, so each pass has its own base and increments
by 1 per video frame. Passes are segmented on seq discontinuities and indexed by
`frameIdx = seq - passBaseSeq`, which lines the five passes up exactly.

Per source frame the LAST editor sample for that seq is used -- i.e. after the per-frame
`lerpAmount = 0.5` slerp has had every editor frame of that packet's lifetime to converge. Taking
the first would measure slerp lag rather than the torso mechanism.

Everything is read from the ACTUAL SKINNED VRM bones via Vrm10Instance.Humanoid, never the control
rig (the V3 correction).

    python analyze_video_sweep_v4.py
"""
import io
import json
import math
import os

TRACE = os.path.join("oak_v4_evidence", "video_trace.jsonl")


def pct(v, p):
    if not v:
        return 0.0
    s = sorted(v)
    return s[int(round((len(s) - 1) * p / 100.0))]


def mean(v):
    return sum(v) / len(v) if v else 0.0


def stdev(v):
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def load():
    rows = []
    for line in io.open(TRACE, encoding="utf-8"):
        line = line.strip()
        if not line or '"err"' in line:
            continue
        rows.append(json.loads(line))
    rows.sort(key=lambda r: r["wall"])
    return rows


def segment(rows):
    """Split into passes on seq discontinuity; return [(scale, {frameIdx: sample})]."""
    passes, cur, prev = [], [], None
    for r in rows:
        if prev is not None and (r["seq"] < prev or r["seq"] - prev > 50):
            if cur:
                passes.append(cur)
            cur = []
        cur.append(r)
        prev = r["seq"]
    if cur:
        passes.append(cur)

    out = []
    for p in passes:
        moving = [r for r in p if r["seq"] != p[0]["seq"]]
        if len(moving) < 200:                      # idle gap between passes, not a real pass
            continue
        base = min(r["seq"] for r in p)
        byidx = {}
        for r in p:
            byidx[r["seq"] - base] = r             # last sample for this source frame wins
        # the scale actually in force is the modal one over the moving part
        scales = {}
        for r in moving:
            scales[r["scale"]] = scales.get(r["scale"], 0) + 1
        scale = max(scales.items(), key=lambda kv: kv[1])[0]
        out.append((scale, byidx))
    return out


def main():
    rows = load()
    passes = segment(rows)
    print("=" * 112)
    print(" VIDEO SCALE SWEEP -- identical input, live skinned VRM bones (%d samples, %d passes)"
          % (len(rows), len(passes)))
    print("=" * 112)
    if not passes:
        print("no usable passes")
        return 1

    # frames common to every pass
    common = set(passes[0][1].keys())
    for _s, d in passes[1:]:
        common &= set(d.keys())
    common = sorted(common)
    print(" frames common to all passes: %d" % len(common))

    # sanity: the source must be identical across passes at the same frame index
    base_scale, base = passes[0]
    src_dev = []
    for s, d in passes[1:]:
        for i in common:
            src_dev.append(abs(wrap(d[i]["srcSh"] - base[i]["srcSh"])))
    print(" source shoulder yaw reproducibility across passes: mean %.4f deg, max %.4f deg" %
          (mean(src_dev), max(src_dev) if src_dev else 0.0))
    print(" (this must be ~0 -- it is what proves the five scales saw the same input)")

    # avatar frontal baseline: the scale-0 pass, whose torso never moves
    zero = None
    for s, d in passes:
        if abs(s) < 1e-6:
            zero = d
    ref = mean([zero[i]["avSh"] for i in common]) if zero else 0.0

    print("\n %6s | %8s %8s | %8s %7s | %8s %8s | %9s %9s | %8s" %
          ("scale", "avSh sd", "avSh max", "gain", "err p50", "err p95", "err max",
           "armL max", "armR max", "handRad"))
    print(" %6s | %8s %8s | %8s %7s | %8s %8s | %9s %9s | %8s" %
          ("", "deg", "deg", "", "deg", "deg", "deg", "deg", "deg", "m"))
    for s, d in sorted(passes, key=lambda x: x[0]):
        av = [wrap(d[i]["avSh"] - ref) for i in common]
        src = [d[i]["srcSh"] for i in common]
        num = sum(a * x for a, x in zip(av, src))
        den = sum(x * x for x in src)
        gain = num / den if den > 1e-9 else 0.0
        err = [abs(wrap(a - x)) for a, x in zip(av, src)]
        armL = [d[i]["armL"] for i in common]
        armR = [d[i]["armR"] for i in common]
        rad = [math.hypot(d[i]["hLx"] - d[i]["spX"], d[i]["hLz"] - d[i]["spZ"]) for i in common]
        print(" %6.2f | %8.2f %8.2f | %8.3f %7.2f | %8.2f %8.2f | %9.4f %9.4f | %8.4f" %
              (s, stdev(av), max(abs(x) for x in av), gain, pct(err, 50), pct(err, 95), max(err),
               max(armL), max(armR), mean(rad)))

    # the frame V3 called out: the worst torso-yaw error in the clip
    print("\n --- the worst-yaw frames (largest |source shoulder yaw|) ---")
    worst = sorted(common, key=lambda i: -abs(base[i]["srcSh"]))[:5]
    print(" %8s %10s |" % ("frameIdx", "src yaw"), end="")
    for s, _d in sorted(passes, key=lambda x: x[0]):
        print(" %7.2f" % s, end="")
    print("   <- avatar shoulder yaw (deg)")
    for i in worst:
        print(" %8d %10.2f |" % (i, base[i]["srcSh"]), end="")
        for s, d in sorted(passes, key=lambda x: x[0]):
            print(" %7.2f" % wrap(d[i]["avSh"] - ref), end="")
        print("")

    print("\n --- hand distance from the spine axis at those frames (m) ---")
    print(" %8s |" % "frameIdx", end="")
    for s, _d in sorted(passes, key=lambda x: x[0]):
        print(" %7.2f" % s, end="")
    print("   <- larger = hand further OUT of the torso")
    for i in worst:
        print(" %8d |" % i, end="")
        for s, d in sorted(passes, key=lambda x: x[0]):
            print(" %7.4f" % math.hypot(d[i]["hLx"] - d[i]["spX"], d[i]["hLz"] - d[i]["spZ"]), end="")
        print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
