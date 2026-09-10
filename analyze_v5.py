#!/usr/bin/env python3
"""TORSO V5 -- measure the new relative-yaw composition in the live Unity rig.

Reads `oak_v4_evidence/v5_trace.jsonl`, sampled from inside `EditorApplication.update` against the
ACTUAL SKINNED VRM bones (Vrm10Instance.Humanoid).

HOW THE COMPOSITION IS SCORED, and why not with the shoulder-line gain
---------------------------------------------------------------------
`avSh` (shoulder-line world yaw) and `avHip` (hip-line world yaw) are read off different bone pairs
with different rest orientations, so their DIFFERENCE is not a twist and their absolute values need
per-run baselines. The composition is therefore scored on the BONE-LOCAL yaws, which are exactly what
the driver writes and carry no such ambiguity:

    hipsY                            <- must equal the conditioned ABSOLUTE hip yaw
    spineY + chestY + upperChestY    <- must equal the conditioned RELATIVE twist (shoulder - hip)
    hipsY + that sum                 <- must equal the conditioned shoulder yaw (bones nest)

The defining property of V5, and the one the old form failed, is DEAD-ZONE INDEPENDENT:

    on a RIGID turn (source hip == source shoulder) the upper-trunk sum must be ~0,
    so the body rotates once instead of being multiplied.

Under the old absolute-sum form the same rigid turn wrote 0.70*shoulderYaw into Spine+Chest on top of
0.70*hipYaw at the Hips, which is where the measured 1.4x came from.

A separate `gain` column IS reported, but only over frames whose source clears the ADR-027 dead zone
(|yaw| > 22 deg); below the knee the conditioner deliberately outputs ~0 and a gain computed there
measures the dead zone, not the composition.

    python analyze_v5.py
"""
import io
import json
import math
import os

TRACE = os.path.join("oak_v4_evidence", "v5_trace.jsonl")
DEADZONE_HI = 22.0
REASONS = {0: "None", 1: "NotFinite", 2: "ShoulderSpan", 3: "HipSpan",
           4: "SpanImplausible", 5: "YawJump"}


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
    return rows


def segment(rows):
    segs, cur, prev = [], [rows[0]], rows[0]["seq"]
    for r in rows[1:]:
        if r["seq"] < prev or r["seq"] - prev > 50:
            segs.append(cur)
            cur = []
        cur.append(r)
        prev = r["seq"]
    segs.append(cur)
    return segs


def per_frame(seg):
    """One sample per source frame -- the LAST, i.e. after the lerpAmount=0.5 slerp has converged."""
    by = {}
    for r in seg:
        by[r["seq"]] = r
    return [by[k] for k in sorted(by)]


def twist_sum(r):
    return wrap(r["spineY"]) + wrap(r["chestY"]) + wrap(r["uchY"])


def gain(av, src):
    den = sum(x * x for x in src)
    return (sum(a * x for a, x in zip(av, src)) / den) if den > 1e-9 else float("nan")


def report_composition(rows):
    print("\n TASK 6 -- composition measured on the BONE-LOCAL yaws the driver actually writes")
    print(" %-30s %5s | %8s %8s | %8s %9s %9s | %8s" %
          ("condition", "n", "src hip", "src sh", "Hips Y", "trunk sum", "src twist", "bodyRel"))
    bins = [
        ("rigid turn      (|sh-hip|<5)", lambda r: abs(r["srcSh"] - r["srcHip"]) < 5.0),
        ("torso twist     (|sh-hip|>15)", lambda r: abs(r["srcSh"] - r["srcHip"]) > 15.0),
        ("opposite signs  (sh*hip<0)", lambda r: r["srcSh"] * r["srcHip"] < 0
                                                 and abs(r["srcSh"]) > 5 and abs(r["srcHip"]) > 5),
        ("ALL FRAMES", lambda r: True),
    ]
    for name, pred in bins:
        sel = [r for r in rows if pred(r)]
        if len(sel) < 5:
            print(" %-30s %5d | (too few frames)" % (name, len(sel)))
            continue
        print(" %-30s %5d | %8.2f %8.2f | %8.2f %9.2f %9.2f | %8.2f" %
              (name, len(sel), mean([r["srcHip"] for r in sel]), mean([r["srcSh"] for r in sel]),
               mean([wrap(r["hipsY"]) for r in sel]), mean([twist_sum(r) for r in sel]),
               mean([r["srcSh"] - r["srcHip"] for r in sel]),
               mean([max(r["bodyRelL"], r["bodyRelR"]) for r in sel])))

    print("\n THE DEFINING PROPERTY -- a rigid turn must NOT be multiplied")
    rigid = [r for r in rows if abs(r["srcSh"] - r["srcHip"]) < 5.0]
    if rigid:
        ts = [abs(twist_sum(r)) for r in rigid]
        print("   |upper-trunk twist| on rigid frames: mean %.3f deg, p95 %.3f, max %.3f  (n=%d)"
              % (mean(ts), pct(ts, 95), max(ts), len(rigid)))
        print("   the OLD form would have written 0.70*shoulderYaw here instead of ~0.")

    print("\n GAIN, restricted to frames whose source clears the %.0f deg dead-zone knee" % DEADZONE_HI)
    above = [r for r in rows if abs(r["srcSh"]) > DEADZONE_HI]
    if len(above) >= 5:
        base = mean([r["avSh"] for r in rows[:40]])
        g = gain([wrap(r["avSh"] - base) for r in above], [r["srcSh"] for r in above])
        print("   n=%d   shoulder-line gain = %.3f" % (len(above), g))
    else:
        print("   n=%d  -- this clip has too few frames above the knee to measure a gain here" % len(above))
        print("   (video.webm: source |yaw| p50 4.2 deg, only ~2.7%% of frames clear 22 deg)")


def report_arms(rows, label):
    print("\n TASK 4 -- arm regression, source -> ACTUAL SKINNED bone (deg)  [%s]" % label)
    print(" %-22s %8s %8s %8s %8s" % ("quantity", "p50", "p95", "p99", "max"))
    for key, name in (("armUpL", "upper arm L"), ("armUpR", "upper arm R"),
                      ("armFoL", "forearm L"), ("armFoR", "forearm R")):
        v = [r[key] for r in rows]
        print(" %-22s %8.4f %8.4f %8.4f %8.4f" % (name, pct(v, 50), pct(v, 95), pct(v, 99), max(v)))
    for key, name in (("elbErrL", "elbow angle err L"), ("elbErrR", "elbow angle err R")):
        v = [abs(r[key]) for r in rows]
        print(" %-22s %8.4f %8.4f %8.4f %8.4f" % (name, pct(v, 50), pct(v, 95), pct(v, 99), max(v)))
    for key, name in (("bodyRelL", "body-rel upper L"), ("bodyRelR", "body-rel upper R")):
        v = [r[key] for r in rows]
        print(" %-22s %8.2f %8.2f %8.2f %8.2f" % (name, pct(v, 50), pct(v, 95), pct(v, 99), max(v)))


def report_ghost(seg):
    """real -> all-zero landmarks -> real, on a live wire. The +90 sentinel must never appear."""
    print("\n TASK 1 -- false-detection (ghost) behaviour")
    sent = [r for r in seg if abs(r["srcSh"] - 90.0) < 0.01]
    real = [r for r in seg if abs(r["srcSh"] - 90.0) >= 0.01]
    if not sent:
        print("   no sentinel frames found in this segment")
        return
    # index-ordered phases
    first_sent = min(i for i, r in enumerate(seg) if abs(r["srcSh"] - 90.0) < 0.01)
    last_sent = max(i for i, r in enumerate(seg) if abs(r["srcSh"] - 90.0) < 0.01)
    pre, ghost, post = seg[:first_sent], seg[first_sent:last_sent + 1], seg[last_sent + 1:]
    base = mean([wrap(r["hipsY"]) for r in pre[:60]]) if len(pre) > 60 else 0.0

    def blk(name, sel):
        if not sel:
            print("   %-20s (empty)" % name)
            return
        h = [wrap(r["hipsY"]) for r in sel]
        t = [twist_sum(r) for r in sel]
        chain = [a + b for a, b in zip(h, t)]
        src = [r["srcSh"] for r in sel]
        near = sum(1 for x in chain if abs(abs(x) - 90.0) < 15.0)
        print("   %-20s n=%5d | src sh %7.2f..%7.2f | Hips Y %6.2f..%6.2f | chain %7.2f..%7.2f | sd %5.2f | within 15deg of +/-90: %d"
              % (name, len(sel), min(src), max(src), min(h), max(h), min(chain), max(chain), stdev(chain), near))

    blk("REAL (before)", pre)
    blk("GHOST (no person)", ghost)
    blk("REAL (after)", post)
    seen = {}
    for r in ghost:
        seen[r["rejWhy"]] = seen.get(r["rejWhy"], 0) + 1
    print("   gate reject reasons during ghost: %s"
          % ", ".join("%s=%d" % (REASONS.get(k, k), v) for k, v in sorted(seen.items())))
    print("   SOURCE during ghost is the +90 sentinel (mean %.2f deg) -- that IS the poisoned input."
          % mean([r["srcSh"] for r in ghost]))
    # rest jitter while the ghost holds
    if ghost:
        print("   avatar chain-yaw stdev while holding: %.4f deg  (0 = perfectly frozen)"
              % stdev([wrap(r["hipsY"]) + twist_sum(r) for r in ghost]))


def main():
    rows = load()
    segs = segment(rows)
    print("=" * 116)
    print(" TORSO V5 -- live rig, skinned bones. %d samples, %d segments" % (len(rows), len(segs)))
    print("=" * 116)
    for i, s in enumerate(segs):
        f = per_frame(s)
        has_sentinel = any(abs(r["srcSh"] - 90.0) < 0.01 for r in f)
        kind = "GHOST test" if has_sentinel else "VIDEO pass"
        print("\n" + "-" * 116)
        print(" SEGMENT %d (%s): %d source frames, scale=%.2f" % (i, kind, len(f), f[0]["scale"]))
        print("-" * 116)
        if kind == "VIDEO pass":
            report_composition(f)
            report_arms(f, "video.webm")
        else:
            report_ghost(s)
            real = [r for r in f if abs(r["srcSh"] - 90.0) >= 0.01]
            if real:
                report_arms(real, "ghost run, REAL phases only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
