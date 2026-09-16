#!/usr/bin/env python3
"""F-19 §13-§15 — analyse the rendered-bone telemetry the Unity recorder wrote.

Design notes on the two measures that are easy to get wrong:

BONE LENGTH (§14). Measured between the WORLD positions of consecutive humanoid bones on the final
rendered skeleton. Under pure FK these are rig constants; any variation at all is an avatar defect,
and the brief is right that it must not be assumed to mean bone scale — so `lossyScale` is recorded
and reported alongside, which separates "the bone was scaled" from "the joint moved".

ELBOW / KNEE PLAUSIBILITY (§12A, §15). An unsigned flexion angle cannot distinguish a normal bend
from a backwards one, and inventing a signed convention per joint invites exactly the kind of
arbitrary choice this programme keeps having to retract. So the test used here is the one that
follows from anatomy without any convention at all: **a human elbow and knee are HINGES**. Their
bend normal — normalize(cross(proximal, distal)) — must stay essentially fixed in the PARENT bone's
own frame. A hinge gives a tight cluster; a retarget producing impossible elbows gives a bend normal
that wanders over the sphere. The spread of that normal is therefore the impossibility measure, and
it is signed-convention-free. Flexion magnitude is reported too, because folding past ~160 deg is
independently impossible.

    python tools/capture/f19_analyze_bones.py f19_evidence/<label>.jsonl
"""
import json
import math
import os
import sys
import statistics as st

CHAINS = [
    ("L upperArm->lowerArm", "LeftUpperArm", "LeftLowerArm"),
    ("L lowerArm->hand", "LeftLowerArm", "LeftHand"),
    ("R upperArm->lowerArm", "RightUpperArm", "RightLowerArm"),
    ("R lowerArm->hand", "RightLowerArm", "RightHand"),
    ("L upperLeg->lowerLeg", "LeftUpperLeg", "LeftLowerLeg"),
    ("L lowerLeg->foot", "LeftLowerLeg", "LeftFoot"),
    ("R upperLeg->lowerLeg", "RightUpperLeg", "RightLowerLeg"),
    ("R lowerLeg->foot", "RightLowerLeg", "RightFoot"),
    ("spine->chest", "Spine", "Chest"),
    ("chest->upperChest", "Chest", "UpperChest"),
    ("neck->head", "Neck", "Head"),
]
HINGES = [
    ("L elbow", "LeftUpperArm", "LeftLowerArm", "LeftHand"),
    ("R elbow", "RightUpperArm", "RightLowerArm", "RightHand"),
    ("L knee", "LeftUpperLeg", "LeftLowerLeg", "LeftFoot"),
    ("R knee", "RightUpperLeg", "RightLowerLeg", "RightFoot"),
]
# A joint must be bent by at least this much before its bend normal means anything.
BEND_MIN_DEG = 15.0
# Folding past this is independently impossible for a human elbow or knee.
FOLD_MAX_DEG = 155.0

TRACKED = ["Hips", "Spine", "Chest", "UpperChest", "Neck", "Head",
           "LeftUpperArm", "LeftLowerArm", "LeftHand",
           "RightUpperArm", "RightLowerArm", "RightHand",
           "LeftUpperLeg", "LeftLowerLeg", "LeftFoot",
           "RightUpperLeg", "RightLowerLeg", "RightFoot"]


def sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def norm(a):
    return math.sqrt(sum(x * x for x in a))


def unit(a):
    n = norm(a)
    return [x / n for x in a] if n > 1e-9 else None


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def ang(a, b):
    d = max(-1.0, min(1.0, dot(a, b)))
    return math.degrees(math.acos(d))


def qconj(q):
    return [-q[0], -q[1], -q[2], q[3]]


def qrot(q, v):
    """Rotate v by quaternion q (x, y, z, w)."""
    x, y, z, w = q
    t = [2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])]
    return [v[0] + w * t[0] + (y * t[2] - z * t[1]),
            v[1] + w * t[1] + (z * t[0] - x * t[2]),
            v[2] + w * t[2] + (x * t[1] - y * t[0])]


def qangle(a, b):
    """Angle between two orientations, degrees."""
    d = abs(sum(a[i] * b[i] for i in range(4)))
    d = max(-1.0, min(1.0, d))
    return math.degrees(2.0 * math.acos(d))


def pct(xs, q):
    if not xs:
        return float("nan")
    ys = sorted(xs)
    return ys[min(len(ys) - 1, int(len(ys) * q))]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path or not os.path.exists(path):
        print("usage: f19_analyze_bones.py <telemetry.jsonl>", file=sys.stderr)
        return 2
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    if not rows:
        print("no records", file=sys.stderr)
        return 1

    frames = [r["f"] for r in rows]
    dts = [r.get("dt", 0.0) for r in rows]
    gaps = [frames[i] - frames[i - 1] for i in range(1, len(frames))]
    print(f"F-19 rendered-bone telemetry: {os.path.basename(path)}")
    print(f"  {len(rows)} records, frames {frames[0]}..{frames[-1]}, "
          f"frame gaps: {min(gaps) if gaps else 0}..{max(gaps) if gaps else 0} "
          f"(1 = every rendered frame captured)")
    print(f"  mean dt {st.mean(dts)*1000:.2f} ms  => {1.0/max(1e-9, st.mean(dts)):.1f} fps")
    valid = sum(1 for r in rows if r.get("srcValid"))
    print(f"  frames with a valid source pose: {valid}/{len(rows)}")

    # ---------------------------------------------------------------- §14 bone length stability
    print("\n§14 BONE-LENGTH STABILITY (world positions of the final rendered skeleton)")
    print(f"  {'chain':26s} {'baseline':>9s} {'min':>9s} {'max':>9s} {'stdev':>9s} {'max dev':>9s}")
    worst_dev = 0.0
    worst_name = ""
    for label, a, b in CHAINS:
        L = []
        for r in rows:
            B = r.get("b", {})
            if a in B and b in B:
                L.append(norm(sub(B[b]["p"], B[a]["p"])))
        if len(L) < 2:
            continue
        base = st.median(L)
        dev = max(abs(x - base) for x in L) / base * 100.0 if base > 1e-9 else float("nan")
        if dev > worst_dev:
            worst_dev, worst_name = dev, label
        print(f"  {label:26s} {base:9.6f} {min(L):9.6f} {max(L):9.6f} "
              f"{st.pstdev(L):9.6f} {dev:8.4f}%")
    print(f"  worst deviation: {worst_dev:.4f}% ({worst_name})")

    # scale, so a length change can be attributed rather than assumed
    sc = {}
    for r in rows:
        for k, v in r.get("b", {}).items():
            s = v.get("s")
            if s:
                sc.setdefault(k, []).append(max(abs(s[0] - 1.0), abs(s[1] - 1.0), abs(s[2] - 1.0)))
    worst_scale = max((max(v) for v in sc.values()), default=0.0)
    print(f"  worst |lossyScale - 1| over every tracked bone and frame: {worst_scale:.6f} "
          f"({'no bone scaling' if worst_scale < 1e-4 else 'SCALING PRESENT'})")

    # ---------------------------------------------------------------- §15 hinge plausibility
    print("\n§15/§12A HINGE PLAUSIBILITY (bend normal in the parent frame; a hinge cannot wander)")
    print(f"  {'joint':10s} {'flex p50':>9s} {'flex p95':>9s} {'flex max':>9s} "
          f"{'spread p95':>11s} {'max':>8s} {'n_bent':>7s} {'n>fold':>7s}")
    hinge_out = {}
    for label, pa, pb, pc in HINGES:
        flex = []
        normals = []
        for r in rows:
            B = r.get("b", {})
            if pa not in B or pb not in B or pc not in B:
                continue
            u = unit(sub(B[pb]["p"], B[pa]["p"]))
            f = unit(sub(B[pc]["p"], B[pb]["p"]))
            if u is None or f is None:
                continue
            flex.append(ang(u, f))          # 0 = straight, grows as the joint folds
            # The bend normal is only defined once the joint is actually bent: cross(u, f) -> 0
            # as the limb straightens and its DIRECTION becomes pure noise there. This is the same
            # degeneracy ArmAimSolver guards with BendSinMin; measuring hinge wander on a straight
            # limb would report ~100 deg of "impossibility" for a perfectly normal relaxed arm.
            if ang(u, f) < BEND_MIN_DEG:
                continue
            n = unit(cross(u, f))
            if n is None:
                continue
            normals.append(qrot(qconj(B[pa]["w"]), n))
        if not flex:
            continue
        med = None
        spread = []
        if len(normals) > 5:
            mx = [st.median([n[i] for n in normals]) for i in range(3)]
            med = unit(mx)
            if med:
                spread = [ang(med, n) for n in normals]
        over = sum(1 for x in flex if x > FOLD_MAX_DEG)
        hinge_out[label] = dict(flex_p50=pct(flex, 0.5), flex_p95=pct(flex, 0.95),
                                flex_max=max(flex),
                                spread_p95=pct(spread, 0.95) if spread else float("nan"),
                                spread_max=max(spread) if spread else float("nan"),
                                over_fold=over, n=len(flex), n_bent=len(normals))
        h = hinge_out[label]
        print(f"  {label:10s} {h['flex_p50']:9.2f} {h['flex_p95']:9.2f} {h['flex_max']:9.2f} "
              f"{h['spread_p95']:11.2f} {h['spread_max']:8.2f} {h['n_bent']:7d} {over:7d}")
    print("  (flex 0 = straight; bend-normal spread near 0 = a clean hinge, tens of degrees = the")
    print("   joint is bending in a direction a human cannot)")

    # ---------------------------------------------------------------- §13 temporal
    print("\n§13 TEMPORAL: frame-to-frame WORLD rotation step per bone (deg)")
    print(f"  {'bone':18s} {'p50':>7s} {'p95':>7s} {'p99':>7s} {'max':>8s} {'>10deg':>7s} {'frozen%':>8s}")
    temporal = {}
    for name in TRACKED:
        steps = []
        for i in range(1, len(rows)):
            A = rows[i - 1].get("b", {}).get(name)
            Bq = rows[i].get("b", {}).get(name)
            if not A or not Bq:
                continue
            steps.append(qangle(A["w"], Bq["w"]))
        if not steps:
            continue
        frozen = sum(1 for s in steps if s < 1e-4) / len(steps) * 100.0
        big = sum(1 for s in steps if s > 10.0)
        temporal[name] = dict(p50=pct(steps, 0.5), p95=pct(steps, 0.95), p99=pct(steps, 0.99),
                              mx=max(steps), big=big, frozen=frozen, n=len(steps))
        t = temporal[name]
        print(f"  {name:18s} {t['p50']:7.3f} {t['p95']:7.3f} {t['p99']:7.3f} {t['mx']:8.3f} "
              f"{big:7d} {frozen:7.1f}%")

    # ---------------------------------------------------------------- §12D cross-body
    print("\n§12D CROSS-BODY: does either hand end up on the wrong side of the body?")
    swaps = 0
    n = 0
    for r in rows:
        B = r.get("b", {})
        if not all(k in B for k in ("LeftHand", "RightHand", "LeftUpperArm", "RightUpperArm", "Hips")):
            continue
        # lateral axis of the shoulder line, and each hand's position along it
        lat = unit(sub(B["LeftUpperArm"]["p"], B["RightUpperArm"]["p"]))
        if lat is None:
            continue
        mid = [(B["LeftUpperArm"]["p"][i] + B["RightUpperArm"]["p"][i]) * 0.5 for i in range(3)]
        dl = dot(sub(B["LeftHand"]["p"], mid), lat)
        dr = dot(sub(B["RightHand"]["p"], mid), lat)
        n += 1
        if dl < dr:                 # the left hand is further toward the body's RIGHT than the right hand
            swaps += 1
    print(f"  {n} frames; {swaps} where the hands are laterally crossed ({swaps/max(1,n)*100:.2f}%)")
    print("  (crossing is a legitimate human pose - this counts occurrences, it is not a defect count)")

    # ---------------------------------------------------------------- torso, the measured part
    print("\nTORSO (driven by MEASURED stereo depth in this replay)")
    yaws = []
    for r in rows:
        B = r.get("b", {})
        if "LeftUpperArm" not in B or "RightUpperArm" not in B:
            continue
        d = sub(B["LeftUpperArm"]["p"], B["RightUpperArm"]["p"])
        flat = unit([d[0], 0.0, d[2]])
        if flat:
            yaws.append(math.degrees(math.atan2(flat[2], flat[0])))
    if yaws:
        steps = [abs((yaws[i] - yaws[i - 1] + 180) % 360 - 180) for i in range(1, len(yaws))]
        print(f"  avatar shoulder-line yaw: p50 {pct([abs(y) for y in yaws],0.5):.2f} deg  "
              f"range {min(yaws):.2f}..{max(yaws):.2f}")
        print(f"  frame-to-frame yaw step:  p50 {pct(steps,0.5):.3f}  p95 {pct(steps,0.95):.3f}  "
              f"max {max(steps):.3f} deg   (>10 deg snaps: {sum(1 for s in steps if s>10)})")

    out = path.replace(".jsonl", "_analysis.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"file": path, "records": len(rows), "worst_bone_dev_pct": worst_dev,
                   "worst_scale_dev": worst_scale, "hinges": hinge_out, "temporal": temporal,
                   "hand_cross_frames": swaps, "hand_cross_total": n}, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
