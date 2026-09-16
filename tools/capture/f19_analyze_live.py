#!/usr/bin/env python3
"""F-19 sections 12-15 and 20 - analyse LIVE rendered-bone telemetry, per protocol block.

This is the live counterpart to f19_analyze_bones.py. The difference is not cosmetic: that file
analysed a REPLAY of F-18 landmark captures, in which elbow, wrist, knee and ankle depth had to be
assumed because F-18 never stored them, so every distal conclusion was explicitly not attributable.
Here the sidecar measured all 33 landmarks live with real stereo depth, so the distal joints ARE
attributable and section 12A can finally be answered.

Two measures that are easy to get wrong, handled the same way as the replay analyser:

  BONE LENGTH (section 14) is measured between WORLD positions on the final rendered skeleton, and
  lossyScale is reported beside it so a length change can be ATTRIBUTED (bone scaled) rather than
  assumed. Under pure FK these are rig constants.

  ELBOW / KNEE PLAUSIBILITY (sections 12A, 15). An unsigned flexion angle cannot tell a normal bend
  from a backwards one, and inventing a per-joint signed convention invites exactly the kind of
  arbitrary choice this programme keeps retracting. The convention-free test is that a human elbow
  and knee are HINGES: their bend normal, normalize(cross(proximal, distal)), must stay essentially
  fixed in the PARENT bone's frame. A clean hinge gives a tight cluster; an impossible joint gives a
  normal that wanders. The normal is undefined on a straight limb (the cross product vanishes), so
  only frames bent past BEND_MIN_DEG contribute - without that guard a perfectly normal relaxed arm
  reports tens of degrees of "impossibility".

    python tools/capture/f19_analyze_live.py evidence/oak_v4/f19/bones_motion.jsonl
"""
import io
import json
import math
import os
import statistics as st
import sys

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
TRACKED = ["Hips", "Spine", "Chest", "UpperChest", "Neck", "Head",
           "LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftHand",
           "RightShoulder", "RightUpperArm", "RightLowerArm", "RightHand",
           "LeftUpperLeg", "LeftLowerLeg", "LeftFoot",
           "RightUpperLeg", "RightLowerLeg", "RightFoot"]
BEND_MIN_DEG = 15.0
FOLD_MAX_DEG = 155.0
SNAP_DEG = 10.0


def sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def norm(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def unit(a):
    n = norm(a)
    return [a[0] / n, a[1] / n, a[2] / n] if n > 1e-9 else None


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def ang(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, dot(a, b)))))


def qconj(q):
    return [-q[0], -q[1], -q[2], q[3]]


def qrot(q, v):
    x, y, z, w = q
    t = [2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])]
    return [v[0] + w * t[0] + (y * t[2] - z * t[1]),
            v[1] + w * t[1] + (z * t[0] - x * t[2]),
            v[2] + w * t[2] + (x * t[1] - y * t[0])]


def qangle(a, b):
    d = abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, d))))


def pct(xs, q):
    if not xs:
        return float("nan")
    ys = sorted(xs)
    return ys[min(len(ys) - 1, int(len(ys) * q))]


def shoulder_yaw(r):
    B = r["b"]
    if "LeftUpperArm" not in B or "RightUpperArm" not in B:
        return None
    d = sub(B["LeftUpperArm"]["p"], B["RightUpperArm"]["p"])
    n = math.hypot(d[0], d[2])
    return math.degrees(math.atan2(d[2] / n, d[0] / n)) if n > 1e-9 else None


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def block_report(name, rows, out):
    if len(rows) < 3:
        return None
    res = {"block": name, "frames": len(rows)}
    dts = [r.get("dt", 0.0) for r in rows]
    fps = 1.0 / max(1e-9, st.mean(dts))
    res["fps"] = fps

    # ---- section 20 continuity, on the rendered avatar -------------------------------------
    yaws = [y for y in (shoulder_yaw(r) for r in rows) if y is not None]
    steps = [abs(wrap(yaws[i] - yaws[i - 1])) for i in range(1, len(yaws))]
    res["yaw_min"], res["yaw_max"] = (min(yaws), max(yaws)) if yaws else (float("nan"),) * 2
    res["yaw_step_p95"] = pct(steps, 0.95)
    res["yaw_step_max"] = max(steps) if steps else float("nan")
    res["snaps"] = sum(1 for s in steps if s > SNAP_DEG)

    # ---- section 14 bone length --------------------------------------------------------------
    worst, wname = 0.0, ""
    for label, a, b in CHAINS:
        L = [norm(sub(r["b"][b]["p"], r["b"][a]["p"])) for r in rows if a in r["b"] and b in r["b"]]
        if len(L) < 2:
            continue
        base = st.median(L)
        dev = max(abs(x - base) for x in L) / base * 100.0 if base > 1e-9 else float("nan")
        if dev > worst:
            worst, wname = dev, label
    res["bone_dev_pct"], res["bone_dev_chain"] = worst, wname
    res["scale_dev"] = max((max(abs(v["s"][i] - 1.0) for i in range(3))
                            for r in rows for v in r["b"].values()), default=0.0)

    # ---- sections 12A / 15 hinge plausibility ------------------------------------------------
    res["hinges"] = {}
    for label, pa, pb, pc in HINGES:
        flex, normals = [], []
        for r in rows:
            B = r["b"]
            if pa not in B or pb not in B or pc not in B:
                continue
            u = unit(sub(B[pb]["p"], B[pa]["p"]))
            f = unit(sub(B[pc]["p"], B[pb]["p"]))
            if u is None or f is None:
                continue
            a = ang(u, f)
            flex.append(a)
            if a < BEND_MIN_DEG:
                continue
            n = unit(cross(u, f))
            if n is not None:
                normals.append(qrot(qconj(B[pa]["w"]), n))
        if not flex:
            continue
        spread = []
        if len(normals) > 5:
            med = unit([st.median([n[i] for n in normals]) for i in range(3)])
            if med:
                spread = [ang(med, n) for n in normals]
        res["hinges"][label] = dict(
            flex_p50=pct(flex, 0.5), flex_p95=pct(flex, 0.95), flex_max=max(flex),
            spread_p95=pct(spread, 0.95) if spread else float("nan"),
            over_fold=sum(1 for x in flex if x > FOLD_MAX_DEG), n_bent=len(normals), n=len(flex))

    # ---- section 12C temporal ------------------------------------------------------------------
    worst_bone, worst_p95, worst_max, snap_total, frozen_worst, frozen_name = "", 0.0, 0.0, 0, 0.0, ""
    res["temporal"] = {}
    for nm in TRACKED:
        s = []
        for i in range(1, len(rows)):
            A = rows[i - 1]["b"].get(nm)
            Bq = rows[i]["b"].get(nm)
            if A and Bq:
                s.append(qangle(A["w"], Bq["w"]))
        if not s:
            continue
        p95, mx = pct(s, 0.95), max(s)
        big = sum(1 for x in s if x > SNAP_DEG)
        frozen = 100.0 * sum(1 for x in s if x < 1e-4) / len(s)
        res["temporal"][nm] = dict(p95=p95, max=mx, snaps=big, frozen=frozen)
        snap_total += big
        if p95 > worst_p95:
            worst_p95, worst_bone, worst_max = p95, nm, mx
        if frozen > frozen_worst:
            frozen_worst, frozen_name = frozen, nm
    res.update(worst_bone=worst_bone, worst_bone_p95=worst_p95, worst_bone_max=worst_max,
               bone_snaps=snap_total, frozen_pct=frozen_worst, frozen_bone=frozen_name)

    # ---- section 12D cross-body ----------------------------------------------------------------
    swaps = n = 0
    for r in rows:
        B = r["b"]
        if not all(k in B for k in ("LeftHand", "RightHand", "LeftUpperArm", "RightUpperArm")):
            continue
        lat = unit(sub(B["LeftUpperArm"]["p"], B["RightUpperArm"]["p"]))
        if lat is None:
            continue
        mid = [(B["LeftUpperArm"]["p"][i] + B["RightUpperArm"]["p"][i]) * 0.5 for i in range(3)]
        n += 1
        if dot(sub(B["LeftHand"]["p"], mid), lat) < dot(sub(B["RightHand"]["p"], mid), lat):
            swaps += 1
    res["cross_frames"], res["cross_total"] = swaps, n

    # ---- P0 LimbGate ---------------------------------------------------------------------------
    g = [r["gates"] for r in rows if "gates" in r]
    if g:
        held = sum(1 for x in g if any(x[:4]))
        res["gate_held_pct"] = 100.0 * held / len(g)
        res["gate_longest"] = [max(x[4 + i] for x in g) for i in range(4)]
        res["gate_longest_s"] = max(res["gate_longest"]) / fps
    return res


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path or not os.path.exists(path):
        print("usage: f19_analyze_live.py <telemetry.jsonl>", file=sys.stderr)
        return 2
    rows = []
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    rows = [r for r in rows if r.get("block") and not r["block"].startswith("(")]
    if not rows:
        print("no labelled frames", file=sys.stderr)
        return 1

    order, groups = [], {}
    for r in rows:
        b = r["block"]
        if b not in groups:
            groups[b] = []
            order.append(b)
        groups[b].append(r)

    out = []
    print("F-19 LIVE rendered-bone telemetry: %s" % os.path.basename(path))
    print("%d labelled frames across %d blocks; rigs: %s"
          % (len(rows), len(order), ",".join(sorted(set(r.get("rig", "?") for r in rows)))))
    print()
    print("%-38s %6s %6s %8s %8s %6s %9s %8s %8s" %
          ("block", "n", "fps", "yawRange", "step p95", "snaps", "bone dev%", "gate%", "gate s"))
    reports = []
    for b in order:
        r = block_report(b, groups[b], out)
        if r is None:
            continue
        reports.append(r)
        print("%-38s %6d %6.0f %8.1f %8.3f %6d %9.4f %8.2f %8.2f" %
              (b[:38], r["frames"], r["fps"], r["yaw_max"] - r["yaw_min"], r["yaw_step_p95"],
               r["snaps"], r["bone_dev_pct"], r.get("gate_held_pct", float("nan")),
               r.get("gate_longest_s", float("nan"))))

    allr = block_report("ALL", rows, out)
    print()
    print("OVERALL")
    print("  frames %d, mean %.0f fps" % (allr["frames"], allr["fps"]))
    print("  section 20 avatar torso yaw: range %.1f..%.1f deg, frame-to-frame p95 %.3f, max %.3f, "
          "snaps >%.0f deg: %d"
          % (allr["yaw_min"], allr["yaw_max"], allr["yaw_step_p95"], allr["yaw_step_max"],
             SNAP_DEG, allr["snaps"]))
    print("  section 14 worst bone deviation %.4f%% (%s); worst |lossyScale-1| %.6f -> %s"
          % (allr["bone_dev_pct"], allr["bone_dev_chain"], allr["scale_dev"],
             "no bone scaling" if allr["scale_dev"] < 1e-4 else "SCALING PRESENT"))
    print("  section 12C worst bone %s: p95 %.3f deg, max %.3f deg; total bone snaps >%.0f deg: %d"
          % (allr["worst_bone"], allr["worst_bone_p95"], allr["worst_bone_max"], SNAP_DEG,
             allr["bone_snaps"]))
    print("  section 12C most frozen bone %s at %.1f%% of frame pairs"
          % (allr["frozen_bone"], allr["frozen_pct"]))
    print("  section 12D hands laterally crossed on %d/%d frames (%.2f%%) - a legitimate pose, counted not judged"
          % (allr["cross_frames"], allr["cross_total"],
             100.0 * allr["cross_frames"] / max(1, allr["cross_total"])))
    print("  P0 LimbGate held a limb on %.2f%% of frames; longest hold %.2f s "
          "(per-limb longest runs lArm/rArm/lLeg/rLeg = %s)"
          % (allr.get("gate_held_pct", float("nan")), allr.get("gate_longest_s", float("nan")),
             allr.get("gate_longest")))
    print()
    print("section 15 / 12A HINGE PLAUSIBILITY (whole capture)")
    print("  %-10s %9s %9s %9s %12s %9s %8s" %
          ("joint", "flex p50", "flex p95", "flex max", "spread p95", "n bent", "n>fold"))
    for k, h in allr["hinges"].items():
        print("  %-10s %9.2f %9.2f %9.2f %12.2f %9d %8d" %
              (k, h["flex_p50"], h["flex_p95"], h["flex_max"], h["spread_p95"], h["n_bent"],
               h["over_fold"]))
    print("  (flex 0 = straight; folding past %.0f deg is independently impossible. bend-normal"
          % FOLD_MAX_DEG)
    print("   spread near 0 = a clean hinge; tens of degrees = bending where a human cannot)")

    dst = path.replace(".jsonl", "_live_analysis.json")
    io.open(dst, "w", encoding="utf-8").write(
        json.dumps({"file": path, "blocks": reports, "all": allr}, indent=2, default=str))
    print("\nwrote %s" % dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
