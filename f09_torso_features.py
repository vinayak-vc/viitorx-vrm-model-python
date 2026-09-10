#!/usr/bin/env python3
"""F-09 -- torso yaw quality audit: extract every candidate predictor, per frame.

Joins `audit_log.jsonl` (per-joint depth-window diagnostics: u, v, confidence, valid-pixel counts,
p30/p50 depth, dmin/dmax, dsd) with `sender_log.jsonl` (the emitted 3-D landmarks that Unity actually
receives) on `seq`. Both are written by the shipping sidecar; nothing here is re-derived from a model.

THE LABEL PROBLEM, and how it is solved without ground truth
------------------------------------------------------------
There is no recorded ground-truth torso yaw. A label built from any depth-difference feature would be
circular, because the depth difference IS what produces the yaw. So the label uses the one signal that
is INDEPENDENT of the per-shoulder depth difference: the SHOULDER SEPARATION IN PIXELS.

A rigid shoulder segment of width S seen at yaw t projects to a horizontal extent S*cos(t). In pixels
at distance Z that is  uSpan = fx*S*cos(t)/Z, so

    cos(t) = uSpan * Z / (fx * S) = uSpan * Z / K,     K = fx*S, one constant per capture

K is calibrated as the p98 of (uSpan * Z) over the capture -- the frames where the subject is most
face-on. Z is `hipZ`, a robust trunk distance, NOT the per-shoulder depth that is under test.

This 2-D estimate is insensitive near t = 0 (cos is flat there) but DECISIVE at large claimed yaw: a
genuine 60 deg turn must narrow the shoulders to 50 % of their face-on pixel width. So the label is
strongest exactly where it matters.

    BAD  = the 3-D yaw claims a large turn that the image does not support
         = |yaw3D| - yaw2D > BAD_DISAGREE_DEG
    GOOD = |yaw3D| - yaw2D| within tolerance

    python f09_torso_features.py
"""
import io
import json
import math
import os

JOINTS = ("L-shoulder", "R-shoulder", "L-hip", "R-hip")
BAD_DISAGREE_DEG = 20.0
CAL_PCT = 98.0
SURFACE_GAP_MM = 100.0          # oak_depth.SURFACE_GAP_MM


def pct(v, p):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[int(round((len(s) - 1) * p / 100.0))]


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def stdev(v):
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def wrapd(a):
    return (a + 180.0) % 360.0 - 180.0


def line_yaw_deg(a, b):
    """Kalidokit CalcHipsAndSpine y-channel, degrees (ported in oak_yaw_v4.py)."""
    r = math.atan2(b[0] - a[0], b[2] - a[2])
    ang = math.fmod(r, 2 * math.pi)
    if ang > math.pi:
        ang -= 2 * math.pi
    elif ang < -math.pi:
        ang += 2 * math.pi
    y = ang / math.pi
    if y > 0.5:
        y -= 2.0
    y += 0.5
    return y * 180.0


def dist3(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def load_capture(d):
    audit, sender = {}, {}
    ap = os.path.join(d, "audit_log.jsonl")
    sp = os.path.join(d, "sender_log.jsonl")
    if not (os.path.exists(ap) and os.path.exists(sp)):
        return [], []
    for line in io.open(ap, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                r = json.loads(line)
                audit[r["seq"]] = r
            except ValueError:
                pass
    for line in io.open(sp, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                r = json.loads(line)
                sender[r["seq"]] = r
            except ValueError:
                pass

    rows, skipped = [], []
    for seq in sorted(set(audit) & set(sender)):
        a, s = audit[seq], sender[seq]
        j = a.get("j") or {}
        if any(n not in j for n in JOINTS):
            continue
        sh, hip = s.get("sh"), s.get("hip")
        if not sh or not hip or len(sh) != 2 or len(hip) != 2:
            continue
        hipZ = s.get("hipZ")
        if not hipZ or hipZ <= 0.1:
            continue
        e = {n: j[n] for n in JOINTS}
        # A joint with no usable depth window carries no p30/dsd/dmin/dmax. Those frames cannot
        # supply the depth features at all, so they are counted and skipped rather than imputed.
        if any(e[n].get("p30") is None or e[n].get("dsd") is None
               or e[n].get("dmin") is None or e[n].get("dmax") is None for n in JOINTS):
            skipped.append(seq)
            continue
        f = {
            "seq": seq, "t": s.get("t", 0.0), "cov": s.get("cov", 0), "hipZ": float(hipZ),
            # --- emitted 3-D landmarks (what Unity receives) ---
            "shL": sh[0][:3], "shR": sh[1][:3], "hipL": hip[0][:3], "hipR": hip[1][:3],
            "shSpan3D": dist3(sh[0], sh[1]),
            "hipSpan3D": dist3(hip[0], hip[1]),
            "shDzEmit": float(sh[0][2]) - float(sh[1][2]),
            "hipDzEmit": float(hip[0][2]) - float(hip[1][2]),
            "yaw3D": line_yaw_deg(sh[0], sh[1]),
            "hipYaw3D": line_yaw_deg(hip[0], hip[1]),
            # --- raw depth-window diagnostics, per torso joint ---
            "shDzRaw": e["L-shoulder"]["p30"] - e["R-shoulder"]["p30"],
            "hipDzRaw": e["L-hip"]["p30"] - e["R-hip"]["p30"],
            "midShZraw": 0.5 * (e["L-shoulder"]["p30"] + e["R-shoulder"]["p30"]),
            "midHipZraw": 0.5 * (e["L-hip"]["p30"] + e["R-hip"]["p30"]),
            "dsdMax": max(e[n]["dsd"] for n in JOINTS),
            "dsdShMax": max(e["L-shoulder"]["dsd"], e["R-shoulder"]["dsd"]),
            "rangeMax": max(e[n]["dmax"] - e[n]["dmin"] for n in JOINTS),
            "rangeShMax": max(e["L-shoulder"]["dmax"] - e["L-shoulder"]["dmin"],
                              e["R-shoulder"]["dmax"] - e["R-shoulder"]["dmin"]),
            "validFracMin": min((e[n]["nv"] / float(e[n]["n"])) if e[n]["n"] else 0.0 for n in JOINTS),
            "confMin": min(e[n]["c"] for n in JOINTS),
            "measuredAll": all(e[n]["m"] for n in JOINTS),
            # --- 2-D, INDEPENDENT of the per-shoulder depth difference ---
            "uSpanSh": abs(e["L-shoulder"]["u"] - e["R-shoulder"]["u"]),
            "uSpanHip": abs(e["L-hip"]["u"] - e["R-hip"]["u"]),
            "vDiffSh": abs(e["L-shoulder"]["v"] - e["R-shoulder"]["v"]),
        }
        f["trunkDz"] = f["midShZraw"] - f["midHipZraw"]           # mm; a standing trunk is near-vertical
        f["surfCrossSh"] = 1 if f["rangeShMax"] > SURFACE_GAP_MM else 0
        rows.append(f)
    return rows, skipped


def annotate(rows):
    """Add the 2-D yaw estimate, the disagreement label, temporal deltas and span stability."""
    if not rows:
        return rows
    prod = [r["uSpanSh"] * r["hipZ"] for r in rows]
    K = pct(prod, CAL_PCT)
    Smed = pct([r["shSpan3D"] for r in rows], 50)
    Hmed = pct([r["hipSpan3D"] for r in rows], 50)
    for i, r in enumerate(rows):
        c = (r["uSpanSh"] * r["hipZ"]) / K if K > 0 else 0.0
        c = max(0.0, min(1.0, c))
        r["yaw2D"] = math.degrees(math.acos(c))
        r["disagree"] = abs(r["yaw3D"]) - r["yaw2D"]
        r["bad"] = 1 if r["disagree"] > BAD_DISAGREE_DEG else 0
        # 3-D segment-length stability: a rigid shoulder line must keep constant length
        r["shLenErr"] = abs(r["shSpan3D"] / Smed - 1.0) if Smed > 1e-6 else 0.0
        r["hipLenErr"] = abs(r["hipSpan3D"] / Hmed - 1.0) if Hmed > 1e-6 else 0.0
        if i > 0:
            dt = r["t"] - rows[i - 1]["t"]
            r["dYaw"] = abs(wrapd(r["yaw3D"] - rows[i - 1]["yaw3D"]))
            r["dYawRate"] = r["dYaw"] / dt if 0 < dt < 1 else 0.0
        else:
            r["dYaw"], r["dYawRate"] = 0.0, 0.0
    return rows


def sep_report(rows, feature, higher_is_worse=True):
    """Separation of a candidate predictor between GOOD and BAD frames, plus a rank AUC."""
    g = [r[feature] for r in rows if not r["bad"]]
    b = [r[feature] for r in rows if r["bad"]]
    if len(g) < 20 or len(b) < 20:
        return None
    # AUC via the Mann-Whitney U statistic on ranks
    pairs = sorted([(v, 0) for v in g] + [(v, 1) for v in b])
    rank, i, rsum = 0, 0, 0.0
    while i < len(pairs):
        jj = i
        while jj + 1 < len(pairs) and pairs[jj + 1][0] == pairs[i][0]:
            jj += 1
        avg = (i + jj) / 2.0 + 1.0
        for kk in range(i, jj + 1):
            if pairs[kk][1] == 1:
                rsum += avg
        i = jj + 1
    n1, n0 = len(b), len(g)
    auc = (rsum - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    if not higher_is_worse:
        auc = 1.0 - auc
    return {"feature": feature, "auc": auc,
            "good_p50": pct(g, 50), "good_p95": pct(g, 95),
            "bad_p50": pct(b, 50), "bad_p95": pct(b, 95), "n_good": n0, "n_bad": n1}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    caps = [d for d in sorted(os.listdir(here))
            if d.startswith("pipeline_logs")
            and os.path.exists(os.path.join(here, d, "audit_log.jsonl"))]
    print("=" * 112)
    print(" F-09 TORSO YAW QUALITY AUDIT -- candidate predictors, %d captures with depth diagnostics" % len(caps))
    print("=" * 112)

    allrows = []
    for d in caps:
        rows, skipped = load_capture(os.path.join(here, d))
        rows = annotate(rows)
        if not rows:
            continue
        nbad = sum(r["bad"] for r in rows)
        print(" %-24s n=%5d  no-depth-window skipped=%4d   BAD (3D claims a turn the image denies) = %5d (%5.2f%%)"
              % (d, len(rows), len(skipped), nbad, 100.0 * nbad / len(rows)))
        for r in rows:
            r["cap"] = d
        allrows.extend(rows)
    if not allrows:
        print(" no usable frames")
        return 1

    print("\n POOLED: %d frames, %d BAD (%.2f%%)"
          % (len(allrows), sum(r["bad"] for r in allrows),
             100.0 * sum(r["bad"] for r in allrows) / len(allrows)))

    # ---- depth quantisation, the structural fact behind all of this ----------------------
    levels = sorted(set(r["midShZraw"] * 2 for r in allrows))
    steps = [levels[i] - levels[i - 1] for i in range(1, len(levels)) if 20 < levels[i] - levels[i - 1] < 400]
    print("\n DEPTH QUANTISATION (distinct p30 sums -> step size, mm): n=%d  p50=%.1f  p90=%.1f"
          % (len(steps), pct(steps, 50), pct(steps, 90)))
    shdz = [abs(r["shDzRaw"]) for r in allrows]
    print(" |raw shoulder depth difference| mm: p50=%.1f p90=%.1f p95=%.1f  -- compare to the step above"
          % (pct(shdz, 50), pct(shdz, 90), pct(shdz, 95)))

    # ---- candidate predictors ------------------------------------------------------------
    cands = [
        ("shLenErr", True, "3-D shoulder segment length error (rigid-length violation)"),
        ("hipLenErr", True, "3-D hip segment length error"),
        ("trunkDz", True, "|midShoulder - midHip| raw depth (mm), trunk should be vertical"),
        ("dsdShMax", True, "worst shoulder depth-window stdev (mm)"),
        ("dsdMax", True, "worst torso depth-window stdev (mm)"),
        ("rangeShMax", True, "worst shoulder window depth range (mm)"),
        ("rangeMax", True, "worst torso window depth range (mm)"),
        ("validFracMin", False, "min valid-pixel fraction over the 4 torso joints"),
        ("confMin", False, "min model confidence over the 4 torso joints"),
        ("dYaw", True, "frame-to-frame |yaw| step (deg)"),
        ("dYawRate", True, "frame-to-frame yaw rate (deg/s)"),
        ("uSpanSh", False, "shoulder pixel span (2-D)"),
        ("hipSpan3D", True, "3-D hip span (m)"),
        ("shSpan3D", True, "3-D shoulder span (m)"),
        ("surfCrossSh", True, "shoulder window crosses a >100 mm discontinuity (0/1)"),
    ]
    print("\n" + "-" * 112)
    print(" CANDIDATE PREDICTOR SEPARATION (AUC 0.5 = useless, 1.0 = perfect; direction normalised)")
    print(" %-14s %6s | %10s %10s | %10s %10s | %s" %
          ("feature", "AUC", "good p50", "good p95", "bad p50", "bad p95", "meaning"))
    res = []
    for f, hw, desc in cands:
        # trunkDz is signed; score its magnitude
        if f == "trunkDz":
            for r in allrows:
                r["_abs_trunkDz"] = abs(r["trunkDz"])
            f2 = "_abs_trunkDz"
        else:
            f2 = f
        s = sep_report(allrows, f2, hw)
        if s is None:
            continue
        s["feature"] = f
        s["desc"] = desc
        res.append(s)
    for s in sorted(res, key=lambda x: -abs(x["auc"] - 0.5)):
        print(" %-14s %6.3f | %10.4f %10.4f | %10.4f %10.4f | %s" %
              (s["feature"], s["auc"], s["good_p50"], s["good_p95"], s["bad_p50"], s["bad_p95"], s["desc"]))

    # ---- dump for the replay stage --------------------------------------------------------
    out = os.path.join(here, "oak_v4_evidence", "f09_features.jsonl")
    with io.open(out, "w", encoding="utf-8") as fh:
        for r in allrows:
            fh.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}) + "\n")
    print("\n features -> %s (%d rows)" % (out, len(allrows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
