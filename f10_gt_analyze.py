#!/usr/bin/env python3
"""F-10 -- score yaw3D, yaw2D and the hybrid against the captured GROUND TRUTH.

Reads `f10_gt_marks_<distance>.json` (written by f10_gt_capture.py) plus the sidecar logs recorded
during the same run, labels every frame with the TRUE shoulder heading, and answers the brief's
critical questions. Ground truth is never derived from yaw3D.

Estimators
----------
  A  yaw3D    = the production estimate, atan2 over the emitted 3-D shoulder pair (Kalidokit form)
  B  yaw2D    = acos(clamp(uSpan * Z / K))   -- shoulder PIXEL span, robust trunk distance Z = hipZ
                K is calibrated from the 0 deg blocks, where cos(heading) == 1 by construction. That
                is the whole point of having ground truth: K stops being a percentile guess.
  C  hybrid   = sign(yaw3D) * |yaw2D|

Each block's first and last 25 % are trimmed, so a mistimed step costs the edges, not the block.

    python f10_gt_analyze.py --dir pipeline_logs_f10_mid --marks oak_v4_evidence/f10_gt_marks_mid.json
"""
import argparse
import glob
import io
import json
import math
import os

JOINTS = ("L-shoulder", "R-shoulder", "L-hip", "R-hip")
TRIM = 0.25


def pct(v, p):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[int(round((len(s) - 1) * p / 100.0))]


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def rmse(v):
    return math.sqrt(sum(x * x for x in v) / len(v)) if v else float("nan")


def stdev(v):
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def wrapd(a):
    return (a + 180.0) % 360.0 - 180.0


def line_yaw_deg(a, b):
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


def load(dirpath):
    audit, sender = {}, {}
    for name, sink in (("audit_log.jsonl", audit), ("sender_log.jsonl", sender)):
        p = os.path.join(dirpath, name)
        if not os.path.exists(p):
            continue
        for line in io.open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                sink[r["seq"]] = r
            except ValueError:
                pass
    rows = []
    for seq in sorted(set(audit) & set(sender)):
        a, s = audit[seq], sender[seq]
        j = a.get("j") or {}
        if any(n not in j for n in JOINTS):
            continue
        sh, hip = s.get("sh"), s.get("hip")
        hipZ = s.get("hipZ")
        if not sh or not hip or not hipZ or hipZ <= 0.1:
            continue
        if any(j[n].get("u") is None for n in JOINTS):
            continue
        rows.append({
            "seq": seq, "t": float(s.get("t", 0.0)), "hipZ": float(hipZ),
            "yaw3D": line_yaw_deg(sh[0], sh[1]),
            "hipYaw3D": line_yaw_deg(hip[0], hip[1]),
            "uSpanSh": abs(j["L-shoulder"]["u"] - j["R-shoulder"]["u"]),
            "uSpanHip": abs(j["L-hip"]["u"] - j["R-hip"]["u"]),
            "confMin": min(j[n]["c"] for n in JOINTS),
        })
    return rows


def label(rows, marks):
    """Attach headingGT + block name, trimming each block's edges."""
    out = []
    for b in marks["blocks"]:
        span = b["t1"] - b["t0"]
        lo, hi = b["t0"] + span * TRIM, b["t1"] - span * TRIM
        for r in rows:
            if lo <= r["t"] <= hi:
                q = dict(r)
                q["block"] = b["name"]
                q["headingGT"] = b["headingGT"]
                out.append(q)
    return out


def calibrate_K(rows):
    """K = fx*S, from the blocks whose true heading is 0 (cos == 1)."""
    z = [r["uSpanSh"] * r["hipZ"] for r in rows if r.get("headingGT") == 0]
    if len(z) < 20:
        return None
    return pct(z, 50)


def add_estimates(rows, K):
    for r in rows:
        c = max(0.0, min(1.0, (r["uSpanSh"] * r["hipZ"]) / K))
        r["yaw2D"] = math.degrees(math.acos(c))
        r["hybrid"] = math.copysign(r["yaw2D"], r["yaw3D"] if r["yaw3D"] != 0 else 1.0)
    return rows


def stats_block(errs):
    a = [abs(e) for e in errs]
    return (mean(a), rmse(errs), pct(a, 50), pct(a, 95), max(a) if a else float("nan"))


def report_magnitude(rows, name, key):
    static = [r for r in rows if r["headingGT"] is not None]
    errs = [abs(r[key]) - abs(r["headingGT"]) for r in static]
    m, rm, p50, p95, mx = stats_block(errs)
    print(" %-10s | %7.2f %7.2f %7.2f %7.2f %7.2f | n=%d" % (name, m, rm, p50, p95, mx, len(static)))
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True, help="capture log dir (repeatable)")
    ap.add_argument("--marks", action="append", required=True, help="marks json (repeatable, paired)")
    a = ap.parse_args()
    if len(a.dir) != len(a.marks):
        print("--dir and --marks must be paired")
        return 1

    runs = []
    for d, mp in zip(a.dir, a.marks):
        if not os.path.isdir(d) or not os.path.exists(mp):
            print("missing: %s / %s" % (d, mp))
            continue
        marks = json.load(io.open(mp, encoding="utf-8"))
        rows = label(load(d), marks)
        if not rows:
            print("no labelled frames for %s" % d)
            continue
        K = calibrate_K(rows)
        if K is None:
            print("%s: cannot calibrate K -- need >=20 frames at heading 0" % d)
            continue
        add_estimates(rows, K)
        runs.append({"dist": marks.get("distance", d), "metres": marks.get("metres", 0.0),
                     "rows": rows, "K": K})

    if not runs:
        print("nothing to analyse")
        return 1

    for run in runs:
        rows, dist = run["rows"], run["dist"]
        print("\n" + "=" * 104)
        print(" DISTANCE '%s' (%.2f m)  --  %d labelled frames, K=%.1f  [hipZ p50=%.2f m]"
              % (dist, run["metres"], len(rows), run["K"],
                 pct([r["hipZ"] for r in rows], 50)))
        print("=" * 104)

        # ---- TASK 3: magnitude accuracy -------------------------------------------------
        print("\n MAGNITUDE ACCURACY vs ground truth (deg)")
        print(" %-10s | %7s %7s %7s %7s %7s |" % ("estimator", "MAE", "RMSE", "p50", "p95", "max"))
        report_magnitude(rows, "yaw3D", "yaw3D")
        report_magnitude(rows, "yaw2D", "yaw2D")
        report_magnitude(rows, "hybrid", "hybrid")

        print("\n BIAS BY HEADING (signed mean error, deg)")
        heads = sorted(set(r["headingGT"] for r in rows if r["headingGT"] is not None))
        print(" %-8s | %8s | %8s %8s %8s | %8s" % ("headGT", "n", "yaw3D", "yaw2D", "hybrid", "sign ok"))
        for h in heads:
            sel = [r for r in rows if r["headingGT"] == h]
            if len(sel) < 5:
                continue
            b3 = mean([abs(r["yaw3D"]) - abs(h) for r in sel])
            b2 = mean([abs(r["yaw2D"]) - abs(h) for r in sel])
            bh = mean([abs(r["hybrid"]) - abs(h) for r in sel])
            if h == 0:
                sok = float("nan")
            else:
                sok = 100.0 * sum(1 for r in sel if (r["yaw3D"] > 0) == (h > 0)) / len(sel)
            print(" %+8d | %8d | %+8.2f %+8.2f %+8.2f | %7.1f%%" %
                  (h, len(sel), b3, b2, bh, sok if sok == sok else -1))

        # ---- TASK 3: sign accuracy -------------------------------------------------------
        signed = [r for r in rows if r["headingGT"] not in (None, 0)]
        if signed:
            tp = sum(1 for r in signed if (r["yaw3D"] > 0) == (r["headingGT"] > 0))
            print("\n SIGN ACCURACY of yaw3D over %d non-zero-heading frames: %.2f%%"
                  % (len(signed), 100.0 * tp / len(signed)))
            print("  confusion (rows = true, cols = yaw3D):   pred+   pred-")
            for s in (+1, -1):
                sel = [r for r in signed if (r["headingGT"] > 0) == (s > 0)]
                pp = sum(1 for r in sel if r["yaw3D"] > 0)
                print("    true%s                                %6d  %6d" % ("+" if s > 0 else "-", pp, len(sel) - pp))
            near0 = [r for r in signed if abs(r["headingGT"]) <= 30]
            near90 = [r for r in signed if abs(r["headingGT"]) >= 90]
            for nm, sel in (("|heading| <= 30", near0), ("|heading| >= 90", near90)):
                if len(sel) >= 5:
                    ok = sum(1 for r in sel if (r["yaw3D"] > 0) == (r["headingGT"] > 0))
                    print("    sign accuracy, %-16s : %6.2f%%  (n=%d)"
                          % (nm, 100.0 * ok / len(sel), len(sel)))

        # ---- TASK 5: torso twist ----------------------------------------------------------
        tw = [r for r in rows if r["block"].startswith("tw_")]
        if tw:
            print("\n TORSO TWIST (TASK 5) -- hip, shoulder and relative yaw must stay separable")
            print(" %-14s | %8s | %9s %9s %9s" % ("block", "n", "hipYaw3D", "shYaw3D", "relative"))
            for b in sorted(set(r["block"] for r in tw)):
                sel = [r for r in tw if r["block"] == b]
                print(" %-14s | %8d | %9.2f %9.2f %9.2f" %
                      (b, len(sel), mean([r["hipYaw3D"] for r in sel]),
                       mean([r["yaw3D"] for r in sel]),
                       mean([r["yaw3D"] - r["hipYaw3D"] for r in sel])))

        # ---- TASK 6: motion --------------------------------------------------------------
        mo = [r for r in rows if r["block"].startswith("m_")]
        if mo:
            print("\n MOTION (TASK 6) -- jitter and step size, no ground-truth heading in these blocks")
            print(" %-16s | %8s | %9s %9s | %9s %9s" %
                  ("block", "n", "yaw3D sd", "yaw2D sd", "d3D p95", "d2D p95"))
            for b in sorted(set(r["block"] for r in mo)):
                sel = sorted([r for r in mo if r["block"] == b], key=lambda r: r["t"])
                d3 = [abs(wrapd(sel[i]["yaw3D"] - sel[i - 1]["yaw3D"])) for i in range(1, len(sel))]
                d2 = [abs(sel[i]["yaw2D"] - sel[i - 1]["yaw2D"]) for i in range(1, len(sel))]
                print(" %-16s | %8d | %9.2f %9.2f | %9.2f %9.2f" %
                      (b, len(sel), stdev([r["yaw3D"] for r in sel]), stdev([r["yaw2D"] for r in sel]),
                       pct(d3, 95) if d3 else 0, pct(d2, 95) if d2 else 0))

    # ---- TASK 4: distance sensitivity ---------------------------------------------------
    if len(runs) > 1:
        print("\n" + "=" * 104)
        print(" DISTANCE SENSITIVITY (TASK 4)")
        print("=" * 104)
        print(" %-8s %7s | %9s %9s %9s | %10s" %
              ("dist", "hipZ", "yaw3D MAE", "yaw2D MAE", "hybrid MAE", "uSpan@0deg"))
        for run in runs:
            rows = run["rows"]
            st = [r for r in rows if r["headingGT"] is not None]
            f = lambda k: mean([abs(abs(r[k]) - abs(r["headingGT"])) for r in st])
            z0 = [r["uSpanSh"] for r in rows if r["headingGT"] == 0]
            print(" %-8s %7.2f | %9.2f %9.2f %9.2f | %10.1f" %
                  (run["dist"], pct([r["hipZ"] for r in rows], 50),
                   f("yaw3D"), f("yaw2D"), f("hybrid"), pct(z0, 50) if z0 else float("nan")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
