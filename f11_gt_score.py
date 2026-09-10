#!/usr/bin/env python3
"""F-11 -- score the 2-D face sign candidates against GROUND TRUTH.

Joins the F-11 capture's `audit_log.jsonl` (now carrying face keypoints, F-11 diagnostic change) with
`sender_log.jsonl`, labels every frame from the capture marks, and answers the questions the video
could not: sign accuracy overall / per direction / at +-90, and what happens when the head and the
torso disagree.

SIGN CONVENTION is inferred empirically from the static +-45 / +-90 blocks (majority vote), never
assumed, and the inferred convention is printed.

The head/torso decoupling blocks are the point of this run:
    dc_body45R_face0 / dc_body45L_face0 : TORSO turned, FACE still on the camera  -> can the nose see it?
    dc_body0_head45R / dc_body0_head45L : TORSO square, HEAD turned              -> does it fire falsely?

    python f11_gt_score.py --dir pipeline_logs_f11 --marks oak_v4_evidence/f10_gt_marks_near.json
"""
import argparse
import io
import json
import math
import os

TRIM = 0.25
CONF_MIN = 0.3


def pct(v, p):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[int(round((len(s) - 1) * p / 100.0))]


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def rmse(v):
    return math.sqrt(sum(x * x for x in v) / len(v)) if v else float("nan")


def load(dirpath):
    audit, sender = {}, {}
    for name, sink in (("audit_log.jsonl", audit), ("sender_log.jsonl", sender)):
        p = os.path.join(dirpath, name)
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
        j, f = a.get("j") or {}, a.get("f") or {}
        if "L-shoulder" not in j or "R-shoulder" not in j or "nose" not in f:
            continue
        hipZ = s.get("hipZ")
        if not hipZ or hipZ <= 0.1:
            continue
        lsh, rsh, nose = j["L-shoulder"], j["R-shoulder"], f["nose"]
        if min(lsh["c"], rsh["c"]) < CONF_MIN:
            continue
        span = abs(lsh["u"] - rsh["u"])
        mid = 0.5 * (lsh["u"] + rsh["u"])
        e = {"seq": seq, "t": float(s.get("t", 0.0)), "hipZ": float(hipZ),
             "span": span, "noseC": nose["c"],
             "A": nose["u"] - mid,
             "B": (nose["u"] - mid) / max(span, 1e-6),
             "earAsym": f["L-ear"]["c"] - f["R-ear"]["c"]}
        if min(f["L-eye"]["c"], f["R-eye"]["c"]) >= CONF_MIN:
            e["Dn"] = (0.5 * (f["L-eye"]["u"] + f["R-eye"]["u"]) - mid) / max(span, 1e-6)
        rows.append(e)
    return rows


def label(rows, marks):
    out = []
    for b in marks["blocks"]:
        sp = b["t1"] - b["t0"]
        lo, hi = b["t0"] + sp * TRIM, b["t1"] - sp * TRIM
        for r in rows:
            if lo <= r["t"] <= hi:
                q = dict(r)
                q["block"] = b["name"]
                q["torsoGT"] = b["headingGT"]
                q["headGT"] = b.get("headGT")
                out.append(q)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--marks", required=True)
    a = ap.parse_args()
    marks = json.load(io.open(a.marks, encoding="utf-8"))
    rows = label(load(a.dir), marks)
    if not rows:
        print("no labelled frames")
        return 1

    # yaw2D, calibrated on the PURE static 0 blocks only (h_0*), not the head-only blocks
    z = [r["span"] * r["hipZ"] for r in rows if r["block"].startswith("h_0")]
    K = pct(z, 50)
    for r in rows:
        c = max(0.0, min(1.0, (r["span"] * r["hipZ"]) / K))
        r["yaw2D"] = math.degrees(math.acos(c))

    print("=" * 108)
    print(" F-11 -- 2-D FACE SIGN vs GROUND TRUTH   (%d labelled frames, K=%.1f)" % (len(rows), K))
    print("=" * 108)

    # ---- sign convention, inferred not assumed ---------------------------------------
    stat = [r for r in rows if r["headGT"] is None and r["torsoGT"] not in (None, 0)]
    agree = sum(1 for r in stat if (r["B"] > 0) == (r["torsoGT"] > 0))
    pol = 1 if agree * 2 >= len(stat) else -1
    print("\n sign convention inferred from the %d static turned frames: B>0 corresponds to a %s turn"
          % (len(stat), "POSITIVE (right)" if pol > 0 else "NEGATIVE (left)"))

    # Each candidate gets its OWN inferred polarity: the nose offset and the ear asymmetry have
    # opposite natural signs, and forcing one polarity on both would score a good candidate as ~0 %.
    def infer_pol(key):
        sel = [r for r in stat if r.get(key) is not None]
        if not sel:
            return 1
        ag = sum(1 for r in sel if (r[key] > 0) == (r["torsoGT"] > 0))
        return 1 if ag * 2 >= len(sel) else -1

    POL = dict((k, infer_pol(k)) for k in ("A", "B", "Dn", "earAsym"))

    def sgn(r, key):
        v = r.get(key)
        if v is None:
            return None
        return POL[key] if v > 0 else -POL[key]

    # ---- per-block behaviour ----------------------------------------------------------
    print("\n PER-BLOCK (median values; 'sign ok' vs the TORSO ground truth)")
    print(" %-18s %-6s %-5s %6s | %8s %8s %8s %8s | %8s" %
          ("block", "torso", "head", "n", "yaw2D", "A px", "B", "earAsym", "sign ok"))
    for b in marks["blocks"]:
        sel = [r for r in rows if r["block"] == b["name"]]
        if len(sel) < 5:
            continue
        t = b["headingGT"]
        ok = "--"
        if t not in (None, 0):
            n_ok = sum(1 for r in sel if sgn(r, "B") == (1 if t > 0 else -1))
            ok = "%.1f%%" % (100.0 * n_ok / len(sel))
        print(" %-18s %-6s %-5s %6d | %8.1f %8.2f %8.4f %8.4f | %8s" %
              (b["name"], t, b.get("headGT"), len(sel), pct([r["yaw2D"] for r in sel], 50),
               pct([r["A"] for r in sel], 50), pct([r["B"] for r in sel], 50),
               pct([r["earAsym"] for r in sel], 50), ok))

    # ---- sign accuracy, the acceptance numbers ---------------------------------------
    print("\n SIGN ACCURACY vs torso ground truth (static blocks only, 0 deg excluded)")
    print("   inferred polarity per candidate: %s"
          % ", ".join("%s=%+d" % (k, v) for k, v in sorted(POL.items())))
    print(" %-10s %8s %10s %10s %10s" % ("candidate", "n", "overall", "positive", "negative"))
    for key in ("B", "A", "Dn", "earAsym"):
        sel = [r for r in stat if r.get(key) is not None]
        if len(sel) < 20:
            continue
        pos = [r for r in sel if r["torsoGT"] > 0]
        neg = [r for r in sel if r["torsoGT"] < 0]
        o = sum(1 for r in sel if sgn(r, key) == (1 if r["torsoGT"] > 0 else -1))
        p = sum(1 for r in pos if sgn(r, key) == 1)
        n = sum(1 for r in neg if sgn(r, key) == -1)
        print(" %-10s %8d %9.2f%% %9.2f%% %9.2f%%" %
              (key, len(sel), 100.0 * o / len(sel),
               100.0 * p / max(1, len(pos)), 100.0 * n / max(1, len(neg))))

    print("\n SIGN ACCURACY at the difficult poses (candidate B)")
    for lab, pred in (("|torso| = 45", lambda r: abs(r["torsoGT"]) == 45),
                      ("|torso| = 90", lambda r: abs(r["torsoGT"]) == 90),
                      ("torso = +90", lambda r: r["torsoGT"] == 90),
                      ("torso = -90", lambda r: r["torsoGT"] == -90)):
        sel = [r for r in stat if pred(r)]
        if len(sel) < 5:
            continue
        ok = sum(1 for r in sel if sgn(r, "B") == (1 if r["torsoGT"] > 0 else -1))
        print("   %-14s n=%4d   %6.2f%%" % (lab, len(sel), 100.0 * ok / len(sel)))

    # ---- THE decoupling test ----------------------------------------------------------
    print("\n" + "-" * 108)
    print(" HEAD / TORSO DECOUPLING -- the mirror failure mode")
    print("-" * 108)
    for nm, desc in (("dc_body45R_face0", "TORSO +45, FACE on camera  -> can the nose see the torso?"),
                     ("dc_body45L_face0", "TORSO -45, FACE on camera  -> can the nose see the torso?")):
        sel = [r for r in rows if r["block"] == nm]
        if len(sel) < 5:
            continue
        t = sel[0]["torsoGT"]
        ok = sum(1 for r in sel if sgn(r, "B") == (1 if t > 0 else -1))
        print("   %-18s n=%4d  B p50=%+.4f  yaw2D p50=%5.1f  sign correct %6.2f%%"
              % (nm, len(sel), pct([r["B"] for r in sel], 50),
                 pct([r["yaw2D"] for r in sel], 50), 100.0 * ok / len(sel)))
        print("      %s" % desc)
    for nm, desc in (("dc_body0_head45R", "TORSO 0, HEAD +45 -> does it FALSELY claim a torso turn?"),
                     ("dc_body0_head45L", "TORSO 0, HEAD -45 -> does it FALSELY claim a torso turn?")):
        sel = [r for r in rows if r["block"] == nm]
        if len(sel) < 5:
            continue
        base = pct([abs(r["B"]) for r in rows if r["block"].startswith("h_0")], 50)
        big = sum(1 for r in sel if abs(r["B"]) > 2 * base)
        print("   %-18s n=%4d  B p50=%+.4f  |B| p95=%.4f  frames with |B| > 2x the frontal median: %.1f%%"
              % (nm, len(sel), pct([r["B"] for r in sel], 50),
                 pct([abs(r["B"]) for r in sel], 95), 100.0 * big / len(sel)))
        print("      %s   (frontal |B| median = %.4f)" % (desc, base))

    # ---- hybrid ------------------------------------------------------------------------
    print("\n HYBRID  sign(faceB) * |yaw2D|  vs torso ground truth, static blocks")
    hy = [abs(sgn(r, "B") * r["yaw2D"] - r["torsoGT"]) for r in stat]
    mag = [abs(r["yaw2D"] - abs(r["torsoGT"])) for r in stat]
    print("   hybrid (signed)        : MAE %6.2f  RMSE %6.2f  p50 %6.2f  p95 %6.2f  max %6.2f"
          % (mean(hy), rmse(hy), pct(hy, 50), pct(hy, 95), max(hy)))
    print("   yaw2D magnitude alone  : MAE %6.2f  RMSE %6.2f  p50 %6.2f  p95 %6.2f  max %6.2f"
          % (mean(mag), rmse(mag), pct(mag, 50), pct(mag, 95), max(mag)))
    print("   (F-10 reference: production yaw3D signed MAE 61.08, depth-sign hybrid 13.09)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
