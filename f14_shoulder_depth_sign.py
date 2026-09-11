#!/usr/bin/env python3
"""F-14 -- does the RAW shoulder depth difference carry the torso-yaw SIGN that yaw3D loses?

THE HYPOTHESIS
    F-10 rejected "depth sign" at 78.49 %, but it tested sign(yaw3D) where
        yaw3D = atan2(shR.x - shL.x, shR.z - shL.z)     (Kalidokit CalcHipsAndSpine y-channel)
    which WRAPS through +-180 as the shoulder line goes edge-on and dx -> 0. F-13's mandated
    comparison found the raw depth difference correct on 9/10 held blocks. F-14 asks whether the
    sign survives in the depth measurement itself, independent of that composition.

    Being right about the sign is not the same as being right for the right reason, so this script
    separates four things that F-13 did not: raw vs smoothed, coverage vs accuracy, frame vs block,
    and "correct" vs "correct where yaw3D was wrong".

GROUND TRUTH
    Block labels only. yaw3D, yaw2D, the hips and the candidate itself are never used to label a
    frame. Blocks are assigned SHOULDER truth by what they constrain THE SHOULDERS to do:
        h_p45/h_p90/h_m45/h_m90       whole body turned            -> shoulder truth = heading
        dc_body45R_face0/..45L..      body turned, face on camera  -> shoulder truth = heading
        h_0/h_0b/h_0c                 square                       -> shoulder truth = 0
        dc_body0_head45R/..45L        body square, HEAD turned     -> shoulder truth = 0
        tw_rigid_p45/tw_sh_p45/tw_sh_m45   SHOULDERS turned        -> shoulder truth = heading,
                                      but recorded in the protocol the subject reported confusion
                                      during, so scored as a SEPARATE robustness group.
    Note this differs from F-13: the tw_sh_* blocks are meaningless for hips (hips stay square) but
    are exactly on-target for shoulders.

ZERO IS UNDEFINED
    A depth difference of exactly 0 is a third state, never "negative" -- the bug that made F-13's
    first run report a stable sign on an all-zero block.

    python f14_shoulder_depth_sign.py
"""
import bisect
import io
import json
import math
import os

TRIM = 0.25
CONF_MIN = 0.3

CAPTURES = [
    ("F-10", "pipeline_logs_f10_near",
     os.path.join("oak_v4_evidence", "f10_gt_marks_near_RECOVERED.json")),
    ("F-11", "pipeline_logs_f11",
     os.path.join("oak_v4_evidence", "f10_gt_marks_near.json")),
]

TURNED = ("h_p45", "h_p90", "h_m45", "h_m90", "dc_body45R_face0", "dc_body45L_face0")
TWIST_TURNED = ("tw_rigid_p45", "tw_sh_p45", "tw_sh_m45")      # shoulders turned, compromised run
ZEROED = ("h_0", "h_0b", "h_0c", "dc_body0_head45R", "dc_body0_head45L")
MOTION = ("m_slow", "m_normal", "m_fast", "m_rev_lr", "m_rev_rl", "m_180",
          "m_turn_arms", "m_turn_still")

CANDS = [
    ("A", "dzP30", "RAW p30(L-sh) - p30(R-sh), mm, pre-smoothing"),
    ("B", "dzP50", "RAW p50(L-sh) - p50(R-sh), mm, pre-smoothing"),
    ("C", "dzCtr", "RAW centre-pixel depth difference, mm"),
    ("D", "dzEmit", "EMITTED shL.z - shR.z, mm, POST one-euro smoothing"),
    ("E", "yaw3Dsig", "sign(yaw3D) -- the F-10 baseline that failed"),
]


def pct(v, p):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[int(round((len(s) - 1) * p / 100.0))]


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def sd(v):
    if len(v) < 2:
        return float("nan")
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def rmse(v):
    return math.sqrt(sum(x * x for x in v) / len(v)) if v else float("nan")


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    s = sorted(neg)
    tot = 0.0
    for v in pos:
        lo, hi = bisect.bisect_left(s, v), bisect.bisect_right(s, v)
        tot += lo + 0.5 * (hi - lo)
    return tot / (len(pos) * len(neg))


def fmt(x):
    return "  none  " if x != x else "%7.2f%%" % x


def line_yaw_deg(a, b):
    """EXACTLY the Kalidokit CalcHipsAndSpine y-channel used by yaw3D (f09_torso_features.py:65)."""
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
    return math.degrees(y * math.pi)


def sgn(v, pol, thr=0.0):
    """Three-valued: exact zero (or within thr) carries NO sign."""
    if v is None or abs(v) <= thr:
        return 0
    return pol if v > 0 else -pol


def load(dirpath):
    audit, sender = {}, {}
    for name, sink in (("audit_log.jsonl", audit), ("sender_log.jsonl", sender)):
        p = os.path.join(dirpath, name)
        if not os.path.exists(p):
            return []
        for line in io.open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "seq" in r and "joint" not in r:
                sink[r["seq"]] = r
    rows = []
    for seq in sorted(set(audit) & set(sender)):
        a, s = audit[seq], sender[seq]
        j = a.get("j") or {}
        if "L-shoulder" not in j or "R-shoulder" not in j:
            continue
        ls, rs = j["L-shoulder"], j["R-shoulder"]
        hipZ = s.get("hipZ")
        if not hipZ or hipZ <= 0.1:
            continue
        if min(ls["c"], rs["c"]) < CONF_MIN:
            continue
        sh = s.get("sh")
        if not sh or len(sh) < 2:
            continue
        e = {
            "seq": seq, "t": float(s.get("t", 0.0)), "hipZ": float(hipZ),
            "confShMin": min(ls["c"], rs["c"]),
            "qShMin": min(ls.get("q", 0.0), rs.get("q", 0.0)),
            "measSh": int(ls.get("m", 0)) + int(rs.get("m", 0)),
            "nvMin": min(ls.get("nv", 0), rs.get("nv", 0)),
            "uSpanSh": abs(ls["u"] - rs["u"]),
            "uL": ls["u"], "uR": rs["u"],
            "dsdShMax": max(ls.get("dsd", 0.0), rs.get("dsd", 0.0)),
            "dzEmit": (float(sh[0][2]) - float(sh[1][2])) * 1000.0,
            "shSepX": abs(float(sh[0][0]) - float(sh[1][0])),
            "yaw3D": line_yaw_deg(sh[0], sh[1]),
        }
        e["yaw3Dsig"] = e["yaw3D"]
        for tag, src in (("P30", "p30"), ("P50", "p50")):
            if src in ls and src in rs:
                e["dz" + tag] = float(ls[src]) - float(rs[src])
                e["zL" + tag] = float(ls[src])
                e["zR" + tag] = float(rs[src])
        if ls.get("ctr") and rs.get("ctr"):
            e["dzCtr"] = float(ls["ctr"]) - float(rs["ctr"])
        rows.append(e)
    return rows


def recover_fx(rows):
    v = [r["uSpanSh"] * r["hipZ"] / r["shSepX"]
         for r in rows if r["shSepX"] > 0.05 and r["uSpanSh"] > 1.0]
    return pct(v, 50) if v else float("nan")


def label(rows, marks, cap):
    out = []
    for b in marks["blocks"]:
        sp = b["t1"] - b["t0"]
        lo, hi = b["t0"] + sp * TRIM, b["t1"] - sp * TRIM
        for r in rows:
            if lo <= r["t"] <= hi:
                q = dict(r)
                q["cap"], q["block"], q["heading"] = cap, b["name"], b["headingGT"]
                if b["name"] in TURNED or b["name"] in TWIST_TURNED:
                    q["shGT"] = b["headingGT"]
                elif b["name"] in ZEROED:
                    q["shGT"] = 0
                else:
                    q["shGT"] = None
                q["grp"] = ("main" if b["name"] in TURNED else
                            "twist" if b["name"] in TWIST_TURNED else
                            "zero" if b["name"] in ZEROED else "motion")
                out.append(q)
    return out


def add_derived(rows):
    z = [r["uSpanSh"] * r["hipZ"] for r in rows if r["block"].startswith("h_0")]
    K = pct(z, 50) if z else float("nan")
    for r in rows:
        c = max(0.0, min(1.0, (r["uSpanSh"] * r["hipZ"]) / K)) if K == K else 0.0
        r["yaw2D"] = math.degrees(math.acos(c))
    return K


def infer_pol(rows, key):
    sel = [r for r in rows if r.get(key) is not None and r[key] != 0
           and r["shGT"] not in (None, 0)]
    if not sel:
        return 1
    ag = sum(1 for r in sel if (r[key] > 0) == (r["shGT"] > 0))
    return 1 if ag * 2 >= len(sel) else -1


def score(rows, key, pol, thr=0.0):
    sel = [r for r in rows if r.get(key) is not None and r["shGT"] not in (None, 0)]
    n_all = len(sel)
    use = [r for r in sel if abs(r[key]) > thr]

    def acc(pred):
        s = [r for r in use if pred(r)]
        if not s:
            return float("nan"), 0
        ok = sum(1 for r in s if sgn(r[key], pol, thr) == (1 if r["shGT"] > 0 else -1))
        return 100.0 * ok / len(s), len(s)

    o, no = acc(lambda r: True)
    p, _ = acc(lambda r: r["shGT"] > 0)
    n, _ = acc(lambda r: r["shGT"] < 0)
    a, _ = acc(lambda r: r["shGT"] == 90)
    b, _ = acc(lambda r: r["shGT"] == -90)
    ok_all = sum(1 for r in use if sgn(r[key], pol, thr) == (1 if r["shGT"] > 0 else -1))
    return (o, p, n, a, b, no, 100.0 * no / n_all if n_all else float("nan"),
            100.0 * ok_all / n_all if n_all else float("nan"), n_all)


def flips(seq_rows, key, pol):
    s = [r for r in seq_rows if r.get(key) is not None]
    if len(s) < 3:
        return 0, float("nan"), 0, 0, float("nan")
    s.sort(key=lambda r: r["t"])
    signs = [sgn(r[key], pol) for r in s]
    d = [x for x in signs if x != 0]
    tr = sum(1 for i in range(1, len(d)) if d[i] != d[i - 1])
    dur = s[-1]["t"] - s[0]["t"]
    run = best = 0
    prev = 0
    for x in signs:
        run = run + 1 if (x != 0 and x == prev) else (1 if x != 0 else run)
        if x != 0:
            prev = x
        best = max(best, run)
    return tr, (tr / dur if dur > 0.5 else float("nan")), best, len(s), 100.0 * len(d) / len(s)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    allrows, meta = [], []
    for cap, d, m in CAPTURES:
        if not os.path.isdir(d) or not os.path.exists(m):
            print("  [skip] %s" % cap)
            continue
        raw = load(d)
        marks = json.load(io.open(m, encoding="utf-8"))
        rows = label(raw, marks, cap)
        K = add_derived(rows)
        meta.append((cap, d, len(raw), len(rows), recover_fx(raw), K,
                     bool(marks.get("RECOVERED"))))
        allrows += rows
    if not allrows:
        print("no labelled frames")
        return 1

    W = 116
    main_t = [r for r in allrows if r["grp"] == "main"]
    twist_t = [r for r in allrows if r["grp"] == "twist"]
    zeros = [r for r in allrows if r["grp"] == "zero"]
    print("=" * W)
    print(" F-14 -- RAW SHOULDER DEPTH-DIFFERENCE SIGN   (offline; no production code touched)")
    print("=" * W)
    for cap, d, nraw, nlab, fx, K, rec in meta:
        print("  %-5s %-22s %6d frames -> %5d labelled | fx=%.1f px  K=%.1f%s"
              % (cap, d, nraw, nlab, fx, K, "   [marks RECOVERED]" if rec else ""))
    print("\n  main scored population : %4d frames (whole-body + body-turned blocks)" % len(main_t))
    print("  twist robustness group : %4d frames (tw_*, shoulders turned, COMPROMISED protocol --"
          % len(twist_t))
    print("                                 scored separately, never pooled into a headline)")
    print("  true-zero population   : %4d frames" % len(zeros))

    POL = dict((k, infer_pol(main_t, k)) for _t, k, _d in CANDS)
    print("\n  polarity inferred per candidate from the main group: %s"
          % ", ".join("%s=%+d" % (k, v) for k, v in sorted(POL.items())))

    # ---------------- 5. static ----------------------------------------------------------------
    print("\n" + "=" * W)
    print(" (5) STATIC SIGN ACCURACY vs shoulder ground truth   (main group; 0 deg excluded)")
    print("=" * W)
    print(" %-3s %-9s %7s %9s %9s %9s %9s %9s %9s %9s"
          % ("id", "signal", "defined", "coverage", "overall", "positive", "negative", "+90",
             "-90", "ALL frm"))
    for tag, key, _d in CANDS:
        o, p, n, a, b, no, cv, oa, _nt = score(main_t, key, POL[key])
        print(" %-3s %-9s %7d %8.1f%% %9s %9s %9s %9s %9s %9s"
              % (tag, key, no, cv, fmt(o), fmt(p), fmt(n), fmt(a), fmt(b), fmt(oa)))
    for tag, key, desc in CANDS:
        print("     %-3s %-9s %s" % (tag, key, desc))
    print("\n  ALL frm counts an undefined sign as wrong -- the production-facing number.")

    print("\n TWIST ROBUSTNESS GROUP (compromised protocol; separate, not pooled)")
    for tag, key, _d in CANDS:
        o, p, n, a, b, no, cv, oa, nt = score(twist_t, key, POL[key])
        if nt:
            print("   %-3s %-9s n=%4d  coverage %5.1f%%  overall %s  ALL %s"
                  % (tag, key, nt, cv, fmt(o), fmt(oa)))

    # per-heading coverage x accuracy
    print("\n PER-HEADING COVERAGE x ACCURACY -- where the pooled row hides an empty cell")
    print(" %-10s %-5s %7s %8s %10s %11s" % ("signal", "head", "n", "defined", "coverage",
                                             "acc|defined"))
    for _t, key, _d in CANDS:
        for nm, val in (("+45", 45), ("-45", -45), ("+90", 90), ("-90", -90)):
            s = [r for r in main_t if r["shGT"] == val and r.get(key) is not None]
            dd = [r for r in s if r[key] != 0]
            if not s:
                continue
            ok = sum(1 for r in dd if sgn(r[key], POL[key]) == (1 if val > 0 else -1))
            print(" %-10s %-5s %7d %8d %9.1f%% %10s"
                  % (key, nm, len(s), len(dd), 100.0 * len(dd) / len(s),
                     ("%.2f%%" % (100.0 * ok / len(dd))) if dd else "  none"))
        print()

    # ---------------- 6. block level ------------------------------------------------------------
    print("=" * W)
    print(" (6) BLOCK-LEVEL EVIDENCE -- the primary unit; frames inside a held pose are duplicates")
    print("=" * W)
    print(" %-6s %-18s %-5s %-6s | %-30s | %-30s"
          % ("cap", "block", "true", "grp", "RAW p30 dz", "EMITTED dz"))
    tally = {"dzP30": [0, 0, 0], "dzEmit": [0, 0, 0], "yaw3Dsig": [0, 0, 0]}
    tally_tw = {"dzP30": [0, 0, 0], "dzEmit": [0, 0, 0], "yaw3Dsig": [0, 0, 0]}
    for cap, _d, _a, _b, _fx, _K, _r in meta:
        for b in sorted(set(r["block"] for r in allrows
                            if r["cap"] == cap and r["grp"] in ("main", "twist"))):
            s = [r for r in allrows if r["cap"] == cap and r["block"] == b]
            if not s:
                continue
            t, cells = s[0]["shGT"], []
            sink = tally if s[0]["grp"] == "main" else tally_tw
            for key in ("dzP30", "dzEmit", "yaw3Dsig"):
                dd = [sgn(r[key], POL[key]) for r in s if r.get(key) is not None and r[key] != 0]
                if not dd:
                    sink[key][2] += 1
                    if key != "yaw3Dsig":
                        cells.append("%-30s" % "NO SIGNAL (0 frames defined)")
                    continue
                maj = 1 if sum(dd) > 0 else -1
                good = maj == (1 if t > 0 else -1)
                sink[key][0 if good else 1] += 1
                if key != "yaw3Dsig":
                    cells.append("%-8s %4d/%4d def, %3d%% agree"
                                 % ("CORRECT" if good else "WRONG", len(dd), len(s),
                                    round(100.0 * sum(1 for x in dd if x == maj) / len(dd))))
            print(" %-6s %-18s %-5s %-6s | %s | %s"
                  % (cap, b, t, s[0]["grp"], cells[0], cells[1]))
    print("\n %-24s %9s %8s %11s" % ("main group", "correct", "wrong", "no signal"))
    for lab, key in (("RAW p30 dz", "dzP30"), ("EMITTED dz", "dzEmit"),
                     ("sign(yaw3D)  [F-10]", "yaw3Dsig")):
        c, w, n = tally[key]
        print(" %-24s %9d %8d %11d   (of %d blocks)" % (lab, c, w, n, c + w + n))
    print(" %-24s %9s %8s %11s" % ("twist group", "correct", "wrong", "no signal"))
    for lab, key in (("RAW p30 dz", "dzP30"), ("EMITTED dz", "dzEmit"),
                     ("sign(yaw3D)  [F-10]", "yaw3Dsig")):
        c, w, n = tally_tw[key]
        print(" %-24s %9d %8d %11d   (of %d blocks)" % (lab, c, w, n, c + w + n))

    # ---------------- 7. distributions ----------------------------------------------------------
    print("\n" + "=" * W)
    print(" (7) RAW SHOULDER DEPTH DISTRIBUTIONS per true heading   (p30, mm)")
    print("=" * W)
    print(" %-9s %5s | %8s %8s | %8s %8s %8s %8s %8s %7s %7s"
          % ("heading", "n", "zL p50", "zR p50", "dz p10", "dz p25", "dz p50", "dz p75",
             "dz p90", "dz sd", "zero%"))
    for nm, val in (("0", 0), ("+45", 45), ("-45", -45), ("+90", 90), ("-90", -90)):
        s = [r for r in allrows if r["shGT"] == val and r["grp"] in ("main", "zero")
             and "dzP30" in r]
        if len(s) < 5:
            continue
        dz = [r["dzP30"] for r in s]
        print(" %-9s %5d | %8.1f %8.1f | %8.1f %8.1f %8.1f %8.1f %8.1f %7.1f %6.1f%%"
              % (nm, len(s), pct([r["zLP30"] for r in s], 50), pct([r["zRP30"] for r in s], 50),
                 pct(dz, 10), pct(dz, 25), pct(dz, 50), pct(dz, 75), pct(dz, 90), sd(dz),
                 100.0 * sum(1 for v in dz if v == 0) / len(dz)))
    print("\n MIRRORED-POSE SEPARABILITY (rank AUC; 0.500 = indistinguishable)")
    for la, lb, va, vb in (("+45", "-45", 45, -45), ("+90", "-90", 90, -90)):
        pa = [r["dzP30"] for r in main_t if r["shGT"] == va and "dzP30" in r]
        pb = [r["dzP30"] for r in main_t if r["shGT"] == vb and "dzP30" in r]
        if len(pa) < 5 or len(pb) < 5:
            continue
        u = auc(pa, pb)
        print("   %s vs %s : AUC %.3f  -> separability %.1f%%" % (la, lb, u,
                                                                  100.0 * max(u, 1 - u)))

    # ---------------- 8. geometry / quantisation ------------------------------------------------
    print("\n" + "=" * W)
    print(" (8) GEOMETRY AND QUANTISATION -- is the shoulder baseline big enough?")
    print("=" * W)
    lv = {}
    for r in allrows:
        for k in ("zLP30", "zRP30"):
            if k in r:
                lv[r[k]] = lv.get(r[k], 0) + 1
    lad = sorted(v for v, c in lv.items() if c >= 30 and 900.0 < v < 2200.0)
    fB = pct([v * round(21220.0 / v) for v in lad], 50) if lad else float("nan")
    zh = pct([r["hipZ"] for r in main_t], 50) * 1000.0
    dh = round(fB / zh)
    near, far = fB / (dh + 1), fB / (dh - 1)
    step = 0.5 * (far - near)
    print("   depth rungs near the torso: %s mm" % ", ".join("%.0f" % v for v in lad[:9]))
    print("   f*B = %.0f mm.px (re-derived here, matching F-13). At %.0f mm, d=%d, ONE STEP = "
          "%.0f-%.0f mm (mean %.0f)." % (fB, zh, dh, zh - near, far - zh, step))
    fx = pct([m[4] for m in meta], 50)
    sepmm = pct([r["uSpanSh"] * r["hipZ"] * 1000.0 / fx
                 for r in allrows if r["grp"] == "zero"], 50)
    print("\n   effective shoulder width, measured FRONT-ON (where cos(yaw)=1): %.0f mm" % sepmm)
    print("   %-8s %12s %14s %16s" % ("theta", "dz = W sin th", "disparity steps", "vs hips (F-13)"))
    for th in (15, 30, 45, 60, 90):
        w = sepmm * math.sin(math.radians(th))
        h = 130.0 * math.sin(math.radians(th))
        print("   %-8s %9.1f mm %13.2f %14.2fx" % ("%d deg" % th, w, w / step, w / h))
    print("\n   The hips gave 1.11 steps at 45 deg (F-13). The shoulders give %.2f."
          % (sepmm * math.sin(math.radians(45)) / step))

    # ---------------- 9. the +-90 test ----------------------------------------------------------
    print("\n" + "=" * W)
    print(" (9) THE +-90 TEST -- does raw dz keep a sign where yaw3D wraps?")
    print("=" * W)
    print(" %-8s %5s %9s %9s %9s %8s %9s %9s %9s"
          % ("heading", "n", "uSpan px", "|yaw3D|", "raw cov", "raw acc", "emit cov", "emit acc",
             "measured"))
    for nm, val in (("0", 0), ("+45", 45), ("-45", -45), ("+90", 90), ("-90", -90)):
        s = [r for r in allrows if r["shGT"] == val and r["grp"] in ("main", "zero")]
        if len(s) < 5:
            continue
        cells = []
        for key in ("dzP30", "dzEmit"):
            ss = [r for r in s if r.get(key) is not None]
            dd = [r for r in ss if r[key] != 0]
            cov = 100.0 * len(dd) / len(ss) if ss else float("nan")
            if val == 0 or not dd:
                cells += [cov, float("nan")]
            else:
                ok = sum(1 for r in dd if sgn(r[key], POL[key]) == (1 if val > 0 else -1))
                cells += [cov, 100.0 * ok / len(dd)]
        print(" %-8s %5d %9.1f %9.1f %8.1f%% %8s %8.1f%% %8s %8.1f%%"
              % (nm, len(s), pct([r["uSpanSh"] for r in s], 50),
                 pct([abs(r["yaw3D"]) for r in s], 50),
                 cells[0], fmt(cells[1]), cells[2], fmt(cells[3]),
                 50.0 * mean([r["measSh"] for r in s])))
    print("\n   Occlusion check at +-90 -- is a shoulder disappearing?")
    print(" %-8s %10s %10s %10s %10s" % ("heading", "conf p50", "conf p10", "valid px p50",
                                         "dsd p50"))
    for nm, val in (("+45", 45), ("-45", -45), ("+90", 90), ("-90", -90)):
        s = [r for r in main_t if r["shGT"] == val]
        if len(s) < 5:
            continue
        print(" %-8s %10.3f %10.3f %10d %10.1f"
              % (nm, pct([r["confShMin"] for r in s], 50), pct([r["confShMin"] for r in s], 10),
                 pct([r["nvMin"] for r in s], 50), pct([r["dsdShMax"] for r in s], 50)))

    # ---------------- 10. true zero -------------------------------------------------------------
    print("\n" + "=" * W)
    print(" (10) TRUE-ZERO SAFETY -- and the question that actually matters, the ANGLE error")
    print("=" * W)
    print(" %-6s %-18s %6s %9s %10s %10s %9s %11s"
          % ("cap", "block", "n", "defined", "|dz| p50", "|dz| p95", "flips/s", "|yaw2D| p50"))
    for cap, _d, _a, _b, _fx, _K, _r in meta:
        for nm in ZEROED:
            s = [r for r in zeros if r["cap"] == cap and r["block"] == nm and "dzP30" in r]
            if len(s) < 5:
                continue
            tr, fps, run, n, dfn = flips(s, "dzP30", POL["dzP30"])
            print(" %-6s %-18s %6d %8.1f%% %10.1f %10.1f %9.2f %11.1f"
                  % (cap, nm, n, dfn, pct([abs(r["dzP30"]) for r in s], 50),
                     pct([abs(r["dzP30"]) for r in s], 95), fps,
                     pct([r["yaw2D"] for r in s], 50)))
    print("\n   An arbitrary sign at a square torso is only a defect if it moves the AVATAR.")
    print("   Hypothetical signed angle at true 0, i.e. |sign(dz) * |yaw2D| - 0|:")
    print(" %-26s %8s %8s %8s %8s %8s" % ("estimator", "MAE", "RMSE", "p50", "p95", "max"))
    for lab, key in (("sign(RAW dz) * |yaw2D|", "dzP30"),
                     ("sign(EMITTED dz) * |yaw2D|", "dzEmit"),
                     ("production yaw3D", None)):
        if key is None:
            e = [abs(r["yaw3D"]) for r in zeros]
        else:
            e = [abs(sgn(r[key], POL[key]) * r["yaw2D"]) for r in zeros if r.get(key) is not None]
        if e:
            print(" %-26s %8.2f %8.2f %8.2f %8.2f %8.2f"
                  % (lab, mean(e), rmse(e), pct(e, 50), pct(e, 95), max(e)))

    # ---------------- 11. dynamic ---------------------------------------------------------------
    print("\n" + "=" * W)
    print(" (11) DYNAMIC BEHAVIOUR -- no instantaneous truth, so behaviour only")
    print("=" * W)
    print(" %-14s %5s | %8s %8s %8s %8s | %9s %10s"
          % ("block", "n", "raw cov", "raw f/s", "emit f/s", "longest", "raw dz sd", "emit dz sd"))
    for nm in MOTION:
        s = [r for r in allrows if r["block"] == nm]
        sr = [r for r in s if "dzP30" in r]
        if len(sr) < 5:
            continue
        _t1, f1, r1, _n1, d1 = flips(sr, "dzP30", POL["dzP30"])
        _t2, f2, _r2, _n2, _d2 = flips(s, "dzEmit", POL["dzEmit"])
        print(" %-14s %5d | %7.1f%% %8.2f %8.2f %8d | %9.1f %10.1f"
              % (nm, len(sr), d1, f1, f2, r1, sd([r["dzP30"] for r in sr]),
                 sd([r["dzEmit"] for r in s])))
    print("\n   Reference only: a subject physically reverses at most ~0.5 times/second (F-12).")

    # ---------------- 12. raw vs smoothed -------------------------------------------------------
    print("\n" + "=" * W)
    print(" (12) RAW vs SMOOTHED -- does the one-euro filter carry information or manufacture it?")
    print("=" * W)
    both = [r for r in main_t if "dzP30" in r]
    rd = [r for r in both if r["dzP30"] != 0]
    ru = [r for r in both if r["dzP30"] == 0]
    ag = sum(1 for r in rd if sgn(r["dzP30"], POL["dzP30"]) == sgn(r["dzEmit"], POL["dzEmit"]))
    print("   where RAW is defined (%d frames): emitted agrees with raw on %.2f%%"
          % (len(rd), 100.0 * ag / max(1, len(rd))))
    fu = [r for r in ru if r["dzEmit"] != 0]
    ok = sum(1 for r in fu if sgn(r["dzEmit"], POL["dzEmit"]) == (1 if r["shGT"] > 0 else -1))
    print("   where RAW is UNDEFINED (%d frames): emitted supplies a sign on %d of them (%.1f%%),"
          % (len(ru), len(fu), 100.0 * len(fu) / max(1, len(ru))))
    print("     and that supplied sign is CORRECT on %.2f%% -- this is the coverage the filter adds."
          % (100.0 * ok / max(1, len(fu))))
    print("\n   Smoothing recovers sub-rung information by averaging a quantised signal over a HELD")
    print("   pose (classic dithering) -- legitimate when static. In motion the same filter lags;")
    print("   compare the flips/s columns in section 11 for the cost.")

    # ---------------- 13. vs yaw3D ---------------------------------------------------------------
    print("\n" + "=" * W)
    print(" (13) AGAINST yaw3D -- does the composition destroy a sign the depth had?")
    print("=" * W)

    def conf_table(pop, lab):
        cells = [[0, 0], [0, 0]]
        for r in pop:
            if "dzP30" not in r or r["dzP30"] == 0:
                continue
            t = 1 if r["shGT"] > 0 else -1
            a = sgn(r["dzP30"], POL["dzP30"]) == t
            b = sgn(r["yaw3Dsig"], POL["yaw3Dsig"]) == t
            cells[0 if a else 1][0 if b else 1] += 1
        tot = sum(sum(x) for x in cells)
        print("   %s  (n=%d, frames where raw dz is defined)" % (lab, tot))
        print("     %-18s %14s %14s" % ("", "yaw3D correct", "yaw3D wrong"))
        print("     %-18s %14d %14d" % ("raw dz correct", cells[0][0], cells[0][1]))
        print("     %-18s %14d %14d" % ("raw dz wrong", cells[1][0], cells[1][1]))
        return cells
    c_all = conf_table(main_t, "ALL turned headings")
    print()
    c_90 = conf_table([r for r in main_t if abs(r["shGT"]) == 90], "ISOLATED to |heading| = 90")
    n_rescue = c_all[0][1]
    n_lose = c_all[1][0]
    print("\n   raw dz right where yaw3D is wrong : %d frames" % n_rescue)
    print("   raw dz wrong where yaw3D is right : %d frames" % n_lose)
    print("   at |heading|=90 specifically      : %d rescued, %d lost" % (c_90[0][1], c_90[1][0]))
    wrapped = [r for r in main_t if abs(r["yaw3D"]) > 150]
    print("\n   frames where |yaw3D| > 150 (the wrap signature): %d of %d = %.1f%%"
          % (len(wrapped), len(main_t), 100.0 * len(wrapped) / len(main_t)))
    if wrapped:
        by = {}
        for r in wrapped:
            by[r["shGT"]] = by.get(r["shGT"], 0) + 1
        print("     by heading: %s" % ", ".join("%+d deg: %d" % (k, v)
                                                for k, v in sorted(by.items())))
        ov = sum(1 for r in wrapped if "dzP30" in r and r["dzP30"] == 0)
        print("   of those wrapped frames, raw dz is ALSO undefined on %d = %.1f%%"
              % (ov, 100.0 * ov / len(wrapped)))
        print("   -> raw dz does not RESCUE the wrap; it goes SILENT in the same place.")

    # ---------------- 14. hybrids ----------------------------------------------------------------
    print("\n" + "=" * W)
    print(" (14) HYPOTHETICAL HYBRIDS -- NOT implemented")
    print("=" * W)
    pop = [r for r in main_t if "dzP30" in r]
    print(" %-32s %8s %8s %8s %8s %8s %10s"
          % ("estimator", "MAE", "RMSE", "p50", "p95", "max", "wrong sign"))

    def hybrid(lab, key, hold):
        e, wrong, last = [], 0, 1
        for r in sorted(pop, key=lambda x: x["t"]):
            s = sgn(r[key], POL[key])
            if s == 0:
                s = last if hold else 1
            else:
                last = s
            e.append(abs(s * r["yaw2D"] - r["shGT"]))
            if s != (1 if r["shGT"] > 0 else -1):
                wrong += 1
        print(" %-32s %8.2f %8.2f %8.2f %8.2f %8.2f %9.2f%%"
              % (lab, mean(e), rmse(e), pct(e, 50), pct(e, 95), max(e),
                 100.0 * wrong / len(e)))
    hybrid("shoulder RAW dz (hold on 0)", "dzP30", True)
    hybrid("shoulder RAW dz (no hold)", "dzP30", False)
    hybrid("shoulder SMOOTHED dz (hold)", "dzEmit", True)
    e = [abs(r["yaw3D"] - r["shGT"]) for r in pop]
    print(" %-32s %8.2f %8.2f %8.2f %8.2f %8.2f %10s"
          % ("production yaw3D (signed)", mean(e), rmse(e), pct(e, 50), pct(e, 95), max(e), "n/a"))
    m = [abs(r["yaw2D"] - abs(r["shGT"])) for r in pop]
    print(" %-32s %8.2f %8.2f %8.2f %8.2f %8.2f %10s"
          % ("yaw2D magnitude alone (unsigned)", mean(m), rmse(m), pct(m, 50), pct(m, 95), max(m),
             "n/a"))
    print("\n   published references on their own populations: F-10 depth-sign hybrid 13.09,")
    print("   F-11 face-sign 31.60, F-12 temporal 9.92 (unstable), F-13 hip-sign 72.46.")

    # ---------------- 14b. THE FAIR COMPARISON ---------------------------------------------------
    print("\n" + "=" * W)
    print(" (14b) THE FAIR COMPARISON -- give EVERY estimator the same hold, including yaw3D")
    print("=" * W)
    print(" Section 14 compares a HELD depth sign against an UNHELD yaw3D, which is not a fair")
    print(" test: F-10 itself proposed a wrap guard (hold the last sign when |yaw3D| > 150).")
    print(" Scoring that guard on the same frames is the experiment that decides F-14.\n")
    print(" %-16s %-34s %8s %8s %9s" % ("population", "estimator", "MAE", "RMSE", "wrong"))
    for plab, pp in (("main only", [r for r in main_t if "dzP30" in r]),
                     ("main + twist", [r for r in allrows if r["grp"] in ("main", "twist")
                                       and "dzP30" in r]),
                     ("twist only", [r for r in twist_t if "dzP30" in r])):
        pp = sorted(pp, key=lambda r: r["t"])
        if not pp:
            continue

        def run(nm, fn):
            e, w, last = [], 0, 1
            for r in pp:
                s = fn(r)
                if s == 0:
                    s = last
                else:
                    last = s
                e.append(abs(s * r["yaw2D"] - r["shGT"]))
                if s != (1 if r["shGT"] > 0 else -1):
                    w += 1
            print(" %-16s %-34s %8.2f %8.2f %8.2f%%"
                  % (plab, nm, mean(e), rmse(e), 100.0 * w / len(e)))
        run("sign(RAW dz), hold", lambda r: sgn(r["dzP30"], POL["dzP30"]))
        run("sign(EMITTED dz), hold", lambda r: sgn(r["dzEmit"], POL["dzEmit"]))
        run("sign(yaw3D) + F-10 WRAP GUARD",
            lambda r: 0 if abs(r["yaw3D"]) > 150 else sgn(r["yaw3D"], POL["yaw3Dsig"]))
        mm = [abs(r["yaw2D"] - abs(r["shGT"])) for r in pp]
        print(" %-16s %-34s %8.2f %8.2f %9s"
              % (plab, "PERFECT-SIGN CEILING (|yaw2D|)", mean(mm), rmse(mm), "--"))
        print()
    print("   If the guarded yaw3D matches the depth sign to the decimal, they are the SAME")
    print("   information and F-14's hypothesis -- that the composition destroys a sign the depth")
    print("   retains -- is refuted.")

    # ---------------- 14c. session dependence ----------------------------------------------------
    print("=" * W)
    print(" (14c) SESSION DEPENDENCE -- acceptance criterion 12")
    print("=" * W)
    print(" %-6s %-22s %7s %9s %10s %10s" % ("cap", "signal", "n", "coverage", "acc|def",
                                             "ALL frm"))
    for cap, _d, _a, _b, _fx, _K, _r in meta:
        sel = [r for r in main_t if r["cap"] == cap]
        for lab, key in (("RAW p30 dz", "dzP30"), ("EMITTED dz", "dzEmit"),
                         ("sign(yaw3D)", "yaw3Dsig")):
            s = [r for r in sel if r.get(key) is not None]
            dd = [r for r in s if r[key] != 0]
            if not s:
                continue
            ok = sum(1 for r in dd if sgn(r[key], POL[key]) == (1 if r["shGT"] > 0 else -1))
            print(" %-6s %-22s %7d %8.1f%% %10s %9.2f%%"
                  % (cap, lab, len(s), 100.0 * len(dd) / len(s),
                     ("%.2f%%" % (100.0 * ok / len(dd))) if dd else "  none", 100.0 * ok / len(s)))
        print()
    print("   Two captures of the SAME subject at the SAME distance on the SAME day. A large gap")
    print("   here means the headline numbers are session-conditioned, not a property of the rig.")

    # ---------------- 15. >90 --------------------------------------------------------------------
    print("\n" + "=" * W)
    print(" (15) THE >90 LIMIT -- unchanged by any sign result")
    print("=" * W)
    m180 = [r for r in allrows if r["block"] == "m_180"]
    if m180:
        y = [r["yaw2D"] for r in m180]
        print("   m_180: |yaw2D| p50 %.1f  p95 %.1f  MAX %.1f deg over %d frames"
              % (pct(y, 50), pct(y, 95), max(y), len(y)))
    print("   SIGN and MAGNITUDE-BEYOND-90 are separate problems. A working sign gives")
    print("   -90..+90 only; acos() cannot represent 120 deg. Do not conflate them.")

    # ---------------- 16. falsification ----------------------------------------------------------
    print("\n" + "=" * W)
    print(" (16) FALSIFICATION")
    print("=" * W)
    pol = POL["dzP30"]

    def case(lab, sel, pred):
        n = len(sel)
        k = sum(1 for r in sel if pred(r))
        print("   %-58s %5d/%-5d = %5.1f%%" % (lab, k, n, 100.0 * k / n if n else float("nan")))
        return k, n
    case("A  true +45 but raw dz undefined or wrong",
         [r for r in main_t if r["shGT"] == 45 and "dzP30" in r],
         lambda r: r["dzP30"] == 0 or sgn(r["dzP30"], pol) != 1)
    case("B  true -45 but raw dz undefined or wrong",
         [r for r in main_t if r["shGT"] == -45 and "dzP30" in r],
         lambda r: r["dzP30"] == 0 or sgn(r["dzP30"], pol) != -1)
    p9 = [r for r in main_t if r["shGT"] == 90 and "dzP30" in r and r["dzP30"] != 0]
    m9 = [r for r in main_t if r["shGT"] == -90 and "dzP30" in r and r["dzP30"] != 0]
    print("   %-58s +90 def %d, -90 def %d" % ("C  +-90 give same sign or no signal", len(p9),
                                               len(m9)))
    if p9 and m9:
        print("      %-55s +90 -> %+d, -90 -> %+d  (%s)" %
              ("", 1 if sum(sgn(r["dzP30"], pol) for r in p9) > 0 else -1,
               1 if sum(sgn(r["dzP30"], pol) for r in m9) > 0 else -1,
               "DISTINCT" if (sum(sgn(r["dzP30"], pol) for r in p9) > 0) !=
                             (sum(sgn(r["dzP30"], pol) for r in m9) > 0) else "SAME -- fails"))
    case("D  true 0 with |dz| >= one disparity step", zeros,
         lambda r: "dzP30" in r and abs(r["dzP30"]) >= step)
    tot_fl = tot_n = 0
    for nm in ZEROED:
        s = [r for r in zeros if r["block"] == nm and "dzP30" in r]
        if len(s) >= 5:
            tr, _f, _r, _n, _d = flips(s, "dzP30", pol)
            tot_fl += tr
            tot_n += 1
    print("   %-58s %d flips over %d held blocks" % ("E  raw sign changes during a STABLE pose",
                                                     tot_fl, tot_n))
    fu2 = [r for r in main_t if "dzP30" in r and r["dzP30"] != 0]
    flip_after = sum(1 for r in fu2
                     if sgn(r["dzP30"], pol) == (1 if r["shGT"] > 0 else -1)
                     and sgn(r["dzEmit"], POL["dzEmit"]) != (1 if r["shGT"] > 0 else -1))
    print("   %-58s %5d/%-5d = %5.1f%%" % ("F  raw CORRECT but smoothing makes it WRONG",
                                           flip_after, len(fu2),
                                           100.0 * flip_after / max(1, len(fu2))))
    print("   %-58s see section 13 (%d frames rescued at |90|)"
          % ("G  yaw3D and raw dz disagree at edge-on poses", c_90[0][1]))
    hset = {}
    for r in main_t:
        if "dzP30" in r and r["dzP30"] != 0:
            hset.setdefault(r["shGT"], {}).setdefault(r["cap"] + "/" + r["block"], []).append(
                sgn(r["dzP30"], pol))
    bad_h = 0
    for h, blocks in sorted(hset.items()):
        majs = set(1 if sum(v) > 0 else -1 for v in blocks.values())
        if len(majs) > 1:
            bad_h += 1
        print("   %-58s %+4d deg -> %d block(s), signs %s"
              % ("H  same heading, different block-level sign" if h == sorted(hset)[0] else "",
                 h, len(blocks), sorted(majs)))
    print("      %d of %d headings show a block-level sign disagreement." % (bad_h, len(hset)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
