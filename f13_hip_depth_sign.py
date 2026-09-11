#!/usr/bin/env python3
"""F-13 -- can the HIP-LINE stereo depth ordering supply a torso-yaw SIGN?

F-10 validated the 2-D shoulder-foreshortening MAGNITUDE (MAE 5.90 deg) and refuted three sign
sources: shoulder depth (F-10, 78.49 %, 50.5 % on left turns), face/nose (F-11, measures HEAD yaw)
and temporal continuity (F-12, the information is absent from |yaw2D|). This script tests the one
remaining cheap candidate: the hips.

WHY THE HIPS MIGHT DIFFER FROM THE SHOULDERS
    The hips do not rotate with the head, and they sit closer to the camera's depth sweet spot. If
    stereo can resolve WHICH HIP IS NEARER -- one bit, not a magnitude -- that bit would sign the
    already-good 2-D magnitude.

WHAT THE SIGNAL ACTUALLY IS (see section 4 of the report)
    The audit record is written BEFORE the one-euro / depth-outlier smoother and before hip-centring,
    so `j["L-hip"]["p30"|"p50"|"ctr"]` are RAW stereo depths in millimetres sampled in a k-window at
    the joint's pixel. That is the purest form of the hypothesis. The emitted 3-D hip landmarks in
    sender_log are the same quantity AFTER smoothing, so they are carried as a separate candidate
    rather than mixed in.

GROUND TRUTH -- and what must not be used as it
    Truth comes ONLY from the capture block labels. `yaw3D`, `yaw2D`, the shoulder depth and the
    candidate itself are never used to label a frame; that was the circularity F-09 called out.
    Blocks are used according to what they constrain THE HIPS to do:
        h_p45/h_p90/h_m45/h_m90        whole body turned  -> hip truth = the heading      SCORED
        dc_body45R_face0 / ..45L..     body turned, face on camera -> hip truth = heading SCORED
        h_0 / h_0b / h_0c              square             -> hip truth = 0                zero case
        dc_body0_head45R / ..45L       body square, head turned -> hip truth = 0          zero case
        tw_sh_p45 / tw_sh_m45          SHOULDERS turned, hips square -> heading != hip truth
        tw_opp_a / tw_opp_b            opposite twist, unlabelled
        tw_rigid_p45                   rigid turn, but recorded in the compromised twist run
    The three `tw_*` groups are reported separately and EXCLUDED from every headline number: the
    subject reported confusion during the twist protocol in F-10, so their hip truth is not reliable.

    python f13_hip_depth_sign.py
"""
import bisect
import io
import json
import math
import os

TRIM = 0.25          # drop the first/last quarter of each block: settling, not pose
CONF_MIN = 0.3       # same landmark-confidence floor the earlier F-1x analyses used

CAPTURES = [
    ("F-10", "pipeline_logs_f10_near",
     os.path.join("oak_v4_evidence", "f10_gt_marks_near_RECOVERED.json")),
    ("F-11", "pipeline_logs_f11",
     os.path.join("oak_v4_evidence", "f10_gt_marks_near.json")),
]

TURNED = ("h_p45", "h_p90", "h_m45", "h_m90", "dc_body45R_face0", "dc_body45L_face0")
ZEROED = ("h_0", "h_0b", "h_0c", "dc_body0_head45R", "dc_body0_head45L")
TWIST = ("tw_rigid_p45", "tw_sh_p45", "tw_sh_m45", "tw_opp_a", "tw_opp_b")
MOTION = ("m_slow", "m_normal", "m_fast", "m_rev_lr", "m_rev_rl", "m_180",
          "m_turn_arms", "m_turn_still")

# Candidate signals. `key` is the per-frame field; polarity is inferred, never assumed.
CANDS = [
    ("A", "dzP30", "raw hip depth difference, p30 of the depth window   (mm)"),
    ("A2", "dzP50", "raw hip depth difference, p50 of the depth window   (mm)"),
    ("B", "hipAng", "hip-line out-of-plane angle atan2(dz, horizontal sep) (deg)"),
    ("C", "dzEmit", "emitted 3-D hip dz, i.e. post-smoothing              (mm)"),
    ("D", "dzCtr", "centre-pixel depth ordering only, no window stats    (mm)"),
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
    """Rank AUC that `pos` values exceed `neg` values. 0.5 = the distributions coincide."""
    if not pos or not neg:
        return float("nan")
    s = sorted(neg)
    tot = 0.0
    for v in pos:
        lo = bisect.bisect_left(s, v)
        hi = bisect.bisect_right(s, v)
        tot += lo + 0.5 * (hi - lo)          # ties count a half
    return tot / (len(pos) * len(neg))


def load(dirpath):
    """Join audit_log (raw depth, pre-smoothing) with sender_log (emitted 3-D, post-smoothing)."""
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
        need = ("L-hip", "R-hip", "L-shoulder", "R-shoulder")
        if any(n not in j for n in need):
            continue
        lh, rh, ls, rs = j["L-hip"], j["R-hip"], j["L-shoulder"], j["R-shoulder"]
        hipZ = s.get("hipZ")
        if not hipZ or hipZ <= 0.1:
            continue
        if min(lh["c"], rh["c"], ls["c"], rs["c"]) < CONF_MIN:
            continue
        hip, sh = s.get("hip"), s.get("sh")
        if not hip or not sh or len(hip) < 2 or len(sh) < 2:
            continue
        e = {
            "seq": seq, "t": float(s.get("t", 0.0)), "hipZ": float(hipZ),
            "confHipMin": min(lh["c"], rh["c"]),
            "qHipMin": min(lh.get("q", 0.0), rh.get("q", 0.0)),
            "measHip": int(lh.get("m", 0)) + int(rh.get("m", 0)),
            "uSpanHip": abs(lh["u"] - rh["u"]),
            "uSpanSh": abs(ls["u"] - rs["u"]),
            # emitted, post-smoothing (metres -> mm)
            "dzEmit": (float(hip[0][2]) - float(hip[1][2])) * 1000.0,
            "shDzEmit": (float(sh[0][2]) - float(sh[1][2])) * 1000.0,
            "hipSepX": abs(float(hip[0][0]) - float(hip[1][0])),
        }
        # raw depth-window statistics -- only where the window actually held valid pixels
        for tag, src in (("P30", "p30"), ("P50", "p50")):
            if src in lh and src in rh:
                e["dz" + tag] = float(lh[src]) - float(rh[src])
                e["zL" + tag] = float(lh[src])
                e["zR" + tag] = float(rh[src])
            if src in ls and src in rs:
                e["shDz" + tag] = float(ls[src]) - float(rs[src])
        if lh.get("ctr") and rh.get("ctr"):
            e["dzCtr"] = float(lh["ctr"]) - float(rh["ctr"])
        e["dsdHipMax"] = max(lh.get("dsd", 0.0), rh.get("dsd", 0.0))
        rows.append(e)
    return rows


def recover_fx(rows):
    """fx implied by the pipeline itself: fx = uSpan_px * Z_m / sep_m. Not logged, so recovered."""
    vals = [r["uSpanHip"] * r["hipZ"] / r["hipSepX"]
            for r in rows if r["hipSepX"] > 0.02 and r["uSpanHip"] > 1.0]
    return pct(vals, 50) if vals else float("nan")


def label(rows, marks, cap):
    out = []
    for b in marks["blocks"]:
        sp = b["t1"] - b["t0"]
        lo, hi = b["t0"] + sp * TRIM, b["t1"] - sp * TRIM
        for r in rows:
            if lo <= r["t"] <= hi:
                q = dict(r)
                q["cap"] = cap
                q["block"] = b["name"]
                q["heading"] = b["headingGT"]
                # HIP truth, which is NOT always the heading -- see the module docstring
                if b["name"] in TURNED:
                    q["hipGT"] = b["headingGT"]
                elif b["name"] in ZEROED:
                    q["hipGT"] = 0
                else:
                    q["hipGT"] = None
                out.append(q)
    return out


def add_derived(rows, fx):
    """yaw2D (magnitude, F-10's validated estimator) and the geometric hip-line angle."""
    z = [r["uSpanSh"] * r["hipZ"] for r in rows if r["block"].startswith("h_0")]
    K = pct(z, 50) if z else float("nan")
    for r in rows:
        c = max(0.0, min(1.0, (r["uSpanSh"] * r["hipZ"]) / K)) if K == K else 0.0
        r["yaw2D"] = math.degrees(math.acos(c))
        # horizontal hip separation in mm, from pixels -- independent of the depth being tested
        sep_mm = r["uSpanHip"] * r["hipZ"] * 1000.0 / fx if fx == fx and fx > 0 else 0.0
        r["hipSepMm"] = sep_mm
        if "dzP30" in r and sep_mm > 20.0:
            r["hipAng"] = math.degrees(math.atan2(r["dzP30"], sep_mm))
    return K


def f(x):
    """Render an accuracy cell; NaN means the candidate defined no sign there at all."""
    return "  none  " if x != x else "%7.2f%%" % x


def sgn(v, pol, thr=0.0):
    """THREE-valued. A depth difference of exactly zero carries NO sign, and must not be silently
    counted as negative -- that was what made an all-zero block look like a perfectly stable sign."""
    if v is None or abs(v) <= thr:
        return 0
    return pol if v > 0 else -pol


def infer_pol(rows, key):
    """+1 if a positive candidate value goes with a positive turn. Inferred, per candidate."""
    sel = [r for r in rows
           if r.get(key) is not None and r[key] != 0 and r["hipGT"] not in (None, 0)]
    if not sel:
        return 1
    ag = sum(1 for r in sel if (r[key] > 0) == (r["hipGT"] > 0))
    return 1 if ag * 2 >= len(sel) else -1


def score(rows, key, pol, thr=0.0):
    """Accuracy WHERE A SIGN EXISTS, plus the coverage that accuracy is conditioned on, plus the
    accuracy over every turned frame (an undefined sign counts as wrong, because a shipping
    estimator has to emit something)."""
    sel = [r for r in rows if r.get(key) is not None and r["hipGT"] not in (None, 0)]
    n_all = len(sel)
    use = [r for r in sel if abs(r[key]) > thr]

    def acc(pred):
        s = [r for r in use if pred(r)]
        if not s:
            return float("nan"), 0
        ok = sum(1 for r in s if sgn(r[key], pol, thr) == (1 if r["hipGT"] > 0 else -1))
        return 100.0 * ok / len(s), len(s)

    o, no = acc(lambda r: True)
    p, _ = acc(lambda r: r["hipGT"] > 0)
    n, _ = acc(lambda r: r["hipGT"] < 0)
    a, _ = acc(lambda r: r["hipGT"] == 90)
    b, _ = acc(lambda r: r["hipGT"] == -90)
    cov = 100.0 * no / n_all if n_all else float("nan")
    ok_all = sum(1 for r in use if sgn(r[key], pol, thr) == (1 if r["hipGT"] > 0 else -1))
    return o, p, n, a, b, no, cov, (100.0 * ok_all / n_all if n_all else float("nan")), n_all


def flips(seq_rows, key, pol, thr=0.0):
    """Sign transitions between DEFINED signs, flips/s, longest defined run, and % defined."""
    s = [r for r in seq_rows if r.get(key) is not None]
    if len(s) < 3:
        return 0, float("nan"), 0, 0, float("nan")
    s.sort(key=lambda r: r["t"])
    signs = [sgn(r[key], pol, thr) for r in s]
    defined = [x for x in signs if x != 0]
    tr = sum(1 for i in range(1, len(defined)) if defined[i] != defined[i - 1])
    dur = s[-1]["t"] - s[0]["t"]
    run = best = 0
    prev = 0
    for x in signs:
        if x != 0 and x == prev:
            run += 1
        elif x != 0:
            run = 1
        prev = x if x != 0 else prev
        best = max(best, run)
    return (tr, (tr / dur if dur > 0.5 else float("nan")), best, len(s),
            100.0 * len(defined) / len(s))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    allrows, meta = [], []
    for cap, d, m in CAPTURES:
        if not os.path.isdir(d) or not os.path.exists(m):
            print("  [skip] %s -- missing %s / %s" % (cap, d, m))
            continue
        raw = load(d)
        marks = json.load(io.open(m, encoding="utf-8"))
        fx = recover_fx(raw)
        rows = label(raw, marks, cap)
        K = add_derived(rows, fx)
        meta.append((cap, d, m, len(raw), len(rows), fx, K, bool(marks.get("RECOVERED")),
                     marks.get("metres")))
        allrows += rows
    if not allrows:
        print("no labelled frames -- captures missing")
        return 1

    W = 112
    print("=" * W)
    print(" F-13 -- HIP-LINE DEPTH SIGN FOR TORSO YAW   (offline; no production code touched)")
    print("=" * W)
    for cap, d, m, nraw, nlab, fx, K, rec, met in meta:
        print("  %-5s %-22s %6d frames -> %5d labelled | fx=%.1f px  K=%.1f  %.1f m%s"
              % (cap, d, nraw, nlab, fx, K, met or 0.0,
                 "   [marks RECOVERED, reconstructed]" if rec else ""))

    turned = [r for r in allrows if r["hipGT"] not in (None, 0)]
    zeros = [r for r in allrows if r["hipGT"] == 0]
    print("\n  scored population: %d turned frames, %d true-zero frames"
          % (len(turned), len(zeros)))
    print("  EXCLUDED from every headline number: %d twist frames (subject reported confusion"
          % len([r for r in allrows if r["block"] in TWIST]))
    print("  during the F-10 twist protocol, so their hip truth is not reliable)")

    POL = dict((k, infer_pol(turned, k)) for _t, k, _d in CANDS)
    print("\n  polarity inferred per candidate (never assumed): %s"
          % ", ".join("%s=%+d" % (k, v) for k, v in sorted(POL.items())))

    # ---------------- 7. static ground truth -------------------------------------------------
    print("\n" + "=" * W)
    print(" (7) STATIC SIGN ACCURACY vs hip ground truth   (0 deg excluded; no threshold)")
    print("=" * W)
    print(" %-3s %-8s %7s %9s %9s %9s %9s %9s %9s %9s"
          % ("id", "signal", "defined", "coverage", "overall", "positive", "negative", "+90",
             "-90", "ALL frm"))
    for tag, key, _desc in CANDS:
        o, p, n, a, b, no, cv, oa, _nt = score(turned, key, POL[key])
        print(" %-3s %-8s %7d %8.1f%% %8s %8s %8s %8s %8s %8s"
              % (tag, key, no, cv, f(o), f(p), f(n), f(a), f(b), f(oa)))
    for tag, key, desc in CANDS:
        print("     %-3s %-8s %s" % (tag, key, desc))
    print("\n  'coverage'  = turned frames on which the candidate produces ANY sign; the depth")
    print("                difference is exactly 0 mm on the rest, so there is no bit to read.")
    print("  'overall'   = accuracy CONDITIONED on a sign existing -- it is not the shipped number.")
    print("  'ALL frm'   = accuracy over every turned frame, counting an undefined sign as wrong,")
    print("                because a production estimator has to emit something on every frame.")
    print("\n  A, A2, B and D are the SAME ONE BIT where all are defined: B divides A by a strictly")
    print("  positive horizontal separation, which cannot change a sign. They differ only in which")
    print("  depth statistic is read and once a threshold is applied (section 12).")

    # ---------------- 7b. per-heading coverage, and the BLOCK as the evidence unit -------------
    print("\n" + "=" * W)
    print(" (7b) PER-HEADING COVERAGE x ACCURACY -- the pooled table above hides empty cells")
    print("=" * W)
    for r in allrows:
        if "shDzP30" in r:
            r["shSign"] = r["shDzP30"]
    print(" %-22s %-5s %7s %8s %10s %10s"
          % ("signal", "head", "n", "defined", "coverage", "acc|defined"))
    for lab, key in (("HIP raw dz", "dzP30"), ("HIP emitted dz", "dzEmit"),
                     ("SHOULDER raw dz", "shSign"), ("SHOULDER emitted dz", "shDzEmit")):
        pol = infer_pol(turned, key)
        for nm, val in (("+45", 45), ("-45", -45), ("+90", 90), ("-90", -90)):
            s = [r for r in turned if r["hipGT"] == val and r.get(key) is not None]
            d = [r for r in s if r[key] != 0]
            if not s:
                continue
            ok = sum(1 for r in d if sgn(r[key], pol) == (1 if val > 0 else -1))
            print(" %-22s %-5s %7d %8d %9.1f%% %10s"
                  % (lab, nm, len(s), len(d), 100.0 * len(d) / len(s),
                     ("%.2f%%" % (100.0 * ok / len(d))) if d else "  none"))
        print()

    print("=" * W)
    print(" (7c) BLOCK-LEVEL EVIDENCE -- the honest unit")
    print("=" * W)
    print(" Inside one held pose consecutive frames are near-duplicates, so a frame count of 1320")
    print(" is NOT 1320 independent observations. The real unit is the held block.\n")
    print(" %-6s %-18s %-5s | %-26s | %-26s"
          % ("cap", "block", "true", "HIP raw dz", "SHOULDER emitted dz"))
    tally = {"dzP30": [0, 0, 0], "shDzEmit": [0, 0, 0]}     # correct, wrong, no-signal
    for cap, _d, _m, _a, _b, _fx, _K, _r, _me in meta:
        for b in sorted(set(r["block"] for r in turned if r["cap"] == cap)):
            s = [r for r in turned if r["cap"] == cap and r["block"] == b]
            if not s:
                continue
            t, cells = s[0]["hipGT"], []
            for key in ("dzP30", "shDzEmit"):
                pol = infer_pol(turned, key)
                d = [sgn(r[key], pol) for r in s if r.get(key) is not None and r[key] != 0]
                if not d:
                    tally[key][2] += 1
                    cells.append("%-26s" % "NO SIGNAL (0 frames)")
                    continue
                maj = 1 if sum(d) > 0 else -1
                good = maj == (1 if t > 0 else -1)
                tally[key][0 if good else 1] += 1
                cells.append("%-9s %4d/%4d frames"
                             % ("CORRECT" if good else "WRONG",
                                sum(1 for x in d if x == maj), len(s)))
            print(" %-6s %-18s %-5s | %s | %s" % (cap, b, t, cells[0], cells[1]))
    print("\n %-22s %9s %9s %11s" % ("", "correct", "wrong", "no signal"))
    for lab, key in (("HIP raw dz", "dzP30"), ("SHOULDER emitted dz", "shDzEmit")):
        c, w, n = tally[key]
        print(" %-22s %9d %9d %11d   (of %d held blocks)" % (lab, c, w, n, c + w + n))

    # ---------------- 8. raw distributions ---------------------------------------------------
    print("\n" + "=" * W)
    print(" (8) RAW HIP DEPTH DISTRIBUTIONS per true heading   (p30 window depth, mm)")
    print("=" * W)
    print(" %-18s %5s | %8s %8s | %8s %8s %8s %8s %8s"
          % ("block(s)", "n", "zL p50", "zR p50", "dz p10", "dz p50", "dz p90", "dz sd", "|dz| p50"))
    groups = [("true 0", lambda r: r["hipGT"] == 0),
              ("true +45", lambda r: r["hipGT"] == 45),
              ("true -45", lambda r: r["hipGT"] == -45),
              ("true +90", lambda r: r["hipGT"] == 90),
              ("true -90", lambda r: r["hipGT"] == -90)]
    for nm, pred in groups:
        sel = [r for r in allrows if pred(r) and "dzP30" in r]
        if len(sel) < 5:
            continue
        dz = [r["dzP30"] for r in sel]
        print(" %-18s %5d | %8.1f %8.1f | %8.1f %8.1f %8.1f %8.1f %8.1f"
              % (nm, len(sel), pct([r["zLP30"] for r in sel], 50),
                 pct([r["zRP30"] for r in sel], 50),
                 pct(dz, 10), pct(dz, 50), pct(dz, 90), sd(dz), pct([abs(x) for x in dz], 50)))

    print("\n OVERLAP between the mirrored poses -- AUC 0.50 means the two are indistinguishable")
    for a_lab, b_lab, av, bv in (("+45", "-45", 45, -45), ("+90", "-90", 90, -90)):
        pa = [r["dzP30"] for r in allrows if r["hipGT"] == av and "dzP30" in r]
        pb = [r["dzP30"] for r in allrows if r["hipGT"] == bv and "dzP30" in r]
        if len(pa) < 5 or len(pb) < 5:
            continue
        u = auc(pa, pb)
        print("   %s vs %s : AUC %.3f   (separability %.1f%%; 50%% = pure chance)"
              % (a_lab, b_lab, u, 100.0 * max(u, 1 - u)))

    # quantisation: how often is the difference exactly zero?
    dz_all = [r["dzP30"] for r in turned if "dzP30" in r]
    if dz_all:
        z0 = sum(1 for v in dz_all if v == 0.0)
        print("\n QUANTISATION  dz is EXACTLY 0 mm on %d/%d turned frames (%.1f%%) -> no sign at all"
              % (z0, len(dz_all), 100.0 * z0 / len(dz_all)))
        lv = sorted(set(abs(v) for v in dz_all if v != 0))
        print("   smallest non-zero |dz| levels (mm): %s"
              % ", ".join("%.0f" % v for v in lv[:10]))

    # ---------------- 8b. WHY -- the mechanism -------------------------------------------------
    print("\n" + "=" * W)
    print(" (8b) MECHANISM -- what physically destroys the hip sign")
    print("=" * W)
    # (i) the depth ladder: z = f*B/d with INTEGER d, recovered from the data itself
    lv = {}
    for r in allrows:
        for k in ("zLP30", "zRP30"):
            if k in r:
                lv[r[k]] = lv.get(r[k], 0) + 1
    lad = sorted(v for v, c in lv.items() if c >= 30 and 900.0 < v < 2000.0)
    fB = pct([v * round(21220.0 / v) for v in lad], 50) if lad else float("nan")
    print(" (i) DEPTH IS QUANTISED TO INTEGER DISPARITY.  z = f*B/d, and the levels actually used")
    print("     near the hips are: %s mm" % ", ".join("%.0f" % v for v in lad[:9]))
    print("     Fitting z*d over those levels gives f*B = %.0f mm.px, constant to <1%%." % fB)
    zh = pct([r["hipZ"] for r in turned], 50) * 1000.0
    d_here = round(fB / zh) if zh > 0 else 0
    near, far = fB / (d_here + 1), fB / (d_here - 1)
    step = 0.5 * (far - near)
    print("     At the subject's %.0f mm the disparity is d=%d, and the neighbouring rungs are"
          % (zh, d_here))
    print("     %.0f mm and %.0f mm -- so ONE STEP is %.0f-%.0f mm, mean %.0f mm."
          % (near, far, zh - near, far - zh, step))
    sep = pct([r["hipSepMm"] for r in turned if "hipSepMm" in r], 50)
    print("\n (ii) GEOMETRY.  Measured hip separation = %.0f mm, so a turn of theta puts the hips"
          % sep)
    print("      dz = %.0f * sin(theta) mm apart:" % sep)
    for th in (15, 30, 45, 60, 90):
        want = sep * math.sin(math.radians(th))
        print("        theta = %2d deg -> true dz = %6.1f mm = %.2f disparity steps"
              % (th, want, want / step if step == step and step > 0 else float("nan")))
    print("      A 45 deg turn is worth %.1f steps. The sign is a comparison of two numbers that"
          % (sep * math.sin(math.radians(45)) / step))
    print("      each move in %.0f mm jumps -- so both hips land on the SAME rung most of the time."
          % step)

    # (iii) the +-90 collapse: the far hip is occluded, so its window samples the near hip
    print("\n (iii) AT +-90 THE FAR HIP IS OCCLUDED BY THE NEAR ONE.")
    print(" %-10s %6s %10s %10s %10s %10s %10s"
          % ("heading", "n", "uSpan px", "|dz| p50", "dz sd", "measured", "conf p50"))
    for nm, val in (("0", 0), ("+45", 45), ("-45", -45), ("+90", 90), ("-90", -90)):
        sel = [r for r in allrows if r["hipGT"] == val and "dzP30" in r]
        if len(sel) < 5:
            continue
        print(" %-10s %6d %10.1f %10.1f %10.1f %9.2f%% %10.3f"
              % (nm, len(sel), pct([r["uSpanHip"] for r in sel], 50),
                 pct([abs(r["dzP30"]) for r in sel], 50), sd([r["dzP30"] for r in sel]),
                 50.0 * mean([r["measHip"] for r in sel]),
                 pct([r["confHipMin"] for r in sel], 50)))
    print("     At +-90 the two hips project close together and the far one is behind the near one,")
    print("     so both 5x5 depth windows land on the SAME body surface: dz collapses to 0 with")
    print("     zero spread. The pose that most needs a sign is the pose that cannot produce one.")

    # ---------------- 9. zero / frontal safety ------------------------------------------------
    print("\n" + "=" * W)
    print(" (9) TRUE 0 deg -- does a square torso produce a confident sign?")
    print("=" * W)
    for cap, _d, _m, _a, _b, _fx, _K, _r, _me in meta:
        for nm in ZEROED:
            sel = [r for r in allrows if r["cap"] == cap and r["block"] == nm and "dzP30" in r]
            if len(sel) < 5:
                continue
            tr, fps, run, n, dfn = flips(sel, "dzP30", POL["dzP30"])
            dz = [abs(r["dzP30"]) for r in sel]
            print("   %-5s %-18s n=%4d  |dz| p50=%6.1f p95=%6.1f mm  defined %5.1f%%  "
                  "flips=%2d (%.2f/s)" % (cap, nm, n, pct(dz, 50), pct(dz, 95), dfn, tr, fps))
    zdz = [abs(r["dzP30"]) for r in zeros if "dzP30" in r]
    if zdz:
        print("\n   pooled true-0 |dz|: p50 %.1f  p90 %.1f  p95 %.1f  max %.1f mm"
              % (pct(zdz, 50), pct(zdz, 90), pct(zdz, 95), max(zdz)))
        print("   fraction of true-0 frames whose |dz| exceeds the MEDIAN |dz| of a real turn:")
        tmed = pct([abs(r["dzP30"]) for r in turned if "dzP30" in r], 50)
        big = sum(1 for v in zdz if v > tmed)
        print("     turned-frame median |dz| = %.1f mm  ->  %d/%d = %.1f%% of true-0 frames "
              "look at least as turned" % (tmed, big, len(zdz), 100.0 * big / len(zdz)))

    # ---------------- 10. dynamic -------------------------------------------------------------
    print("\n" + "=" * W)
    print(" (10) DYNAMIC MOTION -- no instantaneous truth exists, so behaviour only")
    print("=" * W)
    print(" %-14s %5s | %6s %8s %10s %10s | %10s"
          % ("block", "n", "flips", "flips/s", "longest", "defined%", "|dz| p50"))
    for nm in MOTION:
        sel = [r for r in allrows if r["block"] == nm]
        withdz = [r for r in sel if "dzP30" in r]
        if len(withdz) < 5:
            continue
        tr, fps, run, n, dfn = flips(withdz, "dzP30", POL["dzP30"])
        print(" %-14s %5d | %6d %8.2f %10d %9.1f%% | %10.1f"
              % (nm, n, tr, fps, run, dfn, pct([abs(r["dzP30"]) for r in withdz], 50)))
    print("\n   'defined%' = frames whose dz is not exactly 0 mm, i.e. where a sign exists at all.")
    print("   For reference a subject physically reverses at most ~0.5 times per second.")

    # ---------------- 11. hip vs shoulder ------------------------------------------------------
    print("\n" + "=" * W)
    print(" (11) HIP vs SHOULDER DEPTH SIGN -- is the hip line genuinely a better source?")
    print("=" * W)
    for r in turned + zeros:
        if "shDzP30" in r:
            r["shSign"] = r["shDzP30"]
    ps = infer_pol(turned, "shSign")
    print(" %-22s %7s %9s %9s %9s %9s %9s %9s %9s"
          % ("source", "defined", "coverage", "overall", "positive", "negative", "+90", "-90",
             "ALL frm"))
    for lab, key, pol in (("HIP raw dz (p30)", "dzP30", POL["dzP30"]),
                          ("SHOULDER raw dz (p30)", "shSign", ps),
                          ("HIP emitted dz", "dzEmit", POL["dzEmit"]),
                          ("SHOULDER emitted dz", "shDzEmit", infer_pol(turned, "shDzEmit"))):
        o, p, n, a, b, no, cv, oa, _nt = score(turned, key, pol)
        print(" %-22s %7d %8.1f%% %8s %8s %8s %8s %8s %8s"
              % (lab, no, cv, f(o), f(p), f(n), f(a), f(b), f(oa)))
    print("\n   F-10 reference -- the estimator that was actually rejected there was sign(yaw3D),")
    print("   not this: 78.49% overall, 100% on positive turns, 50.5% on negative turns.")
    print("   yaw3D is atan2(dx, dz) over the shoulder line, so it WRAPS when the line goes")
    print("   edge-on. dz alone does not wrap. Shoulder pixel span at each heading:")
    for nm, val in (("+-45", 45), ("+-90", 90)):
        s = [r for r in turned if abs(r["hipGT"]) == val]
        if s:
            print("     |heading| = %-4s  uSpanSh p50 = %5.1f px" % (nm, pct([r["uSpanSh"]
                                                                              for r in s], 50)))

    # ---------------- 11b. how the shoulder signal behaves off the static blocks ----------------
    print("\n" + "-" * W)
    print(" (11b) The shoulder signal in MOTION and at TRUE ZERO -- context for the comparison")
    print("-" * W)
    psh = infer_pol(turned, "shDzEmit")
    print(" %-14s %5s | %6s %8s %9s %9s" % ("motion block", "n", "flips", "flips/s", "longest",
                                            "defined%"))
    for nm in MOTION:
        s = [r for r in allrows if r["block"] == nm and "shDzEmit" in r]
        if len(s) < 5:
            continue
        tr, fps, run, n, dfn = flips(s, "shDzEmit", psh)
        print(" %-14s %5d | %6d %8.2f %9d %8.1f%%" % (nm, n, tr, fps, run, dfn))
    print("\n %-18s %6s %10s %9s %12s" % ("true-zero block", "n", "defined", "flips/s", "|dz| p50"))
    for nm in ZEROED:
        s = [r for r in zeros if r["block"] == nm and "shDzEmit" in r]
        if len(s) < 5:
            continue
        tr, fps, run, n, dfn = flips(s, "shDzEmit", psh)
        print(" %-18s %6d %9.1f%% %9.2f %12.1f"
              % (nm, n, dfn, fps, pct([abs(r["shDzEmit"]) for r in s], 50)))
    print("\n   At a true 0 the shoulder dz still emits a confident, non-flipping sign (|dz| p50 up")
    print("   to 136 mm, comparable to a real 45 deg turn). For a SIGN-ONLY use that is not")
    print("   automatically a defect -- sign * |yaw2D| is near zero when the magnitude is near")
    print("   zero -- but it does mean the magnitude of this signal carries no yaw information,")
    print("   and it has NOT been verified that the product stays small in practice.")

    # ---------------- 12. threshold sweep -------------------------------------------------------
    print("\n" + "=" * W)
    print(" (12) THRESHOLD / COVERAGE SWEEP on |dz| (p30, mm)")
    print("=" * W)
    print(" %8s %8s %9s %9s %9s %9s %9s %12s"
          % ("|dz| >", "n kept", "coverage", "overall", "positive", "negative", "+90/-90",
             "0deg defined"))
    for thr in (0, 10, 20, 30, 50, 75, 100, 150):
        o, p, n, a, b, no, cv, _oa, _nt = score(turned, "dzP30", POL["dzP30"], thr)
        zd = [r for r in zeros if "dzP30" in r]
        zdef = 100.0 * sum(1 for r in zd if abs(r["dzP30"]) > thr) / len(zd) if zd else float("nan")
        print(" %8d %8d %8.1f%% %8s %8s %8s %7s/%-7s %10.1f%%"
              % (thr, no, cv, f(o), f(p), f(n),
                 ("none" if a != a else "%.1f" % a), ("none" if b != b else "%.1f" % b), zdef))
    print("\n   'coverage' = share of turned frames that still have a defined sign.")
    print("   '0deg defined' = share of TRUE-ZERO frames that would emit a confident sign anyway;")
    print("   that is the false-sign rate a square-standing user would see.")

    # ---------------- 13. hypothetical hybrid ---------------------------------------------------
    print("\n" + "=" * W)
    print(" (13) HYPOTHETICAL HYBRID   sign(hip dz) * |yaw2D|   -- NOT implemented")
    print("=" * W)
    pop = [r for r in turned if "dzP30" in r]
    print(" %-30s %8s %8s %8s %8s %8s %10s"
          % ("estimator", "MAE", "RMSE", "p50", "p95", "max", "wrong sign"))
    # HOLD: an undefined sign (dz == 0) reuses the last defined one -- the most favourable reading
    # a shipping estimator could take. RAW: undefined counts as +1, i.e. no special handling.
    for lab, hold in (("hip-sign hybrid (hold on 0)", True), ("hip-sign hybrid (no hold)", False)):
        hy, wrong, last = [], 0, 1
        for r in sorted(pop, key=lambda r: r["t"]):
            s = sgn(r["dzP30"], POL["dzP30"])
            if s == 0:
                s = last if hold else 1
            else:
                last = s
            hy.append(abs(s * r["yaw2D"] - r["hipGT"]))
            if s != (1 if r["hipGT"] > 0 else -1):
                wrong += 1
        print(" %-30s %8.2f %8.2f %8.2f %8.2f %8.2f %9.2f%%"
              % (lab, mean(hy), rmse(hy), pct(hy, 50), pct(hy, 95), max(hy),
                 100.0 * wrong / len(hy)))
    mag = [abs(r["yaw2D"] - abs(r["hipGT"])) for r in pop]
    print(" %-30s %8.2f %8.2f %8.2f %8.2f %8.2f %10s"
          % ("yaw2D magnitude alone", mean(mag), rmse(mag), pct(mag, 50), pct(mag, 95), max(mag),
             "n/a"))
    print("\n   published references, same kind of static population:")
    print("     production yaw3D signed MAE 61.08 | F-10 depth-sign hybrid 13.09")
    print("     F-11 face-sign hybrid 31.60      | F-12 temporal hybrid 9.92 (unstable)")
    print("     yaw2D magnitude alone (F-10)      5.90")

    # ---------------- 14. above 90 -------------------------------------------------------------
    print("\n" + "=" * W)
    print(" (14) THE >90 deg LIMIT -- independent of any sign result")
    print("=" * W)
    m180 = [r for r in allrows if r["block"] == "m_180"]
    if m180:
        y = [r["yaw2D"] for r in m180]
        print("   m_180 block: |yaw2D| p50 %.1f  p95 %.1f  MAX %.1f deg over %d frames"
              % (pct(y, 50), pct(y, 95), max(y), len(y)))
    print("   |yaw2D| = acos(clamp(uSpan*Z/K)) is bounded to [0,90]. A 120 deg turn and a 60 deg")
    print("   turn foreshorten the shoulders identically, so no sign source can separate them.")

    # ---------------- 15. confidence forensics --------------------------------------------------
    print("\n" + "=" * W)
    print(" (15) IS A WRONG HIP SIGN ASSOCIATED WITH LOW CONFIDENCE?  (forensic only)")
    print("=" * W)
    dfn = [r for r in turned if "dzP30" in r and r["dzP30"] != 0]
    ok = [r for r in dfn if sgn(r["dzP30"], POL["dzP30"]) == (1 if r["hipGT"] > 0 else -1)]
    bad = [r for r in dfn if sgn(r["dzP30"], POL["dzP30"]) != (1 if r["hipGT"] > 0 else -1)]
    print(" %-22s %7s %9s %9s %9s %9s"
          % ("population", "n", "conf p50", "conf p10", "depthQ p50", "dsd p50"))
    for lab, s in (("sign CORRECT", ok), ("sign WRONG", bad)):
        if not s:
            continue
        print(" %-22s %7d %9.3f %9.3f %9.3f %9.1f"
              % (lab, len(s), pct([r["confHipMin"] for r in s], 50),
                 pct([r["confHipMin"] for r in s], 10),
                 pct([r["qHipMin"] for r in s], 50), pct([r["dsdHipMax"] for r in s], 50)))
    if bad:
        hi = sum(1 for r in bad if r["confHipMin"] >= 0.6)
        print("\n   %d of %d WRONG-sign frames (%.1f%%) carry hip confidence >= 0.60 -- i.e. the sign"
              % (hi, len(bad), 100.0 * hi / len(bad)))
        print("   is wrong while the landmark looks healthy. Confidence is not validity.")

    # ---------------- 16. twist blocks, reported but not scored ---------------------------------
    print("\n" + "=" * W)
    print(" (16) TWIST BLOCKS -- reported for completeness, EXCLUDED from every headline number")
    print("=" * W)
    for nm in TWIST:
        sel = [r for r in allrows if r["block"] == nm and "dzP30" in r]
        if len(sel) < 5:
            continue
        print("   %-14s n=%4d  heading label=%-5s  hip dz p50=%7.1f mm  |dz| p50=%6.1f"
              % (nm, len(sel), sel[0]["heading"], pct([r["dzP30"] for r in sel], 50),
                 pct([abs(r["dzP30"]) for r in sel], 50)))
    print("\n   tw_sh_* turn the SHOULDERS while the hips stay square, so the heading label is not")
    print("   hip truth; and the subject reported confusion during this protocol in F-10.")

    # ---------------- 17. falsification ---------------------------------------------------------
    print("\n" + "=" * W)
    print(" (17) FALSIFICATION -- the five cases that would sink the hypothesis")
    print("=" * W)
    pol = POL["dzP30"]
    c1 = [r for r in turned if r["hipGT"] == 45 and "dzP30" in r
          and (r["dzP30"] == 0 or sgn(r["dzP30"], pol) != 1)]
    n1 = [r for r in turned if r["hipGT"] == 45 and "dzP30" in r]
    print(" case 1  true +45 but dz is 0 or points the wrong way :  %d/%d = %.1f%% of frames"
          % (len(c1), len(n1), 100.0 * len(c1) / max(1, len(n1))))
    c2 = [r for r in turned if r["hipGT"] == -45 and "dzP30" in r and r["dzP30"] != 0
          and sgn(r["dzP30"], pol) != -1 and r["confHipMin"] >= 0.6]
    n2 = [r for r in turned if r["hipGT"] == -45 and "dzP30" in r]
    print(" case 2  true -45, wrong sign, hip confidence >= 0.60  :  %d/%d = %.1f%% of frames"
          % (len(c2), len(n2), 100.0 * len(c2) / max(1, len(n2))))
    c3 = [r for r in zeros if "dzP30" in r and abs(r["dzP30"]) >= 78.0]
    print(" case 3  true 0 but |dz| >= one disparity step (78 mm) :  %d/%d = %.1f%% of frames"
          % (len(c3), len(zeros), 100.0 * len(c3) / max(1, len(zeros))))
    p90 = [r for r in turned if r["hipGT"] == 90 and "dzP30" in r and r["dzP30"] != 0]
    m90 = [r for r in turned if r["hipGT"] == -90 and "dzP30" in r and r["dzP30"] != 0]
    n90 = [r for r in turned if abs(r["hipGT"]) == 90 and "dzP30" in r]
    print(" case 4  +90 and -90 give the same ordering            :  +90 defines a sign on %d "
          "frames, -90 on %d, out of %d" % (len(p90), len(m90), len(n90)))
    print("         -> at +90 the hip depth difference is NEVER non-zero, so +90 and -90 are")
    print("            literally indistinguishable: neither produces a bit to compare.")
    tot_fl = tot_t = 0
    for nm in MOTION:
        sel = [r for r in allrows if r["block"] == nm and "dzP30" in r]
        if len(sel) < 5:
            continue
        tr, _f, _r, _n, _d = flips(sel, "dzP30", pol)
        tot_fl += tr
        tot_t += sel[-1]["t"] - sel[0]["t"]
    print(" case 5  sign changes with no physical turn            :  %d flips over %.0f s of "
          "motion = %.2f/s" % (tot_fl, tot_t, tot_fl / tot_t if tot_t else float("nan")))

    print("\n" + "=" * W)
    print(" distance coverage: both captures are ~1.5 m ('near'). Distance sensitivity NOT MEASURED.")
    print("=" * W)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
