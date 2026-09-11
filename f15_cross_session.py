#!/usr/bin/env python3
"""F-15 -- does the torso-yaw sign hold up on a SECOND SUBJECT?

WHAT THIS TESTS
    F-14 concluded that raw shoulder dz is not an independent sign source, and that a WRAP-GUARDED
    yaw3D reaches the perfect-sign ceiling. It left one blocker. While preparing F-15 that blocker
    was re-characterised: controlling for heading, F-10 and F-11 agree EXACTLY on +45/-45/+90
    (100.00 % both) and ALL the variance is one -90 block (0.0 % vs 89.1 %). The "27-point session
    gap" was frame-weighting over different block mixes. So the open question is not session drift,
    it is THE -90 POSE -- and whether it depends on the subject's build.

REUSE, NOT REIMPLEMENTATION
    Every estimator, landmark index, depth field, yaw3D definition, normalisation, block trim and
    scoring rule is imported verbatim from f14_shoulder_depth_sign.py. This file adds a capture and
    arranges comparisons; it defines NO new estimator. Two documented deviations, both analysis-only:

      (1) f14.TURNED is EXTENDED with h_p30/h_m30/h_p60/h_m60. F-15 was captured with
          `--headings full`, which adds those static holds; F-10/F-11 used `coarse` and do not
          contain them, so extending the tuple cannot change any F-14 number. They exist to map
          WHERE the shoulder line collapses -- the actual open question.
      (2) Polarity is inferred SEPARATELY per capture, then compared. F-14 inferred it once on a
          pooled population. Inferring per capture is the only way to detect a subject-specific
          sign inversion (acceptance criterion 3); silently reusing a pooled polarity would hide it.

    The headline cross-session comparison is restricted to the headings all three captures share
    (0 / +-45 / +-90) and is MACRO-AVERAGED over headings, because frame-weighting over unequal
    block mixes is exactly what produced the spurious 27-point gap.

    python f15_cross_session.py
"""
import importlib.util
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "f14", os.path.join(HERE, "f14_shoulder_depth_sign.py"))
f14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f14)

pct, mean, sd, rmse, sgn, fmt = f14.pct, f14.mean, f14.sd, f14.rmse, f14.sgn, f14.fmt

CAPTURES = [
    ("F-10", "pipeline_logs_f10_near",
     os.path.join("oak_v4_evidence", "f10_gt_marks_near_RECOVERED.json"), "subject A"),
    ("F-11", "pipeline_logs_f11",
     os.path.join("oak_v4_evidence", "f10_gt_marks_near.json"), "subject A"),
    ("F-15", "pipeline_logs_f15",
     os.path.join("oak_v4_evidence", "f15", "f10_gt_marks_near.json"), "subject B"),
]

SHARED = (45, -45, 90, -90)          # headings all three captures contain
NEW = (30, -30, 60, -60)             # F-15 only, the collapse ramp
EST = [("raw p30 dz", "dzP30"), ("raw p50 dz", "dzP50"), ("centre dz", "dzCtr"),
       ("emitted dz", "dzEmit"), ("sign(yaw3D)", "yaw3Dsig")]


def guarded(r, pol):
    """The F-10 wrap guard, reproduced exactly: refuse the sign when |yaw3D| > 150."""
    return 0 if abs(r["yaw3D"]) > 150 else sgn(r["yaw3D"], pol)


def load_all():
    out = []
    for cap, d, m, who in CAPTURES:
        if not os.path.isdir(d) or not os.path.exists(m):
            print("  [skip] %s -- missing %s or %s" % (cap, d, m))
            continue
        raw = f14.load(d)
        marks = json.load(io.open(m, encoding="utf-8"))
        rows = f14.label(raw, marks, cap)
        K = f14.add_derived(rows)
        for r in rows:
            r["who"] = who
        out.append((cap, who, d, m, len(raw), rows, K, bool(marks.get("RECOVERED")),
                    marks.get("metres")))
    return out


def acc(rows, key, pol, heading=None):
    """(coverage, accuracy|defined, accuracy over ALL frames, n) -- f14 conventions."""
    sel = [r for r in rows if r.get(key) is not None and r["shGT"] not in (None, 0)]
    if heading is not None:
        sel = [r for r in sel if r["shGT"] == heading]
    if not sel:
        return float("nan"), float("nan"), float("nan"), 0
    if key == "yaw3Dsig":
        d = [r for r in sel if sgn(r[key], pol) != 0]
    else:
        d = [r for r in sel if r[key] != 0]
    ok = sum(1 for r in d if sgn(r[key], pol) == (1 if r["shGT"] > 0 else -1))
    return (100.0 * len(d) / len(sel),
            100.0 * ok / len(d) if d else float("nan"),
            100.0 * ok / len(sel), len(sel))


def macro(rows, key, pol, headings=SHARED):
    """Mean of the per-heading ALL-frame accuracies. Immune to block-mix composition."""
    v = [acc(rows, key, pol, h)[2] for h in headings
         if acc(rows, key, pol, h)[3] > 0]
    return mean([x for x in v if x == x]) if v else float("nan")


def main():
    os.chdir(HERE)
    # DEVIATION (1) -- documented in the docstring.
    f14.TURNED = f14.TURNED + ("h_p30", "h_m30", "h_p60", "h_m60")
    data = load_all()
    if len(data) < 3:
        print("\n  F-15 capture not found -- cannot run the cross-session comparison.")
        return 1
    W = 118
    print("=" * W)
    print(" F-15 -- CROSS-SESSION / SECOND-SUBJECT VALIDATION   (offline; no production code touched)")
    print("=" * W)
    for cap, who, d, _m, nraw, rows, K, rec, met in data:
        z = [r["hipZ"] for r in rows]
        print("  %-5s %-10s %-22s %6d->%4d frames | %.2f m (p10 %.2f p90 %.2f) | K=%.1f%s"
              % (cap, who, d, nraw, len(rows), pct(z, 50), pct(z, 10), pct(z, 90), K,
                 "  [marks RECOVERED]" if rec else ""))

    # ---- polarity, inferred per capture -- DEVIATION (2), and criterion 3 --------------------
    print("\n" + "=" * W)
    print(" (A) SIGN-CONVENTION CHECK -- is there a subject-specific inversion?")
    print("=" * W)
    print(" %-6s %-10s %s" % ("cap", "subject", "  ".join("%-13s" % k for _l, k in EST)))
    POLS = {}
    for cap, who, _d, _m, _n, rows, _K, _r, _me in data:
        main_t = [r for r in rows if r["grp"] == "main"]
        POLS[cap] = dict((k, f14.infer_pol(main_t, k)) for _l, k in EST)
        print(" %-6s %-10s %s" % (cap, who,
                                  "  ".join("%-13s" % ("%+d" % POLS[cap][k]) for _l, k in EST)))
    inverted = [k for _l, k in EST
                if len(set(POLS[c][k] for c in POLS)) > 1]
    print("\n   %s" % ("*** SIGN INVERSION between captures on: %s ***" % ", ".join(inverted)
                       if inverted else
                       "No inversion: every estimator infers the SAME polarity on all three "
                       "captures. Criterion 3 passes."))
    POL = POLS["F-10"]

    # ---- per-heading, per-capture -------------------------------------------------------------
    print("\n" + "=" * W)
    print(" (B) PER-HEADING SIGN ACCURACY (ALL frames; undefined counted wrong)")
    print("=" * W)
    for lab, key in EST:
        print("\n  %s" % lab)
        print("   %-6s %s" % ("cap", "".join("%12s" % ("%+d" % h) for h in SHARED) +
                              "%14s" % "macro avg"))
        for cap, _who, _d, _m, _n, rows, _K, _r, _me in data:
            mt = [r for r in rows if r["grp"] == "main"]
            cells = []
            for h in SHARED:
                cv, ad, al, n = acc(mt, key, POL[key], h)
                cells.append("        --  " if n == 0 else "%11.2f%%" % al)
            print("   %-6s %s %13.2f%%" % (cap, "".join(cells), macro(mt, key, POL[key])))

    # ---- the headline comparison ---------------------------------------------------------------
    print("\n" + "=" * W)
    print(" (C) F-10 vs F-11 vs F-15 -- macro-averaged over the shared headings")
    print("=" * W)
    print(" %-24s %10s %10s %10s %12s"
          % ("estimator", "F-10", "F-11", "F-15", "F-15 vs best"))
    for lab, key in EST:
        vals = {}
        for cap, _w, _d, _m, _n, rows, _K, _r, _me in data:
            vals[cap] = macro([r for r in rows if r["grp"] == "main"], key, POL[key])
        best = max(vals["F-10"], vals["F-11"])
        print(" %-24s %9.2f%% %9.2f%% %9.2f%% %+11.2f"
              % (lab, vals["F-10"], vals["F-11"], vals["F-15"], vals["F-15"] - best))
    # wrap-guarded yaw3D + the ceiling
    print("\n %-24s %10s %10s %10s %12s"
          % ("hybrid MAE (deg)", "F-10", "F-11", "F-15", "F-15 vs best"))
    for lab, fn in (("yaw3D raw, no guard",
                     lambda r: sgn(r["yaw3D"], POL["yaw3Dsig"])),
                    ("yaw3D + WRAP GUARD",
                     lambda r: guarded(r, POL["yaw3Dsig"])),
                    ("raw shoulder dz", lambda r: sgn(r["dzP30"], POL["dzP30"])),
                    ("emitted shoulder dz", lambda r: sgn(r["dzEmit"], POL["dzEmit"])),
                    ("perfect-sign ceiling", None)):
        vals = {}
        for cap, _w, _d, _m, _n, rows, _K, _r, _me in data:
            pop = sorted([r for r in rows if r["grp"] == "main"
                          and r["shGT"] in SHARED and "dzP30" in r], key=lambda x: x["t"])
            if fn is None:
                vals[cap] = mean([abs(r["yaw2D"] - abs(r["shGT"])) for r in pop])
                continue
            e, last = [], 1
            for r in pop:
                s = fn(r)
                if s == 0:
                    s = last
                else:
                    last = s
                e.append(abs(s * r["yaw2D"] - r["shGT"]))
            vals[cap] = mean(e)
        best = min(vals["F-10"], vals["F-11"])
        print(" %-24s %10.2f %10.2f %10.2f %+11.2f"
              % (lab, vals["F-10"], vals["F-11"], vals["F-15"], vals["F-15"] - best))
    print("\n   'F-15 vs best' compares against the BETTER of the two prior captures, so a")
    print("   collapse cannot hide behind F-10's weak -90 block. >15 points (or a large MAE")
    print("   rise) is the brief's 'major collapse' trigger.")

    # ---- block level ----------------------------------------------------------------------------
    print("\n" + "=" * W)
    print(" (D) BLOCK-LEVEL EVIDENCE for F-15 -- the primary unit")
    print("=" * W)
    f15 = [d for d in data if d[0] == "F-15"][0][5]
    print(" %-18s %-6s %6s | %-26s | %-26s"
          % ("block", "true", "n", "raw p30 dz", "yaw3D + wrap guard"))
    tally = {"dzP30": [0, 0, 0], "guard": [0, 0, 0]}
    for b in sorted(set(r["block"] for r in f15 if r["grp"] == "main"),
                    key=lambda x: (0 if x.startswith("h_") else 1, x)):
        s = [r for r in f15 if r["block"] == b and r["grp"] == "main"]
        if not s:
            continue
        t = s[0]["shGT"]
        cells = []
        for kind in ("dzP30", "guard"):
            if kind == "dzP30":
                dd = [sgn(r["dzP30"], POL["dzP30"]) for r in s
                      if "dzP30" in r and r["dzP30"] != 0]
            else:
                dd = [x for x in (guarded(r, POL["yaw3Dsig"]) for r in s) if x != 0]
            if not dd:
                tally[kind][2] += 1
                cells.append("%-26s" % "NO SIGNAL")
                continue
            maj = 1 if sum(dd) > 0 else -1
            good = maj == (1 if t > 0 else -1)
            tally[kind][0 if good else 1] += 1
            cells.append("%-8s %4d/%4d def %3d%% agree"
                         % ("CORRECT" if good else "WRONG", len(dd), len(s),
                            round(100.0 * sum(1 for x in dd if x == maj) / len(dd))))
        print(" %-18s %-6s %6d | %s | %s" % (b, t, len(s), cells[0], cells[1]))
    print("\n %-26s %9s %8s %11s" % ("", "correct", "wrong", "no signal"))
    for lab, k in (("raw p30 dz", "dzP30"), ("yaw3D + wrap guard", "guard")):
        c, w, n = tally[k]
        print(" %-26s %9d %8d %11d   (of %d blocks)" % (lab, c, w, n, c + w + n))

    # ---- the +-90 ramp ---------------------------------------------------------------------------
    print("\n" + "=" * W)
    print(" (E) WHERE DOES THE SHOULDER LINE COLLAPSE?  F-15's new +-30/+-60 holds")
    print("=" * W)
    print(" %-8s %6s %10s %11s %11s %11s %11s"
          % ("heading", "n", "span px", "raw cov", "raw acc", "guard acc", "|yaw2D| p50"))
    for h in (0, 30, -30, 45, -45, 60, -60, 90, -90):
        s = [r for r in f15 if r["shGT"] == h and r["grp"] in ("main", "zero")]
        if len(s) < 5:
            continue
        cv, ad, al, n = acc(s, "dzP30", POL["dzP30"], h) if h != 0 else (
            float("nan"),) * 3 + (0,)
        if h == 0:
            d0 = [r for r in s if "dzP30" in r and r["dzP30"] != 0]
            cv = 100.0 * len(d0) / len(s)
        gd = [x for x in (guarded(r, POL["yaw3Dsig"]) for r in s) if x != 0]
        gok = (100.0 * sum(1 for r in s if guarded(r, POL["yaw3Dsig"])
                           == (1 if h > 0 else -1)) / len(s)) if h != 0 else float("nan")
        print(" %-8s %6d %10.1f %10.1f%% %11s %11s %11.1f"
              % ("%+d" % h, len(s), pct([r["uSpanSh"] for r in s], 50), cv,
                 fmt(al) if h != 0 else "   --   ", fmt(gok) if h != 0 else "   --   ",
                 pct([r["yaw2D"] for r in s], 50)))
    print("\n   'span px' is the shoulder separation in pixels. F-14 measured the collapse at")
    print("   +-90 (7.0 px) but had no intermediate holds; these rows locate the knee of the curve.")

    # ---- wrap-guard validation --------------------------------------------------------------------
    print("\n" + "=" * W)
    print(" (F) WRAP-GUARD VALIDATION on F-15")
    print("=" * W)
    mt = [r for r in f15 if r["grp"] == "main"]
    wr = [r for r in mt if abs(r["yaw3D"]) > 150]
    print("   frames triggering the guard (|yaw3D| > 150): %d of %d = %.1f%%"
          % (len(wr), len(mt), 100.0 * len(wr) / len(mt)))
    if wr:
        by = {}
        for r in wr:
            by[r["shGT"]] = by.get(r["shGT"], 0) + 1
        print("     by heading: %s" % ", ".join("%+d: %d" % kv for kv in sorted(by.items())))
        wrong_unguarded = sum(1 for r in wr
                              if sgn(r["yaw3D"], POL["yaw3Dsig"])
                              != (1 if r["shGT"] > 0 else -1))
        print("   of those, UNGUARDED yaw3D would be WRONG on %d (%.1f%%) -- what the guard removes"
              % (wrong_unguarded, 100.0 * wrong_unguarded / len(wr)))
        also = sum(1 for r in wr if "dzP30" in r and r["dzP30"] == 0)
        print("   raw dz is ALSO undefined on %d of them (%.1f%%) -- replicating F-14's finding"
              % (also, 100.0 * also / len(wr)))
    for lab, fn in (("no guard", lambda r: sgn(r["yaw3D"], POL["yaw3Dsig"])),
                    ("with guard", lambda r: guarded(r, POL["yaw3Dsig"]))):
        pop = sorted([r for r in mt if r["shGT"] in SHARED], key=lambda x: x["t"])
        e, w, last, held, run, mx = [], 0, 1, 0, 0, 0
        for r in pop:
            s = fn(r)
            if s == 0:
                s = last
                held += 1
                run += 1
                mx = max(mx, run)
            else:
                last = s
                run = 0
            e.append(abs(s * r["yaw2D"] - r["shGT"]))
            if s != (1 if r["shGT"] > 0 else -1):
                w += 1
        print("   %-11s MAE %6.2f  wrong sign %5.2f%%  frames held %5.1f%%  longest hold %d frames"
              % (lab, mean(e), 100.0 * w / len(e), 100.0 * held / len(e), mx))

    # ---- raw-depth independence -------------------------------------------------------------------
    print("\n" + "=" * W)
    print(" (G) RAW-DEPTH INDEPENDENCE -- does raw dz rescue frames yaw3D gets wrong? (F-14 replication)")
    print("=" * W)
    for lab, pop in (("ALL shared headings", [r for r in mt if r["shGT"] in SHARED]),
                     ("|heading| = 90 only", [r for r in mt if abs(r["shGT"]) == 90])):
        cells = [[0, 0], [0, 0]]
        nosig = 0
        for r in pop:
            if "dzP30" not in r:
                continue
            t = 1 if r["shGT"] > 0 else -1
            if r["dzP30"] == 0:
                nosig += 1
                continue
            a = sgn(r["dzP30"], POL["dzP30"]) == t
            b = sgn(r["yaw3D"], POL["yaw3Dsig"]) == t
            cells[0 if a else 1][0 if b else 1] += 1
        print("   %s  (n=%d defined, %d no-signal)"
              % (lab, sum(sum(x) for x in cells), nosig))
        print("     %-18s %14s %14s" % ("", "yaw3D correct", "yaw3D wrong"))
        print("     %-18s %14d %14d" % ("raw dz correct", cells[0][0], cells[0][1]))
        print("     %-18s %14d %14d" % ("raw dz wrong", cells[1][0], cells[1][1]))
        print("     TRUE RESCUES (dz right, yaw3D wrong): %d      lost: %d\n"
              % (cells[0][1], cells[1][0]))

    # ---- label audit + whole-body-only result ---------------------------------------------------
    print("=" * W)
    print(" (J) LABEL AUDIT -- did subject B actually turn the way the labels say?")
    print("=" * W)
    print(" The nose offset from the shoulder midpoint is an INDEPENDENT read on turn direction:")
    print(" it uses face keypoints, not depth, not yaw3D. For a WHOLE-BODY turn the head goes with")
    print(" the body, so its sign must mirror. It is INVALID for the dc_* blocks, which deliberately")
    print(" hold the face on the camera while the body turns -- those are audited separately or not")
    print(" at all. Convention is read off the POSITIVE blocks, not assumed.\n")
    A = {}
    for line in io.open(os.path.join("pipeline_logs_f15", "audit_log.jsonl"), encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                r = json.loads(line)
                if "j" in r:
                    A[r["seq"]] = r
            except ValueError:
                pass

    def nose_norm(sel):
        o = []
        for r in sel:
            a = A.get(r["seq"])
            if not a:
                continue
            fj, jj = a.get("f") or {}, a.get("j") or {}
            n, ls, rs = fj.get("nose"), jj.get("L-shoulder"), jj.get("R-shoulder")
            if n and ls and rs and n["c"] >= 0.3:
                mid, span = 0.5 * (ls["u"] + rs["u"]), abs(ls["u"] - rs["u"])
                o.append((n["u"] - mid) / max(span, 1e-6))
        return pct(o, 50) if o else float("nan")
    print(" %-20s %-6s %6s %10s %11s %16s"
          % ("block", "label", "n", "|yaw2D|", "nose norm", "label check"))
    bad = []
    for b in sorted(set(r["block"] for r in f15 if r["grp"] == "main")):
        s = [r for r in f15 if r["block"] == b]
        if len(s) < 5:
            continue
        h, nn = s[0]["shGT"], nose_norm(s)
        if b.startswith("dc_"):
            chk = "n/a (decoupled)"
        else:
            chk = "ok" if (1 if nn < 0 else -1) == (1 if h > 0 else -1) else "*** MISLABELLED ***"
            if chk != "ok":
                bad.append(b)
        print(" %-20s %-6s %6d %10.1f %11.3f %16s"
              % (b, h, len(s), pct([r["yaw2D"] for r in s], 50), nn, chk))
    print("\n   %s" % ("Every whole-body block is correctly labelled -- the protocol was followed."
                       if not bad else "MISLABELLED whole-body blocks: %s" % ", ".join(bad)))

    print("\n" + "-" * W)
    print(" WHOLE-BODY (h_*) BLOCKS ONLY -- the label-validated population")
    print("-" * W)
    hb = [r for r in f15 if r["grp"] == "main" and r["block"].startswith("h_")]
    print(" %-26s %9s %8s %11s" % ("", "correct", "wrong", "no signal"))
    for lab, kind in (("raw p30 dz", "dzP30"), ("yaw3D + wrap guard", "guard")):
        c = w = n = 0
        for b in sorted(set(r["block"] for r in hb)):
            s = [r for r in hb if r["block"] == b]
            t = s[0]["shGT"]
            if kind == "dzP30":
                dd = [sgn(r["dzP30"], POL["dzP30"]) for r in s
                      if "dzP30" in r and r["dzP30"] != 0]
            else:
                dd = [x for x in (guarded(r, POL["yaw3Dsig"]) for r in s) if x != 0]
            if not dd:
                n += 1
                continue
            maj = 1 if sum(dd) > 0 else -1
            if maj == (1 if t > 0 else -1):
                c += 1
            else:
                w += 1
        print(" %-26s %9d %8d %11d   (of %d whole-body blocks)" % (lab, c, w, n, c + w + n))
    for band, hs in (("|heading| <= 60", (30, -30, 45, -45, 60, -60)),
                     ("|heading| == 90", (90, -90))):
        c = w = n = 0
        for h in hs:
            s = [r for r in hb if r["shGT"] == h]
            if not s:
                continue
            dd = [x for x in (guarded(r, POL["yaw3Dsig"]) for r in s) if x != 0]
            if not dd:
                n += 1
                continue
            maj = 1 if sum(dd) > 0 else -1
            if maj == (1 if h > 0 else -1):
                c += 1
            else:
                w += 1
        print("   guarded yaw3D, %-16s : %d correct, %d wrong, %d no-signal" % (band, c, w, n))

    # ---- dynamic ------------------------------------------------------------------------------------
    print("=" * W)
    print(" (H) DYNAMIC BEHAVIOUR on F-15 -- behaviour only, no instantaneous truth")
    print("=" * W)
    print(" %-14s %5s | %8s %9s %10s %10s"
          % ("block", "n", "raw cov", "raw f/s", "guard f/s", "longest"))
    for nm in f14.MOTION:
        s = [r for r in f15 if r["block"] == nm]
        sr = [r for r in s if "dzP30" in r]
        if len(sr) < 5:
            continue
        _t1, f1, r1, _n1, d1 = f14.flips(sr, "dzP30", POL["dzP30"])
        gs = sorted(s, key=lambda r: r["t"])
        gsig = [guarded(r, POL["yaw3Dsig"]) for r in gs]
        dd = [x for x in gsig if x != 0]
        tr = sum(1 for i in range(1, len(dd)) if dd[i] != dd[i - 1])
        dur = gs[-1]["t"] - gs[0]["t"]
        print(" %-14s %5d | %7.1f%% %9.2f %10.2f %10d"
              % (nm, len(sr), d1, f1, tr / dur if dur > 0.5 else float("nan"), r1))
    print("\n   Reference only: a subject physically reverses at most ~0.5 times/second (F-12).")

    # ---- true zero ------------------------------------------------------------------------------------
    print("\n" + "=" * W)
    print(" (I) TRUE-ZERO on F-15 -- does an arbitrary sign move the avatar?")
    print("=" * W)
    z = [r for r in f15 if r["grp"] == "zero"]
    print(" %-28s %8s %8s %8s %8s" % ("estimator", "MAE", "RMSE", "p95", "max"))
    for lab, fn in (("sign(raw dz) * |yaw2D|", lambda r: sgn(r["dzP30"], POL["dzP30"])),
                    ("sign(guarded yaw3D) * |yaw2D|", lambda r: guarded(r, POL["yaw3Dsig"])),
                    ("production yaw3D", None)):
        if fn is None:
            e = [abs(r["yaw3D"]) for r in z]
        else:
            e = [abs(fn(r) * r["yaw2D"]) for r in z if "dzP30" in r]
        if e:
            print(" %-28s %8.2f %8.2f %8.2f %8.2f"
                  % (lab, mean(e), rmse(e), pct(e, 95), max(e)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
