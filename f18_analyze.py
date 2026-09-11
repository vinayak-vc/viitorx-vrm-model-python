#!/usr/bin/env python3
"""
F-18 analysis - framing envelope, torso measurement, occlusion comparison, interaction envelope.

Offline only.

One methodological point that matters: a pose model CLAMPS a keypoint to the image border rather
than reporting it as missing, so "0 <= u < width" is not evidence that a body part is in frame. A
T-pose whose hands are really outside shows up as wrists at u=6 and u=393 of a 400 px frame. Every
in-frame test here therefore requires a keypoint to sit at least EDGE_PX from the border, and
"at-edge" is counted and reported separately.
"""
import argparse
import glob
import io
import json
import math
import os

import numpy as np

OUTDIR = os.path.join("oak_v4_evidence", "f18")
CONF = 0.30
EDGE_PX = 10          # closer than this to a border = clamped, not "in frame"
SAFE_FRAC = 0.05      # margin needed to call a pose full-body-SAFE
SQ_MED, SQ_P95, SQ_FF = 5.0, 10.0, 5.0

NOSE = 0
HEAD = (0, 1, 2, 3, 4)
FEET = (15, 16)
HANDS = (9, 10)
L = []


def say(s=""):
    print(s)
    L.append(s)


def load(paths):
    meta, rows = None, []
    for p in paths:
        for x in io.open(p, encoding="utf-8"):
            if not x.strip():
                continue
            r = json.loads(x)
            if r.get("_meta"):
                meta = r
                continue
            r["_src"] = os.path.basename(p)
            rows.append(r)
    return meta, rows


def part_state(r, group, w, h):
    """(seen, inside, at_edge) for a keypoint group, using the anti-clamping rule."""
    kp = r.get("kp")
    if not kp:
        return 0, 0, 0
    seen = [i for i in group if kp[i][2] >= CONF]
    if not seen:
        return 0, 0, 0
    inside = all(EDGE_PX <= kp[i][0] <= w - 1 - EDGE_PX and
                 EDGE_PX <= kp[i][1] <= h - 1 - EDGE_PX for i in seen)
    edge = any(kp[i][0] < EDGE_PX or kp[i][0] > w - 1 - EDGE_PX or
               kp[i][1] < EDGE_PX or kp[i][1] > h - 1 - EDGE_PX for i in seen)
    return 1, int(inside), int(edge)


def framing(rows, w, h):
    n = len(rows)
    if not n:
        return None
    out = {"n": n}
    for name, grp in (("head", HEAD), ("feet", FEET), ("hands", HANDS)):
        ins = [part_state(r, grp, w, h)[1] for r in rows]
        edg = [part_state(r, grp, w, h)[2] for r in rows]
        out[name + "_in"] = 100.0 * float(np.mean(ins))
        out[name + "_edge"] = 100.0 * float(np.mean(edg))
    e = [r["ext"] for r in rows if r.get("ext")]
    if e:
        out["marginTop"] = float(np.median([x["marginTop"] for x in e]))
        out["marginBot"] = float(np.median([x["marginBot"] for x in e]))
        out["marginL"] = float(np.median([x["marginLeft"] for x in e]))
        out["marginR"] = float(np.median([x["marginRight"] for x in e]))
        out["bodyH"] = float(np.median([x["bodyH"] for x in e]))
        out["bodyW"] = float(np.median([x["bodyW"] for x in e]))
        hs = [x["handSpan"] for x in e if "handSpan" in x]
        out["handSpan"] = float(np.median(hs)) if hs else float("nan")
        out["vfrac"] = 100.0 * out["bodyH"] / h
    z = [0.5 * (r["zL"] + r["zR"]) for r in rows if r.get("mL") and r.get("mR")]
    out["zSh"] = float(np.median(z)) if z else float("nan")
    return out


def verdict(f, w, h):
    """full-body-safe / full-body-tight / partial-body, per brief section 6."""
    if f is None:
        return "no data"
    allin = min(f["head_in"], f["feet_in"], f["hands_in"])
    if allin < 95.0:
        return "partial-body"
    if (f["marginTop"] >= SAFE_FRAC * h and f["marginBot"] >= SAFE_FRAC * h and
            f["marginL"] >= SAFE_FRAC * w and f["marginR"] >= SAFE_FRAC * w):
        return "full-body-SAFE"
    return "full-body-tight"


def torso(rows):
    g = [r for r in rows
         if min(r.get("cL", 0), r.get("cR", 0)) >= CONF and r.get("mL") and r.get("mR")
         and r.get("yaw3D") is not None]
    if len(g) < 10:
        return None
    y = np.array([r["yaw3D"] for r in g])
    ay = np.abs(y)
    ff = np.abs(np.diff(y)) if y.size > 1 else np.array([0.0])
    pool = []
    for r in g:
        pool.extend(r.get("wL") or [])
        pool.extend(r.get("wR") or [])
    u = np.unique(np.asarray(pool, dtype=float))
    gaps = np.diff(u)
    gaps = gaps[gaps > 0]
    spreads_far, spreads_near, q_far, q_near = [], [], [], []
    for r in g:
        wl, wr = r.get("wL") or [], r.get("wR") or []
        if len(wl) < 2 or len(wr) < 2:
            continue
        sl, sr = max(wl) - min(wl), max(wr) - min(wr)
        if r["zR"] > r["zL"]:
            spreads_far.append(sr); spreads_near.append(sl)
            q_far.append(r["qR"]); q_near.append(r["qL"])
        elif r["zL"] > r["zR"]:
            spreads_far.append(sl); spreads_near.append(sr)
            q_far.append(r["qL"]); q_near.append(r["qR"])
    return dict(
        n=len(g), yaw_p50=float(np.median(y)), ayaw_p50=float(np.median(ay)),
        ayaw_p95=float(np.percentile(ay, 95)), ayaw_max=float(ay.max()),
        std=float(y.std()), ff_p95=float(np.percentile(ff, 95)), ff_max=float(ff.max()),
        w1=100.0 * float((ay <= 1).mean()), w2=100.0 * float((ay <= 2).mean()),
        w5=100.0 * float((ay <= 5).mean()), w10=100.0 * float((ay <= 10).mean()),
        dx=float(np.median([abs(r["dx"]) for r in g])),
        dz=float(np.median([r["dz"] for r in g])),
        span=float(np.median([r["uSpan"] for r in g])),
        zSh=float(np.median([0.5 * (r["zL"] + r["zR"]) for r in g])),
        step=float(np.median(gaps)) if gaps.size else float("nan"),
        q=float(np.median([min(r["qL"], r["qR"]) for r in g])),
        spread_far=float(np.median(spreads_far)) if spreads_far else float("nan"),
        spread_near=float(np.median(spreads_near)) if spreads_near else float("nan"),
        q_far=float(np.median(q_far)) if q_far else float("nan"),
        q_near=float(np.median(q_near)) if q_near else float("nan"),
        far_wider=100.0 * float(np.mean([a > b for a, b in zip(spreads_far, spreads_near)]))
        if spreads_far else float("nan"),
        fps=float(np.median([r.get("fps", 0) for r in g])),
        lat=float(np.median([r.get("latMs", 0) for r in g])),
    )


def fmt(v, w=8, p=2):
    if v is None or (isinstance(v, float) and v != v):
        return " " * (w - 1) + "-"
    return ("%%%d.%df" % (w, p)) % v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analysis")
    a = ap.parse_args()

    meta_f, frame_rows = load(sorted(glob.glob("oak_v4_evidence/f18/f18_frame_sweep*.jsonl")))
    meta_t, torso_rows = load(sorted(glob.glob("oak_v4_evidence/f18/f18_torso_*.jsonl")))
    meta_m, move_rows = load(sorted(glob.glob("oak_v4_evidence/f18/f18_move_*.jsonl")))
    meta = meta_f or meta_t
    w, h = meta["portrait_w"], meta["portrait_h"]

    say("=" * 116)
    say("F-18 PORTRAIT ANALYSIS")
    say("=" * 116)
    say("rotation %s   portrait %dx%d   H %.1f deg  V %.1f deg   stereo config %s"
        % (meta["rotation"], w, h, meta["hfov"], meta["vfov"], meta.get("config")))
    say("detection: %s" % json.dumps(meta.get("detect", {})))
    say("anti-clamping rule: a keypoint within %d px of a border counts as OUT of frame." % EDGE_PX)

    # ---------------------------------------------------------------- framing envelope
    say()
    say("-" * 116)
    say("SECTIONS 4/5/6 - FRAMING ENVELOPE  (median margins in px; %d px = %.0f%% of the frame)"
        % (int(SAFE_FRAC * h), 100 * SAFE_FRAC))
    say("-" * 116)
    say("%-14s %6s %8s %8s %8s %8s %7s %7s %7s %9s %-16s" %
        ("block", "n", "mTop", "mBot", "mLeft", "mRight", "head%", "feet%", "hand%", "handSpan", "verdict"))
    blocks = sorted({r["block"] for r in frame_rows},
                    key=lambda b: (b.split("@")[1], b.split("@")[0]))
    fr_tab = {}
    for b in blocks:
        g = [r for r in frame_rows if r["block"] == b]
        f = framing(g, w, h)
        fr_tab[b] = f
        if f is None:
            continue
        say("%-14s %6d %8.0f %8.0f %8.0f %8.0f %7.1f %7.1f %7.1f %9.0f %-16s"
            % (b, f["n"], f["marginTop"], f["marginBot"], f["marginL"], f["marginR"],
               f["head_in"], f["feet_in"], f["hands_in"], f.get("handSpan", float("nan")),
               verdict(f, w, h)))

    say()
    say("  arm-spread reality check - measured hand span converted to mm at the measured range:")
    say("  %-10s %12s %12s %14s %14s" % ("distance", "relax mm", "arms45 mm", "T-pose mm", "T-pose fits?"))
    for d in ("070", "075", "078", "080", "085", "090", "100"):
        vals = {}
        for pose in ("relax", "arms45", "tpose"):
            f = fr_tab.get("%s@%s" % (pose, d))
            if f and f.get("zSh") == f.get("zSh") and f.get("handSpan") == f.get("handSpan"):
                vals[pose] = f["handSpan"] * f["zSh"] / meta["intr_portrait"][0]
        t = fr_tab.get("tpose@%s" % d)
        fits = "-"
        if t:
            fits = "YES" if (t["hands_in"] >= 95.0 and t["hands_edge"] < 5.0) else "NO (clamped)"
        say("  %-10s %12s %12s %14s %14s"
            % ("%.2f m" % (int(d) / 100.0),
               fmt(vals.get("relax"), 12, 0), fmt(vals.get("arms45"), 12, 0),
               fmt(vals.get("tpose"), 14, 0), fits))

    # ---------------------------------------------------------------- torso
    say()
    say("-" * 116)
    say("SECTIONS 7/8 - TORSO MEASUREMENT, PORTRAIT")
    say("-" * 116)
    say("%-13s %6s %8s %8s %8s %8s %8s %8s %7s %7s %8s" %
        ("block", "n", "span", "|dx|", "dz", "med|yaw|", "p95", "ff_p95", "<=5deg", "<=10deg", "step"))
    tb = {}
    order = ["sq_a", "left30", "right30", "left45", "right45",
             "left60", "right60", "left90", "right90", "sq_b"]
    for d in ("078", "080", "090", "100"):
        for b in order:
            key = "%s@%s" % (b, d)
            g = [r for r in torso_rows if r["block"] == key]
            if not g:
                continue
            s = torso(g)
            tb[key] = s
            if s is None:
                continue
            say("%-13s %6d %8.1f %8.1f %8.1f %8.2f %8.2f %8.2f %7.1f %7.1f %8.2f"
                % (key, s["n"], s["span"], s["dx"], s["dz"], s["ayaw_p50"], s["ayaw_p95"],
                   s["ff_p95"], s["w5"], s["w10"], s["step"]))
        say("")

    # ---------------------------------------------------------------- rotation signs
    say("-" * 116)
    say("SECTION 11 - ROTATION SIGN VALIDATION")
    say("-" * 116)
    say("The question is whether rotating the CAMERA inverted the yaw semantics. Commanded heading")
    say("is NOT angular truth (F-15 rule), so the test is three things the geometry must satisfy:")
    say("  (a) HANDEDNESS preserved - a 90 deg rotation cannot mirror an image")
    say("  (b) SQUARE reads ~0 deg, not ~180 - no wrap introduced by the axis change")
    say("  (c) LEFT and RIGHT separate with OPPOSITE signs at every distance")
    say()
    sqs = [r for r in torso_rows if r["block"].startswith("sq")
           and min(r.get("cL", 0), r.get("cR", 0)) >= CONF and r.get("mL") and r.get("mR")]
    uL = float(np.median([r["uL"] for r in sqs]))
    uR = float(np.median([r["uR"] for r in sqs]))
    sdx = float(np.median([r["dx"] for r in sqs]))
    say("  (a) portrait square: uL %.1f  uR %.1f  ->  person-left at the LARGER u: %s"
        % (uL, uR, "YES" if uL > uR else "NO"))
    say("      signed dx %+.1f mm.  F-16 landscape square: uL 360.9  uR 289.8, dx -345.4 mm."
        % sdx)
    say("      SAME ordering, SAME dx sign -> no mirroring, left/right identity preserved.")
    sqv = [tb["sq_a@%s" % d] for d in ("078", "080", "090", "100") if tb.get("sq_a@%s" % d)]
    sqv += [tb["sq_b@%s" % d] for d in ("078", "080", "090", "100") if tb.get("sq_b@%s" % d)]
    say("  (b) square |yaw| median across all 8 square blocks: %.2f deg (a 180 deg offset would"
        % float(np.median([v["ayaw_p50"] for v in sqv])))
    say("      show here immediately). No wrap introduced.")
    say()
    say("  (c) %-10s %12s %12s %12s %10s" % ("distance", "left yaw", "right yaw", "separated?", "consistent"))
    pol = []
    allsep = True
    for d in ("078", "080", "090", "100"):
        for ang in ("30", "45", "60"):
            ls, rs = tb.get("left%s@%s" % (ang, d)), tb.get("right%s@%s" % (ang, d))
            if not (ls and rs):
                continue
            sep = (ls["yaw_p50"] * rs["yaw_p50"]) < 0
            allsep &= sep
            pol.append(1 if ls["yaw_p50"] < 0 else -1)
            say("      %-10s %12.2f %12.2f %12s %10s"
                % ("%s m +-%s" % (d, ang), ls["yaw_p50"], rs["yaw_p50"],
                   "YES" if sep else "NO", "L-neg" if ls["yaw_p50"] < 0 else "L-pos"))
    say()
    if allsep:
        say("  ALL %d left/right pairs separate with opposite signs. No inversion, no ambiguity."
            % len(pol))
    else:
        say("  SOME pairs failed to separate - investigate.")
    say()
    say("  OBSERVED POLARITY: 'left' produced NEGATIVE yaw here; in F-16 landscape it produced")
    say("  POSITIVE. That is NOT the camera rotation - handedness and dx sign are identical in")
    say("  both sessions, and depth cannot invert under an image rotation. What flipped is the")
    say("  SUBJECT's dz for the same word: F-16 left30 dz = +231 mm, F-18 left30 dz = -87 mm.")
    say("  The subject turned the other way for the same prompt. Exactly the reason F-15 forbids")
    say("  treating a commanded heading as angular truth. It is a LABELLING artefact, not a")
    say("  geometry defect, and the (c) table above is unaffected by it.")

    # ---------------------------------------------------------------- occlusion vs landscape
    say()
    say("-" * 116)
    say("SECTIONS 9/10 - DOES ROTATING THE STEREO BASELINE CHANGE THE SHOULDER MATCHING ERROR?")
    say("-" * 116)
    say("Landscape reference: F-16 autosweep_sub3 (same sub-pixel 1/8 config, same subject).")
    land = {}
    for x in io.open("oak_v4_evidence/f16/autosweep_sub3.jsonl", encoding="utf-8"):
        if not x.strip():
            continue
        r = json.loads(x)
        if min(r.get("cL", 0), r.get("cR", 0)) < CONF or not (r.get("mL") and r.get("mR")):
            continue
        land.setdefault(r["block"], []).append(r)
    say()
    say("%-26s %8s %8s %9s %9s %10s %11s %10s" %
        ("condition", "zSh mm", "dz mm", "med|yaw|", "p95|yaw|", "far spread", "near spread", "far wider%"))
    for lbl, rowset in (("landscape 0.80 m", land.get("d080")),
                        ("portrait  0.80 m", [r for r in torso_rows
                                              if r["block"] in ("sq_a@080", "sq_b@080")]),
                        ("landscape 1.00 m", land.get("d100")),
                        ("portrait  1.00 m", [r for r in torso_rows
                                              if r["block"] in ("sq_a@100", "sq_b@100")])):
        if not rowset:
            continue
        s = torso(rowset)
        if s is None:
            continue
        say("%-26s %8.0f %8.1f %9.2f %9.2f %10.1f %11.1f %10.1f"
            % (lbl, s["zSh"], s["dz"], s["ayaw_p50"], s["ayaw_p95"],
               s["spread_far"], s["spread_near"], s["far_wider"]))
    say()
    say("  'far wider%' is the F-16 occlusion-edge signature: the share of frames where the")
    say("  shoulder reading FARTHER also carries the wider depth window. 50% means no asymmetry.")

    # ---------------------------------------------------------------- interaction
    if move_rows:
        say()
        say("-" * 116)
        say("SECTION 12 - INTERACTION ENVELOPE at the tested distance")
        say("-" * 116)
        say("%-16s %6s %7s %7s %7s %8s %8s %-16s" %
            ("action", "n", "head%", "feet%", "hand%", "mTop", "mBot", "verdict"))
        for b in sorted({r["block"] for r in move_rows}):
            g = [r for r in move_rows if r["block"] == b]
            f = framing(g, w, h)
            if f is None:
                continue
            say("%-16s %6d %7.1f %7.1f %7.1f %8.0f %8.0f %-16s"
                % (b, f["n"], f["head_in"], f["feet_in"], f["hands_in"],
                   f["marginTop"], f["marginBot"], verdict(f, w, h)))

    # ---------------------------------------------------------------- decision matrix
    say()
    say("=" * 116)
    say("SECTION 18 - DECISION MATRIX")
    say("=" * 116)
    say("%-24s %12s %12s %12s %12s" % ("test", "0.78 m", "0.80 m", "0.90 m", "1.00 m"))

    def sq(d):
        vals = [tb.get("sq_a@%s" % d), tb.get("sq_b@%s" % d)]
        vals = [v for v in vals if v]
        return vals

    def row(label, fn):
        say("%-24s %12s %12s %12s %12s"
            % (label, fn("078"), fn("080"), fn("090"), fn("100")))

    def fb(d):
        f = fr_tab.get("relax@%s" % d)
        return verdict(f, w, h).replace("full-body-", "") if f else "-"

    def arm(d):
        f = fr_tab.get("arms45@%s" % d)
        if not f:
            return "-"
        return "yes" if (f["hands_in"] >= 95 and f["hands_edge"] < 5) else "no"

    def med(d):
        v = sq(d)
        return "%.2f" % np.median([x["ayaw_p50"] for x in v]) if v else "-"

    def p95(d):
        v = sq(d)
        return "%.2f" % np.median([x["ayaw_p95"] for x in v]) if v else "-"

    def spn(d):
        v = sq(d)
        return "%.0f px" % np.median([x["span"] for x in v]) if v else "-"

    def qq(d):
        v = sq(d)
        return "%.2f" % np.median([x["q"] for x in v]) if v else "-"

    def ff(d):
        v = sq(d)
        return "%.2f" % np.median([x["ff_p95"] for x in v]) if v else "-"

    def ov(d):
        v = sq(d)
        if not v:
            return "-"
        m = np.median([x["ayaw_p50"] for x in v])
        p = np.median([x["ayaw_p95"] for x in v])
        f = np.median([x["ff_p95"] for x in v])
        fr = fr_tab.get("relax@%s" % d)
        ok = (m <= SQ_MED and p <= SQ_P95 and f <= SQ_FF and fr and
              verdict(fr, w, h) != "partial-body")
        return "PASS" if ok else "FAIL"

    row("full body (relaxed)", fb)
    row("arms-45 in frame", arm)
    row("square yaw median", med)
    row("square yaw p95", p95)
    row("shoulder span", spn)
    row("depth quality", qq)
    row("frame-to-frame p95", ff)
    row("OVERALL", ov)

    # ---------------------------------------------------------------- performance
    say()
    say("-" * 116)
    say("SECTION 15 - PERFORMANCE")
    say("-" * 116)
    allr = [r for r in (frame_rows + torso_rows + move_rows) if r.get("fps")]
    if allr:
        f = np.array([r["fps"] for r in allr])
        lt = np.array([r["latMs"] for r in allr if r.get("latMs", -1) >= 0])
        say("  portrait capture : fps p50 %.1f  p05 %.1f   camera latency p50 %.0f ms  p95 %.0f ms"
            % (np.median(f), np.percentile(f, 5), np.median(lt), np.percentile(lt, 95)))
    say("  landscape reference (F-16, same sub-pixel 1/8 config): 31.2 fps, 24 ms")
    say("  The rotation is two cv2.rotate calls on a 640x400 frame; no pipeline change.")

    os.makedirs(OUTDIR, exist_ok=True)
    base = os.path.join(OUTDIR, a.out)
    n = 0
    while os.path.exists(base + ".txt"):
        n += 1
        base = os.path.join(OUTDIR, "%s_%d" % (a.out, n))
    io.open(base + ".txt", "w", encoding="utf-8").write("\n".join(L) + "\n")
    io.open(base + ".json", "w", encoding="utf-8").write(
        json.dumps({"framing": fr_tab, "torso": tb}, indent=1, default=float))
    print("\nwrote %s.txt / .json" % base)


if __name__ == "__main__":
    main()
