#!/usr/bin/env python3
"""
F-16 analysis - turns f16_capture JSONL into the per-block metric tables the brief requires
(sections 4, 6, 7, 11, 12). Offline only; reads nothing from production.

Usage:
    python f16_analyze.py oak_v4_evidence/f16/cap_*.jsonl --out <name> [--sq-blocks d080,d100]
"""
import argparse
import glob
import io
import json
import math
import os

import numpy as np

import f16_configs as C

OUTDIR = os.path.join("oak_v4_evidence", "f16")
CONF_MIN = 0.30

# Section 12 proposed engineering thresholds for a genuinely square subject.
SQ_MEDIAN_MAX_DEG = 5.0
SQ_P95_MAX_DEG = 10.0


def load(paths):
    rows = []
    for p in paths:
        for line in io.open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            r["_src"] = os.path.basename(p)
            rows.append(r)
    return rows


def ladder(vals):
    """Gaps between adjacent DISTINCT values of a pool -> the observed quantisation ladder."""
    u = np.unique(np.asarray(vals, dtype=np.float64))
    if u.size < 3:
        return None
    g = np.diff(u)
    g = g[g > 0]
    if g.size == 0:
        return None
    return dict(n_unique=int(u.size), gmin=float(g.min()), gp50=float(np.median(g)),
                gp95=float(np.percentile(g, 95)), gmax=float(g.max()))


def block_stats(rows):
    """All section-4 metrics for one homogeneous block."""
    good = [r for r in rows if r.get("cL", 0) >= CONF_MIN and r.get("cR", 0) >= CONF_MIN]
    s = {"n_raw": len(rows), "n_conf": len(good)}
    if not good:
        return s
    cfg_name = good[0]["cfg"]
    cfg = C.CONFIGS.get(cfg_name, {})
    mono_w = C.MONO_RES.get(cfg.get("mono", "400p"), (None, 640, 400))[1]
    bits = cfg.get("bits", 0) if cfg.get("subpixel") else 0
    s["cfg"] = cfg_name

    span = np.array([r["uSpan"] for r in good])
    s["uSpan_p10"] = float(np.percentile(span, 10))
    s["uSpan_p50"] = float(np.median(span))
    s["uSpan_p90"] = float(np.percentile(span, 90))

    hz = np.array([r["hipZ"] for r in good if r.get("hipZ")], dtype=float)
    s["hipZ_p50_mm"] = float(np.median(hz) * 1000.0) if hz.size else float("nan")
    s["hipZ_n"] = int(hz.size)

    meas = [r for r in good if r.get("mL") and r.get("mR")]
    s["measured_pct"] = 100.0 * len(meas) / len(good)
    s["quality_p50"] = float(np.median([min(r["qL"], r["qR"]) for r in good]))

    if meas:
        zl = np.array([r["zL"] for r in meas], dtype=float)
        zr = np.array([r["zR"] for r in meas], dtype=float)
        s["zSh_p50_mm"] = float(np.median(0.5 * (zl + zr)))
        dz = np.array([r["dz"] for r in meas], dtype=float)
        dx = np.array([abs(r["dx"]) for r in meas], dtype=float)
        s["dz_p50"] = float(np.median(dz))
        s["dz_p05"] = float(np.percentile(dz, 5))
        s["dz_p95"] = float(np.percentile(dz, 95))
        s["dz_absp50"] = float(np.median(np.abs(dz)))
        s["dx_p50"] = float(np.median(dx))
        s["dx_p10"] = float(np.percentile(dx, 10))

        # ---- observed depth ladder, measured AT THE SHOULDERS -----------------------------------
        pool = []
        for r in meas:
            pool.extend(r.get("wL") or [])
            pool.extend(r.get("wR") or [])
        lad = ladder(pool) if len(pool) > 20 else None
        if lad:
            s["step_obs_p50"] = lad["gp50"]
            s["step_obs_min"] = lad["gmin"]
            s["step_uniq"] = lad["n_unique"]
        zc = s.get("zSh_p50_mm") or s["hipZ_p50_mm"]
        if zc and zc == zc:
            s["step_theory"] = C.theoretical_step_mm(zc, mono_w, bits)

        # ---- torso yaw --------------------------------------------------------------------------
        yaw = np.array([r["yaw3D"] for r in meas if r.get("yaw3D") is not None], dtype=float)
        if yaw.size:
            s["yaw_n"] = int(yaw.size)
            s["yaw_p50"] = float(np.median(yaw))
            s["yaw_mean"] = float(np.mean(yaw))
            s["yaw_std"] = float(np.std(yaw))
            ay = np.abs(yaw)
            s["yaw_absp50"] = float(np.median(ay))
            s["yaw_absp95"] = float(np.percentile(ay, 95))
            s["yaw_absmax"] = float(ay.max())
            for t in (1, 2, 5, 10):
                s["within_%d" % t] = 100.0 * float((ay <= t).mean())
            s["yaw_uniq"] = int(np.unique(np.round(yaw, 3)).size)
            d = np.abs(np.diff(yaw))
            if d.size:
                s["dyaw_p50"] = float(np.median(d))
                s["dyaw_p95"] = float(np.percentile(d, 95))
                s["dyaw_max"] = float(d.max())

        # ---- THE headline: degrees of yaw per ONE depth step ------------------------------------
        st = s.get("step_obs_p50") or s.get("step_theory")
        if st and s.get("dx_p50"):
            s["yaw_per_step_deg"] = math.degrees(math.atan(st / s["dx_p50"]))
            s["yaw_per_mm_deg"] = math.degrees(1.0 / s["dx_p50"])
            s["dz_budget_5deg_mm"] = s["dx_p50"] * math.tan(math.radians(SQ_MEDIAN_MAX_DEG))
    return s


def fmt(v, w=8, p=1):
    if v is None or (isinstance(v, float) and v != v):
        return " " * (w - 1) + "-"
    return ("%%%d.%df" % (w, p)) % v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--group", default="block", choices=["block", "cfg", "cfgblock"])
    a = ap.parse_args()

    paths = []
    for p in a.paths:
        paths.extend(sorted(glob.glob(p)))
    rows = load(paths)
    out = []

    def say(s=""):
        print(s)
        out.append(s)

    say("=" * 118)
    say("F-16 CAPTURE ANALYSIS")
    say("=" * 118)
    say("files   : %s" % ", ".join(os.path.basename(p) for p in paths))
    say("frames  : %d" % len(rows))
    if not rows:
        say("NO DATA")
        return

    def key(r):
        if a.group == "block":
            return r["block"]
        if a.group == "cfg":
            return r["cfg"]
        return "%s/%s" % (r["cfg"], r["block"])

    groups = {}
    order = []
    for r in rows:
        k = key(r)
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(r)

    stats = {k: block_stats(groups[k]) for k in order}

    # ---------------------------------------------------------------- geometry + depth
    say()
    say("-" * 118)
    say("GEOMETRY AND DEPTH")
    say("-" * 118)
    say("%-16s %6s %6s %8s %8s %8s %8s %8s %8s %7s %6s" %
        ("group", "n", "conf", "uSpan50", "hipZ", "zSh", "dx", "dz_p50", "|dz|p50", "meas%", "qual"))
    for k in order:
        s = stats[k]
        say("%-16s %6d %6d %s %s %s %s %s %s %s %s" %
            (k[:16], s["n_raw"], s.get("n_conf", 0), fmt(s.get("uSpan_p50")), fmt(s.get("hipZ_p50_mm"), 8, 0),
             fmt(s.get("zSh_p50_mm"), 8, 0), fmt(s.get("dx_p50")), fmt(s.get("dz_p50")),
             fmt(s.get("dz_absp50")), fmt(s.get("measured_pct"), 7), fmt(s.get("quality_p50"), 6, 2)))

    # ---------------------------------------------------------------- quantisation
    say()
    say("-" * 118)
    say("DEPTH QUANTISATION AT THE SHOULDER WINDOWS  (observed ladder vs Z^2*dd/(f*B))")
    say("-" * 118)
    say("%-16s %10s %10s %8s %10s %9s" %
        ("group", "step_obs", "step_min", "uniq", "step_thy", "obs/thy"))
    for k in order:
        s = stats[k]
        o, t = s.get("step_obs_p50"), s.get("step_theory")
        say("%-16s %s %s %s %s %s" %
            (k[:16], fmt(o, 10, 2), fmt(s.get("step_obs_min"), 10, 2), fmt(s.get("step_uniq"), 8, 0),
             fmt(t, 10, 2), fmt((o / t) if (o and t) else None, 9, 2)))

    # ---------------------------------------------------------------- torso yaw
    say()
    say("-" * 118)
    say("TORSO YAW  (yaw3D = the shipped Kalidokit y-channel)")
    say("-" * 118)
    say("%-16s %8s %8s %8s %8s %8s %7s %7s %7s %7s" %
        ("group", "p50", "mean", "std", "|p95|", "|max|", "<=1deg", "<=2deg", "<=5deg", "<=10deg"))
    for k in order:
        s = stats[k]
        say("%-16s %s %s %s %s %s %s %s %s %s" %
            (k[:16], fmt(s.get("yaw_p50")), fmt(s.get("yaw_mean")), fmt(s.get("yaw_std")),
             fmt(s.get("yaw_absp95")), fmt(s.get("yaw_absmax")),
             fmt(s.get("within_1"), 7), fmt(s.get("within_2"), 7),
             fmt(s.get("within_5"), 7), fmt(s.get("within_10"), 7)))

    say()
    say("%-16s %9s %9s %9s %8s" % ("group", "dyaw_p50", "dyaw_p95", "dyaw_max", "uniq"))
    for k in order:
        s = stats[k]
        say("%-16s %s %s %s %s" %
            (k[:16], fmt(s.get("dyaw_p50"), 9, 2), fmt(s.get("dyaw_p95"), 9, 2),
             fmt(s.get("dyaw_max"), 9, 2), fmt(s.get("yaw_uniq"), 8, 0)))

    # ---------------------------------------------------------------- the headline
    say()
    say("=" * 118)
    say("SECTION 7 - HOW MUCH TORSO YAW DOES ONE DEPTH STEP BUY?")
    say("=" * 118)
    say("yaw3D = atan2(dx, dz) + 90deg  =>  d(yaw)/d(dz) = 1/|dx| rad/mm at dz=0.")
    say("A depth step of `step` mm therefore commands atan(step/|dx|) degrees of torso yaw.")
    say()
    say("%-16s %8s %10s %12s %14s %14s" %
        ("group", "dx(mm)", "step(mm)", "deg/step", "deg per mm", "dz budget +-5deg"))
    for k in order:
        s = stats[k]
        say("%-16s %s %s %s %s %s" %
            (k[:16], fmt(s.get("dx_p50")), fmt(s.get("step_obs_p50") or s.get("step_theory"), 10, 2),
             fmt(s.get("yaw_per_step_deg"), 12, 1), fmt(s.get("yaw_per_mm_deg"), 14, 3),
             fmt(s.get("dz_budget_5deg_mm"), 14, 1)))

    # ---------------------------------------------------------------- verdict per group
    say()
    say("=" * 118)
    say("SECTION 12 - ZERO-YAW ACCEPTANCE  (median |yaw| <= %.0f deg AND p95 <= %.0f deg)"
        % (SQ_MEDIAN_MAX_DEG, SQ_P95_MAX_DEG))
    say("=" * 118)
    say("Applies ONLY to blocks where the subject was genuinely square. Others are shown for context.")
    say("%-16s %10s %10s %8s" % ("group", "med|yaw|", "p95|yaw|", "verdict"))
    for k in order:
        s = stats[k]
        m, p = s.get("yaw_absp50"), s.get("yaw_absp95")
        if m is None:
            say("%-16s %10s %10s %8s" % (k[:16], "-", "-", "no data"))
            continue
        v = "PASS" if (m <= SQ_MEDIAN_MAX_DEG and p <= SQ_P95_MAX_DEG) else "FAIL"
        say("%-16s %s %s %8s" % (k[:16], fmt(m, 10, 2), fmt(p, 10, 2), v))

    os.makedirs(OUTDIR, exist_ok=True)
    base = os.path.join(OUTDIR, a.out)
    n = 0
    while os.path.exists(base + ".txt"):
        n += 1
        base = os.path.join(OUTDIR, "%s_%d" % (a.out, n))
    io.open(base + ".txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
    io.open(base + ".json", "w", encoding="utf-8").write(json.dumps(stats, indent=1))
    print("\nwrote %s.txt / .json" % base)


if __name__ == "__main__":
    main()
