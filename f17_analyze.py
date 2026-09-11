#!/usr/bin/env python3
"""
F-17 offline multi-baseline analysis - the decisive section-9 test.

For every captured frame, the SAME shoulder keypoints are measured three ways:

    device  BC 74.99 mm   OAK StereoDepth sub-pixel 1/8, aligned to CAM_A   (the F-16 reference)
    host    BC 74.99 mm   cv2 rectify + SGBM                                (matcher control)
    host    AC 37.64 mm   cv2 rectify + SGBM, IDENTICAL matcher             (HALF the baseline)

and the implied systematic disparity bias is fitted for each:

    dz = Z^2 * dd / (f*B)      =>      dd = dz * (f*B) / Z^2

CASE A  dd is unchanged when the baseline halves  -> the bias lives in the PIXEL domain, so a wider
        baseline divides the angular error proportionally. Wider baseline is the fix.
CASE B  dz is unchanged when the baseline halves  -> the bias lives in the DEPTH domain, so a wider
        baseline buys nothing. The shoulder matching geometry is the fix.

Keypoint handling, so no baseline is advantaged:
  pair AC - CAM_A IS the left camera, so the manifest's CAM_A pixels map exactly into rectified-A
            via undistortPoints(K_A, D_A, R1, P1).
  pair BC - CAM_A is not in the pair, so the pixel is carried into CAM_B through the DEVICE depth
            (used only to LOCATE the window, never to measure it) and then rectified.

Output: oak_v4_evidence/f17/multibaseline.txt / .json
"""
import io
import json
import math
import os
import sys

import numpy as np
import cv2

import f17_stereo_rig as RIG
from f17_tune import load_calib, rectify_pair

OUTDIR = os.path.join("oak_v4_evidence", "f17")
KWIN = 5
CONF_MIN = 0.30
SHOULDER_W_MM = 333.1          # this subject, calibrated in F-16
L = []


def say(s=""):
    print(s)
    L.append(s)


def matcher():
    b = 9
    return cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=192, blockSize=b,
        P1=8 * b * b, P2=32 * b * b, disp12MaxDiff=1, uniquenessRatio=5,
        speckleWindowSize=100, speckleRange=2, preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_HH)


def sample(dep, u, v, k=KWIN, pct=30.0):
    h, w = dep.shape
    x, y = int(round(u)), int(round(v))
    r = k // 2
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    if x1 <= x0 or y1 <= y0:
        return None, 0, None
    win = dep[y0:y1, x0:x1].reshape(-1)
    win = win[win > 0]
    if win.size < 6:
        return None, int(win.size), None
    return (float(np.percentile(win, pct)), int(win.size),
            float(win.max() - win.min()))


def yaw_from(zl, zr, ul, ur, f, cx):
    """Production's geometry: back-project both shoulders, then the shipped Kalidokit y-channel."""
    xl = (ul - cx) * zl / f
    xr = (ur - cx) * zr / f
    return RIG.line_yaw_deg([xl, 0.0, zl], [xr, 0.0, zr]), (xr - xl), (zr - zl)


def rect_points(pts, K, D, R1, P1):
    p = np.array(pts, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(p, K, D, R=R1, P=P1)
    return out.reshape(-1, 2)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "oak_v4_evidence/f17/raw/sweep1"
    K, D, E = load_calib(root)
    rect = {"BC": rectify_pair(K, D, E, "B", "C"),
            "AC": rectify_pair(K, D, E, "A", "C")}
    E_AB = E["A->B"]
    R_AB, T_AB = E_AB[:3, :3], E_AB[:3, 3] * 10.0

    KA = K["A"]
    fxA, cxA, cyA = KA[0, 0], KA[0, 2], KA[1, 2]

    man = [json.loads(x) for x in io.open(os.path.join(root, "manifest.jsonl"), encoding="utf-8")
           if x.strip()]
    m = matcher()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    say("=" * 108)
    say("F-17 MULTI-BASELINE ANALYSIS")
    say("=" * 108)
    say("capture: %s   frame sets: %d" % (root, len(man)))
    say("matcher: SGBM blockSize 9, 192 disparities, MODE_HH, CLAHE - IDENTICAL for both pairs")
    for k, r in rect.items():
        say("  host %-3s  B %8.3f mm  f_rect %8.3f px  f*B %9.1f" % (k, r["B"], r["f"], r["fB"]))
    say("  device BC B   74.992 mm  fx     282.995 px  f*B   21222.2   (sub-pixel 1/8)")

    rows = []
    for rec in man:
        if min(rec["conf_sh"]) < CONF_MIN:
            continue
        bdir = os.path.join(root, rec["block"])
        i = rec["i"]
        a = cv2.imread(os.path.join(bdir, "a_%04d.png" % i), cv2.IMREAD_COLOR)
        gb = cv2.imread(os.path.join(bdir, "b_%04d.png" % i), cv2.IMREAD_GRAYSCALE)
        gc = cv2.imread(os.path.join(bdir, "c_%04d.png" % i), cv2.IMREAD_GRAYSCALE)
        dd = cv2.imread(os.path.join(bdir, "d_%04d.png" % i), cv2.IMREAD_UNCHANGED)
        if a is None or gb is None or gc is None or dd is None:
            continue
        dd = dd.astype(np.float32)
        (uL, vL), (uR, vR) = rec["uv_sh"]

        out = {"block": rec["block"], "target_m": rec["target_m"], "i": i,
               "estRange_m": rec["estRange_m"], "uSpan": rec["uSpan"]}

        # ---------------- device (the F-16 reference) --------------------------------------
        zl, nl, sl = sample(dd, uL, vL)
        zr, nr, sr = sample(dd, uR, vR)
        if zl and zr:
            y, dx, dz = yaw_from(zl, zr, uL, uR, fxA, cxA)
            out["dev"] = dict(zl=zl, zr=zr, dx=dx, dz=dz, yaw=y, sl=sl, sr=sr,
                              z=0.5 * (zl + zr))

        gray = {"A": clahe.apply(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)),
                "B": clahe.apply(gb), "C": clahe.apply(gc)}

        # ---------------- host AC : CAM_A is the left camera, exact mapping -----------------
        r = rect["AC"]
        li = cv2.remap(gray["A"], r["m1"][0], r["m1"][1], cv2.INTER_LINEAR)
        ri = cv2.remap(gray["C"], r["m2"][0], r["m2"][1], cv2.INTER_LINEAR)
        depAC = RIG.depth_from_disp(m.compute(li, ri), r["fB"])
        p = rect_points([(uL, vL), (uR, vR)], K["A"], D["A"], r["R1"], r["P1"])
        zl, nl, sl = sample(depAC, p[0, 0], p[0, 1])
        zr, nr, sr = sample(depAC, p[1, 0], p[1, 1])
        if zl and zr:
            y, dx, dz = yaw_from(zl, zr, p[0, 0], p[1, 0], r["f"], float(r["P1"][0, 2]))
            out["AC"] = dict(zl=zl, zr=zr, dx=dx, dz=dz, yaw=y, sl=sl, sr=sr,
                             z=0.5 * (zl + zr), fB=r["fB"], B=r["B"])

        # ---------------- host BC : carry the pixel into CAM_B via the DEVICE depth ---------
        if "dev" in out:
            r = rect["BC"]
            li = cv2.remap(gray["B"], r["m1"][0], r["m1"][1], cv2.INTER_LINEAR)
            ri = cv2.remap(gray["C"], r["m2"][0], r["m2"][1], cv2.INTER_LINEAR)
            depBC = RIG.depth_from_disp(m.compute(li, ri), r["fB"])
            pb = []
            for (u, v, z) in ((uL, vL, out["dev"]["zl"]), (uR, vR, out["dev"]["zr"])):
                P = np.array([(u - cxA) * z / fxA, (v - cyA) * z / KA[1, 1], z])
                Q = R_AB.dot(P) + T_AB
                if Q[2] <= 1.0:
                    pb = []
                    break
                pb.append((K["B"][0, 0] * Q[0] / Q[2] + K["B"][0, 2],
                           K["B"][1, 1] * Q[1] / Q[2] + K["B"][1, 2]))
            if pb:
                q = rect_points(pb, K["B"], D["B"], r["R1"], r["P1"])
                zl, nl, sl = sample(depBC, q[0, 0], q[0, 1])
                zr, nr, sr = sample(depBC, q[1, 0], q[1, 1])
                if zl and zr:
                    y, dx, dz = yaw_from(zl, zr, q[0, 0], q[1, 0], r["f"],
                                         float(r["P1"][0, 2]))
                    out["BC"] = dict(zl=zl, zr=zr, dx=dx, dz=dz, yaw=y, sl=sl, sr=sr,
                                     z=0.5 * (zl + zr), fB=r["fB"], B=r["B"])
        rows.append(out)

    say("  usable frames: %d" % len(rows))

    # ------------------------------------------------------------------ per-block tables
    blocks = sorted({r["block"] for r in rows})
    FB = {"dev": 21222.2, "BC": rect["BC"]["fB"], "AC": rect["AC"]["fB"]}

    def agg(sel, key):
        g = [r[key] for r in sel if key in r]
        if len(g) < 5:
            return None
        dz = np.array([x["dz"] for x in g])
        dx = np.array([abs(x["dx"]) for x in g])
        z = np.array([x["z"] for x in g])
        yaw = np.array([x["yaw"] for x in g])
        dd = dz * FB[key] / (z * z)
        return dict(n=len(g), dz=float(np.median(dz)), dx=float(np.median(dx)),
                    z=float(np.median(z)), yaw=float(np.median(yaw)),
                    ayaw=float(np.median(np.abs(yaw))),
                    ayaw95=float(np.percentile(np.abs(yaw), 95)),
                    dd=float(np.median(dd)), dd_mean=float(np.mean(dd)),
                    dd_std=float(np.std(dd)),
                    cov=100.0 * len(g) / max(1, len(sel)),
                    spread=float(np.median([max(x["sl"], x["sr"]) for x in g])))

    say()
    say("-" * 108)
    say("PER-DISTANCE, PER-SOURCE")
    say("-" * 108)
    say("%-7s %-8s %5s %7s %8s %9s %9s %9s %9s %8s" %
        ("block", "source", "n", "cov%", "Z mm", "dz mm", "|dx| mm", "med|yaw|", "p95|yaw|", "dd px"))
    table = {}
    for b in blocks:
        sel = [r for r in rows if r["block"] == b]
        table[b] = {}
        for key, lab in (("dev", "deviceBC"), ("BC", "hostBC"), ("AC", "hostAC")):
            s = agg(sel, key)
            table[b][key] = s
            if s is None:
                say("%-7s %-8s     -  (insufficient valid frames)" % (b, lab))
                continue
            say("%-7s %-8s %5d %7.1f %8.0f %9.1f %9.1f %9.2f %9.2f %8.3f" %
                (b, lab, s["n"], s["cov"], s["z"], s["dz"], s["dx"], s["ayaw"],
                 s["ayaw95"], s["dd"]))

    # ------------------------------------------------------------------ THE decisive test
    say()
    say("=" * 108)
    say("SECTION 9 - DOES THE SYSTEMATIC DISPARITY BIAS SURVIVE A BASELINE CHANGE?")
    say("=" * 108)
    say("CASE A: dd unchanged, dz doubles when the baseline halves -> PIXEL-domain bias,")
    say("        a wider baseline divides the angular error proportionally.")
    say("CASE B: dz unchanged, dd halves -> DEPTH-domain bias, a wider baseline buys nothing.")
    say()
    say("%-7s %11s %11s %9s   %11s %11s %9s" %
        ("block", "dz_BC mm", "dz_AC mm", "ratio", "dd_BC px", "dd_AC px", "ratio"))
    rz, rd = [], []
    for b in blocks:
        sb, sa = table[b].get("BC"), table[b].get("AC")
        if not (sb and sa) or abs(sb["dz"]) < 1e-6:
            continue
        r_z = sa["dz"] / sb["dz"]
        r_d = sa["dd"] / sb["dd"] if abs(sb["dd"]) > 1e-9 else float("nan")
        rz.append(r_z)
        rd.append(r_d)
        say("%-7s %11.1f %11.1f %9.3f   %11.3f %11.3f %9.3f"
            % (b, sb["dz"], sa["dz"], r_z, sb["dd"], sa["dd"], r_d))
    if rz:
        say()
        say("  median dz ratio (AC/BC) = %.3f    CASE A predicts ~2.0, CASE B predicts ~1.0"
            % float(np.median(rz)))
        say("  median dd ratio (AC/BC) = %.3f    CASE A predicts ~1.0, CASE B predicts ~0.5"
            % float(np.median(rd)))

    # ------------------------------------------------------------------ shoulder windows
    say()
    say("-" * 108)
    say("SECTION 10 - SHOULDER WINDOW SPREAD (does a different baseline change the edge problem?)")
    say("-" * 108)
    say("%-7s %14s %14s %14s" % ("block", "device spread", "hostBC spread", "hostAC spread"))
    for b in blocks:
        v = []
        for key in ("dev", "BC", "AC"):
            s = table[b].get(key)
            v.append("%.1f mm" % s["spread"] if s else "-")
        say("%-7s %14s %14s %14s" % (b, v[0], v[1], v[2]))

    io.open(os.path.join(OUTDIR, "multibaseline.txt"), "w", encoding="utf-8").write(
        "\n".join(L) + "\n")
    io.open(os.path.join(OUTDIR, "multibaseline.json"), "w", encoding="utf-8").write(
        json.dumps(table, indent=1, default=float))
    print("\nwrote %s" % os.path.join(OUTDIR, "multibaseline.txt"))


if __name__ == "__main__":
    main()
