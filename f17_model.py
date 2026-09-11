#!/usr/bin/env python3
"""
F-17 sections 5 / 17 - what baseline would actually be required, under BOTH models.

The brief forbids recommending a purchase from the ideal formula alone, so this computes the
required baseline two ways and shows how far apart they are:

  IDEAL    the bias is a fixed PIXEL error, so dz ~ 1/(f*B)  ->  dz halves when f*B doubles
  EMPIRICAL the only measurement available of how the bias responds to a change in f*B is F-16's
            resolution sweep: mono 640x400 -> 1280x800 doubled f*B and moved dz 82.2 -> 65.0 mm,
            i.e. -21 %, not the -50 % the ideal model predicts.

Caveat carried into the report: a resolution change is not a baseline change. It is the closest
available probe, not a substitute.

Output: oak_v4_evidence/f17/model.txt
"""
import io
import math
import os

import numpy as np

OUT = os.path.join("oak_v4_evidence", "f17", "model.txt")

FX400 = 282.995        # CAM_C, 640x400, EEPROM
H_PX = 400
B_NOW = 74.992
FB_NOW = FX400 * B_NOW
DD = 0.75              # F-16 measured implied disparity bias, median over 7 distances
W_SH = 333.1           # measured shoulder-keypoint separation of the F-16/F-17 subject
BODY_MM = 1750.0
VFOV = 70.2            # measured

# F-16 measured, production configuration, subject square (device sub-pixel 1/8)
MEASURED = {1.24: 75.0, 1.33: 78.0, 1.50: 88.0, 1.80: 129.0, 2.00: 138.0}   # dz mm, F-17 capture

# F-16 resolution sweep, 1.33 m: f*B doubled, dz fell by this fraction only.
EMPIRICAL_GAIN = 1.0 - (65.0 / 82.2)       # 0.209

L = []


def say(s=""):
    print(s)
    L.append(s)


def yaw_of(dz, w=W_SH):
    return math.degrees(math.atan(abs(dz) / w))


def dz_ideal(z_mm, B):
    return z_mm * z_mm * DD / (FX400 * B)


def b_required(z_mm, target_deg):
    """Ideal model: baseline that brings the square-stance error down to target_deg."""
    dz_max = W_SH * math.tan(math.radians(target_deg))
    return z_mm * z_mm * DD / (dz_max * FX400)


say("=" * 100)
say("F-17 REQUIRED-BASELINE MODEL")
say("=" * 100)
say("current  B = %.3f mm   fx = %.3f px   f*B = %.1f mm*px" % (B_NOW, FX400, FB_NOW))
say("measured implied disparity bias dd = %.2f px (F-16, median over 7 distances)" % DD)
say("measured shoulder separation W = %.1f mm" % W_SH)

# ---------------------------------------------------------------- ideal model vs measurement
say()
say("-" * 100)
say("1. DOES THE IDEAL MODEL REPRODUCE THE MEASUREMENT AT THE CURRENT BASELINE?")
say("-" * 100)
say("%-8s %12s %12s %12s %12s" % ("Z m", "dz measured", "dz ideal", "yaw meas", "yaw ideal"))
for z, dz in sorted(MEASURED.items()):
    di = dz_ideal(z * 1000.0, B_NOW)
    say("%-8.2f %12.1f %12.1f %12.2f %12.2f" % (z, dz, di, yaw_of(dz), yaw_of(di)))
say()
say("  The ideal model reproduces the CURRENT baseline well - that is what F-16 fitted dd on.")
say("  It says nothing about how dd behaves when the baseline CHANGES.")

# ---------------------------------------------------------------- required baseline
say()
say("-" * 100)
say("2. IDEAL MODEL - baseline required for a square-stance error of <=5 deg and <=3 deg")
say("-" * 100)
say("%-8s %14s %14s %16s" % ("Z m", "B for <=5 deg", "B for <=3 deg", "x current"))
for z in (1.24, 1.33, 1.50, 1.80, 2.00):
    b5 = b_required(z * 1000.0, 5.0)
    b3 = b_required(z * 1000.0, 3.0)
    say("%-8.2f %14.1f %14.1f %16s" % (z, b5, b3, "%.2fx / %.2fx" % (b5 / B_NOW, b3 / B_NOW)))

say()
say("  At the MINIMUM full-body distance (1.24 m) the ideal model asks for ~%.0f mm - about"
    % b_required(1240.0, 5.0))
say("  2x the current 75 mm, and within reach of stock wide-baseline stereo hardware.")

# ---------------------------------------------------------------- 150 mm prediction
say()
say("-" * 100)
say("3. IDEAL MODEL - what a 150 mm baseline (a typical wide-baseline stereo head) would give")
say("-" * 100)
say("%-8s %12s %12s %10s" % ("Z m", "dz mm", "yaw deg", "<=5 deg?"))
for z in (1.24, 1.33, 1.50, 1.80, 2.00):
    dz = dz_ideal(z * 1000.0, 150.0)
    say("%-8.2f %12.1f %12.2f %10s" % (z, dz, yaw_of(dz), "PASS" if yaw_of(dz) <= 5.0 else "FAIL"))

# ---------------------------------------------------------------- empirical scaling
say()
say("-" * 100)
say("4. EMPIRICAL SCALING - how the bias ACTUALLY responds to a change in f*B")
say("-" * 100)
say("F-16's 1.33 m configuration sweep contains THREE pairs that differ ONLY by mono resolution,")
say("i.e. three independent doublings of f*B on the same subject in one continuous hold. All")
say("three are used; picking only the favourable one would misstate the result.")
say()
say("%-34s %10s %10s %9s" % ("comparison (f*B x2)", "dz before", "dz after", "ratio"))
PAIRS = [("baseline -> mono800", 89.0, 95.0),
         ("sub3 -> mono800_sub3", 82.2, 65.0),
         ("sub3_rgb800 -> best", 97.0, 86.0)]
ratios = []
for lab, a, b in PAIRS:
    ratios.append(b / a)
    say("%-34s %10.1f %10.1f %9.3f" % (lab, a, b, b / a))
r_mean = float(np.mean(ratios))
r_med = float(np.median(ratios))
say()
say("   mean ratio %.3f   median %.3f   spread %.3f" % (r_mean, r_med, float(np.std(ratios))))
say()
say("   A pure PIXEL-domain bias scales away with f*B            -> ratio 0.50 per doubling")
say("   A pure DEPTH-domain bias (background blending) does not  -> ratio 1.00 per doubling")
say("   MEASURED                                                 -> ratio %.2f per doubling" % r_mean)
say()
say("   Doubling f*B moves the bias by %+.0f %% on average, with one of the three comparisons"
    % (100 * (r_mean - 1.0)))
say("   moving the WRONG WAY (+6.7 %). That is indistinguishable from no effect, and nowhere")
say("   near the -50 % a fixed pixel offset would give. Quantisation fell 2x in every one of")
say("   these pairs, so f*B certainly changed - the BIAS simply does not follow it.")
say()
say("   CAVEAT carried into the report: these doublings come from RESOLUTION, not BASELINE. They")
say("   are the closest available probe of how the bias responds to f*B, not a substitute for")
say("   measuring a wider baseline.")

say()
say("-" * 100)
say("5. IF A BASELINE CHANGE SCALES THE WAY THESE f*B CHANGES DID")
say("-" * 100)
say("%-10s %10s %12s %12s %10s" % ("B mm", "doublings", "dz @1.33", "yaw", "<=5 deg?"))
dz0 = 78.0
for B in (75, 100, 150, 200, 300, 400, 600):
    k = math.log(B / B_NOW, 2.0)
    dz = dz0 * (r_mean ** k)
    say("%-10d %10.2f %12.1f %12.2f %10s"
        % (B, k, dz, yaw_of(dz), "PASS" if yaw_of(dz) <= 5.0 else "FAIL"))
need = math.log(W_SH * math.tan(math.radians(5.0)) / dz0, r_mean)
say()
say("   Under this scaling, reaching 5 deg at 1.33 m needs %.1f doublings = B ~ %.0f mm."
    % (need, B_NOW * (2 ** need)))
say("   No such stereo head exists. THE TWO MODELS DISAGREE BY ORDERS OF MAGNITUDE, and the")
say("   measurement that would settle it is exactly the one this hardware cannot make.")

# ---------------------------------------------------------------- framing coupling
say()
say("-" * 100)
say("6. THE COUPLING THE BRIEF ASKS FOR (section 12): FOV sets the distance, distance sets B")
say("-" * 100)
say("A candidate cannot be judged on baseline alone - a narrower lens pushes the full-body")
say("distance out, and the required baseline grows as Z^2.")
say()
say("%-12s %10s %12s %14s %16s" % ("V-FOV deg", "Z_min m", "dz @Z_min", "yaw @75 mm", "B for <=5 deg"))
for vf in (50.0, 60.0, 70.2, 80.0, 95.0):
    zmin = BODY_MM / (2.0 * math.tan(math.radians(vf / 2.0)))
    dz = dz_ideal(zmin, B_NOW)
    say("%-12.1f %10.2f %12.1f %14.2f %16.1f"
        % (vf, zmin / 1000.0, dz, yaw_of(dz), b_required(zmin, 5.0)))
say()
say("   Closed form:  B_req = H_body^2 * dd / (2 * h_px * W * tan(eps) * tan(vfov/2))")
say("   Widening the LENS is as effective as widening the BASELINE, and cheaper: it lets the")
say("   subject stand closer for the same full-body framing, and the requirement falls as Z^2.")

io.open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\nwrote %s" % OUT)
