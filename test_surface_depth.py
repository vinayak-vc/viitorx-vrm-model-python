#!/usr/bin/env python3
"""Deterministic tests for the F-08 surface-aware depth sampler (oak_depth.sample_depth_surface).

Cases A-F from the F-08 implementation brief, plus compatibility and safety checks. No camera and
no DepthAI device required: `oak_depth` imports depthai at module load, so these tests stub it if
it is unavailable, then exercise pure-numpy code paths.

    python test_surface_depth.py
"""
import sys
import types

import numpy as np

try:
    import depthai  # noqa: F401
except Exception:                                    # pragma: no cover - CI without the SDK
    sys.modules["depthai"] = types.ModuleType("depthai")

import oak_depth as D

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  PASS %-58s %s" % (name, detail))
    else:
        FAIL += 1
        print("  FAIL %-58s %s" % (name, detail))


def win(values, k=5):
    """Build a k*k uint16 depth image; `values` is a flat list of k*k millimetre readings."""
    return np.array(values, dtype=np.uint16).reshape(k, k)


def uniform(mm, k=5):
    return win([mm] * (k * k), k)


def main():
    print("=" * 96)
    print(" F-08 SURFACE-AWARE DEPTH SAMPLER -- deterministic tests")
    print("=" * 96)
    c = k = 2  # centre index of a 5x5 window

    # ---- A. single clean surface -----------------------------------------------------
    a = uniform(1000)
    a[0, 0] = 1001
    a[4, 4] = 1001
    z, q, dg = D.sample_depth_surface(a, c, c)
    check("A single clean surface: depth ~= 1000", abs(z - 1000.0) < 2.0, "z=%.1f" % z)
    check("A single clean surface: one cluster", dg["clusterCount"] == 1,
          "clusters=%d" % dg["clusterCount"])
    check("A single clean surface: high quality", q > 0.9, "q=%.3f" % q)

    # ---- B. two distinct surfaces ----------------------------------------------------
    # Body at 1000 mm occupies the centre; background at 2500 mm fills the rest.
    b = np.full((5, 5), 2500, dtype=np.uint16)
    b[1:4, 1:4] = 1000
    z, q, dg = D.sample_depth_surface(b, c, c)
    check("B two surfaces: both clusters detected", dg["clusterCount"] == 2,
          "clusters=%d" % dg["clusterCount"])
    check("B two surfaces: BODY selected (not background)", abs(z - 1000.0) < 2.0,
          "z=%.1f (body=1000, bg=2500)" % z)
    check("B two surfaces: quality below the clean case", q < 0.9, "q=%.3f" % q)
    check("B two surfaces: occupancy reported", 0.0 < dg["selectedClusterOccupancy"] < 1.0,
          "occ=%.3f" % dg["selectedClusterOccupancy"])

    # ---- B2. NOT "nearest depth wins": an object IN FRONT of the joint ---------------
    # A chair edge at 600 mm sits in a corner; the joint (centre) is on the body at 1500 mm.
    b2 = np.full((5, 5), 1500, dtype=np.uint16)
    b2[0, 0:2] = 600
    b2[1, 0] = 600
    z, q, dg = D.sample_depth_surface(b2, c, c)
    check("B2 nearer foreground does NOT hijack the sample", abs(z - 1500.0) < 2.0,
          "z=%.1f (chair=600, body=1500)" % z)

    # ---- B3. NOT "largest cluster wins": limb edge, background dominates -------------
    b3 = np.full((5, 5), 3000, dtype=np.uint16)
    b3[2, 1:4] = 1200                      # a thin limb across the centre row only
    z, q, dg = D.sample_depth_surface(b3, c, c)
    check("B3 majority background does NOT hijack the sample", abs(z - 1200.0) < 2.0,
          "z=%.1f (limb=1200 in 3px, bg=3000 in 22px)" % z)

    # ---- C. mostly background --------------------------------------------------------
    cbg = np.full((5, 5), 4000, dtype=np.uint16)
    cbg[2, 2] = 1100                       # only the centre pixel is on the person
    z, q, dg = D.sample_depth_surface(cbg, c, c)
    check("C mostly background: centre surface still chosen", abs(z - 1100.0) < 2.0,
          "z=%.1f" % z)
    check("C mostly background: LOW quality", q < 0.25, "q=%.3f" % q)
    check("C mostly background: finite, non-NaN, positive", np.isfinite(z) and z > 0,
          "z=%.1f" % z)

    # ---- D. sparse valid pixels ------------------------------------------------------
    d = np.zeros((5, 5), dtype=np.uint16)
    d[2, 2] = 1000
    d[2, 3] = 1000                          # 2 valid, below min_valid = 6
    z, q, dg = D.sample_depth_surface(d, c, c)
    check("D sparse: returns the 0.0 hole value", z == 0.0, "z=%.1f" % z)
    check("D sparse: quality 0", q == 0.0, "q=%.3f" % q)
    check("D sparse: reason reported", dg["reason"] == "sparse", "reason=%s" % dg["reason"])

    # ---- E. single noisy surface -- must NOT be rejected for small variance ----------
    rng = np.random.RandomState(7)
    e = (1500 + rng.randint(-20, 21, size=(5, 5))).astype(np.uint16)
    z, q, dg = D.sample_depth_surface(e, c, c)
    check("E noisy single surface: still usable", z > 0, "z=%.1f" % z)
    check("E noisy single surface: one cluster", dg["clusterCount"] == 1,
          "clusters=%d spread=%.0f" % (dg["clusterCount"], dg["selectedClusterSpread"]))
    check("E noisy single surface: quality NOT collapsed", q > 0.4, "q=%.3f" % q)

    # ---- F. invalid pixels mixed in --------------------------------------------------
    f = uniform(1800)
    f[0, :] = 0
    f[1, 0:2] = 0                           # 18 valid of 25
    z, q, dg = D.sample_depth_surface(f, c, c)
    check("F invalid pixels: safe usable depth", abs(z - 1800.0) < 2.0, "z=%.1f" % z)
    check("F invalid pixels: counted", dg["validPixelCount"] == 18,
          "valid=%d" % dg["validPixelCount"])
    check("F invalid pixels: quality reduced by validity ratio", q < 0.8, "q=%.3f" % q)

    # ---- G. COMPATIBILITY: single surface == the legacy sampler, exactly -------------
    same = 0
    diff = 0
    rng = np.random.RandomState(11)
    for _ in range(400):
        base = int(rng.randint(700, 3500))
        w = (base + rng.randint(-25, 26, size=(5, 5))).astype(np.uint16)
        zl = D.sample_depth_legacy_mm(w, c, c)
        zn, _q, dg = D.sample_depth_surface(w, c, c)
        if dg["clusterCount"] == 1:
            if abs(zl - zn) < 1e-6:
                same += 1
            else:
                diff += 1
    check("G single-surface output identical to the legacy sampler", diff == 0,
          "identical=%d differing=%d" % (same, diff))

    # ---- H. AMBIGUITY: invalid centre + tied clusters -> legacy value, quality 0 -----
    h = np.zeros((5, 5), dtype=np.uint16)
    h[2, 1] = 1000                          # equidistant from centre, two surfaces
    h[2, 3] = 2600
    h[1, 2] = 1000
    h[3, 2] = 2600
    h[0, 0] = 1000
    h[4, 4] = 2600
    h[0, 4] = 1000
    h[4, 0] = 2600
    z, q, dg = D.sample_depth_surface(h, c, c)
    legacy = D.sample_depth_legacy_mm(h, c, c)
    check("H ambiguous: no depth invented (legacy value returned)", abs(z - legacy) < 1e-6,
          "z=%.1f legacy=%.1f" % (z, legacy))
    check("H ambiguous: quality 0", q == 0.0, "q=%.3f reason=%s" % (q, dg["reason"]))

    # ---- I. SAFETY: never NaN / never negative, across random windows ---------------
    bad = 0
    rng = np.random.RandomState(23)
    for _ in range(3000):
        w = rng.randint(0, 5000, size=(5, 5)).astype(np.uint16)
        z, q, dg = D.sample_depth_surface(w, c, c)
        if not (np.isfinite(z) and z >= 0.0 and np.isfinite(q) and 0.0 <= q <= 1.0):
            bad += 1
    check("I 3000 random windows: finite, z>=0, q in [0,1]", bad == 0, "violations=%d" % bad)

    # ---- J. edge of frame ------------------------------------------------------------
    big = np.full((40, 40), 2000, dtype=np.uint16)
    for (uu, vv) in ((0, 0), (39, 39), (0, 39), (39, 0)):
        z, q, dg = D.sample_depth_surface(big, uu, vv)
        if not (np.isfinite(z) and z > 0):
            bad += 1
    check("J frame corners handled without error", bad == 0, "")

    # ---- K. quality range is documented and respected --------------------------------
    z, q, _ = D.sample_depth_surface(uniform(2000), c, c)
    check("K perfect window -> quality 1.0", abs(q - 1.0) < 1e-6, "q=%.4f" % q)

    # ---- L. backproject keeps its 2-tuple contract ----------------------------------
    uv = np.array([[20.0, 20.0], [21.0, 21.0]], dtype=np.float32)
    intr = (500.0, 500.0, 20.0, 20.0)
    out2 = D.backproject(uv, big, 40, 40, intr)
    check("L backproject default still returns 2 values", isinstance(out2, tuple) and len(out2) == 2,
          "len=%d" % len(out2))
    out4 = D.backproject(uv, big, 40, 40, intr, with_quality=True)
    check("L backproject with_quality returns 4 values", len(out4) == 4, "len=%d" % len(out4))
    check("L backproject quality in range", float(out4[2].min()) >= 0.0 and float(out4[2].max()) <= 1.0,
          "min=%.3f max=%.3f" % (float(out4[2].min()), float(out4[2].max())))
    check("L backproject xyz/measured unchanged by with_quality",
          np.allclose(out2[0], out4[0]) and bool((out2[1] == out4[1]).all()), "")

    # ---- M. cost ---------------------------------------------------------------------
    import time
    # REALISTIC: smooth surfaces with a person-shaped foreground, i.e. what the sensor produces.
    real = np.full((400, 640), 3000, dtype=np.uint16)
    real[80:380, 240:420] = 1800                     # torso/limbs slab
    real += rng.randint(-15, 16, size=real.shape).astype(np.uint16)
    pts = [(float(rng.randint(240, 420)), float(rng.randint(80, 380))) for _ in range(12)]
    # PATHOLOGICAL: uniform random depth -- every window is multi-cluster. Never occurs in practice;
    # kept as an upper bound only.
    patho = rng.randint(500, 4000, size=(400, 640)).astype(np.uint16)
    ppts = [(float(rng.randint(10, 630)), float(rng.randint(10, 390))) for _ in range(12)]

    def bench(frm, points, n=300):
        t0 = time.perf_counter()
        for _ in range(n):
            for (uu, vv) in points:
                D.sample_depth_surface(frm, uu, vv)
        return (time.perf_counter() - t0) / n * 1000.0

    real_ms = bench(real, pts)
    patho_ms = bench(patho, ppts)
    check("M cost, realistic frame, 12 joints < 0.5 ms/frame", real_ms < 0.5,
          "%.3f ms/frame (pathological upper bound %.3f ms)" % (real_ms, patho_ms))

    print("-" * 96)
    print(" %d/%d assertions passed" % (PASS, PASS + FAIL))
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
