#!/usr/bin/env python3
"""
F-16 - SUBJECT-FREE shoulder-pair noise floor.

Isolates the SENSOR from the pose model. On a static rigid scene, sample pairs of pixels separated
by a shoulder-width pixel span and compute the apparent torso yaw the shipped formula would produce:

    yaw = atan( dz / dx ),   dx = span_px * Z / fx

A rigid scene has a FIXED true yaw per pair, so everything that moves frame-to-frame, and every
gap in the value ladder, is sensor error. This is the cleanest possible answer to "how much
apparent torso yaw does one depth step create?" without a human in the loop.

Two numbers are reported per configuration:
  raw    - apparent yaw including the scene's own surface tilt (a constant per pair)
  detilt - apparent yaw after removing each PAIR's own median over the run, i.e. pure instability

Output: oak_v4_evidence/f16/wall_probe_<tag>.txt / .json
"""
import argparse
import io
import json
import math
import os
import time

import numpy as np
import depthai as dai

import f16_configs as C

OUTDIR = os.path.join("oak_v4_evidence", "f16")
FLAT_TOL_MM = 40      # a 5x5 neighbourhood flatter than this is one surface
MIN_SPAN = 20
MAX_SPAN = 220

BANDS = [(700, 900), (900, 1100), (1100, 1300), (1250, 1400), (1400, 1600), (1700, 2100)]


def probe(name, seconds, span_px_at_ref, ref_z, fx640):
    cfg = C.CONFIGS[name]
    rec = dict(config=name, note=cfg["note"], ok=False)
    try:
        pipe, _c, mw, mh, rw, rh = C.build(name)
    except Exception as e:
        rec["error"] = "build: %s" % e
        return rec

    # pair geometry: keep the METRIC baseline fixed at the shoulder width implied by
    # span_px_at_ref at ref_z, so every configuration measures the same physical baseline.
    width_mm = span_px_at_ref * ref_z / fx640

    samples = {b: [] for b in BANDS}          # list of (pair_id, yaw_deg)
    try:
        with dai.Device(pipe) as dev:
            qd = dev.getOutputQueue("depth", maxSize=4, blocking=False)
            fx = fx640 * (rw / 640.0)         # depth is aligned to RGB -> RGB fx applies
            t0 = time.time()
            while time.time() - t0 < 1.0:
                qd.tryGet()
            t0 = time.time()
            nf = 0
            while time.time() - t0 < seconds:
                d = qd.get().getFrame().astype(np.int32)
                nf += 1
                h, w = d.shape

                # ---- FLATNESS MASK ------------------------------------------------------------
                # A pair straddling a depth edge is scene geometry, not sensor error. Keep only
                # endpoints whose own 5x5 neighbourhood is flat and fully valid, so what survives
                # is a measurement of one rigid surface.
                big = np.where(d > 0, d, 10 ** 6)
                lo = big.copy()
                hi = np.where(d > 0, d, -10 ** 6)
                okc = (d > 0)
                for dy in (-2, -1, 0, 1, 2):
                    for dx in (-2, -1, 0, 1, 2):
                        if dy == 0 and dx == 0:
                            continue
                        s = np.roll(np.roll(big, dy, 0), dx, 1)
                        lo = np.minimum(lo, s)
                        s2 = np.roll(np.roll(hi, dy, 0), dx, 1)
                        hi = np.maximum(hi, s2)
                        okc &= np.roll(np.roll(d > 0, dy, 0), dx, 1)
                flat = okc & ((hi - lo) <= FLAT_TOL_MM)
                flat[:3, :] = False
                flat[-3:, :] = False
                flat[:, :3] = False
                flat[:, -3:] = False

                # ---- PAIRS, vectorised over candidate spans ------------------------------------
                # The metric baseline is constant, so the pixel span depends on the endpoint's own
                # Z. Each candidate span is applied as a whole-array shift and kept only where it
                # is the span that endpoint actually requires.
                yy_idx, xx_idx = np.mgrid[0:h, 0:w]
                zl_all = d
                with np.errstate(divide="ignore", invalid="ignore"):
                    want = np.where(zl_all > 0, np.rint(width_mm * fx / np.maximum(zl_all, 1)), 0)
                for span in range(MIN_SPAN, MAX_SPAN + 1):
                    sel = flat & (want == span)
                    sel[:, w - span:] = False
                    if not sel.any():
                        continue
                    right_flat = np.zeros_like(sel)
                    right_flat[:, :w - span] = flat[:, span:]
                    sel &= right_flat
                    if not sel.any():
                        continue
                    zr_all = np.zeros_like(d)
                    zr_all[:, :w - span] = d[:, span:]
                    zl = zl_all[sel].astype(np.float64)
                    zr = zr_all[sel].astype(np.float64)
                    ys = yy_idx[sel]
                    xs = xx_idx[sel]
                    zc = 0.5 * (zl + zr)
                    yaw = np.degrees(np.arctan2(zr - zl, width_mm))
                    pid = ys.astype(np.int64) * 10000 + xs
                    for b in BANDS:
                        m = (zc >= b[0]) & (zc < b[1])
                        if m.any():
                            samples[b].append((pid[m], yaw[m]))
            rec["frames"] = nf
        rec["ok"] = True
    except Exception as e:
        rec["error"] = "run: %s" % e
        return rec

    rec["baseline_mm"] = width_mm
    bands = {}
    for b in BANDS:
        s = samples[b]
        if not s or sum(p[0].size for p in s) < 2000:
            continue
        ids = np.concatenate([p[0] for p in s])
        yy = np.concatenate([p[1] for p in s]).astype(np.float64)
        # per-pair median removal -> pure instability
        order = np.argsort(ids, kind="stable")
        ids_s, yy_s = ids[order], yy[order]
        uniq, start = np.unique(ids_s, return_index=True)
        det = np.empty_like(yy_s)
        for i in range(uniq.size):
            a = start[i]
            bnd = start[i + 1] if i + 1 < uniq.size else yy_s.size
            det[a:bnd] = yy_s[a:bnd] - np.median(yy_s[a:bnd])
        ay, ad = np.abs(yy), np.abs(det)
        bands["%d-%d" % b] = dict(
            n=int(yy.size), pairs=int(uniq.size),
            raw_p50=float(np.median(ay)), raw_p95=float(np.percentile(ay, 95)),
            raw_max=float(ay.max()),
            det_p50=float(np.median(ad)), det_p95=float(np.percentile(ad, 95)),
            det_max=float(ad.max()),
            within5=100.0 * float((ay <= 5).mean()), within10=100.0 * float((ay <= 10).mean()),
            det_within5=100.0 * float((ad <= 5).mean()),
            yaw_uniq=int(np.unique(np.round(yy, 4)).size))
    rec["bands"] = bands
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--configs", default="baseline,sub3,sub5,mono800,mono800_sub3,best")
    ap.add_argument("--span", type=float, default=67.0, help="shoulder pixel span at --refz")
    ap.add_argument("--refz", type=float, default=1330.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    fx640 = 284.6272                               # CAM_A @ 640x400, this device's EEPROM
    names = [n.strip() for n in a.configs.split(",") if n.strip()]
    out = []

    def say(s=""):
        print(s)
        out.append(s)

    say("=" * 110)
    say("F-16 SUBJECT-FREE SHOULDER-PAIR PROBE")
    say("=" * 110)
    say("Pairs of pixels on the STATIC scene, separated by a constant METRIC baseline of")
    say("%.1f mm (= %.0f px at %.0f mm). yaw = atan(dz / baseline) -- the shipped formula."
        % (a.span * a.refz / fx640, a.span, a.refz))
    say("raw    = apparent yaw incl. the scene's own tilt (constant per pair)")
    say("detilt = after removing each pair's own median -> pure sensor instability")

    results = []
    for n in names:
        say()
        say("-" * 110)
        say("[%s] %s" % (n, C.CONFIGS[n]["note"]))
        r = probe(n, a.seconds, a.span, a.refz, fx640)
        results.append(r)
        if not r["ok"]:
            say("   FAILED: %s" % r.get("error"))
            continue
        if not r["bands"]:
            say("   (no band had enough pairs)")
            continue
        say("   %-12s %8s %8s %9s %9s %9s %9s %9s %8s" %
            ("band", "pairs", "n", "raw_p50", "raw_p95", "det_p50", "det_p95", "<=5deg%", "uniq"))
        for k in sorted(r["bands"], key=lambda s: int(s.split("-")[0])):
            s = r["bands"][k]
            say("   %-12s %8d %8d %9.2f %9.2f %9.3f %9.3f %9.1f %8d" %
                (k, s["pairs"], s["n"], s["raw_p50"], s["raw_p95"], s["det_p50"],
                 s["det_p95"], s["within5"], s["yaw_uniq"]))

    say()
    say("=" * 110)
    say("HEADLINE - apparent torso yaw on a RIGID scene, 1250-1400 mm band")
    say("=" * 110)
    say("%-18s %10s %10s %10s %10s %10s" %
        ("config", "raw_p50", "raw_p95", "det_p50", "det_p95", "<=5deg%"))
    for r in results:
        if not r["ok"]:
            say("%-18s FAILED" % r["config"])
            continue
        s = r["bands"].get("1250-1400")
        if not s:
            say("%-18s (band not populated)" % r["config"])
            continue
        say("%-18s %10.2f %10.2f %10.3f %10.3f %10.1f" %
            (r["config"], s["raw_p50"], s["raw_p95"], s["det_p50"], s["det_p95"], s["within5"]))

    os.makedirs(OUTDIR, exist_ok=True)
    tag = a.tag or time.strftime("%H%M%S")
    base = os.path.join(OUTDIR, "wall_probe_%s" % tag)
    n = 0
    while os.path.exists(base + ".txt"):
        n += 1
        base = os.path.join(OUTDIR, "wall_probe_%s_%d" % (tag, n))
    io.open(base + ".txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
    io.open(base + ".json", "w", encoding="utf-8").write(json.dumps(results, indent=1))
    print("\nwrote %s.txt / .json" % base)


if __name__ == "__main__":
    main()
