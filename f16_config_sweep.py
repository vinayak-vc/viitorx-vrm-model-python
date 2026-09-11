#!/usr/bin/env python3
"""
F-16 step 2 - STATIC-SCENE stereo configuration sweep (no subject required).

Measures, for every candidate stereo configuration, the depth quantisation the device ACTUALLY
produces, per range band, and compares it against the analytic prediction Z^2*dd/(f*B).

This isolates the sensor question from the pose question: whatever the room contains, surfaces at
range Z expose the quantisation ladder at Z directly.

Usage:  python f16_config_sweep.py [--seconds 10] [--configs a,b,c] [--tag free]
Output: oak_v4_evidence/f16/config_sweep_<tag>.txt  and  .json
"""
import argparse
import io
import json
import os
import time
import traceback

import numpy as np
import depthai as dai

import f16_configs as C

BANDS = [(700, 900), (900, 1100), (1100, 1300), (1250, 1400),
         (1400, 1600), (1700, 1900), (1900, 2100), (2300, 2700)]
OUTDIR = os.path.join("oak_v4_evidence", "f16")


def unique_step_stats(vals):
    """Quantisation ladder of a pool of depth readings: gaps between adjacent DISTINCT values."""
    u = np.unique(vals)
    if u.size < 3:
        return None
    g = np.diff(u.astype(np.float64))
    g = g[g > 0]
    if g.size == 0:
        return None
    return dict(n_unique=int(u.size), n_samples=int(vals.size),
                gap_min=float(g.min()), gap_p50=float(np.median(g)),
                gap_p95=float(np.percentile(g, 95)), gap_max=float(g.max()),
                gap_mean=float(g.mean()))


def run_config(name, seconds, roi_frac=0.6):
    cfg = C.CONFIGS[name]
    rec = dict(config=name, note=cfg["note"], mono=cfg["mono"], subpixel=cfg["subpixel"],
               bits=cfg["bits"], extended=cfg["extended"], preset=cfg["preset"],
               align=cfg["align"], ok=False)
    try:
        pipe, _cfg, mw, mh, rw, rh = C.build(name)
    except Exception as e:
        rec["error"] = "build: %s" % e
        return rec

    pools = {b: [] for b in BANDS}
    cov = []
    lat = []
    t_frames = []
    try:
        t_open0 = time.time()
        with dai.Device(pipe) as dev:
            rec["boot_s"] = round(time.time() - t_open0, 2)
            qd = dev.getOutputQueue("depth", maxSize=4, blocking=False)
            t0 = time.time()
            while time.time() - t0 < 1.0:      # discard warm-up
                qd.tryGet()
            t0 = time.time()
            last = t0
            while time.time() - t0 < seconds:
                pkt = qd.get()
                now = time.time()
                t_frames.append(now - last)
                last = now
                try:
                    lat.append((dai.Clock.now() - pkt.getTimestamp()).total_seconds() * 1000.0)
                except Exception:
                    pass
                d = pkt.getFrame()
                if "depth_shape" not in rec:
                    rec["depth_shape"] = [int(d.shape[1]), int(d.shape[0])]
                h, w = d.shape
                y0 = int(h * (1 - roi_frac) / 2)
                y1 = int(h * (1 + roi_frac) / 2)
                x0 = int(w * (1 - roi_frac) / 2)
                x1 = int(w * (1 + roi_frac) / 2)
                roi = d[y0:y1, x0:x1]
                nz = roi[roi > 0]
                cov.append(nz.size / float(roi.size))
                for b in BANDS:
                    sel = nz[(nz >= b[0]) & (nz < b[1])]
                    if sel.size:
                        pools[b].append(sel)
        rec["ok"] = True
    except Exception as e:
        rec["error"] = "run: %s" % e
        rec["trace"] = traceback.format_exc()[-600:]
        return rec

    dt = np.array(t_frames[1:], dtype=np.float64)
    rec["frames"] = len(t_frames)
    rec["fps"] = float(1.0 / np.median(dt)) if dt.size else 0.0
    rec["frame_dt_p95_ms"] = float(np.percentile(dt, 95) * 1000.0) if dt.size else 0.0
    rec["latency_ms_p50"] = float(np.median(lat)) if lat else -1.0
    rec["latency_ms_p95"] = float(np.percentile(lat, 95)) if lat else -1.0
    rec["coverage_pct"] = float(np.mean(cov) * 100.0) if cov else 0.0

    mono_w = C.MONO_RES[cfg["mono"]][1]
    bits = cfg["bits"] if cfg["subpixel"] else 0
    bands = {}
    for b in BANDS:
        if not pools[b]:
            continue
        vals = np.concatenate(pools[b])
        if vals.size < 500:
            continue
        s = unique_step_stats(vals)
        if s is None:
            continue
        zc = 0.5 * (b[0] + b[1])
        s["z_centre"] = zc
        s["theory_step_mm"] = C.theoretical_step_mm(zc, mono_w, bits)
        s["ratio_obs_over_theory"] = s["gap_p50"] / s["theory_step_mm"]
        bands["%d-%d" % b] = s
    rec["bands"] = bands
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--configs", default="")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    names = [n.strip() for n in a.configs.split(",") if n.strip()] or list(C.CONFIGS)
    tag = a.tag or time.strftime("%H%M%S")
    os.makedirs(OUTDIR, exist_ok=True)
    base = os.path.join(OUTDIR, "config_sweep_%s" % tag)
    n = 0
    while os.path.exists(base + ".txt"):          # section 15: never overwrite
        n += 1
        base = os.path.join(OUTDIR, "config_sweep_%s_%d" % (tag, n))

    out = []

    def say(s=""):
        print(s)
        out.append(s)

    say("=" * 100)
    say("F-16 STATIC-SCENE STEREO CONFIGURATION SWEEP")
    say("=" * 100)
    say("depthai %s | %.0f s per configuration | central 60%% ROI" % (dai.__version__, a.seconds))
    say("Scene: whatever is in front of the camera. No subject needed - a surface at range Z")
    say("exposes the depth ladder at Z directly.")
    say("baseline = byte-for-byte replica of oak_depth.build_rgbd_pipeline().")

    results = []
    for name in names:
        say()
        say("-" * 100)
        say("[%s] %s" % (name, C.CONFIGS[name]["note"]))
        r = run_config(name, a.seconds)
        results.append(r)
        if not r["ok"]:
            say("   FAILED: %s" % r.get("error"))
            continue
        say("   depth %s  fps %.1f  latency p50 %.0f ms / p95 %.0f ms  coverage %.1f%%  boot %.1fs"
            % (r.get("depth_shape"), r["fps"], r["latency_ms_p50"], r["latency_ms_p95"],
               r["coverage_pct"], r.get("boot_s", 0)))
        if not r["bands"]:
            say("   (no range band had enough samples in this scene)")
            continue
        say("   %-12s %8s %9s %10s %10s %10s %9s" %
            ("band(mm)", "samples", "uniq", "gap_min", "gap_p50", "theory", "obs/thy"))
        for k in sorted(r["bands"], key=lambda s: int(s.split("-")[0])):
            s = r["bands"][k]
            say("   %-12s %8d %9d %10.2f %10.2f %10.2f %9.2f"
                % (k, s["n_samples"], s["n_unique"], s["gap_min"], s["gap_p50"],
                   s["theory_step_mm"], s["ratio_obs_over_theory"]))

    say()
    say("=" * 100)
    say("HEADLINE - quantisation step at the 1250-1400 mm band (production interaction range)")
    say("=" * 100)
    say("%-20s %10s %10s %10s %8s %8s %9s" %
        ("config", "gap_p50", "gap_min", "theory", "uniq", "fps", "cov%"))
    for r in results:
        if not r["ok"]:
            say("%-20s  FAILED: %s" % (r["config"], str(r.get("error"))[:50]))
            continue
        s = r["bands"].get("1250-1400")
        if s is None:
            say("%-20s  (band not populated)   fps %.1f cov %.1f%%"
                % (r["config"], r["fps"], r["coverage_pct"]))
            continue
        say("%-20s %10.2f %10.2f %10.2f %8d %8.1f %9.1f"
            % (r["config"], s["gap_p50"], s["gap_min"], s["theory_step_mm"],
               s["n_unique"], r["fps"], r["coverage_pct"]))

    io.open(base + ".txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
    io.open(base + ".json", "w", encoding="utf-8").write(json.dumps(results, indent=1))
    print("\nwrote %s.txt / .json" % base)


if __name__ == "__main__":
    main()
