#!/usr/bin/env python3
"""F-08 old-vs-new depth sampler A/B, run OFFLINE on identical inputs.

The F-08 audit capture stored the raw 11x11 depth crop around every body keypoint. Running both
samplers over those same crops gives a perfectly controlled A/B -- identical pixels, no human
variation, no second capture session -- which a pair of live runs could never provide.

The production window is the central 5x5 of each stored crop.

    python compare_surface_depth.py --dir pipeline_logs_f08
"""
import argparse
import io
import json
import math
import os
import sys
import types

import numpy as np

try:
    import depthai  # noqa: F401
except Exception:
    sys.modules["depthai"] = types.ModuleType("depthai")

import oak_depth as D

JOINTS = ("L-shoulder", "R-shoulder", "L-elbow", "R-elbow", "L-wrist", "R-wrist",
          "L-hip", "R-hip", "L-knee", "R-knee", "L-ankle", "R-ankle")

MIXED_SPREAD_MM = 150.0      # the audit's definition of a contaminated window


def pct(v, p):
    if not v:
        return 0.0
    s = sorted(v)
    return s[int(round((len(s) - 1) * p / 100.0))]


def sd(v):
    if len(v) < 2:
        return 0.0
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="pipeline_logs_f08")
    a = ap.parse_args()

    path = os.path.join(a.dir, "audit_log.jsonl")
    rows = []
    for ln in io.open(path, encoding="utf-8"):
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if "j" in r:
            rows.append(r)

    blocks = {}
    bp = os.path.join(a.dir, "blocks.json")
    if os.path.exists(bp):
        meta = json.load(io.open(bp, encoding="utf-8"))
        for b in meta.get("blocks", []):
            blocks[b["block"]] = (b["tStart"], b["tEnd"], b["title"])

    print("=" * 108)
    print(" F-08 SAMPLER A/B -- identical raw crops, legacy vs surface-aware")
    print("=" * 108)

    # ---------- compatibility: how much does the emitted depth actually move? ----------
    print()
    print(" 1) DEPTH DELTA old -> new  (0 mm = byte-identical output)")
    print("   %-11s %7s | %8s %8s | %8s %8s %8s %8s | %s"
          % ("joint", "n", "identical", "changed", "p50", "p95", "p99", "max", "changed-only p50"))
    tot_same = tot_diff = 0
    all_delta = []
    for j in JOINTS:
        deltas, changed = [], []
        for r in rows:
            e = r["j"].get(j)
            if not e or "crop" not in e:
                continue
            crop = np.array(e["crop"], dtype=np.uint16)
            n = crop.shape[0]
            c = n // 2
            w5 = crop[c - 2:c + 3, c - 2:c + 3]
            if w5.shape != (5, 5):
                continue
            zo = D.sample_depth_legacy_mm(w5, 2, 2)
            zn, q, dg = D.sample_depth_surface(w5, 2, 2)
            if zo <= 0 and zn <= 0:
                continue
            d = abs(zn - zo)
            deltas.append(d)
            if d > 1e-6:
                changed.append(d)
        if not deltas:
            continue
        same = len(deltas) - len(changed)
        tot_same += same
        tot_diff += len(changed)
        all_delta += deltas
        print("   %-11s %7d | %7.2f%% %7.2f%% | %8.1f %8.1f %8.1f %8.1f | %8.1f"
              % (j, len(deltas), 100.0 * same / len(deltas), 100.0 * len(changed) / len(deltas),
                 pct(deltas, 50), pct(deltas, 95), pct(deltas, 99), max(deltas),
                 pct(changed, 50) if changed else 0.0))
    n_all = tot_same + tot_diff
    if n_all:
        print("   %-11s %7d | %7.2f%% %7.2f%% | %8.1f %8.1f %8.1f %8.1f |"
              % ("ALL", n_all, 100.0 * tot_same / n_all, 100.0 * tot_diff / n_all,
                 pct(all_delta, 50), pct(all_delta, 95), pct(all_delta, 99), max(all_delta)))

    # ---------- does depthQuality separate clean from MIXED? ---------------------------
    print()
    print(" 2) QUALITY SEPARATION -- does depthQuality identify the contaminated windows?")
    print("   %-11s | %-24s | %-24s | %s"
          % ("joint", "CLEAN window quality", "MIXED window quality", "separation"))
    print("   %-11s | %7s %7s %7s | %7s %7s %7s | %s"
          % ("", "n", "mean", "p05", "n", "mean", "p95", "mean ratio"))
    gc, gm = [], []
    for j in JOINTS:
        cq, mq = [], []
        for r in rows:
            e = r["j"].get(j)
            if not e or "crop" not in e:
                continue
            crop = np.array(e["crop"], dtype=np.uint16)
            n = crop.shape[0]
            c = n // 2
            w5 = crop[c - 2:c + 3, c - 2:c + 3]
            if w5.shape != (5, 5):
                continue
            zn, q, dg = D.sample_depth_surface(w5, 2, 2)
            if zn <= 0:
                continue
            (mq if dg["depthWindowSpread"] > MIXED_SPREAD_MM else cq).append(q)
        if not (cq and mq):
            continue
        gc += cq
        gm += mq
        mc, mm = sum(cq) / len(cq), sum(mq) / len(mq)
        print("   %-11s | %7d %7.3f %7.3f | %7d %7.3f %7.3f | %.2fx"
              % (j, len(cq), mc, pct(cq, 5), len(mq), mm, pct(mq, 95), (mc / mm) if mm else 0.0))
    if gc and gm:
        mc, mm = sum(gc) / len(gc), sum(gm) / len(gm)
        print("   %-11s | %7d %7.3f %7.3f | %7d %7.3f %7.3f | %.2fx"
              % ("ALL", len(gc), mc, pct(gc, 5), len(gm), mm, pct(gm, 95), (mc / mm) if mm else 0.0))
        # A single-threshold separability check, reported not applied.
        best_t, best_acc = 0.0, 0.0
        for t in [i / 40.0 for i in range(41)]:
            acc = (sum(1 for q in gc if q >= t) + sum(1 for q in gm if q < t)) / float(len(gc) + len(gm))
            if acc > best_acc:
                best_acc, best_t = acc, t
        print("   -> best single threshold %.3f separates clean/mixed with %.1f%% accuracy"
              % (best_t, 100.0 * best_acc))
        print("      (reported as evidence only -- nothing in the pipeline gates on this)")

    # ---------- per-block, incl. the hand-behind-torso negative test -------------------
    print()
    print(" 3) PER BLOCK -- depth stability old vs new, and quality")
    print("   %-26s | %-21s | %-21s | %s"
          % ("block", "legacy |dz| p95/p99", "surface |dz| p95/p99", "quality mean"))
    for key in ("1", "2", "3", "4", "5", "6", "10"):
        if key not in blocks:
            continue
        t0, t1, title = blocks[key]
        lo, ln_, qs = [], [], []
        # Crops are stored every Nth frame, so iterate ONLY crop-bearing rows: resetting the
        # previous value on the intervening rows leaves no two samples ever adjacent.
        crop_rows = [r for r in rows if t0 <= r.get("t", 0) <= t1]
        for j in JOINTS:
            po = pn = None
            for r in crop_rows:
                e = r["j"].get(j)
                if not e or "crop" not in e:
                    continue
                crop = np.array(e["crop"], dtype=np.uint16)
                n = crop.shape[0]
                c = n // 2
                w5 = crop[c - 2:c + 3, c - 2:c + 3]
                if w5.shape != (5, 5):
                    continue
                zo = D.sample_depth_legacy_mm(w5, 2, 2)
                zn, q, dg = D.sample_depth_surface(w5, 2, 2)
                if zo > 0:
                    if po is not None:
                        lo.append(abs(zo - po))
                    po = zo
                if zn > 0:
                    if pn is not None:
                        ln_.append(abs(zn - pn))
                    pn = zn
                    qs.append(q)
        if not lo:
            continue
        print("   %-26s | %9.0f %9.0f | %9.0f %9.0f | %.3f"
              % (title[:26], pct(lo, 95), pct(lo, 99), pct(ln_, 95), pct(ln_, 99),
                 (sum(qs) / len(qs)) if qs else 0.0))

    # ---------- NEGATIVE TEST: the hallucinated wrist ---------------------------------
    if "10" in blocks:
        t0, t1, _ = blocks["10"]
        print()
        print(" 4) NEGATIVE TEST -- hand behind torso: does depthQuality see the hallucination?")
        for j in ("L-wrist", "R-wrist", "L-hip"):
            qs, cf, spread = [], [], []
            for r in rows:
                if not (t0 <= r.get("t", 0) <= t1):
                    continue
                e = r["j"].get(j)
                if not e or "crop" not in e:
                    continue
                crop = np.array(e["crop"], dtype=np.uint16)
                n = crop.shape[0]
                c = n // 2
                w5 = crop[c - 2:c + 3, c - 2:c + 3]
                if w5.shape != (5, 5):
                    continue
                zn, q, dg = D.sample_depth_surface(w5, 2, 2)
                if zn <= 0:
                    continue
                qs.append(q)
                cf.append(e["c"])
                spread.append(dg["depthWindowSpread"])
            if not qs:
                continue
            print("   %-9s n=%4d  depthQuality mean=%.3f p05=%.3f  conf mean=%.3f  winSpread p50=%.0f mm"
                  % (j, len(qs), sum(qs) / len(qs), pct(qs, 5), sum(cf) / len(cf), pct(spread, 50)))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
