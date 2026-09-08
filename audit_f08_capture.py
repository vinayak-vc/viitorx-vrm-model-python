#!/usr/bin/env python3
"""F-08 AUDIT -- analysis of the instrumented capture (audit_log.jsonl).

Answers the parts that needed new instrumentation:

  Part 2  confidence semantics, measured against recorded data
  Part 3  confidence vs measurement error
  Part 4  depth quality + alternative depth estimators (offline, from the raw crops)
  Part 6  joint-specific reliability
  Part 7  body-edge / depth-window neighbourhood study
  Part 9  confident-but-wrong case study (block 10, sustained hand behind torso)
  Part 10 high-confidence injected fault trace

Read-only post-hoc analysis. Nothing here runs in production.

    python audit_f08_capture.py --dir pipeline_logs_f08
"""
import argparse
import io
import json
import math
import os

JOINTS = ("L-shoulder", "R-shoulder", "L-elbow", "R-elbow", "L-wrist", "R-wrist",
          "L-hip", "R-hip", "L-knee", "R-knee", "L-ankle", "R-ankle")

# A depth window straddling a person/background edge shows a large spread between its nearest and
# farthest valid pixel. 150 mm is well beyond limb thickness at ~2 m and well below room depth.
MIXED_SPREAD_MM = 150.0


def load(p):
    out = []
    if not os.path.exists(p):
        return out
    for ln in io.open(p, encoding="utf-8"):
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
    return out


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


def percentile_of(vals, q):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * q / 100.0
    lo = int(math.floor(k))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def crop_estimators(crop, k):
    """Re-run several depth estimators OFFLINE on the raw crop. Returns {name: mm or 0}."""
    n = len(crop)
    c = n // 2
    out = {}
    for name, half in (("3x3", 1), ("5x5", 2), ("9x9", 4), ("11x11", n // 2)):
        vals = []
        for y in range(max(0, c - half), min(n, c + half + 1)):
            row = crop[y]
            for x in range(max(0, c - half), min(len(row), c + half + 1)):
                if row[x] > 0:
                    vals.append(float(row[x]))
        out[name + "_p30"] = percentile_of(vals, 30.0) if len(vals) >= 6 else 0.0
        if name == "5x5":
            out["5x5_p50"] = percentile_of(vals, 50.0) if len(vals) >= 6 else 0.0
            out["5x5_min"] = min(vals) if vals else 0.0
            out["5x5_mean"] = (sum(vals) / len(vals)) if vals else 0.0
    out["center"] = float(crop[c][c]) if crop[c][c] > 0 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="pipeline_logs_f08")
    a = ap.parse_args()

    A = load(os.path.join(a.dir, "audit_log.jsonl"))
    S = load(os.path.join(a.dir, "sender_log.jsonl"))
    bp = os.path.join(a.dir, "blocks.json")
    meta = json.load(io.open(bp, encoding="utf-8")) if os.path.exists(bp) else {}
    blocks = meta.get("blocks", [])
    print("=" * 104)
    print(" F-08 AUDIT -- %s   audit frames=%d  sender frames=%d" % (a.dir, len(A), len(S)))
    print("=" * 104)

    # ================= PART 2: confidence semantics ==========================
    print()
    print(" PART 2 -- CONFIDENCE: what the recorded values actually look like")
    print("   %-11s %7s | %7s %7s %7s %7s %7s | %8s %8s"
          % ("joint", "n", "min", "p5", "p50", "p95", "max", "<0.3 %", "depthOK%"))
    conf_by_joint = {}
    for j in JOINTS:
        cs = [r["j"][j]["c"] for r in A if j in r["j"]]
        ms = [r["j"][j]["m"] for r in A if j in r["j"]]
        conf_by_joint[j] = cs
        if not cs:
            continue
        print("   %-11s %7d | %7.3f %7.3f %7.3f %7.3f %7.3f | %7.2f%% %7.2f%%"
              % (j, len(cs), min(cs), pct(cs, 5), pct(cs, 50), pct(cs, 95), max(cs),
                 100.0 * sum(1 for c in cs if c < 0.3) / len(cs),
                 100.0 * sum(ms) / len(ms)))
    allc = [c for j in JOINTS for c in conf_by_joint.get(j, [])]
    print("   ALL: max observed = %.4f  (if this exceeds 1.0 the value is NOT a probability)"
          % (max(allc) if allc else 0.0))

    # ================= PART 3: confidence vs measurement error ===============
    print()
    print(" PART 3 -- CONFIDENCE vs MEASUREMENT ERROR")
    print("   Does confidence predict depth-window quality or 2D stability?")
    print("   %-11s | %-28s | %-28s"
          % ("joint", "depth spread (dmax-dmin) mm", "2D step |d(u,v)| px"))
    print("   %-11s | %8s %8s %8s | %8s %8s %8s"
          % ("", "conf<.5", "conf>.8", "ratio", "conf<.5", "conf>.8", "ratio"))
    for j in JOINTS:
        lo_s, hi_s, lo_d, hi_d = [], [], [], []
        prev = None
        for r in A:
            e = r["j"].get(j)
            if not e:
                prev = None
                continue
            c = e["c"]
            if "dmax" in e and "dmin" in e:
                (lo_s if c < 0.5 else hi_s if c > 0.8 else []).append(e["dmax"] - e["dmin"])
            if prev is not None:
                d = math.hypot(e["u"] - prev[0], e["v"] - prev[1])
                (lo_d if c < 0.5 else hi_d if c > 0.8 else []).append(d)
            prev = (e["u"], e["v"])
        if not (lo_s and hi_s):
            continue
        ms, hs = pct(lo_s, 50), pct(hi_s, 50)
        md, hd = pct(lo_d, 50) if lo_d else 0.0, pct(hi_d, 50) if hi_d else 0.0
        print("   %-11s | %8.1f %8.1f %8.2fx | %8.2f %8.2f %8.2fx"
              % (j, ms, hs, (ms / hs) if hs else 0.0, md, hd, (md / hd) if hd else 0.0))

    # ================= PART 4 + 7: depth quality / neighbourhood =============
    print()
    print(" PARTS 4+7 -- DEPTH WINDOW QUALITY  (production sampler = 5x5, 30th pct, min_valid=6)")
    print("   %-11s %7s | %8s %8s | %8s %8s %8s | %9s"
          % ("joint", "n", "nvalid/25", "<6 valid%", "spread p50", "p95", "p99", "MIXED %"))
    mixed_rate = {}
    for j in JOINTS:
        nv, spread, mixed, thin = [], [], 0, 0
        for r in A:
            e = r["j"].get(j)
            if not e or "nv" not in e:
                continue
            nv.append(e["nv"])
            if e["nv"] < 6:
                thin += 1
            if "dmax" in e:
                sp = e["dmax"] - e["dmin"]
                spread.append(sp)
                if sp > MIXED_SPREAD_MM:
                    mixed += 1
        if not nv:
            continue
        mixed_rate[j] = 100.0 * mixed / len(nv)
        print("   %-11s %7d | %8.1f %7.2f%% | %8.1f %8.1f %8.1f | %8.2f%%"
              % (j, len(nv), sum(nv) / float(len(nv)), 100.0 * thin / len(nv),
                 pct(spread, 50), pct(spread, 95), pct(spread, 99), mixed_rate[j]))
    print("   MIXED = the 5x5 window spans > %.0f mm, i.e. it straddles a depth discontinuity"
          % MIXED_SPREAD_MM)

    # ---- does a MIXED window actually produce a worse 3D point? -------------
    print()
    print(" PART 7 -- does a MIXED neighbourhood coincide with a bad measurement?")
    print("   %-11s | %-24s | %s" % ("joint", "frame-to-frame |dz| mm", "confidence"))
    print("   %-11s | %8s %8s %6s | %8s %8s" % ("", "clean", "mixed", "ratio", "clean", "mixed"))
    for j in JOINTS:
        cz, mz, cc, mc = [], [], [], []
        prev = None
        for r in A:
            e = r["j"].get(j)
            if not e or "p30" not in e or e["p30"] <= 0:
                prev = None
                continue
            if prev is not None:
                dz = abs(e["p30"] - prev)
                if e.get("dmax", 0) - e.get("dmin", 0) > MIXED_SPREAD_MM:
                    mz.append(dz)
                    mc.append(e["c"])
                else:
                    cz.append(dz)
                    cc.append(e["c"])
            prev = e["p30"]
        if not (cz and mz):
            continue
        c50, m50 = pct(cz, 50), pct(mz, 50)
        print("   %-11s | %8.1f %8.1f %6.2fx | %8.3f %8.3f"
              % (j, c50, m50, (m50 / c50) if c50 else 0.0,
                 pct(cc, 50) if cc else 0, pct(mc, 50) if mc else 0))

    # ================= PART 4: alternative estimators, static block ==========
    b1 = [b for b in blocks if b["block"] == "1"]
    if b1:
        t0, t1 = b1[0]["tStart"], b1[0]["tEnd"]
        W = [r for r in A if t0 <= r.get("t", 0) <= t1 and "crop" in r["j"].get("R-knee", {})]
        if len(W) > 15:
            print()
            print(" PART 4 -- alternative depth estimators on the SAME raw crops (static block, n=%d)"
                  % len(W))
            print("   temporal sd of the estimated depth, mm -- lower is a steadier estimator")
            names = ("center", "3x3_p30", "5x5_p30", "5x5_p50", "5x5_min", "5x5_mean", "9x9_p30", "11x11_p30")
            print("   %-11s %s" % ("joint", " ".join("%9s" % n for n in names)))
            for j in JOINTS:
                series = dict((n, []) for n in names)
                for r in W:
                    e = r["j"].get(j)
                    if not e or "crop" not in e:
                        continue
                    est = crop_estimators(e["crop"], len(e["crop"]))
                    for n in names:
                        if est.get(n, 0) > 0:
                            series[n].append(est[n])
                if len(series["5x5_p30"]) < 10:
                    continue
                print("   %-11s %s" % (j, " ".join("%9.1f" % sd(series[n]) for n in names)))

    # ================= PART 9: sustained occlusion ===========================
    b10 = [b for b in blocks if b["block"] == "10"]
    if b10:
        t0, t1 = b10[0]["tStart"], b10[0]["tEnd"]
        W = [r for r in A if t0 <= r.get("t", 0) <= t1]
        print()
        print(" PART 9 -- SUSTAINED HAND BEHIND TORSO (block 10, %d frames)" % len(W))
        print("   %-11s | %8s %8s %8s | %8s %8s | %s"
              % ("joint", "conf p50", "conf min", "<0.3 %", "depthOK%", "spread p50", "note"))
        for j in ("L-wrist", "R-wrist", "L-elbow", "R-elbow", "L-hip"):
            cs = [r["j"][j]["c"] for r in W if j in r["j"]]
            ms = [r["j"][j]["m"] for r in W if j in r["j"]]
            sp = [r["j"][j]["dmax"] - r["j"][j]["dmin"] for r in W
                  if j in r["j"] and "dmax" in r["j"][j]]
            if not cs:
                continue
            note = "OCCLUDED HAND" if j == "L-wrist" else ""
            print("   %-11s | %8.3f %8.3f %7.2f%% | %7.2f%% %8.1f | %s"
                  % (j, pct(cs, 50), min(cs), 100.0 * sum(1 for c in cs if c < 0.3) / len(cs),
                     100.0 * sum(ms) / len(ms), pct(sp, 50) if sp else 0.0, note))

    # ================= PART 10: injected fault ===============================
    print()
    print(" PART 10 -- HIGH-CONFIDENCE INJECTED FAULT (block 8)")
    for inj in meta.get("injections", []):
        it, spec = inj["t"], inj["spec"]
        pre = [r for r in A if it - 1.5 <= r.get("t", 0) < it]
        post = [r for r in A if it <= r.get("t", 0) <= it + 2.5]
        if not (pre and post):
            continue
        j = "R-knee"
        pc = [r["j"][j]["c"] for r in pre if j in r["j"]]
        qc = [r["j"][j]["c"] for r in post if j in r["j"]]
        pz = [r["j"][j]["p30"] for r in pre if j in r["j"] and r["j"][j].get("p30", 0) > 0]
        qz = [r["j"][j]["p30"] for r in post if j in r["j"] and r["j"][j].get("p30", 0) > 0]
        print("   %-9s conf before=%.3f after=%.3f | depth-window p30 before=%.0f after=%.0f mm"
              % (spec["mode"], pct(pc, 50) if pc else 0, pct(qc, 50) if qc else 0,
                 pct(pz, 50) if pz else 0, pct(qz, 50) if qz else 0))
        print("             -> the injection is applied to xyz_cam AFTER backproject, so neither the")
        print("                2D confidence nor the depth window can see it. Both stay unchanged.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
