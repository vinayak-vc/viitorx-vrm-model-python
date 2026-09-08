#!/usr/bin/env python3
"""F-08 AUDIT -- offline analysis of the ALREADY-ARCHIVED captures.

Answers the parts of the F-08 brief that existing logs can answer, with no camera and no
production change:

  Part 5  RGB/depth timestamp relationship
  Part 6  joint-specific behaviour (the subset the logs support)
  Part 8  temporal error characterisation
  Part 11 where the instability originates: 2D projection vs depth

The decisive decomposition (Part 11 / root cause): landmarks are logged as
`xyz_cam - mid_hip`, so the unknown hip offset CANCELS in an endpoint difference. That makes
per-bone dx, dy, dz exact, and lets bone-length variation be split into

    L_full    = sqrt(dx^2 + dy^2 + dz^2)   <- what P1-4 rejected on
    L_lateral = sqrt(dx^2 + dy^2)          <- the image-plane part

CAVEAT, stated rather than hidden: x = (u-cx)*z/fx, so dx and dy are themselves scaled by each
endpoint's z. L_lateral is therefore *less* depth-contaminated than L_full, not depth-free. A
large gap between the two is still strong evidence about where the variance lives; it is not a
clean orthogonal decomposition, and this script does not claim one.

    python audit_f08_offline.py
"""
import io
import json
import math
import os

CAPTURES = ("pipeline_logs_p13", "pipeline_logs_p14", "pipeline_logs_rollback")

# (name, sender_log key, index) pairs forming a bone. Index 0 = left, 1 = right.
BONES = (
    ("L-upperarm", ("sh", 0), ("el", 0)),
    ("R-upperarm", ("sh", 1), ("el", 1)),
    ("L-forearm", ("el", 0), ("wr", 0)),
    ("R-forearm", ("el", 1), ("wr", 1)),
    ("L-femur", ("hip", 0), ("kn", 0)),
    ("R-femur", ("hip", 1), ("kn", 1)),
    ("L-shin", ("kn", 0), ("an", 0)),
    ("R-shin", ("kn", 1), ("an", 1)),
)

JOINTS = (("shoulder", "sh"), ("elbow", "el"), ("wrist", "wr"),
          ("hip", "hip"), ("knee", "kn"), ("ankle", "an"))


def load(path):
    out = []
    if not os.path.exists(path):
        return out
    for ln in io.open(path, encoding="utf-8"):
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


def med(v):
    return pct(v, 50)


def _sd(v):
    if len(v) < 2:
        return 0.0
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def get(rec, key, idx):
    a = rec.get(key)
    if not a or idx >= len(a):
        return None
    p = a[idx]
    if not p or len(p) < 3:
        return None
    # An all-zero slot means the joint was dropped (below the confidence gate), not measured at 0.
    if p[0] == 0.0 and p[1] == 0.0 and p[2] == 0.0:
        return None
    return p


def reldev(vals):
    """Relative deviation of each sample from the series median -- the same statistic P1-4 used."""
    if len(vals) < 10:
        return []
    m = med(vals)
    if m <= 1e-6:
        return []
    return [abs(v - m) / m for v in vals]


def main():
    for cap in CAPTURES:
        S = load(os.path.join(cap, "sender_log.jsonl"))
        if not S:
            continue
        print("=" * 104)
        print(" %s   %d frames" % (cap, len(S)))
        print("=" * 104)

        # ---------------- Part 5: RGB/depth timing --------------------------------
        print(" PART 5 -- RGB/depth timestamp relationship")
        for k in ("rgbDepthSyncMs", "poseToDepthMs", "depthWaitMs", "camLatMs", "queueDepth"):
            v = [r[k] for r in S if k in r]
            if v:
                print("   %-15s n=%5d  min=%7.2f  p50=%7.2f  p95=%7.2f  p99=%7.2f  max=%7.2f"
                      % (k, len(v), min(v), pct(v, 50), pct(v, 95), pct(v, 99), max(v)))
        v = [r["rgbDepthSyncMs"] for r in S if "rgbDepthSyncMs" in r]
        if v:
            spread = pct(v, 99) - min(v)
            print("   -> sync spread above the floor = %.2f ms  (floor itself = %.2f ms)"
                  % (spread, min(v)))
            print("      a near-constant floor is a SYSTEMATIC offset, not jitter")

        # ---------------- Part 11 / root cause: where does the variance live? -----
        print()
        print(" PART 11 -- bone-length variation: full 3D vs image-plane only")
        print("   %-12s %8s | %-26s | %-26s | %s"
              % ("bone", "n", "L_full reldev", "L_lateral reldev", "component |d| p99 (m)"))
        print("   %-12s %8s | %8s %8s %8s | %8s %8s %8s | %6s %6s %6s"
              % ("", "", "p50", "p95", "p99", "p50", "p95", "p99", "dx", "dy", "dz"))
        agg_full, agg_lat = [], []
        for name, (ka, ia), (kb, ib) in BONES:
            Lf, Ll, DX, DY, DZ = [], [], [], [], []
            for r in S:
                a, b = get(r, ka, ia), get(r, kb, ib)
                if a is None or b is None:
                    continue
                dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
                Lf.append(math.sqrt(dx * dx + dy * dy + dz * dz))
                Ll.append(math.sqrt(dx * dx + dy * dy))
                DX.append(abs(dx))
                DY.append(abs(dy))
                DZ.append(abs(dz))
            rf, rl = reldev(Lf), reldev(Ll)
            if not rf:
                continue
            agg_full += rf
            agg_lat += rl
            print("   %-12s %8d | %8.3f %8.3f %8.3f | %8.3f %8.3f %8.3f | %6.3f %6.3f %6.3f"
                  % (name, len(Lf), pct(rf, 50), pct(rf, 95), pct(rf, 99),
                     pct(rl, 50), pct(rl, 95), pct(rl, 99),
                     pct(DX, 99), pct(DY, 99), pct(DZ, 99)))
        if agg_full and agg_lat:
            print("   %-12s %8d | %8.3f %8.3f %8.3f | %8.3f %8.3f %8.3f |"
                  % ("ALL BONES", len(agg_full), pct(agg_full, 50), pct(agg_full, 95),
                     pct(agg_full, 99), pct(agg_lat, 50), pct(agg_lat, 95), pct(agg_lat, 99)))
            print("   -> p99 ratio  full / lateral = %.2fx"
                  % (pct(agg_full, 99) / max(1e-9, pct(agg_lat, 99))))

        # ---------------- Parts 6 + 8: per-joint temporal behaviour ---------------
        print()
        print(" PARTS 6+8 -- per-joint 3D motion, dt-normalised (m/s), and depth component")
        print("   %-9s %7s | %8s %8s %8s | %8s %8s | %s"
              % ("joint", "n", "speed p50", "p95", "p99", "|vz| p95", "|vz| p99", "vz share of speed p50"))
        for jname, key in JOINTS:
            sp, vz = [], []
            for side in (0, 1):
                prev, pt = None, None
                for r in S:
                    p = get(r, key, side)
                    t = r.get("t")
                    if p is None or t is None:
                        prev, pt = None, None
                        continue
                    if prev is not None and pt is not None:
                        dt = t - pt
                        if 1e-4 < dt < 0.5:
                            d = [p[i] - prev[i] for i in range(3)]
                            s = math.sqrt(sum(x * x for x in d)) / dt
                            sp.append(s)
                            vz.append(abs(d[2]) / dt)
                    prev, pt = p, t
            if not sp:
                continue
            share = [z / s for z, s in zip(vz, sp) if s > 1e-6]
            print("   %-9s %7d | %8.3f %8.3f %8.3f | %8.3f %8.3f | %.3f"
                  % (jname, len(sp), pct(sp, 50), pct(sp, 95), pct(sp, 99),
                     pct(vz, 95), pct(vz, 99), med(share)))
        print()

        # ---------------- STATIC BLOCK: all variation is noise, by construction ----
        bpath = os.path.join(cap, "blocks.json")
        if os.path.exists(bpath):
            meta = json.load(io.open(bpath, encoding="utf-8"))
            b1 = [b for b in meta.get("blocks", []) if b["block"] == "1"]
            if b1:
                t0, t1 = b1[0]["tStart"], b1[0]["tEnd"]
                W = [r for r in S if t0 <= r.get("t", 0) <= t1]
                print(" STATIC BLOCK 1 (stand still, %d frames) -- ALL variation here is NOISE" % len(W))
                hz = [r["hipZ"] for r in W if "hipZ" in r]
                if hz:
                    m = med(hz)
                    print("   hipZ (subject distance)   median=%.3f m  sd=%.4f m  range=%.4f m"
                          % (m, _sd(hz), max(hz) - min(hz)))
                print("   %-9s %6s | %8s %8s %8s | %s"
                      % ("joint", "n", "sd(x) m", "sd(y) m", "sd(z) m", "sd(z) / mean(sd(x),sd(y))"))
                for jname, key in JOINTS:
                    for side, sl in ((0, "L"), (1, "R")):
                        xs, ys, zs = [], [], []
                        for r in W:
                            q = get(r, key, side)
                            if q is None:
                                continue
                            xs.append(q[0]); ys.append(q[1]); zs.append(q[2])
                        if len(xs) < 20:
                            continue
                        sx, sy, sz = _sd(xs), _sd(ys), _sd(zs)
                        lat = (sx + sy) / 2.0
                        print("   %-9s %6d | %8.4f %8.4f %8.4f | %.2fx"
                              % (sl + "-" + jname, len(xs), sx, sy, sz,
                                 sz / lat if lat > 1e-9 else 0.0))
                print("   %-12s %8s %8s %8s" % ("bone", "median m", "sd m", "sd/median"))
                for name, (ka, ia), (kb, ib) in BONES:
                    L = []
                    for r in W:
                        a, b = get(r, ka, ia), get(r, kb, ib)
                        if a is None or b is None:
                            continue
                        L.append(math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3))))
                    if len(L) < 20:
                        continue
                    m = med(L)
                    print("   %-12s %8.4f %8.4f %8.3f" % (name, m, _sd(L), _sd(L) / m if m else 0))
                print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
