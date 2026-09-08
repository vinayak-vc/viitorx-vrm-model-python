#!/usr/bin/env python3
"""P1-2 A/B comparator — FIFO vs latest-frame, on the same human motion.

Reports freshness, RGB/depth pairing, throughput, and the P1-1 tracking regression.

IMPORTANT — why jitter is reported as VELOCITY, not per-frame displacement:
the latest-frame policy deliberately SKIPS stale frames, so the interval between two
processed frames is longer. Raw per-frame displacement therefore rises even when the
signal is not one bit noisier. Comparing per-frame displacement between the two policies
is not apples-to-apples. Dividing by the real elapsed time (m/s) makes them comparable.
"""
import argparse
import json
import math
import os
import sys

JOINTS = [("L-shoulder", "sh", 0), ("R-shoulder", "sh", 1),
          ("L-elbow", "el", 0), ("R-elbow", "el", 1),
          ("L-wrist", "wr", 0), ("R-wrist", "wr", 1),
          ("L-hip", "hip", 0), ("R-hip", "hip", 1),
          ("L-knee", "kn", 0), ("R-knee", "kn", 1),
          ("L-ankle", "an", 0), ("R-ankle", "an", 1)]


def load(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def stats(v):
    if not v:
        return None
    s = sorted(v)
    n = len(s)

    def q(p):
        return s[min(n - 1, int(p * n))]
    return dict(n=n, median=s[n // 2], p95=q(0.95), p99=q(0.99), max=s[-1])


def nz(p):
    return any(abs(c) > 1e-9 for c in p)


def d3(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def speed_series(rows, key, idx):
    """metres per SECOND between consecutive processed frames (policy-comparable)."""
    out = []
    prev = None
    for r in rows:
        if key not in r or "t" not in r:
            continue
        p = r[key][idx]
        if not nz(p):
            prev = None
            continue
        if prev is not None:
            dt = r["t"] - prev[1]
            if 0.0 < dt < 0.5:
                out.append(d3(prev[0], p) / dt)
        prev = (p, r["t"])
    return out


def line(rows, key):
    st = stats([r[key] for r in rows if key in r and r[key] >= 0])
    if not st:
        return "   (no data)"
    return "%8.2f %8.2f %8.2f %8.2f  (n=%d)" % (st["median"], st["p95"], st["p99"], st["max"], st["n"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fifo", default="p12h_fifo")
    ap.add_argument("--latest", default="p12h_latest")
    a = ap.parse_args()

    A = load(os.path.join(a.fifo, "sender_log.jsonl"))
    B = load(os.path.join(a.latest, "sender_log.jsonl"))
    if not A or not B:
        print("need sender_log.jsonl in BOTH %s and %s" % (a.fifo, a.latest))
        return 2

    def tracked(rows):
        return sum(1 for r in rows if r.get("cov", 0) >= 8)

    print("=" * 92)
    print("P1-2 A/B — FIFO (%s)  vs  LATEST-FRAME (%s)" % (a.fifo, a.latest))
    print("=" * 92)
    for tag, rows in [("FIFO", A), ("LATEST", B)]:
        dur = (rows[-1]["t"] - rows[0]["t"]) if len(rows) > 1 else 0.0
        print("  %-7s frames=%-5d duration=%5.1fs  fps=%5.2f  tracked=%d (%.0f%%)  staleDropped=%d"
              % (tag, len(rows), dur, len(rows) / max(1e-6, dur), tracked(rows),
                 100.0 * tracked(rows) / max(1, len(rows)),
                 sum(r.get("staleDropped", 0) for r in rows)))
    if tracked(A) == 0 or tracked(B) == 0:
        print("\n  *** one or both runs had NO SUBJECT — the jitter comparison below is not valid ***")

    print("\n-- FRESHNESS / LATENCY (ms) --")
    print("  %-22s %8s %8s %8s %8s" % ("metric / policy", "median", "p95", "p99", "max"))
    for key, lbl in [("frameAgeMs", "FRAME AGE"), ("capToPoseMs", "compute host->pose"),
                     ("capToSendMs", "host->UDP"), ("rgbDepthSyncMs", "RGB/depth sync"),
                     ("trackerMs", "P1-1 tracker cost")]:
        for tag, rows in [("FIFO", A), ("LATEST", B)]:
            print("  %-22s %s" % ("%s %s" % (lbl, tag), line(rows, key)))
        print()

    # end-to-end
    print("-- CAMERA -> UDP end-to-end (ms) --")
    for tag, rows in [("FIFO", A), ("LATEST", B)]:
        e = [r["frameAgeMs"] + r["capToSendMs"] for r in rows
             if r.get("frameAgeMs", -1) >= 0 and "capToSendMs" in r]
        st = stats(e)
        if st:
            print("  %-7s median %8.2f  p95 %8.2f  p99 %8.2f  max %8.2f" %
                  (tag, st["median"], st["p95"], st["p99"], st["max"]))

    print("\n-- P1-1 TRACKING REGRESSION — joint SPEED (m/s), policy-comparable --")
    print("  (per-frame displacement is NOT comparable: latest-frame skips frames)")
    print("  %-11s %-27s %-27s %s" % ("joint", "FIFO med/p95/p99/max", "LATEST med/p95/p99/max", "median"))
    worse = 0
    for name, key, idx in JOINTS:
        sa = stats(speed_series(A, key, idx))
        sb = stats(speed_series(B, key, idx))
        if not sa or not sb:
            print("  %-11s (no data)" % name)
            continue
        ch = 100.0 * (sb["median"] - sa["median"]) / max(1e-9, sa["median"])
        if ch > 15:
            worse += 1
        print("  %-11s %6.3f %6.3f %6.3f %6.3f   %6.3f %6.3f %6.3f %6.3f  %+7.1f%%"
              % (name, sa["median"], sa["p95"], sa["p99"], sa["max"],
                 sb["median"], sb["p95"], sb["p99"], sb["max"], ch))
    print("\n  joints with >15%% median speed increase (possible added jitter): %d / 12" % worse)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
