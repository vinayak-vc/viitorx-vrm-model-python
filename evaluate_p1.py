#!/usr/bin/env python3
"""P1-1 EVALUATION — replay + adversarial, against the REAL captured human motion.

Source data: today's live OAK-D capture (recv_log.jsonl + blocks.json), which contains
real dancing, fast arms, fast legs, occlusion and side-on blocks with real depth. The
original video.webm the earlier replay used no longer exists, and real sensor data with
real depth is stronger evidence than a re-inferred RGB video anyway.

Two questions, both evidence-based:

  1. REPLAY / FALSE REJECTION - does P1 suppress LEGITIMATE motion?
     Run every block through the tracker and compare per-joint displacement statistics
     P0 vs P0+P1, plus how many frames P1 spent not-TRACKED during clean motion.

  2. ADVERSARIAL - does P1 catch confident-but-wrong data?
     Inject faults A-F into the real dancing trajectory and measure (a) whether they were
     flagged, and (b) recovery error against the UN-injected real trajectory (ground truth).

Note: recv_log carries per-LIMB confidence (min over the limb's 3 joints), not per-joint,
so each joint is fed its limb's confidence. Stated rather than hidden; it is the same
signal the Unity gate sees.
"""
import argparse
import json
import math
import os
import sys

from joint_tracker import SkeletonTracker, TrackerConfig, TrackingState

# recv_log key/index -> WholeBody index used by the tracker
MAP = [
    ("sh", 0, 5, "L-shoulder", "lArm"), ("sh", 1, 6, "R-shoulder", "rArm"),
    ("el", 0, 7, "L-elbow", "lArm"), ("el", 1, 8, "R-elbow", "rArm"),
    ("wr", 0, 9, "L-wrist", "lArm"), ("wr", 1, 10, "R-wrist", "rArm"),
    ("hip", 0, 11, "L-hip", "lLeg"), ("hip", 1, 12, "R-hip", "rLeg"),
    ("kn", 0, 13, "L-knee", "lLeg"), ("kn", 1, 14, "R-knee", "rLeg"),
    ("an", 0, 15, "L-ankle", "lLeg"), ("an", 1, 16, "R-ankle", "rLeg"),
]
WB2NAME = dict((m[2], m[3]) for m in MAP)


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


def d3(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def nz(p):
    return any(abs(c) > 1e-9 for c in p)


def stats(v):
    if not v:
        return None
    s = sorted(v)
    n = len(s)

    def q(p):
        return s[min(n - 1, int(p * n))]
    return dict(n=n, median=s[n // 2], p95=q(0.95), p99=q(0.99), max=s[-1])


def to_frames(rows):
    """recv_log rows -> [ {wb: (x,y,z)}, {wb: conf}, t ]"""
    frames = []
    for r in rows:
        pos, cnf = {}, {}
        for key, idx, wb, _name, limb in MAP:
            if key not in r:
                continue
            p = r[key][idx]
            c = r.get("cf", {}).get(limb, 0.0)
            if nz(p):
                pos[wb] = (p[0], p[1], p[2])
                cnf[wb] = c
            else:
                cnf[wb] = 0.0
        frames.append((pos, cnf, r.get("tRecv")))
    return frames


def run_tracker(frames, cfg=None):
    """Returns per-wb list of output positions (None where not usable) + the tracker."""
    sk = SkeletonTracker(cfg=cfg)
    out = dict((m[2], []) for m in MAP)
    states = dict((m[2], []) for m in MAP)
    for pos, cnf, t in frames:
        res = sk.update(pos, cnf, t, collect_events=True)
        for wb in out:
            x, y, z, c, st, usable = res[wb]
            out[wb].append((x, y, z) if usable else None)
            states[wb].append(st)
    return out, states, sk


def disp_of(seq):
    v = []
    prev = None
    for p in seq:
        if p is None:
            prev = None
            continue
        if prev is not None:
            v.append(d3(prev, p))
        prev = p
    return v


def disp_raw(frames, wb):
    v = []
    prev = None
    for pos, _c, _t in frames:
        p = pos.get(wb)
        if p is None:
            prev = None
            continue
        if prev is not None:
            v.append(d3(prev, p))
        prev = p
    return v


# ------------------------------------------------------------------ replay
def replay(recv, blocks):
    print("=" * 96)
    print("1. REPLAY / FALSE-REJECTION  (real captured human motion)")
    print("=" * 96)
    grand = {"p0": {}, "p1": {}}
    for b in blocks:
        seg = [r for r in recv if b["tStart"] <= r.get("tRecv", 0) <= b["tEnd"]]
        if len(seg) < 30:
            continue
        frames = to_frames(seg)
        out, states, sk = run_tracker(frames)
        n = len(frames)
        nt = sum(1 for wb in out for s in states[wb] if s != TrackingState.TRACKED)
        tot = n * len(out)
        print("\n-- BLOCK %s %s  (%d frames) --" % (b["block"], b["title"], n))
        print("   frames not TRACKED: %d / %d joint-frames (%.2f%%)" % (nt, tot, 100.0 * nt / max(1, tot)))
        print("   %-11s %27s   %27s" % ("", "P0 (median/p95/p99/max)", "P0+P1 (median/p95/p99/max)"))
        for _k, _i, wb, name, _l in MAP:
            a = stats(disp_raw(frames, wb))
            c = stats(disp_of(out[wb]))
            if not a or not c:
                continue
            grand["p0"].setdefault(name, []).extend(disp_raw(frames, wb))
            grand["p1"].setdefault(name, []).extend(disp_of(out[wb]))
            print("   %-11s %7.4f %7.4f %7.4f %7.4f   %7.4f %7.4f %7.4f %7.4f"
                  % (name, a["median"], a["p95"], a["p99"], a["max"],
                     c["median"], c["p95"], c["p99"], c["max"]))
    print("\n" + "-" * 96)
    print("ALL BLOCKS COMBINED")
    print("   %-11s %27s   %27s   %s" % ("", "P0", "P0+P1", "max change"))
    for _k, _i, wb, name, _l in MAP:
        a = stats(grand["p0"].get(name))
        c = stats(grand["p1"].get(name))
        if not a or not c:
            continue
        ch = 100.0 * (c["max"] - a["max"]) / max(1e-9, a["max"])
        print("   %-11s %7.4f %7.4f %7.4f %7.4f   %7.4f %7.4f %7.4f %7.4f   %+7.1f%%"
              % (name, a["median"], a["p95"], a["p99"], a["max"],
                 c["median"], c["p95"], c["p99"], c["max"], ch))


# ------------------------------------------------------------------ adversarial
def adversarial(recv, blocks):
    print("\n" + "=" * 96)
    print("2. ADVERSARIAL  (faults injected into the REAL dancing trajectory)")
    print("=" * 96)
    dance = None
    for b in blocks:
        if b["block"].startswith("J"):
            dance = b
    if dance is None:
        dance = blocks[-1]
    seg = [r for r in recv if dance["tStart"] <= r.get("tRecv", 0) <= dance["tEnd"]]
    base = to_frames(seg)
    n = len(base)
    if n < 120:
        print("  dancing block too short (%d frames)" % n)
        return
    print("  base: block %s, %d frames of real motion\n" % (dance["block"], n))

    L_WRIST, L_KNEE, L_ANKLE = 9, 13, 15
    START = n // 3

    def clone(frames):
        return [(dict(p), dict(c), t) for p, c, t in frames]

    cases = []

    # A. high-confidence 0.8 m wrist error, single frame
    f = clone(base)
    p, c, t = f[START]
    if L_WRIST in p:
        q = p[L_WRIST]
        p[L_WRIST] = (q[0] + 0.8, q[1], q[2])
        c[L_WRIST] = 0.95
    cases.append(("A  hi-conf 0.8m wrist error (1 frame)", f, L_WRIST, [START]))

    # B. high-confidence FROZEN wrist for 20 frames  <- the real observed failure
    f = clone(base)
    frozen = base[START][0].get(L_WRIST)
    rng = list(range(START, START + 20))
    if frozen:
        for i in rng:
            f[i][0][L_WRIST] = frozen
            f[i][1][L_WRIST] = 0.63          # exactly the live-observed confidence
    cases.append(("B  hi-conf FROZEN wrist, 20 frames", f, L_WRIST, rng))

    # C. 5-frame wrist disappearance
    f = clone(base)
    rng = list(range(START, START + 5))
    for i in rng:
        f[i][0].pop(L_WRIST, None)
        f[i][1][L_WRIST] = 0.0
    cases.append(("C  5-frame wrist disappearance", f, L_WRIST, rng))

    # D. 10-frame knee disappearance
    f = clone(base)
    rng = list(range(START, START + 10))
    for i in rng:
        f[i][0].pop(L_KNEE, None)
        f[i][1][L_KNEE] = 0.0
    cases.append(("D  10-frame knee disappearance", f, L_KNEE, rng))

    # E. high-confidence knee teleport (sustained 6 frames)
    f = clone(base)
    rng = list(range(START, START + 6))
    for i in rng:
        q = f[i][0].get(L_KNEE)
        if q:
            f[i][0][L_KNEE] = (q[0] - 0.55, q[1] + 0.35, q[2])
            f[i][1][L_KNEE] = 0.92
    cases.append(("E  hi-conf knee teleport (6 frames)", f, L_KNEE, rng))

    # F. one-frame ankle spike
    f = clone(base)
    p, c, t = f[START]
    if L_ANKLE in p:
        q = p[L_ANKLE]
        p[L_ANKLE] = (q[0] + 0.45, q[1] - 0.30, q[2] + 0.20)
        c[L_ANKLE] = 0.88
    cases.append(("F  1-frame ankle spike", f, L_ANKLE, [START]))

    clean_out, _clean_st, _sk = run_tracker(base)

    print("  %-38s %8s %9s %11s %11s" %
          ("case", "flagged", "state", "peak err", "resid err"))
    print("  " + "-" * 82)
    results = []
    for label, frames, wb, rng in cases:
        out, states, sk = run_tracker(frames)
        flagged = sum(1 for e in sk.events
                      if e["joint"] == WB2NAME[wb].lower().replace("-", "_")
                      or e["joint"] == _wbname(wb))
        # error against the UN-injected real trajectory = ground truth
        errs = []
        for i in rng:
            a = out[wb][i] if i < len(out[wb]) else None
            g = clean_out[wb][i] if i < len(clean_out[wb]) else None
            if a and g:
                errs.append(d3(a, g))
        after = []
        for i in range(rng[-1] + 1, min(len(out[wb]), rng[-1] + 12)):
            a, g = out[wb][i], clean_out[wb][i]
            if a and g:
                after.append(d3(a, g))
        st_seen = sorted({TrackingState.name(states[wb][i]) for i in rng if i < len(states[wb])})
        peak = max(errs) if errs else float("nan")
        resid = max(after) if after else float("nan")
        detected = flagged > 0 or any(s != "TRACKED" for s in st_seen)
        results.append((label, detected, peak, resid))
        print("  %-38s %8s %9s %9.4fm %9.4fm  %s"
              % (label, "YES" if detected else "NO", ",".join(st_seen)[:9],
                 peak, resid, "" if detected else "<-- MISSED"))
    print()
    ok = sum(1 for _l, d_, _p, _r in results if d_)
    print("  detected %d / %d adversarial cases" % (ok, len(results)))
    return results


def _wbname(wb):
    from joint_tracker import WB_NAMES
    return WB_NAMES.get(wb, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="pipeline_logs")
    a = ap.parse_args()
    recv = load(os.path.join(a.dir, "recv_log.jsonl"))
    bpath = os.path.join(a.dir, "blocks.json")
    if not recv or not os.path.exists(bpath):
        print("need recv_log.jsonl + blocks.json in %s" % a.dir)
        return 2
    blocks = json.load(open(bpath))
    replay(recv, blocks)
    adversarial(recv, blocks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
