#!/usr/bin/env python3
"""P1-4 evaluation — real replay + adversarial injection.

Replays a REAL captured human sequence through:
    A) P1-1 only            (the accepted P1-3 baseline)
    B) P1-1 + P1-4          (this phase)

and then injects the failure modes P1-4 exists for, measuring detection and residual error
against the UN-injected trajectory as ground truth.

LIMITATION, stated rather than hidden: `sender_log.jsonl` does not carry per-joint confidence
(it is not in the log schema), so replay assigns a nominal confidence to every emitted joint.
That makes this replay a test of the GEOMETRIC path specifically -- which is the point of
P1-4 -- but it cannot exercise confidence-driven behaviour. The confidence path is covered by
`test_joint_tracker.py` and by the live captures in the P0/P1-1 reports.

    python evaluate_p14.py --dir p12h_latest
"""
import argparse
import json
import math
import os
import sys
import time

from joint_tracker import SkeletonTracker, TrackerConfig, TrackingState
from kinematic_recovery import (KinematicRecovery, Observation, RecoveryConfig,
                                BONES, _dist)

# sender_log key/index -> WholeBody index used by the trackers
MAP = [("sh", 0, 5), ("sh", 1, 6), ("el", 0, 7), ("el", 1, 8),
       ("wr", 0, 9), ("wr", 1, 10), ("hip", 0, 11), ("hip", 1, 12),
       ("kn", 0, 13), ("kn", 1, 14), ("an", 0, 15), ("an", 1, 16)]
NAMES = {5: "L-shoulder", 6: "R-shoulder", 7: "L-elbow", 8: "R-elbow",
         9: "L-wrist", 10: "R-wrist", 11: "L-hip", 12: "R-hip",
         13: "L-knee", 14: "R-knee", 15: "L-ankle", 16: "R-ankle"}
NOMINAL_CONF = 0.85


def load_frames(path):
    frames = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            pos = {}
            for key, idx, wb in MAP:
                if key not in r:
                    continue
                p = r[key][idx]
                if any(abs(c) > 1e-9 for c in p):
                    pos[wb] = (p[0], p[1], p[2])
            if len(pos) >= 8:
                frames.append((pos, r.get("t", 0.0)))
    return frames


def stats(v):
    if not v:
        return None
    s = sorted(v)
    n = len(s)
    return dict(n=n, median=s[n // 2], p95=s[min(n - 1, int(0.95 * n))],
                p99=s[min(n - 1, int(0.99 * n))], max=s[-1])


def fmt(st, unit="m"):
    if not st:
        return "(no data)"
    return "median %.4f  p95 %.4f  p99 %.4f  max %.4f %s (n=%d)" % (
        st["median"], st["p95"], st["p99"], st["max"], unit, st["n"])


def run(frames, use_p14, rcfg=None, timed=False):
    sk = SkeletonTracker(cfg=TrackerConfig())
    kr = KinematicRecovery(rcfg or RecoveryConfig()) if use_p14 else None
    out = []
    costs = []
    for pos, t in frames:
        cnf = dict((i, NOMINAL_CONF) for i in pos)
        tr = sk.update(pos, cnf, t, collect_events=False)
        if kr is not None:
            t0 = time.perf_counter()
            res = kr.apply(tr, pos, collect_events=False)
            if timed:
                costs.append((time.perf_counter() - t0) * 1000.0)
        else:
            res = dict((k, v + (Observation.VALID,)) for k, v in tr.items())
        out.append(res)
    return out, sk, kr, costs


def state_mix(seq):
    c = {"TRACKED": 0, "WEAK": 0, "PREDICTED": 0, "LOST": 0, "RECOVERING": 0}
    n = 0
    for fr in seq:
        for idx, v in fr.items():
            c[TrackingState.name(v[4])] = c.get(TrackingState.name(v[4]), 0) + 1
            n += 1
    return dict((k, 100.0 * v / max(1, n)) for k, v in c.items())


def seg_deviation(seq, kr):
    """Relative bone-length deviation of the OUTPUT skeleton, vs the learned lengths."""
    devs = []
    for fr in seq:
        for pa, ch in BONES:
            a = fr.get(pa)
            b = fr.get(ch)
            if a is None or b is None or not a[5] or not b[5]:
                continue
            est = kr.lengths[(pa, ch)] if kr is not None else None
            if est is None or not est.ready():
                continue
            L = _dist(a[:3], b[:3])
            if est.length() > 1e-6:
                devs.append(abs(L - est.length()) / est.length())
    return devs


def displacement(seq, wb):
    v = []
    prev = None
    for fr in seq:
        e = fr.get(wb)
        if e is None or not e[5]:
            prev = None
            continue
        p = e[:3]
        if prev is not None:
            v.append(_dist(prev, p))
        prev = p
    return v


# ------------------------------------------------------------------ adversarial
def inject(frames, kind, wb, start, n):
    """Return a copy of `frames` with one failure mode injected."""
    out = [(dict(p), t) for p, t in frames]
    if kind == "missing":
        for i in range(start, min(len(out), start + n)):
            out[i][0].pop(wb, None)
    elif kind == "frozen":
        held = out[start][0].get(wb)
        if held is not None:
            for i in range(start, min(len(out), start + n)):
                out[i][0][wb] = held
    elif kind == "teleport":
        for i in range(start, min(len(out), start + n)):
            p = out[i][0].get(wb)
            if p is not None:
                out[i][0][wb] = (p[0] + 0.45, p[1] - 0.35, p[2])
    elif kind == "drift":
        # slow coherent walk to an impossible place -- the case only geometry can catch
        for i in range(start, min(len(out), start + n)):
            p = out[i][0].get(wb)
            if p is not None:
                f = (i - start + 1) / float(n)
                out[i][0][wb] = (p[0] + 0.30 * f, p[1] - 0.55 * f, p[2])
    elif kind == "baddepth":
        for i in range(start, min(len(out), start + n)):
            p = out[i][0].get(wb)
            if p is not None:
                out[i][0][wb] = (p[0], p[1], p[2] + 0.60)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="p12h_latest")
    a = ap.parse_args()
    path = os.path.join(a.dir, "sender_log.jsonl")
    if not os.path.exists(path):
        print("no sender_log.jsonl in %s" % a.dir)
        return 2
    frames = load_frames(path)
    print("=" * 92)
    print("P1-4 EVALUATION — real capture %s, %d frames" % (a.dir, len(frames)))
    print("=" * 92)
    print("NOTE: sender_log has no per-joint confidence; replay uses a nominal %.2f for every" % NOMINAL_CONF)
    print("      emitted joint. This exercises the GEOMETRIC path, not the confidence path.")

    base, _skb, _krb, _ = run(frames, use_p14=False)
    p14, sk4, kr4, costs = run(frames, use_p14=True, timed=True)

    # ---------------- 1. clean replay: P1-4 must not disturb healthy motion -----------
    print("\n" + "-" * 92)
    print("1. CLEAN REPLAY — does P1-4 disturb legitimate motion?")
    print("-" * 92)
    diffs = []
    changed = 0
    total = 0
    for fb, f4 in zip(base, p14):
        for idx in fb:
            if idx not in f4:
                continue
            total += 1
            if fb[idx][5] and f4[idx][5]:
                d = _dist(fb[idx][:3], f4[idx][:3])
                diffs.append(d)
                if d > 1e-6:
                    changed += 1
    st = stats(diffs)
    print("  P1-4 vs P1-1 output difference : %s" % fmt(st))
    print("  joint-frames modified by P1-4  : %d / %d (%.2f%%)"
          % (changed, total, 100.0 * changed / max(1, total)))
    mb, m4 = state_mix(base), state_mix(p14)
    print("  state mix P1-1 only : " + "  ".join("%s %.1f%%" % (k, mb.get(k, 0)) for k in
                                                 ("TRACKED", "WEAK", "PREDICTED", "LOST")))
    print("  state mix P1-1+P1-4 : " + "  ".join("%s %.1f%%" % (k, m4.get(k, 0)) for k in
                                                 ("TRACKED", "WEAK", "PREDICTED", "LOST", "RECOVERING")))
    geo = sum(js.geo_rejections for js in kr4.joints.values())
    print("  geometric rejections on CLEAN data (false positives): %d / %d joint-frames (%.3f%%)"
          % (geo, total, 100.0 * geo / max(1, total)))

    # ---------------- 2. segment stability -------------------------------------------
    print("\n" + "-" * 92)
    print("2. SEGMENT-LENGTH DEVIATION of the emitted skeleton")
    print("-" * 92)
    db = seg_deviation(base, kr4)
    d4 = seg_deviation(p14, kr4)
    print("  P1-1 only   : %s" % fmt(stats(db), unit="rel"))
    print("  P1-1 + P1-4 : %s" % fmt(stats(d4), unit="rel"))
    print("\n  learned bone lengths (robust median +/- MAD):")
    for k, e in sorted(kr4.lengths.items()):
        if e.ready():
            print("    %-12s %.4f m  mad %.4f  accepted %d  rejected %d"
                  % ("%s-%s" % (NAMES.get(k[0], k[0]), NAMES.get(k[1], k[1])),
                     e.length(), e.spread(), e.accepted, e.rejected))

    # ---------------- 3. per-joint displacement (responsiveness) ---------------------
    print("\n" + "-" * 92)
    print("3. PER-JOINT FRAME-TO-FRAME DISPLACEMENT (responsiveness must not drop)")
    print("-" * 92)
    print("  %-11s %-34s %-34s" % ("joint", "P1-1 only med/p95/max", "P1-1+P1-4 med/p95/max"))
    for _k, _i, wb in MAP:
        sb = stats(displacement(base, wb))
        s4 = stats(displacement(p14, wb))
        if not sb or not s4:
            continue
        print("  %-11s %8.4f %8.4f %8.4f          %8.4f %8.4f %8.4f"
              % (NAMES[wb], sb["median"], sb["p95"], sb["max"],
                 s4["median"], s4["p95"], s4["max"]))

    # ---------------- 4. adversarial --------------------------------------------------
    print("\n" + "-" * 92)
    print("4. ADVERSARIAL — injected into the real trajectory, error vs the UN-injected run")
    print("-" * 92)
    start = len(frames) // 2
    cases = [
        ("missing knee, 5 frames", "missing", 13, 5),
        ("missing knee, 20 frames", "missing", 13, 20),
        ("frozen wrist, 25 frames", "frozen", 9, 25),
        ("hi-conf knee teleport, 10 frames", "teleport", 13, 10),
        ("slow impossible knee drift, 25 frames", "drift", 13, 25),
        ("bad depth on elbow (+0.6 m Z), 10 frames", "baddepth", 7, 10),
    ]
    print("  %-40s %9s %10s %10s %10s %8s" %
          ("failure", "detected", "recovered", "P1-1 only", "P1-1+P1-4", "frames"))
    print("  " + "-" * 88)
    for label, kind, wb, n in cases:
        inj = inject(frames, kind, wb, start, n)
        out, sk_i, kr_i, _ = run(inj, use_p14=True)
        out_nop14, _s, _k, _c = run(inj, use_p14=False)
        # GROUND TRUTH = the raw un-injected measurement. Using a tracker output as the
        # reference makes the "before" column move whenever the tracker is retuned, which
        # is exactly the mistake this replaces.
        errs_before = []
        errs_after = []
        for i in range(start, min(len(frames), start + n)):
            truth = frames[i][0].get(wb)
            if truth is None:
                continue
            b = out_nop14[i].get(wb)
            a4 = out[i].get(wb)
            if b is not None and b[5]:
                errs_before.append(_dist(truth, b[:3]))
            if a4 is not None and a4[5]:
                errs_after.append(_dist(truth, a4[:3]))
        js = kr_i.joints.get(wb)
        detected = bool(js and (js.geo_rejections > 0 or js.reconstructions > 0))
        recovered = bool(js and js.reconstructions > 0)
        eb = max(errs_before) if errs_before else 0.0
        ea = max(errs_after) if errs_after else 0.0
        verdict = "same"
        if ea < eb * 0.9:
            verdict = "BETTER"
        elif ea > eb * 1.1:
            verdict = "WORSE"
        print("  %-40s %9s %10s %9.4fm %9.4fm %8d  %s"
              % (label, "YES" if detected else "no", "YES" if recovered else "no",
                 eb, ea, n, verdict))

    # ---------------- 5. performance --------------------------------------------------
    print("\n" + "-" * 92)
    print("5. PERFORMANCE (P1-4 only, perf_counter)")
    print("-" * 92)
    sc = stats(costs)
    if sc:
        print("  median %.4f  p95 %.4f  p99 %.4f  max %.4f ms  (n=%d)"
              % (sc["median"], sc["p95"], sc["p99"], sc["max"], sc["n"]))
        print("  budget: median < 1 ms  p99 < 2 ms  ->  %s"
              % ("PASS" if sc["median"] < 1.0 and sc["p99"] < 2.0 else "FAIL"))
    print("\n  body consistency (final): %.3f   tracked %.2f  predicted %.2f  lost %.2f"
          % (kr4.body_consistency, kr4.tracked_ratio, kr4.predicted_ratio, kr4.lost_ratio))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
