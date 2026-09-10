#!/usr/bin/env python3
"""F-12 -- recover the F-10 block boundaries, which were overwritten.

The F-11 capture was run with `--distance near` and rewrote `f10_gt_marks_near.json`, destroying the
F-10 block windows. The F-10 FRAME DATA survives in `pipeline_logs_f10_near/`, and the protocol was
deterministic, so the boundaries can be reconstructed rather than recaptured.

METHOD -- and its honesty conditions
------------------------------------
The F-10 protocol at capture time was STATIC_COARSE(7) + TWIST(5) + MOTION(8) = 20 blocks, each
`TRANSITION` seconds of movement then a fixed hold. Given a start time and a small clock-stretch
factor, every boundary follows. Both are recovered by maximising a CONTRAST objective that uses only
the static blocks' expected structure:

    score(offset, stretch) = mean(|yaw2D| over blocks expected at +-90)
                           - mean(|yaw2D| over blocks expected at 0)

This asks only "does the alignment put the wide-shoulder frames where 0 deg is expected and the
narrow-shoulder frames where +-90 is expected". It never uses the heading VALUES it will later score
against, so it cannot manufacture agreement.

The recovery is accepted ONLY if the resulting per-block medians reproduce the values already published
in `f10_near_analysis.txt` (0 deg -> 0.0/4.2/0.0, +45 -> 45.0, +90 -> 80.7, -45 -> 39.3, -90 -> 88.5).
That is an independent check the search does not optimise for. If it fails, the F-10 motion blocks are
declared unrecoverable and the report says so.

    python f12_recover_f10_marks.py
"""
import io
import json
import math
import os

DIR = "pipeline_logs_f10_near"
OUT = os.path.join("oak_v4_evidence", "f10_gt_marks_near_RECOVERED.json")
TRANSITION = 7

# (name, headingGT, hold) exactly as the protocol ran for F-10
PROTOCOL = [
    ("h_0", 0, 10), ("h_p45", 45, 8), ("h_p90", 90, 10), ("h_0b", 0, 8),
    ("h_m45", -45, 8), ("h_m90", -90, 10), ("h_0c", 0, 8),
    ("tw_rigid_p45", 45, 8), ("tw_sh_p45", 45, 8), ("tw_sh_m45", -45, 8),
    ("tw_opp_a", None, 8), ("tw_opp_b", None, 8),
    ("m_slow", None, 12), ("m_normal", None, 10), ("m_fast", None, 10),
    ("m_rev_lr", None, 10), ("m_rev_rl", None, 10), ("m_180", None, 12),
    ("m_turn_arms", None, 10), ("m_turn_still", None, 10),
]
PUBLISHED = {"h_0": 0.0, "h_p45": 45.0, "h_p90": 80.7, "h_0b": 4.2,
             "h_m45": 39.3, "h_m90": 88.5, "h_0c": 0.0}


def pct(v, p):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[int(round((len(s) - 1) * p / 100.0))]


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def load():
    audit, sender = {}, {}
    for nm, sink in (("audit_log.jsonl", audit), ("sender_log.jsonl", sender)):
        for line in io.open(os.path.join(DIR, nm), encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    sink[r["seq"]] = r
                except ValueError:
                    pass
    rows = []
    for seq in sorted(set(audit) & set(sender)):
        a, s = audit[seq], sender[seq]
        j = a.get("j") or {}
        if "L-shoulder" not in j or "R-shoulder" not in j:
            continue
        hipZ = s.get("hipZ")
        if not hipZ or hipZ <= 0.1:
            continue
        lsh, rsh = j["L-shoulder"], j["R-shoulder"]
        if min(lsh["c"], rsh["c"]) < 0.3:
            continue
        rows.append({"seq": seq, "t": float(s.get("t", 0.0)),
                     "span": abs(lsh["u"] - rsh["u"]), "hipZ": float(hipZ)})
    return rows


def windows(t0, stretch):
    out, t = [], t0
    for name, gt, hold in PROTOCOL:
        t += TRANSITION * stretch
        a = t
        t += hold * stretch
        out.append((name, gt, a, t))
    return out


def score(times, prods, cum, t0, stretch):
    """Contrast between the expected-0deg and expected-+-90deg windows.

    `cum` is a prefix sum over `prods` so a window mean is O(log n) via bisect, not O(n). The naive
    form rescanned every frame for all ~9000 candidate alignments and did not finish.
    """
    import bisect
    w = dict((n, (a, b)) for n, _g, a, b in windows(t0, stretch))

    def band(names):
        tot, cnt = 0.0, 0
        for n in names:
            a, b = w[n]
            sp = b - a
            lo = bisect.bisect_left(times, a + sp * 0.25)
            hi = bisect.bisect_right(times, b - sp * 0.25)
            if hi > lo:
                tot += cum[hi] - cum[lo]
                cnt += hi - lo
        return tot, cnt

    zt, zc = band(["h_0", "h_0b", "h_0c"])
    wt, wc = band(["h_p90", "h_m90"])
    if zc < 30 or wc < 30:
        return -1e9
    # frames at 0 deg have the WIDEST shoulder span; at +-90 the narrowest.
    return zt / zc - wt / wc


def main():
    rows = load()
    if not rows:
        print("no usable frames in %s" % DIR)
        return 1
    for r in rows:
        r["prod"] = r["span"] * r["hipZ"]
    rows.sort(key=lambda r: r["t"])
    times = [r["t"] for r in rows]
    prods = [r["prod"] for r in rows]
    cum = [0.0]
    for v in prods:
        cum.append(cum[-1] + v)
    t_lo, t_hi = times[0], times[-1]
    total = sum(TRANSITION + h for _n, _g, h in PROTOCOL)
    print("=" * 96)
    print(" F-10 BLOCK RECOVERY -- log spans %.0f s, protocol needs %d s" % (t_hi - t_lo, total))
    print("=" * 96)

    best = None
    for stretch in [1.0 + i * 0.004 for i in range(0, 12)]:
        span_needed = total * stretch
        t = t_lo
        while t + span_needed <= t_hi + 1.0:
            s = score(times, prods, cum, t, stretch)
            if best is None or s > best[0]:
                best = (s, t, stretch)
            t += 0.25
    if best is None:
        print("no alignment found")
        return 1
    s, t0, stretch = best
    print(" best alignment: start offset %.2f s into the log, stretch %.3f, contrast score %.1f"
          % (t0 - t_lo, stretch, s))

    w = windows(t0, stretch)
    print("\n VALIDATION -- do the recovered static blocks reproduce the PUBLISHED F-10 medians?")
    # calibrate K on the recovered 0-blocks, exactly as the F-10 analyser did
    z = []
    for n, _g, a, b in w:
        if n.startswith("h_0"):
            sp = b - a
            z += [r["prod"] for r in rows if a + sp * 0.25 <= r["t"] <= b - sp * 0.25]
    K = pct(z, 50)
    ok = True
    print(" %-8s %10s %12s %10s" % ("block", "published", "recovered", "delta"))
    for n, _g, a, b in w:
        if n not in PUBLISHED:
            continue
        sp = b - a
        sel = [r for r in rows if a + sp * 0.25 <= r["t"] <= b - sp * 0.25]
        if len(sel) < 10:
            print(" %-8s %10.1f %12s" % (n, PUBLISHED[n], "(no frames)"))
            ok = False
            continue
        y = [math.degrees(math.acos(max(0.0, min(1.0, r["prod"] / K)))) for r in sel]
        got = pct(y, 50)
        d = abs(got - PUBLISHED[n])
        if d > 8.0:
            ok = False
        print(" %-8s %10.1f %12.1f %10.1f%s" % (n, PUBLISHED[n], got, d, "" if d <= 8 else "   <-- MISMATCH"))

    print("\n VERDICT: %s" % ("RECOVERY ACCEPTED" if ok else "RECOVERY REJECTED -- do not use"))
    if not ok:
        print(" The F-10 motion blocks must be treated as unrecoverable.")
        return 2
    blocks = [{"name": n, "headingGT": g, "headGT": None,
               "t0": round(a, 4), "t1": round(b, 4), "instr": "recovered"}
              for n, g, a, b in w]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps({"distance": "near", "metres": 1.5, "headings": "coarse",
                            "RECOVERED": True,
                            "recovery": {"offset_s": round(t0 - t_lo, 3), "stretch": round(stretch, 4),
                                         "contrast": round(s, 2)},
                            "start": round(t0, 4), "end": round(w[-1][3], 4),
                            "transition_s": TRANSITION, "blocks": blocks}, indent=2))
    print(" -> %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
