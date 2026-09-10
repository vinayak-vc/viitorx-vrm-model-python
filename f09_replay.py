#!/usr/bin/env python3
"""F-09 -- (a) does the candidate gate destroy GENUINE turns? (b) counterfactual replay of V5.

(a) THE DISQUALIFYING TEST. A quality gate that also rejects real turning is worse than no gate. The
    image-confirmed turns are the frames where the 2-D shoulder span itself says the subject is
    turned (`yaw2D` large) AND the 3-D yaw agrees with it (`|disagree|` small) -- i.e. both the
    independent 2-D geometry and the depth reconstruction concur. If the candidate predictor fires on
    those, it is rejected outright.

(b) COUNTERFACTUAL REPLAY. The recorded torso yaw is pushed through the SHIPPING conditioner
    (`DampYaw`, port verified to 6.7e-6 deg) and the V5 relative composition, twice:
        A. current system  -- V5 + TrunkGate as shipped
        B. candidate       -- V5 + TrunkGate + torsoYawQuality hold
    and the metrics the brief lists are measured on both.

    python f09_replay.py
"""
import io
import json
import math
import os

from oak_yaw_v4 import DampYaw

FEAT = os.path.join("oak_v4_evidence", "f09_features.jsonl")
WARMUP = 90
MEM = 600
# TrunkGate, as shipped (Runtime/Retargeting/TrunkGate.cs)
MIN_SPAN, MAX_SPAN, MAX_RATE, REACQUIRE = 0.10, 1.0, 500.0, 8


def pct(v, p):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[int(round((len(s) - 1) * p / 100.0))]


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def stdev(v):
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def wrapd(a):
    return (a + 180.0) % 360.0 - 180.0


def add_online(rows):
    for key, out in (("shSpan3D", "shLenErrOnline"), ("hipSpan3D", "hipLenErrOnline")):
        seen = []
        for r in rows:
            seen.append(r[key])
            if len(seen) < WARMUP:
                r[out] = 0.0
                continue
            s = sorted(seen[-MEM:])
            med = s[len(s) // 2]
            r[out] = abs(r[key] / med - 1.0) if med > 1e-6 else 0.0
    return rows


def genuine_turn_test(rows, thrs):
    """Frames the IMAGE confirms are real turns, and how often each threshold would reject them."""
    turns = [r for r in rows if r["yaw2D"] > 25.0 and abs(r["disagree"]) < 10.0]
    still = [r for r in rows if r["yaw2D"] < 10.0 and abs(r["yaw3D"]) < 10.0]
    print("=" * 106)
    print(" (a) DOES THE GATE DESTROY GENUINE TURNS?")
    print("=" * 106)
    print(" image-confirmed genuine turns (2-D says turned AND 3-D agrees): %d frames" % len(turns))
    print(" image-confirmed standing still:                                 %d frames" % len(still))
    if not turns:
        print(" NO image-confirmed turns in this corpus -- the test cannot be run, and no gate")
        print(" should be shipped on that basis alone.")
        return turns
    print("\n %-10s | %-28s | %-24s | %s" %
          ("thr", "rejects GENUINE turns", "catches BAD", "rejects STILL frames"))
    for t in thrs:
        rt = sum(1 for r in turns if r["hipLenErrOnline"] > t)
        bad = [r for r in rows if r["bad"]]
        rb = sum(1 for r in bad if r["hipLenErrOnline"] > t)
        rs = sum(1 for r in still if r["hipLenErrOnline"] > t)
        print(" %-10.2f | %5d / %5d  (%5.1f%%)   | %5d / %5d (%5.1f%%) | %5d / %5d (%5.1f%%)"
              % (t, rt, len(turns), 100.0 * rt / len(turns), rb, len(bad),
                 100.0 * rb / max(1, len(bad)), rs, len(still), 100.0 * rs / max(1, len(still))))
    return turns


def replay(rows, use_quality, thr):
    """V5 composition + TrunkGate (+ optional quality hold). Returns per-frame avatar chain yaw."""
    dh, ds = DampYaw(), DampYaw()
    held_h, held_s, has, streak = 0.0, 0.0, False, 0
    chain, src, rejected, accepted_idx = [], [], [], []
    prev_t = rows[0]["t"]
    for i, r in enumerate(rows):
        dt = r["t"] - prev_t
        prev_t = r["t"]
        if not (0 < dt < 1.0):
            dt = 0.0465
        # --- shipped TrunkGate ------------------------------------------------------------
        ok = (MIN_SPAN < r["shSpan3D"] <= MAX_SPAN) and (MIN_SPAN < r["hipSpan3D"] <= MAX_SPAN)
        if ok and has:
            step = max(abs(wrapd(r["yaw3D"] - held_s)), abs(wrapd(r["hipYaw3D"] - held_h)))
            if step > MAX_RATE * dt:
                ok = False
                streak += 1
                if streak >= REACQUIRE:
                    ok, streak = True, 0
            else:
                streak = 0
        elif not ok:
            streak = 0
        # --- candidate quality hold --------------------------------------------------------
        if ok and use_quality and r["hipLenErrOnline"] > thr:
            ok = False
        if ok:
            held_h, held_s, has = r["hipYaw3D"], r["yaw3D"], True
            accepted_idx.append(i)
        rejected.append(0 if ok else 1)
        h = math.degrees(dh.step(math.radians(held_h if has else 0.0), dt))
        s = math.degrees(ds.step(math.radians(held_s if has else 0.0), dt))
        chain.append(h + (s - h))          # V5: hips + relative twist == shoulder yaw
        src.append(r["yaw3D"])
    return chain, src, rejected, accepted_idx


def recovery_latency(rejected, times):
    """Longest and median gap (s) between consecutive ACCEPTED frames."""
    gaps, last = [], None
    for i, rej in enumerate(rejected):
        if not rej:
            if last is not None and i - last > 1:
                gaps.append(times[i] - times[last])
            last = i
    return gaps


def main():
    rows = [json.loads(l) for l in io.open(FEAT, encoding="utf-8") if l.strip()]
    caps = {}
    for r in rows:
        caps.setdefault(r["cap"], []).append(r)
    for c in caps:
        caps[c].sort(key=lambda r: r["seq"])
        add_online(caps[c])
    pooled = [r for rs in caps.values() for r in rs[WARMUP:]]

    turns = genuine_turn_test(pooled, (0.10, 0.15, 0.17, 0.20, 0.25, 0.30, 0.40))

    print("\n" + "=" * 106)
    print(" (b) COUNTERFACTUAL REPLAY -- A: V5 as shipped   B: V5 + torsoYawQuality hold")
    print("=" * 106)
    print(" %-22s %-10s | %8s %8s | %7s %7s | %8s | %7s %7s" %
          ("capture", "variant", "err mean", "err p95", ">45deg", ">90deg", "rest sd", "rejects", "gap p95"))
    for cap, rs in sorted(caps.items()):
        rs2 = rs[WARMUP:]
        if len(rs2) < 200:
            continue
        for label, useq in (("A shipped", False), ("B +quality", True)):
            chain, src, rej, acc = replay(rs2, useq, 0.17)
            a = [abs(c) for c in chain]
            err = [abs(wrapd(chain[i] - src[i])) for i in range(len(chain)) if not rej[i]]
            rest = [chain[i] for i in range(len(chain)) if abs(src[i]) < 8.0]
            gaps = recovery_latency(rej, [r["t"] for r in rs2])
            print(" %-22s %-10s | %8.2f %8.2f | %6.2f%% %6.2f%% | %8.3f | %6.2f%% %7.3f" %
                  (cap if label.startswith("A") else "", label, mean(err), pct(err, 95),
                   100.0 * sum(1 for x in a if x > 45) / len(a),
                   100.0 * sum(1 for x in a if x > 90) / len(a),
                   stdev(rest), 100.0 * sum(rej) / len(rej),
                   pct(gaps, 95) if gaps else 0.0))
        print("")
    print(" 'err' is measured only on ACCEPTED frames (a held frame has no fresh truth to score).")
    print(" 'gap p95' is the 95th-percentile time between accepted frames = recovery latency.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
