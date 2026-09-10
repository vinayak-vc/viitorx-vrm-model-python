#!/usr/bin/env python3
"""F-09 -- the adversarial question: can ANY available feature tell a fast LEGITIMATE turn from a BAD
measurement?

This is the test that decides whether a torso quality gate is possible at all. A gate only has value
if, among the frames where the yaw is MOVING, it can separate:

    LEGIT : the image agrees the subject is turned      (yaw2D large, |disagree| small)
    BAD   : the depth claims a turn the image denies    (disagree > 20 deg)

If no feature separates those two populations, then a quality score cannot exist on this data,
regardless of how well it separates BAD from the (mostly static) background -- because rejecting BAD
would reject LEGIT at a similar rate, which is the collateral damage measured in f09_replay.py.

Also reports the brief's adversarial conditions that ARE derivable from the recordings.

    python f09_adversarial.py
"""
import io
import json
import math
import os

FEAT = os.path.join("oak_v4_evidence", "f09_features.jsonl")
WARMUP, MEM = 90, 600


def pct(v, p):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[int(round((len(s) - 1) * p / 100.0))]


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def auc(pos, neg):
    """Rank AUC of `pos` (bad) against `neg` (legit); 0.5 = indistinguishable."""
    if len(pos) < 20 or len(neg) < 20:
        return float("nan")
    pairs = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    i, rsum = 0, 0.0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            if pairs[k][1] == 1:
                rsum += avg
        i = j + 1
    n1, n0 = len(pos), len(neg)
    return (rsum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


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


def main():
    rows = [json.loads(l) for l in io.open(FEAT, encoding="utf-8") if l.strip()]
    caps = {}
    for r in rows:
        caps.setdefault(r["cap"], []).append(r)
    for c in caps:
        caps[c].sort(key=lambda r: r["seq"])
        add_online(caps[c])
    pooled = [r for rs in caps.values() for r in rs[WARMUP:]]
    for r in pooled:
        r["_absTrunkDz"] = abs(r["trunkDz"])

    legit = [r for r in pooled if r["yaw2D"] > 25.0 and abs(r["disagree"]) < 10.0]
    bad = [r for r in pooled if r["bad"]]

    print("=" * 104)
    print(" THE DECIDING TEST -- separating a fast LEGITIMATE turn from a BAD measurement")
    print("=" * 104)
    print(" LEGIT (image confirms the turn): %d frames" % len(legit))
    print(" BAD   (image denies the turn)  : %d frames" % len(bad))
    print("\n %-18s %8s | %10s %10s | %10s %10s" %
          ("feature", "AUC", "legit p50", "legit p95", "bad p50", "bad p95"))
    feats = ["hipLenErrOnline", "shLenErrOnline", "dYaw", "dYawRate", "confMin",
             "dsdShMax", "rangeShMax", "validFracMin", "_absTrunkDz", "hipZ", "cov"]
    for f in feats:
        a = auc([r[f] for r in bad], [r[f] for r in legit])
        if a != a:
            continue
        print(" %-18s %8.3f | %10.4f %10.4f | %10.4f %10.4f" %
              (f, a, pct([r[f] for r in legit], 50), pct([r[f] for r in legit], 95),
               pct([r[f] for r in bad], 50), pct([r[f] for r in bad], 95)))
    print("\n AUC near 0.5 means the feature CANNOT tell the two apart. A gate built on such a")
    print(" feature rejects genuine turning at close to the rate it rejects bad measurements.")

    # ---- the brief's adversarial conditions, as far as the recordings support them ----------
    print("\n" + "=" * 104)
    print(" ADVERSARIAL CONDITIONS derivable from the recordings")
    print("=" * 104)
    conds = [
        ("3. standing still", lambda r: r["yaw2D"] < 10 and abs(r["yaw3D"]) < 10),
        ("10. rapid legitimate turn", lambda r: r["yaw2D"] > 25 and abs(r["disagree"]) < 10 and r["dYaw"] > 3),
        ("1/2. turn or twist (any)", lambda r: r["yaw2D"] > 20),
        ("5. one shoulder low-conf", lambda r: r["confMin"] < 0.45),
        ("6. poor depth (window spread)", lambda r: r["dsdShMax"] > 200),
        ("9. depth discontinuity", lambda r: r["surfCrossSh"] == 1),
        ("8. partial person (low cov)", lambda r: r["cov"] < 12),
    ]
    print(" %-32s %7s | %9s %9s | %9s" % ("condition", "frames", "|yaw3D|p50", "|yaw3D|p95", "BAD rate"))
    for name, pred in conds:
        sel = [r for r in pooled if pred(r)]
        if len(sel) < 20:
            print(" %-32s %7d | (too few frames to characterise)" % (name, len(sel)))
            continue
        y = [abs(r["yaw3D"]) for r in sel]
        print(" %-32s %7d | %9.2f %9.2f | %8.2f%%" %
              (name, len(sel), pct(y, 50), pct(y, 95),
               100.0 * sum(r["bad"] for r in sel) / len(sel)))
    print("\n Conditions 4 (partial torso occlusion) and 7 (empty room) are not separable in these")
    print(" recordings -- an empty room produces no landmarks at all, which the shipped TrunkGate")
    print(" already blocks (V5 report sec 5b), and no occlusion ground truth was recorded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
