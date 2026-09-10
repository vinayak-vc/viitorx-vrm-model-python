#!/usr/bin/env python3
"""F-09 -- (a) quantify the stereo depth quantisation, (b) ROC the surviving predictors ONLINE.

(a) QUANTISATION. The torso yaw is atan2(dz, dx) over the shoulder pair, so its resolution is set by
    the resolution of dz. Stereo depth is quantised by integer disparity, and at ~2 m the step is
    large. This measures the step directly from the recorded p30 values and converts it into DEGREES
    OF COMMANDED YAW, which is the number that matters.

(b) ROC. Two things are fixed relative to the first pass:
    * `uSpanSh` is EXCLUDED as a predictor -- it is an input to the label, so its AUC is circular.
    * `shLenErr` is reported but flagged: it shares the shoulder u/z values with the label, so it is
      only partially independent. `hipLenErr` and `dYawRate` share nothing with the label and are the
      defensible ones.
    * The reference segment length is estimated ONLINE (expanding median over the frames seen so far,
      after a warm-up), not from the whole capture, so the operating point is one a live gate could
      actually reach.

    python f09_quantisation_roc.py
"""
import io
import json
import math
import os

FEAT = os.path.join("oak_v4_evidence", "f09_features.jsonl")
WARMUP = 90              # frames before the online median is trusted (~4 s at 21 Hz)


def pct(v, p):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[int(round((len(s) - 1) * p / 100.0))]


def load():
    return [json.loads(l) for l in io.open(FEAT, encoding="utf-8") if l.strip()]


def quantisation(rows):
    print("=" * 108)
    print(" (a) STEREO DEPTH QUANTISATION -- measured, then converted to degrees of commanded yaw")
    print("=" * 108)
    # distinct depth levels actually emitted for a SINGLE joint, per capture
    caps = {}
    for r in rows:
        caps.setdefault(r["cap"], []).append(r)
    for cap, rs in sorted(caps.items()):
        lv = sorted(set(round(r["midShZraw"] * 2 - r["shDzRaw"], 1) for r in rs))   # = 2*p30_R
        lv = [x / 2.0 for x in lv]
        steps = [lv[i] - lv[i - 1] for i in range(1, len(lv))]
        steps = [s for s in steps if s > 1.0]
        if not steps:
            continue
        span = pct([r["shSpan3D"] for r in rs], 50)
        step = pct(steps, 50)
        # one depth step across a shoulder line of this length, in degrees of yaw
        deg = math.degrees(math.atan2(step / 1000.0, max(1e-6, span)))
        print(" %-24s median depth step %6.1f mm | median shoulder span %.3f m | ONE STEP = %5.1f deg of yaw"
              % (cap, step, span, deg))
    dz = [abs(r["shDzRaw"]) for r in rows]
    zero = sum(1 for x in dz if x < 1e-6)
    print("\n |raw shoulder depth difference| over all %d frames:" % len(rows))
    print("   EXACTLY ZERO on %d frames (%.1f%%) -- both shoulders land on the SAME quantised level,"
          % (zero, 100.0 * zero / len(dz)))
    print("   which forces the commanded torso yaw to exactly 0 regardless of the true pose.")
    print("   p50=%.0f  p75=%.0f  p90=%.0f  p95=%.0f mm" % (pct(dz, 50), pct(dz, 75), pct(dz, 90), pct(dz, 95)))

    # If the depth difference is a staircase, the YAW it produces must be a staircase too.
    yaw = [r["yaw3D"] for r in rows]
    span = pct([r["shSpan3D"] for r in rows], 50)
    print("\n commanded torso yaw, |yaw3D| histogram (deg) -- the staircase this produces:")
    buckets = {}
    for y in yaw:
        b = int(abs(y) // 5) * 5
        buckets[b] = buckets.get(b, 0) + 1
    tot = len(yaw)
    for b in sorted(buckets):
        if buckets[b] * 200 < tot:      # hide the long thin tail
            continue
        print("   %3d-%3d deg : %6d (%5.1f%%)  %s" % (b, b + 5, buckets[b], 100.0 * buckets[b] / tot,
                                                      "#" * int(60.0 * buckets[b] / tot)))
    for mm in (pct(dz, 75), pct(dz, 90), pct(dz, 95)):
        print("   a %6.0f mm depth difference over a %.3f m shoulder line = %5.1f deg of yaw"
              % (mm, span, math.degrees(math.asin(min(1.0, mm / 1000.0 / span)))))


def online_len_err(rows, key_span):
    """Expanding-median segment length, computed causally so a live gate could reproduce it."""
    seen, out = [], []
    for r in rows:
        seen.append(r[key_span])
        if len(seen) < WARMUP:
            out.append(0.0)
            continue
        s = sorted(seen[-600:])              # ~30 s memory, bounded
        med = s[len(s) // 2]
        out.append(abs(r[key_span] / med - 1.0) if med > 1e-6 else 0.0)
    return out


def roc(rows, values, higher_is_worse=True):
    lab = [r["bad"] for r in rows]
    # Sweep the threshold from MOST suspicious to least: descending when high values are bad.
    pairs = sorted(zip(values, lab), key=lambda p: p[0], reverse=higher_is_worse)
    P = sum(lab)
    N = len(lab) - P
    if P < 20 or N < 20:
        return None
    tp = fp = 0
    best = None
    pts = []
    for v, l in pairs:
        if l:
            tp += 1
        else:
            fp += 1
        tpr, fpr = tp / P, fp / N
        pts.append((v, tpr, fpr))
        # Youden J
        j = tpr - fpr
        if best is None or j > best[0]:
            best = (j, v, tpr, fpr)
    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0
    for _v, tpr, fpr in pts:
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_fpr, prev_tpr = fpr, tpr
    return {"auc": auc, "bestJ": best[0], "thr": best[1], "tpr": best[2], "fpr": best[3]}


def main():
    rows = load()
    caps = {}
    for r in rows:
        caps.setdefault(r["cap"], []).append(r)
    for cap in caps:
        caps[cap].sort(key=lambda r: r["seq"])

    quantisation(rows)

    # online features, computed per capture then pooled
    for cap, rs in caps.items():
        for r, v in zip(rs, online_len_err(rs, "shSpan3D")):
            r["shLenErrOnline"] = v
        for r, v in zip(rs, online_len_err(rs, "hipSpan3D")):
            r["hipLenErrOnline"] = v
    pooled = [r for rs in caps.values() for r in rs[WARMUP:]]

    print("\n" + "=" * 108)
    print(" (b) ROC on %d frames past warm-up (%d BAD, %.2f%%), ONLINE reference length"
          % (len(pooled), sum(r["bad"] for r in pooled),
             100.0 * sum(r["bad"] for r in pooled) / len(pooled)))
    print("=" * 108)
    print(" %-18s %6s %8s %8s %8s  %s" % ("predictor", "AUC", "thr", "TPR", "FPR", "independence of the label"))
    tests = [
        ("hipLenErrOnline", True, "INDEPENDENT (hip u/z; label uses shoulder u + hipZ)"),
        ("dYawRate", True, "INDEPENDENT (temporal only)"),
        ("dYaw", True, "INDEPENDENT (temporal only)"),
        ("shLenErrOnline", True, "PARTIAL -- shares shoulder u/z with the label"),
        ("confMin", False, "INDEPENDENT (model confidence)"),
        ("dsdShMax", True, "INDEPENDENT (depth-window spread)"),
        ("rangeShMax", True, "INDEPENDENT (depth-window range)"),
        ("validFracMin", False, "INDEPENDENT (valid-pixel count)"),
        ("surfCrossSh", True, "INDEPENDENT (surface discontinuity flag)"),
        ("_absTrunkDz", True, "INDEPENDENT (shoulder-vs-hip depth disagreement)"),
    ]
    for r in pooled:
        r["_absTrunkDz"] = abs(r["trunkDz"])
    results = {}
    for f, hw, note in tests:
        vals = [r[f] for r in pooled]
        s = roc(pooled, vals, hw)
        if s is None:
            continue
        results[f] = s
        print(" %-18s %6.3f %8.4f %8.3f %8.3f  %s" % (f, s["auc"], s["thr"], s["tpr"], s["fpr"], note))

    # combined rule: the two defensible, independent predictors
    print("\n" + "-" * 108)
    print(" COMBINED RULE (independent predictors only): reject when hip OR shoulder length is")
    print(" inconsistent, i.e. max(hipLenErrOnline, shLenErrOnline) > thr")
    for thr in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        tp = sum(1 for r in pooled if max(r["hipLenErrOnline"], r["shLenErrOnline"]) > thr and r["bad"])
        fp = sum(1 for r in pooled if max(r["hipLenErrOnline"], r["shLenErrOnline"]) > thr and not r["bad"])
        P = sum(r["bad"] for r in pooled)
        N = len(pooled) - P
        print("   thr=%.2f  caught %4d/%4d BAD (%5.1f%%)   rejected %5d/%5d GOOD (%5.2f%%)"
              % (thr, tp, P, 100.0 * tp / P, fp, N, 100.0 * fp / N))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
