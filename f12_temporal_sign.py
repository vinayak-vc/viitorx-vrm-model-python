#!/usr/bin/env python3
"""F-12 -- can temporal continuity supply the SIGN of torso yaw from |yaw2D| alone?

Offline only. No production code is touched.

THE METHODOLOGICAL TRAP THIS SCRIPT IS BUILT TO AVOID
-----------------------------------------------------
Every recorded protocol alternates: the subject goes +45, +90, back through 0, then -45, -90. So EVERY
visit to zero in the data is followed by a genuine reversal. A rule as trivial as "flip the sign every
time the magnitude dips near zero" therefore scores ~100 % on this data -- and that is an ARTEFACT OF
THE PROTOCOL, not a property of the method. The case that would break it, "magnitude returns to zero
and then goes back to the SAME side", was never recorded.

So this script reports two things separately and refuses to conflate them:
  (a) how the candidates score on the recorded sequences  -- flattering, and labelled as such;
  (b) whether the information needed to make the decision EXISTS in |yaw2D| at all -- which is the
      real question, and is answered by the zero-crossing and 180 deg analyses.

    python f12_temporal_sign.py
"""
import bisect
import io
import json
import math
import os

CONF_MIN = 0.3
RUNS = [
    ("F-10", "pipeline_logs_f10_near", os.path.join("oak_v4_evidence", "f10_gt_marks_near_RECOVERED.json")),
    ("F-11", "pipeline_logs_f11", os.path.join("oak_v4_evidence", "f10_gt_marks_near.json")),
]


def pct(v, p):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[int(round((len(s) - 1) * p / 100.0))]


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def rmse(v):
    return math.sqrt(sum(x * x for x in v) / len(v)) if v else float("nan")


def load(dirpath):
    audit, sender = {}, {}
    for nm, sink in (("audit_log.jsonl", audit), ("sender_log.jsonl", sender)):
        p = os.path.join(dirpath, nm)
        if not os.path.exists(p):
            return []
        for line in io.open(p, encoding="utf-8"):
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
        if min(lsh["c"], rsh["c"]) < CONF_MIN:
            continue
        rows.append({"seq": seq, "t": float(s.get("t", 0.0)),
                     "prod": abs(lsh["u"] - rsh["u"]) * float(hipZ)})
    rows.sort(key=lambda r: r["t"])
    return rows


def label_and_yaw(rows, marks):
    blocks = marks["blocks"]
    times = [r["t"] for r in rows]
    for r in rows:
        r["block"] = None
        r["gt"] = None
    for b in blocks:
        sp = b["t1"] - b["t0"]
        lo = bisect.bisect_left(times, b["t0"] + sp * 0.25)
        hi = bisect.bisect_right(times, b["t1"] - sp * 0.25)
        for r in rows[lo:hi]:
            r["block"] = b["name"]
            r["gt"] = b["headingGT"]
            r["headGT"] = b.get("headGT")
    z = [r["prod"] for r in rows if r["block"] and r["block"].startswith("h_0")]
    K = pct(z, 50)
    for r in rows:
        c = max(0.0, min(1.0, r["prod"] / K))
        r["yaw2D"] = math.degrees(math.acos(c))
    return K


# --------------------------------------------------------------------- algorithms
def run_alg(rows, kind, Z=5.0, N=2, init=+1):
    """Return per-frame inferred sign. All operate on |yaw2D| ONLY."""
    sign = init
    out = []
    armed = False          # magnitude has entered the near-zero band
    conf = 0
    prev_signed = 0.0
    for r in rows:
        m = r["yaw2D"]
        if kind == "A":
            # simple hold: flip once the magnitude has been near zero and then re-emerges
            if m < Z:
                armed = True
            elif armed:
                sign = -sign
                armed = False
        elif kind in ("B", "C"):
            if m < Z:
                armed = True
                conf = 0
            elif armed:
                conf += 1
                if conf >= (1 if kind == "B" else N):
                    sign = -sign
                    armed = False
                    conf = 0
        elif kind == "D":
            # continuity cost: pick the signed value closest to the previous signed state
            if abs(+m - prev_signed) <= abs(-m - prev_signed):
                sign = +1
            else:
                sign = -1
        out.append(sign)
        prev_signed = sign * m
    return out


# --------------------------------------------------------------------- analyses
def static_score(rows, signs):
    sel = [(r, s) for r, s in zip(rows, signs)
           if r["gt"] not in (None, 0) and r.get("headGT") in (None,)]
    if not sel:
        return None
    ok = sum(1 for r, s in sel if (s > 0) == (r["gt"] > 0))
    pos = [(r, s) for r, s in sel if r["gt"] > 0]
    neg = [(r, s) for r, s in sel if r["gt"] < 0]
    return {
        "n": len(sel), "overall": 100.0 * ok / len(sel),
        "pos": 100.0 * sum(1 for r, s in pos if s > 0) / max(1, len(pos)),
        "neg": 100.0 * sum(1 for r, s in neg if s < 0) / max(1, len(neg)),
        "p90": _acc([(r, s) for r, s in sel if r["gt"] == 90]),
        "m90": _acc([(r, s) for r, s in sel if r["gt"] == -90]),
        "p45": _acc([(r, s) for r, s in sel if r["gt"] == 45]),
        "m45": _acc([(r, s) for r, s in sel if r["gt"] == -45]),
    }


def _acc(pairs):
    if not pairs:
        return float("nan")
    return 100.0 * sum(1 for r, s in pairs if (s > 0) == (r["gt"] > 0)) / len(pairs)


def hybrid_stats(rows, signs):
    errs, wrong = [], 0
    for r, s in zip(rows, signs):
        if r["gt"] in (None,) or r.get("headGT") is not None:
            continue
        e = s * r["yaw2D"] - r["gt"]
        errs.append(abs(e))
        if r["gt"] != 0 and (s > 0) != (r["gt"] > 0):
            wrong += 1
    n = sum(1 for r in rows if r["gt"] not in (None,) and r.get("headGT") is None)
    return {"MAE": mean(errs), "RMSE": rmse(errs), "p50": pct(errs, 50),
            "p95": pct(errs, 95), "max": max(errs) if errs else float("nan"),
            "wrongPct": 100.0 * wrong / max(1, n)}


def main():
    data = {}
    for tag, d, mp in RUNS:
        if not os.path.isdir(d) or not os.path.exists(mp):
            print("missing %s / %s" % (d, mp))
            continue
        rows = load(d)
        marks = json.load(io.open(mp, encoding="utf-8"))
        K = label_and_yaw(rows, marks)
        data[tag] = (rows, marks, K)
        print("%s: %d frames, %d labelled, K=%.1f%s"
              % (tag, len(rows), sum(1 for r in rows if r["block"]), K,
                 "  (marks RECOVERED)" if marks.get("RECOVERED") else ""))

    # ---------------- the information question, asked first ------------------------
    print("\n" + "=" * 104)
    print(" (1) DOES |yaw2D| CONTAIN THE SIGN INFORMATION AT ALL?")
    print("=" * 104)
    rows, marks, _K = data["F-10"]
    seq = [b for b in marks["blocks"] if b["headingGT"] is not None]
    print("\n every zero visit in the recorded protocol is followed by a REVERSAL:")
    prev = None
    rev = same = 0
    for b in seq:
        g = b["headingGT"]
        if g == 0:
            continue
        if prev is not None and prev * g < 0:
            rev += 1
        elif prev is not None and prev * g > 0:
            same += 1
        prev = g
    print("   sign changes between consecutive non-zero blocks: %d reversals, %d same-side" % (rev, same))
    print("   -> a rule that flips at EVERY zero crossing scores ~100 %% on this data by construction.")
    print("   -> the 'return to the same side' case was NEVER RECORDED, so this data cannot")
    print("      distinguish a working method from that trivial rule.")

    print("\n (2) 180 DEG: does the magnitude saturate?")
    y = [r["yaw2D"] for r in rows if r["block"] == "m_180"]
    if y:
        print("   m_180 block: |yaw2D| p50=%.1f p95=%.1f MAX=%.1f deg over %d frames"
              % (pct(y, 50), pct(y, 95), max(y), len(y)))
        print("   acos() is bounded to [0,90], so a 180 deg turn and a 90 deg turn produce the SAME")
        print("   magnitude trace: 0 -> 90 -> 0. The information is lost in the estimator, not the sign.")

    # ---------------- false crossing opportunities ----------------------------------
    print("\n (3) FALSE CROSSINGS -- does |yaw2D| dip into the zero band while the subject is HELD away from 0?")
    print(" %-8s %6s | %s" % ("Z band", "blocks", "frames inside the band during a held +-45/+-90 block"))
    for Z in (3.0, 5.0, 8.0, 10.0, 15.0):
        tot = ins = 0
        for tag in data:
            rs, _m, _k = data[tag]
            held = [r for r in rs if r["gt"] not in (None, 0) and r.get("headGT") is None]
            tot += len(held)
            ins += sum(1 for r in held if r["yaw2D"] < Z)
        print(" %8.1f %6s | %d / %d  (%.2f %%)" % (Z, "-", ins, tot, 100.0 * ins / max(1, tot)))

    # ---------------- parameter sweep ------------------------------------------------
    print("\n" + "=" * 104)
    print(" (4) PARAMETER SWEEP -- scored on the recorded sequences (SEE THE CAVEAT IN (1))")
    print("=" * 104)
    print(" %-4s %5s %3s | %8s %8s %8s | %7s %7s | %8s %8s %8s" %
          ("alg", "Z", "N", "static%", "pos%", "neg%", "+90%", "-90%", "MAE", "RMSE", "wrong%"))
    best = None
    for kind in ("A", "B", "C", "D"):
        Zs = [5.0] if kind == "D" else [3.0, 5.0, 8.0, 10.0, 15.0]
        Ns = [1] if kind in ("A", "B", "D") else [1, 2, 3, 4, 5, 6]
        for Z in Zs:
            for N in Ns:
                allrows, allsigns = [], []
                for tag in ("F-10", "F-11"):
                    rs, _m, _k = data[tag]
                    sg = run_alg(rs, kind, Z, N, init=+1)
                    allrows += rs
                    allsigns += sg
                st = static_score(allrows, allsigns)
                hy = hybrid_stats(allrows, allsigns)
                if st is None:
                    continue
                print(" %-4s %5.1f %3d | %7.2f%% %7.2f%% %7.2f%% | %6.1f%% %6.1f%% | %8.2f %8.2f %7.2f%%" %
                      (kind, Z, N, st["overall"], st["pos"], st["neg"], st["p90"], st["m90"],
                       hy["MAE"], hy["RMSE"], hy["wrongPct"]))
                if best is None or st["overall"] > best[0]["overall"]:
                    best = (st, hy, kind, Z, N)

    # ---------------- dynamic blocks -------------------------------------------------
    print("\n" + "=" * 104)
    print(" (5) DYNAMIC BLOCKS -- sign transitions produced by the best configuration")
    print("=" * 104)
    st, hy, kind, Z, N = best
    print(" best static configuration: alg %s, Z=%.1f, N=%d" % (kind, Z, N))
    rs, _m, _k = data["F-10"]
    sg = run_alg(rs, kind, Z, N)
    print("\n %-16s %6s | %8s %8s | %10s %s" %
          ("block", "n", "yaw p50", "yaw max", "sign flips", "flips per second"))
    for b in _m["blocks"]:
        if not b["name"].startswith("m_"):
            continue
        idx = [i for i, r in enumerate(rs) if r["block"] == b["name"]]
        if len(idx) < 5:
            continue
        y = [rs[i]["yaw2D"] for i in idx]
        fl = sum(1 for k in range(1, len(idx)) if sg[idx[k]] != sg[idx[k - 1]])
        dur = rs[idx[-1]]["t"] - rs[idx[0]]["t"]
        print(" %-16s %6d | %8.1f %8.1f | %10d %14.2f" %
              (b["name"], len(idx), pct(y, 50), max(y), fl, fl / max(0.1, dur)))

    # ---------------- head independence ----------------------------------------------
    print("\n" + "=" * 104)
    print(" (6) HEAD INDEPENDENCE -- F-11 decoupling blocks (torso 0, head turned)")
    print("=" * 104)
    rs11, m11, _k = data["F-11"]
    sg11 = run_alg(rs11, kind, Z, N)
    for nm in ("dc_body0_head45R", "dc_body0_head45L"):
        idx = [i for i, r in enumerate(rs11) if r["block"] == nm]
        if not idx:
            continue
        y = [rs11[i]["yaw2D"] for i in idx]
        fl = sum(1 for k in range(1, len(idx)) if sg11[idx[k]] != sg11[idx[k - 1]])
        print("   %-18s n=%4d  |yaw2D| p50=%5.1f p95=%5.1f  sign flips=%d"
              % (nm, len(idx), pct(y, 50), pct(y, 95), fl))
    print("   (|yaw2D| uses only the SHOULDER span, so head motion cannot enter it by construction;")
    print("    this is a sanity check that the shoulders were not disturbed, not a new hypothesis.)")

    print("\n reference: production yaw3D signed MAE 61.08 | depth-sign hybrid 13.09 | magnitude alone 5.90")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
