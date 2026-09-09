#!/usr/bin/env python3
"""ARM RETARGET V1 -- targeted forensics for the two live findings.

A) ELBOW ANGLE: is the avatar's elbow bend its own invention, or is it faithfully reproducing a
   bent elbow that the UPSTREAM LANDMARKS already contained?

   [RETARGET-TRACE] carries, on the same line, `want` (the raw landmark direction, i.e. what the
   green PoseDebugSkeleton draws) and `got` (the avatar's achieved bone direction). From the two
   arm bones of one side we can reconstruct BOTH elbow angles from the SAME frame:

       landmark elbow angle = 180 - angle(want_upper,  want_lower)
       avatar   elbow angle = 180 - angle(got_upper,   got_lower)

   If they track each other, a wrong-looking elbow is an upstream tracking fault and NOT a
   retargeting fault. That is the distinction the brief's root-cause section demands, and it is
   the reason `upperErrDeg` near zero must never be read as "tracking is correct".

B) OUTLIERS: locate every large arm error and every large single-frame direction step, and say
   what else was true at that instant (gate held, bend angle, roll source).

    python forensics_arm_v1.py --dir pipeline_logs_armv1,pipeline_logs_armv1b
"""
import argparse
import datetime
import io
import json
import os

import analyze_arm_v1_live as A


def elbow_pairs(rt_rows, side):
    """Per trace frame, reconstruct landmark vs avatar elbow angle for one side."""
    up_b = side + "UpperArm"
    lo_b = side + "LowerArm"
    byf = {}
    for r in rt_rows:
        if r["bone"] in (up_b, lo_b):
            byf.setdefault(r["frame"], {})[r["bone"]] = r
    out = []
    for f in sorted(byf):
        d = byf[f]
        if up_b not in d or lo_b not in d:
            continue
        lw = A.ang(d[up_b]["want"], d[lo_b]["want"])
        av = A.ang(d[up_b]["got"], d[lo_b]["got"])
        if lw is None or av is None:
            continue
        out.append((d[up_b]["t"], 180.0 - lw, 180.0 - av))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="pipeline_logs_armv1,pipeline_logs_armv1b")
    ap.add_argument("--date", default="")
    a = ap.parse_args()

    day = (datetime.date.fromisoformat(a.date) if a.date else datetime.date.today())
    ulog = os.path.join(os.path.expanduser("~"), "Documents", "MyMirror", "Logs",
                        "virtual-mirror-%s.log" % day.strftime("%Y%m%d"))
    rt, av, _rest = A.parse_unity_log(ulog, day)

    dirs = [d.strip() for d in a.dir.split(",") if d.strip()]
    chosen, order = {}, []
    for d in dirs:
        meta = json.load(io.open(os.path.join(d, "blocks.json"), encoding="utf-8"))
        for b in meta["blocks"]:
            key = (b["variant"], b["block"])
            if key not in chosen:
                order.append(key)
            b = dict(b)
            b["dir"] = d
            chosen[key] = b
    blocks = [chosen[k] for k in order]
    ml = []
    for d in dirs:
        bs = [b for b in blocks if b["dir"] == d]
        if bs:
            ml += A.load_model_log(os.path.join(d, "model_log.jsonl"),
                                   min(x["tStart"] for x in bs) - 5.0,
                                   max(x["tEnd"] for x in bs) + 5.0)

    # ---------------- A) elbow: landmark vs avatar ----------------------------------------
    print("=" * 108)
    print(" A) ELBOW ANGLE -- does the avatar reproduce the LANDMARK elbow, or invent its own?")
    print("    Both numbers come from the SAME trace line: want=landmarks, got=avatar bone.")
    print("    180 deg = straight arm.  |diff| is the RETARGETING error; a large landmark bend")
    print("    with a small |diff| is an UPSTREAM tracking fault, not a retargeting fault.")
    print("=" * 108)
    print("   %-26s %-6s | %-22s | %-22s | %s"
          % ("motion", "side", "LANDMARK elbow mean/min", "AVATAR elbow mean/min", "|diff| mean/p95/max"))
    per_variant = {}
    for b in blocks:
        rows = A.block_rows(rt, b["tStart"], b["tEnd"])
        for side, lbl in (("left", "left"), ("right", "right")):
            pr = elbow_pairs(rows, side)
            if not pr:
                continue
            lw = [x[1] for x in pr]
            avv = [x[2] for x in pr]
            df = [abs(x[1] - x[2]) for x in pr]
            per_variant.setdefault(b["variant"], []).extend(df)
            print("   %-26s %-6s | %8.1f %8.1f      | %8.1f %8.1f      | %6.1f %6.1f %7.1f  (n=%d)"
                  % ("%s %s" % (b["block"], b["title"][:22]), lbl,
                     A.mean(lw), min(lw), A.mean(avv), min(avv),
                     A.mean(df), A.pct(df, 95), max(df), len(pr)))
    print("   " + "-" * 102)
    for v, df in per_variant.items():
        print("   %-10s elbow reproduction error |diff|: mean=%.2f  p95=%.2f  max=%.2f  (n=%d)"
              % (v.upper(), A.mean(df), A.pct(df, 95), max(df), len(df)))

    # ---------------- B) outliers ----------------------------------------------------------
    print()
    print("=" * 108)
    print(" B) OUTLIERS -- every arm-bone retarget error above 10 deg in the NEW/DENSE variants,")
    print("    with the ARM-V1-TRACE state and the LimbGate state at that instant.")
    print("=" * 108)
    gate_by_t = sorted((r.get("tApply", 0.0), r) for r in ml)
    gt = [x[0] for x in gate_by_t]

    def gate_at(t):
        import bisect
        i = bisect.bisect_left(gt, t)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(gate_by_t) and abs(gate_by_t[j][0] - t) < 0.15:
                if best is None or abs(gate_by_t[j][0] - t) < abs(best[0] - t):
                    best = gate_by_t[j]
        return best[1] if best else None

    av_by_key = {}
    for x in av:
        av_by_key.setdefault((x["frame"], x["arm"]), x)

    n_out = 0
    for b in blocks:
        if b["variant"] == "old":
            continue
        rows = A.block_rows(rt, b["tStart"], b["tEnd"])
        bad = [r for r in rows if r["bone"] in A.ARM_BONES and r["err"] > 10.0]
        if not bad:
            continue
        n_out += len(bad)
        print("   -- %s %s (%s)  : %d of %d arm samples over 10 deg (%.2f%%)"
              % (b["block"], b["title"][:28], b["variant"], len(bad),
                 len([r for r in rows if r["bone"] in A.ARM_BONES]),
                 100.0 * len(bad) / max(1, len([r for r in rows if r["bone"] in A.ARM_BONES]))))
        for r in sorted(bad, key=lambda x: -x["err"])[:8]:
            arm = "leftArm" if r["bone"].startswith("left") else "rightArm"
            x = av_by_key.get((r["frame"], arm))
            g = gate_at(r["t"])
            held = ""
            if g:
                gg = g.get("gate", {})
                held = "gate hLArm=%s hRArm=%s" % (gg.get("hLArm"), gg.get("hRArm"))
            print("      %s  frame=%-7d %-14s errDeg=%6.1f | %s | %s"
                  % (datetime.datetime.fromtimestamp(r["t"]).strftime("%H:%M:%S.%f")[:-3],
                     r["frame"], r["bone"], r["err"],
                     ("bend=%5.1f roll=%7.1f src=%s" % (x["bend"], x["roll"], x["src"])) if x else "no arm trace",
                     held))
    if n_out == 0:
        print("   none")

    # ---------------- C) roll stability vs elbow bend --------------------------------------
    print()
    print("=" * 108)
    print(" C) ROLL STABILITY vs ELBOW BEND (dense trace only, ~40 Hz -- the only sampling rate")
    print("    at which a per-step roll number means anything). BendSinMin=0.2 ~= 11.5 deg.")
    print("=" * 108)
    dense = [b for b in blocks if b["variant"] == "dense"]
    rows = []
    for b in dense:
        rows += A.block_rows(av, b["tStart"], b["tEnd"])
    for arm in ("leftArm", "rightArm"):
        r = sorted([x for x in rows if x["arm"] == arm], key=lambda x: x["frame"])
        buckets = {"straight <15": [], "shallow 15-40": [], "bent >40": []}
        for i in range(1, len(r)):
            d0 = r[i]["roll"] - r[i - 1]["roll"]
            d = abs((d0 + 180.0) % 360.0 - 180.0)
            bend = min(r[i]["bend"], r[i - 1]["bend"])
            key = ("straight <15" if bend < 15 else ("shallow 15-40" if bend < 40 else "bent >40"))
            buckets[key].append(d)
        print("   %s" % arm)
        for k in ("straight <15", "shallow 15-40", "bent >40"):
            v = buckets[k]
            if not v:
                print("      %-16s n=0" % k)
                continue
            print("      %-16s n=%5d  roll step mean=%6.2f p95=%6.2f p99=%6.2f max=%6.1f  >90deg: %d"
                  % (k, len(v), A.mean(v), A.pct(v, 95), A.pct(v, 99), max(v),
                     sum(1 for x in v if x > 90)))
        src = {}
        for x in r:
            key = ("straight <15" if x["bend"] < 15 else ("shallow 15-40" if x["bend"] < 40 else "bent >40"))
            src.setdefault(key, {})
            src[key][x["src"]] = src[key].get(x["src"], 0) + 1
        for k in ("straight <15", "shallow 15-40", "bent >40"):
            if k in src:
                tot = sum(src[k].values())
                print("      %-16s normalSource: %s" % (k, ", ".join(
                    "%s=%.0f%%" % (kk, 100.0 * vv / tot) for kk, vv in sorted(src[k].items()))))

    # ---------------- D) dense single-frame direction pops ----------------------------------
    print()
    print("=" * 108)
    print(" D) SINGLE-FRAME AVATAR POPS (model_log, every render frame ~250 fps). A step this")
    print("    large in one render frame is a visible snap, not motion.")
    print("=" * 108)
    print("   %-26s %-8s | %s" % ("motion", "variant", "forearm dir steps: n  >5deg  >20deg  >45deg   max"))
    for b in blocks:
        w = [r for r in ml if b["tStart"] <= r.get("tApply", 0) <= b["tEnd"]]
        if not w:
            continue
        steps = []
        for key in ("llowF", "rlowF"):
            prev = None
            for r in w:
                v = r.get(key)
                if not v:
                    continue
                v = tuple(v)
                if prev is not None:
                    x = A.ang(prev, v)
                    if x is not None:
                        steps.append(x)
                prev = v
        if not steps:
            continue
        print("   %-26s %-8s | %6d %6d %7d %7d %8.1f"
              % ("%s %s" % (b["block"], b["title"][:22]), b["variant"], len(steps),
                 sum(1 for x in steps if x > 5), sum(1 for x in steps if x > 20),
                 sum(1 for x in steps if x > 45), max(steps)))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
