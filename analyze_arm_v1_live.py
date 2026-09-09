#!/usr/bin/env python3
"""ARM RETARGET V1 -- live A/B analysis (docs/UNITY_ARM_RETARGET_V1_LIVE_VALIDATION_2026-09-09.md).

Reads three streams and joins them on wall-clock time:

  1. Unity's rolling log       -- [RETARGET-TRACE] and [ARM-V1-TRACE], HH:MM:SS.fff local time
  2. pipeline_logs_*/blocks.json -- motion block windows + which arm branch was live
  3. pipeline_logs_*/model_log.jsonl -- per-render-frame avatar state (elbow angles, forearm
     forward vectors, bone lengths, LimbGate states, tApply)

TWO INDEPENDENT ERROR MEASURES, deliberately kept apart:

  RETARGET-TRACE errDeg   want = the RAW landmark direction (exactly what PoseDebugSkeleton draws),
                          got  = the avatar's actual bone direction.
                          -> "does the VRM follow the green skeleton". This is the A/B metric.

  ARM-V1-TRACE  upperErrDeg / forearmErrDeg
                          target = the aim solver's OWN intent, got = the achieved bone.
                          -> solve->rig->bone fidelity only. NEAR ZERO HERE PROVES NOTHING ABOUT
                             UPSTREAM TRACKING. Reported separately, never as correctness.

    python analyze_arm_v1_live.py --dir pipeline_logs_armv1
"""
import argparse
import datetime
import io
import json
import math
import os
import re

ARM_BONES = ("leftUpperArm", "leftLowerArm", "rightUpperArm", "rightLowerArm")
LEG_BONES = ("leftUpperLeg", "leftLowerLeg", "rightUpperLeg", "rightLowerLeg")

RE_RT = re.compile(
    r"^(\d\d):(\d\d):(\d\d)\.(\d\d\d) \[\w+\] \[RETARGET-TRACE\] frame=(\d+) bone=(\S+) "
    r"want=\(([-\d.,]+)\) got=\(([-\d.,]+)\) errDeg=([-\d.]+)")
RE_AV = re.compile(
    r"^(\d\d):(\d\d):(\d\d)\.(\d\d\d) \[\w+\] \[ARM-V1-TRACE\] frame=(\d+) arm=(\S+) "
    r"targetUpperDir=\(([-\d.,]+)\) gotUpperDir=\(([-\d.,]+)\) upperErrDeg=([-\d.]+) "
    r"targetForearmDir=\(([-\d.,]+)\) gotForearmDir=\(([-\d.,]+)\) forearmErrDeg=([-\d.]+) "
    r"bendDeg=([-\d.]+) rollDeg=([-\d.]+) bendNormal=\(([-\d.,]+)\) normalSource=(\S+)")
RE_REST = re.compile(r"^(\d\d):(\d\d):(\d\d)\.(\d\d\d) \[\w+\] \[ARM-V1\] rest basis measured: (.*)$")


def pct(v, p):
    if not v:
        return 0.0
    s = sorted(v)
    return s[int(round((len(s) - 1) * p / 100.0))]


def mean(v):
    return (sum(v) / len(v)) if v else 0.0


def vec(s):
    p = [float(x) for x in s.split(",")]
    return (p[0], p[1], p[2])


def ang(a, b):
    """Unsigned angle between two vectors, degrees. Wrap-free -- no Euler involved."""
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-9 or nb < 1e-9:
        return None
    d = sum(x * y for x, y in zip(a, b)) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, d))))


def parse_unity_log(path, day):
    """Return (retarget rows, armv1 rows, rest-basis lines). Times -> epoch seconds."""
    rt, av, rest = [], [], []
    prev_h = None
    day_off = 0
    for ln in io.open(path, encoding="utf-8", errors="replace"):
        m = RE_RT.match(ln)
        m2 = RE_AV.match(ln) if not m else None
        m3 = RE_REST.match(ln) if not (m or m2) else None
        mm = m or m2 or m3
        if not mm:
            continue
        h, mi, s, ms = int(mm.group(1)), int(mm.group(2)), int(mm.group(3)), int(mm.group(4))
        # The log has no date; roll the day forward if the clock wraps past midnight.
        if prev_h is not None and h < prev_h - 12:
            day_off += 1
        prev_h = h
        t = datetime.datetime.combine(day + datetime.timedelta(days=day_off),
                                      datetime.time(h, mi, s, ms * 1000)).timestamp()
        if m:
            rt.append({"t": t, "frame": int(m.group(5)), "bone": m.group(6),
                       "want": vec(m.group(7)), "got": vec(m.group(8)),
                       "err": float(m.group(9))})
        elif m2:
            av.append({"t": t, "frame": int(m2.group(5)), "arm": m2.group(6),
                       "tUp": vec(m2.group(7)), "gUp": vec(m2.group(8)),
                       "upErr": float(m2.group(9)),
                       "tFore": vec(m2.group(10)), "gFore": vec(m2.group(11)),
                       "foreErr": float(m2.group(12)),
                       "bend": float(m2.group(13)), "roll": float(m2.group(14)),
                       "normal": vec(m2.group(15)), "src": m2.group(16)})
        else:
            rest.append((t, m3.group(5)))
    return rt, av, rest


def load_model_log(path, lo=None, hi=None):
    """Stream-filter by tApply: the log is ~100 MB and only the block windows matter."""
    rows = []
    if not os.path.exists(path):
        return rows
    for ln in io.open(path, encoding="utf-8", errors="replace"):
        ln = ln.strip()
        if not ln:
            continue
        if lo is not None:
            # cheap pre-filter before paying for json.loads
            m = re.search(r'"tApply":(\d+\.\d+)', ln)
            if m:
                t = float(m.group(1))
                if t < lo or t > hi:
                    continue
        try:
            rows.append(json.loads(ln))
        except ValueError:
            continue
    return rows


def in_win(t, a, b):
    return a <= t <= b


def block_rows(rows, a, b):
    return [r for r in rows if in_win(r["t"], a, b)]


def summarise_err(rows, bones):
    """errDeg stats over the given bones."""
    out = {}
    for bn in bones:
        v = [r["err"] for r in rows if r["bone"] == bn]
        out[bn] = (len(v), mean(v), pct(v, 95), max(v) if v else 0.0)
    allv = [r["err"] for r in rows if r["bone"] in bones]
    out["ALL"] = (len(allv), mean(allv), pct(allv, 95), max(allv) if allv else 0.0)
    return out


def asymmetry(rows):
    """|left errDeg - right errDeg| paired per trace frame -- the one-arm-only signature."""
    byf = {}
    for r in rows:
        if r["bone"] in ARM_BONES:
            byf.setdefault(r["frame"], {})[r["bone"]] = r["err"]
    up, lo = [], []
    for f, d in byf.items():
        if "leftUpperArm" in d and "rightUpperArm" in d:
            up.append(abs(d["leftUpperArm"] - d["rightUpperArm"]))
        if "leftLowerArm" in d and "rightLowerArm" in d:
            lo.append(abs(d["leftLowerArm"] - d["rightLowerArm"]))
    return up, lo


def roll_stats(av_rows, arm):
    """Roll continuity between consecutive TRACE samples.

    rollDeg is an ANGLE ON A CIRCLE, so the raw difference is wrong at the +/-180 seam: a roll
    stepping from -179 to +178 is a 3 deg move, not a 357 deg one. Every step below is therefore
    wrapped into (-180, 180]. `wrapped` is the honest continuity measure; `raw` is kept only so
    the report can show how many of the apparent "flips" were seam artefacts.
    A flip = a wrapped step > 90 deg, i.e. a genuine helicopter/pop."""
    r = [x for x in av_rows if x["arm"] == arm]
    r.sort(key=lambda x: x["frame"])
    steps, raw, flips, seam = [], [], 0, 0
    for i in range(1, len(r)):
        d0 = r[i]["roll"] - r[i - 1]["roll"]
        d = abs((d0 + 180.0) % 360.0 - 180.0)
        raw.append(abs(d0))
        steps.append(d)
        if d > 90.0:
            flips += 1
        elif abs(d0) > 90.0:
            seam += 1
    return r, steps, flips, raw, seam


def dir_steps(av_rows, arm, key):
    """Frame-to-frame ANGULAR step of an achieved bone direction (wrap-free)."""
    r = [x for x in av_rows if x["arm"] == arm]
    r.sort(key=lambda x: x["frame"])
    out = []
    for i in range(1, len(r)):
        a = ang(r[i - 1][key], r[i][key])
        if a is not None:
            out.append(a)
    return out


def model_stats(rows, a, b):
    """Dense per-render-frame avatar state inside a window."""
    w = [r for r in rows if a <= r.get("tApply", 0) <= b]
    if not w:
        return None
    st = {"n": len(w)}
    for side, key in (("l", "l"), ("r", "r")):
        st["elbow_" + side] = [r["elbowAng"][key] for r in w if "elbowAng" in r]
    # forearm forward vector: wrap-free per-frame direction step -> jitter, and flips
    for side, key in (("l", "llowF"), ("r", "rlowF")):
        steps = []
        prev = None
        for r in w:
            v = r.get(key)
            if not v:
                continue
            v = tuple(v)
            if prev is not None:
                x = ang(prev, v)
                if x is not None:
                    steps.append(x)
            prev = v
        st["fstep_" + side] = steps
    st["upArm"] = [r["boneLen"]["upArm"] for r in w if "boneLen" in r]
    st["foreArm"] = [r["boneLen"]["foreArm"] for r in w if "boneLen" in r]
    held = sum(1 for r in w if r.get("gate", {}).get("hLArm") or r.get("gate", {}).get("hRArm"))
    st["armHeldPct"] = 100.0 * held / len(w)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="pipeline_logs_armv1")
    ap.add_argument("--unity-log", default="")
    ap.add_argument("--date", default="", help="YYYY-MM-DD of the run (default: today)")
    a = ap.parse_args()

    day = (datetime.date.fromisoformat(a.date) if a.date else datetime.date.today())
    ulog = a.unity_log or os.path.join(
        os.path.expanduser("~"), "Documents", "MyMirror", "Logs",
        "virtual-mirror-%s.log" % day.strftime("%Y%m%d"))
    if not os.path.exists(ulog):
        print("[ERROR] unity log not found: %s" % ulog)
        return 1

    # --dir accepts a comma-separated list. A later directory OVERRIDES an earlier one for the
    # same (variant, block) pair: motions 2-5 had to be recaptured after an 88 s Editor stall
    # (caused by an AssetDatabase refresh from the screenshot tool, not by the pipeline), so the
    # re-run directory supersedes the ruined windows while the rest of the first run stands.
    dirs = [d.strip() for d in a.dir.split(",") if d.strip()]
    chosen, order, overridden = {}, [], []
    for d in dirs:
        meta = json.load(io.open(os.path.join(d, "blocks.json"), encoding="utf-8"))
        for b in meta["blocks"]:
            key = (b["variant"], b["block"])
            if key in chosen:
                overridden.append((key, chosen[key]["dir"], d))
            else:
                order.append(key)
            b = dict(b)
            b["dir"] = d
            chosen[key] = b
    blocks = [chosen[k] for k in order]
    rt, av, rest = parse_unity_log(ulog, day)
    ml = []
    for d in dirs:
        bs = [b for b in blocks if b["dir"] == d]
        if not bs:
            continue
        lo = min(b["tStart"] for b in bs) - 5.0
        hi = max(b["tEnd"] for b in bs) + 5.0
        ml += load_model_log(os.path.join(d, "model_log.jsonl"), lo, hi)

    print("=" * 104)
    print(" ARM RETARGET V1 -- LIVE VALIDATION  (%s)" % day.isoformat())
    print("=" * 104)
    print(" unity log      : %s" % ulog)
    print(" retarget-trace : %d rows   arm-v1-trace: %d rows   model_log: %d rows"
          % (len(rt), len(av), len(ml)))
    print(" capture dirs   : %s" % ", ".join(dirs))
    for key, was, now in overridden:
        print(" SUPERSEDED     : variant=%s motion=%s  %s -> %s" % (key[0], key[1], was, now))
    for t, s in rest:
        print(" rest basis     : %s" % s)
    print()

    variants = []
    for b in blocks:
        if b["variant"] not in variants:
            variants.append(b["variant"])

    # ---------- 1) RETARGETING CORRECTNESS: avatar bone vs RAW landmark direction -----------
    print("=" * 104)
    print(" 1) RETARGETING CORRECTNESS -- angle between the avatar's arm bone and the RAW")
    print("    landmark direction the green PoseDebugSkeleton draws. Lower is better.")
    print("=" * 104)
    hdr = "   %-28s |" % "motion"
    for v in variants:
        hdr += " %-27s |" % ("%s  mean / p95 / max" % v.upper())
    print(hdr)
    per_block = {}
    for b in blocks:
        rows = block_rows(rt, b["tStart"], b["tEnd"])
        per_block[(b["variant"], b["block"])] = summarise_err(rows, ARM_BONES)
    seen = []
    for b in blocks:
        if b["block"] in seen:
            continue
        seen.append(b["block"])
        line = "   %-28s |" % ("%s %s" % (b["block"], b["title"][:24]))
        for v in variants:
            s = per_block.get((v, b["block"]), {}).get("ALL")
            line += (" %6.1f %6.1f %6.1f (n=%4d)|" % (s[1], s[2], s[3], s[0])) if s else " %-27s |" % "-"
        print(line)
    # whole-run totals per variant
    print("   " + "-" * 98)
    for v in variants:
        rows = []
        for b in blocks:
            if b["variant"] == v:
                rows += block_rows(rt, b["tStart"], b["tEnd"])
        s = summarise_err(rows, ARM_BONES)
        print("   %-10s ALL MOTIONS  n=%5d  mean=%6.2f  p95=%6.2f  max=%6.2f"
              % (v.upper(), s["ALL"][0], s["ALL"][1], s["ALL"][2], s["ALL"][3]))
        for bn in ARM_BONES:
            q = s[bn]
            print("               %-14s n=%5d  mean=%6.2f  p95=%6.2f  max=%6.2f"
                  % (bn, q[0], q[1], q[2], q[3]))

    # ---------- 2) CONTROL: legs must be identical across variants --------------------------
    print()
    print("=" * 104)
    print(" 2) WITHIN-RUN CONTROL -- LEG bones were not touched by this change. If the leg")
    print("    error moves as much as the arm error, the A/B is measuring the subject, not the code.")
    print("=" * 104)
    for v in variants:
        rows = []
        for b in blocks:
            if b["variant"] == v:
                rows += block_rows(rt, b["tStart"], b["tEnd"])
        s = summarise_err(rows, LEG_BONES)
        print("   %-10s LEGS  n=%5d  mean=%6.2f  p95=%6.2f  max=%6.2f"
              % (v.upper(), s["ALL"][0], s["ALL"][1], s["ALL"][2], s["ALL"][3]))

    # ---------- 3) L/R ASYMMETRY -- the "one arm only" failure ------------------------------
    print()
    print("=" * 104)
    print(" 3) LEFT/RIGHT ASYMMETRY -- |errDeg(left) - errDeg(right)| on the same trace frame.")
    print("    The SR.mp4 'both arms up, avatar raises one' failure lives here.")
    print("=" * 104)
    print("   %-28s | %-24s | %-24s" % ("motion", "upper arm  mean/p95/max", "forearm  mean/p95/max"))
    for v in variants:
        print("   -- variant %s" % v.upper())
        for b in blocks:
            if b["variant"] != v:
                continue
            up, lo = asymmetry(block_rows(rt, b["tStart"], b["tEnd"]))
            print("   %-28s | %7.1f %7.1f %7.1f | %7.1f %7.1f %7.1f"
                  % ("%s %s" % (b["block"], b["title"][:24]),
                     mean(up), pct(up, 95), max(up) if up else 0.0,
                     mean(lo), pct(lo, 95), max(lo) if lo else 0.0))
        allup, allo = [], []
        for b in blocks:
            if b["variant"] == v:
                u, l = asymmetry(block_rows(rt, b["tStart"], b["tEnd"]))
                allup += u
                allo += l
        print("   %-28s | %7.1f %7.1f %7.1f | %7.1f %7.1f %7.1f"
              % ("   ALL MOTIONS", mean(allup), pct(allup, 95), max(allup) if allup else 0.0,
                 mean(allo), pct(allo, 95), max(allo) if allo else 0.0))

    # ---------- 4) ARM-V1-TRACE: solve->bone fidelity + roll --------------------------------
    print()
    print("=" * 104)
    print(" 4) ARM-V1-TRACE -- solver intent vs achieved bone, and ROLL. NEW variant only")
    print("    (the trace is emitted only while the aim solver is live).")
    print("    NOTE: upperErrDeg/forearmErrDeg do NOT measure upstream tracking correctness.")
    print("=" * 104)
    for v in variants:
        vb = [b for b in blocks if b["variant"] == v]
        rows = []
        for b in vb:
            rows += block_rows(av, b["tStart"], b["tEnd"])
        if not rows:
            print("   variant %s: no ARM-V1-TRACE rows (expected when the old branch is live)" % v.upper())
            continue
        print("   -- variant %s (n=%d trace rows)" % (v.upper(), len(rows)))
        print("   %-26s | %-23s | %-23s | %-22s"
              % ("motion", "upperErrDeg mean/p95/max", "foreErrDeg mean/p95/max", "rollDeg step max / flips"))
        for b in vb:
            r = block_rows(av, b["tStart"], b["tEnd"])
            if not r:
                continue
            ue = [x["upErr"] for x in r]
            fe = [x["foreErr"] for x in r]
            mx, fl = 0.0, 0
            for arm in ("leftArm", "rightArm"):
                _rr, steps, flips, _raw, _seam = roll_stats(r, arm)
                mx = max(mx, max(steps) if steps else 0.0)
                fl += flips
            print("   %-26s | %7.1f %7.1f %7.1f | %7.1f %7.1f %7.1f | %10.1f %6d"
                  % ("%s %s" % (b["block"], b["title"][:22]),
                     mean(ue), pct(ue, 95), max(ue), mean(fe), pct(fe, 95), max(fe), mx, fl))
        ue = [x["upErr"] for x in rows]
        fe = [x["foreErr"] for x in rows]
        bd = [x["bend"] for x in rows]
        print("   %-26s | %7.1f %7.1f %7.1f | %7.1f %7.1f %7.1f |"
              % ("   ALL MOTIONS", mean(ue), pct(ue, 95), max(ue),
                 mean(fe), pct(fe, 95), max(fe)))
        print("      bendDeg  mean=%.1f p05=%.1f p95=%.1f   (BendSinMin=0.2 -> ~11.5 deg threshold)"
              % (mean(bd), pct(bd, 5), pct(bd, 95)))
        src = {}
        for x in rows:
            src[x["src"]] = src.get(x["src"], 0) + 1
        print("      normalSource: %s" % ", ".join("%s=%d (%.1f%%)" % (k, n, 100.0 * n / len(rows))
                                                   for k, n in sorted(src.items())))
        for arm in ("leftArm", "rightArm"):
            r, steps, flips, raw, seam = roll_stats(rows, arm)
            us = dir_steps(rows, arm, "gUp")
            fs = dir_steps(rows, arm, "gFore")
            print("      %-9s roll step (wrapped): max %6.1f  p95 %5.1f  p99 %5.1f  TRUE flips(>90) %d"
                  "   [raw-diff max %6.1f, +/-180 seam artefacts %d]"
                  % (arm, max(steps) if steps else 0.0, pct(steps, 95), pct(steps, 99), flips,
                     max(raw) if raw else 0.0, seam))
            print("      %-9s achieved dir step: upper p95 %5.1f max %6.1f | forearm p95 %5.1f max %6.1f"
                  % ("", pct(us, 95), max(us) if us else 0.0, pct(fs, 95), max(fs) if fs else 0.0))

    # ---------- 5) DENSE per-render-frame avatar state --------------------------------------
    print()
    print("=" * 104)
    print(" 5) DENSE AVATAR STATE (model_log, every render frame) -- elbow angle, forearm")
    print("    direction step (jitter/flips), bone-length constancy, arm gate holds.")
    print("=" * 104)
    print("   %-26s | %-19s | %-25s | %-17s | %s"
          % ("motion", "elbow L/R mean", "forearm dir step p95/max", "upArm/foreArm len", "arm gate held"))
    for v in variants:
        print("   -- variant %s" % v.upper())
        for b in blocks:
            if b["variant"] != v:
                continue
            st = model_stats(ml, b["tStart"], b["tEnd"])
            if not st:
                continue
            fl = st["fstep_l"] + st["fstep_r"]
            ua, fa = st["upArm"], st["foreArm"]
            print("   %-26s | %8.1f %8.1f | %11.2f %11.2f | %6.3f / %6.3f | %6.2f%%  (n=%d)"
                  % ("%s %s" % (b["block"], b["title"][:22]),
                     mean(st["elbow_l"]), mean(st["elbow_r"]),
                     pct(fl, 95), max(fl) if fl else 0.0,
                     mean(ua), mean(fa), st["armHeldPct"], st["n"]))
        rows = []
        for b in blocks:
            if b["variant"] == v:
                st = model_stats(ml, b["tStart"], b["tEnd"])
                if st:
                    rows.append(st)
        if rows:
            fl = [x for st in rows for x in st["fstep_l"] + st["fstep_r"]]
            ua = [x for st in rows for x in st["upArm"]]
            fa = [x for st in rows for x in st["foreArm"]]
            el = [x for st in rows for x in st["elbow_l"]]
            er = [x for st in rows for x in st["elbow_r"]]
            n = sum(st["n"] for st in rows)
            held = sum(st["armHeldPct"] * st["n"] for st in rows) / n
            print("   %-26s | %8.1f %8.1f | %11.2f %11.2f | %6.3f / %6.3f | %6.2f%%  (n=%d)"
                  % ("   ALL MOTIONS", mean(el), mean(er), pct(fl, 95), max(fl) if fl else 0.0,
                     mean(ua), mean(fa), held, n))
            print("      bone length spread: upArm %.5f m (min %.4f max %.4f), foreArm %.5f m"
                  % (max(ua) - min(ua), min(ua), max(ua), max(fa) - min(fa)))
            print("      elbow angle range : L %.1f..%.1f deg   R %.1f..%.1f deg"
                  % (min(el), max(el), min(er), max(er)))

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
