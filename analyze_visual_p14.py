#!/usr/bin/env python3
"""P1-4 VISUAL VALIDATION analyzer -- P1-3 baseline vs P1-4, per block.

Reads the two archived capture dirs produced by visual_p14_capture.py and reports,
per block and overall, the AVATAR-SIDE behaviour (Unity model_log) alongside the
tracker-side events (holds_log) so every P1-4 rejection/recovery can be correlated
with what the avatar actually did.

Nothing here modifies the pipeline; it is pure post-hoc measurement.

    python analyze_visual_p14.py --a pipeline_logs_p13 --b pipeline_logs_p14
"""
import argparse
import collections
import io
import json
import math
import os

# Angular-delta thresholds (degrees, per RENDERED frame). A "snap" is what the eye sees,
# so these are measured on Unity's render cadence, not the 21 fps pose cadence.
SNAP_LEVELS = (10.0, 20.0, 45.0)

# Bone forward vectors logged by AppBootstrap.WriteModelLog (all wrap-free unit vectors).
FWD_KEYS = ("hipsFwd", "lhandF", "rhandF", "llowF", "rlowF",
            "luplegF", "ruplegF", "llowlegF", "rlowlegF")

BONES = ("femur", "shin", "upArm", "foreArm")

# Anatomically implausible interior angles. Vector3.Angle is UNSIGNED (0..180), so this
# detects hyper-FLEXION only; hyperextension direction is not recoverable from this log.
KNEE_MIN_DEG = 25.0
ELBOW_MIN_DEG = 20.0


def load(path):
    out = []
    if not os.path.exists(path):
        return out
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = int(round((len(s) - 1) * p / 100.0))
    return s[k]


def ang(a, b):
    """Unsigned angle in degrees between two 3-vectors."""
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-9 or nb < 1e-9:
        return None
    d = sum(x * y for x, y in zip(a, b)) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, d))))


class Cap(object):
    """One capture pass: the sidecar log, Unity's received/applied logs, events, blocks."""

    def __init__(self, d):
        self.dir = d
        self.sender = load(os.path.join(d, "sender_log.jsonl"))
        self.recv = load(os.path.join(d, "recv_log.jsonl"))
        self.model = load(os.path.join(d, "model_log.jsonl"))
        self.holds = load(os.path.join(d, "holds_log.jsonl"))
        b = os.path.join(d, "blocks.json")
        self.meta = json.load(io.open(b, encoding="utf-8")) if os.path.exists(b) else {}
        self.blocks = self.meta.get("blocks", [])
        self.injections = self.meta.get("injections", [])
        # model_log has no epoch of its own except tApply; use it as the block key.
        for m in self.model:
            m["_t"] = m.get("tApply", 0.0)

    def window(self, rows, t0, t1, key="t"):
        return [r for r in rows if t0 <= r.get(key, 0.0) <= t1]


def model_metrics(rows):
    """Avatar-side metrics from consecutive RENDERED frames."""
    out = {}
    # --- per-bone rotation deltas -------------------------------------------------
    for k in FWD_KEYS:
        deltas = []
        prev = None
        for m in rows:
            v = m.get(k)
            if not v:
                prev = None
                continue
            if prev is not None:
                a = ang(prev, v)
                if a is not None:
                    deltas.append(a)
            prev = v
        d = {"p50": pct(deltas, 50), "p95": pct(deltas, 95),
             "p99": pct(deltas, 99), "max": max(deltas) if deltas else 0.0,
             "n": len(deltas)}
        for lv in SNAP_LEVELS:
            d["gt%g" % lv] = sum(1 for x in deltas if x > lv)
        out[k] = d
    # --- joint interior angles ----------------------------------------------------
    for grp, lo in (("kneeAng", KNEE_MIN_DEG), ("elbowAng", ELBOW_MIN_DEG)):
        for side in ("l", "r"):
            vals, jumps, prev = [], [], None
            for m in rows:
                g = m.get(grp) or {}
                v = g.get(side)
                if v is None or v <= 0.0:
                    prev = None
                    continue
                vals.append(v)
                if prev is not None:
                    jumps.append(abs(v - prev))
                prev = v
            out["%s.%s" % (grp, side)] = {
                "min": min(vals) if vals else 0.0,
                "max": max(vals) if vals else 0.0,
                "implausible": sum(1 for v in vals if v < lo),
                "n": len(vals),
                "jump_p99": pct(jumps, 99), "jump_max": max(jumps) if jumps else 0.0,
                "jump_gt20": sum(1 for x in jumps if x > 20.0),
            }
    # --- bone length consistency (a rotation retarget must keep these CONSTANT) ----
    for b in BONES:
        vals = [m["boneLen"][b] for m in rows
                if m.get("boneLen") and m["boneLen"].get(b, 0.0) > 0.0]
        if vals:
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            out["bone." + b] = {"mean": mean, "cv": (math.sqrt(var) / mean) if mean else 0.0,
                                "min": min(vals), "max": max(vals), "n": len(vals)}
    # --- P0 LimbGate holds --------------------------------------------------------
    n = max(1, len(rows))
    for g in ("hLArm", "hRArm", "hLLeg", "hRLeg"):
        held = sum(1 for m in rows if (m.get("gate") or {}).get(g, 0))
        out["gate." + g] = {"held": held, "pctHeld": 100.0 * held / n}
    out["_frames"] = len(rows)
    return out


def event_counts(rows):
    c = collections.Counter()
    for h in rows:
        c[h.get("event", "?")] += 1
    return c


def latency(cap, t0, t1):
    """camera -> avatar, joined by seq. First apply per seq (a later re-apply of the
    same seq is the render loop repeating, not new latency)."""
    cap_t = {}
    for s in cap.sender:
        if t0 <= s.get("t", 0) <= t1:
            cap_t[s["seq"]] = s["t"] - s.get("capToSendMs", 0.0) / 1000.0
    first = {}
    for m in cap.model:
        q = m.get("seq", -1)
        if q in cap_t and q not in first:
            first[q] = m["_t"]
    vals = [(first[q] - cap_t[q]) * 1000.0 for q in first]
    vals = [v for v in vals if 0.0 < v < 2000.0]
    return {"p50": pct(vals, 50), "p95": pct(vals, 95), "max": max(vals) if vals else 0.0,
            "n": len(vals)}


def fmt_block(name, mm, ev, lat, tms):
    rot = max((mm[k]["p99"] for k in FWD_KEYS if k in mm), default=0.0)
    rmax = max((mm[k]["max"] for k in FWD_KEYS if k in mm), default=0.0)
    snaps = sum(mm[k].get("gt45", 0) for k in FWD_KEYS if k in mm)
    imp = sum(mm[k]["implausible"] for k in mm if k.startswith(("kneeAng", "elbowAng")))
    jmp = sum(mm[k]["jump_gt20"] for k in mm if k.startswith(("kneeAng", "elbowAng")))
    lost = sum(v for k, v in ev.items() if k.endswith("->LOST"))
    return ("  %-22s f=%5d rot99=%6.2f rotMax=%7.2f snap45=%4d implaus=%3d "
            "angJump>20=%4d LOST=%3d lat50=%6.1fms trk50=%5.3fms"
            % (name, mm["_frames"], rot, rmax, snaps, imp, jmp, lost,
               lat["p50"], tms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="pipeline_logs_p13", help="baseline (P1-3) dir")
    ap.add_argument("--b", default="pipeline_logs_p14", help="P1-4 dir")
    a = ap.parse_args()

    A, B = Cap(a.a), Cap(a.b)
    print("=" * 100)
    print(" P1-4 VISUAL VALIDATION -- %s (%s)  vs  %s (%s)"
          % (a.a, A.meta.get("recovery", "?"), a.b, B.meta.get("recovery", "?")))
    print("=" * 100)

    for cap, tag in ((A, "A/P1-3"), (B, "B/P1-4")):
        span = cap.sender[-1]["t"] - cap.sender[0]["t"] if cap.sender else 0.0
        print(" %-7s frames sent=%d recv=%d (loss=%d)  rendered=%d  span=%.1fs  fps=%.1f"
              % (tag, len(cap.sender), len(cap.recv), len(cap.sender) - len(cap.recv),
                 len(cap.model), span, len(cap.sender) / span if span else 0))
    print()

    # ---------------- A) per-block timeline --------------------------------------
    print("-" * 100)
    print(" A) EVENT TIMELINE / PER-BLOCK METRICS")
    print("-" * 100)
    for cap, tag in ((A, "P1-3"), (B, "P1-4")):
        print(" [%s]" % tag)
        for blk in cap.blocks:
            t0, t1 = blk["tStart"], blk["tEnd"]
            mrows = cap.window(cap.model, t0, t1, "_t")
            hrows = cap.window(cap.holds, t0, t1, "t")
            srows = cap.window(cap.sender, t0, t1, "t")
            tms = pct([s.get("trackerMs", 0.0) for s in srows], 50)
            if not mrows:
                print("  %-22s (no rendered frames in window)" % blk["title"][:22])
                continue
            print(fmt_block(blk["title"][:22], model_metrics(mrows),
                            event_counts(hrows), latency(cap, t0, t1), tms))
        print()

    # ---------------- B) whole-run comparison ------------------------------------
    print("-" * 100)
    print(" B) BEFORE / AFTER  (whole run)")
    print("-" * 100)
    mA = model_metrics(A.model)
    mB = model_metrics(B.model)
    tA = A.sender[0]["t"], A.sender[-1]["t"]
    tB = B.sender[0]["t"], B.sender[-1]["t"]
    lA, lB = latency(A, tA[0], tA[1]), latency(B, tB[0], tB[1])

    print(" %-26s %14s %14s" % ("metric", "P1-3", "P1-4"))
    for k in FWD_KEYS:
        print(" %-26s %14s %14s" % ("rot p99  " + k,
                                    "%.2f deg" % mA[k]["p99"], "%.2f deg" % mB[k]["p99"]))
    for k in FWD_KEYS:
        print(" %-26s %14s %14s" % ("rot MAX  " + k,
                                    "%.2f deg" % mA[k]["max"], "%.2f deg" % mB[k]["max"]))
    for lv in SNAP_LEVELS:
        ka = sum(mA[k]["gt%g" % lv] for k in FWD_KEYS)
        kb = sum(mB[k]["gt%g" % lv] for k in FWD_KEYS)
        print(" %-26s %14s %14s" % ("snaps > %g deg/frame" % lv,
                                    "%d (%.3f%%)" % (ka, 100.0 * ka / max(1, mA["_frames"] * len(FWD_KEYS))),
                                    "%d (%.3f%%)" % (kb, 100.0 * kb / max(1, mB["_frames"] * len(FWD_KEYS)))))
    for grp in ("kneeAng.l", "kneeAng.r", "elbowAng.l", "elbowAng.r"):
        print(" %-26s %14s %14s" % (grp + " min",
                                    "%.1f deg" % mA[grp]["min"], "%.1f deg" % mB[grp]["min"]))
        print(" %-26s %14s %14s" % (grp + " implausible",
                                    "%d" % mA[grp]["implausible"], "%d" % mB[grp]["implausible"]))
        print(" %-26s %14s %14s" % (grp + " jump>20/frame",
                                    "%d" % mA[grp]["jump_gt20"], "%d" % mB[grp]["jump_gt20"]))
    for b in BONES:
        ka, kb = mA.get("bone." + b), mB.get("bone." + b)
        if ka and kb:
            print(" %-26s %14s %14s" % ("boneLen CV " + b,
                                        "%.5f" % ka["cv"], "%.5f" % kb["cv"]))
    for g in ("hLArm", "hRArm", "hLLeg", "hRLeg"):
        print(" %-26s %14s %14s" % ("P0 gate held " + g,
                                    "%.2f%%" % mA["gate." + g]["pctHeld"],
                                    "%.2f%%" % mB["gate." + g]["pctHeld"]))
    print(" %-26s %14s %14s" % ("latency p50",
                                "%.1f ms" % lA["p50"], "%.1f ms" % lB["p50"]))
    print(" %-26s %14s %14s" % ("latency p95",
                                "%.1f ms" % lA["p95"], "%.1f ms" % lB["p95"]))
    trA = [s.get("trackerMs", 0.0) for s in A.sender]
    trB = [s.get("trackerMs", 0.0) for s in B.sender]
    print(" %-26s %14s %14s" % ("tracker cost p50",
                                "%.4f ms" % pct(trA, 50), "%.4f ms" % pct(trB, 50)))
    print(" %-26s %14s %14s" % ("tracker cost p99",
                                "%.4f ms" % pct(trA, 99), "%.4f ms" % pct(trB, 99)))

    # ---------------- C) events ---------------------------------------------------
    print()
    print("-" * 100)
    print(" C) TRACKER / RECOVERY EVENTS (whole run)")
    print("-" * 100)
    eA, eB = event_counts(A.holds), event_counts(B.holds)
    for k in sorted(set(eA) | set(eB)):
        print(" %-26s %14d %14d" % (k, eA.get(k, 0), eB.get(k, 0)))

    # ---------------- D) injection correlation ------------------------------------
    print()
    print("-" * 100)
    print(" D) BLOCK 8 INJECTION CORRELATION  (the confidently-wrong path)")
    print("-" * 100)
    for cap, tag in ((A, "P1-3"), (B, "P1-4")):
        for inj in cap.injections:
            t0, sp = inj["t"], inj["spec"]
            pre = [s for s in cap.sender if t0 - 1.5 <= s["t"] < t0]
            post = [s for s in cap.sender if t0 <= s["t"] <= t0 + 3.0]
            if not pre or not post:
                print(" %s %-9s no window" % (tag, sp["mode"]))
                continue
            x0 = pre[-1]["kn"][1][0]
            err = max(abs(s["kn"][1][0] - x0) for s in post)
            hh = [h for h in cap.holds if t0 <= h.get("t", 0) <= t0 + 3.0]
            ec = event_counts(hh)
            rej = ec.get("GEOMETRIC_REJECT", 0)
            rec = ec.get("RECONSTRUCT", 0)
            # avatar side: worst right-leg rotation step during the same window
            mr = [m for m in cap.model if t0 <= m["_t"] <= t0 + 3.0]
            mm = model_metrics(mr) if len(mr) > 2 else None
            rl = mm["rlowlegF"]["max"] if mm else 0.0
            ka = mm["kneeAng.r"]["jump_max"] if mm else 0.0
            print(" %s %-9s injected %.2fm -> WIRE err %.4f m | GEOM_REJECT=%d RECONSTRUCT=%d"
                  " | avatar rlowleg max step %.2f deg, kneeAng.r max jump %.2f deg"
                  % (tag, sp["mode"], sp["meters"], err, rej, rec, rl, ka))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
