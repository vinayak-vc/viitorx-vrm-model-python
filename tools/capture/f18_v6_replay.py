#!/usr/bin/env python3
"""
F-18 section 16 - V6 TorsoYawGuard in OBSERVATION MODE over the portrait yaw stream.

The Unity Editor was not running, so the shipped C# could not be invoked directly as it was for
V6's own validation. This is a line-by-line port of Runtime/Retargeting/TorsoYawGuard.cs, and it is
VERIFIED rather than trusted: it is first replayed over the exact frames V6 published numbers for
(oak_v4_evidence/f15/v6_replay_frames.csv) and the result is compared against the V6 report's
figures. If the port does not reproduce those, its portrait numbers are not reported.

No threshold is modified. WrapGuardDeg = 150, ValidatedRangeDeg = 60, exactly as shipped.

Output: oak_v4_evidence/f18/v6_observation.txt
"""
import io
import math
import os
import sys

import numpy as np

OUT = os.path.join("oak_v4_evidence", "f18", "v6_observation.txt")
WRAP_DEG = 150.0
RANGE_DEG = 60.0
L = []


def say(s=""):
    print(s)
    L.append(s)


class GuardState(object):
    __slots__ = ("HasValid", "HeldHip", "HeldSh", "HoldStreak", "FramesTotal", "FramesHeld",
                 "FramesOutOfRange", "LongestHold", "HoldRuns", "HoldFramesSum")

    def __init__(self):
        self.HasValid = False
        self.HeldHip = 0.0
        self.HeldSh = 0.0
        self.HoldStreak = 0
        self.FramesTotal = 0
        self.FramesHeld = 0
        self.FramesOutOfRange = 0
        self.LongestHold = 0
        self.HoldRuns = 0
        self.HoldFramesSum = 0


def clamp_to_range(rad):
    lim = math.radians(RANGE_DEG)
    return max(-lim, min(lim, rad))


def evaluate(hip, sh, trunk_fresh, st):
    """Port of TorsoYawGuard.Evaluate. Inputs in RADIANS, as the shipped method takes them."""
    st.FramesTotal += 1
    finite = not (math.isnan(hip) or math.isinf(hip) or math.isnan(sh) or math.isinf(sh))
    wrap = math.radians(WRAP_DEG)
    wrapped = (not finite) or abs(hip) > wrap or abs(sh) > wrap

    if wrapped or not trunk_fresh:
        if not st.HasValid:
            return dict(HasValue=False, State="NoValue", Hip=0.0, Sh=0.0, HoldFrames=0)
        if st.HoldStreak == 0:
            st.HoldRuns += 1
        st.HoldStreak += 1
        st.FramesHeld += 1
        st.HoldFramesSum += 1
        if st.HoldStreak > st.LongestHold:
            st.LongestHold = st.HoldStreak
        return dict(HasValue=True, State="WrapGuarded" if wrapped else "Valid",
                    Hip=clamp_to_range(st.HeldHip), Sh=clamp_to_range(st.HeldSh),
                    HoldFrames=st.HoldStreak)

    st.HasValid = True
    st.HeldHip = hip
    st.HeldSh = sh
    st.HoldStreak = 0
    lim = math.radians(RANGE_DEG)
    beyond = abs(hip) > lim or abs(sh) > lim
    if beyond:
        st.FramesOutOfRange += 1
    return dict(HasValue=True, State="OutOfRange" if beyond else "Valid",
                Hip=clamp_to_range(hip), Sh=clamp_to_range(sh), HoldFrames=0)


def replay(samples, fps):
    """samples: [(hip_deg, sh_deg, fresh)] -> tally + rendered-step statistics."""
    st = GuardState()
    states = {"Valid": 0, "WrapGuarded": 0, "OutOfRange": 0, "NoValue": 0}
    out, runs, cur = [], [], 0
    for hip, sh, fresh in samples:
        r = evaluate(math.radians(hip), math.radians(sh), fresh, st)
        states[r["State"]] += 1
        out.append(math.degrees(r["Sh"]))
        if r["HoldFrames"] > 0:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    y = np.array(out)
    d = np.abs(np.diff(y)) if y.size > 1 else np.array([0.0])
    n = max(1, len(samples))
    return dict(n=n, states=states,
                pct={k: 100.0 * v / n for k, v in states.items()},
                max_step=float(d.max()), p95_step=float(np.percentile(d, 95)),
                steps10=int((d > 10).sum()), steps30=int((d > 30).sum()),
                held=st.FramesHeld, held_pct=100.0 * st.FramesHeld / n,
                runs=len(runs), longest=st.LongestHold,
                longest_s=st.LongestHold / fps if fps else float("nan"),
                rendered_p50=float(np.median(np.abs(y))),
                rendered_max=float(np.abs(y).max()))


say("=" * 100)
say("F-18 SECTION 16 - V6 TorsoYawGuard, OBSERVATION ONLY (no threshold changed)")
say("=" * 100)
say("WrapGuardDeg = %.0f   ValidatedRangeDeg = %.0f   (as shipped)" % (WRAP_DEG, RANGE_DEG))

# ---------------------------------------------------------------- verify the port
say()
say("-" * 100)
say("PORT VERIFICATION - replay the exact frames V6 published numbers for")
say("-" * 100)
ref = os.path.join("oak_v4_evidence", "f15", "v6_replay_frames.csv")
verified = False
if os.path.exists(ref):
    import csv
    rows = list(csv.DictReader(io.open(ref, encoding="utf-8")))
    say("  %s : %d rows, columns %s" % (os.path.basename(ref), len(rows),
                                        ",".join(list(rows[0].keys())[:8])))
    hk = next((k for k in rows[0] if "hip" in k.lower() and "yaw" in k.lower()), None)
    sk = next((k for k in rows[0] if ("sh" in k.lower() or "spine" in k.lower()) and "yaw" in k.lower()), None)
    fk = next((k for k in rows[0] if "fresh" in k.lower()), None)
    say("  detected columns: hip=%s shoulder=%s fresh=%s" % (hk, sk, fk))
    if hk and sk:
        def f(v):
            try:
                return float(v)
            except Exception:
                return float("nan")

        def tf(v):
            return str(v).strip().lower() in ("1", "true", "yes")

        samples = [(f(r[hk]), f(r[sk]), tf(r[fk]) if fk else True) for r in rows]
        mx = max(abs(x) for a, b, _ in samples for x in (a, b) if x == x)
        if mx < 7.0:                      # values look like radians -> convert for the deg API
            samples = [(math.degrees(a), math.degrees(b), c) for a, b, c in samples]
            say("  input looked like RADIANS (max |v| = %.3f); converted to degrees" % mx)
        res = replay(samples, 25.5)
        say("  replayed: held %d (%.2f %%), runs %d, longest %d frames = %.2f s"
            % (res["held"], res["held_pct"], res["runs"], res["longest"], res["longest_s"]))
        say("  V6 report published: held 896 (4.31 %), 11 runs, longest 12.33 s")
        verified = (abs(res["held_pct"] - 4.31) < 1.0 and abs(res["longest_s"] - 12.33) < 1.5)
        say("  PORT %s" % ("VERIFIED - reproduces the published figures"
                           if verified else "DID NOT REPRODUCE the published figures"))
else:
    say("  reference file not found: %s" % ref)

# ---------------------------------------------------------------- portrait observation
say()
say("-" * 100)
say("PORTRAIT OBSERVATION")
say("-" * 100)
say("CAVEAT, stated rather than buried: F-18's capture logs the SHOULDER yaw but not a hip yaw")
say("(hip depths were not retained), and TrunkGate's Fresh flag is not reproduced offline. The")
say("replay therefore feeds hipYaw = shoulderYaw and trunkFresh = true. That measures exactly what")
say("section 16 asks - whether the PORTRAIT YAW STREAM triggers new wrap/range guarding - but it")
say("cannot measure TrunkGate-driven holds, so the hold figures are a LOWER BOUND.")

if not verified:
    say()
    say("  Port unverified - portrait figures withheld.")
else:
    import csv
    src = os.path.join("oak_v4_evidence", "f18", "v6_replay_input.csv")
    rows = list(csv.DictReader(io.open(src, encoding="utf-8")))
    say()
    say("  %-22s %7s %8s %8s %9s %9s %9s %8s" %
        ("group", "n", "Valid%", "OutOfR%", "Wrap%", "maxStep", "steps>10", "held%"))
    groups = {"ALL portrait": rows}
    for d in ("078", "080", "090", "100"):
        g = [r for r in rows if r["block"].endswith("@" + d)]
        if g:
            groups["at %.2f m" % (int(d) / 100.0)] = g
    sqr = [r for r in rows if r["block"].startswith("sq")]
    if sqr:
        groups["square blocks only"] = sqr
    mv = [r for r in rows if "@090" in r["block"] and not r["block"].startswith(
        ("sq", "left", "right"))]
    if mv:
        groups["interaction @0.90 m"] = mv
    for name, g in groups.items():
        s = [(float(r["yaw"]), float(r["yaw"]), True) for r in g]
        res = replay(s, 21.6)
        say("  %-22s %7d %8.2f %8.2f %9.2f %9.2f %9d %8.2f"
            % (name, res["n"], res["pct"]["Valid"], res["pct"]["OutOfRange"],
               res["pct"]["WrapGuarded"], res["max_step"], res["steps10"], res["held_pct"]))
    say()
    allres = replay([(float(r["yaw"]), float(r["yaw"]), True) for r in rows], 21.6)
    say("  whole portrait stream: longest hold %d frames = %.2f s over %d hold runs"
        % (allres["longest"], allres["longest_s"], allres["runs"]))
    say("  rendered |yaw| median %.2f deg, max %.2f deg (the clamp caps it at 60)"
        % (allres["rendered_p50"], allres["rendered_max"]))
    sq = replay([(float(r["yaw"]), float(r["yaw"]), True) for r in sqr], 21.6)
    say()
    say("  SQUARE BLOCKS - the public-mirror resting case:")
    say("    Valid %.2f %%   OutOfRange %.2f %%   WrapGuarded %.2f %%   held %.2f %%"
        % (sq["pct"]["Valid"], sq["pct"]["OutOfRange"], sq["pct"]["WrapGuarded"], sq["held_pct"]))
    say("    max guarded step %.2f deg, steps > 10 deg: %d" % (sq["max_step"], sq["steps10"]))

io.open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\nwrote %s" % OUT)
