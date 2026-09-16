#!/usr/bin/env python3
"""F-27 HUMANIZED SKELETON - the before/after report.

WHAT MAKES THIS COMPARISON FAIR. Both streams are measured from the SAME frame, in the same pass, by
the same C# code (HumanizedSkeletonRecorder + PoseMetrics). The obvious alternative - run the session
once with the layer off and once with it on - compares two different takes of a moving subject and
credits the layer with the difference between them.

THE SIX THINGS THE LAYER CLAIMS, and the column that tests each:
  * large bone jumps        -> JUMP   (largest landmark displacement between consecutive poses)
  * impossible joint angles -> HINGE  (elbow below 25 deg, knee below 30 deg - see HumanBodyModel)
  * visible body collapse   -> MINHIP (closest a non-hip joint came to the mid-hip; the frame is
                                       hip-centred, so a wrist at 0.02 m IS the collapse)
  * fixed bone lengths      -> BONE   (worst deviation from that stream's OWN sliding median)
  * missing-joint behaviour -> the HELD and rejection columns of the stage table
  * recovery behaviour      -> the RECOV column
  * response latency        -> UNTOUCHED: the share of frames in which NO temporal stage fired. A
                               frame nothing temporal touched cannot have been delayed by this layer.

A NOTE ON WHAT THIS CANNOT SHOW. It measures the POSE, not the avatar. Whether a more human pose
produces a more human avatar is F-26's fidelity metric's question, and is deliberately kept separate:
conflating the two is how "the numbers look good" survived alongside an avatar that was visibly wrong.

    .venv\\Scripts\\python.exe f27_humanized_analyze.py --in ..\\docs\\evidence\\f27\\seg_all_stages.jsonl
    .venv\\Scripts\\python.exe f27_humanized_analyze.py --selftest
"""
import argparse
import io
import json
import sys

# Anatomical floors, mirrored from HumanBodyModel so the offline report and the live layer cannot
# drift apart silently. An angle below these is not a pose a body can make.
ELBOW_MIN_DEG = 25.0
KNEE_MIN_DEG = 30.0

# A non-hip joint nearer than this to the mid-hip is the collapse signature, not a body part.
COLLAPSE_M = 0.05

BONE_NAMES = ["L thigh", "L shin", "R thigh", "R shin", "L upperarm", "L forearm",
              "R upperarm", "R forearm", "shoulder span", "hip span"]


def load(path):
    rows = []
    try:
        for line in io.open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    except IOError:
        print("cannot read %s" % path)
        sys.exit(2)
    return rows


def quantile(sorted_vals, q):
    if not sorted_vals:
        return None
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def stream_stats(rows, key, warmup):
    """Everything measured on ONE stream. `warmup` frames are skipped: the bone-length medians on both
    sides need a window before their deviations mean anything, and scoring a layer on its own warm-up
    would flatter or damn it for no reason."""
    jumps, bones = [], []
    hinge_bad = 0
    worst_elbow, worst_knee = 180.0, 180.0
    min_hip = None
    collapses = 0
    nans = 0
    n = 0
    for r in rows[warmup:]:
        s = r.get(key)
        if not s:
            continue
        n += 1
        jumps.append(float(s.get("jump", 0.0)))
        bones.append(float(s.get("bone", 0.0)))
        for k in ("elbowL", "elbowR"):
            v = float(s.get(k, 180.0))
            worst_elbow = min(worst_elbow, v)
            if v < ELBOW_MIN_DEG:
                hinge_bad += 1
        for k in ("kneeL", "kneeR"):
            v = float(s.get(k, 180.0))
            worst_knee = min(worst_knee, v)
            if v < KNEE_MIN_DEG:
                hinge_bad += 1
        mh = float(s.get("minhip", 0.0))
        min_hip = mh if min_hip is None else min(min_hip, mh)
        if mh < COLLAPSE_M:
            collapses += 1
        nans += int(s.get("nan", 0))
    jumps.sort()
    bones.sort()
    return dict(n=n,
                jump_med=quantile(jumps, 0.5), jump_p99=quantile(jumps, 0.99),
                jump_max=(jumps[-1] if jumps else None),
                bone_med=quantile(bones, 0.5), bone_p99=quantile(bones, 0.99),
                bone_max=(bones[-1] if bones else None),
                hinge_bad=hinge_bad, worst_elbow=worst_elbow, worst_knee=worst_knee,
                min_hip=min_hip, collapses=collapses, nans=nans)


def stage_stats(rows, warmup):
    keys = ["held", "vclamp", "bone", "hinge", "cone", "knee", "twist", "recov",
            "rejOrigin", "rejFar", "rejNan", "rejConf", "untouched", "boneOnly"]
    total = dict((k, 0) for k in keys)
    frames_with = dict((k, 0) for k in keys)
    n = 0
    for r in rows[warmup:]:
        st = r.get("stage")
        if not st:
            continue
        n += 1
        for k in keys:
            v = int(st.get(k, 0))
            total[k] += v
            if v:
                frames_with[k] += 1
    # Computed HERE rather than read from the layer's own FramesUntouched counter, which is
    # confounded: the sidecar does not emit landmarks 1-10 (the face cluster) at all, so a
    # "rejected: not emitted" fires on every single pose and the C# counter never records an
    # untouched frame. Those landmarks are not in the bone table and are passed straight through -
    # nothing is done to them - so counting them as an intervention understates the result badly
    # (0.0 % where the true figure is ~99 %). What matters for LATENCY is only the stages that carry
    # state between frames.
    temporal = ["held", "vclamp", "recov"]
    structural = ["hinge", "cone", "knee", "twist"]
    latency_free = 0
    fully_untouched = 0
    for r in rows[warmup:]:
        st = r.get("stage")
        if not st:
            continue
        if any(int(st.get(k, 0)) for k in temporal):
            continue
        latency_free += 1
        if not any(int(st.get(k, 0)) for k in structural) and not int(st.get("bone", 0)):
            fully_untouched += 1
    return n, total, frames_with, latency_free, fully_untouched


def fmt(v, digits=4, units=""):
    if v is None:
        return "n/a"
    return ("%." + str(digits) + "f") % v + units


def delta(before, after, lower_is_better=True):
    if before is None or after is None:
        return ""
    if before == 0 and after == 0:
        return "  ="
    if before == 0:
        return "  WORSE" if (after > 0) == lower_is_better else "  better"
    pct = 100.0 * (after - before) / abs(before)
    if abs(pct) < 1.0:
        # A sub-1 % move is noise. Printing "-0% better" next to it would dress up a null result as a
        # win, which is precisely the reading this project has been burned by.
        return "  ="
    good = (pct < 0) if lower_is_better else (pct > 0)
    return "  %+.0f%% %s" % (pct, "better" if good else "WORSE")


def report(rows, warmup):
    raw = stream_stats(rows, "raw", warmup)
    hum = stream_stats(rows, "hum", warmup)
    n, total, frames_with, latency_free, fully_untouched = stage_stats(rows, warmup)

    print("=" * 104)
    print(" F-27 HUMANIZED SKELETON - before / after, measured from the SAME frames in ONE pass")
    print(" %d poses recorded, first %d skipped as warm-up, %d analysed" % (len(rows), warmup, raw["n"]))
    print("=" * 104)
    print(" %-42s %14s %14s %s" % ("measurement", "BEFORE (raw)", "AFTER (human)", "change"))
    print("-" * 104)

    def row(label, key, digits=4, units="m", lower_better=True):
        b, a = raw[key], hum[key]
        print(" %-42s %14s %14s %s"
              % (label, fmt(b, digits, units), fmt(a, digits, units),
                 delta(b, a, lower_better)))

    print(" LARGE BONE JUMPS - landmark displacement between consecutive poses")
    row("   median", "jump_med")
    row("   p99", "jump_p99")
    row("   worst single frame", "jump_max")
    print("")
    print(" FIXED BONE LENGTHS - deviation from that stream's own sliding median")
    row("   median", "bone_med")
    row("   p99", "bone_p99")
    row("   worst single frame", "bone_max")
    print("")
    print(" IMPOSSIBLE JOINT ANGLES - elbow < %.0f deg or knee < %.0f deg" % (ELBOW_MIN_DEG, KNEE_MIN_DEG))
    row("   joint-frames below the anatomical floor", "hinge_bad", 0, "", True)
    row("   tightest elbow seen", "worst_elbow", 1, " deg", False)
    row("   tightest knee seen", "worst_knee", 1, " deg", False)
    print("")
    print(" VISIBLE BODY COLLAPSE - closest a non-hip joint came to the mid-hip")
    row("   closest approach", "min_hip", 4, "m", False)
    row("   frames inside the collapse radius (%.2f m)" % COLLAPSE_M, "collapses", 0, "", True)
    print("")
    print(" NON-FINITE LANDMARKS reaching the retarget")
    row("   NaN/Inf count", "nans", 0, "", True)

    if n:
        print("-" * 104)
        print(" WHAT THE LAYER DID, over %d poses" % n)
        print(" %-28s %12s %12s   %s" % ("stage", "joint-frames", "% of poses", "reading"))
        order = [("held", "joints held (not observed)"),
                 ("vclamp", "velocity clamped"),
                 ("bone", "bone length corrected"),
                 ("hinge", "elbow/knee opened up"),
                 ("cone", "hip/neck cone"),
                 ("knee", "knee un-bent"),
                 ("twist", "torso twist refused"),
                 ("recov", "re-acquire blending"),
                 ("rejOrigin", "rejected: at the origin"),
                 ("rejFar", "rejected: outside a body"),
                 ("rejNan", "rejected: NaN"),
                 ("rejConf", "rejected: not emitted")]
        for k, label in order:
            print(" %-28s %12d %11.1f%%" % (label, total[k], 100.0 * frames_with[k] / n))
        print("-" * 104)
        print(" RESPONSE LATENCY - can this layer have DELAYED the pose?")
        print("   poses no stateful stage touched  %5.1f %%   (no hold, no velocity clamp, no re-acquire"
              % (100.0 * latency_free / n))
        print("                                             blend: nothing that carries state between")
        print("                                             frames, so nothing that can lag)")
        print("   poses nothing at all touched     %5.1f %%   (the above, and no geometric projection"
              % (100.0 * fully_untouched / n))
        print("                                             either - a strictly stronger statement)")
    print("=" * 104)
    return raw, hum


# -------------------------------------------------------------------------------------------------
def selftest():
    """An instrument used to judge a correction layer gets checked first."""
    ok = [0]
    bad = [0]

    def ck(name, got, want):
        good = got == want
        ok[0] += good
        bad[0] += (not good)
        print("  %s %-60s got=%r want=%r" % ("PASS" if good else "FAIL", name, got, want))

    rows = [{"raw": {"jump": 0.5, "bone": 0.10, "elbowL": 10.0, "elbowR": 180.0,
                     "kneeL": 20.0, "kneeR": 180.0, "minhip": 0.01, "nan": 1},
             "hum": {"jump": 0.05, "bone": 0.001, "elbowL": 25.0, "elbowR": 180.0,
                     "kneeL": 30.0, "kneeR": 180.0, "minhip": 0.40, "nan": 0},
             "stage": {"held": 1, "vclamp": 1, "bone": 8, "hinge": 2, "cone": 0, "knee": 1,
                       "twist": 0, "recov": 0, "rejOrigin": 1, "rejFar": 0, "rejNan": 1,
                       "rejConf": 0, "untouched": 0, "boneOnly": 0}}]
    raw = stream_stats(rows, "raw", 0)
    hum = stream_stats(rows, "hum", 0)
    ck("an impossible elbow AND knee are both counted", raw["hinge_bad"], 2)
    ck("  and neither survives in the humanized stream", hum["hinge_bad"], 0)
    ck("a joint on the hip is counted as a collapse", raw["collapses"], 1)
    ck("  and is gone after", hum["collapses"], 0)
    ck("NaN reaching the retarget is counted", raw["nans"], 1)
    ck("  and is gone after", hum["nans"], 0)
    ck("the tightest angle is reported, not averaged", raw["worst_elbow"], 10.0)

    # the warm-up really is skipped, or a layer would be scored on frames its calibration had not seen
    ck("warm-up frames are excluded", stream_stats(rows * 5, "raw", 3)["n"], 2)

    # direction of the verdict: lower jump is better, LARGER clearance from the hip is better
    ck("a smaller jump reads as better", "better" in delta(0.5, 0.05, True), True)
    ck("a larger hip clearance reads as better", "better" in delta(0.01, 0.40, False), True)
    ck("a smaller hip clearance reads as WORSE", "WORSE" in delta(0.40, 0.01, False), True)
    ck("a change under 1 % is not dressed up as an improvement", delta(1.0, 0.999, True).strip(), "=")

    n, total, frames_with, latency_free, fully_untouched = stage_stats(rows, 0)
    ck("stage totals are summed", total["bone"], 8)
    ck("  and per-frame incidence is counted separately", frames_with["knee"], 1)
    ck("a stage that never fired shows zero incidence", frames_with["twist"], 0)

    # A pose whose ONLY intervention was a non-emitted face landmark is latency-free. The layer's own
    # counter says otherwise, which is why this is computed here.
    quiet = [{"stage": {"held": 0, "vclamp": 0, "bone": 8, "hinge": 0, "cone": 0, "knee": 0,
                        "twist": 0, "recov": 0, "rejOrigin": 0, "rejFar": 0, "rejNan": 0,
                        "rejConf": 12, "untouched": 0, "boneOnly": 0}}]
    _, _, _, lf, fu = stage_stats(quiet, 0)
    ck("a pose held up only by an unsent face point is latency-free", lf, 1)
    ck("  but bone-length work still means 'not untouched'", fu, 0)
    held = [{"stage": dict(quiet[0]["stage"], held=1)}]
    _, _, _, lf, _ = stage_stats(held, 0)
    ck("a pose with a HELD joint is not latency-free", lf, 0)

    print("\n%d passed, %d failed" % (ok[0], bad[0]))
    return 0 if not bad[0] else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="path", default="")
    ap.add_argument("--warmup", type=int, default=150,
                    help="poses to skip before scoring; both streams' bone-length medians need a "
                         "window before their deviations mean anything")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.path:
        print("give --in <beforeafter.jsonl>, or --selftest")
        return 2
    rows = load(a.path)
    if not rows:
        print("no records in %s - the recorder wrote nothing. This is NOT a pass." % a.path)
        return 1
    if len(rows) <= a.warmup:
        print("only %d records but warmup is %d - refusing to report on the warm-up alone."
              % (len(rows), a.warmup))
        return 1
    report(rows, a.warmup)
    return 0


if __name__ == "__main__":
    sys.exit(main())
