#!/usr/bin/env python3
"""F-26 S9.1 - POSE FIDELITY: how far the rendered avatar is from the tracked subject, per bone.

THE MEASUREMENT THAT DID NOT EXIST. Every avatar-side number this project has produced measures the
avatar against ITSELF - yaw jitter, snap count, bone-length constancy, left/right swap count. F-19
reported all four as "AVATAR QUALITY: PROVEN GOOD", and every one of them is satisfied by an avatar
that is smoothly, stably and consistently in the WRONG pose, which is what the 2026-09-16 recording
shows it doing. Nothing anywhere compared the avatar's pose to the SUBJECT's pose.

INPUT is AvatarFidelityRecorder's jsonl: one record per rendered frame, each bone's angular error in
degrees between the direction the subject's segment points and the direction the avatar's bone
actually ends up pointing, plus the minimum confidence of the two source landmarks.

WHAT IT REPORTS, and why it is shaped this way:
  * PER BONE, never a single blended score. A mean over nine bones hides exactly the thing the video
    shows - the trunk failing while the arms are fine.
  * A DISTRIBUTION (median / p90 / max), not a mean. The failures here are intermittent and
    pose-dependent; a mean over a session that is mostly standing still reports "good".
  * LOW-CONFIDENCE FRAMES ARE EXCLUDED, not averaged in. A bone whose source is not being observed
    says nothing about the retarget. Absence of observation is not evidence of failure.
  * UNMEASURABLE frames are counted and reported separately. The recorder writes null, never 0, so a
    bone that could not be measured can never read as a perfect one.

WHAT IT IS NOT. An angular error per bone is not a complete fidelity measure: two poses can agree on
every bone DIRECTION and still put the hands in different places, because the avatar's bone LENGTHS
are its own. Direction error is what a rotation-driven rig can be held to. Endpoint error is a
different measurement and is deliberately not conflated with this one.

    .venv\\Scripts\\python.exe f26_fidelity_analyze.py --in evidence/oak_v4/f26/fidelity.jsonl
    .venv\\Scripts\\python.exe f26_fidelity_analyze.py --selftest
"""
import argparse
import io
import json
import os
import sys

BONES = ["leftUpperArm", "leftLowerArm", "rightUpperArm", "rightLowerArm",
         "leftUpperLeg", "leftLowerLeg", "rightUpperLeg", "rightLowerLeg", "trunk"]

# A bone whose two source landmarks are below this is not being observed well enough for its error to
# mean anything. Matches the retarget's own limbConfidenceThreshold (0.3), so this analysis excludes
# exactly the frames the retarget itself would have refused to drive from.
MIN_CONF = 0.3

# The recorder samples every RENDERED frame (~118 Hz in the editor) while poses arrive at ~30 Hz, so
# a source value repeating for three or four frames is normal oversampling. A repeat lasting longer
# than the product's own poseStaleSeconds means NO PACKET ARRIVED - and when the stream goes stale
# AppBootstrap calls ReleaseToRest, which walks the avatar back to its rest pose while the recorded
# `want` stays frozen at the last packet. Those frames manufacture a large error out of the shutdown
# and must not enter the distribution.
#
# A MOTIONLESS SUBJECT DOES NOT LOOK LIKE THIS. The source angles come from RTMW3D and are written at
# two decimal places; its own estimation noise changes them every packet. Bit-identical values across
# ~60 rendered frames is the stream stopping, not a person standing still - so this rule cannot hide
# a real failure to track a stationary subject, which is the case it would be dangerous to drop.
STALE_S = 0.5


def drop_stale(rows, stale_s=STALE_S):
    """-> (kept, dropped). Removes frames in which no new pose packet arrived (see STALE_S)."""
    if not rows:
        return rows, 0
    keys = []
    for r in rows:
        keys.append(tuple(r.get(b + "_src") for b in BONES))
    if all(k == tuple([None] * len(BONES)) for k in keys):
        return rows, 0          # an older recording with no _src columns: nothing to judge on
    keep = [True] * len(rows)
    i = 0
    while i < len(rows):
        j = i + 1
        while j < len(rows) and keys[j] == keys[i]:
            j += 1
        held = rows[j - 1].get("t", 0.0) - rows[i].get("t", 0.0)
        if held > stale_s:
            # keep the first frame of the run - it is the last one a packet actually produced
            for k in range(i + 1, j):
                keep[k] = False
        i = j
    kept = [r for r, k in zip(rows, keep) if k]
    return kept, len(rows) - len(kept)


def load(path):
    out = []
    try:
        for line in io.open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    except IOError:
        print("cannot read %s" % path)
        sys.exit(2)
    return out


def quantile(sorted_vals, q):
    if not sorted_vals:
        return None
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def per_bone(rows, min_conf=MIN_CONF):
    """-> {bone: dict(n, excluded_conf, unmeasurable, median, p90, max, over_10, over_30)}"""
    out = {}
    for bone in BONES:
        vals = []
        low = 0
        unmeasurable = 0
        for r in rows:
            if bone not in r:
                continue
            v = r.get(bone)
            c = r.get(bone + "_conf")
            if v is None:
                unmeasurable += 1
                continue
            if c is not None and c < min_conf:
                low += 1
                continue
            vals.append(float(v))
        vals.sort()
        out[bone] = dict(
            n=len(vals), excluded_low_conf=low, unmeasurable=unmeasurable,
            median=quantile(vals, 0.5), p90=quantile(vals, 0.9),
            max=(vals[-1] if vals else None),
            over_10=sum(1 for v in vals if v > 10.0),
            over_30=sum(1 for v in vals if v > 30.0))
    return out


def follow(rows, min_conf=MIN_CONF):
    """-> {bone: dict(src_p10, src_p90, src_span, rig_p10, rig_p90, rig_span, ratio)}

    A large but nearly CONSTANT error is ambiguous in the error column alone: it is either a fixed
    offset the retarget could be calibrated out of, or a channel that is not moving at all. These two
    read identically. Comparing how far the SUBJECT's segment swings against how far the RIG's does
    separates them - a ratio near 0 is a frozen channel, near 1 is a channel that follows.

    The span is p10..p90, not min..max, so one dropout frame cannot manufacture motion."""
    out = {}
    for bone in BONES:
        src = []
        rig = []
        for r in rows:
            c = r.get(bone + "_conf")
            if c is not None and c < min_conf:
                continue
            s = r.get(bone + "_src")
            g = r.get(bone + "_rig")
            if s is None or g is None:
                continue
            src.append(float(s))
            rig.append(float(g))
        src.sort()
        rig.sort()
        if not src:
            out[bone] = None
            continue
        s10, s90 = quantile(src, 0.1), quantile(src, 0.9)
        g10, g90 = quantile(rig, 0.1), quantile(rig, 0.9)
        s_span = s90 - s10
        g_span = g90 - g10
        out[bone] = dict(n=len(src), src_p10=s10, src_p90=s90, src_span=s_span,
                         rig_p10=g10, rig_p90=g90, rig_span=g_span,
                         ratio=(g_span / s_span if s_span > 1e-6 else None))
    return out


def report(rows, min_conf=MIN_CONF, stale_s=STALE_S):
    recorded = len(rows)
    rows, stale = drop_stale(rows, stale_s)
    stats = per_bone(rows, min_conf)
    print("=" * 100)
    print(" F-26 POSE FIDELITY - avatar bone direction vs subject segment direction")
    print(" %d recorded frames; %d dropped as STALE (no packet arrived, avatar releasing to rest);"
          % (recorded, stale))
    print(" %d analysed. Per-bone frames below confidence %.2f are EXCLUDED, not averaged in."
          % (len(rows), min_conf))
    print("=" * 100)
    print(" %-15s %7s %9s %8s %8s %8s %8s   %s"
          % ("bone", "frames", "median", "p90", "max", ">10deg", ">30deg", "excluded(lowconf/unmeas)"))
    worst = None
    for bone in BONES:
        s = stats[bone]
        if not s["n"]:
            print(" %-15s %7d   %s" % (bone, 0, "NOT MEASURED - no frames above the confidence floor"))
            continue
        pct10 = 100.0 * s["over_10"] / s["n"]
        pct30 = 100.0 * s["over_30"] / s["n"]
        print(" %-15s %7d %8.1f%s %7.1f%s %7.1f%s %7.1f%% %7.1f%%   %d / %d"
              % (bone, s["n"], s["median"], chr(176), s["p90"], chr(176), s["max"], chr(176),
                 pct10, pct30, s["excluded_low_conf"], s["unmeasurable"]))
        if worst is None or s["median"] > stats[worst]["median"]:
            worst = bone
    print("-" * 100)
    if worst is not None:
        print(" WORST BONE BY MEDIAN: %s at %.1f%s" % (worst, stats[worst]["median"], chr(176)))
    print(" No single combined score is produced, on purpose: a mean over nine bones hides a trunk")
    print(" failing while the arms are fine, which is exactly the failure this metric exists to see.")

    fol = follow(rows, min_conf)
    if any(fol[b] for b in BONES):
        print("")
        print(" DOES THE CHANNEL FOLLOW AT ALL - inclination from the rig's vertical, p10..p90")
        print(" A big but CONSTANT error is either a fixed offset or a frozen channel; the error")
        print(" column alone cannot tell them apart. follow = rig swing / subject swing.")
        print(" %-15s %19s %19s %8s   %s" % ("bone", "subject p10..p90", "avatar p10..p90",
                                             "follow", "reading"))
        for bone in BONES:
            f = fol[bone]
            if not f:
                print(" %-15s %s" % (bone, "NOT MEASURED"))
                continue
            if f["ratio"] is None:
                verdict = "subject did not move this bone - says nothing"
            elif f["ratio"] < 0.25:
                verdict = "FROZEN - the avatar barely moves this channel"
            elif f["ratio"] < 0.7:
                verdict = "UNDER-DRIVEN"
            elif f["ratio"] <= 1.4:
                verdict = "follows"
            else:
                verdict = "OVER-DRIVEN"
            print(" %-15s %8.1f..%-8.1f %8.1f..%-8.1f %7s   %s"
                  % (bone, f["src_p10"], f["src_p90"], f["rig_p10"], f["rig_p90"],
                     ("n/a" if f["ratio"] is None else "%.2f" % f["ratio"]), verdict))
    print("=" * 100)
    return stats


# -------------------------------------------------------------------------------------------------
# A/B: the SAME metric on two recordings. Used to answer "did F-27's humanized skeleton make the
# AVATAR better", which is a different question from "did it make the POSE better" (that is F-27's
# own report) and has to be measured separately.
#
# WHAT MAKES THIS COMPARISON FAIR, AND WHERE IT IS NOT:
#   * Both arms are driven from the SAME video clip, from the same offset into it, through the same
#     production wire, and are measured by the same code. Nothing about the instrument differs.
#   * The reference `want` is built from the RAW tracked pose in BOTH arms
#     (KalidokitControlRigDriver.SetFidelityReference). Without that, the humanized arm would be
#     scored against its own input - the retarget handed an easier pose and then marked on how well
#     it followed it - and would win by construction.
#   * It is NOT frame-locked. The two passes are separate playbacks, so they cover the same clip
#     CONTENT but not the identical frames, and small differences are not evidence of anything. The
#     verdict below therefore refuses to call anything a win unless it moves by BOTH at least 1 degree
#     AND at least 10 %.
#   * RAW IS NOT THE SUBJECT. It is the only independent reference available, and F-27 exists because
#     raw contains poses a body cannot make. Every correction that layer makes moves the pose away
#     from raw and can only COST it points here. A tie is therefore a good result for it; a win is a
#     strong one.
MIN_ABS_DEG = 1.0
MIN_REL_PCT = 10.0


def verdict(before, after):
    """-> (text, better) with the two-part guard described above."""
    if before is None or after is None:
        return ("n/a", None)
    diff = after - before
    if abs(diff) < MIN_ABS_DEG:
        return ("=", None)
    if before > 0 and abs(100.0 * diff / before) < MIN_REL_PCT:
        return ("=", None)
    better = diff < 0
    return ("%+.1f%s %s" % (diff, chr(176), "better" if better else "WORSE"), better)


def compare(before_rows, after_rows, before_label, after_label, min_conf=MIN_CONF, stale_s=STALE_S):
    b_all, b_stale = drop_stale(before_rows, stale_s)
    a_all, a_stale = drop_stale(after_rows, stale_s)
    b = per_bone(b_all, min_conf)
    a = per_bone(a_all, min_conf)
    bf = follow(b_all, min_conf)
    af = follow(a_all, min_conf)

    print("=" * 108)
    print(" F-26 POSE FIDELITY, A/B - avatar bone direction vs the RAW tracked segment direction")
    print(" BEFORE = %s   (%d recorded, %d stale dropped, %d analysed)"
          % (before_label, len(before_rows), b_stale, len(b_all)))
    print(" AFTER  = %s   (%d recorded, %d stale dropped, %d analysed)"
          % (after_label, len(after_rows), a_stale, len(a_all)))
    print(" The reference is the RAW pose in BOTH arms, so neither is scored against its own input.")
    print("=" * 108)
    print(" %-15s %19s %19s %18s   %s"
          % ("bone", "BEFORE med / p90", "AFTER med / p90", "change (median)", "follow B -> A"))
    print("-" * 108)

    wins = 0
    losses = 0
    ties = 0
    for bone in BONES:
        sb, sa = b[bone], a[bone]
        if not sb["n"] or not sa["n"]:
            print(" %-15s %s" % (bone, "NOT MEASURED in one or both arms"))
            continue
        text, better = verdict(sb["median"], sa["median"])
        if better is True:
            wins += 1
        elif better is False:
            losses += 1
        else:
            ties += 1
        fb = bf[bone]["ratio"] if bf[bone] and bf[bone]["ratio"] is not None else None
        fa = af[bone]["ratio"] if af[bone] and af[bone]["ratio"] is not None else None
        follow_text = "%s -> %s" % ("n/a" if fb is None else "%.2f" % fb,
                                    "n/a" if fa is None else "%.2f" % fa)
        print(" %-15s %8.1f%s /%7.1f%s %8.1f%s /%7.1f%s %18s   %s"
              % (bone, sb["median"], chr(176), sb["p90"], chr(176),
                 sa["median"], chr(176), sa["p90"], chr(176), text, follow_text))

    print("-" * 108)
    print(" %d bones better, %d worse, %d unchanged" % (wins, losses, ties))
    print(" 'unchanged' means the median moved less than %.0f%s or less than %.0f%% - the two passes are"
          % (MIN_ABS_DEG, chr(176), MIN_REL_PCT))
    print(" separate playbacks of the same clip, not the identical frames, so smaller moves are noise.")
    print("")
    print(" REMEMBER WHAT THE REFERENCE IS. `want` comes from the RAW tracker in both arms, and F-27")
    print(" exists because the raw pose contains shapes a body cannot make. Every correction it makes")
    print(" moves the pose away from that reference, so this test is stacked against it: a TIE means")
    print(" the humanized pose cost the avatar nothing, and a WIN means it helped despite the handicap.")
    print("=" * 108)
    return b, a


# ---------------------------------------------------------------------------------------------
def selftest():
    """Answers known by construction. An instrument used to judge the retarget gets checked first -
    three findings this month came from a wrong instrument rather than wrong code."""
    ok = [0]
    bad = [0]

    def ck(name, got, want):
        good = got == want
        ok[0] += good
        bad[0] += (not good)
        print("  %s %-62s got=%r want=%r" % ("PASS" if good else "FAIL", name, got, want))

    rows = [
        {"frame": 0, "trunk": 5.0, "trunk_conf": 0.9},
        {"frame": 1, "trunk": 15.0, "trunk_conf": 0.9},
        {"frame": 2, "trunk": 45.0, "trunk_conf": 0.9},
        {"frame": 3, "trunk": 99.0, "trunk_conf": 0.1},    # low confidence -> excluded
        {"frame": 4, "trunk": None, "trunk_conf": 0.9},    # unmeasurable -> excluded, NOT 0
    ]
    s = per_bone(rows)["trunk"]
    ck("low-confidence frames are excluded", s["n"], 3)
    ck("  and counted", s["excluded_low_conf"], 1)
    ck("unmeasurable frames are excluded", s["unmeasurable"], 1)
    ck("  a null NEVER reads as a perfect 0", s["max"], 45.0)
    ck("median is the middle of what remains", s["median"], 15.0)
    ck("the 99 deg low-confidence frame cannot reach max", s["max"] != 99.0, True)
    ck(">10 deg counts", s["over_10"], 2)
    ck(">30 deg counts", s["over_30"], 1)

    empty = per_bone([{"frame": 0, "trunk": None, "trunk_conf": 0.9}])["trunk"]
    ck("a bone with nothing measurable reports n=0, not a score", empty["n"], 0)
    ck("  and its median is None, not 0", empty["median"], None)

    # a bone that is simply absent from the record must not be invented
    absent = per_bone([{"frame": 0, "trunk": 5.0, "trunk_conf": 0.9}])["leftUpperArm"]
    ck("a bone absent from the file reports n=0", absent["n"], 0)

    # --- follow ratio: the frozen-vs-offset discriminator -------------------------------------
    # subject swings 0..40 deg, avatar sits at 10 deg throughout -> frozen, ratio 0
    frozen = [{"trunk": 9.0, "trunk_conf": 0.9, "trunk_src": float(d), "trunk_rig": 10.0}
              for d in range(0, 41)]
    f = follow(frozen)["trunk"]
    ck("a frozen channel scores ratio 0", f["ratio"], 0.0)
    ck("  and the subject's own swing is still reported", round(f["src_span"]), 32)

    # subject and avatar swing together -> ratio 1, even with a constant 12 deg offset
    moving = [{"trunk": 12.0, "trunk_conf": 0.9, "trunk_src": float(d), "trunk_rig": d + 12.0}
              for d in range(0, 41)]
    f = follow(moving)["trunk"]
    ck("a following channel scores ratio 1 despite a fixed offset", f["ratio"], 1.0)

    # the two cases above have the SAME median error but opposite diagnoses - the whole point
    ck("  frozen and following can share a median error",
       per_bone(frozen)["trunk"]["median"] < per_bone(moving)["trunk"]["median"], True)

    # low-confidence frames must not contribute motion either
    f = follow([{"trunk": 5.0, "trunk_conf": 0.1, "trunk_src": 0.0, "trunk_rig": 0.0},
                {"trunk": 5.0, "trunk_conf": 0.1, "trunk_src": 90.0, "trunk_rig": 90.0}])["trunk"]
    ck("follow ignores low-confidence frames", f, None)

    # an unmeasurable frame writes null for src/rig and must not read as 0 inclination
    f = follow([{"trunk": None, "trunk_conf": 0.9, "trunk_src": None, "trunk_rig": None},
                {"trunk": 5.0, "trunk_conf": 0.9, "trunk_src": 20.0, "trunk_rig": 20.0}])["trunk"]
    ck("follow ignores unmeasurable frames", f["n"], 1)

    # --- stale tail: the release-to-rest frames must never enter the distribution ---------------
    live = [{"t": i * 0.01, "trunk": 5.0, "trunk_conf": 0.9, "trunk_src": 10.0 + i * 0.01,
             "trunk_rig": 10.0} for i in range(100)]
    # the first frame of the frozen run is the genuine one - it is what the LAST packet produced;
    # only the frames after it are the avatar walking back to rest against a source that has stopped
    tail = [{"t": 1.0, "trunk": 5.0, "trunk_conf": 0.9, "trunk_src": 11.0, "trunk_rig": 10.0}]
    tail += [{"t": 1.0 + i * 0.01, "trunk": 90.0, "trunk_conf": 0.9, "trunk_src": 11.0,
              "trunk_rig": 0.0} for i in range(1, 100)]
    kept, dropped = drop_stale(live + tail)
    ck("a 1.0 s frozen-source tail is dropped", dropped, 99)
    ck("  the last real frame is kept", len(kept), 101)
    ck("  the 90 deg release frames cannot reach the distribution",
       per_bone(kept)["trunk"]["max"], 5.0)

    # oversampling (~4 rendered frames per packet) must NOT be mistaken for staleness
    over = []
    for p in range(30):
        for k in range(4):
            over.append({"t": p * 0.033 + k * 0.008, "trunk": 5.0, "trunk_conf": 0.9,
                         "trunk_src": 10.0 + p * 0.01, "trunk_rig": 10.0})
    kept, dropped = drop_stale(over)
    ck("normal oversampling is not dropped", dropped, 0)

    # a recording made before the _src columns existed must pass through untouched
    kept, dropped = drop_stale([{"t": 0.0, "trunk": 5.0, "trunk_conf": 0.9},
                                {"t": 9.0, "trunk": 5.0, "trunk_conf": 0.9}])
    ck("a recording without _src columns is not silently emptied", (len(kept), dropped), (2, 0))

    # --- the A/B verdict guard: two separate playbacks are not frame-locked ---------------------
    ck("a 5 deg improvement is a win", verdict(20.0, 15.0)[1], True)
    ck("a 5 deg regression is a loss", verdict(15.0, 20.0)[1], False)
    ck("a 0.4 deg move is noise, not a win", verdict(20.0, 19.6)[0], "=")
    ck("a 1.5 deg move on a 20 deg bone is under 10 % - still noise", verdict(20.0, 18.5)[0], "=")
    ck("  but the same 1.5 deg on a 5 deg bone counts", verdict(5.0, 3.5)[1], True)
    ck("a bone with no measurement gives no verdict", verdict(None, 12.0)[1], None)

    print("\n%d passed, %d failed" % (ok[0], bad[0]))
    return 0 if not bad[0] else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="path", default="")
    ap.add_argument("--min-conf", type=float, default=MIN_CONF)
    ap.add_argument("--stale-s", type=float, default=STALE_S,
                    help="drop runs of frames in which the SOURCE pose did not change for longer "
                         "than this - no packet arrived and the avatar is releasing to rest")
    ap.add_argument("--before", default="", help="A/B: the baseline recording")
    ap.add_argument("--after", default="", help="A/B: the recording to judge against the baseline")
    ap.add_argument("--before-label", default="BEFORE")
    ap.add_argument("--after-label", default="AFTER")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.before and a.after:
        before_rows = load(a.before)
        after_rows = load(a.after)
        if not before_rows or not after_rows:
            print("one of the recordings is empty - this is NOT a pass.")
            return 1
        compare(before_rows, after_rows, a.before_label, a.after_label, a.min_conf, a.stale_s)
        return 0
    if not a.path:
        print("give --in <fidelity.jsonl>, or --before X --after Y, or --selftest")
        return 2
    rows = load(a.path)
    if not rows:
        print("no records in %s - the recorder wrote nothing. This is NOT a pass." % a.path)
        return 1
    report(rows, a.min_conf, a.stale_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
