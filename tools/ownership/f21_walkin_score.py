#!/usr/bin/env python3
"""Score an f21_walkin_protocol.py session: the S30 false-hold and silent-wrong-person rates.

Both inputs are stamped with time.time() by their own writer - the protocol writes its timeline, the
SIDECAR writes target_events.jsonl - so the two align on one wall clock with no correlation
heuristic. Nothing here infers what the humans did; it reads what the protocol INSTRUCTED and what
the ownership layer REPORTED, and a rep whose anchor never established is excluded rather than
guessed at.

WHAT COUNTS AS WHAT, stated up front because the interesting rates are opposites:

  OWNER_OCCLUDED / OWNER_TURN   A never left the cross. The chain break lands INSIDE the margin, so
                                ADR-057 says the flag clears and A stays admissible. A path_walked_in
                                rejection here is a FALSE HOLD - the damaging kind, against the most
                                common real-world loss pattern. This is the headline rate.

  IMPOSTOR_WALKIN               B walked to A's spot. A refusal followed by a declared TARGET_SWITCH
                                is CORRECT. A silent reacquire inside the same epoch is the S27
                                failure the gate exists to prevent.

  OWNER_WALKBACK                A walked out past the margin and back. ADR-057 says the gate CANNOT
                                distinguish this from an intruder and deliberately refuses it. Its
                                rate is ~1.0 by construction and proves nothing; this case is scored
                                for its COST - how long the legitimate owner was actually withheld.

Rates from a handful of reps are reported with a Wilson 95% interval, because "0 of 6" is not a
measurement of zero and must not be written down as one.

    .venv\\Scripts\\python.exe f21_walkin_score.py --timeline ... --events ...
"""
import argparse
import io
import json
import math
import os
import sys

REJ = "TARGET_REJECTED_CANDIDATE"
PATH = "path_walked_in"


def load(path):
    out = []
    try:
        for line in io.open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except IOError:
        print("cannot read %s" % path)
        sys.exit(2)
    return out


def wilson(k, n, z=1.96):
    """95% Wilson score interval. Correct at k=0 and k=n, where the normal approximation returns a
    zero-width interval and would let '0 of 6' be reported as a measured zero."""
    if n == 0:
        return (0.0, 1.0)
    p = float(k) / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4.0 * n * n))
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


def reps_from(timeline):
    """Split the timeline into reps, each with its manoeuvre window (first phase after ANCHOR)."""
    reps = {}
    for e in timeline:
        r = e.get("rep")
        if r is None:
            continue
        d = reps.setdefault(r, dict(rep=r, case=e.get("case"), phases=[], valid=None,
                                    t_start=None, t_end=None))
        if e.get("event") == "rep_start":
            d["t_start"] = e["t"]
        elif e.get("event") == "rep_end":
            d["t_end"] = e["t"]
            d["valid"] = e.get("valid")
        elif e.get("event") == "phase_start":
            d["phases"].append((e["t"], e.get("phase"), e.get("actor")))
    for d in reps.values():
        d["phases"].sort()
        names = [p[1] for p in d["phases"]]
        # the manoeuvre is everything after ANCHOR ends
        d["t_man"] = None
        d["t_anchor_end"] = None
        for i, n in enumerate(names):
            if n == "ANCHOR" and i + 1 < len(d["phases"]):
                d["t_man"] = d["phases"][i + 1][0]
                d["t_anchor_end"] = d["t_man"]
                break
    return [reps[k] for k in sorted(reps)]


# S33. HOLD_S must exceed RELEASE_TIMEOUT_S (4.0 s) or a whole lose-release-reacquire cycle could
# pass through the window without leaving a transition inside it. Mirrors f21_walkin_protocol.HOLD_S.
HOLD_S = 5.0
DESTABILISING = ("TARGET_TEMP_LOST", "TARGET_RELEASED", "TARGET_SWITCH")


def anchor_stability(rep, evs, hold_s=HOLD_S):
    """Did the anchor SURVIVE to the manoeuvre? Derived from the ownership events themselves.

    Deliberately NOT read from the timeline's `anchor_held` flag. The protocol only began writing
    that flag after the 2026-09-15 shakedown, and the whole point of that shakedown is that a rep
    can establish a lock and then have it collapse underneath - 6 epochs in 71 s - while the
    timeline still says `valid: true`. Re-deriving it here means older captures are judged by the
    same standard as new ones, and the scorer does not depend on the recorder having been correct.

    -> (ok, transitions). Empty transitions = the machine held a single stable lock into the
    manoeuvre, which is the only condition under which what follows means anything.
    """
    t_man = rep.get("t_man")
    if t_man is None:
        return None, []
    win = [e for e in evs if (t_man - hold_s) <= e.get("t", 0) < t_man]
    bad = [e for e in win if e.get("event") in DESTABILISING]
    return (not bad), bad


SWITCH_MARGIN_M = 0.35        # production OwnershipConfig value - the scale a migration is judged on


def wire_hips(wire):
    """-> [(t, (x, y, z))] of the EMITTED mid-hip, metres, from f24_wire_probe records.

    The packet's `xyz` is the absolute mid-hip in MILLIMETRES, and it is present only on frames the
    sidecar actually sent - which is exactly the population that matters. A withheld frame produces
    no packet, so absence here is F-21 holding, not a gap in the recording.
    """
    out = []
    for d in wire:
        p = d.get("xyz")
        if not p or len(p) < 3:
            continue
        out.append((d.get("_rx") or d.get("t"), (p[0] / 1000.0, p[1] / 1000.0, p[2] / 1000.0)))
    out.sort()
    return out


def migration(rep, hips):
    """Did the EMITTED pose walk away from where the epoch locked on, without saying so?

    S34. The scorer used to ask only "did the path gate fire?", and for OWNER_OCCLUDED / OWNER_TURN
    that was the ONLY question - so a rep in which the sidecar emitted the wrong human for 97
    datagrams scored OK_NO_HOLD, the cleanest result available. The gate cannot see this case by
    construction (ADR-061: the state never leaves LOCKED, and the gate lives in the loss branch), so
    a metric built on the gate is blind to it.

    This asks the question the gate cannot: how far did the EMITTED hip travel from its anchor, and
    did ownership announce anything while it did?

    What this does and does not claim: it detects MIGRATION, not identity. It cannot know who a pose
    belongs to. In a case where the owner is supposed to stand still, a silent excursion past
    SWITCH_MARGIN_M is strong evidence the emitted body changed; in OWNER_WALKBACK the owner is
    INSTRUCTED to leave, so drift is expected there and is reported without a verdict.
    """
    t0, t1 = rep.get("t_man"), rep.get("t_end")
    ta = rep.get("t_anchor_end") or t0
    if t0 is None or not hips:
        return
    anchor = [p for t, p in hips if t is not None and (ta - 3.0) <= t <= ta]
    if not anchor:
        return
    ax = sorted(p[0] for p in anchor)[len(anchor) // 2]
    ay = sorted(p[1] for p in anchor)[len(anchor) // 2]
    az = sorted(p[2] for p in anchor)[len(anchor) // 2]
    man = [(t, p) for t, p in hips if t is not None and t0 <= t <= t1]
    if not man:
        return
    dists = [((p[0] - ax) ** 2 + (p[1] - ay) ** 2 + (p[2] - az) ** 2) ** 0.5 for _t, p in man]
    rep["emitted_frames"] = len(man)
    rep["anchor_hip_z"] = round(az, 3)
    rep["drift_max_m"] = round(max(dists), 3)
    rep["frames_beyond_margin"] = sum(1 for d in dists if d > SWITCH_MARGIN_M)
    announced = rep.get("releases", 0) > 0 or rep.get("switches", 0) > 0
    rep["drift_announced"] = bool(announced)
    if rep["case"] == "OWNER_WALKBACK":
        rep["migration"] = "EXPECTED"          # the owner was told to leave; drift proves nothing
    elif rep["frames_beyond_margin"] and not announced:
        rep["migration"] = "SILENT_MIGRATION"  # the S34 failure, measured per frame
    elif rep["frames_beyond_margin"]:
        rep["migration"] = "DECLARED"
    else:
        rep["migration"] = "NONE"


def score_rep(rep, evs):
    """Ownership events inside the manoeuvre window, and what they add up to."""
    t0, t1 = rep.get("t_man"), rep.get("t_end")
    if t0 is None or t1 is None:
        rep["scored"] = False
        return rep
    held, bad = anchor_stability(rep, evs)
    rep["anchor_stable"] = held
    rep["anchor_instability"] = sorted(set(e.get("event") for e in bad))
    if held is False:
        # the lock collapsed before the manoeuvre began - nothing after it is interpretable
        rep["scored"] = False
        rep["outcome"] = "INVALID_ANCHOR_COLLAPSED"
        return rep
    win = [e for e in evs if t0 <= e.get("t", 0) <= t1]
    rep["scored"] = True
    rep["path_rejections"] = sum(1 for e in win if e.get("event") == REJ
                                 and e.get("reason") == PATH)
    rep["other_rejections"] = sum(1 for e in win if e.get("event") == REJ
                                  and e.get("reason") != PATH)
    rep["releases"] = sum(1 for e in win if e.get("event") == "TARGET_RELEASED")
    rep["switches"] = sum(1 for e in win if e.get("event") == "TARGET_SWITCH")
    rep["reacquires"] = sum(1 for e in win if e.get("event") == "TARGET_REACQUIRED")
    rep["temp_lost"] = sum(1 for e in win if e.get("event") == "TARGET_TEMP_LOST")

    # BLACKOUT: the stretch where ownership was not driving the avatar - from the loss that starts
    # the episode to whatever ends it. This is the thing a visitor would actually notice.
    blackout = None
    t_lost = None
    for e in win:
        k = e.get("event")
        if k == "TARGET_TEMP_LOST" and t_lost is None:
            t_lost = e["t"]
        elif k in ("TARGET_REACQUIRED", "TARGET_LOCKED") and t_lost is not None:
            blackout = e["t"] - t_lost
            break
    if blackout is None and t_lost is not None:
        blackout = t1 - t_lost          # never recovered inside the rep
        rep["blackout_truncated"] = True
    rep["blackout_s"] = None if blackout is None else round(blackout, 3)

    # HOLD: first path refusal -> the event that resolved it.
    hold = None
    t_rej = next((e["t"] for e in win if e.get("event") == REJ and e.get("reason") == PATH), None)
    if t_rej is not None:
        nxt = next((e["t"] for e in win if e.get("t", 0) > t_rej
                    and e.get("event") in ("TARGET_REACQUIRED", "TARGET_RELEASED",
                                           "TARGET_LOCKED")), None)
        hold = (nxt - t_rej) if nxt else (t1 - t_rej)
    rep["hold_s"] = None if hold is None else round(hold, 3)

    case, fired = rep["case"], rep["path_rejections"] > 0

    # "SILENT" means what S27 measured: the human changed INSIDE one ownership epoch with nothing
    # in the log to say so. A TARGET_RELEASED followed by a TARGET_SWITCH is the OPPOSITE of that -
    # it is the consumer being told. The first version of this classifier tested only
    # `not fired and reacquires > 0` and so labelled a released, switched, correctly-declared
    # hand-over SILENT_WRONG_PERSON - a false alarm on the single most serious finding this suite
    # can report. Announcement is the safety property; whether the PATH gate specifically was the
    # thing that refused is a separate question, reported separately.
    declared = rep["releases"] > 0 and rep["switches"] > 0

    if case in ("OWNER_OCCLUDED", "OWNER_TURN"):
        rep["outcome"] = "FALSE_HOLD" if fired else "OK_NO_HOLD"
    elif case == "IMPOSTOR_WALKIN":
        if fired and rep["switches"] > 0:
            rep["outcome"] = "DECLARED_SWITCH"
        elif declared:
            rep["outcome"] = "DECLARED_SWITCH"      # released + switched: announced, so SAFE
        elif fired:
            rep["outcome"] = "HELD_NO_SWITCH"
        elif rep["reacquires"] > 0:
            rep["outcome"] = "SILENT_WRONG_PERSON"  # same epoch, no announcement - the S27 defect
        else:
            rep["outcome"] = "NO_ADMIT"
    else:   # OWNER_WALKBACK
        if fired and rep["switches"] > 0:
            rep["outcome"] = "REFUSED_THEN_DECLARED"
        elif fired:
            rep["outcome"] = "REFUSED_HELD"
        elif declared:
            rep["outcome"] = "RELEASED_THEN_REACQUIRED"   # announced, not silent
        elif rep["reacquires"] > 0:
            rep["outcome"] = "SILENT_READMIT"
        else:
            rep["outcome"] = "NO_ADMIT"
    return rep


def pct(k, n):
    lo, hi = wilson(k, n)
    return "%d/%d = %5.1f%%  [95%% CI %.1f-%.1f%%]" % (k, n, 100.0 * k / n if n else 0.0,
                                                       100 * lo, 100 * hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--wire", default="",
                     help="f24_wire_probe recording of the EMITTED UDP wire. Without it the "
                          "S34 silent-migration check cannot run and is reported NOT MEASURED, "
                          "never as a pass.")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    timeline, evs = load(a.timeline), load(a.events)
    hips = wire_hips(load(a.wire)) if a.wire else []
    reps = [score_rep(r, evs) for r in reps_from(timeline)]
    for _r in reps:
        migration(_r, hips)
    good = [r for r in reps if r.get("valid") and r.get("scored")]
    bad = [r for r in reps if not (r.get("valid") and r.get("scored"))]

    print("=" * 96)
    n_noanchor = sum(1 for r in bad if not r.get("valid"))
    n_collapsed = sum(1 for r in bad if r.get("anchor_stable") is False)
    print("F-21 S30 WALK-IN MATRIX - %d reps recorded, %d scoreable, %d excluded "
          "(%d never anchored, %d anchor collapsed)"
          % (len(reps), len(good), len(bad), n_noanchor, n_collapsed))
    print("=" * 96)
    print("%-4s %-16s %-24s %6s %5s %5s %5s %9s %9s" %
          ("rep", "case", "outcome", "path", "rel", "sw", "reacq", "hold_s", "black_s"))
    for r in reps:
        if not (r.get("valid") and r.get("scored")):
            why = ("ANCHOR COLLAPSED" if r.get("anchor_stable") is False
                   else "ANCHOR NEVER ESTABLISHED" if not r.get("valid") else "INVALID")
            extra = (" [%s]" % ",".join(r.get("anchor_instability") or [])) \
                if r.get("anchor_stable") is False else ""
            print("%-4s %-16s %-24s (excluded)%s" % (r["rep"], r["case"], why, extra))
            continue
        print("%-4s %-16s %-24s %6d %5d %5d %5d %9s %9s" %
              (r["rep"], r["case"], r["outcome"], r["path_rejections"], r["releases"],
               r["switches"], r["reacquires"],
               "-" if r["hold_s"] is None else "%.2f" % r["hold_s"],
               "-" if r["blackout_s"] is None else "%.2f" % r["blackout_s"]))

    def sub(*cases):
        return [r for r in good if r["case"] in cases]

    real = sub("OWNER_OCCLUDED", "OWNER_TURN")
    imp = sub("IMPOSTOR_WALKIN")
    back = sub("OWNER_WALKBACK")

    print("\n" + "-" * 96)
    print("THE HEADLINE NUMBER - false holds against a loss that never left the margin")
    print("  (OWNER_OCCLUDED + OWNER_TURN: A stayed on the cross throughout. ADR-057 says the gate")
    print("   must NOT fire here. This is the rate an installation actually pays.)")
    if real:
        print("    false holds : %s" % pct(sum(1 for r in real if r["outcome"] == "FALSE_HOLD"),
                                           len(real)))
    else:
        print("    NOT TESTED - no scoreable OWNER_OCCLUDED / OWNER_TURN reps")

    # ---- S34: the check a gate-based metric is structurally blind to ---------------------------
    print("\nSILENT MIGRATION (S34) - the EMITTED pose walking off its anchor with nothing said")
    print("  (ADR-061: while LOCKED, owner_pos follows every accepted frame, so a drift slower than")
    print("   %.2f m PER FRAME slides the reference from one human to another and the S30 gate never"
          % SWITCH_MARGIN_M)
    print("   runs. A metric built on the gate cannot see this. Measured from the emitted wire.)")
    measured = [r for r in good if r.get("migration")]
    diagnostic = [r for r in measured if r["migration"] != "EXPECTED"]
    if not measured:
        print("    NOT MEASURED - no --wire recording supplied. This is NOT a pass.")
    elif not diagnostic:
        print("    NOT MEASURED - no reps in a case where drift is diagnostic.")
    else:
        sil = [r for r in diagnostic if r["migration"] == "SILENT_MIGRATION"]
        print("    silent migration : %s" % pct(len(sil), len(diagnostic)))
        for r in diagnostic:
            print("      rep %-3s %-16s drift_max %5.2f m   beyond-margin %4d/%-4d frames   %s"
                  % (r["rep"], r["case"], r.get("drift_max_m", 0.0),
                     r.get("frames_beyond_margin", 0), r.get("emitted_frames", 0),
                     r["migration"]))
        for r in measured:
            if r["migration"] == "EXPECTED":
                print("      rep %-3s %-16s drift_max %5.2f m   (owner INSTRUCTED to leave - not a "
                      "migration signal)" % (r["rep"], r["case"], r.get("drift_max_m", 0.0)))

    print("\nSAFETY - the failure S27 found and S30 fixed")
    if imp:
        print("    silent wrong-person : %s"
              % pct(sum(1 for r in imp if r["outcome"] == "SILENT_WRONG_PERSON"), len(imp)))
        print("    declared hand-overs : %s"
              % pct(sum(1 for r in imp if r["outcome"] == "DECLARED_SWITCH"), len(imp)))
    else:
        print("    NOT TESTED - no scoreable IMPOSTOR_WALKIN reps")

    print("\nCOST - the deliberate walk-out-and-back, which ADR-057 says is refused BY DESIGN")
    if back:
        bl = sorted(r["blackout_s"] for r in back if r["blackout_s"] is not None)
        print("    refused : %s" % pct(sum(1 for r in back
                                           if r["outcome"].startswith("REFUSED")), len(back)))
        if bl:
            print("    blackout: median %.2f s   min %.2f   max %.2f   (n=%d)"
                  % (bl[len(bl) // 2], bl[0], bl[-1], len(bl)))
            print("    offline baseline on 123.webm was 1.60 s withheld (S30.9).")
    else:
        print("    NOT TESTED - no scoreable OWNER_WALKBACK reps")
    print("-" * 96)
    print("Rates are Wilson 95%. A '0 of n' is NOT a measured zero - read the interval.")
    print("TARGET_SWITCH=0 is not evidence of correctness on its own (S30.2); the recorded")
    print("preview still has to be reviewed by eye for wrong-person frames.")

    if a.json_out:
        with io.open(a.json_out, "w", encoding="utf-8") as f:
            f.write(json.dumps(dict(reps=reps, n_scoreable=len(good)), indent=2))
        print("\nwrote %s" % a.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
