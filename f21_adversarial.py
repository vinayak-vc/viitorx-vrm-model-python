#!/usr/bin/env python3
"""F-21 offline completion - deterministic multi-target adversarial scenario suite.

WHAT THIS PROVES, STATED UP FRONT: the OWNERSHIP LOGIC, not real two-person detector behaviour.
There is no second detector output to test against - RTMW3D-x is a single-person top-down model
(one inference, one candidate per frame, see target_ownership.py's module docstring). A "second
person" here is therefore represented the only way it can ever actually reach F-21 in production:
the OBSERVATION STREAM SWITCHES SOURCE. That is precisely the F-19 failure mechanism - the M15 crop
silently re-centres on a different body and every downstream stage keeps treating it as the same
person - so driving F-21 with a labelled, ground-truth-known source switch is a faithful model of
the input F-21 actually sees, and nothing more is claimed.

Because every observation carries a ground-truth label here (which real human produced it), this
harness measures the number the production logs CANNOT measure:

    SILENT WRONG-PERSON EMISSION - a frame that F-21 let through (should_emit=True) whose underlying
    human is not the human this ownership epoch locked onto.

That is the real F-19 defect. It is NOT the same as the `TARGET_SWITCH` event counter, which only
counts a fresh acquisition after a release - a switch F-21 has already declared out loud. A system
can have TARGET_SWITCH == 0 and still be silently emitting the wrong person; this suite separates
the two and reports both.

    python f21_adversarial.py
    python f21_adversarial.py --out-dir oak_v4_evidence/f21/adversarial
"""
import argparse
import io
import json
import os
import sys

import target_ownership as TO
from target_ownership import (NO_TARGET, ACQUIRING, LOCKED, TEMPORARILY_LOST,
                              REACQUIRING, RELEASED)

DT = 1.0 / 30.0                      # production frame interval (F-19 measured 29.9-30.0 fps)

# Two people standing 0.80 m apart at 2.0 m from the camera - a realistic installation spacing, and
# comfortably outside SWITCH_MARGIN_M (0.35) so position continuity CAN separate them. Cases where
# it deliberately cannot (crossing, same-spot handoff) set the positions equal on purpose.
A_POS = (0.00, -0.15, 2.00)
B_POS = (0.80, -0.15, 2.00)
A_SCALE = 0.46                       # torso span, metres
B_SCALE = 0.46

_results = []


def check(name, cond, detail=""):
    _results.append((name, bool(cond), detail))
    print("  %s  %s%s" % ("PASS" if cond else "FAIL", name, ("   [%s]" % detail) if detail else ""))


# --------------------------------------------------------------------------------------------
# scenario construction helpers - every step is (ground_truth_label, Observation)
# --------------------------------------------------------------------------------------------
def steps_person(label, pos, n, conf=0.85, scale=A_SCALE):
    return [(label, TO.Observation(True, pos, conf, scale)) for _ in range(n)]


def steps_absent(n):
    return [(None, TO.Observation(False, None, 0.0, None)) for _ in range(n)]


def steps_lowconf(label, pos, n, conf=0.15, scale=A_SCALE):
    """Present but below MIN_TARGET_CONFIDENCE - the machine must treat this as no observation."""
    return [(label, TO.Observation(True, pos, conf, scale)) for _ in range(n)]


def steps_path(label, p0, p1, n, conf=0.85, scale=A_SCALE):
    """Linear walk from p0 to p1 over n frames (inclusive of p0, exclusive of p1)."""
    out = []
    for i in range(n):
        f = float(i) / max(1, n)
        out.append((label, TO.Observation(
            True,
            tuple(p0[k] + (p1[k] - p0[k]) * f for k in range(3)),
            conf, scale)))
    return out


class Run(object):
    """One executed scenario, with every per-frame fact retained so metrics are derived from the
    record rather than asserted from memory."""

    def __init__(self, name, steps, cfg=None, dt=DT):
        self.name = name
        self.dt = dt
        self.own = TO.TargetOwnership(cfg or TO.OwnershipConfig())
        self.frames = []        # dicts: i, t, label, state, emit, epoch
        self.events = []        # ownership events + frame/label context
        for i, (label, obs) in enumerate(steps):
            t = i * dt
            state, emit = self.own.update(obs, t)
            for e in self.own.drain_events():
                e = dict(e)
                e["frame"] = i
                e["t"] = round(t, 4)
                e["label"] = label
                self.events.append(e)
            self.frames.append(dict(i=i, t=t, label=label, state=state, emit=bool(emit),
                                    epoch=self.own.epoch))
        self.m = self._metrics()

    # -- metric derivation -------------------------------------------------------------------
    def _epoch_owner_labels(self):
        """label of the human each ownership epoch actually locked onto (from the TARGET_LOCKED
        event's own frame), so a later emitted frame can be compared against it."""
        out = {}
        for e in self.events:
            if e["event"] == "TARGET_LOCKED":
                out[e["target_id"]] = e["label"]
        return out

    def _metrics(self):
        owners = self._epoch_owner_labels()
        emitted = [f for f in self.frames if f["emit"]]

        wrong = [f for f in emitted
                 if owners.get(f["epoch"]) is not None and f["label"] != owners[f["epoch"]]]
        episodes = 0
        prev = None
        for f in wrong:
            if prev is None or f["i"] != prev + 1:
                episodes += 1
            prev = f["i"]

        declared = [e for e in self.events if e["event"] == "TARGET_SWITCH"]
        releases = [e for e in self.events if e["event"] == "TARGET_RELEASED"]
        reacquires = [e for e in self.events if e["event"] == "TARGET_REACQUIRED"]
        temp_lost = [e for e in self.events if e["event"] == "TARGET_TEMP_LOST"]
        rejected = [e for e in self.events if e["event"] == "TARGET_REJECTED_CANDIDATE"]

        # FALSE RELEASE: released while the owner of that epoch was in fact observed, at its own
        # position, within the release window that just elapsed.
        false_releases = 0
        for r in releases:
            owner_label = owners.get(r["target_id"])
            lo = r["t"] - self.own.cfg.release_timeout_s
            present = [f for f in self.frames
                       if lo <= f["t"] <= r["t"] and f["label"] == owner_label]
            if present:
                false_releases += 1

        # FALSE REACQUISITION: relocked onto a human who is not the one the epoch owns.
        false_reacquires = sum(1 for e in reacquires
                               if owners.get(e["target_id"]) is not None
                               and e["label"] != owners[e["target_id"]])

        # OSCILLATION: completed LOCKED -> (TEMPORARILY_LOST|REACQUIRING) -> LOCKED round trips.
        osc = 0
        seen_lost = False
        for f in self.frames:
            if f["state"] == LOCKED and seen_lost:
                osc += 1
                seen_lost = False
            elif f["state"] in (TEMPORARILY_LOST, REACQUIRING):
                seen_lost = True

        # ACQUISITION LATENCY: first valid observation of a fresh acquisition -> first emitted frame.
        acq = []
        for e in self.events:
            if e["event"] != "TARGET_ACQUIRED":
                continue
            first_emit = next((f["i"] for f in self.frames if f["i"] >= e["frame"] and f["emit"]), None)
            j = e["frame"]
            while j > 0 and self.frames[j - 1]["label"] == e["label"]:
                j -= 1
            if first_emit is not None:
                acq.append(first_emit - j)

        # RELEASE LATENCY: last observation of the owner -> the TARGET_RELEASED event.
        rel = []
        for r in releases:
            owner_label = owners.get(r["target_id"])
            last_seen = None
            for f in self.frames:
                if f["i"] > r["frame"]:
                    break
                if f["label"] == owner_label:
                    last_seen = f["t"]
            if last_seen is not None:
                rel.append(round(r["t"] - last_seen, 4))

        # RECOVERY LATENCY: the frame the owner became re-matchable (the first REACQUIRING frame of
        # that episode) -> the first emitted frame. Derived from the state record rather than from
        # the label run, because during a CONFIDENCE dip the label is still the owner's while the
        # observation is not usable - walking back by label alone would silently measure the dip too.
        rec = []
        for e in reacquires:
            first_emit = next((f["i"] for f in self.frames
                               if f["i"] >= e["frame"] and f["emit"]), None)
            if first_emit is None:
                continue
            j = first_emit
            while j > 0 and self.frames[j - 1]["state"] == REACQUIRING:
                j -= 1
            rec.append(first_emit - j)

        return dict(
            frames=len(self.frames),
            emitted_frames=len(emitted),
            epochs=self.own.epoch,
            wrong_person_frames=len(wrong),
            wrong_person_episodes=episodes,
            declared_switch_events=len(declared),
            false_releases=false_releases,
            false_reacquisitions=false_reacquires,
            oscillation_cycles=osc,
            temp_lost_events=len(temp_lost),
            rejected_candidate_events=len(rejected),
            released_events=len(releases),
            reacquired_events=len(reacquires),
            # S30 aliases + the path gate's own counter, so a path scenario reads as what it tests
            releases=len(releases),
            reacquires=len(reacquires),
            path_rejections=sum(1 for e in self.events
                                if e.get("reason") == "path_walked_in"),
            acquisition_latency_frames=acq,
            release_latency_s=rel,
            recovery_latency_frames=rec,
            final_state=self.frames[-1]["state"] if self.frames else None,
        )

    # -- convenience queries -----------------------------------------------------------------
    def emitted_labels(self):
        return set(f["label"] for f in self.frames if f["emit"])

    def first_emit_of(self, label):
        return next((f["i"] for f in self.frames if f["emit"] and f["label"] == label), None)

    def first_event(self, kind):
        return next((e for e in self.events if e["event"] == kind), None)

    def count(self, kind):
        return sum(1 for e in self.events if e["event"] == kind)


# ================================================================================================
# 17 scenarios, in the brief's own order
# ================================================================================================
def s01_A_acquired_and_locked():
    r = Run("s01", steps_person("A", A_POS, 40))
    check("01 A locks", r.m["epochs"] == 1 and r.m["final_state"] == LOCKED,
          "epochs=%d state=%s" % (r.m["epochs"], r.m["final_state"]))
    # ACQUIRE_CONFIRM_FRAMES=5 means the 5th consecutive frame is the first emitted one, i.e. 4 frame
    # INTERVALS (133 ms at 30 fps) of latency - reported as intervals, not as a frame count, so it
    # can be compared against the 150/500/2000 ms F-20A budgets in the same units.
    check("01 acquisition latency is ACQUIRE_CONFIRM_FRAMES-1 intervals (133 ms at 30 fps)",
          r.m["acquisition_latency_frames"] == [4], str(r.m["acquisition_latency_frames"]))
    check("01 no wrong-person frames", r.m["wrong_person_frames"] == 0)
    check("01 every emitted frame is A", r.emitted_labels() == set(["A"]), str(r.emitted_labels()))
    return r


def s02_B_enters():
    # A is locked; the crop then hands off to B (the F-19 mechanism) 0.80 m away.
    r = Run("s02", steps_person("A", A_POS, 40) + steps_person("B", B_POS, 40))
    check("02 B is never emitted while A is owned", r.first_emit_of("B") is None)
    check("02 B is explicitly rejected, not silently absorbed",
          r.count("TARGET_REJECTED_CANDIDATE") > 0, "n=%d" % r.count("TARGET_REJECTED_CANDIDATE"))
    check("02 rejection reason is position_jump",
          r.first_event("TARGET_REJECTED_CANDIDATE")["reason"] == "position_jump",
          r.first_event("TARGET_REJECTED_CANDIDATE")["reason"])
    check("02 zero silent wrong-person frames", r.m["wrong_person_frames"] == 0)
    return r


def s03_B_more_centered():
    # A off-centre, B dead-centre. "More centred" must carry ZERO weight (report SS7).
    a_off = (0.45, -0.15, 2.00)
    r = Run("s03", steps_person("A", a_off, 40) + steps_person("B", (0.0, -0.15, 2.00), 40))
    check("03 a more-centred candidate does not win ownership", r.first_emit_of("B") is None)
    check("03 zero silent wrong-person frames", r.m["wrong_person_frames"] == 0)
    return r


def s04_B_closer_and_larger():
    # (a) B genuinely closer -> position separates them.
    r = Run("s04a", steps_person("A", A_POS, 40, scale=0.46)
            + steps_person("B", (0.10, -0.15, 1.20), 40, scale=0.62))
    check("04a a closer/larger candidate does not win ownership", r.first_emit_of("B") is None)
    check("04a zero silent wrong-person frames", r.m["wrong_person_frames"] == 0)

    # (b) B standing at A's OWN position but grossly larger - position cannot separate them, so the
    #     scale corroborator is the only thing left. This is the case SCALE_MARGIN_RATIO exists for.
    r2 = Run("s04b", steps_person("A", A_POS, 40, scale=0.40)
             + steps_person("B", A_POS, 40, scale=0.75))
    check("04b gross scale mismatch at the owner's own position is rejected",
          r2.first_emit_of("B") is None)
    ev = r2.first_event("TARGET_REJECTED_CANDIDATE")
    check("04b rejection reason is scale_mismatch", ev is not None and ev["reason"] == "scale_mismatch",
          ev["reason"] if ev else "no rejection event")
    return r, r2


def s05_B_higher_confidence():
    # The brief's critical rule: confidence must NEVER be sufficient to take ownership.
    r = Run("s05", steps_person("A", A_POS, 40, conf=0.55)
            + steps_person("B", B_POS, 40, conf=0.99))
    check("05 higher instantaneous confidence never wins ownership", r.first_emit_of("B") is None)
    check("05 zero silent wrong-person frames", r.m["wrong_person_frames"] == 0)
    return r


def s06_A_loses_confidence_temporarily():
    r = Run("s06", steps_person("A", A_POS, 40)
            + steps_lowconf("A", A_POS, 10)          # 0.33 s below the 0.3 conf gate
            + steps_person("A", A_POS, 40))
    check("06 a confidence dip causes TEMPORARILY_LOST, not a release",
          r.count("TARGET_TEMP_LOST") == 1 and r.count("TARGET_RELEASED") == 0,
          "temp_lost=%d released=%d" % (r.count("TARGET_TEMP_LOST"), r.count("TARGET_RELEASED")))
    check("06 same ownership epoch survives the dip", r.m["epochs"] == 1, "epochs=%d" % r.m["epochs"])
    check("06 recovery latency is REACQUIRE_CONFIRM_FRAMES-1 intervals (133 ms at 30 fps)",
          r.m["recovery_latency_frames"] == [4], str(r.m["recovery_latency_frames"]))
    check("06 zero false releases", r.m["false_releases"] == 0)
    return r


def s07_A_disappears_temporarily():
    r = Run("s07", steps_person("A", A_POS, 40) + steps_absent(15)
            + steps_person("A", A_POS, 40))
    check("07 a 0.5 s disappearance is survived without release",
          r.count("TARGET_RELEASED") == 0 and r.m["epochs"] == 1,
          "released=%d epochs=%d" % (r.count("TARGET_RELEASED"), r.m["epochs"]))
    check("07 reacquired to the SAME epoch", r.count("TARGET_REACQUIRED") == 1)
    check("07 zero false reacquisitions", r.m["false_reacquisitions"] == 0)
    return r


def s08_A_returns():
    # (a) inside REACQUIRE_WINDOW (2.0 s) -> fast path relock, epoch preserved.
    r_in = Run("s08a", steps_person("A", A_POS, 40) + steps_absent(54)     # 1.80 s
               + steps_person("A", A_POS, 40))
    check("08a return inside REACQUIRE_WINDOW relocks the same epoch",
          r_in.count("TARGET_REACQUIRED") == 1 and r_in.m["epochs"] == 1,
          "reacq=%d epochs=%d" % (r_in.count("TARGET_REACQUIRED"), r_in.m["epochs"]))

    # (b) REGRESSION GUARD for the defect this suite found. Before the fix in target_ownership.py's
    #     TEMPORARILY_LOST branch, an owner who stepped away for 2.33 s and returned to their EXACT
    #     original position at full confidence was rejected 51 consecutive times as "not_owner", held
    #     un-emitted for 1.87 s, RELEASED, and then re-acquired with a TARGET_SWITCH logged against a
    #     person who never moved. All four of those are asserted against here.
    r_out = Run("s08b", steps_person("A", A_POS, 40) + steps_absent(70)    # 2.33 s > 2.0 s window
                + steps_person("A", A_POS, 200))
    back = next(f["i"] for f in r_out.frames if f["i"] > 40 and f["label"] == "A")
    first_after = next(f["i"] for f in r_out.frames if f["i"] >= back and f["emit"])
    r_out.m["owner_unemitted_frames_after_return"] = first_after - back
    r_out.m["owner_unemitted_seconds_after_return"] = round((first_after - back) * DT, 3)

    check("08b a returning owner past REACQUIRE_WINDOW is still matched, not released",
          r_out.count("TARGET_RELEASED") == 0, "released=%d" % r_out.count("TARGET_RELEASED"))
    check("08b it is a REACQUIRE of the same epoch, not a new acquisition",
          r_out.count("TARGET_REACQUIRED") == 1 and r_out.m["epochs"] == 1,
          "reacq=%d epochs=%d" % (r_out.count("TARGET_REACQUIRED"), r_out.m["epochs"]))
    check("08b no TARGET_SWITCH is logged for a person who never moved",
          r_out.count("TARGET_SWITCH") == 0, "switches=%d" % r_out.count("TARGET_SWITCH"))
    check("08b the returning owner is not rejected as not_owner even once",
          r_out.count("TARGET_REJECTED_CANDIDATE") == 0,
          "rejections=%d" % r_out.count("TARGET_REJECTED_CANDIDATE"))
    check("08b the owner is un-emitted only for the confirm window, not seconds",
          r_out.m["owner_unemitted_frames_after_return"] == 4,
          "%d frames (%.2f s)" % (r_out.m["owner_unemitted_frames_after_return"],
                                  r_out.m["owner_unemitted_seconds_after_return"]))
    check("08b zero silent wrong-person frames", r_out.m["wrong_person_frames"] == 0)
    check("08b zero false releases", r_out.m["false_releases"] == 0)
    print("      [measured] owner returning after %.2f s away is re-emitted %d frames (%.2f s) later"
          % (70 * DT, r_out.m["owner_unemitted_frames_after_return"],
             r_out.m["owner_unemitted_seconds_after_return"]))
    return r_in, r_out


def s09_A_and_B_cross():
    """The named §11 limitation, MEASURED instead of assumed, in both discriminable and
    non-discriminable form."""
    # (a) equal scale: at the crossing point the two humans are geometrically indistinguishable.
    cross_a = (steps_person("A", A_POS, 40)
               + steps_path("A", A_POS, (0.40, -0.15, 2.00), 12)
               + steps_path("B", (0.40, -0.15, 2.00), B_POS, 12, scale=A_SCALE))
    r_same = Run("s09a", cross_a)
    leak_same = r_same.m["wrong_person_frames"]

    # (b) different builds: scale is then the only remaining discriminator.
    cross_b = (steps_person("A", A_POS, 40, scale=0.40)
               + steps_path("A", A_POS, (0.40, -0.15, 2.00), 12, scale=0.40)
               + steps_path("B", (0.40, -0.15, 2.00), B_POS, 12, scale=0.75))
    r_diff = Run("s09b", cross_b)

    check("09a crossing with an IDENTICAL build is NOT separable by geometry (measured, expected)",
          leak_same > 0, "wrong-person frames leaked=%d - the SS11 limitation, quantified" % leak_same)
    check("09b crossing with a DIFFERENT build is separated by the scale corroborator",
          r_diff.m["wrong_person_frames"] == 0,
          "wrong-person frames=%d" % r_diff.m["wrong_person_frames"])
    check("09a the leak is bounded, not a permanent hand-off",
          leak_same <= 12, "leaked=%d of 12 crossing frames" % leak_same)
    return r_same, r_diff


def s10_A_leaves_permanently():
    r = Run("s10", steps_person("A", A_POS, 40) + steps_absent(200))       # 6.67 s of nothing
    check("10 a permanent departure releases exactly once", r.count("TARGET_RELEASED") == 1)
    # The clock starts on the first MISSING frame (one interval after the last seen one) and fires on
    # the first frame strictly past the timeout - so measured from the owner's last observation the
    # latency is RELEASE_TIMEOUT + 2 frame intervals, not +1. Stated exactly rather than rounded.
    lat = r.m["release_latency_s"]
    check("10 release latency is RELEASE_TIMEOUT + 2 frame intervals",
          len(lat) == 1 and abs(lat[0] - (4.0 + 2 * DT)) < 1e-4, str(lat))
    check("10 machine settles in NO_TARGET", r.m["final_state"] == NO_TARGET, r.m["final_state"])
    check("10 zero false releases (the owner really had left)", r.m["false_releases"] == 0)
    return r


def s11_B_remains():
    # A leaves; B has been standing there the whole time. B must wait out A's full release budget.
    r = Run("s11", steps_person("A", A_POS, 40) + steps_person("B", B_POS, 200))
    check("11 B eventually acquires once A is genuinely gone",
          r.first_emit_of("B") is not None and r.m["epochs"] == 2,
          "epochs=%d first_B=%s" % (r.m["epochs"], r.first_emit_of("B")))
    check("11 zero silent wrong-person frames throughout", r.m["wrong_person_frames"] == 0)
    rel = r.first_event("TARGET_RELEASED")
    protect = (r.first_emit_of("B") - 40) * DT
    r.m["b_denied_seconds"] = round(protect, 3)
    check("11 B is denied for the full release budget plus a fresh confirm",
          protect >= 4.0, "B denied for %.2f s" % protect)
    assert rel is not None
    return r


def s12_B_acquires_only_after_explicit_release():
    r = Run("s12", steps_person("A", A_POS, 40) + steps_person("B", B_POS, 200))
    rel = r.first_event("TARGET_RELEASED")
    first_b = r.first_emit_of("B")
    check("12 no B frame is emitted before TARGET_RELEASED is logged",
          rel is not None and first_b is not None and first_b > rel["frame"],
          "released@%s first_B_emit@%s" % (rel["frame"] if rel else None, first_b))
    names = [e["event"] for e in r.events]
    rel_i = names.index("TARGET_RELEASED")
    lock_i = names.index("TARGET_LOCKED", rel_i)     # the SECOND lock, i.e. B's
    check("12 the release is explicit and ordered before the new lock",
          rel_i < lock_i and r.events[lock_i]["label"] == "B",
          "released@ev%d newlock@ev%d label=%s" % (rel_i, lock_i, r.events[lock_i]["label"]))
    return r


def s13_repeated_A_B_transitions():
    steps = steps_person("A", A_POS, 40)
    for _ in range(12):                                   # 12 full A/B/A/B cycles
        steps += steps_person("B", B_POS, 5)
        steps += steps_person("A", A_POS, 5)
    r = Run("s13", steps)
    check("13 repeated A/B/A/B transitions never emit B", r.first_emit_of("B") is None)
    check("13 zero silent wrong-person frames", r.m["wrong_person_frames"] == 0)
    check("13 ownership never actually changes hands", r.m["epochs"] == 1, "epochs=%d" % r.m["epochs"])
    check("13 no release despite 12 interruptions", r.count("TARGET_RELEASED") == 0)
    print("      [measured] oscillation cycles=%d over %d frames (%.1f s); B rejected %d times"
          % (r.m["oscillation_cycles"], r.m["frames"], r.m["frames"] * DT,
             r.count("TARGET_REJECTED_CANDIDATE")))
    return r


def s14_long_no_target_interval():
    r = Run("s14", steps_absent(900))                      # 30 s of an empty room
    check("14 an empty room produces no state at all",
          r.m["epochs"] == 0 and r.m["emitted_frames"] == 0 and len(r.events) == 0,
          "epochs=%d emitted=%d events=%d" % (r.m["epochs"], r.m["emitted_frames"], len(r.events)))
    check("14 machine stays in NO_TARGET", r.m["final_state"] == NO_TARGET, r.m["final_state"])
    return r


def s15_new_target_after_long_release():
    r = Run("s15", steps_person("A", A_POS, 40) + steps_absent(600)        # 20 s empty
            + steps_person("B", B_POS, 40))
    check("15 a new person after a long release acquires normally",
          r.m["epochs"] == 2 and r.first_emit_of("B") is not None,
          "epochs=%d" % r.m["epochs"])
    check("15 acquisition latency is unchanged by the long gap",
          r.m["acquisition_latency_frames"] == [4, 4], str(r.m["acquisition_latency_frames"]))
    check("15 zero silent wrong-person frames", r.m["wrong_person_frames"] == 0)
    check("15 the genuine hand-over IS declared as a TARGET_SWITCH",
          r.count("TARGET_SWITCH") == 1, "switches=%d" % r.count("TARGET_SWITCH"))
    return r


def s16_sidecar_restart_while_ownership_exists():
    """Process-level: ownership is per-process state with no persistence path of any kind.
    A restarted sidecar must therefore start from zero - no owner, no epoch, nothing emitted until a
    full fresh acquisition - and must announce itself with a NEW F-20A session id."""
    before = Run("s16-before", steps_person("A", A_POS, 40))
    check("16 ownership exists before the restart",
          before.m["final_state"] == LOCKED and before.own.epoch == 1)

    after = TO.TargetOwnership(TO.OwnershipConfig())       # exactly what a new process constructs
    check("16 a restarted process owns nothing",
          after.state == NO_TARGET and after.epoch == 0 and after.owner_pos is None
          and after.owner_since is None and after.switch_count == 0,
          "state=%s epoch=%d" % (after.state, after.epoch))
    check("16 no ownership persistence path exists in the module",
          not any(hasattr(TO, n) for n in ("save", "load", "restore", "dump", "STATE_FILE")),
          "target_ownership.py exposes no save/load")

    emits = []
    for i in range(10):
        _st, em = after.update(TO.Observation(True, A_POS, 0.85, A_SCALE), i * DT)
        emits.append(em)
    check("16 the restarted process re-earns the lock from scratch (no stale carry-over)",
          emits[:4] == [False] * 4 and emits[4] is True, str(emits))

    # F-20A: a new process must present a NEW producer session id, so Unity flushes its pose buffer
    # rather than blending a pre-restart pose across the boundary. Two REAL interpreter launches.
    py = sys.executable
    sids = []
    for _ in range(2):
        import subprocess
        out = subprocess.check_output(
            [py, "-c", "import wholebody_udp_sender as W; print(W.SESSION_ID)"],
            cwd=os.path.dirname(os.path.abspath(__file__)))
        sids.append(out.decode().strip())
    check("16 each sidecar process presents a distinct F-20A session id",
          len(sids) == 2 and sids[0] != sids[1] and all(len(s) == 12 for s in sids),
          "sids=%s" % sids)
    return before


def s17_unity_restart_while_ownership_exists():
    """The consumer restarting must not disturb the producer's ownership. Two things are actually
    exercised here, over a REAL loopback socket rather than by inspection:
      (a) the producer's socket survives the receiver vanishing (Windows maps an ICMP port-unreachable
          onto the sending UDP socket - an unhandled raise here would kill the sidecar, and
          wholebody_udp_sender.py L938's sendto is NOT wrapped in try/except);
      (b) ownership state is bit-identical across the outage, because the UDP path is one-way and
          carries no consumer feedback at all."""
    import socket
    own = TO.TargetOwnership(TO.OwnershipConfig())
    for i in range(40):
        own.update(TO.Observation(True, A_POS, 0.85, A_SCALE), i * DT)
    snap_before = dict(state=own.state, epoch=own.epoch, owner_pos=own.owner_pos,
                       owner_since=own.owner_since, switch_count=own.switch_count)
    check("17 ownership is LOCKED before the consumer restarts", own.state == LOCKED)

    addr = ("127.0.0.1", 8913)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(addr)
    payload = json.dumps({"lm": [[0.0, 0.0, 0.0, 0.0]] * 33, "sid": "f21test"}).encode("utf-8")
    for _ in range(30):
        tx.sendto(payload, addr)
    rx.close()                                             # Unity quits

    send_error = None
    sent_during_outage = 0
    for i in range(300):                                   # 10 s of production rate against nothing
        try:
            tx.sendto(payload, addr)
            sent_during_outage += 1
        except Exception as e:                             # noqa: BLE001 - the point is to catch ANY
            send_error = "%s: %s" % (type(e).__name__, e)
            break
        own.update(TO.Observation(True, A_POS, 0.85, A_SCALE), (40 + i) * DT)

    check("17 the producer's socket survives the consumer disappearing",
          send_error is None and sent_during_outage == 300,
          send_error or "%d/300 datagrams sent into a closed port" % sent_during_outage)

    rx2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Unity comes back
    rx2.bind(addr)
    rx2.settimeout(1.0)
    tx.sendto(payload, addr)
    got = None
    try:
        got, _ = rx2.recvfrom(65535)
    except Exception:
        pass
    check("17 the restarted consumer receives the very next datagram", got is not None,
          "%d bytes" % (len(got) if got else 0))

    snap_after = dict(state=own.state, epoch=own.epoch, owner_pos=own.owner_pos,
                      owner_since=own.owner_since, switch_count=own.switch_count)
    check("17 ownership is untouched by the consumer restart",
          snap_after == snap_before, "before=%s after=%s" % (snap_before, snap_after))
    check("17 no TARGET_SWITCH/RELEASE was produced by the consumer outage",
          own.switch_count == 0 and not any(e["event"] in ("TARGET_SWITCH", "TARGET_RELEASED")
                                            for e in own.events))
    tx.close()
    rx2.close()
    return None


# ================================================================================================
# ==============================================================================================
# S30 PATH-CONSISTENCY SCENARIOS (P1-P11)
#
# S27 measured a silent wrong-person hand-off on real footage: the owner is lost, a DIFFERENT body
# is correctly refused while it is far away, it keeps walking, it arrives inside the frozen owner's
# position margin, and it is re-locked under the SAME target_id with TARGET_SWITCH still reading 0.
# The path-consistency gate closes that by asking where a candidate came FROM, using only positions
# the module already holds.
#
# These cases pin BOTH directions of that gate, because a gate that only ever refuses is not a fix,
# it is an outage: P1/P2/P7/P8 must still REACQUIRE, P3/P4/P6 must REFUSE, P5/P9/P10 check that
# refusing does not become oscillation or identity inheritance, and P11 is the control that proves
# the gate is what changed the outcome.
# ==============================================================================================
NEAR_A = (0.10, -0.15, 2.00)          # 0.10 m from A_POS - inside SWITCH_MARGIN_M
FAR_LEFT = (-1.60, -0.15, 2.00)       # well outside it, on the opposite side from B_POS


def p01_stationary_owner_returns():
    """A locked, A vanishes entirely (no observation at all), A reappears at the same spot.
    Nothing was tracked walking in, so the gate must not fire."""
    steps = (steps_person("A", A_POS, 8) + steps_absent(20) + steps_person("A", A_POS, 12))
    r = Run("p01_stationary_owner_returns", steps)
    check("P1 owner is re-locked, not held out", r.m["reacquires"] == 1,
          "reacquires=%d" % r.m["reacquires"])
    check("P1 same epoch - a reacquire, not a new acquisition", r.m["epochs"] == 1,
          "epochs=%d" % r.m["epochs"])
    check("P1 no path rejection fired on a body that never walked in",
          r.m["path_rejections"] == 0, "path_rejections=%d" % r.m["path_rejections"])
    check("P1 no wrong-person frames", r.m["wrong_person_frames"] == 0)
    return r


def p02_owner_moves_slightly_while_absent():
    """A returns a little away from where it left, still inside the margin, with no observed
    approach. Still the owner."""
    steps = (steps_person("A", A_POS, 8) + steps_absent(15) + steps_person("A", NEAR_A, 12))
    r = Run("p02_owner_moves_slightly_while_absent", steps)
    check("P2 owner re-locked after a small displacement", r.m["reacquires"] == 1,
          "reacquires=%d" % r.m["reacquires"])
    check("P2 same epoch", r.m["epochs"] == 1, "epochs=%d" % r.m["epochs"])
    check("P2 no path rejection", r.m["path_rejections"] == 0)
    return r


def p03_impostor_walks_into_frozen_position():
    """THE S27 CASE, in deterministic form. A is lost; B appears far away and walks continuously to
    A's frozen position over 40 frames (1.33 s), arriving dead on it at full confidence and an
    identical torso span. Position and scale both say yes. The path says no."""
    steps = (steps_person("A", A_POS, 8) + steps_absent(4)
             + steps_path("B", B_POS, A_POS, 40) + steps_person("B", A_POS, 30))
    r = Run("p03_impostor_walks_into_frozen_position", steps)
    check("P3 B is NEVER locked as the owner", r.m["reacquires"] == 0,
          "reacquires=%d" % r.m["reacquires"])
    check("P3 the path gate is what refused it", r.m["path_rejections"] > 0,
          "path_rejections=%d" % r.m["path_rejections"])
    check("P3 ZERO wrong-person frames emitted", r.m["wrong_person_frames"] == 0,
          "wrong=%d" % r.m["wrong_person_frames"])
    check("P3 no frame is emitted at all while B is refused",
          r.m["emitted_frames"] == 4,
          "emitted=%d - A's 8 frames minus the 4 spent confirming acquisition"
          % r.m["emitted_frames"])
    return r


def p04_impostor_enters_from_the_opposite_side():
    """Same as P3 mirrored. The rule must not be 'arrives from the direction the owner left'."""
    steps = (steps_person("A", A_POS, 8) + steps_absent(4)
             + steps_path("B", FAR_LEFT, A_POS, 40) + steps_person("B", A_POS, 30))
    r = Run("p04_impostor_enters_from_the_opposite_side", steps)
    check("P4 B is never locked, approaching from the other side", r.m["reacquires"] == 0)
    check("P4 refused by the path gate", r.m["path_rejections"] > 0,
          "path_rejections=%d" % r.m["path_rejections"])
    check("P4 zero wrong-person frames", r.m["wrong_person_frames"] == 0)
    return r


def p05_A_B_A_rapid_transitions_do_not_oscillate():
    """A present, B walks in, A returns, B walks in again. Ownership must not flap."""
    steps = (steps_person("A", A_POS, 8)
             + steps_absent(3) + steps_path("B", B_POS, A_POS, 20)
             + steps_absent(3) + steps_person("A", A_POS, 10)
             + steps_absent(3) + steps_path("B", B_POS, A_POS, 20)
             + steps_absent(3) + steps_person("A", A_POS, 10))
    r = Run("p05_A_B_A_rapid_transitions_do_not_oscillate", steps)
    check("P5 ownership never leaves A", r.m["epochs"] == 1, "epochs=%d" % r.m["epochs"])
    check("P5 zero wrong-person frames across the whole sequence",
          r.m["wrong_person_frames"] == 0, "wrong=%d" % r.m["wrong_person_frames"])
    check("P5 no release - A keeps coming back inside the budget", r.m["releases"] == 0,
          "releases=%d" % r.m["releases"])
    check("P5 every emitted frame is A",
          all(f["label"] == "A" for f in r.frames if f["emit"]))
    return r


def p06_repeated_touches_accumulate_no_credit():
    """B walks in, touches the frozen position, retreats, and does it again, six times. Confirm
    credit must not accumulate across the attempts."""
    steps = steps_person("A", A_POS, 8) + steps_absent(3)
    for _ in range(6):
        steps = (steps + steps_path("B", B_POS, A_POS, 8) + steps_person("B", A_POS, 3)
                 + steps_path("B", A_POS, B_POS, 8))
    r = Run("p06_repeated_touches_accumulate_no_credit", steps)
    check("P6 six approaches never add up to a reacquire", r.m["reacquires"] == 0,
          "reacquires=%d" % r.m["reacquires"])
    check("P6 zero wrong-person frames", r.m["wrong_person_frames"] == 0)
    check("P6 each approach is refused on its own merits", r.m["path_rejections"] >= 6,
          "path_rejections=%d" % r.m["path_rejections"])
    return r


def p07_owner_returns_by_an_unusual_route():
    """The gate must not be 'must return the way you left'. A drifts while still the owner, is
    unobserved for a while, and reappears inside the margin having (as far as the camera is
    concerned) come from nowhere - which is what stepping back out from behind an obstruction
    actually looks like."""
    steps = (steps_path("A", A_POS, (0.30, -0.15, 2.00), 6)
             + steps_absent(18)
             + steps_person("A", A_POS, 14))
    r = Run("p07_owner_returns_by_an_unusual_route", steps)
    check("P7 an unusual but unobserved return still reacquires", r.m["reacquires"] == 1,
          "reacquires=%d" % r.m["reacquires"])
    check("P7 same epoch", r.m["epochs"] == 1)
    check("P7 no path rejection", r.m["path_rejections"] == 0)
    return r


def p08_owner_returns_past_reacquire_window_within_release():
    """ADR-054's case, now with the gate present: 2.33 s is past REACQUIRE_WINDOW_S (2.0) but well
    inside RELEASE_TIMEOUT_S (4.0). The frozen reference must still be usable and the returning
    owner must still be admitted - the gate must not have quietly re-broken this."""
    cfg = TO.OwnershipConfig()
    gap = int(round(2.33 / DT))
    steps = steps_person("A", A_POS, 8) + steps_absent(gap) + steps_person("A", A_POS, 12)
    r = Run("p08_owner_returns_past_reacquire_window_within_release", steps, cfg=cfg)
    check("P8 loss exceeded REACQUIRE_WINDOW_S", gap * DT > cfg.reacquire_window_s,
          "%.2fs > %.2fs" % (gap * DT, cfg.reacquire_window_s))
    check("P8 loss stayed inside RELEASE_TIMEOUT_S", gap * DT < cfg.release_timeout_s,
          "%.2fs < %.2fs" % (gap * DT, cfg.release_timeout_s))
    check("P8 owner re-locked, not released", r.m["reacquires"] == 1 and r.m["releases"] == 0,
          "reacquires=%d releases=%d" % (r.m["reacquires"], r.m["releases"]))
    check("P8 no TARGET_SWITCH for a person who never moved",
          r.m["declared_switch_events"] == 0)
    return r


def p09_true_release_when_owner_never_returns():
    """Nobody returns at all. The machine must release on schedule and go quiet - the gate must not
    hold an epoch open forever just because it is busy refusing candidates."""
    steps = (steps_person("A", A_POS, 8) + steps_absent(int(round(5.0 / DT))))
    r = Run("p09_true_release_when_owner_never_returns", steps)
    check("P9 released", r.m["releases"] == 1, "releases=%d" % r.m["releases"])
    check("P9 ends with no target", r.frames[-1]["state"] in (NO_TARGET, RELEASED),
          "state=%s" % r.frames[-1]["state"])
    check("P9 no emission after the owner vanished",
          all(not f["emit"] for f in r.frames[8:]))
    return r


def p10_new_person_after_true_release_acquires_fresh():
    """After a genuine release, B must acquire as a NEW epoch with a DECLARED switch - the gate must
    not leak the old identity onto whoever shows up next, and must not block them either."""
    steps = (steps_person("A", A_POS, 8) + steps_absent(int(round(4.5 / DT)))
             + steps_path("B", B_POS, A_POS, 20) + steps_person("B", A_POS, 20))
    r = Run("p10_new_person_after_true_release_acquires_fresh", steps)
    check("P10 released first", r.m["releases"] == 1, "releases=%d" % r.m["releases"])
    check("P10 B acquires as a NEW epoch", r.m["epochs"] == 2, "epochs=%d" % r.m["epochs"])
    check("P10 the hand-over is DECLARED, not silent", r.m["declared_switch_events"] == 1,
          "switches=%d" % r.m["declared_switch_events"])
    check("P10 no wrong-person frames - B is emitted as B, under B's own epoch",
          r.m["wrong_person_frames"] == 0, "wrong=%d" % r.m["wrong_person_frames"])
    return r


def p11_control_gate_off_reproduces_the_silent_handoff():
    """The same P3 input with path_consistency=False must reproduce the OLD behaviour exactly - the
    silent hand-off. This is the control: without it, 'the gate fixed it' is unfalsifiable."""
    steps = (steps_person("A", A_POS, 8) + steps_absent(4)
             + steps_path("B", B_POS, A_POS, 40) + steps_person("B", A_POS, 30))
    off = Run("p11_control_path_off", steps, cfg=TO.OwnershipConfig(path_consistency=False))
    on = Run("p11_control_path_on", steps, cfg=TO.OwnershipConfig(path_consistency=True))
    check("P11 CONTROL: with the gate OFF the wrong person IS emitted",
          off.m["wrong_person_frames"] > 0, "wrong=%d" % off.m["wrong_person_frames"])
    check("P11 CONTROL: and it is SILENT - no switch, no release, same epoch",
          off.m["declared_switch_events"] == 0 and off.m["releases"] == 0
          and off.m["epochs"] == 1,
          "switches=%d releases=%d epochs=%d" % (off.m["declared_switch_events"],
                                                 off.m["releases"], off.m["epochs"]))
    check("P11 with the gate ON the same input emits zero wrong-person frames",
          on.m["wrong_person_frames"] == 0, "wrong=%d" % on.m["wrong_person_frames"])
    return off, on


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join("oak_v4_evidence", "f21", "adversarial"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    print("=" * 94)
    print(" F-21 deterministic multi-target adversarial suite")
    print(" proves the OWNERSHIP LOGIC - not real two-person detector behaviour (see module docstring)")
    print("=" * 94)

    runs = {}
    order = [
        ("01 A acquired and locked", s01_A_acquired_and_locked),
        ("02 B enters", s02_B_enters),
        ("03 B becomes more centered", s03_B_more_centered),
        ("04 B becomes closer/larger", s04_B_closer_and_larger),
        ("05 B has higher instantaneous confidence", s05_B_higher_confidence),
        ("06 A temporarily loses confidence", s06_A_loses_confidence_temporarily),
        ("07 A temporarily disappears", s07_A_disappears_temporarily),
        ("08 A returns", s08_A_returns),
        ("09 A and B cross", s09_A_and_B_cross),
        ("10 A leaves permanently", s10_A_leaves_permanently),
        ("11 B remains", s11_B_remains),
        ("12 B acquires only after explicit A release", s12_B_acquires_only_after_explicit_release),
        ("13 A/B/A/B repeated transitions", s13_repeated_A_B_transitions),
        ("14 Long no-target interval", s14_long_no_target_interval),
        ("15 New target after long release", s15_new_target_after_long_release),
        ("16 Sidecar/session restart while ownership exists", s16_sidecar_restart_while_ownership_exists),
        ("17 Unity restart while ownership exists", s17_unity_restart_while_ownership_exists),
        ("P1 stationary owner returns", p01_stationary_owner_returns),
        ("P2 owner moves slightly while absent", p02_owner_moves_slightly_while_absent),
        ("P3 impostor walks into the frozen position", p03_impostor_walks_into_frozen_position),
        ("P4 impostor enters from the opposite side", p04_impostor_enters_from_the_opposite_side),
        ("P5 A/B/A rapid transitions do not oscillate", p05_A_B_A_rapid_transitions_do_not_oscillate),
        ("P6 repeated touches accumulate no credit", p06_repeated_touches_accumulate_no_credit),
        ("P7 owner returns by an unusual route", p07_owner_returns_by_an_unusual_route),
        ("P8 owner returns past the reacquire window",
         p08_owner_returns_past_reacquire_window_within_release),
        ("P9 true release when the owner never returns", p09_true_release_when_owner_never_returns),
        ("P10 new person after a true release", p10_new_person_after_true_release_acquires_fresh),
        ("P11 CONTROL - gate off reproduces the silent hand-off",
         p11_control_gate_off_reproduces_the_silent_handoff),
    ]
    # P11 deliberately runs the machine with the S30 gate DISABLED, to prove the gate is what
    # changed the outcome. Its wrong-person frames are the OLD behaviour being demonstrated on
    # purpose - folding them into an aggregate that gets read as "what production does" would be
    # a straightforwardly false number. That one run is excluded, and the exclusion is printed
    # rather than left implicit.
    EXCLUDE_FROM_AGGREGATE = ("p11_control_path_off",)
    excluded = []
    for title, fn in order:
        print("\n-- %s" % title)
        out = fn()
        for r in (out if isinstance(out, tuple) else (out,)):
            if isinstance(r, Run):
                if r.name in EXCLUDE_FROM_AGGREGATE:
                    excluded.append(r.name)
                    continue
                runs[r.name] = r.m

    # ---- aggregate the seven metrics the brief asks for, across every scenario ----------------
    agg = dict(
        scenarios=len(runs),
        excluded_control_runs=excluded,
        total_frames=sum(m["frames"] for m in runs.values()),
        total_emitted_frames=sum(m["emitted_frames"] for m in runs.values()),
        wrong_person_switches=sum(m["wrong_person_episodes"] for m in runs.values()),
        wrong_person_frames=sum(m["wrong_person_frames"] for m in runs.values()),
        false_releases=sum(m["false_releases"] for m in runs.values()),
        false_reacquisitions=sum(m["false_reacquisitions"] for m in runs.values()),
        ownership_oscillation_cycles=sum(m["oscillation_cycles"] for m in runs.values()),
        declared_switch_events=sum(m["declared_switch_events"] for m in runs.values()),
        acquisition_latency_frames=sorted(set(
            f for m in runs.values() for f in m["acquisition_latency_frames"])),
        release_latency_s=sorted(set(
            round(v, 4) for m in runs.values() for v in m["release_latency_s"])),
        recovery_latency_frames=sorted(set(
            f for m in runs.values() for f in m["recovery_latency_frames"])),
    )
    # s09a's crossing leak is a KNOWN, deliberately-provoked geometric limitation, not a defect of
    # the logic - it is reported separately so the headline number stays honest in both directions.
    agg["wrong_person_frames_excluding_identical_build_crossing"] = (
        agg["wrong_person_frames"] - runs.get("s09a", {}).get("wrong_person_frames", 0))
    agg["identical_build_crossing_leak_frames"] = runs.get("s09a", {}).get("wrong_person_frames", 0)

    n = len(_results)
    p = sum(1 for _, ok, _ in _results if ok)
    print("\n" + "=" * 94)
    print(" AGGREGATE METRICS (all scenarios)")
    print("=" * 94)
    for k in ("total_frames", "total_emitted_frames", "wrong_person_switches", "wrong_person_frames",
              "wrong_person_frames_excluding_identical_build_crossing",
              "identical_build_crossing_leak_frames", "false_releases", "false_reacquisitions",
              "ownership_oscillation_cycles", "declared_switch_events",
              "acquisition_latency_frames", "release_latency_s", "recovery_latency_frames"):
        print("  %-52s %s" % (k, agg[k]))
    print("-" * 94)
    print(" %d/%d assertions passed" % (p, n))
    if p != n:
        print("\nFAILED:")
        for name, ok, detail in _results:
            if not ok:
                print("  %s  %s" % (name, detail))
    print("=" * 94)

    with io.open(os.path.join(a.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(aggregate=agg, per_scenario=runs,
                       assertions_total=n, assertions_passed=p), f, indent=2, default=str)
    with io.open(os.path.join(a.out_dir, "assertions.txt"), "w", encoding="utf-8") as f:
        for name, ok, detail in _results:
            f.write("%s  %s%s\n" % ("PASS" if ok else "FAIL", name,
                                    ("   [%s]" % detail) if detail else ""))
    print(" evidence -> %s" % a.out_dir)
    return 0 if p == n else 1


if __name__ == "__main__":
    sys.exit(main())
