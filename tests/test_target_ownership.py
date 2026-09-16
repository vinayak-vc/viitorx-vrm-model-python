#!/usr/bin/env python3

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(_os.path.join(_d, "_sidecar_path.py")):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import _sidecar_path  # noqa: F401  - puts the sidecar root and every tools/ group on sys.path
"""F-21 deterministic unit tests for target_ownership.py.

Self-contained runner (no pytest dependency), matching test_joint_tracker.py's pattern. Covers as
many of the F-21 brief's SS13 adversarial cases (1-12 of 15) as are expressible as a pure sequence of
per-frame Observations; the remaining three (sidecar restart, Unity restart, F-20A stale interaction)
are integration/architectural concerns tested against the real sidecar and Unity instead - see
docs/F21_SINGLE_PERSON_TARGET_OWNERSHIP_2026-09-14.md SS13/SS14.

    python test_target_ownership.py
"""
import sys

from target_ownership import (TargetOwnership, OwnershipConfig, Observation,
                              NO_TARGET, ACQUIRING, LOCKED, TEMPORARILY_LOST, REACQUIRING, RELEASED)

DT = 1.0 / 30.0
_results = []

A_POS = (0.0, 0.0, 0.90)
B_POS = (0.60, 0.0, 0.90)     # 0.60 m from A - beyond the default 0.35 m switch margin
A_SCALE = 0.55
B_SCALE_CLOSE = 0.50          # within the default 0.45 scale-margin ratio of A_SCALE
B_SCALE_FAR = 0.25            # well outside it


def check(name, cond, detail=""):
    _results.append((name, bool(cond), detail))
    print("  %-4s %-56s %s" % ("PASS" if cond else "FAIL", name, detail))


def new_owner(cfg=None):
    return TargetOwnership(cfg or OwnershipConfig())


def feed(own, frames, t0=0.0, dt=DT):
    """frames: list of (pos_or_None, conf, scale_or_None). Returns list of (state, emit)."""
    out = []
    t = t0
    for pos, conf, scale in frames:
        obs = Observation(valid=pos is not None, pos=pos, conf=conf, scale=scale)
        st, emit = own.update(obs, t)
        out.append((st, emit))
        t += dt
    return out


def const(pos, conf, scale, n):
    return [(pos, conf, scale)] * n


def absent(n):
    return [(None, 0.0, None)] * n


def events_of(own, kind=None):
    return [e for e in own.events if kind is None or e["event"] == kind]


# ---------------------------------------------------------------- 1
def t01_basic_acquisition():
    own = new_owner()
    res = feed(own, const(A_POS, 0.9, A_SCALE, 5))
    check("01 locks after exactly ACQUIRE_CONFIRM_FRAMES",
          [s for s, _ in res] == [ACQUIRING, ACQUIRING, ACQUIRING, ACQUIRING, LOCKED],
          "states=%s" % [s for s, _ in res])
    check("01 emits only on the locking frame",
          [e for _, e in res] == [False, False, False, False, True])
    check("01 TARGET_ACQUIRED + TARGET_LOCKED logged once",
          len(events_of(own, "TARGET_ACQUIRED")) == 1 and len(events_of(own, "TARGET_LOCKED")) == 1)


# ---------------------------------------------------------------- 2 (SS8 scenario A)
def t02_no_oscillation_on_entry():
    own = new_owner()
    res = feed(own, const(A_POS, 0.9, A_SCALE, 60))
    check("02 locks exactly once, then holds LOCKED",
          res[4][0] == LOCKED and all(s == LOCKED for s, _ in res[5:]),
          "states unique=%s" % sorted(set(s for s, _ in res)))
    check("02 no TARGET_SWITCH on first-ever acquisition", len(events_of(own, "TARGET_SWITCH")) == 0)


# ---------------------------------------------------------------- 3 (SS13.1, SS8 scenario B)
def t03_similar_confidence_second_person_rejected():
    own = new_owner()
    feed(own, const(A_POS, 0.9, A_SCALE, 5))
    res = feed(own, const(B_POS, 0.95, A_SCALE, 60), t0=5 * DT)   # B: higher conf, same scale
    check("03 owner NOT stolen by a higher-confidence second person",
          all(s != LOCKED or own.epoch == 1 for s, _ in res))
    check("03 candidate explicitly rejected, not silently ignored",
          len(events_of(own, "TARGET_REJECTED_CANDIDATE")) >= 1)
    check("03 zero silent switches", own.switch_count == 0)


# ---------------------------------------------------------------- 4 (SS13.2)
def t04_different_scale_rejected_at_owner_position():
    own = new_owner()
    feed(own, const(A_POS, 0.9, A_SCALE, 5))
    # same POSITION as the owner (impossible for a real person - scale can't jump), different scale
    res = feed(own, const(A_POS, 0.9, B_SCALE_FAR, 3), t0=5 * DT)
    check("04 gross scale mismatch at owner position -> temp-lost, not accepted",
          res[0][0] == TEMPORARILY_LOST, "state=%s" % res[0][0])
    check("04 reason recorded as scale_mismatch",
          any(e.get("reason") == "scale_mismatch" for e in events_of(own, "TARGET_TEMP_LOST")))


# ---------------------------------------------------------------- 5 (SS13.3)
def t05_confidence_dip_within_gate_does_not_lose_owner():
    own = new_owner()
    feed(own, const(A_POS, 0.9, A_SCALE, 5))
    # confidence dips but stays >= MIN_TARGET_CONFIDENCE (0.3) - same person, just less confident
    res = feed(own, const(A_POS, 0.31, A_SCALE, 30), t0=5 * DT)
    check("05 owner retained through an in-gate confidence dip",
          all(s == LOCKED for s, _ in res), "states=%s" % sorted(set(s for s, _ in res)))
    check("05 no TEMP_LOST events", len(events_of(own, "TARGET_TEMP_LOST")) == 0)


# ---------------------------------------------------------------- 6 (SS13.4, SS8 scenario C)
def t06_partial_occlusion_then_return():
    own = new_owner()
    feed(own, const(A_POS, 0.9, A_SCALE, 5))
    res = feed(own, absent(6), t0=5 * DT)                          # brief occlusion
    check("06 goes TEMPORARILY_LOST on occlusion, not RELEASED",
          res[0][0] == TEMPORARILY_LOST and all(s != RELEASED for s, _ in res))
    res2 = feed(own, const(A_POS, 0.9, A_SCALE, 5), t0=11 * DT)    # A returns at the same spot
    check("06 reacquires the SAME owner (epoch unchanged)",
          res2[-1][0] == LOCKED and own.epoch == 1, "epoch=%d" % own.epoch)
    check("06 owner_since is the ORIGINAL acquisition time, not reset by the reacquire",
          abs(own.owner_since - 4 * DT) < 1e-9, "owner_since=%s expected=%s" % (own.owner_since, 4 * DT))


# ---------------------------------------------------------------- 7 (SS13.5, SS8 scenario D)
def t07_owner_leaves_release_then_new_acquire():
    cfg = OwnershipConfig(release_timeout_s=1.0)
    own = new_owner(cfg)
    feed(own, const(A_POS, 0.9, A_SCALE, 5))
    n_absent = int(1.2 / DT) + 2
    res = feed(own, absent(n_absent), t0=5 * DT)
    check("07 releases after RELEASE_TIMEOUT with nobody returning",
          any(s == RELEASED for s, _ in res), "states tail=%s" % [s for s, _ in res[-3:]])
    res2 = feed(own, const(B_POS, 0.9, A_SCALE, 5), t0=(5 + n_absent) * DT)
    check("07 a NEW person can acquire only after release",
          res2[-1][0] == LOCKED and own.epoch == 2, "epoch=%d" % own.epoch)
    check("07 TARGET_SWITCH logged with a reason",
          len(events_of(own, "TARGET_SWITCH")) == 1 and "reason" in events_of(own, "TARGET_SWITCH")[0])


# ---------------------------------------------------------------- 8 (crossing, SS8 scenario E)
def t08_crossing_does_not_transfer_ownership():
    own = new_owner()
    feed(own, const(A_POS, 0.9, A_SCALE, 5))
    frames = []
    for i in range(10):                                    # A drifts slightly (normal walking)
        frames.append(((0.02 * i, 0.0, 0.90), 0.9, A_SCALE))
    frames.append((B_POS, 0.95, A_SCALE))                   # one frame: model briefly latches onto B
    for i in range(10):                                     # A continues from where it left off
        frames.append(((0.18 + 0.02 * i, 0.0, 0.90), 0.9, A_SCALE))
    res = feed(own, frames, t0=5 * DT)
    check("08 momentary jump to B does not lock B",
          own.epoch == 1, "epoch=%d" % own.epoch)
    check("08 the B-frame itself is rejected, not accepted",
          any(e.get("reason") == "position_jump" for e in
              events_of(own, "TARGET_REJECTED_CANDIDATE") + events_of(own, "TARGET_TEMP_LOST")))
    check("08 owner is LOCKED again by the end (A's own return)",
          res[-1][0] == LOCKED)
    check("08 zero switches during the crossing", own.switch_count == 0)


# ---------------------------------------------------------------- 9 (SS13.7, SS8 scenario H building block)
def t09_flickering_candidate_never_locks():
    own = new_owner()
    frames = []
    for _ in range(20):
        frames += [(B_POS, 0.9, A_SCALE), (None, 0.0, None), (None, 0.0, None)]
    res = feed(own, frames)
    check("09 a candidate that never holds ACQUIRE_CONFIRM_FRAMES in a row never locks",
          own.epoch == 0 and all(s != LOCKED for s, _ in res))


# ---------------------------------------------------------------- 10 (SS13.8)
def t10_owner_gone_then_candidate_appears_not_reassigned():
    own = new_owner()
    feed(own, const(A_POS, 0.9, A_SCALE, 5))
    res = feed(own, absent(3) + const(B_POS, 0.9, A_SCALE, 20), t0=5 * DT)
    check("10 owner absence immediately followed by a DIFFERENT candidate stays lost, not B",
          own.epoch == 1 and all(s != LOCKED for s, _ in res),
          "states unique=%s" % sorted(set(s for s, _ in res)))
    check("10 candidate rejected every time it's checked",
          len(events_of(own, "TARGET_REJECTED_CANDIDATE")) >= 1)


# ---------------------------------------------------------------- 11 (SS13.9)
def t11_owner_returns_after_candidate_reacquires_owner_not_candidate():
    own = new_owner()
    feed(own, const(A_POS, 0.9, A_SCALE, 5))
    frames = absent(3) + const(B_POS, 0.9, A_SCALE, 5) + const(A_POS, 0.9, A_SCALE, 6)
    res = feed(own, frames, t0=5 * DT)
    check("11 reacquisition matches the FROZEN owner reference, not whoever arrived first",
          res[-1][0] == LOCKED and own.epoch == 1, "epoch=%d state=%s" % (own.epoch, res[-1][0]))


# ---------------------------------------------------------------- 12 (SS13.10, SS8 scenario H)
def t12_rapid_A_B_A_cycle_is_rate_limited_not_oscillating():
    cfg = OwnershipConfig(release_timeout_s=0.5)
    own = new_owner(cfg)
    t = 0.0
    # A locks
    feed(own, const(A_POS, 0.9, A_SCALE, 5), t0=t); t += 5 * DT
    check("12 A locks", own.epoch == 1 and own.state == LOCKED)
    # A leaves long enough to release, B locks
    n = int(0.6 / DT) + 2
    feed(own, absent(n), t0=t); t += n * DT
    feed(own, const(B_POS, 0.9, A_SCALE, 5), t0=t); t += 5 * DT
    check("12 B acquires only after A's release", own.epoch == 2 and own.state == LOCKED)
    # B leaves long enough to release, A returns
    feed(own, absent(n), t0=t); t += n * DT
    res = feed(own, const(A_POS, 0.9, A_SCALE, 5), t0=t)
    check("12 A re-acquires as a NEW epoch (B genuinely released first)",
          own.epoch == 3 and res[-1][0] == LOCKED, "epoch=%d" % own.epoch)
    check("12 exactly two TARGET_SWITCH events for two genuine hand-offs",
          own.switch_count == 2, "switch_count=%d" % own.switch_count)


# ---------------------------------------------------------------- 13 (SS13.11)
def t13_no_person_extended_time_stays_quiet():
    own = new_owner()
    res = feed(own, absent(300))          # 10 s of nobody, from a cold start
    check("13 stays NO_TARGET the whole time, no spurious events",
          all(s == NO_TARGET for s, _ in res) and len(own.events) == 0)


# ---------------------------------------------------------------- 14 (SS13.12)
def t14_new_person_after_long_empty_period_acquires_normally():
    own = new_owner()
    feed(own, absent(300))
    res = feed(own, const(A_POS, 0.9, A_SCALE, 5), t0=300 * DT)
    check("14 acquires normally after a long empty period (nothing stale carried over)",
          res[-1][0] == LOCKED and own.epoch == 1)


# ---------------------------------------------------------------- 15 (scale corroborator, positive case)
def t15_close_scale_variation_does_not_reject():
    own = new_owner()
    feed(own, const(A_POS, 0.9, A_SCALE, 5))
    res = feed(own, const(A_POS, 0.9, B_SCALE_CLOSE, 30), t0=5 * DT)
    check("15 ordinary scale measurement noise (same person) does not trip the scale margin",
          all(s == LOCKED for s, _ in res))


# ---------------------------------------------------------------- 16 (release timing precision)
def t16_release_timeout_boundary():
    cfg = OwnershipConfig(release_timeout_s=1.0)
    own = new_owner(cfg)
    feed(own, const(A_POS, 0.9, A_SCALE, 5))              # locks; owner_since = 4*DT
    lost_at = feed(own, absent(1), t0=5 * DT)              # -> TEMPORARILY_LOST; lost_since = 5*DT
    check("16 setup: entered TEMPORARILY_LOST", lost_at[-1][0] == TEMPORARILY_LOST)
    just_under = feed(own, absent(1), t0=5 * DT + 0.99)    # loss_elapsed = 0.99 < 1.0
    check("16 still TEMPORARILY_LOST just under the release timeout",
          just_under[-1][0] == TEMPORARILY_LOST, "state=%s" % just_under[-1][0])

    own2 = new_owner(cfg)
    feed(own2, const(A_POS, 0.9, A_SCALE, 5))
    feed(own2, absent(1), t0=5 * DT)
    just_over = feed(own2, absent(1), t0=5 * DT + 1.01)    # loss_elapsed = 1.01 > 1.0
    check("16 RELEASED once the timeout is exceeded",
          just_over[-1][0] == RELEASED, "state=%s" % just_over[-1][0])


# ---------------------------------------------------------------- 17 (regression guard)
def t17_owner_returning_past_reacquire_window_is_not_released():
    """REGRESSION GUARD for the defect f21_adversarial.py's scenario s08b found offline.

    Before the fix, matching against the frozen owner reference was gated on
    `loss_elapsed <= reacquire_window_s`, so an owner who stepped away for longer than
    REACQUIRE_WINDOW (2.0 s) and returned to their EXACT original position at full confidence was
    rejected 51 consecutive times as "not_owner", held un-emitted for 1.87 s, RELEASED, and then
    re-acquired with a TARGET_SWITCH logged - against a person who never moved. RELEASE_TIMEOUT
    (4.0 s) is the budget that bounds a loss; REACQUIRE_WINDOW only selects the confirm count."""
    own = new_owner()
    feed(own, const(A_POS, 0.9, A_SCALE, 10))
    check("17 setup: locked", own.state == LOCKED)

    # away for 2.33 s - past REACQUIRE_WINDOW (2.0 s), well inside RELEASE_TIMEOUT (4.0 s)
    feed(own, absent(70), t0=10 * DT)
    check("17 setup: temporarily lost, not yet released", own.state == TEMPORARILY_LOST,
          own.state)

    back = feed(own, const(A_POS, 0.9, A_SCALE, 10), t0=80 * DT)
    check("17 the returning owner is re-locked, not released",
          own.state == LOCKED and not events_of(own, "TARGET_RELEASED"),
          "state=%s released=%d" % (own.state, len(events_of(own, "TARGET_RELEASED"))))
    check("17 it is the SAME ownership epoch (a reacquire, not a new acquisition)",
          own.epoch == 1 and len(events_of(own, "TARGET_REACQUIRED")) == 1,
          "epoch=%d reacquired=%d" % (own.epoch, len(events_of(own, "TARGET_REACQUIRED"))))
    check("17 no TARGET_SWITCH is logged for a person who never moved",
          own.switch_count == 0 and not events_of(own, "TARGET_SWITCH"),
          "switch_count=%d" % own.switch_count)
    check("17 the returning owner is never rejected as a candidate",
          not events_of(own, "TARGET_REJECTED_CANDIDATE"),
          "rejections=%d" % len(events_of(own, "TARGET_REJECTED_CANDIDATE")))
    check("17 emission resumes after the confirm window, not after a release",
          [e for _, e in back] == [False] * 4 + [True] * 6, str([e for _, e in back]))


# ---------------------------------------------------------------- 18 (the widened window is not a hole)
def t18_widened_match_window_still_rejects_a_different_body():
    """The fix in t17 widens WHEN a frozen-reference match is attempted; it must not widen WHAT
    passes. A different body arriving in the same >REACQUIRE_WINDOW interval must still be rejected
    on position, and must still have to wait out the full release before it can acquire."""
    own = new_owner()
    feed(own, const(A_POS, 0.9, A_SCALE, 10))
    feed(own, absent(70), t0=10 * DT)
    out = feed(own, const(B_POS, 0.95, A_SCALE, 10), t0=80 * DT)   # B, 0.60 m away, higher conf
    check("18 a different body past the reacquire window is still rejected",
          all(not e for _, e in out) and own.state == TEMPORARILY_LOST,
          "state=%s emitted=%d" % (own.state, sum(1 for _, e in out if e)))
    rej = events_of(own, "TARGET_REJECTED_CANDIDATE")
    check("18 the rejection now names the real discriminator, not a flat 'not_owner'",
          rej and rej[0]["reason"] == "position_jump",
          rej[0]["reason"] if rej else "no rejection")
    check("18 no reacquire and no switch were produced by the intruder",
          not events_of(own, "TARGET_REACQUIRED") and own.switch_count == 0)


# ----------------------------------------------------------------------------------------------
# S30 path consistency. The gate these pin exists because S27 measured a SILENT wrong-person
# hand-off on real footage: a different body walked into the lost owner's frozen position, passed
# the position and scale gates, and was re-locked under the same target_id with TARGET_SWITCH still
# reading 0. These four hold both directions of the fix - it must refuse a tracked walk-in, and it
# must NOT refuse an owner who simply reappears.
# ----------------------------------------------------------------------------------------------
def _walk(p0, p1, n, conf=0.85, scale=A_SCALE):
    out = []
    for i in range(n):
        f = float(i) / max(1, n)
        out.append((tuple(p0[k] + (p1[k] - p0[k]) * f for k in range(3)), conf, scale))
    return out


def t19_candidate_that_walks_into_the_frozen_position_is_refused():
    own = new_owner()
    feed(own, [(A_POS, 0.85, A_SCALE)] * 6)
    check("19 setup: locked on A", own.state == LOCKED, own.state)
    out = feed(own, [(None, 0.0, None)] * 3 + _walk(B_POS, A_POS, 25)
               + [(A_POS, 0.85, A_SCALE)] * 20, t0=6 * DT)
    check("19 the walked-in body never becomes the owner",
          own.state in (TEMPORARILY_LOST, REACQUIRING, RELEASED, NO_TARGET),
          "state=%s" % own.state)
    check("19 not one frame of it is emitted", not any(e for _s, e in out),
          "emitted=%d" % sum(1 for _s, e in out if e))
    check("19 and the log names the path gate as the reason",
          any(e.get("reason") == "path_walked_in" for e in own.events),
          "reasons=%s" % sorted(set(e.get("reason") for e in own.events
                                    if e["event"] == "TARGET_REJECTED_CANDIDATE")))


def t20_owner_reappearing_without_an_observed_approach_still_reacquires():
    own = new_owner()
    feed(own, [(A_POS, 0.85, A_SCALE)] * 6)
    out = feed(own, [(None, 0.0, None)] * 20 + [(A_POS, 0.85, A_SCALE)] * 12, t0=6 * DT)
    check("20 an owner who simply reappears is re-locked", own.state == LOCKED, own.state)
    check("20 it is the same epoch, not a new acquisition", own.epoch == 1, "epoch=%d" % own.epoch)
    check("20 the path gate did not fire on it",
          not any(e.get("reason") == "path_walked_in" for e in own.events))
    check("20 emission resumes", any(e for _s, e in out))


def t21_path_consistency_can_be_disabled_and_the_old_behaviour_returns():
    """The control. Without it, 'the gate fixed it' cannot be distinguished from 'the input
    changed'."""
    steps = ([(A_POS, 0.85, A_SCALE)] * 6 + [(None, 0.0, None)] * 3
             + _walk(B_POS, A_POS, 25) + [(A_POS, 0.85, A_SCALE)] * 20)
    off = new_owner(OwnershipConfig(path_consistency=False))
    out_off = feed(off, steps)
    on = new_owner(OwnershipConfig(path_consistency=True))
    out_on = feed(on, steps)
    check("21 CONTROL gate OFF: the walked-in body IS re-locked, silently",
          off.state == LOCKED and off.epoch == 1 and off.switch_count == 0,
          "state=%s epoch=%d switches=%d" % (off.state, off.epoch, off.switch_count))
    check("21 CONTROL gate OFF emits it", sum(1 for _s, e in out_off if e) > 6,
          "emitted=%d" % sum(1 for _s, e in out_off if e))
    check("21 gate ON emits only the 2 genuine A frames after acquisition",
          sum(1 for _s, e in out_on if e) == 2,
          "emitted=%d" % sum(1 for _s, e in out_on if e))


def t22_a_long_observation_gap_breaks_the_chain():
    """CHAIN_GAP_S is what stops a stale position asserting continuity. A candidate seen far away,
    then NOT seen for longer than the gap, then appearing inside the margin, is judged fresh - that
    pattern is indistinguishable from the owner stepping back out, so it stays admissible."""
    cfg = OwnershipConfig()
    own = new_owner(cfg)
    feed(own, [(A_POS, 0.85, A_SCALE)] * 6)
    gap_frames = int(cfg.chain_gap_s / DT) + 4
    feed(own, [(B_POS, 0.85, A_SCALE)] * 4, t0=6 * DT)                 # seen far away
    out = feed(own, [(None, 0.0, None)] * gap_frames + [(A_POS, 0.85, A_SCALE)] * 12,
               t0=10 * DT)
    check("22 the gap is longer than CHAIN_GAP_S", gap_frames * DT > cfg.chain_gap_s,
          "%.3fs > %.3fs" % (gap_frames * DT, cfg.chain_gap_s))
    check("22 the chain is broken, so the returning owner is admitted", own.state == LOCKED,
          "state=%s" % own.state)
    check("22 same epoch", own.epoch == 1, "epoch=%d" % own.epoch)
    check("22 emission resumes", any(e for _s, e in out))


# ----------------------------------------------------------------------------------------------
# S34 / ADR-061 owner-reference DRIFT. The live 2026-09-16 session emitted person B under person A's
# identity for ~97 datagrams inside one epoch with nothing logged, because the LOCKED branch updates
# owner_pos to every accepted frame - so the reference walks with the observation. Measured on that
# trace: 0.015 m per frame against a 0.35 m margin (23x under), and B's apparent torso span was
# +39 % against a +/-45 % scale margin. Neither existing gate can see it, by construction.
#
# t23 pins that the DEFAULT is unchanged - drift is measured but nothing acts on it.
# t24-t26 pin the opt-in mechanism against the real trace's shape.
# ----------------------------------------------------------------------------------------------
def _migrate(own, z_from, z_to, n, t0, conf=0.85, scale=A_SCALE):
    """Walk the observation from z_from to z_to over n frames, as a real migration does - in steps
    far under SWITCH_MARGIN_M, so every individual frame passes the position test."""
    out = []
    for i in range(n):
        z = z_from + (z_to - z_from) * (float(i + 1) / n)
        out.append(((A_POS[0], A_POS[1], z), conf, scale))
    return feed(own, out, t0=t0)


def t23_drift_is_measured_but_changes_nothing_by_default():
    own = new_owner()
    feed(own, [(A_POS, 0.85, A_SCALE)] * 6)
    check("23 setup: locked", own.state == LOCKED, own.state)
    out = _migrate(own, A_POS[2], A_POS[2] - 0.45, 30, t0=6 * DT)
    check("23 every migration frame is still EMITTED (behaviour unchanged)",
          all(e for _s, e in out), "emitted=%d/30" % sum(1 for _s, e in out if e))
    check("23 the machine never left LOCKED", own.state == LOCKED, own.state)
    check("23 no drift event was emitted",
          not any(e["event"] == "TARGET_DRIFT_EXCEEDED" for e in own.events))
    check("23 but the drift IS measured and available", own.owner_drift_max_m > 0.4,
          "max drift=%.3f m" % own.owner_drift_max_m)
    check("23 and it is on the snapshot for the HUD",
          own.snapshot(1.0)["owner_drift_max_m"] > 0.4,
          "snapshot=%.3f" % own.snapshot(1.0)["owner_drift_max_m"])


def t24_no_single_frame_step_could_have_caught_it():
    """The measurement that rules out a per-frame fix, asserted rather than asserted-about."""
    own = new_owner()
    feed(own, [(A_POS, 0.85, A_SCALE)] * 6)
    steps = []
    prev = A_POS[2]
    for i in range(30):
        z = A_POS[2] - 0.45 * (float(i + 1) / 30)
        steps.append(abs(z - prev))
        prev = z
    check("24 the largest single-frame step is far under SWITCH_MARGIN_M",
          max(steps) < 0.35 / 10.0,
          "max step=%.4f m vs margin 0.35 m (%.0fx under)" % (max(steps), 0.35 / max(steps)))


def t25_a_drift_budget_converts_the_silent_slide_into_a_declared_one():
    own = new_owner(OwnershipConfig(drift_budget_m=0.30))
    feed(own, [(A_POS, 0.85, A_SCALE)] * 6)
    out = _migrate(own, A_POS[2], A_POS[2] - 0.45, 30, t0=6 * DT)
    check("25 the budget fires", any(e["event"] == "TARGET_DRIFT_EXCEEDED" for e in own.events),
          "events=%s" % sorted(set(e["event"] for e in own.events)))
    check("25 it stops emitting the migrated pose", not all(e for _s, e in out),
          "emitted=%d/30" % sum(1 for _s, e in out if e))
    check("25 and it routes through the loss branch, where ADR-057's gate lives",
          any(e["event"] == "TARGET_TEMP_LOST" and e.get("reason") == "drift_budget"
              for e in own.events))
    # NOT "state != LOCKED": recovering is correct, and the machine SHOULD re-lock on the body it
    # can still see. What the budget buys is that the slide is no longer SILENT - the consumer gets
    # TARGET_DRIFT_EXCEEDED + TARGET_TEMP_LOST where previously there was nothing at all.
    kinds = [e["event"] for e in own.events]
    check("25 the slide is ANNOUNCED, which is the whole point",
          "TARGET_DRIFT_EXCEEDED" in kinds and "TARGET_TEMP_LOST" in kinds,
          "events=%s" % sorted(set(kinds)))
    # Honest limit, pinned so nobody reads more into the budget than it gives: a reacquire
    # RE-ANCHORS, so a long migration is announced once per budget-length rather than refused. It
    # converts silence into a repeating announcement; it does not by itself change identity.
    check("25 a reacquire re-anchors, so the budget is a RATCHET not a refusal",
          own.owner_drift_max_m < 0.45,
          "drift after re-anchor=%.3f m (was 0.450 before the budget fired)"
          % own.owner_drift_max_m)


def t26_a_budget_does_not_fire_on_an_owner_who_stays_put():
    own = new_owner(OwnershipConfig(drift_budget_m=0.30))
    feed(own, [(A_POS, 0.85, A_SCALE)] * 6)
    jitter = [((A_POS[0] + 0.02 * ((i % 3) - 1), A_POS[1],
                A_POS[2] + 0.01 * ((i % 5) - 2)), 0.85, A_SCALE)
              for i in range(40)]
    out = feed(own, jitter, t0=6 * DT)
    check("26 an owner standing still never trips the budget",
          not any(e["event"] == "TARGET_DRIFT_EXCEEDED" for e in own.events),
          "max drift=%.3f m" % own.owner_drift_max_m)
    check("26 and keeps being emitted", all(e for _s, e in out))


def main():
    print("=" * 78)
    print("F-21 TargetOwnership unit tests")
    print("=" * 78)
    for fn in [t01_basic_acquisition, t02_no_oscillation_on_entry,
               t03_similar_confidence_second_person_rejected,
               t04_different_scale_rejected_at_owner_position,
               t05_confidence_dip_within_gate_does_not_lose_owner,
               t06_partial_occlusion_then_return,
               t07_owner_leaves_release_then_new_acquire,
               t08_crossing_does_not_transfer_ownership,
               t09_flickering_candidate_never_locks,
               t10_owner_gone_then_candidate_appears_not_reassigned,
               t11_owner_returns_after_candidate_reacquires_owner_not_candidate,
               t12_rapid_A_B_A_cycle_is_rate_limited_not_oscillating,
               t13_no_person_extended_time_stays_quiet,
               t14_new_person_after_long_empty_period_acquires_normally,
               t15_close_scale_variation_does_not_reject,
               t16_release_timeout_boundary,
               t17_owner_returning_past_reacquire_window_is_not_released,
               t18_widened_match_window_still_rejects_a_different_body,
               t19_candidate_that_walks_into_the_frozen_position_is_refused,
               t20_owner_reappearing_without_an_observed_approach_still_reacquires,
               t21_path_consistency_can_be_disabled_and_the_old_behaviour_returns,
               t22_a_long_observation_gap_breaks_the_chain,
               t23_drift_is_measured_but_changes_nothing_by_default,
               t24_no_single_frame_step_could_have_caught_it,
               t25_a_drift_budget_converts_the_silent_slide_into_a_declared_one,
               t26_a_budget_does_not_fire_on_an_owner_who_stays_put]:
        fn()
    n = len(_results)
    p = sum(1 for _, ok, _ in _results if ok)
    print("-" * 78)
    print("%d/%d assertions passed" % (p, n))
    if p != n:
        print("\nFAILED:")
        for name, ok, detail in _results:
            if not ok:
                print("  %s  %s" % (name, detail))
    return 0 if p == n else 1


if __name__ == "__main__":
    sys.exit(main())
