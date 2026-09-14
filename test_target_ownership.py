#!/usr/bin/env python3
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
               t16_release_timeout_boundary]:
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
