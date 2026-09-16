#!/usr/bin/env python3
"""F-22 deterministic unit tests for pose_validation.py.

Self-contained runner (no pytest), matching test_joint_tracker.py/test_target_ownership.py's pattern.
Covers the F-22 brief's SS26 categories as far as they apply to a PURE per-chain angular validator:
Cartesian velocity/acceleration and bone-length-RATIO checks are deliberately NOT re-tested here -
those are P1-1's job (joint_tracker.py), already covered by its own test suite, and duplicating them
would be exactly the "duplicate an existing mechanism" the brief's SS1 warns against. F-20A/F-21
interaction is architectural (this module never touches either) and is exercised in
f22_video_replay.py instead, not as a pure-module unit test.

    python test_pose_validation.py
"""

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(_os.path.join(_d, "_sidecar_path.py")):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import _sidecar_path  # noqa: F401  - puts the sidecar root and every tools/ group on sys.path

import sys

from pose_validation import (PoseValidator, ChainValidator, ChainConfig,
                             VALID, HELD, REJECTED,
                             TRACK_LOST, NONFINITE, INVALID_CONFIDENCE, ANGLE_LIMIT,
                             ANGLE_RATE_IMPOSSIBLE)

DT = 1.0 / 30.0
_results = []

# A generic "elbow-shaped" config reused across tests that don't care about elbow vs. knee specifics.
CFG = ChainConfig(warn_deg=150.0, reject_deg=160.0,
                  rate_suspicious_deg_s=900.0, rate_impossible_deg_s=1800.0,
                  min_confidence=0.3, hold_max_frames=8)

SHOULDER = (0.0, 0.0, 0.0)


def check(name, cond, detail=""):
    _results.append((name, bool(cond), detail))
    print("  %-4s %-58s %s" % ("PASS" if cond else "FAIL", name, detail))


def elbow_at(bend_deg, forearm_len=0.28, upper_len=0.30):
    """Shoulder fixed at origin, upper arm along +X. `bend_deg` is the TRUE vector angle between the
    upper-arm and forearm vectors (0 = straight, 180 = folded flat back onto the upper arm) - the
    exact convention pose_validation.py and Arm V2's BendDeg both use."""
    import math
    elbow = (upper_len, 0.0, 0.0)
    theta = math.radians(bend_deg)   # direction of the forearm vector from +X: 0 continues straight
                                      # (same direction as the upper arm), 180 folds fully back on it
    wrist = (elbow[0] + forearm_len * math.cos(theta), elbow[1] + forearm_len * math.sin(theta), 0.0)
    return SHOULDER, elbow, wrist


def feed(cv, bends, conf=0.9, t0=0.0, dt=DT, measured3=(True, True, True)):
    out = []
    t = t0
    for b in bends:
        proximal, joint, wrist = elbow_at(b)
        out.append(cv.update(proximal, joint, wrist, conf, measured3, t))
        t += dt
    return out


# ---------------------------------------------------------------- 1 joint validity / recovery
def t01_straight_arm_always_valid():
    cv = ChainValidator("test", CFG)
    res = feed(cv, [0.0] * 30)
    check("01 dead-straight arm stays VALID every frame",
          all(r[0] == VALID for r in res), "states=%s" % sorted(set(r[0] for r in res)))


def t02_normal_range_of_motion_valid():
    cv = ChainValidator("test", CFG)
    # a slow, legitimate wave: 20 -> 110 -> 20 deg over 2 s, well under WARN
    bends = [20 + 90 * abs(((i / 30.0) % 2.0) - 1.0) for i in range(60)]
    res = feed(cv, bends)
    check("02 normal wave-like motion (<=110 deg) never rejected",
          all(r[0] == VALID for r in res), "max_state=%s" % sorted(set(r[0] for r in res)))


# ---------------------------------------------------------------- 2 nonfinite
def t03_nonfinite_rejected():
    cv = ChainValidator("test", CFG)
    state, reason, bend, rate = cv.update((0, 0, 0), (float("nan"), 0, 0), (0.3, 0, 0), 0.9,
                                          (True, True, True), 0.0)
    check("03 NaN coordinate -> HELD/REJECTED with NONFINITE",
          state != VALID and reason == NONFINITE, "state=%s reason=%s" % (state, reason))


def t04_degenerate_segment_rejected():
    cv = ChainValidator("test", CFG)
    # elbow coincides with shoulder -> zero-length upper-arm segment
    state, reason, bend, rate = cv.update((0, 0, 0), (0, 0, 0), (0.3, 0, 0), 0.9,
                                          (True, True, True), 0.0)
    check("04 zero-length segment -> rejected as NONFINITE (segment continuity)",
          state != VALID and reason == NONFINITE, "state=%s reason=%s" % (state, reason))


# ---------------------------------------------------------------- 3 confidence
def t05_low_confidence_held():
    cv = ChainValidator("test", CFG)
    proximal, joint, wrist = elbow_at(20.0)
    state, reason, bend, rate = cv.update(proximal, joint, wrist, 0.1, (True, True, True), 0.0)
    check("05 confidence below gate -> HELD/REJECTED with INVALID_CONFIDENCE",
          state != VALID and reason == INVALID_CONFIDENCE, "state=%s reason=%s" % (state, reason))


# ---------------------------------------------------------------- 4 depth / track invalidity
def t06_unmeasured_joint_track_lost():
    cv = ChainValidator("test", CFG)
    proximal, joint, wrist = elbow_at(20.0)
    state, reason, bend, rate = cv.update(proximal, joint, wrist, 0.9, (True, False, True), 0.0)
    check("06 an unmeasured chain point -> TRACK_LOST",
          state != VALID and reason == TRACK_LOST, "state=%s reason=%s" % (state, reason))


# ---------------------------------------------------------------- 5/10/11 absolute angle limits
def t07_elbow_at_f19_median_is_rejected():
    """F-19's own live finding: hands-near-face MEDIAN was 167.8/169.4 deg - not an occasional
    spike, the whole block. This is the exact defect F-22 exists to catch."""
    cv = ChainValidator("test", CFG)
    res = feed(cv, [168.0] * 20)
    check("07 F-19's own measured hands-near-face median (168 deg) is rejected",
          all(r[0] != VALID for r in res) and any(r[1] == ANGLE_LIMIT for r in res),
          "states=%s" % sorted(set(r[0] for r in res)))


def t08_elbow_at_f19_max_is_rejected():
    cv = ChainValidator("test", CFG)
    res = feed(cv, [178.0] * 20)
    check("08 F-19's own measured hands-near-face max (178 deg) is rejected",
          all(r[0] != VALID for r in res))


def t09_knee_legitimate_walking_max_not_rejected():
    """F-19 measured a LEGITIMATE knee max of 176 deg during ordinary walking (rare, 30/85k
    frames) - the knee REJECT threshold (178) must sit above this or real walking false-rejects."""
    from pose_validation import KNEE_CONFIG
    cv = ChainValidator("knee", KNEE_CONFIG)
    res = feed(cv, [176.0] * 5)
    check("09 knee's own observed legitimate walking max (176 deg) is NOT rejected",
          all(r[0] == VALID for r in res), "states=%s" % [r[0] for r in res])


def t10_boundary_just_under_and_over_reject():
    cv1 = ChainValidator("t", CFG)
    s1, r1, _, _ = feed(cv1, [159.9])[0]
    cv2 = ChainValidator("t", CFG)
    s2, r2, _, _ = feed(cv2, [160.1])[0]
    check("10 159.9 deg (just under REJECT) stays VALID", s1 == VALID, "state=%s" % s1)
    check("10 160.1 deg (just over REJECT) is rejected", s2 != VALID and r2 == ANGLE_LIMIT)


# ---------------------------------------------------------------- 6 chain consistency
def t11_chain_consistency_is_the_bend_angle_itself():
    """The bend angle IS the shoulder-elbow-wrist chain-consistency check - an internally
    inconsistent chain (a single bad depth point dragging the wrist far out of any plausible arm
    configuration) manifests as exactly this signal. No separate chain check is implemented on top
    of it (F-22 brief SS2: do not invent an unproven constraint)."""
    cv = ChainValidator("test", CFG)
    # a single bad wrist depth sample throws the forearm vector into an impossible fold
    proximal, joint, _ = elbow_at(20.0)
    bad_wrist = (joint[0] - 0.5, joint[1], joint[2])   # wrist snaps back past the shoulder line
    state, reason, bend, rate = cv.update(proximal, joint, bad_wrist, 0.9, (True, True, True), 0.0)
    check("11 a single bad depth point creating an impossible fold is caught",
          state != VALID and reason == ANGLE_LIMIT, "state=%s reason=%s bend=%.1f" % (state, reason, bend))


# ---------------------------------------------------------------- 8/9 temporal / angular velocity
def t12_gradual_but_still_impossible_is_rejected():
    """F-22 brief SS15: temporal smoothness alone cannot establish validity. A slow, smooth ramp
    into an impossible angle must still be rejected once it crosses the absolute threshold."""
    cv = ChainValidator("test", CFG)
    bends = [140 + i * 1.0 for i in range(30)]   # 140 -> 169 deg over 1 s, 1 deg/frame - very gentle
    res = feed(cv, bends)
    # Find the crossing from the ACTUAL reported bend (not the nominal input), since an acos/cos
    # round-trip can land a hair off an exact integer degree near the boundary.
    idx = next(i for i, r in enumerate(res) if r[1] == ANGLE_LIMIT)
    check("12 a SLOW smooth ramp still gets rejected once past the absolute limit",
          res[idx][0] != VALID, "crossed at reported bend=%.2f state=%s" % (res[idx][2], res[idx][0]))
    check("12 frames clearly before the crossing (bend <= 158 deg) stayed VALID",
          all(r[0] == VALID for r in res[:idx] if r[2] is not None and r[2] <= 158.0),
          "pre-crossing states=%s" % [r[0] for r in res[:idx]])


def t13_instant_impossible_jump_rejected_by_rate():
    """Within the absolute-plausible range (<=160) but arriving in one frame from a very different
    angle - an implausible RATE even though the endpoint alone would pass the absolute check."""
    cv = ChainValidator("test", CFG)
    feed(cv, [20.0] * 5)                      # settle at a plausible, steady 20 deg
    proximal, joint, wrist = elbow_at(155.0)   # next frame: 135 deg jump in 1/30 s = ~4000 deg/s
    state, reason, bend, rate = cv.update(proximal, joint, wrist, 0.9, (True, True, True), 5 * DT)
    check("13 an instant large jump (within absolute range) is caught by the RATE check",
          state != VALID and reason == ANGLE_RATE_IMPOSSIBLE,
          "state=%s reason=%s rate=%.0fdeg/s" % (state, reason, rate or -1))


def t14_fast_but_legitimate_motion_not_rejected():
    """MUST-TEST false positive (SS27): a fast but humanly-plausible arm motion (e.g. a quick
    straighten) must NOT trip the rate check."""
    cv = ChainValidator("test", CFG)
    # 90 -> 10 deg over 4 frames (~133 ms) = 20 deg/frame = ~600 deg/s - fast, still human
    bends = [90, 70, 50, 30, 10]
    res = feed(cv, bends)
    check("14 a fast legitimate arm-straighten (~600 deg/s) is accepted, not rejected",
          all(r[0] == VALID for r in res), "states=%s" % [r[0] for r in res])


# ---------------------------------------------------------------- 13/17 localized / multi-joint failure
def t15_local_failure_does_not_affect_other_chains():
    import numpy as np
    pv = PoseValidator()
    xyz = np.zeros((17, 3), dtype=np.float32)
    # right elbow (6-8-10): impossible fold. Every other chain: straight/plausible.
    for (sh, el, wr) in [(5, 7, 9), (6, 8, 10)]:
        xyz[sh] = [0, 0, 0]
        xyz[el] = [0.30, 0, 0]
        xyz[wr] = [0.58, 0, 0]   # straight
    xyz[8] = [0.30, 0, 0]   # right shoulder set below; right elbow position
    xyz[6] = [0.0, 0, 0]
    xyz[10] = [0.02, 0, 0]  # right wrist snapped back near the shoulder -> impossible fold at elbow
    for (hp, kn, an) in [(11, 13, 15), (12, 14, 16)]:
        xyz[hp] = [0, -0.9, 0]
        xyz[kn] = [0, -0.5, 0]
        xyz[an] = [0, -0.05, 0]
    measured = np.ones(17, dtype=bool)
    conf = np.full(17, 0.9, dtype=np.float32)
    out = pv.update(xyz, measured, conf, 0.0)
    check("15 the bad chain (right elbow, idx 8) is not VALID",
          out[8][0] != VALID, "state=%s" % out[8][0])
    check("15 every OTHER chain (left elbow, both knees) stays VALID",
          out[7][0] == VALID and out[13][0] == VALID and out[14][0] == VALID,
          "states=%s" % dict((k, v[0]) for k, v in out.items()))


def t16_all_chains_failing_simultaneously_each_localized():
    """'Global' failure is not a special code path - it falls naturally out of every chain failing
    independently, each with its OWN correct reason (F-22 brief SS13: only invalidate globally when
    the evidence shows a global failure - here that's simply every local check failing on its own
    merits, not a bespoke all-or-nothing mechanism)."""
    import numpy as np
    pv = PoseValidator()
    xyz = np.zeros((17, 3), dtype=np.float32)
    xyz[5] = [0, 0, 0]; xyz[7] = [0.30, 0, 0]; xyz[9] = [0.02, 0, 0]
    xyz[6] = [0, 0, 0]; xyz[8] = [0.30, 0, 0]; xyz[10] = [0.02, 0, 0]
    xyz[11] = [0, -0.9, 0]; xyz[13] = [0, -0.5, 0]; xyz[15] = [0, -0.9 + 0.02, 0]
    xyz[12] = [0, -0.9, 0]; xyz[14] = [0, -0.5, 0]; xyz[16] = [0, -0.9 + 0.02, 0]
    measured = np.ones(17, dtype=bool)
    conf = np.full(17, 0.9, dtype=np.float32)
    out = pv.update(xyz, measured, conf, 0.0)
    check("16 all four chains independently rejected, each with ANGLE_LIMIT",
          all(out[j][0] != VALID and out[j][1] == ANGLE_LIMIT for j in (7, 8, 13, 14)),
          "states=%s" % dict((k, v[:2]) for k, v in out.items()))


# ---------------------------------------------------------------- 15/16 hold + recovery
def t17_hold_then_reject_after_window():
    cv = ChainValidator("test", CFG)
    res = feed(cv, [170.0] * (CFG.hold_max_frames + 3))
    held = res[:CFG.hold_max_frames]
    rejected = res[CFG.hold_max_frames:]
    check("17 first hold_max_frames failures are HELD, not REJECTED",
          all(r[0] == HELD for r in held), "states=%s" % [r[0] for r in held])
    check("17 failures past the hold window escalate to REJECTED",
          all(r[0] == REJECTED for r in rejected), "states=%s" % [r[0] for r in rejected])


def t18_recovery_is_immediate_on_one_good_frame():
    cv = ChainValidator("test", CFG)
    feed(cv, [170.0] * 3, t0=0.0)          # HELD x3
    res = feed(cv, [20.0], t0=3 * DT)      # one good frame
    check("18 recovers to VALID on the very next good frame (no confirm-frames needed)",
          res[0][0] == VALID, "state=%s" % res[0][0])


def main():
    print("=" * 78)
    print("F-22 PoseValidator unit tests")
    print("=" * 78)
    for fn in [t01_straight_arm_always_valid, t02_normal_range_of_motion_valid,
               t03_nonfinite_rejected, t04_degenerate_segment_rejected,
               t05_low_confidence_held, t06_unmeasured_joint_track_lost,
               t07_elbow_at_f19_median_is_rejected, t08_elbow_at_f19_max_is_rejected,
               t09_knee_legitimate_walking_max_not_rejected, t10_boundary_just_under_and_over_reject,
               t11_chain_consistency_is_the_bend_angle_itself,
               t12_gradual_but_still_impossible_is_rejected,
               t13_instant_impossible_jump_rejected_by_rate,
               t14_fast_but_legitimate_motion_not_rejected,
               t15_local_failure_does_not_affect_other_chains,
               t16_all_chains_failing_simultaneously_each_localized,
               t17_hold_then_reject_after_window, t18_recovery_is_immediate_on_one_good_frame]:
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
