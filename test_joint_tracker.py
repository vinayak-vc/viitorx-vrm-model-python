#!/usr/bin/env python3
"""P1-1 deterministic unit tests for joint_tracker.

Self-contained runner (no pytest dependency). Covers the 15 cases the P1-1 brief lists.
    python test_joint_tracker.py
"""
import math
import sys

from joint_tracker import (JointTracker, SkeletonTracker, TrackerConfig,
                           TrackingState)

DT = 1.0 / 30.0
_results = []


def check(name, cond, detail=""):
    _results.append((name, bool(cond), detail))
    print("  %-4s %-52s %s" % ("PASS" if cond else "FAIL", name, detail))


def feed(tr, pts, conf=0.9, t0=0.0, dt=DT, depth_valid=True):
    """Feed a list of (x,y,z) (or None for 'no measurement'). Returns list of states."""
    states = []
    t = t0
    for p in pts:
        if p is None:
            st, _ = tr.update(0.0, 0.0, 0.0, 0.0, t, depth_valid)
        else:
            st, _ = tr.update(p[0], p[1], p[2], conf, t, depth_valid)
        states.append(st)
        t += dt
    return states


def d(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ---------------------------------------------------------------- 1
def t01_stationary():
    tr = JointTracker("wrist")
    pts = [(0.4, 0.1, 2.0)] * 60
    st = feed(tr, pts)
    check("01 stationary joint stays TRACKED",
          all(s == TrackingState.TRACKED for s in st[1:]),
          "states unique=%s" % sorted({TrackingState.name(s) for s in st}))
    check("01 stationary output does not drift",
          d(tr.position(), (0.4, 0.1, 2.0)) < 1e-6,
          "drift=%.2emm" % (d(tr.position(), (0.4, 0.1, 2.0)) * 1000))


# ---------------------------------------------------------------- 2
def t02_constant_velocity():
    tr = JointTracker("wrist")
    pts = [(0.4 + 0.01 * i, 0.1, 2.0) for i in range(60)]   # 0.3 m/s
    st = feed(tr, pts)
    check("02 constant velocity stays TRACKED",
          all(s == TrackingState.TRACKED for s in st[2:]))
    check("02 velocity estimate ~0.3 m/s",
          abs(tr.vx - 0.3) < 0.05, "vx=%.3f" % tr.vx)
    check("02 tracks position without lag",
          d(tr.position(), pts[-1]) < 1e-6, "err=%.4f" % d(tr.position(), pts[-1]))


# ---------------------------------------------------------------- 3
def t03_fast_motion():
    tr = JointTracker("wrist")
    # 0.06 m/frame = 1.8 m/s -- a hard dance swing, must NOT be rejected
    pts = [(0.4 + 0.06 * i, 0.1, 2.0) for i in range(40)]
    st = feed(tr, pts)
    bad = [i for i, s in enumerate(st[3:], 3) if s == TrackingState.PREDICTED]
    check("03 fast legitimate motion is not rejected", len(bad) <= 1,
          "predicted frames=%d" % len(bad))
    check("03 fast motion final position accurate",
          d(tr.position(), pts[-1]) < 0.02, "err=%.4f m" % d(tr.position(), pts[-1]))


# ---------------------------------------------------------------- 4
def t04_isolated_spike():
    tr = JointTracker("wrist")
    pts = [(0.4, 0.1, 2.0)] * 20 + [(1.2, 0.1, 2.0)] + [(0.4, 0.1, 2.0)] * 10
    feed(tr, pts)
    err = d(tr.position(), (0.4, 0.1, 2.0))
    check("04 isolated 0.8m spike suppressed", err < 0.02, "final err=%.4f m" % err)
    check("04 spike counted as suspicious", tr.suspicious_count >= 1,
          "suspicious=%d" % tr.suspicious_count)


# ---------------------------------------------------------------- 5
def t05_high_confidence_spike():
    """THE headline case: confidence 0.95 but kinematically impossible."""
    tr = JointTracker("wrist")
    pts = [(0.4, 0.1, 2.0)] * 20 + [(1.2, 0.1, 2.0)] + [(0.4, 0.1, 2.0)] * 10
    feed(tr, pts, conf=0.95)
    err = d(tr.position(), (0.4, 0.1, 2.0))
    check("05 HIGH-CONFIDENCE spike still rejected", err < 0.02,
          "conf=0.95, final err=%.4f m" % err)
    check("05 flagged despite high confidence", tr.suspicious_count >= 1,
          "suspicious=%d" % tr.suspicious_count)


# ---------------------------------------------------------------- 6,7,8
def _dropout(n):
    tr = JointTracker("wrist")
    pts = [(0.4 + 0.01 * i, 0.1, 2.0) for i in range(20)]
    feed(tr, pts)
    st = feed(tr, [None] * n, t0=20 * DT)
    return tr, st


def t06_dropout3():
    tr, st = _dropout(3)
    check("06 3-frame dropout -> PREDICTED",
          all(s == TrackingState.PREDICTED for s in st),
          "states=%s" % [TrackingState.name(s) for s in st])
    check("06 3-frame dropout stays usable", tr.is_usable())


def t07_dropout5():
    tr, st = _dropout(5)
    check("07 5-frame dropout -> PREDICTED throughout",
          all(s == TrackingState.PREDICTED for s in st),
          "states=%s" % [TrackingState.name(s) for s in st])


def t08_dropout10():
    tr, st = _dropout(10)
    check("08 10-frame dropout eventually LOST",
          st[-1] == TrackingState.LOST,
          "final=%s after %d" % (TrackingState.name(st[-1]), len(st)))
    check("08 LOST is not usable (no zero emitted)", not tr.is_usable())


# ---------------------------------------------------------------- 9
def t09_prediction_extrapolates():
    tr = JointTracker("wrist")
    pts = [(0.4 + 0.01 * i, 0.1, 2.0) for i in range(20)]
    feed(tr, pts)
    before = tr.position()[0]
    feed(tr, [None] * 3, t0=20 * DT)
    after = tr.position()[0]
    check("09 prediction moves along velocity", after > before,
          "x %.4f -> %.4f" % (before, after))
    check("09 prediction is bounded", (after - before) < 0.1,
          "advance=%.4f m" % (after - before))


# ---------------------------------------------------------------- 10
def t10_prediction_expiry():
    cfg = TrackerConfig(max_predict_frames=4)
    tr = JointTracker("wrist", cfg)
    feed(tr, [(0.4 + 0.01 * i, 0.1, 2.0) for i in range(20)])
    st = feed(tr, [None] * 8, t0=20 * DT)
    n_pred = sum(1 for s in st if s == TrackingState.PREDICTED)
    check("10 prediction window respected", n_pred == 4, "predicted=%d (cfg=4)" % n_pred)
    check("10 expiry -> LOST", st[-1] == TrackingState.LOST)


# ---------------------------------------------------------------- 11
def t11_reacquisition_blended():
    tr = JointTracker("wrist")
    feed(tr, [(0.4, 0.1, 2.0)] * 20)
    feed(tr, [None] * 3, t0=20 * DT)
    target = (0.55, 0.1, 2.0)
    pos_before = tr.position()
    st, _ = tr.update(target[0], target[1], target[2], 0.9, 23 * DT)
    jump = d(tr.position(), pos_before)
    full = d(target, pos_before)
    check("11 reacquisition does NOT teleport", jump < full * 0.9,
          "moved %.4f of %.4f m" % (jump, full))
    check("11 reacquisition counted", tr.reacquire_count == 1)
    for i in range(8):
        tr.update(target[0], target[1], target[2], 0.9, (24 + i) * DT)
    check("11 reacquisition converges", d(tr.position(), target) < 0.01,
          "err=%.4f m" % d(tr.position(), target))


# ---------------------------------------------------------------- 12
def t12_reacquisition_large_error():
    tr = JointTracker("wrist")
    feed(tr, [(0.4, 0.1, 2.0)] * 20)
    feed(tr, [None] * 8, t0=20 * DT)          # -> LOST
    check("12 long gap reached LOST", tr.state == TrackingState.LOST)
    far = (-0.5, 0.4, 2.4)
    prev = tr.position()
    tr.update(far[0], far[1], far[2], 0.9, 30 * DT)
    step = d(tr.position(), prev)
    check("12 large-error reacquire is blended, not a jump",
          step < d(far, prev) * 0.9, "step %.3f of %.3f m" % (step, d(far, prev)))
    for i in range(12):
        tr.update(far[0], far[1], far[2], 0.9, (31 + i) * DT)
    check("12 converges to the new location", d(tr.position(), far) < 0.02,
          "err=%.4f m" % d(tr.position(), far))


# ---------------------------------------------------------------- 13
def t13_irregular_timestamps():
    tr = JointTracker("wrist")
    t = 0.0
    ok = True
    for i in range(40):
        dt = [DT, 0.0, -0.01, 5.0, DT * 2][i % 5]      # zero, negative, huge, doubled
        t += dt
        st, _ = tr.update(0.4 + 0.005 * i, 0.1, 2.0, 0.9, t)
        if st == TrackingState.LOST:
            ok = False
    fin = tr.position()
    check("13 irregular/zero/negative timestamps do not break tracking", ok)
    check("13 no NaN/inf under bad dt",
          all(math.isfinite(v) for v in fin) and math.isfinite(tr.vx),
          "pos=%s vx=%.3f" % ([round(v, 3) for v in fin], tr.vx))


# ---------------------------------------------------------------- 14
def t14_zero_invalid_measurement():
    tr = JointTracker("wrist")
    feed(tr, [(0.4, 0.1, 2.0)] * 20)
    st, usable = tr.update(0.0, 0.0, 0.0, 0.0, 20 * DT)     # the sidecar's zero-fill
    check("14 zero-filled measurement is NOT accepted",
          d(tr.position(), (0.0, 0.0, 0.0)) > 1.0,
          "pos=%s" % [round(v, 3) for v in tr.position()])
    check("14 zero-fill routed to PREDICTED", st == TrackingState.PREDICTED)
    tr2 = JointTracker("cold")
    st2, usable2 = tr2.update(0.0, 0.0, 0.0, 0.0, 0.0)
    check("14 zero at cold start -> LOST and NOT usable",
          st2 == TrackingState.LOST and not usable2)


# ---------------------------------------------------------------- 15
def t15_depth_inconsistency():
    """Isolate the DEPTH signal. A large Z move would trip the residual gate first, so
    these use a SMALL move that passes residual -- then only depth can flag it."""
    # control: small move, depth measured -> clean
    ctl = JointTracker("ctl")
    feed(ctl, [(0.4, 0.1, 2.0)] * 20)
    ctl.update(0.4, 0.1, 2.02, 0.95, 20 * DT, depth_valid=True)
    check("15 control: small move w/ valid depth is clean",
          ctl.last_suspicion == 0.0 and ctl.state == TrackingState.TRACKED,
          "suspicion=%.2f reason=%s" % (ctl.last_suspicion, ctl.last_reason))

    # same move, depth NOT measured -> must be flagged by the depth signal alone
    tr = JointTracker("wrist")
    feed(tr, [(0.4, 0.1, 2.0)] * 20)
    tr.update(0.4, 0.1, 2.02, 0.95, 20 * DT, depth_valid=False)
    check("15 depth-unmeasured flagged (residual alone would pass)",
          tr.last_suspicion > 0.0 and "depth_unmeasured" in tr.last_reason,
          "suspicion=%.2f reason=%s" % (tr.last_suspicion, tr.last_reason))
    check("15 depth-unmeasured degrades to WEAK, not rejected",
          tr.state == TrackingState.WEAK, "state=%s" % TrackingState.name(tr.state))

    # Z contradicts the prediction by more than depth_z_tol, but residual is widened so
    # only the depth term can fire.
    # widen residual, speed AND accel so the depth term is the only one that can fire
    tr2 = JointTracker("wrist2", TrackerConfig(resid_floor=1.0, resid_hard=10.0,
                                               max_speed=100.0, max_accel=1e6))
    feed(tr2, [(0.4, 0.1, 2.0)] * 20)
    tr2.update(0.4, 0.1, 2.30, 0.95, 20 * DT, depth_valid=True)
    check("15 depth-z contradiction flagged by the depth term",
          "depth_z" in tr2.last_reason, "reason=%s" % tr2.last_reason)


# ---------------------------------------------------------------- bonus
def t16_neighbour_constraint():
    sk = SkeletonTracker(indices=[5, 7, 9])            # shoulder, elbow, wrist (left)
    sh, el, wr = (0.2, -0.5, 2.0), (0.35, -0.3, 2.0), (0.45, -0.1, 2.0)
    for i in range(40):
        sk.update({5: sh, 7: el, 9: wr}, {5: 0.9, 7: 0.9, 9: 0.9}, i * DT)
    # elbow teleports somewhere geometrically absurd, at high confidence
    bad_el = (0.35, -0.3, 3.2)
    sk.update({5: sh, 7: bad_el, 9: wr}, {5: 0.9, 7: 0.95, 9: 0.9}, 40 * DT)
    tr = sk.trackers[7]
    err = d((tr.x, tr.y, tr.z), el)
    check("16 neighbour-inconsistent elbow suppressed", err < 0.15,
          "err=%.3f m reason=%s" % (err, tr.last_reason))


def t17_no_allocation_smoke():
    """Update path must be cheap; also a crude timing sanity check."""
    import time
    sk = SkeletonTracker()
    pos = dict((i, (0.1 * i, 0.2, 2.0)) for i in sk.indices)
    cnf = dict((i, 0.9) for i in sk.indices)
    for i in range(50):
        sk.update(pos, cnf, i * DT, collect_events=False)
    n = 2000
    t0 = time.time()
    for i in range(n):
        sk.update(pos, cnf, (50 + i) * DT, collect_events=False)
    per = (time.time() - t0) / n * 1000.0
    check("17 full-skeleton update < 1 ms", per < 1.0,
          "%.4f ms/frame for %d joints" % (per, len(sk.indices)))


def main():
    print("=" * 74)
    print("P1-1 JointTracker unit tests")
    print("=" * 74)
    for fn in [t01_stationary, t02_constant_velocity, t03_fast_motion,
               t04_isolated_spike, t05_high_confidence_spike,
               t06_dropout3, t07_dropout5, t08_dropout10,
               t09_prediction_extrapolates, t10_prediction_expiry,
               t11_reacquisition_blended, t12_reacquisition_large_error,
               t13_irregular_timestamps, t14_zero_invalid_measurement,
               t15_depth_inconsistency, t16_neighbour_constraint,
               t17_no_allocation_smoke]:
        fn()
    n = len(_results)
    p = sum(1 for _, ok, _ in _results if ok)
    print("-" * 74)
    print("%d/%d assertions passed" % (p, n))
    if p != n:
        print("\nFAILED:")
        for name, ok, detail in _results:
            if not ok:
                print("  %s  %s" % (name, detail))
    return 0 if p == n else 1


if __name__ == "__main__":
    sys.exit(main())
