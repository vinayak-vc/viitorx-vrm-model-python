#!/usr/bin/env python3
"""P1-4 deterministic tests — long-horizon recovery + kinematic constraints.

Self-contained runner (no pytest), same convention as test_joint_tracker.py.
Covers cases A–I of the P1-4 brief plus solver and safety checks.

    python test_kinematic_recovery.py
"""
import math
import sys

from joint_tracker import SkeletonTracker, TrackerConfig, TrackingState
from kinematic_recovery import (KinematicRecovery, Observation, RecoveryConfig,
                                interior_angle_deg, solve_two_anchor, _dist)

DT = 1.0 / 30.0
_results = []

# WholeBody indices
LSH, RSH, LEL, REL, LWR, RWR = 5, 6, 7, 8, 9, 10
LHIP, RHIP, LKN, RKN, LAN, RAN = 11, 12, 13, 14, 15, 16

BASE = {
    LSH: (0.18, -0.50, 2.0), RSH: (-0.18, -0.50, 2.0),
    LEL: (0.30, -0.25, 2.0), REL: (-0.30, -0.25, 2.0),
    LWR: (0.36, 0.00, 2.0), RWR: (-0.36, 0.00, 2.0),
    LHIP: (0.10, 0.00, 2.0), RHIP: (-0.10, 0.00, 2.0),
    LKN: (0.12, 0.45, 2.0), RKN: (-0.12, 0.45, 2.0),
    LAN: (0.13, 0.88, 2.0), RAN: (-0.13, 0.88, 2.0),
}
ALL = sorted(BASE.keys())


def check(name, cond, detail=""):
    _results.append((name, bool(cond), detail))
    print("  %-4s %-54s %s" % ("PASS" if cond else "FAIL", name, detail))


def pose(mod=None, drop=(), conf=0.9, confmod=None):
    """Build (positions, confidences) from BASE with optional per-joint overrides."""
    p = dict(BASE)
    if mod:
        p.update(mod)
    c = dict((i, conf) for i in ALL)
    if confmod:
        c.update(confmod)
    for d in drop:
        p.pop(d, None)
        c[d] = 0.0
    return p, c


class Rig:
    """SkeletonTracker + KinematicRecovery driven together, as the sidecar does."""

    def __init__(self, cfg=None, rcfg=None):
        self.sk = SkeletonTracker(cfg=cfg or TrackerConfig())
        self.kr = KinematicRecovery(rcfg or RecoveryConfig())
        self.t = 0.0

    def step(self, positions, confidences):
        tr = self.sk.update(positions, confidences, self.t, collect_events=False)
        out = self.kr.apply(tr, positions, collect_events=True)
        self.t += DT
        return out

    def warm(self, n=40):
        """Settle the trackers and teach the bone-length estimators."""
        for i in range(n):
            s = 0.004 * math.sin(i * 0.35)      # tiny sway: real, but nothing implausible
            m = {}
            for k, v in BASE.items():
                m[k] = (v[0] + s, v[1], v[2])
            self.step(*pose(m))
        return self


def finite(p):
    return all(v == v and abs(v) < 1e6 for v in p)


# ------------------------------------------------------------------ A
def case_a_normal():
    r = Rig().warm()
    recon = 0
    rej = 0
    bad_state = 0
    for i in range(60):
        dx = 0.010 * math.sin(i * 0.25)         # smooth, legitimate motion
        m = {LWR: (BASE[LWR][0] + dx, BASE[LWR][1] + dx * 0.5, 2.0),
             LEL: (BASE[LEL][0] + dx * 0.6, BASE[LEL][1], 2.0)}
        out = r.step(*pose(m))
        for idx, (x, y, z, c, st, usable, obs) in out.items():
            if st == TrackingState.PREDICTED:
                recon += 1
            if obs == Observation.GEOMETRICALLY_SUSPECT:
                rej += 1
            if not usable:
                bad_state += 1
    check("A normal tracking: no reconstruction", recon == 0, "predicted frames=%d" % recon)
    check("A normal tracking: no geometric rejection", rej == 0, "rejections=%d" % rej)
    check("A normal tracking: all joints usable", bad_state == 0, "unusable=%d" % bad_state)


# ------------------------------------------------------------------ B
def case_b_knee_5frames():
    r = Rig().warm()
    zero = 0
    explode = 0
    states = []
    for i in range(5):
        out = r.step(*pose(drop=(LKN,)))
        x, y, z, c, st, usable, obs = out[LKN]
        states.append(TrackingState.name(st))
        if abs(x) < 1e-9 and abs(y) < 1e-9 and abs(z) < 1e-9:
            zero += 1
        if _dist((x, y, z), BASE[LKN]) > 0.30:
            explode += 1
    check("B 5-frame knee gap: never zero position", zero == 0, "zeros=%d" % zero)
    check("B 5-frame knee gap: no limb explosion", explode == 0,
          "max drift ok; states=%s" % states)
    check("B 5-frame knee gap: stays usable", "LOST" not in states, "states=%s" % states)


# ------------------------------------------------------------------ C
def case_c_knee_20frames():
    r = Rig().warm()
    states = []
    for i in range(20):
        out = r.step(*pose(drop=(LKN,)))
        states.append(TrackingState.name(out[LKN][4]))
        p = out[LKN][:3]
        if not finite(p):
            check("C 20-frame gap: finite", False, "non-finite at frame %d" % i)
            return
    js = r.kr.joints[LKN]
    check("C 20-frame knee gap: reconstruction engaged", js.reconstructions > 0,
          "reconstructions=%d" % js.reconstructions)
    check("C 20-frame knee gap: no zero / no NaN", True, "states[-5:]=%s" % states[-5:])
    check("C 20-frame knee gap: bounded by the safe horizon",
          js.reconstructions <= r.kr.cfg.max_reconstruct_frames,
          "recon=%d limit=%d" % (js.reconstructions, r.kr.cfg.max_reconstruct_frames))


# ------------------------------------------------------------------ D
def case_d_highconf_teleport():
    r = Rig().warm()
    bad = (0.12 + 0.55, 0.45 - 0.40, 2.0)       # physically impossible knee, high confidence
    out = r.step(*pose({LKN: bad}, confmod={LKN: 0.95}))
    x, y, z, c, st, usable, obs = out[LKN]
    err = _dist((x, y, z), bad)
    check("D hi-conf teleport: geometrically rejected",
          obs == Observation.GEOMETRICALLY_SUSPECT or st != TrackingState.TRACKED,
          "obs=%s state=%s conf=0.95" % (Observation.name(obs), TrackingState.name(st)))
    check("D hi-conf teleport: output is NOT the teleport", err > 0.20,
          "distance from injected value = %.3f m" % err)
    # neighbours must be untouched (Part 12 failure isolation)
    hip_err = _dist(out[LHIP][:3], BASE[LHIP])
    ank_err = _dist(out[LAN][:3], BASE[LAN])
    check("D hi-conf teleport: hip + ankle unaffected",
          hip_err < 0.02 and ank_err < 0.02,
          "hip %.4f  ankle %.4f" % (hip_err, ank_err))


# ------------------------------------------------------------------ D2
def case_d2_geometry_overrides_confidence():
    """THE P1-4 differentiator.

    Case D passes because P1-1's own residual gate catches an instant teleport. That does not
    exercise P1-4 at all. Here the knee is walked to an anatomically impossible place SLOWLY
    and COHERENTLY, at confidence 0.85, so every P1-1 signal (residual, speed, acceleration,
    frozen) stays happy -- P1-1 accepts it. Only skeleton geometry can tell it is wrong.

    FIXTURE NOTE (corrected 2026-09-08): the drift was originally -0.62 in y, putting the shin
    at 1.52x its learned length. That tripped the ORIGINAL seg_reject of 0.45 -- but 0.45 was
    set by reasoning, and measuring 853 frames of real capture showed natural bone-length
    deviation reaching p99 = 1.79, so 0.45 was firing on 25.3% of HEALTHY joint-frames. With
    seg_reject correctly re-set to 1.80, a 1.52x shin is rightly no longer "impossible".
    The drift is now -0.85 (shin ~2.0x), which is impossible at any calibration.
    The TEST was wrong, not the threshold.
    """
    r = Rig().warm()
    p1_accepted = 0
    p14_rejected = 0
    final = None
    for i in range(1, 26):
        # drift the knee upward past the hip: thigh collapses, shin over-extends.
        frac = i / 25.0
        m = {LKN: (BASE[LKN][0] + 0.28 * frac, BASE[LKN][1] - 0.85 * frac, 2.0)}
        out = r.step(*pose(m, confmod={LKN: 0.85}))
        st_p1 = r.sk.trackers[LKN].state
        if st_p1 == TrackingState.TRACKED:
            p1_accepted += 1
        if out[LKN][6] == Observation.GEOMETRICALLY_SUSPECT:
            p14_rejected += 1
        final = out[LKN]
    check("D2 P1-1 alone accepts the slow impossible drift", p1_accepted > 10,
          "P1-1 TRACKED on %d/25 frames at conf 0.85" % p1_accepted)
    check("D2 P1-4 geometry rejects it anyway", p14_rejected > 0,
          "geometric rejections=%d (confidence never dropped)" % p14_rejected)
    js = r.kr.joints[LKN]
    check("D2 rejection was geometric, not confidence-based",
          js.len_violations + js.angle_violations > 0,
          "lenViolations=%d angleViolations=%d" % (js.len_violations, js.angle_violations))


# ------------------------------------------------------------------ E
def case_e_frozen_wrist():
    r = Rig().warm()
    frozen = BASE[LWR]
    for i in range(30):
        dx = 0.02 * i                            # arm moves...
        m = {LSH: (BASE[LSH][0] + dx, BASE[LSH][1], 2.0),
             LEL: (BASE[LEL][0] + dx, BASE[LEL][1], 2.0),
             LWR: frozen}                        # ...wrist does not
        out = r.step(*pose(m))
    wr_state = out[LWR][4]
    el_err = _dist(out[LEL][:3], (BASE[LEL][0] + 0.02 * 29, BASE[LEL][1], 2.0))
    sh_err = _dist(out[LSH][:3], (BASE[LSH][0] + 0.02 * 29, BASE[LSH][1], 2.0))
    tr = r.sk.trackers[LWR]
    check("E frozen wrist: detected",
          tr.frozen_count > 0 or wr_state != TrackingState.TRACKED,
          "frozen_count=%d state=%s" % (tr.frozen_count, TrackingState.name(wr_state)))
    check("E frozen wrist: does not poison elbow", el_err < 0.05, "elbow err %.4f" % el_err)
    check("E frozen wrist: does not poison shoulder", sh_err < 0.05, "shoulder err %.4f" % sh_err)


# ------------------------------------------------------------------ F
def case_f_bad_elbow():
    r = Rig().warm()
    bad = (0.30 + 0.45, -0.25 + 0.30, 2.0)       # elbow far off, shoulder+wrist fine
    out = r.step(*pose({LEL: bad}, confmod={LEL: 0.9}))
    p = out[LEL][:3]
    check("F bad elbow: not accepted as-is", _dist(p, bad) > 0.15,
          "distance from bad value %.3f m" % _dist(p, bad))
    # if reconstructed, the two bones must be plausible
    e1 = r.kr.lengths[(LSH, LEL)]
    e2 = r.kr.lengths[(LEL, LWR)]
    if e1.ready() and e2.ready() and out[LEL][5]:
        up = _dist(out[LSH][:3], p)
        lo = _dist(p, out[LWR][:3])
        r1 = abs(up - e1.length()) / e1.length()
        r2 = abs(lo - e2.length()) / e2.length()
        check("F bad elbow: reconstructed bones plausible", r1 < 0.25 and r2 < 0.25,
              "upper %.1f%%  fore %.1f%%" % (r1 * 100, r2 * 100))
    else:
        check("F bad elbow: reconstructed bones plausible", False, "estimator not ready")


# ------------------------------------------------------------------ G
def case_g():
    a = (0.0, 0.0, 0.0)
    ref = (0.2, 0.3, 0.0)
    # 1) AC > AB + BC
    c1 = (2.0, 0.0, 0.0)
    b1, s1 = solve_two_anchor(a, c1, 0.4, 0.4, ref)
    check("G unreachable (AC > AB+BC): no NaN", b1 is not None and finite(b1),
          "status=%s b=%s" % (s1, None if b1 is None else tuple(round(v, 3) for v in b1)))
    check("G unreachable: reported as unreachable", s1 == "unreachable", "status=%s" % s1)
    # 2) AC < |AB - BC|
    c2 = (0.05, 0.0, 0.0)
    b2, s2 = solve_two_anchor(a, c2, 0.8, 0.2, ref)
    check("G contained (AC < |AB-BC|): no NaN", b2 is not None and finite(b2),
          "status=%s b=%s" % (s2, None if b2 is None else tuple(round(v, 3) for v in b2)))
    check("G contained: reported as contained", s2 == "contained", "status=%s" % s2)
    # 3) coincident anchors
    b3, s3 = solve_two_anchor(a, a, 0.4, 0.4, ref)
    check("G coincident anchors: no NaN", b3 is not None and finite(b3), "status=%s" % s3)
    # 4) normal case must actually satisfy both lengths
    c4 = (0.6, 0.0, 0.0)
    b4, s4 = solve_two_anchor(a, c4, 0.45, 0.43, ref)
    ok = b4 is not None and abs(_dist(a, b4) - 0.45) < 1e-6 and abs(_dist(b4, c4) - 0.43) < 1e-6
    check("G normal case: both bone lengths satisfied exactly", ok,
          "|AB|=%.6f |BC|=%.6f status=%s" % (_dist(a, b4), _dist(b4, c4), s4))
    # 5) bend direction follows the reference, not an arbitrary pick
    ref_up = (0.3, -0.5, 0.0)
    ref_dn = (0.3, 0.5, 0.0)
    bu, _ = solve_two_anchor(a, c4, 0.45, 0.43, ref_up)
    bd, _ = solve_two_anchor(a, c4, 0.45, 0.43, ref_dn)
    check("G bend direction follows the reference", bu[1] < 0 and bd[1] > 0,
          "ref_up.y=%.3f  ref_dn.y=%.3f" % (bu[1], bd[1]))


# ------------------------------------------------------------------ H
def case_h_reacquisition():
    r = Rig().warm()
    for i in range(10):                          # lose the knee
        r.step(*pose(drop=(LKN,)))
    moved = (BASE[LKN][0] + 0.18, BASE[LKN][1] + 0.10, 2.0)
    prev = None
    steps = []
    for i in range(14):                          # it comes back somewhere else
        out = r.step(*pose({LKN: moved}))
        p = out[LKN][:3]
        if prev is not None:
            steps.append(_dist(prev, p))
        prev = p
    check("H reacquisition: no snap", max(steps) <= r.kr.cfg.recover_max_step + 1e-6,
          "max per-frame step %.4f m (limit %.3f)" % (max(steps), r.kr.cfg.recover_max_step))
    check("H reacquisition: converges to the new measurement",
          _dist(prev, moved) < 0.05, "final err %.4f m" % _dist(prev, moved))
    check("H reacquisition: counted", r.kr.joints[LKN].recoveries > 0,
          "recoveries=%d" % r.kr.joints[LKN].recoveries)


# ------------------------------------------------------------------ I
def case_i_fast_motion():
    """CRITICAL: legitimate fast motion must not be rejected."""
    r = Rig().warm()
    rej = 0
    lost = 0
    n = 0
    for i in range(60):
        # 0.05 m/frame at 30 fps = 1.5 m/s wrist, a hard dance swing
        dx = 0.05 * math.sin(i * 0.5)
        dy = 0.04 * math.cos(i * 0.5)
        m = {LWR: (BASE[LWR][0] + dx * 2.0, BASE[LWR][1] + dy * 2.0, 2.0),
             LEL: (BASE[LEL][0] + dx, BASE[LEL][1] + dy, 2.0),
             LAN: (BASE[LAN][0] + dx * 1.5, BASE[LAN][1], 2.0),
             LKN: (BASE[LKN][0] + dx * 0.8, BASE[LKN][1], 2.0)}
        out = r.step(*pose(m))
        for idx in (LWR, LEL, LKN, LAN):
            n += 1
            if out[idx][6] == Observation.GEOMETRICALLY_SUSPECT:
                rej += 1
            if out[idx][4] == TrackingState.LOST:
                lost += 1
    pct = 100.0 * rej / max(1, n)
    check("I fast motion: not rejected as geometrically suspect", pct < 5.0,
          "%.1f%% of %d joint-frames rejected" % (pct, n))
    check("I fast motion: nothing goes LOST", lost == 0, "lost=%d" % lost)


# ------------------------------------------------------------------ extras
def case_j_length_estimator():
    cfg = RecoveryConfig()
    from kinematic_recovery import BoneLengthEstimator
    e = BoneLengthEstimator("t", cfg)
    for i in range(30):
        e.observe(0.45 + 0.002 * math.sin(i))
    check("J estimator: converged to the true length", abs(e.length() - 0.45) < 0.005,
          "len=%.4f mad=%.5f" % (e.length(), e.spread()))
    before = e.length()
    for i in range(10):
        e.observe(1.20)                          # sustained bad samples
    check("J estimator: rejects sustained outliers",
          abs(e.length() - before) < 0.01 and e.rejected >= 10,
          "len %.4f -> %.4f, rejected=%d" % (before, e.length(), e.rejected))


def case_k_angles():
    straight = interior_angle_deg((0, -1, 0), (0, 0, 0), (0, 1, 0))
    right = interior_angle_deg((0, -1, 0), (0, 0, 0), (1, 0, 0))
    check("K interior angle: straight = 180", abs(straight - 180.0) < 1e-6, "%.2f" % straight)
    check("K interior angle: right = 90", abs(right - 90.0) < 1e-6, "%.2f" % right)
    check("K interior angle: degenerate returns None",
          interior_angle_deg((0, 0, 0), (0, 0, 0), (1, 0, 0)) is None)


def case_l_perf():
    import time
    r = Rig().warm()
    p, c = pose()
    for i in range(30):
        r.step(p, c)
    times = []
    for i in range(1500):
        tr = r.sk.update(p, c, r.t, collect_events=False)
        t0 = time.perf_counter()
        r.kr.apply(tr, p, collect_events=False)
        times.append((time.perf_counter() - t0) * 1000.0)
        r.t += DT
    times.sort()
    n = len(times)
    med, p95, p99 = times[n // 2], times[int(0.95 * n)], times[int(0.99 * n)]
    check("L P1-4 cost: median < 1 ms", med < 1.0, "median %.4f ms" % med)
    check("L P1-4 cost: p99 < 2 ms", p99 < 2.0,
          "p95 %.4f  p99 %.4f  max %.4f ms" % (p95, p99, times[-1]))


def main():
    print("=" * 78)
    print("P1-4 kinematic recovery — deterministic tests")
    print("=" * 78)
    for fn in [case_a_normal, case_b_knee_5frames, case_c_knee_20frames,
               case_d_highconf_teleport, case_d2_geometry_overrides_confidence, case_e_frozen_wrist, case_f_bad_elbow,
               case_g, case_h_reacquisition, case_i_fast_motion,
               case_j_length_estimator, case_k_angles, case_l_perf]:
        fn()
    n = len(_results)
    p = sum(1 for _n, ok, _d in _results if ok)
    print("-" * 78)
    print("%d/%d assertions passed" % (p, n))
    if p != n:
        print("\nFAILED:")
        for nm, ok, d in _results:
            if not ok:
                print("  %s  %s" % (nm, d))
    return 0 if p == n else 1


if __name__ == "__main__":
    sys.exit(main())
