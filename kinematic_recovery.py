#!/usr/bin/env python3
"""P1-4 — LONG-HORIZON JOINT RECOVERY + KINEMATIC CONSTRAINTS.

P1-1 (`joint_tracker.py`) reasons about ONE joint at a time. That is enough for a spike or a
short dropout, but it cannot solve the case P1-4 exists for:

    hip    = correct
    ankle  = correct
    knee   = CONFIDENTLY WRONG, and stays wrong past the prediction horizon

A single-joint tracker has no evidence that the knee is wrong — its own velocity history is
self-consistent. Only the SKELETON knows: a knee that makes the thigh 0.9 m and the shin 0.2 m
is wrong no matter what the model's confidence says.

This module adds skeleton-level reasoning ON TOP of P1-1. It does not replace it, does not
re-filter it, and does not touch P0 or the Unity side.

    RTMW3D -> depth -> P0 smoother -> P1-1 JointTracker -> [P1-4 THIS MODULE] -> build msg -> UDP

WHAT THIS IS NOT
----------------
* Not IK. Nothing here drives the avatar rig; it repairs a LANDMARK before it is transmitted,
  which is what P1-4 Part 5/6 specify. Unity's `useIkDriver` stays 0.
* Not another smoothing stage. It only acts on joints P1-1 has already declared unusable or
  that fail a geometric test. A healthy joint passes through BYTE-IDENTICAL.
* Not a biomechanics engine. The angle limits are deliberately loose — they exist to reject
  physically impossible frames, not to model a real knee.

THE CENTRAL RULE
----------------
    high confidence != correct

The geometric consistency score is computed WITHOUT reference to model confidence, so a joint
at confidence 0.8 whose geometry is impossible is still rejected.
"""

import math

from joint_tracker import TrackingState

# --------------------------------------------------------------------------- skeleton

# (proximal anchor, middle, distal anchor) — the chains P1-4 Part 2 names.
# Indices are WholeBody, matching joint_tracker.WB_NAMES.
CHAINS = [
    (5, 7, 9),      # left  shoulder -> elbow -> wrist
    (6, 8, 10),     # right shoulder -> elbow -> wrist
    (11, 13, 15),   # left  hip -> knee -> ankle
    (12, 14, 16),   # right hip -> knee -> ankle
]

# bone (parent, child) -> stable key for the length estimator
BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]

# torso quad, used only for the body-level consistency score (P1-4 Part 11)
TORSO = [(5, 6), (11, 12), (5, 11), (6, 12)]

MIDDLE_OF = {}      # middle index -> (A, C)
for _a, _b, _c in CHAINS:
    MIDDLE_OF[_b] = (_a, _c)

END_OF = {}         # distal index -> (parent, chain root)
for _a, _b, _c in CHAINS:
    END_OF[_c] = (_b, _a)


class Observation:
    """How P1-4 classifies ONE measurement, independent of the model's own confidence."""
    VALID = 0
    WEAK = 1
    GEOMETRICALLY_SUSPECT = 2
    INVALID = 3

    NAMES = {0: "VALID", 1: "WEAK", 2: "GEOMETRICALLY_SUSPECT", 3: "INVALID"}

    @staticmethod
    def name(v):
        return Observation.NAMES.get(v, "?")


class RecoveryConfig:
    """Every threshold here must be justified by replay evidence (AGENTS.md sec.5)."""

    def __init__(self, **kw):
        # --- robust bone-length estimator (Part 3) -------------------------------
        self.len_window = 64          # samples kept for the median/MAD estimate
        self.len_min_samples = 12     # below this the estimate is not trusted at all
        # A sample only teaches the estimator when BOTH endpoints are healthy. Learning from a
        # reconstructed or predicted joint would let the skeleton drift to fit its own errors.
        self.len_outlier_mad = 4.0    # reject a learning sample this many MADs from the median
        # TRUSTWORTHINESS GATE. Reconstruction is only as good as the length prior it uses.
        # MEASURED: on real capture the hip-knee estimator accepted just 45 of 818 samples
        # (5.5%) because this pipeline's landmarks are metrically unstable -- and
        # reconstructing a 20-frame knee gap from that 5% estimate was WORSE than simply
        # holding (max error 0.170 m -> 0.327 m). A bone whose samples are mostly rejected
        # does not describe a real limb, so it must not be used to invent one.
        self.len_min_accept_ratio = 0.30   # accepted / (accepted+rejected) required to reconstruct

        # --- segment-length constraint (Part 4) ----------------------------------
        # MEASURED on 853 frames of real human capture (p12h_latest), bone length relative
        # deviation from its OWN median:
        #     median 0.101   p95 0.589   p99 1.791   max 3.528
        #     worst bone (R-shoulder-R-elbow) p95 = 1.514
        # i.e. this pipeline's landmarks are NOT metrically stable -- a "bone" routinely
        # varies by 50-150% on perfectly healthy frames. An earlier 0.45 threshold, set by
        # reasoning rather than measurement, therefore fired on 25.3% of clean joint-frames
        # and drove TRACKED from 95.7% down to 67.6%. That was a false-positive machine.
        #
        # DECISION: bone length is NOT a veto on this pipeline. seg_reject now sits ABOVE the
        # measured p99 of natural variation, so it only fires when a bone is ~3x wrong --
        # genuinely impossible, not merely noisy. The scale-free ANGLE check (below) carries
        # the real detection load.
        # FALSE-POSITIVE TRADE: a moderately wrong joint now passes the length test. Accepted
        # deliberately: on this data a tighter length gate costs far more than it catches.
        self.seg_reject = 1.80        # > measured p99 (1.791) of NATURAL variation
        self.seg_warn = 0.60          # ~ measured p95 (0.589); evidence only, never a veto

        # --- angle plausibility (Part 7) -----------------------------------------
        # Interior angle at the middle joint (hip-knee-ankle / shoulder-elbow-wrist).
        # 180 deg = fully straight. Humans hyperextend a knee ~5-10 deg, so >195 is impossible.
        # Deep flexion (heel to buttock) reaches ~30 deg, so <20 is impossible.
        # MEASURED on the same 853 real frames (3409 chain-samples): min 30.5, max 179.3,
        # p1 71.8, p99 176.9 -> these limits produce ZERO false positives on real motion.
        # This is the scale-free signal and it does the real work; unlike bone length it is
        # unaffected by the pipeline's metric instability.
        self.angle_min_deg = 20.0
        self.angle_max_deg = 195.0
        # One-frame change limit. A hard kick changes knee angle ~90 deg in ~0.15 s
        # (~28 deg/frame at 21 fps), so 120 only catches true flips -- it must NOT clip a
        # fast kick. FALSE-POSITIVE TRADE: a genuine sub-120-deg/frame flip would pass.
        self.angle_max_rate_deg = 120.0

        # --- staged recovery (Part 9) --------------------------------------------
        # 0..max_predict_frames is P1-1's own short prediction (unchanged, 6 frames).
        # Beyond that P1-4 reconstructs geometrically for up to max_reconstruct_frames.
        # Past that: LOST and hold, rather than drifting indefinitely.
        self.max_reconstruct_frames = 30   # ~1.4 s at 21 fps
        self.reconstruct_needs_anchors = True

        # --- recovery blending (Part 10) -----------------------------------------
        self.recover_frames = 8       # blend reconstructed -> measured over N frames
        self.recover_max_step = 0.06  # m per frame during RECOVERING; bounds any snap

        # --- solver ---------------------------------------------------------------
        self.solver_eps = 1e-6

        for k, v in kw.items():
            if not hasattr(self, k):
                raise KeyError("unknown RecoveryConfig field: %s" % k)
            setattr(self, k, v)


# --------------------------------------------------------------------------- vec helpers

def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _dist(a, b):
    return _norm(_sub(a, b))


def _unit(a, eps=1e-9):
    n = _norm(a)
    if n < eps:
        return None
    return (a[0] / n, a[1] / n, a[2] / n)


def _finite(p):
    for c in p:
        if not (c == c) or c in (float("inf"), float("-inf")):
            return False
    return True


def interior_angle_deg(a, b, c):
    """Angle at B in the chain A-B-C. 180 = straight. Returns None when degenerate."""
    u = _unit(_sub(a, b))
    v = _unit(_sub(c, b))
    if u is None or v is None:
        return None
    d = _dot(u, v)
    if d < -1.0:
        d = -1.0
    elif d > 1.0:
        d = 1.0
    return math.degrees(math.acos(d))


# --------------------------------------------------------------------------- Part 3

class BoneLengthEstimator:
    """Robust running length for ONE bone: median + MAD over a bounded window.

    Deliberately NOT an EMA. An EMA chases every measurement, including the bad ones, so the
    'expected' length drifts toward whatever error is currently happening -- which defeats the
    whole point of having an expectation. A median over trusted samples does not move for a
    single bad frame, and MAD gives an honest spread to gate on.
    """

    __slots__ = ("cfg", "name", "_buf", "_i", "_n", "_med", "_mad", "accepted", "rejected")

    def __init__(self, name, cfg):
        self.cfg = cfg
        self.name = name
        self._buf = [0.0] * cfg.len_window
        self._i = 0
        self._n = 0
        self._med = 0.0
        self._mad = 0.0
        self.accepted = 0
        self.rejected = 0

    def ready(self):
        return self._n >= self.cfg.len_min_samples

    def trustworthy(self):
        """Ready AND built from a representative sample. Only a trustworthy estimator may
        drive reconstruction; a merely `ready` one is still fine for plausibility scoring."""
        if not self.ready():
            return False
        seen = self.accepted + self.rejected
        if seen < self.cfg.len_min_samples:
            return False
        return (self.accepted / float(seen)) >= self.cfg.len_min_accept_ratio

    def accept_ratio(self):
        seen = self.accepted + self.rejected
        return (self.accepted / float(seen)) if seen else 0.0

    def length(self):
        return self._med

    def spread(self):
        return self._mad

    def observe(self, length):
        """Feed a TRUSTED sample only (both endpoints healthy + geometry plausible)."""
        if length <= 0.0 or not (length == length):
            return False
        # Once established, refuse a sample far outside the current distribution -- otherwise
        # a sustained bad limb slowly teaches the estimator its own error.
        if self.ready() and self._mad > 1e-6:
            if abs(length - self._med) > self.cfg.len_outlier_mad * self._mad:
                self.rejected += 1
                return False
        self._buf[self._i] = length
        self._i = (self._i + 1) % self.cfg.len_window
        if self._n < self.cfg.len_window:
            self._n += 1
        self.accepted += 1
        self._recompute()
        return True

    def _recompute(self):
        s = sorted(self._buf[:self._n])
        n = len(s)
        self._med = s[n // 2]
        dev = sorted(abs(x - self._med) for x in s)
        self._mad = dev[n // 2]
        if self._mad < 1e-4:
            self._mad = 1e-4        # floor: a perfectly still limb must not give a zero gate


# --------------------------------------------------------------------------- Part 6

def solve_two_anchor(a, c, r_ab, r_bc, reference, eps=1e-6):
    """Reconstruct B in the chain A-B-C from both anchors and the expected bone lengths.

    Classic two-sphere intersection: the solution set is a CIRCLE, and we pick the point on it
    closest to `reference` (the previous/predicted B). That is what preserves the bend
    direction -- picking arbitrarily is how a knee ends up inverted.

    Returns (B, status) where status is one of:
        'exact'        - the circle existed and we picked from it
        'unreachable'  - d > r_ab + r_bc  : anchors too far apart, limb fully extended
        'contained'    - d < |r_ab - r_bc|: one sphere inside the other
        'degenerate'   - A and C coincide, or no usable reference direction
    Never returns NaN/Inf (P1-4 Part 13).
    """
    ac = _sub(c, a)
    d = _norm(ac)

    if d < eps:
        # A and C coincide: the circle is undefined. Keep the reference offset from A at r_ab.
        u = _unit(_sub(reference, a)) if reference is not None else None
        if u is None:
            return None, "degenerate"
        return _add(a, _scale(u, r_ab)), "degenerate"

    n = _scale(ac, 1.0 / d)

    # --- degenerate ranges: clamp onto the A->C line, never fabricate a circle -------------
    if d > r_ab + r_bc - eps:
        # Unreachable: the limb cannot span the gap. Fully extend along A->C and place B at
        # the proportional split so BOTH bones stay as close to expectation as possible.
        t = r_ab / max(eps, r_ab + r_bc)
        return _add(a, _scale(ac, t)), "unreachable"
    if d < abs(r_ab - r_bc) + eps:
        # One sphere contains the other. Place B along A->C at r_ab from A (the closest point
        # on the AB sphere to the axis) -- safe, continuous, and never NaN.
        return _add(a, _scale(n, r_ab)), "contained"

    # --- the normal case: intersection circle -----------------------------------------------
    # x = distance from A to the circle plane along n
    x = (d * d + r_ab * r_ab - r_bc * r_bc) / (2.0 * d)
    h_sq = r_ab * r_ab - x * x
    if h_sq <= 0.0:
        return _add(a, _scale(n, r_ab)), "contained"
    h = math.sqrt(h_sq)
    centre = _add(a, _scale(n, x))

    # Pick the circle point nearest the reference: project (reference - centre) onto the plane
    # perpendicular to n, then walk out by h.
    radial = None
    if reference is not None:
        rv = _sub(reference, centre)
        rv = _sub(rv, _scale(n, _dot(rv, n)))       # remove the axial component
        radial = _unit(rv)
    if radial is None:
        # No usable reference direction (reference sits on the axis). Any perpendicular is
        # equally valid; choose a stable one derived from the axis itself.
        seed = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0)
        radial = _unit(_cross(n, seed))
        if radial is None:
            return _add(a, _scale(n, r_ab)), "degenerate"

    b = _add(centre, _scale(radial, h))
    if not _finite(b):
        return _add(a, _scale(n, r_ab)), "degenerate"
    return b, "exact"


# --------------------------------------------------------------------------- Part 8 + 5

class JointRecoveryState:
    """P1-4's per-joint bookkeeping. P1-1 keeps its own state; this is additive."""

    __slots__ = ("name", "recovering_left", "recon_frames", "last_good",
                 "reconstructions", "geo_rejections", "len_violations",
                 "angle_violations", "recoveries", "lost_frames", "recon_err_sum",
                 "recon_err_max", "recon_err_n", "observation")

    def __init__(self, name):
        self.name = name
        self.recovering_left = 0
        self.recon_frames = 0
        self.last_good = None
        self.reconstructions = 0
        self.geo_rejections = 0
        self.len_violations = 0
        self.angle_violations = 0
        self.recoveries = 0
        self.lost_frames = 0
        self.recon_err_sum = 0.0
        self.recon_err_max = 0.0
        self.recon_err_n = 0
        self.observation = Observation.VALID


class KinematicRecovery:
    """Skeleton-level constraint + long-horizon recovery layer (P1-4).

    Consumes the output of `SkeletonTracker.update()` and returns a repaired joint set.
    Healthy joints pass through unchanged.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or RecoveryConfig()
        self.lengths = {}
        for pa, ch in BONES:
            self.lengths[(pa, ch)] = BoneLengthEstimator("%d-%d" % (pa, ch), self.cfg)
        self.joints = {}
        self.events = []
        self.frame = 0
        # body-level (Part 11)
        self.body_consistency = 1.0
        self.tracked_ratio = 0.0
        self.predicted_ratio = 0.0
        self.lost_ratio = 0.0

    def _js(self, idx, name):
        js = self.joints.get(idx)
        if js is None:
            js = JointRecoveryState(name)
            self.joints[idx] = js
        return js

    def reset(self):
        for e in self.lengths.values():
            e.__init__(e.name, self.cfg)
        self.joints = {}
        self.events = []
        self.frame = 0

    # ------------------------------------------------------------------ geometry score
    def geometric_score(self, idx, pos, tracked, distrust=None):
        """Consistency of `pos` with the skeleton, computed WITHOUT model confidence.

        Returns (observation, relative_length_error, angle_deg_or_None).
        This is the Part 8 requirement: a joint at confidence 0.8 that fails here is still
        rejected, because nothing in this function can see the model's confidence.
        """
        cfg = self.cfg
        worst = 0.0
        ang = None

        # --- bone lengths touching this joint ------------------------------------
        for pa, ch in BONES:
            if idx not in (pa, ch):
                continue
            other = ch if idx == pa else pa
            op = tracked.get(other)
            if op is None:
                continue
            # ENDPOINT ARBITRATION: if the OTHER endpoint is itself untrustworthy, the bone
            # violation is evidence against IT, not against this joint. Without this a frozen
            # wrist makes its (perfectly good) elbow look implausible and gets it rebuilt.
            if distrust is not None and other in distrust and idx not in distrust:
                continue
            est = self.lengths[(pa, ch)]
            if not est.ready():
                continue
            measured = _dist(pos, op)
            exp = est.length()
            if exp <= 1e-6:
                continue
            rel = abs(measured - exp) / exp
            if rel > worst:
                worst = rel

        # --- interior angle when this joint is a chain middle ---------------------
        if idx in MIDDLE_OF:
            a_i, c_i = MIDDLE_OF[idx]
            ap = tracked.get(a_i)
            cp = tracked.get(c_i)
            if ap is not None and cp is not None:
                ang = interior_angle_deg(ap, pos, cp)

        obs = Observation.VALID
        if worst > cfg.seg_reject:
            obs = Observation.GEOMETRICALLY_SUSPECT
        elif worst > cfg.seg_warn:
            obs = Observation.WEAK
        if ang is not None and (ang < cfg.angle_min_deg or ang > cfg.angle_max_deg):
            obs = Observation.GEOMETRICALLY_SUSPECT
        return obs, worst, ang

    # ------------------------------------------------------------------ main entry
    def apply(self, tracker_out, positions_raw, collect_events=True):
        """tracker_out: {idx: (x, y, z, conf, state, usable)} straight from SkeletonTracker.

        Returns {idx: (x, y, z, conf, state, usable, observation)}.
        """
        from joint_tracker import WB_NAMES
        cfg = self.cfg
        self.frame += 1

        # Trusted positions this frame = joints P1-1 considers healthy. Anchors must come from
        # here, never from a joint we are about to repair.
        trusted = {}
        for idx, (x, y, z, _c, st, usable) in tracker_out.items():
            if usable and st in (TrackingState.TRACKED, TrackingState.WEAK):
                trusted[idx] = (x, y, z)

        # --- 1. learn bone lengths from clean pairs only (Part 3) ------------------
        for pa, ch in BONES:
            a = trusted.get(pa)
            b = trusted.get(ch)
            if a is None or b is None:
                continue
            # only learn when BOTH endpoints are strictly TRACKED (not WEAK, not recovered)
            if tracker_out[pa][4] != TrackingState.TRACKED or tracker_out[ch][4] != TrackingState.TRACKED:
                continue
            self.lengths[(pa, ch)].observe(_dist(a, b))

        # Joints P1-1 already doubts. Used for endpoint arbitration below.
        distrust = set()
        for idx, (_x, _y, _z, _c, st, usable) in tracker_out.items():
            if (not usable) or st != TrackingState.TRACKED:
                distrust.add(idx)

        out = {}
        n_tracked = n_pred = n_lost = 0
        seg_bad = 0
        seg_total = 0

        for idx, (x, y, z, conf, st, usable) in tracker_out.items():
            name = WB_NAMES.get(idx, "j%d" % idx)
            js = self._js(idx, name)
            pos = (x, y, z)
            p_present = positions_raw.get(idx) is not None
            obs = Observation.VALID
            new_state = st
            new_usable = usable
            new_pos = pos

            # ---------- geometric verdict, independent of model confidence --------
            if usable and st in (TrackingState.TRACKED, TrackingState.WEAK):
                peers = dict(trusted)
                peers.pop(idx, None)
                obs, rel, ang = self.geometric_score(idx, pos, peers, distrust)
                seg_total += 1
                if obs == Observation.GEOMETRICALLY_SUSPECT:
                    seg_bad += 1
                    js.geo_rejections += 1
                    if ang is not None and (ang < cfg.angle_min_deg or ang > cfg.angle_max_deg):
                        js.angle_violations += 1
                    else:
                        js.len_violations += 1
                    # A high-confidence but geometrically impossible sample is DEMOTED here.
                    # This is the core P1-4 behaviour: geometry overrides confidence.
                    new_usable = False
                    if collect_events:
                        self.events.append({
                            "frame": self.frame, "joint": name, "event": "GEOMETRIC_REJECT",
                            "confidence": round(conf, 3), "relLenErr": round(rel, 3),
                            "angleDeg": None if ang is None else round(ang, 1),
                            "reason": "angle" if js.angle_violations and ang is not None
                                      and (ang < cfg.angle_min_deg or ang > cfg.angle_max_deg)
                                      else "segment",
                        })

            # ---------- mid-recovery: keep easing toward the measurement ----------
            if js.recovering_left > 0 and js.last_good is not None and p_present:
                step = _dist(pos, js.last_good)
                if step > cfg.recover_max_step:
                    u = _unit(_sub(pos, js.last_good))
                    new_pos = (_add(js.last_good, _scale(u, cfg.recover_max_step))
                               if u is not None else js.last_good)
                else:
                    new_pos = pos
                js.recovering_left -= 1
                new_usable = True
                new_state = (TrackingState.RECOVERING if js.recovering_left > 0
                             else TrackingState.TRACKED)
                if not _finite(new_pos):
                    new_pos = js.last_good
                js.last_good = new_pos
                js.observation = obs
                if new_state == TrackingState.TRACKED:
                    n_tracked += 1
                out[idx] = (new_pos[0], new_pos[1], new_pos[2], conf, new_state, True, obs)
                continue

            # ---------- reconstruct when unusable --------------------------------
            if not new_usable:
                rec, status = self._reconstruct(idx, js, trusted, pos)
                if rec is not None:
                    js.recon_frames += 1
                    if js.recon_frames <= cfg.max_reconstruct_frames:
                        js.reconstructions += 1
                        # error vs what the (rejected) measurement claimed -- diagnostic only
                        if st in (TrackingState.TRACKED, TrackingState.WEAK):
                            e = _dist(rec, pos)
                            js.recon_err_sum += e
                            js.recon_err_n += 1
                            if e > js.recon_err_max:
                                js.recon_err_max = e
                        new_pos = rec
                        new_usable = True
                        new_state = TrackingState.PREDICTED
                        obs = Observation.GEOMETRICALLY_SUSPECT if obs == Observation.VALID else obs
                        if collect_events and js.recon_frames == 1:
                            self.events.append({
                                "frame": self.frame, "joint": name,
                                "event": "RECONSTRUCT", "solver": status,
                            })
                    else:
                        # Past the safe horizon: stop inventing motion (Part 9).
                        new_usable = False
                        new_state = TrackingState.LOST
                        js.lost_frames += 1
                else:
                    new_state = TrackingState.LOST
                    new_usable = False
                    js.lost_frames += 1
            else:
                # measurement accepted -> if we had been reconstructing, blend back (Part 10)
                if js.recon_frames > 0:
                    js.recovering_left = cfg.recover_frames
                    js.recoveries += 1
                    js.recon_frames = 0
                    if collect_events:
                        self.events.append({
                            "frame": self.frame, "joint": name, "event": "RECOVERING",
                        })
                if js.recovering_left > 0 and js.last_good is not None:
                    step = _dist(pos, js.last_good)
                    limit = cfg.recover_max_step
                    if step > limit:
                        u = _unit(_sub(pos, js.last_good))
                        if u is not None:
                            new_pos = _add(js.last_good, _scale(u, limit))
                    js.recovering_left -= 1
                    new_state = (TrackingState.RECOVERING if js.recovering_left > 0
                                 else TrackingState.TRACKED)

            # ---------- safety: never emit NaN/Inf (Part 13) ----------------------
            if not _finite(new_pos):
                new_pos = js.last_good if js.last_good is not None else pos
                if not _finite(new_pos):
                    new_pos = (0.0, 0.0, 0.0)
                    new_usable = False
                    new_state = TrackingState.LOST

            if new_usable:
                js.last_good = new_pos
            js.observation = obs

            if new_state == TrackingState.TRACKED:
                n_tracked += 1
            elif new_state == TrackingState.PREDICTED:
                n_pred += 1
            elif new_state == TrackingState.LOST:
                n_lost += 1

            out[idx] = (new_pos[0], new_pos[1], new_pos[2], conf, new_state, new_usable, obs)

        # --- body-level diagnostics (Part 11) -- DIAGNOSTIC ONLY, never suppresses ----
        total = max(1, len(tracker_out))
        self.tracked_ratio = n_tracked / float(total)
        self.predicted_ratio = n_pred / float(total)
        self.lost_ratio = n_lost / float(total)
        seg_ok = 1.0 - (seg_bad / float(seg_total)) if seg_total else 1.0
        self.body_consistency = max(0.0, min(1.0,
                                             0.5 * self.tracked_ratio + 0.3 * seg_ok
                                             + 0.2 * (1.0 - self.lost_ratio)))
        return out

    # ------------------------------------------------------------------ reconstruction
    def _reconstruct(self, idx, js, trusted, fallback):
        """Return (position, solver_status) or (None, reason)."""
        cfg = self.cfg
        ref = js.last_good if js.last_good is not None else fallback

        if idx in MIDDLE_OF:
            a_i, c_i = MIDDLE_OF[idx]
            a = trusted.get(a_i)
            c = trusted.get(c_i)
            if a is not None and c is not None:
                e1 = self.lengths.get((a_i, idx))
                e2 = self.lengths.get((idx, c_i))
                if (e1 is not None and e2 is not None
                        and e1.trustworthy() and e2.trustworthy()):
                    b, status = solve_two_anchor(a, c, e1.length(), e2.length(), ref,
                                                 cfg.solver_eps)
                    if b is not None and _finite(b):
                        return b, status
            # only one anchor available -> fall through to the parent-projection below

        # END joint (wrist/ankle), or a middle joint with a single anchor: keep the bone
        # length and the previous DIRECTION from the parent. This preserves limb length
        # without inventing a new orientation.
        pa = None
        if idx in END_OF:
            pa = END_OF[idx][0]
        elif idx in MIDDLE_OF:
            pa = MIDDLE_OF[idx][0]
        if pa is not None:
            p = trusted.get(pa)
            key = (pa, idx)
            est = self.lengths.get(key)
            if p is not None and est is not None and est.trustworthy() and ref is not None:
                u = _unit(_sub(ref, p))
                if u is not None:
                    return _add(p, _scale(u, est.length())), "projected"
        return None, "no_anchor"

    # ------------------------------------------------------------------ telemetry
    def telemetry(self):
        """Aggregate snapshot (Part 15). Caller decides how often to emit it."""
        rows = {}
        for idx, js in self.joints.items():
            mean_err = (js.recon_err_sum / js.recon_err_n) if js.recon_err_n else 0.0
            rows[js.name] = dict(
                reconstructions=js.reconstructions,
                geoRejections=js.geo_rejections,
                lenViolations=js.len_violations,
                angleViolations=js.angle_violations,
                recoveries=js.recoveries,
                lostFrames=js.lost_frames,
                meanReconErr=round(mean_err, 4),
                maxReconErr=round(js.recon_err_max, 4),
                observation=Observation.name(js.observation),
            )
        bones = {}
        for k, e in self.lengths.items():
            if e.ready():
                bones["%d-%d" % k] = dict(length=round(e.length(), 4),
                                          mad=round(e.spread(), 4),
                                          accepted=e.accepted, rejected=e.rejected)
        return dict(joints=rows, bones=bones,
                    bodyConsistencyScore=round(self.body_consistency, 3),
                    trackedJointRatio=round(self.tracked_ratio, 3),
                    predictedJointRatio=round(self.predicted_ratio, 3),
                    lostJointRatio=round(self.lost_ratio, 3))
