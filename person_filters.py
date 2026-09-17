#!/usr/bin/env python3
"""F-33 - PER-PERSON SIGNAL CONDITIONING. The measured single-person filter stack, once per identity.

F-32 shipped multi-person tracking with a stated hole: the pose for each person went to the wire
RAW. None of P0 smoothing, P1-1 joint tracking, P1-4 recovery or F-22 validation ran, because every
one of them is a TEMPORAL filter carrying per-joint state, and state belongs to a person - not to a
slot in a list that a re-ordering of the tracker's output silently reassigns. This module closes
that hole: one complete filter chain per track id, created when the identity is born and destroyed
when it dies.

    xyz_cam (back-projected, noisy)
        -> P0  KeypointSmoother      per-keypoint One-Euro + displacement cap + hold-on-dropout
        -> mid-hip (with the M11 hold, PER PERSON)
        -> P1-1 SkeletonTracker      per-joint temporal validation; LOST -> emit confidence zeroed
        -> P1-4 KinematicRecovery    OPTIONAL, default OFF, exactly as in production
        -> F-22 PoseValidator        elbow/knee biomechanics on the geometry actually emitted
        -> build_body_landmarks

THE ORDER IS COPIED FROM wholebody_udp_sender.main() AND MATTERS. mid-hip is taken AFTER P0 and
BEFORE P1-1, so the origin every landmark is expressed relative to is smoothed but not
tracker-modified. Reversing those two makes the whole body breathe whenever the tracker corrects a
hip, which reads as the floor moving rather than the person.

THE ONE THING THAT IS NOT A STRAIGHT COPY: SAMPLE RATE.

KeypointSmoother takes freq once at construction and the single-person sender leaves it at the
default 30.0, which is true there - that loop runs at camera rate. It is NOT true here. Three people
cost three sequential 20.7 ms pose solves, so the loop runs at ~16 fps, and a person below the
--max-poses cap is updated rarer still. One-Euro derives velocity as delta * freq, so a filter told
30 while actually sampled at 16 over-estimates speed by 1.9x, inflates its adaptive cutoff, and
OPENS UP exactly when it should be damping.

Measured on a synthetic 0.5 Hz reach with 10 mm of noise (docs/evidence/f33/freq_probe.txt) - note
that the wrong rate is worse on BOTH axes at once, which is not the usual smoothing trade-off:

    actual rate   told freq=30 (naive port)      told the truth
    30 fps        13.5 mm jitter / 167 ms lag    13.5 mm / 167 ms   (identical - nothing to fix)
    16 fps        32.8 mm jitter / 250 ms lag    25.1 mm / 188 ms
    10 fps        59.7 mm jitter / 300 ms lag    39.1 mm / 200 ms

So each person MEASURES ITS OWN update cadence and retunes its own filters. That also handles the
--max-poses case for free: a person solved every third frame simply reports a third of the rate.

Pure logic - no camera, no socket, no wall-clock read. Every method takes the time, exactly like
person_tracker.py and target_ownership.py, so the whole chain is unit-testable.
"""

import numpy as np

import joint_tracker as JT
import kinematic_recovery as KR
import pose_validation as PV
import smoothing

#: Keypoints that get the HEAVY depth One-Euro and the bounded hold-on-dropout: trunk (5,6,11,12),
#: arms (7,8,9,10), legs (13,14,15,16) and both hands (91-132). Copied from the single-person
#: sender's limb_idx; tests/test_person_filters.py fails if the two ever disagree.
SINGLE_PERSON_LIMB_INDICES = frozenset(list(range(5, 17)) + list(range(91, 133)))

#: The six COCO-WholeBody FOOT keypoints (17-22: big toe, small toe, heel, each side). The wire
#: carries four of them - FOOT_TO_JOINTID sends big toes to JointId 31/32 and heels to 29/30.
FOOT_INDICES = frozenset(range(17, 23))

#: The five COCO-WholeBody HEAD keypoints (0 nose, 1-2 eyes, 3-4 ears). Like the feet, these were in
#: no filter group at all - see LIMB_INDICES.
FACE_INDICES = frozenset(range(0, 5))

#: ADR-071. F-33 ADDS the feet and the head to that set. This is the ONE place the chain
#: deliberately differs from the single-person sender, so it is measured rather than assumed.
#:
#: The feet and the head were in no filter group at all: not trunk, not arm, not leg, not hand. They
#: still got the light image-plane One-Euro - KeypointSmoother covers all 133 slots - but no heavy
#: depth cutoff, no bounded hold, and no tightened displacement cap. Both then produced the worst
#: artefacts in the whole filtered run, and for DIFFERENT reasons, which is why they need different
#: treatment:
#:   * FEET fail with src=0. No depth that frame, no hold to fall back on, so the emitted point was
#:     rebuilt from the model's raw monocular z and moved metres. The hold is what fixes this.
#:   * HEAD fails with src=1. Depth WAS sampled - from a window straddling the edge of the head and
#:     catching the wall behind it. A hold cannot help; the displacement cap is what fixes this.
#:
#: Measured over 2060 person-frames of the seven-dancer clip, identical input to every run
#: (docs/evidence/f33/video_ab_456.txt), counting single-frame steps over 300 mm - a joint crossing
#: 300 mm in 33 ms is moving at 9 m/s and is not a person:
#:
#:     no filters at all (raw F-32)               2158   worst step 2091 mm
#:     filters, single-person grouping             630   worst step 2018 mm
#:     filters + feet + head (this)                 77   worst step  960 mm
#:
#: The last step changed the distal, trunk and hip figures by 0.0% to the decimal and emitted exactly
#: the same 40200 joints - it moved only the joints it targets, which is what a correct grouping fix
#: should look like and is the reason to believe it rather than a tuning accident.
#:
#: This matters beyond tidiness: F-29 put the feet on screen and the Footprints experience reads heel
#: and toe directly, so an unfiltered heel is not a number in a log, it is a footprint appearing two
#: metres from the person who made it. The head drives the avatar's gaze.
LIMB_INDICES = SINGLE_PERSON_LIMB_INDICES | FOOT_INDICES | FACE_INDICES

#: Distal joints whose per-frame displacement cap is tighter than the global one (P0-2): elbows and
#: wrists take arm_max_jump, knees and ankles take leg_max_jump. Feet take the leg cap - they are
#: the far end of the same chain and move no faster than the ankle they hang off. The head takes
#: head_max_jump: unlike the feet its failures come with src=1, a depth window straddling the edge
#: of the head and sampling the wall behind it, so what bites is the displacement cap and not the
#: hold. A nose crossing 350 mm in one 33 ms frame is 10 m/s, which no neck does.
ARM_INDICES = (7, 8, 9, 10)
LEG_INDICES = (13, 14, 15, 16)
HEAD_INDICES = (0, 1, 2, 3, 4)

#: How many recent inter-update gaps a person keeps to estimate its own sample rate. 15 at ~16 fps
#: is about a second - long enough that one late frame cannot move the estimate, short enough that
#: the rate change from a fourth person walking in is picked up within a second.
CADENCE_WINDOW = 15


class FilterConfig(object):
    """Every knob, defaulted to the value the single-person sender ships.

    These are DUPLICATED from wholebody_udp_sender.main()'s argparse defaults rather than imported,
    because that parser is built inside main() and importing it would mean running it. Duplication
    invites drift, so tests/test_person_filters.py reads the sender's source and asserts every one
    of these still matches. Change a number there and that test fails here.
    """

    __slots__ = ("min_cutoff", "beta", "max_jump", "arm_max_jump", "leg_max_jump", "head_max_jump",
                 "depth_min_cutoff", "depth_beta", "max_hold_frames",
                 "smooth", "tracker", "tracker_predict_frames", "tracker_reacquire_frames",
                 "recovery", "recovery_max_frames", "recovery_blend_frames", "pose_validation",
                 "filter_feet", "filter_head",
                 "adaptive_rate", "min_freq", "max_freq", "default_freq", "idle_release_seconds")

    def __init__(self, min_cutoff=0.5, beta=0.4, max_jump=1.5, arm_max_jump=0.35,
                 leg_max_jump=0.35, head_max_jump=0.35, depth_min_cutoff=0.3, depth_beta=0.1,
                 max_hold_frames=8,
                 smooth=True, tracker=True, tracker_predict_frames=6, tracker_reacquire_frames=5,
                 recovery=False, recovery_max_frames=30, recovery_blend_frames=8,
                 pose_validation=True, filter_feet=True, filter_head=True,
                 adaptive_rate=True, min_freq=4.0, max_freq=60.0,
                 default_freq=30.0, idle_release_seconds=3.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.max_jump = max_jump
        self.arm_max_jump = arm_max_jump
        self.leg_max_jump = leg_max_jump
        #: F-33 only; the single-person sender has no equivalent flag. See HEAD_INDICES.
        self.head_max_jump = head_max_jump
        self.depth_min_cutoff = depth_min_cutoff
        self.depth_beta = depth_beta
        self.max_hold_frames = max_hold_frames
        self.smooth = smooth
        self.tracker = tracker
        self.tracker_predict_frames = tracker_predict_frames
        self.tracker_reacquire_frames = tracker_reacquire_frames
        #: P1-4 is REJECTED for production (docs/P1_4_CLOSEOUT_2026-09-08.md). It is reachable here
        #: for the same research A/B the single-person sender allows, and defaults OFF for the same
        #: reason. Gated on tracker because it consumes P1-1's output and is meaningless without it.
        self.recovery = recovery
        self.recovery_max_frames = recovery_max_frames
        self.recovery_blend_frames = recovery_blend_frames
        self.pose_validation = pose_validation
        #: ADR-071, and the off-switches its measurement was taken with. See FOOT_INDICES and
        #: FACE_INDICES: both groups are OUTSIDE the single-person sender's limb set, and turning
        #: them off restores that grouping exactly.
        self.filter_feet = filter_feet
        self.filter_head = filter_head
        #: See the module docstring. Off restores the naive port (every filter told 30 fps), which
        #: is what an A/B needs to show the difference rather than assert it.
        self.adaptive_rate = adaptive_rate
        #: Clamp on the measured rate. Below ~4 fps One-Euro is not meaningfully filtering anything
        #: and a wilder estimate would only amplify noise in the estimate itself; above 60 it is a
        #: timing glitch, not a camera.
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.default_freq = default_freq
        #: A bank is dropped this long after its person stops being SEEN - last_seen_at, not
        #: last_update_at, so a few frames whose pose failed do not release a person standing right
        #: there. Tracks are released by PersonTracker after 1.2 s, so 3.0 s keeps the chain alive
        #: across a release-reacquire without holding the memory of everyone who ever walked past.
        self.idle_release_seconds = idle_release_seconds


class PersonFilters(object):
    """One person's complete filter chain, with its own temporal state and its own measured rate."""

    def __init__(self, person_id, cfg=None, now=0.0):
        self.id = int(person_id)
        self.cfg = cfg or FilterConfig()
        self.born_at = now
        #: TWO clocks, and conflating them is a bug this module's tests caught. `last_seen_at` is
        #: LIFECYCLE - the pool asks "is this identity still around?" and a person whose pose failed
        #: this frame is still around. `last_update_at` is the RATE ESTIMATE - it advances only when
        #: the chain actually filtered something, because a frame that produced no sample is not a
        #: sample interval. Driving the idle timer off the rate clock released the bank of anyone
        #: whose hips went unconfident for a few seconds, mid-session, while they stood there.
        self.last_seen_at = now
        self.last_update_at = now
        self.updates = 0
        #: M11, per person: the last mid-hip that was actually measured, held through a transient
        #: hip depth-hole. Single-person holds one of these; N people need N, and sharing one is the
        #: bug that would place everybody at whoever was seen last.
        self.last_mid_hip = None
        self.freq = self.cfg.default_freq
        self._gaps = []
        self._applied_freq = None

        self._smoother = None
        if self.cfg.smooth:
            limbs = set(SINGLE_PERSON_LIMB_INDICES)
            overrides = {}
            for idx in ARM_INDICES:
                overrides[idx] = self.cfg.arm_max_jump
            for idx in LEG_INDICES:
                overrides[idx] = self.cfg.leg_max_jump
            if self.cfg.filter_feet:
                limbs |= FOOT_INDICES
                for idx in FOOT_INDICES:
                    overrides[idx] = self.cfg.leg_max_jump
            if self.cfg.filter_head:
                limbs |= FACE_INDICES
                for idx in HEAD_INDICES:
                    overrides[idx] = self.cfg.head_max_jump
            self._smoother = smoothing.KeypointSmoother(
                133, freq=self.cfg.default_freq, min_cutoff=self.cfg.min_cutoff,
                beta=self.cfg.beta, max_jump=self.cfg.max_jump,
                depth_min_cutoff=self.cfg.depth_min_cutoff, depth_beta=self.cfg.depth_beta,
                limb_indices=limbs, max_hold=self.cfg.max_hold_frames,
                max_jump_overrides=overrides)

        self._skel = None
        if self.cfg.tracker:
            self._skel = JT.SkeletonTracker(cfg=JT.TrackerConfig(
                max_predict_frames=self.cfg.tracker_predict_frames,
                reacquire_frames=self.cfg.tracker_reacquire_frames))

        self._recovery = None
        if self.cfg.tracker and self.cfg.recovery:
            self._recovery = KR.KinematicRecovery(KR.RecoveryConfig(
                max_reconstruct_frames=self.cfg.recovery_max_frames,
                recover_frames=self.cfg.recovery_blend_frames))

        self._validator = PV.PoseValidator() if self.cfg.pose_validation else None

        #: DIAG-ONLY counters. Read by the sender's periodic log line; nothing reads them back.
        self.counts = {"smoothed": 0, "rate_limited": 0, "held": 0, "dropped": 0,
                       "lost": 0, "rejected": 0}

    # ---- sample rate -------------------------------------------------------------------------

    def observe_rate(self, now):
        """Fold this update's gap into the person's own rate estimate and retune their filters.

        Uses the MEDIAN of the recent gaps, not a mean or an EMA. One person being skipped for a
        frame produces a single gap three times the others; a mean is dragged by it for as long as
        the window lasts, a median ignores it. The estimate must describe the steady cadence,
        because that is what One-Euro's velocity term divides by.
        """
        gap = now - self.last_update_at
        self.last_update_at = now
        self.updates = self.updates + 1
        if gap <= 0.0 or gap > 2.0:
            # First update, a clock that went backwards, or a gap so long the person was effectively
            # absent. None of those describe a sample rate; leave the estimate alone.
            return self.freq
        self._gaps.append(gap)
        if len(self._gaps) > CADENCE_WINDOW:
            del self._gaps[0]
        if not self.cfg.adaptive_rate or len(self._gaps) < 3:
            return self.freq
        median_gap = float(np.median(np.asarray(self._gaps, dtype=np.float64)))
        if median_gap <= 0.0:
            return self.freq
        self.freq = max(self.cfg.min_freq, min(self.cfg.max_freq, 1.0 / median_gap))
        # Retune only on a real change. Writing the same value 399 times a frame is pure waste, and
        # 2% hysteresis keeps normal frame-time noise from counting as a change.
        if self._smoother is not None:
            if self._applied_freq is None or abs(self.freq - self._applied_freq) > 0.02 * self.freq:
                self._smoother.set_freq(self.freq)
                self._applied_freq = self.freq
        return self.freq

    # ---- stage 1: P0 -------------------------------------------------------------------------

    def smooth(self, xyz_cam, measured):
        """P0 in place: One-Euro per keypoint, displacement cap, bounded hold-on-dropout.

        measured is written back because a held limb is reported valid: the build step must then use
        the held value rather than falling through to the zrel back-projection, which is the 8-12 m
        spike the hold exists to prevent.
        """
        if self._smoother is None:
            return
        n = int(xyz_cam.shape[0])
        i = 0
        while i < n:
            sx, sy, sz, eff, action, _disp = self._smoother.filter(
                i, float(xyz_cam[i, 0]), float(xyz_cam[i, 1]), float(xyz_cam[i, 2]),
                bool(measured[i]))
            xyz_cam[i, 0] = sx
            xyz_cam[i, 1] = sy
            xyz_cam[i, 2] = sz
            measured[i] = eff
            if action == "RATE_LIMIT":
                self.counts["rate_limited"] = self.counts["rate_limited"] + 1
            elif action == "HOLD":
                self.counts["held"] = self.counts["held"] + 1
            elif action == "DROP":
                self.counts["dropped"] = self.counts["dropped"] + 1
            i = i + 1
        self.counts["smoothed"] = self.counts["smoothed"] + 1

    # ---- stage 2: the origin ----------------------------------------------------------------

    def mid_hip(self, xyz_cam, measured, fallback=None):
        """The person's mid-hip in camera metres, or None if it cannot be established.

        M11, per person: both hips -> midpoint; one hip -> that hip; neither, but this person has a
        recent one -> HOLD it through the depth-hole rather than stalling their whole body.
        fallback is the multi-person addition - the tracker's own depth-derived hip, used only when
        this person has never had a measured one.
        """
        if bool(measured[11]) and bool(measured[12]):
            self.last_mid_hip = (xyz_cam[11] + xyz_cam[12]) / 2.0
        elif bool(measured[11]):
            self.last_mid_hip = xyz_cam[11].copy()
        elif bool(measured[12]):
            self.last_mid_hip = xyz_cam[12].copy()
        elif self.last_mid_hip is None:
            return fallback
        return self.last_mid_hip

    # ---- stage 3: P1-1, P1-4, F-22 ------------------------------------------------------------

    def refine(self, xyz_cam, measured, conf, now):
        """Temporal validation and biomechanics on the geometry about to be emitted.

        Returns (conf_emit, track_res). conf_emit is a COPY: a joint P1-1 calls LOST, or F-22 calls
        anatomically impossible, has its emit confidence zeroed so build_body_landmarks drops it -
        which hands the decision to Unity's P0 LimbGate exactly as a real occlusion would
        (AGENTS.md section 2: never emit a fabricated position to keep a joint alive). The INPUT
        confidences are never modified, so the hand builder still sees what the model said.

        track_res is captured AFTER P1-4 so the st field on the wire describes the geometry that was
        actually sent, not an earlier opinion of it.
        """
        conf_emit = conf.copy()
        track_res = None
        if self._skel is not None:
            positions = {}
            confidences = {}
            depth_valid = {}
            for j in self._skel.indices:
                positions[j] = (float(xyz_cam[j, 0]), float(xyz_cam[j, 1]), float(xyz_cam[j, 2]))
                confidences[j] = float(conf[j]) if bool(measured[j]) else 0.0
                depth_valid[j] = bool(measured[j])
            res = self._skel.update(positions, confidences, now, depth_valid=depth_valid,
                                    collect_events=False)
            if self._recovery is not None:
                rec = self._recovery.apply(res, positions, collect_events=False)
                res = dict((k, v[:6]) for k, v in rec.items())
            track_res = res
            for j, entry in res.items():
                if entry[5]:
                    xyz_cam[j, 0] = entry[0]
                    xyz_cam[j, 1] = entry[1]
                    xyz_cam[j, 2] = entry[2]
                    measured[j] = True
                else:
                    measured[j] = False
                    conf_emit[j] = 0.0
                    self.counts["lost"] = self.counts["lost"] + 1

        if self._validator is not None:
            out = self._validator.update(xyz_cam, measured, conf_emit, now)
            for j, entry in out.items():
                if entry[0] != PV.VALID:
                    conf_emit[j] = 0.0
                    self.counts["rejected"] = self.counts["rejected"] + 1
            self._validator.drain_events()
        return conf_emit, track_res

    def snapshot(self):
        """Everything a HUD or a log line needs, and nothing that decides behaviour."""
        return {"id": self.id, "freq": round(self.freq, 1), "updates": self.updates,
                "rate_limited": self.counts["rate_limited"], "held": self.counts["held"],
                "lost": self.counts["lost"], "rejected": self.counts["rejected"]}


class PersonFilterPool(object):
    """The filter chains, keyed by track id, with a lifecycle tied to the identity.

    WHY A POOL AND NOT A LIST. PersonTracker.update() returns people sorted most-established-first,
    so a person's INDEX in that list changes the moment somebody else gains a hit. A filter bank
    addressed by index would hand person A's One-Euro history to person B on that frame - every
    joint would appear to teleport, and the displacement cap would then spend several frames slewing
    the two bodies into each other. Track ids are stable and never reused, which is exactly the
    property a temporal filter needs.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or FilterConfig()
        self.banks = {}
        self.created = 0
        self.released = 0

    def acquire(self, person_id, now):
        """This person's chain, creating it on first sight."""
        key = int(person_id)
        bank = self.banks.get(key)
        if bank is None:
            bank = PersonFilters(key, self.cfg, now)
            self.banks[key] = bank
            self.created = self.created + 1
        bank.last_seen_at = now
        return bank

    def release(self, person_id):
        """Drop one chain, for a caller that knows the track is gone (a RELEASED event)."""
        if self.banks.pop(int(person_id), None) is not None:
            self.released = self.released + 1
            return True
        return False

    def release_idle(self, now):
        """Drop every chain whose person has not been updated recently; returns the dropped ids.

        Idle-based rather than event-based ON PURPOSE. Forwarding the tracker's RELEASED events
        would work until the day a caller forgets to drain them, and the failure mode of that is a
        slow leak over an evening's run - the kind nobody notices until the process is hours old.
        Time since last use needs no cooperation from the caller and cannot be forgotten.
        """
        dropped = []
        for key, bank in list(self.banks.items()):
            if (now - bank.last_seen_at) > self.cfg.idle_release_seconds:
                del self.banks[key]
                self.released = self.released + 1
                dropped.append(key)
        return dropped

    def snapshot(self):
        return {"banks": len(self.banks), "created": self.created, "released": self.released}
