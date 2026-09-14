#!/usr/bin/env python3
"""F-21 - Single-Person Lock / Target Ownership.

RTMW3D-x (rtmw3d_pose.py) is a single-person TOP-DOWN 3D pose model: one inference per frame, one set
of keypoints, no detector, no track id, no multi-person output of any kind (verified against source,
not assumed - see docs/F21_SINGLE_PERSON_TARGET_OWNERSHIP_2026-09-14.md SS2). The existing "M15"
person-box loop in wholebody_udp_sender.py just re-centres its crop on whichever body is currently
most confident, with zero notion of whether it is still the same body. That silent hand-off is the
exact F-19 failure this module closes.

This module is deliberately the ONLY place identity is decided. It is pure and clock-injected (no
camera, no socket, no wall-clock reads) - the same shape as joint_tracker.py's JointTracker and
Unity's TrackingStreamHealth.cs - so it is fully unit-testable and so the sender only has to feed it
one observation per frame and act on the verdict.

Identity signals actually available (and no others - do not add a track id here without a source
change that provides one):
  - RAW mid-hip camera-space position (metres) - read BEFORE P0's smoother rate-limits it, or a real
    person-swap looks like fast continuous motion over a couple of frames instead of a jump.
  - body_conf_mean (the same 0..1-ish raw SimCC confidence M15 already gates on).
  - torso span (shoulder<->hip distance, metres) as a coarse body-scale corroborator.
There is no appearance/ReID signal and no detector-provided id anywhere in this codebase. Candidate
count is therefore always 0 or 1, never N - this module does not pretend otherwise.

States:
    NO_TARGET -> ACQUIRING -> LOCKED -> TEMPORARILY_LOST -> REACQUIRING -> (LOCKED | RELEASED)
    RELEASED is transient: it is returned/logged for exactly one update() call, then the machine
    folds back to NO_TARGET on the next call so a new acquisition can start immediately.

Every transition is logged (self.events, drained by the caller) with an explicit reason - "Unexpected
switching must be extremely easy to detect in logs" is a hard requirement, not a nice-to-have.
"""
import math

NO_TARGET = "NO_TARGET"
ACQUIRING = "ACQUIRING"
LOCKED = "LOCKED"
TEMPORARILY_LOST = "TEMPORARILY_LOST"
REACQUIRING = "REACQUIRING"
RELEASED = "RELEASED"


def _dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


class Observation(object):
    """One frame's single candidate (or an empty/invalid one - never more than one, see module doc).

    pos: (x, y, z) camera-space metres, mid-hip, RAW (pre-smoothing). conf: body_conf_mean.
    scale: torso span metres, or None if not computable this frame.
    """
    __slots__ = ("valid", "pos", "conf", "scale")

    def __init__(self, valid, pos=None, conf=0.0, scale=None):
        self.valid = valid
        self.pos = pos
        self.conf = conf
        self.scale = scale


class OwnershipConfig(object):
    """All six SS5 parameters, plus the scale-margin corroborator. Every default is derived and cited
    in the class docstring below and in the report - none is an arbitrary round number.

    MIN_TARGET_CONFIDENCE = 0.3      reuses the EXISTING --conf gate M15 already applies (production
                                      value, not a new number).
    ACQUIRE_CONFIRM_FRAMES = 5       mirrors P1-1's own --tracker-reacquire-frames default (5).
    REACQUIRE_CONFIRM_FRAMES = 5     same standard as acquisition - no special-casing without evidence.
    SWITCH_MARGIN_M = 0.35           starting value: comfortably above per-frame depth/position noise
                                      (P0's OWN --arm-max-jump/--leg-max-jump distal caps are 0.35 m,
                                      tuned against this exact sensor's real noise floor) while staying
                                      well below "a different person is standing where the owner was."
                                      This is a STARTING value tuned further against live evidence
                                      (report SS6/SS16), deliberately smaller than the upstream
                                      smoother's 1.5 m/frame cap since it is evaluated pre-smoothing.
    SCALE_MARGIN_RATIO = 0.45        corroborating only, generous on purpose (adults vary in torso
                                      span more than this within normal measurement noise; this catches
                                      a grossly different body, not a discriminator on its own).
    REACQUIRE_WINDOW_S = 2.0         starting value: an occlusion-recovery budget, reasoned from P1-1's
                                      6-frame (~0.2 s) PER-JOINT prediction horizon scaled up to a
                                      WHOLE-BODY, human-behaviour timescale (stepping behind an
                                      obstruction, not a single dropped joint) - tuned live.
    RELEASE_TIMEOUT_S = 4.0          starting value: deliberately LONGER than F-20A's 2 s
                                      STALE_FAILSAFE, because this is about human behaviour (did they
                                      leave), not transport health - Unity already shows neutral well
                                      before this via F-20A regardless, so a longer window costs
                                      nothing visually while avoiding a premature release.
    """
    def __init__(self, **kw):
        self.min_confidence = kw.get("min_confidence", 0.3)
        self.acquire_confirm_frames = kw.get("acquire_confirm_frames", 5)
        self.reacquire_confirm_frames = kw.get("reacquire_confirm_frames", 5)
        self.switch_margin_m = kw.get("switch_margin_m", 0.35)
        self.scale_margin_ratio = kw.get("scale_margin_ratio", 0.45)
        self.reacquire_window_s = kw.get("reacquire_window_s", 2.0)
        self.release_timeout_s = kw.get("release_timeout_s", 4.0)


class TargetOwnership(object):
    def __init__(self, cfg=None):
        self.cfg = cfg or OwnershipConfig()
        self.state = NO_TARGET
        self.owner_pos = None
        self.owner_scale = None
        self.owner_conf = 0.0
        self.owner_since = None
        self.lost_since = None
        self.confirm_count = 0
        self.switch_count = 0
        self.epoch = 0          # LOCAL ownership-epoch counter for logs/HUD - NOT a re-identified
                                 # person; it only distinguishes "this lock" from "the previous lock".
        self._acquiring_ref = None
        self._acquiring_scale = None
        self._had_owner_before = False
        self._released_at = None
        self.events = []

    # ---- internal -----------------------------------------------------------------------------
    def _emit(self, kind, **fields):
        rec = dict(event=kind, state=self.state, target_id=self.epoch)
        rec.update(fields)
        self.events.append(rec)

    def _matches_owner(self, obs):
        if self.owner_pos is None or obs.pos is None:
            return False, "no_reference"
        d = _dist(obs.pos, self.owner_pos)
        if d > self.cfg.switch_margin_m:
            return False, "position_jump"
        if self.owner_scale is not None and obs.scale is not None:
            lo = self.owner_scale * (1.0 - self.cfg.scale_margin_ratio)
            hi = self.owner_scale * (1.0 + self.cfg.scale_margin_ratio)
            if not (lo <= obs.scale <= hi):
                return False, "scale_mismatch"
        return True, ""

    def _lock(self, obs, t, reacquired):
        self.owner_pos = obs.pos
        self.owner_scale = obs.scale if obs.scale is not None else self.owner_scale
        self.owner_conf = obs.conf
        self.state = LOCKED
        if reacquired:
            loss_duration = t - self.lost_since if self.lost_since is not None else 0.0
            self._emit("TARGET_REACQUIRED", pos=obs.pos, loss_duration=round(loss_duration, 3))
        else:
            self.epoch += 1
            self.owner_since = t
            self._emit("TARGET_ACQUIRED", pos=obs.pos)
            self._emit("TARGET_LOCKED", pos=obs.pos)
            if self._had_owner_before and self._released_at is not None:
                self.switch_count += 1
                self._emit("TARGET_SWITCH",
                           reason="new acquisition %.1fs after previous release"
                                  % (t - self._released_at))
            self._had_owner_before = True
        self.lost_since = None
        self.confirm_count = 0

    # ---- public ---------------------------------------------------------------------------------
    def update(self, obs, t):
        """obs: Observation for this frame. t: seconds, any monotonic clock the caller chooses.
        Returns (state, should_emit): should_emit is True exactly when the caller should treat this
        frame's pose as the owner's and let it flow downstream (build the packet); False means hold/
        withhold - the caller reuses its EXISTING no-op path (the same `continue` used today when
        mid_hip can't be computed), so F-20A's stale watchdog does the rest with no new Unity code."""
        if self.state == RELEASED:
            self.state = NO_TARGET

        valid = bool(obs.valid and obs.conf >= self.cfg.min_confidence and obs.pos is not None)

        if self.state == NO_TARGET:
            if valid:
                self.state = ACQUIRING
                self.confirm_count = 1
                self._acquiring_ref = obs.pos
                self._acquiring_scale = obs.scale
            return self.state, False

        if self.state == ACQUIRING:
            if not valid:
                self.state = NO_TARGET
                self.confirm_count = 0
                return self.state, False
            if _dist(obs.pos, self._acquiring_ref) <= self.cfg.switch_margin_m:
                self.confirm_count += 1
                self._acquiring_ref = obs.pos      # allow drift while genuinely walking in
            else:
                # A DIFFERENT plausible body appeared mid-acquisition. Nobody owns anything yet, so
                # this is not a "switch" - just restart the confirm count on the new candidate.
                # Two bodies alternating every frame will therefore never accumulate enough
                # confirmations to lock (SS3/SS14: prefer no lock over an arbitrary one).
                self.confirm_count = 1
                self._acquiring_ref = obs.pos
                self._acquiring_scale = obs.scale
            if self.confirm_count >= self.cfg.acquire_confirm_frames:
                self._lock(obs, t, reacquired=False)
                return self.state, True
            return self.state, False

        if self.state == LOCKED:
            if valid:
                ok, reason = self._matches_owner(obs)
                if ok:
                    self.owner_pos = obs.pos
                    self.owner_scale = obs.scale if obs.scale is not None else self.owner_scale
                    self.owner_conf = obs.conf
                    return self.state, True
                self._emit("TARGET_REJECTED_CANDIDATE", reason=reason, pos=obs.pos)
            else:
                reason = "no_observation"
            self.state = TEMPORARILY_LOST
            self.lost_since = t
            self._emit("TARGET_TEMP_LOST", reason=reason)
            return self.state, False

        if self.state in (TEMPORARILY_LOST, REACQUIRING):
            loss_elapsed = t - self.lost_since
            if loss_elapsed > self.cfg.release_timeout_s:
                self._emit("TARGET_RELEASED", loss_duration=round(loss_elapsed, 3))
                self.state = RELEASED
                self._released_at = t
                self.owner_pos = None
                self.owner_scale = None
                self.owner_since = None
                self.lost_since = None
                self.confirm_count = 0
                return self.state, False

            matched = False
            if valid and loss_elapsed <= self.cfg.reacquire_window_s:
                matched, reason = self._matches_owner(obs)

            if matched:
                if self.state == TEMPORARILY_LOST:
                    self.state = REACQUIRING
                    self.confirm_count = 1
                else:
                    self.confirm_count += 1
                if self.confirm_count >= self.cfg.reacquire_confirm_frames:
                    self._lock(obs, t, reacquired=True)
                    return self.state, True
                return self.state, False

            if self.state == REACQUIRING:
                # the match broke before confirming - fall back rather than keep a partial credit,
                # so a flickering false match can't slowly accumulate into a wrong reacquire.
                self.state = TEMPORARILY_LOST
                self.confirm_count = 0
            if valid:
                self._emit("TARGET_REJECTED_CANDIDATE", reason="not_owner", pos=obs.pos)
            return self.state, False

        return self.state, False

    def drain_events(self):
        ev = self.events
        self.events = []
        return ev

    def snapshot(self, t):
        """Everything the SS17 HUD needs, in one call - diagnostic only, never consumed to decide
        behaviour (the HUD reads this; nothing reads the HUD)."""
        owner_age = None
        loss_duration = None
        release_remaining = None
        if self.state in (LOCKED, TEMPORARILY_LOST, REACQUIRING) and self.owner_since is not None:
            owner_age = t - self.owner_since
        if self.state in (TEMPORARILY_LOST, REACQUIRING) and self.lost_since is not None:
            loss_duration = t - self.lost_since
            release_remaining = max(0.0, self.cfg.release_timeout_s - loss_duration)
        acquire_progress = self.confirm_count if self.state in (ACQUIRING, REACQUIRING) else 0
        acquire_needed = (self.cfg.acquire_confirm_frames if self.state == ACQUIRING else
                          self.cfg.reacquire_confirm_frames if self.state == REACQUIRING else None)
        return dict(
            state=self.state,
            target_id=self.epoch if self.state in (LOCKED, TEMPORARILY_LOST, REACQUIRING) else None,
            owner_confidence=(self.owner_conf if self.state in
                              (LOCKED, TEMPORARILY_LOST, REACQUIRING) else 0.0),
            owner_age_s=owner_age,
            loss_duration_s=loss_duration,
            release_remaining_s=release_remaining,
            acquire_progress=acquire_progress,
            acquire_needed=acquire_needed,
            switch_count=self.switch_count,
        )
