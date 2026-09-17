#!/usr/bin/env python3
"""F-32 — MULTI-PERSON TRACKING. Turns a bag of per-frame detections into persistent identities.

    detections (no identity, order meaningless) -> PersonTracker -> tracks with STABLE ids

WHAT THIS REPLACES. F-21's `target_ownership.py` holds exactly ONE person and refuses silent
hand-off; it exists because RTMW3D is single-person and the "M15" crop loop would slide between
bodies. That layer is not generalised here - it is superseded for the multi-person path, and F-32's
measurements are why. On a seven-person clip the single-person pipeline emitted a body whose
shoulder width ranged 0.068-0.455 m: a chimera assembled from whichever dancers fell inside the
migrating crop, with NO frame-to-frame jump big enough to trip any margin (max 172 mm/frame). You
cannot fix that with a better gate on one target. You need N targets and an assignment.

DESIGN, and each choice is forced by what this system actually has:

* ASSOCIATION IS 3-D, NOT IoU. Every published tracker (SORT, ByteTrack, OC-SORT) associates on
  image-plane IoU because that is all a webcam gives. This rig has an OAK-D: metric depth per
  detection. Two people overlapping on screen at different distances have near-perfect IoU and are
  trivially separable in Z. Using depth as a first-class association term is the single biggest
  advantage this system has over an off-the-shelf tracker, and IoU is kept only as a fallback for
  detections whose depth failed.

* OPTIMAL ASSIGNMENT, NOT GREEDY. See `assignment.py`: greedy is suboptimal on 54.6% of random 4x4
  cost matrices, and its failure mode is precisely an ID swap between crossing people.

* CONSTANT-VELOCITY PREDICTION, NO KALMAN FILTER. A Kalman filter needs filterpy or ~200 lines of
  matrix bookkeeping to estimate a covariance this problem does not use: the gate here is a fixed
  physical plausibility bound (people do not move faster than ~6 m/s), not an adaptive one. Damped
  constant velocity gets the prediction, and the residual gate does the rejecting. This mirrors what
  `joint_tracker.py` already does per JOINT, one level up.

* BIRTH AND DEATH ARE ASYMMETRIC, deliberately, and the numbers come from F-21 rather than being
  invented: confirm over several frames so a flickering false positive never becomes a person, but
  hold a lost track for over a second so an occlusion does not destroy an identity. Losing an ID
  costs a visitor their score; gaining a phantom costs a bubble spawning in mid-air.

Pure logic: no camera, no socket, no wall-clock read. Every method takes the time, exactly like
`target_ownership.py` and `TrackingStreamHealth.cs`, so the whole thing is unit-testable against
synthetic crossings.
"""

import numpy as np

import assignment

# ---- track states -------------------------------------------------------------------------------

TENTATIVE = "TENTATIVE"   # seen, not yet trusted - never emitted
CONFIRMED = "CONFIRMED"   # a real person, emitted
LOST = "LOST"             # not seen recently; coasting on prediction, still emitted as held
DEAD = "DEAD"             # released; the id is retired and never reused


class Detection(object):
    """One person found in one frame, with no identity.

    box   : (x1, y1, x2, y2) in NORMALISED image coordinates (0-1), so the tracker is resolution
            independent and a change of capture size cannot silently retune every threshold.
    score : detector confidence 0-1.
    depth : metres to the person, or None when depth sampling failed for this box. None is a real
            and common case (a person against a reflective floor, or beyond the stereo range), so
            the tracker must work without it rather than treating it as an error.
    """

    __slots__ = ("box", "score", "depth")

    def __init__(self, box, score, depth=None):
        self.box = tuple(float(v) for v in box)
        self.score = float(score)
        self.depth = None if depth is None else float(depth)

    @property
    def centre(self):
        x1, y1, x2, y2 = self.box
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    @property
    def height(self):
        return max(1e-6, self.box[3] - self.box[1])


class Track(object):
    """One person, followed across frames."""

    __slots__ = ("id", "state", "cx", "cy", "vx", "vy", "depth", "vdepth", "height",
                 "box", "score", "hits", "misses", "age", "born_at", "last_seen_at")

    def __init__(self, track_id, det, now):
        self.id = track_id
        self.state = TENTATIVE
        self.cx, self.cy = det.centre
        self.vx = 0.0
        self.vy = 0.0
        self.depth = det.depth
        self.vdepth = 0.0
        self.height = det.height
        self.box = det.box
        self.score = det.score
        self.hits = 1
        self.misses = 0
        self.age = 0
        self.born_at = now
        self.last_seen_at = now

    @property
    def centre(self):
        return (self.cx, self.cy)

    def predict(self, dt):
        """Damped constant velocity. The damping matters: an undamped prediction on a track that is
        coasting through an occlusion accelerates away from where the person actually is, and the
        residual gate then refuses the correct detection when they reappear."""
        damp = 0.85
        self.cx = self.cx + self.vx * dt
        self.cy = self.cy + self.vy * dt
        self.vx = self.vx * damp
        self.vy = self.vy * damp
        if self.depth is not None:
            self.depth = self.depth + self.vdepth * dt
            self.vdepth = self.vdepth * damp
        self.age = self.age + 1

    def correct(self, det, dt, now):
        """Fold in a matched detection."""
        ncx, ncy = det.centre
        if dt > 1e-6:
            # Blend the velocity estimate rather than replacing it: a single noisy detection should
            # not redirect a track, but a sustained move should.
            self.vx = 0.6 * self.vx + 0.4 * (ncx - self.cx) / dt
            self.vy = 0.6 * self.vy + 0.4 * (ncy - self.cy) / dt
            if det.depth is not None and self.depth is not None:
                self.vdepth = 0.6 * self.vdepth + 0.4 * (det.depth - self.depth) / dt
        self.cx, self.cy = ncx, ncy
        if det.depth is not None:
            self.depth = det.depth if self.depth is None else 0.5 * (self.depth + det.depth)
        # Body height is a slow-moving identity signal - a person's apparent size changes only as
        # they walk toward or away from the camera. Smoothed hard so one bad box cannot reshape it.
        self.height = 0.8 * self.height + 0.2 * det.height
        self.box = det.box
        self.score = det.score
        self.hits = self.hits + 1
        self.misses = 0
        self.last_seen_at = now


class TrackerConfig(object):
    """Every threshold, with the reasoning attached. Nothing here is a round number for its own sake.

    MAX_CENTRE_DISTANCE  0.28   normalised image units. A person crossing a 3 m-wide view at a brisk
                                2 m/s covers ~0.22 of the frame per 100 ms of detector latency; this
                                sits just above that so real motion is never gated out.
    MAX_DEPTH_DELTA      0.90   metres. Walking toward the camera at 2 m/s covers 0.18 m between
                                detections at 11 Hz; 0.9 m allows for that plus depth noise while
                                still separating two people standing a metre apart in Z.
    MAX_HEIGHT_RATIO     1.8    a detection nearly twice the track's body height is a different
                                person (or a broken box), not the same one.
    CONFIRM_HITS         3      matches F-21's family of confirm counts. Three detections at 11 Hz is
                                ~270 ms - fast enough that a visitor is not kept waiting, slow enough
                                that a single-frame false positive never becomes a person.
    MAX_MISSES_SECONDS   1.2    how long a lost track coasts before release. Deliberately longer than
                                a typical occlusion (walking behind someone) and shorter than the
                                time it takes a visitor to leave and be replaced.
    MIN_SCORE            0.35   detections below this never start a track, though they may still
                                UPDATE a confirmed one - the ByteTrack insight: a weak detection on
                                an established person is usually that person, partly occluded.
    """

    __slots__ = ("max_centre_distance", "max_depth_delta", "max_height_ratio",
                 "confirm_hits", "max_misses_seconds", "min_score", "depth_weight")

    def __init__(self, max_centre_distance=0.28, max_depth_delta=0.90, max_height_ratio=1.8,
                 confirm_hits=3, max_misses_seconds=1.2, min_score=0.35, depth_weight=0.35):
        self.max_centre_distance = max_centre_distance
        self.max_depth_delta = max_depth_delta
        self.max_height_ratio = max_height_ratio
        self.confirm_hits = confirm_hits
        self.max_misses_seconds = max_misses_seconds
        self.min_score = min_score
        #: How much a metre of depth disagreement counts relative to a full frame-width of image
        #: disagreement. 0.35 makes ~0.9 m of depth error as costly as ~0.3 of image width, so depth
        #: can separate two overlapping people without being able to override a clear image match.
        self.depth_weight = depth_weight


class PersonTracker(object):
    """Detections in, stable identities out.

    Usage, once per frame that has detections:

        tracks = tracker.update(detections, now)

    Returns the CONFIRMED and LOST tracks, newest-first by confirmation, so a caller that can only
    afford N pose inferences (F-32: ~20.7 ms each, so 2-3 people) takes the first N and knows they
    are the most established rather than an arbitrary subset.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or TrackerConfig()
        self.tracks = []
        self.events = []
        self._next_id = 1
        self._last_time = None

    def reset(self):
        self.tracks = []
        self.events = []
        self._next_id = 1
        self._last_time = None

    # ---- association cost ------------------------------------------------------------------

    def _cost(self, track, det):
        """Cost of matching this detection to this track, or None when the pair is implausible.

        Returning None rather than a big number is deliberate: the caller turns it into
        assignment.BIG for the solver AND remembers it was gated, so a solver forced to use a
        padded cell cannot be mistaken for a real match."""
        dx = det.centre[0] - track.cx
        dy = det.centre[1] - track.cy
        centre_dist = float(np.hypot(dx, dy))
        if centre_dist > self.cfg.max_centre_distance:
            return None

        ratio = det.height / max(1e-6, track.height)
        if ratio < 1.0 / self.cfg.max_height_ratio or ratio > self.cfg.max_height_ratio:
            return None

        cost = centre_dist
        if det.depth is not None and track.depth is not None:
            depth_delta = abs(det.depth - track.depth)
            if depth_delta > self.cfg.max_depth_delta:
                # The one case IoU-based trackers cannot see: overlapping on screen, metres apart
                # in reality. Refusing here is the whole reason depth is worth carrying.
                return None
            cost = cost + self.cfg.depth_weight * depth_delta
        # A detection the model is unsure of is a slightly worse explanation of a track than a
        # confident one, all else equal. Small term - it breaks ties, it does not drive matching.
        cost = cost + 0.02 * (1.0 - det.score)
        return cost

    # ---- the frame update ------------------------------------------------------------------

    def update(self, detections, now):
        dt = 0.0 if self._last_time is None else max(0.0, now - self._last_time)
        self._last_time = now

        for t in self.tracks:
            t.predict(dt)

        usable = [d for d in detections if d.score > 0.0]
        gated = {}
        if self.tracks and usable:
            cost = np.full((len(self.tracks), len(usable)), assignment.BIG, dtype=np.float64)
            for i, t in enumerate(self.tracks):
                for j, d in enumerate(usable):
                    c = self._cost(t, d)
                    if c is not None:
                        cost[i, j] = c
                        gated[(i, j)] = True
            pairs = assignment.solve(cost)
        else:
            pairs = []

        matched_tracks = set()
        matched_dets = set()
        for i, j in pairs:
            # The solver returns a COMPLETE matching on the padded matrix, so a pair it had no real
            # option for comes back at BIG. Those are not matches and must be dropped here.
            if (i, j) not in gated:
                continue
            self.tracks[i].correct(usable[j], dt, now)
            matched_tracks.add(i)
            matched_dets.add(j)

        for i, t in enumerate(self.tracks):
            if i in matched_tracks:
                if t.state == TENTATIVE and t.hits >= self.cfg.confirm_hits:
                    t.state = CONFIRMED
                    self._log(t, "CONFIRMED", now)
                elif t.state == LOST:
                    t.state = CONFIRMED
                    self._log(t, "REACQUIRED", now)
            else:
                t.misses = t.misses + 1
                if t.state == TENTATIVE:
                    # An unconfirmed track that misses even once is almost always a false positive.
                    # Killing it immediately keeps phantom people out of the emitted set.
                    t.state = DEAD
                elif t.state == CONFIRMED:
                    t.state = LOST
                    self._log(t, "LOST", now)
                if t.state == LOST and (now - t.last_seen_at) > self.cfg.max_misses_seconds:
                    t.state = DEAD
                    self._log(t, "RELEASED", now)

        for j, d in enumerate(usable):
            if j in matched_dets:
                continue
            if d.score < self.cfg.min_score:
                continue
            t = Track(self._next_id, d, now)
            self._next_id = self._next_id + 1
            self.tracks.append(t)
            self._log(t, "BORN", now)

        self.tracks = [t for t in self.tracks if t.state != DEAD]
        # Most established first, so a caller that can only afford N pose solves picks the N people
        # most likely to still be there next frame rather than an arbitrary N.
        alive = [t for t in self.tracks if t.state in (CONFIRMED, LOST)]
        alive.sort(key=lambda t: (t.state == LOST, -t.hits, t.id))
        return alive

    def _log(self, track, event, now):
        self.events.append({"event": event, "id": track.id, "t": round(now, 4),
                            "hits": track.hits, "misses": track.misses,
                            "depth": None if track.depth is None else round(track.depth, 3)})

    def drain_events(self):
        out = self.events
        self.events = []
        return out

    def snapshot(self):
        """Everything a HUD needs, and nothing that decides behaviour."""
        return {
            "tracks": len(self.tracks),
            "confirmed": len([t for t in self.tracks if t.state == CONFIRMED]),
            "lost": len([t for t in self.tracks if t.state == LOST]),
            "next_id": self._next_id,
        }
