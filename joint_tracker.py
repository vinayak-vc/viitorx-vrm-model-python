#!/usr/bin/env python3
"""P1-1 — PER-JOINT TEMPORAL TRACKING + PLAUSIBILITY.

Solves the problem the live human acceptance run proved (P0_ACCEPTANCE §4d, Finding A):
a limb can sit at a WRONG position while RTMW3D confidence stays high (~0.63), so a
confidence gate never fires. Confidence alone is not validity.

    RAW MEASUREMENT -> JointTracker -> TRACKED / WEAK / PREDICTED / LOST -> STABLE JOINT

ONE reusable tracker for every joint; wrist/elbow/knee/ankle differ only by config.
This sits UPSTREAM of the P0 LimbGate and does not replace it:

    P1 JointTracker -> stable joint -> PoseFrame -> P0 LimbGate -> Kalidokit -> VRM

Design notes that matter:

* CAUSAL ONLY. The brief's "compare N-2..N+2" would need lookahead, and lookahead is
  latency. Instead the A->B->X->B (spike) vs A->B->C->D->E (real motion) distinction is
  made *causally*: a suspicious sample is not rejected, it is DOWN-WEIGHTED into WEAK.
  If following samples form a coherent trajectory with it, the tracker promotes back to
  TRACKED within a frame or two (real fast motion survives). If the next sample snaps
  back to the predicted track, the outlier was damped and never propagated. This gets
  the same discrimination with zero added latency.
* ADAPTIVE TOLERANCE. Residual tolerance is a running robust scale (MAD-like) of that
  joint's own prediction residuals, so a fast-moving wrist earns a wider gate than a
  still hip. Fixed metre thresholds would either clip dancing or pass spikes.
* NO ALLOCATION in update(): scalars only, fixed-size ring buffers preallocated.
"""

import math

# ---------------------------------------------------------------- states


class TrackingState:
    TRACKED = 0     # healthy measurement, used directly
    WEAK = 1        # low confidence or questionable plausibility -> reduced influence
    PREDICTED = 2   # measurement unusable; extrapolating from velocity, bounded
    LOST = 3        # prediction window exhausted -> explicitly INVALID (never zero-filled)

    NAMES = {0: "TRACKED", 1: "WEAK", 2: "PREDICTED", 3: "LOST"}

    @staticmethod
    def name(s):
        return TrackingState.NAMES.get(s, "?")


class TrackerConfig:
    """All tunables in one place. Same object can be shared by every joint, or cloned
    per joint class (a wrist tolerates more speed than a hip)."""

    def __init__(self, **kw):
        # --- confidence bands ---
        self.conf_invalid = 0.01     # at/below: no measurement at all
        self.conf_min = 0.30         # matches the sidecar emit gate + Unity LimbGate
        self.conf_weak = 0.45        # below this the sample is down-weighted

        # --- kinematic limits (human joint, generous: dancing must pass) ---
        self.max_speed = 8.0         # m/s
        self.max_accel = 120.0       # m/s^2

        # --- adaptive residual gate ---
        self.resid_floor = 0.05      # m, tolerance ceiling when the joint is dead still
        self.resid_k = 4.0           # multiples of the joint's own robust residual scale
        self.resid_hard = 0.45       # m, beyond this a sample is suspicious regardless

        # --- prediction ---
        self.max_predict_frames = 6  # brief: 2-6 frames
        self.max_predict_dist = 0.30 # m, total displacement prediction may invent
        self.predict_decay = 0.80    # per-frame velocity decay while predicting
        self.use_accel_in_predict = True
        self.max_accel_predict = 20.0

        # --- reacquisition ---
        self.reacquire_frames = 5    # blend predicted -> measured over N frames

        # --- depth consistency ---
        self.depth_z_tol = 0.20      # m, |z - predicted z| beyond this is suspicious

        # --- neighbour (segment length) plausibility ---
        self.segment_tol = 0.40      # fractional deviation from the joint's own median segment length
        self.segment_max_score = 0.5 # CAP: the neighbour check is corroborating evidence, never a
                                     # veto. The brief says an inconsistent elbow should "reduce
                                     # confidence", not be rejected outright -- and measured on real
                                     # motion an uncapped segment term caused 3170 false rejections
                                     # (real depth noise moves apparent bone length a lot).

        # --- STUCK / frozen detection (the signature actually observed live:
        #     P0_ACCEPTANCE Finding A -- a hidden wrist froze at a WRONG position for
        #     840 frames while confidence stayed ~0.63. Nothing else catches this,
        #     because a frozen joint has tiny residual, speed and acceleration.) ---
        self.frozen_eps = 0.004      # m, per-frame motion below this counts as "not moving"
        self.frozen_min_frames = 12  # consecutive still frames before we get suspicious
        self.frozen_parent_move = 0.05  # m, parent must have moved at least this meanwhile

        # --- coherence promotion (fast-motion rescue) ---
        self.coherence_frames = 2    # consecutive coherent suspicious samples -> accept
        self.coherence_tol = 0.12    # m, agreement between consecutive suspicious samples

        # --- OUTPUT STEP LIMIT -------------------------------------------------
        # Invariant: P1's own output must never move much further in one frame than the
        # raw measurement did. WEAK deliberately under-follows a questionable sample,
        # which builds an offset; paying that offset back in a single frame produced a
        # catch-up spike that made P1's max displacement WORSE than P0's (measured:
        # L-wrist +62%). Bounding the catch-up keeps reacquisition smooth AND guarantees
        # P1 cannot introduce a peak that P0 did not have.
        self.out_step_factor = 1.2   # multiples of the measurement's own step (measured: 1.5 still let a
                                     # WEAK catch-up step 1.8x the real motion through)
        self.out_step_floor = 0.010  # m, so a still joint can still creep back

        # --- timing ---
        self.default_dt = 1.0 / 30.0
        self.max_dt = 0.5            # clamp pathological timestamp gaps

        self.history = 5
        for k, v in kw.items():
            if not hasattr(self, k):
                raise KeyError("unknown TrackerConfig field: %s" % k)
            setattr(self, k, v)


class JointTracker:
    """Temporal state + plausibility for ONE joint. Reusable for every joint type."""

    __slots__ = (
        "cfg", "name",
        "x", "y", "z",                      # current output (stable) position
        "fx", "fy", "fz",                   # filtered position (same as output; kept explicit)
        "vx", "vy", "vz",                   # velocity  m/s
        "ax", "ay", "az",                   # acceleration m/s^2
        "confidence",
        "lvx", "lvy", "lvz",                # lastValidPosition
        "lvvx", "lvvy", "lvvz",             # lastValidVelocity
        "last_measurement_t", "last_valid_t",
        "invalid_frames", "predicted_frames",
        "state", "has_valid",
        "_resid_scale", "_seg_med", "_seg_n",
        "_susp_px", "_susp_py", "_susp_pz", "_susp_run",
        "_frozen_frames", "_frozen_par_x", "_frozen_par_y", "_frozen_par_z", "frozen_count",
        "_mx", "_my", "_mz", "_has_meas",
        "_reacq_left", "_reacq_total", "_reacq_err0",
        "transitions", "suspicious_count", "predicted_count", "lost_count",
        "reacquire_count", "last_reason", "last_suspicion",
    )

    def __init__(self, name="joint", cfg=None):
        self.cfg = cfg or TrackerConfig()
        self.name = name
        self.reset()

    # ------------------------------------------------------------------ life-cycle
    def reset(self):
        self.x = self.y = self.z = 0.0
        self.fx = self.fy = self.fz = 0.0
        self.vx = self.vy = self.vz = 0.0
        self.ax = self.ay = self.az = 0.0
        self.confidence = 0.0
        self.lvx = self.lvy = self.lvz = 0.0
        self.lvvx = self.lvvy = self.lvvz = 0.0
        self.last_measurement_t = None
        self.last_valid_t = None
        self.invalid_frames = 0
        self.predicted_frames = 0
        self.state = TrackingState.LOST
        self.has_valid = False
        self._resid_scale = 0.0
        self._seg_med = 0.0
        self._seg_n = 0
        self._susp_px = self._susp_py = self._susp_pz = 0.0
        self._susp_run = 0
        self._frozen_frames = 0
        self._mx = self._my = self._mz = 0.0
        self._has_meas = False
        self._frozen_par_x = self._frozen_par_y = self._frozen_par_z = 0.0
        self.frozen_count = 0
        self._reacq_left = 0
        self._reacq_total = 0
        self._reacq_err0 = 0.0
        self.transitions = 0
        self.suspicious_count = 0
        self.predicted_count = 0
        self.lost_count = 0
        self.reacquire_count = 0
        self.last_reason = ""
        self.last_suspicion = 0.0

    # ------------------------------------------------------------------ helpers
    def _set_state(self, s):
        if s != self.state:
            self.state = s
            self.transitions += 1
            return True
        return False

    def position(self):
        return (self.x, self.y, self.z)

    def velocity(self):
        return (self.vx, self.vy, self.vz)

    def is_usable(self):
        """True when the tracker has something trustworthy to emit. LOST is explicitly
        NOT usable -- the caller must mark the joint invalid rather than emit zeros."""
        return self.state != TrackingState.LOST and self.has_valid

    # ------------------------------------------------------------------ update
    def update(self, px, py, pz, conf, t, depth_valid=True, parent=None, raw_disp=None):
        """One measurement. Returns (state, usable).

        px,py,pz : measured position (metres). conf<=conf_invalid or measured==False
                   means "no measurement this frame".
        t        : timestamp in SECONDS (real timestamps preferred over frame counting).
        depth_valid : whether depth was actually measured for this joint this frame.
        parent   : (x,y,z) of the connected parent joint (shoulder for elbow, elbow for
                   wrist, hip for knee, knee for ankle) or None. Used for a lightweight
                   segment-length plausibility check -- NOT an IK solve.
        raw_disp : the pre-rate-limit raw displacement from the sidecar, when available;
                   an extra hint that the source sample was a spike.
        """
        cfg = self.cfg

        # ---- dt from real timestamps, defended against irregular/absent stamps ----
        if t is None or self.last_measurement_t is None:
            dt = cfg.default_dt
        else:
            dt = t - self.last_measurement_t
            if dt <= 0.0 or dt > cfg.max_dt:
                dt = cfg.default_dt
        if t is not None:
            self.last_measurement_t = t

        have_meas = (conf is not None and conf > cfg.conf_invalid
                     and not (px == 0.0 and py == 0.0 and pz == 0.0))

        # ---- cold start ------------------------------------------------------
        if not self.has_valid:
            if have_meas:
                self._accept(px, py, pz, conf, t, first=True)
                self._set_state(TrackingState.TRACKED)
                return self.state, True
            self._set_state(TrackingState.LOST)
            return self.state, False

        # ---- predict where the joint should be -------------------------------
        prx = self.lvx + self.vx * dt
        pry = self.lvy + self.vy * dt
        prz = self.lvz + self.vz * dt
        if cfg.use_accel_in_predict:
            k = 0.5 * dt * dt
            prx += _clamp(self.ax, -cfg.max_accel_predict, cfg.max_accel_predict) * k
            pry += _clamp(self.ay, -cfg.max_accel_predict, cfg.max_accel_predict) * k
            prz += _clamp(self.az, -cfg.max_accel_predict, cfg.max_accel_predict) * k

        if not have_meas:
            self.invalid_frames += 1
            self.last_reason = "no_measurement"
            self.last_suspicion = 1.0
            return self._predict_or_lose(prx, pry, prz, dt)

        # ---- GAP RECOVERY ----------------------------------------------------
        # While PREDICTED/LOST we have no information about where the joint really went,
        # so the tight residual gate (tuned for continuous tracking) must NOT judge the
        # returning sample -- it would reject every reacquisition and force the coherence
        # path, which is a hard jump. Sanity-check the returning sample against what
        # max_speed allows over the elapsed gap, then blend into it.
        if self.state in (TrackingState.PREDICTED, TrackingState.LOST) and conf >= cfg.conf_min:
            gap = cfg.default_dt * max(1, self.predicted_frames + self.invalid_frames)
            if self.last_valid_t is not None and t is not None and t > self.last_valid_t:
                gap = min(cfg.max_dt * 4.0, t - self.last_valid_t)
            step = _dist(px, py, pz, self.lvx, self.lvy, self.lvz)
            if step / max(1e-6, gap) <= cfg.max_speed:
                self._reacq_left = max(1, cfg.reacquire_frames)
                self._reacq_total = self._reacq_left
                self._reacq_err0 = _dist(px, py, pz, self.x, self.y, self.z)
                self.reacquire_count += 1
                self.invalid_frames = 0
                self._susp_run = 0
                self.last_reason = "reacquire(err=%.3fm)" % self._reacq_err0
                self.last_suspicion = 0.0
                return self._blend_reacquire(px, py, pz, conf, t, dt)
            self.invalid_frames += 1
            self.last_reason = "reacquire_rejected_speed(%.1fm/s)" % (step / max(1e-6, gap))
            self.last_suspicion = 1.0
            return self._predict_or_lose(prx, pry, prz, dt)

        # ---- plausibility ----------------------------------------------------
        susp, reason = self._suspicion(px, py, pz, conf, dt, prx, pry, prz,
                                       depth_valid, parent, raw_disp)
        self.last_suspicion = susp
        self.last_reason = reason

        if susp >= 1.0:
            # Implausible. Do NOT accept. Predict instead -- but remember the rejected
            # sample so a genuine fast move can be recognised on the next frame.
            self.suspicious_count += 1
            if self._susp_run > 0 and _dist(px, py, pz,
                                            self._susp_px, self._susp_py, self._susp_pz) <= cfg.coherence_tol:
                self._susp_run += 1
            else:
                self._susp_run = 1
            self._susp_px, self._susp_py, self._susp_pz = px, py, pz
            if self._susp_run >= cfg.coherence_frames:
                # Consecutive "implausible" samples that AGREE with each other are not a
                # spike -- they are real motion the model under-predicted. Accept and
                # re-seed velocity from them (this is what keeps dancing responsive).
                self._susp_run = 0
                # Preserve motion: re-seed velocity from the agreeing samples rather than
                # zeroing it (a hard _accept would stall the joint mid-swing).
                self.vx = (px - self.lvx) / max(1e-6, dt)
                self.vy = (py - self.lvy) / max(1e-6, dt)
                self.vz = (pz - self.lvz) / max(1e-6, dt)
                self.x = self.fx = px
                self.y = self.fy = py
                self.z = self.fz = pz
                self.lvx, self.lvy, self.lvz = px, py, pz
                self.confidence = conf
                self.last_valid_t = t
                self.has_valid = True
                self.invalid_frames = 0
                self.predicted_frames = 0
                self._resid_scale = max(self._resid_scale, _dist(px, py, pz, prx, pry, prz))
                self._set_state(TrackingState.TRACKED)
                self.last_reason = "coherent_fast_motion"
                return self.state, True
            self.invalid_frames += 1
            return self._predict_or_lose(prx, pry, prz, dt)

        # plausible enough to use, possibly with reduced influence
        self._susp_run = 0
        self.invalid_frames = 0

        weak = (conf < cfg.conf_weak) or (susp > 0.0)
        if self._reacq_left > 0:
            return self._blend_reacquire(px, py, pz, conf, t, dt)

        if weak:
            w = 1.0 - min(0.75, susp)              # down-weight, never fully discard
            nx = self.x + (px - self.x) * w
            ny = self.y + (py - self.y) * w
            nz = self.z + (pz - self.z) * w
            ms = self._meas_step(px, py, pz)
            self._commit(nx, ny, nz, conf, t, dt, seed_from=(nx, ny, nz), meas_step=ms)
            self._set_state(TrackingState.WEAK)
            return self.state, True

        ms = self._meas_step(px, py, pz)
        self._commit(px, py, pz, conf, t, dt, seed_from=(px, py, pz), meas_step=ms)
        self._set_state(TrackingState.TRACKED)
        return self.state, True

    # ------------------------------------------------------------------ internals
    def _suspicion(self, px, py, pz, conf, dt, prx, pry, prz, depth_valid, parent, raw_disp):
        """0.0 = fully plausible, >=1.0 = reject. Between = down-weight (WEAK)."""
        cfg = self.cfg
        score = 0.0
        reason = ""

        if conf < cfg.conf_min:
            score += 0.5
            reason = "low_conf"

        # 1) residual against the kinematic prediction, scaled by this joint's own history
        resid = _dist(px, py, pz, prx, pry, prz)
        tol = cfg.resid_floor + cfg.resid_k * self._resid_scale
        if resid > cfg.resid_hard:
            self._push_resid(resid)
            return 1.0, "residual_hard(%.2fm)" % resid
        if resid > tol:
            over = (resid - tol) / max(1e-6, tol)
            score += min(1.0, over)
            reason = reason or "residual(%.3f>%.3f)" % (resid, tol)

        # 2) implied speed
        step = _dist(px, py, pz, self.lvx, self.lvy, self.lvz)
        speed = step / max(1e-6, dt)
        if speed > cfg.max_speed:
            self._push_resid(resid)
            return 1.0, "speed(%.1fm/s)" % speed

        # 3) implied acceleration (validation signal only -- not required to be exact)
        nvx = (px - self.lvx) / max(1e-6, dt)
        nvy = (py - self.lvy) / max(1e-6, dt)
        nvz = (pz - self.lvz) / max(1e-6, dt)
        acc = _dist(nvx, nvy, nvz, self.vx, self.vy, self.vz) / max(1e-6, dt)
        if acc > cfg.max_accel:
            score += 0.6
            reason = reason or "accel(%.0fm/s2)" % acc

        # 4) depth consistency -- uses the EXISTING depth result, does not redesign it.
        #    RGB says the joint moved a lot in Z but depth was not actually measured, or
        #    the measured Z contradicts the prediction -> suspicious.
        if not depth_valid:
            score += 0.35
            reason = reason or "depth_unmeasured"
        elif abs(pz - prz) > cfg.depth_z_tol:
            score += 0.4
            reason = reason or "depth_z(%.2fm)" % abs(pz - prz)

        # 5) neighbour / segment-length plausibility (lightweight, not IK)
        if parent is not None:
            seg = _dist(px, py, pz, parent[0], parent[1], parent[2])
            if self._seg_n >= 8 and self._seg_med > 1e-4:
                dev = abs(seg - self._seg_med) / self._seg_med
                if dev > cfg.segment_tol:
                    score += min(cfg.segment_max_score,
                                 (dev - cfg.segment_tol) / cfg.segment_tol)
                    reason = reason or "segment(%.0f%%)" % (dev * 100.0)

        # 6) STUCK detection. A joint pinned at one position while its parent keeps
        #    moving is the confident-but-wrong signature from the live run. Residual,
        #    speed and acceleration are all ~0 for a frozen joint, so only this catches it.
        step_now = _dist(px, py, pz, self.lvx, self.lvy, self.lvz)
        if step_now < cfg.frozen_eps:
            if self._frozen_frames == 0 and parent is not None:
                self._frozen_par_x, self._frozen_par_y, self._frozen_par_z = parent
            self._frozen_frames += 1
            if self._frozen_frames >= cfg.frozen_min_frames and parent is not None:
                par_moved = _dist(parent[0], parent[1], parent[2],
                                  self._frozen_par_x, self._frozen_par_y, self._frozen_par_z)
                if par_moved >= cfg.frozen_parent_move:
                    self.frozen_count += 1
                    score += 0.6
                    reason = reason or ("frozen(%df,parent%+.2fm)"
                                        % (self._frozen_frames, par_moved))
        else:
            self._frozen_frames = 0

        # 7) the sidecar already flagged this sample as a rate-limited spike
        if raw_disp is not None and raw_disp > cfg.resid_hard:
            score += 0.5
            reason = reason or "raw_spike(%.2fm)" % raw_disp

        self._push_resid(resid)
        return (min(1.0, score) if score < 1.0 else 1.0), (reason or "ok")

    def _push_resid(self, r):
        # robust-ish running scale: slow up, slower down. No allocation, no sort.
        if self._resid_scale <= 0.0:
            self._resid_scale = r
        elif r > self._resid_scale:
            self._resid_scale += (r - self._resid_scale) * 0.20
        else:
            self._resid_scale += (r - self._resid_scale) * 0.05

    def _push_segment(self, seg):
        if self._seg_n == 0:
            self._seg_med = seg
        else:
            self._seg_med += (seg - self._seg_med) * 0.05
        if self._seg_n < 1000:
            self._seg_n += 1

    def observe_segment(self, px, py, pz, parent):
        """Feed a KNOWN-GOOD sample's segment length into the running median."""
        if parent is not None:
            self._push_segment(_dist(px, py, pz, parent[0], parent[1], parent[2]))

    def _accept(self, px, py, pz, conf, t, first=False):
        self.x = self.fx = px
        self.y = self.fy = py
        self.z = self.fz = pz
        self.confidence = conf
        self.lvx, self.lvy, self.lvz = px, py, pz
        if first:
            self.vx = self.vy = self.vz = 0.0
            self.ax = self.ay = self.az = 0.0
        self.lvvx, self.lvvy, self.lvvz = self.vx, self.vy, self.vz
        self.last_valid_t = t
        self.has_valid = True
        self.invalid_frames = 0
        self.predicted_frames = 0
        self._reacq_left = 0

    def _commit(self, nx, ny, nz, conf, t, dt, seed_from, meas_step=None):
        """Write the accepted output and update velocity/acceleration from it.

        The output is rate-limited relative to the MEASUREMENT's own step so catching up
        an accumulated WEAK/reacquire offset can never out-run the real motion."""
        sx, sy, sz = seed_from
        if meas_step is not None:
            allowed = self.cfg.out_step_factor * meas_step + self.cfg.out_step_floor
            step = _dist(nx, ny, nz, self.x, self.y, self.z)
            if step > allowed and step > 1e-9:
                f = allowed / step
                nx = self.x + (nx - self.x) * f
                ny = self.y + (ny - self.y) * f
                nz = self.z + (nz - self.z) * f
        nvx = (sx - self.lvx) / max(1e-6, dt)
        nvy = (sy - self.lvy) / max(1e-6, dt)
        nvz = (sz - self.lvz) / max(1e-6, dt)
        # velocity smoothing: light, deliberately NOT another big low-pass (would add lag)
        b = 0.5
        ovx, ovy, ovz = self.vx, self.vy, self.vz
        self.vx = ovx + (nvx - ovx) * b
        self.vy = ovy + (nvy - ovy) * b
        self.vz = ovz + (nvz - ovz) * b
        self.ax = (self.vx - ovx) / max(1e-6, dt)
        self.ay = (self.vy - ovy) / max(1e-6, dt)
        self.az = (self.vz - ovz) / max(1e-6, dt)
        self.x = self.fx = nx
        self.y = self.fy = ny
        self.z = self.fz = nz
        self.confidence = conf
        self.lvx, self.lvy, self.lvz = sx, sy, sz
        self.lvvx, self.lvvy, self.lvvz = self.vx, self.vy, self.vz
        self.last_valid_t = t
        self.predicted_frames = 0

    def _meas_step(self, px, py, pz):
        """Distance between THIS raw measurement and the PREVIOUS raw measurement --
        i.e. how far the sensor says the joint actually moved. This is the budget the
        output step limiter is allowed to spend; using lastValid instead would include
        the WEAK offset and the limiter would never bind."""
        if not self._has_meas:
            self._mx, self._my, self._mz = px, py, pz
            self._has_meas = True
            return 0.0
        d = _dist(px, py, pz, self._mx, self._my, self._mz)
        self._mx, self._my, self._mz = px, py, pz
        return d

    def _blend_reacquire(self, px, py, pz, conf, t, dt):
        """Move a fraction of the way toward the measurement each frame so a recovered
        joint eases in instead of teleporting. Progress is 1/N, 2/N ... so the blend
        completes in exactly _reacq_total frames."""
        done = self._reacq_total - self._reacq_left + 1
        a = float(done) / float(self._reacq_total)
        nx = self.x + (px - self.x) * a
        ny = self.y + (py - self.y) * a
        nz = self.z + (pz - self.z) * a
        self._reacq_left -= 1
        # no step limit during reacquisition: the blend IS the limiter, and clamping it
        # again would stall recovery after a long gap.
        self._commit(nx, ny, nz, conf, t, dt, seed_from=(px, py, pz))
        self._set_state(TrackingState.TRACKED if self._reacq_left <= 0 else TrackingState.WEAK)
        return self.state, True

    def _predict_or_lose(self, prx, pry, prz, dt):
        cfg = self.cfg
        if self.predicted_frames >= cfg.max_predict_frames:
            self._set_state(TrackingState.LOST)
            self.lost_count += 1
            return self.state, False
        travelled = _dist(prx, pry, prz, self.lvx, self.lvy, self.lvz)
        if travelled > cfg.max_predict_dist:
            self._set_state(TrackingState.LOST)
            self.lost_count += 1
            return self.state, False
        self.predicted_frames += 1
        self.predicted_count += 1
        # decay velocity so a long gap coasts to a stop instead of flying off
        self.vx *= cfg.predict_decay
        self.vy *= cfg.predict_decay
        self.vz *= cfg.predict_decay
        self.x = self.fx = prx
        self.y = self.fy = pry
        self.z = self.fz = prz
        self.confidence *= cfg.predict_decay
        self._set_state(TrackingState.PREDICTED)
        return self.state, True


def _dist(ax, ay, az, bx, by, bz):
    dx = ax - bx
    dy = ay - by
    dz = az - bz
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


# ------------------------------------------------------------------ skeleton

# WholeBody index -> connected PARENT index, for segment-length plausibility.
# shoulder->elbow->wrist and hip->knee->ankle, exactly the chains the brief names.
WB_PARENT = {
    7: 5, 8: 6,        # elbow  <- shoulder
    9: 7, 10: 8,       # wrist  <- elbow
    13: 11, 14: 12,    # knee   <- hip
    15: 13, 16: 14,    # ankle  <- knee
}

WB_NAMES = {
    5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle",
}

DEFAULT_TRACKED = sorted(WB_NAMES.keys())


class SkeletonTracker:
    """Holds one JointTracker per tracked index and applies them in parent-before-child
    order so the neighbour check always sees an already-stabilised parent."""

    def __init__(self, indices=None, cfg=None):
        self.cfg = cfg or TrackerConfig()
        self.indices = list(indices if indices is not None else DEFAULT_TRACKED)
        self.trackers = {}
        for i in self.indices:
            self.trackers[i] = JointTracker(WB_NAMES.get(i, "j%d" % i), self.cfg)
        # parents first, so a child sees the stabilised parent this same frame
        self._order = sorted(self.indices, key=lambda i: (i in WB_PARENT, i))
        self.events = []          # state transitions + suspicious samples (diagnostics)
        self.frame = 0

    def reset(self):
        for t in self.trackers.values():
            t.reset()
        self.events = []
        self.frame = 0

    def update(self, positions, confidences, t, depth_valid=None, raw_disp=None,
               collect_events=True):
        """positions: {idx: (x,y,z)}, confidences: {idx: conf}. Returns
        {idx: (x, y, z, conf, state, usable)} for every tracked index."""
        out = {}
        self.frame += 1
        for i in self._order:
            tr = self.trackers[i]
            p = positions.get(i)
            c = confidences.get(i, 0.0)
            dv = True if depth_valid is None else bool(depth_valid.get(i, True))
            rd = None if raw_disp is None else raw_disp.get(i)
            par = None
            pi = WB_PARENT.get(i)
            if pi is not None and pi in self.trackers and self.trackers[pi].has_valid:
                ptr = self.trackers[pi]
                par = (ptr.x, ptr.y, ptr.z)
            prev_state = tr.state
            if p is None:
                st, usable = tr.update(0.0, 0.0, 0.0, 0.0, t, dv, par, rd)
            else:
                st, usable = tr.update(p[0], p[1], p[2], c, t, dv, par, rd)
                if st in (TrackingState.TRACKED, TrackingState.WEAK) and par is not None:
                    # also learn during WEAK: freezing the median while a joint is noisy
                    # lets it drift away from the body's real proportions.
                    tr.observe_segment(tr.x, tr.y, tr.z, par)
            out[i] = (tr.x, tr.y, tr.z, tr.confidence, st, usable)
            if collect_events:
                if st != prev_state:
                    self.events.append({
                        "frame": self.frame, "t": t, "joint": tr.name,
                        "event": "%s->%s" % (TrackingState.name(prev_state),
                                             TrackingState.name(st)),
                        "confidence": round(c, 3),
                        "reason": tr.last_reason,
                    })
                elif tr.last_suspicion >= 1.0:
                    self.events.append({
                        "frame": self.frame, "t": t, "joint": tr.name,
                        "event": "SUSPICIOUS", "confidence": round(c, 3),
                        "position": None if p is None else [round(v, 4) for v in p],
                        "velocity": [round(tr.vx, 3), round(tr.vy, 3), round(tr.vz, 3)],
                        "reason": tr.last_reason,
                    })
        return out

    def counters(self):
        c = {}
        for i, tr in self.trackers.items():
            c[tr.name] = dict(suspicious=tr.suspicious_count,
                              predicted=tr.predicted_count,
                              lost=tr.lost_count,
                              reacquired=tr.reacquire_count,
                              transitions=tr.transitions,
                              state=TrackingState.name(tr.state))
        return c
