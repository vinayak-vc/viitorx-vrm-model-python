#!/usr/bin/env python3
"""
Temporal smoothing for the whole-body sidecar (jitter fix).

RTMW3D keypoints + measured stereo depth are noisy frame-to-frame (depth especially quantizes into
coarse steps at a distance), so the avatar shakes even when the user is still. Unity's One-Euro is too
light for this and, being velocity-adaptive, actually lets the large depth spikes through. This module
smooths the metric camera-space keypoints at the SOURCE with a per-coordinate One-Euro filter plus a
3D outlier gate that rejects sudden depth jumps (holding the last good value).
"""

import math


class _LowPass:
    def __init__(self):
        self.y = None

    def filter(self, x, alpha):
        if self.y is None:
            self.y = x
        else:
            self.y = alpha * x + (1.0 - alpha) * self.y
        return self.y

    def reset(self):
        self.y = None


class OneEuro:
    """Standard 1€ filter for a single scalar signal (Casiez et al.)."""

    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = _LowPass()
        self._dx = _LowPass()
        self._last = None

    def _alpha(self, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x):
        if self._last is None:
            dx = 0.0
        else:
            dx = (x - self._last) * self.freq
        edx = self._dx.filter(dx, self._alpha(self.d_cutoff))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        y = self._x.filter(x, self._alpha(cutoff))
        self._last = x
        return y

    def reset(self):
        self._x.reset()
        self._dx.reset()
        self._last = None


class KeypointSmoother:
    """Per-keypoint One-Euro (x,y,z) with a 3D outlier gate, for N metric keypoints.

    Call filter(index, x, y, z, valid) each frame. When valid is False the keypoint's filters reset so
    a re-acquired point does not lerp from a stale pose. When the raw point jumps more than max_jump
    metres from the last smoothed point it is treated as an outlier and the last smoothed value is held.
    """

    # LOW-D: default max_jump aligned to the sender's --max-jump default (1.5). The old 0.5 froze
    # keypoints whenever the coarse depth-quantization step at distance exceeded it every frame.
    def __init__(self, count, freq=30.0, min_cutoff=1.0, beta=0.02, max_jump=1.5,
                 depth_min_cutoff=None, depth_beta=None, limb_indices=None, max_hold=0):
        self.max_jump = max_jump
        self.max_hold = max_hold
        self.limb_indices = set(limb_indices) if limb_indices else set()
        dmc = depth_min_cutoff if depth_min_cutoff is not None else min_cutoff
        dbeta = depth_beta if depth_beta is not None else beta
        self._fx = [OneEuro(freq, min_cutoff, beta) for _ in range(count)]
        self._fy = [OneEuro(freq, min_cutoff, beta) for _ in range(count)]
        # LIMB depth (z) gets a HEAVIER (lower-cutoff) One-Euro: measured live, limb depth jitters ~5x the
        # image plane for small/distant hands, so damp z harder on arms+hands. Other joints keep the normal
        # z filter (shoulders/hips have their z flattened at build anyway).
        self._fz = [OneEuro(freq,
                            dmc if i in self.limb_indices else min_cutoff,
                            dbeta if i in self.limb_indices else beta) for i in range(count)]
        self._last = [None] * count
        self._hold = [0] * count

    def filter(self, index, x, y, z, valid):
        # Returns (x, y, z, effective_valid). effective_valid can be True even when the raw point was
        # invalid, when a LIMB dropout is being held (below) — the caller should then treat it as measured.
        if not valid:
            # HOLD-ON-DROPOUT (limbs only): a limb keypoint with no measured depth would otherwise fall
            # back to the model's zrel back-projection (garbage — 8-12 m spikes). Instead hold the last
            # good smoothed value for up to max_hold frames, reported valid so build uses it, NOT the
            # fallback. Bounded, so a genuinely-gone limb still drops (never a permanent freeze — ADR-018).
            prev = self._last[index]
            if index in self.limb_indices and self.max_hold > 0 and prev is not None and self._hold[index] < self.max_hold:
                self._hold[index] = self._hold[index] + 1
                return prev[0], prev[1], prev[2], True
            self._fx[index].reset()
            self._fy[index].reset()
            self._fz[index].reset()
            self._last[index] = None
            self._hold[index] = 0
            return x, y, z, False
        self._hold[index] = 0
        prev = self._last[index]
        if prev is not None and self.max_jump > 0.0:
            dx = x - prev[0]
            dy = y - prev[1]
            dz = z - prev[2]
            d2 = dx * dx + dy * dy + dz * dz
            if d2 > (self.max_jump * self.max_jump):
                # Big step (depth spike OR the coarse depth-quantization step at distance): RATE-LIMIT it
                # (move at most max_jump toward the target) instead of holding — holding froze keypoints
                # permanently when the quantization step exceeded max_jump every frame.
                t = self.max_jump / math.sqrt(d2)
                x = prev[0] + dx * t
                y = prev[1] + dy * t
                z = prev[2] + dz * t
        sx = self._fx[index].filter(x)
        sy = self._fy[index].filter(y)
        sz = self._fz[index].filter(z)
        self._last[index] = (sx, sy, sz)
        return sx, sy, sz, True
