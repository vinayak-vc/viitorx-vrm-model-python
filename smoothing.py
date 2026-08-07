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

    def __init__(self, count, freq=30.0, min_cutoff=1.0, beta=0.02, max_jump=0.5):
        self.max_jump = max_jump
        self._fx = [OneEuro(freq, min_cutoff, beta) for _ in range(count)]
        self._fy = [OneEuro(freq, min_cutoff, beta) for _ in range(count)]
        self._fz = [OneEuro(freq, min_cutoff, beta) for _ in range(count)]
        self._last = [None] * count

    def filter(self, index, x, y, z, valid):
        if not valid:
            self._fx[index].reset()
            self._fy[index].reset()
            self._fz[index].reset()
            self._last[index] = None
            return x, y, z
        prev = self._last[index]
        if prev is not None and self.max_jump > 0.0:
            dx = x - prev[0]
            dy = y - prev[1]
            dz = z - prev[2]
            d2 = dx * dx + dy * dy + dz * dz
            if d2 > (self.max_jump * self.max_jump):
                # Big step (depth spike OR the coarse depth-quantization step at distance): RATE-LIMIT
                # it — move at most max_jump toward the target — instead of holding. Holding froze
                # keypoints permanently when the quantization step exceeded max_jump every frame.
                t = self.max_jump / math.sqrt(d2)
                x = prev[0] + dx * t
                y = prev[1] + dy * t
                z = prev[2] + dz * t
        sx = self._fx[index].filter(x)
        sy = self._fy[index].filter(y)
        sz = self._fz[index].filter(z)
        self._last[index] = (sx, sy, sz)
        return sx, sy, sz
