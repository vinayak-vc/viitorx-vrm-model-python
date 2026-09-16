#!/usr/bin/env python3
"""F-22 - Human Pose Validation / Biomechanical Validation Layer.

Audited before writing a line of this (docs/F22_HUMAN_POSE_VALIDATION_2026-09-14.md SS3): nothing in
the live pipeline checks whether a tracking measurement is a physically plausible human pose. P0
LimbGate (Runtime/Retargeting/LimbGate.cs) is confidence-only. P1-1 (joint_tracker.py) is per-joint
Cartesian plausibility (residual/velocity/acceleration + a lightweight parent-distance check) - zero
angle math anywhere. Arm V2 (Runtime/Retargeting/ArmAimSolver.cs) already computes the elbow bend
angle every frame (`BendDeg`, L289) but only as a diagnostic - never used to gate anything. F-19 found
this gap live: hands-near-face elbows rendered at 167.8/169.4 deg median, 177.5/179.6 deg max - LimbGate
held 0.00% of that block, because the CONFIDENCE was high even though the ANGLE was anatomically
impossible.

This module is the fix, scoped to exactly F-19's evidence (elbows + knees): pure, clock-injected, no
I/O (same shape as joint_tracker.py/target_ownership.py), fed from RAW camera-space xyz_cam between
P1-1's skel.update() and build_body_landmarks() in wholebody_udp_sender.py. It does NOT duplicate
P1-1's Cartesian checks - its whole job is the ANGULAR dimension P1-1 doesn't touch: is this bend
angle anatomically plausible, and did it get there at a physically plausible rate.

A rejected/held joint zeros ONLY that joint's own conf_emit slot - reusing P1-1's OWN existing
contract for signalling an invalid joint ("LOST -> drop -> P0 LimbGate holds", wholebody_udp_sender.py
L825) - which Unity's P0 LimbGate already correctly consumes. No Runtime/ change, no UDP change.

Bend-angle convention (matches Arm V2's ArmAimSolver.BendDeg exactly, so a threshold derived from
F-19's rendered evidence applies to this raw pre-retargeting signal unchanged): angle between
(joint - proximal) and (distal - joint), both vectors pointing "forward" along the chain. A straight
limb reads ~0 deg; a fully folded one approaches 180 deg. F-19's own words: "a human elbow reaches
~150 deg" in this exact convention.
"""
import math

VALID = "VALID"
HELD = "HELD"
REJECTED = "REJECTED"

# Explicit reason taxonomy (F-22 brief SS21: "do not collapse everything into LOW_CONFIDENCE").
TRACK_LOST = "TRACK_LOST"
NONFINITE = "NONFINITE"
INVALID_CONFIDENCE = "INVALID_CONFIDENCE"
ANGLE_LIMIT = "ANGLE_LIMIT"
ANGLE_RATE_IMPOSSIBLE = "ANGLE_RATE_IMPOSSIBLE"


def _finite3(p):
    return all(math.isfinite(v) for v in p)


def _bend_deg(proximal, joint, distal):
    """None if a segment is degenerate (near-zero length) - caller treats that as NONFINITE."""
    ux, uy, uz = joint[0] - proximal[0], joint[1] - proximal[1], joint[2] - proximal[2]
    lx, ly, lz = distal[0] - joint[0], distal[1] - joint[1], distal[2] - joint[2]
    nu = math.sqrt(ux * ux + uy * uy + uz * uz)
    nl = math.sqrt(lx * lx + ly * ly + lz * lz)
    if nu < 1e-6 or nl < 1e-6:
        return None
    cos_a = (ux * lx + uy * ly + uz * lz) / (nu * nl)
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.degrees(math.acos(cos_a))


class ChainConfig(object):
    """One config per chain family (elbow/knee). Every default is cited in the module docstring and
    the report - none is a visual guess (F-22 brief SS4/SS22)."""
    def __init__(self, warn_deg, reject_deg, rate_suspicious_deg_s, rate_impossible_deg_s,
                min_confidence=0.3, hold_max_frames=8):
        self.warn_deg = warn_deg
        self.reject_deg = reject_deg
        self.rate_suspicious_deg_s = rate_suspicious_deg_s
        self.rate_impossible_deg_s = rate_impossible_deg_s
        self.min_confidence = min_confidence
        self.hold_max_frames = hold_max_frames


# F-19 SS14's own recommendation, adopted directly (not re-derived): "hard flexion clamp ~150 deg".
ELBOW_CONFIG = ChainConfig(warn_deg=150.0, reject_deg=160.0,
                           rate_suspicious_deg_s=900.0, rate_impossible_deg_s=1800.0)
# F-19 measured a LEGITIMATE max of 176 deg during ordinary walking (rare, 30/85k frames) - REJECT
# must sit above that observed legitimate value or it would false-reject real walking.
KNEE_CONFIG = ChainConfig(warn_deg=165.0, reject_deg=178.0,
                          rate_suspicious_deg_s=900.0, rate_impossible_deg_s=1800.0)


class ChainValidator(object):
    """One instance per (side, joint) - e.g. left elbow. update() takes this frame's three raw
    camera-space positions plus validity/confidence for the MIDDLE (hinge) joint, and returns
    (state, reason, bend_deg, rate_deg_s). No side effects on caller data - the caller decides what
    to do with conf_emit."""
    def __init__(self, name, cfg):
        self.name = name
        self.cfg = cfg
        self.state = VALID
        self.last_valid_bend = None
        self.last_valid_t = None
        self.hold_streak = 0
        self.events = []

    def _emit(self, kind, **fields):
        rec = dict(event=kind, chain=self.name)
        rec.update(fields)
        self.events.append(rec)

    def update(self, proximal, joint, distal, joint_conf, measured3, t):
        """measured3: (proximal_ok, joint_ok, distal_ok) bools."""
        cfg = self.cfg
        bend = None
        rate = None
        warn = None   # set to a WARN-band reason when the frame is still ACCEPTED but flagged -
                       # SS7's "normal / warning / reject range": WARN never blocks emission on its
                       # own, it is diagnostic-only, logged so an operator/report can see a joint
                       # trending toward REJECT before it actually gets there.

        if not (measured3[0] and measured3[1] and measured3[2]):
            ok, reason = False, TRACK_LOST
        elif not (_finite3(proximal) and _finite3(joint) and _finite3(distal)):
            ok, reason = False, NONFINITE
        elif joint_conf < cfg.min_confidence:
            ok, reason = False, INVALID_CONFIDENCE
        else:
            bend = _bend_deg(proximal, joint, distal)
            if bend is None:
                ok, reason = False, NONFINITE
            elif bend > cfg.reject_deg:
                ok, reason = False, ANGLE_LIMIT
            else:
                ok, reason = True, None
                if bend > cfg.warn_deg:
                    warn = ANGLE_LIMIT
                if self.last_valid_bend is not None and self.last_valid_t is not None:
                    dt = t - self.last_valid_t
                    if dt > 1e-6:
                        rate = abs(bend - self.last_valid_bend) / dt
                        if rate > cfg.rate_impossible_deg_s:
                            ok, reason = False, ANGLE_RATE_IMPOSSIBLE
                        elif rate > cfg.rate_suspicious_deg_s:
                            warn = ANGLE_RATE_IMPOSSIBLE

        if ok:
            if self.state != VALID:
                self._emit("CHAIN_RECOVERED", bend=round(bend, 2))
            elif warn is not None:
                self._emit("CHAIN_WARN", reason=warn, bend=round(bend, 2),
                           rate=(round(rate, 1) if rate is not None else None))
            self.state = VALID
            self.hold_streak = 0
            self.last_valid_bend = bend
            self.last_valid_t = t
            return self.state, None, bend, rate

        self.hold_streak += 1
        prev_state = self.state
        self.state = HELD if self.hold_streak <= cfg.hold_max_frames else REJECTED
        if self.state != prev_state or self.hold_streak == 1:
            self._emit("CHAIN_%s" % self.state, reason=reason,
                       bend=(round(bend, 2) if bend is not None else None),
                       rate=(round(rate, 1) if rate is not None else None),
                       hold_streak=self.hold_streak)
        return self.state, reason, bend, rate

    def drain_events(self):
        ev = self.events
        self.events = []
        return ev


# WholeBody/COCO indices - identical mapping used throughout wholebody_udp_sender.py.
CHAIN_DEFS = {
    "left_elbow":  (5, 7, 9, ELBOW_CONFIG),    # L shoulder -> L elbow -> L wrist
    "right_elbow": (6, 8, 10, ELBOW_CONFIG),   # R shoulder -> R elbow -> R wrist
    "left_knee":   (11, 13, 15, KNEE_CONFIG),  # L hip -> L knee -> L ankle
    "right_knee":  (12, 14, 16, KNEE_CONFIG),  # R hip -> R knee -> R ankle
}


class PoseValidator(object):
    """Owns one ChainValidator per chain in CHAIN_DEFS. update() reads xyz_cam/measured/conf directly
    (RAW, pre-smoothing - see module docstring) and returns {joint_idx: (state, reason, bend, rate)}
    for the MIDDLE joint of every chain. The caller zeros conf_emit for any non-VALID joint index -
    this module never mutates caller arrays."""
    def __init__(self, chain_defs=None):
        self.chains = dict((name, ChainValidator(name, cfg))
                           for name, (_, _, _, cfg) in (chain_defs or CHAIN_DEFS).items())
        self._defs = chain_defs or CHAIN_DEFS

    def update(self, xyz_cam, measured, conf, t):
        out = {}
        for name, (p_idx, j_idx, d_idx, _cfg) in self._defs.items():
            proximal = (float(xyz_cam[p_idx, 0]), float(xyz_cam[p_idx, 1]), float(xyz_cam[p_idx, 2]))
            joint = (float(xyz_cam[j_idx, 0]), float(xyz_cam[j_idx, 1]), float(xyz_cam[j_idx, 2]))
            distal = (float(xyz_cam[d_idx, 0]), float(xyz_cam[d_idx, 1]), float(xyz_cam[d_idx, 2]))
            measured3 = (bool(measured[p_idx]), bool(measured[j_idx]), bool(measured[d_idx]))
            state, reason, bend, rate = self.chains[name].update(
                proximal, joint, distal, float(conf[j_idx]), measured3, t)
            out[j_idx] = (state, reason, bend, rate)
        return out

    def drain_events(self):
        ev = []
        for chain in self.chains.values():
            ev.extend(chain.drain_events())
        return ev
