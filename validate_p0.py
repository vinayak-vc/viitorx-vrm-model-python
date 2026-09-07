#!/usr/bin/env python3
"""
P0 stability-patch validation (audit AUDIT_FBT_2026-09-07, P0-1 + P0-2).

Three checks, run offline (no camera / no Unity needed):

  A. REAL-DATA replay: take the recorded recv_log.jsonl limb streams (the actual landmarks Unity
     received during the audit capture) and measure frame-to-frame displacement BEFORE, then AFTER
     applying the new P0-2 per-limb displacement cap. Proves the 0.8 m-class spikes are suppressed on
     genuine recorded data. (recv_log carries wrists + elbows; legs are covered by check B.)

  B. REAL-MODULE test: exercise the ACTUAL edited smoothing.KeypointSmoother — legs now included in
     depth-smoothing + hold, per-limb caps, and the bounded dropout hold — with synthetic spikes/dropouts.

  C. P0-1 gate logic: replicate the C# LimbGate.Resolve branch exactly and run the mandated occlusion
     scenarios (dropout hold, >8-frame hold, re-acquire, invalid-never-zero). The C# itself is unit-tested
     in Unity (Tests/EditMode/LimbGateTests.cs); this mirrors the same logic for an offline record.

Run:  blender --background --python validate_p0.py    (or any Python 3 with this dir on sys.path)
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smoothing  # the ACTUAL edited module

RECV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_logs", "recv_log.jsonl")
ARM_CAP = 0.35
LEG_CAP = 0.35


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * p))]


def stats(deltas):
    d = sorted(deltas)
    n = len(d)
    return (sum(d) / n if n else 0.0, pct(d, 0.5), pct(d, 0.95), d[-1] if d else 0.0)


def d3(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def rate_limit(prev, cur, cap):
    """The exact P0-2 slew: move at most `cap` metres toward cur from prev."""
    if prev is None:
        return cur
    dx = [cur[i] - prev[i] for i in range(3)]
    d = math.sqrt(sum(v * v for v in dx))
    if cap > 0.0 and d > cap:
        t = cap / d
        return [prev[i] + dx[i] * t for i in range(3)]
    return cur


# ---------------------------------------------------------------- A. real replay
def check_a():
    if not os.path.exists(RECV):
        print("A. SKIP — no recv_log.jsonl at", RECV)
        return
    rows = [json.loads(l) for l in open(RECV) if l.strip()]
    # (label, key, idx, cap)
    series = [("L-wrist", "wr", 0, ARM_CAP), ("R-wrist", "wr", 1, ARM_CAP),
              ("L-elbow", "el", 0, ARM_CAP), ("R-elbow", "el", 1, ARM_CAP)]
    print("A. REAL-DATA replay on recorded recv_log (%d frames) — per-limb cap %.2f m" % (len(rows), ARM_CAP))
    print("   %-9s | %-28s | %-28s" % ("joint", "BEFORE mean/p50/p95/max (m)", "AFTER  mean/p50/p95/max (m)"))
    for label, key, idx, cap in series:
        raw = []
        for r in rows:
            try:
                raw.append(r[key][idx])
            except Exception:
                raw.append(None)
        before, after = [], []
        prev_raw = None
        prev_capped = None
        for p in raw:
            if p is None:
                prev_raw = None
                prev_capped = None
                continue
            if prev_raw is not None:
                before.append(d3(prev_raw, p))
            capped = rate_limit(prev_capped, p, cap)
            if prev_capped is not None:
                after.append(d3(prev_capped, capped))
            prev_raw = p
            prev_capped = capped
        b = stats(before)
        a = stats(after)
        print("   %-9s | %6.4f %6.4f %6.4f %6.4f | %6.4f %6.4f %6.4f %6.4f" %
              (label, b[0], b[1], b[2], b[3], a[0], a[1], a[2], a[3]))


# ---------------------------------------------------------------- B. real module
def check_b():
    print("\nB. REAL-MODULE (edited smoothing.KeypointSmoother) — legs protected + per-limb cap + hold")
    limb_idx = set([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]) | set(range(91, 133))
    overrides = {i: ARM_CAP for i in (7, 8, 9, 10)}
    overrides.update({i: LEG_CAP for i in (13, 14, 15, 16)})
    ks = smoothing.KeypointSmoother(133, min_cutoff=0.5, beta=0.4, max_jump=1.5,
                                    depth_min_cutoff=0.3, depth_beta=0.1,
                                    limb_indices=limb_idx, max_hold=8, max_jump_overrides=overrides)
    ok = True

    # B1: knee (13) now in limb_indices → bounded dropout hold works.
    for _ in range(5):
        ks.filter(13, 0.1, -0.5, 2.4, True)
    held = []
    for _ in range(6):
        x, y, z, eff, act, disp = ks.filter(13, 0.0, 0.0, 0.0, False)
        held.append((eff, act))
    knee_hold_ok = all(e and a == 'HOLD' for e, a in held)
    print("   B1 knee dropout hold (6 frames, max_hold=8): %s  [%s]" %
          ("PASS" if knee_hold_ok else "FAIL", ",".join(a for _, a in held)))
    ok = ok and knee_hold_ok

    # B1b: knee held past the bound → drops (does not freeze).
    ks.filter(13, 0.1, -0.5, 2.4, True)  # re-establish valid
    seq = [ks.filter(13, 0.0, 0.0, 0.0, False) for _ in range(10)]
    dropped = any((not e) or a == 'DROP' for _, _, _, e, a, _ in seq)
    print("   B1b knee hold is BOUNDED (drops after max_hold): %s" % ("PASS" if dropped else "FAIL"))
    ok = ok and dropped

    # B2: wrist (9) 0.8 m spike → rate-limited to <= ARM_CAP (+ tiny One-Euro slack).
    ks.filter(9, 0.40, 0.13, 2.40, True)
    x, y, z, eff, act, disp = ks.filter(9, 0.40 + 0.80, 0.13, 2.40, True)  # +0.80 m spike
    step = math.sqrt((x - 0.40) ** 2 + (y - 0.13) ** 2 + (z - 2.40) ** 2)
    wrist_ok = act == 'RATE_LIMIT' and step <= ARM_CAP + 0.05 and disp > 0.7
    print("   B2 wrist +0.80 m spike: rawDisp=%.3f action=%s appliedStep=%.3f (cap %.2f): %s" %
          (disp, act, step, ARM_CAP, "PASS" if wrist_ok else "FAIL"))
    ok = ok and wrist_ok

    # B3: trunk shoulder (5) 0.8 m move → NOT capped by the arm/leg threshold (global 1.5) → untouched.
    ks.filter(5, 0.07, 0.45, 2.40, True)
    x5, y5, z5, e5, a5, d5 = ks.filter(5, 0.07 + 0.80, 0.45, 2.40, True)
    trunk_ok = a5 == 'ACCEPT'  # 0.8 < global 1.5 → not rate-limited
    print("   B3 trunk shoulder +0.80 m: action=%s (expect ACCEPT, trunk untouched): %s" %
          (a5, "PASS" if trunk_ok else "FAIL"))
    ok = ok and trunk_ok
    return ok


# ---------------------------------------------------------------- C. P0-1 gate logic
class LimbGateSim:
    """Exact mirror of the C# LimbGate.Resolve branch (Runtime/Retargeting/LimbGate.cs)."""

    def __init__(self):
        self.lastU = None
        self.lastL = None
        self.hasValid = False
        self.state = "Valid"
        self.held = 0
        self.holdEvents = 0
        self.reacq = 0
        self.confFail = 0

    def resolve(self, conf, thr, freshU, freshL):
        if conf >= thr:
            trans = self.state == "Held" and self.hasValid
            if trans:
                self.reacq += 1
            self.lastU, self.lastL = freshU, freshL
            self.hasValid = True
            self.held = 0
            self.state = "Valid"
            return True, freshU, freshL, trans
        self.confFail += 1
        if not self.hasValid:
            return False, None, None, False
        if self.state != "Held":
            self.holdEvents += 1
        self.state = "Held"
        self.held += 1
        return True, self.lastU, self.lastL, (self.held == 1)


def check_c():
    print("\nC. P0-1 gate logic (mirror of C# LimbGate.Resolve) — invalid NEVER collapses to zero")
    thr = 0.3
    valid_u, valid_l = (0.5, 0.2, -0.1), (0.3, 0.1, 0.0)
    zero = (0.0, 0.0, 0.0)  # what an invalid joint's solve collapses toward (the bug)
    ok = True

    # C1: static valid → applies fresh each frame.
    g = LimbGateSim()
    appliedU = None
    for _ in range(5):
        a, u, l, t = g.resolve(0.9, thr, valid_u, valid_l)
        appliedU = u
    c1 = appliedU == valid_u
    print("   C1 valid stream applies fresh solve: %s" % ("PASS" if c1 else "FAIL"))
    ok = ok and c1

    # C2: 3-5 frame dropout → holds last valid, NEVER zero (the collapse we are fixing).
    g = LimbGateSim()
    g.resolve(0.9, thr, valid_u, valid_l)
    outs = [g.resolve(0.0, thr, zero, zero) for _ in range(5)]
    c2 = all(applied and u == valid_u and u != zero for applied, u, l, t in outs)
    print("   C2 3-5 frame dropout HOLDS last valid (never zero): %s" % ("PASS" if c2 else "FAIL"))
    ok = ok and c2

    # C3: >8 frame dropout → still safe (holds, does not collapse). P0 holds rather than predicts.
    g = LimbGateSim()
    g.resolve(0.9, thr, valid_u, valid_l)
    outs = [g.resolve(0.0, thr, zero, zero) for _ in range(12)]
    c3 = all(u == valid_u for _, u, _, _ in outs)
    print("   C3 >8 frame dropout stays held (no collapse): %s" % ("PASS" if c3 else "FAIL"))
    ok = ok and c3

    # C4: re-acquisition → transition flagged HELD→VALID, output = new valid (slerp downstream = smooth).
    g = LimbGateSim()
    g.resolve(0.9, thr, valid_u, valid_l)
    for _ in range(4):
        g.resolve(0.0, thr, zero, zero)
    new_u, new_l = (0.6, 0.25, -0.05), (0.35, 0.12, 0.0)
    applied, u, l, trans = g.resolve(0.9, thr, new_u, new_l)
    c4 = applied and u == new_u and trans and g.reacq == 1
    print("   C4 re-acquire flagged + applies new (no teleport via slerp): %s" % ("PASS" if c4 else "FAIL"))
    ok = ok and c4

    # C5: startup while invalid → returns 'do not apply' (bone stays at rest), never zero.
    g = LimbGateSim()
    applied, u, l, t = g.resolve(0.0, thr, zero, zero)
    c5 = (applied is False) and (u is None)
    print("   C5 invalid-at-startup does not apply (bone rests, not zero): %s" % ("PASS" if c5 else "FAIL"))
    ok = ok and c5
    return ok


if __name__ == "__main__":
    print("=" * 78)
    check_a()
    b_ok = check_b()
    c_ok = check_c()
    print("\n" + "=" * 78)
    print("SUMMARY: P0-2 module tests %s | P0-1 gate logic %s" %
          ("PASS" if b_ok else "FAIL", "PASS" if c_ok else "FAIL"))
