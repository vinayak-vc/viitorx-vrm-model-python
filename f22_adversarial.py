#!/usr/bin/env python3
"""F-22 offline completion - deterministic adversarial perturbation suite on REAL video frames.

The gap this closes, stated plainly: test_pose_validation.py exercises the validator in ISOLATION on
synthetic limbs, and f22_video_replay.py exercises real footage that never actually violates a
threshold. Neither one drives the thing that matters - the PRODUCTION INSERTION POINT and what
reaches the consumer when a chain is suppressed. This harness does, end to end, for every case in
the brief's own list.

Each case proves all four stages, in order, and fails if any one of them disagrees:

    1. RAW OBSERVATION   the bend angle actually present in the perturbed skeleton (measured back
                         out of the geometry, never assumed from the value that was requested)
    2. F-22              pose_validation.PoseValidator - the SAME class wholebody_udp_sender.py
                         constructs, unmodified, no replay-only variant
    3. conf_emit         the five-line insertion block from wholebody_udp_sender.py L880-884,
                         reproduced exactly: `if _state != PV.VALID: conf_emit[_j] = 0.0`
    4. DOWNSTREAM        wholebody_udp_sender.build_body_landmarks - the REAL function, imported
                         from the real module - asserting the joint the consumer receives is
                         [0,0,0,0] with src=0, which is what makes Unity's P0 LimbGate hold

BASE GEOMETRY IS REAL, NOT INVENTED. Frames come from the actual replay clip through the actual
RTMW3D model. They are lifted into metric camera space the same way production does it - pinhole
backprojection, per-joint depth taken from the model's own root-relative z about a nominal hip
plane - rather than f22_video_replay.py's `zrel * 200` pixel-space proxy, so segment lengths and
bend angles here are in real metres and real degrees. The nominal hip depth (2.0 m) stands in for
the stereo measurement an arbitrary video cannot provide; that substitution is the single synthetic
element in the base pose and is stated rather than buried.

PERTURBATION preserves the real limb: the distal joint is rotated about the hinge, in the limb's own
existing bend plane, keeping the REAL forearm/shin length measured from that frame. Only the angle
changes.

    python f22_adversarial.py --video "D:\\...\\video\\video.webm"
"""
import argparse
import io
import json
import math
import os
import sys

import numpy as np

import rtmw3d_pose as R
import pose_validation as PV
import wholebody_udp_sender as W

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "..", "..", "..", "SentisModel", "rtmw3d-x.onnx")

DT = 1.0 / 30.0
CONF_THR = 0.3          # production --conf default
FLATTEN_TRUNK = False   # production default (Milestone-2)
USE_ZREL = True         # production --zrel-fallback default
NOMINAL_HIP_Z = 2.0     # metres - stands in for the stereo hip depth a plain video cannot provide

# COCO-WholeBody -> the JointId slot the consumer actually reads (rtmw3d_pose.COCO17_TO_JOINTID).
CHAIN_JOINTID = {7: 13, 8: 14, 13: 25, 14: 26}          # L/R elbow, L/R knee
CHAIN_NAME = {7: "left_elbow", 8: "right_elbow", 13: "left_knee", 14: "right_knee"}
CHAIN_TRIPLE = {7: (5, 7, 9), 8: (6, 8, 10), 13: (11, 13, 15), 14: (12, 14, 16)}

_results = []


def check(name, cond, detail=""):
    _results.append((name, bool(cond), detail))
    print("    %s  %s%s" % ("PASS" if cond else "FAIL", name, ("   [%s]" % detail) if detail else ""))


# ---------------------------------------------------------------------------------------------
# real base frames
# ---------------------------------------------------------------------------------------------
def load_base_frame(video, model, want_index):
    """One REAL frame, lifted into metric camera space by the same pinhole backprojection
    production uses. Returns everything build_body_landmarks needs."""
    import cv2
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit("could not open %s" % video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bbox = R.center_bbox(w, h)
    uv = zrel = conf = None
    for i in range(want_index + 1):
        ok, frame = cap.read()
        if not ok:
            break
        uv, zrel, conf = model.infer(frame, bbox)
        ref = R.bbox_from_keypoints(uv, conf, w, h, thr=CONF_THR)
        if ref is not None:
            bbox = tuple(0.7 * np.array(bbox) + 0.3 * np.array(ref))
    cap.release()
    if uv is None:
        raise SystemExit("no frames read from %s" % video)

    # A plausible pinhole for a 1080-class sensor; only the RATIO matters for angles, and it is held
    # fixed across every case so no case can be advantaged by the choice.
    fx = fy = 800.0
    intr = (fx, fy, w / 2.0, h / 2.0)
    zrel_hip = float((zrel[11] + zrel[12]) / 2.0)

    xyz = np.zeros((133, 3), dtype=np.float64)
    for i in range(133):
        z = NOMINAL_HIP_Z + (float(zrel[i]) - zrel_hip)
        xyz[i] = [(float(uv[i, 0]) - intr[2]) * z / fx,
                  (float(uv[i, 1]) - intr[3]) * z / fy, z]
    measured = np.array([bool(conf[i] > CONF_THR) for i in range(133)])
    mid_hip = (xyz[11] + xyz[12]) / 2.0
    return dict(uv=uv, zrel=zrel, conf=np.array(conf, dtype=np.float64), xyz=xyz,
                measured=measured, mid_hip=mid_hip, hip_z=float(mid_hip[2]),
                zrel_hip=zrel_hip, intr=intr, w=w, h=h, index=want_index)


def bend_of(xyz, triple):
    p, j, d = triple
    return PV._bend_deg(tuple(xyz[p]), tuple(xyz[j]), tuple(xyz[d]))


def set_bend(xyz, triple, target_deg):
    """Rotate the distal joint about the hinge to an EXACT target bend, in the limb's own existing
    bend plane, preserving the real distal segment length measured from this frame."""
    p_i, j_i, d_i = triple
    prox, joint, dist = xyz[p_i], xyz[j_i], xyz[d_i]
    u = joint - prox
    nu = np.linalg.norm(u)
    if nu < 1e-9:
        raise ValueError("degenerate proximal segment in the base frame")
    u = u / nu
    f = dist - joint
    L = float(np.linalg.norm(f))
    if L < 1e-9:
        raise ValueError("degenerate distal segment in the base frame")
    perp = f - float(np.dot(f, u)) * u
    if np.linalg.norm(perp) < 1e-9:          # limb dead straight - any perpendicular will do
        seed = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp = seed - float(np.dot(seed, u)) * u
    perp = perp / np.linalg.norm(perp)
    th = math.radians(target_deg)
    out = xyz.copy()
    out[d_i] = joint + L * (math.cos(th) * u + math.sin(th) * perp)
    return out


# ---------------------------------------------------------------------------------------------
# the production insertion point + the production downstream, driven frame by frame
# ---------------------------------------------------------------------------------------------
class Pipeline(object):
    """wholebody_udp_sender.py's F-22 block (L880-884) and its build_body_landmarks call (L892),
    with nothing else between them - so a case that passes here passes in the sender."""

    def __init__(self, base):
        self.base = base
        self.validator = PV.PoseValidator()
        self.t = 0.0
        self.events = []
        self.frame = 0

    def step(self, xyz=None, measured=None, conf=None):
        b = self.base
        xyz = b["xyz"] if xyz is None else xyz
        measured = b["measured"] if measured is None else measured
        conf = b["conf"] if conf is None else conf

        conf_emit = conf.copy()                                   # P1-1's own contract
        pv_out = self.validator.update(xyz, measured, conf_emit, self.t)
        for _j, (_state, _reason, _bend, _rate) in pv_out.items():
            if _state != PV.VALID:
                conf_emit[_j] = 0.0                               # <- the entire F-22 side effect
        for e in self.validator.drain_events():
            e["frame"] = self.frame
            self.events.append(e)

        lm, src = W.build_body_landmarks(
            b["uv"], xyz, measured, conf_emit, b["zrel"], b["zrel_hip"],
            b["mid_hip"], b["hip_z"], b["intr"], CONF_THR, FLATTEN_TRUNK, USE_ZREL)

        self.t += DT
        self.frame += 1
        return dict(pv=pv_out, conf_emit=conf_emit, lm=lm, src=src)


def prove_chain(tag, out, coco_j, expect_valid, expect_reason=None, expect_bend=None):
    """The four-stage proof, asserted as one unit so a case cannot pass on a partial agreement."""
    jid = CHAIN_JOINTID[coco_j]
    state, reason, bend, _rate = out["pv"][coco_j]
    ce = float(out["conf_emit"][coco_j])
    vis = out["lm"][jid][3]
    src = out["src"][jid]

    if expect_bend is not None:
        check("%s stage1 raw observation is %.1f deg as constructed" % (tag, expect_bend),
              bend is not None and abs(bend - expect_bend) < 0.15,
              "measured %s" % (("%.2f" % bend) if bend is not None else "None"))
    if expect_valid:
        check("%s stage2 F-22 verdict VALID" % tag, state == PV.VALID, "state=%s" % state)
        check("%s stage3 conf_emit preserved" % tag, ce > 0.0, "conf_emit=%.3f" % ce)
        check("%s stage4 consumer receives the joint" % tag, vis > 0.0 and src == 1,
              "vis=%.3f src=%d" % (vis, src))
    else:
        check("%s stage2 F-22 suppresses (HELD or REJECTED)" % tag,
              state in (PV.HELD, PV.REJECTED), "state=%s reason=%s" % (state, reason))
        if expect_reason is not None:
            check("%s stage2 reason is %s" % (tag, expect_reason), reason == expect_reason,
                  "reason=%s" % reason)
        check("%s stage3 conf_emit zeroed" % tag, ce == 0.0, "conf_emit=%.3f" % ce)
        check("%s stage4 consumer receives [0,0,0,0] src=0 (P0 LimbGate holds)" % tag,
              out["lm"][jid] == [0.0, 0.0, 0.0, 0.0] and src == 0,
              "lm=%s src=%d" % (out["lm"][jid], src))
    return dict(state=state, reason=reason, bend=bend, conf_emit=ce, vis=vis, src=src)


# ---------------------------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------------------------
def ramp_to(pipe, base, coco_j, target, start=None, step_deg=20.0, settle=1):
    """Walk the bend from `start` (default: this frame's REAL angle) to the target at a rate the
    ANGULAR-RATE check cannot object to (20 deg/frame = 600 deg/s, below the 900 deg/s WARN band),
    so an absolute-angle case is never accidentally decided by the rate check instead.

    `start` must be where the limb ACTUALLY is, not where the base frame was - ramping down from a
    held 155 deg by restarting at the base pose would teleport the joint 125 deg in one frame, and
    the rate check would (correctly) reject it, which would measure the harness rather than F-22."""
    triple = CHAIN_TRIPLE[coco_j]
    if start is None:
        start = bend_of(base["xyz"], triple)
    n = max(1, int(math.ceil(abs(target - start) / step_deg)))
    out = None
    for k in range(1, n + 1):
        ang = start + (target - start) * (float(k) / n)
        out = pipe.step(xyz=set_bend(base["xyz"], triple, ang))
    for _ in range(settle):
        out = pipe.step(xyz=set_bend(base["xyz"], triple, target))
    return out


def case_angle(base, coco_j, target, expect_valid, records):
    cfg = PV.CHAIN_DEFS[CHAIN_NAME[coco_j]][3]
    tag = "%s@%.0fdeg" % (CHAIN_NAME[coco_j], target)
    print("\n-- %s  (WARN %.0f / REJECT %.0f)" % (tag, cfg.warn_deg, cfg.reject_deg))
    pipe = Pipeline(base)
    pipe.step()                                             # one real, unperturbed baseline frame
    out = ramp_to(pipe, base, coco_j, target)
    rec = prove_chain(tag, out, coco_j, expect_valid,
                      expect_reason=(None if expect_valid else PV.ANGLE_LIMIT),
                      expect_bend=target)

    # recovery on the FIRST valid frame - required for every case by the brief.
    back = ramp_to(pipe, base, coco_j, 30.0, start=target)
    st2 = back["pv"][coco_j][0]
    check("%s recovery: VALID again on the first good frame" % tag, st2 == PV.VALID, "state=%s" % st2)
    check("%s recovery: consumer receives the joint again" % tag,
          back["lm"][CHAIN_JOINTID[coco_j]][3] > 0.0)
    rec["recovered"] = st2 == PV.VALID
    records[tag] = rec
    return rec


def case_single_frame_spike(base, records):
    coco_j, tag = 7, "single_frame_angle_spike"
    print("\n-- %s" % tag)
    pipe = Pipeline(base)
    triple = CHAIN_TRIPLE[coco_j]
    for _ in range(5):
        pipe.step(xyz=set_bend(base["xyz"], triple, 40.0))
    spike = pipe.step(xyz=set_bend(base["xyz"], triple, 175.0))
    rec = prove_chain(tag, spike, coco_j, False, expect_reason=PV.ANGLE_LIMIT, expect_bend=175.0)
    check("%s a one-frame spike is HELD, never escalated to REJECTED" % tag,
          spike["pv"][coco_j][0] == PV.HELD, "state=%s" % spike["pv"][coco_j][0])
    nxt = pipe.step(xyz=set_bend(base["xyz"], triple, 40.0))
    check("%s recovery is immediate on the very next frame" % tag,
          nxt["pv"][coco_j][0] == PV.VALID and nxt["lm"][CHAIN_JOINTID[coco_j]][3] > 0.0,
          "state=%s" % nxt["pv"][coco_j][0])
    records[tag] = rec
    return rec


def case_multi_frame_violation(base, records):
    coco_j, tag = 7, "multi_frame_angle_violation"
    print("\n-- %s  (hold_max_frames=%d)" % (tag, PV.ELBOW_CONFIG.hold_max_frames))
    pipe = Pipeline(base)
    triple = CHAIN_TRIPLE[coco_j]
    for _ in range(5):
        pipe.step(xyz=set_bend(base["xyz"], triple, 40.0))
    states = []
    for _ in range(12):
        o = pipe.step(xyz=set_bend(base["xyz"], triple, 175.0))
        states.append(o["pv"][coco_j][0])
    rec = prove_chain(tag, o, coco_j, False, expect_reason=PV.ANGLE_LIMIT, expect_bend=175.0)
    held = states.count(PV.HELD)
    check("%s HELD for exactly hold_max_frames then REJECTED" % tag,
          held == PV.ELBOW_CONFIG.hold_max_frames and states[held:] == [PV.REJECTED] * (12 - held),
          "held=%d then %s" % (held, set(states[held:])))
    check("%s conf_emit stays zeroed for the whole violation" % tag,
          float(o["conf_emit"][coco_j]) == 0.0)
    nxt = pipe.step(xyz=set_bend(base["xyz"], triple, 40.0))
    check("%s recovery from REJECTED is still immediate on one good frame" % tag,
          nxt["pv"][coco_j][0] == PV.VALID, "state=%s" % nxt["pv"][coco_j][0])
    rec["held_frames"] = held
    records[tag] = rec
    return rec


def case_rate_spike(base, records):
    """A jump that stays INSIDE the legal absolute band but gets there impossibly fast. The absolute
    check is evaluated first in ChainValidator.update, so this case is only meaningful below REJECT -
    it is what isolates the rate check from the angle check."""
    coco_j, tag = 7, "excessive_angular_rate_spike"
    print("\n-- %s  (impossible %.0f deg/s)" % (tag, PV.ELBOW_CONFIG.rate_impossible_deg_s))
    pipe = Pipeline(base)
    triple = CHAIN_TRIPLE[coco_j]
    for _ in range(5):
        pipe.step(xyz=set_bend(base["xyz"], triple, 20.0))
    # 20 -> 155 deg in one 33.3 ms frame = 4050 deg/s, and 155 is still under the 160 REJECT line.
    out = pipe.step(xyz=set_bend(base["xyz"], triple, 155.0))
    rate = out["pv"][coco_j][3]
    check("%s the constructed rate really is impossible" % tag,
          rate is not None and rate > PV.ELBOW_CONFIG.rate_impossible_deg_s,
          "%.0f deg/s" % (rate or -1))
    check("%s the absolute angle alone would NOT have caught it" % tag,
          out["pv"][coco_j][2] <= PV.ELBOW_CONFIG.reject_deg,
          "bend=%.1f <= reject %.0f" % (out["pv"][coco_j][2], PV.ELBOW_CONFIG.reject_deg))
    rec = prove_chain(tag, out, coco_j, False, expect_reason=PV.ANGLE_RATE_IMPOSSIBLE,
                      expect_bend=155.0)
    # Recovery from a RATE rejection is bounded but NOT single-frame, and the mechanism is worth
    # recording because it is not obvious from the code: last_valid_t is only advanced on an ACCEPTED
    # frame, so while the joint is held the measured rate decays as delta/dt with dt growing one
    # frame at a time. It therefore self-heals in ceil(delta / rate_limit / DT) frames with no
    # special-case logic - here 135 deg / 1800 deg/s = 75 ms = 3 frames at 30 fps.
    frames_to_recover = 0
    for _ in range(10):
        nxt = pipe.step(xyz=set_bend(base["xyz"], triple, 155.0))
        frames_to_recover += 1
        if nxt["pv"][coco_j][0] == PV.VALID:
            break
    bound = int(math.ceil(135.0 / PV.ELBOW_CONFIG.rate_impossible_deg_s / DT))
    check("%s holding still at the same angle self-heals (the RATE was the problem, not the pose)"
          % tag, nxt["pv"][coco_j][0] == PV.VALID, "state=%s" % nxt["pv"][coco_j][0])
    check("%s recovery is bounded by the decaying-rate arithmetic (<=%d frames)" % (tag, bound),
          frames_to_recover <= bound, "recovered in %d frames, bound %d" % (frames_to_recover, bound))
    check("%s the consumer gets the joint back on that frame" % tag,
          nxt["lm"][CHAIN_JOINTID[coco_j]][3] > 0.0)
    rec["rate_recovery_frames"] = frames_to_recover
    print("      [measured] rate-rejection self-heal = %d frames (%.0f ms) while the pose is static"
          % (frames_to_recover, frames_to_recover * DT * 1000.0))
    records[tag] = rec
    return rec


def case_missing(base, coco_j, missing_idx, label, records):
    tag = "missing_%s" % label
    print("\n-- %s" % tag)
    pipe = Pipeline(base)
    triple = CHAIN_TRIPLE[coco_j]
    pipe.step(xyz=set_bend(base["xyz"], triple, 40.0))
    m = base["measured"].copy()
    m[missing_idx] = False
    out = pipe.step(xyz=set_bend(base["xyz"], triple, 40.0), measured=m)
    rec = prove_chain(tag, out, coco_j, False, expect_reason=PV.TRACK_LOST)
    nxt = pipe.step(xyz=set_bend(base["xyz"], triple, 40.0))
    check("%s recovery on the first frame the joint is measured again" % tag,
          nxt["pv"][coco_j][0] == PV.VALID, "state=%s" % nxt["pv"][coco_j][0])
    records[tag] = rec
    return rec


def case_invalid_confidence(base, records):
    coco_j, tag = 7, "invalid_confidence"
    print("\n-- %s  (min_confidence=%.1f)" % (tag, PV.ELBOW_CONFIG.min_confidence))
    pipe = Pipeline(base)
    triple = CHAIN_TRIPLE[coco_j]
    pipe.step(xyz=set_bend(base["xyz"], triple, 40.0))
    c = base["conf"].copy()
    c[coco_j] = 0.10
    out = pipe.step(xyz=set_bend(base["xyz"], triple, 40.0), conf=c)
    rec = prove_chain(tag, out, coco_j, False, expect_reason=PV.INVALID_CONFIDENCE)
    check("%s the OTHER arm is untouched by this arm's failure" % tag,
          out["pv"][8][0] == PV.VALID and float(out["conf_emit"][8]) > 0.0,
          "right_elbow=%s" % out["pv"][8][0])
    nxt = pipe.step(xyz=set_bend(base["xyz"], triple, 40.0))
    check("%s recovery when confidence returns" % tag, nxt["pv"][coco_j][0] == PV.VALID)
    records[tag] = rec
    return rec


def case_degenerate_segment(base, records):
    coco_j, tag = 7, "degenerate_segment"
    print("\n-- %s" % tag)
    pipe = Pipeline(base)
    triple = CHAIN_TRIPLE[coco_j]
    pipe.step(xyz=set_bend(base["xyz"], triple, 40.0))
    xyz = base["xyz"].copy()
    xyz[9] = xyz[7]                                   # wrist collapsed onto the elbow
    out = pipe.step(xyz=xyz)
    rec = prove_chain(tag, out, coco_j, False, expect_reason=PV.NONFINITE)
    check("%s a zero-length forearm yields no bend angle at all" % tag,
          out["pv"][coco_j][2] is None, "bend=%s" % out["pv"][coco_j][2])
    nxt = pipe.step(xyz=set_bend(base["xyz"], triple, 40.0))
    check("%s recovery once the segment is real again" % tag, nxt["pv"][coco_j][0] == PV.VALID)
    records[tag] = rec
    return rec


def case_nonfinite(base, records):
    coco_j, tag = 7, "nonfinite_coordinate"
    print("\n-- %s" % tag)
    pipe = Pipeline(base)
    triple = CHAIN_TRIPLE[coco_j]
    pipe.step(xyz=set_bend(base["xyz"], triple, 40.0))
    xyz = base["xyz"].copy()
    xyz[9, 2] = float("nan")
    out = pipe.step(xyz=xyz)
    rec = prove_chain(tag, out, coco_j, False, expect_reason=PV.NONFINITE)
    records[tag] = rec
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(
        "D:", os.sep, "Unity", "viitorx-vrm-avtar-unity-base-project", "Assets", "Games", "video",
        "video.webm"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--frame", type=int, default=120, help="which real frame to derive the base pose from")
    ap.add_argument("--out-dir", default=os.path.join("oak_v4_evidence", "f22", "adversarial"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    print("=" * 100)
    print(" F-22 adversarial perturbation suite - real frames, real insertion point, real downstream")
    print("=" * 100)
    print("[load] RTMW3D ...")
    model = R.RTMW3D(a.model)
    base = load_base_frame(a.video, model, a.frame)
    print("[base] %s frame %d  (%dx%d)" % (os.path.basename(a.video), a.frame, base["w"], base["h"]))
    for j in (7, 8, 13, 14):
        print("       %-12s real bend = %.1f deg   conf=%.2f   measured=%s"
              % (CHAIN_NAME[j], bend_of(base["xyz"], CHAIN_TRIPLE[j]), base["conf"][j],
                 bool(base["measured"][j])))
    # Sanity: the real frame must itself be clean, or every case below is measuring a dirty baseline.
    for j in (7, 8, 13, 14):
        cfg = PV.CHAIN_DEFS[CHAIN_NAME[j]][3]
        check("base frame %s is within its own REJECT limit" % CHAIN_NAME[j],
              bend_of(base["xyz"], CHAIN_TRIPLE[j]) <= cfg.reject_deg,
              "%.1f <= %.0f" % (bend_of(base["xyz"], CHAIN_TRIPLE[j]), cfg.reject_deg))

    records = {}
    # 1-7: elbow sweep across and through the 160 deg REJECT line. Thresholds are NOT adjusted to
    #      make any of these pass - 150/155/160 are expected VALID because that is what the shipped
    #      threshold says, and they are asserted as VALID.
    for deg in (150.0, 155.0, 160.0, 165.0, 170.0, 175.0, 179.0):
        case_angle(base, 7, deg, expect_valid=(deg <= PV.ELBOW_CONFIG.reject_deg), records=records)
    # 8-10: knee sweep. 176 and 178 are expected VALID on purpose - F-19 measured 176 deg as a
    #       LEGITIMATE walking maximum, so rejecting them would be a false positive, not a catch.
    for deg in (176.0, 178.0, 179.0):
        case_angle(base, 13, deg, expect_valid=(deg <= PV.KNEE_CONFIG.reject_deg), records=records)
    # 11-17
    case_single_frame_spike(base, records)
    case_multi_frame_violation(base, records)
    case_rate_spike(base, records)
    case_missing(base, 7, 7, "elbow", records)
    case_missing(base, 7, 9, "wrist", records)
    case_invalid_confidence(base, records)
    case_degenerate_segment(base, records)
    case_nonfinite(base, records)

    n = len(_results)
    p = sum(1 for _, ok, _ in _results if ok)
    print("\n" + "=" * 100)
    print(" CASE SUMMARY   (raw observation -> F-22 -> conf_emit -> downstream)")
    print("=" * 100)
    print("  %-32s %-9s %-24s %-10s %-8s %s" % ("case", "bend", "F-22 verdict", "conf_emit",
                                                "vis", "src"))
    for tag, r in records.items():
        print("  %-32s %-9s %-24s %-10s %-8s %s"
              % (tag, ("%.1f" % r["bend"]) if r["bend"] is not None else "n/a",
                 "%s/%s" % (r["state"], r["reason"] or "-"), "%.3f" % r["conf_emit"],
                 "%.3f" % r["vis"], r["src"]))
    print("-" * 100)
    print(" %d/%d assertions passed across %d adversarial cases" % (p, n, len(records)))
    if p != n:
        print("\nFAILED:")
        for name, ok, detail in _results:
            if not ok:
                print("  %s  %s" % (name, detail))
    print("=" * 100)

    with io.open(os.path.join(a.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(video=a.video, base_frame=a.frame,
                       base_bends=dict((CHAIN_NAME[j], round(bend_of(base["xyz"],
                                                                     CHAIN_TRIPLE[j]), 2))
                                       for j in (7, 8, 13, 14)),
                       thresholds=dict(
                           elbow=dict(warn=PV.ELBOW_CONFIG.warn_deg, reject=PV.ELBOW_CONFIG.reject_deg,
                                      rate_warn=PV.ELBOW_CONFIG.rate_suspicious_deg_s,
                                      rate_reject=PV.ELBOW_CONFIG.rate_impossible_deg_s),
                           knee=dict(warn=PV.KNEE_CONFIG.warn_deg, reject=PV.KNEE_CONFIG.reject_deg)),
                       cases=records, assertions_total=n, assertions_passed=p),
                  f, indent=2, default=str)
    with io.open(os.path.join(a.out_dir, "assertions.txt"), "w", encoding="utf-8") as f:
        for name, ok, detail in _results:
            f.write("%s  %s%s\n" % ("PASS" if ok else "FAIL", name,
                                    ("   [%s]" % detail) if detail else ""))
    print(" evidence -> %s" % a.out_dir)
    return 0 if p == n else 1


if __name__ == "__main__":
    sys.exit(main())
