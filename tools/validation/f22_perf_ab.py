#!/usr/bin/env python3

import evidence_paths as EV

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(_os.path.join(_d, "_sidecar_path.py")):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import _sidecar_path  # noqa: F401  - puts the sidecar root and every tools/ group on sys.path
"""F-22 performance A/B - pose validation OFF vs ON on identical replay input.

The F-22 report's own SS18 says performance "was not independently re-measured this session". This
closes that gap, and it does so in two separate measurements rather than one, because a single
end-to-end number would be actively misleading here:

  MEASUREMENT 1 - ISOLATED INSERTION-POINT COST (the number that actually answers the question).
    Real per-frame geometry is extracted ONCE and cached, so both arms replay byte-identical input
    with no model inference in the loop. Only the sender's F-22 block and its downstream
    build_body_landmarks call are timed. This is the true added cost, resolvable in microseconds.

  MEASUREMENT 2 - END-TO-END REPLAY THROUGHPUT (reported, with its limitation stated).
    A full replay pass with inference in the loop. RTMW3D dominates this by three orders of
    magnitude, so the two arms are expected to be indistinguishable - and an "indistinguishable"
    result here is NOT evidence that F-22 is free, only that it is far below the noise floor of a
    measurement dominated by something else. Measurement 1 is what carries the claim.

FRAME AGE and CAPTURE-TO-SEND LATENCY are deliberately NOT reported: both are properties of the live
OAK-D capture path (a real sensor timestamp and a real socket send), and neither exists in a video
replay. What measurement 1 DOES bound is how much F-22 can possibly add to capture-to-send latency
once that path is live, which is the part a replay can honestly speak to.

    python tools/validation/f22_perf_ab.py --video "../../video/video.webm"
"""
import argparse
import io
import json
import os
import sys
import time

import cv2
import numpy as np

import rtmw3d_pose as R
import pose_validation as PV
import target_ownership as TO
import wholebody_udp_sender as W

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "..", "..", "..", "SentisModel", "rtmw3d-x.onnx")
NOMINAL_HIP_Z = 2.0
CONF_THR = 0.3
FLATTEN_TRUNK = False
USE_ZREL = True
DT = 1.0 / 30.0


def extract(model, path, limit=None):
    """Run the model once and cache every frame's real geometry, so the A/B arms differ ONLY in
    whether F-22 runs."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit("could not open %s" % path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fx = fy = 800.0
    intr = (fx, fy, w / 2.0, h / 2.0)
    bbox = R.center_bbox(w, h)
    out = []
    while True:
        ok, frame = cap.read()
        if not ok or (limit and len(out) >= limit):
            break
        uv, zrel, conf = model.infer(frame, bbox)
        ref = R.bbox_from_keypoints(uv, conf, w, h, thr=CONF_THR)
        if ref is not None:
            bbox = tuple(0.7 * np.array(bbox) + 0.3 * np.array(ref))
        zrel_hip = float((zrel[11] + zrel[12]) / 2.0)
        xyz = np.zeros((133, 3), dtype=np.float64)
        for i in range(133):
            z = NOMINAL_HIP_Z + (float(zrel[i]) - zrel_hip)
            xyz[i] = [(float(uv[i, 0]) - intr[2]) * z / fx,
                      (float(uv[i, 1]) - intr[3]) * z / fy, z]
        measured = np.array([bool(conf[i] > CONF_THR) for i in range(133)])
        if not (measured[11] and measured[12]):
            continue
        mid_hip = (xyz[11] + xyz[12]) / 2.0
        out.append(dict(uv=uv, zrel=zrel, conf=np.array(conf, dtype=np.float64), xyz=xyz,
                        measured=measured, mid_hip=mid_hip, hip_z=float(mid_hip[2]),
                        zrel_hip=zrel_hip, intr=intr))
    cap.release()
    return out


def arm(frames, validation_on, repeats):
    """The sender's tail, exactly as wholebody_udp_sender.py runs it: conf_emit, the F-22 block
    (L880-884) when enabled, then build_body_landmarks (L892)."""
    validator = PV.PoseValidator() if validation_on else None
    per_frame_us = []
    pv_only_us = []
    suppressed = 0
    t = 0.0
    for _ in range(repeats):
        for f in frames:
            t0 = time.perf_counter()
            conf_emit = f["conf"].copy()
            if validator is not None:
                t_pv0 = time.perf_counter()
                pv_out = validator.update(f["xyz"], f["measured"], conf_emit, t)
                for _j, (_state, _r, _b, _rt) in pv_out.items():
                    if _state != PV.VALID:
                        conf_emit[_j] = 0.0
                        suppressed += 1
                pv_only_us.append((time.perf_counter() - t_pv0) * 1e6)
                validator.drain_events()
            W.build_body_landmarks(f["uv"], f["xyz"], f["measured"], conf_emit, f["zrel"],
                                   f["zrel_hip"], f["mid_hip"], f["hip_z"], f["intr"],
                                   CONF_THR, FLATTEN_TRUNK, USE_ZREL)
            per_frame_us.append((time.perf_counter() - t0) * 1e6)
            t += DT
    return per_frame_us, pv_only_us, suppressed


def stats(v):
    if not v:
        return dict(n=0)
    a = np.array(v)
    return dict(n=len(v), mean_us=round(float(a.mean()), 2), p50_us=round(float(np.percentile(a, 50)), 2),
                p95_us=round(float(np.percentile(a, 95)), 2),
                p99_us=round(float(np.percentile(a, 99)), 2), max_us=round(float(a.max()), 2))


def throughput(model, path, validation_on):
    """Full replay with inference in the loop - the measurement RTMW3D dominates."""
    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fx = fy = 800.0
    intr = (fx, fy, w / 2.0, h / 2.0)
    diag = (w * w + h * h) ** 0.5
    own = TO.TargetOwnership(TO.OwnershipConfig(min_confidence=CONF_THR,
                                                switch_margin_m=diag * 0.064,
                                                scale_margin_ratio=10.0))
    validator = PV.PoseValidator() if validation_on else None
    bbox = R.center_bbox(w, h)
    n = 0
    owned = 0
    suppressed = 0
    t = 0.0
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        uv, zrel, conf = model.infer(frame, bbox)
        ref = R.bbox_from_keypoints(uv, conf, w, h, thr=CONF_THR)
        body_conf_mean = float(np.mean(conf[0:17]))
        zrel_hip = float((zrel[11] + zrel[12]) / 2.0)
        xyz = np.zeros((133, 3), dtype=np.float64)
        for i in range(133):
            z = NOMINAL_HIP_Z + (float(zrel[i]) - zrel_hip)
            xyz[i] = [(float(uv[i, 0]) - intr[2]) * z / fx,
                      (float(uv[i, 1]) - intr[3]) * z / fy, z]
        measured = np.array([bool(conf[i] > CONF_THR) for i in range(133)])
        hip = tuple(((xyz[11] + xyz[12]) / 2.0).tolist()) if (measured[11] and measured[12]) else None
        state, emit = own.update(TO.Observation(hip is not None, hip, body_conf_mean, None), t)
        own.drain_events()
        if (ref is not None and body_conf_mean >= CONF_THR
                and state in (TO.ACQUIRING, TO.LOCKED, TO.REACQUIRING)):
            bbox = tuple(0.7 * np.array(bbox) + 0.3 * np.array(ref))
        elif state in (TO.NO_TARGET, TO.RELEASED):
            bbox = R.center_bbox(w, h)
        if emit and hip is not None:
            owned += 1
            conf_emit = np.array(conf, dtype=np.float64)
            if validator is not None:
                pv_out = validator.update(xyz, measured, conf_emit, t)
                for _j, (_s, _r, _b, _rt) in pv_out.items():
                    if _s != PV.VALID:
                        conf_emit[_j] = 0.0
                        suppressed += 1
                validator.drain_events()
            mid_hip = (xyz[11] + xyz[12]) / 2.0
            W.build_body_landmarks(uv, xyz, measured, conf_emit, zrel, zrel_hip, mid_hip,
                                   float(mid_hip[2]), intr, CONF_THR, FLATTEN_TRUNK, USE_ZREL)
        n += 1
        t += DT
    cap.release()
    el = time.perf_counter() - t0
    return dict(frames=n, owned_frames=owned, suppressed_joint_frames=suppressed,
                elapsed_s=round(el, 3), fps=round(n / el, 2),
                ms_per_frame=round(el * 1000.0 / max(1, n), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(
        "D:", os.sep, "Unity", "viitorx-vrm-avtar-unity-base-project", "Assets", "Games", "video",
        "video.webm"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--repeats", type=int, default=40)
    ap.add_argument("--out-dir", default=EV.oak_v4("f22", "perf_ab"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    print("=" * 100)
    print(" F-22 performance A/B  -  pose validation OFF vs ON, identical replay input")
    print("=" * 100)
    print("[load] RTMW3D ...")
    model = R.RTMW3D(a.model)
    print("[cache] extracting real per-frame geometry from %s ..." % os.path.basename(a.video))
    frames = extract(model, a.video)
    print("[cache] %d usable frames cached" % len(frames))

    # warm both paths so neither arm pays first-call import/JIT costs the other avoids
    arm(frames[:20], False, 1)
    arm(frames[:20], True, 1)

    print("\n-- MEASUREMENT 1: isolated insertion-point cost (%d frames x %d repeats)"
          % (len(frames), a.repeats))
    off_us, _, _ = arm(frames, False, a.repeats)
    on_us, pv_us, suppressed = arm(frames, True, a.repeats)
    s_off, s_on, s_pv = stats(off_us), stats(on_us), stats(pv_us)
    delta_mean = s_on["mean_us"] - s_off["mean_us"]

    print("  %-34s %12s %12s %12s" % ("", "OFF", "ON", "delta"))
    for k in ("n", "mean_us", "p50_us", "p95_us", "p99_us", "max_us"):
        d = (round(s_on[k] - s_off[k], 2) if k != "n" else "")
        print("  %-34s %12s %12s %12s" % (k, s_off[k], s_on[k], d))
    print("  %-34s %12s" % ("F-22 block alone, mean_us", s_pv["mean_us"]))
    print("  %-34s %12s" % ("F-22 block alone, p95_us", s_pv["p95_us"]))
    print("  %-34s %12s" % ("F-22 block alone, max_us", s_pv["max_us"]))
    budget_pct = 100.0 * (delta_mean / 1e6) / DT
    print("  added cost vs a 33.3 ms frame budget: %.4f %% (%.1f us of 33333 us)"
          % (budget_pct, delta_mean))
    print("  suppressed joint-frames in the ON arm: %d of %d chain evaluations"
          % (suppressed, len(frames) * a.repeats * 4))

    print("\n-- MEASUREMENT 2: end-to-end replay throughput (RTMW3D-dominated, see module docstring)")
    t_off = throughput(model, a.video, False)
    t_on = throughput(model, a.video, True)
    print("  %-28s %14s %14s" % ("", "OFF", "ON"))
    for k in ("frames", "owned_frames", "suppressed_joint_frames", "elapsed_s", "fps",
              "ms_per_frame"):
        print("  %-28s %14s %14s" % (k, t_off[k], t_on[k]))
    print("  fps delta: %+.2f  (%.2f %% of the OFF arm) - dominated by inference, NOT a measure of"
          % (t_on["fps"] - t_off["fps"],
             100.0 * (t_on["fps"] - t_off["fps"]) / max(1e-9, t_off["fps"])))
    print("             F-22's cost; measurement 1 carries that claim.")

    print("\n" + "=" * 100)
    print(" CONCLUSION")
    print("=" * 100)
    print("  F-22's measured added cost is %.1f us/frame (mean), %.1f us at p95 - %.4f %% of a"
          % (delta_mean, s_pv["p95_us"], budget_pct))
    print("  33.3 ms production frame. That is the upper bound it can add to capture-to-send")
    print("  latency once the live path is running; frame age and capture-to-send themselves are")
    print("  sensor-path properties and are not measurable from a video replay.")
    print("=" * 100)

    with io.open(os.path.join(a.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(video=a.video, cached_frames=len(frames), repeats=a.repeats,
                       isolated=dict(off=s_off, on=s_on, f22_block_only=s_pv,
                                     delta_mean_us=round(delta_mean, 2),
                                     pct_of_frame_budget=round(budget_pct, 5),
                                     suppressed_joint_frames=suppressed),
                       end_to_end=dict(off=t_off, on=t_on)), f, indent=2)
    print(" evidence -> %s" % a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
