#!/usr/bin/env python3

import evidence_paths as EV

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(_os.path.join(_d, "_sidecar_path.py")):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import _sidecar_path  # noqa: F401  - puts the sidecar root and every tools/ group on sys.path
"""F-22 offline replay + threshold-derivation harness — reuses f21_video_replay.py's video-input
pattern (RTMW3D + M15 bbox loop, no OAK-D/depth needed) and composes F-21 ownership + F-22 pose
validation in PRODUCTION'S OWN ORDER (brief SS20: "F-22 should validate the owned target, not select
the target"). The SAME pose_validation.PoseValidator class that runs in wholebody_udp_sender.py runs
here, unmodified — no separate "fake" validation algorithm for replay (brief SS19).

Position/geometry here is the model's own raw camera-space-shaped xyz (image-plane x,y + the model's
root-relative zrel as z) — there is no real depth channel for an arbitrary video, exactly as
f21_video_replay.py already established and documented.

Produces, per the brief's SS22/SS23:
  - real elbow/knee bend-angle and angular-velocity distributions (p50/p95) from this footage,
  - a baseline-vs-F22-enabled comparison (how many frames would have carried an anatomically
    impossible angle un-gated, vs how many F-22 actually suppressed),
  - a false-reject count on this (entirely legitimate, no "hands near face") footage.

    python f22_video_replay.py --video "path\to\video.webm"
"""
import argparse
import io
import json
import os
import sys

import cv2
import numpy as np

import rtmw3d_pose as R
import target_ownership as TO
import pose_validation as PV

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "..", "..", "..", "SentisModel", "rtmw3d-x.onnx")


def _percentile(values, p):
    if not values:
        return None
    return float(np.percentile(np.array(values), p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--switch-margin-ratio", type=float, default=0.25,
                    help="F-21 ownership margin, image-plane fraction of frame diagonal - see "
                         "f21_video_replay.py's module docstring for why this differs from the "
                         "production metric value")
    ap.add_argument("--out-dir", default=EV.oak_v4("f22", "video_regression"))
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        print("ERROR: could not open %s" % a.video)
        return 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    diag = (w * w + h * h) ** 0.5
    switch_margin = diag * a.switch_margin_ratio

    print("[f22-video] loading RTMW3D ...")
    model = R.RTMW3D(a.model)
    print("[f22-video] %s: %dx%d @ %.1f fps, %d frames" % (a.video, w, h, fps, total))

    ownership = TO.TargetOwnership(TO.OwnershipConfig(min_confidence=a.conf,
                                                       switch_margin_m=switch_margin,
                                                       scale_margin_ratio=10.0))
    validator = PV.PoseValidator()

    bbox = R.center_bbox(w, h)
    pose_events = []
    # per-chain accumulators
    bends = dict((name, []) for name in PV.CHAIN_DEFS)
    rates = dict((name, []) for name in PV.CHAIN_DEFS)
    state_counts = dict((name, {}) for name in PV.CHAIN_DEFS)
    baseline_impossible = dict((name, 0) for name in PV.CHAIN_DEFS)   # bend > reject_deg, unconditional
    f22_suppressed = dict((name, 0) for name in PV.CHAIN_DEFS)        # F-22 actually zeroed conf_emit
    owned_frames = 0
    frame_i = 0
    t = 0.0
    dt = 1.0 / fps
    idx_to_chain = dict((j_idx, name) for name, (_, j_idx, _, _) in PV.CHAIN_DEFS.items())

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        uv, zrel, conf = model.infer(frame, bbox)
        refined = R.bbox_from_keypoints(uv, conf, w, h, thr=a.conf)
        body_conf_mean = float(np.mean(conf[0:17]))

        # Build a 17-joint xyz array in a consistent (image-plane-x, image-plane-y, model zrel)
        # space - same scope limitation f21_video_replay.py already documents (no real depth for an
        # arbitrary video). zrel IS the model's own metric-ish root-relative z (rtmw3d_pose.py), so
        # unlike F-21's replay this gives F-22's angle math a genuine (if unverified) 3rd dimension.
        xyz = np.zeros((17, 3), dtype=np.float32)
        for i in range(17):
            xyz[i] = [uv[i, 0], uv[i, 1], zrel[i] * 200.0]  # crude px-per-metre scale for a 3D angle
        measured = conf[:17] > a.conf

        hip_pos = None
        hip_valid = False
        scale = None
        if conf[11] > a.conf and conf[12] > a.conf:
            hip_pos = tuple(((uv[11] + uv[12]) / 2.0).tolist()) + (0.0,)
            hip_valid = True
        if hip_valid and conf[5] > a.conf and conf[6] > a.conf:
            shoulder = (uv[5] + uv[6]) / 2.0
            scale = float(np.linalg.norm(shoulder - np.array(hip_pos[:2])))

        obs = TO.Observation(valid=hip_valid, pos=hip_pos, conf=body_conf_mean, scale=scale)
        own_state, should_emit = ownership.update(obs, t)

        if (refined is not None and body_conf_mean >= a.conf
                and own_state in (TO.ACQUIRING, TO.LOCKED, TO.REACQUIRING)):
            bbox = tuple(0.7 * np.array(bbox) + 0.3 * np.array(refined))
        elif own_state in (TO.NO_TARGET, TO.RELEASED):
            bbox = R.center_bbox(w, h)

        if should_emit:
            owned_frames += 1
            conf_emit = conf.copy()
            pv_out = validator.update(xyz, measured, conf_emit, t)
            for j_idx, (state, reason, bend, rate) in pv_out.items():
                name = idx_to_chain[j_idx]
                state_counts[name][state] = state_counts[name].get(state, 0) + 1
                if bend is not None:
                    bends[name].append(bend)
                    cfg = PV.CHAIN_DEFS[name][3]
                    if bend > cfg.reject_deg:
                        baseline_impossible[name] += 1
                if rate is not None:
                    rates[name].append(rate)
                if state != PV.VALID:
                    f22_suppressed[name] += 1
            for e in validator.drain_events():
                e["frame"] = frame_i
                e["t"] = round(t, 3)
                pose_events.append(e)

        frame_i += 1
        t += dt
        if frame_i % 120 == 0:
            print("[f22-video] frame %d/%d own_state=%s" % (frame_i, total, own_state))

    cap.release()

    with io.open(os.path.join(a.out_dir, "pose_events.jsonl"), "w", encoding="utf-8") as f:
        for e in pose_events:
            f.write(json.dumps(e) + "\n")

    summary = dict(video=a.video, frames=frame_i, owned_frames=owned_frames, fps=fps, chains={})
    print("=" * 78)
    print(" F-22 offline threshold-derivation summary (%s)" % a.video)
    print("=" * 78)
    for name in PV.CHAIN_DEFS:
        cfg = PV.CHAIN_DEFS[name][3]
        b = bends[name]
        r = rates[name]
        chain_summary = dict(
            n_samples=len(b),
            bend_p50=_percentile(b, 50), bend_p95=_percentile(b, 95), bend_max=(max(b) if b else None),
            rate_p50=_percentile(r, 50), rate_p95=_percentile(r, 95), rate_max=(max(r) if r else None),
            state_counts=state_counts[name],
            baseline_impossible_frames=baseline_impossible[name],
            f22_suppressed_frames=f22_suppressed[name],
            reject_deg=cfg.reject_deg, warn_deg=cfg.warn_deg,
        )
        summary["chains"][name] = chain_summary
        print(" %-14s bend p50=%.1f p95=%.1f max=%.1f | rate p50=%.0f p95=%.0f max=%.0f deg/s"
              % (name, chain_summary["bend_p50"] or -1, chain_summary["bend_p95"] or -1,
                 chain_summary["bend_max"] or -1, chain_summary["rate_p50"] or -1,
                 chain_summary["rate_p95"] or -1, chain_summary["rate_max"] or -1))
        print("   states=%s  baseline_impossible=%d  f22_suppressed=%d"
              % (state_counts[name], baseline_impossible[name], f22_suppressed[name]))
    print("-" * 78)
    total_suppressed = sum(f22_suppressed.values())
    total_baseline = sum(baseline_impossible.values())
    print(" owned_frames=%d  total_baseline_impossible=%d  total_f22_suppressed=%d"
          % (owned_frames, total_baseline, total_suppressed))
    print("=" * 78)

    with io.open(os.path.join(a.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
