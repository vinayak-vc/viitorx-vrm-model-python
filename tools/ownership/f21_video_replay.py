#!/usr/bin/env python3
"""F-21 single-person REGRESSION replay against an arbitrary video file (no OAK-D, no depth).

Scope, deliberately narrow: this does NOT reproduce the live two-person protocol - a plain video has
no depth channel, so there is no metric camera-space position and the production switch_margin_m
(0.35 m, tuned against real backprojected depth) does not apply here. What this DOES usefully test:
whether F-21's identity-continuity check produces any FALSE temporary-loss/release/switch on a single
continuously-moving person during fast, energetic real human motion (jumps, spins, arm-waves) - a
harder stress case for "ordinary movement must not cause switching" (SS7 of the brief) than a calm
walk-in. It reuses the REAL RTMW3D model and the REAL M15 bbox loop, just fed from cv2.VideoCapture
instead of an OAK-D device, and target_ownership.TargetOwnership unmodified.

Position signal here is IMAGE-PLANE (pixel) continuity only - no depth is available. switch_margin is
a fraction of the frame diagonal, not the production metric value; this is stated plainly rather than
implied to be equivalent.

    python tools/ownership/f21_video_replay.py --video "path\to\video.webm"
"""

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(_os.path.join(_d, "_sidecar_path.py")):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import _sidecar_path  # noqa: F401  - puts the sidecar root and every tools/ group on sys.path
import evidence_paths as EV

import argparse
import io
import json
import os
import sys

import cv2
import numpy as np

import rtmw3d_pose as R
import target_ownership as TO

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = EV.DEFAULT_MODEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--conf", type=float, default=0.3, help="matches production's --conf default")
    ap.add_argument("--switch-margin-ratio", type=float, default=0.25,
                    help="max plausible per-frame hip displacement, as a fraction of the frame "
                         "diagonal - NOT the production 0.35 m metric value, see module docstring")
    ap.add_argument("--out-dir", default=EV.oak_v4("f21", "video_regression"))
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

    print("[f21-video] loading RTMW3D ...")
    model = R.RTMW3D(a.model)
    print("[f21-video] %s: %dx%d @ %.1f fps, %d frames" % (a.video, w, h, fps, total))
    print("[f21-video] switch_margin=%.1fpx (%.0f%% of frame diagonal, IMAGE-PLANE only)"
          % (switch_margin, a.switch_margin_ratio * 100))

    cfg = TO.OwnershipConfig(min_confidence=a.conf, switch_margin_m=switch_margin,
                             scale_margin_ratio=10.0)  # scale check effectively off - no reliable
                                                        # metric scale baseline without depth
    ownership = TO.TargetOwnership(cfg)

    bbox = R.center_bbox(w, h)
    events_out = []
    state_counts = {}
    frame_i = 0
    t = 0.0
    dt = 1.0 / fps
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        uv, zrel, conf = model.infer(frame, bbox)
        refined = R.bbox_from_keypoints(uv, conf, w, h, thr=a.conf)
        body_conf_mean = float(np.mean(conf[0:17]))

        valid = False
        pos = None
        scale = None
        if conf[11] > a.conf and conf[12] > a.conf:
            hip = (uv[11] + uv[12]) / 2.0
            pos = (float(hip[0]), float(hip[1]), 0.0)
            valid = True
        elif conf[11] > a.conf:
            pos = (float(uv[11, 0]), float(uv[11, 1]), 0.0)
            valid = True
        elif conf[12] > a.conf:
            pos = (float(uv[12, 0]), float(uv[12, 1]), 0.0)
            valid = True
        if valid and conf[5] > a.conf and conf[6] > a.conf:
            shoulder = (uv[5] + uv[6]) / 2.0
            scale = float(np.linalg.norm(shoulder - np.array(pos[:2])))

        obs = TO.Observation(valid=valid, pos=pos, conf=body_conf_mean, scale=scale)
        state, should_emit = ownership.update(obs, t)
        state_counts[state] = state_counts.get(state, 0) + 1
        for e in ownership.drain_events():
            e["frame"] = frame_i
            e["t"] = round(t, 3)
            events_out.append(e)

        if (refined is not None and body_conf_mean >= a.conf
                and state in (TO.ACQUIRING, TO.LOCKED, TO.REACQUIRING)):
            bbox = tuple(0.7 * np.array(bbox) + 0.3 * np.array(refined))
        elif state in (TO.NO_TARGET, TO.RELEASED):
            bbox = R.center_bbox(w, h)

        frame_i += 1
        t += dt
        if frame_i % 120 == 0:
            print("[f21-video] frame %d/%d state=%s target_id=%s" % (
                frame_i, total, state, ownership.epoch))

    cap.release()

    with io.open(os.path.join(a.out_dir, "target_events.jsonl"), "w", encoding="utf-8") as f:
        for e in events_out:
            f.write(json.dumps(e) + "\n")

    switches = sum(1 for e in events_out if e["event"] == "TARGET_SWITCH")
    temp_lost = sum(1 for e in events_out if e["event"] == "TARGET_TEMP_LOST")
    reacquired = sum(1 for e in events_out if e["event"] == "TARGET_REACQUIRED")
    released = sum(1 for e in events_out if e["event"] == "TARGET_RELEASED")
    rejected = sum(1 for e in events_out if e["event"] == "TARGET_REJECTED_CANDIDATE")

    summary = dict(
        video=a.video, frames=frame_i, fps=fps, final_target_id=ownership.epoch,
        state_counts=state_counts, target_switch_count=switches,
        temp_lost_events=temp_lost, reacquired_events=reacquired, released_events=released,
        rejected_candidate_events=rejected,
        switch_margin_px=switch_margin, switch_margin_ratio=a.switch_margin_ratio,
    )
    with io.open(os.path.join(a.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 78)
    print(" F-21 single-person video regression summary")
    print("=" * 78)
    for k, v in summary.items():
        print("  %-26s %s" % (k, v))
    print("-" * 78)
    ok = (switches == 0 and ownership.epoch <= 1)
    print(" %s: %d frames, target_id never exceeded 1, %d TARGET_SWITCH events"
          % ("PASS" if ok else "FAIL", frame_i, switches))
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
