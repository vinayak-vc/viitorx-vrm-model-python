#!/usr/bin/env python3
"""F-21 REAL MULTI-PERSON video replay - ownership ON vs OFF on genuinely crowded footage.

Why this is a different evidence class from f21_video_replay.py: that harness replays a SINGLE
person and proves F-21 does not false-trip on energetic motion. This one replays footage with
roughly eight real humans in frame simultaneously, which is the actual condition F-19's failure
needs - the M15 crop has other real bodies to wander onto, in real imagery, with real occlusion
and real crossing. It exercises exactly the loop that failed in F-19: crop -> single-person
inference -> crop refinement.

WHAT IT CANNOT PROVE, said plainly: there is no identity ground truth here. RTMW3D-x emits one
skeleton per frame with no track id, so nothing in this pipeline can state which human any given
frame belongs to. What IS measurable without ground truth is the geometric signature of a
body-to-body hand-off:

    the image-plane hip displacement BETWEEN CONSECUTIVE EMITTED FRAMES.

A real human at 30 fps moves a bounded distance per frame; a crop that jumps from one dancer to
another two metres away produces a displacement far outside that distribution. Counting how large a
jump survives into the OUTPUT stream, with ownership ON versus OFF, measures whether F-21 stops
body-to-body hand-offs reaching the consumer - on real multi-person imagery rather than on labels
this harness made up itself.

There is no depth channel in an arbitrary video, so the margin is a fraction of the frame diagonal
rather than production's calibrated 0.35 m - the same scope limitation f21_video_replay.py already
documents, restated rather than quietly reused.

    python tools/ownership/f21_multiperson_replay.py --video "../../video/456.webm"
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
import time

import cv2
import numpy as np

import rtmw3d_pose as R
import target_ownership as TO

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = EV.DEFAULT_MODEL


def run_pass(model, video, conf_thr, switch_margin_ratio, ownership_on, log):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit("could not open %s" % video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    diag = (w * w + h * h) ** 0.5
    margin = diag * switch_margin_ratio

    own = None
    if ownership_on:
        own = TO.TargetOwnership(TO.OwnershipConfig(
            min_confidence=conf_thr, switch_margin_m=margin,
            scale_margin_ratio=10.0))          # no reliable metric scale without depth

    bbox = R.center_bbox(w, h)
    lowconf_streak = 0
    dt = 1.0 / fps
    t = 0.0
    frame_i = 0
    emitted = []          # (frame_i, hip_uv) for frames that reached the consumer
    events = []
    state_counts = {}
    t0 = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        uv, zrel, conf = model.infer(frame, bbox)
        refined = R.bbox_from_keypoints(uv, conf, w, h, thr=conf_thr)
        body_conf_mean = float(np.mean(conf[0:17]))

        hip = None
        scale = None
        if conf[11] > conf_thr and conf[12] > conf_thr:
            hip = (uv[11] + uv[12]) / 2.0
        elif conf[11] > conf_thr:
            hip = uv[11].copy()
        elif conf[12] > conf_thr:
            hip = uv[12].copy()
        if hip is not None and conf[5] > conf_thr and conf[6] > conf_thr:
            scale = float(np.linalg.norm((uv[5] + uv[6]) / 2.0 - hip))

        if own is not None:
            obs = TO.Observation(valid=hip is not None,
                                 pos=(float(hip[0]), float(hip[1]), 0.0) if hip is not None else None,
                                 conf=body_conf_mean, scale=scale)
            state, should_emit = own.update(obs, t)
            state_counts[state] = state_counts.get(state, 0) + 1
            for e in own.drain_events():
                e["frame"] = frame_i
                e["t"] = round(t, 3)
                events.append(e)
            # identity-gated M15 refinement - production's own rule
            if (refined is not None and body_conf_mean >= conf_thr
                    and state in (TO.ACQUIRING, TO.LOCKED, TO.REACQUIRING)):
                bbox = tuple(0.7 * np.array(bbox) + 0.3 * np.array(refined))
            elif state in (TO.NO_TARGET, TO.RELEASED):
                bbox = R.center_bbox(w, h)
        else:
            # pre-F-21 M15 behaviour, copied verbatim from wholebody_udp_sender.py's else-branch
            should_emit = hip is not None
            state = "M15"
            state_counts[state] = state_counts.get(state, 0) + 1
            if refined is not None and body_conf_mean >= conf_thr:
                lowconf_streak = 0
                bbox = tuple(0.7 * np.array(bbox) + 0.3 * np.array(refined))
            else:
                lowconf_streak += 1
                if refined is None or lowconf_streak >= 20:
                    bbox = R.center_bbox(w, h)
                    lowconf_streak = 0
                else:
                    bbox = tuple(0.7 * np.array(bbox) + 0.3 * np.array(refined))

        if should_emit and hip is not None:
            emitted.append((frame_i, float(hip[0]), float(hip[1])))

        frame_i += 1
        t += dt
        if frame_i % 100 == 0:
            log("    frame %d  state=%s" % (frame_i, state))
    cap.release()
    elapsed = time.perf_counter() - t0

    # displacement between CONSECUTIVE EMITTED frames, normalised per frame gap so a legitimate
    # multi-frame hold is not counted as a teleport.
    jumps = []
    for k in range(1, len(emitted)):
        f0, x0, y0 = emitted[k - 1]
        f1, x1, y1 = emitted[k]
        gap = max(1, f1 - f0)
        jumps.append(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / gap)

    return dict(
        ownership=("ON" if ownership_on else "OFF"),
        width=w, height=h, fps=fps, frames=frame_i,
        emitted_frames=len(emitted),
        emission_rate=round(len(emitted) / float(max(1, frame_i)), 4),
        switch_margin_px=round(margin, 1),
        state_counts=state_counts,
        target_switch_events=sum(1 for e in events if e["event"] == "TARGET_SWITCH"),
        temp_lost_events=sum(1 for e in events if e["event"] == "TARGET_TEMP_LOST"),
        rejected_candidate_events=sum(1 for e in events
                                      if e["event"] == "TARGET_REJECTED_CANDIDATE"),
        released_events=sum(1 for e in events if e["event"] == "TARGET_RELEASED"),
        reacquired_events=sum(1 for e in events if e["event"] == "TARGET_REACQUIRED"),
        final_target_id=(own.epoch if own is not None else None),
        jump_p50_px=(round(float(np.percentile(jumps, 50)), 1) if jumps else None),
        jump_p95_px=(round(float(np.percentile(jumps, 95)), 1) if jumps else None),
        jump_p99_px=(round(float(np.percentile(jumps, 99)), 1) if jumps else None),
        jump_max_px=(round(max(jumps), 1) if jumps else None),
        jumps_over_margin=sum(1 for j in jumps if j > margin),
        elapsed_s=round(elapsed, 2),
        fps_processed=round(frame_i / elapsed, 2) if elapsed > 0 else None,
        events=events,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--conf", type=float, default=0.3)
    # 0.064 of this clip's diagonal is ~140 px, which is what production's 0.35 m switch margin
    # actually subtends at the installation's ~2 m working depth through a ~800 px focal length
    # (0.35 * 800 / 2.0 = 140 px). Sweeping it against a deliberately loose 0.25 shows whether the
    # PRODUCTION-equivalent margin false-trips on real crowded footage - the question a single loose
    # margin cannot answer.
    ap.add_argument("--margins", default="0.064,0.10,0.25",
                    help="comma-separated ownership margins as a fraction of the frame diagonal")
    ap.add_argument("--out-dir", default=EV.oak_v4("f21", "multiperson"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    margins = [float(x) for x in a.margins.split(",")]

    def log(msg):
        print(msg, flush=True)

    log("=" * 100)
    log(" F-21 REAL MULTI-PERSON replay  (ownership ON vs OFF)  %s" % a.video)
    log("=" * 100)
    log("[load] RTMW3D ...")
    model = R.RTMW3D(a.model)

    results = {}
    log("\n-- pass: ownership OFF (pre-F-21 M15 behaviour)")
    results["OFF"] = run_pass(model, a.video, a.conf, margins[-1], False, log)
    for m in margins:
        key = "ON@%.3f" % m
        log("\n-- pass: ownership ON, margin=%.3f of diagonal" % m)
        results[key] = run_pass(model, a.video, a.conf, m, True, log)

    for key, r in results.items():
        safe = key.replace("@", "_").replace(".", "p")
        with io.open(os.path.join(a.out_dir, "events_%s.jsonl" % safe), "w", encoding="utf-8") as f:
            for e in r.pop("events"):
                f.write(json.dumps(e) + "\n")

    cols = ["OFF"] + ["ON@%.3f" % m for m in margins]
    log("\n" + "=" * 100)
    log(" RESULT - image-plane hip displacement between consecutive EMITTED frames")
    log("=" * 100)
    log("  %-28s %s" % ("metric", "".join("%-16s" % c for c in cols)))
    for k in ("switch_margin_px", "frames", "emitted_frames", "emission_rate", "jump_p50_px",
              "jump_p95_px", "jump_p99_px", "jump_max_px", "jumps_over_margin",
              "target_switch_events", "temp_lost_events", "rejected_candidate_events",
              "released_events", "reacquired_events", "final_target_id", "fps_processed"):
        log("  %-28s %s" % (k, "".join("%-16s" % results[c].get(k) for c in cols)))
    log("=" * 100)

    with io.open(os.path.join(a.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(video=a.video, margins=margins, results=results), f, indent=2)
    log(" evidence -> %s" % a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
