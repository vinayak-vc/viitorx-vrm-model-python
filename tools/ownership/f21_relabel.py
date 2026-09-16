#!/usr/bin/env python3
"""Recompute GROUND TRUTH LABELS for an existing f21_wrongperson_replay run, without re-inferring.

The pipeline's behaviour and the instrument that scores it are independent, and they should be
iterated independently. The *_rows.jsonl files already record every frame's ownership state, emit
decision, epoch and emitted hip position; only the label depends on f21_ground_truth.py. So a change
to the labeller costs one pass of background subtraction, not four passes of RTMW3D.

That separation is also the honest one: re-running the whole replay after every labeller tweak would
make it impossible to tell whether a number moved because the FIX changed or because the RULER did.
Here the pipeline output is fixed input, read from disk.

    python tools/ownership/f21_relabel.py --video ...\\123.webm
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

import cv2

from f21_ground_truth import GroundTruth, classify, nearest
from f21_wrongperson_replay import wrong_person, summarise

ARMS = ["OWNERSHIP_OFF", "PATH_OFF", "PATH_ON", "PATH_ON_NO_F22"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--dir", default=EV.oak_v4("f21", "wrongperson"))
    ap.add_argument("--labels", default="W=55:90:0.15")
    ap.add_argument("--default-label", default="M")
    ap.add_argument("--label-radius", type=float, default=0.06)
    a = ap.parse_args()

    windows = []
    for part in [x for x in a.labels.split(",") if x.strip()]:
        name, rng = part.split("=")
        lo, hi, ms = rng.split(":")
        windows.append((name, float(lo), float(hi), float(ms)))

    base = os.path.splitext(os.path.basename(a.video))[0]
    gt = GroundTruth(a.video)

    # one pass over the video, labelling every frame's blobs once for every arm
    rows_by_arm = {}
    for arm in ARMS:
        p = os.path.join(a.dir, "%s_%s_rows.jsonl" % (base, arm))
        if os.path.isfile(p):
            rows_by_arm[arm] = [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]
    if not rows_by_arm:
        raise SystemExit("no *_rows.jsonl found in %s" % a.dir)

    cap = cv2.VideoCapture(a.video)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        blobs = None
        for arm, rows in rows_by_arm.items():
            if i >= len(rows):
                continue
            r = rows[i]
            if r["hip_u"] is None:
                r["label"], r["margin"] = "NONE", 0.0
                continue
            if blobs is None:
                blobs = gt.blobs(frame)
            b = nearest(blobs, r["hip_u"], r["hip_v"], a.label_radius, gt.w, gt.h)
            lab, m = ("NONE", 0.0) if b is None else classify(b, windows, a.default_label)
            r["label"], r["margin"] = lab, round(m, 4)
        i += 1
    cap.release()

    out = {}
    print("=" * 104)
    print(" RELABELLED  %s   labels=%s default=%s" % (base, windows, a.default_label))
    print("=" * 104)
    for arm in ARMS:
        if arm not in rows_by_arm:
            continue
        rows = rows_by_arm[arm]
        evp = os.path.join(a.dir, "%s_%s_events.jsonl" % (base, arm))
        events = [json.loads(l) for l in io.open(evp, encoding="utf-8") if l.strip()] \
            if os.path.isfile(evp) else []
        wp = wrong_person(rows)
        s = summarise(arm, rows, events, wp, [], 1.0, gt.fps)
        out[arm] = s
        io.open(os.path.join(a.dir, "%s_%s_rows.jsonl" % (base, arm)), "w",
                encoding="utf-8").write("".join(json.dumps(r) + "\n" for r in rows))
        print("\n-- %s" % arm)
        print("   emitted=%d held=%d  WRONG=%d in %d episode(s)  first=%s  dur=%.2fs"
              % (s["emitted"], s["held_frames"], s["wrong_person_frames"],
                 s["wrong_person_episodes"], s["first_wrong_person_frame"],
                 s["wrong_person_duration_s"]))
        print("   unlabelled_emitted=%d  epochs=%d releases=%d switches=%d reacquires=%d "
              "path_rejections=%d" % (s["unlabelled_emitted"], s["epochs"], s["released"],
                                      s["switches"], s["reacquired"],
                                      s["rejected_path_walked_in"]))
        for ep, an in sorted(s["epoch_anchors"].items()):
            print("   epoch %s owner=%s majority=%.2f%s  hist=%s  (first labelled f%d -> %s)"
                  % (ep, an["anchored_on"], an["majority"],
                     "  << AMBIGUOUS, EXCLUDED" if an["ambiguous"] else "",
                     an["epoch_label_histogram"], an["first_labelled_frame"], an["first_label"]))
        if wp["episodes"]:
            for e in wp["episodes"]:
                print("   wrong-person episode f%d-f%d (%d frames, %.2fs) label=%s"
                      % (e[0]["frame"], e[-1]["frame"], len(e), len(e) / gt.fps, e[0]["label"]))

    io.open(os.path.join(a.dir, "%s_summary.json" % base), "w",
            encoding="utf-8").write(json.dumps(out, indent=2))
    print("\n" + "=" * 104)
    print(" %-18s %8s %8s %8s %10s %9s %8s %9s" % ("ARM", "emitted", "held", "WRONG", "episodes",
                                                   "releases", "switch", "pathrej"))
    for arm in ARMS:
        if arm not in out:
            continue
        s = out[arm]
        print(" %-18s %8d %8d %8d %10d %9d %8d %9d"
              % (arm, s["emitted"], s["held_frames"], s["wrong_person_frames"],
                 s["wrong_person_episodes"], s["released"], s["switches"],
                 s["rejected_path_walked_in"]))
    print("=" * 104)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
