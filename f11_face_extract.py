#!/usr/bin/env python3
"""F-11 -- extract face/shoulder PIXEL geometry from a video, offline.

The archived F-10 capture does not contain the face keypoints (`AUDIT_JOINTS` starts at COCO index 5;
nose/eyes/ears are 0-4), and no RGB was recorded, so the F-11 candidates cannot be computed from it.
This runs the SAME RTMW3D model the sidecar runs, over a video file, and dumps the pixel coordinates
the candidates need. It touches no production code and needs no capture.

WHAT IT CANNOT DO: supply a ground-truth heading. Sign ACCURACY (F-11 sec 6/7/11) needs known
headings, which only a capture provides. What this DOES answer is F-11 sec 8 -- whether the face
signal is measuring yaw at all, or tracking a confounder.

COCO-WholeBody indices, confirmed against rtmw3d_pose.COCO17_TO_JOINTID:
    0 nose | 1 L-eye | 2 R-eye | 3 L-ear | 4 R-ear | 5 L-shoulder | 6 R-shoulder | 11/12 hips

    python f11_face_extract.py --video ../../video/video.webm --out oak_v4_evidence/f11_video_face.jsonl
"""
import argparse
import json
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rtmw3d_pose as R

IDX = {"nose": 0, "Leye": 1, "Reye": 2, "Lear": 3, "Rear": 4,
       "Lsh": 5, "Rsh": 6, "Lhip": 11, "Rhip": 12}
CONF_THR = 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default=os.path.join("..", "..", "..", "SentisModel", "rtmw3d-x.onnx"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if not os.path.exists(a.video):
        print("[ERROR] video not found: %s" % a.video)
        return 1
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        print("[ERROR] cannot open video")
        return 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("video %dx%d  ->  %s" % (w, h, a.out))
    model = R.RTMW3D(a.model)
    print("providers: %s" % (model.active_providers,))

    bbox = R.center_bbox(w, h)
    lowconf = 0
    n = 0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            uv, zrel, conf = model.infer(frame, bbox)
            refined = R.bbox_from_keypoints(uv, conf, w, h, thr=CONF_THR)
            if refined is not None and float(np.mean(conf[0:17])) >= CONF_THR:
                bbox = refined
                lowconf = 0
            else:
                lowconf += 1
                if refined is None or lowconf >= 20:
                    bbox = R.center_bbox(w, h)
                    lowconf = 0
            rec = {"frame": n}
            for name, i in IDX.items():
                rec[name] = [round(float(uv[i][0]), 2), round(float(uv[i][1]), 2),
                             round(float(conf[i]), 4)]
            # the model's own root-relative depth, for the 3-D comparison only
            rec["zrel"] = [round(float(zrel[IDX["Lsh"]]), 4), round(float(zrel[IDX["Rsh"]]), 4)]
            fh.write(json.dumps(rec) + "\n")
            if n % 100 == 0:
                print("  %d frames" % n)
    cap.release()
    print("done: %d frames -> %s" % (n, a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
