#!/usr/bin/env python3

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(_os.path.join(_d, "_sidecar_path.py")):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import _sidecar_path  # noqa: F401  - puts the sidecar root and every tools/ group on sys.path
"""F-22 - classify every ABSOLUTE-angle rejection found on real footage. True catch, or false one?

f22_threshold_analysis.py found the thing the F-22 report said it had never had: the absolute-angle
REJECT path firing on REAL video (elbow bends of 163-174 deg, past the shipped 160 deg line). A count
alone cannot say whether those are anatomically impossible poses F-22 correctly caught, or artifacts
of the offline geometry F-22 wrongly rejected - and a report that guessed either way would be
worthless. This script decides it with a measurement plus a picture.

THE DISCRIMINATOR. The offline 3D lift takes each joint's depth from the model's own root-relative z
about a nominal hip plane; production instead measures depth with real stereo. So an "impossible"
3D angle can arise two different ways, and they are separable:

    the 2D IMAGE-PLANE bend angle is ALSO extreme  -> the forearm really is folded back against the
                                                      upper arm in the picture. The pose is impossible
                                                      in the raw image, independent of any depth
                                                      channel. F-22 caught a genuine bad pose.
    the 2D bend is ordinary, only 3D is extreme    -> the fold exists only along z, i.e. it came from
                                                      the model's zrel, which production does not use
                                                      for this. Evidence about THIS HARNESS, not about
                                                      the production signal - and a reason to expect a
                                                      different rate live.

Each flagged frame is written out with its skeleton drawn on it, so the classification can be
checked by eye rather than taken on trust.

    python f22_inspect_rejections.py --video "D:\\...\\video\\123.webm"
"""
import argparse
import io
import json
import math
import os
import sys

import cv2
import numpy as np

import rtmw3d_pose as R
import pose_validation as PV

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "..", "..", "..", "SentisModel", "rtmw3d-x.onnx")
NOMINAL_HIP_Z = 2.0
CONF_THR = 0.3
CHAIN_TRIPLE = {"left_elbow": (5, 7, 9), "right_elbow": (6, 8, 10),
                "left_knee": (11, 13, 15), "right_knee": (12, 14, 16)}
CHAIN_CFG = {"left_elbow": PV.ELBOW_CONFIG, "right_elbow": PV.ELBOW_CONFIG,
             "left_knee": PV.KNEE_CONFIG, "right_knee": PV.KNEE_CONFIG}


def bend2d(uv, triple):
    p, j, d = triple
    u = np.array(uv[j], dtype=np.float64) - np.array(uv[p], dtype=np.float64)
    l = np.array(uv[d], dtype=np.float64) - np.array(uv[j], dtype=np.float64)
    nu, nl = np.linalg.norm(u), np.linalg.norm(l)
    if nu < 1e-6 or nl < 1e-6:
        return None
    c = float(np.dot(u, l)) / (nu * nl)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def draw(frame, uv, conf, triple, name, b3, b2):
    img = frame.copy()
    for a, b in R.JOINTID_EDGES if hasattr(R, "JOINTID_EDGES") else []:
        pass
    skel = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)]
    for a, b in skel:
        if conf[a] > CONF_THR and conf[b] > CONF_THR:
            cv2.line(img, (int(uv[a][0]), int(uv[a][1])), (int(uv[b][0]), int(uv[b][1])),
                     (90, 90, 90), 2)
    p, j, d = triple
    for idx, col in ((p, (255, 200, 0)), (j, (0, 0, 255)), (d, (0, 255, 255))):
        cv2.circle(img, (int(uv[idx][0]), int(uv[idx][1])), 9, col, -1)
    cv2.line(img, (int(uv[p][0]), int(uv[p][1])), (int(uv[j][0]), int(uv[j][1])), (0, 0, 255), 4)
    cv2.line(img, (int(uv[j][0]), int(uv[j][1])), (int(uv[d][0]), int(uv[d][1])), (0, 255, 255), 4)
    txt = "%s  3D=%.1fdeg  2D=%.1fdeg  (REJECT %.0f)" % (name, b3, b2, CHAIN_CFG[name].reject_deg)
    cv2.rectangle(img, (0, 0), (img.shape[1], 46), (0, 0, 0), -1)
    cv2.putText(img, txt, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    vdir = os.path.join("D:", os.sep, "Unity", "viitorx-vrm-avtar-unity-base-project", "Assets",
                        "Games", "video")
    ap.add_argument("--videos", default=",".join(os.path.join(vdir, v) for v in
                                                 ("video.webm", "123.webm", "456.webm")))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-images", type=int, default=6, help="annotated frames saved per video")
    ap.add_argument("--out-dir", default=os.path.join("oak_v4_evidence", "f22", "rejections"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    print("=" * 100)
    print(" F-22 - classifying every ABSOLUTE-angle rejection on real footage")
    print("=" * 100)
    model = R.RTMW3D(a.model)

    all_hits = []
    per_video = {}
    for path in a.videos.split(","):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise SystemExit("could not open %s" % path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fx = fy = 800.0
        cx, cy = w / 2.0, h / 2.0
        bbox = R.center_bbox(w, h)
        hits = []
        saved = 0
        fi = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            uv, zrel, conf = model.infer(frame, bbox)
            ref = R.bbox_from_keypoints(uv, conf, w, h, thr=CONF_THR)
            if ref is not None:
                bbox = tuple(0.7 * np.array(bbox) + 0.3 * np.array(ref))
            zrel_hip = float((zrel[11] + zrel[12]) / 2.0)
            xyz = np.zeros((17, 3), dtype=np.float64)
            for i in range(17):
                z = NOMINAL_HIP_Z + (float(zrel[i]) - zrel_hip)
                xyz[i] = [(float(uv[i, 0]) - cx) * z / fx, (float(uv[i, 1]) - cy) * z / fy, z]
            for name, triple in CHAIN_TRIPLE.items():
                p, j, d = triple
                if not (conf[p] > CONF_THR and conf[j] > CONF_THR and conf[d] > CONF_THR):
                    continue
                b3 = PV._bend_deg(tuple(xyz[p]), tuple(xyz[j]), tuple(xyz[d]))
                if b3 is None or b3 <= CHAIN_CFG[name].reject_deg:
                    continue
                b2 = bend2d(uv, triple)
                hit = dict(video=os.path.basename(path), frame=fi, chain=name,
                           bend3d=round(b3, 2), bend2d=(round(b2, 2) if b2 is not None else None),
                           reject_deg=CHAIN_CFG[name].reject_deg,
                           conf=[round(float(conf[k]), 3) for k in triple],
                           # a fold that is real in the picture is real regardless of any z channel
                           classification=("IMPOSSIBLE_IN_2D_IMAGE" if (b2 is not None and
                                                                       b2 > CHAIN_CFG[name].reject_deg)
                                           else "DEPTH_DERIVED_ONLY"))
                hits.append(hit)
                if saved < a.max_images:
                    img = draw(frame, uv, conf, triple, name, b3, b2 if b2 else -1)
                    fn = "%s_f%05d_%s.png" % (os.path.splitext(os.path.basename(path))[0], fi, name)
                    cv2.imwrite(os.path.join(a.out_dir, fn), img)
                    hit["image"] = fn
                    saved += 1
            fi += 1
        cap.release()
        per_video[os.path.basename(path)] = dict(frames=fi, rejections=len(hits))
        all_hits.extend(hits)
        print("  %-12s %4d frames  %3d absolute-angle rejections" % (
            os.path.basename(path), fi, len(hits)))

    print("\n" + "=" * 100)
    print(" CLASSIFICATION")
    print("=" * 100)
    real2d = [x for x in all_hits if x["classification"] == "IMPOSSIBLE_IN_2D_IMAGE"]
    depthonly = [x for x in all_hits if x["classification"] == "DEPTH_DERIVED_ONLY"]
    print("  total absolute-angle rejections        : %d" % len(all_hits))
    print("  impossible in the 2D IMAGE too         : %d  -> a genuine bad pose; F-22 caught it,"
          % len(real2d))
    print("                                             and would catch it with real stereo depth too")
    print("  extreme only after the offline z lift  : %d  -> attributable to this harness's zrel"
          % len(depthonly))
    print("                                             proxy, NOT to production's measured depth")
    if all_hits:
        print("\n  %-12s %7s %-13s %9s %9s  %s" % ("video", "frame", "chain", "3D deg", "2D deg",
                                                   "classification"))
        for x in sorted(all_hits, key=lambda r: -r["bend3d"])[:20]:
            print("  %-12s %7d %-13s %9.1f %9s  %s" % (x["video"], x["frame"], x["chain"],
                                                       x["bend3d"], x["bend2d"], x["classification"]))
    print("=" * 100)

    with io.open(os.path.join(a.out_dir, "rejections.json"), "w", encoding="utf-8") as f:
        json.dump(dict(per_video=per_video, total=len(all_hits),
                       impossible_in_2d=len(real2d), depth_derived_only=len(depthonly),
                       hits=all_hits), f, indent=2)
    print(" evidence -> %s  (%d annotated frames saved)" % (
        a.out_dir, sum(1 for x in all_hits if "image" in x)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
