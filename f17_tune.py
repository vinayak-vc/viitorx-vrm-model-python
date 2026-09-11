#!/usr/bin/env python3
"""
F-17 - is the half-baseline CAM_A/CAM_C pair usable at all?

CAM_A is a COLOUR sensor with an IR-cut filter; CAM_B/CAM_C are unfiltered mono. If that spectral
mismatch defeats block matching, the half-baseline route dies here and the finding is that this
hardware cannot supply a second baseline for comparison.

Runs several matcher configurations over the captured frames and reports, per pair, the depth
coverage of the whole frame and of the torso band specifically. Offline: no camera, no subject.

Output: oak_v4_evidence/f17/tuning.txt
"""
import io
import json
import os
import sys

import numpy as np
import cv2

import f17_stereo_rig as RIG

OUT = os.path.join("oak_v4_evidence", "f17", "tuning.txt")
L = []


def say(s=""):
    print(s)
    L.append(s)


def load_calib(root):
    c = json.load(io.open(os.path.join(root, "calib.json"), encoding="utf-8"))
    K = {k: np.array(v, dtype=np.float64) for k, v in c["intrinsics"].items()}
    D = {}
    for k, v in c["distortion"].items():
        d = np.array(v, dtype=np.float64)
        if d.size not in (4, 5, 8, 12, 14):
            d = d[:8] if d.size > 8 else d[:4]
        D[k] = d.reshape(1, -1)
    E = {k: np.array(v, dtype=np.float64) for k, v in c["extrinsics"].items()}
    return K, D, E


def rectify_pair(K, D, E, ln, rn):
    M = E["%s->%s" % (ln, rn)]
    Rm = M[:3, :3]
    T = M[:3, 3] * 10.0                      # depthai gives cm
    R1, R2, P1, P2, Q, _a, _b = cv2.stereoRectify(
        K[ln], D[ln], K[rn], D[rn], (RIG.W, RIG.H), Rm, T.reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    m1 = cv2.initUndistortRectifyMap(K[ln], D[ln], R1, P1, (RIG.W, RIG.H), cv2.CV_16SC2)
    m2 = cv2.initUndistortRectifyMap(K[rn], D[rn], R2, P2, (RIG.W, RIG.H), cv2.CV_16SC2)
    f = float(P1[0, 0])
    B = abs(float(P2[0, 3]) / P2[0, 0])
    return dict(R1=R1, R2=R2, P1=P1, P2=P2, m1=m1, m2=m2, f=f, B=B, fB=f * B)


CONFIGS = [
    ("sgbm_b5_d160",    dict(blockSize=5,  numDisparities=160, uniquenessRatio=10,
                             mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY, clahe=False)),
    ("sgbm_b7_d192",    dict(blockSize=7,  numDisparities=192, uniquenessRatio=5,
                             mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY, clahe=False)),
    ("sgbm_b7_clahe",   dict(blockSize=7,  numDisparities=192, uniquenessRatio=5,
                             mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY, clahe=True)),
    ("sgbm_b9_hh",      dict(blockSize=9,  numDisparities=192, uniquenessRatio=5,
                             mode=cv2.STEREO_SGBM_MODE_HH, clahe=True)),
    ("sgbm_b11_clahe",  dict(blockSize=11, numDisparities=192, uniquenessRatio=3,
                             mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY, clahe=True)),
]


def matcher_from(c):
    b = c["blockSize"]
    return cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=c["numDisparities"], blockSize=b,
        P1=8 * b * b, P2=32 * b * b, disp12MaxDiff=1,
        uniquenessRatio=c["uniquenessRatio"], speckleWindowSize=100, speckleRange=2,
        preFilterCap=63, mode=c["mode"])


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "oak_v4_evidence/f17/raw/sweep1"
    K, D, E = load_calib(root)
    rect = {"BC": rectify_pair(K, D, E, "B", "C"),
            "AC": rectify_pair(K, D, E, "A", "C")}

    man = [json.loads(x) for x in io.open(os.path.join(root, "manifest.jsonl"), encoding="utf-8")
           if x.strip()]
    say("=" * 96)
    say("F-17 MATCHER VIABILITY / TUNING  (offline, on captured frames)")
    say("=" * 96)
    say("capture : %s   frame sets: %d" % (root, len(man)))
    for k, r in rect.items():
        say("  pair %-3s  B_rect %8.3f mm   f_rect %9.4f px   f*B %10.1f"
            % (k, r["B"], r["f"], r["fB"]))

    blocks = sorted({m["block"] for m in man})
    sel = []
    for b in blocks:
        g = [m for m in man if m["block"] == b]
        sel.extend(g[::max(1, len(g) // 6)][:6])
    say("  evaluating %d frames spread over %s" % (len(sel), ", ".join(blocks)))
    say()
    say("  %-18s %-30s %-30s" % ("config", "pair BC (74.99 mm)", "pair AC (37.64 mm)"))

    clahe_op = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    for name, c in CONFIGS:
        m = matcher_from(c)
        st = {"BC": [[], []], "AC": [[], []]}
        for rec in sel:
            bdir = os.path.join(root, rec["block"])
            i = rec["i"]
            a = cv2.imread(os.path.join(bdir, "a_%04d.png" % i), cv2.IMREAD_COLOR)
            gb = cv2.imread(os.path.join(bdir, "b_%04d.png" % i), cv2.IMREAD_GRAYSCALE)
            gc = cv2.imread(os.path.join(bdir, "c_%04d.png" % i), cv2.IMREAD_GRAYSCALE)
            if a is None or gb is None or gc is None:
                continue
            gray = {"A": cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), "B": gb, "C": gc}
            if c["clahe"]:
                gray = {k: clahe_op.apply(v) for k, v in gray.items()}
            for pk, (ln, rn) in (("BC", ("B", "C")), ("AC", ("A", "C"))):
                r = rect[pk]
                li = cv2.remap(gray[ln], r["m1"][0], r["m1"][1], cv2.INTER_LINEAR)
                ri = cv2.remap(gray[rn], r["m2"][0], r["m2"][1], cv2.INTER_LINEAR)
                dep = RIG.depth_from_disp(m.compute(li, ri), r["fB"])
                h, w = dep.shape
                roi = dep[int(h * .2):int(h * .9), int(w * .2):int(w * .8)]
                band = dep[int(h * .25):int(h * .55), int(w * .35):int(w * .65)]
                st[pk][0].append(float((roi > 0).mean()))
                st[pk][1].append(float((band > 0).mean()))
        say("  %-18s frame %5.1f%%  torso %5.1f%%        frame %5.1f%%  torso %5.1f%%"
            % (name, 100 * np.mean(st["BC"][0]), 100 * np.mean(st["BC"][1]),
               100 * np.mean(st["AC"][0]), 100 * np.mean(st["AC"][1])))

    io.open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
