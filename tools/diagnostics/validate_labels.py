#!/usr/bin/env python3

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(_os.path.join(_d, "_sidecar_path.py")):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import _sidecar_path  # noqa: F401  - puts the sidecar root and every tools/ group on sys.path
"""Validate f21_ground_truth.py against frames checked BY EYE, and report its error rate.

An instrument used to score a safety property has to be scored itself first, or a labelling bug gets
reported as a pipeline defect. That already happened once in this work: the first version of the
labeller called six frames of 123.webm a WRONG-PERSON EMISSION while the woman was alone in the room
and the pipeline was correctly locked on her (see the module docstring in f21_ground_truth.py).

TRUTH below is not derived from the pipeline, the labeller, or any threshold. Every entry was read
off the two contact sheets written to oak_v4_evidence/f21/labels/ and checked by eye:

    123_contact_sheet.png   f0..f774 at 50-frame intervals, plus the hand-off region
    123_gaps.png            f190..f270 and f430..f520, the two empty-room stretches

    W     the woman (green t-shirt) is the only person in frame
    M     the man (orange/cream striped shirt) is the only person in frame
    WM    both are in frame
    -     the room is empty

    python validate_labels.py
"""
import os
import sys

import cv2

from f21_ground_truth import GroundTruth, classify

VIDEO = os.path.join("D:" + os.sep, "Unity", "viitorx-vrm-avtar-unity-base-project", "Assets",
                     "Games", "video", "123.webm")

# 123.webm: the woman's green shirt is the one colour belonging to exactly one person in the scene.
WINDOWS = [("W", 55, 90, 0.15)]
DEFAULT = "M"

TRUTH = {
    0: "W", 50: "W", 100: "W", 111: "W", 128: "W", 150: "W", 190: "W",
    210: "-", 220: "-", 230: "-", 240: "-", 250: "-", 260: "-",
    300: "W", 350: "W", 400: "W", 430: "W",
    450: "-", 460: "-", 470: "-", 480: "-", 490: "-", 500: "-",
    520: "W", 525: "W", 530: "W", 550: "W", 594: "W",
    600: "WM", 620: "WM", 626: "WM", 630: "WM", 650: "WM", 660: "WM", 690: "WM",
    712: "M", 720: "M",
    740: "-", 750: "-", 760: "-", 774: "-",
}


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else VIDEO
    gt = GroundTruth(video)
    cap = cv2.VideoCapture(video)
    i = 0
    rows = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in TRUTH:
            labs = [classify(b, WINDOWS, DEFAULT)[0] for b in gt.blobs(f)]
            rows.append((i, TRUTH[i], labs))
        i += 1
    cap.release()

    print("=" * 92)
    print(" f21_ground_truth.py validation against %d eye-checked frames of %s"
          % (len(TRUTH), os.path.basename(video)))
    print("=" * 92)
    ok_n = 0
    fails = []
    for fi, truth, labs in rows:
        got = set(l for l in labs if l != "?")
        if truth == "-":
            good = not labs
            want = "no blob"
        elif truth == "WM":
            good = ("W" in got and "M" in got)
            want = "W and M"
        else:
            good = (got == set([truth]))
            want = "exactly %s" % truth
        # Two error classes, and only one of them can corrupt the metric:
        #   CONSERVATIVE - a person who is present was not labelled at all (dropped below the area
        #                  floor while half out of frame). wrong_person() excludes unlabelled frames,
        #                  so this costs coverage and can never invent a wrong-person frame.
        #   DANGEROUS    - somebody was labelled as the OTHER person. This one can.
        got_list = [l for l in labs if l != "?"]
        if truth == "-":
            dangerous = bool(got_list)
        else:
            allowed = set(truth) if truth == "WM" else {truth}
            dangerous = any(l not in allowed for l in got_list)
        ok_n += good
        flag = "ok " if good else ("BAD" if dangerous else "cons")
        print("  %s f%-5d truth=%-3s want %-10s got %s" % (flag, fi, truth, want, labs or "[]"))
        if not good:
            fails.append((fi, truth, labs, "DANGEROUS" if dangerous else "conservative"))
    print("-" * 92)
    print(" %d/%d correct  (%.1f%%)" % (ok_n, len(rows), 100.0 * ok_n / max(1, len(rows))))
    dang = [f for f in fails if f[3] == "DANGEROUS"]
    if fails:
        print(" residual errors, reported not hidden:")
        for fi, truth, labs, kind in fails:
            print("   f%-5d truth=%-3s got=%-22s %s" % (fi, truth, labs, kind))
    print(" %d conservative (person dropped -> frame excluded from scoring, cannot invent a "
          "wrong-person frame)" % (len(fails) - len(dang)))
    print(" %d DANGEROUS  (a person labelled as the other person)" % len(dang))
    print("=" * 92)
    return 0 if not dang else 1


if __name__ == "__main__":
    sys.exit(main())
