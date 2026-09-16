#!/usr/bin/env python3
"""
F-16 - is the residual ~0.75 px disparity error an OCCLUSION-EDGE artefact?

The shoulder keypoints sit on the left and right silhouette edges of the body, which is where
block-matching stereo is worst: the two mono cameras see different amounts of background beside
each shoulder, so the matcher can bias one edge foreground and the other background.

If that is the mechanism, then on each frame the shoulder reading FARTHER should show the wider /
more contaminated depth window. Tested on data already captured -- no extra subject time.

Output: evidence/oak_v4/f16/edge_check.txt
"""

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(_os.path.join(_d, "_sidecar_path.py")):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import _sidecar_path  # noqa: F401  - puts the sidecar root and every tools/ group on sys.path
import evidence_paths as EV

import glob
import io
import json
import os

import numpy as np

import f16_configs as C

OUT = EV.oak_v4("f16", "edge_check.txt")
CONF = 0.30
out = []


def say(s=""):
    print(s)
    out.append(s)


def load(pat):
    """Thin alias for the shared loader in f16_configs, kept so the call sites below read the same
    as they did when every F-16 script carried its own copy."""
    return C.load_rows(pat, CONF)


say("=" * 100)
say("F-16 OCCLUSION-EDGE CHECK")
say("=" * 100)
say("Hypothesis: the shoulder that reads FARTHER does so because its 5x5 depth window straddles")
say("the body/background edge. If true, the farther window is the wider / lower-quality one.")

for name, pat in (("sub-pixel distance sweep", EV.oak_v4("f16/autosweep_sub3.jsonl")),
                  ("config sweep 1.33 m", EV.oak_v4("f16/cfgsweep_133.jsonl")),
                  ("pose protocol 1.33 m", EV.oak_v4("f16/pose_sub3_133.jsonl"))):
    rows = load(pat)
    if not rows:
        continue
    say()
    say("-" * 100)
    say("%s   (n=%d)" % (name, len(rows)))
    say("-" * 100)

    far_wider = 0
    far_narrower = 0
    tie = 0
    spreads_far = []
    spreads_near = []
    q_far = []
    q_near = []
    for r in rows:
        wl = r.get("wL") or []
        wr = r.get("wR") or []
        if len(wl) < 2 or len(wr) < 2:
            continue
        sl = max(wl) - min(wl)
        sr = max(wr) - min(wr)
        if r["zR"] > r["zL"]:            # right shoulder is the FARTHER one
            sf, sn, qf, qn = sr, sl, r["qR"], r["qL"]
        elif r["zL"] > r["zR"]:
            sf, sn, qf, qn = sl, sr, r["qL"], r["qR"]
        else:
            continue
        spreads_far.append(sf)
        spreads_near.append(sn)
        q_far.append(qf)
        q_near.append(qn)
        if sf > sn:
            far_wider += 1
        elif sf < sn:
            far_narrower += 1
        else:
            tie += 1

    n = far_wider + far_narrower + tie
    if not n:
        say("  no usable frames")
        continue
    say("  frames where the FARTHER shoulder had the WIDER depth window : %5d  (%.1f %%)"
        % (far_wider, 100.0 * far_wider / n))
    say("  frames where it had the NARROWER window                      : %5d  (%.1f %%)"
        % (far_narrower, 100.0 * far_narrower / n))
    say("  ties                                                         : %5d  (%.1f %%)"
        % (tie, 100.0 * tie / n))
    say("  window spread (mm)   farther p50 %7.1f   nearer p50 %7.1f"
        % (np.median(spreads_far), np.median(spreads_near)))
    say("  F-08 depth quality   farther p50 %7.3f   nearer p50 %7.3f"
        % (np.median(q_far), np.median(q_near)))

say()
say("=" * 100)
say("WHICH ANATOMICAL SIDE READS FARTHER, AND DOES IT TRACK THE IMAGE SIDE?")
say("=" * 100)
rows = load(EV.oak_v4("f16/pose_sub3_133.jsonl"))
by = {}
for r in rows:
    by.setdefault(r["block"], []).append(r)
say("%-9s %9s %11s %11s %9s" % ("block", "L at u", "dz_p50", "farther is", "n"))
for blk in ("sq_a", "sq_b", "sq_c", "back"):
    g = by.get(blk)
    if not g:
        continue
    uL = np.median([r["uL"] for r in g])
    uR = np.median([r["uR"] for r in g])
    dz = np.median([r["dz"] for r in g])
    side = "image-%s" % ("RIGHT" if ((dz > 0) == (uR > uL)) else "LEFT")
    say("%-9s %9s %11.1f %11s %9d"
        % (blk, "right" if uL > uR else "left", dz,
           ("anat-R " if dz > 0 else "anat-L ") + side, len(g)))
say()
say("If 'farther' keeps the same ANATOMICAL side across the 180 deg turn but swaps IMAGE side,")
say("the bias is tied to the body, not to the sensor's left/right. The reverse means the opposite.")

io.open(OUT, "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\nwrote %s" % OUT)
