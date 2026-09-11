#!/usr/bin/env python3
"""
F-17 - validate the host multi-baseline rig BEFORE spending any subject time.

A host-rectified stereo pair is only worth measuring if it agrees with the device on the same
scene. This runs all three depth sources on a static scene and compares them:

    device  BC 74.99 mm   OAK StereoDepth, sub-pixel 1/8, aligned to CAM_A
    host    BC 74.99 mm   cv2 rectify + SGBM
    host    AC 37.64 mm   cv2 rectify + SGBM, identical matcher

If host-BC does not track the device, the rectification or the matcher is wrong and nothing
downstream can be trusted. No subject required.

Output: oak_v4_evidence/f17/rig_validation.txt
"""
import io
import os
import time

import numpy as np
import cv2
import depthai as dai

import f17_stereo_rig as RIG

OUT = os.path.join("oak_v4_evidence", "f17", "rig_validation.txt")
L = []


def say(s=""):
    print(s)
    L.append(s)


say("=" * 96)
say("F-17 HOST STEREO RIG VALIDATION  (static scene, no subject)")
say("=" * 96)

pipe = RIG.build_pipeline(subpixel=True, bits=3)
matcher = RIG.make_matcher()

with dai.Device(pipe) as dev:
    calib = dev.readCalibration()
    rect = RIG.build_rectification(calib)

    say()
    say("rectified geometry derived from the device's own EEPROM:")
    say("  %-6s %-12s %12s %12s %14s" % ("pair", "cams", "B_rect mm", "f_rect px", "f*B mm*px"))
    for k, r in rect.items():
        say("  %-6s %-12s %12.3f %12.4f %14.1f"
            % (k, "%s(L) %s(R)" % (r["left"], r["right"]), r["B"], r["f"], r["fB"]))
    say()
    say("  device StereoDepth f*B (CAM_C fx x 74.99 mm) = 21222.2 mm*px  [F-16 measured]")

    q = {n: dev.getOutputQueue(n, maxSize=4, blocking=False)
         for n in ("rgb", "monoB", "monoC", "depth")}

    t0 = time.time()
    while time.time() - t0 < 1.5:
        for v in q.values():
            v.tryGet()

    acc = {"device": [], "hostBC": [], "hostAC": []}
    cov = {"device": [], "hostBC": [], "hostAC": []}
    t_host = []
    n = 0
    t0 = time.time()
    while time.time() - t0 < 10.0:
        rgb = q["rgb"].get().getCvFrame()
        gb = q["monoB"].get().getCvFrame()
        gc = q["monoC"].get().getCvFrame()
        dd = q["depth"].get().getFrame()
        n += 1

        ga = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        gray = {"A": ga, "B": gb, "C": gc}

        t_s = time.time()
        host = {}
        for k, r in rect.items():
            li = cv2.remap(gray[r["left"]], r["m1"][0], r["m1"][1], cv2.INTER_LINEAR)
            ri = cv2.remap(gray[r["right"]], r["m2"][0], r["m2"][1], cv2.INTER_LINEAR)
            disp = matcher.compute(li, ri)
            host[k] = RIG.depth_from_disp(disp, r["fB"])
        t_host.append(time.time() - t_s)

        def roi(a):
            h, w = a.shape
            return a[int(h * .3):int(h * .7), int(w * .3):int(w * .7)]

        for name, arr in (("device", dd.astype(np.float32)),
                          ("hostBC", host["BC"]), ("hostAC", host["AC"])):
            r_ = roi(arr)
            nz = r_[r_ > 0]
            cov[name].append(nz.size / float(r_.size))
            if nz.size:
                acc[name].append(float(np.median(nz)))

say()
say("frames: %d    host stereo cost: %.1f ms/frame for BOTH pairs (%.1f fps ceiling)"
    % (n, 1000.0 * np.mean(t_host), 1.0 / np.mean(t_host)))
say()
say("central-ROI median depth over the run (the three sources looking at the same scene):")
say("  %-10s %12s %12s %12s %10s" % ("source", "median mm", "p05", "p95", "coverage"))
for name in ("device", "hostBC", "hostAC"):
    a = np.array(acc[name], dtype=float)
    if not a.size:
        say("  %-10s  no valid depth" % name)
        continue
    say("  %-10s %12.1f %12.1f %12.1f %9.1f%%"
        % (name, np.median(a), np.percentile(a, 5), np.percentile(a, 95),
           100.0 * np.mean(cov[name])))

d = np.median(acc["device"]) if acc["device"] else float("nan")
hb = np.median(acc["hostBC"]) if acc["hostBC"] else float("nan")
ha = np.median(acc["hostAC"]) if acc["hostAC"] else float("nan")
say()
say("AGREEMENT CHECKS")
say("  host BC vs device  : %+.1f mm  (%.2f %%)" % (hb - d, 100.0 * (hb - d) / d))
say("  host AC vs host BC : %+.1f mm  (%.2f %%)" % (ha - hb, 100.0 * (ha - hb) / hb))
say()
say("  host BC and the device share a baseline and should agree closely; a large gap means the")
say("  host rectification or matcher is wrong. host AC sees the same scene from half the")
say("  baseline and should agree on DEPTH while being ~2x coarser in resolution.")

io.open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\nwrote %s" % OUT)
