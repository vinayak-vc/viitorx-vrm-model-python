#!/usr/bin/env python3
"""F-44 - what does raising the resolution actually cost, and what does it buy?

Measures the CAMERA PIPELINE only. The pose model is deliberately out of the loop because its cost
is constant: RTMW3D always resizes its crop to 288x384 and the detector always runs at 544x320, so
neither sees the source resolution. What changes is the VPU's stereo matcher and the USB link, and
that is exactly what this measures.

Reported per config:
  fps            sustained RGB + depth delivery over the measurement window
  valid%         fraction of depth pixels with a non-zero reading -- the IR projector and the
                 mono resolution both move this, and it is the number RETARGET_AUDIT's
                 "~63% of keypoints get measured depth" ultimately comes from
  px/m @2m,4m    pixels a 1.7 m body occupies, which is what the pose model actually sees
"""
import sys, time, os
sys.path.insert(0, r"D:\Unity\viitorx-vrm-avtar-unity-base-project\Assets\Games\viitorx-vrm-avtar-unity\python-sidecar~")

import numpy as np
import depthai as dai
import oak_depth as D

WARM = 30          # frames discarded before timing (pipeline start-up, AE settle)
MEASURE = 150      # ~5 s at 30 fps

CONFIGS = [
    ("400p", (1, 2), "PRODUCTION as shipped"),
    ("800p", (1, 2), "mono 800p only"),
    ("400p", (1, 1), "rgb native only"),
    ("800p", (1, 1), "both"),
]


def run(mono_res, isp, ir_on=True):
    D.STEREO_CONFIG["monoRes"] = mono_res
    D.STEREO_CONFIG["rgbIsp"] = isp
    D.STEREO_CONFIG["subpixel"] = True
    D.STEREO_CONFIG["subpixelBits"] = 3

    pipeline = D.build_rgbd_pipeline()
    with dai.Device(pipeline) as dev:
        ir = D.enable_ir_dot_projector(dev, 0.8 if ir_on else 0.0)
        q_rgb = dev.getOutputQueue("rgb", 4, False)
        q_depth = dev.getOutputQueue("depth", 4, False)

        rgb_shape = depth_shape = None
        valid = []
        for _ in range(WARM):
            rgb_shape = q_rgb.get().getCvFrame().shape[:2]
            depth_shape = q_depth.get().getFrame().shape

        t0 = time.time()
        for _ in range(MEASURE):
            q_rgb.get().getCvFrame()
            d = q_depth.get().getFrame()
            valid.append(float(np.count_nonzero(d)) / d.size)
        dt = time.time() - t0

    intr = None
    return {
        "fps": MEASURE / dt,
        "valid": 100.0 * float(np.mean(valid)),
        "rgb": rgb_shape,
        "depth": depth_shape,
        "ir": ir,
    }


def main():
    print("F-44 resolution A/B  (sub-pixel 1/8 + IR dot ON in every row)")
    print("=" * 96)
    print("%-22s %-12s %-12s %7s %8s %9s %9s" %
          ("config", "rgb", "depth", "fps", "valid%", "px@2m", "px@4m"))
    print("-" * 96)
    rows = []
    for mono_res, isp, label in CONFIGS:
        try:
            r = run(mono_res, isp)
        except Exception as e:
            print("%-22s FAILED: %s" % (label, e))
            continue
        h, w = r["rgb"]
        # A 1.7 m body, through the RGB focal length implied by this frame height.
        # fy scales linearly with the frame's long axis: 284.6 px at 640 tall.
        fy = 284.627 * (max(w, h) / 640.0)
        px2 = 1.7 * fy / 2.0
        px4 = 1.7 * fy / 4.0
        rows.append((label, r, px2, px4))
        print("%-22s %-12s %-12s %7.1f %8.1f %9.0f %9.0f" %
              (label, "%dx%d" % (w, h), "%dx%d" % (r["depth"][1], r["depth"][0]),
               r["fps"], r["valid"], px2, px4))
        print("%-22s   IR: %s" % ("", r["ir"]))
        time.sleep(1.0)

    print("-" * 96)
    if rows:
        base = rows[0][1]
        for label, r, px2, px4 in rows[1:]:
            print("%-22s fps %+.1f%%   valid %+.1f pp   px-on-target %.2fx"
                  % (label, 100.0 * (r["fps"] - base["fps"]) / base["fps"],
                     r["valid"] - base["valid"],
                     (1.7 * 284.627 * (max(r["rgb"]) / 640.0)) /
                     (1.7 * 284.627 * (max(base["rgb"]) / 640.0))))


if __name__ == "__main__":
    main()
