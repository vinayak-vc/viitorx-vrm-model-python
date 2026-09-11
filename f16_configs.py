#!/usr/bin/env python3
"""
F-16 shared stereo-configuration matrix.

An ISOLATED diagnostic builder. It deliberately does NOT modify oak_depth.build_rgbd_pipeline():
production keeps its exact pipeline, and `baseline` here is a byte-for-byte replica of it so every
other configuration is measured against the shipping one.
"""
import depthai as dai

MONO_RES = {
    "400p": (dai.MonoCameraProperties.SensorResolution.THE_400_P, 640, 400),
    "720p": (dai.MonoCameraProperties.SensorResolution.THE_720_P, 1280, 720),
    "800p": (dai.MonoCameraProperties.SensorResolution.THE_800_P, 1280, 800),
}

# fx (px) per socket at 640x400, read from this device's EEPROM (f16_probe_device.py).
# fx scales LINEARLY with output width, so fx(W) = FX640 * W / 640.
FX640_CAM_C = 282.9949
BASELINE_MM = 75.0


def fb_product(mono_w):
    """f*B in mm*px for the rectified-right camera at a given mono width."""
    return FX640_CAM_C * (mono_w / 640.0) * BASELINE_MM


def theoretical_step_mm(z_mm, mono_w, subpixel_bits=0):
    """Depth quantisation step at range z for one disparity increment.

    step = Z^2 * delta_d / (f*B), with delta_d = 1 px (integer) or 2^-bits (sub-pixel).
    """
    fb = fb_product(mono_w)
    dd = 1.0 if subpixel_bits <= 0 else 1.0 / float(1 << subpixel_bits)
    return (z_mm * z_mm) * dd / fb


# name -> dict. `align` RGB is the production path (depth reprojected into CAM_A).
RGB_ISP = {"640x400": (1, 2), "1280x800": (1, 1)}

CONFIGS = {
    # ---- the shipping configuration, replicated exactly -------------------------------------
    "baseline": dict(mono="400p", subpixel=False, bits=0, extended=False,
                     preset="HIGH_DENSITY", lrcheck=True, align="rgb",
                     note="PRODUCTION (oak_depth.build_rgbd_pipeline)"),
    # ---- single-variable changes -------------------------------------------------------------
    "sub3": dict(mono="400p", subpixel=True, bits=3, extended=False,
                 preset="HIGH_DENSITY", lrcheck=True, align="rgb",
                 note="+ sub-pixel 1/8 px"),
    "sub5": dict(mono="400p", subpixel=True, bits=5, extended=False,
                 preset="HIGH_DENSITY", lrcheck=True, align="rgb",
                 note="+ sub-pixel 1/32 px"),
    "mono800": dict(mono="800p", subpixel=False, bits=0, extended=False,
                    preset="HIGH_DENSITY", lrcheck=True, align="rgb",
                    note="+ mono 1280x800 (2x fx)"),
    "accuracy": dict(mono="400p", subpixel=False, bits=0, extended=False,
                     preset="HIGH_ACCURACY", lrcheck=True, align="rgb",
                     note="+ HIGH_ACCURACY preset"),
    "noalign": dict(mono="400p", subpixel=False, bits=0, extended=False,
                    preset="HIGH_DENSITY", lrcheck=True, align="right",
                    note="DIAG: depth NOT reprojected into RGB"),
    # ---- combinations -------------------------------------------------------------------------
    "mono800_sub3": dict(mono="800p", subpixel=True, bits=3, extended=False,
                         preset="HIGH_DENSITY", lrcheck=True, align="rgb",
                         note="1280x800 + sub-pixel 1/8"),
    "mono800_sub5": dict(mono="800p", subpixel=True, bits=5, extended=False,
                         preset="HIGH_DENSITY", lrcheck=True, align="rgb",
                         note="1280x800 + sub-pixel 1/32"),
    "mono800_sub3_ext": dict(mono="800p", subpixel=True, bits=3, extended=True,
                             preset="HIGH_DENSITY", lrcheck=True, align="rgb",
                             note="1280x800 + sub-pixel + extended disparity"),
    # ---- RGB-resolution variants (change the POSE/pixel side, not the depth ladder) ------------
    "rgb800": dict(mono="400p", subpixel=False, bits=0, extended=False,
                   preset="HIGH_DENSITY", lrcheck=True, align="rgb", rgb="1280x800",
                   note="RGB 1280x800 (2x shoulder pixel span), stereo unchanged"),
    "best": dict(mono="800p", subpixel=True, bits=3, extended=False,
                 preset="HIGH_DENSITY", lrcheck=True, align="rgb", rgb="1280x800",
                 note="RGB 1280x800 + mono 1280x800 + sub-pixel 1/8"),
    "sub3_rgb800": dict(mono="400p", subpixel=True, bits=3, extended=False,
                        preset="HIGH_DENSITY", lrcheck=True, align="rgb", rgb="1280x800",
                        note="RGB 1280x800 + sub-pixel 1/8 at mono 400p (cheap best)"),
}


def build(cfg_name, rgb_fps=30, mono_fps=30):
    """Return (pipeline, cfg, mono_w, mono_h, rgb_w, rgb_h)."""
    cfg = CONFIGS[cfg_name]
    mono_enum, mw, mh = MONO_RES[cfg["mono"]]
    rgb_key = cfg.get("rgb", "640x400")
    isp_n, isp_d = RGB_ISP[rgb_key]
    rgb_w, rgb_h = (int(v) for v in rgb_key.split("x"))

    p = dai.Pipeline()

    cam = p.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
    cam.setIspScale(isp_n, isp_d)             # production = 1/2 -> 640x400 full FOV
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(rgb_fps)

    ml = p.create(dai.node.MonoCamera)
    mr = p.create(dai.node.MonoCamera)
    for m, sock in ((ml, dai.CameraBoardSocket.CAM_B), (mr, dai.CameraBoardSocket.CAM_C)):
        m.setResolution(mono_enum)
        m.setBoardSocket(sock)
        m.setFps(mono_fps)

    st = p.create(dai.node.StereoDepth)
    st.setDefaultProfilePreset(getattr(dai.node.StereoDepth.PresetMode, cfg["preset"]))
    st.setLeftRightCheck(cfg["lrcheck"])
    st.setSubpixel(cfg["subpixel"])
    if cfg["subpixel"] and cfg["bits"]:
        st.setSubpixelFractionalBits(cfg["bits"])
    st.setExtendedDisparity(cfg["extended"])
    if cfg["align"] == "rgb":
        st.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    else:
        st.setDepthAlign(dai.StereoDepthProperties.DepthAlign.RECTIFIED_RIGHT)
    ml.out.link(st.left)
    mr.out.link(st.right)

    xr = p.create(dai.node.XLinkOut)
    xr.setStreamName("rgb")
    cam.isp.link(xr.input)

    xd = p.create(dai.node.XLinkOut)
    xd.setStreamName("depth")
    st.depth.link(xd.input)

    return p, cfg, mw, mh, rgb_w, rgb_h
