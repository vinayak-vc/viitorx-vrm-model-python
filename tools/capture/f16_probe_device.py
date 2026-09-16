#!/usr/bin/env python3
"""
F-16 step 1 - device / calibration probe.

Reads the OAK-D calibration and enumerates the stereo configuration options that the INSTALLED
depthai build actually exposes. Touches no production code and needs no subject in frame.

Output: oak_v4_evidence/f16/device_probe.txt
"""
import io
import os
import sys
import traceback

import numpy as np
import depthai as dai

OUT = os.path.join("oak_v4_evidence", "f16", "device_probe.txt")
L = []


def say(s=""):
    print(s)
    L.append(s)


def hdr(s):
    say()
    say("=" * 78)
    say(s)
    say("=" * 78)


hdr("F-16 DEVICE / CALIBRATION PROBE")
say("depthai runtime : %s" % dai.__version__)
say("python          : %s" % sys.version.split()[0])

# ---------------------------------------------------------------- enumerate SDK capabilities
hdr("1. SDK capability enumeration (what this depthai build exposes)")

mono_res = [n for n in dir(dai.MonoCameraProperties.SensorResolution) if n.startswith("THE_")]
say("MonoCamera resolutions : %s" % ", ".join(sorted(mono_res)))
col_res = [n for n in dir(dai.ColorCameraProperties.SensorResolution) if n.startswith("THE_")]
say("ColorCamera resolutions: %s" % ", ".join(sorted(col_res)))
presets = [n for n in dir(dai.node.StereoDepth.PresetMode) if not n.startswith("_")]
say("StereoDepth presets    : %s" % ", ".join(sorted(presets)))
try:
    med = [n for n in dir(dai.StereoDepthProperties.MedianFilter) if not n.startswith("_")]
    say("Median filters         : %s" % ", ".join(sorted(med)))
except Exception as e:
    say("Median filters         : <%s>" % e)

setters = [n for n in dir(dai.node.StereoDepth) if n.startswith("set")]
say("StereoDepth setters    : %s" % ", ".join(sorted(setters)))

# ---------------------------------------------------------------- device
hdr("2. Device")
infos = dai.Device.getAllAvailableDevices()
if not infos:
    say("NO DEVICE FOUND - cannot continue.")
    io.open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    sys.exit(2)
for i in infos:
    say("found: name=%s mxid=%s state=%s protocol=%s" % (i.name, i.getMxId(), i.state, i.protocol))

with dai.Device() as dev:
    say("device name      : %s" % dev.getDeviceName())
    try:
        say("board name       : %s" % dev.readCalibration().getEepromData().boardName)
        say("board rev        : %s" % dev.readCalibration().getEepromData().boardRev)
        say("product name     : %s" % dev.readCalibration().getEepromData().productName)
    except Exception as e:
        say("eeprom           : <%s>" % e)
    say("connected cameras: %s" % ", ".join(str(c) for c in dev.getConnectedCameras()))
    try:
        for f in dev.getConnectedCameraFeatures():
            say("  socket=%-7s sensor=%-10s width=%-5d height=%-5d supported=%s"
                % (f.socket, f.sensorName, f.width, f.height,
                   ",".join("%dx%d" % (c.width, c.height) for c in f.configs)))
    except Exception as e:
        say("camera features  : <%s>" % e)

    calib = dev.readCalibration()

    # ------------------------------------------------------------ baseline
    hdr("3. Stereo geometry (from on-device calibration EEPROM)")
    baseline_cm = calib.getBaselineDistance()
    say("getBaselineDistance()            : %.6f cm  = %.4f mm" % (baseline_cm, baseline_cm * 10.0))
    try:
        say("stereo left  socket             : %s" % calib.getStereoLeftCameraId())
        say("stereo right socket             : %s" % calib.getStereoRightCameraId())
    except Exception as e:
        say("stereo sockets                  : <%s>" % e)

    # ------------------------------------------------------------ intrinsics at every res we care about
    hdr("4. Intrinsics (fx in PIXELS at the given output size)")
    say("%-8s %-9s %9s %9s %9s %9s" % ("socket", "size", "fx", "fy", "cx", "cy"))
    sizes = [(640, 400), (640, 480), (1280, 720), (1280, 800)]
    fx_table = {}
    for sock_name in ("CAM_A", "CAM_B", "CAM_C"):
        sock = getattr(dai.CameraBoardSocket, sock_name)
        for (w, h) in sizes:
            try:
                m = np.array(calib.getCameraIntrinsics(sock, w, h), dtype=np.float64)
                fx_table[(sock_name, w, h)] = m[0][0]
                say("%-8s %-9s %9.3f %9.3f %9.3f %9.3f"
                    % (sock_name, "%dx%d" % (w, h), m[0][0], m[1][1], m[0][2], m[1][2]))
            except Exception as e:
                say("%-8s %-9s  <%s>" % (sock_name, "%dx%d" % (w, h), str(e)[:40]))

    # ------------------------------------------------------------ the f*B product
    hdr("5. f x B products  (depth_mm = f_px * B_mm / disparity)")
    B_mm = baseline_cm * 10.0
    say("baseline B = %.4f mm" % B_mm)
    say()
    say("%-8s %-9s %12s %16s" % ("socket", "size", "fx_px", "f*B (mm*px)"))
    for k in sorted(fx_table):
        sock_name, w, h = k
        say("%-8s %-9s %12.4f %16.1f" % (sock_name, "%dx%d" % (w, h), fx_table[k], fx_table[k] * B_mm))

    say()
    say("F-09..F-15 used an EMPIRICALLY FITTED f*B = 21216 mm*px (from observed depth quantisation).")
    key = ("CAM_C", 640, 400)
    if key in fx_table:
        fb = fx_table[key] * B_mm
        say("CAM_C @ 640x400 gives f*B = %.1f  -> ratio to empirical = %.4f" % (fb, fb / 21216.0))
    key = ("CAM_A", 640, 400)
    if key in fx_table:
        fb = fx_table[key] * B_mm
        say("CAM_A @ 640x400 gives f*B = %.1f  -> ratio to empirical = %.4f" % (fb, fb / 21216.0))

    # ------------------------------------------------------------ quantisation table
    hdr("6. THEORETICAL depth quantisation, integer disparity, mono 400P")
    key = ("CAM_C", 640, 400)
    fb = fx_table.get(key, 21216.0 / 1.0) * B_mm if key in fx_table else 21216.0
    say("using f*B = %.1f mm*px" % fb)
    say()
    say("%-8s %10s %10s %12s %12s" % ("Z (mm)", "d (px)", "d_round", "Z_quant", "step @Z"))
    for Z in (800, 1000, 1200, 1330, 1500, 1800, 2000, 2500, 3000):
        d = fb / Z
        dr = round(d)
        zq = fb / dr if dr > 0 else float("nan")
        step = fb / dr - fb / (dr + 1) if dr > 0 else float("nan")
        say("%-8d %10.3f %10d %12.1f %12.1f" % (Z, d, dr, zq, step))

    # ------------------------------------------------------------ subpixel availability
    hdr("7. Sub-pixel disparity availability")
    st = dai.node.StereoDepth
    say("setSubpixel present            : %s" % hasattr(st, "setSubpixel"))
    say("setSubpixelFractionalBits       : %s" % hasattr(st, "setSubpixelFractionalBits"))
    try:
        p = dai.Pipeline()
        s = p.create(dai.node.StereoDepth)
        cfg = s.initialConfig.get()
        say("initialConfig.algorithmControl fields:")
        for n in dir(cfg.algorithmControl):
            if n.startswith("_"):
                continue
            try:
                say("   %-42s = %s" % (n, getattr(cfg.algorithmControl, n)))
            except Exception:
                pass
        say()
        say("costMatching.enableCompanding   = %s" % cfg.costMatching.enableCompanding)
        say("costMatching.confidenceThreshold= %s" % cfg.costMatching.confidenceThreshold)
    except Exception as e:
        say("initialConfig probe failed: %s" % e)
        say(traceback.format_exc()[-900:])

io.open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\nwrote %s" % OUT)
