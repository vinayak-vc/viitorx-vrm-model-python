#!/usr/bin/env python3
"""
F-17 step 1 - hardware inventory + every stereo pair this hardware physically supports.

Section 4 of the brief says: inventory what stereo hardware is ACTUALLY available before
predicting anything. This reads the EEPROM extrinsics so the available baselines are measured,
not assumed from a product page.

Output: oak_v4_evidence/f17/inventory.txt
"""
import io
import math
import os
import sys

import numpy as np
import depthai as dai

OUT = os.path.join("oak_v4_evidence", "f17", "inventory.txt")
L = []


def say(s=""):
    print(s)
    L.append(s)


def hdr(s):
    say()
    say("=" * 88)
    say(s)
    say("=" * 88)


hdr("F-17 HARDWARE INVENTORY")
say("depthai %s" % dai.__version__)

infos = dai.Device.getAllAvailableDevices()
say()
say("USB/network stereo devices visible to depthai: %d" % len(infos))
for i in infos:
    say("   name=%-8s mxid=%-22s state=%s protocol=%s"
        % (i.name, i.getMxId(), i.state, i.protocol))
if not infos:
    say("NONE - cannot continue")
    io.open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    sys.exit(2)
if len(infos) == 1:
    say()
    say("ONLY ONE STEREO DEVICE IS PRESENT. No second/wider-baseline camera is attached, so the")
    say("brief's candidate list (OAK-D-LR, OAK-D-W, custom pair, second camera) cannot be measured")
    say("directly. Section 4 asks for the inventory first precisely so this is stated, not assumed.")

with dai.Device() as dev:
    calib = dev.readCalibration()
    hdr("DEVICE")
    say("device name  : %s" % dev.getDeviceName())
    e = calib.getEepromData()
    say("board        : %s rev %s   product %s" % (e.boardName, e.boardRev, e.productName))
    say("cameras      : %s" % ", ".join(str(c) for c in dev.getConnectedCameras()))
    say("stereo pair set in calibration: left=%s right=%s"
        % (calib.getStereoLeftCameraId(), calib.getStereoRightCameraId()))
    say("getBaselineDistance(): %.4f cm = %.2f mm"
        % (calib.getBaselineDistance(), calib.getBaselineDistance() * 10.0))

    # ---------------------------------------------------------------- extrinsics between sockets
    hdr("MEASURED EXTRINSICS - every camera pair on this board")
    socks = [("CAM_A", dai.CameraBoardSocket.CAM_A),
             ("CAM_B", dai.CameraBoardSocket.CAM_B),
             ("CAM_C", dai.CameraBoardSocket.CAM_C)]
    say("%-14s %10s %10s %10s %12s" % ("pair", "tx mm", "ty mm", "tz mm", "|T| mm"))
    pairs = {}
    for i in range(len(socks)):
        for j in range(len(socks)):
            if i >= j:
                continue
            an, a = socks[i]
            bn, b = socks[j]
            try:
                # getCameraTranslationVector returns CENTIMETRES; convert once, here.
                T = np.array(calib.getCameraTranslationVector(a, b, False),
                             dtype=np.float64) * 10.0
                mag = float(np.linalg.norm(T))
                pairs["%s-%s" % (an, bn)] = (T, mag)
                say("%-14s %10.3f %10.3f %10.3f %12.3f"
                    % ("%s->%s" % (an, bn), T[0], T[1], T[2], mag))
            except Exception as ex:
                say("%-14s  <%s>" % ("%s->%s" % (an, bn), str(ex)[:50]))

    # ---------------------------------------------------------------- f*B for each pair
    hdr("STEREO PAIRS AVAILABLE ON THIS HARDWARE  (f*B at 640x400)")
    fx = {}
    for n, s in socks:
        try:
            m = np.array(calib.getCameraIntrinsics(s, 640, 400), dtype=np.float64)
            fx[n] = m[0][0]
        except Exception:
            pass
    say("%-14s %10s %12s %14s %16s" % ("pair", "B mm", "fx px", "f*B mm*px", "vs B-C"))
    base_fb = None
    for k in sorted(pairs):
        T, mag = pairs[k]
        rect_fx = fx.get(k.split("-")[1], float("nan"))
        fb = rect_fx * mag
        if k == "CAM_B-CAM_C":
            base_fb = fb
    for k in sorted(pairs):
        T, mag = pairs[k]
        rect_fx = fx.get(k.split("-")[1], float("nan"))
        fb = rect_fx * mag
        say("%-14s %10.3f %12.4f %14.1f %16s"
            % (k, mag, rect_fx, fb,
               ("%.3fx" % (fb / base_fb)) if base_fb else "-"))

    say()
    say("NOTE: only horizontally-separated, rectifiable pairs are usable for StereoDepth. The")
    say("|ty| and |tz| components show how far each pair is from a pure horizontal baseline.")
    for k in sorted(pairs):
        T, mag = pairs[k]
        say("   %-14s horizontal fraction |tx|/|T| = %.4f   (vertical %.4f, depth %.4f)"
            % (k, abs(T[0]) / mag, abs(T[1]) / mag, abs(T[2]) / mag))

io.open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\nwrote %s" % OUT)
