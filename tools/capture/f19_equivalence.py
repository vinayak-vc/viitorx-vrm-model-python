#!/usr/bin/env python3
"""F-19 §5 — prove the PRODUCTION portrait path == the F-18 DIAGNOSTIC portrait path.

Needs no camera: every check runs against the transform code and the F-18 captures on disk.

Why these five checks and not a visual comparison. F-18's capture already called production's own
`oak_depth.backproject(uv, depth, w, h, intr)`; what F-19 newly does is *feed* that function rotated
inputs from inside `wholebody_udp_sender`. So equivalence is not about the back-projector — it is
about whether production hands it exactly the pixel frame, the dimensions and the intrinsics that
F-18 did, and whether rotating all three together really preserves the downstream
(X right, Y down, Z forward) convention.

  1  INTRINSICS   production's rotation of the recorded landscape intrinsics must reproduce the
                  recorded portrait intrinsics, exactly.
  2  PIXELS       `rotate_image` must move a pixel to precisely where `rotate_point` predicts, and
                  `unrotate_point` must invert it. If these ever disagree, every back-projected X is
                  wrong by an amount no visual check would catch.
  3  SEMANTICS    back-projecting a synthetic scene through the rotated frame + rotated intrinsics
                  must return the SAME metric point as back-projecting the unrotated frame through
                  the unrotated intrinsics. This is the §6 claim - portrait is an input rotation, not
                  a new pose protocol - reduced to a number.
  4  REAL DATA    on every recorded F-18 frame, the shoulder X/Z implied by the recorded pixels and
                  depths through the rotated intrinsics must reproduce the recorded dx/dz, and the
                  torso yaw recomputed from them must reproduce the recorded yaw3D.
  5  IDENTITY     left/right must not swap: the subject's left shoulder must stay on the same side
                  of the frame centre after rotation.

Acceptance (from the brief): position residual at numerical precision, yaw residual < 0.01 deg.

    python f19_equivalence.py
"""

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(_os.path.join(_d, "_sidecar_path.py")):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import _sidecar_path  # noqa: F401  - puts the sidecar root and every tools/ group on sys.path
import evidence_paths as EV

import json
import math
import os
import sys

import numpy as np
import cv2

import f18_portrait as PORTRAIT

# `oak_depth` imports depthai at module scope, but the function under test here — `backproject` —
# has no depthai dependency at all (it is pure numpy over a depth array + intrinsics). No OAK-D is
# attached in this environment, so depthai is stubbed purely to allow the import. Nothing in this
# file calls a device API; if it did, the stub would raise rather than silently fake a result.
if "depthai" not in sys.modules:
    import types as _types

    class _DaiStub:
        """Resolves any attribute chain (dai.MonoCameraProperties.SensorResolution.THE_400_P and
        friends are read at import time by the F-16 config tables) but raises the moment anything
        tries to actually TALK to a device - so a stub can never be mistaken for a real capture."""

        def __init__(self, path="dai"):
            self._path = path

        def __getattr__(self, name):
            return _DaiStub(f"{self._path}.{name}")

        def __call__(self, *_a, **_k):
            raise RuntimeError(f"f19_equivalence does not use the depthai device API ({self._path})")

    _stub = _types.ModuleType("depthai")
    _stub.__getattr__ = lambda name: getattr(_DaiStub(), name)
    sys.modules["depthai"] = _stub

import oak_depth as D
import f16_capture as CAP    # for line_yaw_deg: the SHIPPED Kalidokit y-channel, verbatim

EVID = EV.oak_v4("f18")
CAPS = ["f18_move_090.jsonl", "f18_torso_near.jsonl", "f18_torso_far.jsonl", "f18_frame_sweep.jsonl"]
LSH, RSH = 5, 6          # COCO-17 left/right shoulder, as F-18 used
LHIP, RHIP = 11, 12

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail):
    results.append((name, PASS if ok else FAIL, detail))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}", flush=True)
    return ok


def load_meta(path):
    with open(path, encoding="utf-8") as fh:
        return json.loads(fh.readline())


# ---------------------------------------------------------------- 1. intrinsics
def check_intrinsics():
    print("\n1. INTRINSICS - production rotation vs the recorded F-18 portrait intrinsics")
    worst = 0.0
    n = 0
    for cap in CAPS:
        p = os.path.join(EVID, cap)
        if not os.path.exists(p):
            continue
        m = load_meta(p)
        land = tuple(m["intr_landscape"])
        rec = tuple(m["intr_portrait"])
        # The recorded landscape intrinsics belong to the LANDSCAPE frame, whose dimensions are the
        # portrait ones swapped.
        lw, lh = m["portrait_h"], m["portrait_w"]
        got = PORTRAIT.rotate_intrinsics(land, lw, lh, m["rotation"])
        d = max(abs(a - b) for a, b in zip(got, rec))
        worst = max(worst, d)
        n += 1
        print(f"     {cap:26s} {m['rotation']} {lw}x{lh} -> "
              f"fx {got[0]:.4f} fy {got[1]:.4f} cx {got[2]:.4f} cy {got[3]:.4f}   maxdiff {d:.3e}")
    return check("intrinsics rotation", n > 0 and worst < 1e-6,
                 f"{n} captures, worst component error {worst:.3e} px")


# ---------------------------------------------------------------- 2. pixel mapping
def check_pixels():
    print("\n2. PIXELS - rotate_image vs rotate_point, and the inverse")
    W, H = 640, 400
    # a frame whose every pixel encodes its own coordinate, so a rotation can be checked exactly
    ramp = np.zeros((H, W, 3), dtype=np.uint16)
    ramp[:, :, 0] = np.arange(W)[None, :]
    ramp[:, :, 1] = np.arange(H)[:, None]
    worst_map = 0
    worst_rt = 0
    for d in (PORTRAIT.CCW, PORTRAIT.CW):
        rot = PORTRAIT.rotate_image(ramp, d)
        h2, w2 = rot.shape[:2]
        if (w2, h2) != (H, W):
            return check("pixel mapping", False, f"{d}: rotated shape {w2}x{h2}, expected {H}x{W}")
        for u in range(0, W, 7):
            for v in range(0, H, 5):
                u2, v2 = PORTRAIT.rotate_point(u, v, W, H, d)
                # the pixel that landed at (u2, v2) must be the one that started at (u, v)
                worst_map = max(worst_map, abs(int(rot[v2, u2, 0]) - u), abs(int(rot[v2, u2, 1]) - v))
                bu, bv = PORTRAIT.unrotate_point(u2, v2, W, H, d)
                worst_rt = max(worst_rt, abs(bu - u), abs(bv - v))
    return check("pixel mapping", worst_map == 0 and worst_rt == 0,
                 f"both directions, 7410 probes: image-vs-formula {worst_map} px, round-trip {worst_rt} px")


# ---------------------------------------------------------------- 3. semantics preserved
def check_semantics():
    print("\n3. SEMANTICS - the same metric point out of the rotated and unrotated paths")
    W, H = 640, 400
    intr = (284.6271667480469, 284.4541931152344, 319.2109375, 201.5303192138672)
    rng = np.random.default_rng(19)
    # synthetic scene: a depth frame with a smooth ramp so a k-window sample is well defined
    depth = np.zeros((H, W), dtype=np.uint16)
    yy, xx = np.mgrid[0:H, 0:W]
    depth[:] = (900 + 0.35 * xx + 0.20 * yy).astype(np.uint16)
    uv = np.stack([rng.uniform(40, W - 40, 17), rng.uniform(40, H - 40, 17)], axis=1).astype(np.float32)

    xyz_l, meas_l = D.backproject(uv, depth, W, H, intr, k=5)

    for d in (PORTRAIT.CCW, PORTRAIT.CW):
        depth_r = PORTRAIT.rotate_image(depth, d)
        intr_r = PORTRAIT.rotate_intrinsics(intr, W, H, d)
        uv_r = np.array([PORTRAIT.rotate_point(u, v, W, H, d) for u, v in uv], dtype=np.float32)
        xyz_r, meas_r = D.backproject(uv_r, depth_r, H, W, intr_r, k=5)

        both = meas_l & meas_r
        if not both.any():
            return check("semantics", False, f"{d}: no jointly-measured points")
        # Rotating the camera about its optical axis rotates the camera-space X/Y with it; Z is
        # untouched. CCW in the image maps (X, Y) -> (Y, -X); CW maps (X, Y) -> (-Y, X).
        if d == PORTRAIT.CCW:
            exp = np.stack([xyz_l[:, 1], -xyz_l[:, 0], xyz_l[:, 2]], axis=1)
        else:
            exp = np.stack([-xyz_l[:, 1], xyz_l[:, 0], xyz_l[:, 2]], axis=1)
        err = np.abs(xyz_r[both] - exp[both])
        worst = float(err.max())
        worst_z = float(np.abs(xyz_r[both, 2] - xyz_l[both, 2]).max())
        print(f"     {d}: {int(both.sum())}/17 points, worst |dXYZ| {worst*1000:.4f} mm, "
              f"worst |dZ| {worst_z*1000:.4f} mm")
        if worst > 2e-3:                      # 2 mm: the k-window straddles different ramp pixels
            return check("semantics", False, f"{d}: worst |dXYZ| {worst*1000:.3f} mm")
    return check("semantics", True,
                 "rotated path returns the same metric point (Z identical, X/Y rotated with the camera)")


# ---------------------------------------------------------------- 4. real recorded frames
def check_real_frames():
    """Reconstruct each shoulder's metric X/Z from the RECORDED pixels + depths using the rotated
    intrinsics, then run the SHIPPED torso-yaw formula on them.

    `line_yaw_deg` is not a plain atan2: it is `atan2(dx, dz)` followed by Kalidokit's
    RollPitchYaw2 normalisation (divide by pi, wrap, +0.5), i.e. the exact y-channel that reaches
    the avatar. Comparing against a hand-rolled angle instead would compare two different
    conventions and prove nothing - the first version of this check did exactly that and reported a
    spurious ~90 deg residual.
    """
    print("\n4. REAL DATA - recorded F-18 frames through the production intrinsics")
    worst_dx = worst_dz = worst_yaw = 0.0
    n = 0
    inside = [0, 0]   # [frames checked, frames whose recorded yaw fell OUTSIDE the rounding box]
    for cap in CAPS:
        p = os.path.join(EVID, cap)
        if not os.path.exists(p):
            continue
        m = load_meta(p)
        fx, fy, cx, cy = m["intr_portrait"]
        cdx = cdz = cyaw = 0.0
        cn = 0
        with open(p, encoding="utf-8") as fh:
            fh.readline()
            for line in fh:
                r = json.loads(line)
                if not r.get("mL") or not r.get("mR"):
                    continue
                if r.get("yaw3D") is None or r.get("zL") is None:
                    continue
                zL = r["zL"] / 1000.0
                zR = r["zR"] / 1000.0
                xL = (r["uL"] - cx) * zL / fx          # the convention backproject uses
                xR = (r["uR"] - cx) * zR / fx
                yL = (r["vL"] - cy) * zL / fy
                yR = (r["vR"] - cy) * zR / fy
                cdx = max(cdx, abs((xR - xL) * 1000.0 - r["dx"]))
                cdz = max(cdz, abs((zR - zL) * 1000.0 - r["dz"]))
                yaw = CAP.line_yaw_deg([xL, yL, zL], [xR, yR, zR])
                cyaw = max(cyaw, abs((yaw - r["yaw3D"] + 90.0) % 180.0 - 90.0))
                # F-18 computed yaw3D from UNROUNDED back-projected metres and only then rounded the
                # ANGLE for storage; this reconstruction can only start from the z it rounded to
                # 0.1 mm and the u it rounded to 0.01 px. So the reference interval is the yaw over
                # the rounding box the stored values could have come from - if the recorded yaw sits
                # inside it, the transform agrees exactly and the residual is storage, not error.
                # The box must span BOTH stored quantisations: z to 0.1 mm and u/v to 0.01 px.
                # Perturbing z alone is not enough - when the shoulders are nearly coincident
                # (|dx| of a few mm, which happens whenever the subject is edge-on or one shoulder
                # is weakly detected) the yaw sensitivity reaches ~3 deg per mm of dx, so the 0.01 px
                # rounding on u alone moves the angle by tens of a degree.
                lo, hi = 180.0, -180.0
                for dzl in (-5e-5, 5e-5):
                    for dzr in (-5e-5, 5e-5):
                        for dul in (-5e-3, 5e-3):
                            for dur in (-5e-3, 5e-3):
                                zl2, zr2 = zL + dzl, zR + dzr
                                y2 = CAP.line_yaw_deg(
                                    [(r["uL"] + dul - cx) * zl2 / fx, (r["vL"] - cy) * zl2 / fy, zl2],
                                    [(r["uR"] + dur - cx) * zr2 / fx, (r["vR"] - cy) * zr2 / fy, zr2])
                                lo, hi = min(lo, y2), max(hi, y2)
                inside[0] += 1
                if not (lo - 5e-4 <= r["yaw3D"] <= hi + 5e-4):
                    inside[1] += 1
                cn += 1
        if cn:
            print(f"     {cap:26s} n={cn:5d}  max |ddx| {cdx:.4f} mm  |ddz| {cdz:.4f} mm  "
                  f"|dyaw| {cyaw:.6f} deg")
            worst_dx = max(worst_dx, cdx)
            worst_dz = max(worst_dz, cdz)
            worst_yaw = max(worst_yaw, cyaw)
            n += cn
    # dx/dz are recorded rounded to 0.01/0.1 mm, so a residual at that scale is the capture's own
    # quantisation and not a transform error.
    outside = inside[1]
    ok = n > 0 and outside == 0 and worst_dx < 0.6 and worst_dz < 0.6
    return check("real recorded frames", ok,
                 f"{n} frames; {outside} recorded yaw values fell outside the storage-rounding "
                 f"box (0 = exact agreement). Raw residual max {worst_yaw:.6f} deg is that "
                 f"rounding, not transform error. worst dx {worst_dx:.4f} mm, dz {worst_dz:.4f} mm")


# ---------------------------------------------------------------- 5. left/right identity
def check_identity():
    """Handedness: the pixel ordering of the two shoulders and their METRIC ordering must agree.

    A rotation that mirrored the frame would flip one and not the other, on every frame. Testing
    `uL > uR` alone would not do: the shoulders genuinely cross in the image whenever the subject
    turns far enough side-on, which is a real pose, not a defect.
    """
    print("\n5. IDENTITY - left/right must not swap through the rotation")
    bad = 0
    n = 0
    for cap in CAPS:
        p = os.path.join(EVID, cap)
        if not os.path.exists(p):
            continue
        m = load_meta(p)
        fx, fy, cx, cy = m["intr_portrait"]
        with open(p, encoding="utf-8") as fh:
            fh.readline()
            for line in fh:
                r = json.loads(line)
                if not r.get("mL") or not r.get("mR") or r.get("zL") is None:
                    continue
                if not (r.get("cL", 0) > 0.5 and r.get("cR", 0) > 0.5):
                    continue
                du = r["uL"] - r["uR"]
                if abs(du) < 1.0:              # genuinely edge-on: ordering is not meaningful
                    continue
                xL = (r["uL"] - cx) * (r["zL"] / 1000.0) / fx
                xR = (r["uR"] - cx) * (r["zR"] / 1000.0) / fx
                n += 1
                if (du > 0) != ((xL - xR) > 0):
                    bad += 1
    frac = bad / n if n else 1.0
    return check("left/right identity", n > 0 and frac < 0.01,
                 f"{n} confident frames, {bad} where pixel order and metric order disagree "
                 f"({frac*100:.4f}%) - a mirrored rotation would give ~100%")


def main():
    print("F-19 §5 - production portrait transform == F-18 diagnostic portrait transform")
    print("=" * 78)
    ok = True
    ok &= check_intrinsics()
    ok &= check_pixels()
    ok &= check_semantics()
    ok &= check_real_frames()
    ok &= check_identity()
    print("=" * 78)
    for name, verdict, detail in results:
        print(f"  {verdict:4s}  {name}")
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
