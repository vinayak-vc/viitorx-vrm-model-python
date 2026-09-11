#!/usr/bin/env python3
"""
F-16 diagnostic capture core.

An ISOLATED capture harness. It reuses production's pose model (rtmw3d_pose) and production's F-08
surface-aware depth sampler (oak_depth.backproject) UNCHANGED, but drives them from a configurable
stereo pipeline (f16_configs) so the stereo variable can be swept without touching production.

Nothing here sends UDP, and no production module is modified.

Per-frame JSONL record (all lengths in mm unless noted):
  seq t cfg block            capture identity
  uL vL uR vR cL cR uSpan    shoulder pixels + SimCC confidence + pixel span
  zL zR qL qR mL mR          F-08 sampled depth, quality, measured flag (EXACTLY production's)
  wL wR                      sorted UNIQUE raw depth values in the 5x5 window (the ladder itself)
  p30L p30R p50L p50R ctrL ctrR nvL nvR
  hipZ                       mid-hip depth (m)
  dx dz                      landmark-space shoulder delta -> the yaw triangle
  yaw3D                      the SHIPPED Kalidokit y-channel, reproduced exactly
  latMs fps
"""
import argparse
import io
import json
import math
import os
import time

import numpy as np
import cv2
try:
    import winsound
except ImportError:
    winsound = None
import depthai as dai

import rtmw3d_pose as R
import oak_depth as D
import f16_configs as C

OUTDIR = os.path.join("oak_v4_evidence", "f16")


def beep(freq=880, ms=150):
    """Audible block cue - the subject is metres from the screen and cannot read the console."""
    if winsound is not None:
        try:
            winsound.Beep(freq, ms)
        except Exception:
            pass
WB_L_SHOULDER = 5      # COCO-WholeBody
WB_R_SHOULDER = 6
WB_L_HIP = 11
WB_R_HIP = 12
KWIN = 5               # production default (--kwin)


def line_yaw_deg(a, b):
    """The SHIPPED Kalidokit CalcHipsAndSpine y-channel (KMath.RollPitchYaw2 -> rigHips), verbatim.
    Validated against 11,477 live frames at 0.000000 deg residual."""
    r = math.atan2(b[0] - a[0], b[2] - a[2])
    ang = math.fmod(r, 2 * math.pi)
    if ang > math.pi:
        ang -= 2 * math.pi
    elif ang < -math.pi:
        ang += 2 * math.pi
    y = ang / math.pi
    if y > 0.5:
        y -= 2.0
    y += 0.5
    return math.degrees(y * math.pi)


def window_unique(depth, u, v, k=KWIN):
    """Sorted unique NON-ZERO depths of the kxk window -- the quantisation ladder at that point."""
    dh, dw = depth.shape
    x, y = int(round(u)), int(round(v))
    r = k // 2
    x0, x1 = max(0, x - r), min(dw, x + r + 1)
    y0, y1 = max(0, y - r), min(dh, y + r + 1)
    if x1 <= x0 or y1 <= y0:
        return []
    w = depth[y0:y1, x0:x1].reshape(-1)
    w = w[w > 0]
    return [int(z) for z in np.unique(w)]


def pct(depth, u, v, q, k=KWIN):
    dh, dw = depth.shape
    x, y = int(round(u)), int(round(v))
    r = k // 2
    x0, x1 = max(0, x - r), min(dw, x + r + 1)
    y0, y1 = max(0, y - r), min(dh, y + r + 1)
    if x1 <= x0 or y1 <= y0:
        return None, 0
    w = depth[y0:y1, x0:x1].reshape(-1)
    w = w[w > 0]
    if w.size == 0:
        return None, 0
    return float(np.percentile(w, q)), int(w.size)


def run(config, blocks, gap, model, fh, quiet=False, warmup=1.0):
    """Capture `blocks` = [(label, seconds), ...] under one stereo configuration.

    Returns a dict of per-block frame counts. Prints a prompt + countdown between blocks so the
    subject can move. Self-terminating: the loop is bounded by the block durations.
    """
    pipe, cfg, mw, mh, rw, rh = C.build(config)
    counts = {}
    with dai.Device(pipe) as dev:
        q_rgb = dev.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_dep = dev.getOutputQueue("depth", maxSize=4, blocking=False)
        intr = D.read_rgb_intrinsics(dev, rw, rh)
        bbox = R.center_bbox(rw, rh)
        seq = 0
        t_last = time.time()

        for bi, (label, secs) in enumerate(blocks):
            if gap > 0:          # countdown before EVERY block, including the first
                for s in range(int(gap), 0, -1):
                    if not quiet:
                        print("\r  >>> NEXT: %-22s in %2d s   " % (label, s), end="", flush=True)
                    t_end = time.time() + 1.0
                    while time.time() < t_end:      # keep draining so queues stay fresh
                        q_rgb.tryGet()
                        q_dep.tryGet()
                if not quiet:
                    print("\r  >>> RECORDING: %-30s" % label)
            t0 = time.time()
            while time.time() - t0 < warmup:
                q_rgb.tryGet()
                q_dep.tryGet()
            t0 = time.time()
            n = 0
            while time.time() - t0 < secs:
                rgb_pkt = q_rgb.get()
                dep_pkt = q_dep.get()
                now = time.time()
                fps = 1.0 / max(1e-6, now - t_last)
                t_last = now
                try:
                    lat = (dai.Clock.now() - rgb_pkt.getTimestamp()).total_seconds() * 1000.0
                except Exception:
                    lat = -1.0
                frame = rgb_pkt.getCvFrame()
                depth = dep_pkt.getFrame()
                seq += 1
                n += 1

                uv, zrel, conf = model.infer(frame, bbox)
                refined = R.bbox_from_keypoints(uv, conf, rw, rh, thr=0.3)
                if refined is not None and float(np.mean(conf[0:17])) >= 0.3:
                    bbox = tuple(0.7 * np.array(bbox) + 0.3 * np.array(refined))
                else:
                    bbox = R.center_bbox(rw, rh)

                xyz, meas, qual, _diag = D.backproject(
                    uv, depth, rw, rh, intr, k=KWIN, with_quality=True, legacy=False)

                sxd = depth.shape[1] / float(rw)
                syd = depth.shape[0] / float(rh)
                L, Rr = WB_L_SHOULDER, WB_R_SHOULDER
                uL, vL = float(uv[L, 0]), float(uv[L, 1])
                uR, vR = float(uv[Rr, 0]), float(uv[Rr, 1])
                p30L, nvL = pct(depth, uL * sxd, vL * syd, 30.0)
                p30R, nvR = pct(depth, uR * sxd, vR * syd, 30.0)
                p50L, _ = pct(depth, uL * sxd, vL * syd, 50.0)
                p50R, _ = pct(depth, uR * sxd, vR * syd, 50.0)
                ctrL = depth[min(depth.shape[0] - 1, max(0, int(round(vL * syd)))),
                             min(depth.shape[1] - 1, max(0, int(round(uL * sxd))))]
                ctrR = depth[min(depth.shape[0] - 1, max(0, int(round(vR * syd)))),
                             min(depth.shape[1] - 1, max(0, int(round(uR * sxd))))]

                hip_ok = meas[WB_L_HIP] and meas[WB_R_HIP]
                hipZ = float(0.5 * (xyz[WB_L_HIP, 2] + xyz[WB_R_HIP, 2])) if hip_ok else None

                a = [float(xyz[L, 0]), float(xyz[L, 1]), float(xyz[L, 2])]
                b = [float(xyz[Rr, 0]), float(xyz[Rr, 1]), float(xyz[Rr, 2])]
                yaw = line_yaw_deg(a, b) if (meas[L] and meas[Rr]) else None

                rec = {
                    "seq": seq, "t": round(now, 4), "cfg": config, "block": label,
                    "uL": round(uL, 2), "vL": round(vL, 2), "uR": round(uR, 2), "vR": round(vR, 2),
                    "cL": round(float(conf[L]), 3), "cR": round(float(conf[Rr]), 3),
                    "uSpan": round(abs(uL - uR), 2),
                    "zL": round(float(xyz[L, 2]) * 1000.0, 1),
                    "zR": round(float(xyz[Rr, 2]) * 1000.0, 1),
                    "qL": round(float(qual[L]), 3), "qR": round(float(qual[Rr]), 3),
                    "mL": int(bool(meas[L])), "mR": int(bool(meas[Rr])),
                    "p30L": p30L, "p30R": p30R, "p50L": p50L, "p50R": p50R,
                    "ctrL": int(ctrL), "ctrR": int(ctrR), "nvL": nvL, "nvR": nvR,
                    "wL": window_unique(depth, uL * sxd, vL * syd),
                    "wR": window_unique(depth, uR * sxd, vR * syd),
                    "hipZ": round(hipZ, 4) if hipZ is not None else None,
                    "dx": round((b[0] - a[0]) * 1000.0, 2),
                    "dz": round((b[2] - a[2]) * 1000.0, 2),
                    "yaw3D": round(yaw, 3) if yaw is not None else None,
                    "latMs": round(lat, 1), "fps": round(fps, 2),
                    "depthW": int(depth.shape[1]), "depthH": int(depth.shape[0]),
                }
                fh.write(json.dumps(rec) + "\n")

                if not quiet and n % 15 == 0:
                    print("\r   %-12s n=%4d  HIP %5.2f m  uSpan %5.1f px  zL %6.0f zR %6.0f  "
                          "dz %+7.1f  dx %6.1f  YAW %+7.1f  %4.1f fps   "
                          % (label, n, hipZ if hipZ is not None else float("nan"),
                             abs(uL - uR), float(xyz[L, 2]) * 1000.0,
                             float(xyz[Rr, 2]) * 1000.0, (b[2] - a[2]) * 1000.0,
                             abs(b[0] - a[0]) * 1000.0, yaw if yaw is not None else float("nan"),
                             fps), end="", flush=True)
            counts[label] = n
            beep(500, 120)
            beep(500, 120)
            if not quiet:
                print("\r   %-14s DONE  %d frames%s" % (label, n, " " * 60))
    return counts


def open_out(name):
    os.makedirs(OUTDIR, exist_ok=True)
    base = os.path.join(OUTDIR, name)
    n = 0
    path = base + ".jsonl"
    while os.path.exists(path):                  # section 15: unique filename per run
        n += 1
        path = "%s_%d.jsonl" % (base, n)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"..\..\..\SentisModel\rtmw3d-x.onnx")
    ap.add_argument("--config", default="baseline")
    ap.add_argument("--blocks", required=True,
                    help="comma list label:seconds, e.g. d080:12,d100:12")
    ap.add_argument("--gap", type=float, default=8.0, help="seconds between blocks (move time)")
    ap.add_argument("--name", default="", help="output basename")
    a = ap.parse_args()

    blocks = []
    for tok in a.blocks.split(","):
        tok = tok.strip()
        if not tok:
            continue
        lab, _, secs = tok.partition(":")
        blocks.append((lab, float(secs or 12)))

    print("[f16] loading RTMW3D ...")
    model = R.RTMW3D(a.model)
    name = a.name or ("cap_%s_%s" % (a.config, time.strftime("%H%M%S")))
    path = open_out(name)
    print("[f16] config=%s  blocks=%s" % (a.config, blocks))
    print("[f16] writing %s" % path)
    with io.open(path, "w", encoding="utf-8") as fh:
        counts = run(a.config, blocks, a.gap, model, fh)
    print("[f16] done: %s" % counts)
    print("[f16] %s" % path)


if __name__ == "__main__":
    main()
