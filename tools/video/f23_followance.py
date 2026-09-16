#!/usr/bin/env python3
"""Measure how well the rendered avatar FOLLOWS the video, instead of judging it by eye.

"Does the avatar do the same thing" is a question a side-by-side can only answer impressionistically,
and impressions are exactly what this project's rules distrust. So: take one scalar that is defined
identically on both sides, and cross-correlate it.

    SIGNAL   normalised arm span - horizontal wrist-to-wrist distance over shoulder width.
             Scale-free (so the avatar's proportions and the dancer's need not match), rotation-
             tolerant, and it is the single most active degree of freedom in this clip.

    INPUT    computed from the payloads actually sent on the wire (f23_video_to_unity --dump), so
             it is the tracker's opinion of the dancer, not a re-derivation.
    OUTPUT   measured from the RECORDED PIXELS of the Unity game view by segmenting the avatar
             against the checkerboard. Deliberately measured from what was DRAWN rather than from a
             bone log: a bone log would prove the retarget emitted something, not that the viewer
             saw it.

Reported: best lag, Pearson r at that lag, and r at zero lag. A high r at a consistent lag means the
avatar follows faithfully but late; a low r at every lag means it does not follow.

    .venv\\Scripts\\python.exe f23_followance.py --dump <payloads.jsonl> --capture <unity.mkv> --lead 4.08
"""
import argparse
import io
import json
import sys

import cv2
import numpy as np

L_WRIST, R_WRIST, L_SH, R_SH = 15, 16, 11, 12


def input_signal(dump_path):
    """Normalised arm span per video frame, from the wire payloads."""
    xs = []
    for line in io.open(dump_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        lm = d["lm"]
        lw, rw, ls, rs = lm[L_WRIST], lm[R_WRIST], lm[L_SH], lm[R_SH]
        if min(lw[3], rw[3], ls[3], rs[3]) <= 0.0:
            xs.append(np.nan)
            continue
        sh = abs(ls[0] - rs[0])
        xs.append(abs(lw[0] - rw[0]) / sh if sh > 1e-6 else np.nan)
    return np.array(xs, dtype=float)


def _frames(cap_path, crop, skip):
    x, y, w, h = crop
    cap = cv2.VideoCapture(cap_path)
    for _ in range(skip):
        cap.read()
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield f[y:y + h, x:x + w]
    cap.release()


def output_signal(cap_path, crop, skip):
    """Normalised avatar width per rendered frame, measured from the pixels.

    BACKGROUND SUBTRACTION against a static median plate, not a colour threshold. A first attempt
    thresholded saturation on the assumption that the checkerboard was neutral grey; it is not
    (median S = 56, and S > 70 selected 119k pixels spanning the full width), so the signal came
    back CONSTANT and the correlation it produced was meaningless. The game camera does not move,
    so the plate is well defined - the same technique f21_ground_truth.py uses, and for the same
    reason.

    Width is normalised by the avatar's own median width, so the measure is scale-free on this side
    as well as on the input side."""
    plate_src = list(_frames(cap_path, crop, skip))
    if len(plate_src) < 10:
        return np.array([], dtype=float)
    sample = plate_src[::max(1, len(plate_src) // 40)][:40]
    plate = np.median(np.stack(sample), axis=0).astype(np.uint8)

    out = []
    for g in plate_src:
        d = cv2.absdiff(g, plate).max(axis=2)
        mask = (d > 30).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cols = np.where(mask.sum(axis=0) > 3)[0]
        out.append((cols.max() - cols.min()) if len(cols) > 3 else np.nan)
    a = np.array(out, dtype=float)
    med = np.nanmedian(a)
    return a / med if med and not np.isnan(med) else a


def pearson(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 30:
        return float("nan")
    a, b = a[m], b[m]
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--lead", type=float, default=0.0)
    ap.add_argument("--crop", default="0,136,540,870")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max-lag", type=int, default=30)
    a = ap.parse_args()

    crop = tuple(int(v) for v in a.crop.split(","))
    src = input_signal(a.dump)
    dst = output_signal(a.capture, crop, int(round(a.lead * a.fps)))
    n = min(len(src), len(dst))
    src, dst = src[:n], dst[:n]
    print("frames compared: %d  (input %d, output %d)" % (n, len(src), len(dst)))
    print("input  arm-span: median %.2f  range %.2f-%.2f"
          % (np.nanmedian(src), np.nanmin(src), np.nanmax(src)))
    print("output avatar w: median %.2f  range %.2f-%.2f"
          % (np.nanmedian(dst), np.nanmin(dst), np.nanmax(dst)))

    best = (-2.0, 0)
    rows = []
    for lag in range(-a.max_lag, a.max_lag + 1):
        if lag >= 0:
            r = pearson(src[:n - lag], dst[lag:])
        else:
            r = pearson(src[-lag:], dst[:n + lag])
        rows.append((lag, r))
        if not np.isnan(r) and r > best[0]:
            best = (r, lag)
    r0 = dict(rows)[0]
    print()
    print("Pearson r at zero lag : %+.3f" % r0)
    print("best r                : %+.3f at lag %+d frames (%.0f ms)"
          % (best[0], best[1], best[1] * 1000.0 / a.fps))
    print()
    print("lag(frames)  r")
    for lag, r in rows:
        if lag % 3 == 0:
            bar = "#" * int(max(0.0, r) * 40)
            print("  %+4d      %+.3f %s%s" % (lag, r, bar, "   <-- best" if lag == best[1] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
