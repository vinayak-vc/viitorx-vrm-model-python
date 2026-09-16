#!/usr/bin/env python3
"""Compose the SOURCE video and the recorded Unity avatar into one side-by-side clip.

Written in OpenCV rather than as an ffmpeg filtergraph on purpose: the graph needed an escaped
Windows font path AND quoted drawtext values in the same expression, and ffmpeg's parser rejected
the combination three different ways. Compositing here costs a few lines and removes a whole class
of quoting failure from an artefact that is going in front of people.

The caveat band is burned into every frame deliberately. This clip is going to be shown out of
context - that is what demo clips are for - and the one thing a viewer must not take away is that
the stereo sensor path was demonstrated. It was not: the avatar here is driven from RTMW3D's
monocular root-relative z, because a webm has one camera.

    .venv\\Scripts\\python.exe f23_compose_sbs.py --source ..\\..\\video\\video.webm \\
        --capture evidence/oak_v4/f23/unity_XXXX.mkv --lead 4.08
"""
import argparse
import os
import sys

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
CAVEAT = "NO STEREO DEPTH - monocular depth only. Retarget/avatar demo, NOT the sensor path."


def autocrop(path, pad=6):
    """Bounding box of everything that MOVES during the capture.

    The desktop is static except for the rendered avatar, so the moving pixels ARE the Game view.
    Found this way rather than from the window rect because gdigrab records the whole desktop
    regardless of the region flags, and a hard-coded rect silently recorded the wrong window once
    the editor was no longer where it had been."""
    cap = cv2.VideoCapture(path)
    prev, acc = None, None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            d = cv2.absdiff(g, prev)
            acc = d if acc is None else np.maximum(acc, d)
        prev = g
    cap.release()
    if acc is None:
        raise SystemExit("capture has no frames")
    mask = (acc > 28).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    ys, xs = np.where(mask > 0)
    if len(xs) < 500:
        raise SystemExit("nothing moved in the capture - was the avatar visible?")
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad)
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def verify_gameview(cap_path, crop, skip, need=0.80):
    """Refuse to compose unless the crop really is the Unity Game view.

    gdigrab records the DESKTOP, so if the editor loses foreground mid-run the crop silently
    contains whatever covered it - and the composite still renders, looking plausible, showing the
    wrong window. That happened twice. The Game view has a signature the chat window does not: a
    large near-neutral checkerboard floor under a saturated blue band. Checked on sampled frames;
    below `need` agreement this raises instead of producing a misleading artefact."""
    x, y, w, h = crop
    cap = cv2.VideoCapture(cap_path)
    for _ in range(skip):
        cap.read()
    hits = total = 0
    idx = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        idx += 1
        if idx % 10:
            continue
        g = f[y:y + h, x:x + w]
        if g.size == 0:
            break
        hsv = cv2.cvtColor(g, cv2.COLOR_BGR2HSV)
        sky = ((hsv[:int(h * 0.14), :, 0] > 100) & (hsv[:int(h * 0.14), :, 0] < 130)
               & (hsv[:int(h * 0.14), :, 1] > 60)).mean()
        floor = hsv[int(h * 0.25):, :, 1].mean()      # checkerboard: low saturation
        hits += 1 if (sky > 0.30 and floor < 90) else 0
        total += 1
    cap.release()
    frac = (float(hits) / total) if total else 0.0
    if frac < need:
        raise SystemExit(
            "REFUSING TO COMPOSE: only %.0f%% of sampled frames look like the Unity Game "
            "view. The capture is probably of another window - the editor lost foreground. "
            "Re-run with Unity visible and in front, and do not click away during it."
            % (100 * frac))
    print("game-view check: %.0f%% of sampled frames OK" % (100 * frac))


def label(img, text, scale, colour, thick, cx, y):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    x = int(cx - tw / 2)
    cv2.rectangle(img, (x - 12, y - th - 12), (x + tw + 12, y + 12), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), FONT, scale, colour, thick, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--lead", type=float, default=0.0,
                    help="seconds of the capture before the first packet (from the demo runner)")
    ap.add_argument("--crop", default="auto",
                    help="x,y,w,h of the Game view in the capture, or 'auto' to find it from "
                         "where the picture actually moves")
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()

    if a.crop == "auto":
        cx, cy, cw, ch = autocrop(a.capture)
        print("auto crop: %d,%d,%d,%d" % (cx, cy, cw, ch))
    else:
        cx, cy, cw, ch = (int(v) for v in a.crop.split(","))
    verify_gameview(a.capture, (cx, cy, cw, ch), int(round(a.lead * 30.0)))
    src = cv2.VideoCapture(a.source)
    cap = cv2.VideoCapture(a.capture)
    if not src.isOpened() or not cap.isOpened():
        print("cannot open inputs")
        return 2

    fps = src.get(cv2.CAP_PROP_FPS) or 30.0
    cap_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    skip = int(round(a.lead * cap_fps))
    for _ in range(skip):
        cap.read()
    print("source %.1f fps, capture %.1f fps, skipping %d capture frames (%.2f s lead)"
          % (fps, cap_fps, skip, a.lead))

    out_path = a.out or os.path.splitext(a.capture)[0].replace("unity_", "side_by_side_") + ".mp4"
    writer = None
    n = 0
    while True:
        ok_s, fs = src.read()
        ok_c, fc = cap.read()
        if not ok_s or not ok_c:
            break
        fc = fc[cy:cy + ch, cx:cx + cw]
        sh_, sw_ = fs.shape[:2]
        fs = cv2.resize(fs, (int(sw_ * a.height / sh_), a.height))
        ch_, cw_ = fc.shape[:2]
        fc = cv2.resize(fc, (int(cw_ * a.height / ch_), a.height))
        frame = np.hstack([fs, fc])

        w = frame.shape[1]
        label(frame, "SOURCE VIDEO", 0.9, (255, 255, 255), 2, fs.shape[1] / 2, 44)
        label(frame, "UNITY AVATAR", 0.9, (255, 255, 255), 2, fs.shape[1] + fc.shape[1] / 2, 44)
        label(frame, CAVEAT, 0.6, (0, 220, 255), 2, w / 2, frame.shape[0] - 22)

        if writer is None:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (w, frame.shape[0]))
        writer.write(frame)
        n += 1

    src.release()
    cap.release()
    if writer is not None:
        writer.release()
    print("wrote %s  (%d frames, %.1f s)" % (out_path, n, n / fps))
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
