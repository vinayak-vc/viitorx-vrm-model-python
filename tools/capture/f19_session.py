#!/usr/bin/env python3
"""F-19 sections 10, 11, 17, 18, 19 - drive a live protocol and label the Unity recorder's frames.

This is a PROMPTER, not a capture: the sidecar owns the camera and Unity owns the avatar, so all
this does is tell the subject what to do next and write the current cue where F19BoneRecorder can
read it. Labelling by file rather than over the editor bridge keeps the render loop untouched
between blocks.

The window is deliberately small and NOT fullscreen: for sections 10 and 11 the subject has to be
able to see the avatar, which is the thing under test. Put it where it does not cover the mirror.

    python f19_session.py --protocol square
    python f19_session.py --protocol motion
    python f19_session.py --blocks "STAND STILL:15,RAISE ARMS:10"
"""
import argparse
import io
import os
import sys
import time

import cv2
import numpy as np

W, H = 900, 340
WHITE = (255, 255, 255)
GREEN = (90, 230, 90)
AMBER = (0, 190, 255)
GREY = (150, 150, 150)
BLOCK_FILE = os.path.join("oak_v4_evidence", "f19", "block.txt")
COUNTDOWN = 4.0

# Section 10: the highest-value test - a physically square human must give a visually square avatar.
SQUARE = [("STAND SQUARE - STILL - HANDS RELAXED", 20.0)]

# Section 11: whole-body motion. Ordered so the trunk work happens while the subject is freshest and
# the awkward poses (crouch, edge of frame) come last.
MOTION = [
    ("STAND STILL", 8.0),
    ("STEP FORWARD, THEN BACK", 10.0),
    ("STEP LEFT, THEN RIGHT", 10.0),
    ("ARMS RELAXED", 8.0),
    ("ARMS OUT AT 45 DEGREES", 10.0),
    ("ARMS STRAIGHT FORWARD", 10.0),
    ("REACH LEFT, THEN RIGHT", 10.0),
    ("HANDS NEAR YOUR FACE", 8.0),
    ("HANDS NEAR YOUR WAIST", 8.0),
    ("ROTATE TORSO LEFT", 8.0),
    ("ROTATE TORSO RIGHT", 8.0),
    ("BACK TO CENTRE", 6.0),
    ("TURN LEFT-RIGHT SLOWLY", 12.0),
    ("TURN LEFT-RIGHT AT NORMAL SPEED", 12.0),
    ("BEND FORWARD", 8.0),
    ("CROUCH DOWN", 8.0),
    ("PARTIAL SQUAT", 8.0),
    ("STEP AROUND WHILE MOVING YOUR ARMS", 12.0),
]

# Section 17/18: uncontrolled visitors, single-user follow/lose/re-enter.
PUBLIC = [
    ("STAND CENTRED", 8.0),
    ("STAND SLIGHTLY LEFT", 8.0),
    ("STAND SLIGHTLY RIGHT", 8.0),
    ("STAND AT THE EDGE OF VIEW", 8.0),
    ("CROSS YOUR ARMS", 8.0),
    ("PUT ONE HAND BEHIND YOUR BACK", 8.0),
    ("HOLD ONE HAND AGAINST YOUR TORSO", 8.0),
    ("TURN SIDE-ON", 8.0),
    ("WALK OUT OF VIEW", 8.0),
    ("WALK BACK IN AND STOP", 10.0),
    ("WALK OUT, THEN RE-ENTER TWICE", 14.0),
]

# Section 19: the person-switching question. Needs a second person.
MULTI = [
    ("PRIMARY CENTRED - SECOND PERSON STANDS BEHIND", 15.0),
    ("SECOND PERSON STEPS BESIDE YOU", 15.0),
    ("SECOND PERSON WALKS THROUGH THE FRAME", 15.0),
    ("SECOND PERSON BRIEFLY BLOCKS YOU", 15.0),
    ("SECOND PERSON LEAVES", 10.0),
]

PROTOCOLS = {"square": SQUARE, "motion": MOTION, "public": PUBLIC, "multi": MULTI}


def draw(cue, sub, prog, colour, idx, total):
    img = np.zeros((H, W, 3), np.uint8)
    scale = 1.5 if len(cue) <= 26 else (1.05 if len(cue) <= 40 else 0.8)
    (tw, _), _ = cv2.getTextSize(cue, cv2.FONT_HERSHEY_SIMPLEX, scale, 4)
    cv2.putText(img, cue, (max(20, (W - tw) // 2), 150),
                cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 4, cv2.LINE_AA)
    (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(img, sub, ((W - sw) // 2, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, "block %d / %d      SPACE = skip      Q = stop" % (idx, total),
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2, cv2.LINE_AA)
    bw = int((W - 40) * max(0.0, min(1.0, prog)))
    cv2.rectangle(img, (20, H - 60), (W - 20, H - 25), (55, 55, 55), -1)
    cv2.rectangle(img, (20, H - 60), (20 + bw, H - 25), colour, -1)
    cv2.imshow("F-19 PROTOCOL", img)
    return cv2.waitKey(1) & 0xFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="", choices=sorted(PROTOCOLS.keys()))
    ap.add_argument("--blocks", default="", help="'CUE:seconds,CUE:seconds' overriding --protocol")
    ap.add_argument("--block-file", default=BLOCK_FILE)
    a = ap.parse_args()

    if a.blocks:
        blocks = []
        for part in a.blocks.split(","):
            name, _, secs = part.rpartition(":")
            blocks.append((name.strip(), float(secs)))
    elif a.protocol:
        blocks = PROTOCOLS[a.protocol]
    else:
        print("give --protocol or --blocks", file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(a.block_file), exist_ok=True)
    total_s = sum(s for _, s in blocks) + COUNTDOWN * len(blocks)
    print("[f19] %d blocks, about %.0f s total" % (len(blocks), total_s), flush=True)

    def label(name):
        io.open(a.block_file, "w", encoding="utf-8").write(name)

    cv2.namedWindow("F-19 PROTOCOL", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("F-19 PROTOCOL", W, H)
    stopped = False
    done = []
    for i, (name, secs) in enumerate(blocks):
        # The transition itself is labelled separately so it never contaminates a block.
        label("(between)")
        t0 = time.time()
        while time.time() - t0 < COUNTDOWN:
            left = COUNTDOWN - (time.time() - t0)
            k = draw(name, "get ready - starts in %.0f" % left, 1.0 - left / COUNTDOWN,
                     AMBER, i + 1, len(blocks))
            if k in (27, ord("q")):
                stopped = True
                break
            if k == 32:
                break
        if stopped:
            break
        label(name)
        print("[f19] block %d/%d  %s  (%.0f s)" % (i + 1, len(blocks), name, secs), flush=True)
        t0 = time.time()
        while time.time() - t0 < secs:
            k = draw(name, "%.0f s left" % (secs - (time.time() - t0)),
                     (time.time() - t0) / secs, GREEN, i + 1, len(blocks))
            if k in (27, ord("q")):
                stopped = True
                break
            if k == 32:
                break
        done.append(name)
        if stopped:
            break
    label("(done)")
    cv2.destroyAllWindows()
    print("[f19] completed %d/%d blocks%s" % (len(done), len(blocks),
                                              " (stopped early)" if stopped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
