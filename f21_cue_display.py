#!/usr/bin/env python3
"""F-21 live-session CUE PANEL - big instruction + top-down diagram, drawn next to the camera feed.

WHY IT EXISTS. The sidecar's own banner (wholebody_udp_sender._draw_cue) is correctly sized for an
OPERATOR AT THE DESK and roughly 4x too small for a SUBJECT AT THE CAMERA, which is who has to read
it during a two-person protocol where nobody is at the keyboard. MEASURED: the preview is 640x400
landscape, 400x640 under --portrait, shown through cv2.imshow with no namedWindow - WINDOW_AUTOSIZE,
so a 400 px wide window at native size. The instruction line is FONT_HERSHEY_SIMPLEX scale 0.55 =
13 px cap-height with a 1 px stroke. On a 24" 1080p monitor that is 3.6 mm; at the 2-3 m the subjects
stand from it that subtends 4.1-6.2 arcmin, against ~5 arcmin for the 20/20 threshold of RESOLUTION
and 15-25 arcmin for comfortable glanceable reading. This panel renders the instruction at ~60 px
cap-height = 20-26 arcmin at 2.5 m, which is the band that can actually be read while walking.

THREE CHANNELS, deliberately redundant, because text is the first thing to fail at distance:

    COLOUR    a cyan / magenta field down both edges - registers peripherally, before any glyph
    GLYPH     a ~300 px A or B - says WHOSE TURN at any distance in the room
    DIAGRAM   a top-down schematic of the camera, the floor cross and the movement - says WHAT TO DO
              without depending on reading a sentence at all

The sentence is then the fourth channel, not the only one. This matters because S30's measurement is
only valid if the RIGHT person makes the RIGHT move; a misread cue does not produce a wrong number,
it produces a rep that has to be thrown away, and thrown-away reps are what the last two live
attempts consisted of.

Used two ways:
  * imported by wholebody_udp_sender.py (lazily, only under --cue-panel) so the feed and the cue are
    ONE window - the operator and both subjects watch a single thing
  * run standalone against a cue file, to check the layout without the camera:
        .venv\\Scripts\\python.exe f21_cue_display.py --cue-file <path> --demo
"""
import argparse
import io
import json
import sys

import cv2
import numpy as np

WIN = "F-21 CUE"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR. Distinguishable under the common red-green deficiencies: these differ in blue and in
# lightness, not in red-vs-green.
COL = {
    "A":      (255, 255, 0),      # cyan
    "B":      (255, 0, 255),      # magenta
    "BOTH":   (0, 215, 255),      # amber
    "NOBODY": (150, 150, 150),    # grey
}
_DEF = (255, 255, 255)
_DIM = (90, 90, 90)


def read_cue(path):
    """Never raise: a partial read of an atomically-replaced file just means the last cue is redrawn.

    Deliberately NOT a bare `except Exception` - F-21 S31.4 is the story of one of those turning a
    NameError into a silent, permanent feature outage in this exact code path. Only the errors a
    racing reader can actually produce are caught."""
    try:
        with io.open(path, encoding="utf-8") as f:
            return json.load(f)
    except (IOError, OSError, ValueError):
        return None


def _wrap(text, scale, thick, max_w):
    out, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        (tw, _), _ = cv2.getTextSize(trial, FONT, scale, thick)
        if tw > max_w and cur:
            out.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        out.append(cur)
    return out


def _fit(text, max_w, start, thick, floor=0.6):
    s = start
    while s > floor:
        (tw, _), _ = cv2.getTextSize(text, FONT, s, thick)
        if tw <= max_w:
            break
        s -= 0.05
    return s


def _centred(img, text, y, scale, colour, thick, x0=0, w=None):
    w = img.shape[1] if w is None else w
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    cv2.putText(img, text, (x0 + (w - tw) // 2, y), FONT, scale, colour, thick, cv2.LINE_AA)
    return th


def _person(img, xy, colour, label, r=22, dim=False):
    """A person, top-down: a head-sized disc with their letter in it."""
    c = _DIM if dim else colour
    cv2.circle(img, xy, r, c, -1)
    cv2.circle(img, xy, r, (0, 0, 0), 2)
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.9, 2)
    cv2.putText(img, label, (xy[0] - tw // 2, xy[1] + th // 2), FONT, 0.9, (0, 0, 0), 2,
                cv2.LINE_AA)


def draw_diagram(img, x, y, w, h, phase, actor):
    """Top-down schematic: where the camera is, where the floor cross is, and who moves where.

    Phase-keyed rather than case-keyed, because what a subject must DO is a property of the phase.
    An unknown phase draws the empty stage rather than a wrong picture - a misleading diagram is
    worse than none, and this is the channel that is trusted when the sentence cannot be read."""
    cx = x + w // 2
    cam_y = y + h - 34
    cross = (cx, y + int(h * 0.34))

    # camera + field of view
    cv2.line(img, (cx - 150, y + 6), (cx - 26, cam_y), (70, 70, 70), 2, cv2.LINE_AA)
    cv2.line(img, (cx + 150, y + 6), (cx + 26, cam_y), (70, 70, 70), 2, cv2.LINE_AA)
    cv2.rectangle(img, (cx - 26, cam_y - 12), (cx + 26, cam_y + 14), (200, 200, 200), -1)
    cv2.circle(img, (cx, cam_y + 1), 8, (40, 40, 40), -1)
    (tw, _), _ = cv2.getTextSize("CAMERA", FONT, 0.5, 1)
    cv2.putText(img, "CAMERA", (cx - tw // 2, cam_y + 32), FONT, 0.5, (150, 150, 150), 1,
                cv2.LINE_AA)

    # the floor cross
    # drawn larger than a person disc (r=22) so the X still reads when someone is standing ON it
    cv2.line(img, (cross[0] - 34, cross[1] - 34), (cross[0] + 34, cross[1] + 34),
             (255, 255, 255), 4, cv2.LINE_AA)
    cv2.line(img, (cross[0] - 34, cross[1] + 34), (cross[0] + 34, cross[1] - 34),
             (255, 255, 255), 4, cv2.LINE_AA)

    right = (x + w - 42, cross[1])
    left = (x + 42, cross[1])
    between = (cx, (cross[1] + cam_y) // 2)
    A, B = COL["A"], COL["B"]
    arrow = (0, 215, 255)

    def arr(p, q):
        cv2.arrowedLine(img, p, q, arrow, 5, cv2.LINE_AA, tipLength=0.22)

    if phase == "CLEAR":
        _person(img, left, A, "A", dim=True)
        _person(img, right, B, "B", dim=True)
        arr((left[0] + 40, left[1]), (left[0] - 10, left[1]))
        arr((right[0] - 40, right[1]), (right[0] + 10, right[1]))
    elif phase == "ANCHOR":
        _person(img, cross, A, "A")
        _person(img, right, B, "B", dim=True)
    elif phase == "A_LEAVE":
        _person(img, cross, A, "A", dim=True)
        arr((cross[0] + 40, cross[1]), (right[0] - 30, right[1]))
        _person(img, right, A, "A")
    elif phase == "A_RETURN":
        _person(img, right, A, "A", dim=True)
        arr((right[0] - 30, right[1]), (cross[0] + 40, cross[1]))
        _person(img, cross, A, "A")
    elif phase == "B_ENTER":
        _person(img, right, B, "B", dim=True)
        arr((right[0] - 30, right[1]), (cross[0] + 40, cross[1]))
        _person(img, cross, B, "B")
        _person(img, left, A, "A", dim=True)
    elif phase == "B_CROSS":
        _person(img, cross, A, "A")
        _person(img, (between[0] - 150, between[1]), B, "B", dim=True)
        arr((between[0] - 110, between[1]), (between[0] + 110, between[1]))
        _person(img, (between[0] + 150, between[1]), B, "B")
    elif phase == "A_TURN":
        _person(img, cross, A, "A")
        cv2.ellipse(img, cross, (52, 52), 0, 200, 480, arrow, 5, cv2.LINE_AA)
        cv2.arrowedLine(img, (cross[0] + 36, cross[1] - 38), (cross[0] + 52, cross[1] - 14),
                        arrow, 5, cv2.LINE_AA, tipLength=0.5)
    elif phase == "SETTLE":
        _person(img, cross, COL.get(actor, _DEF), actor if actor in ("A", "B") else "A")
        (tw, _), _ = cv2.getTextSize("HOLD STILL", FONT, 0.7, 2)
        cv2.putText(img, "HOLD STILL", (cx - tw // 2, cross[1] - 44), FONT, 0.7, arrow, 2,
                    cv2.LINE_AA)


def render(w, h, cue):
    """The cue panel at any size. Keeps working down to a small panel by shrinking, never by
    silently dropping the instruction."""
    img = np.zeros((h, w, 3), np.uint8)
    if not cue:
        _centred(img, "WAITING", h // 2, min(3.0, w / 320.0), _DEF, 4)
        return img

    actor = (cue.get("actor") or "").upper()
    colour = COL.get(actor, _DEF)
    pad = max(10, int(w * 0.018))
    if actor in COL:
        cv2.rectangle(img, (0, 0), (pad, h), colour, -1)
        cv2.rectangle(img, (w - pad, 0), (w, h), colour, -1)
    inner = w - 4 * pad

    # ---- context line: operator/record, not a subject instruction
    head = cue.get("title", "")
    if cue.get("case"):
        head += "   [%s]" % cue["case"]
    if cue.get("rep"):
        head += "   rep %s" % cue["rep"]
    y = int(h * 0.055)
    _centred(img, head, y, _fit(head, inner, 1.0, 2), (150, 150, 150), 2)

    # ---- the giant actor glyph
    y = int(h * 0.27)
    if actor:
        label = {"NOBODY": "NOBODY", "BOTH": "BOTH"}.get(actor, actor)
        thick = 26 if actor in ("A", "B") else 8
        _centred(img, label, y, _fit(label, inner, 9.0 if actor in ("A", "B") else 2.6, thick),
                 colour, thick)

    # ---- the instruction, sized for the back of the room
    scale = max(1.1, h / 430.0)
    y = int(h * 0.36)
    for line in _wrap(cue.get("instruction", ""), scale, 4, inner)[:2]:
        y += _centred(img, line, y, scale, (255, 255, 255), 4) + int(h * 0.028)
    if cue.get("detail"):
        for line in _wrap(cue["detail"], scale * 0.5, 2, inner)[:1]:
            y += _centred(img, line, y + int(h * 0.006), scale * 0.5, (0, 215, 255), 2)

    # ---- the diagram: what to do, without reading
    dy = int(h * 0.45)
    draw_diagram(img, 2 * pad, dy, w - 4 * pad, int(h * 0.45), cue.get("title", ""), actor)

    # ---- countdown + progress
    secs = cue.get("seconds_left")
    if secs is not None:
        secs = max(0.0, float(secs))
        _centred(img, "%.0f" % secs, h - int(h * 0.028), scale * 1.15, (255, 255, 255), 6)
        total = cue.get("seconds_total")
        if total:
            frac = 1.0 - min(1.0, secs / float(total))
            bh = max(5, h // 110)
            cv2.rectangle(img, (2 * pad, h - bh - 3), (w - 2 * pad, h - 3), (55, 55, 55), -1)
            cv2.rectangle(img, (2 * pad, h - bh - 3),
                          (2 * pad + int((w - 4 * pad) * frac), h - 3), colour, -1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cue-file", required=True)
    ap.add_argument("--width", type=int, default=1290)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--demo", action="store_true",
                    help="cycle every phase against a synthetic cue, to check the layout and the "
                         "diagrams with no camera and no protocol running")
    a = ap.parse_args()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, a.width, a.height)

    if a.demo:
        demo = [("CLEAR", "NOBODY", "Everyone OUT of view", "wait"),
                ("ANCHOR", "A", "A stand on the X", "B stay out"),
                ("A_LEAVE", "A", "A walk OUT to the right", "keep going"),
                ("A_RETURN", "A", "A walk back to the X", "slowly"),
                ("B_ENTER", "B", "B walk to the X", "slowly, same path"),
                ("B_CROSS", "B", "B walk past, in front of A", "A does NOT move"),
                ("A_TURN", "A", "A turn around on the X", "stay on the X"),
                ("SETTLE", "A", "Stand still", "hold")]
        i = 0
        while True:
            t, act, ins, det = demo[i % len(demo)]
            img = render(a.width, a.height,
                         dict(title=t, actor=act, instruction=ins, detail=det, case="DEMO",
                              rep="%d/%d" % (i % len(demo) + 1, len(demo)),
                              seconds_left=3, seconds_total=4))
            cv2.imshow(WIN, img)
            k = cv2.waitKey(1400) & 0xFF
            if k in (ord("q"), 27):
                break
            i += 1
    else:
        while True:
            cv2.imshow(WIN, render(a.width, a.height, read_cue(a.cue_file)))
            if (cv2.waitKey(50) & 0xFF) in (ord("q"), 27):
                break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
