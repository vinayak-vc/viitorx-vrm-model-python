#!/usr/bin/env python3
"""F-21 offline GROUND TRUTH labeller - "which human is this frame's skeleton actually on?".

Why this exists, and what it is NOT. The F-21 report's headline safety number was TARGET_SWITCH, and
SS27 showed that number reading 0 across a real person change: the ownership epoch stayed open while
the human underneath it changed. TARGET_SWITCH counts DECLARED hand-overs; it is structurally
incapable of counting SILENT ones. The metric that can is:

    WRONG_PERSON_FRAMES - frames emitted whose actual human is not the human this epoch locked onto.

That needs a per-frame identity label, and the pipeline has none. This module supplies one OFFLINE,
for test harnesses only. It is emphatically NOT a new identity signal for F-21:

  - nothing here is importable by, or reachable from, wholebody_udp_sender.py;
  - it needs the whole clip up front (it builds a background plate from a static median) and a
    static camera, so it could not run live even if someone wanted it to;
  - it is a MEASURING INSTRUMENT. Using appearance to SCORE the pipeline is the opposite of using
    appearance to DRIVE it, which the brief forbids and which this does not do.

How it labels, in three steps that are each checkable by eye (--dump writes the annotated frames):
  1. background plate = per-pixel median of frames sampled across the clip. 123.webm/456.webm are
     static-camera clips with genuinely empty stretches, so the median is the real empty room.
  2. foreground blobs = |frame - plate| thresholded, opened, connected components above a minimum
     area, then filtered for PERSON PLAUSIBILITY. Both filters exist because of measured failures,
     not in anticipation of them: a warm wall/lamp region and a bare-arm fragment of a
     half-out-of-frame dancer both produce blobs carrying no class-specific colour, and calling those
     "the person without the green shirt" is how absence of evidence turns into a wrong-person count.
       - area floor 2 % of frame. Measured: real full bodies here run 46 k - 275 k px; fragments of a
         half-exited person run 10 k - 27 k.
     A blob below the floor is dropped, not labelled '?' and argued about, so the frame reports no
     ground truth there and is excluded from scoring entirely.

     A hue-based "is this person-shaped" filter was tried here too and REMOVED, because the measurement
     said it was wrong: the spurious regions sit at hue-0-12 share 0.78-0.93 and real bodies looked
     like 0.00-0.48, but that sample only covered the two dancers' SOLO stretches. Across the lamp-lit
     left of the room the man's own striped shirt goes past 0.6 as well, and the filter deleted him
     from every two-person frame - the exact frames the whole measurement exists for. The area floor
     alone is kept. It leaves one known spurious blob (f594, 43 k px), which cannot reach the metric
     because nearest() scores distance-to-bbox and the real body containing the emitted hip always
     wins at distance 0.
  3. identity = the share of saturated torso-band pixels falling in a class's SIGNATURE hue window -
     a colour that belongs to exactly one person in the scene. Declaration order is priority order.
     A blob with NO signature colour gets the default label, but only from CLEAR absence: a share
     below NOT_SIGNATURE (0.05), with the band between that and min_share left as '?' rather than
     resolved. Absence of a colour is weaker evidence than its presence, so it is given a dead zone.

     A stricter version of that idea was tried and REMOVED, and the removal is the useful record.
     It required a blob to be a COMPLETE body (bounding box clear of the left/right frame edges)
     before absence-of-green could name anyone, reasoning that a woman entering with her shirt still
     outside the frame would otherwise score as "the person not wearing green". Sound reasoning, and
     measurably wrong here: 123.webm is 1080 px wide and portrait, and through the whole two-person
     stretch BOTH dancers have a bounding box on an edge (f620: man x=0-464, woman x=699-1080). The
     completeness rule deleted the man from every frame the measurement exists for - the same shape
     of mistake as the hue-plausibility filter above, and caught the same way, by checking what the
     rule did to the two-person frames rather than only to the solo ones.

     What actually protects against the entering-woman case is the area floor (her fragments are
     14 k-34 k px, well under the 2 % floor) plus a per-epoch modal anchor instead of a first-frame
     anchor (see f21_wrongperson_replay.wrong_person).

NOT the median hue, which is what this did first and which was wrong in a way worth recording,
because it is the kind of error that quietly inflates a safety number: a dancer's FACE and BARE ARMS
sit inside the torso band, and skin sits at hue ~10-20 in OpenCV's space - right on top of the orange
striped shirt that identifies the OTHER person. On 123.webm f108-f113 the median flipped to the man's
window while the woman was demonstrably alone in the room (verified by eye - the frame is
evidence/oak_v4/f21/wrongperson/frames/123_PATH_ON_f00111.png, she is in her green t-shirt and the
pipeline is correctly locked on her), and those six frames were being counted as WRONG-PERSON
EMISSIONS by the pipeline. They were nothing of the sort; the instrument was broken.

Switching from the median to a share fixed only half of it, and the other half is the part actually
worth knowing: SKIN IS NOT CLASS-SPECIFIC. Both dancers have faces and bare arms, so the orange
window contains evidence for BOTH of them and its share discriminates nothing - at f111 the woman's
own skin took 0.570 of her torso band against 0.405 for her shirt, and a largest-share rule still
called her the man. The green t-shirt is the only colour in this scene that belongs to exactly one
person, so it is the only one allowed to decide. Measured, with the threshold placed afterwards:

    woman, 31 blobs across two solo stretches   green share  p10 0.553  median 0.688  max 0.817
    man,   13 blobs across his solo stretch     green share  max 0.001

Threshold 0.15 - 3.7x below the woman's p10 and 150x above the man's maximum. Corrected by finding
the mechanism, not by moving a number until 123.webm read what was wanted, and then validated against
an eye-checked truth set with the residual error rate reported (validate_labels.py) rather than
assumed to be zero.

A frame whose emitted hip is not inside/near any blob is labelled NONE, not guessed.
"""
import numpy as np
import cv2


class GroundTruth(object):
    def __init__(self, video, sample=40, fg_thresh=28, min_area_frac=0.02, torso_band=(0.08, 0.42)):
        self.video = video
        self.fg_thresh = fg_thresh
        self.torso_band = torso_band
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise SystemExit("could not open %s" % video)
        self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.min_area = int(min_area_frac * self.w * self.h)
        idx = set(int(round(i)) for i in np.linspace(0, max(0, self.n - 1), sample))
        got, i = [], 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i in idx:
                got.append(f)
            i += 1
        cap.release()
        # median, not mean: a mean smears a person who stood still into a permanent ghost.
        self.plate = np.median(np.stack(got), axis=0).astype(np.uint8)
        self._kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    def blobs(self, frame):
        """-> list of dict(cx, cy, x, y, w, h, area, hue, sat). Sorted by area, largest first."""
        d = cv2.absdiff(frame, self.plate)
        g = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        _, m = cv2.threshold(g, self.fg_thresh, 255, cv2.THRESH_BINARY)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self._kern)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self._kern, iterations=5)
        nlab, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        out = []
        for i in range(1, nlab):
            x, y, bw, bh, area = stats[i]
            if area < self.min_area:
                continue
            y0 = int(y + self.torso_band[0] * bh)
            y1 = int(y + self.torso_band[1] * bh)
            sub = (lab[y0:y1, x:x + bw] == i)
            if sub.sum() < 50:
                continue
            band = hsv[y0:y1, x:x + bw]
            sat = band[..., 1][sub]
            keep = sat > 60          # ignore the white/cream pixels both dancers wear below the waist
            hue_px = band[..., 0][sub][keep]
            hist = (np.bincount(hue_px.astype(np.int32), minlength=180)
                    if hue_px.size else np.zeros(180, dtype=np.int64))
            complete = (x > EDGE_PX and (x + bw) < (frame.shape[1] - EDGE_PX))   # reported only
            out.append(dict(cx=float(cent[i][0]), cy=float(cent[i][1]), x=int(x), y=int(y),
                            w=int(bw), h=int(bh), area=int(area), complete=bool(complete),
                            hue=(float(np.median(hue_px)) if hue_px.size >= 30 else None),
                            hist=hist, sat=float(np.median(sat)), npx=int(hue_px.size)))
        out.sort(key=lambda b: -b["area"])
        return out


# hue windows in OpenCV's 0..179 space, measured from the clip (see the calibration in the F-21
# report S30) and applied as SHARES, never as a silent constant on a median.
MIN_PIXELS = 400          # too few saturated torso pixels to decide anything -> '?'
EDGE_PX = 4               # a bbox this close to the left/right edge is a partially-seen body
NOT_SIGNATURE = 0.05      # below this a signature colour is CLEARLY absent, not just weak


def share(hist, lo, hi):
    tot = float(hist.sum())
    if tot <= 0:
        return 0.0
    lo_i, hi_i = int(round(lo)), int(round(hi))
    n = (hist[lo_i:hi_i + 1].sum() if lo_i <= hi_i
         else hist[lo_i:].sum() + hist[:hi_i + 1].sum())     # wraps through red at 0/180
    return float(n) / tot


def classify(blob, windows, default=None):
    """windows: ordered [(label, lo, hi, min_share)] - SIGNATURE colours, each unique to one person.
    -> (label, share). '?' is a deliberate non-answer; wrong_person() excludes it from the comparison
    rather than scoring a guess."""
    if blob is None or blob.get("hist") is None or int(blob["hist"].sum()) < MIN_PIXELS:
        return "?", 0.0
    weakest = 1.0
    for name, lo, hi, min_share in windows:
        sh = share(blob["hist"], lo, hi)
        if sh >= min_share:
            return name, round(sh, 4)       # positive evidence - valid even on a partial body
        weakest = min(weakest, sh)
    if default and (not windows or weakest < NOT_SIGNATURE):
        # no windows at all = a single-person clip, where there is no signature to be absent and
        # every person-sized blob is the same person by construction (video.webm, 456.webm - see
        # the F-21 report S30 on why those two get no wrong-person ground truth).
        return default, round(-weakest, 4)
    return "?", 0.0                          # the dead zone between - refuse to call it


def nearest(blobs, u, v, max_frac, w, h):
    """The blob an emitted hip belongs to, or None. max_frac is of the frame diagonal."""
    if u is None or v is None:
        return None
    lim = max_frac * (w * w + h * h) ** 0.5
    best, bd = None, 1e18
    for b in blobs:
        # inside the bbox counts as distance 0 - a hip is often off the blob centroid
        dx = max(b["x"] - u, 0, u - (b["x"] + b["w"]))
        dy = max(b["y"] - v, 0, v - (b["y"] + b["h"]))
        d = (dx * dx + dy * dy) ** 0.5
        if d < bd:
            best, bd = b, d
    return best if (best is not None and bd <= lim) else None
