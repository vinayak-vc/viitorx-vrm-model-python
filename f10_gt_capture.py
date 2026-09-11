#!/usr/bin/env python3
"""F-10 -- ground-truth torso-heading capture. TEXT ONLY (no audio), readable from ~2 m.

Drives the subject through the marked headings and motions the F-10 brief specifies, and writes the
wall-clock window for each so every frame can be labelled with the TRUE heading afterwards. It does
NOT touch the production pipeline -- the OAK sidecar runs unmodified alongside.

The heading is rendered in a large block font because the subject is standing 2 m from the screen and
there is no speaker. Holds are long and transitions are generous, so a mis-timed frame costs accuracy
at the edges of a block, not the block itself; the analyser trims each block's first and last 25 %.

FLOOR SETUP (do this once, it is the ground truth)
--------------------------------------------------
  * Mark a spot on the floor for the subject's feet. They stay on it for the whole run.
  * From that spot, lay a protractor of headings on the floor: 0 deg facing the camera, then
    +-30, +-45, +-60, +-90. Painter's tape and a printed protractor are enough; what matters is that
    the SHOULDER LINE is what turns, and that the subject can register against the marks repeatably.
  * POSITIVE is defined here as the subject turning their LEFT shoulder TOWARDS the camera.
    (The analyser only needs consistency; it reports the convention it inferred.)
  * Keep camera position, subject distance, lighting and framing fixed within a run.
  * Repeat the whole run at each distance you want (see --distance), moving only the foot spot.

    python f10_gt_capture.py --distance near
    python f10_gt_capture.py --distance mid
    python f10_gt_capture.py --distance far
"""
import argparse
import json
import os
import sys
import time

# 5x7 block font. ASCII '#' on purpose: the Windows console is cp1252 and cannot encode
# the U+2588 block character, which crashes the whole capture mid-run.
FONT = {
    "0": ["###", "# #", "# #", "# #", "# #", "# #", "###"],
    "1": ["  #", "  #", "  #", "  #", "  #", "  #", "  #"],
    "2": ["###", "  #", "  #", "###", "#  ", "#  ", "###"],
    "3": ["###", "  #", "  #", "###", "  #", "  #", "###"],
    "4": ["# #", "# #", "# #", "###", "  #", "  #", "  #"],
    "5": ["###", "#  ", "#  ", "###", "  #", "  #", "###"],
    "6": ["###", "#  ", "#  ", "###", "# #", "# #", "###"],
    "7": ["###", "  #", "  #", "  #", "  #", "  #", "  #"],
    "8": ["###", "# #", "# #", "###", "# #", "# #", "###"],
    "9": ["###", "# #", "# #", "###", "  #", "  #", "###"],
    "+": ["   ", "   ", " # ", "###", " # ", "   ", "   "],
    "-": ["   ", "   ", "   ", "###", "   ", "   ", "   "],
    " ": ["   ", "   ", "   ", "   ", "   ", "   ", "   "],
    "L": ["#  ", "#  ", "#  ", "#  ", "#  ", "#  ", "###"],
    "R": ["###", "# #", "# #", "###", "# #", "# #", "# #"],
    "O": ["###", "# #", "# #", "# #", "# #", "# #", "###"],
    "*": [" # ", "###", " # ", "   ", "   ", "   ", "   "],
}


def big(text):
    rows = ["" for _ in range(7)]
    for ch in text:
        g = FONT.get(ch.upper(), FONT[" "])
        for i in range(7):
            rows[i] += g[i] + " "
    return rows



# --- top-down diagram -----------------------------------------------------------------------
# Console cells are about twice as tall as wide, so x is scaled by 2 or every angle looks wrong.
W, H, CX, CY, XS = 31, 14, 15, 8, 2.0


def draw_topdown(heading):
    """Bird's-eye view: where the shoulders point, and which way the face looks.

    Screen axes are x right / y DOWN, camera at the top. Facing angle is measured from
    'toward the camera', positive = the subject turning to their own RIGHT:
        face = (sin h, -cos h)      left shoulder = (fy, -fx)
    At h = +90 that puts the LEFT shoulder toward the camera, which is what standing
    side-on to the right actually looks like from above.
    """
    import math as _m
    g = [[" "] * W for _ in range(H)]
    lab = "[ C A M E R A ]"
    x0 = (W - len(lab)) // 2
    for i, ch in enumerate(lab):
        g[0][x0 + i] = ch
    for y in (1, 2):
        g[y][CX] = "|"
    g[2][CX] = "v"

    if heading is None:
        for i, ch in enumerate("<< keep moving >>"):
            g[CY][(W - 17) // 2 + i] = ch
        return ["   " + "".join(r).rstrip() for r in g]

    h = _m.radians(heading)
    fx, fy = _m.sin(h), -_m.cos(h)
    lx, ly = fy, -fx                      # the subject's LEFT
    for t in [i * 0.25 for i in range(-16, 17)]:
        x = int(round(CX + lx * t * XS))
        y = int(round(CY + ly * t))
        if 4 <= y < H and 0 <= x < W and abs(t) <= 4:
            g[y][x] = "#"
    # shoulder ends
    for t, mark in ((4.0, "L"), (-4.0, "R")):
        x = int(round(CX + lx * t * XS))
        y = int(round(CY + ly * t))
        if 4 <= y < H and 0 <= x < W:
            g[y][x] = mark
    # nose
    nx = int(round(CX + fx * 3.0 * XS))
    ny = int(round(CY + fy * 3.0))
    if 4 <= ny < H and 0 <= nx < W:
        g[ny][nx] = "@"
    return ["   " + "".join(r).rstrip() for r in g]


def show(label, sub, remaining, total, heading=0):
    os.system("cls" if os.name == "nt" else "clear")
    print()
    for line in big(label):
        print("   " + line)
    print()
    for line in draw_topdown(heading):
        print(line)
    print()
    print("   " + sub)
    width = 46
    done = int(width * (total - remaining) / max(1, total))
    print("   [" + "#" * done + "." * (width - done) + "]  %2ds left" % remaining)
    sys.stdout.flush()


# name, heading in degrees (None = not a static heading), hold seconds, instruction
#
# FULL set -- needs floor marks at every angle.
STATIC_FULL = [
    ("h_0",    0,   8, "FACE THE CAMERA squarely"),
    ("h_p30", +30,  8, "turn LEFT shoulder toward camera, 30 mark"),
    ("h_p45", +45,  8, "same direction, 45 mark"),
    ("h_p60", +60,  8, "same direction, 60 mark"),
    ("h_p90", +90,  8, "same direction, 90 mark (side on)"),
    ("h_0b",   0,   6, "back to FACING THE CAMERA"),
    ("h_m30", -30,  8, "turn RIGHT shoulder toward camera, 30 mark"),
    ("h_m45", -45,  8, "same direction, 45 mark"),
    ("h_m60", -60,  8, "same direction, 60 mark"),
    ("h_m90", -90,  8, "same direction, 90 mark (side on)"),
    ("h_0c",   0,   6, "back to FACING THE CAMERA"),
]
# COARSE set -- NO floor marks required. 0 and +-90 are PHYSICALLY SELF-EVIDENT (square to the lens,
# and shoulders edge-on to the lens), which is what makes them usable as ground truth with no setup.
# The +-45 blocks are eyeballed halfway positions and carry perhaps +-5-10 deg of their own error;
# they are recorded and reported as ESTIMATED so no conclusion rests on them alone.
STATIC_COARSE = [
    ("h_0",    0,  10, "LOOK STRAIGHT AT THE CAMERA"),
    ("h_p45", +45,  8, "TURN HALFWAY TO YOUR RIGHT"),
    ("h_p90", +90, 10, "TURN ALL THE WAY RIGHT - stand sideways"),
    ("h_0b",   0,   8, "FACE THE CAMERA AGAIN"),
    ("h_m45", -45,  8, "TURN HALFWAY TO YOUR LEFT"),
    ("h_m90", -90, 10, "TURN ALL THE WAY LEFT - stand sideways"),
    ("h_0c",   0,   8, "FACE THE CAMERA AGAIN"),
]
# Which headings were eyeballed rather than physically anchored -- carried into the marks file so the
# analyser and the report can weight them correctly.
ESTIMATED_HEADINGS = (45, -45)

# F-11 -- HEAD/TORSO DECOUPLING. The failure mode that matters for a mirror product: the user turns
# their BODY while keeping their FACE on the mirror. A nose-based sign source goes blind exactly there.
# Recorded as (name, TORSO heading = what we estimate, HEAD heading, hold, instruction).
DECOUPLE = [
    ("dc_body45R_face0",  +45,   0, 10, "BODY halfway RIGHT but KEEP LOOKING AT THE CAMERA"),
    ("dc_body45L_face0",  -45,   0, 10, "BODY halfway LEFT but KEEP LOOKING AT THE CAMERA"),
    ("dc_body0_head45R",    0, +45, 10, "BODY STAYS FACING THE CAMERA - turn only your HEAD right"),
    ("dc_body0_head45L",    0, -45, 10, "BODY STAYS FACING THE CAMERA - turn only your HEAD left"),
]

# TASK 5 -- torso twist: hips and shoulders independently. heading = SHOULDER heading.
TWIST = [
    ("tw_rigid_p45", +45, 8, "TURN HALFWAY RIGHT - move your FEET too"),
    ("tw_sh_p45",    +45, 8, "FEET STAY STILL - twist only your CHEST halfway right"),
    ("tw_sh_m45",    -45, 8, "FEET STAY STILL - twist only your CHEST halfway left"),
    ("tw_opp_a",     None, 8, "HIPS point LEFT while your CHEST turns RIGHT"),
    ("tw_opp_b",     None, 8, "HIPS point RIGHT while your CHEST turns LEFT"),
]
# TASK 6 -- realistic motion. No static heading, so heading is None.
MOTION = [
    ("m_slow",      None, 12, "TURN SLOWLY left and right, keep going"),
    ("m_normal",    None, 10, "TURN AT NORMAL SPEED, left and right"),
    ("m_fast",      None, 10, "TURN FAST, left and right"),
    ("m_rev_lr",    None, 10, "TURN LEFT then SNAP BACK RIGHT, repeat"),
    ("m_rev_rl",    None, 10, "TURN RIGHT then SNAP BACK LEFT, repeat"),
    ("m_180",       None, 12, "TURN RIGHT AROUND - show the camera your back - then come back"),
    ("m_turn_arms", None, 10, "TURN while WAVING BOTH ARMS"),
    ("m_turn_still", None, 10, "TURN with your ARMS HANGING STILL"),
]
TRANSITION = 7          # seconds to move between marks (generous: a rushed move costs a block)


def _label(heading):
    """Plain L/R labels: '45R' reads instantly, '+45' needs a convention explained first."""
    if heading is None:
        return "* *"
    if heading == 0:
        return "0"
    return "%d%s" % (abs(heading), "R" if heading > 0 else "L")


def run_block(name, heading, hold, instr, marks, head_gt=None):
    label = _label(heading)
    for r in range(TRANSITION, 0, -1):
        show(label, "GET READY: " + instr, r, TRANSITION, heading)
        time.sleep(1.0)
    t0 = time.time()
    for r in range(hold, 0, -1):
        show(label, "NOW: " + instr, r, hold, heading)
        time.sleep(1.0)
    t1 = time.time()
    marks.append({"name": name, "headingGT": heading, "headGT": head_gt,
                  "t0": round(t0, 4), "t1": round(t1, 4), "instr": instr})



def intro_screen():
    """Show every position up front, then WAIT. The subject sets the pace, not the script."""
    os.system("cls" if os.name == "nt" else "clear")
    print()
    print("   ================= WHAT YOU ARE GOING TO DO =================")
    print()
    print("   You stand in one spot and turn your body to 5 positions.")
    print("   The screen shows a bird's-eye view, as if looking down from")
    print("   the ceiling.  '@' is your face.  'L'/'R' are your shoulders.")
    print()
    for h, name, what in ((0, "0", "look straight at the camera"),
                          (45, "45R", "turn HALFWAY to your RIGHT"),
                          (90, "90R", "turn ALL THE WAY right - stand sideways"),
                          (-45, "45L", "turn HALFWAY to your LEFT"),
                          (-90, "90L", "turn ALL THE WAY left - sideways")):
        print("   ---- %-4s %s" % (name, what))
        for line in draw_topdown(h):
            print(line)
        print()
    print("   Hold each one still until the bar runs out. Feet may move.")
    print("   Then some easier ones: twist your chest with feet still, and")
    print("   turning left/right at different speeds. The screen will say.")
    print()
    print("   ============================================================")
    print()
    try:
        input("   Read it, then press ENTER when you are ready to walk over... ")
    except EOFError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", required=True, choices=("near", "mid", "far"),
                    help="which foot spot this run uses; recorded in the marks file")
    ap.add_argument("--metres", type=float, default=0.0, help="measured distance, if known")
    ap.add_argument("--out-dir", default="oak_v4_evidence")
    ap.add_argument("--skip-motion", action="store_true")
    ap.add_argument("--focus", default="all", choices=("all", "sign"),
                    help="sign = static headings + head/torso decoupling only (the F-11 question), "
                         "skipping twist and motion so the run is ~3 min")
    ap.add_argument("--headings", default="coarse", choices=("coarse", "full"),
                    help="coarse = 0/+-45/+-90 with NO floor marks needed (0 and +-90 are physically "
                         "self-evident); full = the nine-heading protocol, which needs marks")
    ap.add_argument("--lead", type=int, default=30, help="seconds to walk to the mark before it starts")
    a = ap.parse_args()

    static = STATIC_FULL if a.headings == "full" else STATIC_COARSE
    if a.focus == "sign":
        blocks = static + [(n, t, h, i) for (n, t, _hg, h, i) in DECOUPLE]
        head_gt = dict((d[0], d[2]) for d in DECOUPLE)
    else:
        blocks = static + [(n, t, h, i) for (n, t, _hg, h, i) in DECOUPLE] + TWIST +             (([] if a.skip_motion else MOTION))
        head_gt = dict((d[0], d[2]) for d in DECOUPLE)
    total = sum(b[2] + TRANSITION for b in blocks) + a.lead
    os.system("cls" if os.name == "nt" else "clear")
    print("\n  F-10 GROUND-TRUTH TORSO HEADING CAPTURE   distance=%s" % a.distance)
    print("  %d blocks, about %d s. Stand on the foot mark, full body in frame." % (len(blocks), total))
    print("  The big number is the FLOOR MARK to turn your SHOULDERS to.")
    print("  '* *' means a moving block -- follow the line underneath instead.\n")
    intro_screen()
    for r in range(a.lead, 0, -1):
        show("0", "walk to your spot and FACE THE CAMERA", r, a.lead, 0)
        time.sleep(1.0)

    marks = []
    t_start = time.time()
    for name, heading, hold, instr in blocks:
        run_block(name, heading, hold, instr, marks, head_gt.get(name))

    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, "f10_gt_marks_%s.json" % a.distance)
    # F-15 GUARD. The marks filename is derived only from --distance, so a second run at the same
    # distance silently overwrote an earlier capture's block windows -- that is exactly how the F-10
    # marks were destroyed (recovered later by f12_recover_f10_marks.py; see ADR-041). The subject's
    # time is already spent by this point, so refusing here would throw the capture away: side-step
    # to a timestamped name instead, and say so loudly.
    if os.path.exists(out):
        alt = os.path.join(a.out_dir, "f10_gt_marks_%s_%s.json"
                           % (a.distance, time.strftime("%Y%m%d_%H%M%S")))
        print("\n  !! %s ALREADY EXISTS -- refusing to overwrite a previous capture." % out)
        print("  !! writing to %s instead. Pass --out-dir next time." % alt)
        out = alt
    with open(out, "w") as f:
        json.dump({"distance": a.distance, "metres": a.metres, "headings": a.headings,
                   "estimatedHeadings": list(ESTIMATED_HEADINGS) if a.headings == "coarse" else [],
                   "start": round(t_start, 4), "end": round(time.time(), 4),
                   "transition_s": TRANSITION, "blocks": marks}, f, indent=2)
    os.system("cls" if os.name == "nt" else "clear")
    print("\n  DONE -- %d blocks, %.0f s" % (len(marks), time.time() - t_start))
    print("  marks -> %s\n" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
