#!/usr/bin/env python3
"""ARM RETARGET V2 -- controlled before/after for the roll state machine.

Replays the SAME recorded source arm directions through the V1 and the V2 roll logic, so the two are
compared on byte-identical input. A live A/B could not do this: the human moves differently in the two
passes, which is exactly the confound that muddied the V1 live report's motions 4 and 5.

Input:  pipeline_logs_v2/video_source_dirs.jsonl  -- per render frame, the mirrored source upper/forearm
        directions in the control-rig frame, logged straight out of the running driver.

V1 logic: single threshold, sin(bend) >= 0.2 -> take cross(upper, fore) verbatim; else carry.
V2 logic: hysteresis enter 0.35 / exit 0.15, continuity guard (reject a reversed candidate or one
          implying > 90 deg of roll in a frame), debounced re-acquisition, 30 deg/frame bound.

Both are driven from the same vectors; only the state machine differs.

    python ab_roll_v1_v2.py
"""
import io
import json
import math
import os

BEND_SIN_MIN_V1 = 0.20
BEND_SIN_ENTER = 0.35
BEND_SIN_EXIT = 0.15
ROLL_STEP_MAX = 90.0
FLIP_CONFIRM = 6
ROLL_SLEW_MAX = 30.0
MIN_SEG = 1e-4


def dot(a, b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def mag(a): return math.sqrt(dot(a, a))


def norm(a):
    m = mag(a)
    return (0.0, 0.0, 0.0) if m < 1e-12 else (a[0]/m, a[1]/m, a[2]/m)


def signed_angle(a, b, axis):
    a, b = norm(a), norm(b)
    ang = math.degrees(math.acos(max(-1.0, min(1.0, dot(a, b)))))
    return ang if dot(cross(a, b), axis) >= 0 else -ang


def rotate_about(v, axis, deg):
    axis = norm(axis)
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return tuple(v[i]*c + cross(axis, v)[i]*s + axis[i]*dot(axis, v)*(1-c) for i in range(3))


def orthogonalise(v, axis):
    if dot(v, v) < MIN_SEG:
        return (0.0, 0.0, 0.0)
    p = tuple(v[i] - axis[i]*dot(v, axis) for i in range(3))
    return (0.0, 0.0, 0.0) if dot(p, p) < 1e-8 else norm(p)


class V1:
    """The shipped V1 logic: one threshold, no guard."""
    name = "V1 (single threshold sin 0.20)"

    def __init__(self):
        self.bend_normal = (0.0, 0.0, 0.0)

    def step(self, upper, fore):
        c = cross(upper, fore)
        bs = mag(c)
        if bs >= BEND_SIN_MIN_V1:
            n = tuple(x/bs for x in c)
            self.bend_normal = n
            return n, "ElbowPlane"
        p = orthogonalise(self.bend_normal, upper)
        if dot(p, p) > 1e-8:
            return p, "Held"
        return (0.0, 0.0, 0.0), "RestCarried"


class V2:
    """Hysteresis + continuity guard + debounced, rate-bounded re-acquisition."""
    name = "V2 (hysteresis 0.35/0.15 + guard)"

    def __init__(self):
        self.bend_normal = (0.0, 0.0, 0.0)
        self.locked = False
        self.pending = (0.0, 0.0, 0.0)
        self.count = 0

    def step(self, upper, fore):
        c = cross(upper, fore)
        bs = mag(c)
        gate = BEND_SIN_EXIT if self.locked else BEND_SIN_ENTER
        if bs < gate:
            self.locked = False
            self.pending = (0.0, 0.0, 0.0)
            self.count = 0
            p = orthogonalise(self.bend_normal, upper)
            if dot(p, p) > 1e-8:
                return p, "Held"
            return (0.0, 0.0, 0.0), "RestCarried"
        self.locked = True
        cand = tuple(x/bs for x in c)
        held = orthogonalise(self.bend_normal, upper)
        fresh = orthogonalise(cand, upper)
        if dot(held, held) < MIN_SEG or dot(fresh, fresh) < MIN_SEG:
            self.pending = (0.0, 0.0, 0.0)
            self.count = 0
            self.bend_normal = cand
            return cand, "ElbowPlane"
        signed = signed_angle(held, fresh, upper)
        step = abs(signed)
        believable = step <= ROLL_STEP_MAX and dot(cand, self.bend_normal) >= 0.0
        if believable:
            self.pending = (0.0, 0.0, 0.0)
            self.count = 0
        else:
            if bs >= BEND_SIN_ENTER and dot(self.pending, self.pending) > MIN_SEG and dot(cand, self.pending) > 0.0:
                self.count += 1
            else:
                self.count = 1 if bs >= BEND_SIN_ENTER else 0
            self.pending = cand
            if self.count < FLIP_CONFIRM:
                return held, "Guarded"
        if step > ROLL_SLEW_MAX:
            sign = -1.0 if signed < 0 else 1.0
            stepped = rotate_about(held, upper, ROLL_SLEW_MAX*sign)
            self.bend_normal = stepped
            return stepped, "Slewed"
        self.pending = (0.0, 0.0, 0.0)
        self.count = 0
        self.bend_normal = cand
        return cand, "ElbowPlane"


def pct(v, p):
    if not v:
        return 0.0
    s = sorted(v)
    return s[int(round((len(s)-1)*p/100.0))]


# Rest bases as measured live from the shipping VRM (logged by the driver at every avatar load):
#   left (upper=( 1.00,-0.05,0.00)  hinge=( 0.05, 0.97,-0.24))
#   right(upper=(-1.00,-0.05,0.00)  hinge=( 0.05,-0.97, 0.24))
REST = {
    "left":  (norm((1.0, -0.05, 0.0)), norm((0.05, 0.97, -0.24))),
    "right": (norm((-1.0, -0.05, 0.0)), norm((0.05, -0.97, 0.24))),
}


def carried_reference(rest_upper, rest_normal, upper):
    """FromToRotation(restUpper -> upper) applied to restNormal.

    This is the reference ArmAimSolver itself uses for RollDeg, and it is the ONLY correct one here:
    a FIXED world axis projected onto the bone (the obvious choice) becomes degenerate whenever the arm
    swings near that axis, and the fallback to a second axis injects a ~180 deg discontinuity of the
    MEASUREMENT. That artefact is symmetric across V1 and V2 and would mask the very difference this
    A/B exists to measure -- observed as an identical 173.91 deg 'max step' for both variants.
    """
    ax = cross(rest_upper, upper)
    if mag(ax) < 1e-9:
        return rest_normal
    ang = math.degrees(math.acos(max(-1.0, min(1.0, dot(rest_upper, upper)))))
    return norm(rotate_about(rest_normal, ax, ang))


def run(rows, machine_cls, arm):
    """Return per-frame roll (deg about the upper-arm axis, vs the roll-free reference) + sources."""
    m = machine_cls()
    rest_upper, rest_normal = REST[arm]
    rolls, srcs, bends = [], [], []
    for r in rows:
        if r["arm"] != arm:
            continue
        upper = norm(tuple(r["up"]))
        fore = norm(tuple(r["fore"]))
        if mag(upper) < 0.5:
            continue
        n, src = m.step(upper, fore)
        if mag(n) < 0.5:
            rolls.append(rolls[-1] if rolls else 0.0)
            srcs.append(src)
            bends.append(r["bend"])
            continue
        ref = carried_reference(rest_upper, rest_normal, upper)
        rolls.append(signed_angle(ref, n, upper))
        srcs.append(src)
        bends.append(r["bend"])
    return rolls, srcs, bends


def stats(rolls, bends):
    steps, bend_at = [], []
    for i in range(1, len(rolls)):
        d0 = rolls[i] - rolls[i-1]
        d = abs((d0 + 180.0) % 360.0 - 180.0)
        steps.append(d)
        bend_at.append(min(bends[i], bends[i-1]))
    return steps, bend_at


def main():
    path = "pipeline_logs_v2/video_source_dirs.jsonl"
    if not os.path.exists(path):
        print("missing %s" % path)
        return 1
    rows = [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]
    print("=" * 104)
    print(" ROLL STATE MACHINE A/B -- V1 vs V2 on IDENTICAL recorded video input (n=%d frames)" % len(rows))
    print("=" * 104)
    print(" %-6s %-34s %7s %7s %7s %7s %7s %7s %7s"
          % ("arm", "variant", "n", "p50", "p95", "p99", "max", ">45", ">90"))
    totals = {}
    for arm in ("left", "right"):
        for cls in (V1, V2):
            rolls, srcs, bends = run(rows, cls, arm)
            steps, bend_at = stats(rolls, bends)
            n45 = sum(1 for x in steps if x > 45)
            n90 = sum(1 for x in steps if x > 90)
            totals.setdefault(cls.name, [0, 0, 0.0, 0])
            totals[cls.name][0] += n45
            totals[cls.name][1] += n90
            totals[cls.name][2] = max(totals[cls.name][2], max(steps) if steps else 0.0)
            totals[cls.name][3] += len(steps)
            print(" %-6s %-34s %7d %7.2f %7.2f %7.2f %7.2f %7d %7d"
                  % (arm, cls.name, len(steps), pct(steps, 50), pct(steps, 95), pct(steps, 99),
                     max(steps) if steps else 0.0, n45, n90))
            big = [(s, b) for s, b in zip(steps, bend_at) if s > 45]
            if big:
                bs = [b for _s, b in big]
                print("        %-32s bend at the >45 deg events: min=%.1f p50=%.1f max=%.1f"
                      % ("", min(bs), pct(bs, 50), max(bs)))
            cnt = {}
            for s in srcs:
                cnt[s] = cnt.get(s, 0) + 1
            print("        %-32s sources: %s" % ("", ", ".join("%s=%.1f%%" % (k, 100.0*v/len(srcs))
                                                              for k, v in sorted(cnt.items()))))
    print("-" * 104)
    for name, (n45, n90, mx, n) in totals.items():
        print(" TOTAL %-40s frames=%6d  >45deg=%3d  >90deg=%3d  max step=%7.2f"
              % (name, n, n45, n90, mx))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
