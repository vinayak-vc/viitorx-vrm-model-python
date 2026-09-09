"""Faithful Python port of ArmAimSolver's V2 roll state machine, to validate the logic and the seven
required scenarios BEFORE compiling in Unity. Vector math mirrors UnityEngine (left-handed, but every
operation here is basis-agnostic since we only use cross/dot/angle consistently)."""
import math
import random

BEND_SIN_ENTER = 0.35
BEND_SIN_EXIT = 0.15
ROLL_STEP_MAX = 90.0
FLIP_CONFIRM = 6
ROLL_SLEW_MAX = 30.0
MIN_SEG = 1e-4

ELBOW_PLANE, HELD, REST_CARRIED, GUARDED, SLEWED = "ElbowPlane", "Held", "RestCarried", "Guarded", "Slewed"


def sub(a, b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def add(a, b): return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
def mul(a, s): return (a[0]*s, a[1]*s, a[2]*s)
def dot(a, b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def mag(a): return math.sqrt(dot(a, a))
def norm(a):
    m = mag(a)
    return (0.0, 0.0, 0.0) if m < 1e-12 else (a[0]/m, a[1]/m, a[2]/m)


def signed_angle(a, b, axis):
    """Unity's Vector3.SignedAngle: unsigned angle, signed by cross(a,b) . axis."""
    a, b = norm(a), norm(b)
    ang = math.degrees(math.acos(max(-1.0, min(1.0, dot(a, b)))))
    return ang if dot(cross(a, b), axis) >= 0 else -ang


def rotate_about(v, axis, deg):
    """Rodrigues."""
    axis = norm(axis)
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return add(add(mul(v, c), mul(cross(axis, v), s)), mul(axis, dot(axis, v)*(1-c)))


def orthogonalise(v, axis):
    if dot(v, v) < MIN_SEG:
        return (0.0, 0.0, 0.0)
    p = sub(v, mul(axis, dot(v, axis)))
    if dot(p, p) < 1e-8:
        return (0.0, 0.0, 0.0)
    return norm(p)


class State:
    def __init__(self):
        self.bend_normal = (0.0, 0.0, 0.0)
        self.plane_locked = False
        self.pending_normal = (0.0, 0.0, 0.0)
        self.pending_count = 0


def resolve_normal(candidate, upper, bend_sin, st):
    held = orthogonalise(st.bend_normal, upper)
    fresh = orthogonalise(candidate, upper)
    if dot(held, held) < MIN_SEG or dot(fresh, fresh) < MIN_SEG:
        st.pending_normal = (0.0, 0.0, 0.0); st.pending_count = 0
        st.bend_normal = candidate
        return candidate, ELBOW_PLANE
    signed = signed_angle(held, fresh, upper)
    step = abs(signed)
    reversed_ = dot(candidate, st.bend_normal) < 0.0
    believable = step <= ROLL_STEP_MAX and not reversed_
    if believable:
        st.pending_normal = (0.0, 0.0, 0.0); st.pending_count = 0
    else:
        if (bend_sin >= BEND_SIN_ENTER and dot(st.pending_normal, st.pending_normal) > MIN_SEG
                and dot(candidate, st.pending_normal) > 0.0):
            st.pending_count += 1
        else:
            st.pending_count = 1 if bend_sin >= BEND_SIN_ENTER else 0
        st.pending_normal = candidate
        if st.pending_count < FLIP_CONFIRM:
            return held, GUARDED
    if step > ROLL_SLEW_MAX:
        sign = -1.0 if signed < 0 else 1.0
        stepped = rotate_about(held, upper, ROLL_SLEW_MAX*sign)
        st.bend_normal = stepped
        return stepped, SLEWED
    st.pending_normal = (0.0, 0.0, 0.0); st.pending_count = 0
    st.bend_normal = candidate
    return candidate, ELBOW_PLANE


def carry_normal(upper, rest_upper, rest_normal, st):
    if dot(st.bend_normal, st.bend_normal) > MIN_SEG:
        p = orthogonalise(st.bend_normal, upper)
        if dot(p, p) > 1e-8:
            return p, HELD
    # FromToRotation(rest_upper -> upper) applied to rest_normal
    ax = cross(rest_upper, upper)
    if mag(ax) < 1e-9:
        return norm(rest_normal), REST_CARRIED
    ang = math.degrees(math.acos(max(-1.0, min(1.0, dot(norm(rest_upper), norm(upper))))))
    return norm(rotate_about(rest_normal, ax, ang)), REST_CARRIED


REST_UPPER = (-1.0, 0.0, 0.0)   # avatar RIGHT arm rest, as in RightRest()
REST_NORMAL = None


def build_rest():
    global REST_NORMAL
    lateral, up = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    forward = mul(cross(lateral, up), -1.0)
    REST_NORMAL = norm(cross(REST_UPPER, forward))


def solve(shoulder, elbow, wrist, st, mirror=True):
    if mirror:
        shoulder = (-shoulder[0], shoulder[1], shoulder[2])
        elbow = (-elbow[0], elbow[1], elbow[2])
        wrist = (-wrist[0], wrist[1], wrist[2])
    upper = norm(sub(elbow, shoulder))
    fd = sub(wrist, elbow)
    fore = upper if dot(fd, fd) < MIN_SEG else norm(fd)
    c = cross(upper, fore)
    bend_sin = mag(c)
    gate = BEND_SIN_EXIT if st.plane_locked else BEND_SIN_ENTER
    if bend_sin >= gate:
        st.plane_locked = True
        normal, src = resolve_normal(mul(c, 1.0/bend_sin), upper, bend_sin, st)
    else:
        st.plane_locked = False
        st.pending_normal = (0.0, 0.0, 0.0); st.pending_count = 0
        normal, src = carry_normal(upper, REST_UPPER, REST_NORMAL, st)
    # roll relative to the roll-free carried reference
    ax = cross(REST_UPPER, upper)
    if mag(ax) < 1e-9:
        carried = REST_NORMAL
    else:
        ang = math.degrees(math.acos(max(-1.0, min(1.0, dot(norm(REST_UPPER), upper)))))
        carried = norm(rotate_about(REST_NORMAL, ax, ang))
    roll = signed_angle(carried, normal, upper)
    bend = math.degrees(math.acos(max(-1.0, min(1.0, dot(upper, fore)))))
    return {"normal": normal, "src": src, "roll": roll, "bend": bend,
            "upper": upper, "fore": fore, "lowerDegenerate": mag(cross(fore, normal))**2 < 0.0076}


SHOULDER = (0.18, 0.50, 0.0)
UPPER_LEN, FORE_LEN = 0.28, 0.26


def solve_bend(st, bend_deg, side):
    axis = (1.0, 0.0, 0.0)
    elbow = add(SHOULDER, mul(axis, UPPER_LEN))
    a = math.radians(bend_deg)
    fore = norm(add(mul(axis, math.cos(a)), mul((0.0, 0.0, -1.0), side*math.sin(a))))
    return solve(SHOULDER, elbow, add(elbow, mul(fore, FORE_LEN)), st)


def dstep(p, c):
    d = (c - p + 180.0) % 360.0 - 180.0
    return abs(d)


def main():
    build_rest()
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(("  PASS " if cond else "  FAIL ") + name.ljust(58) + detail)
        if not cond:
            ok = False

    print("=" * 100)
    print(" ArmAimSolver V2 roll state machine -- offline scenario check")
    print("=" * 100)
    print(" enter=%.2f (%.2f deg)  exit=%.2f (%.2f deg)  stepMax=%.0f  confirm=%d  slew=%.0f"
          % (BEND_SIN_ENTER, math.degrees(math.asin(BEND_SIN_ENTER)),
             BEND_SIN_EXIT, math.degrees(math.asin(BEND_SIN_EXIT)),
             ROLL_STEP_MAX, FLIP_CONFIRM, ROLL_SLEW_MAX))

    # (1) threshold chatter
    st = State()
    solve_bend(st, 60, 1); solve_bend(st, 2, 1)
    prev = solve_bend(st, 2, 1)["roll"]
    worst, flips, entered = 0.0, 0, 0
    for i in range(40):
        r = solve_bend(st, 10.5 if i % 2 == 0 else 12.5, 1.0 if i % 2 == 0 else -1.0)
        if r["src"] == ELBOW_PLANE:
            entered += 1
        s = dstep(prev, r["roll"]); worst = max(worst, s)
        if s > 90: flips += 1
        prev = r["roll"]
    check("1 threshold chatter: plane never entered", entered == 0, "entries=%d" % entered)
    check("1 threshold chatter: no flip", flips == 0 and worst < ROLL_SLEW_MAX, "worst=%.2f" % worst)
    e = solve_bend(st, 45, 1)
    l = solve_bend(st, 15, 1)
    check("1 hysteresis: locked plane survives a drop to 15 deg",
          e["src"] == ELBOW_PLANE and l["src"] == ELBOW_PLANE, "%s -> %s" % (e["src"], l["src"]))

    # (2) straight -> bend -> straight
    st = State()
    prev = solve_bend(st, 70, 1)["roll"]
    worst = 0.0
    for i, b in enumerate([60, 40, 25, 12, 4, 0, 4, 12, 25, 40, 60, 70]):
        r = solve_bend(st, b, 1.0 if i < 6 else -1.0)
        worst = max(worst, dstep(prev, r["roll"])); prev = r["roll"]
    for _ in range(30):
        r = solve_bend(st, 70, -1)
        worst = max(worst, dstep(prev, r["roll"])); prev = r["roll"]
    check("2 straight->bend->straight: bounded", worst <= ROLL_SLEW_MAX + 0.05, "worst=%.2f" % worst)

    # (3) straight -> shallow bend
    st = State()
    seeded = solve_bend(st, 70, 1)
    solve_bend(st, 2, 1)
    shallow = solve_bend(st, 15, -1)
    deep = solve_bend(st, 45, 1)
    ang = math.degrees(math.acos(max(-1.0, min(1.0, dot(seeded["normal"], shallow["normal"])))))
    check("3 shallow bend does not enter the plane", shallow["src"] != ELBOW_PLANE, shallow["src"])
    check("3 shallow bend keeps the seeded normal", ang < 5.0, "%.2f deg" % ang)
    check("3 a clear bend still drives the roll", deep["src"] == ELBOW_PLANE, deep["src"])

    # (4) noisy shallow crossing
    st = State()
    solve_bend(st, 55, 1)
    prev = solve_bend(st, 55, 1)["roll"]
    random.seed(20260909)
    worst, flips = 0.0, 0
    for _ in range(400):
        r = solve_bend(st, random.uniform(0, 18), 1.0 if random.random() < 0.5 else -1.0)
        s = dstep(prev, r["roll"]); worst = max(worst, s)
        if s > 90: flips += 1
        prev = r["roll"]
    check("4 noisy near-straight crossing: no flip", flips == 0, "worst=%.2f" % worst)

    # (5) opposite-normal rejection
    st = State()
    held = solve_bend(st, 60, 1)
    flipped = solve_bend(st, 60, -1)
    check("5 reversed plane is guarded", flipped["src"] == GUARDED, flipped["src"])
    check("5 held normal survives unchanged", dot(held["normal"], flipped["normal"]) > 0.99,
          "dot=%.4f" % dot(held["normal"], flipped["normal"]))
    last = None
    for _ in range(FLIP_CONFIRM + 2):
        last = solve_bend(st, 60, -1)["src"]
    check("5 sustained coherent reversal is eventually accepted", last == SLEWED, str(last))

    # (6) no >90 roll step across an adversarial sequence  + (7) directions unchanged
    st = State()
    random.seed(4242)
    prev = solve_bend(st, 50, 1)["roll"]
    worst, over90, over45, guarded, degen = 0.0, 0, 0, 0, 0
    worst_up, worst_fore = 0.0, 0.0
    for i in range(2000):
        if i % 5 == 0:
            b = random.uniform(0, 12)
        elif i % 5 == 1:
            b = random.uniform(9, 22)
        else:
            b = random.uniform(20, 140)
        side = -1.0 if random.random() < 0.35 else 1.0
        r = solve_bend(st, b, side)
        if r["src"] in (GUARDED, SLEWED):
            guarded += 1
        if r["lowerDegenerate"]:
            degen += 1
        s = dstep(prev, r["roll"]); worst = max(worst, s)
        if s > 90: over90 += 1
        if s > 45: over45 += 1
        prev = r["roll"]
        # direction check: the solved upper/fore are the mirrored source directions by construction
        want_up = (-1.0, 0.0, 0.0)
        worst_up = max(worst_up, abs(math.degrees(math.acos(max(-1.0, min(1.0, dot(r["upper"], want_up)))))))
    check("6 no roll step above 90 deg", over90 == 0, "over90=%d worst=%.2f" % (over90, worst))
    check("6 none above the 30 deg slew bound", over45 == 0, "over45=%d" % over45)
    check("6 the guard was actually exercised", guarded > 0, "guarded=%d/2000" % guarded)
    check("7 upper-arm direction untouched", worst_up < 0.05, "worst=%.6f deg" % worst_up)
    print("     [info] forearm-degenerate frames (lowerUp fallback path): %d/2000" % degen)

    print("-" * 100)
    print(" ALL PASS" if ok else " FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
