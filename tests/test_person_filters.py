#!/usr/bin/env python3
"""F-33 tests for person_filters.py - the per-person filter chain.

AGENTS.md section 8 names the cases new signal-processing logic must cover: steady state, legitimate
fast motion, an isolated spike, a HIGH-CONFIDENCE wrong value, a short gap, a long gap, and recovery.
Each is below with a synthetic joint and an injected clock.

Two more are specific to this module and are the ones that would actually bite in production:

  * THE POOL IS KEYED BY IDENTITY, NOT POSITION. PersonTracker emits most-established-first, so a
    person's index changes whenever somebody else gains a hit. A bank addressed by index hands one
    person's filter history to another; the test drives that exact re-ordering.
  * THE CONFIG HAS NOT DRIFTED from the single-person sender. FilterConfig duplicates that parser's
    defaults, so this reads wholebody_udp_sender.py's SOURCE and compares them. Change a number
    there without changing it here and this fails.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import person_filters as PF

PASS = 0
FAIL = 0


def check(cond, label, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  PASS %-58s %s" % (label, detail))
    else:
        FAIL += 1
        print("  FAIL %-58s %s" % (label, detail))


#: A tracked joint with no parent complications. WholeBody 9 is the left wrist: in DEFAULT_TRACKED,
#: in the limb group, and the joint whose spikes P0-2 was built for.
WRIST = 9
FPS = 16.0          # the multi-person loop rate with three people, not 30
DT = 1.0 / FPS


def blank():
    """A plausible standing body: everything measured, everything confident."""
    xyz = np.zeros((133, 3), dtype=np.float32)
    for i in range(133):
        xyz[i] = (0.0, 0.0, 2.5)
    xyz[11] = (-0.12, 0.0, 2.5)     # left hip
    xyz[12] = (0.12, 0.0, 2.5)      # right hip
    xyz[5] = (-0.18, -0.50, 2.5)    # left shoulder
    xyz[6] = (0.18, -0.50, 2.5)
    xyz[7] = (-0.22, -0.25, 2.5)    # left elbow
    xyz[WRIST] = (-0.25, 0.0, 2.5)  # left wrist
    measured = np.ones(133, dtype=bool)
    conf = np.full(133, 0.9, dtype=np.float32)
    return xyz, measured, conf


def drive(bank, frames, mutate=None, t0=0.0):
    """Run `frames` updates through the whole chain; return the emitted wrist track per frame."""
    out = []
    t = t0
    for k in range(frames):
        t = t + DT
        xyz, measured, conf = blank()
        if mutate is not None:
            mutate(k, xyz, measured, conf)
        bank.observe_rate(t)
        bank.smooth(xyz, measured)
        mid = bank.mid_hip(xyz, measured)
        conf_emit, res = bank.refine(xyz, measured, conf, t)
        out.append({"t": t, "xyz": xyz[WRIST].copy(), "mid": None if mid is None else mid.copy(),
                    "conf": float(conf_emit[WRIST]), "measured": bool(measured[WRIST]),
                    "state": None if res is None else int(res[WRIST][4])})
    return out


print("")
print("=== the sample-rate estimate ===")

b = PF.PersonFilters(1, PF.FilterConfig(), now=0.0)
drive(b, 20)
check(abs(b.freq - FPS) < 0.5, "a person measures their OWN update rate",
      "%.2f fps (fed %.1f)" % (b.freq, FPS))

b = PF.PersonFilters(2, PF.FilterConfig(), now=0.0)
t = 0.0
for k in range(24):
    # Every third frame this person is skipped by --max-poses, so the gap TRIPLES. A mean would be
    # dragged by that for the length of the window; the median must ignore it.
    t = t + (DT * 3.0 if k % 3 == 0 else DT)
    b.observe_rate(t)
check(abs(b.freq - FPS) < 1.0, "an occasional skipped frame does not move the estimate",
      "%.2f fps" % b.freq)

b = PF.PersonFilters(3, PF.FilterConfig(adaptive_rate=False), now=0.0)
drive(b, 20)
check(abs(b.freq - 30.0) < 1e-6, "--no-adaptive-rate pins the filters at 30 fps (the A/B arm)",
      "%.2f fps" % b.freq)

b = PF.PersonFilters(4, PF.FilterConfig(), now=0.0)
t = 0.0
for _ in range(20):
    t = t + 1.0 / 400.0     # absurd, e.g. a replay with a broken clock
    b.observe_rate(t)
check(b.freq <= PF.FilterConfig().max_freq, "the rate estimate is clamped, not trusted blindly",
      "%.1f fps, clamp %.0f" % (b.freq, PF.FilterConfig().max_freq))


print("")
print("=== AGENTS.md section 8: the required cases ===")

b = PF.PersonFilters(10, PF.FilterConfig(), now=0.0)
rows = drive(b, 60)
tail = np.asarray([r["xyz"] for r in rows[20:]])
spread = float(np.max(np.linalg.norm(tail - tail.mean(axis=0), axis=1)))
check(spread < 0.001, "STEADY STATE: a still joint stays still", "spread %.4f m" % spread)
check(all(r["state"] == 0 for r in rows[10:]), "STEADY STATE: and reports TRACKED")


def fast(k, xyz, measured, conf):
    # 1.5 m/s to the left, which is a brisk reach, not a spike.
    xyz[WRIST] = (-0.25 - 1.5 * k * DT, 0.0, 2.5)


b = PF.PersonFilters(11, PF.FilterConfig(), now=0.0)
rows = drive(b, 40)          # settle
rows = drive(b, 40, fast, t0=rows[-1]["t"])
travelled = abs(float(rows[-1]["xyz"][0] - rows[0]["xyz"][0]))
expected = 1.5 * 39 * DT
check(travelled > 0.80 * expected, "FAST MOTION: a genuine 1.5 m/s reach is followed, not eaten",
      "%.2f m of an expected %.2f m" % (travelled, expected))
check(all(r["conf"] > 0.0 for r in rows), "FAST MOTION: and is never gated off")


def spike(k, xyz, measured, conf):
    if k == 20:
        xyz[WRIST] = (-0.25, 0.0, 9.0)      # a classic depth blow-out


b = PF.PersonFilters(12, PF.FilterConfig(), now=0.0)
rows = drive(b, 40, spike)
worst = max(float(np.linalg.norm(rows[i]["xyz"] - rows[i - 1]["xyz"])) for i in range(1, len(rows)))
check(worst < 0.40, "ISOLATED SPIKE: a 6.5 m depth jump is slewed, not emitted",
      "worst single-frame step %.3f m (cap %.2f)" % (worst, PF.FilterConfig().arm_max_jump))
after = np.asarray([r["xyz"] for r in rows[30:]])
check(float(np.max(np.abs(after[:, 2] - 2.5))) < 0.25,
      "ISOLATED SPIKE: and the joint returns to the truth afterwards",
      "residual %.3f m" % float(np.max(np.abs(after[:, 2] - 2.5))))


# "Confidently wrong" has two shapes and the chain answers them in two different places, so both
# are driven. F-22's convention (pose_validation.py): a STRAIGHT limb reads ~0 deg and a fully
# folded one approaches 180; an elbow is rejected past 160, which is the forearm doubled back
# further than an arm goes.
def folded_elbow(xyz):
    upper = xyz[7] - xyz[5]
    upper = upper / float(np.linalg.norm(upper))
    ang = np.radians(170.0)
    rot = np.array([upper[0] * np.cos(ang) - upper[1] * np.sin(ang),
                    upper[0] * np.sin(ang) + upper[1] * np.cos(ang), 0.0], dtype=np.float32)
    return xyz[7] + 0.25 * rot


b = PF.PersonFilters(13, PF.FilterConfig(), now=0.0)
t = 0.0
elbow_conf = []
for k in range(45):
    t = t + DT
    xyz, measured, conf = blank()
    if k >= 20:
        xyz[WRIST] = folded_elbow(xyz)
        conf[7] = 0.99                      # the model is SURE about an impossible arm
    b.observe_rate(t)
    b.smooth(xyz, measured)
    b.mid_hip(xyz, measured)
    ce, _res = b.refine(xyz, measured, conf, t)
    elbow_conf.append(float(ce[7]))
check(elbow_conf[10] > 0.0, "HIGH-CONFIDENCE WRONG: a healthy elbow is emitted")
check(min(elbow_conf[35:]) == 0.0,
      "HIGH-CONFIDENCE WRONG (angle): an impossible elbow is REFUSED at conf 0.99",
      "emit confidence %.2f" % min(elbow_conf[35:]))


def drift(k, xyz, measured, conf):
    # The sender's own --inject-drift-file case: 0.85 m of linear drift over 25 frames, held at high
    # confidence throughout. No single step is big enough for the displacement cap to notice.
    if k >= 15:
        step = min(1.0, (k - 14) / 25.0)
        xyz[WRIST] = (-0.25 + 0.85 * step, 0.0, 2.5)
        conf[WRIST] = 0.85


b = PF.PersonFilters(16, PF.FilterConfig(), now=0.0)
rows = drive(b, 60, drift)
drifted = abs(float(rows[-1]["xyz"][0] + 0.25))
# THIS PINS A LIMITATION, NOT A CAPABILITY, and it is worth more than a green tick that hides it.
# The chain follows the drift almost exactly: no single 34 mm step is large enough for the
# displacement cap, and P1-1's residual gate adapts to the joint's own residual scale, so a ramp
# looks like motion to every layer that is running. Catching this is precisely what P1-4 does, and
# P1-4 is REJECTED for production (docs/P1_4_CLOSEOUT_2026-09-08.md), so the honest statement is
# that multi-person inherits the single-person behaviour here - including its blind spot.
check(drifted > 0.80, "HIGH-CONFIDENCE WRONG (drift): a slow confident drift is NOT caught - "
                      "inherited limit, P1-4's job",
      "wrist followed %.0f mm of an injected 850 mm" % (drifted * 1000.0))
b = PF.PersonFilters(17, PF.FilterConfig(recovery=True), now=0.0)
rows = drive(b, 60, drift)
with_p14 = abs(float(rows[-1]["xyz"][0] + 0.25))
check(with_p14 <= drifted, "HIGH-CONFIDENCE WRONG (drift): P1-4 is the layer that would, if enabled",
      "%.0f mm with --filter-recovery vs %.0f mm without"
      % (with_p14 * 1000.0, drifted * 1000.0))


def short_gap(k, xyz, measured, conf):
    if 20 <= k < 24:            # 4 frames, well inside max_hold_frames=8
        measured[WRIST] = False
        conf[WRIST] = 0.0


b = PF.PersonFilters(14, PF.FilterConfig(), now=0.0)
rows = drive(b, 45, short_gap)
held = rows[20:24]
check(all(r["measured"] for r in held), "SHORT GAP: a 4-frame dropout is HELD, not dropped")
check(all(float(np.linalg.norm(r["xyz"] - rows[19]["xyz"])) < 0.01 for r in held),
      "SHORT GAP: and holds the last good value, not a back-projected guess")


def long_gap(k, xyz, measured, conf):
    if 20 <= k < 40:            # 20 frames, far past the hold AND past the predict horizon
        measured[WRIST] = False
        conf[WRIST] = 0.0


b = PF.PersonFilters(15, PF.FilterConfig(), now=0.0)
rows = drive(b, 60, long_gap)
check(rows[38]["conf"] == 0.0,
      "LONG GAP: a 20-frame dropout is eventually REFUSED, never frozen forever",
      "state %s, emit confidence %.2f" % (rows[38]["state"], rows[38]["conf"]))
check(any(r["conf"] > 0.0 for r in rows[45:]),
      "RECOVERY: and the joint comes back once it is measured again",
      "emitting again by frame %d"
      % next(i for i, r in enumerate(rows) if i > 40 and r["conf"] > 0.0))


print("")
print("=== the pool: a chain belongs to an IDENTITY ===")

pool = PF.PersonFilterPool()
a1 = pool.acquire(7, 0.0)
a2 = pool.acquire(7, 0.5)
check(a1 is a2, "the same id gets the same chain")
check(pool.acquire(8, 0.5) is not a1, "a different id gets a different chain")

# The re-ordering trap, driven directly. Person 7 is at index 0, then person 8 overtakes them.
pool = PF.PersonFilterPool()
order = [(7, 8), (7, 8), (8, 7), (8, 7), (7, 8)]
seen = {}
t = 0.0
for ids in order:
    t = t + DT
    for pid in ids:
        bank = pool.acquire(pid, t)
        seen.setdefault(pid, []).append(id(bank))
check(all(len(set(v)) == 1 for v in seen.values()),
      "re-ordering the emitted list never swaps two people's history",
      "ids %s kept %d chains" % (sorted(seen), len(pool.banks)))

pool = PF.PersonFilterPool()
pool.acquire(1, 0.0)
pool.acquire(2, 0.0)
pool.acquire(2, 5.0)
dropped = pool.release_idle(5.0)
check(dropped == [1], "an idle chain is released; a live one is kept", "dropped %s" % dropped)
check(pool.snapshot()["banks"] == 1, "and the pool does not leak",
      "%s" % pool.snapshot())
check(pool.release(2) and not pool.release(2), "an explicit release is idempotent")

# The M11 hold is PER PERSON: sharing one would place everybody at whoever was seen last.
pool = PF.PersonFilterPool()
xyz_a, meas_a, _c = blank()
xyz_b, meas_b, _c = blank()
xyz_b[11] = (2.88, 0.0, 4.0)
xyz_b[12] = (3.12, 0.0, 4.0)
mid_a = pool.acquire(1, 0.0).mid_hip(xyz_a, meas_a)
mid_b = pool.acquire(2, 0.0).mid_hip(xyz_b, meas_b)
gone = np.zeros(133, dtype=bool)
hold_a = pool.acquire(1, DT).mid_hip(xyz_a, gone)
check(float(np.linalg.norm(mid_a - mid_b)) > 1.0, "two people have two different origins",
      "%.2f m apart" % float(np.linalg.norm(mid_a - mid_b)))
check(float(np.linalg.norm(hold_a - mid_a)) < 1e-6,
      "a hip depth-hole holds THAT person's origin, not the other person's",
      "held %s" % np.round(hold_a, 2).tolist())


print("")
print("=== ADR-071: the feet and the head are in a filter group ===")

check(PF.FOOT_INDICES.isdisjoint(PF.SINGLE_PERSON_LIMB_INDICES),
      "the feet really were in NO single-person filter group")
check(PF.FACE_INDICES.isdisjoint(PF.SINGLE_PERSON_LIMB_INDICES),
      "the head really was in NO single-person filter group")
b_on = PF.PersonFilters(20, PF.FilterConfig())
b_off = PF.PersonFilters(21, PF.FilterConfig(filter_feet=False, filter_head=False))
check(PF.FOOT_INDICES <= b_on._smoother.limb_indices
      and PF.FACE_INDICES <= b_on._smoother.limb_indices,
      "by default F-33 puts both in one")
check(b_off._smoother.limb_indices == set(PF.SINGLE_PERSON_LIMB_INDICES),
      "--no-filter-feet --no-filter-head restores the single-person grouping EXACTLY",
      "%d indices" % len(b_off._smoother.limb_indices))


def heel_gap(k, xyz, measured, conf):
    if k >= 20:
        measured[19] = False       # left heel loses depth and never gets it back
        conf[19] = 0.0


for feet, label, want_hold in ((True, "with ADR-071", True), (False, "single-person grouping", False)):
    b = PF.PersonFilters(22, PF.FilterConfig(filter_feet=feet), now=0.0)
    t = 0.0
    heels = []
    for k in range(30):
        t = t + DT
        xyz, measured, conf = blank()
        xyz[19] = (-0.14, 0.42, 2.5)
        heel_gap(k, xyz, measured, conf)
        b.observe_rate(t)
        b.smooth(xyz, measured)
        heels.append(bool(measured[19]))
    check(heels[22] == want_hold,
          "a heel dropout is %s (%s)" % ("HELD" if want_hold else "dropped to the fallback", label),
          "measured=%s at frame 22" % heels[22])


print("")
print("=== the config has not drifted from the single-person sender ===")

src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "wholebody_udp_sender.py"), encoding="utf-8").read()
cfg = PF.FilterConfig()
#: (sender flag, FilterConfig attribute). Only the knobs F-33 reuses; head_max_jump has no
#: single-person counterpart by design (ADR-071) and is deliberately absent.
SHARED = (("--min-cutoff", "min_cutoff"), ("--beta", "beta"), ("--max-jump", "max_jump"),
          ("--arm-max-jump", "arm_max_jump"), ("--leg-max-jump", "leg_max_jump"),
          ("--depth-min-cutoff", "depth_min_cutoff"), ("--depth-beta", "depth_beta"),
          ("--max-hold-frames", "max_hold_frames"),
          ("--tracker-predict-frames", "tracker_predict_frames"),
          ("--tracker-reacquire-frames", "tracker_reacquire_frames"),
          ("--recovery-max-frames", "recovery_max_frames"),
          ("--recovery-blend-frames", "recovery_blend_frames"))
for flag, attr in SHARED:
    m = re.search(r'add_argument\("' + re.escape(flag) + r'",[^)]*?default=([0-9.]+)', src,
                  re.S)
    if m is None:
        check(False, "found %s in the sender" % flag, "regex did not match - has it been renamed?")
        continue
    sender = float(m.group(1))
    mine = float(getattr(cfg, attr))
    check(abs(sender - mine) < 1e-9, "%s matches the single-person default" % flag,
          "sender %.3f == FilterConfig.%s %.3f" % (sender, attr, mine))

check(not cfg.recovery, "P1-4 defaults OFF here, as it does in production")
check(cfg.pose_validation and cfg.tracker and cfg.smooth,
      "P0, P1-1 and F-22 default ON here, as they do in production")

print("")
print("-" * 78)
print("%d/%d assertions passed" % (PASS, PASS + FAIL))
sys.exit(0 if FAIL == 0 else 1)
