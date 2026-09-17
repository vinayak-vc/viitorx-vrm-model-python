#!/usr/bin/env python3
"""F-32 tests for assignment.py + person_tracker.py.

The case that matters is CROSSING: two people who swap screen positions must keep their ids. That is
the failure this module exists to prevent, and it is the one no amount of eyeballing a video catches
reliably, so it is pinned here with synthetic trajectories and an injected clock.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools

import numpy as np

import assignment
import person_tracker as PT

PASS = 0
FAIL = 0


def check(cond, label, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  PASS %-60s %s" % (label, detail))
    else:
        FAIL += 1
        print("  FAIL %-60s %s" % (label, detail))


def box(cx, cy, w=0.12, h=0.45):
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def brute_force(cost):
    n_r, n_c = cost.shape
    best = None
    for cols in itertools.permutations(range(n_c), min(n_r, n_c)):
        for rows in itertools.combinations(range(n_r), len(cols)):
            c = sum(cost[r, cc] for r, cc in zip(rows, cols))
            if best is None or c < best:
                best = c
    return best


print("ASSIGNMENT")
rng = np.random.RandomState(11)
worst = 0.0
for _ in range(200):
    m = rng.rand(rng.randint(1, 6), rng.randint(1, 6)) * 10
    got = assignment.total_cost(m, assignment.solve(m))
    worst = max(worst, abs(got - brute_force(m)))
check(worst < 1e-9, "optimal on 200 random rectangular matrices", "max error %.2e" % worst)

m = np.array([[1.0, 2.0], [2.0, 100.0]])
check(assignment.solve(m) == [(0, 1), (1, 0)],
      "prefers the globally cheap matching over the greedy one", "101 -> 4")
check(assignment.solve(np.zeros((0, 0))) == [], "empty matrix is legal")
try:
    assignment.solve(np.array([[np.inf]]))
    check(False, "rejects non-finite costs")
except ValueError:
    check(True, "rejects non-finite costs", "inf would propagate to nan")

print("")
print("TRACK LIFECYCLE")
t = PT.PersonTracker()
out = t.update([PT.Detection(box(0.3, 0.5), 0.9, 2.0)], 0.0)
check(out == [], "a single sighting is NOT yet a person", "TENTATIVE is never emitted")

for i in range(1, 3):
    out = t.update([PT.Detection(box(0.3, 0.5), 0.9, 2.0)], i * 0.09)
check(len(out) == 1 and out[0].state == PT.CONFIRMED,
      "confirmed after CONFIRM_HITS sightings", "id=%d" % out[0].id)

first_id = out[0].id
out = t.update([], 0.30)
check(len(out) == 1 and out[0].state == PT.LOST,
      "a missed frame makes it LOST, not gone", "still emitted, so a visitor does not flicker out")

out = t.update([PT.Detection(box(0.32, 0.5), 0.9, 2.0)], 0.40)
check(len(out) == 1 and out[0].id == first_id and out[0].state == PT.CONFIRMED,
      "reappearing keeps the SAME id", "id=%d" % out[0].id)

for k in range(20):
    out = t.update([], 0.5 + k * 0.1)
check(out == [], "released after the miss timeout", "%.1f s" % t.cfg.max_misses_seconds)

t2 = PT.PersonTracker()
t2.update([PT.Detection(box(0.3, 0.5), 0.9, 2.0)], 0.0)
out = t2.update([], 0.09)
check(out == [] and len(t2.tracks) == 0,
      "an unconfirmed blip dies immediately", "phantom people never reach the wire")

print("")
print("CROSSING - the case this module exists for")
t3 = PT.PersonTracker()
now = 0.0
for step in range(4):
    t3.update([PT.Detection(box(0.25 + step * 0.02, 0.5), 0.9, 2.0),
               PT.Detection(box(0.75 - step * 0.02, 0.5), 0.9, 3.0)], now)
    now += 0.09
tracks = t3.update([PT.Detection(box(0.31, 0.5), 0.9, 2.0),
                    PT.Detection(box(0.69, 0.5), 0.9, 3.0)], now)
near_id = [tr.id for tr in tracks if tr.depth < 2.5][0]
far_id = [tr.id for tr in tracks if tr.depth >= 2.5][0]
check(len(tracks) == 2, "two people tracked", "ids %d and %d" % (near_id, far_id))

for step in range(1, 6):
    now += 0.09
    lx = 0.31 + step * 0.08
    rx = 0.69 - step * 0.08
    tracks = t3.update([PT.Detection(box(lx, 0.5), 0.9, 2.0),
                        PT.Detection(box(rx, 0.5), 0.9, 3.0)], now)

by_depth = {}
for tr in tracks:
    by_depth[int(round(tr.depth))] = tr.id
check(by_depth.get(2) == near_id and by_depth.get(3) == far_id,
      "ids follow the PERSON through the crossing, not the screen position",
      "near=%s far=%s" % (by_depth.get(2), by_depth.get(3)))

print("")
print("DEPTH AS AN ASSOCIATION SIGNAL")
t4 = PT.PersonTracker()
now = 0.0
for _ in range(4):
    t4.update([PT.Detection(box(0.5, 0.5), 0.9, 1.5),
               PT.Detection(box(0.52, 0.5), 0.9, 3.5)], now)
    now += 0.09
tracks = t4.update([PT.Detection(box(0.5, 0.5), 0.9, 1.5),
                    PT.Detection(box(0.52, 0.5), 0.9, 3.5)], now)
check(len(tracks) == 2,
      "two people at the SAME screen position but 2 m apart stay separate",
      "image-plane IoU alone could not do this")

t5 = PT.PersonTracker()
now = 0.0
for _ in range(4):
    t5.update([PT.Detection(box(0.4, 0.5), 0.9, 2.0)], now)
    now += 0.09
tracks = t5.update([PT.Detection(box(0.42, 0.5), 0.9, 6.0)], now)
check(len(tracks) == 1 and tracks[0].state == PT.LOST,
      "a detection that teleports 4 m in depth is REFUSED", "held as LOST rather than adopted")

print("")
print("DEGRADATION")
t6 = PT.PersonTracker()
now = 0.0
for _ in range(4):
    t6.update([PT.Detection(box(0.3, 0.5), 0.9, None),
               PT.Detection(box(0.7, 0.5), 0.9, None)], now)
    now += 0.09
tracks = t6.update([PT.Detection(box(0.31, 0.5), 0.9, None),
                    PT.Detection(box(0.71, 0.5), 0.9, None)], now)
check(len(tracks) == 2, "works with NO depth at all", "falls back to image-plane association")

t7 = PT.PersonTracker()
now = 0.0
for _ in range(4):
    t7.update([PT.Detection(box(0.3, 0.5), 0.9, 2.0)], now)
    now += 0.09
before = t7.update([PT.Detection(box(0.3, 0.5), 0.9, 2.0)], now)[0].id
now += 0.09
after = t7.update([PT.Detection(box(0.3, 0.5), 0.20, 2.0)], now)
check(len(after) == 1 and after[0].id == before,
      "a WEAK detection still updates an established person",
      "a partly-occluded person is still that person")

t8 = PT.PersonTracker()
out = t8.update([PT.Detection(box(0.3, 0.5), 0.20, 2.0)], 0.0)
for i in range(1, 5):
    out = t8.update([PT.Detection(box(0.3, 0.5), 0.20, 2.0)], i * 0.09)
check(out == [], "but a weak detection never BIRTHS a person", "min_score gates birth only")

print("")
print("EMISSION ORDER AND IDS")
t9 = PT.PersonTracker()
now = 0.0
for _ in range(6):
    t9.update([PT.Detection(box(0.2, 0.5), 0.9, 2.0)], now)
    now += 0.09
for _ in range(3):
    t9.update([PT.Detection(box(0.2, 0.5), 0.9, 2.0),
               PT.Detection(box(0.8, 0.5), 0.9, 2.5)], now)
    now += 0.09
tracks = t9.update([PT.Detection(box(0.2, 0.5), 0.9, 2.0),
                    PT.Detection(box(0.8, 0.5), 0.9, 2.5)], now)
check(tracks[0].hits > tracks[1].hits,
      "most-established person is emitted FIRST",
      "a host that can only afford N poses takes the N most likely to persist")

ids = set()
t10 = PT.PersonTracker()
now = 0.0
for cycle in range(3):
    for _ in range(4):
        t10.update([PT.Detection(box(0.3, 0.5), 0.9, 2.0)], now)
        now += 0.09
    for tr in t10.update([PT.Detection(box(0.3, 0.5), 0.9, 2.0)], now):
        ids.add(tr.id)
    for _ in range(20):
        now += 0.1
        t10.update([], now)
check(len(ids) == 3, "a retired id is NEVER reused", "saw ids %s" % sorted(ids))

print("")
print("-" * 78)
print("%d/%d assertions passed" % (PASS, PASS + FAIL))
sys.exit(0 if FAIL == 0 else 1)
