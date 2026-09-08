#!/usr/bin/env python3
"""Correlate P1-4 rejection/recovery events with what the Unity avatar actually did.

The two capture passes are two different human performances, so a raw block-vs-block
comparison cannot separate "P1-4 caused this" from "the operator danced harder".
This script controls for that by comparing, WITHIN A SINGLE PASS, the avatar's
frame-to-frame rotation in windows that follow a P1-4 event against windows that
do not. Same body, same performance, same block -- only the event differs.

    python correlate_p14_visual.py --dir pipeline_logs_p14 --ctrl pipeline_logs_p13
"""
import argparse
import collections
import io
import json
import math
import os

FWD_KEYS = ("lhandF", "rhandF", "llowF", "rlowF",
            "luplegF", "ruplegF", "llowlegF", "rlowlegF")

# A P1-4 event is followed by the avatar for this long. 250 ms comfortably covers the
# 40 ms interpolation delay plus the 8-frame recovery blend at ~21 fps.
WINDOW_S = 0.25

P14_EVENTS = ("GEOMETRIC_REJECT", "RECONSTRUCT", "RECOVERING",
              "TRACKED->LOST", "WEAK->LOST", "PREDICTED->LOST", "SUSPICIOUS")


def load(p):
    out = []
    if not os.path.exists(p):
        return out
    for ln in io.open(p, encoding="utf-8"):
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
    return out


def pct(v, p):
    if not v:
        return 0.0
    s = sorted(v)
    return s[int(round((len(s) - 1) * p / 100.0))]


def ang(a, b):
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-9 or nb < 1e-9:
        return None
    d = sum(x * y for x, y in zip(a, b)) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, d))))


def frame_steps(model):
    """(tApply, max rotation step across all logged bones) per rendered frame."""
    out, prev = [], {}
    for m in model:
        t = m.get("tApply", 0.0)
        worst = 0.0
        for k in FWD_KEYS:
            v = m.get(k)
            if not v:
                continue
            if k in prev:
                a = ang(prev[k], v)
                if a is not None:
                    worst = max(worst, a)
            prev[k] = v
        out.append((t, worst))
    return out


def summarize(vals, label):
    return ("  %-34s n=%6d  p50=%5.2f  p95=%6.2f  p99=%6.2f  max=%7.2f  "
            ">10=%4d  >20=%4d  >45=%3d"
            % (label, len(vals), pct(vals, 50), pct(vals, 95), pct(vals, 99),
               max(vals) if vals else 0.0,
               sum(1 for x in vals if x > 10), sum(1 for x in vals if x > 20),
               sum(1 for x in vals if x > 45)))


def split(steps, marks, window):
    """Partition per-frame rotation steps into (follows an event, does not)."""
    marks = sorted(marks)
    near, far, i = [], [], 0
    for t, s in steps:
        while i < len(marks) and marks[i] < t - window:
            i += 1
        near.append(s) if (i < len(marks) and marks[i] <= t) else far.append(s)
    return near, far


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="pipeline_logs_p14")
    ap.add_argument("--ctrl", default="pipeline_logs_p13")
    a = ap.parse_args()

    for d, tag in ((a.dir, "P1-4"), (a.ctrl, "P1-3")):
        model = load(os.path.join(d, "model_log.jsonl"))
        holds = load(os.path.join(d, "holds_log.jsonl"))
        # P1-4 stage events carry seq but no epoch; recover it from the sender log so
        # every event kind lands on the same timeline as the rendered frames.
        seq_t = {}
        for r in load(os.path.join(d, "sender_log.jsonl")):
            seq_t[r.get("seq")] = r.get("t")
        keep = []
        for h in holds:
            if "t" not in h:
                tt = seq_t.get(h.get("seq"))
                if tt is None:
                    continue
                h["t"] = tt
            keep.append(h)
        holds = keep
        steps = frame_steps(model)
        print("=" * 100)
        print(" %s  (%s)   rendered frames=%d   events=%d" % (tag, d, len(steps), len(holds)))
        print("=" * 100)

        marks = [h["t"] for h in holds if h.get("event") in P14_EVENTS]
        near, far = split(steps, marks, WINDOW_S)
        print(" WITHIN %.0f ms AFTER a P1-4-class event vs everywhere else:" % (WINDOW_S * 1000))
        print(summarize(near, "follows event"))
        print(summarize(far, "no event"))
        if near and far:
            print("  -> p99 ratio %.2fx, >20deg rate %.3f%% vs %.3f%%"
                  % (pct(near, 99) / max(1e-6, pct(far, 99)),
                     100.0 * sum(1 for x in near if x > 20) / len(near),
                     100.0 * sum(1 for x in far if x > 20) / len(far)))
        print()

        # attribution: what preceded each large snap?
        big = [(t, s) for t, s in steps if s > 20.0]
        by_ev = collections.Counter()
        ev_at = sorted((h["t"], h.get("event", "?")) for h in holds)
        for t, s in big:
            hits = [e for tt, e in ev_at if t - WINDOW_S <= tt <= t]
            by_ev[hits[-1] if hits else "(no preceding event)"] += 1
        print(" ATTRIBUTION of the %d snaps > 20 deg/frame (last event within %.0f ms):"
              % (len(big), WINDOW_S * 1000))
        for k, v in by_ev.most_common(10):
            print("   %-28s %4d" % (k, v))
        print()

        # per-event-kind: worst avatar step in the 250 ms following each occurrence
        print(" PER-EVENT-KIND avatar response (worst rotation step in the next %.0f ms):"
              % (WINDOW_S * 1000))
        kinds = collections.defaultdict(list)
        for h in holds:
            e = h.get("event", "?")
            if e not in P14_EVENTS:
                continue
            t0 = h["t"]
            w = [s for t, s in steps if t0 <= t <= t0 + WINDOW_S]
            if w:
                kinds[e].append(max(w))
        for e in sorted(kinds):
            v = kinds[e]
            print("   %-22s n=%4d  p50=%5.2f  p95=%6.2f  max=%7.2f"
                  % (e, len(v), pct(v, 50), pct(v, 95), max(v)))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
