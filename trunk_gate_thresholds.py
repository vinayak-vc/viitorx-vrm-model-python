#!/usr/bin/env python3
"""TORSO V5 -- derive the TrunkGate thresholds from the recorded corpus, rather than guessing them.

The gate must separate a REAL torso measurement from a degenerate one. Two families of threshold are
needed and each is taken from the data:

  * span floors   -- how wide is a real shoulder / hip line, in the landmark metric space?
  * jump ceiling  -- how fast does a REAL torso yaw actually move, frame to frame?

A threshold is only useful if the two populations separate. This prints both distributions so the
separation (or lack of it) is visible, and reports what each candidate threshold would cost in
false rejections of good frames.

    python trunk_gate_thresholds.py
"""
import io
import json
import math
import os

from oak_yaw_v4 import load, pct, mean, wrapped_delta


def stdev(v):
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    dirs = sorted(d for d in os.listdir(here)
                  if d.startswith("pipeline_logs") and
                  os.path.exists(os.path.join(here, d, "sender_log.jsonl")))
    sh_spans, hip_spans, degen_sh, degen_hip = [], [], [], []
    steps, rates = [], []
    n_total = 0
    for d in dirs:
        rows = load(os.path.join(here, d, "sender_log.jsonl"))
        n_total += len(rows)
        dts = [rows[i]["t"] - rows[i - 1]["t"] for i in range(1, len(rows))]
        dts = [x for x in dts if 0 < x < 1.0]
        med_dt = pct(dts, 50) or 0.0465
        for r in rows:
            # "real" = the frame has a plausible human torso at all; classified by BOTH spans being
            # non-trivial, which is deliberately a weaker test than the gate being designed.
            if r["sh_span"] > 0.15 and r["hip_span"] > 0.08:
                sh_spans.append(r["sh_span"])
                hip_spans.append(r["hip_span"])
            else:
                degen_sh.append(r["sh_span"])
                degen_hip.append(r["hip_span"])
        for i in range(1, len(rows)):
            a, b = rows[i - 1], rows[i]
            if not (a["valid"] and b["valid"]):
                continue
            dt = b["t"] - a["t"]
            if not (0 < dt < 0.5):
                continue
            s = abs(wrapped_delta(a["sh_yaw"], b["sh_yaw"]))
            steps.append(s)
            rates.append(s / dt)

    print("=" * 96)
    print(" TRUNK GATE THRESHOLDS -- from %d recorded frames across %d captures" % (n_total, len(dirs)))
    print("=" * 96)

    print("\n SHOULDER SPAN |lm[11].x - lm[12].x|  (metres, landmark space)")
    print("   plausible-torso frames (n=%d):  p01=%.4f p05=%.4f p50=%.4f p95=%.4f max=%.4f"
          % (len(sh_spans), pct(sh_spans, 1), pct(sh_spans, 5), pct(sh_spans, 50),
             pct(sh_spans, 95), max(sh_spans)))
    print("   degenerate frames      (n=%d):  p50=%.4f p95=%.4f max=%.4f"
          % (len(degen_sh), pct(degen_sh, 50), pct(degen_sh, 95), max(degen_sh) if degen_sh else 0))

    print("\n HIP SPAN |lm[23].x - lm[24].x|")
    print("   plausible-torso frames:  p01=%.4f p05=%.4f p50=%.4f p95=%.4f max=%.4f"
          % (pct(hip_spans, 1), pct(hip_spans, 5), pct(hip_spans, 50), pct(hip_spans, 95), max(hip_spans)))
    print("   degenerate frames:       p50=%.4f p95=%.4f max=%.4f"
          % (pct(degen_hip, 50), pct(degen_hip, 95), max(degen_hip) if degen_hip else 0))

    print("\n cost of each candidate SPAN floor (fraction of plausible frames it would reject):")
    print("   %-10s %12s %12s" % ("floor (m)", "shoulder", "hip"))
    for f in (0.02, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15):
        rs = 100.0 * sum(1 for v in sh_spans if v < f) / len(sh_spans)
        rh = 100.0 * sum(1 for v in hip_spans if v < f) / len(hip_spans)
        print("   %-10.2f %11.3f%% %11.3f%%" % (f, rs, rh))

    print("\n SHOULDER-YAW FRAME STEP, valid->valid only (n=%d)" % len(steps))
    print("   deg   : p50=%.2f p90=%.2f p95=%.2f p99=%.2f p99.9=%.2f max=%.2f"
          % (pct(steps, 50), pct(steps, 90), pct(steps, 95), pct(steps, 99), pct(steps, 99.9), max(steps)))
    print("   deg/s : p50=%.1f p90=%.1f p95=%.1f p99=%.1f p99.9=%.1f max=%.1f"
          % (pct(rates, 50), pct(rates, 90), pct(rates, 95), pct(rates, 99), pct(rates, 99.9), max(rates)))

    print("\n cost of each candidate JUMP ceiling (fraction of valid transitions it would reject):")
    print("   %-14s %12s" % ("ceiling (deg/s)", "rejected"))
    for c in (140, 200, 300, 400, 500, 600, 800, 1000):
        r = 100.0 * sum(1 for v in rates if v > c) / len(rates)
        print("   %-14d %11.3f%%" % (c, r))
    print("\n NOTE: the conditioner already rate-limits at 140 deg/s. The GATE's ceiling is a different")
    print("       job -- it rejects the physically IMPOSSIBLE so it never enters the filter state at")
    print("       all, so it must sit well ABOVE the conditioner's slew or it would fight it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
