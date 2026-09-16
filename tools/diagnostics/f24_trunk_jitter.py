#!/usr/bin/env python3
"""Measure the OAK-D's LIVE trunk-yaw rest jitter, from the production wire.

THE NUMBER THIS EXISTS TO GET. KalidokitControlRigDriver gates trunk yaw through a deadzone:

    yawDeadzoneLoDeg = 8    |yaw| below this -> forced to frontal
    yawDeadzoneHiDeg = 22   full gain only above this

8 deg was chosen against a V4-era measurement of 5.4-7.6 deg of REST JITTER - the yaw the pipeline
reports while the subject stands still. Everything below that threshold is noise by assumption. But
the F-23 dance replay measured a real human's pelvic twist at std 7.4 deg, i.e. THE SAME SIZE as the
assumed noise, which is why 76.6 % of that dancer's hip rotation was multiplied by zero.

So the deadzone cannot be argued about further without a current measurement of the noise floor on
THIS rig, with THIS configuration. That is what this produces:

    hip yaw and shoulder yaw, per frame, while the subject holds still
    -> std, p95 and frame-to-frame step, in degrees

Yaw is computed exactly as TrunkGate does - atan2(dx, dz) over the landmark pair, reported as
deviation from frontal - so the number is directly comparable to the deadzone it informs.

Run the STILL block and the MOVE block in one session and pass --split, or analyse a single block.

    .venv\\Scripts\\python.exe f24_trunk_jitter.py --wire oak_v4_evidence/f24/wire.jsonl
"""
import argparse
import io
import json
import math
import sys

import numpy as np

L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24
DEADZONE_LO, DEADZONE_HI = 8.0, 22.0


def load(path):
    out = []
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def yaw_series(rows, a, b):
    """Deviation from frontal, degrees. Frontal is atan2(span, 0) = 90 deg."""
    out = []
    for d in rows:
        lm = d.get("lm")
        if not lm:
            out.append(np.nan)
            continue
        p, q = lm[a], lm[b]
        if p[3] <= 0.0 or q[3] <= 0.0:
            out.append(np.nan)
            continue
        dx, dz = p[0] - q[0], p[2] - q[2]
        if abs(dx) < 1e-6 and abs(dz) < 1e-6:
            out.append(np.nan)
            continue
        out.append(math.degrees(math.atan2(dx, dz)) - 90.0)
    return np.array(out, dtype=float)


def describe(name, v):
    m = ~np.isnan(v)
    if m.sum() < 10:
        print("  %-16s NOT MEASURABLE - only %d valid frames" % (name, m.sum()))
        return None
    v = v[m]
    step = np.abs(np.diff(v))
    print("  %-16s std %5.2f deg   p95|yaw| %5.2f   peak-to-peak %6.2f   "
          "frame-step p95 %5.2f deg   n=%d"
          % (name, v.std(), np.percentile(np.abs(v), 95), v.max() - v.min(),
             np.percentile(step, 95) if len(step) else float("nan"), len(v)))
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wire", required=True)
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the recording")
    ap.add_argument("--end", type=float, default=0.0, help="0 = to the end")
    ap.add_argument("--label", default="BLOCK")
    a = ap.parse_args()

    rows = load(a.wire)
    if not rows:
        print("no packets in %s" % a.wire)
        return 2
    t0 = rows[0].get("_rx", 0)
    lo = t0 + a.start
    hi = t0 + a.end if a.end > 0 else float("inf")
    rows = [r for r in rows if lo <= r.get("_rx", 0) <= hi]
    if not rows:
        print("no packets in the requested window")
        return 2

    dur = rows[-1]["_rx"] - rows[0]["_rx"]
    measured = sum(1 for r in rows if sum(r.get("src", [])) > 0)
    print("=" * 92)
    print("%s - %d frames over %.1f s (%.1f fps), %d frames carry measured stereo depth"
          % (a.label, len(rows), dur, len(rows) / max(1e-6, dur), measured))
    print("=" * 92)

    hip = describe("HIP yaw", yaw_series(rows, L_HIP, R_HIP))
    sh = describe("SHOULDER yaw", yaw_series(rows, L_SH, R_SH))

    print()
    print("Against the driver's deadzone (lo %.0f deg, hi %.0f deg):" % (DEADZONE_LO, DEADZONE_HI))
    for name, v in (("hip", hip), ("shoulder", sh)):
        if v is None:
            continue
        inside = 100.0 * (np.abs(v) <= DEADZONE_LO).mean()
        above = 100.0 * (np.abs(v) >= DEADZONE_HI).mean()
        print("  %-9s %5.1f %% of frames sit INSIDE the deadzone, %4.1f %% reach full gain"
              % (name, inside, above))
    print()
    print("READ IT THIS WAY. On a STILL subject, everything here is noise: a deadzone at or above")
    print("the p95 keeps the avatar steady. On a MOVING subject the same figures are signal, and")
    print("whatever the deadzone removes is real motion the viewer never sees. The V4-era figure")
    print("the current 8 deg was set against was 5.4-7.6 deg.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
