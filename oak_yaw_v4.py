#!/usr/bin/env python3
"""TORSO YAW V4 -- characterise the REAL OAK-D torso-yaw signal from recorded sidecar logs.

WHY THIS EXISTS
---------------
`UNITY_TORSO_YAW_V4_VALIDATION_2026-09-09.md` sec 5a states no recorded log carries a usable OAK
torso-yaw signal, because "recv_log.jsonl has shoulder and hip z = 0.0". That is true of the HIPS
only. `sender_log.jsonl` logs `sh = [lm[11][:3], lm[12][:3]]` AFTER `build_body_landmarks`, and with
`--flatten-trunk` defaulting to FALSE (Milestone-2) the SHOULDER z is the measured stereo depth.
So the real depth-derived yaw signal IS recorded, across tens of thousands of frames.

WHAT IS COMPUTED
----------------
Exactly Unity's own derivation, ported from the shipping C#:

  KMath.Find2DAngle(cx, cy, ex, ey) = atan2(ey - cy, ex - cx)
  KMath.RollPitchYaw2(a, b).y       = NormalizeAngle(Find2DAngle(a.z, a.x, b.z, b.x))
  KMath.NormalizeAngle(r)           = wrap(r, -pi, pi) / pi          -> roughly [-1, 1]

  KalidokitPoseSolver.CalcHipsAndSpine:
      spine.y = RollPitchYaw2(lm[11], lm[12]).y
      if spine.y > 0.5: spine.y -= 2
      spine.y += 0.5
      spineEuler.y = spine.y * pi                                     -> RADIANS

THE DEGENERATE CASE, which drives the headline result
-----------------------------------------------------
`CalcHipsAndSpine` reads lm[11], lm[12], lm[23], lm[24] with NO confidence or visibility check
(P0 LimbGate gates arms and legs only). When a landmark pair is undetected the sidecar emits
[0,0,0], and then atan2(0,0) = 0 -> normalize 0 -> +0.5 -> *pi = +pi/2, i.e. the solver commands
EXACTLY +90 deg of torso yaw. Verified numerically in `degenerate_check`. At torsoYawScale = 0 that
is multiplied away; at 0.75 it is a real command. Frames are therefore classified before any
statistic is taken.

NOTE ON SIGN: `mirrorSagittal` negates X downstream, which flips the SIGN of both yaws but changes
no magnitude, no step size and no stability property. Every metric here is sign-symmetric or
reported as |.|, so the mirror does not affect any conclusion.

    python oak_yaw_v4.py
"""
import io
import json
import math
import os

PI = math.pi
HIPS_DAMP = 0.70
SPINE_DAMP = 0.45
CHEST_DAMP = 0.25
MAX_RATE_DEG = 140.0
TAU = 0.15
DZ_LO = 8.0
DZ_HI = 22.0
MIN_SPAN_M = 0.05          # a real shoulder/hip line is far wider than this
SENTINEL_DEG = 90.0


def normalize_angle(r):
    """KMath.NormalizeAngle -- wrap to (-pi, pi], then divide by pi."""
    a = math.fmod(r, 2.0 * PI)
    if a > PI:
        a -= 2.0 * PI
    elif a < -PI:
        a += 2.0 * PI
    return a / PI


def line_yaw_rad(a, b):
    """CalcHipsAndSpine's y channel, in radians, for a landmark pair (a=left, b=right)."""
    y = normalize_angle(math.atan2(b[0] - a[0], b[2] - a[2]))
    if y > 0.5:
        y -= 2.0
    y += 0.5
    return y * PI


def smoothstep(e0, e1, x):
    t = (x - e0) / (e1 - e0) if (e1 - e0) != 0 else 0.0
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


class DampYaw:
    """Verbatim port of KalidokitControlRigDriver.DampYaw. Radians in, radians out."""

    def __init__(self, max_rate_deg=MAX_RATE_DEG, tau=TAU, lo=DZ_LO, hi=DZ_HI):
        self.rate_limited = None
        self.smoothed = None
        self.max_rate = math.radians(max_rate_deg)
        self.tau = tau
        self.lo = math.radians(lo)
        self.hi = math.radians(hi)

    def step(self, raw, dt):
        if self.rate_limited is None:
            self.rate_limited = raw
            self.smoothed = raw
        max_step = self.max_rate * dt
        d = raw - self.rate_limited
        self.rate_limited += max(-max_step, min(max_step, d))
        a = 1.0 - math.exp(-dt / max(0.02, self.tau))
        self.smoothed += a * (self.rate_limited - self.smoothed)
        gate = smoothstep(0.0, 1.0, (abs(self.smoothed) - self.lo) / max(0.001, self.hi - self.lo))
        return self.smoothed * gate


def wrapped_delta(a, b):
    """Shortest signed difference b - a, in degrees, on the circle."""
    return (b - a + 180.0) % 360.0 - 180.0


def pct(v, p):
    if not v:
        return 0.0
    s = sorted(v)
    return s[int(round((len(s) - 1) * p / 100.0))]


def mean(v):
    return sum(v) / len(v) if v else 0.0


def load(path):
    rows = []
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        sh, hip = d.get("sh"), d.get("hip")
        if not sh or not hip or len(sh) != 2 or len(hip) != 2:
            continue
        sh_span = abs(float(sh[0][0]) - float(sh[1][0]))
        hip_span = abs(float(hip[0][0]) - float(hip[1][0]))
        sh_zero = all(abs(float(c)) < 1e-9 for p in sh for c in p[:3])
        rows.append({
            "t": float(d.get("t", 0.0)),
            "seq": int(d.get("seq", 0)),
            "cov": int(d.get("cov", 0)),
            "sh_yaw": math.degrees(line_yaw_rad(sh[0], sh[1])),
            "hip_yaw": math.degrees(line_yaw_rad(hip[0], hip[1])),
            "sh_dz": float(sh[0][2]) - float(sh[1][2]),
            "sh_span": sh_span,
            "hip_span": hip_span,
            "sh_zero": sh_zero,
            # VALID = a real, non-degenerate shoulder line the solver can be trusted on
            "valid": sh_span > MIN_SPAN_M and hip_span > MIN_SPAN_M,
        })
    return rows


def analyse(name, rows):
    if len(rows) < 10:
        print("  %-28s (too few usable frames: %d)" % (name, len(rows)))
        return None

    dts = [rows[i]["t"] - rows[i - 1]["t"] for i in range(1, len(rows))]
    dts = [d for d in dts if 0.0 < d < 1.0]
    med_dt = pct(dts, 50) if dts else 0.033
    dur = rows[-1]["t"] - rows[0]["t"]

    valid = [r for r in rows if r["valid"]]
    degen = [r for r in rows if not r["valid"]]
    sentinel = [r for r in rows if abs(r["sh_yaw"] - SENTINEL_DEG) < 1e-6]
    if not valid:
        print("  %-28s n=%-6d  NO VALID FRAMES (%d degenerate) -- excluded" % (name, len(rows), len(degen)))
        return None

    # transitions into the degenerate state while the stream is otherwise live
    drops = sum(1 for i in range(1, len(rows)) if rows[i - 1]["valid"] and not rows[i]["valid"])

    sh = [r["sh_yaw"] for r in valid]
    hp = [r["hip_yaw"] for r in valid]
    flat_sh = sum(1 for r in valid if abs(r["sh_dz"]) < 1e-9)

    # steps only between frames that are BOTH valid and temporally adjacent
    steps_sh, steps_hip, flips, revs = [], [], 0, 0
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        if not (a["valid"] and b["valid"]):
            continue
        dt = b["t"] - a["t"]
        if not (0.0 < dt < 0.5):
            continue
        d = wrapped_delta(a["sh_yaw"], b["sh_yaw"])
        steps_sh.append(abs(d))
        steps_hip.append(abs(wrapped_delta(a["hip_yaw"], b["hip_yaw"])))
        if abs(d) > 150.0:
            flips += 1
        if abs(a["sh_yaw"]) > 20 and abs(b["sh_yaw"]) > 20 and a["sh_yaw"] * b["sh_yaw"] < 0:
            revs += 1
    rate_sh = [s / med_dt for s in steps_sh] if steps_sh else [0.0]

    print("  %-28s n=%-6d %6.1fs @%4.1fHz   valid=%5d (%5.1f%%)  degenerate=%5d  90deg-sentinel=%5d" %
          (name, len(rows), dur, (1.0 / med_dt if med_dt else 0), len(valid),
           100.0 * len(valid) / len(rows), len(degen), len(sentinel)))
    print("      detection drops while streaming: %d      flat shoulder Z within valid: %d (%.1f%%)" %
          (drops, flat_sh, 100.0 * flat_sh / len(valid)))
    print("      shoulder yaw  p05=%7.1f p50=%7.1f p95=%7.1f  min=%7.1f max=%7.1f  |yaw|>22: %4.1f%%" %
          (pct(sh, 5), pct(sh, 50), pct(sh, 95), min(sh), max(sh),
           100.0 * sum(1 for v in sh if abs(v) > 22) / len(sh)))
    print("      hip yaw       p05=%7.1f p50=%7.1f p95=%7.1f  min=%7.1f max=%7.1f" %
          (pct(hp, 5), pct(hp, 50), pct(hp, 95), min(hp), max(hp)))
    if steps_sh:
        print("      frame steps   sh p50=%6.2f p95=%6.2f max=%7.2f deg  (p95 %5.0f deg/s)  hip p95=%6.2f" %
              (pct(steps_sh, 50), pct(steps_sh, 95), max(steps_sh), pct(rate_sh, 95), pct(steps_hip, 95)))
        print("      >150deg flips=%d   sign reversals(|yaw|>20)=%d   steps>140deg/s=%d (%.1f%%)" %
              (flips, revs, sum(1 for r in rate_sh if r > MAX_RATE_DEG),
               100.0 * sum(1 for r in rate_sh if r > MAX_RATE_DEG) / len(rate_sh)))
    return {"rows": rows, "med_dt": med_dt, "sh": sh, "name": name,
            "flips": flips, "revs": revs, "drops": drops, "valid": len(valid), "n": len(rows)}


def sweep(name, rows, med_dt, scales=(0.0, 0.25, 0.5, 0.75, 0.813, 1.0), gate_degenerate=False):
    """Push the recorded OAK yaw through the shipping conditioner + the V4 sec 2b composition.

    gate_degenerate=False reproduces the SHIPPING behaviour: the solver has no confidence gate, so
    the +90 deg sentinel is fed to DampYaw exactly as it would be in Unity.
    gate_degenerate=True holds the previous valid yaw instead -- the counterfactual for a gate.
    """
    tag = "WITH a confidence gate (counterfactual)" if gate_degenerate else "SHIPPING (no gate)"
    print("\n  --- avatar shoulder-line yaw, %s -- %s ---" % (tag, name))
    print("  %6s | %8s %8s %8s | %8s %8s | %7s" %
          ("scale", "err mean", "err p95", "err max", "out p95", "out max", "gain"))
    out_rows = []
    for sc in scales:
        dh, ds = DampYaw(), DampYaw()
        errs, outs, gnum, gden = [], [], 0.0, 0.0
        prev_t = rows[0]["t"]
        last_sh, last_hip = 0.0, 0.0
        for r in rows:
            dt = r["t"] - prev_t
            prev_t = r["t"]
            if not (0.0 < dt < 1.0):
                dt = med_dt
            if r["valid"]:
                raw_sh, raw_hip = r["sh_yaw"], r["hip_yaw"]
                last_sh, last_hip = raw_sh, raw_hip
            elif gate_degenerate:
                raw_sh, raw_hip = last_sh, last_hip
            else:
                raw_sh, raw_hip = r["sh_yaw"], r["hip_yaw"]     # the +90 sentinel, as shipped
            h = math.degrees(dh.step(math.radians(raw_hip), dt))
            s = math.degrees(ds.step(math.radians(raw_sh), dt))
            out = min(1.0, sc) * (HIPS_DAMP * h + SPINE_DAMP * s + CHEST_DAMP * s)
            outs.append(abs(out))
            if r["valid"]:                                       # score only where truth is known
                errs.append(abs(wrapped_delta(out, r["sh_yaw"])))
                if abs(r["sh_yaw"]) > 22.0:
                    gnum += out * r["sh_yaw"]
                    gden += r["sh_yaw"] * r["sh_yaw"]
        print("  %6.3f | %8.2f %8.2f %8.2f | %8.2f %8.2f | %7.3f" %
              (sc, mean(errs), pct(errs, 95), max(errs), pct(outs, 95), max(outs),
               (gnum / gden if gden > 0 else 0.0)))
        out_rows.append((sc, mean(errs), max(errs), pct(outs, 95), max(outs)))
    return out_rows


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    dirs = sorted(d for d in os.listdir(here)
                  if d.startswith("pipeline_logs") and
                  os.path.exists(os.path.join(here, d, "sender_log.jsonl")))
    print("=" * 112)
    print(" REAL OAK-D TORSO YAW -- recorded sidecar logs, Unity's own derivation, degeneracy-classified")
    print("=" * 112)
    keep = []
    for d in dirs:
        r = analyse(d, load(os.path.join(here, d, "sender_log.jsonl")))
        if r:
            keep.append(r)
        print("")

    # the capture with the most valid frames AND real excursion is the one to sweep
    best = max(keep, key=lambda r: r["valid"] * (max(r["sh"]) - min(r["sh"])))
    print("=" * 112)
    print(" SWEEP CAPTURE: %s  (%d valid frames, yaw span %.1f deg)" %
          (best["name"], best["valid"], max(best["sh"]) - min(best["sh"])))
    print("=" * 112)
    sweep(best["name"], best["rows"], best["med_dt"], gate_degenerate=False)
    sweep(best["name"], best["rows"], best["med_dt"], gate_degenerate=True)

    print("\n" + "=" * 112)
    print(" POOLED over %d captures: %d frames, %d valid | >150deg flips=%d  reversals=%d  drops=%d"
          % (len(keep), sum(r["n"] for r in keep), sum(r["valid"] for r in keep),
             sum(r["flips"] for r in keep), sum(r["revs"] for r in keep), sum(r["drops"] for r in keep)))
    print("=" * 112)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
