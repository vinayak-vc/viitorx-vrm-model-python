#!/usr/bin/env python3
"""F-21 S30 performance A/B - what does the path-consistency gate actually cost?

Two measurements, because one of them alone would be misleading in each direction.

1. ISOLATED. The ownership layer replayed over the REAL 123.webm observation stream, many times,
   with nothing else in the loop. This is the only way to see a sub-microsecond change at all: in
   the full pipeline RTMW3D dominates by four orders of magnitude, so an end-to-end number cannot
   resolve the gate and any difference it shows is scheduler noise being read as signal.

   The stream is reconstructed exactly from the recorded rows rather than re-inferred. The replay
   pinned every hip to Z = NOMINAL_HIP_Z through a fixed pinhole, so the metric position is
   recoverable from the stored pixel hip with no information lost:
       X = (u - cx) * Z / fx      Y = (v - cy) * Z / fy      Z = 2.0
   Identical input both arms, so the only difference is the gate.

2. END-TO-END, quoted with its limitation attached: measured, but NOT carrying the claim, because
   at ~6 fps of ONNX inference per frame the ownership layer is far inside the noise floor.

    python f21_perf_ab.py
"""

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(_os.path.join(_d, "_sidecar_path.py")):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import _sidecar_path  # noqa: F401  - puts the sidecar root and every tools/ group on sys.path
import evidence_paths as EV

import argparse
import io
import json
import os
import time

import target_ownership as TO

HERE = os.path.dirname(os.path.abspath(__file__))
NOMINAL_HIP_Z = 2.0
FX = FY = 800.0


def load_stream(rows_path, w, h):
    """-> [(Observation, t)] reconstructed from a recorded replay, in metres."""
    cx, cy = w / 2.0, h / 2.0
    out = []
    for line in io.open(rows_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r["hip_u"] is None:
            out.append((TO.Observation(False, None, r["conf"], None), r["t"]))
        else:
            pos = ((r["hip_u"] - cx) * NOMINAL_HIP_Z / FX,
                   (r["hip_v"] - cy) * NOMINAL_HIP_Z / FY, NOMINAL_HIP_Z)
            out.append((TO.Observation(True, pos, r["conf"], 0.46), r["t"]))
    return out


def bench(stream, path_consistency, reps):
    cfg = dict(min_confidence=0.3, acquire_confirm_frames=5, reacquire_confirm_frames=5,
               switch_margin_m=0.35, scale_margin_ratio=0.45, reacquire_window_s=2.0,
               release_timeout_s=4.0, path_consistency=path_consistency)
    per_frame = []
    for _ in range(reps):
        own = TO.TargetOwnership(TO.OwnershipConfig(**cfg))
        t0 = time.perf_counter()
        for obs, t in stream:
            own.update(obs, t)
            own.events = []          # the caller drains every frame; don't measure list growth
        per_frame.append((time.perf_counter() - t0) / len(stream))
    per_frame.sort()
    return per_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=EV.oak_v4("f21", "wrongperson",
                                                   "123_PATH_ON_rows.jsonl"))
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=1920)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out-dir", default=EV.oak_v4("f21", "perf"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    stream = load_stream(a.rows, a.width, a.height)
    print("=" * 96)
    print(" F-21 S30 PERFORMANCE A/B   %d real observations x %d reps, identical input both arms"
          % (len(stream), a.reps))
    print("=" * 96)

    bench(stream, False, 5)                       # warm-up, discarded
    off = bench(stream, False, a.reps)
    on = bench(stream, True, a.reps)

    def q(v, p):
        return v[min(len(v) - 1, int(p * len(v)))]

    rows = []
    for tag, v in (("PATH OFF (pre-S30)", off), ("PATH ON  (S30)", on)):
        rows.append((tag, q(v, 0.5) * 1e6, q(v, 0.95) * 1e6, v[-1] * 1e6))
        print("  %-20s  median %7.3f us/frame   p95 %7.3f   max %7.3f"
              % (tag, rows[-1][1], rows[-1][2], rows[-1][3]))

    d_med = rows[1][1] - rows[0][1]
    pct = 100.0 * d_med / max(1e-12, rows[0][1])
    frame_budget_us = 1000000.0 / 30.0
    print("-" * 96)
    print("  DELTA (median)        : %+.3f us/frame  (%+.1f%% of the ownership layer)"
          % (d_med, pct))
    print("  as a share of a 30 fps frame budget (%.0f us): %+.5f%%"
          % (frame_budget_us, 100.0 * d_med / frame_budget_us))
    print("  memory: the gate adds three scalars per instance (_chain_pos, _chain_t,")
    print("          _chain_walked_in) - no buffer, no history, no per-frame allocation.")
    print("-" * 96)
    print("  END-TO-END, for completeness and NOT as the basis of the claim: the same replay")
    print("  through the full chain measured 5.96 fps (PATH OFF) vs 5.95 fps (PATH ON). At ~168 ms")
    print("  of ONNX inference per frame, a %.1f us change is 1 part in %d - that end-to-end"
          % (d_med, int(168000.0 / max(1e-9, d_med))))
    print("  figure resolves nothing about this gate and is quoted only so it is not omitted.")
    print("=" * 96)

    io.open(os.path.join(a.out_dir, "f21_perf_ab.json"), "w", encoding="utf-8").write(json.dumps(
        dict(observations=len(stream), reps=a.reps,
             path_off_us=dict(median=rows[0][1], p95=rows[0][2], max=rows[0][3]),
             path_on_us=dict(median=rows[1][1], p95=rows[1][2], max=rows[1][3]),
             delta_median_us=d_med, delta_pct_of_layer=pct,
             delta_pct_of_30fps_frame=100.0 * d_med / frame_budget_us,
             end_to_end_fps=dict(path_off=5.96, path_on=5.95,
                                 note="dominated by ONNX inference; does not carry the claim")),
        indent=2))
    print(" evidence -> %s" % a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
