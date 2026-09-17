#!/usr/bin/env python3
"""F-33 - what the per-person filter chain costs and what it buys, against KNOWN ground truth.

WHY THIS EXISTS AND THE VIDEO A/B DOES NOT REPLACE IT. On a recorded clip there is no truth, so the
only thing measurable is how much the output moves per frame - and any filter can drive that to zero
by refusing to move at all. A smoother that adds 300 ms of lag scores beautifully on jitter and is
unusable. This bench synthesises people whose exact position is known at every instant, so jitter and
LAG are measured together and the trade-off is visible instead of assumed.

WHAT IS REAL: person_filters.py, smoothing.py, joint_tracker.py, pose_validation.py - the production
modules, unmodified, driven exactly as multiperson_udp_sender.py drives them.

WHAT IS NOT: the people. Their trajectories are analytic and their sensor noise is a model, described
with its assumptions at FX_BASELINE_SUBPIXEL below. This measures the FILTER, not the camera and not
RTMW3D. A number from here is a statement about signal processing and must never be quoted as a
tracking result.

    .venv\\Scripts\\python.exe tools/diagnostics/f33_filter_bench.py
"""

import os as _os
import sys as _sys

_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(_os.path.join(_d, "_sidecar_path.py")):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import _sidecar_path  # noqa: F401

import argparse
import time

import numpy as np

import person_filters as PF

#: The noise model, stated once so every number below can be read against it.
#:   * DEPTH QUANTISATION grows with the square of distance - one disparity step is
#:     z^2 / (fx * baseline * subpixel) metres, ~16 mm at 2 m and ~62 mm at 4 m on this rig. This is
#:     the dominant term and the reason the depth axis gets its own heavier filter.
#:   * IMAGE-PLANE noise is roughly constant in pixels, so it shrinks with distance in metres.
#:   * DROPOUTS: a keypoint loses depth entirely on some frames (a dark sleeve, a reflective floor).
#:   * SPIKES: the depth window straddles an edge and returns the wall behind the person.
#:
#: THE ASSUMPTION, AND WHICH WAY IT IS WRONG: this uses 1/8-pixel subpixel. Production stereo has
#: subpixel OFF (oak_depth.STEREO_CONFIG), so the real quantisation step is up to 8x coarser than
#: modelled here - the HIGH_DENSITY preset and the F-08 percentile window recover some of that, by
#: an amount nobody has measured. So this bench UNDERSTATES the noise the filter faces. It shifts
#: every arm of the A/B together, which is what matters for comparing them.
FX_BASELINE_SUBPIXEL = 430.0 * 0.075 * 8.0
IMAGE_NOISE_PX = 1.2
IMAGE_FOCAL_PX = 430.0
DROPOUT_RATE = 0.06
SPIKE_RATE = 0.004
SPIKE_METRES = 1.5

#: The joints scored. P1-1 tracks exactly these twelve, so they are the ones every layer of the chain
#: has an opinion about; scoring a face keypoint would mix filtered and unfiltered behaviour.
SCORED = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
WRISTS = (9, 10)


def truth(person, t):
    """Where person `person` ACTUALLY is at time t, in camera metres. 133 keypoints.

    Person 0 stands at 2.2 m and reaches - the case that exposes lag, because a 0.5 Hz reach is fast
    enough that over-smoothing shows up as the hand arriving late.
    Person 1 walks across the view at 3.4 m - the case that exposes depth noise, because the
    quantisation step there is ~2.5x the near person's.
    """
    xyz = np.zeros((133, 3), dtype=np.float32)
    if person == 0:
        base_x, base_z = -0.55, 2.2
        sway = 0.04 * np.sin(2.0 * np.pi * 0.25 * t)
    else:
        base_x, base_z = -1.4 + 0.45 * t, 3.4
        sway = 0.09 * np.sin(2.0 * np.pi * 0.9 * t)      # a walking gait
    reach = 0.32 * np.sin(2.0 * np.pi * 0.5 * t)
    hip_y = 0.0 + sway
    layout = {5: (-0.19, -0.52), 6: (0.19, -0.52), 7: (-0.24, -0.26), 8: (0.24, -0.26),
              9: (-0.26, 0.0), 10: (0.26, 0.0), 11: (-0.12, 0.0), 12: (0.12, 0.0),
              13: (-0.13, 0.44), 14: (0.13, 0.44), 15: (-0.14, 0.86), 16: (0.14, 0.86),
              0: (0.0, -0.72), 19: (-0.14, 0.92), 22: (0.14, 0.92)}
    for j, (dx, dy) in layout.items():
        xyz[j] = (base_x + dx, hip_y + dy, base_z)
    # Both wrists reach forward and back: the motion the lag figure is measured on.
    xyz[9] = (base_x - 0.26, hip_y - 0.10 - abs(reach) * 0.3, base_z - abs(reach))
    xyz[10] = (base_x + 0.26, hip_y - 0.10 - abs(reach) * 0.3, base_z - abs(reach))
    for j in range(133):
        if not np.any(xyz[j]):
            xyz[j] = (base_x, hip_y, base_z)
    return xyz


def observe(xyz_true, rng):
    """The truth as the sensor would report it: noise, dropouts, spikes."""
    xyz = xyz_true.copy()
    measured = np.ones(133, dtype=bool)
    conf = np.full(133, 0.85, dtype=np.float32)
    z = xyz_true[:, 2]
    depth_sigma = (z * z) / FX_BASELINE_SUBPIXEL
    plane_sigma = z * (IMAGE_NOISE_PX / IMAGE_FOCAL_PX)
    xyz[:, 0] += rng.normal(0.0, 1.0, 133) * plane_sigma
    xyz[:, 1] += rng.normal(0.0, 1.0, 133) * plane_sigma
    xyz[:, 2] += rng.normal(0.0, 1.0, 133) * depth_sigma
    drop = rng.random_sample(133) < DROPOUT_RATE
    measured[drop] = False
    conf[drop] = 0.0
    spike = rng.random_sample(133) < SPIKE_RATE
    xyz[spike, 2] += SPIKE_METRES
    return xyz, measured, conf


def score(est, tru, dt):
    """(jitter mm, lag ms, rms mm) for one joint's track against its truth.

    JITTER is the frame-to-frame step ERROR - the estimate's own step minus the truth's step - so
    real motion is subtracted out and what remains is the shake a viewer sees. BOTH its sd and its
    median are returned, because they answer different questions: the sd is dominated by the rare
    1.5 m spikes (0.4% of frames at 1.5 m contributes ~134 mm of sd on its own), so it measures the
    worst behaviour, while the median measures the ordinary frame. A filter has to improve both.
    LAG is the whole-frame shift that best aligns the estimate to the truth. Reporting it next to
    jitter is the point of this bench: a filter can always buy one with the other.
    """
    est = np.asarray(est)
    tru = np.asarray(tru)
    d_est = np.diff(est, axis=0)
    d_tru = np.diff(tru, axis=0)
    step_err = np.linalg.norm(d_est - d_tru, axis=1)
    jitter = float(np.std(step_err)) * 1000.0
    jitter_med = float(np.median(step_err)) * 1000.0
    best_err = None
    best_k = 0
    for k in range(0, int(round(0.5 / dt)) + 1):
        if k == 0:
            err = float(np.mean(np.linalg.norm(est - tru, axis=1)))
        else:
            err = float(np.mean(np.linalg.norm(est[k:] - tru[:len(tru) - k], axis=1)))
        if best_err is None or err < best_err:
            best_err = err
            best_k = k
    return jitter, jitter_med, best_k * dt * 1000.0, best_err * 1000.0


def run(fps, seconds, cfg, seed, people=2):
    """One arm of the A/B. cfg=None means no filters at all (raw F-32 behaviour)."""
    dt = 1.0 / fps
    n = int(seconds * fps)
    pool = PF.PersonFilterPool(cfg) if cfg is not None else None
    est = dict((p, dict((j, []) for j in SCORED)) for p in range(people))
    tru = dict((p, dict((j, []) for j in SCORED)) for p in range(people))
    dropped = 0
    elapsed = 0.0
    for p in range(people):
        rng = np.random.RandomState(seed + p)
        t = 0.0
        for _ in range(n):
            t = t + dt
            xyz_true = truth(p, t)
            xyz, measured, conf = observe(xyz_true, rng)
            t0 = time.perf_counter()
            if pool is not None:
                bank = pool.acquire(p, t)
                bank.observe_rate(t)
                bank.smooth(xyz, measured)
                bank.mid_hip(xyz, measured)
                conf_emit, _res = bank.refine(xyz, measured, conf, t)
            else:
                conf_emit = conf
            elapsed += time.perf_counter() - t0
            for j in SCORED:
                if conf_emit[j] > 0.0 and bool(measured[j]):
                    est[p][j].append(xyz[j].copy())
                    tru[p][j].append(xyz_true[j].copy())
                else:
                    dropped += 1
    rows = []
    for p in range(people):
        for j in SCORED:
            if len(est[p][j]) > n // 2:
                rows.append((p, j) + score(est[p][j], tru[p][j], dt))
    arr = np.asarray([[r[2], r[3], r[4], r[5]] for r in rows])
    wr = np.asarray([[r[2], r[3], r[4], r[5]] for r in rows if r[1] in WRISTS])
    return {"jitter": float(np.median(arr[:, 0])), "jitter_med": float(np.median(arr[:, 1])),
            "lag": float(np.median(arr[:, 2])), "rms": float(np.median(arr[:, 3])),
            "wrist_jitter": float(np.median(wr[:, 0])) if len(wr) else float("nan"),
            "wrist_lag": float(np.median(wr[:, 2])) if len(wr) else float("nan"),
            "dropped_pct": 100.0 * dropped / float(n * people * len(SCORED)),
            "ms_per_person_frame": 1000.0 * elapsed / float(n * people)}


def main():
    ap = argparse.ArgumentParser(description="F-33 filter bench against known ground truth")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--rates", default="30,16,10",
                    help="loop rates to test. 30 is one person, 16 is three (the --max-poses "
                         "default at 20.7 ms a solve), 10 is a person being starved by the cap.")
    a = ap.parse_args()

    print("")
    print("F-33 filter bench - SYNTHETIC people, REAL filters. See the module docstring for what")
    print("this does and does not measure.")
    print("  %.0f s per arm, seed %d, depth noise ~%.0f mm at 2.2 m and ~%.0f mm at 3.4 m,"
          % (a.seconds, a.seed, 1000.0 * 2.2 * 2.2 / FX_BASELINE_SUBPIXEL,
             1000.0 * 3.4 * 3.4 / FX_BASELINE_SUBPIXEL))
    print("  %.0f%% of keypoint-frames lose depth, %.1f%% take a %.1f m spike."
          % (100.0 * DROPOUT_RATE, 100.0 * SPIKE_RATE, SPIKE_METRES))

    arms = (("no filters (raw F-32)", None),
            ("F-33, rate pinned at 30", PF.FilterConfig(adaptive_rate=False)),
            ("F-33, adaptive rate", PF.FilterConfig(adaptive_rate=True)))

    for fps in [float(v) for v in a.rates.split(",")]:
        print("")
        print("=== loop running at %.0f fps ===" % fps)
        print("  %-26s %11s %11s %9s %9s %10s %9s"
              % ("", "jitter sd", "jitter med", "lag", "rms", "wrist lag", "us/person"))
        for label, cfg in arms:
            r = run(fps, a.seconds, cfg, a.seed)
            print("  %-26s %8.1f mm %8.1f mm %6.0f ms %6.1f mm %7.0f ms %8.0f"
                  % (label, r["jitter"], r["jitter_med"], r["lag"], r["rms"], r["wrist_lag"],
                     r["ms_per_person_frame"] * 1000.0))
    print("")
    print("Read jitter and lag TOGETHER. A filter that lowers one by raising the other has not")
    print("improved anything; the claim is only meaningful where jitter falls and lag does not rise.")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
