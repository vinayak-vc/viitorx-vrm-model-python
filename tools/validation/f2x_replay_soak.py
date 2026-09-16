#!/usr/bin/env python3
"""OFFLINE REPLAY SOAK - F-21 + F-22 running together over a long looped replay.

LABEL, up front and not negotiable: this is an OFFLINE REPLAY SOAK. It is NOT equivalent to a
hardware soak and must never be quoted as one. What it genuinely exercises is every pure-software
stage - the M15 crop loop, F-21 ownership, F-22 validation, and the real build_body_landmarks - over
tens of thousands of consecutive frames, which is enough to expose unbounded growth, accumulating
state, leaked events and exception paths. What it CANNOT exercise is the OAK-D itself: no USB, no
DepthAI pipeline, no stereo depth, no thermal drift, no real socket send. Every failure mode that
lives in those is untouched here and stays a live-hardware question.

The loop deliberately cycles through ALL available clips rather than repeating one. Each clip change
is a total change of person and scene, so the ownership machine is driven through its whole
lifecycle - release, re-acquire, new epoch - repeatedly instead of sitting in LOCKED for the entire
run, which is what makes accumulating state visible.

Measured: exceptions, resident-memory growth, state-machine oscillation, ownership changes,
validator suppression rate, recovery behaviour, and processing throughput.

    python tools/validation/f2x_replay_soak.py --minutes 15
"""

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(_os.path.join(_d, "_sidecar_path.py")):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import _sidecar_path  # noqa: F401  - puts the sidecar root and every tools/ group on sys.path
import evidence_paths as EV

import argparse
import ctypes
import ctypes.wintypes
import gc
import io
import json
import os
import sys
import time

import cv2
import numpy as np

import rtmw3d_pose as R
import pose_validation as PV
import target_ownership as TO
import wholebody_udp_sender as W

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = EV.DEFAULT_MODEL
NOMINAL_HIP_Z = 2.0
CONF_THR = 0.3
FLATTEN_TRUNK = False
USE_ZREL = True


class _PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.wintypes.DWORD),
                ("PageFaultCount", ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


# The signature MUST be declared. The obvious spelling - ctypes.windll.psapi.GetProcessMemoryInfo
# with no argtypes - silently FAILS on 64-bit Windows: ctypes narrows the process HANDLE to a C int,
# the call returns 0, and the struct is left zeroed. It does not raise, so an undeclared version
# reports a confident, wrong "0.0 MB" and a soak would announce "no memory growth" having measured
# nothing at all. Caught exactly that way on the first 16-minute run.
_K32 = ctypes.WinDLL("kernel32", use_last_error=True)
_K32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
_K32.K32GetProcessMemoryInfo.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(_PMC),
                                         ctypes.wintypes.DWORD]
_K32.K32GetProcessMemoryInfo.restype = ctypes.wintypes.BOOL


def rss_mb():
    """Working set in MB. psutil is not installed in this venv, so this reads the same counter
    straight from the Win32 API rather than adding a dependency for one number. Raises rather than
    returning a plausible zero if the call fails - a silently-zero memory metric is worse than none."""
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(_PMC)
    if not _K32.K32GetProcessMemoryInfo(_K32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        raise OSError("K32GetProcessMemoryInfo failed: %d" % ctypes.get_last_error())
    return pmc.WorkingSetSize / (1024.0 * 1024.0)


def main():
    ap = argparse.ArgumentParser()
    vdir = os.path.join("D:", os.sep, "Unity", "viitorx-vrm-avtar-unity-base-project", "Assets",
                        "Games", "video")
    ap.add_argument("--videos", default=",".join(os.path.join(vdir, v) for v in
                                                 ("video.webm", "123.webm", "456.webm")))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--minutes", type=float, default=15.0)
    # The default (one ownership instance carried across every clip) soaks CONTINUITY, but it
    # also means each clip starts holding an owner position from a DIFFERENT clip in a
    # different pixel space, so no clip ever replays its own scenario and the S30 path gate is
    # never reached. Resetting makes every cycle a faithful repeat of that clip, which is what
    # soaking the GATE needs. Both are run and reported separately - see F-21 report S30.13.
    ap.add_argument("--reset-per-clip", action="store_true",
                     help="rebuild the ownership machine at the start of every clip")
    ap.add_argument("--out-dir", default=EV.oak_v4("soak"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    videos = a.videos.split(",")

    print("=" * 100)
    print(" OFFLINE REPLAY SOAK - NOT a hardware soak (see module docstring)")
    print(" target %.1f min over %d clips, looping" % (a.minutes, len(videos)))
    print("=" * 100)
    model = R.RTMW3D(a.model)

    own = TO.TargetOwnership(TO.OwnershipConfig(min_confidence=CONF_THR,
                                                switch_margin_m=0.35, scale_margin_ratio=0.45))
    validator = PV.PoseValidator()
    fx = fy = 800.0

    deadline = time.time() + a.minutes * 60.0
    t_sim = 0.0
    total_frames = 0
    owned_frames = 0
    exceptions = []
    state_counts = {}
    ownership_events = {}
    suppress_counts = {}
    hold_episodes = 0
    reject_reasons = {}
    open_holds = set()
    recoveries = 0
    samples = []
    cycle = 0
    gc.collect()
    rss0 = rss_mb()
    t0 = time.perf_counter()
    idx_to_chain = dict((j, n) for n, (_, j, _, _) in PV.CHAIN_DEFS.items())
    print(" start RSS = %.1f MB" % rss0)

    while time.time() < deadline:
        cycle += 1
        for path in videos:
            if time.time() >= deadline:
                break
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                exceptions.append("cycle %d: could not open %s" % (cycle, path))
                continue
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            # UNIT BUG, found 2026-09-15 and fixed here: this used to be
            #     own.cfg.switch_margin_m = diag * 0.064
            # a PIXEL figure, while every position fed to ownership below is lifted into METRES by
            # the pinhole backprojection (xyz[i] = (u-cx)*z/fx, ...). The margin was therefore ~141
            # METRES and nothing could ever fail _matches_owner - which is why every soak to date
            # reported "candidate rejection reasons: none", 0 releases, and 0 S30 path rejections
            # while replaying a clip that demonstrably contains a person swap. The soak was
            # exercising F-21's PLUMBING and never its DISCRIMINATION, and the clean numbers looked
            # like evidence of correctness. Use the real production value: the lift is metric, so
            # the production constant applies directly, with no conversion to get wrong.
            own.cfg.switch_margin_m = 0.35             # production SWITCH_MARGIN_M, metres
            own.cfg.scale_margin_ratio = 0.45          # production value; was 10.0 (= disabled)
            if a.reset_per_clip:
                _cfg = own.cfg
                own = TO.TargetOwnership(_cfg)         # same config, fresh state machine
            cx, cy = w / 2.0, h / 2.0
            bbox = R.center_bbox(w, h)
            dt = 1.0 / fps
            while time.time() < deadline:
                ok, frame = cap.read()
                if not ok:
                    break
                try:
                    uv, zrel, conf = model.infer(frame, bbox)
                    ref = R.bbox_from_keypoints(uv, conf, w, h, thr=CONF_THR)
                    bcm = float(np.mean(conf[0:17]))
                    zrel_hip = float((zrel[11] + zrel[12]) / 2.0)
                    xyz = np.zeros((133, 3), dtype=np.float64)
                    for i in range(133):
                        z = NOMINAL_HIP_Z + (float(zrel[i]) - zrel_hip)
                        xyz[i] = [(float(uv[i, 0]) - cx) * z / fx,
                                  (float(uv[i, 1]) - cy) * z / fy, z]
                    measured = np.array([bool(conf[i] > CONF_THR) for i in range(133)])
                    hip = None
                    scale = None
                    if measured[11] and measured[12]:
                        hip = tuple(((xyz[11] + xyz[12]) / 2.0).tolist())
                    elif measured[11]:
                        hip = tuple(xyz[11].tolist())
                    elif measured[12]:
                        hip = tuple(xyz[12].tolist())
                    if hip is not None and measured[5] and measured[6]:
                        scale = float(np.linalg.norm((xyz[5] + xyz[6]) / 2.0 - np.array(hip)))

                    state, emit = own.update(
                        TO.Observation(hip is not None, hip, bcm, scale), t_sim)
                    state_counts[state] = state_counts.get(state, 0) + 1
                    for e in own.drain_events():
                        ownership_events[e["event"]] = ownership_events.get(e["event"], 0) + 1
                        if e["event"] == "TARGET_REJECTED_CANDIDATE":
                            r = e.get("reason", "?")
                            reject_reasons[r] = reject_reasons.get(r, 0) + 1
                    if (ref is not None and bcm >= CONF_THR
                            and state in (TO.ACQUIRING, TO.LOCKED, TO.REACQUIRING)):
                        bbox = tuple(0.7 * np.array(bbox) + 0.3 * np.array(ref))
                    elif state in (TO.NO_TARGET, TO.RELEASED):
                        bbox = R.center_bbox(w, h)

                    if emit and hip is not None:
                        owned_frames += 1
                        conf_emit = np.array(conf, dtype=np.float64)
                        pv_out = validator.update(xyz, measured, conf_emit, t_sim)
                        for j, (st, reason, _b, _r) in pv_out.items():
                            name = idx_to_chain[j]
                            if st != PV.VALID:
                                conf_emit[j] = 0.0
                                suppress_counts[reason] = suppress_counts.get(reason, 0) + 1
                                if name not in open_holds:
                                    open_holds.add(name)
                                    hold_episodes += 1
                            elif name in open_holds:
                                open_holds.discard(name)
                                recoveries += 1
                        validator.drain_events()
                        mid_hip = (xyz[11] + xyz[12]) / 2.0
                        W.build_body_landmarks(uv, xyz, measured, conf_emit, zrel, zrel_hip,
                                               mid_hip, float(mid_hip[2]),
                                               (fx, fy, cx, cy), CONF_THR, FLATTEN_TRUNK, USE_ZREL)
                except Exception as ex:                      # noqa: BLE001 - a soak must not stop
                    exceptions.append("frame %d (%s): %s: %s"
                                      % (total_frames, os.path.basename(path),
                                         type(ex).__name__, ex))
                total_frames += 1
                t_sim += dt
                if total_frames % 1000 == 0:
                    el = time.perf_counter() - t0
                    samples.append(dict(frames=total_frames, minutes=round(el / 60.0, 2),
                                        rss_mb=round(rss_mb(), 1),
                                        fps=round(total_frames / el, 2),
                                        owned=owned_frames, exceptions=len(exceptions),
                                        ownership_epochs=own.epoch))
                    s = samples[-1]
                    print("  %7d frames  %5.1f min  RSS %7.1f MB  %5.1f fps  epochs=%d  exc=%d"
                          % (s["frames"], s["minutes"], s["rss_mb"], s["fps"],
                             s["ownership_epochs"], s["exceptions"]))
            cap.release()

    el = time.perf_counter() - t0
    gc.collect()
    rss1 = rss_mb()
    peak = max([s["rss_mb"] for s in samples] + [rss1])
    # growth measured from the FIRST sample, not from process start: the first 1000 frames include
    # ONNX session warm-up and arena allocation, which is one-time and would masquerade as a leak.
    base = samples[0]["rss_mb"] if samples else rss0
    growth = rss1 - base

    osc = ownership_events.get("TARGET_TEMP_LOST", 0)
    print("\n" + "=" * 100)
    print(" OFFLINE REPLAY SOAK RESULT")
    print("=" * 100)
    print("  duration                      : %.1f min (%d clip cycles)" % (el / 60.0, cycle))
    print("  frames processed              : %d" % total_frames)
    print("  owned frames (F-21 emitted)   : %d  (%.1f%%)"
          % (owned_frames, 100.0 * owned_frames / max(1, total_frames)))
    print("  throughput                    : %.2f fps" % (total_frames / el))
    print("  EXCEPTIONS                    : %d" % len(exceptions))
    for e in exceptions[:5]:
        print("      %s" % e)
    print("  RSS start / after warm-up     : %.1f / %.1f MB" % (rss0, base))
    print("  RSS end / peak                : %.1f / %.1f MB" % (rss1, peak))
    print("  RSS growth after warm-up      : %+.1f MB over %d frames (%+.4f MB per 1000 frames)"
          % (growth, total_frames, growth / max(1.0, total_frames / 1000.0)))
    print("  ownership state distribution  : %s" % state_counts)
    print("  ownership events              : %s" % ownership_events)
    print("  candidate rejection reasons   : %s" % (reject_reasons or "none"))
    print("  S30 path-gate rejections      : %d" % reject_reasons.get("path_walked_in", 0))
    print("  WRONG-PERSON FRAMES           : not measurable here, and not reported as zero.")
    print("                                  The soak loops three clips and only 123.webm has")
    print("                                  ground-truth identity labels. That number lives in")
    print("                                  f21_wrongperson_replay.py; quoting a soak figure for")
    print("                                  it would be inventing evidence. What the soak IS for")
    print("                                  is stability: exceptions, memory, oscillation, and")
    print("                                  whether the new gate ever wedges the machine.")
    print("  ownership epochs (changes)    : %d" % own.epoch)
    print("  state-machine oscillation     : %d TEMP_LOST episodes over %.1f min" % (osc, el / 60.0))
    print("  validator suppression reasons : %s" % suppress_counts)
    print("  suppression rate              : %.3f%% of chain evaluations"
          % (100.0 * sum(suppress_counts.values()) / max(1, owned_frames * 4)))
    print("  hold episodes / recoveries    : %d / %d" % (hold_episodes, recoveries))
    print("  unrecovered holds at end      : %s" % (sorted(open_holds) or "none"))
    print("=" * 100)
    print(" LABEL: OFFLINE REPLAY SOAK - no OAK-D, no USB, no stereo depth, no real socket send.")
    print("        Not equivalent to a hardware soak and not to be quoted as one.")
    print("=" * 100)

    with io.open(os.path.join(a.out_dir, "soak_summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(label="OFFLINE REPLAY SOAK", videos=videos,
                       minutes=round(el / 60.0, 2), cycles=cycle, frames=total_frames,
                       owned_frames=owned_frames, fps=round(total_frames / el, 2),
                       exceptions=exceptions, rss_start_mb=round(rss0, 1),
                       rss_after_warmup_mb=round(base, 1), rss_end_mb=round(rss1, 1),
                       rss_peak_mb=round(peak, 1), rss_growth_mb=round(growth, 1),
                       state_counts=state_counts, ownership_events=ownership_events,
                       rejection_reasons=reject_reasons,
                       path_gate_rejections=reject_reasons.get("path_walked_in", 0),
                       wrong_person_frames="NOT MEASURABLE IN SOAK - see "
                                           "f21_wrongperson_replay.py (only 123.webm is labelled)",
                       ownership_epochs=own.epoch, suppression=suppress_counts,
                       hold_episodes=hold_episodes, recoveries=recoveries,
                       unrecovered_holds=sorted(open_holds), samples=samples), f, indent=2)
    print(" evidence -> %s" % a.out_dir)
    return 0 if not exceptions else 1


if __name__ == "__main__":
    sys.exit(main())
