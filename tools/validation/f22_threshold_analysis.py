#!/usr/bin/env python3

import evidence_paths as EV

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(_os.path.join(_d, "_sidecar_path.py")):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import _sidecar_path  # noqa: F401  - puts the sidecar root and every tools/ group on sys.path
"""F-22 threshold analysis - REAL VIDEO evidence and SYNTHETIC ADVERSARIAL evidence, kept apart.

The two populations answer different questions and must never be averaged into one accuracy number
(the brief is explicit about this, and it would be a genuinely misleading statistic):

    REAL VIDEO EVIDENCE          entirely legitimate human motion. Every rejection here is a
                                 candidate FALSE rejection. It can measure false-positive rate,
                                 warning/reject band occupancy, and how close ordinary motion comes
                                 to the shipped thresholds. It CANNOT measure true-positive rate,
                                 because these clips contain no known-bad pose.
    SYNTHETIC ADVERSARIAL        f22_adversarial.py's constructed violations, where the correct
                                 answer is known by construction. It measures true-positive
                                 behaviour and the reason taxonomy. It CANNOT measure false-positive
                                 rate, because none of it is real motion.

GEOMETRY, and why it differs from f22_video_replay.py: that harness placed keypoints at
(image_u, image_v, zrel*200) - a pixel-space proxy that mixes units, so its "degrees" are not real
degrees. This harness lifts every frame into metric camera space the same way production does -
pinhole backprojection with per-joint depth taken from the model's own root-relative z about a
nominal hip plane - so the distributions below are in real metres and real degrees and are directly
comparable to the shipped thresholds. Both sets of numbers are reported in the F-22 doc; this one
supersedes the other for threshold reasoning, and the reason is stated rather than the old table
being quietly replaced.

    python tools/validation/f22_threshold_analysis.py --videos a.webm,b.webm,c.webm
"""
import argparse
import io
import json
import os
import sys

import cv2
import numpy as np

import rtmw3d_pose as R
import pose_validation as PV
import target_ownership as TO

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "..", "..", "..", "SentisModel", "rtmw3d-x.onnx")
NOMINAL_HIP_Z = 2.0
CONF_THR = 0.3


def pct(v, p):
    return round(float(np.percentile(np.array(v), p)), 2) if v else None


def analyse_video(model, path, switch_margin_ratio=0.064):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit("could not open %s" % path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    diag = (w * w + h * h) ** 0.5
    fx = fy = 800.0
    intr = (fx, fy, w / 2.0, h / 2.0)

    own = TO.TargetOwnership(TO.OwnershipConfig(
        min_confidence=CONF_THR, switch_margin_m=diag * switch_margin_ratio,
        scale_margin_ratio=10.0))
    validator = PV.PoseValidator()
    bbox = R.center_bbox(w, h)
    dt = 1.0 / fps
    t = 0.0

    bends = dict((n, []) for n in PV.CHAIN_DEFS)
    rates = dict((n, []) for n in PV.CHAIN_DEFS)
    warn_band = dict((n, 0) for n in PV.CHAIN_DEFS)     # warn_deg < bend <= reject_deg
    reject_band = dict((n, 0) for n in PV.CHAIN_DEFS)   # bend > reject_deg  (absolute violations)
    rate_warn = dict((n, 0) for n in PV.CHAIN_DEFS)
    rate_reject = dict((n, 0) for n in PV.CHAIN_DEFS)
    suppressed = dict((n, 0) for n in PV.CHAIN_DEFS)
    reasons = {}
    holds = []            # (chain, start_frame, length_frames)
    open_hold = {}
    owned = 0
    frame_i = 0
    idx_to_chain = dict((j, n) for n, (_, j, _, _) in PV.CHAIN_DEFS.items())

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        uv, zrel, conf = model.infer(frame, bbox)
        refined = R.bbox_from_keypoints(uv, conf, w, h, thr=CONF_THR)
        body_conf_mean = float(np.mean(conf[0:17]))
        zrel_hip = float((zrel[11] + zrel[12]) / 2.0)

        xyz = np.zeros((17, 3), dtype=np.float64)
        for i in range(17):
            z = NOMINAL_HIP_Z + (float(zrel[i]) - zrel_hip)
            xyz[i] = [(float(uv[i, 0]) - intr[2]) * z / fx,
                      (float(uv[i, 1]) - intr[3]) * z / fy, z]
        measured = np.array([bool(conf[i] > CONF_THR) for i in range(17)])

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

        state, should_emit = own.update(
            TO.Observation(hip is not None, hip, body_conf_mean, scale), t)
        own.drain_events()
        if (refined is not None and body_conf_mean >= CONF_THR
                and state in (TO.ACQUIRING, TO.LOCKED, TO.REACQUIRING)):
            bbox = tuple(0.7 * np.array(bbox) + 0.3 * np.array(refined))
        elif state in (TO.NO_TARGET, TO.RELEASED):
            bbox = R.center_bbox(w, h)

        if should_emit:
            owned += 1
            out = validator.update(xyz, measured, conf[:17].copy(), t)
            for j, (st, reason, bend, rate) in out.items():
                name = idx_to_chain[j]
                cfg = PV.CHAIN_DEFS[name][3]
                if bend is not None:
                    bends[name].append(bend)
                    if bend > cfg.reject_deg:
                        reject_band[name] += 1
                    elif bend > cfg.warn_deg:
                        warn_band[name] += 1
                if rate is not None:
                    rates[name].append(rate)
                    if rate > cfg.rate_impossible_deg_s:
                        rate_reject[name] += 1
                    elif rate > cfg.rate_suspicious_deg_s:
                        rate_warn[name] += 1
                if st != PV.VALID:
                    suppressed[name] += 1
                    reasons[reason] = reasons.get(reason, 0) + 1
                    if name not in open_hold:
                        open_hold[name] = frame_i
                elif name in open_hold:
                    holds.append((name, open_hold[name], frame_i - open_hold[name]))
                    del open_hold[name]
            validator.drain_events()
        frame_i += 1
        t += dt
    cap.release()
    for name, start in open_hold.items():
        holds.append((name, start, frame_i - start))

    chains = {}
    for name in PV.CHAIN_DEFS:
        cfg = PV.CHAIN_DEFS[name][3]
        b, r = bends[name], rates[name]
        chains[name] = dict(
            n=len(b), warn_deg=cfg.warn_deg, reject_deg=cfg.reject_deg,
            bend_p50=pct(b, 50), bend_p95=pct(b, 95), bend_p99=pct(b, 99),
            bend_max=(round(max(b), 2) if b else None),
            headroom_to_reject_deg=(round(cfg.reject_deg - max(b), 2) if b else None),
            rate_p50=pct(r, 50), rate_p95=pct(r, 95), rate_p99=pct(r, 99),
            rate_max=(round(max(r), 1) if r else None),
            warn_band_frames=warn_band[name],
            warn_band_pct=round(100.0 * warn_band[name] / max(1, len(b)), 3),
            reject_band_frames=reject_band[name],
            reject_band_pct=round(100.0 * reject_band[name] / max(1, len(b)), 3),
            rate_warn_frames=rate_warn[name], rate_reject_frames=rate_reject[name],
            suppressed_frames=suppressed[name],
            suppressed_pct=round(100.0 * suppressed[name] / max(1, owned), 3),
        )
    hold_lengths = [n for _, _, n in holds]
    return dict(
        video=os.path.basename(path), width=w, height=h, fps=round(fps, 2),
        frames=frame_i, owned_frames=owned, chains=chains, reasons=reasons,
        hold_episodes=len(holds),
        hold_len_frames=dict(p50=pct(hold_lengths, 50), p95=pct(hold_lengths, 95),
                             max=(max(hold_lengths) if hold_lengths else 0)),
        hold_len_ms=dict(p50=(round(pct(hold_lengths, 50) * 1000.0 / fps, 1)
                              if hold_lengths else None),
                         max=(round(max(hold_lengths) * 1000.0 / fps, 1) if hold_lengths else 0)),
        recovery_frames_after_hold=1,   # by construction: VALID on the first passing frame (SS10)
    )


def main():
    ap = argparse.ArgumentParser()
    vdir = os.path.join("D:", os.sep, "Unity", "viitorx-vrm-avtar-unity-base-project", "Assets",
                        "Games", "video")
    ap.add_argument("--videos", default=",".join(os.path.join(vdir, v) for v in
                                                 ("video.webm", "123.webm", "456.webm")))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adversarial-summary",
                    default=EV.oak_v4("f22", "adversarial", "summary.json"))
    ap.add_argument("--out-dir", default=EV.oak_v4("f22", "thresholds"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    print("=" * 104)
    print(" F-22 threshold analysis")
    print("=" * 104)
    print("[load] RTMW3D ...")
    model = R.RTMW3D(a.model)

    vids = []
    for p in a.videos.split(","):
        print("\n[real] %s" % p)
        r = analyse_video(model, p)
        vids.append(r)
        print("   frames=%d owned=%d" % (r["frames"], r["owned_frames"]))

    print("\n" + "=" * 104)
    print(" REAL VIDEO EVIDENCE - legitimate motion only. Every rejection here is a candidate")
    print(" FALSE rejection. This population cannot measure true-positive rate.")
    print("=" * 104)
    print("  %-11s %-13s %7s %7s %7s %8s %9s %9s %9s %9s"
          % ("video", "chain", "p50", "p95", "max", "headrm", "warn%", "reject%", "rate_max", "suppr"))
    totals = dict(owned=0, warn=0, reject=0, suppressed=0, samples=0)
    for r in vids:
        for name, c in r["chains"].items():
            print("  %-11s %-13s %7s %7s %7s %8s %9s %9s %9s %9s"
                  % (r["video"], name, c["bend_p50"], c["bend_p95"], c["bend_max"],
                     c["headroom_to_reject_deg"], c["warn_band_pct"], c["reject_band_pct"],
                     c["rate_max"], c["suppressed_frames"]))
            totals["warn"] += c["warn_band_frames"]
            totals["reject"] += c["reject_band_frames"]
            totals["suppressed"] += c["suppressed_frames"]
            totals["samples"] += c["n"]
        totals["owned"] += r["owned_frames"]
    print("-" * 104)
    print("  TOTAL   owned_frames=%d  chain_samples=%d  warn_band=%d (%.3f%%)  "
          "ABSOLUTE reject_band=%d (%.3f%%)  suppressed=%d (%.3f%%)"
          % (totals["owned"], totals["samples"], totals["warn"],
             100.0 * totals["warn"] / max(1, totals["samples"]), totals["reject"],
             100.0 * totals["reject"] / max(1, totals["samples"]), totals["suppressed"],
             100.0 * totals["suppressed"] / max(1, totals["samples"])))
    for r in vids:
        print("  %-11s hold episodes=%d  len p50=%s p95=%s max=%s frames (max %s ms)  reasons=%s"
              % (r["video"], r["hold_episodes"], r["hold_len_frames"]["p50"],
                 r["hold_len_frames"]["p95"], r["hold_len_frames"]["max"],
                 r["hold_len_ms"]["max"], r["reasons"]))

    adv = None
    if os.path.exists(a.adversarial_summary):
        adv = json.load(io.open(a.adversarial_summary, encoding="utf-8"))
        print("\n" + "=" * 104)
        print(" SYNTHETIC ADVERSARIAL EVIDENCE - constructed violations with a known correct answer.")
        print(" This population measures TRUE-positive behaviour. It cannot measure false-positive")
        print(" rate, because none of it is real human motion. NOT combined with the table above.")
        print("=" * 104)
        cases = adv["cases"]
        expected_suppressed = [k for k, v in cases.items() if v["state"] != "VALID"]
        expected_valid = [k for k, v in cases.items() if v["state"] == "VALID"]
        print("  constructed cases            : %d" % len(cases))
        print("  correctly SUPPRESSED         : %d  %s" % (len(expected_suppressed),
                                                           sorted(expected_suppressed)))
        print("  correctly PASSED (by design) : %d  %s" % (len(expected_valid),
                                                           sorted(expected_valid)))
        rc = {}
        for v in cases.values():
            if v["reason"]:
                rc[v["reason"]] = rc.get(v["reason"], 0) + 1
        print("  rejection reason taxonomy    : %s" % rc)
        print("  assertions                   : %d/%d" % (adv["assertions_passed"],
                                                          adv["assertions_total"]))

    print("\n" + "=" * 104)
    print(" WHAT THE TWO POPULATIONS JOINTLY SUPPORT")
    print("=" * 104)
    print("  - false rejection  : measurable ONLY from the real-video table above")
    print("  - true rejection   : measurable ONLY from the synthetic table above")
    print("  - a single combined 'accuracy' figure is NOT computed, on purpose")
    print("=" * 104)

    out = dict(real_video_evidence=vids, real_video_totals=totals,
               synthetic_adversarial_evidence=(adv["cases"] if adv else None),
               synthetic_assertions=((adv["assertions_passed"], adv["assertions_total"])
                                     if adv else None),
               thresholds=dict(
                   elbow=dict(warn=PV.ELBOW_CONFIG.warn_deg, reject=PV.ELBOW_CONFIG.reject_deg,
                              rate_warn=PV.ELBOW_CONFIG.rate_suspicious_deg_s,
                              rate_reject=PV.ELBOW_CONFIG.rate_impossible_deg_s),
                   knee=dict(warn=PV.KNEE_CONFIG.warn_deg, reject=PV.KNEE_CONFIG.reject_deg,
                             rate_warn=PV.KNEE_CONFIG.rate_suspicious_deg_s,
                             rate_reject=PV.KNEE_CONFIG.rate_impossible_deg_s)))
    with io.open(os.path.join(a.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(" evidence -> %s" % a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
