#!/usr/bin/env python3
"""
Offline dancing-video P0 stress test (RGB replay — NO depth, NO Unity).

Runs the REAL RTMW3D pose model (rtmw3d_pose.RTMW3D, unchanged) + the REAL person-box tracking on every
frame of a dancing video, then exercises the ACTUAL P0 code:
  * P0-2 smoothing.KeypointSmoother in BOTH pre-P0 and P0 configs (before/after limb caps + leg protection),
  * P0-1 LimbGate decision logic on the real per-frame confidences.

IMPORTANT LIMITATION — this is RGB video with NO OAK-D depth and NO camera intrinsics, so there is no
measured metric XYZ. x/y are model pixels; z is the model's root-relative metric depth (zrel). To exercise
the metric 0.35 m caps we build an ESTIMATED hip-relative metric signal: x/y pixels are scaled by a global
torso-length normalization (median |midShoulder-midHip| px -> 0.5 m nominal), z = zrel (already metres). This
is clearly an ESTIMATE; it validates the P0 LOGIC on real dance motion, NOT depth quality.

Outputs (pipeline_logs/):
  replay_metrics.json   — all numeric results
  replay_raw.npz        — per-frame uv/zrel/conf (reproducible)
And an annotated MP4 (skeleton + confidence + P0 state) next to the video.

Run:  .venv\Scripts\python replay_video.py --video <path>
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rtmw3d_pose as R
import smoothing

WB = {5: "L-shoulder", 6: "R-shoulder", 7: "L-elbow", 8: "R-elbow", 9: "L-wrist", 10: "R-wrist",
      11: "L-hip", 12: "R-hip", 13: "L-knee", 14: "R-knee", 15: "L-ankle", 16: "R-ankle"}
JOINTS = list(WB.keys())
ARMS = [7, 8, 9, 10]
LEGS = [13, 14, 15, 16]
NOMINAL_TORSO_M = 0.5
CONF_THR = 0.3

# LimbGate driving joints (WB indices), mirroring the C# cross-map (KalidokitControlRigDriver):
#   leftArm bones <- {6,8,10}  rightArm <- {5,7,9}  leftLeg <- {12,14,16}  rightLeg <- {11,13,15}
LIMB_DRIVERS = {"leftArm": (6, 8, 10), "rightArm": (5, 7, 9), "leftLeg": (12, 14, 16), "rightLeg": (11, 13, 15)}


def percentiles(vals, ps):
    if len(vals) == 0:
        return {p: 0.0 for p in ps}
    a = np.sort(np.asarray(vals, dtype=np.float64))
    return {p: float(a[min(len(a) - 1, int(len(a) * p))]) for p in ps}


# ----------------------------------------------------------------- inference
def run_inference(video, cache):
    if os.path.exists(cache):
        d = np.load(cache)
        return d["uv"], d["zrel"], d["conf"], int(d["w"]), int(d["h"]), float(d["fps"])
    model = R.RTMW3D(os.environ.get("RTMW3D_MODEL",
                     r"C:/Unity/viitorx-vrm-avtar-unity-base-project/Assets/SentisModel/rtmw3d-x.onnx"))
    print("[replay] providers:", model.active_providers)
    cap = cv2.VideoCapture(video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    bbox = R.center_bbox(w, h)
    lowconf = 0
    uv_all, z_all, c_all = [], [], []
    t0 = time.time()
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        uv, zrel, conf = model.infer(frame, bbox)
        # person-box tracking WITH recovery — identical policy to wholebody_udp_sender.
        refined = R.bbox_from_keypoints(uv, conf, w, h, thr=CONF_THR)
        body_conf_mean = float(np.mean(conf[0:17]))
        if refined is not None and body_conf_mean >= CONF_THR:
            lowconf = 0
            bbox = tuple(0.7 * np.array(bbox) + 0.3 * np.array(refined))
        else:
            lowconf += 1
            if refined is None or lowconf >= 20:
                bbox = R.center_bbox(w, h)
                lowconf = 0
            else:
                bbox = tuple(0.7 * np.array(bbox) + 0.3 * np.array(refined))
        uv_all.append(uv.copy())
        z_all.append(zrel.copy())
        c_all.append(conf.copy())
        i += 1
        if i % 60 == 0:
            print("[replay] %d frames  (%.1f fps infer)" % (i, i / (time.time() - t0 + 1e-9)))
    cap.release()
    uv = np.stack(uv_all)
    zrel = np.stack(z_all)
    conf = np.stack(c_all)
    np.savez_compressed(cache, uv=uv, zrel=zrel, conf=conf, w=w, h=h, fps=fps)
    print("[replay] inference done: %d frames in %.1fs" % (i, time.time() - t0))
    return uv, zrel, conf, w, h, fps


# ----------------------------------------------------------------- metric estimate
def build_metric(uv, zrel, conf):
    """ESTIMATED hip-relative metric xyz[F,133,3]. x/y px -> m by global torso scale; z = zrel - zrel_hip."""
    F = uv.shape[0]
    # global scale from confident torso frames
    torso_px = []
    for f in range(F):
        if conf[f, 5] >= CONF_THR and conf[f, 6] >= CONF_THR and conf[f, 11] >= CONF_THR and conf[f, 12] >= CONF_THR:
            msh = 0.5 * (uv[f, 5] + uv[f, 6])
            mhip = 0.5 * (uv[f, 11] + uv[f, 12])
            torso_px.append(float(np.linalg.norm(msh - mhip)))
    med_torso = float(np.median(torso_px)) if torso_px else 200.0
    scale = NOMINAL_TORSO_M / max(1.0, med_torso)
    xyz = np.zeros((F, 133, 3), dtype=np.float32)
    last_hip_px = None
    for f in range(F):
        if conf[f, 11] >= CONF_THR and conf[f, 12] >= CONF_THR:
            hip_px = 0.5 * (uv[f, 11] + uv[f, 12])
            last_hip_px = hip_px
        elif last_hip_px is not None:
            hip_px = last_hip_px
        else:
            hip_px = np.array([uv[f, :, 0].mean(), uv[f, :, 1].mean()])
        zhip = 0.5 * (zrel[f, 11] + zrel[f, 12])
        xyz[f, :, 0] = (uv[f, :, 0] - hip_px[0]) * scale
        xyz[f, :, 1] = (uv[f, :, 1] - hip_px[1]) * scale
        xyz[f, :, 2] = zrel[f, :] - zhip
    return xyz, scale, med_torso


# ----------------------------------------------------------------- smoother pass
def smooth_pass(xyz, conf, cfg):
    """Run the REAL KeypointSmoother across all frames. cfg dict -> KeypointSmoother kwargs (+ limb_indices).
    Returns smoothed xyz + per-frame per-joint action code."""
    F = xyz.shape[0]
    ks = smoothing.KeypointSmoother(133, min_cutoff=cfg["min_cutoff"], beta=cfg["beta"], max_jump=cfg["max_jump"],
                                    depth_min_cutoff=cfg["depth_min_cutoff"], depth_beta=cfg["depth_beta"],
                                    limb_indices=cfg["limb_indices"], max_hold=cfg["max_hold"],
                                    max_jump_overrides=cfg.get("overrides"))
    out = np.zeros_like(xyz)
    actions = {}
    for j in JOINTS:
        actions[j] = []
    for f in range(F):
        for j in JOINTS:
            valid = bool(conf[f, j] >= CONF_THR)  # NO depth -> confidence is the validity proxy (stated)
            sx, sy, sz, eff, act, disp = ks.filter(j, float(xyz[f, j, 0]), float(xyz[f, j, 1]), float(xyz[f, j, 2]), valid)
            out[f, j] = (sx, sy, sz)
            actions[j].append(act)
    return out, actions


def disp_series(xyz, joints, valid_mask=None):
    """Frame-to-frame displacement per joint (metres). Skips a step if either endpoint is invalid."""
    F = xyz.shape[0]
    res = {}
    for j in joints:
        ds = []
        for f in range(1, F):
            if valid_mask is not None and (not valid_mask[f, j] or not valid_mask[f - 1, j]):
                continue
            ds.append(float(np.linalg.norm(xyz[f, j] - xyz[f - 1, j])))
        res[j] = ds
    return res


# ----------------------------------------------------------------- spike classification
def classify_spikes(xyz, conf, joints, big_thr=0.20):
    """Isolated spike vs legitimate fast motion via a +/-2 window on the RAW estimated positions."""
    F = xyz.shape[0]
    out = {}
    for j in joints:
        big = 0
        iso = 0
        legit = 0
        for f in range(2, F - 2):
            if conf[f, j] < CONF_THR or conf[f - 1, j] < CONF_THR:
                continue
            d = float(np.linalg.norm(xyz[f, j] - xyz[f - 1, j]))
            if d < big_thr:
                continue
            big += 1
            # isolated: jumps at f, returns near f-1 within 2 frames, and neighbours are small steps
            back = float(np.linalg.norm(xyz[f + 1, j] - xyz[f - 1, j]))
            fwd = float(np.linalg.norm(xyz[f + 1, j] - xyz[f, j]))
            if back < big_thr and fwd >= big_thr:
                iso += 1
            else:
                legit += 1
        out[j] = {"big": big, "isolated": iso, "legit": legit,
                  "spike_pct": (100.0 * iso / big) if big else 0.0}
    return out


# ----------------------------------------------------------------- dropout / occlusion
def dropout_runs(conf, actions, joints):
    """valid->invalid->valid runs per joint, with what the P0 smoother did (HOLD/DROP counts)."""
    F = conf.shape[0]
    out = {}
    for j in joints:
        runs = []
        f = 0
        max_hold = 0
        max_invalid = 0
        holds = sum(1 for a in actions[j] if a == "HOLD")
        drops = sum(1 for a in actions[j] if a == "DROP")
        rlim = sum(1 for a in actions[j] if a == "RATE_LIMIT")
        while f < F:
            if conf[f, j] < CONF_THR:
                start = f
                while f < F and conf[f, j] < CONF_THR:
                    f += 1
                dur = f - start
                max_invalid = max(max_invalid, dur)
                if start > 0 and f < F:
                    runs.append((start, f - 1, dur))
            else:
                f += 1
        # max continuous HOLD action
        cur = 0
        for a in actions[j]:
            if a == "HOLD":
                cur += 1
                max_hold = max(max_hold, cur)
            else:
                cur = 0
        out[j] = {"dropout_runs": len(runs), "max_invalid_frames": max_invalid,
                  "holds": holds, "drops": drops, "rate_limited": rlim, "max_continuous_hold": max_hold,
                  "examples": runs[:5]}
    return out


# ----------------------------------------------------------------- P0-1 gate sim
class LimbGateSim:
    def __init__(self):
        self.lastU = None; self.hasValid = False; self.state = "Valid"
        self.held = 0; self.holdEvents = 0; self.reacq = 0; self.confFail = 0; self.maxHold = 0

    def resolve(self, conf, thr):
        if conf >= thr:
            if self.state == "Held" and self.hasValid:
                self.reacq += 1
            self.hasValid = True; self.held = 0; self.state = "Valid"; return "APPLY"
        self.confFail += 1
        if not self.hasValid:
            return "SKIP"
        if self.state != "Held":
            self.holdEvents += 1
        self.state = "Held"; self.held += 1; self.maxHold = max(self.maxHold, self.held); return "HOLD"


def gate_analysis(conf):
    F = conf.shape[0]
    res = {}
    for name, (a, b, c) in LIMB_DRIVERS.items():
        g = LimbGateSim()
        applied = held = skip = 0
        for f in range(F):
            lc = min(conf[f, a], conf[f, b], conf[f, c])
            act = g.resolve(float(lc), CONF_THR)
            if act == "APPLY":
                applied += 1
            elif act == "HOLD":
                held += 1
            else:
                skip += 1
        res[name] = {"applied": applied, "held": held, "skip_startup": skip,
                     "holdEvents": g.holdEvents, "reacquisitions": g.reacq,
                     "max_continuous_hold": g.maxHold, "conf_failures": g.confFail,
                     "held_pct": 100.0 * held / F, "valid_pct": 100.0 * applied / F,
                     "destructive_zero_events": 0}  # by construction: gate never emits zero
    return res


# ----------------------------------------------------------------- rotation proxy
def rotation_proxy(xyz, conf):
    segs = {"upperArmR": (5, 7), "forearmR": (7, 9), "upperArmL": (6, 8), "forearmL": (8, 10),
            "upperLegR": (11, 13), "lowerLegR": (13, 15), "upperLegL": (12, 14), "lowerLegL": (14, 16)}
    F = xyz.shape[0]
    out = {}
    for name, (a, b) in segs.items():
        deltas = []
        e45 = e90 = e135 = e180 = 0
        prev = None
        for f in range(F):
            if conf[f, a] < CONF_THR or conf[f, b] < CONF_THR:
                prev = None
                continue
            v = xyz[f, b] - xyz[f, a]
            n = np.linalg.norm(v)
            if n < 1e-6:
                prev = None
                continue
            v = v / n
            if prev is not None:
                d = float(np.degrees(np.arccos(np.clip(np.dot(v, prev), -1, 1))))
                deltas.append(d)
                if d > 45: e45 += 1
                if d > 90: e90 += 1
                if d > 135: e135 += 1
                if d > 170: e180 += 1
            prev = v
        p = percentiles(deltas, [0.5, 0.95, 0.99])
        out[name] = {"median": round(p[0.5], 1), "p95": round(p[0.95], 1), "p99": round(p[0.99], 1),
                     "max": round(max(deltas), 1) if deltas else 0.0,
                     ">45": e45, ">90": e90, ">135": e135, "~180": e180}
    return out


# ----------------------------------------------------------------- annotated video
def annotate(video, uv, conf, actions_p0, out_path, fps):
    cap = cv2.VideoCapture(video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    colors = {"ACCEPT": (0, 220, 0), "RATE_LIMIT": (0, 200, 255), "HOLD": (0, 165, 255), "DROP": (0, 0, 255)}
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok or f >= uv.shape[0]:
            break
        for (a, b) in R.JOINTID_EDGES:
            pass  # edges are JointId space; we draw WB body edges below instead
        edges = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
                 (11, 13), (13, 15), (12, 14), (14, 16)]
        for a, b in edges:
            if conf[f, a] >= CONF_THR and conf[f, b] >= CONF_THR:
                pa = (int(uv[f, a, 0]), int(uv[f, a, 1]))
                pb = (int(uv[f, b, 0]), int(uv[f, b, 1]))
                cv2.line(frame, pa, pb, (200, 200, 200), 2)
        for j in JOINTS:
            act = actions_p0[j][f] if f < len(actions_p0[j]) else "ACCEPT"
            col = colors.get(act, (0, 220, 0)) if conf[f, j] >= CONF_THR else (0, 0, 255)
            p = (int(uv[f, j, 0]), int(uv[f, j, 1]))
            cv2.circle(frame, p, 5, col, -1)
            if act in ("RATE_LIMIT", "HOLD") or conf[f, j] < CONF_THR:
                cv2.putText(frame, WB[j].split("-")[1][:2] + ":" + (act[:4] if conf[f, j] >= CONF_THR else "INV"),
                            (p[0] + 6, p[1]), cv2.FONT_HERSHEY_PLAIN, 0.9, col, 1)
        cv2.putText(frame, "f=%d t=%.2fs" % (f, f / fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "GREEN=accept ORANGE=hold YELLOW=ratelimit RED=invalid", (10, h - 15),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
        vw.write(frame)
        f += 1
    cap.release()
    vw.release()
    return out_path


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=r"C:/Unity/viitorx-vrm-avtar-unity-base-project/Assets/Games/video/video.webm")
    ap.add_argument("--out", default="pipeline_logs")
    ap.add_argument("--no-video", action="store_true", help="skip annotated video render")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cache = os.path.join(args.out, "replay_raw.npz")

    uv, zrel, conf, w, h, fps = run_inference(args.video, cache)
    F = uv.shape[0]
    xyz, scale, med_torso = build_metric(uv, zrel, conf)

    p0_cfg = dict(min_cutoff=0.5, beta=0.4, max_jump=1.5, depth_min_cutoff=0.3, depth_beta=0.1, max_hold=8,
                  limb_indices=set([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]) | set(range(91, 133)),
                  overrides={i: 0.35 for i in ARMS} | {i: 0.35 for i in LEGS})
    pre_cfg = dict(min_cutoff=0.5, beta=0.4, max_jump=1.5, depth_min_cutoff=0.3, depth_beta=0.1, max_hold=8,
                   limb_indices=set([5, 6, 7, 8, 9, 10, 11, 12]) | set(range(91, 133)),  # legs EXCLUDED
                   overrides=None)

    xyz_p0, act_p0 = smooth_pass(xyz, conf, p0_cfg)
    xyz_pre, act_pre = smooth_pass(xyz, conf, pre_cfg)

    valid_mask = conf >= CONF_THR
    raw_d = disp_series(xyz, JOINTS, valid_mask)
    pre_d = disp_series(xyz_pre, JOINTS, valid_mask)
    p0_d = disp_series(xyz_p0, JOINTS, valid_mask)

    def summ(dseries):
        r = {}
        for j in JOINTS:
            p = percentiles(dseries[j], [0.5, 0.9, 0.95, 0.99])
            arr = dseries[j]
            r[WB[j]] = {"mean": round(float(np.mean(arr)) if arr else 0, 4), "median": round(p[0.5], 4),
                        "p90": round(p[0.9], 4), "p95": round(p[0.95], 4), "p99": round(p[0.99], 4),
                        "max": round(max(arr) if arr else 0, 4)}
        return r

    spikes = classify_spikes(xyz, conf, JOINTS)
    drop = dropout_runs(conf, act_p0, JOINTS)
    gates = gate_analysis(conf)
    rot = rotation_proxy(xyz, conf)

    # confidence distribution per joint
    cstats = {}
    for j in JOINTS:
        cj = conf[:, j]
        pc = percentiles(cj.tolist(), [0.05, 0.25, 0.5, 0.75, 0.95])
        cstats[WB[j]] = {"min": round(float(cj.min()), 3), "p5": round(pc[0.05], 3), "p25": round(pc[0.25], 3),
                         "median": round(pc[0.5], 3), "p75": round(pc[0.75], 3), "p95": round(pc[0.95], 3),
                         "pct_below_0.3": round(100.0 * float((cj < CONF_THR).mean()), 1)}

    # legitimate motion above 0.35 (est) that P0 would clip
    clipped = {}
    for j in ARMS + LEGS:
        cnt = 0
        exU = []
        arr = raw_d[j]
        for k, d in enumerate(arr):
            if d > 0.35:
                cnt += 1
                if len(exU) < 5:
                    exU.append(round(d, 3))
        clipped[WB[j]] = {"count_gt_0.35": cnt, "examples": exU}

    report = {
        "video": {"path": args.video, "frames_decoded": F, "fps": fps, "resolution": "%dx%d" % (w, h),
                  "duration_s": round(F / fps, 2), "orientation": "portrait" if h > w else "landscape"},
        "depth_validation": "NOT POSSIBLE FROM THIS VIDEO (RGB only, no OAK stereo, no intrinsics)",
        "metric_note": "x/y are ESTIMATED metres (global torso scale %.5f m/px, median torso %.1f px -> %.2f m); z = model root-relative zrel. Estimate exercises the P0 caps; it is NOT real depth." % (scale, med_torso, NOMINAL_TORSO_M),
        "confidence_threshold": CONF_THR,
        "motion_raw_est_m": summ(raw_d),
        "motion_preP0_est_m": summ(pre_d),
        "motion_P0_est_m": summ(p0_d),
        "spikes": {WB[j]: spikes[j] for j in JOINTS},
        "confidence": cstats,
        "dropout_occlusion": {WB[j]: drop[j] for j in JOINTS},
        "p0_1_limb_gate": gates,
        "rotation_proxy_deg": rot,
        "legit_motion_gt_0.35_est": clipped,
    }
    with open(os.path.join(args.out, "replay_metrics.json"), "w") as f_:
        json.dump(report, f_, indent=2)
    print(json.dumps(report, indent=2))

    if not args.no_video:
        outv = os.path.join(os.path.dirname(args.video), "annotated_p0.mp4")
        annotate(args.video, uv, conf, act_p0, outv, fps)
        print("[replay] annotated video ->", outv)


if __name__ == "__main__":
    main()
