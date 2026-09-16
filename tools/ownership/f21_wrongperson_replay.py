#!/usr/bin/env python3

import evidence_paths as EV

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(_os.path.join(_d, "_sidecar_path.py")):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import _sidecar_path  # noqa: F401  - puts the sidecar root and every tools/ group on sys.path
"""F-21 S30 - WRONG_PERSON_FRAMES on real footage, through the whole production chain.

This is the harness the S27 finding needed and did not have. Three things make it a different
evidence class from f21_multiperson_replay.py / f21_handoff_forensics.py:

1. GROUND TRUTH. f21_ground_truth.py labels which HUMAN each emitted frame is actually on, offline,
   from appearance. So the metric is the one the brief asks for -

       wrong-person frames = emitted frames whose human != the human this ownership epoch locked on

   - and not TARGET_SWITCH, which counts only DECLARED hand-overs and read 0 straight through the
   real person change in S27. A label is never guessed: no blob near the emitted hip -> NONE.

2. REAL METRES, NOT A PIXEL PROXY. Earlier F-21 replays set switch_margin_m to a fraction of the
   frame diagonal because a webm has no depth, then reasoned about the production equivalence in
   prose. This one lifts every frame into metric camera space with the SAME pinhole backprojection
   production uses (f22_adversarial.py's load_base_frame, fx=fy=800, hip at 2.0 m), so ownership runs
   on the UNMODIFIED production config - switch_margin_m=0.35 and all - and no conversion is being
   trusted. Stated limitation, because it matters: RTMW3D's zrel is relative TO the hip, so the hip
   itself has no observable depth here and sits on a fixed 2.0 m plane. Hip motion in this replay is
   lateral only; torso scale carries what little distance information there is. A real stereo depth
   channel would make the gate strictly more selective, not less, so this is the pessimistic case.

3. THE WHOLE CHAIN (brief S12). crop -> RTMW3D -> M15 refinement -> F-21 -> F-22 -> the real
   build_body_landmarks. What is counted is what Unity would actually have received.
"""
import argparse
import io
import json
import os
import time

import cv2
import numpy as np

import rtmw3d_pose as R
import target_ownership as TO
import pose_validation as PV
import wholebody_udp_sender as W
from f21_ground_truth import GroundTruth, classify, nearest

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "..", "..", "..", "SentisModel", "rtmw3d-x.onnx")
CONF_THR = 0.3
NOMINAL_HIP_Z = 2.0
FX = FY = 800.0
SKEL = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)]


def lift(uv, zrel, conf, w, h):
    """The identical pinhole lift f22_adversarial.load_base_frame uses, applied per frame."""
    intr = (FX, FY, w / 2.0, h / 2.0)
    zrel_hip = float((zrel[11] + zrel[12]) / 2.0)
    xyz = np.zeros((133, 3), dtype=np.float64)
    zs = NOMINAL_HIP_Z + (np.asarray(zrel, dtype=np.float64).reshape(-1) - zrel_hip)
    xyz[:, 2] = zs
    xyz[:, 0] = (np.asarray(uv[:, 0], dtype=np.float64) - intr[2]) * zs / FX
    xyz[:, 1] = (np.asarray(uv[:, 1], dtype=np.float64) - intr[3]) * zs / FY
    measured = np.array([bool(conf[i] > CONF_THR) for i in range(133)])
    return xyz, measured, intr, zrel_hip


def run(model, gt, video, cfg_kw, ownership_on, f22_on, windows, label_radius, a_default="M"):
    cap = cv2.VideoCapture(video)
    w, h = gt.w, gt.h
    dt = 1.0 / gt.fps
    own = TO.TargetOwnership(TO.OwnershipConfig(**cfg_kw)) if ownership_on else None
    val = PV.PoseValidator() if f22_on else None
    bbox = R.center_bbox(w, h)
    lowconf_streak = 0
    t = 0.0
    fi = 0
    rows, events, own_us = [], [], []
    t_wall0 = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        uv, zrel, conf = model.infer(frame, bbox)
        ref = R.bbox_from_keypoints(uv, conf, w, h, thr=CONF_THR)
        bcm = float(np.mean(conf[0:17]))
        xyz, measured, intr, zrel_hip = lift(uv, zrel, conf, w, h)

        raw_hip, raw_scale, raw_valid = None, None, False
        if measured[11] and measured[12]:
            raw_hip = (xyz[11] + xyz[12]) / 2.0
            raw_valid = True
        elif measured[11]:
            raw_hip, raw_valid = xyz[11].copy(), True
        elif measured[12]:
            raw_hip, raw_valid = xyz[12].copy(), True
        if raw_valid and measured[5] and measured[6]:
            raw_scale = float(np.linalg.norm((xyz[5] + xyz[6]) / 2.0 - raw_hip))

        emit = True
        state = None
        path_blocked = False
        if own is not None:
            obs = TO.Observation(raw_valid,
                                 (tuple(float(v) for v in raw_hip) if raw_valid else None),
                                 bcm, raw_scale)
            _t0 = time.perf_counter()
            state, emit = own.update(obs, t)
            own_us.append((time.perf_counter() - _t0) * 1e6)
            path_blocked = bool(own.snapshot(t).get("path_blocked"))
            for e in own.drain_events():
                e["frame"] = fi
                events.append(e)
            # production M15 acceptance, identity-gated (wholebody_udp_sender.py:731-737)
            if ref is not None and bcm >= CONF_THR and state in (TO.ACQUIRING, TO.LOCKED,
                                                                 TO.REACQUIRING):
                bbox = tuple(0.7 * np.array(bbox) + 0.3 * np.array(ref))
            elif state in (TO.NO_TARGET, TO.RELEASED):
                bbox = R.center_bbox(w, h)
            # else TEMPORARILY_LOST: hold the crop where it was
        else:
            # pre-F-21 M15 behaviour (wholebody_udp_sender.py:749-760), for the OFF arm
            if ref is not None and bcm >= CONF_THR:
                lowconf_streak = 0
                bbox = tuple(0.7 * np.array(bbox) + 0.3 * np.array(ref))
            else:
                lowconf_streak += 1
                if ref is None or lowconf_streak >= 20:
                    bbox = R.center_bbox(w, h)
                    lowconf_streak = 0
                else:
                    bbox = tuple(0.7 * np.array(bbox) + 0.3 * np.array(ref))

        f22_suppressed = 0
        if emit:
            conf_emit = np.array(conf, dtype=np.float64).reshape(-1)
            if val is not None:
                pv_out = val.update(xyz, measured, conf_emit, t)
                for _j, (_s, _r, _b, _rt) in pv_out.items():
                    if _s != PV.VALID:
                        conf_emit[_j] = 0.0
                        f22_suppressed += 1
                val.drain_events()
            mid_hip = (xyz[11] + xyz[12]) / 2.0
            W.build_body_landmarks(uv, xyz, measured, conf_emit, zrel, zrel_hip,
                                   mid_hip, float(mid_hip[2]), intr, CONF_THR, False, True)

        # ---- ground truth ----------------------------------------------------------------------
        hip_uv = None
        if measured[11] and measured[12]:
            hip_uv = (uv[11] + uv[12]) / 2.0
        elif measured[11]:
            hip_uv = uv[11]
        elif measured[12]:
            hip_uv = uv[12]
        blobs = gt.blobs(frame)
        b = nearest(blobs, float(hip_uv[0]), float(hip_uv[1]), label_radius, w, h) \
            if hip_uv is not None else None
        lab, marg = ("NONE", 0.0) if b is None else classify(b, windows, a_default)

        rows.append(dict(frame=fi, t=round(t, 4), state=state, emit=bool(emit),
                         epoch=(own.epoch if own is not None else 0),
                         label=lab, margin=round(marg, 2), n_blobs=len(blobs),
                         hip_u=(float(hip_uv[0]) if hip_uv is not None else None),
                         hip_v=(float(hip_uv[1]) if hip_uv is not None else None),
                         conf=round(bcm, 3), f22_suppressed=f22_suppressed,
                         path_blocked=path_blocked))
        fi += 1
        t += dt
    cap.release()
    return rows, events, own_us, time.perf_counter() - t_wall0


ANCHOR_FRAMES = 15
ANCHOR_MAJORITY = 0.70


def wrong_person(rows, anchor_frames=ANCHOR_FRAMES):
    """Per-epoch: the human the epoch belongs to, then every later emitted frame that disagrees.
    NONE/?-labelled frames are excluded from the comparison (absence of ground truth is not evidence
    of a wrong person) and counted separately so they cannot hide inside a clean number.

    The epoch's owner is the MODAL label over ALL of that epoch's labelled emitted frames, not the
    label of its first one. Anchoring on the first frame makes an entire epoch's score hostage to one
    labelling error, and that is not a theoretical risk - it happened: a single mislabelled frame at
    f4 of 123.webm anchored epoch 1 to the man and scored the woman's own 184 frames as wrong-person
    emissions, reporting 429. An ownership epoch is by construction meant to be ONE person, so the
    person it predominantly emitted is the person it owns, and frames disagreeing with that are the
    wrong-person frames.

    The limit of that definition, stated rather than left for someone to find: if an epoch emitted
    MORE wrong-person frames than owner frames, the mode names the intruder and the count inverts.
    That is why the full per-epoch label histogram is reported next to every result and any epoch
    whose mode is under a %.0f%% majority is FLAGGED AND EXCLUDED rather than scored - a weak anchor
    must not be allowed to produce a confident number.""" % (ANCHOR_MAJORITY * 100)
    by_epoch = {}
    for r in rows:
        if r["emit"] and r["label"] not in ("NONE", "?"):
            by_epoch.setdefault(r["epoch"], []).append(r)
    owner_of, anchors = {}, {}
    for ep, rs in by_epoch.items():
        hist = {}
        for r in rs:
            hist[r["label"]] = hist.get(r["label"], 0) + 1
        best = max(hist.items(), key=lambda kv: kv[1])
        majority = float(best[1]) / max(1, len(rs))
        anchors[ep] = dict(epoch_label_histogram=hist, labelled_frames=len(rs),
                           majority=round(majority, 3), anchored_on=best[0],
                           first_labelled_frame=rs[0]["frame"], first_label=rs[0]["label"],
                           ambiguous=bool(majority < ANCHOR_MAJORITY))
        owner_of[ep] = best[0]

    wrong, none_emitted, episodes = [], 0, []
    for r in rows:
        if not r["emit"]:
            continue
        ep = r["epoch"]
        if r["label"] in ("NONE", "?"):
            none_emitted += 1
            continue
        if owner_of.get(ep) is None or anchors[ep]["ambiguous"]:
            continue
        if r["label"] != owner_of[ep]:
            wrong.append(r)
    for r in wrong:
        if episodes and r["frame"] == episodes[-1][-1]["frame"] + 1:
            episodes[-1].append(r)
        else:
            episodes.append([r])
    return dict(owner_of_epoch=owner_of, wrong=wrong, episodes=episodes,
                unlabelled_emitted=none_emitted, anchors=anchors)


def summarise(tag, rows, events, wp, own_us, wall, fps):
    ev = lambda k: sum(1 for e in events if e["event"] == k)
    emitted = sum(1 for r in rows if r["emit"])
    n = len(rows)
    first = wp["wrong"][0]["frame"] if wp["wrong"] else None
    dur = sum(len(e) for e in wp["episodes"]) / fps if wp["episodes"] else 0.0
    d = dict(tag=tag, frames=n, emitted=emitted,
             wrong_person_frames=len(wp["wrong"]),
             wrong_person_episodes=len(wp["episodes"]),
             first_wrong_person_frame=first,
             wrong_person_duration_s=round(dur, 3),
             unlabelled_emitted=wp["unlabelled_emitted"],
             owner_of_epoch=wp["owner_of_epoch"],
             epoch_anchors=wp["anchors"],
             ambiguous_epochs=[e for e, a in wp["anchors"].items() if a["ambiguous"]],
             epochs=max([r["epoch"] for r in rows] or [0]),
             temp_lost=ev("TARGET_TEMP_LOST"), reacquired=ev("TARGET_REACQUIRED"),
             released=ev("TARGET_RELEASED"), switches=ev("TARGET_SWITCH"),
             rejections=ev("TARGET_REJECTED_CANDIDATE"),
             rejected_path_walked_in=sum(1 for e in events
                                         if e.get("reason") == "path_walked_in"),
             held_frames=n - emitted,
             path_blocked_frames=sum(1 for r in rows if r["path_blocked"]),
             f22_suppressed_joint_frames=sum(r["f22_suppressed"] for r in rows),
             f22_frames_with_suppression=sum(1 for r in rows if r["f22_suppressed"]),
             wall_s=round(wall, 2), throughput_fps=round(n / wall, 2))
    if own_us:
        s = sorted(own_us)
        d["ownership_us_mean"] = round(sum(s) / len(s), 3)
        d["ownership_us_p95"] = round(s[min(len(s) - 1, int(0.95 * len(s)))], 3)
        d["ownership_us_max"] = round(s[-1], 3)
    return d


def dump_frames(video, rows, frames, out_dir, prefix):
    if not frames:
        return []
    os.makedirs(out_dir, exist_ok=True)
    want = dict((f, None) for f in frames)
    by = dict((r["frame"], r) for r in rows)
    cap = cv2.VideoCapture(video)
    i = 0
    written = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            r = by.get(i, {})
            img = fr.copy()
            if r.get("hip_u") is not None:
                cv2.circle(img, (int(r["hip_u"]), int(r["hip_v"])), 18, (0, 0, 255), -1)
                cv2.circle(img, (int(r["hip_u"]), int(r["hip_v"])), 26, (0, 255, 255), 4)
            cv2.rectangle(img, (0, 0), (img.shape[1], 54), (0, 0, 0), -1)
            cv2.putText(img, "f%d %s emit=%s label=%s ep=%s%s"
                        % (i, r.get("state"), r.get("emit"), r.get("label"), r.get("epoch"),
                           " PATH-BLOCKED" if r.get("path_blocked") else ""),
                        (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            p = os.path.join(out_dir, "%s_f%05d.png" % (prefix, i))
            cv2.imwrite(p, cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
            written.append(p)
        i += 1
    cap.release()
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--labels", default="W=55:90:0.15",
                    help="SIGNATURE hue windows, NAME=lo:hi:min_share, comma separated, in priority "
                         "order. A signature colour must belong to exactly one person in the scene.")
    ap.add_argument("--default-label", default="M",
                    help="label for a person-sized blob carrying no signature colour")
    ap.add_argument("--label-radius", type=float, default=0.06,
                    help="max hip->blob distance as a fraction of the frame diagonal")
    ap.add_argument("--out-dir", default=EV.oak_v4("f21", "wrongperson"))
    ap.add_argument("--dump", action="store_true", help="write annotated evidence frames")
    a = ap.parse_args()

    windows = []
    for part in [x for x in a.labels.split(",") if x.strip()]:
        name, rng = part.split("=")
        lo, hi, ms = rng.split(":")
        windows.append((name, float(lo), float(hi), float(ms)))

    os.makedirs(a.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(a.video))[0]
    print("=" * 102)
    print(" F-21 S30 WRONG-PERSON REPLAY - %s   labels=%s" % (os.path.basename(a.video), windows))
    print("=" * 102)
    gt = GroundTruth(a.video)
    print(" ground truth: %dx%d %d frames @ %.0f fps, min blob area %d px"
          % (gt.w, gt.h, gt.n, gt.fps, gt.min_area))
    model = R.RTMW3D(a.model)

    PROD = dict(min_confidence=0.3, acquire_confirm_frames=5, reacquire_confirm_frames=5,
                switch_margin_m=0.35, scale_margin_ratio=0.45,
                reacquire_window_s=2.0, release_timeout_s=4.0)
    arms = [
        ("OWNERSHIP_OFF", dict(PROD), False, True),
        ("PATH_OFF", dict(PROD, path_consistency=False), True, True),
        ("PATH_ON", dict(PROD, path_consistency=True), True, True),
        ("PATH_ON_NO_F22", dict(PROD, path_consistency=True), True, False),
    ]
    out = {}
    for tag, cfg, on, f22 in arms:
        rows, events, own_us, wall = run(model, gt, a.video, cfg, on, f22, windows,
                                         a.label_radius, a.default_label)
        wp = wrong_person(rows)
        s = summarise(tag, rows, events, wp, own_us, wall, gt.fps)
        out[tag] = s
        io.open(os.path.join(a.out_dir, "%s_%s_rows.jsonl" % (base, tag)), "w",
                encoding="utf-8").write("".join(json.dumps(r) + "\n" for r in rows))
        io.open(os.path.join(a.out_dir, "%s_%s_events.jsonl" % (base, tag)), "w",
                encoding="utf-8").write("".join(json.dumps(e) + "\n" for e in events))
        print("\n-- %s" % tag)
        for k in ("frames", "emitted", "held_frames", "wrong_person_frames",
                  "wrong_person_episodes", "first_wrong_person_frame", "wrong_person_duration_s",
                  "unlabelled_emitted", "owner_of_epoch", "epoch_anchors", "ambiguous_epochs",
                  "epochs", "temp_lost", "reacquired",
                  "released", "switches", "rejections", "rejected_path_walked_in",
                  "path_blocked_frames", "f22_frames_with_suppression", "throughput_fps",
                  "ownership_us_mean", "ownership_us_p95", "ownership_us_max"):
            if k in s:
                print("   %-32s %s" % (k, s[k]))
        if a.dump and wp["episodes"]:
            ep = wp["episodes"][0]
            fs = [ep[0]["frame"] - 2, ep[0]["frame"], ep[len(ep) // 2]["frame"], ep[-1]["frame"]]
            for p in dump_frames(a.video, rows, [f for f in fs if f >= 0],
                                 os.path.join(a.out_dir, "frames"), "%s_%s" % (base, tag)):
                print("   evidence -> %s" % p)

    io.open(os.path.join(a.out_dir, "%s_summary.json" % base), "w",
            encoding="utf-8").write(json.dumps(out, indent=2))
    print("\n" + "=" * 102)
    print(" %-18s %8s %8s %8s %10s %9s %8s" % ("ARM", "emitted", "held", "WRONG", "episodes",
                                               "releases", "switch"))
    for tag, _c, _o, _f in arms:
        s = out[tag]
        print(" %-18s %8d %8d %8d %10d %9d %8d"
              % (tag, s["emitted"], s["held_frames"], s["wrong_person_frames"],
                 s["wrong_person_episodes"], s["released"], s["switches"]))
    print("=" * 102)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
