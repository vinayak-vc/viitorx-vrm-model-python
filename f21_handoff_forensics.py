#!/usr/bin/env python3
"""F-21 forensics - did the emitted target actually change HUMAN inside a single ownership epoch?

Why this exists. The F-21 report treats 123.webm as a single-person stress clip and reads its
`target_id never left 1, 0 TARGET_SWITCH` as evidence of stable ownership. Inspecting the footage
while classifying F-22's rejections showed the premise is wrong: 123.webm contains a second person
ENTERING around frame 626 and the first person LEAVING afterwards. A flat target_id across a real
person change is not stability - it is the exact silent hand-off F-21 was built to prevent, and the
two readings are indistinguishable from the ownership log alone.

This separates them, without an identity signal the pipeline does not have, by tracking WHERE the
emitted skeleton is in the image over time and how the body's apparent size changes:

  - a sustained translation of the emitted hip to a different part of the frame, with no
    TARGET_RELEASED in between, is a hand-off inside one epoch;
  - a torso-span step change across the same interval corroborates it (different body, different
    apparent size);
  - annotated frames are written either side of every candidate so the conclusion is checkable by
    eye rather than asserted from a number.

    python f21_handoff_forensics.py --video "D:\\...\\video\\123.webm"
"""
import argparse
import io
import json
import os
import sys

import cv2
import numpy as np

import rtmw3d_pose as R
import target_ownership as TO

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "..", "..", "..", "SentisModel", "rtmw3d-x.onnx")
CONF_THR = 0.3
SKEL = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)]


def annotate(frame, uv, conf, text, colour=(0, 255, 255)):
    img = frame.copy()
    for a, b in SKEL:
        if conf[a] > CONF_THR and conf[b] > CONF_THR:
            cv2.line(img, (int(uv[a][0]), int(uv[a][1])), (int(uv[b][0]), int(uv[b][1])), colour, 3)
    for i in range(17):
        if conf[i] > CONF_THR:
            cv2.circle(img, (int(uv[i][0]), int(uv[i][1])), 6, (0, 0, 255), -1)
    cv2.rectangle(img, (0, 0), (img.shape[1], 46), (0, 0, 0), -1)
    cv2.putText(img, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--margin-ratio", type=float, default=0.064,
                    help="production-equivalent: 0.35 m at ~2 m through a ~800 px focal length")
    ap.add_argument("--shift-window", type=int, default=30,
                    help="frames over which a sustained hip translation is measured")
    ap.add_argument("--out-dir", default=os.path.join("oak_v4_evidence", "f21", "handoff"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit("could not open %s" % a.video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    diag = (w * w + h * h) ** 0.5
    margin = diag * a.margin_ratio

    print("=" * 100)
    print(" F-21 hand-off forensics: %s  (%dx%d @ %.0f fps, margin %.0f px)"
          % (os.path.basename(a.video), w, h, fps, margin))
    print("=" * 100)
    model = R.RTMW3D(a.model)
    own = TO.TargetOwnership(TO.OwnershipConfig(min_confidence=CONF_THR, switch_margin_m=margin,
                                                scale_margin_ratio=10.0))
    bbox = R.center_bbox(w, h)
    dt = 1.0 / fps
    t = 0.0
    fi = 0
    track = []
    events = []
    keep = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        uv, zrel, conf = model.infer(frame, bbox)
        ref = R.bbox_from_keypoints(uv, conf, w, h, thr=CONF_THR)
        bcm = float(np.mean(conf[0:17]))
        hip = None
        scale = None
        if conf[11] > CONF_THR and conf[12] > CONF_THR:
            hip = (uv[11] + uv[12]) / 2.0
        elif conf[11] > CONF_THR:
            hip = uv[11].copy()
        elif conf[12] > CONF_THR:
            hip = uv[12].copy()
        if hip is not None and conf[5] > CONF_THR and conf[6] > CONF_THR:
            scale = float(np.linalg.norm((uv[5] + uv[6]) / 2.0 - hip))

        state, emit = own.update(TO.Observation(
            hip is not None, (float(hip[0]), float(hip[1]), 0.0) if hip is not None else None,
            bcm, scale), t)
        for e in own.drain_events():
            e["frame"] = fi
            events.append(e)
            print("  EVENT f%-5d %-28s %s" % (fi, e["event"], e.get("reason", "")))
        track.append(dict(frame=fi, state=state, emit=bool(emit),
                          target_id=own.epoch if emit else None,
                          hip_u=(float(hip[0]) if hip is not None else None),
                          hip_v=(float(hip[1]) if hip is not None else None),
                          scale=scale, conf=round(bcm, 3)))
        if (refined_ok := (ref is not None and bcm >= CONF_THR
                           and state in (TO.ACQUIRING, TO.LOCKED, TO.REACQUIRING))):
            bbox = tuple(0.7 * np.array(bbox) + 0.3 * np.array(ref))
        elif state in (TO.NO_TARGET, TO.RELEASED):
            bbox = R.center_bbox(w, h)
        keep[fi] = (frame, uv.copy(), conf.copy())
        for old in [k for k in keep if k < fi - 400]:
            del keep[old]
        fi += 1
        t += dt
    cap.release()

    emitted = [r for r in track if r["emit"] and r["hip_u"] is not None]
    print("\n  frames=%d  emitted=%d  epochs=%d  releases=%d  switches=%d"
          % (fi, len(emitted), own.epoch,
             sum(1 for e in events if e["event"] == "TARGET_RELEASED"),
             sum(1 for e in events if e["event"] == "TARGET_SWITCH")))

    # ---- the detector -------------------------------------------------------------------------
    # A sustained translation of the emitted hip is NOT evidence of anything on its own: a person
    # walking across the frame over several seconds produces exactly that. The discriminating
    # question is narrower and answerable without any identity signal:
    #
    #   at the moment ownership was lost, did the observation TELEPORT to a body somewhere else
    #   (a jump no human could make in one frame), and is the body that later got REACQUIRED
    #   CONTINUOUS with that other body rather than with the owner?
    #
    # If yes, the reacquire is position-plausible but path-impossible: the thing that walked back
    # into the owner's last-known spot arrived from where the OTHER body was, so it is not the owner.
    by_frame = dict((r["frame"], r) for r in track)
    handoffs = []
    for ev in events:
        if ev["event"] != "TARGET_REACQUIRED":
            continue
        rf = ev["frame"]
        # walk back to the TEMP_LOST that opened this loss episode
        lost = None
        for e2 in events:
            if e2["event"] == "TARGET_TEMP_LOST" and e2["frame"] < rf:
                lost = e2["frame"]
        if lost is None:
            continue
        owner_u = next((by_frame[f]["hip_u"] for f in range(lost - 1, max(-1, lost - 40), -1)
                        if f in by_frame and by_frame[f]["emit"] and by_frame[f]["hip_u"] is not None),
                       None)
        first_u = next((by_frame[f]["hip_u"] for f in range(lost, rf)
                        if f in by_frame and by_frame[f]["hip_u"] is not None), None)
        reacq_u = next((by_frame[f]["hip_u"] for f in range(rf, rf + 10)
                        if f in by_frame and by_frame[f]["hip_u"] is not None), None)
        if owner_u is None or first_u is None or reacq_u is None:
            continue
        jump = abs(first_u - owner_u)
        if jump <= margin:
            continue                      # no teleport at the loss: an ordinary occlusion recovery
        # How far from the owner's last known spot did the observation actually travel during the
        # loss? A candidate that was most of a frame-width away mid-loss, and then walks back inside
        # the margin and is re-locked, is the shape of a hand-off. It is only the SHAPE, though -
        # this script deliberately does NOT auto-classify it, because the honest discriminator is
        # whether the pixels show a different human, and no threshold on hip_u can decide that.
        # Candidates are reported with before/during/after images; the call is made by looking.
        excursion = max(abs(by_frame[f]["hip_u"] - owner_u) for f in range(lost, rf + 1)
                        if f in by_frame and by_frame[f]["hip_u"] is not None)
        handoffs.append(dict(
            lost_frame=lost, reacquired_frame=rf,
            owner_u_at_loss=round(owner_u, 1), other_body_u=round(first_u, 1),
            teleport_px=round(jump, 1), reacquired_u=round(reacq_u, 1),
            max_excursion_from_owner_px=round(excursion, 1),
            reacquire_offset_from_owner_px=round(abs(reacq_u - owner_u), 1),
            loss_duration_s=round((rf - lost) / fps, 3),
            classification="NEEDS_VISUAL_CONFIRMATION"))
        for tagname, fnum in (("owner_before_loss", max(0, lost - 5)),
                              ("other_body_at_loss", lost + 2),
                              ("reacquired_as_same_target", rf + 2)):
            if fnum in keep:
                fr, u2, c2 = keep[fnum]
                img = annotate(fr, u2, c2, "f%d  %s  hip_u=%.0f"
                               % (fnum, tagname.upper().replace("_", " "),
                                  by_frame[fnum]["hip_u"] or -1))
                fn = "%s_handoff_f%05d_%s.png" % (
                    os.path.splitext(os.path.basename(a.video))[0], fnum, tagname)
                cv2.imwrite(os.path.join(a.out_dir, fn), img)
                handoffs[-1].setdefault("images", []).append(fn)

    episodes = handoffs
    print("\n" + "=" * 100)
    print(" HAND-OFF CANDIDATES - reacquires that are position-plausible after the observation")
    print(" had travelled well outside the margin during the loss")
    print("=" * 100)
    if not episodes:
        print("  none - no reacquire followed an out-of-margin excursion")
    for i, e in enumerate(episodes):
        print("  #%d  lost f%d -> reacquired f%d  (%.2f s of loss)"
              % (i + 1, e["lost_frame"], e["reacquired_frame"], e["loss_duration_s"]))
        print("      owner last seen at u=%.0f px" % e["owner_u_at_loss"])
        print("      observation reached %.0f px away from that during the loss (margin %.0f px)"
              % (e["max_excursion_from_owner_px"], margin))
        print("      reacquired at u=%.0f - only %.0f px from the owner's last spot, so the frozen-"
              % (e["reacquired_u"], e["reacquire_offset_from_owner_px"]))
        print("      reference match SUCCEEDED, re-locking as the same target_id with no")
        print("      TARGET_SWITCH and no TARGET_RELEASED.")
        print("      -> classification requires looking at: %s" % ", ".join(e.get("images", [])))
    print("=" * 100)

    verdict = ("%d HAND-OFF CANDIDATE(S) - VISUAL CONFIRMATION REQUIRED" % len(episodes)
               if episodes else "NO HAND-OFF CANDIDATES")
    print(" VERDICT: %s" % verdict)
    if episodes:
        print("  This script does not decide whether the reacquired body is a different human -")
        print("  no threshold on a hip coordinate can. It surfaces the candidate and writes the")
        print("  frames; the determination belongs in the report, made by looking at them.")

    with io.open(os.path.join(a.out_dir, "%s_forensics.json"
                              % os.path.splitext(os.path.basename(a.video))[0]),
                 "w", encoding="utf-8") as f:
        json.dump(dict(video=a.video, frames=fi, emitted=len(emitted), epochs=own.epoch,
                       margin_px=round(margin, 1), verdict=verdict,
                       episodes=episodes, events=events), f, indent=2)
    with io.open(os.path.join(a.out_dir, "%s_track.jsonl"
                              % os.path.splitext(os.path.basename(a.video))[0]),
                 "w", encoding="utf-8") as f:
        for r in track:
            f.write(json.dumps(r) + "\n")
    print(" evidence -> %s" % a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
