#!/usr/bin/env python3
"""F-32 — MULTI-PERSON sidecar. Detect N people, track their identities, pose each one, stream all.

    OAK-D VPU : stereo depth (30 fps)  +  person detection (11.4 fps)   <- both free, VPU was idle
    host      : PersonTracker assigns persistent ids                    <- pure numpy, microseconds
    host GPU  : RTMW3D per tracked person, N capped                     <- 20.7 ms each, the budget
    wire      : every person in one datagram, backward compatible

WHY THIS IS A SEPARATE FILE FROM wholebody_udp_sender.py. That sender is 1200+ lines carrying the
whole measured single-person stack - P0 smoothing, P1-1 joint tracking, P1-4 recovery, F-21
ownership, F-22 validation, F-08 surface depth - all of it accepted on live evidence. Rewriting it
in place to be N-person would put every one of those results at risk for a feature that is new and
unproven. This file reuses its BUILDING BLOCKS by import (build_body_landmarks, build_hand,
build_joint_states) so there is exactly one definition of the landmark contract, and leaves the
production path byte-identical. The two can be compared side by side, which is the point.

F-33 CLOSED THE FILTER GAP. Each tracked person now carries a full PersonFilters chain - P0
smoothing, P1-1 joint tracking, optional P1-4 recovery, F-22 validation - keyed by track id and
living exactly as long as that identity does. See person_filters.py, and --no-filters for the A/B
against the raw F-32 behaviour. What is still deliberately absent:
  * No F-21 ownership. It is superseded here - see person_tracker.py's header.
  * P1-1's horizons are counted in FRAMES, and this loop runs slower than the single-person one
    (~16 fps with three people), so a 6-frame prediction horizon spans ~375 ms here against ~200 ms
    there. The P0 smoother adapts to the real rate; P1-1 does not, and is more forgiving here as a
    result. Stated rather than silently accepted.

MEASURED BUDGET (F-32, this machine): RTMW3D is 20.7 ms p50 with a FIXED batch of 1, so N people
cost N sequential inferences: 2 -> 24 fps, 3 -> 16 fps, 4 -> 12 fps. --max-poses defaults to 3.
The tracker emits most-established-first, so the cap drops the people least likely to still be
there, not an arbitrary subset.
"""

import argparse
import json
import os
import socket
import sys
import time
import uuid

import numpy as np
import cv2
import depthai as dai

import evidence_paths as EV
import oak_depth as D
import person_filters as PF
import person_tracker as PT
import rtmw3d_pose as R
import wholebody_udp_sender as W

#: Detector input. 544x320 is the model's native size; feeding anything else costs an extra resize.
DET_W, DET_H = 544, 320

#: Where the detector blob lives. Fetched once with blobconverter (see docs/F32); not committed
#: because it is a 2.3 MB binary, and the setup script can re-fetch it.
DEFAULT_DETECTOR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "depthai_blazepose", "models",
                                "person-detection-retail-0013_openvino_2022.1_6shave.blob")

SESSION_ID = uuid.uuid4().hex[:12]


def letterbox(frame, tw, th):
    """Fit a frame into the detector's input WITHOUT distorting it, and return the mapping back.

    This is not a detail. An earlier F-32 run squashed a portrait frame into the model's landscape
    input and measured 0.95 people per frame on a clip containing seven; letterboxing took the same
    detector to 7.06. Aspect distortion is the single easiest way to make a good detector look bad.
    """
    h, w = frame.shape[:2]
    s = min(tw / float(w), th / float(h))
    nw, nh = int(w * s), int(h * s)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    ox, oy = (tw - nw) // 2, (th - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(frame, (nw, nh))
    return canvas, s, ox, oy


def build_pipeline(detector_blob, conf):
    """The production RGB-D pipeline plus a person detector on the same VPU.

    Purely ADDITIVE: no existing node is reconfigured, so the F-19 stereo contract is untouched.
    """
    pipeline = D.build_rgbd_pipeline()
    cam = None
    for node in pipeline.getAllNodes():
        if isinstance(node, dai.node.ColorCamera):
            cam = node
    if cam is None:
        raise RuntimeError("no ColorCamera in the production pipeline - did oak_depth change?")

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(DET_W, DET_H)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    manip.setMaxOutputFrameSize(DET_W * DET_H * 3)
    # NON-BLOCKING, QUEUE 1, ON BOTH NODES. Measured: with a blocking NN input the detector
    # back-pressures the ImageManip, which back-pressures ColorCamera.preview - which the production
    # RGB XLinkOut also consumes - and the RGB stream collapses from 29.8 to 12.3 fps. The detector
    # MUST be allowed to drop frames. It is an enrolment source, not a per-frame dependency.
    manip.inputImage.setBlocking(False)
    manip.inputImage.setQueueSize(1)
    cam.preview.link(manip.inputImage)

    nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    nn.setBlobPath(detector_blob)
    nn.setConfidenceThreshold(conf)
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)
    nn.input.setQueueSize(1)
    manip.out.link(nn.input)

    out = pipeline.createXLinkOut()
    out.setStreamName("det")
    nn.out.link(out.input)
    return pipeline


def detections_from(packet, rgb_w, rgb_h, depth_frame, intr):
    """Detector output -> tracker Detections, with metric depth sampled per person where possible.

    Depth is taken at the box CENTRE-UPPER region rather than the centroid: the centroid of a
    standing person's box often lands between the legs and samples the floor behind them, which
    reads as a person several metres further away and poisons the association.
    """
    out = []
    for d in packet.detections:
        x1 = max(0.0, min(1.0, d.xmin))
        y1 = max(0.0, min(1.0, d.ymin))
        x2 = max(0.0, min(1.0, d.xmax))
        y2 = max(0.0, min(1.0, d.ymax))
        if x2 <= x1 or y2 <= y1:
            continue
        depth_m = None
        if depth_frame is not None:
            u = int(0.5 * (x1 + x2) * rgb_w)
            v = int((y1 + 0.35 * (y2 - y1)) * rgb_h)     # chest height, not between the legs
            u = max(0, min(rgb_w - 1, u))
            v = max(0, min(rgb_h - 1, v))
            mm = D.sample_depth_mm(depth_frame, u, v)
            if mm is not None and mm > 0:
                depth_m = mm / 1000.0
        out.append(PT.Detection((x1, y1, x2, y2), d.confidence, depth_m))
    return out


def track_to_bbox(track, rgb_w, rgb_h):
    """A track's normalised box -> the pixel bbox RTMW3D wants, at its required aspect.

    The detector's box is tight around the person; RTMW3D expects a 3:4 portrait crop. Handing it a
    differently-shaped box silently letterboxes inside _preprocess and wastes resolution on padding.
    """
    x1, y1, x2, y2 = track.box
    cx = 0.5 * (x1 + x2) * rgb_w
    cy = 0.5 * (y1 + y2) * rgb_h
    w = (x2 - x1) * rgb_w
    h = (y2 - y1) * rgb_h
    # A little margin: the detector box clips fingers and feet, and RTMW3D needs them in frame.
    w = w * 1.25
    h = h * 1.15
    return R.adjust_bbox_to_aspect(cx, cy, w, h)


def hip_from_track_depth(uv, track, intr):
    """A mid-hip inferred from the TRACKER's own depth, for a person whose hips have no stereo.

    Last resort, and only ever an ORIGIN: it places the body in the room, it does not articulate it.
    Returns None when the track has no depth either (every detection on a video file, and any live
    person the stereo cannot reach).
    """
    if track.depth is None:
        return None
    hip_uv = 0.5 * (uv[11] + uv[12])
    z = float(track.depth)
    return np.array([(hip_uv[0] - intr[2]) * z / intr[0],
                     (hip_uv[1] - intr[3]) * z / intr[1], z], dtype=np.float32)


def build_person_payload(model, frame, track, rgb_w, rgb_h, intr, depth_frame, conf_thr,
                         fallback_depth=True, filters=None, now=None):
    """Pose ONE tracked person and build their slice of the wire payload, or None.

    Shared by the live sender and the video harness so there is exactly one definition of what a
    person on the wire looks like - the same reason f23_video_to_unity.py imports
    build_body_landmarks rather than copying it.

    `filters` is this person's F-33 PersonFilters, or None for the raw F-32 behaviour. When present
    the chain runs in the single-person sender's order: P0 -> mid-hip -> P1-1 -> P1-4 -> F-22. When
    absent every line below behaves exactly as it did before F-33, which is what makes the
    --no-filters A/B a real comparison rather than two different programs.

    Returns (person_dict_or_None, pose_ms, filter_ms).
    """
    bbox = track_to_bbox(track, rgb_w, rgb_h)
    t0 = time.perf_counter()
    uv, zrel, conf = model.infer(frame, bbox)
    pose_ms = (time.perf_counter() - t0) * 1000.0

    if conf[11] < conf_thr or conf[12] < conf_thr:
        return None, pose_ms, 0.0    # no confident hip: nothing downstream can be anchored

    measured = np.zeros(133, dtype=bool)
    xyz_cam = np.zeros((133, 3), dtype=np.float32)
    if depth_frame is not None:
        xyz_cam, measured = D.backproject(uv, depth_frame, rgb_w, rgb_h, intr)

    conf_emit = conf
    track_res = None
    filter_ms = 0.0
    if filters is None:
        if bool(measured[11]) and bool(measured[12]):
            mid_hip = (xyz_cam[11] + xyz_cam[12]) / 2.0
        elif fallback_depth:
            mid_hip = hip_from_track_depth(uv, track, intr)
        else:
            mid_hip = None
    else:
        t_f = time.perf_counter()
        filters.observe_rate(now if now is not None else time.time())
        filters.smooth(xyz_cam, measured)
        fallback = hip_from_track_depth(uv, track, intr) if fallback_depth else None
        mid_hip = filters.mid_hip(xyz_cam, measured, fallback)
        if mid_hip is not None:
            conf_emit, track_res = filters.refine(xyz_cam, measured, conf,
                                                  now if now is not None else time.time())
        filter_ms = (time.perf_counter() - t_f) * 1000.0
    if mid_hip is None:
        return None, pose_ms, filter_ms

    hip_z = float(mid_hip[2])
    zrel_hip = 0.5 * (float(zrel[11]) + float(zrel[12]))
    lm, src = W.build_body_landmarks(uv, xyz_cam, measured, conf_emit, zrel, zrel_hip, mid_hip,
                                     hip_z, intr, conf_thr, flatten_trunk=False, use_zrel=True)
    lm = [[round(v, 4) for v in pt] for pt in lm]
    # Hands take the MODEL's confidences, not conf_emit: P1-1 and F-22 have an opinion about the 12
    # body joints they watch and none at all about the 42 finger landmarks, so gating a hand on a
    # rejected elbow would suppress a hand nothing has found fault with.
    lh = W.build_hand(uv, xyz_cam, measured, conf, zrel, zrel_hip, mid_hip, hip_z, intr, 91,
                      conf_thr, use_zrel=True)
    rh = W.build_hand(uv, xyz_cam, measured, conf, zrel, zrel_hip, mid_hip, hip_z, intr, 112,
                      conf_thr, use_zrel=True)

    person = {
        "id": track.id,
        "lm": lm,
        "xyz": [round(float(mid_hip[0] * 1000.0), 1),
                round(float(mid_hip[1] * 1000.0), 1),
                round(float(mid_hip[2] * 1000.0), 1)],
        "src": src,
        "st": W.build_joint_states(track_res),
        "state": track.state,
    }
    if lh is not None:
        person["lh"] = lh
    if rh is not None:
        person["rh"] = rh
    return person, pose_ms, filter_ms


def add_legacy_primary(message, persons):
    """Publish the most-established person at the ROOT in the exact single-person shape.

    THE BACKWARD-COMPATIBILITY CONTRACT. This is what lets OakDUdpPoseProvider, the VRM mirror app
    and all nine experience scenes consume a multi-person sender with NO change at all - they simply
    see one person, as they always have. It costs one duplicated person (~6 KB of a ~25 KB payload).
    """
    if not persons:
        return
    primary = persons[0]
    message["lm"] = primary["lm"]
    message["xyz"] = primary["xyz"]
    message["src"] = primary["src"]
    message["st"] = primary["st"]
    if "lh" in primary:
        message["lh"] = primary["lh"]
    if "rh" in primary:
        message["rh"] = primary["rh"]


def main():
    ap = argparse.ArgumentParser(description="F-32 multi-person OAK-D sidecar")
    ap.add_argument("--model", default=EV.DEFAULT_MODEL, help="RTMW3D onnx")
    ap.add_argument("--detector", default=DEFAULT_DETECTOR, help="person detector blob")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--det-conf", type=float, default=0.45,
                    help="detector confidence. 0.45 balances recall against phantom people; the "
                         "tracker's confirm-over-N-frames rule catches what slips through.")
    ap.add_argument("--conf", type=float, default=0.3, help="RTMW3D keypoint confidence gate")
    ap.add_argument("--max-poses", type=int, default=3,
                    help="how many tracked people get a pose solve per frame. MEASURED: 20.7 ms "
                         "each, so 2->24fps, 3->16fps, 4->12fps. The tracker emits "
                         "most-established-first, so the cap drops the least-established people.")
    ap.add_argument("--filters", action=argparse.BooleanOptionalAction, default=True,
                    help="F-33: run the single-person filter chain (P0 smoothing, P1-1 joint "
                         "tracking, F-22 validation) once PER TRACKED PERSON. --no-filters "
                         "restores the raw F-32 output for an A/B.")
    ap.add_argument("--adaptive-rate", action=argparse.BooleanOptionalAction, default=True,
                    help="F-33: let each person retune their One-Euro filters from their OWN "
                         "measured update rate. --no-adaptive-rate pins every filter at 30 fps, "
                         "which is what the single-person sender assumes and what this loop is not.")
    ap.add_argument("--filter-feet", action=argparse.BooleanOptionalAction, default=True,
                    help="ADR-071: give the FOOT keypoints the same depth filter and bounded hold "
                         "the limbs get. --no-filter-feet restores the single-person grouping, "
                         "where the feet are in no filter group at all.")
    ap.add_argument("--filter-head", action=argparse.BooleanOptionalAction, default=True,
                    help="ADR-071: same, for the nose/eyes/ears. --no-filter-head restores the "
                         "single-person grouping.")
    ap.add_argument("--filter-recovery", action="store_true", default=False,
                    help="also run P1-4 kinematic recovery per person. REJECTED for production "
                         "(docs/P1_4_CLOSEOUT_2026-09-08.md); research A/B only, as single-person.")
    ap.add_argument("--legacy-primary", action=argparse.BooleanOptionalAction, default=True,
                    help="also publish the most-established person at the ROOT of the payload, in "
                         "the exact single-person shape. This is what lets the existing Unity "
                         "provider and the VRM mirror app consume this sender unchanged.")
    ap.add_argument("--show", action="store_true", help="preview window with track ids")
    args = ap.parse_args()

    if not os.path.isfile(args.detector):
        print("detector blob not found: %s" % args.detector)
        print("fetch it with:  .venv\\Scripts\\python -c \"import blobconverter,shutil;"
              "shutil.copy(blobconverter.from_zoo(name='person-detection-retail-0013',shaves=6,"
              "zoo_type='intel'), r'%s')\"" % args.detector)
        return 2

    import onnxruntime as ort
    eps = ort.get_available_providers()
    print("interpreter : %s" % sys.executable)
    print("providers   : %s" % ", ".join(eps))
    if "DmlExecutionProvider" not in eps:
        print("  !! no DirectML - RTMW3D will run on CPU (~5 fps) and multi-person will be unusable")

    print("loading RTMW3D ...")
    model = R.RTMW3D(args.model)
    tracker = PT.PersonTracker()
    pool = None
    if args.filters:
        pool = PF.PersonFilterPool(PF.FilterConfig(adaptive_rate=args.adaptive_rate,
                                                   recovery=args.filter_recovery,
                                                   filter_feet=args.filter_feet,
                                                   filter_head=args.filter_head))
        print("[mp] F-33 per-person filters ON  (P0 smoothing + P1-1 tracker + F-22 validation%s, "
              "adaptive rate %s)" % (" + P1-4 recovery" if args.filter_recovery else "",
                                     "ON" if args.adaptive_rate else "OFF (pinned 30 fps)"))
    else:
        print("[mp] *** F-33 per-person filters OFF (--no-filters) - raw F-32 output. Every person "
              "streams unsmoothed and untracked; A/B use only.")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (args.host, args.port)

    pipeline = build_pipeline(args.detector, args.det_conf)
    frames = 0
    sent = 0
    t_log = time.time()
    win = 0
    pose_ms_total = 0.0
    filter_ms_total = 0.0
    pose_count = 0

    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb", 4, False)
        q_depth = device.getOutputQueue("depth", 4, False)
        q_det = device.getOutputQueue("det", 4, False)
        rgb_w = rgb_h = None
        intr = None
        latest_dets = []

        print("streaming multi-person to %s:%d  (max %d poses/frame)" % (addr[0], addr[1], args.max_poses))
        while True:
            in_rgb = q_rgb.get()
            t_cap = time.time()
            frame = in_rgb.getCvFrame()
            if rgb_w is None:
                rgb_h, rgb_w = frame.shape[:2]
                intr = D.read_rgb_intrinsics(device, rgb_w, rgb_h)
                print("rgb %dx%d  intrinsics fx=%.1f fy=%.1f" % (rgb_w, rgb_h, intr[0], intr[1]))

            in_depth = q_depth.tryGet()
            depth_frame = in_depth.getFrame() if in_depth is not None else None

            # The detector runs at ~11 Hz against 30 Hz RGB, so most frames reuse the last
            # detections. That is the intended design - tracks carry identity between detections.
            in_det = q_det.tryGet()
            if in_det is not None:
                latest_dets = detections_from(in_det, rgb_w, rgb_h, depth_frame, intr)

            tracks = tracker.update(latest_dets, time.time())
            for e in tracker.drain_events():
                print("[mp] TRACK %-10s id=%-3d hits=%-3d depth=%s"
                      % (e["event"], e["id"], e["hits"], e["depth"]), flush=True)

            persons = []
            now = time.time()
            for track in tracks[:max(1, args.max_poses)]:
                bank = pool.acquire(track.id, now) if pool is not None else None
                person, pose_ms, filter_ms = build_person_payload(
                    model, frame, track, rgb_w, rgb_h, intr, depth_frame, args.conf,
                    filters=bank, now=now)
                pose_ms_total += pose_ms
                filter_ms_total += filter_ms
                pose_count += 1
                if person is not None:
                    persons.append(person)
            if pool is not None:
                # Chains outlive their person by idle_release_seconds, so a track released during a
                # brief absence is reacquired with its history intact rather than starting cold.
                pool.release_idle(now)

            frames += 1
            win += 1
            if not persons:
                continue

            message = {
                "persons": persons,
                "n": len(persons),
                "ndet": len(latest_dets),
                "ntrack": len(tracks),
                "seq": frames,
                "sid": SESSION_ID,
                "t": round(time.time(), 4),
                "lat": round((time.time() - t_cap) * 1000.0, 1),
            }
            if args.legacy_primary:
                add_legacy_primary(message, persons)

            blob = json.dumps(message).encode("utf-8")
            sock.sendto(blob, addr)
            sent += 1

            if time.time() - t_log > 2.0:
                el = time.time() - t_log
                avg_pose = pose_ms_total / max(1, pose_count)
                avg_filter = filter_ms_total / max(1, pose_count)
                print("[mp] frames=%d sent=%d tracks=%d dets=%d posed=%d fps~%.1f pose~%.1fms "
                      "filter~%.2fms bytes=%d"
                      % (frames, sent, len(tracks), len(latest_dets), len(persons),
                         win / el, avg_pose, avg_filter, len(blob)), flush=True)
                if pool is not None:
                    # DIAG-ONLY. One line per person: their own measured rate and what their chain
                    # actually did. A person whose freq reads far below the loop rate is being
                    # starved by --max-poses, which is the number to look at before blaming jitter.
                    for bank in list(pool.banks.values())[:6]:
                        print("[mp]   filters %s" % bank.snapshot(), flush=True)
                win = 0
                pose_ms_total = 0.0
                filter_ms_total = 0.0
                pose_count = 0
                t_log = time.time()

            if args.show:
                vis = frame.copy()
                for track in tracks:
                    x1, y1, x2, y2 = track.box
                    c = [(60, 220, 60), (255, 140, 40), (60, 160, 255), (220, 60, 220),
                         (60, 230, 230), (200, 200, 60)][track.id % 6]
                    cv2.rectangle(vis, (int(x1 * rgb_w), int(y1 * rgb_h)),
                                  (int(x2 * rgb_w), int(y2 * rgb_h)), c, 2)
                    label = "#%d %s" % (track.id, "" if track.depth is None else "%.1fm" % track.depth)
                    cv2.putText(vis, label, (int(x1 * rgb_w), max(int(y1 * rgb_h) - 8, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
                cv2.imshow("multi-person", vis)
                if cv2.waitKey(1) in (27, ord("q")):
                    break

    sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
