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

MEASURED BUDGET. N people still cost N sequential inferences — the batch stays 1, and F-45 measured
why that is the right choice rather than a limitation: making the ONNX batch dimension dynamic costs
**+42% at batch 1** (DirectML can no longer specialise on a fixed shape) and wins only 7% back at
batch 3, so batching is a large regression in the common case for a rounding error in the rare one.

What DID move is precision. F-45's FP16 build is 2.97x faster on the GPU stage, which is 88% of the
per-person cost:

    per person, end to end     fp32 21.51 ms      fp16 8.99 ms
    1 person                        46.5 fps           111.3 fps
    3 people                        15.5 fps            37.1 fps   <- past the camera's 30
    5 people                         9.3 fps            22.3 fps

`evidence_paths.DEFAULT_MODEL` prefers `rtmw3d-x-fp16.onnx` when it exists (build it with
tools/model/f45_make_fp16.py), and both senders print which weights they loaded. --max-poses still
defaults to 3: the GPU now affords more, but F-34's crowd experiences were designed around three and
that is a separate decision. The tracker emits most-established-first, so the cap drops the people
least likely to still be there, not an arbitrary subset.

P1-1's frame-counted horizons are the thing that most benefits: they were tuned at ~21 fps and have
been running at 16, and now run at the camera's 30.
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
import f18_portrait as PORTRAIT   # the transform VERIFIED in F-18, imported, not re-derived
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

_COCO_PAIRS = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
)


def letterbox_mapping(src_w, src_h, tw=DET_W, th=DET_H):
    """The scale and padding a letterbox from (src_w,src_h) into (tw,th) applies.

    ONE definition, because two places have to agree on it exactly: the host-side `letterbox` below
    (video harness) and `detections_from` (live), which has to invert whatever the VPU's
    ImageManip.setResizeThumbnail did on device. When those two disagree every box is wrong by a
    fixed affine amount, which looks like a bad detector rather than like a mapping bug.

    Returns (scale, offset_x, offset_y) in DETECTOR-INPUT pixels.
    """
    s = min(tw / float(src_w), th / float(src_h))
    return s, (tw - src_w * s) * 0.5, (th - src_h * s) * 0.5


def letterbox(frame, tw, th):
    """Fit a frame into the detector's input WITHOUT distorting it, and return the mapping back.

    This is not a detail. An earlier F-32 run squashed a portrait frame into the model's landscape
    input and measured 0.95 people per frame on a clip containing seven; letterboxing took the same
    detector to 7.06. Aspect distortion is the single easiest way to make a good detector look bad.
    """
    h, w = frame.shape[:2]
    s, ox_f, oy_f = letterbox_mapping(w, h, tw, th)
    nw, nh = int(w * s), int(h * s)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    ox, oy = int(ox_f), int(oy_f)
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

    # THE DETECTOR MUST SEE THE SAME PICTURE THE POSE STAGE DOES, UNDISTORTED AND UNCROPPED.
    #
    # It did not. `cam.preview` is never sized by build_rgbd_pipeline, so it sat at DepthAI's
    # default 300x300 with keepAspectRatio ON - which CENTRE-CROPS the 640x400 ISP to 400x400 and
    # throws 120 px away off each side - and `setResize(544, 320)` then stretched that square to
    # 17:10. Two compounding errors: the detector saw people 1.7x too wide, anyone in the outer
    # 120 px was invisible to it, and `detections_from` mapped the resulting boxes onto the FULL
    # 640x400 frame, so every box was also displaced outward from centre by 640/400.
    #
    # Measured cost of exactly this class of mistake, from this file's own F-32 run: 0.95 people
    # per frame on a clip containing seven, against 7.06 once letterboxed. The validated video
    # harness (tools/video/f32_multiperson_video.py) always letterboxed; the live path never did.
    #
    # So: preview at the ISP's own size with no aspect crop, then setResizeThumbnail - the VPU's
    # own letterbox - into the detector input. `detections_from` inverts it via letterbox_mapping.
    cam.setPreviewSize(cam.getIspWidth(), cam.getIspHeight())
    cam.setPreviewKeepAspectRatio(False)

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResizeThumbnail(DET_W, DET_H)
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


def rotate_box_normalised(box, direction):
    """A normalised box in the RAW camera frame -> the same box in the host's ROTATED frame.

    The detector runs ON DEVICE, on the unrotated camera stream, so its boxes are always in raw
    coordinates no matter what the host does afterwards. When the host rotates for portrait, the
    boxes have to come with it or every crop lands somewhere the person is not.

    Mirrors f18_portrait's pixel mapping exactly, in normalised terms:
        ccw:  u' = v,        v' = W-1-u    ->    x' = y,      y' = 1-x
        cw :  u' = H-1-v,    v' = u        ->    x' = 1-y,    y' = x
    Each mapping flips one axis, so the corners are re-paired to keep x1<x2 and y1<y2.
    """
    x1, y1, x2, y2 = box
    if direction == PORTRAIT.CW:
        return (1.0 - y2, x1, 1.0 - y1, x2)
    return (y1, 1.0 - x2, y2, 1.0 - x1)


def detections_from(packet, det_w, det_h, frame_w, frame_h, depth_frame, intr,
                    portrait_dir=None):
    """Detector output -> tracker Detections, with metric depth sampled per person where possible.

    Depth is taken at the box CENTRE-UPPER region rather than the centroid: the centroid of a
    standing person's box often lands between the legs and samples the floor behind them, which
    reads as a person several metres further away and poisons the association.

    TWO COORDINATE SPACES, and they are not the same one in portrait:
      det_w/det_h     the RAW camera size the on-device detector saw, which is what the letterbox
                      was computed against.
      frame_w/frame_h the host frame everything downstream uses - equal to the raw size in
                      landscape, and swapped in portrait.

    The detector's coordinates are normalised to its LETTERBOXED input, not to the camera frame, so
    every box is un-letterboxed here before it means anything. Skipping either that or the portrait
    rotation does not fail loudly - it just shifts and widens every box by a fixed amount, which
    reads as a detector that cannot find people properly. See build_pipeline.
    """
    s, ox, oy = letterbox_mapping(det_w, det_h)
    inner_w = det_w * s
    inner_h = det_h * s
    out = []
    for d in packet.detections:
        x1 = max(0.0, min(1.0, (d.xmin * DET_W - ox) / inner_w))
        y1 = max(0.0, min(1.0, (d.ymin * DET_H - oy) / inner_h))
        x2 = max(0.0, min(1.0, (d.xmax * DET_W - ox) / inner_w))
        y2 = max(0.0, min(1.0, (d.ymax * DET_H - oy) / inner_h))
        if x2 <= x1 or y2 <= y1:
            continue
        if portrait_dir is not None:
            x1, y1, x2, y2 = rotate_box_normalised((x1, y1, x2, y2), portrait_dir)
        depth_m = None
        if depth_frame is not None:
            u = int(0.5 * (x1 + x2) * frame_w)
            v = int((y1 + 0.35 * (y2 - y1)) * frame_h)   # chest height, not between the legs
            u = max(0, min(frame_w - 1, u))
            v = max(0, min(frame_h - 1, v))
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
                         fallback_depth=True, filters=None, now=None, poses_out=None):
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
    if poses_out is not None:
        poses_out[track.id] = (uv, conf)

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
    # SAME FLAG, SAME SEMANTICS, SAME DEFAULT as wholebody_udp_sender.py - deliberately, because the
    # two senders share STEREO_CONFIG and a flag that meant something different here would be worse
    # than no flag at all.
    #
    # THIS SENDER HAD NO WAY TO ASK FOR SUB-PIXEL AT ALL, which meant every crowd scene ran on
    # whole-pixel disparity while the single-person path ran on eighths - Unity passes
    # --subpixel-bits 3 and only one sender could act on it. Stereo depth error goes as
    # distance^2 * disparity_step / (focal_px * baseline), so the disparity step is a straight
    # multiplier on depth error at EVERY range: 1 px -> 1/8 px is 8x better, everywhere. F-16
    # measured it as a large quantisation win and F-19 deferred it; nothing deferred it here, it
    # was simply never wired.
    ap.add_argument("--subpixel-bits", type=int, default=-1,
                    help="F-19: -1 (default) = production stereo settings untouched. "
                         "0 = sub-pixel off, 3 = 1/8 px (the configuration F-18 validated).")
    # F-43: the OAK-D-PRO's IR dot projector, which had never been switched on. It textures blank
    # walls and plain clothing for the mono pair without appearing in the IR-cut RGB frame, so it
    # buys depth exactly where block matching fails. Default ON; 0 restores the previous behaviour.
    # It matters more here than for the single-person sender: crowd scenes run at 2-2.5 m, where the
    # quantisation step is already 4-8x what it is at the 0.90 m single-person distance.
    # F-44 WORKING VOLUME. See the single-person sender for the full note. It matters more here:
    # crowd scenes run at 2-2.5 m where a body is only ~194-242 px tall, and RTMW3D then UPSCALES
    # that crop to its 384 px input - interpolating detail that was never captured. --rgb-isp 1/1
    # is the direct fix for that, and it costs no GPU time because the pose input size is fixed.
    ap.add_argument("--mono-res", default="", choices=["", "400p", "480p", "720p", "800p"],
                    help="F-44: stereo mono resolution. Empty (default) = untouched. 800p halves "
                         "depth error at every range. Costs VPU time.")
    ap.add_argument("--rgb-isp", default="", choices=["", "1/1", "1/2"],
                    help="F-44: RGB ISP scale. Empty (default) = untouched (1/2 -> 640x400). 1/1 "
                         "keeps native 1280x800 at full FOV: 2x pixels on a body, so 2x the "
                         "distance for the same detail. Costs USB bandwidth.")
    ap.add_argument("--ir-dot", type=float, default=D.IR_DOT_INTENSITY,
                    help="F-43: IR laser dot projector intensity, 0..1 (default %(default)s). "
                         "0 turns it off, restoring the pre-F-43 unassisted stereo pair. "
                         "Ignored with a banner note on boards that have no IR driver.")
    # SAME FLAGS AS THE SINGLE-PERSON SENDER. This sender had NO portrait support at all, which is
    # not the same as having it off: the two senders produced DIFFERENT coordinate frames from the
    # same camera, and `SidecarProcessLauncher`'s claim that this one "handles portrait internally"
    # was simply untrue. The rotation itself is f18_portrait's - image, depth AND intrinsics
    # together, because rotating the image alone corrupts every back-projected X.
    #
    # NOTE FOR PORTRAIT + MULTI-PERSON: the person detector runs ON DEVICE, on the unrotated
    # stream, so in portrait it sees people lying sideways and its recall suffers. The boxes are
    # rotated correctly (rotate_box_normalised), so nothing is misplaced - but a portrait-mounted
    # camera is a worse detector input than a landscape one, which is why landscape is the sensible
    # mount for crowd scenes and portrait for the single-person path that uses no detector at all.
    ap.add_argument("--portrait", action=argparse.BooleanOptionalAction, default=False,
                    help="the camera is physically rotated 90 degrees; rotate image, depth and "
                         "intrinsics back. OFF = landscape (the sensible mount for crowd scenes).")
    ap.add_argument("--portrait-dir", default="ccw", choices=["ccw", "cw"],
                    help="which way the camera is rotated, as seen from behind it.")
    ap.add_argument("--det-timeout", type=float, default=8.0,
                    help="seconds of SILENCE on the detector queue, while RGB is still arriving, "
                         "before treating the on-device detector as stalled and exiting so the "
                         "supervisor rebuilds the pipeline. The detector emits a packet per frame "
                         "even with nobody in view, so silence here is never an empty room. "
                         "0 disables.")
    args = ap.parse_args()

    if args.subpixel_bits >= 0:
        # Applied to the shared STEREO_CONFIG BEFORE build_pipeline -> build_rgbd_pipeline reads it,
        # so the startup banner and the pipeline cannot disagree.
        D.STEREO_CONFIG["subpixel"] = args.subpixel_bits > 0
        D.STEREO_CONFIG["subpixelBits"] = args.subpixel_bits
    # F-44: same rule - mutate the shared config, never the pipeline call, so one place decides.
    if args.mono_res:
        D.STEREO_CONFIG["monoRes"] = args.mono_res
    if args.rgb_isp:
        D.STEREO_CONFIG["rgbIsp"] = tuple(int(v) for v in args.rgb_isp.split("/"))

    #: The rotation direction, or None for landscape. One value carries both "is portrait on" and
    #: "which way", so no call site can apply one without the other.
    portrait_dir = args.portrait_dir if args.portrait else None

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
    print("[mp] model:", model.describe(), flush=True)  # F-45: fp16 vs fp32 is 2.97x
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
    last_bytes = 0
    pose_ms_total = 0.0
    filter_ms_total = 0.0
    pose_count = 0

    device = None
    print("[mp] connecting to OAK-D camera...", flush=True)
    while True:
        try:
            device = dai.Device(pipeline)
            break
        except RuntimeError as e:
            if "Cannot find any device" in str(e):
                print("[mp] OAK-D camera not found. Please connect the OAK-D camera via USB. Retrying in 2s...", flush=True)
                time.sleep(2.0)
            else:
                raise

    with device:
        q_rgb = device.getOutputQueue("rgb", 4, False)
        q_depth = device.getOutputQueue("depth", 4, False)
        q_det = device.getOutputQueue("det", 4, False)
        # det_* is the RAW camera size (what the on-device detector sees); rgb_* is the host
        # frame size, which is the same in landscape and swapped in portrait.
        det_w = det_h = None
        rgb_w = rgb_h = None
        intr = None
        latest_dets = []
        tracks = []
        t_last_det = time.time()

        print("streaming multi-person to %s:%d  (max %d poses/frame)" % (addr[0], addr[1], args.max_poses))
        # Announced, not assumed. Sub-pixel is the difference between whole-pixel and 1/8-pixel
        # disparity - an 8x multiplier on depth error at every range - and it is invisible in the
        # stream, so the only way to tell it took effect is to print what the pipeline was built
        # with. The single-person sender has always done this; this one never did, which is part of
        # why it ran on whole-pixel disparity unnoticed.
        print("[mp]   stereo: %s" % D.stereo_config_str(), flush=True)
        # F-43. Reports what the DEVICE actually did, not what was asked for, so a board without the
        # emitter says so here instead of silently looking identical to one that has it.
        print("[mp]   IR dot projector: %s" % D.enable_ir_dot_projector(device, args.ir_dot),
              flush=True)
        # The supervisor's readiness contract (sidecar_supervisor.py: SID_MARKER then FRAME_MARKER).
        # It watches stdout for this banner and then for the first `frames=` line, and treats a
        # child that never prints both as a stuck startup. Printed in the single-person sender's
        # exact wording so one marker serves both senders.
        print("[mp]   producer session id = %s  (F-20A: new on every process start)" % SESSION_ID,
              flush=True)
        while True:
            in_rgb = q_rgb.get()
            t_cap = time.time()
            frame = in_rgb.getCvFrame()
            if det_w is None:
                # RAW camera size, captured BEFORE any rotation: the on-device detector sees this
                # one, and read_rgb_intrinsics scales the calibration to it. Rotating the
                # intrinsics is a separate step and must come after, not instead.
                det_h, det_w = frame.shape[:2]
                intr = D.read_rgb_intrinsics(device, det_w, det_h)
                if portrait_dir is not None:
                    intr = PORTRAIT.rotate_intrinsics(intr, det_w, det_h, portrait_dir)
                    rgb_w, rgb_h = det_h, det_w
                else:
                    rgb_w, rgb_h = det_w, det_h
                print("rgb %dx%d  intrinsics fx=%.1f fy=%.1f  (%s)"
                      % (rgb_w, rgb_h, intr[0], intr[1],
                         "PORTRAIT_" + portrait_dir.upper() if portrait_dir else "LANDSCAPE"),
                      flush=True)
            if portrait_dir is not None:
                frame = PORTRAIT.rotate_image(frame, portrait_dir)

            in_depth = q_depth.tryGet()
            depth_frame = in_depth.getFrame() if in_depth is not None else None
            if depth_frame is not None and portrait_dir is not None:
                # Depth is aligned to CAM_A, so it MUST be rotated with the colour frame - the two
                # are indexed by the same (u,v) everywhere downstream.
                depth_frame = PORTRAIT.rotate_image(depth_frame, portrait_dir)

            # THE TRACKER IS FED ONCE PER DETECTION, NOT ONCE PER RGB FRAME.
            #
            # The detector runs at ~11 Hz against 30 Hz RGB. Re-feeding the same `latest_dets` on
            # every frame - which is what this did - counts ONE detection as three, so
            # `confirm_hits = 3` was satisfied by a single detector packet and the tracker's own
            # "a flickering false positive never becomes a person" rule could never fire. A chair
            # or a stack of cartons clipped once became a CONFIRMED person in ~100 ms, and the
            # LOST/REACQUIRED churn in the log was the same list being re-scored at the wrong rate.
            #
            # Updating only on a new packet also puts the tracker's dt on the clock that actually
            # produces evidence, which is what its velocity and miss thresholds were tuned against.
            # Between packets the tracks stand, and the pose stage keeps using their boxes - that
            # is the part that was always meant by "tracks carry identity between detections".
            in_det = q_det.tryGet()
            if in_det is not None:
                t_last_det = time.time()
                latest_dets = detections_from(in_det, det_w, det_h, rgb_w, rgb_h,
                                              depth_frame, intr, portrait_dir)
                tracks = tracker.update(latest_dets, time.time())
                for e in tracker.drain_events():
                    print("[mp] TRACK %-10s id=%-3d hits=%-3d depth=%s"
                          % (e["event"], e["id"], e["hits"], e["depth"]), flush=True)
            else:
                # A PRODUCER THAT CANNOT PRODUCE MUST SAY SO.
                #
                # Measured 2026-09-17: the on-device detector stopped emitting after 1457 frames and
                # never resumed. The process stayed alive at a healthy 30 fps for another 59,000
                # frames with dets=0, streaming nothing, and Unity sat on "WAITING for the sidecar"
                # for half an hour. Nothing was logged by DepthAI and nothing noticed.
                #
                # It used to be caught BY ACCIDENT: no detections meant no people meant no stdout,
                # and the supervisor's heartbeat killed and restarted it. Fixing the heartbeat so an
                # empty room does not trigger a restart removed that accident, so the real check has
                # to exist deliberately.
                #
                # THE TEST IS UNAMBIGUOUS because the detector emits a packet per input frame
                # whether or not it finds anybody - an empty room still produces ImgDetections with
                # an empty list. So silence on this queue is never "nobody is here"; it is always
                # the detector having stopped. Exiting non-zero hands the problem to the supervisor,
                # which rebuilds the whole pipeline - the only thing known to clear it.
                silent = time.time() - t_last_det
                if args.det_timeout > 0 and silent > args.det_timeout:
                    print("[mp] FATAL detector silent for %.1fs while RGB still arrives "
                          "(frames=%d, fps~%.1f). The on-device detector has stalled; exiting so "
                          "the supervisor rebuilds the pipeline."
                          % (silent, frames, win / max(1e-6, time.time() - t_log)), flush=True)
                    sock.close()
                    return 3

            # ONLY PEOPLE THE DETECTOR CAN CURRENTLY SEE GET POSED AND SENT. `update` also returns
            # LOST tracks - coasting on prediction, with no detection behind them - and posing one
            # runs RTMW3D on a box that is drifting away from any body, which is where the second
            # skeleton on a single person came from. They stay in the tracker so an occlusion does
            # not destroy the identity; they just do not become people on the wire.
            posable = [t for t in tracks if t.state == PT.CONFIRMED]

            persons = []
            poses = {}
            now = time.time()
            for track in posable[:max(1, args.max_poses)]:
                bank = pool.acquire(track.id, now) if pool is not None else None
                person, pose_ms, filter_ms = build_person_payload(
                    model, frame, track, rgb_w, rgb_h, intr, depth_frame, args.conf,
                    filters=bank, now=now, poses_out=poses)
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

            if args.show:
                vis = frame.copy()
                for track in tracks:
                    x1, y1, x2, y2 = track.box
                    lost = track.state != PT.CONFIRMED
                    # A held track and a sent one must not look alike: the whole confusion this
                    # preview is meant to resolve is "why are there two boxes on one person".
                    c = (130, 130, 130) if lost else [
                        (60, 220, 60), (255, 140, 40), (60, 160, 255), (220, 60, 220),
                        (60, 230, 230), (200, 200, 60)][track.id % 6]
                    cv2.rectangle(vis, (int(x1 * rgb_w), int(y1 * rgb_h)),
                                  (int(x2 * rgb_w), int(y2 * rgb_h)), c, 1 if lost else 2)
                    label = "#%d %s%s" % (track.id,
                                          "" if track.depth is None else "%.1fm" % track.depth,
                                          " held" if lost else "")
                    cv2.putText(vis, label, (int(x1 * rgb_w), max(int(y1 * rgb_h) - 8, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
                    pose_info = poses.get(track.id)
                    if pose_info is not None:
                        uv_kpts, conf_kpts = pose_info
                        for p1, p2 in _COCO_PAIRS:
                            if conf_kpts[p1] > args.conf and conf_kpts[p2] > args.conf:
                                pt1 = (int(uv_kpts[p1, 0]), int(uv_kpts[p1, 1]))
                                pt2 = (int(uv_kpts[p2, 0]), int(uv_kpts[p2, 1]))
                                cv2.line(vis, pt1, pt2, c, 2, cv2.LINE_AA)
                        for i in range(min(133, uv_kpts.shape[0])):
                            if conf_kpts[i] > args.conf:
                                cv2.circle(vis, (int(uv_kpts[i, 0]), int(uv_kpts[i, 1])), 2,
                                           c if i < 91 else (255, 255, 255), -1)
                hud = "tracks=%d (held %d) dets=%d posed=%d sent=%d   [q] close preview" % (
                    len(tracks), len(tracks) - len(posable), len(latest_dets), len(persons), sent)
                cv2.putText(vis, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("multi-person", vis)
                # ESC IS NOT A QUIT KEY HERE, deliberately. Unity's own on-screen hint is "ESC menu",
                # so ESC is the key a person naturally presses while this window happens to have
                # focus - and it used to end the sidecar, taking tracking down for the rest of the
                # session. Closing the preview must never be the same gesture as killing the
                # producer, so only `q` does it, and it closes the WINDOW rather than the process.
                if cv2.waitKey(1) == ord("q"):
                    args.show = False
                    try:
                        cv2.destroyWindow("multi-person")
                    except Exception:
                        pass

            # THE HEARTBEAT MUST NOT DEPEND ON ANYBODY BEING IN FRAME.
            #
            # This block used to sit BELOW the `if not persons: continue` a few lines down, so an
            # empty room produced no stdout at all. That was harmless while this sender ran
            # unsupervised and fatal the moment it did not: sidecar_supervisor.py watches stdout
            # growth and kills a child that goes quiet for --heartbeat-timeout (10 s). Measured on
            # 2026-09-17 - "HEARTBEAT TIMEOUT pid=41620 no stdout growth for 10s - killing for
            # restart", rc=1 after 463.9 s - which reads to a person as the skeleton freezing for no
            # reason. Walking out of frame for ten seconds must not restart the producer.
            #
            # So it runs BEFORE the empty-frame skip, and reports the empty case honestly rather
            # than being suppressed by it.
            if time.time() - t_log > 2.0:
                el = time.time() - t_log
                avg_pose = pose_ms_total / max(1, pose_count)
                avg_filter = filter_ms_total / max(1, pose_count)
                print("[mp] frames=%d sent=%d tracks=%d dets=%d posed=%d fps~%.1f pose~%.1fms "
                      "filter~%.2fms bytes=%d detAge~%.1fs"
                      % (frames, sent, len(tracks), len(latest_dets), len(persons),
                         win / el, avg_pose, avg_filter, last_bytes,
                         time.time() - t_last_det), flush=True)
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
            last_bytes = len(blob)

    if args.show:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
