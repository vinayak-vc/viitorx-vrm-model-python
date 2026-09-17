#!/usr/bin/env python3
"""F-32 — drive the MULTI-PERSON wire from a recorded video file.

The multi-person analogue of f23_video_to_unity.py, and the only way to test this work while nobody
is standing in front of the camera. Same honesty caveat as that file, and one more:

WHAT IS REAL
  * person-detection-retail-0013 on the OAK-D VPU - the production detector, on the real device
  * PersonTracker - the production tracker, unmodified
  * build_person_payload / add_legacy_primary - the PRODUCTION payload builders, imported
  * the wire shape and the UDP port - byte-identical in structure to the live sender

WHAT IS NOT
  * NO STEREO DEPTH. A video file has one camera, so every detection carries depth=None and the
    tracker runs its DEGRADED image-plane-only association. Depth is the tracker's strongest
    discriminator, so live performance should be BETTER than this, not worse - but that is an
    expectation, not a measurement.
  * The hip depth is derived from the model's own monocular z against an assumed subject height,
    exactly as f23 does. It scales the scene; it does not invent articulation.
  * --synth-depth (F-33) SYNTHESISES a depth frame: each tracked person's box filled with their own
    estimated distance plus a stereo-like noise model. Everything downstream of it is the real
    production code - oak_depth.backproject, the F-08 surface sampler, the P0 smoother, P1-1, F-22 -
    but the depth going IN is a model, so a `src` flag of 1 here means "sampled the synthetic depth
    frame", not "measured by stereo". It exists because a depth-less clip gives P0 nothing to smooth
    and starves P1-1 into marking every joint LOST; without it this harness cannot exercise F-33 at
    all. Numbers from it describe the FILTER CHAIN, never the camera.

    .venv\\Scripts\\python.exe tools/video/f32_multiperson_video.py --video ..\\..\\video\\456.webm
"""

import os as _os
import sys as _sys

_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(_os.path.join(_d, "_sidecar_path.py")):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import _sidecar_path  # noqa: F401

import argparse
import json
import math
import socket
import time
import uuid

import numpy as np
import cv2
import depthai as dai

import evidence_paths as EV
import multiperson_udp_sender as MP
import person_filters as PF
import person_tracker as PT
import rtmw3d_pose as R

#: Stereo depth quantises in steps that grow with the SQUARE of distance: one disparity unit is
#: z^2 / (fx * baseline * subpixel_steps) metres. With the OAK-D-PRO-W's 75 mm baseline, a ~430 px
#: focal at the 640x400 depth resolution and 1/8-pixel subpixel, that is ~16 mm at 2 m and ~62 mm at
#: 4 m. The synthetic frame reproduces that shape so the smoother meets noise of a plausible SIZE
#: and DISTANCE DEPENDENCE. It is a model of the sensor, not a capture from it.
#:
#: THE ASSUMPTION, AND WHICH WAY IT IS WRONG: this uses 1/8-pixel subpixel. Production stereo has
#: subpixel OFF (oak_depth.STEREO_CONFIG), so the real quantisation step is up to 8x coarser than
#: modelled here - the HIGH_DENSITY preset and the F-08 percentile window recover some of that, by
#: an amount nobody has measured. So this bench UNDERSTATES the noise the filter faces. It shifts
#: every arm of the A/B together, which is what matters for comparing them.
SYNTH_DEPTH_FX_BASELINE = 430.0 * 0.075 * 8.0


def synth_depth_frame(tracks, w, h, rng):
    """A depth image built from what the tracker believes, for a clip that has no depth of its own.

    Painted FAR TO NEAR so a nearer person overwrites a further one where their boxes overlap -
    which is what occlusion does, and the case the whole multi-person design exists to handle.
    Pixels outside every box stay 0, which oak_depth reads as a hole, exactly as it would off the
    real camera.
    """
    depth = np.zeros((h, w), dtype=np.uint16)
    ordered = sorted([t for t in tracks if t.depth is not None], key=lambda t: -t.depth)
    for t in ordered:
        x1 = max(0, int(t.box[0] * w))
        y1 = max(0, int(t.box[1] * h))
        x2 = min(w, int(t.box[2] * w))
        y2 = min(h, int(t.box[3] * h))
        if x2 <= x1 or y2 <= y1:
            continue
        z = float(t.depth)
        sigma_mm = 1000.0 * (z * z) / SYNTH_DEPTH_FX_BASELINE
        patch = rng.normal(z * 1000.0, sigma_mm, (y2 - y1, x2 - x1))
        depth[y1:y2, x1:x2] = np.clip(patch, 1.0, 65000.0).astype(np.uint16)
    return depth


SID = "mp%09x" % (uuid.uuid4().int & 0xFFFFFFFFF)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default=EV.DEFAULT_MODEL)
    ap.add_argument("--detector", default=MP.DEFAULT_DETECTOR)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--det-conf", type=float, default=0.45)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--max-poses", type=int, default=3)
    ap.add_argument("--hfov", type=float, default=60.0,
                    help="assumed horizontal FOV of the source camera; scale only")
    ap.add_argument("--subject-height", type=float, default=1.70)
    ap.add_argument("--loop", type=int, default=1)
    ap.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--legacy-primary", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--filters", action=argparse.BooleanOptionalAction, default=True,
                    help="F-33 per-person filter chain. --no-filters for the raw F-32 A/B.")
    ap.add_argument("--adaptive-rate", action=argparse.BooleanOptionalAction, default=True,
                    help="F-33: each person retunes their filters from their own measured rate.")
    ap.add_argument("--filter-feet", action=argparse.BooleanOptionalAction, default=True,
                    help="ADR-071: give the FOOT keypoints the same depth filter and bounded hold "
                         "the limbs get. --no-filter-feet restores the single-person grouping, "
                         "where the feet are in no filter group at all.")
    ap.add_argument("--filter-head", action=argparse.BooleanOptionalAction, default=True,
                    help="ADR-071: same, for the nose/eyes/ears. --no-filter-head restores the "
                         "single-person grouping.")
    ap.add_argument("--synth-depth", action=argparse.BooleanOptionalAction, default=True,
                    help="synthesise a depth frame from the tracker (see the module header). "
                         "REQUIRED for --filters to do anything: with no depth at all the smoother "
                         "has nothing to smooth and P1-1 starves to LOST.")
    ap.add_argument("--seed", type=int, default=12345,
                    help="synthetic depth noise seed. Fixed so an A/B feeds both runs IDENTICAL "
                         "input, which AGENTS.md section 8 requires of any A/B.")
    ap.add_argument("--dump", default="", help="also write each payload to this jsonl")
    a = ap.parse_args()

    if not _os.path.isfile(a.detector):
        print("detector blob not found: %s" % a.detector)
        return 2

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        print("cannot open %s" % a.video)
        return 2
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fx = fy = 0.5 * W / math.tan(math.radians(a.hfov) * 0.5)
    intr = (fx, fy, W * 0.5, H * 0.5)
    print("video       : %dx%d @ %.1f fps   assumed fx=fy=%.1f" % (W, H, src_fps, fx))

    print("loading RTMW3D ...")
    model = R.RTMW3D(a.model)
    tracker = PT.PersonTracker()
    pool = None
    if a.filters:
        pool = PF.PersonFilterPool(PF.FilterConfig(adaptive_rate=a.adaptive_rate,
                                                   filter_feet=a.filter_feet,
                                                   filter_head=a.filter_head))
    rng = np.random.RandomState(a.seed)
    print("filters     : %s   adaptive rate: %s   synth depth: %s"
          % ("ON" if a.filters else "OFF", "ON" if a.adaptive_rate else "OFF (pinned 30 fps)",
             "ON" if a.synth_depth else "OFF"))
    if a.filters and not a.synth_depth:
        print("  !! --filters with --no-synth-depth: there is no depth to filter. P1-1 will mark")
        print("     every tracked joint LOST within a few frames and the body will stop emitting.")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dump = open(a.dump, "w", encoding="utf-8") if a.dump else None

    # Detector-only pipeline: no camera, no stereo. Frames are fed from the host over XLinkIn.
    pipeline = dai.Pipeline()
    nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    nn.setBlobPath(_os.path.abspath(a.detector))
    nn.setConfidenceThreshold(a.det_conf)
    nn.setNumInferenceThreads(2)
    xin = pipeline.createXLinkIn()
    xin.setStreamName("in")
    xin.out.link(nn.input)
    xout = pipeline.createXLinkOut()
    xout.setStreamName("out")
    nn.out.link(xout.input)

    seq = 0
    sent = 0
    skipped = 0
    t_start = time.time()
    frame_i = 0
    pose_ms_total = 0.0
    filter_ms_total = 0.0
    pose_count = 0
    biggest = 0

    with dai.Device(pipeline) as device:
        qin = device.getInputQueue("in")
        qout = device.getOutputQueue("out", 4, True)

        for _ in range(max(1, a.loop)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                t_frame = time.time()

                canvas, s, ox, oy = MP.letterbox(frame, MP.DET_W, MP.DET_H)
                img = dai.ImgFrame()
                img.setType(dai.ImgFrame.Type.BGR888p)
                img.setWidth(MP.DET_W)
                img.setHeight(MP.DET_H)
                img.setData(canvas.transpose(2, 0, 1).flatten())
                qin.send(img)
                packet = qout.get()

                dets = []
                for d in packet.detections:
                    x1 = ((d.xmin * MP.DET_W - ox) / s) / W
                    y1 = ((d.ymin * MP.DET_H - oy) / s) / H
                    x2 = ((d.xmax * MP.DET_W - ox) / s) / W
                    y2 = ((d.ymax * MP.DET_H - oy) / s) / H
                    x1 = max(0.0, min(1.0, x1)); y1 = max(0.0, min(1.0, y1))
                    x2 = max(0.0, min(1.0, x2)); y2 = max(0.0, min(1.0, y2))
                    if x2 > x1 and y2 > y1:
                        dets.append(PT.Detection((x1, y1, x2, y2), d.confidence, None))

                # THE VIDEO's clock, not the wall clock. A file replay that reads the wall clock is
                # not reproducible: the tracker's constant-velocity prediction is scaled by dt, so
                # the same clip yields different identities on a faster machine, or simply on a
                # second run - which showed up here as an A/B whose two halves disagreed about
                # which ids existed. F-32 measured this harness on the wall clock and its id counts
                # were therefore machine-dependent; the live sender is unaffected, where the wall
                # clock IS the frame clock.
                tracks = tracker.update(dets, frame_i / src_fps)
                for e in tracker.drain_events():
                    print("[mpv] TRACK %-10s id=%-3d hits=%d" % (e["event"], e["id"], e["hits"]),
                          flush=True)

                # No stereo: give each track a hip depth from its own apparent height, so people
                # further from the camera are placed further away instead of all on one plane.
                # Done for EVERY track before any posing, because the synthetic depth frame is
                # shared and a nearer person must be able to occlude a further one in it.
                for track in tracks:
                    span_px = (track.box[3] - track.box[1]) * H
                    hip_z = (a.subject_height * fy / span_px) if span_px > 40 else 2.0
                    track.depth = min(max(hip_z, 0.6), 6.0)
                depth_frame = synth_depth_frame(tracks, W, H, rng) if a.synth_depth else None

                persons = []
                # The filters are given the VIDEO's own timeline, not the wall clock. Two reasons.
                # (1) Honesty: these frames really are 1/src_fps apart, however long the host takes
                # to chew through them, and that interval is what a temporal filter should model.
                # (2) An A/B has to feed both runs identical input (AGENTS.md section 8), and a wall
                # clock does not: the --filters run is slower than the --no-filters run BECAUSE of
                # the thing being measured, so it would be handed different timestamps.
                # The tracker keeps its wall clock, unchanged, so F-32's measured behaviour stands.
                now = frame_i / src_fps
                for track in tracks[:max(1, a.max_poses)]:
                    bank = pool.acquire(track.id, now) if pool is not None else None
                    person, pose_ms, filter_ms = MP.build_person_payload(
                        model, frame, track, W, H, intr, depth_frame, a.conf,
                        filters=bank, now=now)
                    pose_ms_total += pose_ms
                    filter_ms_total += filter_ms
                    pose_count += 1
                    if person is not None:
                        persons.append(person)
                if pool is not None:
                    pool.release_idle(now)

                frame_i += 1
                if not persons:
                    skipped += 1
                    continue

                seq += 1
                message = {
                    "persons": persons,
                    "n": len(persons),
                    "ndet": len(dets),
                    "ntrack": len(tracks),
                    "seq": seq,
                    "sid": SID,
                    "t": round(time.time(), 4),
                    "lat": round((time.time() - t_frame) * 1000.0, 1),
                }
                if a.legacy_primary:
                    MP.add_legacy_primary(message, persons)

                blob = json.dumps(message).encode("utf-8")
                biggest = max(biggest, len(blob))
                sock.sendto(blob, (a.host, a.port))
                sent += 1
                if dump:
                    dump.write(json.dumps(dict(message, frame=frame_i)) + "\n")

                if a.realtime:
                    lag = (frame_i / src_fps) - (time.time() - t_start)
                    if lag > 0:
                        time.sleep(min(lag, 0.25))
                if sent % 60 == 0:
                    el = time.time() - t_start
                    print("  sent=%-5d frame=%-5d people=%d tracks=%-2d %.1f fps  pose~%.1fms "
                          "filter~%.2fms"
                          % (sent, frame_i, len(persons), len(tracks), sent / max(1e-6, el),
                             pose_ms_total / max(1, pose_count),
                             filter_ms_total / max(1, pose_count)), flush=True)

    cap.release()
    if dump:
        dump.close()
    el = time.time() - t_start
    print("")
    print("sent %d packets, skipped %d (nobody posed), %.1f s, %.1f fps"
          % (sent, skipped, el, sent / max(1e-6, el)))
    print("mean pose inference %.1f ms over %d solves" % (pose_ms_total / max(1, pose_count), pose_count))
    print("mean F-33 filter    %.2f ms per person-frame" % (filter_ms_total / max(1, pose_count)))
    print("largest datagram %d bytes  (UDP practical limit is 65507)" % biggest)
    if pool is not None:
        print("filter banks: %s" % pool.snapshot())
        for bank in sorted(pool.banks.values(), key=lambda b: -b.updates)[:6]:
            print("  %s" % bank.snapshot())
    if a.synth_depth:
        print("src=1 here means SYNTHETIC depth (see the module header), never stereo.")
    else:
        print("every src flag is 0: no joint on this wire carries a depth measurement.")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
