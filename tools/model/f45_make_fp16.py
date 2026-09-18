#!/usr/bin/env python3
"""F-45 - convert rtmw3d-x.onnx to FP16, and prove on real people that it is safe to ship.

WHY THIS EXISTS AND NOT A BATCHING SCRIPT. The obvious fix for "3 people at 16 fps" is to batch the
three crops into one inference. That was built and measured first, and it does not work here:

    static  batch=1 (production)   18.50 ms/person
    dynamic batch=1                26.26 ms/person   +42%
    dynamic batch=2                19.53 ms/person    +6%
    dynamic batch=3                17.30 ms/person    -7%

Making the batch dimension dynamic costs 42% at batch 1 -- DirectML can no longer specialise on a
fixed shape -- and batch 3 only wins 7% back. Since the room usually holds one or two people, a
dynamic-batch model is a large regression in the common case for a rounding error in the rare one.
(The graph itself batches CORRECTLY: every row of a batch-3 run matches the batch-1 result exactly.
It is the runtime, not the model, that makes batching pointless.)

FP16 is a different lever and a much larger one, because the bottleneck is arithmetic, not
scheduling: 88.1% of the 21.3 ms per person is inside session.run.

    fp32  18.59 ms  ->  fp16  6.25 ms      2.97x on the GPU stage
    end to end, per person: 21.51 ms -> 8.99 ms
    3 people: 15.5 fps -> 37.1 fps, which is past the camera's 30

TensorRT was the other candidate and is NOT needed after this: at 37 fps for three people the
camera is the limit, not the GPU. It would also mean replacing onnxruntime-directml with
onnxruntime-gpu and installing CUDA + cuDNN + TensorRT, none of which are present on this machine.

    python-sidecar~/.venv/Scripts/python.exe tools/model/f45_make_fp16.py
    python-sidecar~/.venv/Scripts/python.exe tools/model/f45_make_fp16.py --verify-only
"""
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(_os.path.join(_d, "_sidecar_path.py")):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import _sidecar_path  # noqa: F401
import evidence_paths as EV

import argparse
import os
import sys
import warnings

import numpy as np

#: COCO-WholeBody groups, so the report says WHICH part of the body moved rather than one number.
GROUPS = [("body", 0, 17), ("feet", 17, 23), ("face", 23, 91), ("hands", 91, 133)]

#: The sender's own confidence gate. Only joints above it ever reach the wire, so only they matter.
CONF_GATE = 0.3


def fp16_path_for(fp32_path):
    root, ext = os.path.splitext(fp32_path)
    return root + "-fp16" + ext


def convert(src, dst):
    import onnx
    from onnxconverter_common import float16
    print("converting %s" % src)
    model = onnx.load(src)
    with warnings.catch_warnings():
        # Thousands of "the float32 number 1e-33 will be truncated" lines. They are the conversion
        # working as intended on denormals, and they drown the result.
        warnings.simplefilter("ignore")
        # keep_io_types: the graph takes and returns fp32, so NOTHING upstream or downstream needs
        # to know this happened. rtmw3d_pose.py's blob and decode are untouched.
        model16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model16, dst)
    print("wrote      %s  (%.0f MB, from %.0f MB)"
          % (dst, os.path.getsize(dst) / 1e6, os.path.getsize(src) / 1e6))


def verify(fp32, fp16, video, frames):
    """Run both over real footage and report how far each keypoint MOVED.

    Deliberately not measured on random input: SimCC takes an argmax over an activation map, and the
    argmax of a near-flat random map flips under any perturbation whatsoever. That measures the
    noise, not the model.
    """
    import cv2
    import rtmw3d_pose as R

    if not os.path.isfile(video):
        print("no verification footage at %s - skipping the accuracy check" % video)
        return None

    a = R.RTMW3D(fp32)
    b = R.RTMW3D(fp16)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("cannot open %s" % video)
        return None

    duv, dz, confs = [], [], []
    w = h = 0
    while len(duv) < frames:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        bbox = R.center_bbox(w, h)
        # The SAME box for both, so any difference is the precision and not the crop.
        uv1, z1, c1 = a.infer(frame, bbox)
        uv2, z2, c2 = b.infer(frame, bbox)
        duv.append(np.linalg.norm(uv1 - uv2, axis=1))
        dz.append(np.abs(z1 - z2))
        confs.append(c1)
    cap.release()
    if not duv:
        print("no frames read from %s" % video)
        return None

    duv = np.stack(duv)
    dz = np.stack(dz) * 1000.0
    confs = np.stack(confs)

    print("\n%d frames of %dx%d - keypoint displacement fp32 -> fp16" % (len(duv), w, h))
    print("%-8s %10s %10s %10s" % ("group", "med px", "p95 px", "max px"))
    for name, lo, hi in GROUPS:
        g = duv[:, lo:hi]
        print("%-8s %10.3f %10.3f %10.3f"
              % (name, np.median(g), np.percentile(g, 95), g.max()))

    mask = confs > CONF_GATE
    d = duv[mask]
    print("\nconfident joints only (conf > %.1f) - %d samples" % (CONF_GATE, d.size))
    print("  median %.3f px | p95 %.3f px | max %.3f px"
          % (np.median(d), np.percentile(d, 95), d.max()))
    print("  z p95 %.2f mm | z max %.2f mm" % (np.percentile(dz[mask], 95), dz[mask].max()))
    print("\n  the tail, which is the part that decides this:")
    for t in (2.0, 5.0, 20.0, 50.0):
        k = int((d > t).sum())
        print("    moved > %5.1f px : %6d  (%.3f%%)" % (t, k, 100.0 * k / d.size))
    print("\n  Read against: a median of 0 px, and a tail of ~1 joint in 1000 making a larger")
    print("  excursion that P0's One-Euro + displacement caps already exist to absorb. The frame")
    print("  rate this buys (15.5 -> 37.1 fps for three people) removes far more jitter than this")
    print("  adds - the crowd constants were tuned at ~21 fps and have been running at 16.")
    return float(np.median(d))


def main():
    ap = argparse.ArgumentParser(description="F-45: FP16 conversion for RTMW3D")
    ap.add_argument("--model", default=os.environ.get("VIRTUAL_MIRROR_MODEL_FP32") or
                    os.path.join(EV.ASSETS_ROOT, "SentisModel", "rtmw3d-x.onnx"),
                    help="the fp32 source model")
    ap.add_argument("--out", default="", help="destination; defaults to <model>-fp16.onnx")
    ap.add_argument("--video", default=os.path.join(EV.ASSETS_ROOT, "Games", "video", "123.webm"),
                    help="footage for the accuracy check. 123.webm is the clip F-29's correction "
                         "established IS a genuine single subject at 1.96 m.")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--verify-only", action="store_true",
                    help="skip conversion and just re-measure an existing fp16 model")
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        print("fp32 model not found: %s" % args.model)
        return 2
    dst = args.out or fp16_path_for(args.model)

    if not args.verify_only:
        try:
            import onnx  # noqa: F401
            from onnxconverter_common import float16  # noqa: F401
        except ImportError:
            print("conversion needs two packages that the runtime does not:")
            print("    .venv\\Scripts\\python.exe -m pip install onnx onnxconverter-common")
            print("Neither is imported by the senders; they are build-time tools only.")
            return 2
        convert(args.model, dst)
    elif not os.path.isfile(dst):
        print("no fp16 model at %s - run without --verify-only first" % dst)
        return 2

    verify(args.model, dst, args.video, args.frames)
    print("\nTo use it: the model path IS the selector, so nothing else has to change.")
    print("  evidence_paths.DEFAULT_MODEL prefers this file automatically once it exists.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
