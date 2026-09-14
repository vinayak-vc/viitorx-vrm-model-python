#!/usr/bin/env python3
"""F-19 — replay the F-18 PORTRAIT captures onto the real UDP wire, so the complete production
Unity pipeline (P1-3 buffer -> TrunkGate -> V5 torso -> V6 guard -> Arm V2 -> VRM) can be exercised
and its RENDERED BONES recorded, with no camera attached.

WHAT IS REAL HERE AND WHAT IS NOT — read this before trusting any number downstream.

  REAL, measured by the OAK-D in portrait:
    * every keypoint's PIXEL position and confidence (`kp`, 17 COCO joints)
    * the portrait intrinsics
    * the two SHOULDER depths (zL, zR) and the HIP depth (hipZ), which are what the torso-yaw
      signal is actually built from

  NOT recorded by F-18, therefore RECONSTRUCTED here:
    * per-joint depth for every joint that is not a shoulder or a hip. F-18 stored the summary
      fields it needed, not the depth frame, so elbow/wrist/knee/ankle Z cannot be recovered.
      Each such joint is placed on the nearest measured trunk depth plane (shoulder plane for the
      upper body, hip plane for the lower body).

  CONSEQUENCE, stated once and repeated in the report: TORSO results from this replay are driven by
  genuinely measured depth and are meaningful. ARM and LEG results are driven by a depth assumption,
  so they characterise the RETARGET AND THE AVATAR (which is what F-19 §12-§15 audit) but they do
  NOT characterise the tracker's distal accuracy. Do not read an elbow angle here as a tracking
  result.

The wire payload is the production contract, byte-identical in shape to what
`wholebody_udp_sender` emits: {"lm": [[x,y,z,vis] x33], "xyz": [...], "src": [...], "seq", "t"}.

    python f19_replay_portrait.py --capture oak_v4_evidence/f18/f18_move_090.jsonl --block natural@090
    python f19_replay_portrait.py --capture ... --list
"""
import argparse
import json
import math
import os
import socket
import sys
import time

NUM = 33

# COCO-17 -> the 33-slot MediaPipe-style wire indices the sidecar fills (same mapping the production
# sender uses via rtmw3d_pose.COCO17_TO_JOINTID).
COCO17_TO_JOINTID = {
    0: 0,                                     # nose
    5: 11, 6: 12,                             # shoulders  (L, R)
    7: 13, 8: 14,                             # elbows
    9: 15, 10: 16,                            # wrists
    11: 23, 12: 24,                           # hips
    13: 25, 14: 26,                           # knees
    15: 27, 16: 28,                           # ankles
    1: 2, 2: 5, 3: 7, 4: 8,                   # eyes / ears
}
UPPER_COCO = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}      # placed on the shoulder depth plane
LOWER_COCO = {11, 12, 13, 14, 15, 16}                # placed on the hip depth plane


def load(path):
    with open(path, encoding="utf-8") as fh:
        meta = json.loads(fh.readline())
        rows = [json.loads(l) for l in fh]
    return meta, rows


def build_payload(meta, r, conf_thr):
    """Back-project one recorded frame into the production camera-space convention (X right, Y down,
    Z forward, metres), then express it hip-relative exactly as the production sender does."""
    fx, fy, cx, cy = meta["intr_portrait"]
    kp = r.get("kp")
    if not kp:
        return None
    zL = r.get("zL")
    zR = r.get("zR")
    hipZ = r.get("hipZ")
    if zL is None or zR is None:
        return None
    zL, zR = zL / 1000.0, zR / 1000.0
    shoulder_plane = 0.5 * (zL + zR)
    if shoulder_plane <= 0.05:
        return None
    hip_plane = hipZ if (hipZ is not None and hipZ > 0.05) else shoulder_plane

    pts = {}
    for ci, (u, v, c) in enumerate(kp):
        if c < conf_thr:
            continue
        jid = COCO17_TO_JOINTID.get(ci)
        if jid is None:
            continue
        if ci == 5:
            z = zL
        elif ci == 6:
            z = zR
        elif ci in LOWER_COCO:
            z = hip_plane
        else:
            z = shoulder_plane
        pts[jid] = ((u - cx) * z / fx, (v - cy) * z / fy, z, float(c))

    if 23 not in pts or 24 not in pts:
        return None
    mid = [(pts[23][i] + pts[24][i]) * 0.5 for i in range(3)]

    lm = [[0.0, 0.0, 0.0, 0.0] for _ in range(NUM)]
    src = [0] * NUM
    for jid, (x, y, z, c) in pts.items():
        lm[jid] = [round(x - mid[0], 5), round(y - mid[1], 5), round(z - mid[2], 5), round(c, 3)]
        # src flag: 1 where the depth is a genuine stereo measurement, 0 where it was assumed
        src[jid] = 1 if jid in (11, 12, 23, 24) else 0
    return {"lm": lm, "xyz": [0.0, 0.0, round(mid[2] * 1000.0, 1)], "src": src}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default=os.path.join("oak_v4_evidence", "f18", "f18_move_090.jsonl"))
    ap.add_argument("--block", default="", help="only this capture block (see --list)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--rate", type=float, default=30.0, help="send Hz")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--loop", type=int, default=1, help="repeat the block N times")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    meta, rows = load(a.capture)
    if a.list:
        seen = {}
        for r in rows:
            b = r.get("block")
            seen[b] = seen.get(b, 0) + 1
        print(f"{a.capture}  rotation={meta.get('rotation')} "
              f"{meta.get('portrait_w')}x{meta.get('portrait_h')}")
        for b, n in seen.items():
            print(f"   {b:22s} {n:5d} frames")
        return 0

    sel = [r for r in rows if (not a.block or r.get("block") == a.block)]
    payloads = []
    skipped = 0
    for r in sel:
        p = build_payload(meta, r, a.conf)
        if p is None:
            skipped += 1
            continue
        p["_block"] = r.get("block")
        p["_capseq"] = r.get("seq")
        payloads.append(p)
    if not payloads:
        print("no usable frames", file=sys.stderr)
        return 1

    measured = sum(1 for v in payloads[0]["src"] if v)
    filled = sum(1 for v in payloads[0]["lm"] if v[3] > 0)
    print(f"replaying {len(payloads)} frames ({skipped} skipped) from {a.capture}"
          f"{' block=' + a.block if a.block else ''} at {a.rate:.0f} Hz -> {a.host}:{a.port}")
    print(f"  {filled}/33 slots filled; {measured} carry MEASURED stereo depth "
          f"(shoulders+hips), the rest sit on a trunk depth plane")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # P1-3's PoseBuffer drops any packet whose seq is below the newest it holds, so seq must be
    # monotonic across process restarts, not restarted at 0.
    seq = int((time.time() - 1788900000.0) * 100.0)
    period = 1.0 / a.rate
    nxt = time.time()
    sent = 0
    marks = []
    for rep in range(a.loop):
        cur_block = None
        for p in payloads:
            if p["_block"] != cur_block:
                cur_block = p["_block"]
                marks.append({"block": cur_block, "t": round(time.time(), 4), "seq": seq})
                print(f"  -> {cur_block}", flush=True)
            msg = {"lm": p["lm"], "xyz": p["xyz"], "src": p["src"],
                   "seq": seq, "t": round(time.time(), 4)}
            sock.sendto(json.dumps(msg).encode("utf-8"), (a.host, a.port))
            seq += 1
            sent += 1
            nxt += period
            d = nxt - time.time()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.time()
    with open(os.path.join("f19_evidence", "replay_marks.json"), "w", encoding="utf-8") as fh:
        json.dump({"capture": a.capture, "block": a.block, "rate": a.rate,
                   "sent": sent, "marks": marks}, fh, indent=2)
    print(f"done: {sent} packets")
    return 0


if __name__ == "__main__":
    os.makedirs("f19_evidence", exist_ok=True)
    sys.exit(main())
