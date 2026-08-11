#!/usr/bin/env python3
"""
Compare the 3-stage pipeline logs to localize instability (jitter / wrist-spin / facing), with per-AXIS
depth analysis, distance/coverage, gimbal-free model metrics, spike locations, and an auto-verdict.

  1. sender_log.jsonl  -> Python sidecar SENDS   (wholebody_udp_sender.py --log-dir <dir>)   [or run_capture.bat]
  2. recv_log.jsonl    -> Unity RECEIVES+converts (AppBootstrap.pipelineLogging = true)
  3. model_log.jsonl   -> avatar bone RESULT      (same toggle)

Key idea: the 2D Python preview shows only the image plane (x,y). What Unity applies also includes DEPTH
(z), which the preview can't show. Splitting jitter into x / y / z reveals that limb DEPTH is the noisy
axis. Model orientation is logged as a gimbal-free FORWARD vector (euler-deltas explode near gimbal lock).

Usage:  python compare_logs.py [--dir pipeline_logs]
"""

import argparse
import json
import math
import os


def load(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def stats(vals):
    if not vals:
        return (0.0, 0.0, 0.0, 0)
    s = sorted(vals)
    n = len(s)
    return (sum(s) / n, s[min(n - 1, int(0.95 * n))], s[-1], n)


def ang_wrap(d):
    return abs((d + 180.0) % 360.0 - 180.0)


def quat_angle(q1, q2):
    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


def vec_angle(a, b):
    na = math.sqrt(sum(v * v for v in a))
    nb = math.sqrt(sum(v * v for v in b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b)) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def series(rows, key, idx=None):
    out = []
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        out.append(v[idx] if idx is not None else v)
    return out


def axis_jit(rows, key, idx):
    s = series(rows, key, idx)
    dxs, dys, dzs = [], [], []
    for i in range(1, len(s)):
        if len(s[i]) < 3 or len(s[i - 1]) < 3:
            continue
        dxs.append(abs(s[i][0] - s[i - 1][0]))
        dys.append(abs(s[i][1] - s[i - 1][1]))
        dzs.append(abs(s[i][2] - s[i - 1][2]))
    mx = sum(dxs) / len(dxs) if dxs else 0.0
    my = sum(dys) / len(dys) if dys else 0.0
    mz = sum(dzs) / len(dzs) if dzs else 0.0
    return mx, my, mz, len(dxs)


def worst_axis_spike(rows, key, idx, axis):
    s = series(rows, key, idx)
    seqs = [r.get("seq", -1) for r in rows if r.get(key) is not None]
    best, at = 0.0, -1
    for i in range(1, len(s)):
        if len(s[i]) < 3 or len(s[i - 1]) < 3:
            continue
        d = abs(s[i][axis] - s[i - 1][axis])
        if d > best:
            best, at = d, (seqs[i] if i < len(seqs) else -1)
    return best, at


def quat_jit(rows, key):
    s = series(rows, key)
    return stats([quat_angle(s[i], s[i - 1]) for i in range(1, len(s)) if len(s[i]) == 4 and len(s[i - 1]) == 4])


def vec_jit(rows, key):
    s = series(rows, key)
    return stats([vec_angle(s[i], s[i - 1]) for i in range(1, len(s)) if len(s[i]) == 3 and len(s[i - 1]) == 3])


def scalar_jit(rows, key):
    s = series(rows, key)
    return stats([ang_wrap(s[i] - s[i - 1]) for i in range(1, len(s))])


def fps(rows):
    ts = [r.get("t") for r in rows if r.get("t") is not None]
    if len(ts) < 2:
        return 0.0
    span = ts[-1] - ts[0]
    return (len(ts) - 1) / span if span > 1e-6 else 0.0


def axis_row(label, mx, my, mz, n):
    plane = (mx + my) / 2.0
    ratio = mz / plane if plane > 1e-6 else 0.0
    return "  %-12s |dx|=%.4f  |dy|=%.4f  |dz|=%.4f m/frame   depth=%.1fx in-plane  (n=%d)" % (label, mx, my, mz, ratio, n)


def st_row(label, unit, st):
    m, p95, mx, n = st
    return "  %-16s mean=%7.3f  p95=%7.3f  max=%7.3f %-4s (n=%d)" % (label, m, p95, mx, unit, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="pipeline_logs")
    args = ap.parse_args()
    send = load(os.path.join(args.dir, "sender_log.jsonl"))
    recv = load(os.path.join(args.dir, "recv_log.jsonl"))
    model = load(os.path.join(args.dir, "model_log.jsonl"))

    print("Loaded: sender=%d  recv=%d  model=%d frames  (dir=%s)" % (len(send), len(recv), len(model), args.dir))
    sset = set(r.get("seq") for r in send if "seq" in r)
    rset = set(r.get("seq") for r in recv if "seq" in r)
    if sset and rset:
        arr = len(sset & rset)
        print("Wire  : %d/%d sent seqs arrived in recv (%.1f%%; the rest is UDP loss, not jitter)" % (arr, len(sset), 100.0 * arr / max(1, len(sset))))
    print("Rate  : sender ~%.0f fps | recv ~%.0f fps | model(Unity) %d frames" % (fps(send), fps(recv), len(model)))

    # distance + coverage from the sender log
    hz = [r["hipZ"] for r in send if "hipZ" in r]
    if hz:
        avg = sum(hz) / len(hz)
        flag = "  <-- TOO FAR (aim ~2 m; depth noise ~ scales with distance)" if avg > 2.5 else "  (good range)"
        print("Dist  : hip avg=%.2f m  (min=%.2f max=%.2f)%s" % (avg, min(hz), max(hz), flag))
    cov = [r["cov"] for r in send if "cov" in r]
    if cov:
        avg = sum(cov) / len(cov)
        print("Cover : avg %.1f/33 keypoints measured (%.0f%% depth-holes filled by fallback/hold)" % (avg, 100.0 * (33 - avg) / 33.0))

    print("\n== STAGE 1  SEND (sidecar) - per-axis motion; z = DEPTH (invisible in the 2D preview) ==")
    for name, key in [("R shoulder", "sh"), ("R elbow", "el"), ("R wrist", "wr")]:
        print(axis_row(name, *axis_jit(send, key, 1)))
    ws, wseq = worst_axis_spike(send, "wr", 1, 2)
    print("  worst R-wrist DEPTH spike: |dz|=%.2f m at seq=%s" % (ws, wseq))

    print("\n== STAGE 2  RECV (Unity, converted) - positions should ~match send; palm = wrist orientation ==")
    for name, key in [("R shoulder", "sh"), ("R elbow", "el"), ("R wrist", "wr")]:
        print(axis_row(name, *axis_jit(recv, key, 1)))
    print(st_row("L palm quat", "deg", quat_jit(recv, "lpalm")))
    print(st_row("R palm quat", "deg", quat_jit(recv, "rpalm")))

    print("\n== STAGE 3  MODEL (applied bones) - forward-vector angle is gimbal-free (trust it over euler) ==")
    print(st_row("hips yaw", "deg", scalar_jit(model, "hipsY")))
    if any("rhandF" in r for r in model):
        print(st_row("L hand dir", "deg", vec_jit(model, "lhandF")))
        print(st_row("R hand dir", "deg", vec_jit(model, "rhandF")))
        print(st_row("L forearm dir", "deg", vec_jit(model, "llowF")))
        print(st_row("R forearm dir", "deg", vec_jit(model, "rlowF")))
    else:
        print("  (no forward-vector fields yet - re-capture after the Unity rebuild; showing euler, gimbal-prone:)")
        print(st_row("R hand euler", "deg", stats([math.sqrt(sum(ang_wrap(a[k] - b[k]) ** 2 for k in range(3))) for a, b in zip(series(model, "rhand")[1:], series(model, "rhand")) if len(a) == 3 and len(b) == 3])))

    print("\n== VERDICT ==")
    smx, smy, smz, _ = axis_jit(send, "wr", 1)
    plane = (smx + smy) / 2.0
    if plane > 1e-6 and smz > 2.0 * plane:
        print("  * Wrist DEPTH (%.3f m/frame) is %.1fx the image-plane (%.3f) -> source depth noise; NOT a Unity bug." % (smz, smz / plane, plane))
        print("    -> stand ~2 m; --depth-min-cutoff lower; wrist stays off for POC (ADR-024).")
    pj = quat_jit(recv, "rpalm")
    if pj[0] > 5.0:
        print("  * Palm quat jitter %.1f deg/frame (spikes %.0f) -> noisy hand landmarks; the wrist follows this." % (pj[0], pj[2]))
    hj = scalar_jit(model, "hipsY")
    print("  * Facing (hips-yaw) jitter %.2f deg/frame, max %.0f -> %s" % (hj[0], hj[2], "STABLE/frontal-locked" if hj[0] < 1.0 else "check flatten/hip-yaw"))
    if hz and sum(hz) / len(hz) > 2.5:
        print("  * Biggest free win: you are at %.2f m avg - get to ~2 m." % (sum(hz) / len(hz)))


if __name__ == "__main__":
    main()
