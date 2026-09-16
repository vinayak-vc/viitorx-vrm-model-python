#!/usr/bin/env python3
"""F-20A - summarise a phase-labelled bone capture: tracking state, rest convergence, snaps.

Writes a small text summary beside the capture so the report's numbers stay reproducible after the
large per-frame telemetry is pruned.

    python tools/deployment/f20a_analyze.py evidence/oak_v4/f20a/bones_f20a.jsonl
"""
import collections, io, json, math, os, sys

BONES = ["Spine", "Chest", "LeftUpperArm", "RightUpperArm", "LeftLowerArm", "RightLowerArm"]
MOTION = ("Hips", "LeftUpperArm", "RightUpperArm", "Head")


def rest_angle(r):
    v = [abs(r["b"][b]["l"][3]) for b in BONES if b in r["b"]]
    if not v:
        return None
    return sum(math.degrees(2 * math.acos(max(-1.0, min(1.0, x)))) for x in v) / len(v)


def yaw(r):
    B = r["b"]
    if "LeftUpperArm" not in B or "RightUpperArm" not in B:
        return None
    d = [B["LeftUpperArm"]["p"][i] - B["RightUpperArm"]["p"][i] for i in (0, 1, 2)]
    n = math.hypot(d[0], d[2])
    return math.degrees(math.atan2(d[2] / n, d[0] / n)) if n > 1e-9 else None


def motion(rs):
    out = []
    for i in range(1, len(rs)):
        s = 0.0
        n = 0
        for b in MOTION:
            A = rs[i - 1]["b"].get(b)
            B = rs[i]["b"].get(b)
            if not A or not B:
                continue
            d = abs(sum(A["w"][k] * B["w"][k] for k in range(4)))
            s += math.degrees(2 * math.acos(max(-1.0, min(1.0, d))))
            n += 1
        if n:
            out.append(s / n)
    return sum(out) / len(out) if out else 0.0


def main():
    path = sys.argv[1]
    rows = [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]
    order, g = [], {}
    for r in rows:
        b = r.get("block", "")
        if not b:
            continue
        if b not in g:
            g[b] = []
            order.append(b)
        g[b].append(r)
    L = []
    L.append("F-20A phase summary: %s  (%d records)" % (os.path.basename(path), len(rows)))
    L.append("")
    L.append("%-18s %7s %-34s %9s %9s %9s %7s" %
             ("phase", "n", "tracking states", "moved/f", "restEnd", "maxStep", "snaps"))
    for b in order:
        rs = g[b]
        st = collections.Counter(r.get("track", "?") for r in rs)
        ys = [y for y in (yaw(r) for r in rs) if y is not None]
        steps = [abs((ys[i] - ys[i - 1] + 180) % 360 - 180) for i in range(1, len(ys))]
        rr = [x for x in (rest_angle(r) for r in rs) if x is not None]
        L.append("%-18s %7d %-34s %9.4f %9.2f %9.3f %7d" %
                 (b[:18], len(rs), ",".join("%s:%d" % (k[:11], v) for k, v in st.most_common(2)),
                  motion(rs), rr[-1] if rr else -1.0,
                  max(steps) if steps else 0.0, sum(1 for s in steps if s > 10)))
    L.append("")
    L.append("moved/f  = mean per-frame world rotation of hips+arms+head (deg); 0 = frozen")
    L.append("restEnd  = mean bone LOCAL angle from the normalized rig rest pose at the phase end;")
    L.append("           0.00 = the avatar reached the neutral failsafe pose exactly")
    L.append("snaps    = rendered torso-yaw steps > 10 deg")
    out = path.replace(".jsonl", "_phases.txt")
    io.open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
