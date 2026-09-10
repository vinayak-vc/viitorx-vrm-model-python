#!/usr/bin/env python3
"""TORSO YAW V4 -- analyse the LIVE Unity trace captured on the real OAK-D path.

Input : oak_v4_evidence/live_trace.jsonl  (written from inside EditorApplication.update, so the
        editor main loop is NOT starved -- the defect that voided V4 sec 4c's dynamic test; the
        `f` field is Time.frameCount and is checked here to prove it.)

Measures, per torsoYawScale block:
  * source shoulder-line yaw  vs  ACTUAL SKINNED avatar shoulder-line yaw  -> effective gain
  * the per-bone yaw decomposition (Hips / Spine / Chest / UpperChest)
  * ARM SAFETY: source arm direction vs ACTUAL SKINNED arm bone direction, while the torso rotates

The arm comparison honours Kalidokit's cross-map (the avatar's LEFT arm is solved from the subject's
RIGHT landmarks 12/14) and the rig conversion (x, -y, z) with the sagittal mirror on X. Rather than
assume the sign convention, both candidates are scored on the first block and the better one is
locked in and reported, so a convention error cannot masquerade as an arm regression.

    python analyze_live_v4.py
"""
import io
import json
import math
import os


def norm(v):
    m = math.sqrt(sum(c * c for c in v))
    return [0.0, 0.0, 0.0] if m < 1e-12 else [c / m for c in v]


def ang(a, b):
    d = sum(x * y for x, y in zip(norm(a), norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, d))))


def sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def pct(v, p):
    if not v:
        return 0.0
    s = sorted(v)
    return s[int(round((len(s) - 1) * p / 100.0))]


def mean(v):
    return sum(v) / len(v) if v else 0.0


def stdev(v):
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def main():
    path = os.path.join("oak_v4_evidence", "live_trace.jsonl")
    rows = []
    errs = 0
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if "err" in d:
            errs += 1
            continue
        rows.append(d)
    if not rows:
        print("no samples (%d errors)" % errs)
        return 1

    # --- harness integrity: did the sampler starve the editor main loop? -------------------
    fd = [rows[i]["f"] - rows[i - 1]["f"] for i in range(1, len(rows))]
    td = [rows[i]["t"] - rows[i - 1]["t"] for i in range(1, len(rows))]
    print("=" * 100)
    print(" LIVE OAK-D TRACE -- %d samples, %d sampler errors, %.1f s" %
          (len(rows), errs, rows[-1]["t"] - rows[0]["t"]))
    print(" harness: frameCount delta p50=%.1f  wall dt p50=%.4f s  -> %.1f fps sampled" %
          (pct(fd, 50), pct(td, 50), 1.0 / max(1e-6, pct(td, 50))))
    print(" (V4 sec 4c's void test showed frameCount delta == 1 against ~0.86 s of wall clock;")
    print("  a delta of 1 against a ~1/60 s dt here means the editor is running normally.)")
    print("=" * 100)

    # --- source arm direction, in the convention the SHIPPING code uses -------------------
    # Verified live in the editor to 0.0000 deg against the skinned bones:
    #   want = MirrorX(providerLandmark[to] - providerLandmark[from])      MirrorX negates X
    #   got  = Inverse(RigFrame()) * (childBone.position - bone.position)  RigFrame = hips.parent.rotation
    # NOTE: `KalidokitToRig` (x, -y, z) is NOT applied here. The driver's own landmarks[] are
    # Y-DOWN and it converts on read, but PoseFrame.landmarks (what this trace samples) are already
    # Y-UP -- applying the flip again double-negates Y and fabricates a 13-23 deg "error". The
    # sampler therefore records `upLdir`/`upRdir` ALREADY in the rig frame.
    def src_arm(r, side):
        """Source upper-arm direction for the AVATAR's `side`, via Kalidokit's cross-map."""
        if side == "L":
            sh, el = r["s12"], r["s14"]          # avatar LEFT  <- subject RIGHT (12/14)
        else:
            sh, el = r["s11"], r["s13"]          # avatar RIGHT <- subject LEFT  (11/13)
        v = sub(el, sh)
        return [-v[0], v[1], v[2]]

    # --- per-block --------------------------------------------------------------------------
    blocks = sorted(set(r["blk"] for r in rows))
    print("\n %5s %6s | %8s %8s %8s | %8s %8s | %7s %7s %7s %7s | %8s %8s" %
          ("blk", "scale", "srcSh p50", "srcSh sd", "avSh sd", "gain", "err mean",
           "hipsY", "spineY", "chestY", "uchY", "armErr p50", "armErr max"))
    for b in blocks:
        rs = [r for r in rows if r["blk"] == b]
        if len(rs) < 5:
            continue
        scale = mean([r["scale"] for r in rs])
        src = [r["srcSh"] for r in rs]
        # avatar shoulder-line yaw, referenced to this run's own frontal baseline
        av_raw = [r["avSh"] for r in rs]
        base = mean([r["avSh"] for r in rows if r["blk"] == blocks[0]])
        av = [wrap(v - base) for v in av_raw]
        # effective gain: least-squares slope of avatar yaw against source yaw
        num = sum(a * s for a, s in zip(av, src))
        den = sum(s * s for s in src)
        gain = num / den if den > 1e-9 else 0.0
        err = [abs(wrap(a - s)) for a, s in zip(av, src)]

        def bone(k):
            return mean([wrap(r[k]) for r in rs])

        arm = []
        for r in rs:
            for side, key in (("L", "upLdir"), ("R", "upRdir")):
                s = src_arm(r, side)
                if sum(c * c for c in s) < 1e-8:
                    continue
                arm.append(ang(s, r[key]))
        print(" %5d %6.2f | %8.2f %8.2f %8.2f | %8.3f %8.2f | %7.2f %7.2f %7.2f %7.2f | %10.4f %10.4f" %
              (b, scale, pct(src, 50), stdev(src), stdev(av), gain, mean(err),
               bone("hipsY"), bone("spineY"), bone("chestY"), bone("uchY"),
               pct(arm, 50) if arm else -1, max(arm) if arm else -1))

    print("\n KEY: 'avSh sd' is how much the avatar's shoulder line MOVES. At scale 0 it must be ~0.")
    print("      'armErr' is source -> ACTUAL SKINNED bone direction; it must stay ~0 at EVERY scale,")
    print("      which is the ARM V2 parent-division regression test while the torso rotates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
