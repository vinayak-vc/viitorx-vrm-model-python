#!/usr/bin/env python3
"""F-11 -- characterise the candidate 2-D face sign signals.

Runs on `f11_video_face.jsonl` (offline RTMW3D over video.webm). There is NO ground-truth heading in a
video, so this CANNOT score sign accuracy -- that needs a capture at known headings. What it can do,
and what F-11 sec 8 explicitly asks for, is establish whether the face signal is measuring yaw at all
or tracking a confounder, and whether the candidates agree with an INDEPENDENT 2-D sign source.

Yaw magnitude, scale-invariant (the subject's distance changes through the clip, so a raw pixel span
conflates distance with yaw):

    spanRatio = shoulderPixelSpan / torsoPixelHeight      (constant for a given person, face-on)
    cos(t)    = spanRatio / spanRatio_faceOn              (calibrated as the p98)

Independent sign cross-check -- EAR VISIBILITY ASYMMETRY. Turning right hides the right ear and
exposes the left, so conf(L-ear) - conf(R-ear) carries the sign through a completely different
mechanism (model confidence) than the nose offset (pixel geometry). Agreement between the two is
mutual corroboration; it is NOT proof, because both could share a bias.

    python f11_face_analyze.py
"""
import io
import json
import math
import os

FEAT = os.path.join("oak_v4_evidence", "f11_video_face.jsonl")
CONF_MIN = 0.3


def pct(v, p):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[int(round((len(s) - 1) * p / 100.0))]


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def stdev(v):
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def pearson(a, b):
    if len(a) < 3:
        return float("nan")
    ma, mb = mean(a), mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da > 0 and db > 0 else float("nan")


def main():
    rows = [json.loads(l) for l in io.open(FEAT, encoding="utf-8") if l.strip()]
    good = []
    for r in rows:
        lsh, rsh = r["Lsh"], r["Rsh"]
        lh, rh = r["Lhip"], r["Rhip"]
        nose = r["nose"]
        if min(lsh[2], rsh[2], lh[2], rh[2]) < CONF_MIN:
            continue
        shMidX = 0.5 * (lsh[0] + rsh[0])
        shMidY = 0.5 * (lsh[1] + rsh[1])
        hipMidY = 0.5 * (lh[1] + rh[1])
        span = abs(lsh[0] - rsh[0])
        torso = abs(hipMidY - shMidY)
        if torso < 1e-3:
            continue
        e = {
            "frame": r["frame"], "noseConf": nose[2],
            "span": span, "torso": torso, "ratio": span / torso,
            "A": nose[0] - shMidX,
            "B": (nose[0] - shMidX) / max(span, 1e-6),
            "C1": nose[0] - lsh[0], "C2": nose[0] - rsh[0],
            "LearC": r["Lear"][2], "RearC": r["Rear"][2],
            "earAsym": r["Lear"][2] - r["Rear"][2],
        }
        # D: face centre from the eyes, when both are present
        if min(r["Leye"][2], r["Reye"][2]) >= CONF_MIN:
            e["D"] = 0.5 * (r["Leye"][0] + r["Reye"][0]) - shMidX
            e["Dn"] = e["D"] / max(span, 1e-6)
        good.append(e)

    print("=" * 104)
    print(" F-11 -- candidate 2-D face sign signals, video.webm (%d of %d frames usable)"
          % (len(good), len(rows)))
    print(" NO GROUND TRUTH in a video: this tests PHYSICALITY and CONFOUNDERS, not sign accuracy.")
    print("=" * 104)

    # ---- availability ------------------------------------------------------------------
    nose_ok = sum(1 for e in good if e["noseConf"] >= CONF_MIN)
    print("\n AVAILABILITY")
    print("   nose conf >= %.1f      : %d / %d (%.1f%%)   conf p05=%.3f p50=%.3f"
          % (CONF_MIN, nose_ok, len(good), 100.0 * nose_ok / len(good),
             pct([e["noseConf"] for e in good], 5), pct([e["noseConf"] for e in good], 50)))
    d_ok = sum(1 for e in good if "D" in e)
    print("   both eyes usable (D)  : %d / %d (%.1f%%)" % (d_ok, len(good), 100.0 * d_ok / len(good)))
    print("   ear conf L p50=%.3f  R p50=%.3f"
          % (pct([e["LearC"] for e in good], 50), pct([e["RearC"] for e in good], 50)))

    # ---- yaw magnitude, scale invariant -------------------------------------------------
    rmax = pct([e["ratio"] for e in good], 98)
    for e in good:
        c = max(0.0, min(1.0, e["ratio"] / rmax))
        e["yaw2D"] = math.degrees(math.acos(c))
    y = [e["yaw2D"] for e in good]
    print("\n YAW MAGNITUDE in this clip (scale-invariant, faceOn ratio = %.3f)" % rmax)
    print("   |yaw2D| p50=%.1f p90=%.1f p95=%.1f max=%.1f deg" % (pct(y, 50), pct(y, 90), pct(y, 95), max(y)))
    print("   NOTE: this clip barely turns. Nothing here speaks to the +-90 deg behaviour that")
    print("   F-10 showed is where the depth estimator collapses.")

    # ---- sec 8: is the signal measuring yaw, or a confounder? ---------------------------
    print("\n IS THE SIGNAL MEASURING YAW?  (F-11 sec 8) -- correlation of |candidate| with |yaw2D|")
    print(" %-6s %-34s %9s %9s" % ("cand", "definition", "r vs yaw", "r vs span"))
    for key, desc in (("A", "noseX - shoulderMidX  (px)"),
                      ("B", "A / shoulderSpanPx  (normalised)"),
                      ("Dn", "faceCentreX - shMidX, normalised")):
        sel = [e for e in good if key in e]
        if len(sel) < 20:
            print(" %-6s %-34s   (too few frames)" % (key, desc))
            continue
        r1 = pearson([abs(e[key]) for e in sel], [e["yaw2D"] for e in sel])
        r2 = pearson([abs(e[key]) for e in sel], [e["span"] for e in sel])
        print(" %-6s %-34s %9.3f %9.3f" % (key, desc, r1, r2))
    print("\n   'r vs span' is the confounder check: a signal that tracks the shoulder span rather")
    print("   than the yaw would be measuring how far away / how wide the subject is, not heading.")

    # ---- candidate agreement with the INDEPENDENT ear-asymmetry sign --------------------
    print("\n AGREEMENT WITH AN INDEPENDENT 2-D SIGN SOURCE (ear visibility asymmetry)")
    strong = [e for e in good if abs(e["earAsym"]) > 0.10 and e["yaw2D"] > 8.0]
    print("   frames where the ear asymmetry is decisive (|dConf|>0.10) and yaw2D>8 deg: %d" % len(strong))
    if len(strong) >= 20:
        for key in ("A", "B", "Dn"):
            sel = [e for e in strong if key in e]
            if len(sel) < 20:
                continue
            agree = sum(1 for e in sel if (e[key] > 0) == (e["earAsym"] > 0))
            print("   %-4s agrees with ear asymmetry on %4d / %4d frames (%5.1f%%)"
                  % (key, agree, len(sel), 100.0 * agree / len(sel)))
        print("   (50%% = no relationship. This is corroboration between two 2-D signals, NOT truth.)")

    # ---- stability ----------------------------------------------------------------------
    print("\n STABILITY of the sign, frame to frame")
    for key in ("A", "B", "Dn"):
        sel = [e for e in good if key in e]
        if len(sel) < 20:
            continue
        flips = sum(1 for i in range(1, len(sel)) if sel[i][key] * sel[i - 1][key] < 0)
        near = sum(1 for e in sel if abs(e[key]) < (0.02 if key != "A" else 1.0))
        print("   %-4s sign flips %4d / %4d frames (%5.2f%%)   |value| near zero on %d frames"
              % (key, flips, len(sel) - 1, 100.0 * flips / max(1, len(sel) - 1), near))

    print("\n %-6s %10s %10s %10s %10s" % ("cand", "p05", "p50", "p95", "sd"))
    for key in ("A", "B", "C1", "C2", "Dn", "earAsym"):
        sel = [e[key] for e in good if key in e]
        if len(sel) < 20:
            continue
        print(" %-6s %10.4f %10.4f %10.4f %10.4f" % (key, pct(sel, 5), pct(sel, 50), pct(sel, 95), stdev(sel)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
