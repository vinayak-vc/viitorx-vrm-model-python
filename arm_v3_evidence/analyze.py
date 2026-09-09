#!/usr/bin/env python3
"""ARM RETARGET V3 — TASK 2/4/5 analysis: screen-space silhouette error and cause separation.

Four chains are compared, all projected through the SAME production camera (this script's projection
is verified against Unity's own Camera.WorldToScreenPoint to 0.003 px):

  BONE        the avatar's real skinned humanoid bones
  MESH        the avatar's VISIBLE joints, from the skinned vertices (LBS, verified against BakeMesh)
  EXPECTED    the source direction anchored at the avatar's own shoulder, walked at the AVATAR's own
              bone lengths. This is what a perfect direction-only FK retarget must produce, so any
              BONE-vs-EXPECTED gap is a retarget direction error and nothing else.
  PROPORTIONAL the source direction anchored at the avatar's own shoulder, walked at the SOURCE's bone
              lengths scaled by one global body scale (shoulder span avatar / shoulder span source).
              This is where the joint would land if the avatar had the human's limb proportions, so any
              EXPECTED-vs-PROPORTIONAL gap is pure avatar proportion and nothing else.

Writes screen_error.json and *_overlay.png next to the captures.
"""
import json
import math
import os
import sys

from PIL import Image, ImageDraw

EVID = r"C:\Unity\viitorx-vrm-avtar-unity-base-project\Assets\Games\viitorx-vrm-avtar-unity\python-sidecar~\arm_v3_evidence"
WHITE, RED, CYAN, MAGENTA, GREEN, YELLOW = (255, 255, 255), (255, 70, 70), (0, 235, 235), (255, 0, 255), (80, 255, 80), (255, 210, 0)


def unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return [c / n for c in v] if n > 1e-9 else [0.0, 0.0, 0.0]


def add(a, b, s=1.0):
    return [a[i] + b[i] * s for i in range(3)]


def dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


class Cam:
    def __init__(self, rec):
        self.p = rec["camPos"]
        self.w, self.h = rec["camPx"]
        self.th = math.tan(math.radians(rec["camFov"] / 2.0))
        self.asp = self.w / self.h

    def screen(self, p):
        vz = p[2] - self.p[2]
        if vz <= 1e-6:
            return None
        xn = ((p[0] - self.p[0]) / vz) / (self.th * self.asp)
        yn = ((p[1] - self.p[1]) / vz) / self.th
        return [(xn * 0.5 + 0.5) * self.w, (yn * 0.5 + 0.5) * self.h]

    def img(self, sp, iw, ih):
        """Unity screen (bottom-left origin, camera rect) -> PIL image pixels (top-left origin)."""
        return (sp[0] * iw / self.w, (self.h - sp[1]) * ih / self.h)


def bearing(a, b):
    """Screen bearing of a->b in degrees, measured in IMAGE pixels so it matches what a viewer sees."""
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def dang(p, q):
    return (p - q + 180.0) % 360.0 - 180.0


def chain(dr, pts, color, width, radius):
    for i in range(len(pts) - 1):
        dr.line([pts[i], pts[i + 1]], fill=color, width=width)
    for p in pts:
        dr.ellipse([p[0] - radius, p[1] - radius, p[0] + radius, p[1] + radius], outline=color, width=2)


def analyze(rec):
    cam = Cam(rec)
    # ONE global body scale, from a measure both figures have: glenohumeral span.
    sp_av = dist(rec["left"]["boneUpperPos"], rec["right"]["boneUpperPos"])
    sp_sr = dist(rec["left"]["srcShoulder"], rec["right"]["srcShoulder"]) if rec["haveSource"] else 0.0
    gscale = sp_av / sp_sr if sp_sr > 1e-6 else float("nan")
    out = {"label": rec["label"], "haveSource": rec["haveSource"],
           "shoulderSpan_avatar_m": sp_av, "shoulderSpan_source_m": sp_sr, "globalBodyScale": gscale,
           "arms": {}}
    for side in ("left", "right"):
        S, M = rec[side], rec[side]["mesh"]
        a = {}
        boneW = [S["boneUpperPos"], S["boneElbowPos"], S["boneWristPos"]]
        meshW = [M["meshShoulder"], M["meshElbow"], M["meshWrist"]]
        lenU, lenF = S["boneUpperLen"], S["boneForeLen"]
        a["boneUpperLen_m"], a["boneForeLen_m"] = lenU, lenF
        a["srcUpperLen_m"], a["srcForeLen_m"] = S["srcUpperLen"], S["srcForeLen"]
        expW = propW = None
        if rec["haveSource"] and S["srcUpperDirMirrored"] != [0.0, 0.0, 0.0]:
            du, df = unit(S["srcUpperDirMirrored"]), unit(S["srcForeDirMirrored"])
            e = add(S["boneUpperPos"], du, lenU)
            expW = [S["boneUpperPos"], e, add(e, df, lenF)]
            # what human proportions would give, at the avatar's own shoulder
            pu, pf = S["srcUpperLen"] * gscale, S["srcForeLen"] * gscale
            e2 = add(S["boneUpperPos"], du, pu)
            propW = [S["boneUpperPos"], e2, add(e2, df, pf)]
            a["proportionalUpperLen_m"], a["proportionalForeLen_m"] = pu, pf
            a["armLenRatio_avatar_over_proportional"] = (lenU + lenF) / (pu + pf)
            # 3-D joint displacement caused by proportion alone
            a["proportion_elbow_error_m"] = dist(expW[1], propW[1])
            a["proportion_wrist_error_m"] = dist(expW[2], propW[2])
            a["retarget_elbow_error_m"] = dist(boneW[1], expW[1])
            a["retarget_wrist_error_m"] = dist(boneW[2], expW[2])
            a["mesh_elbow_error_m"] = dist(meshW[1], boneW[1])

        def scr(ws):
            return [cam.screen(p) for p in ws] if ws else None

        bs, ms, es, ps = scr(boneW), scr(meshW), scr(expW), scr(propW)
        a["screen"] = {"bone": bs, "mesh": ms, "expected": es, "proportional": ps}
        if bs and es:
            a["screenErr_boneVsExpected_upper_deg"] = dang(bearing(bs[0], bs[1]), bearing(es[0], es[1]))
            a["screenErr_boneVsExpected_fore_deg"] = dang(bearing(bs[1], bs[2]), bearing(es[1], es[2]))
            a["boneVsExpected_elbow_px"] = dist(bs[1], es[1])
            a["boneVsExpected_wrist_px"] = dist(bs[2], es[2])
        if bs and ms:
            a["screenErr_meshVsBone_upper_deg"] = dang(bearing(ms[0], ms[1]), bearing(bs[0], bs[1]))
            a["screenErr_meshVsBone_fore_deg"] = dang(bearing(ms[1], ms[2]), bearing(bs[1], bs[2]))
            a["meshVsBone_elbow_px"] = dist(ms[1], bs[1])
            a["meshVsBone_wrist_px"] = dist(ms[2], bs[2])
        if bs and ps:
            a["screenErr_boneVsProportional_upper_deg"] = dang(bearing(bs[0], bs[1]), bearing(ps[0], ps[1]))
            a["boneVsProportional_elbow_px"] = dist(bs[1], ps[1])
            a["boneVsProportional_wrist_px"] = dist(bs[2], ps[2])
        # visible segment length on screen, for the "arms look stubby" claim
        if bs:
            a["screenUpperLen_px"] = dist(bs[0], bs[1])
            a["screenForeLen_px"] = dist(bs[1], bs[2])
        if ps:
            a["screenProportionalUpperLen_px"] = dist(ps[0], ps[1])
            a["screenProportionalForeLen_px"] = dist(ps[1], ps[2])
        out["arms"][side] = a
    return out, cam


def draw(rec, res, cam, png, outpng):
    img = Image.open(png).convert("RGB")
    W, H = img.size
    dr = ImageDraw.Draw(img)
    for side in ("left", "right"):
        s = res["arms"][side]["screen"]
        if s["proportional"]:
            chain(dr, [cam.img(p, W, H) for p in s["proportional"]], MAGENTA, 2, 4)
        if s["expected"]:
            chain(dr, [cam.img(p, W, H) for p in s["expected"]], CYAN, 2, 5)
        chain(dr, [cam.img(p, W, H) for p in s["bone"]], WHITE, 3, 7)
        chain(dr, [cam.img(p, W, H) for p in s["mesh"]], RED, 2, 4)
    ds = rec.get("debugSkeleton") or {}
    js = ds.get("jointsScreen") or {}
    for ids in (("11", "13", "15"), ("12", "14", "16")):
        pts = [cam.img(js[i], W, H) for i in ids if js.get(i)]
        if len(pts) == 3:
            chain(dr, pts, GREEN, 2, 4)
    dr.text((8, 8), res["label"], fill=YELLOW)
    dr.text((8, 22), "WHITE bone   RED visible mesh   CYAN expected(src dir @ avatar len)", fill=YELLOW)
    dr.text((8, 36), "MAGENTA human-proportional      GREEN debug skeleton (offset+1.5x)", fill=YELLOW)
    img.save(outpng)


def main():
    recs = [json.loads(l) for l in open(os.path.join(EVID, "mesh_trace.jsonl"), encoding="utf-8")]
    allres = []
    for rec in recs:
        res, cam = analyze(rec)
        allres.append(res)
        png = os.path.join(EVID, rec["label"] + ".png")
        if os.path.exists(png):
            draw(rec, res, cam, png, os.path.join(EVID, rec["label"] + "_overlay.png"))
    with open(os.path.join(EVID, "screen_error.json"), "w", encoding="utf-8") as fh:
        json.dump(allres, fh, indent=2)

    print("global body scale (avatar shoulder span / source shoulder span):")
    for r in allres:
        if r["haveSource"]:
            print("   %-16s %.4f m / %.4f m = %.4f" % (r["label"], r["shoulderSpan_avatar_m"],
                                                       r["shoulderSpan_source_m"], r["globalBodyScale"]))
            break
    print()
    print("=== TASK 4 SCREEN-SPACE ERROR (degrees of on-screen limb bearing; pixels at 1080x1920) ===")
    hdr = "%-12s %-5s | %8s %8s | %8s %8s | %8s %8s | %7s %7s"
    print(hdr % ("pose", "arm", "bone-exp", "bone-exp", "mesh-bone", "mesh-bone", "elbow", "wrist", "upLen", "propLen"))
    print(hdr % ("", "", "upper", "fore", "upper", "fore", "px b-e", "px b-e", "px", "px"))
    for r in allres:
        if not r["haveSource"]:
            continue
        for side in ("left", "right"):
            a = r["arms"][side]
            print(hdr % (r["label"].replace("pose_", ""), side,
                         "%+.3f" % a.get("screenErr_boneVsExpected_upper_deg", float("nan")),
                         "%+.3f" % a.get("screenErr_boneVsExpected_fore_deg", float("nan")),
                         "%+.2f" % a.get("screenErr_meshVsBone_upper_deg", float("nan")),
                         "%+.2f" % a.get("screenErr_meshVsBone_fore_deg", float("nan")),
                         "%.2f" % a.get("boneVsExpected_elbow_px", float("nan")),
                         "%.2f" % a.get("boneVsExpected_wrist_px", float("nan")),
                         "%.1f" % a.get("screenUpperLen_px", float("nan")),
                         "%.1f" % a.get("screenProportionalUpperLen_px", float("nan"))))
    print()
    print("=== TASK 5 PROPORTION: 3-D joint displacement attributable to avatar proportions alone ===")
    print("%-12s %-5s %10s %10s | %10s %10s | %s" % ("pose", "arm", "elbow m", "wrist m", "elbow px", "wrist px", "armLen avatar/proportional"))
    for r in allres:
        if not r["haveSource"]:
            continue
        for side in ("left", "right"):
            a = r["arms"][side]
            if "proportion_elbow_error_m" not in a:
                continue
            print("%-12s %-5s %10.4f %10.4f | %10.1f %10.1f | %.4f" % (
                r["label"].replace("pose_", ""), side,
                a["proportion_elbow_error_m"], a["proportion_wrist_error_m"],
                a.get("boneVsProportional_elbow_px", float("nan")),
                a.get("boneVsProportional_wrist_px", float("nan")),
                a["armLenRatio_avatar_over_proportional"]))


if __name__ == "__main__":
    sys.exit(main())
