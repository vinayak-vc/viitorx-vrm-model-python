#!/usr/bin/env python3
"""Record the Unity avatar while a video drives it, and build a SIDE-BY-SIDE comparison.

Produces exactly one artefact: source video on the left, the rendered VRM avatar on the right, on a
common clock. That is the only honest way to answer "is the avatar doing the same thing" - a bone
log answers a different, narrower question (what the retarget emitted), and cannot show a viewer
that an arm went the wrong way.

WHAT THIS DOES NOT DEMONSTRATE, and the caption says so on every frame: there is NO STEREO DEPTH in
this path. See f23_video_to_unity.py's header. The avatar is driven from RTMW3D's monocular
root-relative z, so this exercises the retarget and the avatar, not the torso-yaw signal that
F-16/F-17/F-18 measured as the actual blocker.

Unity must already be in PLAY MODE with the Bootstrap scene (oakUdpPort 8899) before this runs;
this script cannot press Play.

    .venv\\Scripts\\python.exe f23_dance_demo.py --video ..\\..\\video\\video.webm
"""

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(_os.path.join(_d, "_sidecar_path.py")):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import _sidecar_path  # noqa: F401  - puts the sidecar root and every tools/ group on sys.path
import evidence_paths as EV

import argparse
import ctypes
import ctypes.wintypes as wt
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, ".venv", "Scripts", "python.exe")
OUT = EV.oak_v4("f23")
UNITY_TITLE_HINT = "Windows, Mac, Linux"


def unity_rect():
    """Screen rect of the Unity Editor window, clamped to the visible desktop.

    Captured by REGION rather than by `gdigrab -i title=`: the editor's title contains spaces,
    dashes and the project path, and quoting that through a shell is a reliable way to record
    nothing at all."""
    u = ctypes.windll.user32
    u.SetProcessDPIAware()
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
    def cb(h, _):
        if not u.IsWindowVisible(h):
            return True
        n = u.GetWindowTextLengthW(h)
        if not n:
            return True
        b = ctypes.create_unicode_buffer(n + 1)
        u.GetWindowTextW(h, b, n + 1)
        if UNITY_TITLE_HINT in b.value:
            r = wt.RECT()
            u.GetWindowRect(h, ctypes.byref(r))
            found.append((b.value, r.left, r.top, r.right - r.left, r.bottom - r.top))
        return True

    u.EnumWindows(cb, 0)
    if not found:
        return None
    title, x, y, w, h = max(found, key=lambda f: f[3] * f[4])
    sw, sh = u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    x, y = max(0, x), max(0, y)
    w, h = min(w, sw - x) & ~1, min(h, sh - y) & ~1      # even dims: yuv420p needs them
    return title, x, y, w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--settle", type=float, default=2.0,
                    help="seconds to record before and after the drive, so the clip does not start "
                         "mid-pose or clip the last move")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    cap_path = os.path.join(OUT, "unity_%s.mkv" % stamp)
    side_path = os.path.join(OUT, "side_by_side_%s.mp4" % stamp)

    rect = unity_rect()
    if rect is None:
        print("Unity editor window not found. Is the editor open?")
        return 2
    title, x, y, w, h = rect
    print("unity window : %s" % title[:70])
    print("capturing    : %dx%d at (%d,%d)" % (w, h, x, y))

    # Keep Unity ON TOP for the whole capture. gdigrab records the DESKTOP, so anything that
    # covers the editor is what lands in the file - twice this recorded the chat window instead of
    # the avatar, and the composite rendered happily from the wrong pixels. SetForegroundWindow is
    # not enough: Windows lets focus move straight back. HWND_TOPMOST keeps the editor above other
    # windows regardless of focus, and is undone in the finally block.
    HWND_TOPMOST, HWND_NOTOPMOST = -1, -2
    SWP = 0x0001 | 0x0002 | 0x0010          # NOSIZE | NOMOVE | NOACTIVATE
    u = ctypes.windll.user32
    hwnd = u.FindWindowW(None, title)
    if hwnd:
        u.ShowWindow(hwnd, 9)               # SW_RESTORE - a minimised editor renders nothing
        u.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP)
        u.SetForegroundWindow(hwnd)
        time.sleep(1.2)
        print("editor pinned on top for the capture")
    else:
        print("editor window not found by title - capture may record the wrong window")

    enc = ["-c:v", "libx264", "-preset", "ultrafast"]
    try:
        if b"h264_nvenc" in subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"],
                                                    stderr=subprocess.STDOUT):
            enc = ["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ull"]
    except (OSError, subprocess.CalledProcessError):
        pass
    print("encoder      : %s" % enc[1])

    log = open(cap_path + ".ffmpeg.log", "wb")
    rec = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "gdigrab", "-framerate", str(a.fps),
         "-offset_x", str(x), "-offset_y", str(y), "-video_size", "%dx%d" % (w, h),
         "-i", "desktop"] + enc + ["-pix_fmt", "yuv420p", cap_path],
        stdin=subprocess.PIPE, stdout=log, stderr=log)

    t_rec = time.time()
    time.sleep(a.settle)
    mark = os.path.join(OUT, "first_packet_%s.json" % stamp)
    print("\ndriving the avatar from %s ...\n" % a.video)
    rc = subprocess.call([PY, "-u", os.path.join(HERE, "f23_video_to_unity.py"),
                          "--video", a.video, "--mark", mark], cwd=HERE)
    time.sleep(a.settle)

    # Align the two clips on the FIRST PACKET rather than on a guessed lead-in: model load is ~10 s
    # and varies run to run, and a side-by-side that drifts is worse than no side-by-side.
    lead = a.settle
    try:
        import json as _json
        with open(mark, encoding="utf-8") as mf:
            lead = max(0.0, _json.load(mf)["t_first"] - t_rec)
        print("first packet landed %.2f s into the capture" % lead)
    except (IOError, OSError, ValueError, KeyError):
        print("no first-packet mark - falling back to --settle for alignment")

    try:
        rec.communicate(b"q", timeout=20)
    except Exception:
        rec.kill()
    log.close()
    if hwnd:
        u.SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, SWP)   # always give the desktop back
    print("\ncaptured %s" % cap_path)
    if rc != 0:
        print("the driver exited %d - the capture may be incomplete" % rc)

    # ---- side by side, trimmed to the drive window, labelled with the caveat -------------------
    font = "C\:/Windows/Fonts/arialbd.ttf"
    vf = (
        "[1:v]trim=start=%0.2f,setpts=PTS-STARTPTS,scale=-2:720[u];"
        "[0:v]scale=-2:720,setpts=PTS-STARTPTS[s];"
        "[s][u]hstack=inputs=2,"
        "drawtext=fontfile=%s:text='SOURCE':x=(w/4)-(tw/2):y=24:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8,"
        "drawtext=fontfile=%s:text='UNITY AVATAR':x=(3*w/4)-(tw/2):y=24:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8,"
        "drawtext=fontfile=%s:text='NO STEREO DEPTH - monocular zrel only. Retarget demo\\, not the sensor path.':"
        "x=(w/2)-(tw/2):y=h-46:fontsize=22:fontcolor=yellow:box=1:boxcolor=black@0.7:boxborderw=8"
        % (lead, font, font, font)
    )
    cmd = ["ffmpeg", "-y", "-i", a.video, "-i", cap_path, "-filter_complex", vf,
           "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
           "-an", side_path]
    print("\ncomposing side-by-side ...")
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        print("compose failed:\n%s" % p.stderr.decode("utf-8", "replace")[-1500:])
        return 1
    print("wrote %s" % side_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
