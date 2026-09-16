#!/usr/bin/env python3
"""Toggle Unity's Play mode by sending Ctrl+P to the editor window.

WHY THIS AND NOT THE MCP BRIDGE. Unity's MCP bridge is listening on 6400 and greets with
"WELCOME UNITY-MCP 1 FRAMING=1", but no Unity MCP server is wired into this session, and every
framing I tried after the welcome was reset by the peer. Reverse-engineering an undocumented wire
format to save one keystroke is the wrong trade. Ctrl+P is Unity's own documented Play shortcut.

SAFETY, because this synthesises input on someone's desktop: the foreground window is CHECKED
against the editor's own handle immediately before the keystroke, and the keystroke is not sent if
anything else is in front. A stray Ctrl+P into an editor or a browser is a printing dialog or worse.

    .venv\\Scripts\\python.exe f25_unity_play.py --restart
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import sys
import time

TITLE_HINT = "Windows, Mac, Linux"
VK_CONTROL, VK_P = 0x11, 0x50
KEYEVENTF_KEYUP = 0x0002


def find_editor():
    u = ctypes.windll.user32
    u.SetProcessDPIAware()
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
    def cb(h, _):
        n = u.GetWindowTextLengthW(h)
        if n and u.IsWindowVisible(h):
            b = ctypes.create_unicode_buffer(n + 1)
            u.GetWindowTextW(h, b, n + 1)
            if TITLE_HINT in b.value:
                found.append((h, b.value))
        return True

    u.EnumWindows(cb, 0)
    return found[0] if found else (None, None)


def send_ctrl_p(hwnd):
    """Send Ctrl+P, but only while the editor is genuinely the foreground window."""
    u = ctypes.windll.user32
    u.ShowWindow(hwnd, 9)              # SW_RESTORE
    u.SetForegroundWindow(hwnd)
    time.sleep(0.6)
    if u.GetForegroundWindow() != hwnd:
        print("REFUSING: the editor is not in the foreground - a stray Ctrl+P could hit any app.")
        return False
    u.keybd_event(VK_CONTROL, 0, 0, 0)
    u.keybd_event(VK_P, 0, 0, 0)
    time.sleep(0.05)
    u.keybd_event(VK_P, 0, KEYEVENTF_KEYUP, 0)
    u.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart", action="store_true",
                    help="toggle twice: stop Play, then start it again (reloads the scene, which is "
                         "what picks up an edited Bootstrap.unity value)")
    ap.add_argument("--gap", type=float, default=3.0, help="seconds between the two toggles")
    ap.add_argument("--settle", type=float, default=12.0,
                    help="seconds to wait after entering Play for the scene and avatar to load")
    a = ap.parse_args()

    hwnd, title = find_editor()
    if hwnd is None:
        print("Unity editor window not found.")
        return 2
    print("editor: %s" % title[:70])

    if not send_ctrl_p(hwnd):
        return 1
    print("sent Ctrl+P (stop)")
    if a.restart:
        time.sleep(a.gap)
        if not send_ctrl_p(hwnd):
            return 1
        print("sent Ctrl+P (play)")
        print("waiting %.0f s for the scene to load ..." % a.settle)
        time.sleep(a.settle)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
