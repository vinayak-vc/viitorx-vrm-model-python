#!/usr/bin/env python3
"""F-21 S30 - the FALSE-HOLD measurement, as a repeated matched-pair experiment.

WHAT THIS EXISTS TO MEASURE, and why f21_live_protocol.py could not measure it.

ADR-057's path-consistency gate refuses a candidate that was tracked continuously from outside
SWITCH_MARGIN_M to inside it. It refuses an intruder walking in and a legitimate owner walking back
identically, because from a hip position those ARE the same observation stream, and it resolves the
ambiguity towards a DECLARED hand-over. Offline that cost 96 frames (1.60 s) of a legitimate owner on
one clip. Its rate in a real room is the open number.

f21_live_protocol.py has WALK_IN_IMPOSTOR and WALK_IN_OWNER, but as sequenced they cannot produce
that number, for two reasons found by reading the protocol against the state machine rather than by
running it:

  1. WALK_IN_IMPOSTOR runs 18 s against RELEASE_TIMEOUT_S = 4.0 s, so it ENDS - by design, this is
     the fix working - with a release, a TARGET_SWITCH and the impostor as owner. The next cue then
     says "Owner returns to that same spot". The humans read "Owner" as the first person; the system's
     owner is now the second. The phase measures a SECOND IMPOSTOR WALK-IN under the name of the
     returning-owner case. The word "owner" is a ROLE that changes hands mid-protocol and must never
     appear in a subject-facing instruction.
  2. One rep of each, last, in a 194 s protocol. A rate needs repetitions; getting N costs N x 194 s
     and drags 15 unrelated phases through each one.

So: fixed person labels A and B for the whole session, never a role word. Every rep re-anchors and
VERIFIES the anchor from the sidecar's own log before the manoeuvre, so a rep can never be scored
against an ownership state nobody checked. Cases alternate so conditions are matched.

THE FOUR CASES, and which one is the number that actually matters:

    OWNER_WALKBACK   A anchors, A leaves past the margin, A walks back to the cross.
                     The gate SHOULD refuse - ADR-057 says this is indistinguishable and is
                     deliberately resolved towards a declared hand-over. So its "false-hold rate"
                     is expected to be ~1.0 BY CONSTRUCTION, and measuring that rate proves nothing.
                     What this case is really for is the COST: how long the legitimate owner is
                     actually withheld end to end, and whether a visitor would notice.

    IMPOSTOR_WALKIN  A anchors, A hides, B walks to the cross. The safety case. A SILENT admission
                     of B as A is the failure S27 found and S30 fixed; a refusal followed by a
                     declared TARGET_SWITCH is the correct outcome.

    OWNER_OCCLUDED   A anchors ON the cross AND STAYS THERE; B walks between A and the camera.
    OWNER_TURN       A anchors ON the cross AND STAYS THERE; A turns away or crouches.

                     *** THESE TWO ARE THE REAL FALSE-HOLD RATE. *** In an installation an owner
                     almost never walks fully outside 0.35 m and back - they get briefly occluded by
                     a passer-by, turn away, crouch, are blocked. Those are chain breaks that land
                     INSIDE the margin, and ADR-057 is explicit that such a break CLEARS the flag and
                     the candidate stays admissible. So the gate MUST NOT fire here. If it does, that
                     is a false hold in the damaging sense - a real availability defect against the
                     most common real-world loss pattern - and it is not something any offline clip
                     has ever exercised.

A PHYSICAL FLOOR MARK IS REQUIRED. The gate is a 0.35 m position test, so "the same spot" has to be
an objective place, not a memory. Tape a cross on the floor at the capture position before starting;
every cue refers to it. Without it the manoeuvre is unrepeatable and the reps are not matched.

    .venv\\Scripts\\python.exe f21_walkin_protocol.py --supervised --reps 16
"""
import argparse
import io
import json
import os
import subprocess
import sys
import time

import f21_live_protocol as LP     # reuse the launch definition so it cannot drift from the direct
                                   # protocol: PY / SCRIPT / SUPERVISOR / MODEL / ProducerDied

HERE = os.path.dirname(os.path.abspath(__file__))
EVIDENCE = os.path.join(HERE, "oak_v4_evidence", "f21", "walkin")
TIMELINE = os.path.join(EVIDENCE, "timeline.jsonl")
CUE_FILE = os.path.join(EVIDENCE, "cue.json")
SUP_EVIDENCE = os.path.join(EVIDENCE, "supervisor")

CASES = ["OWNER_WALKBACK", "IMPOSTOR_WALKIN", "OWNER_OCCLUDED", "OWNER_TURN"]

SIDECAR_WINDOW = "whole-body OAK sidecar"
WINDOW_AT = (0, 0)   # the composite window is 1900 px wide; AUTOSIZE opens it wherever Windows
                     # likes, which on this desktop put a third of the camera feed off-screen.

# (phase, actor, instruction, detail, seconds). Short enough to read at a glance at 2-3 m; the
# cue display sizes the instruction at ~60 px cap-height, which is ~20-26 arcmin at 2.5 m.
BLOCK = {
    "OWNER_WALKBACK": [
        ("CLEAR",    "NOBODY", "Everyone OUT of view",       "wait",                  5),
        ("ANCHOR",   "A",      "A stand on the X",           "B stay out of view",   12),
        ("A_LEAVE",  "A",      "A walk OUT to the right",    "keep going, well out",  6),
        ("A_RETURN", "A",      "A walk back to the X",       "slowly",               10),
        ("SETTLE",   "A",      "A stand still on the X",     "B stay out",           10),
    ],
    "IMPOSTOR_WALKIN": [
        ("CLEAR",    "NOBODY", "Everyone OUT of view",       "wait",                  5),
        ("ANCHOR",   "A",      "A stand on the X",           "B stay out of view",   12),
        ("A_LEAVE",  "A",      "A walk OUT to the right",    "keep going, well out",  6),
        ("B_ENTER",  "B",      "B walk to the X",            "slowly, same way A did", 10),
        ("SETTLE",   "B",      "B stand still on the X",     "A stay out",           10),
    ],
    # A NEVER LEAVES THE X in these two - that is the whole point. The chain break lands inside the
    # margin, so ADR-057 says the flag clears and A stays admissible.
    "OWNER_OCCLUDED": [
        ("CLEAR",    "NOBODY", "Everyone OUT of view",       "wait",                  5),
        ("ANCHOR",   "A",      "A stand on the X",           "B stay out of view",   12),
        ("B_CROSS",  "B",      "B walk past, in front of A", "A does NOT move",       8),
        ("SETTLE",   "A",      "A still on X. B go out",     "A never left the X",   10),
    ],
    "OWNER_TURN": [
        ("CLEAR",    "NOBODY", "Everyone OUT of view",       "wait",                  5),
        ("ANCHOR",   "A",      "A stand on the X",           "B stay out of view",   12),
        ("A_TURN",   "A",      "A turn around on the X",     "feet stay on the X",    8),
        ("SETTLE",   "A",      "A face the camera again",    "A never left the X",   10),
    ],
}


events = []
_mode = "direct"
OWNERSHIP_LOG_DIR = {"direct": EVIDENCE,
                     "supervised": os.path.join(HERE, "oak_v4_evidence", "f21")}


# NO AUDIO CUES. Verified on this machine: winsound.Beep returns without error but there is no
# output device attached, so it is silent. A cue channel that cannot be heard is not a cue channel,
# and winsound.Beep BLOCKS for its full duration - it would have added ~90 ms of jitter per phase
# boundary to produce nothing. The cue panel is the only channel, which is why it carries colour,
# a giant letter and a diagram as well as the sentence.


def place_window(timeout=90.0):
    """Move the sidecar's window to the top-left corner once it exists.

    cv2.imshow with no namedWindow is WINDOW_AUTOSIZE and opens wherever Windows decides. That was
    harmless at 400 px; the composite is 1900 px and the first run put the whole camera half past
    the right edge of a 1920 px screen - the operator lost the F-21 HUD and the recording captured
    a cropped session. Moved from here via user32 rather than by editing the sidecar; S31.3 already
    enumerates this window from user32, so it is an established handle in this project.

    A False is cosmetic, never fatal: the protocol runs either way and the window can be dragged."""
    import ctypes
    u = ctypes.windll.user32
    deadline = time.time() + timeout
    while time.time() < deadline:
        hwnd = u.FindWindowW(None, SIDECAR_WINDOW)
        if hwnd:
            u.SetWindowPos(hwnd, 0, WINDOW_AT[0], WINDOW_AT[1], 0, 0, 0x0001 | 0x0004)
            return True
        time.sleep(0.5)
    return False


def preflight():
    """Prove the interpreter and the accelerator, and put the proof IN THE EVIDENCE.

    The handoff's S30.11 trap: .venv\Scripts\python.exe carries onnxruntime with
    DmlExecutionProvider, the system Python does not, and rtmw3d_pose.RTMW3D asks for DirectML and
    falls back to CPU SILENTLY - 183 ms/frame instead of ~31. Ownership decisions are deterministic
    so correctness survives, but the frame RATE does not, and this protocol measures DURATIONS
    (hold, blackout) in seconds against a stream whose frame rate is exactly what changes. A 6x
    slower producer is a different experiment.

    This has already bitten twice, and leftover sidecars under the system Python were found running
    on this machine today. So it is asserted here rather than remembered, and the answer is written
    into the timeline so any number from this session can be audited later against the interpreter
    that produced it."""
    import onnxruntime as ort
    info = dict(event="PREFLIGHT", t=round(time.time(), 4),
                protocol_exe=sys.executable, sidecar_exe=LP.PY,
                providers=list(ort.get_available_providers()))
    info["directml"] = "DmlExecutionProvider" in info["providers"]
    info["venv"] = os.path.normcase(sys.executable) == os.path.normcase(LP.PY)
    events.append(info)
    print("preflight: exe=%s" % sys.executable)
    print("           providers=%s" % ", ".join(info["providers"]))
    if not info["venv"]:
        print("  !! NOT the sidecar venv. Run with .venv\Scripts\python.exe.")
    if not info["directml"]:
        print("  !! NO DirectML - inference will silently fall back to CPU (~5 fps, not ~31).")
        print("     Durations measured against that stream are NOT comparable. Fix before running.")
    return info["venv"] and info["directml"]


WIRE_PORT = 8900          # the sidecar sends here; the probe forwards to Unity on 8899


def start_wire_probe(out_path, seconds):
    """Insert f24_wire_probe between the sidecar and Unity so the EMITTED pose is recorded.

    S34 is the reason this exists. The silent migration was only visible in the EMITTED hip, and it
    had to be read off a recorded preview at 2 Hz because nothing logged it. The probe records every
    packet and forwards it BYTE-FOR-BYTE, so Unity still sees exactly what the sidecar sent.

    Returns the process, or None. A None here ABORTS the session rather than running it unmeasured -
    the whole point is that a run which looks captured and is not is worse than no run at all, which
    is the same lesson start_recorder() below carries.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cmd = [LP.PY, "-u", os.path.join(HERE, "f24_wire_probe.py"),
           "--listen-port", str(WIRE_PORT), "--forward-port", "8899",
           "--out", out_path, "--seconds", str(int(seconds))]
    try:
        proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.DEVNULL,
                                stderr=subprocess.STDOUT)
    except OSError as ex:
        print("WIRE PROBE FAILED TO START: %s" % ex)
        return None
    time.sleep(1.0)                      # it binds immediately; catch an instant death
    if proc.poll() is not None:
        print("WIRE PROBE EXITED IMMEDIATELY (rc=%s). Is port %d already in use?"
              % (proc.returncode, WIRE_PORT))
        return None
    return proc


def start_recorder(path):
    """Screen-record the session with ffmpeg gdigrab.

    The whole desktop, not just one window: it captures the cue and the preview+HUD in one frame on
    one clock, which is what makes an after-the-fact eye review of wrong-person frames possible at
    all. Nothing in wholebody_udp_sender.py writes video - no VideoWriter, no imwrite - so without
    this the preview is gone the moment the subjects leave, and S30.2's point that TARGET_SWITCH=0
    is not evidence of correctness would have nothing to check against.

    MATROSKA, NOT MP4, and this was learned the hard way: an MP4 only becomes playable when ffmpeg
    writes the moov atom at the END, so the first test recording that got killed mid-run produced
    113 MB of "moov atom not found" - a whole session's evidence, unreadable. Matroska is written
    incrementally and a truncated file still plays up to the cut. Over a two-hour session with two
    people, "the recording survives an abrupt stop" is worth more than the container being tidy.

    Returns None if ffmpeg is absent; the session is still worth running, the eye review is not.

    ffmpeg's stderr goes to a LOG FILE, never to DEVNULL. A first attempt silenced it and a
    recording came back 15 s long for a 43 s rep with no way to find out why - a recorder that fails
    quietly is worse than no recorder, because the session looks captured until someone tries to
    review it. The log sits next to the recording it belongs to.

    HARDWARE ENCODER WHERE THERE IS ONE, and this was measured rather than assumed. libx264 at
    1920x1080 competes with the sidecar's own inference for CPU and never catches up: ffmpeg's own
    speed= went 0.161x -> 0.976x and a two-rep run recorded 70.2 s of an ~86 s session. Roughly a
    sixth of the session simply was not captured, which for an eye-review recording means the
    interesting seconds might be the missing ones. h264_nvenc runs on the GPU's dedicated encoder
    block - not the shaders, so it does not contend with DirectML - and captured 225 of 225 frames
    at 0.99x. Falls back to libx264 where no NVENC exists; the choice is logged either way.
    """
    enc = ["-c:v", "libx264", "-preset", "ultrafast"]
    try:
        have = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"],
                                       stderr=subprocess.STDOUT)
        if b"h264_nvenc" in have:
            enc = ["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ull"]
    except (OSError, ValueError, subprocess.CalledProcessError):
        pass
    print("recorder encoder: %s" % enc[1])
    try:
        log = io.open(path + ".ffmpeg.log", "wb")
        return subprocess.Popen(
            ["ffmpeg", "-y", "-f", "gdigrab", "-framerate", "15", "-i", "desktop"]
            + enc + ["-pix_fmt", "yuv420p", path],
            stdin=subprocess.PIPE, stdout=log, stderr=log)
    except (OSError, ValueError):
        return None


_cue_write_failures = [0]


def write_cue(**kw):
    """Publish the cue atomically, and NEVER die because a write lost a race.

    Found the hard way: os.replace() raised PermissionError [WinError 5] mid-protocol and took the
    whole run down. On Windows a rename over a file another process holds open fails, and the
    sidecar re-reads cue.json on EVERY preview frame (~30 Hz) while this writes it at 10 Hz - so the
    collision is not rare, it is expected, and over a two-hour session it is a certainty.

    A cue update is worth ~0.1 s of countdown; the session is worth two people's afternoon. So a
    lost write is retried briefly and then SKIPPED, never raised. Skips are counted and written into
    the timeline rather than swallowed - a silent failure here would degrade the instructions the
    subjects are following with nothing in the evidence to say so."""
    tmp = CUE_FILE + ".tmp"
    try:
        with io.open(tmp, "w", encoding="utf-8") as f:
            json.dump(kw, f)
        for _ in range(5):
            try:
                os.replace(tmp, CUE_FILE)
                return True
            except PermissionError:
                time.sleep(0.01)
    except (IOError, OSError):
        pass
    _cue_write_failures[0] += 1
    return False


def read_events():
    """Every ownership event the sidecar has written so far. Read from its log, never from memory."""
    path = os.path.join(OWNERSHIP_LOG_DIR[_mode], "target_events.jsonl")
    out = []
    try:
        for line in io.open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue          # a torn last line while the sidecar is mid-write
    except IOError:
        pass
    return out


def events_since(t0):
    return [e for e in read_events() if e.get("t", 0) >= t0]


def alive(proc, phase, what):
    """LP.check_alive, tolerant of --no-launch (a dry run has no child to watch)."""
    if proc is not None:
        LP.check_alive(proc, phase, what)


# ---------------------------------------------------------------------------------------------
# S33 ANCHOR STABILITY. The 2026-09-15 shakedown established an anchor in every rep and was scored
# 4/4 scoreable - while the ownership machine was churning through SIX epochs in 71 s underneath it.
# `valid = anchor is not None` only asked whether a lock was EVER seen, never whether it SURVIVED
# to the manoeuvre. A rep whose anchor collapsed two seconds later is not a measurement of anything.
#
# The test uses the production machine's OWN verdict rather than a threshold invented here: over the
# closing window of the ANCHOR phase, F-21 must emit NO destabilising transition at all. That is
# exactly the condition under which the following manoeuvre is interpretable, and it introduces no
# new tunable.
#
# HOLD_S is the one new number and it is derived, not picked: it must EXCEED RELEASE_TIMEOUT_S
# (4.0 s, target_ownership.OwnershipConfig) or a complete lose-release-reacquire cycle could pass
# through the window without leaving a transition inside it.
# ---------------------------------------------------------------------------------------------
HOLD_S = 5.0
DESTABILISING = ("TARGET_TEMP_LOST", "TARGET_RELEASED", "TARGET_SWITCH")


def anchor_held(t_phase_end, hold_s=HOLD_S):
    """Did the anchor SURVIVE the rest of the ANCHOR phase?

    -> (ok, transitions, diagnosis). `transitions` are the destabilising events that disqualify it;
    `diagnosis` is operator-facing text naming the likeliest physical cause, derived from the
    rejection reasons and the hip depths the sidecar itself reported.
    """
    t_from = t_phase_end - hold_s
    win = [e for e in read_events() if t_from <= e.get("t", 0) <= t_phase_end]
    bad = [e for e in win if e.get("event") in DESTABILISING]
    if not bad:
        return True, [], ""

    reasons = {}
    for e in win:
        if e.get("event") == "TARGET_REJECTED_CANDIDATE":
            r = e.get("reason", "?")
            reasons[r] = reasons.get(r, 0) + 1
    zs = sorted(e["pos"][2] for e in win if e.get("pos"))
    diag = "transitions=%s" % ",".join(sorted(set(e["event"] for e in bad)))
    if reasons:
        diag += "  rejections=%s" % reasons
    if len(zs) >= 5:
        lo, hi = zs[int(0.1 * len(zs))], zs[int(0.9 * len(zs))]
        diag += "  hip_z p10-p90 %.2f-%.2f m (spread %.2f)" % (lo, hi, hi - lo)
        if (hi - lo) > 0.35:
            # 0.35 m is SWITCH_MARGIN_M: a depth swing that size IS a position_jump by itself.
            diag += "\n      -> DEPTH IS UNSTABLE. The hip depth is swinging by more than the "
            diag += "0.35 m ownership\n         margin while the subject stands still, so F-21 "
            diag += "cannot hold a lock.\n         Move the subject OFF the back wall (>=1 m of "
            diag += "separation), out from under a\n         bright linear light, and onto the "
            diag += "cross at 2.0-2.5 m. A featureless white\n         wall is the classic stereo "
            diag += "failure case and its depth leaks into the hip sample."
    elif reasons.get("scale_mismatch"):
        diag += "\n      -> torso span is unstable; check the whole body is in frame."
    return False, bad, diag


def run_phase(proc, what, case, rep, total_reps, phase, actor, instruction, detail, seconds,
              wait_for_anchor=False):
    alive(proc, phase, what)
    t0 = time.time()
    events.append(dict(t=round(t0, 4), rep=rep, case=case, phase=phase, actor=actor,
                       instruction=instruction, event="phase_start"))
    print("  [%-8s] %-6s %s" % (phase, actor, instruction), flush=True)

    anchor = None
    while True:
        left = seconds - (time.time() - t0)
        if left <= 0:
            break
        alive(proc, phase, what)
        write_cue(title=phase, instruction=instruction, detail=detail, actor=actor,
                  seconds_left=left, seconds_total=seconds,
                  case=case, rep="%d/%d" % (rep, total_reps))
        if wait_for_anchor and anchor is None:
            for e in events_since(t0):
                if e.get("event") in ("TARGET_LOCKED", "TARGET_ACQUIRED"):
                    anchor = e
                    break
        time.sleep(0.1)
    alive(proc, phase, what)
    events.append(dict(t=round(time.time(), 4), rep=rep, case=case, phase=phase,
                       event="phase_end", anchor=anchor))
    return anchor


def run_rep(proc, what, case, rep, total_reps):
    print("\n--- rep %d/%d  %s ---" % (rep, total_reps, case), flush=True)
    t_rep = time.time()
    events.append(dict(t=round(t_rep, 4), rep=rep, case=case, event="rep_start"))
    anchor = None
    for phase, actor, instruction, detail, seconds in BLOCK[case]:
        got = run_phase(proc, what, case, rep, total_reps, phase, actor, instruction, detail,
                        seconds, wait_for_anchor=(phase == "ANCHOR"))
        if phase == "ANCHOR":
            anchor = got
            if anchor is None:
                print("    ANCHOR NEVER ESTABLISHED - this rep is INVALID and will be excluded",
                      flush=True)
            else:
                held, bad, diag = anchor_held(time.time())
                if not held:
                    print("    ANCHOR DID NOT HOLD (%d destabilising events in the last %.0f s) - "
                          "this rep is INVALID" % (len(bad), HOLD_S), flush=True)
                    print("      %s" % diag, flush=True)
                    anchor = None
    events.append(dict(t=round(time.time(), 4), rep=rep, case=case, event="rep_end",
                       valid=anchor is not None,
                       anchor_held=anchor is not None,
                       anchor_target_id=(anchor or {}).get("target_id")))
    return anchor is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--supervised", action="store_true",
                    help="run under sidecar_supervisor.py (F-20B) so a DepthAI crash is a restart")
    ap.add_argument("--reps", type=int, default=16, help="total reps, split evenly across --cases")
    ap.add_argument("--cases", default=",".join(CASES),
                    help="comma-separated subset of: " + ",".join(CASES))
    ap.add_argument("--no-launch", action="store_true",
                    help="assume a sidecar is already running; only drive the cues (dry run)")
    ap.add_argument("--force", action="store_true",
                    help="run even if preflight fails (wrong interpreter / no DirectML). Any "
                         "duration measured that way must be reported as such.")
    ap.add_argument("--no-wire", action="store_true",
                     help="do NOT record the emitted UDP wire. Default is to record it, because "
                          "S34's silent migration was only visible in the EMITTED hip position and "
                          "had to be read off a preview at 2 Hz. With the wire it is measured per "
                          "frame. Recording inserts f24_wire_probe between the sidecar (port 8900) "
                          "and Unity (8899); the forward is byte-identical.")
    ap.add_argument("--no-record", action="store_true",
                    help="do not screen-record. The recording is the ONLY way to review "
                         "wrong-person frames by eye afterwards; nothing else captures the preview.")
    a = ap.parse_args()

    cases = [c.strip().upper() for c in a.cases.split(",") if c.strip()]
    for c in cases:
        if c not in BLOCK:
            print("unknown case %r; known: %s" % (c, ", ".join(CASES)))
            return 2

    global _mode
    _mode = "supervised" if a.supervised else "direct"
    os.makedirs(EVIDENCE, exist_ok=True)

    if not preflight() and not a.force:
        print("")
        print("Refusing to run. Pass --force to override (and say so in the report).")
        return 2

    order = [cases[i % len(cases)] for i in range(a.reps)]     # alternate, so conditions are matched

    # The probe must be listening BEFORE the sidecar starts sending, and its lifetime has to cover
    # the whole session - the reps plus model load plus the settle at the end.
    wire_proc, wire_path = None, None
    if not a.no_wire and not a.no_launch:
        wire_path = os.path.join(EVIDENCE, "wire_%s.jsonl" % time.strftime("%Y%m%d_%H%M%S"))
        budget = 120 + int(a.reps * 45)
        wire_proc = start_wire_probe(wire_path, budget)
        if wire_proc is None:
            print("\nABORTING: the wire recording is the only per-frame evidence of what was "
                  "EMITTED,\nand S34's silent migration is invisible without it. Re-run with "
                  "--no-wire to\nproceed deliberately unmeasured.")
            return 2
        print("wire probe: recording %s (sidecar -> :%d -> Unity :8899)"
              % (os.path.basename(wire_path), WIRE_PORT))

    proc = None
    what = "sidecar"
    if not a.no_launch:
        if a.supervised:
            os.makedirs(SUP_EVIDENCE, exist_ok=True)
            what = "supervisor"
            cmd = [LP.PY, "-u", LP.SUPERVISOR, "--model", LP.MODEL, "--portrait",
                   "--portrait-dir", "ccw", "--subpixel-bits", "3",
                   "--evidence-dir", SUP_EVIDENCE, "--show", "--cue-file", CUE_FILE,
                   "--cue-panel"]
            if wire_proc is not None:
                # the probe is deliberately bound to WIRE_PORT; that is the point of a relay
                cmd += ["--port", str(WIRE_PORT), "--allow-port-listener"]
        else:
            cmd = [LP.PY, "-u", LP.SCRIPT, "--model", LP.MODEL, "--portrait",
                   "--portrait-dir", "ccw", "--subpixel-bits", "3", "--seconds", "0", "--show",
                   "--ownership-log-dir", EVIDENCE, "--cue-file", CUE_FILE, "--cue-panel"]
            if wire_proc is not None:
                cmd += ["--port", str(WIRE_PORT)]
        write_cue(title="STARTING", instruction="Loading the model", detail="stand by",
                  actor="NOBODY", seconds_left=8, seconds_total=8)
        proc = subprocess.Popen(cmd, cwd=HERE)
    else:
        write_cue(title="DRY RUN", instruction="No sidecar", detail="cues only",
                  actor="NOBODY", seconds_left=3, seconds_total=3)

    rec_proc, rec_path = None, None
    if not a.no_record:
        os.makedirs(EVIDENCE, exist_ok=True)
        rec_path = os.path.join(EVIDENCE, "session_%s.mkv" % time.strftime("%Y%m%d_%H%M%S"))
        rec_proc = start_recorder(rec_path)
        events.append(dict(event="RECORDER_START", t=round(time.time(), 4), path=rec_path,
                           started=rec_proc is not None))
    if wire_path:
        events.append(dict(event="WIRE_START", t=round(time.time(), 4), path=wire_path))
        print("recording: %s" % (rec_path if rec_proc else "FFMPEG NOT AVAILABLE - no recording"))
    print("\nF-21 WALK-IN MATRIX  %d reps: %s" % (a.reps, ", ".join("%s" % c for c in cases)))
    print("Cross taped on the floor at the capture position? Everything below assumes it.")
    if proc is not None:
        print("window placement: %s"
              % ("ok" if place_window() else "not found - drag it to the top-left corner"))
    print("ONE window opens: big instruction + diagram on the LEFT, camera feed + F-21 HUD on the")
    print("RIGHT. The giant letter is who moves. Watch only that window.\n")

    aborted = None
    valid = 0
    try:
        for i, case in enumerate(order, 1):
            if run_rep(proc, what, case, i, a.reps):
                valid += 1
    except LP.ProducerDied as e:
        aborted = dict(event="WALKIN_ABORTED", reason="%s exited" % e.what, returncode=e.rc,
                       phase=e.phase, t=round(time.time(), 4),
                       wall_clock=time.strftime("%Y-%m-%d %H:%M:%S"),
                       completed_reps=valid)
        events.append(aborted)
        print("\n" + "!" * 78)
        print("ABORTED: %s EXITED rc=%s during %s" % (e.what.upper(), e.rc, e.phase))
        print("  %d reps completed. Everything after this is UNOBSERVED. Do not score it." % valid)
        print("!" * 78)
    except KeyboardInterrupt:
        events.append(dict(event="WALKIN_INTERRUPTED", t=round(time.time(), 4),
                           completed_reps=valid))
        print("\nInterrupted by operator after %d reps." % valid)
    finally:
        write_cue(title="DONE", instruction="Protocol complete", detail="thank you",
                  actor="NOBODY", seconds_left=0, seconds_total=1)
        if proc is not None and proc.poll() is None:
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                proc.wait(timeout=8)
            except Exception:
                pass
        if rec_proc is not None and rec_proc.poll() is None:
            # 'q' on ffmpeg's stdin rather than a kill, so it closes the file cleanly. With
            # Matroska a hard kill is survivable too, but a clean stop is still preferred.
            try:
                rec_proc.communicate(b"q", timeout=15)
            except Exception:
                rec_proc.kill()
            events.append(dict(event="RECORDER_STOP", t=round(time.time(), 4)))
            print("recording finalised: %s" % rec_path)
        events.append(dict(event="CUE_WRITE_SKIPS", t=round(time.time(), 4),
                           count=_cue_write_failures[0]))
        if _cue_write_failures[0]:
            print("cue writes skipped (lost the race with the sidecar's reader): %d"
                  % _cue_write_failures[0])
        io.open(TIMELINE, "w", encoding="utf-8").write(
            "".join(json.dumps(e) + "\n" for e in events))
        print("\nwrote %s" % TIMELINE)
        print("score it with:")
        print("  .venv\\Scripts\\python.exe f21_walkin_score.py --timeline %s --events %s"
              % (TIMELINE, os.path.join(OWNERSHIP_LOG_DIR[_mode], "target_events.jsonl")))
    return 1 if aborted else 0


if __name__ == "__main__":
    sys.exit(main())
