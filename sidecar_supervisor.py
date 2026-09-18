#!/usr/bin/env python3
"""F-20B - Sidecar Supervisor / Watchdog.

F-20A (docs/F20A_SESSION_STALE_POSE_RECOVERY_2026-09-14.md) made Unity self-recovering once the
producer comes back: session ids, a stale-pose watchdog, a neutral failsafe. It explicitly left one
gap - "nothing restarts the sidecar", and a USB unplug kills the sidecar process (rc=1). This file is
the producer-side fix: start `wholebody_udp_sender.py` with the exact production configuration,
detect when it dies (crash, forced kill, camera unplug, DepthAI init failure), and restart it with
backoff - so an unattended installation recovers without an operator touching Unity, the scene, or a
command prompt.

Scope, deliberately narrow (see the F-20B brief SS11/16/23): this process supervises the SIDECAR
PROCESS ONLY. It never talks to Unity, never reinterprets tracking quality (that is F-20A's
TrackingStreamHealth, over on the consumer side), and never redesigns the OAK-D pipeline. It answers
exactly one question - "is the producer process alive and producing output" - and restarts it when the
answer is no.

State machine (SS8):
    STOPPED -> STARTING -> RUNNING -> EXITED -> BACKOFF -> STARTING -> ...
                                                    \\-> FAILED_PERMANENT (crash loop; keeps retrying
                                                         at a long, throttled interval - SS10)
A STATIC failure found before ever spawning a process (missing python.exe/script/model, or
`import depthai/cv2/numpy` failing) goes straight to a TERMINAL FAILED_PERMANENT and the supervisor
process exits - SS7 is explicit that these must not loop, and none of them self-heal by retrying. A
camera-not-found / DepthAI init failure is NOT treated as static: it goes through the ordinary
backoff/restart path like any other crash, because it CAN self-heal (the camera gets plugged back in) -
this is what makes SS17 test F (start with the camera absent, then plug it in) recover on its own.

Run:
    python-sidecar~/.venv/Scripts/python.exe sidecar_supervisor.py --portrait --portrait-dir ccw
        --subpixel-bits 3
(all defaults already match the F-19/F-20A production configuration; see run_supervisor.bat)

Multi-person:
    python-sidecar~/.venv/Scripts/python.exe sidecar_supervisor.py
        --script multiperson_udp_sender.py --max-poses 3
Either sender can be supervised. They take different flags, so `--script` (or `--sender`) selects
the command shape - see SENDER_BY_SCRIPT and build_command. Nothing else differs: one state
machine, one backoff table, one readiness contract, which both senders satisfy.
"""

import evidence_paths as EV

import argparse
import io
import json
import os
import signal
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PYTHON = os.path.join(HERE, ".venv", "Scripts", "python.exe")
DEFAULT_SCRIPT = os.path.join(HERE, "wholebody_udp_sender.py")
DEFAULT_MODEL = EV.DEFAULT_MODEL
DEFAULT_EVIDENCE_DIR = EV.oak_v4("f20b")

# ---- which sender is being supervised (F-36 follow-up) -------------------------------------------
#
# The supervisor used to be able to launch exactly one program. Its command builder emits
# --portrait/--portrait-dir/--subpixel-bits/--seconds, which only wholebody_udp_sender.py's argparse
# defines, so pointing --script at the multi-person sender made Python exit on an unrecognised flag
# before the camera was ever opened. That is why the multi-person path was launched DIRECTLY from
# Unity instead - and a directly launched sidecar has no watchdog at all, so anything that ended the
# process (including its own preview window) took tracking down for the rest of the session.
#
# Rather than a generic "--extra-args" passthrough - rejected below for good reasons that still
# hold - each sender declares the flags it accepts. Everything else is unchanged: same state
# machine, same backoff, same readiness and heartbeat contract, which both senders now satisfy by
# printing the SID banner and `frames=` lines.
SENDER_WHOLEBODY = "wholebody"
SENDER_MULTIPERSON = "multiperson"

#: Script basename -> sender kind, so --script alone picks the right command shape and an operator
#: cannot silently get the wrong one by forgetting a second flag.
SENDER_BY_SCRIPT = {
    "wholebody_udp_sender.py": SENDER_WHOLEBODY,
    "multiperson_udp_sender.py": SENDER_MULTIPERSON,
}

#: Must match multiperson_udp_sender.DEFAULT_DETECTOR. Spelled out rather than imported: importing
#: that module pulls in depthai, cv2 and onnxruntime, and the supervisor deliberately loads none of
#: them - its whole job is to survive a child that cannot. Checked in _validate so a missing blob is
#: a TERMINAL failure with a useful message, rather than five silent crash-loop restarts (SS7).
MULTIPERSON_DETECTOR_BLOB = os.path.join(
    HERE, "depthai_blazepose", "models",
    "person-detection-retail-0013_openvino_2022.1_6shave.blob")

STOPPED = "STOPPED"
STARTING = "STARTING"
RUNNING = "RUNNING"
EXITED = "EXITED"
BACKOFF = "BACKOFF"
FAILED_PERMANENT = "FAILED_PERMANENT"

# SS9: "restart 1 -> 1s, 2 -> 2s, 3 -> 5s, 4 -> 10s, 5+ -> 30s" - checked against F-20A's own measured
# device-init time (~4s: F20A report SS7, "the ~4s of StaleFailsafe inside each restart phase is the
# sidecar's own model load and device initialisation"). Every attempt costs at least that ~4s before
# it can prove itself either way, so this table never produces a tight start/crash/start loop.
BACKOFF_SCHEDULE_SECONDS = [1.0, 2.0, 5.0, 10.0, 30.0]

SID_MARKER = "producer session id = "
FRAME_MARKER = "frames="


def _now_str():
    return time.strftime("%H:%M:%S")


class Supervisor(object):
    def __init__(self, args):
        self.args = args
        self.evidence_dir = args.evidence_dir or DEFAULT_EVIDENCE_DIR
        os.makedirs(self.evidence_dir, exist_ok=True)
        os.makedirs(os.path.join(self.evidence_dir, "logs"), exist_ok=True)
        self._log_path = os.path.join(self.evidence_dir, "supervisor.log")
        self._events_path = os.path.join(self.evidence_dir, "supervisor_events.jsonl")
        self._state_path = os.path.join(self.evidence_dir, "supervisor_state.json")

        self.state = STOPPED
        self.proc = None
        self._stdout_fh = None
        self._stdout_log_path = None
        self.sid = None
        self.ready = False
        self.pid = None
        self.start_time = None
        self.restart_count = 0
        self.last_exit_code = None
        self.last_failure_reason = ""
        self.backoff_seconds = 0.0
        self.backoff_tier = 0
        self.exit_history = []           # timestamps, for the crash-loop window (SS10)
        self.running_since = None
        self._stop_requested = False
        self._lock_socket = None
        # Sticky across the whole degraded retry cycle - self.state itself is NOT sticky (it also
        # has to say STARTING/EXITED while a degraded retry attempt is in flight), so without this
        # flag the persisted SupervisorState flips FAILED_PERMANENT -> BACKOFF -> STARTING within
        # milliseconds of tripping and an external reader (an operator's poll, this file's own test
        # harness) almost never observes it. Cleared only on a real recovery (_mark_ready).
        self.degraded = False

    # ---- logging / evidence -----------------------------------------------------------------
    def log(self, msg):
        line = "[%s] %s" % (_now_str(), msg)
        print(line, flush=True)
        with io.open(self._log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def event(self, kind, **fields):
        rec = dict(t=round(time.time(), 3), event=kind)
        rec.update(fields)
        with io.open(self._events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def write_state(self):
        uptime = 0.0
        if self.start_time and self.proc is not None and self.proc.poll() is None:
            uptime = round(time.time() - self.start_time, 1)
        st = dict(
            SupervisorState=self.state,
            SidecarPid=self.pid,
            SidecarUptime=uptime,
            RestartCount=self.restart_count,
            LastExitCode=self.last_exit_code,
            LastFailureReason=self.last_failure_reason,
            BackoffSeconds=self.backoff_seconds,
            SidecarReady=self.ready,
            CurrentSessionId=self.sid,
        )
        tmp = self._state_path + ".tmp"
        # Windows can transiently refuse os.replace() with WinError 5 (Access is denied) if a
        # concurrent reader (an operator's own polling script, a future HUD) has the target file
        # open at that exact instant - observed live during the E_CRASH_LOOP test. The diagnostics
        # file must never be allowed to crash the watchdog itself, so this retries briefly and then
        # gives up quietly; supervisor.log / supervisor_events.jsonl (append-only, no replace race)
        # remain the authoritative record even if a single state snapshot is missed.
        for attempt in range(5):
            try:
                with io.open(tmp, "w", encoding="utf-8") as f:
                    json.dump(st, f, indent=2)
                os.replace(tmp, self._state_path)
                return
            except OSError:
                if attempt == 4:
                    return
                time.sleep(0.05)

    # ---- single-instance / single-sidecar guards (SS14) --------------------------------------
    def acquire_single_instance_lock(self):
        """One supervisor per machine: hold a loopback TCP bind for this process's lifetime. A
        bound socket cannot be stolen by a stale PID file after a crash, which a lock FILE can."""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", self.args.lock_port))
            s.listen(1)
        except OSError:
            self.log("SUPERVISOR ABORT another supervisor already owns lock_port=%d"
                      % self.args.lock_port)
            return False
        self._lock_socket = s
        return True

    def _kill_tracked_process(self):
        """Kills ONLY the PID this supervisor itself spawned (+ its process tree). Never a broad
        `taskkill /IM python.exe` - SS15's explicit warning."""
        if self.proc is None:
            return
        pid = self.proc.pid
        subprocess.call(["taskkill", "/F", "/T", "/PID", str(pid)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            self.proc.wait(timeout=8)
        except Exception:
            pass

    # ---- environment validation (SS7) ---------------------------------------------------------
    def _probe_port_available(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.bind((self.args.host, self.args.port))
            return True
        except OSError:
            return False
        finally:
            s.close()

    def _validate(self, full):
        if not os.path.isfile(self.args.python):
            return False, "python executable not found: %s" % self.args.python
        if not os.path.isfile(self.args.script):
            return False, "sidecar script not found: %s" % self.args.script
        if self.args.model and not os.path.isfile(self.args.model):
            return False, "model file not found: %s" % self.args.model
        if (self.sender_kind() == SENDER_MULTIPERSON
                and not os.path.isfile(MULTIPERSON_DETECTOR_BLOB)):
            return False, ("person detector blob not found: %s (fetch it with blobconverter - see "
                           "docs/F32)" % MULTIPERSON_DETECTOR_BLOB)
        if full:
            # SS7: "depthai imports" - checked once at startup and again whenever we resume from a
            # crash loop, not on every ordinary restart (that would slow down the common case for a
            # check that almost never newly fails between one restart and the next).
            try:
                r = subprocess.run([self.args.python, "-c", "import depthai, cv2, numpy"],
                                    capture_output=True, timeout=30)
            except Exception as e:
                return False, "could not probe python dependencies: %s" % e
            if r.returncode != 0:
                stderr_lines = r.stderr.decode("utf-8", "replace").strip().splitlines()
                tail = stderr_lines[-1] if stderr_lines else "unknown import error"
                return False, "import depthai/cv2/numpy failed: %s" % tail
        return True, ""

    def _port_ready(self, attempts=5, delay=1.0):
        """SS15: verify the UDP socket is actually free before respawning. Transient (the previous
        process's socket hasn't been released by the OS yet) - NOT a terminal failure, so this just
        retries a few times rather than escalating to FAILED_PERMANENT.

        F-21 S34: --allow-port-listener skips this. The check binds the DESTINATION port to infer
        that no stale producer is holding it, but a UDP sender never binds its destination - what
        this actually detects is a LISTENER on the far end. That is normally nobody, so the proxy
        works; it stops working the moment something is deliberately listening there. Two real
        cases: f24_wire_probe relaying sidecar -> probe -> Unity, and Unity itself while it is in
        Play mode holding the port. In both, an occupied port is the HEALTHY state and refusing to
        launch is the wrong call. Off by default, so F-20B's behaviour is unchanged.
        """
        if getattr(self.args, "allow_port_listener", False):
            self.log("PORT CHECK SKIPPED (--allow-port-listener): a listener on %s:%d is expected"
                     % (self.args.host, self.args.port))
            return True
        for i in range(attempts):
            if self._probe_port_available():
                return True
            self.log("PORT IN USE %s:%d retry %d/%d" % (self.args.host, self.args.port, i + 1, attempts))
            self.event("PortInUse", host=self.args.host, port=self.args.port, attempt=i + 1)
            time.sleep(delay)
        return False

    # ---- launch config (SS6) ------------------------------------------------------------------
    def sender_kind(self):
        """Which sender --script names. Explicit --sender wins; otherwise the basename decides."""
        if getattr(self.args, "sender", None):
            return self.args.sender
        return SENDER_BY_SCRIPT.get(os.path.basename(self.args.script or ""), SENDER_WHOLEBODY)

    def build_command(self):
        """Explicit and logged, per SS6 - and restricted to flags the TARGET sender's own argparse
        already defines (SS6: "do not invent arguments"). Each branch below lists only what that
        sender accepts, which is the whole reason this is a branch and not a shared tail."""
        cmd = [self.args.python, "-u", self.args.script,
               "--host", self.args.host, "--port", str(self.args.port)]
        if self.args.model:
            cmd += ["--model", self.args.model]

        if self.sender_kind() == SENDER_MULTIPERSON:
            # multiperson_udp_sender.py does not take --seconds (it runs until stopped), so that one
            # stays out. Everything else is now the SAME flag in both senders and is forwarded to
            # both, because the alternative is what happened twice already: Unity asks for a
            # configuration, one sender honours it, the other silently does not, and the two produce
            # different data from the same camera.
            #   --subpixel-bits : whole-pixel vs 1/8-pixel disparity, an 8x multiplier on depth
            #                     error at every range. Only the single-person sender could act on
            #                     it, so every crowd scene ran coarse.
            #   --portrait      : this sender had NO portrait support at all, so a rotated camera
            #                     gave the two senders different coordinate frames.
            #   --ir-dot        : F-43's IR dot projector. Forwarded to both for the same reason as
            #                     --subpixel-bits: it changes the DEPTH the sender measures, so one
            #                     sender running textured and the other unassisted would mean the
            #                     two columns disagree about the same room.
            # --max-poses is multi-person only, forwarded by name for the same auditability reason.
            cmd += ["--max-poses", str(self.args.max_poses)]
            cmd += ["--subpixel-bits", str(self.args.subpixel_bits)]
            cmd += ["--ir-dot", str(self.args.ir_dot)]
            cmd += ["--mono-res", self.args.mono_res, "--rgb-isp", self.args.rgb_isp]
            if self.args.portrait:
                cmd += ["--portrait", "--portrait-dir", self.args.portrait_dir]
            else:
                cmd += ["--no-portrait"]
            if self.args.show:
                cmd += ["--show"]
            return cmd

        if self.args.portrait:
            cmd += ["--portrait", "--portrait-dir", self.args.portrait_dir]
        else:
            cmd += ["--no-portrait"]
        cmd += ["--subpixel-bits", str(self.args.subpixel_bits), "--seconds", "0"]
        cmd += ["--ir-dot", str(self.args.ir_dot)]
        cmd += ["--mono-res", self.args.mono_res, "--rgb-isp", self.args.rgb_isp]
        # F-21 S31: the two live-session flags, forwarded explicitly.
        #
        # Explicitly, and NOT as a generic "--extra-args" passthrough, which was the obvious
        # smaller change and is the wrong one: a supervisor that forwards arbitrary strings into
        # a supervised process can also forward --no-ownership or --seconds 30, and then the
        # thing under test is not the thing that was configured. Two named flags stay auditable
        # in the SIDECAR CMD log line, and anything else still has to be added here deliberately.
        #
        # Neither flag touches watchdog policy (S10) or the readiness/heartbeat contract (S8):
        # --show only opens a preview window and --cue-file only reads a JSON file the protocol
        # writes. Both are already argparse flags of wholebody_udp_sender.py, so S6's "do not
        # invent arguments" rule still holds.
        if self.args.show:
            cmd += ["--show"]
        if self.args.cue_file:
            cmd += ["--cue-file", self.args.cue_file]
        if getattr(self.args, "cue_panel", False):
            cmd += ["--cue-panel"]
        return cmd

    # ---- lifecycle ------------------------------------------------------------------------------
    def _spawn(self, run_index):
        self.sid = None
        self.ready = False
        log_path = os.path.join(self.evidence_dir, "logs", "run_%03d.txt" % run_index)
        self._stdout_fh = io.open(log_path, "w", encoding="utf-8")
        self._stdout_log_path = log_path
        cmd = self.build_command()
        self.log("SIDECAR CMD %s" % " ".join(cmd))
        self.state = STARTING
        self.proc = subprocess.Popen(cmd, cwd=HERE, stdout=self._stdout_fh,
                                      stderr=subprocess.STDOUT)
        self.pid = self.proc.pid
        self.start_time = time.time()
        self.log("SIDECAR START pid=%d run=%d" % (self.pid, run_index))
        self.event("SidecarStart", pid=self.pid, run=run_index, cmd=cmd)
        self.write_state()

    def _mark_ready(self):
        self.ready = True
        elapsed = round(time.time() - self.start_time, 3)
        self.log("SIDECAR READY sid=%s pid=%d elapsed=%.3fs" % (self.sid, self.pid, elapsed))
        self.event("SidecarReady", sid=self.sid, pid=self.pid, elapsed=elapsed)
        if self.degraded:
            self.log("RECOVERED FROM FAILED_PERMANENT sid=%s" % self.sid)
            self.event("RecoveredFromFailedPermanent", sid=self.sid)
            self.exit_history = []
            self.backoff_tier = 0
            self.degraded = False
        self.state = RUNNING
        self.running_since = time.time()
        self.write_state()

    def _monitor_until_exit(self):
        """One loop, two jobs: (a) before ready, watch stdout for the two markers the real sidecar
        already prints - SID banner then the first `frames=` line - exactly the two-stage check
        f20a_failure_injection.py's wait_until_sending() proved live (SS12: never call it healthy
        from process-alive alone); (b) after ready, watch for stdout GROWTH as a lightweight
        tracking-health heartbeat (SS5) - a process can stay alive while stuck, and this catches
        that without re-implementing any of F-20A's own interpretation of tracking quality."""
        last_size = 0
        last_growth_t = time.time()
        last_state_write = time.time()
        while True:
            rc = self.proc.poll()
            if rc is not None:
                return rc, time.time() - self.start_time, "exited"
            if self._stop_requested:
                return None, time.time() - self.start_time, "stop_requested"

            try:
                size = os.path.getsize(self._stdout_log_path)
            except OSError:
                size = last_size

            if not self.ready:
                if size != last_size:
                    text = io.open(self._stdout_log_path, encoding="utf-8", errors="replace").read()
                    if self.sid is None and SID_MARKER in text:
                        self.sid = text.split(SID_MARKER)[1].split()[0]
                    if self.sid is not None and FRAME_MARKER in text:
                        self._mark_ready()
                    last_growth_t = time.time()
                elif time.time() - self.start_time > self.args.ready_timeout:
                    self.log("STARTUP STUCK pid=%d not ready after %.0fs - killing for restart"
                              % (self.pid, self.args.ready_timeout))
                    self.event("StartupStuck", pid=self.pid, timeout=self.args.ready_timeout)
                    self._kill_tracked_process()
                    return self.proc.poll(), time.time() - self.start_time, "startup_stuck"
            else:
                if size != last_size:
                    last_growth_t = time.time()
                elif self.args.heartbeat_timeout > 0 and \
                        time.time() - last_growth_t > self.args.heartbeat_timeout:
                    self.log("HEARTBEAT TIMEOUT pid=%d no stdout growth for %.0fs - killing for restart"
                              % (self.pid, self.args.heartbeat_timeout))
                    self.event("HeartbeatTimeout", pid=self.pid, timeout=self.args.heartbeat_timeout)
                    self._kill_tracked_process()
                    return self.proc.poll(), time.time() - self.start_time, "heartbeat_timeout"
            last_size = size

            if time.time() - last_state_write > 2.0:
                self.write_state()
                last_state_write = time.time()
            time.sleep(0.3)

    def _handle_exit(self, rc, uptime, cause):
        self.state = EXITED
        self.last_exit_code = rc
        self.last_failure_reason = cause
        if self._stdout_fh is not None and not self._stdout_fh.closed:
            self._stdout_fh.close()
        now = time.time()
        self.exit_history = [t for t in self.exit_history
                              if now - t < self.args.crash_loop_window] + [now]
        self.log("SIDECAR EXIT pid=%s rc=%s uptime=%.1fs cause=%s" % (self.pid, rc, uptime, cause))
        self.event("SidecarExit", pid=self.pid, rc=rc, uptime=round(uptime, 1), cause=cause)

        if uptime >= self.args.running_reset_seconds:
            # SS9: a sustained healthy run forgives past failures so one old blip doesn't inflate
            # every future restart's backoff.
            self.backoff_tier = 0
            self.exit_history = [now]

        if len(self.exit_history) >= self.args.crash_loop_count:
            if not self.degraded:
                # Logged/emitted ONCE, at the transition - not on every subsequent exit while
                # still degraded, which is exactly the "spam logs" failure mode SS10 forbids.
                self.log("CRASH LOOP DETECTED (%d exits within %.0fs) -> FAILED_PERMANENT, "
                          "retrying every %.0fs"
                          % (len(self.exit_history), self.args.crash_loop_window,
                             self.args.failed_permanent_retry))
                self.event("CrashLoop", count=len(self.exit_history),
                            window=self.args.crash_loop_window)
            self.degraded = True
            self.state = FAILED_PERMANENT

        self.restart_count += 1
        self.write_state()

    def _sleep_backoff(self):
        if self.degraded:
            # Stays reported as FAILED_PERMANENT for the whole wait (not BACKOFF) - an external
            # reader of supervisor_state.json must see the degraded state persist, not flicker
            # through a generic "BACKOFF" that looks identical to an ordinary retry.
            delay = self.args.failed_permanent_retry
            self.state = FAILED_PERMANENT
        else:
            tier = min(self.backoff_tier, len(BACKOFF_SCHEDULE_SECONDS) - 1)
            delay = BACKOFF_SCHEDULE_SECONDS[tier]
            self.backoff_tier += 1
            self.state = BACKOFF
        self.backoff_seconds = delay
        self.log("RESTART scheduled delay=%.0fs" % delay)
        self.event("Backoff", delay=delay)
        self.write_state()
        t0 = time.time()
        while time.time() - t0 < delay and not self._stop_requested:
            time.sleep(0.2)

    def _enter_failed_permanent(self, reason, terminal):
        self.state = FAILED_PERMANENT
        self.degraded = True
        self.last_failure_reason = reason
        self.log("FAILED_PERMANENT reason=%s terminal=%s" % (reason, terminal))
        self.event("FailedPermanent", reason=reason, terminal=terminal)
        self.write_state()
        if terminal:
            self._stop_requested = True

    def _install_signal_handlers(self):
        def handler(signum, _frame):
            self.log("SUPERVISOR STOP requested (signal %s)" % signum)
            self._stop_requested = True
        signal.signal(signal.SIGINT, handler)
        sigbreak = getattr(signal, "SIGBREAK", None)   # Windows console close/Ctrl+Break
        if sigbreak is not None:
            signal.signal(sigbreak, handler)

    def _shutdown(self):
        if self.proc is not None and self.proc.poll() is None:
            self._kill_tracked_process()
        if self._stdout_fh is not None and not self._stdout_fh.closed:
            self._stdout_fh.close()
        if self.state != FAILED_PERMANENT:
            # Don't downgrade a genuine terminal failure to a generic "STOPPED" - an operator (or
            # this project's own test harness) reading the final supervisor_state.json needs to be
            # able to tell "shut down cleanly" from "gave up because of an unrecoverable failure"
            # without cross-referencing the log.
            self.state = STOPPED
        self.write_state()
        self.log("SUPERVISOR STOP pid=%d" % os.getpid())
        self.event("SupervisorStop", pid=os.getpid())
        if self._lock_socket is not None:
            try:
                self._lock_socket.close()
            except Exception:
                pass

    def run(self):
        if not self.acquire_single_instance_lock():
            return 1
        self.write_state()
        self.log("SUPERVISOR START pid=%d lock_port=%d evidence=%s"
                  % (os.getpid(), self.args.lock_port, self.evidence_dir))
        self.event("SupervisorStart", pid=os.getpid())
        self._install_signal_handlers()

        ok, reason = self._validate(full=True)
        if not ok:
            self._enter_failed_permanent(reason, terminal=True)
            self._shutdown()
            return 2

        run_index = 0
        try:
            while not self._stop_requested:
                run_index += 1
                ok, reason = self._validate(full=self.degraded)
                if not ok:
                    self._enter_failed_permanent(reason, terminal=True)
                    break
                if not self._port_ready():
                    self._handle_exit(None, 0.0, "port_unavailable")
                    self._sleep_backoff()
                    continue

                self._spawn(run_index)
                rc, uptime, cause = self._monitor_until_exit()
                if self._stop_requested:
                    break
                self._handle_exit(rc, uptime, cause)
                self._sleep_backoff()
        finally:
            self._shutdown()
        return 0


def build_arg_parser():
    ap = argparse.ArgumentParser(description="F-20B sidecar supervisor/watchdog")
    ap.add_argument("--python", default=DEFAULT_PYTHON)
    ap.add_argument("--script", default=DEFAULT_SCRIPT)
    ap.add_argument("--sender", default=None, choices=[SENDER_WHOLEBODY, SENDER_MULTIPERSON],
                     help="which sender --script names, and therefore which flags it is launched "
                          "with. Inferred from the script's basename when omitted.")
    ap.add_argument("--max-poses", type=int, default=3,
                     help="multi-person only: people posed per frame, forwarded to the sender. "
                          "Ignored for the single-person sender, which has no such flag.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--portrait", action=argparse.BooleanOptionalAction, default=True,
                     help="production default is portrait ON (F-19/F-20A); --no-portrait for dev")
    ap.add_argument("--portrait-dir", default="ccw", choices=["ccw", "cw"])
    ap.add_argument("--subpixel-bits", type=int, default=3)
    ap.add_argument("--mono-res", default="800p", choices=["400p", "480p", "720p", "800p"],
                     help="F-44: stereo mono resolution, forwarded to whichever sender runs. "
                          "800p halves depth error at every range; 400p is the pre-F-44 default.")
    ap.add_argument("--rgb-isp", default="1/1", choices=["1/1", "1/2"],
                     help="F-44: RGB ISP scale. 1/1 is native 1280x800 at full FOV (2x pixels "
                          "on a body); 1/2 is the pre-F-44 640x400.")
    ap.add_argument("--ir-dot", type=float, default=0.8,
                     help="F-43: IR dot projector intensity 0..1, forwarded to whichever sender "
                          "runs. 0 disables. Mirrors oak_depth.IR_DOT_INTENSITY.")
    ap.add_argument("--evidence-dir", default=None,
                     help="default evidence/oak_v4/f20b; tests point this at their own subfolder")
    ap.add_argument("--lock-port", type=int, default=8897,
                     help="loopback TCP port used as the single-supervisor-instance mutex (SS14)")
    ap.add_argument("--ready-timeout", type=float, default=45.0,
                     help="seconds to wait for the SID+frames markers before treating startup as stuck")
    ap.add_argument("--heartbeat-timeout", type=float, default=10.0,
                     help="restart if stdout stops growing for this long once ready (0 = disabled)")
    ap.add_argument("--running-reset-seconds", type=float, default=60.0,
                     help="a run this long resets the backoff tier and crash-loop window (SS9)")
    ap.add_argument("--crash-loop-count", type=int, default=5,
                     help="N exits within --crash-loop-window -> FAILED_PERMANENT (SS10)")
    ap.add_argument("--crash-loop-window", type=float, default=300.0)
    ap.add_argument("--show", action="store_true",
                     help="F-21 S31: forward --show to the sidecar so a live operator gets the "
                          "preview window and the F-21 HUD while under supervision")
    ap.add_argument("--allow-port-listener", action="store_true",
                     help="skip the pre-launch UDP port-availability check. Use when something is "
                          "DELIBERATELY listening on --port: a recording relay (f24_wire_probe), or "
                          "Unity itself in Play mode. See _port_ready().")
    ap.add_argument("--cue-file", default=None,
                     help="F-21 S31: forward --cue-file to the sidecar, so f21_live_protocol.py "
                          "can draw its phase banner on that same window")
    ap.add_argument("--cue-panel", action="store_true",
                     help="F-21 S32: forward --cue-panel, so the cue is drawn as a LARGE side "
                          "panel next to the feed in one window instead of a 13 px banner on it. "
                          "Preview-only, same class of flag as --show.")
    ap.add_argument("--failed-permanent-retry", type=float, default=60.0,
                     help="retry interval while FAILED_PERMANENT - long and throttled, never zero, "
                          "so a camera that comes back after a long absence still self-heals (SS20)")
    return ap


def main():
    args = build_arg_parser().parse_args()
    return Supervisor(args).run()


if __name__ == "__main__":
    sys.exit(main())
