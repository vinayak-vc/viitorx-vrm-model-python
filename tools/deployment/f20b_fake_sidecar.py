#!/usr/bin/env python3
"""F-20B TEST DOUBLE for sidecar_supervisor.py - never used in production.

Stands in for wholebody_udp_sender.py so the supervisor's OWN state machine (readiness detection,
backoff, crash-loop protection, the duplicate-instance guard) can be proven fast and deterministically
without cycling the real OAK-D camera dozens of times per test run. It accepts the same CLI shape the
supervisor invokes with (so `build_command()` needs no test-only branch) and prints the same two
readiness markers the real sidecar prints (`producer session id = ...` then a `frames=...` line).

  --mode ok      idles forever printing a frames= heartbeat, like a healthy sidecar. Used for the
                 duplicate-supervisor-guard test (no camera/model dependency, no need to wait ~4s
                 for real device init).
  --mode crash   prints its startup line then exits 1 immediately. Used for the crash-loop test -
                 five of these in a row inside the crash-loop window must trip FAILED_PERMANENT.
"""
import argparse
import os
import sys
import time
import uuid

ap = argparse.ArgumentParser()
# The supervisor's own build_command() only knows the real sidecar's flags, so a test harness that
# wants THIS process to crash sets F20B_FAKE_MODE on the supervisor's own environment before
# launching it - subprocess.Popen inherits the parent environment by default, so it reaches here
# without the supervisor needing to know this test double exists.
ap.add_argument("--mode", choices=["ok", "crash"], default=os.environ.get("F20B_FAKE_MODE", "ok"))
ap.add_argument("--model", default="")
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8899)
ap.add_argument("--portrait", action="store_true")
ap.add_argument("--no-portrait", dest="portrait", action="store_false")
ap.add_argument("--portrait-dir", default="ccw")
ap.add_argument("--subpixel-bits", type=int, default=3)
# F-43. This stub uses strict parse_args on purpose - it is the thing that catches the supervisor
# forwarding a flag the real sender would reject. So every flag build_command() emits has to be
# declared here too, and adding one without this line is the failure it exists to find.
ap.add_argument("--ir-dot", type=float, default=0.8)
ap.add_argument("--mono-res", default="800p")
ap.add_argument("--rgb-isp", default="1/1")
ap.add_argument("--seconds", type=float, default=0.0)
args = ap.parse_args()

sid = uuid.uuid4().hex[:12]
print("[wb]   producer session id = %s  (F-20B fake double, mode=%s)" % (sid, args.mode), flush=True)

if args.mode == "crash":
    time.sleep(0.2)
    print("[wb] frames=0 sent=0 hip_z=0.00m measured_body=0/33 fps~0.0 age=0ms stale=0", flush=True)
    print("[fake] simulated crash", flush=True)
    sys.exit(1)

frames = 0
while True:
    time.sleep(0.5)
    frames += 1
    print("[wb] frames=%d sent=%d hip_z=0.90m measured_body=33/33 fps~2.0 age=10ms stale=0"
          % (frames, frames), flush=True)
