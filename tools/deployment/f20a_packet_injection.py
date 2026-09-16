#!/usr/bin/env python3
"""F-20A section 14 - stale/malformed packet safety, injected into the REAL Unity provider.

The unit tests exercise the session logic in isolation. This drives the SHIPPING receive path over a
real socket with deliberately hostile packets, so the counters Unity reports can be checked against
what was actually sent. The sidecar must not be running: this script owns the stream.

Each case is announced by writing a phase name into the block file, so the Unity-side telemetry can
be segmented the same way the failure-injection run is.

    python f20a_packet_injection.py --port 8899
"""
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(_os.path.join(_d, "_sidecar_path.py")):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import _sidecar_path  # noqa: F401  - puts the sidecar root and every tools/ group on sys.path
import evidence_paths as EV

import argparse
import io
import json
import os
import socket
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
BLOCK_FILE = EV.oak_v4("f20a", "block.txt")
OUT = EV.oak_v4("f20a", "injection_sent.json")


def landmarks(phase):
    """33 valid landmarks; the value encodes the phase so a wrong pose would be visible."""
    return [[0.10 + phase * 0.001, -0.20, 1.00, 0.9] for _ in range(33)]


def packet(sid, seq, t, phase=0):
    return json.dumps({
        "lm": landmarks(phase),
        "xyz": [0.0, 0.0, 1200.0],
        "src": [1] * 33,
        "seq": seq,
        "sid": sid,
        "t": round(t, 4),
    }).encode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--rate", type=float, default=30.0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(BLOCK_FILE), exist_ok=True)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (a.host, a.port)
    dt = 1.0 / a.rate
    sid_a = "inj" + uuid.uuid4().hex[:9]
    sid_b = "inj" + uuid.uuid4().hex[:9]
    sent = {}

    def label(name):
        io.open(BLOCK_FILE, "w", encoding="utf-8").write("INJ_" + name)
        print("[inj] %s" % name, flush=True)

    def burst(name, items, pause=dt):
        label(name)
        n = 0
        for sid, seq, t in items:
            s.sendto(packet(sid, seq, t), addr)
            n += 1
            time.sleep(pause)
        sent[name] = n
        time.sleep(0.4)

    now = time.time()

    # 1. a clean run so there is a known-good baseline and a known newestSeq
    burst("BASELINE", [(sid_a, i, now + i * dt) for i in range(1, 91)])

    # 2. duplicate: the same seq repeated. Must be rejected as duplicate, not accepted.
    burst("DUPLICATE", [(sid_a, 90, time.time()) for _ in range(20)])

    # 3. old packets: seq well below newestSeq, SAME session. Ordering must still reject these.
    burst("OLD_SEQ", [(sid_a, 5 + i, time.time()) for i in range(20)])

    # 4. delayed packet: arrives late but with a seq just below newest. Same rule.
    burst("DELAYED", [(sid_a, 89, time.time())], pause=0.1)

    # 5. a packet from the PREVIOUS session after a new one has started.
    burst("NEW_SESSION", [(sid_b, i, time.time() + i * dt) for i in range(1, 31)])
    burst("PREV_SESSION_PKT", [(sid_a, 500 + i, time.time()) for i in range(10)])

    # 6. large forward sequence jump, same session - a burst loss. MUST be accepted.
    burst("SEQ_JUMP", [(sid_b, 50000 + i, time.time() + i * dt) for i in range(10)])

    # 7. backward TIMESTAMP with a forward seq. The buffer pushes on the RECEIVE clock, so this
    #    tests that a bogus sender timestamp cannot poison anything downstream.
    burst("TS_JUMP", [(sid_b, 60000 + i, now - 5000.0) for i in range(10)])

    # 8. a burst loss: 60 packets of a gap, then resume. Must recover without a session change.
    label("BURST_LOSS")
    time.sleep(2.0)
    burst("AFTER_LOSS", [(sid_b, 70000 + i, time.time() + i * dt) for i in range(30)])

    label("DONE")
    io.open(OUT, "w", encoding="utf-8").write(json.dumps(
        {"sid_a": sid_a, "sid_b": sid_b, "sent": sent}, indent=2))
    print("[inj] wrote %s" % OUT)
    s.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
