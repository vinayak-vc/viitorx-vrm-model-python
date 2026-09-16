#!/usr/bin/env python3
"""Record the production UDP wire live, and forward it to Unity unchanged.

WHY A RELAY RATHER THAN A LOG FLAG. The sidecar's --log-dir writes a summary of selected landmarks;
this records the EXACT bytes the consumer receives, which is the only thing that can be said to be
"what production sent". It also keeps Unity driven, so the avatar can be watched and recorded during
the same run that produces the numbers - one session, one clock, no second capture to align.

    sidecar --port 8900  ->  this probe (records)  ->  127.0.0.1:8899  ->  Unity

The forward is byte-for-byte: this process never parses a packet before passing it on, so it cannot
alter what Unity sees. Parsing happens only for the recorded copy.

    .venv\\Scripts\\python.exe f24_wire_probe.py --seconds 60 --out oak_v4_evidence/f24/wire.jsonl
"""
import argparse
import io
import json
import os
import socket
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen-port", type=int, default=8900)
    ap.add_argument("--forward-host", default="127.0.0.1")
    ap.add_argument("--forward-port", type=int, default=8899,
                    help="0 = record only, do not forward")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    rx.bind(("127.0.0.1", a.listen_port))
    rx.settimeout(0.5)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if a.forward_port else None

    print("probe listening on %d, forwarding to %s:%s"
          % (a.listen_port, a.forward_host, a.forward_port or "(off)"))
    print("recording to %s for %.0f s" % (a.out, a.seconds))

    n = bad = 0
    t0 = time.time()
    with io.open(a.out, "w", encoding="utf-8") as f:
        while time.time() - t0 < a.seconds:
            try:
                data, _ = rx.recvfrom(262144)
            except socket.timeout:
                continue
            if tx is not None:
                tx.sendto(data, (a.forward_host, a.forward_port))   # forward FIRST, unparsed
            try:
                msg = json.loads(data.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                bad += 1
                continue
            msg["_rx"] = round(time.time(), 4)
            f.write(json.dumps(msg) + "\n")
            n += 1
            if n % 150 == 0:
                print("  %d packets, %.1f pkt/s" % (n, n / max(1e-6, time.time() - t0)), flush=True)
    rx.close()
    dt = time.time() - t0
    print("\nrecorded %d packets in %.1f s (%.1f pkt/s), %d unparseable" % (n, dt, n / dt, bad))
    if n == 0:
        print("NOTHING ARRIVED. Is the sidecar running with --port %d ?" % a.listen_port)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
