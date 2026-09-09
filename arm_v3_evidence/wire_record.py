#!/usr/bin/env python3
"""Record the UDP pose wire verbatim, then replay one recorded frame as a static hold.

Why record rather than re-derive: the landmark payload is produced by video_udp_sender.py's estimator
(RTMW3D + torso-normalised metric scale). Re-implementing that here would risk drifting from what the
production sender actually emits. Instead the unmodified sender is pointed at this recorder's port, so
every byte captured is exactly what it would have sent to Unity, and the replay is bit-identical
except for `seq`/`t`, which MUST be re-stamped (P1-3's PoseBuffer drops any packet whose seq is below
the newest it already holds, so a replay starting at an old seq would be silently ignored).

    # 1. capture (in one shell)
    python wire_record.py record --out wire.jsonl --port 9999
    #    (in another) python video_udp_sender.py --video ... --port 9999
    # 2. inspect
    python wire_record.py list --dump wire.jsonl
    # 3. hold one frame on the real wire
    python wire_record.py hold --dump wire.jsonl --index 137 --seconds 20
"""
import argparse
import json
import socket
import sys
import time


def cmd_record(a):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((a.host, a.port))
    sock.settimeout(a.idle_timeout)
    n = 0
    with open(a.out, "w", encoding="utf-8") as fh:
        print(f"recording on {a.host}:{a.port} -> {a.out} (stops after {a.idle_timeout}s idle)", flush=True)
        while True:
            try:
                data, _ = sock.recvfrom(262144)
            except socket.timeout:
                break
            try:
                msg = json.loads(data.decode("utf-8"))
            except Exception:
                continue
            fh.write(json.dumps({"index": n, "recvT": round(time.time(), 4), "msg": msg}) + "\n")
            n += 1
            if n % 50 == 0:
                print(f"  {n} packets", flush=True)
    print(f"done: {n} packets", flush=True)
    return 0


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def cmd_list(a):
    recs = load(a.dump)
    print(f"{len(recs)} packets")
    for r in recs[:: a.every]:
        lm = r["msg"]["lm"]
        filled = sum(1 for p in lm if p[3] > 0.0)
        print(f"  index={r['index']:4d} seq={r['msg']['seq']} filled={filled}/33")
    return 0


def cmd_hold(a):
    recs = load(a.dump)
    match = [r for r in recs if r["index"] == a.index]
    if not match:
        print(f"index {a.index} not in dump (have 0..{recs[-1]['index']})", file=sys.stderr)
        return 1
    msg = dict(match[0]["msg"])
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (a.host, a.port)
    seq = int((time.time() - 1788900000.0) * 100.0)   # monotonic across restarts, see module docstring
    t0 = time.time()
    sent = 0
    while time.time() - t0 < a.seconds:
        msg["seq"] = seq
        msg["t"] = round(time.time(), 4)
        sock.sendto(json.dumps(msg).encode("utf-8"), addr)
        seq += 1
        sent += 1
        time.sleep(1.0 / a.rate)
    print(f"held index {a.index} for {a.seconds:.1f}s ({sent} packets)", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record")
    r.add_argument("--out", required=True)
    r.add_argument("--host", default="127.0.0.1")
    r.add_argument("--port", type=int, default=9999)
    r.add_argument("--idle-timeout", type=float, default=20.0)
    r.set_defaults(fn=cmd_record)

    l = sub.add_parser("list")
    l.add_argument("--dump", required=True)
    l.add_argument("--every", type=int, default=25)
    l.set_defaults(fn=cmd_list)

    h = sub.add_parser("hold")
    h.add_argument("--dump", required=True)
    h.add_argument("--index", type=int, required=True)
    h.add_argument("--host", default="127.0.0.1")
    h.add_argument("--port", type=int, default=8899)
    h.add_argument("--seconds", type=float, default=20.0)
    h.add_argument("--rate", type=float, default=40.0)
    h.set_defaults(fn=cmd_hold)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
