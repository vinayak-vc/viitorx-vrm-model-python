#!/usr/bin/env python3
"""Direct client for the MCP-for-Unity editor bridge (TCP 127.0.0.1:6401).

The MCP *client* in this Claude Code session is CONNECTION_CLOSED, but the Unity-side bridge host is
still listening. This speaks its wire protocol directly so the forensics work can proceed:

  * server greets with the ASCII line  "WELCOME UNITY-MCP 1 FRAMING=1\n"   (unframed)
  * every subsequent message, both directions, is  <8-byte big-endian length><UTF-8 payload>
  * request payload is  {"type": "<tool_name>", "params": {...}}
  * response payload is {"status": "success"|"error", "result"|"error": ...}

Source of truth: Library/PackageCache/com.coplaydev.unity-mcp@*/Editor/Services/Transport/Transports/
StdioBridgeHost.cs (WriteFrameAsync / ReadFrameAsUtf8Async / handshake) and
Editor/Services/Transport/TransportCommandDispatcher.cs (command.type / command.params).

Usage:
    python ubridge.py ping
    python ubridge.py exec <file.cs.txt>          # execute_code, body is a METHOD BODY (no usings)
    python ubridge.py call <tool> <json-params>

NOTE: execute_code runs the file contents as the body of a generated method, so top-level `using`
directives are a compile error — fully qualify every type.
"""
import json
import socket
import struct
import sys

HOST = "127.0.0.1"
PORT = 6401
# Unity's own FrameIOTimeoutMs is 60 s for a command; allow a little more so we see its timeout
# response rather than our own socket error.
TIMEOUT_S = 300.0


class BridgeError(RuntimeError):
    pass


def _read_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise BridgeError(f"connection closed after {len(buf)}/{n} bytes")
        buf += chunk
    return buf


def _read_line(sock):
    buf = b""
    while not buf.endswith(b"\n"):
        c = sock.recv(1)
        if not c:
            raise BridgeError("connection closed during handshake")
        buf += c
        if len(buf) > 256:
            raise BridgeError(f"handshake line too long: {buf!r}")
    return buf.decode("ascii", "replace").strip()


def call(tool, params=None, port=PORT, timeout=TIMEOUT_S):
    """Send one command and return the parsed response dict. One connection per call.

    The bridge closes stale clients whenever a new one connects, so a short-lived connection is the
    correct pattern here and cannot wedge a concurrently-running real MCP client.
    """
    payload = json.dumps({"type": tool, "params": params or {}}).encode("utf-8")
    with socket.create_connection((HOST, port), timeout=10.0) as sock:
        sock.settimeout(timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        greeting = _read_line(sock)
        if "FRAMING=1" not in greeting:
            raise BridgeError(f"unexpected handshake: {greeting!r}")
        sock.sendall(struct.pack(">Q", len(payload)) + payload)
        (length,) = struct.unpack(">Q", _read_exact(sock, 8))
        if length > (64 << 20):
            raise BridgeError(f"response frame implausibly large: {length}")
        return json.loads(_read_exact(sock, length).decode("utf-8"))


def execute_code(body, port=PORT, timeout=TIMEOUT_S):
    """Run a C# method body in the editor. Raises BridgeError on a bridge- or compile-level failure."""
    resp = call("execute_code", {"action": "execute", "code": body}, port=port, timeout=timeout)
    if resp.get("status") != "success":
        raise BridgeError(json.dumps(resp, indent=2)[:4000])
    return resp.get("result")


def _main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "ping":
        print(json.dumps(call("ping"), indent=2))
    elif cmd == "exec":
        with open(argv[2], "r", encoding="utf-8") as fh:
            body = fh.read()
        try:
            print(json.dumps(execute_code(body), indent=2))
        except BridgeError as exc:
            print(f"BRIDGE/COMPILE ERROR:\n{exc}", file=sys.stderr)
            return 1
    elif cmd == "call":
        params = json.loads(argv[3]) if len(argv) > 3 else {}
        print(json.dumps(call(argv[2], params), indent=2))
    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
