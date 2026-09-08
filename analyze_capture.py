#!/usr/bin/env python3
"""P0 ACCEPTANCE ANALYZER (diagnostic-only, read-only).

Consumes the capture logs and emits the acceptance evidence tables:
  sender_log.jsonl  - sidecar SENT   (seq, t=epoch, sh/el/hip/wr, hipZ, cov, stage timings)
  holds_log.jsonl   - P0-2 events    (RATE_LIMIT / HOLD / DROP per joint, epoch t)
  recv_log.jsonl    - Unity RECEIVED (+DIAG: kn, an, cf{per-limb conf}, tSend, tRecv)
  model_log.jsonl   - avatar APPLIED (+DIAG: gate{}, boneLen{}, leg eulers, tApply)
  blocks.json       - guided_capture.py block boundaries (epoch tStart/tEnd)

  --blocks   segment every metric per test block A-J
  --baseline compare trunk/limb stability against a previous capture dir
"""
import argparse
import collections
import json
import math
import os

# ---- the 12 tracked joints, as (label, recv_log key, index-within-pair) ----
JOINTS = [
    ("L-shoulder", "sh", 0), ("R-shoulder", "sh", 1),
    ("L-elbow", "el", 0), ("R-elbow", "el", 1),
    ("L-wrist", "wr", 0), ("R-wrist", "wr", 1),
    ("L-hip", "hip", 0), ("R-hip", "hip", 1),
    ("L-knee", "kn", 0), ("R-knee", "kn", 1),
    ("L-ankle", "an", 0), ("R-ankle", "an", 1),
]
TRUNK = ["L-shoulder", "R-shoulder", "L-hip", "R-hip"]
GATE_KEYS = ["lArm", "rArm", "lLeg", "rLeg"]


def load(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def stats(v):
    if not v:
        return None
    s = sorted(v)
    n = len(s)

    def q(p):
        return s[min(n - 1, int(p * n))]
    return dict(n=n, mean=sum(s) / n, median=s[n // 2],
                p95=q(0.95), p99=q(0.99), max=s[-1])


def row(label, st, unit="m", width=12):
    if not st:
        return "  %-*s      (no data)" % (width, label)
    return ("  %-*s %8.4f %8.4f %8.4f %8.4f %8.4f  %6d"
            % (width, label, st["mean"], st["median"], st["p95"], st["p99"], st["max"], st["n"]))


def header(unit="m", width=12):
    return ("  %-*s %8s %8s %8s %8s %8s  %6s"
            % (width, "joint", "mean", "median", "p95", "p99", "max", "n")
            + "\n  " + "-" * (width + 52) + "   (%s)" % unit)


def d3(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def nonzero(p):
    return any(abs(c) > 1e-9 for c in p)


def quat_angle(q1, q2):
    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


def displacement(rows, key, idx):
    """Frame-to-frame displacement. Pairs where either sample is all-zero are skipped:
    zero means 'joint not emitted', so including it manufactures a fake ~1 m delta."""
    out = []
    prev = None
    for r in rows:
        if key not in r:
            continue
        p = r[key][idx]
        if not nonzero(p):
            prev = None
            continue
        if prev is not None:
            out.append(d3(prev, p))
        prev = p
    return out


def palm_deltas(rows, qk, tk):
    out = []
    prev = None
    for r in rows:
        if qk not in r or not r.get(tk, False):
            prev = None
            continue
        q = r[qk]
        if prev is not None:
            out.append(quat_angle(prev, q))
        prev = q
    return out


def bucket_of(n):
    if n <= 2:
        return "1-2"
    if n <= 5:
        return "3-5"
    if n <= 8:
        return "6-8"
    if n <= 11:
        return "9-11"
    return "12+"


def gate_analysis(model):
    """Per-limb Unity LimbGate behaviour. This is the P0-1 proof: it reads the gate's OWN
    state as recorded at apply time, never the Python-side hold."""
    res = {}
    for k in GATE_KEYS:
        res[k] = dict(applied=0, held=0, holds=0, reacq=0, runs=[])
    prev = dict((k, 0) for k in GATE_KEYS)
    cur = dict((k, 0) for k in GATE_KEYS)
    for r in model:
        g = r.get("gate")
        if g is None:
            continue
        for k in GATE_KEYS:
            v = g.get(k, 0)
            res[k]["applied"] += 1
            if v:
                res[k]["held"] += 1
            if v and not prev[k]:
                res[k]["holds"] += 1
                cur[k] = 1
            elif v and prev[k]:
                cur[k] += 1
            elif (not v) and prev[k]:
                res[k]["reacq"] += 1
                res[k]["runs"].append(cur[k])
                cur[k] = 0
            prev[k] = v
    for k in GATE_KEYS:
        if cur[k]:
            res[k]["runs"].append(cur[k])
    return res


def report_section(recv, model, holds, sender, title, indent=""):
    """All per-window metrics. Used for the whole capture and for each block."""
    print("\n%s== %s ==" % (indent, title))
    has_diag = bool(recv and "cf" in recv[0])

    if sender:
        cov = [r.get("cov", 0) for r in sender]
        tracked = sum(1 for c in cov if c >= 8)
        hz = sorted(r.get("hipZ", 0) for r in sender)
        print("%s  tracked frames %d/%d (%.1f%%)   max joints %d/33   distance median %.2f m"
              % (indent, tracked, len(cov), 100.0 * tracked / max(1, len(cov)),
                 max(cov) if cov else 0, hz[len(hz) // 2] if hz else 0))
        if tracked == 0:
            print("%s  *** NO SUBJECT TRACKED - metrics below are NOT valid ***" % indent)

    print("\n%s-- per-joint frame-to-frame displacement (m) --" % indent)
    print(header())
    for label, key, idx in JOINTS:
        if key in ("kn", "an") and not has_diag:
            print("  %-12s      (not logged - pre-instrumentation build)" % label)
            continue
        print(row(label, stats(displacement(recv, key, idx))))

    print("\n%s-- palm / wrist rotation delta (deg) --" % indent)
    print(header(unit="deg"))
    for label, qk, tk in [("L-palm", "lpalm", "ltrk"), ("R-palm", "rpalm", "rtrk")]:
        print(row(label, stats(palm_deltas(recv, qk, tk))))

    if has_diag:
        print("\n%s-- per-limb confidence reaching the Unity gate --" % indent)
        print(header(unit="0..1"))
        for k in GATE_KEYS:
            vals = [r["cf"][k] for r in recv if "cf" in r and k in r["cf"]]
            print(row(k, stats(vals)))

    g = gate_analysis(model)
    if any(g[k]["applied"] for k in GATE_KEYS):
        print("\n%s-- UNITY LimbGate (P0-1 proof: gate's own state at apply time) --" % indent)
        print("%s  %-6s %9s %9s %8s %8s %8s  %s"
              % (indent, "limb", "applied", "held", "holds", "reacq", "longest", "hold-run lengths"))
        tot_h = tot_r = 0
        for k in GATE_KEYS:
            d = g[k]
            tot_h += d["holds"]
            tot_r += d["reacq"]
            b = collections.Counter(bucket_of(x) for x in d["runs"])
            print("%s  %-6s %9d %9d %8d %8d %8d  %s"
                  % (indent, k, d["applied"], d["held"], d["holds"], d["reacq"],
                     max(d["runs"]) if d["runs"] else 0, dict(b) if b else "-"))
        print("%s  TOTAL holds=%d  reacquires=%d" % (indent, tot_h, tot_r))
        if tot_h == 0:
            print("%s  *** UNITY GATE NEVER FIRED in this window ***" % indent)
        else:
            print("%s  >>> UNITY GATE FIRED (%d hold events) <<<" % (indent, tot_h))

    if holds:
        print("\n%s-- P0-2 sidecar events (rate-limit / hold / drop) --" % indent)
        per = collections.defaultdict(collections.Counter)
        disp = collections.defaultdict(list)
        for r in holds:
            per[r["joint"]][r["action"]] += 1
            if r["action"] == "RATE_LIMIT":
                disp[r["joint"]].append(r["displacement"])
        print("%s  %-15s %9s %7s %6s   rate-limited displacement (m)"
              % (indent, "joint", "RATE_LIM", "HOLD", "DROP"))
        for j in sorted(per):
            dd = sorted(disp[j])
            if dd:
                ds = "median %.3f  p95 %.3f  max %.3f" % (
                    dd[len(dd) // 2], dd[min(len(dd) - 1, int(0.95 * len(dd)))], dd[-1])
            else:
                ds = "-"
            print("%s  %-15s %9d %7d %6d   %s"
                  % (indent, j, per[j]["RATE_LIMIT"], per[j]["HOLD"], per[j]["DROP"], ds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="pipeline_logs")
    ap.add_argument("--blocks", default="", help="blocks.json from guided_capture.py")
    ap.add_argument("--baseline", default="", help="baseline capture dir for the before/after table")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    d = a.dir

    sender = load(os.path.join(d, "sender_log.jsonl"))
    recv = load(os.path.join(d, "recv_log.jsonl"))
    model = load(os.path.join(d, "model_log.jsonl"))
    holds = load(os.path.join(d, "holds_log.jsonl"))

    print("=" * 78)
    print("P0 ACCEPTANCE ANALYSIS   dir=%s   %s" % (d, a.label))
    print("=" * 78)
    print("rows: sender=%d recv=%d model=%d holds=%d"
          % (len(sender), len(recv), len(model), len(holds)))
    has_diag = bool(recv and "cf" in recv[0])
    has_gate = bool(model and "gate" in model[0])
    print("recv  DIAG (kn/an/cf/tSend/tRecv): %s" % ("PRESENT" if has_diag else "ABSENT"))
    print("model DIAG (gate/boneLen/tApply) : %s" % ("PRESENT" if has_gate else "ABSENT"))

    # ---------- whole capture ----------
    report_section(recv, model, holds, sender, "WHOLE CAPTURE")

    # ---------- bone length ----------
    print("\n== BONE LENGTH (squash vs rotation) ==")
    if model and "boneLen" in model[0]:
        for bone in ["femur", "shin", "upArm", "foreArm"]:
            vals = [r["boneLen"][bone] for r in model
                    if "boneLen" in r and r["boneLen"].get(bone, 0) > 0]
            if vals:
                s = sorted(vals)
                spread = s[-1] - s[0]
                print("  %-8s min %.5f  max %.5f  spread %.6f m  -> %s"
                      % (bone, s[0], s[-1], spread,
                         "CONSTANT (no squash)" if spread < 1e-4 else "VARYING (investigate)"))
    else:
        print("  not logged (needs the DIAG build)")

    # ---------- packet loss ----------
    print("\n== PACKET LOSS / ORDER ==")
    if sender and recv:
        s_seq = [r["seq"] for r in sender if "seq" in r]
        r_seq = [r["seq"] for r in recv if "seq" in r]
        if s_seq and r_seq:
            lo, hi = max(min(s_seq), min(r_seq)), min(max(s_seq), max(r_seq))
            sent_in = set(x for x in s_seq if lo <= x <= hi)
            recv_in = set(x for x in r_seq if lo <= x <= hi)
            missing = sent_in - recv_in
            ooo = sum(1 for i in range(1, len(r_seq)) if r_seq[i] < r_seq[i - 1])
            print("  window seq %d..%d   sent=%d received=%d missing=%d (%.2f%%) "
                  "out-of-order=%d duplicates=%d"
                  % (lo, hi, len(sent_in), len(recv_in), len(missing),
                     100.0 * len(missing) / max(1, len(sent_in)), ooo,
                     len(r_seq) - len(set(r_seq))))
    else:
        print("  need BOTH sender_log and recv_log from the same run")

    # ---------- latency ----------
    print("\n== END-TO-END LATENCY ==")
    if sender and "camLatMs" in (sender[0] if sender else {}):
        for key, label in [("camLatMs", "camera sensor -> host"),
                           ("capToPoseMs", "host frame -> pose solved"),
                           ("poseToDepthMs", "pose -> depth backprojection"),
                           ("capToSendMs", "host frame -> UDP sent")]:
            v = sorted(r[key] for r in sender if key in r and r[key] >= 0)
            if v:
                n = len(v)
                print("  %-28s median %7.2f  p95 %7.2f  max %8.2f ms  (n=%d)"
                      % (label, v[n // 2], v[min(n - 1, int(0.95 * n))], v[-1], n))
    if has_diag:
        udp = [(r["tRecv"] - r["tSend"]) * 1000.0 for r in recv
               if r.get("tSend", 0) > 1e6 and r.get("tRecv", 0) > 1e6]
        udp = [x for x in udp if -1000 < x < 5000]
        st = stats(udp)
        if st:
            print("  %-28s median %7.2f  p95 %7.2f  max %8.2f ms  (n=%d)"
                  % ("UDP send -> Unity receive", st["median"], st["p95"], st["max"], st["n"]))
        if has_gate:
            # Only the FIRST apply of a seq is that pose's latency; the control rig re-applies
            # the same pose every render frame, so later applies just measure packet interval.
            first = {}
            for m in model:
                s = m.get("seq")
                if s is None or "tApply" not in m:
                    continue
                if s not in first or m["tApply"] < first[s]:
                    first[s] = m["tApply"]
            rmap = dict((r["seq"], r) for r in recv if "seq" in r)
            a2 = [(t - rmap[s]["tRecv"]) * 1000.0 for s, t in first.items()
                  if s in rmap and "tRecv" in rmap[s]]
            a2 = [x for x in a2 if 0 <= x < 5000]
            st2 = stats(a2)
            if st2:
                print("  %-28s median %7.2f  p95 %7.2f  max %8.2f ms  (n=%d)"
                      % ("Unity receive -> avatar", st2["median"], st2["p95"], st2["max"], st2["n"]))

    # ---------- per-block ----------
    if a.blocks and os.path.exists(a.blocks):
        blocks = json.load(open(a.blocks))
        print("\n" + "=" * 78)
        print("PER-BLOCK BREAKDOWN")
        print("=" * 78)
        for b in blocks:
            t0, t1 = b["tStart"], b["tEnd"]
            s_b = [r for r in sender if t0 <= r.get("t", 0) <= t1]
            r_b = [r for r in recv if t0 <= r.get("tRecv", 0) <= t1]
            m_b = [r for r in model if t0 <= r.get("tApply", 0) <= t1]
            h_b = [r for r in holds if t0 <= r.get("t", 0) <= t1]
            report_section(r_b, m_b, h_b, s_b,
                           "BLOCK %s - %s  (%.1fs)" % (b["block"], b["title"], b["seconds"]))
    elif a.blocks:
        print("\n[blocks file not found: %s]" % a.blocks)

    # ---------- baseline comparison ----------
    if a.baseline and os.path.isdir(a.baseline):
        b_recv = load(os.path.join(a.baseline, "recv_log.jsonl"))
        if b_recv:
            print("\n" + "=" * 78)
            print("BEFORE (baseline %s)  vs  P0 (%s)" % (a.baseline, d))
            print("=" * 78)
            print("  %-12s %26s %26s" % ("", "-- baseline --", "-- P0 --"))
            print("  %-12s %8s %8s %8s %8s %8s %8s   %s"
                  % ("joint", "median", "p95", "max", "median", "p95", "max", "change (median)"))
            for label, key, idx in JOINTS:
                bs = stats(displacement(b_recv, key, idx))
                ps = stats(displacement(recv, key, idx))
                if not bs and not ps:
                    continue
                if not bs:
                    print("  %-12s %8s %8s %8s %8.4f %8.4f %8.4f   NEW (no baseline)"
                          % (label, "-", "-", "-", ps["median"], ps["p95"], ps["max"]))
                elif not ps:
                    print("  %-12s %8.4f %8.4f %8.4f %8s %8s %8s   NOT MEASURED"
                          % (label, bs["median"], bs["p95"], bs["max"], "-", "-", "-"))
                else:
                    ch = 100.0 * (ps["median"] - bs["median"]) / max(1e-9, bs["median"])
                    print("  %-12s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f   %+7.1f%% %s"
                          % (label, bs["median"], bs["p95"], bs["max"],
                             ps["median"], ps["p95"], ps["max"], ch,
                             "BETTER" if ch < -5 else ("WORSE" if ch > 5 else "same")))
            print("\n  TRUNK REGRESSION CHECK (must not get worse):")
            for label, key, idx in JOINTS:
                if label not in TRUNK:
                    continue
                bs = stats(displacement(b_recv, key, idx))
                ps = stats(displacement(recv, key, idx))
                if bs and ps:
                    ch = 100.0 * (ps["median"] - bs["median"]) / max(1e-9, bs["median"])
                    print("    %-12s %+7.1f%%  ->  %s"
                          % (label, ch, "OK" if ch <= 5 else "REGRESSED"))
    print()


if __name__ == "__main__":
    main()
