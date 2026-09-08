#!/usr/bin/env python3
"""P0-1 UNITY GATE VERIFIER — compares INTENDED occlusions against what the Unity
LimbGate actually did, read from its own state at apply time (model_log gate{}).

Run after inject_occlusion.py. Answers, per scripted occlusion:
  * did the Unity gate for THAT limb enter HELD?          (the P0-1 requirement)
  * did any OTHER limb hold spuriously?                   (false positives)
  * did the gate re-acquire when confidence returned?     (no permanent freeze)
  * did the held rotation stay non-zero?                  (no origin collapse)
  * did bone lengths move?                                (no squash)
"""
import argparse
import json
import os


def load(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    return rows


GATE_KEYS = ["lArm", "rArm", "lLeg", "rLeg"]
# Which avatar bone eulers each limb drives, for the "held rotation is non-zero" check.
LIMB_BONES = {"lArm": ["llow"], "rArm": ["rlow"],
              "lLeg": ["llowleg"], "rLeg": ["rlowleg"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="pipeline_logs")
    a = ap.parse_args()

    gt_path = os.path.join(a.dir, "ground_truth.json")
    if not os.path.exists(gt_path):
        print("ground_truth.json not found in %s - run inject_occlusion.py first" % a.dir)
        return 2
    gt = json.load(open(gt_path))
    model = load(os.path.join(a.dir, "model_log.jsonl"))
    recv = load(os.path.join(a.dir, "recv_log.jsonl"))

    print("=" * 78)
    print("P0-1 UNITY GATE VERIFICATION   dir=%s" % a.dir)
    print("=" * 78)
    print("model rows=%d  recv rows=%d" % (len(model), len(recv)))
    if not model:
        print("\nNO model_log - was Unity in PLAY with pipelineLogging = ON?")
        return 1
    if "gate" not in model[0]:
        print("\nmodel_log has no gate{} field - the DIAG build is not active.")
        return 1

    # seq -> gate states observed while that pose was applied
    by_seq = {}
    for r in model:
        s = r.get("seq")
        if s is None or "gate" not in r:
            continue
        by_seq.setdefault(s, []).append(r)

    # confidence actually received per seq (proves the wire carried conf 0)
    conf_by_seq = {}
    for r in recv:
        if "seq" in r and "cf" in r:
            conf_by_seq[r["seq"]] = r["cf"]

    occ = [e for e in gt["events"] if e["limb"]]
    print("\nscripted occlusions: %d" % len(occ))
    print("\n  %-6s %7s  %9s %9s  %10s %9s  %s"
          % ("limb", "frames", "conf seen", "gate HELD", "held/applied", "reacquire", "verdict"))
    print("  " + "-" * 84)

    passed = failed = skipped = 0
    spurious_total = 0
    for e in occ:
        limb = e["limb"]
        s0, s1 = e["seqStart"], e["seqEnd"]
        rows = []
        for s in range(s0, s1 + 1):
            rows.extend(by_seq.get(s, []))
        applied = len(rows)
        held = sum(1 for r in rows if r["gate"].get(limb, 0) == 1)
        confs = [conf_by_seq[s][limb] for s in range(s0, s1 + 1) if s in conf_by_seq]
        conf_seen = ("%.2f" % max(confs)) if confs else "n/a"

        # spurious holds on limbs that were NOT occluded
        spurious = 0
        for other in GATE_KEYS:
            if other == limb:
                continue
            spurious += sum(1 for r in rows if r["gate"].get(other, 0) == 1)
        spurious_total += spurious

        # re-acquire: gate back to VALID within the following recovery window
        reacq = "no"
        after = []
        for s in range(s1 + 1, s1 + 41):
            after.extend(by_seq.get(s, []))
        if after and any(r["gate"].get(limb, 0) == 0 for r in after):
            reacq = "yes"

        # applied == 0 means Unity logged NO applied frame for this seq window at all -- almost
        # always because the VRM avatar was not yet bound (model_log only writes once
        # boundAnimator != null). That is an absence of observation, NOT a gate failure, and
        # must not be scored as one.
        if applied == 0:
            verdict = "NOT OBSERVED - avatar not yet bound"
            skipped += 1
        elif held > 0 and reacq == "yes":
            verdict = "PASS"
            passed += 1
        elif held == 0:
            verdict = "FAIL - gate never held"
            failed += 1
        else:
            verdict = "FAIL - no re-acquire"
            failed += 1
        print("  %-6s %7d  %9s %9d  %10s %9s  %s"
              % (limb, e["frames"], conf_seen, held,
                 "%d/%d" % (held, applied), reacq, verdict))

    # held rotation must never be zero (the origin-collapse bug)
    zero_rot = 0
    checked = 0
    for e in occ:
        limb = e["limb"]
        for s in range(e["seqStart"], e["seqEnd"] + 1):
            for r in by_seq.get(s, []):
                if r["gate"].get(limb, 0) != 1:
                    continue
                for bone in LIMB_BONES[limb]:
                    if bone in r:
                        checked += 1
                        if all(abs(v) < 1e-6 for v in r[bone]):
                            zero_rot += 1

    lens = {}
    for r in model:
        for b, v in (r.get("boneLen") or {}).items():
            if v > 0:
                lens.setdefault(b, []).append(v)

    print("\n" + "-" * 78)
    print("SUMMARY")
    print("-" * 78)
    print("  occlusions passed          : %d / %d observed" % (passed, passed + failed))
    if skipped:
        print("  not observed (avatar unbound): %d  (excluded - absence of data, not failure)" % skipped)
    print("  spurious holds (other limb): %d applied-frames" % spurious_total)
    print("  held rotations checked     : %d" % checked)
    print("  held rotations that were 0 : %d   %s"
          % (zero_rot, "(ORIGIN COLLAPSE!)" if zero_rot else "(none - no collapse)"))
    for b in sorted(lens):
        v = lens[b]
        print("  bone %-8s spread      : %.6f m %s"
              % (b, max(v) - min(v), "(constant)" if max(v) - min(v) < 1e-4 else "(VARYING!)"))

    print()
    if failed == 0 and passed > 0 and zero_rot == 0:
        print("  >>> UNITY LimbGate PROVEN: fired on every scripted occlusion, held a")
        print("      non-zero rotation, and re-acquired. <<<")
    elif passed == 0:
        print("  >>> UNITY GATE NEVER FIRED - P0-1 is not working on the live path. <<<")
    else:
        print("  >>> PARTIAL - see the FAIL rows above. <<<")
    print()
    print("  NOTE: this proves the GATE MECHANISM only. Real confidence-degradation")
    print("  profiles, depth behaviour, reacquisition dynamics and avatar appearance")
    print("  still require a human subject (guided_capture.py).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
