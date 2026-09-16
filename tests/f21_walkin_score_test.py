#!/usr/bin/env python3

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(_os.path.join(_d, "_sidecar_path.py")):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import _sidecar_path  # noqa: F401  - puts the sidecar root and every tools/ group on sys.path
"""Verify f21_walkin_score.py against synthetic sessions whose answers are known by construction.

This project has three findings that came from a wrong instrument rather than wrong code (S30.6's
labeller, S31.3's two bad assertions, S31.4's dead --cue-file), and the S30 false-hold rate is about
to be read off this scorer. So the scorer gets the same treatment: sessions are built with the
outcome decided in advance, and the scorer has to agree.

    .venv\\Scripts\\python.exe f21_walkin_score_test.py
"""
import io
import json
import os
import sys
import tempfile

import f21_walkin_score as S

PASS = []
FAIL = []


def check(name, got, want):
    (PASS if got == want else FAIL).append((name, got, want))
    print("  %s %-58s got=%r want=%r" % ("PASS" if got == want else "FAIL", name, got, want))


class Session(object):
    """Builds a timeline + event log on one clock, the way the protocol and the sidecar do."""

    def __init__(self):
        self.t = 1000.0
        self.tl = []
        self.ev = []
        self.rep = 0

    def rep_block(self, case, manoeuvre_events, valid=True, anchor_at_start=True):
        self.rep += 1
        r = self.rep
        self.tl.append(dict(t=self.t, rep=r, case=case, event="rep_start"))
        self.tl.append(dict(t=self.t, rep=r, case=case, event="phase_start", phase="ANCHOR",
                            actor="A"))
        if anchor_at_start:
            self.t += 1.0
            self.ev.append(dict(t=self.t, event="TARGET_LOCKED", state="LOCKED", target_id=r))
        self.t += 2.0
        self.tl.append(dict(t=self.t, rep=r, case=case, event="phase_start", phase="MANOEUVRE",
                            actor="B"))
        for dt, kind, extra in manoeuvre_events:
            self.t += dt
            e = dict(t=self.t, event=kind, state="X", target_id=r)
            e.update(extra or {})
            self.ev.append(e)
        self.t += 1.0
        self.tl.append(dict(t=self.t, rep=r, case=case, event="rep_end", valid=valid))
        self.t += 1.0
        return r

    def score(self):
        d = tempfile.mkdtemp()
        tp, ep = os.path.join(d, "t.jsonl"), os.path.join(d, "e.jsonl")
        io.open(tp, "w", encoding="utf-8").write(
            "".join(json.dumps(x) + "\n" for x in self.tl))
        io.open(ep, "w", encoding="utf-8").write(
            "".join(json.dumps(x) + "\n" for x in self.ev))
        return {r["rep"]: r for r in
                [S.score_rep(x, S.load(ep)) for x in S.reps_from(S.load(tp))]}


REJ = ("TARGET_REJECTED_CANDIDATE", {"reason": "path_walked_in"})
REJ_OTHER = ("TARGET_REJECTED_CANDIDATE", {"reason": "position_jump"})
LOST = ("TARGET_TEMP_LOST", {"reason": "no_observation"})


def main():
    print("f21_walkin_score.py - scorer verification\n")

    # ---- outcome classification, one rep per class -------------------------------------------
    print("outcome classification:")
    s = Session()
    # 1 the damaging false hold: A never left the cross, gate fired anyway
    r1 = s.rep_block("OWNER_OCCLUDED", [(0.5, LOST[0], LOST[1]), (0.5,) + REJ,
                                        (1.0, "TARGET_REACQUIRED", {})])
    # 2 the correct non-event: occluded, gate stayed out of the way
    r2 = s.rep_block("OWNER_OCCLUDED", [(0.5, LOST[0], LOST[1]),
                                        (0.8, "TARGET_REACQUIRED", {})])
    # 3 impostor correctly refused and DECLARED
    r3 = s.rep_block("IMPOSTOR_WALKIN", [(0.5, LOST[0], LOST[1]), (0.5,) + REJ,
                                         (4.0, "TARGET_RELEASED", {}),
                                         (0.5, "TARGET_SWITCH", {}),
                                         (0.1, "TARGET_LOCKED", {})])
    # 4 the S27 failure: impostor silently admitted inside the same epoch
    r4 = s.rep_block("IMPOSTOR_WALKIN", [(0.5, LOST[0], LOST[1]),
                                         (1.0, "TARGET_REACQUIRED", {})])
    # 5 walkback refused by design, then declared
    r5 = s.rep_block("OWNER_WALKBACK", [(0.5, LOST[0], LOST[1]), (0.5,) + REJ,
                                        (4.0, "TARGET_RELEASED", {}),
                                        (0.5, "TARGET_SWITCH", {}),
                                        (0.1, "TARGET_LOCKED", {})])
    # 6 turn on the spot, gate stayed out of the way
    r6 = s.rep_block("OWNER_TURN", [(0.5, LOST[0], LOST[1]), (0.6, "TARGET_REACQUIRED", {})])
    got = s.score()
    check("OWNER_OCCLUDED + path rejection -> FALSE_HOLD", got[r1]["outcome"], "FALSE_HOLD")
    check("OWNER_OCCLUDED clean            -> OK_NO_HOLD", got[r2]["outcome"], "OK_NO_HOLD")
    check("IMPOSTOR refused + switch       -> DECLARED_SWITCH", got[r3]["outcome"],
          "DECLARED_SWITCH")
    check("IMPOSTOR silent reacquire       -> SILENT_WRONG_PERSON", got[r4]["outcome"],
          "SILENT_WRONG_PERSON")
    check("WALKBACK refused + switch       -> REFUSED_THEN_DECLARED", got[r5]["outcome"],
          "REFUSED_THEN_DECLARED")
    check("OWNER_TURN clean                -> OK_NO_HOLD", got[r6]["outcome"], "OK_NO_HOLD")

    # ---- S33 REGRESSION: a DECLARED hand-over must never be scored SILENT ---------------------
    # This is the guard for a real false alarm. On the 2026-09-15 shakedown the classifier tested
    # only `not fired and reacquires > 0`, so a rep that RELEASED and emitted TARGET_SWITCH - the
    # consumer being told, i.e. the correct safe behaviour - was reported as SILENT_WRONG_PERSON,
    # the most serious finding this suite can produce. None of the tests above covered it, which is
    # why it survived: r3 declares AND fires the path gate, so it never isolated announcement from
    # refusal. These four do.
    print("\ndeclared hand-overs are NOT silent (S33 regression):")
    s = Session()
    # the gate did NOT fire, but ownership released and announced a switch -> SAFE, not silent
    d1 = s.rep_block("IMPOSTOR_WALKIN", [(0.5, LOST[0], LOST[1]), (0.5,) + REJ_OTHER,
                                         (4.0, "TARGET_RELEASED", {}),
                                         (0.5, "TARGET_SWITCH", {}),
                                         (0.2, "TARGET_REACQUIRED", {})])
    # the genuine S27 defect: readmitted inside the SAME epoch, nothing announced
    d2 = s.rep_block("IMPOSTOR_WALKIN", [(0.5, LOST[0], LOST[1]),
                                         (0.8, "TARGET_REACQUIRED", {})])
    # walkback, released and announced -> not a silent readmit
    d3 = s.rep_block("OWNER_WALKBACK", [(0.5, LOST[0], LOST[1]),
                                        (4.0, "TARGET_RELEASED", {}),
                                        (0.5, "TARGET_SWITCH", {}),
                                        (0.2, "TARGET_REACQUIRED", {})])
    # walkback, readmitted silently inside the epoch -> still caught
    d4 = s.rep_block("OWNER_WALKBACK", [(0.5, LOST[0], LOST[1]),
                                        (0.8, "TARGET_REACQUIRED", {})])
    got = s.score()
    check("IMPOSTOR released+switched       -> DECLARED_SWITCH", got[d1]["outcome"],
          "DECLARED_SWITCH")
    check("IMPOSTOR same-epoch readmit      -> SILENT_WRONG_PERSON", got[d2]["outcome"],
          "SILENT_WRONG_PERSON")
    check("WALKBACK released+switched       -> RELEASED_THEN_REACQUIRED", got[d3]["outcome"],
          "RELEASED_THEN_REACQUIRED")
    check("WALKBACK same-epoch readmit      -> SILENT_READMIT", got[d4]["outcome"],
          "SILENT_READMIT")
    check("a release WITHOUT a switch is not a declaration",
          S.score_rep(dict(rep=9, case="IMPOSTOR_WALKIN", t_man=0.0, t_end=9.0),
                      [dict(t=1.0, event="TARGET_RELEASED"),
                       dict(t=2.0, event="TARGET_REACQUIRED")])["outcome"],
          "SILENT_WRONG_PERSON")

    # ---- S33: an anchor that ESTABLISHES and then COLLAPSES is not a measurement --------------
    # The 2026-09-15 shakedown scored 4/4 reps "scoreable" while ownership churned through six
    # epochs underneath. A lock that is seen once and gone two seconds later makes everything after
    # it uninterpretable, and the scorer must say so rather than classify the noise.
    print("\nan anchor that collapses before the manoeuvre is excluded (S33):")
    s = Session()
    c1 = s.rep_block("OWNER_OCCLUDED", [(0.5, LOST[0], LOST[1]),
                                        (0.8, "TARGET_REACQUIRED", {})])
    got = s.score()
    check("a stable anchor is still scoreable", got[c1]["scored"], True)
    check("  and is reported stable", got[c1]["anchor_stable"], True)

    # now the same rep, but ownership released + switched DURING the anchor hold
    s2 = Session()
    s2.rep = 0
    s2.tl.append(dict(t=s2.t, rep=1, case="OWNER_OCCLUDED", event="rep_start"))
    s2.tl.append(dict(t=s2.t, rep=1, case="OWNER_OCCLUDED", event="phase_start",
                      phase="ANCHOR", actor="A"))
    s2.ev.append(dict(t=s2.t + 0.5, event="TARGET_LOCKED", state="LOCKED", target_id=1))
    # the collapse, inside the 5 s before the manoeuvre
    s2.ev.append(dict(t=s2.t + 1.0, event="TARGET_TEMP_LOST", state="X", target_id=1))
    s2.ev.append(dict(t=s2.t + 2.0, event="TARGET_RELEASED", state="X", target_id=1))
    s2.ev.append(dict(t=s2.t + 2.5, event="TARGET_SWITCH", state="X", target_id=2))
    s2.t += 3.0
    s2.tl.append(dict(t=s2.t, rep=1, case="OWNER_OCCLUDED", event="phase_start",
                      phase="MANOEUVRE", actor="B"))
    s2.ev.append(dict(t=s2.t + 0.5, event="TARGET_REACQUIRED", state="X", target_id=2))
    s2.t += 2.0
    s2.tl.append(dict(t=s2.t, rep=1, case="OWNER_OCCLUDED", event="rep_end", valid=True))
    g2 = s2.score()
    check("a collapsed anchor is NOT scoreable", g2[1]["scored"], False)
    check("  outcome names the reason", g2[1]["outcome"], "INVALID_ANCHOR_COLLAPSED")
    check("  even though the timeline said valid=True", g2[1]["valid"], True)

    # ---- S34 SILENT MIGRATION: the failure a gate-based metric cannot see --------------------
    # The shape below is taken from the real 2026-09-16 rep 7, read off the recorded preview:
    # the EMITTED hip walked 1.61 -> 1.16 -> 1.58 m while F-21 stayed LOCKED on one epoch and the
    # ownership log recorded nothing at all. The old scorer asked only "did the path gate fire?" for
    # OWNER_OCCLUDED, so that rep scored OK_NO_HOLD - the cleanest possible result for a silent
    # wrong-person emission. These pin the replacement check.
    print("\nsilent migration, measured from the emitted wire (S34):")

    def hips(anchor_z, man_zs, ta=100.0, t0=101.0):
        h = [(ta - 1.0, (0.0, 0.0, anchor_z)), (ta - 0.5, (0.0, 0.0, anchor_z))]
        for i, z in enumerate(man_zs):
            h.append((t0 + i * 0.1, (0.0, 0.0, z)))
        return h

    def rep(case, rel=0, sw=0, n=7):
        return dict(rep=n, case=case, t_man=101.0, t_end=110.0, t_anchor_end=100.0,
                    releases=rel, switches=sw)

    r = rep("OWNER_OCCLUDED")
    S.migration(r, hips(1.61, [1.61, 1.34, 1.16, 1.19, 1.45, 1.58]))
    check("the real rep-7 shape          -> SILENT_MIGRATION", r["migration"], "SILENT_MIGRATION")
    check("  drift_max is the excursion", r["drift_max_m"], 0.45)
    check("  frames beyond the margin counted", r["frames_beyond_margin"], 2)

    r = rep("OWNER_OCCLUDED", rel=1, sw=1, n=3)
    S.migration(r, hips(1.61, [1.61, 1.16, 1.61]))
    check("same drift but ANNOUNCED      -> DECLARED", r["migration"], "DECLARED")

    r = rep("OWNER_WALKBACK", n=1)
    S.migration(r, hips(1.61, [1.61, 3.00, 1.61]))
    check("owner told to leave           -> EXPECTED, no verdict", r["migration"], "EXPECTED")

    r = rep("OWNER_TURN", n=4)
    S.migration(r, hips(1.61, [1.61, 1.66, 1.58, 1.62]))
    check("owner stayed put              -> NONE", r["migration"], "NONE")

    r = rep("OWNER_TURN", n=5)
    S.migration(r, [])
    check("NO WIRE is not a pass         -> no verdict at all", r.get("migration"), None)

    # ---- a rejection for a DIFFERENT reason must not read as a path hold ----------------------
    print("\nthe gate is not the only thing that rejects:")
    s = Session()
    r = s.rep_block("OWNER_OCCLUDED", [(0.5, LOST[0], LOST[1]), (0.5,) + REJ_OTHER,
                                       (0.8, "TARGET_REACQUIRED", {})])
    got = s.score()
    check("position_jump rejection is NOT a false hold", got[r]["outcome"], "OK_NO_HOLD")
    check("  and is counted separately", got[r]["other_rejections"], 1)
    check("  path count stays 0", got[r]["path_rejections"], 0)

    # ---- durations --------------------------------------------------------------------------
    print("\ndurations:")
    s = Session()
    r = s.rep_block("OWNER_WALKBACK", [(0.5, LOST[0], LOST[1]), (0.5,) + REJ,
                                       (2.0, "TARGET_REACQUIRED", {})])
    got = s.score()
    check("blackout = TEMP_LOST -> REACQUIRED", round(got[r]["blackout_s"], 2), 2.5)
    check("hold     = rejection -> REACQUIRED", round(got[r]["hold_s"], 2), 2.0)

    # ---- an unresolved episode must be marked, not silently short ----------------------------
    s = Session()
    r = s.rep_block("OWNER_WALKBACK", [(0.5, LOST[0], LOST[1]), (0.5,) + REJ])
    got = s.score()
    check("never-recovered blackout is flagged truncated",
          got[r].get("blackout_truncated"), True)

    # ---- validity gating ---------------------------------------------------------------------
    print("\nvalidity gating (the guard that makes a rep scoreable at all):")
    s = Session()
    r = s.rep_block("IMPOSTOR_WALKIN", [(0.5, LOST[0], LOST[1]),
                                        (1.0, "TARGET_REACQUIRED", {})],
                    valid=False, anchor_at_start=False)
    got = s.score()
    check("rep with no anchor is marked invalid", got[r]["valid"], False)

    # ---- events OUTSIDE the manoeuvre window must not leak in --------------------------------
    print("\nwindowing:")
    s = Session()
    s.ev.append(dict(t=999.0, event="TARGET_REJECTED_CANDIDATE", reason="path_walked_in",
                     state="X", target_id=0))          # before the session began
    r = s.rep_block("OWNER_OCCLUDED", [(0.5, LOST[0], LOST[1]),
                                       (0.8, "TARGET_REACQUIRED", {})])
    got = s.score()
    check("a rejection before the rep does not count", got[r]["path_rejections"], 0)
    check("  so the rep still reads clean", got[r]["outcome"], "OK_NO_HOLD")

    s = Session()
    r = s.rep_block("OWNER_OCCLUDED", [(0.5, LOST[0], LOST[1]),
                                       (0.8, "TARGET_REACQUIRED", {})])
    s.ev.append(dict(t=s.t + 50.0, event="TARGET_REJECTED_CANDIDATE", reason="path_walked_in",
                     state="X", target_id=r))          # after the rep ended
    got = s.score()
    check("a rejection after the rep does not count", got[r]["path_rejections"], 0)

    # ---- an ANCHOR-phase rejection must not be scored as the manoeuvre -----------------------
    print("\nthe anchor phase is not the manoeuvre:")
    s = Session()
    s.rep += 1
    r = s.rep
    s.tl.append(dict(t=s.t, rep=r, case="OWNER_OCCLUDED", event="rep_start"))
    s.tl.append(dict(t=s.t, rep=r, case="OWNER_OCCLUDED", event="phase_start", phase="ANCHOR",
                     actor="A"))
    s.ev.append(dict(t=s.t + 0.5, event="TARGET_REJECTED_CANDIDATE", reason="path_walked_in",
                     state="X", target_id=r))
    s.ev.append(dict(t=s.t + 1.0, event="TARGET_LOCKED", state="LOCKED", target_id=r))
    s.t += 2.0
    s.tl.append(dict(t=s.t, rep=r, case="OWNER_OCCLUDED", event="phase_start", phase="MANOEUVRE",
                     actor="B"))
    s.t += 2.0
    s.tl.append(dict(t=s.t, rep=r, case="OWNER_OCCLUDED", event="rep_end", valid=True))
    got = s.score()
    check("a rejection during ANCHOR is outside the window", got[r]["path_rejections"], 0)

    # ---- Wilson, the part that stops '0 of 6' becoming 'zero' --------------------------------
    print("\nWilson interval:")
    lo, hi = S.wilson(0, 6)
    check("0/6 lower bound is 0", round(lo, 4), 0.0)
    check("0/6 upper bound is NOT 0 (n=6 cannot show a zero rate)", hi > 0.35, True)
    lo, hi = S.wilson(6, 6)
    check("6/6 upper bound is 1", round(hi, 4), 1.0)
    check("6/6 lower bound is NOT 1", lo < 0.65, True)
    lo, hi = S.wilson(0, 0)
    check("0/0 is maximally uncertain", (round(lo, 3), round(hi, 3)), (0.0, 1.0))

    print("\n%s  %d/%d" % ("ALL PASS" if not FAIL else "FAILURES PRESENT",
                           len(PASS), len(PASS) + len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
