# AGENTS.md — Python tracking sidecar

## Purpose

This file defines how AI agents must behave when working on **this repository**
(`vinayak-vc/viitorx-vrm-model-python`).

It is the Python counterpart to the Unity project's `AGENTS.md`. The two repos have **separate
responsibilities** (§2) and **separate rules**. Do not apply the Unity C# style rules here, and do not
apply these rules to the Unity repo.

Rules take priority over generated output. If a rule conflicts with what you were about to write, the
rule wins.

---

# 1. Global Behavior

* Act as a **senior computer-vision / real-time systems engineer**
* Produce **production-ready code only** — no pseudo-code, no placeholders, no `TODO` stubs
* Do not guess missing requirements; ask
* Match **existing patterns in this repo exactly** (§3)
* Prefer **clarity over brevity** — this is real-time perception code that others must debug at 2 a.m.

## The rule that matters most here

> **No measurement, no claim.**

This is a perception pipeline. Class names, plausible-looking code and "it should be smoother" are not
evidence. Every behavioural claim must be backed by a number from a real capture or a deterministic
test. If you cannot measure it, say so explicitly and mark it **NOT TESTED** — never imply it works.

Corollary: **absence of observation is not evidence of failure** (and vice versa). A window with no
logged frames is missing data, not a defect.

---

# 2. Repository responsibility boundary

| | This repo (Python sidecar) | Unity repo |
|---|---|---|
| Owns | sensor, depth, inference, signal conditioning, the wire format | retarget, avatar, rendering, the final safety gate |
| Stages | OAK-D capture → queue policy → RTMW3D → depth fuse → P0 smoother → P1-1 tracker → UDP send | UDP receive → pose buffer → Kalidokit solve → **P0 LimbGate** → control rig → VRM |
| May change | anything up to and including the datagram | anything from the datagram onward |
| Must NOT change | Unity C#, VRM, Kalidokit, the LimbGate | the sidecar's models or filters |

**The contract between them is the UDP JSON datagram** (documented in `README.md` → *UDP wire
contract*). Changing that format is a **breaking change across two repos**: it requires an ADR in
*both* `docs/decisions.md` files and a coordinated commit.

**Safety layering — do not violate:**

> P1 improves the **signal**. P0 (in Unity) remains the **safety mechanism**.

A joint this repo marks `LOST` has its **emit confidence zeroed**, producing `[0,0,0,0]` — byte-identical
to a real occlusion — so Unity's LimbGate still makes the final call. **Never move safety into this
repo**, and never emit a fabricated position to keep a joint "alive".

---

# 3. Code Style (STRICT — derived from the existing code, not from PEP-8 defaults)

Measured across `wholebody_udp_sender.py`, `smoothing.py`, `oak_depth.py`, `rtmw3d_pose.py`,
`joint_tracker.py` (1672 lines):

## Typing

* **NO type hints.** The codebase has **zero** annotated signatures. Do not introduce them piecemeal —
  a half-annotated codebase is worse than an unannotated one. Document types in the docstring instead.

## Formatting

* **`%`-formatting**, not f-strings (measured 37 `%` vs 5 f-strings) and never `.format()`
* Bare class declarations: `class Foo:` — **not** `class Foo(object):`
* 4-space indent, `snake_case` functions/variables, `UPPER_CASE` module constants
* Soft line limit **~110 chars** for code. Long explanatory comment prose is accepted and common —
  clarity beats column count here.
* Standard library first, then third-party (`numpy`, `cv2`, `depthai`), then local modules

## Comments and docstrings

This repo deliberately carries **rationale**, not restatement. Follow it.

* Module docstring: what the file owns and **why it exists**
* Function docstring: behaviour, units, and the meaning of the return tuple
* Inline comments explain **why a value or branch exists**, tagged with the finding they resolve —
  e.g. `# P0-2 (audit F-03/F-04): ...`, `# M16: ...`, `# LOW-D: ...`
* **Always state units** (`m`, `mm`, `ms`, `m/s`, `deg`) — mixing metres and millimetres is the single
  easiest bug to introduce here
* Never write a comment that only repeats the code

## Disallowed

* ❌ type hints (until the whole repo is converted in one deliberate pass)
* ❌ f-strings / `.format()`
* ❌ `class Foo(object):`
* ❌ silent `except:` — catch a specific exception, or let it raise
* ❌ magic numbers without a comment giving the measurement that justifies them

---

# 4. Real-time constraints

This code runs once per camera frame at ~21–30 fps. A regression here is a latency regression.

* **No allocation in the per-frame path** where avoidable — prefer scalars and preallocated buffers
  (see `JointTracker.__slots__`)
* **No blocking calls** in the frame loop other than the one deliberate `q.get()`
* **Prefer freshness over completeness** — dropping a stale frame is correct; processing it is not
  (ADR-030 in `docs/decisions.md`)
* Measure cost with **`time.perf_counter()`**, never `time.time()` — the latter has ~1 ms resolution on
  Windows and will quantise any sub-millisecond measurement into meaningless 0.000/1.000 values
* State a budget for any new per-frame work and prove you are inside it

---

# 5. Filters, gates and thresholds

* **Every threshold must cite the measurement that set it**, in a comment. `0.35` is not a number, it
  is "below the observed 0.5–0.9 m spikes and 4.7× above peak legitimate motion".
* Prefer **rate-limiting (slew)** over dropping — a hard reject can freeze a joint permanently if the
  condition persists. This has bitten the repo before.
* **Do not add another smoothing stage.** The sidecar is the single smoothing owner (ADR-020). More
  filtering buys stillness by spending responsiveness; add *memory*, not *lag*.
* A plausibility signal should **reduce confidence**, not veto, unless it is provably decisive —
  an uncapped neighbour check once caused 3170 false rejections of legitimate motion.

---

# 6. Error Handling

* Do not fail silently
* Validate array shapes and index ranges at boundaries
* A recoverable per-frame failure logs once (or at a sparse interval) and continues — **never log every
  frame**; healthy frames must stay silent or the log is useless
* A device/model failure at startup should abort loudly with the actual cause

---

# 7. Diagnostics

Diagnostics are a feature of this repo, not an afterthought.

* Event streams log **only interesting events** (spike, hold, drop, state transition) — never healthy frames
* Permanent production metrics belong in `sender_log.jsonl` (see `README.md` → *Diagnostics*)
* New diagnostic-only code must be marked `DIAG-ONLY` in a comment and must be **purely additive** —
  it may not change any decision the pipeline makes
* Test-only instruments (e.g. `--inject-load-ms`) must say **TEST ONLY** in their `--help` text

---

# 8. Testing

* `test_joint_tracker.py` is self-contained (no pytest dependency) — keep it runnable as
  `python test_joint_tracker.py`, exit 0 on pass
* New signal-processing logic needs deterministic tests covering: steady state, legitimate fast motion,
  an isolated spike, a **high-confidence** wrong value, a short gap, a long gap, and recovery
* **When a test fails, first ask whether the test is wrong.** Two of this repo's bugs were caught that
  way, and one "failure" was a fixture that compared a value against itself.
* Offline evaluation against real captured logs (`evaluate_p1.py`) is preferred over synthetic data
  where real data exists — but use **identical input** for any A/B, since a human cannot repeat a
  performance closely enough to measure a filter change

---

# 9. File Creation Rules

* Do not create new files unless necessary
* One responsibility per module; keep the frame loop in `wholebody_udp_sender.py` readable
* Analysis/validation tools are top-level scripts with `argparse` and a `main()`
* Capture output goes to a `--log-dir`; it is git-ignored and must never be committed

---

# 10. Refactoring Rules

* Do not break the UDP wire contract (§2)
* Do not change a tuned constant without a new measurement and an ADR
* Preserve flag names; add new behaviour behind a flag with the safe default, and keep an off-switch
  for A/B (`--tracker` / `--no-tracker`, `--latest-frame` / `--no-latest-frame`)

---

# 11. Priority Order

1. This file (`AGENTS.md`)
2. Existing code patterns in this repo
3. The Unity repo's `AGENTS.md` §16 documentation discipline (mirrored here in §12)
4. General Python best practice / PEP-8

---

# 12. Documentation

Mirror the Unity repo's documentation discipline. Maintain:

- `README.md` — how to run it, the wire contract, known limits
- `docs/project-overview.md`
- `docs/architecture.md`
- `docs/roadmap.md`
- `docs/tasks.md`
- `docs/decisions.md`
- `docs/ai_handoff.md`

## Before coding

Read `docs/architecture.md`, `docs/roadmap.md`, `docs/ai_handoff.md`. Summarise your understanding
before implementing.

## During coding

Follow §3–§8. Update `docs/tasks.md`. Add an ADR to `docs/decisions.md` for any tuned constant, filter,
queue policy or wire-format change.

## After coding

Update `docs/ai_handoff.md`: files modified, what was **measured**, what is still **NOT TESTED**, and
the next recommended task. Update `README.md` if flags, files or the wire contract changed.

**An ADR here must record the measurement, the honest limit, and an explicit DO-NOT list.** A decision
without a number is an opinion.

Assume another AI agent continues tomorrow, in the *other* repo, without this conversation.
Optimise for agent portability across the boundary.
