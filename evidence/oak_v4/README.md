# `evidence/oak_v4/` — what is here and what must not be deleted

Evidence backing the V4/V5 torso work and the F-09…F-12 investigations.
Tracked in git: `*.txt`, `*.json`, `*.md` (the published evidence).
Ignored but kept on disk: the bulk per-frame traces and screenshot dumps the reports were computed from.

## Irreplaceable — do not delete, do not overwrite

| File | Describes | Used by |
|---|---|---|
| `f10_gt_marks_near_RECOVERED.json` | **the F-10 capture** (`pipeline_logs_f10_near/`) | `f12_temporal_sign.py` |
| `f10_gt_marks_near.json` | **the F-11 capture** (`pipeline_logs_f11/`) | `f11_gt_score.py`, `f12_temporal_sign.py` |

> **Naming trap.** `f10_gt_marks_near.json` does **not** describe the F-10 capture. The F-11 run was
> started with `--distance near` and `f10_gt_capture.py` wrote its marks to that same path, destroying
> the original. The F-10 windows in `..._RECOVERED.json` were reconstructed by
> `f12_recover_f10_marks.py` (offset 100.75 s, stretch 1.000; all seven static blocks within 3.8° of
> the medians published in `f10_near_analysis.txt`) and the file is flagged `"RECOVERED": true`.
> Every F-10 number downstream inherits that reconstruction.
>
> Both files are copied to `_marks_backup/`. Before any run of `f10_gt_capture.py`, pass an
> `--out-dir` that cannot collide, or back these up again first.

## Distilled data (keeps the raw captures deletable)

- `f09_features.jsonl` — 22,727 frames of per-frame torso features distilled from `pipeline_logs`,
  `pipeline_logs_f08`, `pipeline_logs_surface`, `pipeline_logs_ab_surface`, `pipeline_logs_ab_legacy`.
  Those five raw captures were archived out of the repo on 2026-09-10; this file is what
  `f09_replay.py`, `f09_quantisation_roc.py`, `f09_adversarial.py` and `f10_resolution_sign.py` read.
  It carries `hipL`/`hipR`/`hipDzRaw`/`hipDzEmit`/`hipSpan3D`/`hipYaw3D`, so the **hip-line depth sign**
  hypothesis can be explored on it — but scoring it needs ground truth, i.e. the two captures above.
- `probe_trace.jsonl` + `torso_probe_marks*.json` — the controlled probe that measured the V5
  composition gain at 1.000 (vs the old form's 1.400 / 0.700 / 0.000). Backs shipping behaviour.
- `shots/` — V5 visual-acceptance frames.
