# Changelog

All notable changes to the Virtual Mirror Python sidecar are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This component versions alongside the
[Unity SDK](https://github.com/vinayak-vc/viitorx-vrm-avtar-unity). The **UDP wire contract** is the
public interface and is what semver applies to here.

## [Unreleased]

## [0.1.0] — 2026-09-16

### Added

- `setup_sidecar.ps1` — one-time machine setup. Creates the virtualenv, installs pinned
  dependencies, and verifies that `DmlExecutionProvider` actually loaded. That last check is the
  point: onnxruntime falls back to CPU **without raising**, turning ~31 ms/frame into ~183 ms, so
  the failure is detected at install time instead of surfacing later as unexplained slowness.
- `requirements.lock.txt` — exact `pip freeze` of the working environment.
- `evidence_paths` — the single source of truth for capture locations.
  `VIRTUAL_MIRROR_EVIDENCE_DIR` redirects the whole tree.
- `_sidecar_path` — import shim letting harnesses under `tools/` and `tests/` resolve project
  modules.
- `pyproject.toml`, `LICENSE` (MIT), `.gitattributes` and this changelog.

### Changed

- **Repository layout.** The root now holds only the production path: `sidecar_supervisor.py`,
  `wholebody_udp_sender.py` and the nine modules the sender imports. The 72 investigation harnesses
  moved into `tools/` (grouped by dependency cluster: `capture`, `ownership`, `validation`, `video`,
  `diagnostics`, `deployment`, `armaim`) and `tests/`. Root went from 96 files to 18.

  This is enforced, not merely tidy: the Unity build copies the sidecar root verbatim, so anything
  left there ships to every player.
- **Evidence.** `oak_v4_evidence/`, `arm_v1_evidence/`, `arm_v2_evidence/` and `arm_v3_evidence/`
  are now `evidence/oak_v4`, `evidence/arm_v1`, `evidence/arm_v2` and `evidence/arm_v3`.
- Launcher scripts moved to `scripts/` and re-based with `cd /d "%~dp0.."` so they still resolve the
  repository root from one level down.
- `README.md` — layout, Tests and device-free sections rewritten; they had described roughly a dozen
  scripts deleted in `29ec57e` and printed commands that no longer ran.

### Fixed

- **Seven modules hardcoded an absolute `C:\Unity\...` path** to a checkout present on no machine —
  this project lives on `D:` — making them dead on arrival for every user including their author.
- **Thirteen harnesses built evidence paths from a per-file `HERE`.** After the reorganisation those
  resolved to `tools/<group>/oak_v4_evidence/`, which does not exist. Silent, and invisible to the
  self-tests because the affected harnesses need hardware.
- `requirements.txt` had `onnxruntime-directml` **commented out** while the working environment had
  it installed. A setup performed from that file produced a sidecar with no inference at all.
- Docstring usage examples pointed at `"D:\...\video\123.webm"`, a placeholder nobody could paste
  and run; they now use the real relative path and the post-reorganisation script locations.
- `f16_consolidate` and `f16_edge_check` carried byte-identical 13-line `load()` functions; the
  shared implementation is now `f16_configs.load_rows()`.

### Notes

- Python **3.10 exactly**. `depthai` publishes `cp310` wheels only, and on 3.11+ pip cannot resolve
  it — with an error that never mentions the Python version.
- `depthai_blazepose/` is vendored MIT source from
  [geaxgx/depthai_blazepose](https://github.com/geaxgx/depthai_blazepose). It is the superseded
  Phase-1 path and is not imported by the production pipeline. Its `LICENSE.txt` must stay with it.

[Unreleased]: https://github.com/vinayak-vc/viitorx-vrm-model-python/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/vinayak-vc/viitorx-vrm-model-python/releases/tag/v0.1.0
