"""Makes the sidecar's modules importable from a harness that lives in a subfolder.

Python puts the SCRIPT's own directory on sys.path, not the project root, so a harness under
tools/<group>/ cannot `import rtmw3d_pose` on its own. Importing this module fixes that for the
root and for every tools/ group, which is what the handful of cross-group imports need
(f21_supervisor_integration -> f20b_deployment_verify, f24_jitter_session -> f21_live_protocol,
and the capture cluster's internal web).

Harnesses reach this file via the two-line prelude at the top of each one.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_TOOLS = os.path.join(_ROOT, "tools")

_paths = [_ROOT]
if os.path.isdir(_TOOLS):
    _paths.append(_TOOLS)
    for _name in sorted(os.listdir(_TOOLS)):
        _candidate = os.path.join(_TOOLS, _name)
        if os.path.isdir(_candidate):
            _paths.append(_candidate)
_paths.append(os.path.join(_ROOT, "tests"))

for _p in _paths:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
