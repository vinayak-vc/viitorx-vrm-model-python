"""Single source of truth for where the sidecar reads and writes evidence.

Before this module the evidence root appeared as 115 literals across 60 files, in three different
forms: relative to the caller's working directory, relative to a per-file `HERE`, and as embedded
path strings. Two consequences, both of which had already bitten:

  * `os.path.join("evidence", ...)` silently depends on the CWD, so a harness produced output in a
    different place depending on where it was launched from.
  * `os.path.join(HERE, "evidence", ...)` bakes in the file's own depth, so moving a harness into a
    subfolder pointed it at `tools/<group>/evidence/` - a directory that does not exist. The
    ADR-065 reorganisation broke thirteen files exactly this way, silently, because those harnesses
    need hardware and so are not covered by the self-tests.

Everything here resolves from THIS file's location, which is the repository root, so the answer no
longer depends on the caller's CWD or on how deep the calling script happens to sit.

Set VIRTUAL_MIRROR_EVIDENCE_DIR to redirect the tree - useful for keeping captures off the repo
volume, and for tests that need an isolated directory.
"""
import os

__all__ = ["PROJECT_ROOT", "UNITY_PROJECT_ROOT", "EVIDENCE_ROOT", "evidence",
           "oak_v4", "arm_v1", "arm_v2", "arm_v3", "ensure_dir"]

#: Repository root. This module deliberately lives there so the answer is a single dirname.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

#: The Unity game repository containing this sidecar, i.e. python-sidecar~'s parent.
#: Several harnesses drive Unity and need to reach its Assets tree. They previously hardcoded an
#: absolute path to a checkout that does not exist on any other machine - and, since the project
#: moved from C: to D:, does not exist on this one either.
UNITY_PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

#: Root of the capture tree. Override with VIRTUAL_MIRROR_EVIDENCE_DIR.
EVIDENCE_ROOT = os.environ.get("VIRTUAL_MIRROR_EVIDENCE_DIR") or os.path.join(PROJECT_ROOT, "evidence")


def evidence(*parts):
    """Absolute path inside the evidence tree.

    Accepts embedded separators, so both evidence("oak_v4", "f16") and
    evidence("oak_v4/f16/run.jsonl") work.
    """
    return os.path.join(EVIDENCE_ROOT, *parts)


def oak_v4(*parts):
    """Absolute path inside the OAK-D v4 capture tree (formerly `oak_v4_evidence/`)."""
    return os.path.join(EVIDENCE_ROOT, "oak_v4", *parts)


def arm_v1(*parts):
    """ARM V1 aiming measurements (formerly `arm_v1_evidence/`)."""
    return os.path.join(EVIDENCE_ROOT, "arm_v1", *parts)


def arm_v2(*parts):
    """ARM V2 aiming measurements (formerly `arm_v2_evidence/`)."""
    return os.path.join(EVIDENCE_ROOT, "arm_v2", *parts)


def arm_v3(*parts):
    """ARM V3 aiming measurements (formerly `arm_v3_evidence/`)."""
    return os.path.join(EVIDENCE_ROOT, "arm_v3", *parts)


def ensure_dir(path):
    """Creates `path` if absent and returns it, so a caller can write straight into the result."""
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path
