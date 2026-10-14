"""FROZEN classification & fork-detection rules for the fork-suppression analysis.

============================================================================
FREEZE DECLARATION (do NOT edit the constants below after seeing any result).
Frozen: 2026-08-12, before any metric was computed. Editing a threshold, a
tool->class mapping, or a marker list after inspecting NEP/uplift numbers is
p-hacking and is prohibited (plan §10). Sensitivity/robustness is expressed by
sweeping the parameters that are ALREADY declared sweepable here
(FORK_PERCENTILES, and the SUBMIT-as-3rd-class split), never by editing rules.
============================================================================

Two things are frozen here:

(1) §4 token classification
    (a) action-onset  -> EXPLORE | COMMIT | SUBMIT | OTHER
    (b) deliberation-markers -> EXPLORE (anchored at the marker's FIRST token)

(2) §5 fork detection
    - PRIMARY: student full-vocab entropy H_t
        * M2/M4 use a PERCENTILE cut (top-p%, p in FORK_PERCENTILES)
        * M3 uses an ABSOLUTE threshold, calibrated ONCE on base (its 90th pct
          H) then fixed; M3 never uses a percentile.
    - REPORTED ALONGSIDE (paper arXiv:2607.05184 §4.1, cross-check only, NOT the
      NEP judge): teacher top-1 prob < 0.25 => fork, > 0.65 => lock.

Scaffold semantics are taken verbatim from swe_ladder.stage_classifier (parse_action,
_EDIT_COMMANDS, _EXPLORE_BASH_HEADS) — we REUSE it, we do not reimplement it.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable

# --- reuse the existing scaffold parser / command sets (do NOT reimplement) ---
_ROOT = os.environ.get("SWE_LADDER_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from swe_ladder.stage_classifier import (  # noqa: E402
    _EDIT_COMMANDS,
    _EXPLORE_BASH_HEADS,
    parse_action,
)

# ---------------------------------------------------------------------------
# (1a) action-onset position classes
# ---------------------------------------------------------------------------
EXPLORE = "EXPLORE"
COMMIT = "COMMIT"
SUBMIT = "SUBMIT"   # finish/submit: reported as a separate 3rd class (sensitivity)
OTHER = "OTHER"     # everything else (e.g. running scripts/pytest = verification)

# file_editor commands that mutate code -> COMMIT. undo_edit is mutating too.
# NOTE this is a SUPERSET of stage_classifier._EDIT_COMMANDS ({str_replace,insert,
# create}) by exactly {"undo_edit"}: the stage machine only treats create/edit as a
# forward stage MARKER, but for the EXPLORE/COMMIT dichotomy an undo is still a code
# commit action, so we include it. Asserted below so a future edit to the upstream
# set is caught instead of silently drifting.
COMMIT_EDIT_COMMANDS = frozenset(_EDIT_COMMANDS | {"undo_edit"})
assert _EDIT_COMMANDS == {"str_replace", "insert", "create"}, (
    "upstream _EDIT_COMMANDS changed; re-freeze COMMIT_EDIT_COMMANDS deliberately"
)

_FILE_EDITOR_FNS = ("file_editor", "str_replace_editor")
_SUBMIT_FNS = ("finish", "submit")


def classify_action_onset(action_text: str) -> str:
    """Classify one step's action-onset (the tool call) into a position class.

    Uses stage_classifier.parse_action (handles standard + inline + the malformed
    hybrid `<parameter=command=str_replace</parameter>`), so we read the ACTUAL
    scaffold tool names, never a guess.

        search                         -> EXPLORE
        file_editor view               -> EXPLORE   (read-only inspection)
        file_editor create/str_replace/insert/undo_edit -> COMMIT
        execute_bash <read-only head>  -> EXPLORE   (find/grep/ls/cat/... )
        execute_bash python/pytest/... -> OTHER      (verification, not a commit)
        finish / submit                -> SUBMIT
        anything else / unparseable    -> OTHER
    """
    fn, params = parse_action(action_text)
    if fn is None:
        return OTHER
    if fn == "search":
        return EXPLORE
    if fn in _FILE_EDITOR_FNS:
        cmd = (params.get("command") or "").strip().lower()
        if cmd == "view":
            return EXPLORE
        if cmd in COMMIT_EDIT_COMMANDS:
            return COMMIT
        return OTHER
    if fn == "execute_bash":
        cmd = params.get("cmd") or params.get("command") or ""
        head = cmd.strip().split()[0] if cmd.strip() else ""
        if head in _EXPLORE_BASH_HEADS:
            return EXPLORE
        return OTHER
    if fn in _SUBMIT_FNS:
        return SUBMIT
    return OTHER


# The literal action-onset marker in the r2egym scaffold; the tool-name first token
# begins immediately after this string (R2E-Gym/.../action.py:52-59).
FUNCTION_ONSET_LITERAL = "<function="

# ---------------------------------------------------------------------------
# (1b) deliberation-markers  (frozen from Qwen3-4B-Instruct-2507 tokenizer)
# ---------------------------------------------------------------------------
# Single-word markers are each a SINGLE token; first-token match is exact. We list
# BOTH the no-leading-space and the leading-space (Ġ) variant ids. Only the FIRST
# token is counted as the marker position.
#   Wait 14190/13824 · Actually 52592/33763 · Hmm 80022/88190 · However 11209/4354
#   · But 3983/1988 · Alternatively 92014/38478 · Instead 30787/12090
FROZEN_MARKER_SINGLE_TOKEN_IDS = frozenset(
    {14190, 13824, 52592, 33763, 80022, 88190, 11209, 4354,
     3983, 1988, 92014, 38478, 30787, 12090}
)

# Multi-word markers share an ambiguous first token ("Let"/"On"), so matching the
# first token alone would over-count every "Let me look..."/"On the...". We therefore
# require the FULL token sequence to match; the counted position is still its FIRST
# token. Both leading-space variants included.
#   "Let me also"          [10061,752,1083] / [6771,752,1083]
#   "Let me double-check"  [10061,752,1990,15934] / [6771,752,1990,15934]
#   "On second thought"    [1925,2086,3381] / [1913,2086,3381]
FROZEN_MARKER_MULTI_SEQS = (
    (10061, 752, 1083), (6771, 752, 1083),
    (10061, 752, 1990, 15934), (6771, 752, 1990, 15934),
    (1925, 2086, 3381), (1913, 2086, 3381),
)


def find_marker_token_positions(token_ids: list[int]) -> list[int]:
    """Return sorted anchor indices (each marker's FIRST-token position) in token_ids.

    Single-token markers match directly; multi-word markers require their full
    sequence. Matches are non-overlapping and a position is reported at most once.
    """
    ids = list(token_ids)
    n = len(ids)
    hits: set[int] = set()
    # multi-word first (they consume a longer span)
    consumed = [False] * n
    for i in range(n):
        for seq in FROZEN_MARKER_MULTI_SEQS:
            L = len(seq)
            if i + L <= n and tuple(ids[i:i + L]) == seq:
                hits.add(i)
                for j in range(i, i + L):
                    consumed[j] = True
                break
    for i in range(n):
        if not consumed[i] and ids[i] in FROZEN_MARKER_SINGLE_TOKEN_IDS:
            hits.add(i)
    return sorted(hits)


# ---------------------------------------------------------------------------
# (2) fork detection
# ---------------------------------------------------------------------------
# M2/M4 percentile cuts (sweepable — declared here, part of the frozen protocol).
FORK_PERCENTILES = (5, 10, 20)  # top-p% of student entropy H_t counts as a fork

# M3 absolute threshold: calibrated ONCE as the 90th percentile of base-model H, then
# frozen. Value filled by run.py's one-time calibration and pinned into results.json;
# it must NOT be recomputed per checkpoint (that would be a moving percentile).
M3_BASE_CALIBRATION_PERCENTILE = 90

# Paper arXiv:2607.05184 §4.1 (cross-check ONLY, never the NEP judge):
PAPER_FORK_TEACHER_TOP1_MAX = 0.25   # teacher top-1 prob < 0.25  => fork
PAPER_LOCK_TEACHER_TOP1_MIN = 0.65   # teacher top-1 prob > 0.65  => lock


def student_fork_mask_percentile(entropies, p: int):
    """Boolean mask: True where H_t is in the top-p% of the given entropy array.

    entropies: 1-D sequence/np.ndarray of per-token student full-vocab entropy.
    Cut is computed on the SUPPLIED pool (caller decides the pool: whole batch for
    M2, per-stage subset for M4 — document whichever is used).
    """
    import numpy as np

    h = np.asarray(entropies, dtype=float)
    if h.size == 0:
        return np.zeros(0, dtype=bool)
    thr = np.percentile(h, 100 - p)
    return h >= thr


def paper_teacher_fork_lock(teacher_top1_prob):
    """(is_fork, is_lock) per the paper's teacher-top-1 diagnostic. Reported alongside
    the student-entropy primary in M1/M3; NOT used to select NEP fork positions."""
    import numpy as np

    q = np.asarray(teacher_top1_prob, dtype=float)
    return (q < PAPER_FORK_TEACHER_TOP1_MAX), (q > PAPER_LOCK_TEACHER_TOP1_MIN)
