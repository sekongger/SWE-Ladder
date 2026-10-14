"""Rule-based per-step stage classifier for end-to-end OPSD trajectories.

Assigns each trajectory step to one of the three staged-OPD stages so that the OPD
loss can route each step's tokens to the stage-appropriate privileged information
(gold locations / gold tests / gold patch) without running a teacher model per step.

Approach: a forward-only (monotonic) phase state machine driven by *unambiguous*
tool-based transition markers. Ambiguous steps (file view, generic bash, test run,
finish, plain reasoning) inherit the current phase. This mirrors the natural agent
workflow localize -> reproduce -> patch, resolves the view-before-edit and
python-run ambiguities by context, and is O(1) per step.

Stage order: causal_localization < localized_issue_reproduction < patch_generation.
A backtracking variant can later relax the monotonic constraint; kept forward-only
here for a validated first version.
"""

from __future__ import annotations

import os
import re

LOCALIZATION = "causal_localization"
REPRODUCTION = "localized_issue_reproduction"
PATCH = "patch_generation"

STAGE_ORDER = {LOCALIZATION: 0, REPRODUCTION: 1, PATCH: 2}

_FUNC_RE = re.compile(r"<function=([^>]+)>")
# Standard form: <parameter=key>value</parameter> (key has no '='; value between tags).
_PARAM_STD_RE = re.compile(r"<parameter=([^=>]+)>(.*?)</parameter>", re.S)
# Inline form the base model frequently emits (r2egym tolerates it): <parameter=key=value>
# e.g. <parameter=command=str_replace>. Without handling this, `command`/`cmd` are missed
# and stage detection (esp. patch edits) silently fails.
# NOTE: the model very often emits a MALFORMED hybrid that mixes the inline '=' with a
# standard-style close, e.g. `<parameter=command=str_replace</parameter>`. The greedy
# `[^>]*` then captures the value as `str_replace</parameter` (up to the '>' of the
# closing tag), so the value no longer matches the edit-command set and the patch stage is
# never detected. We strip everything from the first '<' in the value (see parse_action)
# so `str_replace</parameter` -> `str_replace`. Inline values are simple tokens
# (command/cmd/short paths) that never legitimately contain '<', so this is safe.
_PARAM_INLINE_RE = re.compile(r"<parameter=([^=>\s]+)=([^>]*)>")
# repro scripts: reproduce_issue.py (env convention), repro*.py, reproduce*.py, bug_repro.py ...
_REPRO_BASENAME_RE = re.compile(r"(reproduce|repro)", re.I)
_EDIT_COMMANDS = {"str_replace", "insert", "create"}
# bash commands that are pure repository exploration (localization work)
_EXPLORE_BASH_HEADS = {
    "find", "grep", "egrep", "fgrep", "rg", "ls", "cat", "head", "tail",
    "wc", "tree", "sed", "awk", "nl", "which", "dir", "less", "more",
}


def parse_action(action_text: str):
    """Extract (function_name, params) from an r2egym XML action string."""
    if not action_text:
        return None, {}
    m = _FUNC_RE.search(action_text)
    if not m:
        return None, {}
    fn = m.group(1).strip().lower()
    params = {k.strip().lower(): v.strip() for k, v in _PARAM_STD_RE.findall(action_text)}
    # Fill in inline-form params the model emits (e.g. <parameter=command=str_replace>),
    # without overriding a standard-form value if both somehow appear. Strip everything
    # from the first '<' so the malformed hybrid `<parameter=command=str_replace</parameter>`
    # yields `str_replace` instead of `str_replace</parameter` (see _PARAM_INLINE_RE note).
    for k, v in _PARAM_INLINE_RE.findall(action_text):
        params.setdefault(k.strip().lower(), v.split("<")[0].strip())
    return fn, params


def _basename(path: str) -> str:
    return os.path.basename((path or "").strip().rstrip("/"))


def is_repro_path(path: str) -> bool:
    b = _basename(path)
    return bool(b) and bool(_REPRO_BASENAME_RE.search(b))


def is_test_path(path: str) -> bool:
    p = (path or "").lower()
    b = _basename(p)
    return (
        "/tests/" in p
        or "/test/" in p
        or b.startswith("test_")
        or b.endswith("_test.py")
        or b == "conftest.py"
    )


def _bash_script_arg(cmd: str) -> str:
    """First .py argument in a bash command (the script being run)."""
    for tok in (cmd or "").split()[1:]:
        if tok.endswith(".py"):
            return tok
    return ""


def step_transition(fn: str | None, params: dict, observation: str | None = None):
    """Phase this step *transitions to*, or None if ambiguous (inherit current phase).

    Only two unambiguous forward markers exist:
      - reproduction: create/edit or run a reproduce_issue-style script
      - patch:        edit a source file (non-test, non-repro)
    Everything else (view, search, find/grep/cat, pytest, finish) returns None.
    """
    if fn == "file_editor" or fn == "str_replace_editor":
        cmd = (params.get("command") or "").strip().lower()
        path = params.get("path") or params.get("file_path") or ""
        if cmd in _EDIT_COMMANDS:
            if is_repro_path(path):
                return REPRODUCTION
            if is_test_path(path):
                return None  # editing a test: off-task in both stages; inherit rather than mislabel
            if path:
                return PATCH  # editing a source file
        return None  # view / unknown command
    if fn == "execute_bash":
        cmd = params.get("cmd") or params.get("command") or ""
        head = cmd.strip().split()[0] if cmd.strip() else ""
        if head in ("python", "python3"):
            script = _bash_script_arg(cmd)
            if script and is_repro_path(script):
                return REPRODUCTION
        return None  # find/grep/ls/cat/pytest/... -> ambiguous
    return None  # search / finish / unknown


def is_exploration_action(fn: str | None, params: dict) -> bool:
    """True if the step is pure repository exploration (localization work):
    search, file_editor view, or a read-only exploration bash command."""
    if fn == "search":
        return True
    if fn in ("file_editor", "str_replace_editor"):
        return (params.get("command") or "").strip().lower() == "view"
    if fn == "execute_bash":
        cmd = params.get("cmd") or params.get("command") or ""
        head = cmd.strip().split()[0] if cmd.strip() else ""
        return head in _EXPLORE_BASH_HEADS
    return False


class StageStateMachine:
    """Streaming version of the phase machine for online (per-step) classification.

    Feed one step at a time via ``update(action_text, observation)``; it returns the
    step's stage label. Used by the rollout engine to emit per-token stage labels
    aligned with response tokens without a second pass. ``classify_trajectory`` below
    is the batch equivalent and shares this exact logic.
    """

    def __init__(self, backtrack_threshold: int = 0, start_stage: str = LOCALIZATION):
        self.phase = start_stage
        self.backtrack_threshold = int(backtrack_threshold or 0)
        self._explore_run = 0

    def update(self, action_text: str, observation: str | None = None) -> str:
        fn, params = parse_action(action_text)
        target = step_transition(fn, params, observation)
        if target is not None and STAGE_ORDER[target] > STAGE_ORDER[self.phase]:
            self.phase = target
            self._explore_run = 0
        elif (
            self.backtrack_threshold
            and STAGE_ORDER[self.phase] > STAGE_ORDER[LOCALIZATION]
            and is_exploration_action(fn, params)
        ):
            self._explore_run += 1
            if self._explore_run >= self.backtrack_threshold:
                self.phase = LOCALIZATION
                self._explore_run = 0
        else:
            self._explore_run = 0
        return self.phase


def expand_step_labels_to_tokens(step_labels, step_token_lengths):
    """Expand per-step stage labels to a flat per-token label list.

    step_labels[i] applies to all tokens produced at step i (the step's assistant +
    environment tokens as concatenated into the response tensor during trajectory
    transform). step_token_lengths[i] is that step's token count in the response.
    Returns a flat list of length sum(step_token_lengths), aligned with response tokens.
    """
    if len(step_labels) != len(step_token_lengths):
        raise ValueError(
            f"step_labels ({len(step_labels)}) and step_token_lengths "
            f"({len(step_token_lengths)}) length mismatch"
        )
    tokens = []
    for label, n in zip(step_labels, step_token_lengths):
        tokens.extend([label] * int(n))
    return tokens


def classify_trajectory(steps, start_stage: str = LOCALIZATION, backtrack_threshold: int = 0):
    """Label each step with a stage using a phase machine.

    steps: iterable of (action_text, observation_text) or dicts with 'action'/'observation'.
    backtrack_threshold: 0 disables backtracking (forward-only/monotonic). When >0, after
      the phase has advanced beyond localization, N *consecutive* exploration steps move it
      back to localization (the agent has clearly returned to exploring a new area). A single
      view before an edit does not trigger it; forward markers re-advance the phase afterward.
    Returns a list of stage labels (same length as steps).
    """
    sm = StageStateMachine(backtrack_threshold=backtrack_threshold, start_stage=start_stage)
    labels = []
    for step in steps:
        if isinstance(step, dict):
            action_text = step.get("action", "")
            obs = step.get("observation", "")
        else:
            action_text, obs = step
        labels.append(sm.update(action_text, obs))
    return labels
