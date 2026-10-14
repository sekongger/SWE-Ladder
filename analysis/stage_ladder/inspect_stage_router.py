"""Inspect the rule-based stage state machine on real end-to-end rollouts.

Three views, all over trajectories dumped by a training run
(`<run_dir>/chat_completions/<step>_readable.json`):

trace       pick one solved trajectory and print a step-by-step trace -- the action
            taken, the stage the machine assigns, where it transitions
            (localization -> reproduction -> patch), and which privileged evidence
            that step therefore routes to. This is the eyeball check: is each step
            judged the way a reader would judge it?

accuracy    step-level agreement over every trajectory in the dump, cross-checking
            the machine's labels against INDEPENDENT ground-truth signals per step:
              - a create/edit of a reproduce_* script => that step should be
                reproduction or later
              - a successful source-file edit (observation says "has been edited",
                on a non-test / non-repro path) => that step should be patch
            Neither signal is an input to the classifier, so agreement is not
            circular.

trajectory  trajectory-level failure modes, which the step view cannot see: a
            trajectory that successfully edited source but never reached patch
            (false negative), one labelled patch with no edit action at all
            (false positive), and stage flip-flopping under backtracking.

Usage:
  python analysis/stage_ladder/inspect_stage_router.py <run_dir>            # all views
  python analysis/stage_ladder/inspect_stage_router.py <run_dir> --raw      # trace raw tool calls
  python analysis/stage_ladder/inspect_stage_router.py <run_dir> --view accuracy --step 30
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

# The classifier under inspection is the released training module, imported verbatim.
ROOT = os.environ.get("SWE_LADDER_ROOT") or str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from swe_ladder.stage_classifier import (  # noqa: E402
    LOCALIZATION,
    PATCH,
    REPRODUCTION,
    STAGE_ORDER,
    StageStateMachine,
    is_repro_path,
    is_test_path,
    parse_action,
)

SHORT = {LOCALIZATION: "LOC", REPRODUCTION: "REPRO", PATCH: "PATCH"}

# What the stage-routed teacher is allowed to see at each rung (see swe_ladder/staged_privileged.py).
PRIVILEGE = {
    LOCALIZATION: "gold files / line spans / bug lines (to_inspect)",
    REPRODUCTION: "reference test code, test_patch, failing test names",
    PATCH: "golden_patch, the full fix diff",
}


def steps_from_messages(msgs):
    """Pair each assistant action with the observation that follows it."""
    steps = []
    pending = None
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content") or ""
        if role == "assistant":
            pending = content
        elif role in ("tool", "user") and pending is not None:
            steps.append((pending, content))
            pending = None
    if pending is not None:
        steps.append((pending, ""))
    return steps


def action_brief(action_text):
    """One-line summary of the tool call: name, command, and the argument that matters."""
    fn, p = parse_action(action_text)
    if not fn:
        return "(no tool call -- reasoning only)"
    if fn in ("file_editor", "str_replace_editor"):
        cmd = (p.get("command") or "").strip()
        path = p.get("path") or p.get("file_path") or ""
        return f"{fn} {cmd} {os.path.basename(path)}"
    if fn == "execute_bash":
        cmd = (p.get("cmd") or p.get("command") or "").strip().replace("\n", " ")
        return f"execute_bash: {cmd[:70]}"
    if fn == "search":
        return f"search {p.get('search_term', '') or p.get('pattern', '')}"[:80]
    return f"{fn} {str(dict(list(p.items())[:2]))[:60]}"


def action_raw(action_text):
    """The raw <function=...>...</function> block, compacted to one line."""
    m = re.search(r"<function=.*?</function>", action_text, re.S)
    s = m.group(0) if m else "(no tool call)"
    s = re.sub(r"\s+", " ", s)
    # Trim long payloads (old_str/new_str/file_text) so the call shape stays readable.
    s = re.sub(r"(<parameter=(?:file_text|old_str|new_str)[=>])(.{40}).*?(</parameter>|>)", r"\1\2...\3", s)
    return s[:160]


def count_edit_actions(steps):
    """Assistant actions that edit a source file (not a test, not a repro script)."""
    n = 0
    for action, _observation in steps:
        fn, params = parse_action(action)
        if fn not in ("file_editor", "str_replace_editor"):
            continue
        cmd = (params.get("command") or "").strip().lower()
        path = params.get("path") or params.get("file_path") or ""
        if cmd in ("str_replace", "insert", "create") and path \
                and not is_repro_path(path) and not is_test_path(path):
            n += 1
    return n


def rollout_files(run_dir, step=None):
    files = sorted(
        glob.glob(os.path.join(run_dir, "chat_completions", "*_readable.json")),
        key=lambda p: int(os.path.basename(p).split("_")[0]),
    )
    if step:
        files = [f for f in files if os.path.basename(f).startswith(f"{step}_")]
    return files


def labelled_trajectories(files):
    """Yield (file, index, record, steps, labels) for every trajectory in the dump."""
    for f in files:
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        for index, record in enumerate(data):
            steps = steps_from_messages(record.get("messages") or [])
            if not steps:
                continue
            sm = StageStateMachine(backtrack_threshold=3)
            yield f, index, record, steps, [sm.update(a, o) for a, o in steps]


def pick_trajectory(files, *, min_steps, max_steps):
    """Prefer a short solved trajectory that exercises all three rungs; fall back to
    any solved trajectory that reaches patch."""
    fallback = None
    for f, index, record, steps, labels in labelled_trajectories(files):
        if (record.get("traj_score") or 0) < 1:
            continue
        if PATCH not in labels:
            continue
        if fallback is None:
            fallback = (f, index, record, steps, labels)
        if not min_steps <= len(steps) <= max_steps:
            continue
        if not any(parse_action(a)[0] == "search" for a, _ in steps):
            continue
        if {LOCALIZATION, REPRODUCTION, PATCH} <= set(labels):
            return f, index, record, steps, labels
    return fallback


def trace(files, *, raw, min_steps, max_steps):
    picked = pick_trajectory(files, min_steps=min_steps, max_steps=max_steps)
    if picked is None:
        print("no solved trajectory reaching patch_generation in this dump")
        return
    f, index, record, steps, labels = picked
    render = action_raw if raw else action_brief
    print(f"===== {os.path.basename(f)} traj#{index}  score={record.get('traj_score')} "
          f"term={record.get('termination_reason')} steps={len(steps)} =====")
    prev = None
    for i, ((action, _), label) in enumerate(zip(steps, labels), start=1):
        entered = "   <<< enters" if label != prev else ""
        prev = label
        print(f"  step{i:>2} [{SHORT[label]:<5}] {render(action)}{entered}")
        if raw:
            print(f"          routes privileged evidence: {PRIVILEGE[label]}")
    print("  stage distribution: " + ", ".join(
        f"{SHORT[s]}={labels.count(s)} steps" for s in (LOCALIZATION, REPRODUCTION, PATCH)))


def accuracy(files):
    n_traj = 0
    repro_events = repro_correct = 0
    patch_events = patch_correct = 0
    solved = solved_patch = 0
    for _f, _index, record, steps, labels in labelled_trajectories(files):
        n_traj += 1
        if (record.get("traj_score") or 0) >= 1:
            solved += 1
            if PATCH in labels:
                solved_patch += 1
        for (action, observation), label in zip(steps, labels):
            fn, p = parse_action(action)
            path = p.get("path") or p.get("file_path") or ""
            cmd = (p.get("command") or "").strip().lower()
            # Ground truth: the environment confirmed a source edit -> this step is patching.
            edited_ok = (
                "has been edited" in (observation or "").lower()
                and cmd in ("str_replace", "insert", "create")
                and path
                and not is_repro_path(path)
                and not is_test_path(path)
            )
            if edited_ok:
                patch_events += 1
                patch_correct += label == PATCH
            # Ground truth: writing a reproduce_* script -> this step is reproduction or later.
            if (fn in ("file_editor", "str_replace_editor")
                    and cmd in ("create", "str_replace", "insert")
                    and is_repro_path(path)):
                repro_events += 1
                repro_correct += STAGE_ORDER[label] >= STAGE_ORDER[REPRODUCTION]
    print(f"\n===== agreement with independent ground truth ({n_traj} real trajectories) =====")
    print(f"  confirmed source edit        -> labelled PATCH:          "
          f"{patch_correct}/{patch_events} = {100 * patch_correct / max(1, patch_events):.1f}%")
    print(f"  wrote/edited a repro script  -> labelled REPRO or later: "
          f"{repro_correct}/{repro_events} = {100 * repro_correct / max(1, repro_events):.1f}%")
    print(f"  solved trajectories reaching PATCH:                      "
          f"{solved_patch}/{solved} = {100 * solved_patch / max(1, solved):.1f}%")


def trajectory_view(files):
    """Trajectory-level failure modes of the labelling, with the per-trajectory offenders."""
    total = solved = solved_reached_patch = 0
    edited_total = edited_reached_patch = 0
    stage_hist = Counter()
    reached = Counter()
    flipflops = 0
    false_neg = []  # confirmed source edit, yet never labelled patch
    false_pos = []  # labelled patch, yet no source-edit action at all
    for f, index, record, steps, labels in labelled_trajectories(files):
        total += 1
        step_id = os.path.basename(f).split("_")[0]
        stage_hist.update(labels)
        reached[max(STAGE_ORDER[l] for l in labels)] += 1
        reached_patch = PATCH in labels
        score = record.get("traj_score") or 0
        n_edit_actions = count_edit_actions(steps)
        if any("has been edited" in (o or "").lower() for _, o in steps):
            edited_total += 1
            if reached_patch:
                edited_reached_patch += 1
            else:
                false_neg.append((step_id, index, score, len(steps), n_edit_actions))
        if reached_patch and n_edit_actions == 0:
            false_pos.append((step_id, index, score, len(steps)))
        if score > 0:
            solved += 1
            solved_reached_patch += reached_patch
        # Flip-flop: the stage falls back twice or more, which only backtracking allows.
        order = [STAGE_ORDER[l] for l in labels]
        if sum(1 for a, b in zip(order, order[1:]) if b < a) >= 2:
            flipflops += 1

    steps_total = max(1, sum(stage_hist.values()))
    print(f"\n===== trajectory-level view ({total} trajectories, "
          f"{len(files)} rollout dump(s)) =====")
    print("  step distribution (backtrack=3): " + ", ".join(
        f"{SHORT[s]}={stage_hist[s]} ({100 * stage_hist[s] / steps_total:.1f}%)"
        for s in (LOCALIZATION, REPRODUCTION, PATCH)))
    print("  deepest rung reached:            " + ", ".join(
        f"{SHORT[s]}={reached[STAGE_ORDER[s]]} ({100 * reached[STAGE_ORDER[s]] / max(1, total):.1f}%)"
        for s in (LOCALIZATION, REPRODUCTION, PATCH)))
    print(f"  solved trajectories: {solved}, of which reached patch: {solved_reached_patch} "
          f"({100 * solved_reached_patch / max(1, solved):.1f}%)")
    print(f"  trajectories with a confirmed source edit: {edited_total}, of which reached patch: "
          f"{edited_reached_patch} ({100 * edited_reached_patch / max(1, edited_total):.1f}%)")
    print(f"  FALSE NEGATIVES (edited source, never labelled patch): {len(false_neg)}")
    for step_id, index, score, n_steps, n_edits in false_neg[:8]:
        print(f"      step{step_id} traj{index} score={score} steps={n_steps} edit_actions={n_edits}")
    print(f"  FALSE POSITIVES (labelled patch, zero edit actions): {len(false_pos)}")
    for step_id, index, score, n_steps in false_pos[:8]:
        print(f"      step{step_id} traj{index} score={score} steps={n_steps}")
    print(f"  trajectories with >=2 stage drops (flip-flop risk): {flipflops}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", help="training run dir containing chat_completions/")
    parser.add_argument("--view", choices=("all", "trace", "accuracy", "trajectory"), default="all")
    parser.add_argument("--step", help="only read this global step's rollout dump")
    parser.add_argument("--raw", action="store_true",
                        help="trace raw tool-call blocks and the routed privilege, not brief actions")
    parser.add_argument("--min-steps", type=int, default=6, help="shortest trajectory to trace")
    parser.add_argument("--max-steps", type=int, default=14, help="longest trajectory to trace")
    args = parser.parse_args(argv)

    files = rollout_files(args.run_dir, args.step)
    if not files:
        parser.error(f"no *_readable.json under {args.run_dir}/chat_completions")
    if args.view in ("all", "trace"):
        trace(files, raw=args.raw, min_steps=args.min_steps, max_steps=args.max_steps)
    if args.view in ("all", "accuracy"):
        accuracy(files)
    if args.view in ("all", "trajectory"):
        trajectory_view(files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
