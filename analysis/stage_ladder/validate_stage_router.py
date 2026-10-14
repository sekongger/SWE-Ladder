"""Validate the rule-based stage classifier against known single-stage rollouts.

Ground truth: each staged run was trained on ONE stage's task, and the staged prompt
forbids cross-stage actions. So we expect:
  - location run:      steps stay in causal_localization (no repro, no source edit)
  - reproduction run:  reaches localized_issue_reproduction, NEVER patch_generation
  - patch run:         reaches patch_generation
Early exploration in reproduction/patch tasks is legitimately localization, so we do
NOT require 100% of steps to equal the task stage. The meaningful metrics are:
  - false transitions (location->repro/patch, reproduction->patch)  [precision]
  - recall of the target transition (reproduction reaches repro, patch reaches patch)

This is the silver-label check: ground truth is the run's own task stage, so it
covers every step of every rollout but cannot see within-trajectory nuance. The
gold-label check (119 hand-labelled steps) is reported in the top-level README.

Usage:
  python analysis/stage_ladder/validate_stage_router.py \
      --loc-run   <train_output>/<loc_run>/chat_completions \
      --repro-run <train_output>/<repro_run>/chat_completions \
      --patch-run <train_output>/<patch_run>/chat_completions
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

# The classifier under test is the released training module, imported verbatim.
ROOT = os.environ.get("SWE_LADDER_ROOT") or str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from swe_ladder.stage_classifier import (  # noqa: E402
    LOCALIZATION,
    REPRODUCTION,
    PATCH,
    classify_trajectory,
)


def steps_from_chat_completion(cc):
    """Pair assistant actions with the following tool observation."""
    steps = []
    pending = None
    for m in cc:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = str(m.get("content", ""))
        if role == "assistant":
            if pending is not None:
                steps.append((pending, ""))
            pending = content
        elif role == "tool":
            if pending is not None:
                steps.append((pending, content))
                pending = None
    if pending is not None:
        steps.append((pending, ""))
    return steps


def load_trajectories(cc_dir, max_files=12):
    trajs = []
    for f in sorted(glob.glob(os.path.join(cc_dir, "*.jsonl")))[:max_files]:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                cc = r.get("chat_completion") or r.get("messages")
                if cc:
                    trajs.append(steps_from_chat_completion(cc))
    return trajs


def summarize(gt_stage, trajs):
    step_labels = Counter()
    terminal = Counter()
    total_steps = 0
    reached_repro = reached_patch = 0
    for steps in trajs:
        if not steps:
            continue
        labels = classify_trajectory(steps)
        step_labels.update(labels)
        total_steps += len(labels)
        terminal[labels[-1]] += 1
        reached_repro += 1 if REPRODUCTION in labels else 0
        reached_patch += 1 if PATCH in labels else 0
    n = len([t for t in trajs if t])
    print(f"\n===== ground-truth stage: {gt_stage}  (trajectories={n}, steps={total_steps}) =====")
    print("  step label share: " + ", ".join(
        f"{'loc' if k == LOCALIZATION else k.split('_')[0]}={step_labels.get(k, 0) / max(total_steps, 1):.1%}"
        for k in (LOCALIZATION, REPRODUCTION, PATCH)))
    print("  terminal label:   " + ", ".join(f"{k}={terminal.get(k, 0)}" for k in (LOCALIZATION, REPRODUCTION, PATCH)))
    print(f"  reached reproduction: {reached_repro}/{n} = {reached_repro / max(n, 1):.1%}")
    print(f"  reached patch:        {reached_patch}/{n} = {reached_patch / max(n, 1):.1%}")
    return reached_repro, reached_patch, n


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--loc-run", default=os.environ.get("STAGE_RUN_LOC", ""),
                        help="chat_completions dir of a LOC-stage run")
    parser.add_argument("--repro-run", default=os.environ.get("STAGE_RUN_REPRO", ""),
                        help="chat_completions dir of a REPRO-stage run")
    parser.add_argument("--patch-run", default=os.environ.get("STAGE_RUN_PATCH", ""),
                        help="chat_completions dir of a PATCH-stage run")
    parser.add_argument("--max-files", type=int, default=12,
                        help="rollout files to read per run (one file per global step)")
    args = parser.parse_args(argv)

    runs = {LOCALIZATION: args.loc_run, REPRODUCTION: args.repro_run, PATCH: args.patch_run}
    missing = [stage for stage, path in runs.items() if not path or not os.path.isdir(path)]
    if missing:
        parser.error("missing rollout dir for: " + ", ".join(missing))

    results = {}
    for gt, cc_dir in runs.items():
        results[gt] = summarize(gt, load_trajectories(cc_dir, max_files=args.max_files))

    print("\n\n########## headline correctness ##########")
    rr, rp, n = results[LOCALIZATION]
    print(f"[location run]     false reproduction: {rr}/{n}={rr / max(n, 1):.1%}   "
          f"false patch: {rp}/{n}={rp / max(n, 1):.1%}   (both should be ~0)")
    rr, rp, n = results[REPRODUCTION]
    print(f"[reproduction run] reproduction recall: {rr}/{n}={rr / max(n, 1):.1%} (should be high)   "
          f"false patch: {rp}/{n}={rp / max(n, 1):.1%} (should be ~0)")
    rr, rp, n = results[PATCH]
    print(f"[patch run]        patch recall: {rp}/{n}={rp / max(n, 1):.1%} (should be high)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
