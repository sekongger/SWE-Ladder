#!/usr/bin/env python3
"""Validate the stage router against the 119-step human-adjudicated gold set.

The rule router is the component every experiment in this repository depends on:
it decides which rung of the ladder each step belongs to, and therefore which
privileged context the teacher sees. This re-measures its accuracy on
`data/stage_router_gold/gold_119.jsonl` — ten complete trajectories, every step
adjudicated as LOCATE / REPRODUCE / PATCH / VERIFY by hand.

The classifier under test is imported from `swe_ladder/`, so this scores the
released code, not a copy of it. It needs no accelerator, no training stack and
no network.

  python3 analysis/stage_ladder/validate_router_gold.py
  python3 analysis/stage_ladder/validate_router_gold.py --version v1 --with-observations
  python3 analysis/stage_ladder/validate_router_gold.py --expect 98   # exits 1 if it moved

VERIFY maps onto `patch_generation`: a step that re-runs a test after a
successful edit is closing out the patch, and the gold patch is the right
evidence for it. That mapping is part of what is being scored, not a free
parameter — see `swe_ladder/semantic_stage_api.py`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("SWE_LADDER_ROOT") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT))

from swe_ladder.stage_classifier import StageStateMachine  # noqa: E402

GOLD_TO_STAGE = {
    "LOCATE": "causal_localization",
    "REPRODUCE": "localized_issue_reproduction",
    "PATCH": "patch_generation",
    "VERIFY": "patch_generation",
}
STAGES = ("causal_localization", "localized_issue_reproduction", "patch_generation")
SHORT = {"causal_localization": "LOC", "localized_issue_reproduction": "REPRO", "patch_generation": "PATCH"}


def check_demo(path: Path, backtrack: int, version: str) -> int:
    """Recompute the demo trajectory's stored rule labels.

    `demo_trajectory_labeled.json` ships the rule router's own label for each of
    its 23 steps, recorded when the trajectory was adjudicated. Recomputing them
    is a regression test on the released classifier: if this fails, the router
    no longer behaves the way the runs in this repository were trained.
    """
    traj = json.loads(path.read_text(encoding="utf-8"))
    machine = StageStateMachine(backtrack_threshold=backtrack, version=version)
    predicted = [machine.update(s["action_raw"], s.get("observation_raw")) for s in traj["steps"]]
    stored = [s["rule_router_stage"] for s in traj["steps"]]
    gold = [GOLD_TO_STAGE[s["subtask_gold"]] for s in traj["steps"]]
    agree = sum(p == g for p, g in zip(predicted, gold))
    print(f"demo trajectory ({traj['metadata']['instance_id']}, {len(stored)} steps, "
          f"{traj['metadata']['traj_score']:.0f} reward)")
    print(f"  stored rule labels reproduced: {'yes' if predicted == stored else 'NO'}")
    print(f"  agreement with gold: {agree}/{len(gold)} = {agree / len(gold) * 100:.1f}%"
          "   (a PATCH-heavy trajectory: 18 of 23 steps are gold PATCH)")
    if predicted != stored:
        for i, (p, s) in enumerate(zip(predicted, stored)):
            if p != s:
                print(f"  step {i}: stored {SHORT[s]}, now {SHORT[p]}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default=str(ROOT / "data/stage_router_gold/gold_119.jsonl"))
    ap.add_argument("--version", default="v2", choices=["v1", "v2"], help="rule set (training used v2)")
    ap.add_argument("--backtrack", type=int, default=3, help="backtrack_threshold (training used 3)")
    ap.add_argument("--with-observations", action="store_true",
                    help="also feed each step's observation, as the rollout engine does")
    ap.add_argument("--expect", type=int, default=None, help="fail if the number of agreeing steps differs")
    ap.add_argument("--show-errors", action="store_true", help="print each disagreement with its action head")
    ap.add_argument("--demo", action="store_true",
                    help="also recompute the 23-step demo trajectory's stored labels")
    args = ap.parse_args()

    gold_path = Path(args.gold)
    if not gold_path.exists():
        print(f"ERROR: gold set not found: {gold_path}", file=sys.stderr)
        return 2
    trajectories = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    confusion = {(g, p): 0 for g in STAGES for p in STAGES}
    total = agree = 0
    print(f"rule router: version={args.version} backtrack_threshold={args.backtrack} "
          f"observations={'yes' if args.with_observations else 'no'}")
    print(f"{'traj':>4}  {'steps':>5}  {'agree':>5}   {'acc':>6}   issue")
    for traj in trajectories:
        machine = StageStateMachine(backtrack_threshold=args.backtrack, version=args.version)
        hits = 0
        for step in traj["steps"]:
            observation = step.get("observation") if args.with_observations else None
            predicted = machine.update(step["action"], observation)
            expected = GOLD_TO_STAGE[step["gold"]]
            confusion[(expected, predicted)] += 1
            if predicted == expected:
                hits += 1
            elif args.show_errors:
                head = " ".join(step["action"].split())[:90]
                print(f"       step {step['step_idx']:>2}: gold {SHORT[expected]:<5} "
                      f"got {SHORT[predicted]:<5} | {head}")
        total += len(traj["steps"])
        agree += hits
        print(f"{traj['traj']:>4}  {len(traj['steps']):>5}  {hits:>5}   "
              f"{hits / len(traj['steps']) * 100:5.1f}%   {traj.get('issue_title', '')[:60]}")

    print(f"\nagreement: {agree}/{total} = {agree / total * 100:.1f}%")
    print("\nconfusion (rows = gold, cols = rule router)")
    print(f"{'':>7}" + "".join(f"{SHORT[p]:>7}" for p in STAGES))
    for g in STAGES:
        print(f"{SHORT[g]:>7}" + "".join(f"{confusion[(g, p)]:>7}" for p in STAGES))

    if args.expect is not None and agree != args.expect:
        print(f"\nFAIL: expected {args.expect} agreeing steps, got {agree}", file=sys.stderr)
        return 1
    if args.demo:
        print()
        return check_demo(gold_path.with_name("demo_trajectory_labeled.json"), args.backtrack, args.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
