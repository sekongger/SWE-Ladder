#!/usr/bin/env python3
"""Rebuild gold_119.jsonl from the training rollout dumps it came from.

The 119-step gold set is ten complete agent trajectories whose every step was
hand-adjudicated into one of LOCATE / REPRODUCE / PATCH / VERIFY. The labels were
recorded separately from the trajectories (as `subtask_gold_10traj.json`: a file,
a line number and a label sequence per trajectory); this script joins the two and
writes the self-contained `gold_119.jsonl` that ships in this directory.

It is here for provenance, not for you to run: the dumps it reads are per-step
rollout logs from one of our training runs, tens of GB that this repository does
not ship. `gold_119.jsonl` is its output and is all the validator needs.

  python3 data/stage_router_gold/extract_gold_from_dumps.py \
      --dumps <run>/chat_completions --labels subtask_gold_10traj.json \
      -o data/stage_router_gold/gold_119.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Both routers read at most 200 characters of an observation (see
# swe_ladder.semantic_stage_api.build_window), so capping the stored text cannot
# change a label; it keeps the shipped file readable and small.
OBS_CAP = 4000
ISSUE_RE = re.compile(r"<issue_description>\s*(.+?)\s*$", re.S | re.M)


def issue_title(messages: list[dict]) -> str:
    """First line of the issue, as human-readable provenance for a trajectory."""
    for message in messages:
        if message.get("role") != "user":
            continue
        match = ISSUE_RE.search(str(message.get("content", "")))
        if match:
            for line in match.group(1).splitlines():
                line = line.strip().lstrip("#").strip()
                if line:
                    return line[:200]
    return ""


def read_line(path: Path, index: int) -> dict:
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i == index:
                return json.loads(line)
    raise IndexError(f"{path} has no line {index}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps", required=True, help="a run's chat_completions/ directory")
    ap.add_argument("--labels", required=True, help="per-trajectory gold label sequences")
    ap.add_argument("--run-name", default="", help="human-readable run id, stored as provenance")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    dumps = Path(args.dumps)
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    out: list[dict] = []

    for entry in labels:
        record = read_line(dumps / entry["file"], entry["line"])
        messages = record["chat_completion"]
        actions = [m for m in messages if m.get("role") == "assistant"]
        gold = entry["gold"]
        if len(actions) != len(gold):
            raise SystemExit(
                f"trajectory {entry['traj']}: {len(actions)} actions but {len(gold)} gold labels"
            )

        steps = []
        for i, (action, label) in enumerate(zip(actions, gold)):
            position = messages.index(action)
            following = messages[position + 1] if position + 1 < len(messages) else {}
            observation = str(following.get("content", "")) if following.get("role") in ("tool", "user") else ""
            steps.append({
                "step_idx": i,
                "gold": label,
                "action": str(action.get("content", "")),
                "observation": observation[:OBS_CAP],
                "observation_truncated": len(observation) > OBS_CAP,
            })

        out.append({
            "traj": entry["traj"],
            "n_steps": len(steps),
            "traj_score": record.get("traj_score"),
            "termination_reason": record.get("termination_reason"),
            "issue_title": issue_title(messages),
            "source": {"run": args.run_name, "dump": entry["file"], "line": entry["line"]},
            "steps": steps,
        })

    output = Path(args.output)
    with output.open("w", encoding="utf-8") as handle:
        for record in out:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {output}: {len(out)} trajectories, {sum(r['n_steps'] for r in out)} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
