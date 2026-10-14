"""Run the issue-reproduction verifier against a container the model already acted in.

The REPRODUCE rung is scored by executing the agent's `reproduce_issue.py` twice —
on the buggy tree and on the gold-patched tree — and requiring fail-then-pass
(`swe_ladder/staged_eval.py`, mirroring the in-training reward in
`rllm/environments/swe/issue_reproduction_verifier.py`). This runs that check by
hand on one eval record, which is how to debug a stage whose reward looks wrong.

  python3 scripts/data/run_issue_reproduction_verifier.py \\
      --eval-jsonl eval_data/localized_issue_reproduction/valid_eval.jsonl \\
      --container <container-after-rollout> --output /tmp/verifier.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# See the note in build_staged_verl_dataset.py: `swe_ladder` here, `rllm.opd` once installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from swe_ladder.staged_eval import run_issue_reproduction_verifier_in_container
except ModuleNotFoundError:  # running from inside an installed workspace
    from rllm.opd.staged_eval import run_issue_reproduction_verifier_in_container


def _docker_client() -> Any:
    import docker

    return docker.from_env()


def load_eval_record(path: str | Path, *, index: int = 0, task_id: str | None = None) -> dict[str, Any]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            if task_id is not None:
                if str(record.get("task_id") or "") == task_id:
                    return record
                continue
            if row_index == index:
                return record
    selector = f"task_id={task_id}" if task_id is not None else f"index={index}"
    raise ValueError(f"No eval record found in {input_path} for {selector}")


def run_record_in_existing_container(record: dict[str, Any], *, container_name: str) -> dict[str, Any]:
    verifier = record.get("verifier")
    if not isinstance(verifier, dict):
        raise ValueError(f"Eval record {record.get('task_id', '<unknown>')} is missing verifier metadata")
    if verifier.get("kind") != "issue_reproduction_buggy_then_gold_patch":
        raise ValueError(f"Unsupported verifier kind: {verifier.get('kind')!r}")

    client = _docker_client()
    container = client.containers.get(container_name)
    result = run_issue_reproduction_verifier_in_container(container, verifier)
    return {
        "task_id": record.get("task_id"),
        "stage": record.get("stage"),
        "verifier_kind": verifier.get("kind"),
        **result,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a staged issue-reproduction verifier in an existing Docker container."
    )
    parser.add_argument("--eval-jsonl", required=True, help="Evaluation JSONL containing verifier metadata.")
    parser.add_argument("--container", required=True, help="Existing Docker container id or name after model rollout.")
    parser.add_argument("--index", type=int, default=0, help="Record index to evaluate when --task-id is omitted.")
    parser.add_argument("--task-id", default=None, help="Task id to evaluate.")
    parser.add_argument("--output", required=True, help="Output JSON path for verifier result.")
    args = parser.parse_args(argv)

    record = load_eval_record(args.eval_jsonl, index=args.index, task_id=args.task_id)
    result = run_record_in_existing_container(record, container_name=args.container)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
