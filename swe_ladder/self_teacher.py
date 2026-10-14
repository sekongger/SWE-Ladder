from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "opd_self_teacher_v1"
TOPK_REQUEST_SCHEMA_VERSION = "opd_teacher_topk_request_v1"

DEFAULT_LOSS_MASK_POLICY: dict[str, Any] = {
    "mask_prompt": True,
    "mask_observations": True,
    "mask_tool_wrappers": True,
    "mask_special_tokens": True,
    "train_assistant_thoughts": True,
    "train_assistant_actions": True,
    "masked_text_markers": [
        "Generic sandbox policy violation",
        "[STDOUT]",
        "[STDERR]",
        "<tool_response>",
        "</tool_response>",
    ],
}

_PRIVILEGED_FIELDS = ("patch", "test_patch", "FAIL_TO_PASS")


class MissingPrivilegedContext(ValueError):
    """Raised when a trajectory cannot produce a privileged teacher record."""


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _coerce_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    return [value]


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return text[:max_chars] + f"\n...[truncated {omitted} chars]"


def _task_from_record(record: dict[str, Any]) -> dict[str, Any]:
    for key in ("ds", "task"):
        value = record.get(key)
        if isinstance(value, dict):
            return value
    return record


def _trajectory_steps_from_record(record: dict[str, Any]) -> list[Any]:
    steps = record.get("trajectory_steps")
    if isinstance(steps, list):
        return steps

    trajectory = record.get("trajectory")
    if isinstance(trajectory, dict):
        nested_steps = trajectory.get("steps")
        if isinstance(nested_steps, list):
            return nested_steps

    steps = record.get("steps")
    if isinstance(steps, list):
        return steps
    return []


def _step_field(step: Any, name: str) -> str:
    if isinstance(step, dict):
        return _coerce_text(step.get(name))
    return _coerce_text(getattr(step, name, ""))


def summarize_trajectory_steps(steps: Iterable[Any], *, max_chars: int = 6000) -> str:
    lines: list[str] = []
    for index, step in enumerate(steps):
        if isinstance(step, dict):
            step_id = step.get("step_idx", step.get("step_count", index))
        else:
            step_id = getattr(step, "step_idx", getattr(step, "step_count", index))
        thought = _step_field(step, "thought")
        action = _step_field(step, "action")
        observation = _step_field(step, "observation")
        lines.append(f"Step {step_id}:")
        if thought:
            lines.append(f"Thought: {thought}")
        if action:
            lines.append(f"Action: {action}")
        if observation:
            lines.append(f"Observation: {_truncate(observation, 1000)}")
    return _truncate("\n".join(lines), max_chars)


def _append_trainable_segment(parts: list[str], spans: list[dict[str, Any]], *, segment_type: str, text: str, step_idx: Any = None) -> None:
    if not text:
        return
    if parts:
        parts.append("\n\n")
    start = sum(len(part) for part in parts)
    parts.append(text)
    end = start + len(text)
    span: dict[str, Any] = {
        "start": start,
        "end": end,
        "type": segment_type,
        "trainable": True,
    }
    if step_idx is not None:
        span["step_idx"] = step_idx
    spans.append(span)


def build_student_target(trajectory_record: dict[str, Any], *, max_patch_chars: int = 12000) -> dict[str, Any]:
    """Build the assistant-only text that teacher-forced top-k extraction scores."""

    parts: list[str] = []
    spans: list[dict[str, Any]] = []
    for index, step in enumerate(_trajectory_steps_from_record(trajectory_record)):
        if isinstance(step, dict):
            step_id = step.get("step_idx", step.get("step_count", index))
        else:
            step_id = getattr(step, "step_idx", getattr(step, "step_count", index))
        thought = _step_field(step, "thought")
        action = _step_field(step, "action")
        if thought:
            _append_trainable_segment(
                parts,
                spans,
                segment_type="thought",
                step_idx=step_id,
                text=f"Step {step_id} Thought:\n{thought}",
            )
        if action:
            _append_trainable_segment(
                parts,
                spans,
                segment_type="action",
                step_idx=step_id,
                text=f"Step {step_id} Action:\n{action}",
            )

    output_patch = _coerce_text(trajectory_record.get("output_patch"))
    if output_patch:
        _append_trainable_segment(
            parts,
            spans,
            segment_type="output_patch",
            text="Final output patch:\n" + _truncate(output_patch, max_patch_chars),
        )
    return {"text": "".join(parts), "trainable_spans": spans}


def _validate_privileged_context(task: dict[str, Any]) -> None:
    missing = [field for field in _PRIVILEGED_FIELDS if not task.get(field)]
    if missing:
        instance_id = task.get("instance_id") or task.get("uid") or "<unknown>"
        raise MissingPrivilegedContext(f"{instance_id} is missing privileged teacher fields: {missing}")


def _build_teacher_messages(
    *,
    student_visible: dict[str, Any],
    teacher_only: dict[str, Any],
    max_patch_chars: int,
) -> list[dict[str, str]]:
    system = (
        "You are a coding agent working on a repository task. "
        "Solve the problem directly using the task, prior attempt, and verified repair evidence below. "
        "Focus on repair-relevant next actions and patch-oriented reasoning, not on running tests or creating reproduce scripts."
    )
    user = "\n\n".join(
        [
            "## Task",
            f"Instance: {student_visible['instance_id']}",
            f"Repository: {student_visible['repo']}",
            f"Base commit: {student_visible.get('base_commit', '')}",
            "Problem statement:\n" + student_visible.get("problem_statement", ""),
            "Hints:\n" + student_visible.get("hints_text", ""),
            "Prior attempt summary:\n" + student_visible.get("trajectory_summary", ""),
            "Prior output patch:\n" + _truncate(student_visible.get("output_patch", ""), max_patch_chars),
            "## Verified repair evidence",
            "Reference source diff:\n" + _truncate(teacher_only.get("golden_patch", ""), max_patch_chars),
            "Reference test diff:\n" + _truncate(teacher_only.get("test_patch", ""), max_patch_chars),
            "FAIL_TO_PASS:\n" + json.dumps(teacher_only.get("fail_to_pass", []), ensure_ascii=False, indent=2),
            "PASS_TO_PASS:\n" + json.dumps(teacher_only.get("pass_to_pass", []), ensure_ascii=False, indent=2),
            (
                "## Required artifact\n"
                "Use the evidence to continue toward the minimal source-code repair path. "
                "Do not suggest executing Python, pytest, package managers, "
                "or ad-hoc reproduce/test files because the generic sandbox is non-executable."
            ),
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_self_teacher_record(
    trajectory_record: dict[str, Any],
    *,
    max_trajectory_chars: int = 6000,
    max_patch_chars: int = 12000,
) -> dict[str, Any]:
    task = _task_from_record(trajectory_record)
    _validate_privileged_context(task)

    instance_id = task.get("instance_id") or task.get("uid") or trajectory_record.get("instance_id") or ""
    repo = task.get("repo") or trajectory_record.get("repo") or ""
    steps = _trajectory_steps_from_record(trajectory_record)
    student_visible = {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": task.get("base_commit") or task.get("commit_hash") or "",
        "problem_statement": task.get("problem_statement") or trajectory_record.get("problem_statement") or "",
        "hints_text": task.get("hints_text") or "",
        "trajectory_summary": summarize_trajectory_steps(steps, max_chars=max_trajectory_chars),
        "output_patch": trajectory_record.get("output_patch") or "",
        "exit_reason": trajectory_record.get("exit_reason") or trajectory_record.get("termination_reason") or "",
        "reward": trajectory_record.get("reward", 0.0),
    }
    teacher_only = {
        "golden_patch": task.get("patch") or "",
        "test_patch": task.get("test_patch") or "",
        "fail_to_pass": _coerce_list(task.get("FAIL_TO_PASS")),
        "pass_to_pass": _coerce_list(task.get("PASS_TO_PASS")),
        "run_tests": task.get("run_tests") or "",
    }
    student_target = build_student_target(trajectory_record, max_patch_chars=max_patch_chars)
    return {
        "schema_version": SCHEMA_VERSION,
        "instance_id": instance_id,
        "repo": repo,
        "source_exp_name": trajectory_record.get("exp_name", ""),
        "student_visible": student_visible,
        "student_target": student_target,
        "teacher_only": teacher_only,
        "teacher_messages": _build_teacher_messages(
            student_visible=student_visible,
            teacher_only=teacher_only,
            max_patch_chars=max_patch_chars,
        ),
        "loss_mask_policy": copy.deepcopy(DEFAULT_LOSS_MASK_POLICY),
        "metadata": {
            "source_exit_reason": student_visible["exit_reason"],
            "source_reward": student_visible["reward"],
            "num_trajectory_steps": len(steps),
        },
    }


def build_teacher_topk_request(
    self_teacher_record: dict[str, Any],
    *,
    topk_k: int = 32,
    tokenizer_path: str = "",
) -> dict[str, Any]:
    if topk_k <= 0:
        raise ValueError(f"topk_k must be positive, got {topk_k}")

    student_target = self_teacher_record.get("student_target") or {}
    target_text = student_target.get("text") or ""
    trainable_spans = student_target.get("trainable_spans") or []
    if not target_text:
        raise ValueError(f"{self_teacher_record.get('instance_id', '<unknown>')} has empty student target text")

    return {
        "schema_version": TOPK_REQUEST_SCHEMA_VERSION,
        "source_schema_version": self_teacher_record.get("schema_version", ""),
        "instance_id": self_teacher_record.get("instance_id", ""),
        "repo": self_teacher_record.get("repo", ""),
        "topk_k": topk_k,
        "tokenizer_path": tokenizer_path,
        "teacher_messages": self_teacher_record.get("teacher_messages", []),
        "target_text": target_text,
        "trainable_spans": copy.deepcopy(trainable_spans),
        "loss_mask_policy": copy.deepcopy(self_teacher_record.get("loss_mask_policy", DEFAULT_LOSS_MASK_POLICY)),
        "metadata": copy.deepcopy(self_teacher_record.get("metadata", {})),
    }


def build_self_teacher_records(
    trajectory_records: Iterable[dict[str, Any]],
    *,
    skip_missing_privileged: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for trajectory_record in trajectory_records:
        if limit is not None and len(records) >= limit:
            break
        try:
            records.append(build_self_teacher_record(trajectory_record))
        except MissingPrivilegedContext:
            if not skip_missing_privileged:
                raise
    return records


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build OPD privileged self-teacher query records from SWE trajectory JSONL.")
    parser.add_argument("--trajectories", required=True, help="Input trajectory JSONL.")
    parser.add_argument("--output", required=True, help="Output teacher-query JSONL.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of output records.")
    parser.add_argument("--emit", choices=["self_teacher", "topk_request"], default="self_teacher", help="Output schema to write.")
    parser.add_argument("--topk-k", type=int, default=32, help="Top-k size when --emit=topk_request.")
    parser.add_argument("--tokenizer-path", default="", help="Student tokenizer path to store in top-k requests.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of skipping rows that do not contain privileged SWE fields.",
    )
    args = parser.parse_args(argv)

    records = build_self_teacher_records(
        iter_jsonl(args.trajectories),
        skip_missing_privileged=not args.strict,
        limit=args.limit,
    )
    if args.emit == "topk_request":
        records = [
            build_teacher_topk_request(
                record,
                topk_k=args.topk_k,
                tokenizer_path=args.tokenizer_path,
            )
            for record in records
        ]
    write_jsonl(args.output, records)
    print(f"wrote {len(records)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
