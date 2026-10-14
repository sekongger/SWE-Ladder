from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Iterable

from rllm.agents.system_prompts import SWE_SYSTEM_PROMPT


SCHEMA_VERSION = "staged_opd_sample_v1"
SUPPORTED_STAGES = (
    "causal_localization",
    "localized_issue_reproduction",
    "patch_generation",
)
SUPPORTED_SPLITS = ("train", "valid")


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _compact_reference_test_code(test_code: Any) -> list[dict[str, str]]:
    """Keep only executable/reference code content in teacher hints."""

    if isinstance(test_code, str):
        return [{"code": test_code}] if test_code.strip() else []
    if not isinstance(test_code, list):
        return []

    compact: list[dict[str, str]] = []
    for item in test_code:
        if isinstance(item, dict):
            code = item.get("code")
        else:
            code = item
        if code:
            compact.append({"code": str(code)})
    return compact


def _validate_stage(stage: str) -> None:
    if stage != "all" and stage not in SUPPORTED_STAGES:
        supported = ", ".join((*SUPPORTED_STAGES, "all"))
        raise ValueError(f"Unsupported OPD stage {stage!r}. Supported stages: {supported}")


def _validate_split(split: str) -> None:
    if split not in SUPPORTED_SPLITS:
        supported = ", ".join(SUPPORTED_SPLITS)
        raise ValueError(f"Unsupported OPD split {split!r}. Supported splits: {supported}")


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


def _stage_path(stages_root: str | Path, *, stage: str, split: str) -> Path:
    return Path(stages_root) / stage / f"{split}.jsonl"


def _record_stage(record: dict[str, Any]) -> str:
    stage = record.get("stage")
    if not isinstance(stage, str) or not stage:
        raise ValueError(f"Staged OPD record is missing string field 'stage': {record.get('task_id', '<unknown>')}")
    _validate_stage(stage)
    if stage == "all":
        raise ValueError("Individual staged OPD records cannot use stage='all'")
    return stage


def _student_input(record: dict[str, Any]) -> dict[str, Any]:
    student_input = record.get("student_input")
    if not isinstance(student_input, dict):
        raise ValueError(f"Staged OPD record is missing object field 'student_input': {record.get('task_id', '<unknown>')}")
    return student_input


def _teacher_pi(record: dict[str, Any]) -> dict[str, Any]:
    teacher_pi = record.get("teacher_pi")
    if teacher_pi is None:
        return {}
    if not isinstance(teacher_pi, dict):
        raise ValueError(f"Staged OPD record field 'teacher_pi' must be an object: {record.get('task_id', '<unknown>')}")
    return teacher_pi


def _tool_protocol_prompt() -> str:
    return """We have access to the following functions:

–– BEGIN FUNCTION #1: file_editor ––
Description: Custom editing tool for viewing, creating and editing files.
- command (string, required): one of [view, create, str_replace, insert, undo_edit].
- path (string, required): absolute path to a file or directory, e.g. /testbed/file.py.
- view_range (array, optional): line range for view, e.g. [11, 20] or [11, -1].
- concise (boolean, optional): defaults to True; set False for full content in the requested range.
- file_text, old_str, new_str, insert_line are used only for create/edit commands.
For localization, prefer view on specific files or ranges after search has identified likely targets.
–– END FUNCTION #1 ––

–– BEGIN FUNCTION #2: execute_bash ––
Description: Execute a bash command in the terminal.
- command (string, required): command to execute. Use concise repository-inspection commands such as pwd, ls, find, grep, or sed.
- command can be empty to retrieve logs from a still-running command, or ctrl+c to interrupt it.
–– END FUNCTION #2 ––

–– BEGIN FUNCTION #3: search ––
Description: Search for a term in a directory or single file.
- search_term (string, required): term or string to search for.
- path (string, optional): file or directory to search in. Defaults to . if omitted.
Use search to find symbols, error text, and related callers before opening files.
–– END FUNCTION #3 ––

–– BEGIN FUNCTION #4: finish ––
Description: Finish once the requested artifact is complete.
- command (string, required): currently allowed value is submit.
- result (string, optional): final artifact text.
–– END FUNCTION #4 ––

If you choose to call a function, reply in this XML format and stop immediately after </function>:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
multi-line value
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>.
- Required parameters MUST be specified.
- Only call one function at a time.
- Put reasoning before the function call, and end the response immediately after </function>.
- Do not output any text after </function>.
- Never fabricate tool output text, command output, file contents, or Execution output lines.
- For the first exploration action, prefer execute_bash to inspect the repository.
- Do not call file_editor view on the /testbed directory root as the first step.
"""


def _causal_localization_required_artifact() -> str:
    return (
        "Identify the causal bug location. Return suspected non-test files, hunk or line ranges, "
        "evidence from code inspection, and uncertainty notes. Do not include a final patch.\n\n"
        "Before calling finish, your final artifact must include at least one concrete non-test source file path, "
        "a line or hunk range, and code evidence from a viewed file. Verify that each cited line range is inside "
        "the relevant class or function body rather than a nearby unrelated definition. If evidence is missing, "
        "keep using the tools instead of finalizing."
        "\n\nYour first action must be repository inspection, not finish. Before finalizing, first make at least "
        "one successful non-finish repository inspection call such as search, file_editor view, or execute_bash. "
        "To finalize, call the finish function with command=submit and put the artifact in result: "
        "<function=finish><parameter=command>submit</parameter><parameter=result>...</parameter></function>. "
        "Never use <function=submit>."
    )


def _localized_issue_reproduction_required_artifact() -> str:
    return (
        "Design a localized issue reproduction artifact, not a source patch. Do not edit source files or tests in "
        "this stage, and do not apply patches. Your job is not to repair the bug; your job is to write "
        "reproduce_issue.py so that it exposes the bug. Use source_line_clues as starting points, but ground the "
        "final artifact in repository "
        "evidence you observed through tools.\n\n"
        "Operational constraints:\n"
        "- When you output a tool call, keep the surrounding text short: at most one short sentence explaining the "
        "immediate action. Do not add explanatory paragraphs before or after the XML function call.\n"
        "- source_line_clues are hints only; do not treat them as proof that the reproduction is valid. The "
        "reproduction is valid only if reproduce_issue.py runs and exposes the target failure.\n"
        "- reproduce_issue.py should exit non-zero on the buggy repository when the target issue is present. The "
        "external verifier may apply a repair patch after you finish and rerun the same script; the same script "
        "should exit 0 after that external repair is applied. Do not modify source files to make this happen.\n"
        "- If reproduce_issue.py fails because of syntax, import, path, or setup mistakes unrelated to the target "
        "bug, fix the script and rerun it before finishing.\n\n"
        "Before finish:\n"
        "1. inspect the localized source file with tools;\n"
        "2. create reproduce_issue.py;\n"
        "3. run reproduce_issue.py.\n\n"
        "Before calling finish, run the reproduction script and finish with a short result summary containing only: "
        "script path, command you ran, and the failure or error observed. Do not include a long investigation report."
    )


def _student_system_prompt(stage: str) -> str:
    if stage in {"localized_issue_reproduction", "patch_generation"}:
        return SWE_SYSTEM_PROMPT.strip()

    base = (
        "You are in the causal localization stage. "
        "Your job is to identify the minimal causal source-code location that explains the issue. "
        "Inspect the repository with tools before finalizing; do not assume repo_tree or preloaded code_context is provided. "
        "Every assistant turn must end with exactly one XML function call. Do not write bare commands, fake bash output, "
        "or bare finish text. The first assistant turn must be a non-finish repository inspection call. "
        "Do not state a suspected file, class, function, line, or hunk range before reading a real tool result. "
        "To finalize, call the finish function with command=submit. Never use <function=submit>."
    )
    if stage == "causal_localization":
        return base + " Your objective is to produce a causal localization artifact, not a patch."
    raise ValueError(f"Unsupported OPD stage {stage!r}")


def _source_line_clues_from_localized_files(localized_buggy_files: Any) -> list[dict[str, Any]]:
    """Extract old source lines from localized patch metadata for reproduction prompts."""

    if not isinstance(localized_buggy_files, list):
        return []

    clues: list[dict[str, Any]] = []
    for file_entry in localized_buggy_files:
        if not isinstance(file_entry, dict):
            continue
        file_path = file_entry.get("file")
        old_buggy_lines = file_entry.get("old_buggy_lines")
        if not file_path or not isinstance(old_buggy_lines, list):
            continue
        for hunk_entry in old_buggy_lines:
            if not isinstance(hunk_entry, dict):
                continue
            removed_lines = hunk_entry.get("removed_lines")
            if not removed_lines:
                continue
            clues.append(
                {
                    "file": file_path,
                    "old_start": hunk_entry.get("old_start"),
                    "old_end": hunk_entry.get("old_end"),
                    "removed_lines": removed_lines,
                }
            )
    return clues


def _localized_files_with_removed_lines_only(localized_buggy_files: Any) -> list[dict[str, Any]]:
    """Keep reproduction prompts focused on old changed lines, not hunk coordinates."""

    if not isinstance(localized_buggy_files, list):
        return []

    filtered: list[dict[str, Any]] = []
    for file_entry in localized_buggy_files:
        if not isinstance(file_entry, dict):
            continue
        file_path = file_entry.get("file")
        if not file_path:
            continue
        removed_lines: list[str] = []
        old_buggy_lines = file_entry.get("old_buggy_lines")
        if isinstance(old_buggy_lines, list):
            for hunk_entry in old_buggy_lines:
                if not isinstance(hunk_entry, dict):
                    continue
                lines = hunk_entry.get("removed_lines")
                if isinstance(lines, list):
                    removed_lines.extend(str(line) for line in lines)
        filtered.append({"file": file_path, "removed_lines": removed_lines})
    return filtered


def build_student_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """Build stage-aware student messages from record.student_input only."""

    stage = _record_stage(record)
    student_input = copy.deepcopy(_student_input(record))
    issue = _coerce_text(student_input.get("issue_statement"))
    hints = _coerce_text(student_input.get("hints_text"))

    common = [
        f"## Stage\n{stage}",
        "## Issue statement\n" + issue,
    ]

    if stage == "causal_localization":
        instructions = "## Required artifact\n" + _causal_localization_required_artifact()
        body = "\n\n".join([*common, instructions])
    elif stage == "localized_issue_reproduction":
        common.append("## Hints\n" + hints)
        localized_buggy_files = student_input.get("localized_buggy_files", [])
        source_line_clues = _json_block(_source_line_clues_from_localized_files(localized_buggy_files))
        instructions = "## Required artifact\n" + _localized_issue_reproduction_required_artifact()
        body = "\n\n".join(
            [
                *common,
                (
                    "## source_line_clues\n"
                    "These are old source lines from the buggy version that were later changed. "
                    "Use them to identify behavior that the reproduction script should exercise; "
                    "they are not observed evidence by themselves. Verify the surrounding source with tools "
                    "before writing reproduce_issue.py.\n"
                    + source_line_clues
                ),
                instructions,
            ]
        )
    elif stage == "patch_generation":
        common.append("## Hints\n" + hints)
        localized = _json_block(student_input.get("localized_buggy_files", []))
        instructions = (
            "## Required artifact\n"
            "Generate a minimal source patch, explain the patch rationale, and provide a verification plan. "
            "Use the issue statement, hints, and localized_buggy_files as repair guidance. "
            "Do not edit tests or assume any hidden test content."
        )
        body = "\n\n".join(
            [
                *common,
                "## localized_buggy_files\n" + localized,
                instructions,
            ]
        )
    else:
        raise ValueError(f"Unsupported OPD stage {stage!r}")

    return [{"role": "system", "content": _student_system_prompt(stage)}, {"role": "user", "content": body}]


def _verified_repository_evidence(stage: str, teacher_pi: dict[str, Any]) -> dict[str, Any]:
    # Process-graph mode (OPD_TEACHER_EVIDENCE_MODE=graph): the per-stage
    # privileged block becomes the stage-visible subgraph of the instance's
    # latent process graph (P2T-style G*), with a monotone privilege ladder --
    # localization sees facts only, reproduction adds repro milestones, patch
    # sees the full chain incl. byte-exact edits. Falls back to the flat
    # evidence below when the mode is off or no graph is attached.
    from .process_graph_evidence import maybe_graph_evidence

    graph_evidence = maybe_graph_evidence(stage, teacher_pi)
    if graph_evidence is not None:
        return graph_evidence
    if stage == "causal_localization":
        return {
            "candidate_non_test_source_files_to_inspect": teacher_pi.get("gold_changed_non_test_files", []),
            "candidate_hunk_ranges_to_inspect": teacher_pi.get("gold_hunk_ranges", []),
            "candidate_modified_source_segments_to_inspect": teacher_pi.get("old_buggy_lines", []),
        }
    if stage == "localized_issue_reproduction":
        evidence: dict[str, Any] = {}
        test_code = teacher_pi.get("fail_to_pass_test_code") or teacher_pi.get("test_code") or []
        compact_test_code = _compact_reference_test_code(test_code)
        if compact_test_code:
            evidence["reference_test_code_to_adapt_into_reproduction"] = compact_test_code

        test_change_diff = teacher_pi.get("fail_to_pass_tests", "") or teacher_pi.get("test_patch", "")
        failing_test_names = teacher_pi.get("fail_to_pass_test_names", [])
        failing_test_ids = teacher_pi.get("fail_to_pass_test_ids", [])
        if test_change_diff:
            evidence["candidate_reproduction_test_changes_to_consider"] = test_change_diff
        if failing_test_ids:
            evidence["candidate_failing_test_ids_to_consider"] = failing_test_ids
        if failing_test_names:
            evidence["candidate_failing_test_names_to_consider"] = failing_test_names
        if not evidence:
            evidence["candidate_reproduction_test_changes_to_consider"] = ""
        return evidence
    if stage == "patch_generation":
        return {
            "verified_repair_diff": teacher_pi.get("golden_patch", ""),
        }
    raise ValueError(f"Unsupported OPD stage {stage!r}")


def _prompt_task_context(student_input: dict[str, Any]) -> dict[str, Any]:
    """Return task context suitable for model-visible prompts.

    Repository checkout metadata is kept in extra_info for the environment, not
    shown to the model as a pseudo-tool or prompt field.
    """

    context = copy.deepcopy(student_input)
    context.pop("codebase_access", None)
    return context


def _teacher_stage_instruction(stage: str) -> str:
    if stage == "causal_localization":
        return _causal_localization_required_artifact()
    if stage == "localized_issue_reproduction":
        return _localized_issue_reproduction_required_artifact()
    if stage == "patch_generation":
        return (
            "Generate a minimal source patch. Use the verified repair diff as repository evidence for the intended "
            "fix, then provide the minimal repair, rationale, and verification plan."
        )
    raise ValueError(f"Unsupported OPD stage {stage!r}")


def _teacher_evidence_heading(stage: str) -> str:
    if stage == "causal_localization":
        return "Code context hints"
    if stage == "localized_issue_reproduction":
        return "Reproduction starting points"
    if stage == "patch_generation":
        return "Verified repository evidence"
    raise ValueError(f"Unsupported OPD stage {stage!r}")


def _append_reproduction_hints(student_user: str, evidence: dict[str, Any]) -> str:
    hints_marker = "## Hints\n"
    next_section_marker = "\n\n## source_line_clues"
    if hints_marker not in student_user or next_section_marker not in student_user:
        return student_user + "\n\n## Hints\nAdditional reproduction hints:\n" + _json_block(evidence)

    before_hints, hints_and_after = student_user.split(hints_marker, 1)
    hints_text, after_hints = hints_and_after.split(next_section_marker, 1)
    hints_text = hints_text.rstrip()
    addition = "Additional reproduction hints:\n" + _json_block(evidence)
    merged_hints = f"{hints_text}\n\n{addition}" if hints_text else addition
    return before_hints + hints_marker + merged_hints + next_section_marker + after_hints


def build_teacher_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """Build solve-oriented messages from the task context and verified repository evidence."""

    stage = _record_stage(record)
    teacher_pi = copy.deepcopy(_teacher_pi(record))
    evidence = _verified_repository_evidence(stage, teacher_pi)
    student_messages = build_student_messages(record)
    student_system = student_messages[0]["content"].strip()
    student_user = student_messages[1]["content"].strip()
    if "We have access to the following functions" in student_system:
        system = student_system
    else:
        system = f"{student_system}\n\n{SWE_SYSTEM_PROMPT}" if student_system else SWE_SYSTEM_PROMPT
    if stage == "localized_issue_reproduction":
        user = _append_reproduction_hints(student_user, evidence)
    else:
        user = (
            student_user
            + "\n\n"
            + f"## {_teacher_evidence_heading(stage)}\n"
            + _json_block(evidence)
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_staged_opd_sample(record: dict[str, Any], *, split: str) -> dict[str, Any]:
    _validate_split(split)
    stage = _record_stage(record)
    student_input = copy.deepcopy(_student_input(record))
    teacher_pi = copy.deepcopy(_teacher_pi(record))
    task_id = record.get("task_id") or record.get("instance_id") or ""
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "stage": stage,
        "split": split,
        "student_input": student_input,
        "teacher_pi": teacher_pi,
        "raw_record": copy.deepcopy(record),
        "student_messages": build_student_messages(record),
        "teacher_messages": build_teacher_messages(record),
        "privileged_context": {
            "student_input": student_input,
            "teacher_pi": teacher_pi,
        },
        "metadata": {
            "codebase_access": copy.deepcopy(student_input.get("codebase_access", {})),
        },
    }


def load_staged_opd_dataset(
    stages_root: str | Path,
    *,
    stage: str = "all",
    split: str = "train",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    _validate_stage(stage)
    _validate_split(split)
    stages = SUPPORTED_STAGES if stage == "all" else (stage,)
    samples: list[dict[str, Any]] = []
    for stage_name in stages:
        path = _stage_path(stages_root, stage=stage_name, split=split)
        if not path.exists():
            raise FileNotFoundError(f"Missing staged OPD dataset file: {path}")
        for record in iter_jsonl(path):
            if limit is not None and len(samples) >= limit:
                return samples
            samples.append(build_staged_opd_sample(record, split=split))
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build staged OPD samples from staged coding-agent JSONL data.")
    parser.add_argument("--stages-root", default="stages", help="Root directory containing <stage>/<split>.jsonl files.")
    parser.add_argument("--stage", choices=[*SUPPORTED_STAGES, "all"], default="all", help="Stage to load.")
    parser.add_argument("--split", choices=SUPPORTED_SPLITS, default="train", help="Dataset split to load.")
    parser.add_argument("--output", required=True, help="Output staged OPD samples JSONL.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of output samples.")
    args = parser.parse_args(argv)

    records = load_staged_opd_dataset(
        args.stages_root,
        stage=args.stage,
        split=args.split,
        limit=args.limit,
    )
    write_jsonl(args.output, records)
    print(f"wrote {len(records)} staged OPD sample(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
