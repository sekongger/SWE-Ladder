from __future__ import annotations

import json
import re
import io
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EVAL_STAGES = (
    "causal_localization",
    "localized_issue_reproduction",
    "patch_generation",
)


@dataclass(frozen=True)
class EvalLineRange:
    file: str
    start: int
    end: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "file", normalize_repo_path(self.file))
        start = int(self.start)
        end = int(self.end)
        if end < start:
            start, end = end, start
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)


@dataclass(frozen=True)
class IssueReproductionVerifierSpec:
    script_path: str
    timeout: int
    golden_patch: str
    test_patch: str
    fail_to_pass_test_ids: tuple[str, ...]

    @property
    def has_gold_patch(self) -> bool:
        return bool(self.golden_patch.strip())


def normalize_repo_path(path: str) -> str:
    path = str(path).strip().strip("`'\"")
    path = path.removeprefix("/testbed/")
    path = path.removeprefix("testbed/")
    path = path.removeprefix("./")
    return path


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _sanitize_public_text(text: str) -> str:
    """Remove pasted patch snippets from model-visible eval prompts."""

    return re.sub(
        r"```[^\n`]*\n(?:(?!```).)*diff --git(?:(?!```).)*```",
        "```text\n[diff snippet omitted for evaluation]\n```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )


def _json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _student_input(sample: dict[str, Any]) -> dict[str, Any]:
    value = sample.get("student_input", sample)
    return value if isinstance(value, dict) else {}


def _issue_statement(sample: dict[str, Any]) -> str:
    student = _student_input(sample)
    return _sanitize_public_text(
        _coerce_text(
        student.get("issue_statement")
        or sample.get("problem_statement")
        or sample.get("issue_statement")
        or ""
        )
    )


def _hints(sample: dict[str, Any]) -> str:
    student = _student_input(sample)
    return _sanitize_public_text(_coerce_text(student.get("hints_text") or sample.get("hints_text") or ""))


def _source_line_clues(sample: dict[str, Any]) -> list[dict[str, Any]]:
    student = _student_input(sample)
    clues = student.get("source_line_clues") or sample.get("source_line_clues")
    if isinstance(clues, list):
        return clues

    localized = student.get("localized_buggy_files") or []
    if not isinstance(localized, list):
        return []

    extracted: list[dict[str, Any]] = []
    for file_entry in localized:
        if not isinstance(file_entry, dict):
            continue
        file_path = file_entry.get("file")
        old_buggy_lines = file_entry.get("old_buggy_lines")
        if not file_path or not isinstance(old_buggy_lines, list):
            continue
        for hunk in old_buggy_lines:
            if not isinstance(hunk, dict):
                continue
            removed_lines = hunk.get("removed_lines")
            if removed_lines:
                extracted.append(
                    {
                        "file": file_path,
                        "old_start": hunk.get("old_start"),
                        "old_end": hunk.get("old_end"),
                        "removed_lines": removed_lines,
                    }
                )
    return extracted


def _system_prompt(stage: str) -> str:
    return (
        "You are a coding agent running an offline stage-specific evaluation. "
        "Use only the task information in the user message and repository evidence observed through tools. "
        "Do not mention training objectives, hidden labels, or dataset internals. "
        f"Evaluation stage: {stage}."
    )


def build_eval_messages(stage: str, sample: dict[str, Any]) -> list[dict[str, str]]:
    """Build model-facing evaluation prompts independent from training prompts."""

    if stage not in EVAL_STAGES:
        raise ValueError(f"Unsupported evaluation stage {stage!r}. Supported: {', '.join(EVAL_STAGES)}")

    issue = _issue_statement(sample)
    hints = _hints(sample)
    common = [
        f"## Evaluation stage\n{stage}",
        f"## Issue statement\n{issue}",
        f"## Hints\n{hints}",
    ]

    if stage == "causal_localization":
        body = "\n\n".join(
            [
                *common,
                (
                    "## Task\n"
                    "Identify the causal bug file path and the smallest source line ranges that explain the issue. "
                    "Inspect repository files before submitting. Do not create a patch or reproduction script.\n\n"
                    "Return a JSON object in the finish result with this schema:\n"
                    '{\n  "locations": [\n'
                    '    {"file": "relative/non_test_file.py", "start": 10, "end": 20, "evidence": "observed code evidence"}\n'
                    '  ],\n  "uncertainty": "short note"\n}\n\n'
                    "The verifier scores bug file path hits separately from line range hits. "
                    "Line range score is matched gold source lines divided by total gold source lines."
                ),
            ]
        )
    elif stage == "localized_issue_reproduction":
        body = "\n\n".join(
            [
                *common,
                (
                    "## source_line_clues\n"
                    "These are old source lines from the buggy version that were later changed. "
                    "Use them as starting clues for the behavior your reproduction should exercise; "
                    "confirm surrounding code with tools.\n"
                    + _json_block(_source_line_clues(sample))
                ),
                (
                    "## Task\n"
                    "Create a localized issue reproduction artifact. Do not edit source files or existing tests, "
                    "and do not apply patches yourself.\n\n"
                    "Before finish:\n"
                    "1. inspect the localized source file with tools;\n"
                    "2. create /testbed/reproduce_issue.py;\n"
                    "3. run /testbed/reproduce_issue.py.\n\n"
                    "The external verifier will run /testbed/reproduce_issue.py on the buggy repository, then apply a "
                    "repair patch when available and run the same script again. A strong reproduction exits non-zero "
                    "on the buggy repo and exits 0 after that external repair. Your task is not to repair the bug."
                ),
            ]
        )
    else:
        student = _student_input(sample)
        reproduction_tests = student.get("reproduction_tests") or sample.get("reproduction_tests") or {}
        body = "\n\n".join(
            [
                *common,
                "## Public reproduction context\n" + _json_block(reproduction_tests),
                (
                    "## Task\n"
                    "Produce a minimal source patch. Do not edit tests. Run relevant local checks if available.\n\n"
                    "The verifier will use FAIL_TO_PASS tests to measure whether the regression is fixed and "
                    "PASS_TO_PASS tests to measure whether existing behavior is preserved."
                ),
            ]
        )

    return [{"role": "system", "content": _system_prompt(stage)}, {"role": "user", "content": body}]


def build_eval_record(stage: str, sample: dict[str, Any], *, split: str = "valid") -> dict[str, Any]:
    """Build an offline evaluation record with model prompt and verifier-only metadata."""

    if stage not in EVAL_STAGES:
        raise ValueError(f"Unsupported evaluation stage {stage!r}. Supported: {', '.join(EVAL_STAGES)}")

    task_id = str(sample.get("task_id") or sample.get("instance_id") or "")
    student = _student_input(sample)
    record: dict[str, Any] = {
        "task_id": task_id,
        "stage": stage,
        "split": split,
        "eval_messages": build_eval_messages(stage, sample),
        "metadata": {
            "codebase_access": student.get("codebase_access", {}),
        },
        "verifier": _build_verifier_metadata(stage, sample),
    }
    return record


def iter_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_eval_records_from_jsonl(
    path: str | Path,
    *,
    stage: str,
    split: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample in iter_jsonl(path):
        sample_stage = sample.get("stage") or stage
        if sample_stage != stage:
            raise ValueError(f"Expected stage={stage!r}, got {sample_stage!r} in {path}")
        records.append(build_eval_record(stage, sample, split=split))
        if limit is not None and len(records) >= limit:
            break
    return records


def write_eval_records_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def _build_verifier_metadata(stage: str, sample: dict[str, Any]) -> dict[str, Any]:
    if stage == "causal_localization":
        return {
            "kind": "location_file_and_line_overlap",
            "gold_ranges": [
                {"file": item.file, "start": item.start, "end": item.end}
                for item in gold_ranges_from_sample(sample)
            ],
        }
    if stage == "localized_issue_reproduction":
        spec = build_issue_reproduction_verifier_spec(sample)
        return {
            "kind": "issue_reproduction_buggy_then_gold_patch",
            "script_path": spec.script_path,
            "timeout": spec.timeout,
            "golden_patch": spec.golden_patch,
            "test_patch": spec.test_patch,
            "fail_to_pass_test_ids": list(spec.fail_to_pass_test_ids),
        }
    if stage == "patch_generation":
        return {
            "kind": "fail_to_pass_and_pass_to_pass",
            "FAIL_TO_PASS": _list_from_sample(sample, "FAIL_TO_PASS"),
            "PASS_TO_PASS": _list_from_sample(sample, "PASS_TO_PASS"),
        }
    raise ValueError(f"Unsupported evaluation stage {stage!r}")


def _list_from_sample(sample: dict[str, Any], key: str) -> list[str]:
    value = sample.get(key)
    if value is None:
        teacher_pi = sample.get("teacher_pi")
        if isinstance(teacher_pi, dict):
            value = teacher_pi.get(key)
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value] if value else []
        value = parsed
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def build_issue_reproduction_verifier_spec(
    sample: dict[str, Any],
    *,
    script_path: str = "/testbed/reproduce_issue.py",
    timeout: int = 120,
) -> IssueReproductionVerifierSpec:
    teacher_pi = sample.get("teacher_pi", {})
    if not isinstance(teacher_pi, dict):
        teacher_pi = {}
    golden_patch = str(
        teacher_pi.get("golden_patch")
        or teacher_pi.get("patch")
        or sample.get("golden_patch")
        or sample.get("patch")
        or ""
    )
    test_patch = str(teacher_pi.get("test_patch") or sample.get("test_patch") or "")
    fail_to_pass = teacher_pi.get("fail_to_pass_test_ids") or sample.get("FAIL_TO_PASS") or []
    if isinstance(fail_to_pass, str):
        try:
            fail_to_pass = json.loads(fail_to_pass)
        except json.JSONDecodeError:
            fail_to_pass = [fail_to_pass]
    if not isinstance(fail_to_pass, (list, tuple)):
        fail_to_pass = [fail_to_pass]
    return IssueReproductionVerifierSpec(
        script_path=script_path,
        timeout=max(1, int(timeout)),
        golden_patch=golden_patch,
        test_patch=test_patch,
        fail_to_pass_test_ids=tuple(str(item) for item in fail_to_pass if str(item)),
    )


def build_issue_reproduction_verifier_shell(
    *,
    script_path: str = "/testbed/reproduce_issue.py",
    gold_patch_path: str = "/tmp/rllm_gold.patch",
) -> str:
    """Build a shell verifier body that expects the gold patch file to exist in the container."""

    return f"""set +e
cd /testbed || exit 0
SCRIPT={json.dumps(script_path)}
PATCH={json.dumps(gold_patch_path)}
if [ ! -s "$SCRIPT" ]; then
  echo "__RLLM_ISSUE_REPRODUCTION_FILE_EXISTS__=0"
  echo "__RLLM_ISSUE_REPRODUCTION_BUGGY_EXIT__="
  echo "__RLLM_ISSUE_REPRODUCTION_PATCH_APPLY_EXIT__="
  echo "__RLLM_ISSUE_REPRODUCTION_PATCHED_EXIT__="
  exit 0
fi
echo "__RLLM_ISSUE_REPRODUCTION_FILE_EXISTS__=1"
if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
$PY "$SCRIPT"
BUGGY_EXIT=$?
echo "__RLLM_ISSUE_REPRODUCTION_BUGGY_EXIT__=$BUGGY_EXIT"
if [ ! -s "$PATCH" ]; then
  echo "__RLLM_ISSUE_REPRODUCTION_PATCH_APPLY_EXIT__="
  echo "__RLLM_ISSUE_REPRODUCTION_PATCHED_EXIT__="
  exit 0
fi
git apply --whitespace=nowarn "$PATCH"
PATCH_APPLY_EXIT=$?
echo "__RLLM_ISSUE_REPRODUCTION_PATCH_APPLY_EXIT__=$PATCH_APPLY_EXIT"
if [ "$PATCH_APPLY_EXIT" -ne 0 ]; then
  echo "__RLLM_ISSUE_REPRODUCTION_PATCHED_EXIT__="
  exit 0
fi
$PY "$SCRIPT"
PATCHED_EXIT=$?
echo "__RLLM_ISSUE_REPRODUCTION_PATCHED_EXIT__=$PATCHED_EXIT"
exit 0
"""


def parse_issue_reproduction_verifier_output(output: str) -> dict[str, Any]:
    markers = {
        "file_exists": "__RLLM_ISSUE_REPRODUCTION_FILE_EXISTS__",
        "buggy_exit_code": "__RLLM_ISSUE_REPRODUCTION_BUGGY_EXIT__",
        "patch_apply_exit_code": "__RLLM_ISSUE_REPRODUCTION_PATCH_APPLY_EXIT__",
        "patched_exit_code": "__RLLM_ISSUE_REPRODUCTION_PATCHED_EXIT__",
    }

    parsed: dict[str, Any] = {}
    for key, marker in markers.items():
        match = re.search(re.escape(marker) + r"=(?P<value>[^\n\r]*)", output)
        value = match.group("value").strip() if match else ""
        if key == "file_exists":
            parsed[key] = value == "1"
        else:
            parsed[key] = _parse_int_or_none(value)
    parsed.setdefault("file_exists", False)
    parsed.setdefault("buggy_exit_code", None)
    parsed.setdefault("patch_apply_exit_code", None)
    parsed.setdefault("patched_exit_code", None)
    return parsed


def run_issue_reproduction_verifier_in_container(
    container: Any,
    verifier: dict[str, Any],
    *,
    gold_patch_path: str = "/tmp/rllm_gold.patch",
) -> dict[str, Any]:
    """Run the stage2 two-phase issue-reproduction verifier in an existing Docker container."""

    script_path = str(verifier.get("script_path") or "/testbed/reproduce_issue.py")
    timeout = int(verifier.get("timeout") or 120)
    golden_patch = str(verifier.get("golden_patch") or "")
    if golden_patch.strip():
        _copy_text_to_container(container, gold_patch_path, golden_patch)

    shell_body = build_issue_reproduction_verifier_shell(
        script_path=script_path,
        gold_patch_path=gold_patch_path,
    )
    command = ["bash", "-lc", shell_body]
    exec_result = container.exec_run(command)
    output = _docker_exec_output(exec_result)
    parsed = parse_issue_reproduction_verifier_output(output)
    score = score_issue_reproduction(
        file_exists=parsed["file_exists"],
        buggy_exit_code=parsed["buggy_exit_code"],
        patched_exit_code=parsed["patched_exit_code"],
    )
    return {
        **score,
        "patch_apply_exit_code": parsed["patch_apply_exit_code"],
        "timeout": timeout,
        "command": command,
        "output_preview": output[:4000],
    }


def _copy_text_to_container(container: Any, target_path: str, text: str) -> None:
    target = Path(target_path)
    arcname = str(target).lstrip("/")
    data = text.encode("utf-8")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        info = tarfile.TarInfo(name=arcname)
        info.size = len(data)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(data))
    stream.seek(0)
    container.put_archive("/", stream.getvalue())


def _docker_exec_output(exec_result: Any) -> str:
    if isinstance(exec_result, tuple):
        output = exec_result[1] if len(exec_result) > 1 else b""
    else:
        output = getattr(exec_result, "output", b"")
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return str(output or "")


def _parse_int_or_none(value: str) -> int | None:
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def extract_location_predictions(text: str) -> list[EvalLineRange]:
    """Extract file/range predictions from finish text or a JSON finish artifact."""

    result_text = _extract_finish_result(text)
    ranges: list[EvalLineRange] = []
    ranges.extend(_extract_json_locations(result_text))
    ranges.extend(_extract_text_locations(result_text))
    return _dedupe_ranges(ranges)


def _extract_finish_result(text: str) -> str:
    match = re.search(
        r"<parameter=result>\s*(?P<result>.*?)\s*</parameter>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group("result") if match else text


def _extract_json_locations(text: str) -> list[EvalLineRange]:
    stripped = text.strip()
    if not stripped:
        return []

    candidates = [stripped]
    object_match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if object_match:
        candidates.append(object_match.group(0))

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        locations = payload.get("locations") if isinstance(payload, dict) else payload
        if not isinstance(locations, list):
            continue
        parsed: list[EvalLineRange] = []
        for item in locations:
            if not isinstance(item, dict):
                continue
            file_path = item.get("file") or item.get("path")
            start = item.get("start") or item.get("line_start") or item.get("old_start")
            end = item.get("end") or item.get("line_end") or item.get("old_end") or start
            if file_path and start is not None:
                parsed.append(EvalLineRange(file=str(file_path), start=int(start), end=int(end)))
        if parsed:
            return parsed
    return []


def _extract_text_locations(text: str) -> list[EvalLineRange]:
    path_pattern = r"(?P<file>(?:/testbed/|testbed/|\./)?[A-Za-z0-9_./-]+\.(?:py|pyi|js|ts|tsx|jsx|java|go|rs|c|cc|cpp|h|hpp|rb|php|scala|kt|swift))"
    line_pattern = (
        r"(?:line(?:s| range)?|start)\s*[:=]?\s*[`'\"]?"
        r"(?P<start>\d+)[`'\"]?"
        r"(?:\s*(?:-|–|—|to|,|end[:=]?)\s*[`'\"]?(?P<end>\d+)[`'\"]?)?"
    )
    patterns = [
        re.compile(path_pattern + r"[^\n]{0,80}?" + line_pattern, re.IGNORECASE),
        re.compile(r"file\s*[:=]\s*" + path_pattern + r"[^\n]{0,80}?" + line_pattern, re.IGNORECASE),
        re.compile(path_pattern + r"[\s\S]{0,200}?" + line_pattern, re.IGNORECASE),
        re.compile(r"file\s*[:=]\s*" + path_pattern + r"[\s\S]{0,200}?" + line_pattern, re.IGNORECASE),
    ]
    ranges: list[EvalLineRange] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            start = int(match.group("start"))
            end = int(match.group("end") or start)
            ranges.append(EvalLineRange(file=match.group("file"), start=start, end=end))
    return ranges


def _dedupe_ranges(ranges: list[EvalLineRange]) -> list[EvalLineRange]:
    seen: set[EvalLineRange] = set()
    deduped: list[EvalLineRange] = []
    for item in ranges:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def exact_file_match_reward(predicted: list[EvalLineRange], gold: list[EvalLineRange]) -> float:
    """Return a binary reward when the predicted and gold bug-file sets match."""

    gold_files = {item.file for item in gold}
    predicted_files = {item.file for item in predicted}
    return float(bool(gold_files) and predicted_files == gold_files)


def score_location_prediction(predicted: list[EvalLineRange], gold: list[EvalLineRange]) -> dict[str, Any]:
    gold_files = {item.file for item in gold}
    predicted_files = {item.file for item in predicted}
    file_hits = gold_files & predicted_files

    gold_lines = _line_set(gold)
    predicted_lines = _line_set(predicted)
    hit_lines = gold_lines & predicted_lines

    gold_file_count = len(gold_files)
    gold_line_count = len(gold_lines)
    predicted_line_count = len(predicted_lines)
    return {
        "gold_file_count": gold_file_count,
        "predicted_file_count": len(predicted_files),
        "file_hit_count": len(file_hits),
        "file_hit_rate": len(file_hits) / gold_file_count if gold_file_count else 0.0,
        "gold_line_count": gold_line_count,
        "predicted_line_count": predicted_line_count,
        "line_hit_count": len(hit_lines),
        "line_hit_rate": len(hit_lines) / gold_line_count if gold_line_count else 0.0,
        "range_precision_reward": len(hit_lines) / predicted_line_count if predicted_line_count else 0.0,
        "any_file_hit": bool(file_hits),
        "any_line_hit": bool(hit_lines),
    }


def _line_set(ranges: list[EvalLineRange]) -> set[tuple[str, int]]:
    lines: set[tuple[str, int]] = set()
    for item in ranges:
        for line in range(item.start, item.end + 1):
            lines.add((item.file, line))
    return lines


def gold_ranges_from_sample(sample: dict[str, Any]) -> list[EvalLineRange]:
    teacher_pi = sample.get("teacher_pi", {})
    if not isinstance(teacher_pi, dict):
        return []

    ranges: list[EvalLineRange] = []
    for item in teacher_pi.get("gold_hunk_ranges", []) or []:
        if not isinstance(item, dict) or not item.get("file"):
            continue
        start = item.get("old_start")
        if start is None:
            continue
        end = item.get("old_end")
        if end is None:
            count = item.get("old_count")
            end = int(start) + int(count) - 1 if count else start
        ranges.append(EvalLineRange(file=item["file"], start=int(start), end=int(end)))
    return ranges


def score_issue_reproduction(
    *,
    file_exists: bool,
    buggy_exit_code: int | None,
    patched_exit_code: int | None,
) -> dict[str, Any]:
    buggy_failed = file_exists and buggy_exit_code is not None and buggy_exit_code != 0
    patched_passed = patched_exit_code is not None and patched_exit_code == 0
    if not file_exists:
        score = 0.0
    elif patched_exit_code is None:
        score = 1.0 if buggy_failed else 0.0
    else:
        score = 1.0 if buggy_failed and patched_passed else 0.0

    return {
        "file_exists": file_exists,
        "buggy_exit_code": buggy_exit_code,
        "patched_exit_code": patched_exit_code,
        "buggy_failed": buggy_failed,
        "patched_passed": patched_passed,
        "score": score,
    }


def score_patch_tests(
    *,
    fail_to_pass_results: dict[str, bool],
    pass_to_pass_results: dict[str, bool],
) -> dict[str, Any]:
    fail_to_pass_rate = _pass_rate(fail_to_pass_results)
    pass_to_pass_rate = _pass_rate(pass_to_pass_results)
    resolved = bool(fail_to_pass_results) and all(fail_to_pass_results.values()) and all(pass_to_pass_results.values())
    score = 0.5 * fail_to_pass_rate + 0.5 * pass_to_pass_rate
    return {
        "fail_to_pass_total": len(fail_to_pass_results),
        "fail_to_pass_passed": sum(1 for ok in fail_to_pass_results.values() if ok),
        "fail_to_pass_rate": fail_to_pass_rate,
        "pass_to_pass_total": len(pass_to_pass_results),
        "pass_to_pass_passed": sum(1 for ok in pass_to_pass_results.values() if ok),
        "pass_to_pass_rate": pass_to_pass_rate,
        "resolved": resolved,
        "score": score,
    }


def _pass_rate(results: dict[str, bool]) -> float:
    if not results:
        return 0.0
    return sum(1 for ok in results.values() if ok) / len(results)
