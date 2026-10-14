from __future__ import annotations

import re

ACTION_FUNCTION_NAME_PATTERN = re.compile(r"<function\s*=\s*([^>]+)>")
TOOL_OUTPUT_NAME_PATTERN = re.compile(r"Execution output of \[([^\]]+)\]:")

FINISH_FUNCTION_NAMES = {"finish", "submit"}
# `file_editor`/`search` are the r2egym scaffold's tools; `str_replace_editor` is the
# OpenHands/SWE-agent editor. The observation header is
# "Execution output of [<function_name>]", so both names must be recognized or an
# OpenHands run scores zero repository-inspection and zero edit calls.
EDITOR_TOOL_NAMES = {"file_editor", "str_replace_editor"}
REWARDED_REPOSITORY_TOOL_NAMES = {"search", "execute_bash"} | EDITOR_TOOL_NAMES

FAILED_TOOL_RESULT_MARKERS = (
    "exit code: error",
    "usage:",
    " error:",
    "not allowed",
    "no matches found",
    "not found or not a directory",
    "no such file or directory",
    "filenotfounderror",
)


def _normalize_tool_name(name: str | None) -> str:
    return str(name or "").strip().lower()


def count_non_finish_tool_calls(messages: list[dict] | None) -> int:
    """Count assistant XML tool calls, excluding finish/submit."""

    if not messages:
        return 0

    count = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content", "")
        for function_name in ACTION_FUNCTION_NAME_PATTERN.findall(str(content)):
            normalized = _normalize_tool_name(function_name)
            if normalized and normalized not in FINISH_FUNCTION_NAMES:
                count += 1
    return count


def _tool_name_from_result(content: str) -> str:
    match = TOOL_OUTPUT_NAME_PATTERN.search(content)
    if not match:
        return ""
    return _normalize_tool_name(match.group(1))


def is_successful_tool_result(content: str | None) -> bool:
    text = str(content or "")
    if not text or "<<< Finished >>>" in text:
        return False

    tool_name = _tool_name_from_result(text)
    if tool_name not in REWARDED_REPOSITORY_TOOL_NAMES:
        return False

    lowered = text.lower()
    if any(marker in lowered for marker in FAILED_TOOL_RESULT_MARKERS):
        return False
    if "exit code:" in lowered and "exit code: 0" not in lowered:
        return False

    return True


def count_successful_non_finish_tool_calls(messages: list[dict] | None) -> int:
    """Count successful repository-inspection tool results, excluding finish."""

    if not messages:
        return 0

    return sum(
        1
        for message in messages
        if isinstance(message, dict)
        and message.get("role") == "tool"
        and is_successful_tool_result(message.get("content", ""))
    )


# Only `str_replace`/`insert` on an existing file emit "has been edited"; a brand-new
# file (`create`, e.g. a repro script) emits "File created at ..." and is usually an
# untracked file that never lands in the final git-diff patch. So counting only the
# "has been edited" marker is a robust proxy for "the trajectory modified source that
# would appear in the output patch".
FILE_EDIT_SUCCESS_MARKERS = ("has been edited",)


def count_successful_edit_calls(messages: list[dict] | None) -> int:
    """Count successful file edits (str_replace/insert) as a nonempty-patch proxy."""

    if not messages:
        return 0

    count = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        content = str(message.get("content", ""))
        if _tool_name_from_result(content) not in EDITOR_TOOL_NAMES:
            continue
        lowered = content.lower()
        if any(marker in lowered for marker in FILE_EDIT_SUCCESS_MARKERS):
            count += 1
    return count
