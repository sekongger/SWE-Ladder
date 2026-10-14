from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from .staged_dataset import (
    _causal_localization_required_artifact,
    _json_block,
    _localized_issue_reproduction_required_artifact,
    _source_line_clues_from_localized_files,
    _student_system_prompt,
    _teacher_evidence_heading,
    _verified_repository_evidence,
    build_student_messages,
    build_teacher_messages,
)


_END_TO_END_STAGES = frozenset(
    {
        "causal_localization",
        "localized_issue_reproduction",
        "patch_generation",
    }
)


_SWEAGENT_EVAL_PROMPT_DIR = Path(__file__).resolve().parent / "prompt_templates_sweagent"

# This text is deliberately role- and stage-neutral.  It describes only the
# notation used by the structured evidence that follows it, so the canonical
# SWE-agent system prompt remains byte-identical for both message variants.
_SWEAGENT_PRIVILEGED_INFORMATION_GUIDE = """<privileged_information_guide>
Read the following block as verified repository evidence. Paths are relative to
the repository root. Hunk ranges and the numbers in removed_lines refer to the
old file. In a unified diff, lines beginning with '-' are old content and lines
beginning with '+' are new content. Test IDs are pytest or unittest selectors.
Reference test code is executable code that can be adapted into a check. JSON
keys describe how to interpret the value that follows them.
</privileged_information_guide>"""


def _sweagent_eval_prompt_mode(record: dict[str, Any] | None = None) -> bool:
    """Use the byte-canonical SWE-agent evaluation prompts.

    The regular staged prompts intentionally contain stage-specific task
    instructions.  The SWE-agent baseline requires the student messages to be
    exactly the messages used by R2E-Gym-eval, so this mode is opt-in and keeps
    that contract separate from the existing staged prompt family.
    """

    if os.environ.get("SWE_SCAFFOLD", "").strip().lower() == "leaf":
        return True
    if isinstance(record, dict) and "sweagent_eval_prompt" in record:
        return _flag_enabled(record.get("sweagent_eval_prompt"), default=False)
    return _flag_enabled(os.environ.get("OPD_SWEAGENT_EVAL_PROMPT"), default=False)


def _render_sweagent_eval_student_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    student_input = record.get("student_input") if isinstance(record.get("student_input"), dict) else {}
    issue = str(student_input.get("issue_statement") or record.get("problem_statement") or "")
    # Keep the canonical eval-prompt switch usable for Leaf as well.  Leaf is
    # a separate protocol: reusing the SWE-agent prompt here would advertise
    # execute_bash/str_replace_editor/submit while the Leaf runner exposes
    # Read/Write/Edit/Glob/Bash.  The scaffold selector is already exported to
    # every rollout worker by the launcher.
    if os.environ.get("SWE_SCAFFOLD", "").strip().lower() == "leaf":
        from rllm.agents.system_prompts import LEAF_SYSTEM_PROMPT, LEAF_USER_PROMPT

        return [
            {"role": "system", "content": LEAF_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": LEAF_USER_PROMPT.replace("{problem_statement}", issue),
            },
        ]

    system_path = _SWEAGENT_EVAL_PROMPT_DIR / "system_prompt.md"
    user_path = _SWEAGENT_EVAL_PROMPT_DIR / "instance_prompt.md"
    if not system_path.exists() or not user_path.exists():
        raise FileNotFoundError(
            "SWE-agent canonical prompt assets are missing: "
            f"{system_path} and/or {user_path}"
        )
    context = {"problem_statement": issue}
    return [
        {"role": "system", "content": system_path.read_text(encoding="utf-8")},
        {"role": "user", "content": user_path.read_text(encoding="utf-8").format_map(_SafeFormatDict(context))},
    ]


def _sweagent_teacher_messages_with_user_privilege(
    record: dict[str, Any], *, evidence_stage: str | None = None, evidence: dict[str, Any] | None = None
) -> list[dict[str, str]]:
    """Keep the system prompt byte-identical and append readable evidence to user input."""

    student = _render_sweagent_eval_student_messages(record)
    stage = evidence_stage or str(record.get("stage") or "patch_generation")
    teacher_pi = record.get("teacher_pi") if isinstance(record.get("teacher_pi"), dict) else {}
    evidence = evidence if evidence is not None else _verified_repository_evidence(stage, teacher_pi)
    privilege = (
        "\n\n"
        f"{_SWEAGENT_PRIVILEGED_INFORMATION_GUIDE}\n"
        "<privileged_information>\n"
        f"{_json_block(evidence)}\n"
        "</privileged_information>"
    )
    return [
        {"role": "system", "content": student[0]["content"]},
        {"role": "user", "content": student[1]["content"] + privilege},
    ]


def _flag_enabled(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _messages_to_text(messages: list[dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = str(message.get("role", "")).upper()
        content = str(message.get("content", ""))
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _record_from_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    raw_record = entry.get("raw_record")
    if isinstance(raw_record, dict) and raw_record.get("stage"):
        record = copy.deepcopy(raw_record)
        if isinstance(entry.get("student_input"), dict):
            record["student_input"] = copy.deepcopy(entry["student_input"])
        if isinstance(entry.get("teacher_pi"), dict):
            record["teacher_pi"] = copy.deepcopy(entry["teacher_pi"])
        if "sweagent_eval_prompt" in entry:
            record["sweagent_eval_prompt"] = entry["sweagent_eval_prompt"]
        return record

    stage = entry.get("stage")
    student_input = entry.get("student_input")
    if not isinstance(stage, str) or not isinstance(student_input, dict):
        return None
    teacher_pi = entry.get("teacher_pi") if isinstance(entry.get("teacher_pi"), dict) else {}
    return {
        "task_id": entry.get("task_id") or entry.get("instance_id") or "",
        "stage": stage,
        "student_input": copy.deepcopy(student_input),
        "teacher_pi": copy.deepcopy(teacher_pi),
        "sweagent_eval_prompt": entry.get("sweagent_eval_prompt", False),
    }


def _required_artifact(stage: str) -> str:
    if stage == "causal_localization":
        return _causal_localization_required_artifact()
    if stage == "localized_issue_reproduction":
        return _localized_issue_reproduction_required_artifact()
    if stage == "patch_generation":
        return (
            "Generate a minimal source patch, explain the patch rationale, and provide a verification plan. "
            "Use the issue statement, hints, and localized_buggy_files as repair guidance. "
            "Do not edit tests or assume any hidden test content."
        )
    return ""


def _template_path(stage: str, role: str) -> Path | None:
    root = os.environ.get("OPD_PROMPT_TEMPLATE_DIR")
    template_root = Path(root) if root else Path(__file__).resolve().parent / "prompt_templates"
    path = template_root / f"{stage}_{role}.md"
    return path if path.exists() else None


def _template_context(
    record: dict[str, Any], *, default_system_prompt: str = "", evidence_stage: str | None = None
) -> dict[str, str]:
    stage = str(record.get("stage") or "")
    evidence_stage = evidence_stage or stage
    student_input = record.get("student_input") if isinstance(record.get("student_input"), dict) else {}
    teacher_pi = record.get("teacher_pi") if isinstance(record.get("teacher_pi"), dict) else {}
    localized_buggy_files = student_input.get("localized_buggy_files", [])
    teacher_evidence = _verified_repository_evidence(evidence_stage, teacher_pi)
    return {
        "task_id": str(record.get("task_id") or record.get("instance_id") or ""),
        "stage": stage,
        "issue_statement": str(student_input.get("issue_statement") or ""),
        "hints_text": str(student_input.get("hints_text") or ""),
        "localized_buggy_files_json": _json_block(localized_buggy_files),
        "source_line_clues_json": _json_block(_source_line_clues_from_localized_files(localized_buggy_files)),
        "teacher_evidence_json": _json_block(teacher_evidence),
        "teacher_evidence_heading": _teacher_evidence_heading(evidence_stage),
        "required_artifact": _required_artifact(stage),
        "default_system_prompt": default_system_prompt,
    }


def _render_template(path: Path, context: dict[str, str]) -> str:
    return path.read_text(encoding="utf-8").format_map(_SafeFormatDict(context))


def _end_to_end_enabled() -> bool:
    return os.environ.get("OPD_END_TO_END_PROMPT", "0") == "1"


def _stage_routing_enabled() -> bool:
    return os.environ.get("OPSD_STAGE_ROUTING", "0") == "1"


def _opsd_privileged_stages() -> frozenset[str]:
    raw = os.environ.get("OPSD_PRIVILEGED_STAGES")
    if raw is None or not raw.strip():
        return _END_TO_END_STAGES

    stages = frozenset(part.strip().lower() for part in raw.split(",") if part.strip())
    unknown = stages - _END_TO_END_STAGES
    if unknown:
        raise ValueError(
            "OPSD_PRIVILEGED_STAGES contains unsupported stage(s): "
            f"{', '.join(sorted(unknown))}"
        )
    return stages


def _opsd_privileged_evidence_stage(stage: str) -> str:
    """Return the evidence source for one privileged routing stage.

    ``OPSD_PRIVILEGED_EVIDENCE_STAGE_MAP`` is a comma-separated ``target=source``
    mapping. It changes only the teacher evidence block; the target stage keeps
    its own task instruction and token routing. Unmapped stages use their own
    evidence, preserving the existing behavior.
    """

    raw = os.environ.get("OPSD_PRIVILEGED_EVIDENCE_STAGE_MAP", "").strip()
    if not raw:
        return stage

    mapping: dict[str, str] = {}
    for item in raw.split(","):
        target, separator, source = item.partition("=")
        target = target.strip().lower()
        source = source.strip().lower()
        if not separator or not target or not source:
            raise ValueError(
                "OPSD_PRIVILEGED_EVIDENCE_STAGE_MAP entries must use target=source"
            )
        unknown = {target, source} - _END_TO_END_STAGES
        if unknown:
            raise ValueError(
                "OPSD_PRIVILEGED_EVIDENCE_STAGE_MAP contains unsupported stage(s): "
                f"{', '.join(sorted(unknown))}"
            )
        if target in mapping:
            raise ValueError(
                f"OPSD_PRIVILEGED_EVIDENCE_STAGE_MAP maps {target!r} more than once"
            )
        mapping[target] = source
    return mapping.get(stage, stage)


def render_end_to_end_teacher_messages_with_evidence(
    record: dict[str, Any], evidence: dict[str, Any]
) -> list[dict[str, str]]:
    """Dynamic-frontier variant (OPSD_GRAPH_FRONTIER=1): render the end-to-end
    teacher prompt with an EXPLICIT evidence dict (one establishment epoch's
    established + frontier snapshot) instead of a stage-derived one. Everything
    else — templates, student prompt, structure — is identical, so teacher
    variants still differ from the student prompt only by the privileged block."""
    if _sweagent_eval_prompt_mode(record):
        return _sweagent_teacher_messages_with_user_privilege(record, evidence=evidence)

    rec = dict(record)
    rec["stage"] = "patch_generation"  # template context needs a valid stage tag
    system_path = _template_path("end_to_end", "teacher_system")
    user_path = _template_path("end_to_end", "teacher_user")
    if system_path is None or user_path is None:
        return render_student_messages_at_runtime(rec)
    context = _template_context(rec, default_system_prompt="")
    context["teacher_evidence_json"] = _json_block(evidence)
    context["teacher_evidence_heading"] = "Verified process knowledge (current investigation state)"
    return [
        {"role": "system", "content": _render_template(system_path, context)},
        {"role": "user", "content": _render_template(user_path, context)},
    ]


def render_end_to_end_combined_teacher_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """Render one no-router teacher prompt containing all privileged evidence.

    The end-to-end student conversation is rendered with the patch-stage task
    instruction, then receives one stable, JSON-encoded evidence block.  The
    evidence keys are deliberately ordered LOC, REPRO, PATCH so the teacher
    context is deterministic and easy to inspect.  Existing stage-routed
    callers continue using :func:`render_end_to_end_teacher_messages`.
    """

    if _sweagent_eval_prompt_mode(record):
        teacher_pi = record.get("teacher_pi") if isinstance(record.get("teacher_pi"), dict) else {}
        combined_evidence = {
            "LOC privileged evidence": _verified_repository_evidence("causal_localization", teacher_pi),
            "REPRO privileged evidence": _verified_repository_evidence("localized_issue_reproduction", teacher_pi),
            "PATCH privileged evidence": _verified_repository_evidence("patch_generation", teacher_pi),
        }
        return _sweagent_teacher_messages_with_user_privilege(
            record, evidence_stage="patch_generation", evidence=combined_evidence
        )

    rec = dict(record)
    rec["stage"] = "patch_generation"
    system_path = _template_path("end_to_end", "teacher_system")
    user_path = _template_path("end_to_end", "teacher_user")
    if system_path is None or user_path is None:
        return build_teacher_messages(rec)

    teacher_pi = rec.get("teacher_pi") if isinstance(rec.get("teacher_pi"), dict) else {}
    combined_evidence = {
        "LOC privileged evidence": _verified_repository_evidence("causal_localization", teacher_pi),
        "REPRO privileged evidence": _verified_repository_evidence("localized_issue_reproduction", teacher_pi),
        "PATCH privileged evidence": _verified_repository_evidence("patch_generation", teacher_pi),
    }
    context = _template_context(rec, default_system_prompt="", evidence_stage="patch_generation")
    context["teacher_evidence_json"] = _json_block(combined_evidence)
    context["teacher_evidence_heading"] = "Combined privileged evidence (LOC, REPRO, PATCH)"
    return [
        {"role": "system", "content": _render_template(system_path, context)},
        {"role": "user", "content": _render_template(user_path, context)},
    ]


def render_end_to_end_teacher_messages(record: dict[str, Any], stage: str) -> list[dict[str, str]]:
    """Render one stage-routed end-to-end teacher prompt.

    Stages excluded by ``OPSD_PRIVILEGED_STAGES`` use the exact student messages. Included
    stages use the student prompt plus their configured privileged evidence block. When the
    variable is unset, all stages remain privileged for backward compatibility.
    """
    if _sweagent_eval_prompt_mode(record):
        if stage not in _opsd_privileged_stages():
            return _render_sweagent_eval_student_messages(record)
        return _sweagent_teacher_messages_with_user_privilege(record, evidence_stage=_opsd_privileged_evidence_stage(stage))

    rec = dict(record)
    rec["stage"] = stage  # preserve the target stage's task instruction and artifact.
    if stage not in _opsd_privileged_stages():
        return render_student_messages_at_runtime(rec)
    system_path = _template_path("end_to_end", "teacher_system")
    user_path = _template_path("end_to_end", "teacher_user")
    if system_path is None or user_path is None:
        # fall back to the end-to-end student prompt with no evidence if templates are absent
        return render_student_messages_at_runtime(rec)
    evidence_stage = _opsd_privileged_evidence_stage(stage)
    context = _template_context(rec, default_system_prompt="", evidence_stage=evidence_stage)
    return [
        {"role": "system", "content": _render_template(system_path, context)},
        {"role": "user", "content": _render_template(user_path, context)},
    ]


def render_student_messages_at_runtime(record: dict[str, Any]) -> list[dict[str, str]]:
    if _sweagent_eval_prompt_mode(record):
        return _render_sweagent_eval_student_messages(record)

    stage = str(record.get("stage") or "")
    template_stage = "end_to_end" if _end_to_end_enabled() else stage
    system_path = _template_path(template_stage, "student_system")
    user_path = _template_path(template_stage, "student_user") or _template_path(template_stage, "student")
    if system_path is None and user_path is None:
        return build_student_messages(record)
    try:
        default_system = _student_system_prompt(stage)
    except Exception:
        default_system = ""  # end-to-end / unknown stage: templates supply the system prompt
    context = _template_context(record, default_system_prompt=default_system)
    system = _render_template(system_path, context) if system_path is not None else default_system
    user = _render_template(user_path, context) if user_path is not None else build_student_messages(record)[1]["content"]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def render_teacher_messages_at_runtime(record: dict[str, Any]) -> list[dict[str, str]]:
    if _sweagent_eval_prompt_mode(record):
        return _sweagent_teacher_messages_with_user_privilege(record)

    stage = str(record.get("stage") or "")
    system_path = _template_path(stage, "teacher_system")
    user_path = _template_path(stage, "teacher_user") or _template_path(stage, "teacher")
    if system_path is None and user_path is None:
        return build_teacher_messages(record)
    default_teacher_messages = build_teacher_messages(record)
    default_system = default_teacher_messages[0]["content"]
    context = _template_context(record, default_system_prompt=default_system)
    system = _render_template(system_path, context) if system_path is not None else default_system
    user = _render_template(user_path, context) if user_path is not None else default_teacher_messages[1]["content"]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def render_runtime_staged_prompt_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Optionally refresh staged OPD prompts from structured fields at runtime.

    This keeps old parquet files usable while allowing prompt text changes via
    code or markdown templates without rebuilding the parquet.
    """

    if not _flag_enabled(entry.get("opd_render_prompt_at_runtime"), default=_flag_enabled(os.environ.get("OPD_RENDER_PROMPT_AT_RUNTIME"))):
        return entry

    # Attach the instance's process graph (OPD_TEACHER_EVIDENCE_MODE=graph) to
    # every teacher_pi in the entry BEFORE rendering, so both the dataset-time
    # teacher messages and the trainer's per-stage routed variants (which read
    # privileged_context["teacher_pi"]) see it. No-op when the mode is off.
    from .process_graph_evidence import inject_graph_into_entry

    entry = inject_graph_into_entry(entry)

    record = _record_from_entry(entry)
    if record is None:
        return entry

    rendered = copy.deepcopy(entry)
    student_messages = render_student_messages_at_runtime(record)
    if not _stage_routing_enabled() and _flag_enabled(os.environ.get("OPSD_COMBINED_PRIVILEGED_TEACHER")):
        teacher_messages = render_end_to_end_combined_teacher_messages(record)
    else:
        teacher_messages = render_teacher_messages_at_runtime(record)
    rendered["student_messages"] = student_messages
    rendered["teacher_messages"] = teacher_messages
    rendered["problem_statement"] = _messages_to_text(student_messages)
    privileged_context = rendered.get("privileged_context")
    if not isinstance(privileged_context, dict):
        privileged_context = {}
    privileged_context = copy.deepcopy(privileged_context)
    privileged_context["teacher_messages"] = teacher_messages
    privileged_context.setdefault("student_input", copy.deepcopy(record.get("student_input", {})))
    privileged_context.setdefault("teacher_pi", copy.deepcopy(record.get("teacher_pi", {})))
    if "sweagent_eval_prompt" in record:
        privileged_context["sweagent_eval_prompt"] = record["sweagent_eval_prompt"]
    rendered["privileged_context"] = privileged_context
    return rendered
